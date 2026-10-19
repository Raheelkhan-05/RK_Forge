# forge/llm.py

"""Minimal Ollama client (stdlib only): streaming chat, JSON-schema output, instant cancellation, retries."""
from __future__ import annotations

import http.client
import json
import queue
import re
import socket
import threading
import time
import urllib.request
from typing import Callable, Iterator, Protocol
from urllib.parse import urlsplit

from .config import Config, RoleCfg
from .telemetry import Tracer, ollama_metrics, preview_messages

# Conservative on purpose: code, Hindi and Hinglish tokenise worse than English prose. Guessing too high makes
# Ollama silently cut the START of the prompt (the instructions), which looks like the model "forgetting".
CHARS_PER_TOKEN = 2.8


class LLMError(RuntimeError):
    pass


class FatalLLMError(LLMError):
    """Retrying cannot help (for example: the model is not installed)."""


class Cancelled(Exception):
    """Raised when the user cancels a run."""


RETRYABLE = (OSError, http.client.HTTPException, LLMError)


def est_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


class LLM(Protocol):
    def chat(self, role: str, messages: list[dict], schema: dict | None = None,
             on_token: Callable[[str, str], None] | None = None) -> str: ...


_END = object()


def _detail(raw: str) -> str:
    try:
        return str(json.loads(raw).get("error") or raw)[:300]
    except (ValueError, AttributeError):
        return raw[:300]


def post_stream(url: str, payload: dict, should_stop: Callable[[], bool], idle_timeout: float) -> Iterator[bytes]:
    """POST and yield the response line by line.

    The network work happens in a helper thread so a stop request is noticed within ~0.25 s even while Ollama is still
    loading the model or reading the prompt (no byte has arrived yet). On stop the socket is shut down, which makes
    Ollama abandon the generation instead of finishing it for nobody.
    """
    u = urlsplit(url)
    body = json.dumps(payload).encode("utf-8")
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=10)
    q: queue.Queue = queue.Queue()
    stopped = threading.Event()

    def pump() -> None:
        try:
            conn.connect()
            conn.sock.settimeout(None)
            conn.request("POST", u.path or "/", body=body, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            if resp.status != 200:
                msg = f"Ollama answered HTTP {resp.status}: {_detail(resp.read(2000).decode('utf-8', 'replace'))}"
                q.put((FatalLLMError if 400 <= resp.status < 500 else LLMError)(msg))
                return
            while True:
                line = resp.readline()
                if not line:
                    break
                q.put(line)
        except BaseException as e:  # noqa: BLE001
            if not stopped.is_set():
                q.put(e)
        finally:
            q.put(_END)

    threading.Thread(target=pump, daemon=True, name="forge-ollama-io").start()
    last = time.time()
    try:
        while True:
            if should_stop():
                raise Cancelled()
            try:
                item = q.get(timeout=0.25)
            except queue.Empty:
                if time.time() - last > idle_timeout:
                    raise TimeoutError(f"Ollama sent nothing for {int(idle_timeout)} seconds")
                continue
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            last = time.time()
            yield item
    finally:
        stopped.set()
        sock = getattr(conn, "sock", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


class OllamaLLM:
    def __init__(self, cfg: Config, should_stop: Callable[[], bool] | None = None):
        self.tracer = Tracer()
        self.cfg = cfg
        self.should_stop = should_stop or (lambda: False)
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
        self.last_truncated = False      # True when the last reply stopped because it hit num_predict

    # ------------------------------------------------------------------ health
    def _get(self, path: str, timeout: float = 5.0) -> dict:
        with urllib.request.urlopen(self.cfg.host + path, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def health(self) -> dict:
        try:
            tags = self._get("/api/tags")
            installed = [m["name"] for m in tags.get("models", [])]
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"Cannot reach Ollama at {self.cfg.host}: {e}", "installed": [], "loaded": [],
                    "missing": [], "advice": None}
        try:
            ps = self._get("/api/ps").get("models", [])
            loaded = [m["name"] for m in ps]
            placement = [{"name": m["name"], "size_gb": round(m.get("size", 0) / 1e9, 1),
                          "gpu_pct": round(100 * m.get("size_vram", 0) / m["size"]) if m.get("size") else None,
                          "context": m.get("context_length")} for m in ps]
        except Exception:  # noqa: BLE001
            loaded, placement = [], []
        advice = None
        for p in placement:
            if p["gpu_pct"] is not None and p["gpu_pct"] < 95:
                advice = (f"Only {p['gpu_pct']}% of {p['name']} is on the GPU. The rest runs on the CPU, which is why replies are slow. "
                          "Run: python -m forge tune")
        needed = sorted({rc.model for rc in self.cfg.roles.values()})
        missing = [m for m in needed if m not in installed and f"{m}:latest" not in installed]
        return {"ok": not missing, "installed": installed, "loaded": loaded, "missing": missing, "placement": placement,
                "advice": advice,
                "error": f"Missing models: {', '.join(missing)}. Run: ollama pull {missing[0]}" if missing else None}

    # ------------------------------------------------------------------ chat
    @staticmethod
    def _options(rc: RoleCfg, num_predict: int | None = None) -> dict:
        o = {"temperature": rc.temperature, "num_ctx": rc.num_ctx, "num_predict": num_predict or rc.num_predict,
             "seed": 7, "repeat_penalty": rc.repeat_penalty}
        if rc.num_gpu >= 0:
            o["num_gpu"] = rc.num_gpu
        if rc.num_thread > 0:
            o["num_thread"] = rc.num_thread
        return o

    def chat(self, role: str, messages: list[dict], schema: dict | None = None,
             on_token: Callable[[str, str], None] | None = None, num_predict: int | None = None) -> str:
        rc: RoleCfg = self.cfg.roles[role]
        attempts = 3
        last: Exception | None = None
        think = rc.think
        for i in range(attempts):
            try:
                text = self._stream(rc, messages, schema, think, on_token, role=role, attempt=i + 1, num_predict=num_predict)
                if not text.strip() and think:          # some builds return empty content when think+format
                    think = False
                    continue
                if not text.strip():
                    raise LLMError("Model returned an empty response")
                return text
            except (Cancelled, FatalLLMError):
                raise
            except RETRYABLE as e:
                last = e
                time.sleep(min(2 ** i, 8))
        raise LLMError(f"Ollama request failed after {attempts} attempts: {last}")

    def chat_stream(self, role: str, messages: list[dict],
                    on_token: Callable[[str, str], None] | None = None, think: bool | None = None) -> tuple[str, bool]:
        """Free-text streaming for conversation replies. Returns (text, truncated).

        A failure after the first visible token is NOT retried (that would repeat text on screen).
        """
        rc: RoleCfg = self.cfg.roles[role]
        use_think = rc.think if think is None else think
        attempts = 3
        last: Exception | None = None
        for i in range(attempts):
            shown = [False]

            def tap(kind: str, text: str) -> None:
                if kind == "content":
                    shown[0] = True
                if on_token:
                    on_token(kind, text)
            try:
                text = self._stream(rc, messages, None, use_think, tap, mark_truncation=False, role=role, attempt=i + 1)
                if not text.strip():
                    if use_think:
                        use_think = False
                        continue
                    raise LLMError("Model returned an empty response")
                return text, self.last_truncated
            except (Cancelled, FatalLLMError):
                raise
            except RETRYABLE as e:
                last = e
                if shown[0]:
                    raise LLMError(f"The reply was interrupted: {e}") from e
                time.sleep(min(2 ** i, 8))
        raise LLMError(f"Ollama request failed after {attempts} attempts: {last}")

    def _stream(self, rc: RoleCfg, messages, schema, think: bool, on_token, mark_truncation: bool = True,
                role: str = "", attempt: int = 1, num_predict: int | None = None) -> str:
        body = {"model": rc.model, "messages": messages, "stream": True, "think": think, "keep_alive": "30m",
                "options": self._options(rc, num_predict)}
        if schema:
            body["format"] = schema
        out: list[str] = []
        t0 = time.time()
        self.last_truncated = False
        est = sum(est_tokens(m.get("content") or "") for m in messages)
        sp = self.tracer.open(f"{role or 'model'} call", "llm", agent=role or "llm", model=rc.model, num_ctx=rc.num_ctx,
                              num_predict=num_predict or rc.num_predict, think=think, structured=bool(schema), attempt=attempt,
                              est_prompt_tokens=est, prompt=preview_messages(messages), live="waiting")
        first, n_c, n_t, last_u, done_chunk = None, 0, 0, 0.0, None
        lines = post_stream(self.cfg.host + "/api/chat", body, self.should_stop, self.cfg.limits.llm_idle_timeout)
        try:
            for raw in lines:
                if not raw.strip():
                    continue
                try:
                    chunk = json.loads(raw.decode("utf-8"))
                except ValueError as e:
                    raise LLMError(f"Ollama sent a broken line: {e}") from e
                if "error" in chunk:
                    raise LLMError(chunk["error"])
                msg = chunk.get("message", {})
                if msg.get("thinking"):
                    n_t += 1
                    if on_token:
                        on_token("thinking", msg["thinking"])
                piece = msg.get("content", "")
                if piece:
                    out.append(piece)
                    n_c += 1
                    if on_token:
                        on_token("content", piece)
                if piece or msg.get("thinking"):
                    now = time.time()
                    if first is None:
                        first = now
                        sp.update(first_token_s=round(now - t0, 3), live="generating")
                    elif now - last_u > 0.3:
                        sp.update(content_chunks=n_c, thinking_chunks=n_t)
                        last_u = now
                if chunk.get("done"):
                    done_chunk = chunk
                    self.stats["calls"] += 1
                    self.stats["prompt_tokens"] += chunk.get("prompt_eval_count", 0)
                    self.stats["completion_tokens"] += chunk.get("eval_count", 0)
                    self.stats["seconds"] += time.time() - t0
                    if chunk.get("done_reason") == "length":
                        self.last_truncated = True          # also for JSON calls: a cut-off JSON must not be retried blindly
                        if mark_truncation and not schema:
                            out.append("\n\n[[TRUNCATED]]")
        except BaseException as e:
            sp.close(error=f"{type(e).__name__}: {str(e)[:200]}", cancelled=isinstance(e, Cancelled),
                     wall_s=round(time.time() - t0, 3), content_chunks=n_c, thinking_chunks=n_t)
            raise
        finally:
            lines.close()
        text = "".join(out)
        shown = text.replace("\n\n[[TRUNCATED]]", "")
        m = ollama_metrics(done_chunk or {}, time.time() - t0, est)
        if ((done_chunk or {}).get("prompt_eval_count") or 0) >= int(rc.num_ctx * 0.95):
            m["hint"] += " The prompt filled almost the whole context window, so Ollama may have cut off the start of it."
        sp.close(response=shown[:1500], response_chars=len(shown), content_chunks=n_c, thinking_chunks=n_t,
                 truncated=self.last_truncated, **m)
        return text

    # ------------------------------------------------------------------ JSON helper
    def chat_json(self, role: str, messages: list[dict], schema: dict, on_token=None, retries: int = 2) -> dict:
        rc = self.cfg.roles[role]
        msgs = list(messages)
        last_err, boost = "", None
        for _ in range(retries + 1):
            text = self.chat(role, msgs, schema, on_token, num_predict=boost)
            cut = self.last_truncated
            try:
                data = parse_json(text)
                if isinstance(data, dict):
                    return data
                last_err = "top-level value must be a JSON object"
            except ValueError as e:
                last_err = str(e)
            if cut:
                # Same prompt + same limit would be cut off again. Allow a longer answer and ask for a shorter one.
                boost = min(int((boost or rc.num_predict) * 1.5), max(rc.num_predict, rc.num_ctx // 2))
                msgs = list(messages) + [{"role": "user", "content": "Your previous reply was cut off because it was too long. "
                                                                      "Reply again with ONLY the JSON object and keep every field short."}]
            else:
                msgs = msgs + [{"role": "assistant", "content": text[:2000]},
                               {"role": "user", "content": f"Your reply was not valid JSON ({last_err}). Reply again with ONLY the JSON object."}]
        raise LLMError(f"Could not obtain valid JSON from the model: {last_err}")


def parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        frag = text[start:end + 1]
        try:
            return json.loads(frag)
        except json.JSONDecodeError:
            frag = re.sub(r",\s*([}\]])", r"\1", frag)   # trailing commas
            try:
                return json.loads(frag)
            except json.JSONDecodeError as e:
                raise ValueError(str(e)) from e
    raise ValueError("no JSON object found")