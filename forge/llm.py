# forge/llm.py

"""Minimal Ollama client (stdlib only): streaming chat, JSON-schema output, cancellation, retries."""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Callable, Protocol

from .config import Config, RoleCfg

from .telemetry import NULL_SPAN, Tracer, ollama_metrics, preview_messages


class LLMError(RuntimeError):
    pass


class Cancelled(Exception):
    """Raised when the user cancels a run."""


def est_tokens(text: str) -> int:
    """Cheap token estimate; code averages ~3.2 chars/token for Qwen tokenizers."""
    return int(len(text) / 3.2) + 1


class LLM(Protocol):
    def chat(self, role: str, messages: list[dict], schema: dict | None = None,
             on_token: Callable[[str, str], None] | None = None) -> str: ...


class OllamaLLM:
    def __init__(self, cfg: Config, should_stop: Callable[[], bool] | None = None):
        self.tracer = Tracer()
        self.cfg = cfg
        self.should_stop = should_stop or (lambda: False)
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
        self.last_truncated = False      # True when the last streamed reply stopped because it hit num_predict

    # ------------------------------------------------------------------ health
    def _get(self, path: str, timeout: float = 5.0) -> dict:
        with urllib.request.urlopen(self.cfg.host + path, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def health(self) -> dict:
        try:
            tags = self._get("/api/tags")
            installed = [m["name"] for m in tags.get("models", [])]
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"Cannot reach Ollama at {self.cfg.host}: {e}", "installed": [], "loaded": [], "missing": []}
        try:
            ps = self._get("/api/ps").get("models", [])
            loaded = [m["name"] for m in ps]
            placement = [{"name": m["name"], "size_gb": round(m.get("size", 0) / 1e9, 1),
                          "gpu_pct": round(100 * m.get("size_vram", 0) / m["size"]) if m.get("size") else None,
                          "context": m.get("context_length")} for m in ps]
        except Exception:  # noqa: BLE001
            loaded, placement = [], []
        needed = sorted({rc.model for rc in self.cfg.roles.values()})
        missing = [m for m in needed if m not in installed and f"{m}:latest" not in installed]
        return {"ok": not missing, "installed": installed, "loaded": loaded, "missing": missing, "placement": placement,
                "error": f"Missing models: {', '.join(missing)}. Run: ollama pull {missing[0]}" if missing else None}

    # ------------------------------------------------------------------ chat
    def chat(self, role: str, messages: list[dict], schema: dict | None = None,
             on_token: Callable[[str, str], None] | None = None) -> str:
        rc: RoleCfg = self.cfg.roles[role]
        attempts = 3
        last: Exception | None = None
        think = rc.think
        for i in range(attempts):
            try:
                text = self._stream(rc, messages, schema, think, on_token, role=role, attempt=i + 1)
                if not text.strip() and think:          # some builds return empty content when think+format
                    think = False
                    continue
                if not text.strip():
                    raise LLMError("Model returned an empty response")
                return text
            except Cancelled:
                raise
            except (urllib.error.URLError, ConnectionError, TimeoutError, LLMError) as e:
                last = e
                time.sleep(min(2 ** i, 8))
        raise LLMError(f"Ollama request failed after {attempts} attempts: {last}")

    def chat_stream(self, role: str, messages: list[dict],
                    on_token: Callable[[str, str], None] | None = None, think: bool | None = None) -> tuple[str, bool]:
        """Free-text streaming for conversation replies. Returns (text, truncated).

        Unlike chat(), a failure after the first visible token is NOT retried (that would repeat text on screen);
        it is raised so the caller can keep what was already shown.
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
                    if use_think:       # the thinking phase used the whole budget: answer again without it
                        use_think = False
                        continue
                    raise LLMError("Model returned an empty response")
                return text, self.last_truncated
            except Cancelled:
                raise
            except (urllib.error.URLError, ConnectionError, TimeoutError, LLMError) as e:
                last = e
                if shown[0]:
                    raise LLMError(f"The reply was interrupted: {e}") from e
                time.sleep(min(2 ** i, 8))
        raise LLMError(f"Ollama request failed after {attempts} attempts: {last}")

    def _stream(self, rc: RoleCfg, messages, schema, think: bool, on_token, mark_truncation: bool = True,
                role: str = "", attempt: int = 1) -> str:
        body = {
            "model": rc.model, "messages": messages, "stream": True, "think": think,
            "keep_alive": "30m",
            "options": {"temperature": rc.temperature, "num_ctx": rc.num_ctx, "num_predict": rc.num_predict,
                        "seed": 7, "repeat_penalty": 1.05},
        }
        if schema:
            body["format"] = schema
        req = urllib.request.Request(self.cfg.host + "/api/chat", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        out: list[str] = []
        t0 = time.time()
        self.last_truncated = False
        est = sum(est_tokens(m.get("content") or "") for m in messages)
        sp = self.tracer.open(f"{role or 'model'} call", "llm", agent=role or "llm", model=rc.model, num_ctx=rc.num_ctx,
                              num_predict=rc.num_predict, think=think, structured=bool(schema), attempt=attempt,
                              est_prompt_tokens=est, prompt=preview_messages(messages), live="waiting")
        first, n_c, n_t, last_u, done_chunk = None, 0, 0, 0.0, None
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.limits.llm_idle_timeout) as resp:
                for raw in resp:
                    if self.should_stop():
                        raise Cancelled()
                    if not raw.strip():
                        continue
                    chunk = json.loads(raw.decode("utf-8"))
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
                        if chunk.get("done_reason") == "length" and not schema:
                            self.last_truncated = True
                            if mark_truncation:
                                out.append("\n\n[[TRUNCATED]]")
        except BaseException as e:
            sp.close(error=f"{type(e).__name__}: {str(e)[:200]}", cancelled=isinstance(e, Cancelled),
                     wall_s=round(time.time() - t0, 3), content_chunks=n_c, thinking_chunks=n_t)
            raise
        text = "".join(out)
        shown = text.replace("\n\n[[TRUNCATED]]", "")
        sp.close(response=shown[:1500], response_chars=len(shown), content_chunks=n_c, thinking_chunks=n_t,
                 truncated=self.last_truncated, **ollama_metrics(done_chunk or {}, time.time() - t0, est))
        return text

    # ------------------------------------------------------------------ JSON helper
    def chat_json(self, role: str, messages: list[dict], schema: dict,
                  on_token=None, retries: int = 2) -> dict:
        msgs = list(messages)
        last_err = ""
        for _ in range(retries + 1):
            text = self.chat(role, msgs, schema, on_token)
            try:
                data = parse_json(text)
                if isinstance(data, dict):
                    return data
                last_err = "top-level value must be a JSON object"
            except ValueError as e:
                last_err = str(e)
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