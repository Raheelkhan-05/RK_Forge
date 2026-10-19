"""Real-time tracing. Every span is measured; nothing is simulated.

A trace = one user message (or one task run). Spans nest: step -> llm call. Events go to a sink
(chat: the SSE hub, runs: the event log), so the UI sees them as they happen.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager


def _id() -> str:
    return uuid.uuid4().hex[:8]


def preview_messages(messages: list[dict], head: int = 360) -> list[dict]:
    out = []
    for i, m in enumerate(messages):
        c = m.get("content") or ""
        out.append({"role": m.get("role"), "chars": len(c), "head": c[: 700 if i == len(messages) - 1 else head]})
    return out


def ollama_metrics(chunk: dict, wall_s: float, est_prompt: int) -> dict:
    """Turn Ollama's own final-chunk numbers (nanoseconds) into a breakdown and a plain-language diagnosis."""
    ns = lambda k: (chunk.get(k) or 0) / 1e9  # noqa: E731
    total, load, pre, gen = ns("total_duration"), ns("load_duration"), ns("prompt_eval_duration"), ns("eval_duration")
    pt, et = chunk.get("prompt_eval_count", 0) or 0, chunk.get("eval_count", 0) or 0
    queue = max(0.0, wall_s - total) if total else 0.0
    parts = {"waiting for Ollama (it was busy with another request)": queue, "loading the model into memory": load,
             "reading the prompt": max(0.0, pre), "writing the answer": gen}
    main = max(parts, key=parts.get)
    ptps, gtps = (round(pt / pre, 1) if pre else None), (round(et / gen, 1) if gen else None)
    if parts[main] <= 0:
        hint = "Ollama did not report timings for this call."
    elif main.startswith("waiting"):
        hint = "Another request (usually a background job or an earlier call) was still running in Ollama when this was sent."
    elif main.startswith("loading"):
        hint = "The model was not resident in memory and had to be loaded first (keep_alive expired, or another model evicted it)."
    elif main.startswith("reading"):
        hint = f"The prompt was {pt} tokens, read at {ptps} tok/s. Shorter prompts, a smaller context or more GPU layers speed this up."
    else:
        hint = f"Writing {et} tokens at {gtps} tok/s. That is this machine's generation speed for this model."
    if est_prompt and pt and pt < 0.5 * est_prompt:
        hint += " Part of the prompt was reused from Ollama's cache."
    return {"wall_s": round(wall_s, 3), "queue_s": round(queue, 3), "load_s": round(load, 3), "prefill_s": round(pre, 3),
            "gen_s": round(gen, 3), "prompt_tokens": pt, "gen_tokens": et, "prefill_tps": ptps, "gen_tps": gtps,
            "bottleneck": main, "bottleneck_share": round(parts[main] / wall_s, 2) if wall_s else 0, "hint": hint}


class _Null:
    id = None

    def update(self, **kv): pass
    def set(self, **kv): pass
    def close(self, **kv): pass


NULL_SPAN = _Null()


class Span:
    def __init__(self, tr: "Tracer", name, kind, agent, parent, trace, scope, meta):
        self.tr, self.id, self.name, self.kind, self.agent = tr, _id(), name, kind, agent
        self.parent, self.trace, self.scope, self.meta = parent, trace or self.id, scope, dict(meta)
        self.t0, self.done = time.time(), False

    def set(self, **kv) -> None:                 # merged into the final "end" event
        self.meta.update(kv)

    def update(self, **kv) -> None:              # streamed immediately (live tokens, state)
        self.meta.update(kv)
        self.tr._emit("update", self, kv)

    def close(self, **kv) -> None:
        if self.done:
            return
        self.done = True
        self.meta.update(kv)
        st = self.tr._stack()
        if self in st:
            st.remove(self)
        self.tr._emit("end", self, {**self.meta, "dur": round(time.time() - self.t0, 3)})


class Tracer:
    def __init__(self, sink=None, keep: int = 1500):
        self.sink, self.keep = sink, keep
        self.tl = threading.local()
        self.history: dict[str, deque] = {}
        self.lock = threading.Lock()

    def _stack(self) -> list:
        if not hasattr(self.tl, "stack"):
            self.tl.stack = []
        return self.tl.stack

    def _emit(self, phase: str, sp: Span, extra: dict) -> None:
        ev = {"phase": phase, "id": sp.id, "parent": sp.parent, "trace": sp.trace, "name": sp.name, "kind": sp.kind,
              "agent": sp.agent, "t": time.time(), **extra}
        if phase != "update":
            with self.lock:
                self.history.setdefault(sp.scope, deque(maxlen=self.keep)).append(ev)
        if self.sink:
            try:
                self.sink(sp.scope, ev)
            except Exception:  # noqa: BLE001
                pass

    def open(self, name: str, kind: str = "step", agent: str | None = None, scope: str | None = None, **meta):
        scope = scope or getattr(self.tl, "scope", None)
        if not scope:
            return NULL_SPAN
        st = self._stack()
        parent = st[-1] if st else None
        sp = Span(self, name, kind, agent, parent.id if parent else None,
                  st[0].trace if st else None, scope, meta)
        st.append(sp)
        self._emit("start", sp, dict(sp.meta))
        return sp

    @contextmanager
    def span(self, name: str, kind: str = "step", agent: str | None = None, **meta):
        sp = self.open(name, kind, agent, **meta)
        try:
            yield sp
        except BaseException as e:
            sp.set(error=f"{type(e).__name__}: {str(e)[:200]}")
            raise
        finally:
            sp.close()

    def note(self, name: str, agent: str | None = None, **meta) -> None:
        self.open(name, "tool", agent, **meta).close()

    def begin(self, scope: str, name: str, **meta):
        self.end()
        self.tl.scope = scope
        return self.open(name, "trace", None, scope=scope, **meta)

    def end(self) -> None:
        for sp in reversed(list(self._stack())):
            sp.close()
        self.tl.scope = None

    def recent(self, scope: str) -> list[dict]:
        with self.lock:
            return list(self.history.get(scope, []))

    def forget(self, scope: str) -> None:
        with self.lock:
            self.history.pop(scope, None)