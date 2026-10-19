"""Chat layer: talk first, build only when the intent is clear.

A message goes through one pipeline:

  1. wording signals (intent.py) + a small JSON "router" call decide: discuss / task / confirm / revise / cancel
  2. a controller (this file, not the model) turns that into an action and applies safety rules:
       - a task never starts from hedging wording, a pure question, or a low-confidence reading
       - unclear tasks produce clarifying questions instead of a run
       - clear tasks produce a brief card the user confirms (or start directly when the user said "go ahead")
  3. discussion replies are grounded in: recalled facts, the running summary, retrieved older messages,
     the newest messages, the real code (when the question is about it) and the state of any task

Background jobs (title, rolling summary, fact extraction) run only when no reply is waiting, and are
pre-empted the moment the user sends something.
"""
from __future__ import annotations

import logging
import re
import threading
import json
import time
from collections import deque
from contextlib import contextmanager

from .indexer import query_terms
from .intent import Signals, analyze, code_names, looks_like_code_talk
from .telemetry import Tracer
from .llm import Cancelled, LLMError
from .memory import CHAT_KINDS, Memory
from .workspace import TxError, norm_rel

log = logging.getLogger("forge.chat")

TERMINAL_RUN = {"done", "failed", "cancelled", "rejected", "conflict", "undone", "interrupted"}
ATTENTION_RUN = {"awaiting_user", "awaiting_approval", "blocked"}

# ------------------------------------------------------------------------------------------------ schemas
S_STR = {"type": "string"}
S_STRS = {"type": "array", "items": S_STR}
ROUTER_SCHEMA = {"type": "object", "properties": {
    "intent": {"type": "string", "enum": ["discuss", "task", "confirm", "revise", "cancel"]},
    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    "execute_now": {"type": "boolean"},
    "blocking_questions": S_STRS, "needs_code": {"type": "boolean"}, "code_query": S_STR, "reason": S_STR},
    "required": ["intent", "confidence", "execute_now"]}
BRIEF_SCHEMA = {"type": "object", "properties": {
    "title": S_STR, "request": S_STR, "keep_working": S_STRS, "assumptions": S_STRS, "open_questions": S_STRS},
    "required": ["title", "request"]}
FACTS_SCHEMA = {"type": "object", "properties": {
    "facts": {"type": "array", "items": {"type": "object", "properties": {
        "kind": {"type": "string", "enum": list(CHAT_KINDS)}, "text": S_STR, "supersedes": S_STRS}, "required": ["kind", "text"]}},
    "forget": S_STRS}, "required": ["facts"]}
TITLE_SCHEMA = {"type": "object", "properties": {"title": S_STR}, "required": ["title"]}
DIGEST_SCHEMA = {"type": "object", "properties": {"summary": S_STR}, "required": ["summary"]}

# ------------------------------------------------------------------------------------------------ prompts
CHAT_SYS = """You are Forge, a senior software engineer who talks with the user about their own codebase before any code is changed. You can discuss freely: explain how things work, compare options, review ideas, answer follow-up and counter-questions, and help the user work out what they actually want.

How to behave:
- Think with the user. Be direct, specific and honest. Say when you are unsure. Never invent files, functions, APIs or behaviour: your facts about the code are <code_evidence> and the conversation. If they do not cover something, say what you would need to look at.
- Use <project_memory>, <conversation_summary>, <earlier_excerpts> and the recent messages. When the user says "that", "the second option", "your last point", or asks about something from earlier, resolve it from those before answering. If you still cannot tell what they mean, ask one short question.
- You cannot edit files in this chat. Code only changes when a task runs, and a task only starts after the user confirms. Never claim you changed, ran or applied anything unless a "[Task ...]" result in the conversation says so.
- If the discussion has reached a concrete change, say so in one sentence and tell the user they can press "Make this a task" or say "go ahead". Do not push for a task while they are still exploring.
- If the user asks about a finished task, use the task result and <last_change_diff> to explain what changed and why, and point out risks honestly.
- Match the user's language and tone (English, Hindi, Hinglish...). Be concise: short answers for simple questions, structured answers (short lists, small code blocks) for complex ones. No filler, do not repeat the question.
- If the user asks you to remember or forget something, confirm in one short sentence; it is stored automatically."""

ROUTER_SYS = """You are the intake step of a coding assistant. The assistant can (a) discuss freely with the user (explain code, compare options, answer follow-ups, brainstorm) and (b) run a coding task that edits the repository.
Decide what the user's LATEST message needs. Read the recent conversation, the pending task brief (if any), the task state and the wording signals.

intent:
- "discuss": questions, explanations, opinions, brainstorming, comparisons, follow-ups about earlier answers or finished work, small talk, or ideas the user is still exploring.
- "task": the user wants a change made to the code (feature, fix, refactor, rename...), now or as the conclusion of the discussion.
- "confirm": the user approves the pending task brief as it stands.
- "revise": the user changes the pending brief (adds, removes or changes requirements, or answers its open questions).
- "cancel": the user drops the pending brief.

Rules:
- Exploring ("what if", "should we", "how would", "thinking about") or asking for an opinion is "discuss", never "task".
- "task" needs a concrete outcome in the code. If the outcome, scope or a decisive rule is unclear, list the missing points in blocking_questions (max 3, specific, only those that change the implementation) and lower the confidence.
- execute_now is true ONLY when the user explicitly tells you to carry it out now ("do it", "go ahead", "start", "implement this"), not when they merely describe what they want.
- confidence is how sure you are about BOTH the intent and whether the request is complete enough to act on.
- needs_code is true when a good answer or brief needs the real repository (specific files, functions or behaviour). code_query: 2-6 keywords or names to look for.
- Without a pending brief never answer confirm, revise or cancel.
- When unsure between discuss and task, choose discuss."""

BRIEF_SYS = """You write the task brief for a coding pipeline from a conversation. A separate agent will plan and implement it without seeing this chat.
- request: a complete, self-contained description (2-10 sentences) of exactly what to build or change, including every decision and constraint the user stated or agreed to. Mention file or component names only if the user or the code evidence gave them.
- keep_working: existing behaviour that must not break (only if the user said so or it is obvious from the request).
- assumptions: choices you had to make that the user did not state (short).
- open_questions: things only the user can decide that would change the implementation (max 3, else empty).
- title: at most 8 words.
Never invent requirements, file names or APIs. If a previous brief is given, apply the user's latest message to it."""

FACTS_SYS = """You maintain the long-term memory of a coding assistant. From the NEW MESSAGES extract durable facts worth remembering in future conversations.
Keep: what the project is and how it is built; decisions the user made and why; the user's goals; naming and code conventions the user states; the user's preferences about how you should explain or answer; constraints (hardware, deadlines, tools).
Rules:
- Only facts the USER said or clearly agreed to. Never store your own suggestions that the user did not accept.
- One fact per item, self-contained, under 200 characters, specific (include file, component or feature names).
- No secrets (keys, passwords, tokens), no one-off details, no chit-chat, nothing already in KNOWN FACTS.
- If a new message corrects, replaces or cancels a known fact, give the new fact and put the old fact's id in "supersedes". If the user asks you to forget something, put that fact's id in "forget".
- If the user explicitly asks you to remember something, always store it.
- kind: preference (how to talk or explain), decision, project (what exists / how it works), goal, convention (code rules), fact (anything else).
At most 5 facts. If nothing is worth saving return {"facts": [], "forget": []}."""

SUMMARY_SYS = """You keep a running summary of a conversation between a user and a coding assistant. Update the previous summary with the new messages.
Keep: the user's goals, decisions made and why, constraints, names of files/components/functions discussed, options that were compared and what was chosen, open questions, tasks that were started and their results, and the user's preferences. Drop greetings and filler. Plain text, short lines, at most 280 words."""

TITLE_SYS = "Write a short title (2-6 words, no quotes, no trailing punctuation) for this conversation, naming its main topic."

FINISH_HINT = ('answer = short notes of the relevant facts found in the code; evidence = list of "path:startLine-endLine"; '
               'confidence = high only when the code states it explicitly.')

SECRET_RE = re.compile(r"(?i)(?:api[_-]?key|secret|passw(?:or)?d|token|bearer|private[_ ]key)\s*[:=]|sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{12,}|eyJ[A-Za-z0-9_-]{20,}")
FILLER = re.compile(r"(?i)^(?:(?:hey|hi|hello|hii|ok(?:ay)?|so|please|pls|can you|could you|would you|i want to|i need to|i would like to|let'?s|help me(?: to)?|tell me|explain(?: to me)?)\b[\s,:-]*)+")


# ------------------------------------------------------------------------------------------------ helpers
def clip(text: str | None, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:max(0, n - 1)].rstrip() + "…"


def clip_mid(text: str | None, n: int) -> str:
    text = text or ""
    if len(text) <= n:
        return text
    half = max(1, (n - 30) // 2)
    return text[:half].rstrip() + "\n… (middle omitted) …\n" + text[-half:].lstrip()


def one_line(text: str | None, n: int) -> str:
    return clip(re.sub(r"\s+", " ", text or ""), n)


def strip_think(text: str) -> str:
    return re.sub(r"(?s)<think>.*?</think>\s*", "", text or "").strip()


def heuristic_title(text: str) -> str:
    t = FILLER.sub("", one_line(text, 200)).strip(" .?!,:;-\"'")
    if not t:
        t = one_line(text, 60).strip(" .?!,:;-\"'")
    if not t:
        return "New chat"
    t = clip(" ".join(t.split()[:7]), 52)
    return t[:1].upper() + t[1:]


def clean_title(raw: str) -> str:
    t = re.sub(r"\s+", " ", (raw or "")).strip(" \t\"'`.,:;-")
    t = clip(" ".join(t.split()[:8]), 60)
    return t[:1].upper() + t[1:] if t else ""


def brief_text(b: dict) -> str:
    lines = [f"Title: {b.get('title', '')}", f"Task: {b.get('request', '')}"]
    if b.get("keep_working"):
        lines.append("Must keep working: " + "; ".join(b["keep_working"]))
    if b.get("assumptions"):
        lines.append("Assumptions: " + "; ".join(b["assumptions"]))
    if b.get("open_questions"):
        lines.append("Open questions: " + "; ".join(b["open_questions"]))
    return "\n".join(lines)


def brief_to_request(b: dict) -> str:
    parts = [(b.get("request") or "").strip()]
    if b.get("keep_working"):
        parts.append("Must keep working / must not change: " + "; ".join(b["keep_working"]) + ".")
    if b.get("assumptions"):
        parts.append("Decisions already made with the user: " + "; ".join(b["assumptions"]) + ".")
    return "\n\n".join(p for p in parts if p)


def clean_list(items, n: int, each: int) -> list[str]:
    out = []
    for x in items or []:
        s = one_line(str(x), each)
        if s and s not in out:
            out.append(s)
    return out[:n]


def compose_result(run: dict) -> str:
    st = run.get("status")
    files = run.get("files") or []
    err = (run.get("error") or "").strip()
    if st == "done":
        out = [f"**Done.** The task is applied to your project ({len(files)} file{'' if len(files) == 1 else 's'} changed)."]
        if run.get("summary"):
            out += ["", run["summary"]]
        steps = [t for t in run.get("tasks", []) if t["status"] == "accepted"]
        if steps:
            out += ["", "**What was done**"]
            out += [f"- {t['title']}" + (f": {clip(t['summary'], 160)}" if t.get("summary") else "") for t in steps[:12]]
        if files:
            out += ["", "**Files changed**"] + [f"- `{p}`" for p in files[:20]]
        out += ["", "Ask me to explain any part of it, review it, or tell me what to adjust."]
        return "\n".join(out)
    if st == "failed":
        return ("The task stopped and **your files were not changed**.\n\n" + (clip(err, 700) or "No reason was recorded.")
                + "\n\nTell me what to do differently, or ask me why it failed.")
    if st == "cancelled":
        return "The task was cancelled. Nothing was changed in your project."
    if st == "rejected":
        return "You discarded the result. Nothing was changed in your project."
    if st == "conflict":
        return ("Nothing was written: files changed on disk while the task was running. "
                "Tell me to run it again and it will start from your latest code.")
    if st == "undone":
        return "The changes from that task were reverted. Your files are back to how they were before it."
    if st == "interrupted":
        return "The task was interrupted (the server restarted). Nothing was changed. You can resume it from the task card."
    return f"The task ended with status {st}."


def friendly_error(e: Exception) -> str:
    s = str(e)
    if "Ollama request failed" in s or "Cannot reach" in s:
        return ("I can't reach the model right now. Make sure Ollama is running (`ollama serve`) and the model is pulled, "
                "then press Retry.\n\nDetails: " + clip(s, 240))
    return "The model did not answer properly: " + clip(s, 300)


class RepoView:
    """Read-only view of the real repository with the same read/exists surface the context builder expects."""

    def __init__(self, ix):
        self.ix = ix

    def read(self, rel: str) -> str | None:
        try:
            return self.ix.read(norm_rel(rel))
        except TxError:
            return None

    def exists(self, rel: str) -> bool:
        try:
            return (self.ix.root / norm_rel(rel)).is_file()
        except TxError:
            return False


class ChatHub:
    """Thread-safe publish/subscribe with a per-chat sequence number (lets a reconnecting client skip what it has)."""

    def __init__(self):
        self.lock = threading.RLock()
        self.seq: dict[str, int] = {}
        self.listeners: dict[str, list] = {}

    def subscribe(self, cid: str, fn):
        with self.lock:
            self.listeners.setdefault(cid, []).append(fn)

        def unsubscribe():
            with self.lock:
                if fn in self.listeners.get(cid, []):
                    self.listeners[cid].remove(fn)
        return unsubscribe

    def publish(self, cid: str, type_: str, payload: dict) -> int:
        with self.lock:
            n = self.seq.get(cid, 0) + 1
            self.seq[cid] = n
            fns = list(self.listeners.get(cid, []))
        ev = {"seq": n, "type": type_, "payload": payload}
        for fn in fns:
            try:
                fn(ev)
            except Exception:  # noqa: BLE001
                pass
        return n


# ------------------------------------------------------------------------------------------------ service
class ChatService:
    def __init__(self, engine, start_worker: bool = True):
        self.engine = engine
        self.cfg = engine.cfg
        self.mem: Memory = engine.mem
        self.hub = ChatHub()
        self.live: dict[str, dict] = {}
        self.slock = threading.RLock()
        self.cv = threading.Condition()
        self.fg: deque = deque()
        self.bg: deque = deque()
        self._bg_keys: set[tuple] = set()
        self.closed = False
        self.current: str | None = None
        self._stop_turn = threading.Event()
        self.llm_fg = engine.llm_factory(lambda: self._stop_turn.is_set() or self.closed)
        self.llm_bg = engine.llm_factory(lambda: bool(self.fg) or self.closed)
        self.tracer = Tracer(lambda scope, ev: self.hub.publish(scope, "trace", ev))
        self.llm_fg.tracer = self.tracer
        self.llm_bg.tracer = self.tracer
        self.last_active = time.time()
        self._run_conv: dict[str, str | None] = {}
        self._tools = None
        self.mem.listeners.append(self._on_event)
        try:
            self.reconcile()
        except Exception:  # noqa: BLE001
            log.exception("reconcile failed")
        self.worker: threading.Thread | None = None
        if start_worker:
            self.worker = threading.Thread(target=self._loop, daemon=True, name="forge-chat")
            self.worker.start()

    def close(self) -> None:
        self.closed = True
        with self.cv:
            self.cv.notify_all()
        if self._on_event in self.mem.listeners:
            self.mem.listeners.remove(self._on_event)

    # ------------------------------------------------------------------ worker
    def _loop(self) -> None:
        while not self.closed:
            job = self._next()
            if job is not None:
                self._run_job(job)

    BG_IDLE = 6.0

    def _next(self):
        with self.cv:
            while True:
                if self.closed:
                    return None
                if self.fg:
                    return self.fg.popleft()
                if self.bg and time.time() - self.last_active >= self.BG_IDLE:
                    job = self.bg.popleft()
                    self._bg_keys.discard((job[0], job[1]))
                    return job
                self.cv.wait(1.0)

    def _run_job(self, job) -> None:
        kind, cid, arg, extra = job
        if kind != "turn":
            self.tracer.begin(cid, f"Background: {kind}", job=kind)
        try:
            if kind == "turn":
                self._turn(cid, arg, retry=bool(extra))
            elif kind == "title":
                self._job_title(cid)
            elif kind == "fold":
                self._job_fold(cid)
            elif kind == "extract":
                self._job_extract(cid)
        except Cancelled:
            if kind != "turn" and (extra or 0) < 3:
                self.schedule(kind, cid, (extra or 0) + 1)
        except Exception:  # noqa: BLE001
            log.exception("chat job %s failed", kind)
        finally:
            if kind != "turn":
                self.tracer.end()

    def schedule(self, kind: str, cid: str, tries: int = 0) -> None:
        with self.cv:
            key = (kind, cid)
            if key in self._bg_keys:
                return
            self._bg_keys.add(key)
            self.bg.append((kind, cid, None, tries))
            self.cv.notify()

    def drain(self, max_jobs: int = 200) -> int:
        """Run queued jobs on the calling thread (used by tests and by `forge` tooling)."""
        n = 0
        while n < max_jobs:
            with self.cv:
                if self.fg:
                    job = self.fg.popleft()
                elif self.bg:
                    job = self.bg.popleft()
                    self._bg_keys.discard((job[0], job[1]))
                else:
                    return n
            self._run_job(job)
            n += 1
        return n

    # ------------------------------------------------------------------ events to the UI
    def pub(self, cid: str, type_: str, **payload) -> int:
        with self.hub.lock:
            live = self.live.get(cid)
            if type_ == "turn_start":
                self.live[cid] = {"text": "", "phase": payload.get("phase", ""), "reply_to": payload.get("reply_to"), "activity": []}
            elif live is not None:
                if type_ == "phase":
                    live["phase"] = payload.get("label", "")
                elif type_ == "delta":
                    live["text"] += payload.get("text", "")
                elif type_ == "activity":
                    live["activity"].append(payload.get("label", ""))
            if type_ == "turn_end":
                self.live.pop(cid, None)
            return self.hub.publish(cid, type_, payload)

    def _phase(self, cid: str, label: str) -> None:
        self.pub(cid, "phase", label=label)

    # ------------------------------------------------------------------ public API
    def public_conv(self, c: dict) -> dict:
        rid = c["state"].get("active_run")
        st = self.mem.run_status(rid) if rid else None
        return {"id": c["id"], "title": c["title"], "locked": c["title_locked"], "created": c["created"], "updated": c["updated"],
                "preview": c.get("preview", ""), "messages": c.get("count", 0), "run": rid if st and st not in TERMINAL_RUN else None,
                "running": bool(st and st not in TERMINAL_RUN), "attention": st in ATTENTION_RUN}

    def new_chat(self, title: str = "New chat") -> dict:
        return self.public_conv(self.mem.create_conv(clip(title, 80) or "New chat"))

    def list_chats(self, q: str | None = None) -> list[dict]:
        rows = self.mem.search_convs(q) if q else self.mem.list_convs()
        return [self.public_conv(c) for c in rows]

    def snapshot(self, cid: str) -> dict | None:
        conv = self.mem.get_conv(cid)
        if not conv:
            return None
        with self.hub.lock:                    # seq and live text are captured together, so no delta is lost or doubled
            seq = self.hub.seq.get(cid, 0)
            live = dict(self.live[cid]) if cid in self.live else None
            if live:
                live["activity"] = list(live["activity"])
            busy = self.current == cid
            queued = sum(1 for j in self.fg if j[1] == cid)
        msgs = self.mem.recent_msgs(cid, 200)
        return {"conv": self.public_conv(conv), "messages": msgs, "has_more": bool(msgs) and self.mem.has_older(cid, msgs[0]["id"]), "trace": self.tracer.recent(cid),
                "max_id": msgs[-1]["id"] if msgs else 0, "live": live, "seq": seq, "busy": busy, "queued": queued}

    def older(self, cid: str, before: int, limit: int = 100) -> dict:
        msgs = self.mem.recent_msgs(cid, max(1, min(limit, 300)), before=before)
        return {"messages": msgs, "has_more": bool(msgs) and self.mem.has_older(cid, msgs[0]["id"])}

    def rename(self, cid: str, title: str) -> dict:
        title = clean_title(title)
        if not title or not self.mem.get_conv(cid):
            raise ValueError("Give the chat a name.")
        self.mem.update_conv(cid, title=title, title_locked=True)
        self.pub(cid, "title", title=title)
        return self.public_conv(self.mem.get_conv(cid))

    def delete(self, cid: str) -> None:
        if self.current == cid:
            self._stop_turn.set()
        with self.cv:
            self.fg = deque(j for j in self.fg if j[1] != cid)
            self.bg = deque(j for j in self.bg if j[1] != cid)
            self._bg_keys = {k for k in self._bg_keys if k[1] != cid}
        self.mem.delete_conv(cid)
        self.tracer.forget(cid)
        self.pub(cid, "deleted")

    def stop(self, cid: str) -> None:
        with self.cv:
            self.fg = deque(j for j in self.fg if j[1] != cid)
        if self.current == cid:
            self._stop_turn.set()
        self.pub(cid, "queued", count=0)

    def post(self, cid: str, text: str, mode: str = "chat") -> dict:
        text = (text or "").strip()
        if not text:
            raise ValueError("Write a message first.")
        if len(text) > 20000:
            raise ValueError("That message is too long (limit 20,000 characters).")
        conv = self.mem.get_conv(cid)
        if not conv:
            raise KeyError("chat not found")
        self.last_active = time.time()
        mode = mode if mode in ("chat", "task", "brief") else "chat"
        low = text.lower()
        if low.startswith("/task ") and len(text) > 6:
            mode, text = "task", text[6:].strip()
        elif low.startswith("/remember ") and len(text) > 10:
            return self._slash_remember(cid, conv, text)
        elif low.startswith("/forget ") and len(text) > 8:
            return self._slash_forget(cid, conv, text)
        m = self.mem.add_msg(cid, "user", "text", text, {"mode": mode})
        self.pub(cid, "message", message=m)
        if conv["title"] == "New chat" and not conv["title_locked"]:
            title = heuristic_title(text)
            self.mem.update_conv(cid, title=title)
            self.pub(cid, "title", title=title)
        with self.cv:
            self.fg.append(("turn", cid, m["id"], False))
            queued = sum(1 for j in self.fg if j[1] == cid)
            self.cv.notify()
        self.pub(cid, "queued", count=queued + (1 if self.current == cid else 0))
        return m

    def retry(self, cid: str, mid: int) -> None:
        m = self.mem.get_msg(mid)
        if not m or m["conv_id"] != cid or m["role"] != "user":
            raise ValueError("Nothing to retry.")
        with self.cv:
            self.fg.append(("turn", cid, mid, True))
            self.cv.notify()
        self.pub(cid, "queued", count=sum(1 for j in self.fg if j[1] == cid))

    def brief_action(self, cid: str, action: str, request: str | None = None) -> dict:
        with self.slock:
            conv = self.mem.get_conv(cid)
            brief = (conv or {}).get("state", {}).get("brief")
            if not conv or not brief:
                raise ValueError("There is no task brief waiting in this chat.")
            if action == "cancel":
                self._set_brief_status(brief, "dismissed")
                self.mem.patch_state(cid, brief=None)
                return {"ok": True}
            if action != "start":
                raise ValueError("Unknown action.")
            if request and request.strip():
                brief = dict(brief, request=request.strip()[:6000])
            rid = self.start_task(cid, brief)
            return {"ok": True, "run": rid}

    # ------------------------------------------------------------------ slash commands
    def _slash_remember(self, cid: str, conv: dict, text: str) -> dict:
        fact = " ".join(text[10:].split())[:300]
        m = self.mem.add_msg(cid, "user", "text", text, {"mode": "chat"})
        self.pub(cid, "message", message=m)
        if SECRET_RE.search(fact):
            reply = "I did not save that: it looks like it contains a secret (key, token or password)."
        else:
            self.mem.add_fact("fact", fact, "user")
            reply = f"Saved to memory: {fact}"
            self.pub(cid, "memory")
        self._say(cid, reply, m["id"])
        self._after_turn(cid)
        return m

    def _slash_forget(self, cid: str, conv: dict, text: str) -> dict:
        words = set(query_terms(text[8:], 20))
        m = self.mem.add_msg(cid, "user", "text", text, {"mode": "chat"})
        self.pub(cid, "message", message=m)
        scored = []
        for f in self.mem.list_facts():
            if f["source"].startswith(("chat", "user")):
                ov = len(words & set(query_terms(f["text"], 40)))
                if ov:
                    scored.append((ov, f))
        scored.sort(key=lambda t: -t[0])
        if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            self.mem.forget(scored[0][1]["id"])
            reply = f"Forgot: {scored[0][1]['text']}"
            self.pub(cid, "memory")
        elif scored:
            reply = "Several memories match. Be more specific, or remove one in the Memory tab:\n" + "\n".join(f"- {f['text']}" for _, f in scored[:4])
        else:
            reply = "I found nothing in memory that matches. You can browse and remove items in the Memory tab."
        self._say(cid, reply, m["id"])
        return m

    # ------------------------------------------------------------------ messages
    def _say(self, cid: str, text: str, reply_to: int | None = None, kind: str = "text", **meta) -> dict | None:
        if not self.mem.get_conv(cid):
            return None
        m = self.mem.add_msg(cid, "assistant", kind, text, {"reply_to": reply_to, **meta})
        self.pub(cid, "message", message=m)
        return m

    def _notice(self, cid: str, text: str, error: bool = False, retry_of: int | None = None) -> None:
        self._say(cid, text, retry_of, kind="notice", error=error, retry_of=retry_of)

    def _set_brief_status(self, brief: dict, status: str) -> None:
        mid = brief.get("msg_id")
        if not mid:
            return
        m = self.mem.get_msg(mid)
        if m:
            meta = dict(m["meta"], status=status)
            m2 = self.mem.update_msg(mid, meta=meta)
            if m2:
                self.pub(m["conv_id"], "message_update", message=m2)

    # ------------------------------------------------------------------ one turn
    def _turn(self, cid: str, mid: int, retry: bool = False) -> None:
        conv = self.mem.get_conv(cid)
        msg = self.mem.get_msg(mid)
        if not conv or not msg:
            return
        self._stop_turn.clear()
        self.current = cid
        self.tracer.begin(cid, "Message", mid=mid, text=clip(msg["content"], 140), retry=retry)
        pending = conv["state"].get("brief")
        with self.tracer.span("Wording signals", "step", agent="signals") as sp:
            sig = analyze(msg["content"])
            sp.set(response=sig.describe(bool(pending)))
        self.pub(cid, "turn_start", reply_to=mid, phase="Reading your message")
        try:
            if retry:
                for n in self.mem.find_msgs_conv(cid, "notice", 20):
                    if n["meta"].get("retry_of") == mid:
                        self.mem.delete_msg(n["id"])
                        self.pub(cid, "message_delete", id=n["id"])
            self._run_turn(cid, conv, msg, sig, pending)
        except Cancelled:
            pass
        except LLMError as e:
            self._notice(cid, friendly_error(e), error=True, retry_of=mid)
        except Exception as e:  # noqa: BLE001
            log.exception("turn failed")
            self._notice(cid, "Something went wrong while answering: " + clip(str(e), 240), error=True, retry_of=mid)
        finally:
            self.tracer.end()
            self.last_active = time.time()
            self.current = None
            self.pub(cid, "turn_end", reply_to=mid)
            self.pub(cid, "queued", count=sum(1 for j in self.fg if j[1] == cid))
            self._after_turn(cid)

    def _after_turn(self, cid: str) -> None:
        self.schedule("title", cid)
        self.schedule("extract", cid)
        self.schedule("fold", cid)

    def _run_turn(self, cid: str, conv: dict, msg: dict, sig: Signals, pending: dict | None) -> None:
        mid, text = msg["id"], msg["content"].strip()
        mode = msg["meta"].get("mode", "chat")
        if mode in ("task", "brief"):
            self._phase(cid, "Preparing the task brief")
            brief = self._make_brief(cid, conv, msg, sig, None, [])
            self._post_brief(cid, brief, mid)
            if mode == "task":
                self.start_task(cid, brief)
            return
        if pending and sig.decline:
            self._set_brief_status(pending, "dismissed")
            self.mem.patch_state(cid, brief=None)
            self._say(cid, "Okay, I won't start that. We can keep talking, and tell me whenever you want to adjust it.", mid)
            return
        if pending and sig.confirm:
            self.start_task(cid, pending)
            return
        self._phase(cid, "Understanding your message")
        quick = (not pending and not sig.imperative and not sig.go and sig.words <= 40 and not looks_like_code_talk(text))
        if quick:    # plain conversation: skip the router model call entirely
            route = {"intent": "discuss", "confidence": "high", "execute_now": False, "blocking_questions": [],
                     "needs_code": False, "code_query": ""}
            self.tracer.note("Router skipped", agent="router",
                             response="Plain conversation: no change request, no code names, nothing pending. Went straight to the chat model.")
        else:
            with self.tracer.span("Intent router", "step", agent="router") as sp:
                route = self._route(cid, conv, msg, sig, pending)
                sp.set(response=json.dumps({k: route[k] for k in ("intent", "confidence", "execute_now", "needs_code")}))
            if route["needs_code"] and not looks_like_code_talk(text):
                route["needs_code"] = False        # the model asked to search the repo for chit-chat
        with self.tracer.span("Controller decision", "step", agent="controller") as sp:
            action = self._decide(route, sig, pending)
            sp.set(response=f"intent={route['intent']} confidence={route['confidence']} -> action={action}")
        log.info("chat %s: intent=%s conf=%s -> %s", cid, route["intent"], route["confidence"], action)
        if action == "cancel":
            self._set_brief_status(pending, "dismissed")
            self.mem.patch_state(cid, brief=None)
            self._say(cid, "Okay, dropped. Nothing was started.", mid)
        elif action == "start_pending":
            self.start_task(cid, pending)
        elif action == "ask_confirm":
            self._say(cid, "Just to be sure: should I start this task now? Press **Start task** on the brief, or reply \"go ahead\".", mid)
        elif action == "clarify":
            self._say(cid, self._clarify_text(route), mid)
        elif action in ("propose", "auto_start", "revise"):
            blocks: list[str] = []
            if route.get("needs_code") or code_names(text):
                blocks, _paths, _labels = self._gather(cid, text, route)
            self._phase(cid, "Writing the task brief")
            brief = self._make_brief(cid, conv, msg, sig, pending if action == "revise" else None, blocks)
            self._post_brief(cid, brief, mid)
            go = action == "auto_start" or (action == "revise" and self._may_auto(route, sig))
            if brief["open_questions"]:
                qs = "\n".join(f"{i}. {q}" for i, q in enumerate(brief["open_questions"], 1))
                self._say(cid, "Before this runs, a few things only you can decide:\n\n" + qs
                          + "\n\nAnswer here and I'll update the brief, or press **Start task** to let me use my best judgement.", mid)
            elif go:
                self.start_task(cid, brief)
        else:
            self._discuss(cid, conv, msg, sig, route, pending)

    # ------------------------------------------------------------------ routing + control
    def _route(self, cid: str, conv: dict, msg: dict, sig: Signals, pending: dict | None) -> dict:
        hist = self.mem.recent_msgs(cid, 8, before=msg["id"])
        lines = []
        for m in hist[-6:]:
            if m["meta"].get("error"):
                continue
            who = "user" if m["role"] == "user" else "assistant"
            lines.append(f"{who}: {clip(self._plain(m), 650 if who == 'user' else 450)}")
        parts = []
        if conv.get("summary"):
            parts.append(f"<conversation_summary>\n{clip(conv['summary'], 900)}\n</conversation_summary>")
        if lines:
            parts.append("<recent_messages>\n" + "\n".join(lines) + "\n</recent_messages>")
        if pending:
            parts.append("<pending_task_brief>\n" + clip(brief_text(pending), 900) + "\n</pending_task_brief>")
        state = self._task_state(conv)
        if state:
            parts.append(state)
        parts.append(f"<signals>{sig.describe(bool(pending))}</signals>")
        parts.append(f"<latest_message>\n{clip(msg['content'], 2500)}\n</latest_message>")
        msgs = [{"role": "system", "content": ROUTER_SYS}, {"role": "user", "content": "\n\n".join(parts)}]
        try:
            d = self.llm_fg.chat_json("router", msgs, ROUTER_SCHEMA, None, retries=1)
        except LLMError as e:
            if "valid JSON" not in str(e):
                raise
            d = {"intent": "task" if (sig.imperative and not sig.hedge) else "discuss", "confidence": "low", "execute_now": False}
        intent = d.get("intent") if d.get("intent") in ("discuss", "task", "confirm", "revise", "cancel") else "discuss"
        conf = d.get("confidence") if d.get("confidence") in ("high", "medium", "low") else "low"
        return {"intent": intent, "confidence": conf, "execute_now": bool(d.get("execute_now")),
                "blocking_questions": clean_list(d.get("blocking_questions"), 3, 220),
                "needs_code": bool(d.get("needs_code")), "code_query": one_line(d.get("code_query") or "", 160)}

    def _may_auto(self, route: dict, sig: Signals) -> bool:
        return bool(self.cfg.chat.auto_start and route["execute_now"] and route["confidence"] == "high"
                    and not sig.hedge and not (sig.question and not sig.imperative))

    def _decide(self, route: dict, sig: Signals, pending: dict | None) -> str:
        intent, conf = route["intent"], route["confidence"]
        if intent in ("confirm", "revise", "cancel") and not pending:
            intent = "discuss"
        if intent == "cancel":
            return "cancel"
        if intent == "confirm":
            return "start_pending" if conf != "low" and not (sig.question or sig.hedge) else "ask_confirm"
        if intent == "revise":
            return "revise"
        if intent == "task":
            if route["blocking_questions"] or conf == "low":
                return "clarify"
            return "auto_start" if self._may_auto(route, sig) else "propose"
        return "discuss"

    @staticmethod
    def _clarify_text(route: dict) -> str:
        qs = route["blocking_questions"][:3]
        if qs:
            body = "\n".join(f"{i}. {q}" for i, q in enumerate(qs, 1))
            return ("Before I turn this into a task, I want to get it right. Could you tell me:\n\n" + body
                    + "\n\nAnswer in your own words, or say \"go ahead\" and I'll use my best judgement.")
        return ("I want to be sure I understand what you're after. Should I change the code for this, or would you rather talk it through first? "
                "Tell me a bit more about the result you want.")

    # ------------------------------------------------------------------ task briefs and starting runs
    def _make_brief(self, cid: str, conv: dict, msg: dict, sig: Signals, previous: dict | None, blocks: list[str]) -> dict:
        extra = []
        if previous:
            extra.append("<previous_brief>\n" + brief_text(previous) + "\n</previous_brief>")
        extra += blocks
        text = msg["content"]
        final = f"Latest user message:\n{clip(text, 3000)}\n\nWrite the task brief now as JSON."
        msgs, _ = self._assemble(cid, conv, role="planner", system=BRIEF_SYS, history_upto=msg["id"], final_user=final,
                                 extra=extra, query=text)
        try:
            d = self.llm_fg.chat_json("planner", msgs, BRIEF_SCHEMA, None, retries=1)
        except LLMError as e:
            if "valid JSON" not in str(e):
                raise
            d = {}
        request = (d.get("request") or "").strip()
        if len(request) < 15:
            request = text.strip()
        return {"title": clip(d.get("title") or heuristic_title(text), 70), "request": clip(request, 6000),
                "keep_working": clean_list(d.get("keep_working"), 5, 200), "assumptions": clean_list(d.get("assumptions"), 6, 220),
                "open_questions": clean_list(d.get("open_questions"), 3, 220)}

    def _post_brief(self, cid: str, brief: dict, reply_to: int) -> dict | None:
        with self.slock:
            conv = self.mem.get_conv(cid)
            if not conv:
                return None
            old = conv["state"].get("brief")
            if old:
                self._set_brief_status(old, "superseded")
            m = self.mem.add_msg(cid, "assistant", "brief", brief_text(brief), {"brief": brief, "status": "pending", "reply_to": reply_to})
            brief["msg_id"] = m["id"]
            self.mem.patch_state(cid, brief=brief)
        self.pub(cid, "message", message=m)
        return m

    def start_task(self, cid: str, brief: dict) -> str:
        request = brief_to_request(brief)
        ahead = [r for r in self.mem.list_runs(30) if r["status"] not in TERMINAL_RUN]
        rid = self.engine.submit(request, {"conv_id": cid, "title": brief.get("title", "")})
        self._run_conv[rid] = cid
        with self.slock:
            self._set_brief_status(brief, "started")
            self.mem.patch_state(cid, brief=None, active_run=rid)
            m = self.mem.add_msg(cid, "assistant", "run", "Started task: " + brief.get("title", ""),
                                 {"run_id": rid, "title": brief.get("title", ""), "queued_behind": len(ahead)})
        self.pub(cid, "message", message=m)
        return rid

    # ------------------------------------------------------------------ run -> chat
    def _conv_of(self, rid: str) -> str | None:
        if rid in self._run_conv:
            return self._run_conv[rid]
        run = self.mem.get_run(rid)
        cid = (run or {}).get("options", {}).get("conv_id")
        self._run_conv[rid] = cid if cid and self.mem.get_conv(cid) else None
        return self._run_conv[rid]

    def _on_event(self, ev: dict) -> None:
        try:
            rid = ev["run_id"]
            cid = self._conv_of(rid)
            if not cid:
                return
            t, p = ev["type"], ev["payload"]
            if t == "status":
                st = p.get("status")
                self.pub(cid, "run_status", run_id=rid, status=st)
                if st in TERMINAL_RUN:
                    self._finish_run(cid, rid)
            elif t in ("question", "approval", "blocker"):
                self.pub(cid, "run_status", run_id=rid, status={"question": "awaiting_user", "approval": "awaiting_approval", "blocker": "blocked"}[t])
        except Exception:  # noqa: BLE001
            log.exception("run event handling failed")

    def _has_result(self, cid: str, rid: str, status: str) -> bool:
        return any(m["meta"].get("run_id") == rid and m["meta"].get("status") == status for m in self.mem.find_msgs_conv(cid, "run_result", 100))

    def _finish_run(self, cid: str, rid: str) -> None:
        with self.slock:
            run = self.mem.get_run(rid)
            if not run or not self.mem.get_conv(cid) or run["status"] not in TERMINAL_RUN or self._has_result(cid, rid, run["status"]):
                return
            m = self.mem.add_msg(cid, "assistant", "run_result", compose_result(run),
                                 {"run_id": rid, "status": run["status"], "files": run["files"], "title": run["options"].get("title", "")})
            conv = self.mem.get_conv(cid)
            if conv and conv["state"].get("active_run") == rid:
                self.mem.patch_state(cid, active_run=None)
        self.pub(cid, "message", message=m)
        self.pub(cid, "run_status", run_id=rid, status=run["status"])
        self.schedule("extract", cid)
        self.schedule("fold", cid)

    def reconcile(self) -> None:
        """After a restart: post results for tasks that ended while the server was down."""
        for m in self.mem.find_msgs("run", 400):
            rid, cid = m["meta"].get("run_id"), m["conv_id"]
            if not rid:
                continue
            self._run_conv[rid] = cid
            st = self.mem.run_status(rid)
            if st in TERMINAL_RUN:
                self._finish_run(cid, rid)

    # ------------------------------------------------------------------ discussion
    def _discuss(self, cid: str, conv: dict, msg: dict, sig: Signals, route: dict, pending: dict | None) -> None:
        mid, text = msg["id"], msg["content"].strip()
        extra: list[str] = []
        labels: list[str] = []
        if code_names(text) or route.get("needs_code"):
            blocks, _paths, labels = self._gather(cid, text, route)
            extra += blocks
        if pending:
            extra.append("<pending_task_brief status=\"waiting for the user to confirm\">\n" + clip(brief_text(pending), 1200) + "\n</pending_task_brief>")
        state = self._task_state(conv)
        if state:
            extra.append(state)
        if sig.change_ref:
            d = self._diff_block(cid)
            if d:
                extra.append(d)
        if sig.memory_ref:
            allf = [f for f in self.mem.list_facts() if f["source"].startswith(("chat", "user"))][:25]
            if allf:
                extra.append("<all_saved_memory>\n" + clip("\n".join(f"- [{f['kind']}] {f['text']}" for f in allf), 2400) + "\n</all_saved_memory>")
        directive = ""
        if sig.remember or sig.correction:
            directive = "The user is asking you to remember, correct or forget something. Confirm in one short sentence what you will keep or drop; it is saved automatically."
        prev_user = ""
        for m in reversed(self.mem.recent_msgs(cid, 6, before=mid)):
            if m["role"] == "user" and m["kind"] == "text":
                prev_user = m["content"]
                break
        self._phase(cid, "Recalling our conversation")
        msgs, used = self._assemble(cid, conv, role="chat", system=CHAT_SYS, directive=directive, history_upto=mid,
                                    final_user=clip(text, 6000), extra=extra, query=f"{text} {prev_user}")
        self._phase(cid, "Writing")
        out, truncated, stopped = self._stream(cid, msgs)
        out = strip_think(out)
        if not out:
            return
        meta = {"reply_to": mid, "stopped": stopped, "truncated": truncated,
                "activity": labels[:8], "memory": [{"id": f["id"], "kind": f["kind"], "text": clip(f["text"], 140)} for f in used[:6]]}
        self._say(cid, out, mid, **{k: v for k, v in meta.items() if k != "reply_to"})

    def _stream(self, cid: str, msgs: list[dict]) -> tuple[str, bool, bool]:
        buf: list[str] = []
        last = [time.time()]
        thinking = [False]

        def flush() -> None:
            if buf:
                s = "".join(buf)
                buf.clear()
                self.pub(cid, "delta", text=s)

        def tok(kind: str, text: str) -> None:
            if kind == "thinking":
                if not thinking[0]:
                    thinking[0] = True
                    self._phase(cid, "Thinking it through")
                return
            if thinking[0]:
                thinking[0] = False
                self._phase(cid, "Writing")
            buf.append(text)
            if time.time() - last[0] > 0.05:
                flush()
                last[0] = time.time()
        try:
            text, truncated = self.llm_fg.chat_stream("chat", msgs, tok)
            flush()
            return text, truncated, False
        except Cancelled:
            flush()
            with self.hub.lock:
                partial = (self.live.get(cid) or {}).get("text", "")
            return partial, False, True

    # ------------------------------------------------------------------ context assembly
    @staticmethod
    def _plain(m: dict) -> str:
        if m["kind"] == "brief":
            return "[task brief] " + m["content"]
        if m["kind"] == "run":
            return "[task started] " + m["meta"].get("title", "")
        return m["content"]

    @staticmethod
    def _turn_of(m: dict) -> tuple[str, str]:
        k = m["kind"]
        if k == "text":
            return ("user" if m["role"] == "user" else "assistant"), m["content"]
        if k == "brief":
            return "assistant", f"[Task brief - {m['meta'].get('status', 'pending')}]\n" + m["content"]
        if k == "run":
            return "assistant", "[Task started] " + m["meta"].get("title", "")
        return "assistant", m["content"]

    @contextmanager
    def _repo(self, timeout: float = 4.0):
        ok = self.engine.lock.acquire(timeout=timeout)
        try:
            yield ok
        finally:
            if ok:
                self.engine.lock.release()

    def _recall(self, query: str, max_chars: int) -> list[dict]:
        ix = getattr(self.engine, "ix", None)
        with self._repo(2.0) as ok:
            if ok and ix is not None:
                def fn(p: str):
                    return (ix.db.execute("SELECT hash FROM files WHERE path=?", (p,)).fetchone() or [None])[0]
            else:
                def fn(p: str):
                    return None
            try:
                return self.mem.recall(query, [], fn, max_chars=max_chars, always=("preference",))
            except Exception:  # noqa: BLE001
                log.exception("recall failed")
                return []

    def _assemble_inner(self, cid: str, conv: dict, *, role: str, system: str, history_upto: int, final_user: str,
                  extra=(), query: str = "", directive: str = "") -> tuple[list[dict], list[dict]]:
        """System prompt + memory + summary + gap digest + retrieved excerpts + newest messages + the final user turn, in budget."""
        rc, cc = self.cfg.roles[role], self.cfg.chat
        total = int(rc.prompt_budget() * 3.0)
        facts = self._recall(query, cc.memory_chars)
        mem_block = ("<project_memory>\n" + "\n".join(f"- [{f['kind']}] {f['text']}" for f in facts) + "\n</project_memory>") if facts else ""
        summary = clip(conv.get("summary", ""), min(3000, int(total * 0.10)))
        summary_block = f"<conversation_summary>\n{summary}\n</conversation_summary>" if summary else ""
        extra_text = "\n\n".join(x for x in extra if x)
        extra_text = clip_mid(extra_text, int(total * 0.28)) if extra_text else ""
        head = "\n\n".join(p for p in (system, directive) if p)
        reserve = int(total * 0.17)
        room = total - (len(head) + len(mem_block) + len(summary_block) + len(extra_text) + len(final_user) + reserve + 400)
        if room < 3000 and extra_text:
            extra_text = clip_mid(extra_text, max(600, len(extra_text) - (3000 - room)))
            room = total - (len(head) + len(mem_block) + len(summary_block) + len(extra_text) + len(final_user) + reserve + 400)
        room = max(room, 1500)

        cand = [m for m in self.mem.recent_msgs(cid, 80, before=history_upto) if not m["meta"].get("error")]
        chosen: list[tuple[dict, str, str]] = []
        used = 0
        for m in reversed(cand):
            r, t = self._turn_of(m)
            t = clip_mid(t, 5000 if len(chosen) < 2 else 2200)
            cost = len(t) + 12
            if chosen and used + cost > room:
                break
            chosen.append((m, r, t))
            used += cost
        chosen.reverse()
        first_id = chosen[0][0]["id"] if chosen else history_upto

        gap_block = ""
        gap_msgs = [m for m in self.mem.msgs_range(cid, after=conv.get("summary_upto", 0), before=first_id, limit=80) if not m["meta"].get("error")]
        if gap_msgs:
            lines, size = [], 0
            for m in reversed(gap_msgs):
                ln = f"[#{m['id']} {'user' if m['role'] == 'user' else 'assistant'}] {one_line(self._plain(m), 220)}"
                if size + len(ln) > int(total * 0.07):
                    lines.append("(older lines omitted)")
                    break
                lines.append(ln)
                size += len(ln)
            gap_block = "<earlier_messages_not_yet_summarized>\n" + "\n".join(reversed(lines)) + "\n</earlier_messages_not_yet_summarized>"

        exc_block = ""
        if self.mem.has_older(cid, first_id):
            prev_user = next((t for _, r, t in reversed(chosen) if r == "user"), "")
            hits = self.mem.search_msgs(cid, f"{query} {prev_user}", first_id, 5)
            lines, size = [], 0
            for m in hits:
                r, t = self._turn_of(m)
                ln = f"[#{m['id']} {r}] {clip(t, 500)}"
                if size + len(ln) > int(total * 0.10):
                    break
                lines.append(ln)
                size += len(ln)
            if lines:
                exc_block = "<earlier_excerpts>\n" + "\n".join(lines) + "\n</earlier_excerpts>"

        system_text = "\n\n".join(p for p in (head, mem_block, summary_block, gap_block, exc_block, extra_text) if p)
        msgs = [{"role": "system", "content": system_text}]
        msgs += [{"role": r, "content": t} for _, r, t in chosen]
        msgs.append({"role": "user", "content": final_user})
        return msgs, facts

    def _task_state(self, conv: dict) -> str:
        rid = conv["state"].get("active_run")
        run = self.mem.get_run(rid) if rid else None
        if not run or run["status"] in TERMINAL_RUN:
            return ""
        lines = [f"Task in progress (status: {run['status']}): {clip(run['request'], 300)}"]
        lines += [f"- {t['id']} {t['title']}: {t['status']}" for t in run["tasks"][:10]]
        pend = getattr(self.engine, "pending", lambda r: None)(rid)
        if pend:
            lines.append(f"The task is waiting for the user ({pend.get('kind')}): " + clip(" | ".join(pend.get("questions", []) or [pend.get("title", "")]), 500))
        return "<task_in_progress>\n" + "\n".join(lines) + "\n</task_in_progress>"

    def _diff_block(self, cid: str) -> str:
        for m in reversed(self.mem.find_msgs_conv(cid, "run_result", 10)):
            if m["meta"].get("status") == "done" and m["meta"].get("run_id"):
                p = self.cfg.forge_dir / "runs" / m["meta"]["run_id"] / "changes.patch"
                try:
                    txt = p.read_text(encoding="utf-8")
                except OSError:
                    return ""
                return f'<last_change_diff run="{m["meta"]["run_id"]}">\n{clip_mid(txt, 4500)}\n</last_change_diff>'
        return ""

    # ------------------------------------------------------------------ grounding in the real code
    def _tooling(self):
        if self._tools is None:
            ix = getattr(self.engine, "ix", None)
            if ix is None:
                return None
            try:
                from .agents import Agents
                from .context import ContextBuilder
            except Exception:  # noqa: BLE001
                return None
            view = RepoView(ix)
            ctx = ContextBuilder(ix, view, self.mem)
            agents = Agents(self.llm_fg, self.cfg, ix, view, ctx, self.mem)
            original = agents.tools.run

            def locked(*a, **k):
                with self._repo(4.0) as ok:
                    return original(*a, **k) if ok else "error: the code index is busy, try again"
            agents.tools.run = locked
            self._tools = (agents, ctx)
        return self._tools

    def _gather_inner(self, cid: str, text: str, route: dict) -> tuple[list[str], list[str], list[str]]:
        """Code evidence for a question: things the user named first (instant), then a short read-only investigation."""
        blocks: list[str] = []
        paths: list[str] = []
        labels: list[str] = []
        tools = self._tooling()
        if tools is None:
            return blocks, paths, labels
        budget = int(self.cfg.roles["chat"].prompt_budget() * 3.0 * 0.22)
        names = code_names(text)
        if names:
            self._phase(cid, "Looking at the code you mentioned")
            with self._repo(3.0) as ok:
                if ok:
                    try:
                        blocks, paths = self._named(tools[1], names, budget)
                    except Exception:  # noqa: BLE001
                        log.exception("named lookup failed")
            labels += [f"Opened {p}" for p in paths]
            for lb in labels:
                self.pub(cid, "activity", label=lb)
        steps = self.cfg.chat.research_steps
        if not blocks and route.get("needs_code") and steps > 0:
            self._phase(cid, "Reading the relevant code")
            query = route.get("code_query") or one_line(text, 160)

            def emit(t: str, **kw) -> None:
                if t == "tool":
                    lb = f"{kw.get('action')}: {kw.get('arg', '')}"[:90]
                    labels.append(lb)
                    self.tracer.note(lb, agent="context", thought=kw.get("thought", ""), response=kw.get("result", "")[:500])
                    self.pub(cid, "activity", label=lb)
            card = self.mem.meta_get("project_card", "") or ""
            seed = f"<project_map>\n{card[:2500]}\n</project_map>\n<question>\n{clip(text, 800)}\n</question>"
            obs, final = tools[0].investigate("context", f"Find the code needed to answer the user's question: {query}",
                                              FINISH_HINT, seed, steps, emit, None)
            if obs:
                body = "\n\n".join(obs)
                if final and final.get("answer"):
                    body += "\n\nInvestigator notes: " + clip(final["answer"], 600)
                blocks.append("<code_evidence>\n" + clip_mid(body, budget) + "\n</code_evidence>")
        return blocks, paths, labels
    
    def _assemble(self, cid, conv, **kw):
        with self.tracer.span("Assemble prompt", "step", agent="context") as sp:
            msgs, facts = self._assemble_inner(cid, conv, **kw)
            chars = sum(len(m["content"]) for m in msgs)
            sp.set(response=f"{len(msgs)} messages · {chars:,} chars · {len(facts)} memory facts recalled")
            return msgs, facts

    def _gather(self, cid, text, route):
        with self.tracer.span("Read code", "step", agent="context") as sp:
            blocks, paths, labels = self._gather_inner(cid, text, route)
            sp.set(response=f"{len(blocks)} code blocks · " + (", ".join(paths) or "investigation notes"))
            return blocks, paths, labels

    def _named(self, ctx, names: list[str], budget: int) -> tuple[list[str], list[str]]:
        from .patching import Lookup
        ix = self.engine.ix
        blocks: list[str] = []
        paths: list[str] = []
        used = 0
        use = names[:4]
        per = max(600, budget // max(1, len(use)))
        terms = query_terms(" ".join(use), 12)
        for nm in use:
            if used >= budget:
                break
            if "/" in nm or re.search(r"\.[A-Za-z]{1,5}$", nm):
                want = nm.lstrip("./")
                cands = sorted((p for p in ix.all_paths() if p == want or p.endswith("/" + want)), key=len)
                for p in cands[:1]:
                    text, _mode, _tok = ctx.render_file(p, int(per / 3.0), terms, [])
                    if text:
                        blocks.append(text)
                        paths.append(p)
                        used += len(text)
            else:
                exact = [s for s in ix.find_symbol(nm, limit=6) if s.name.lower() == nm.lower()]
                for s in exact[:2]:
                    text, found = ctx.resolve_lookup(Lookup("symbol", s.name, s.path))
                    if found and text:
                        blocks.append(clip(text, per))
                        paths.append(s.path)
                        used += min(len(text), per)
        return blocks, paths

    # ------------------------------------------------------------------ background jobs
    def _job_title(self, cid: str) -> None:
        conv = self.mem.get_conv(cid)
        if not conv or conv["title_locked"]:
            return
        n_user = self.mem.count_msgs(cid, role="user", kind="text")
        if n_user == 0 or self.mem.count_msgs(cid, role="assistant", kind="text") == 0:
            return
        at = conv["titled_at"]
        if not (at == 0 or (at < 3 and n_user >= 3)):
            return
        sample = []
        for m in self.mem.msgs_range(cid, limit=14):
            if m["kind"] == "text" and not m["meta"].get("error"):
                sample.append(f"{'user' if m['role'] == 'user' else 'assistant'}: {one_line(m['content'], 300)}")
        msgs = [{"role": "system", "content": TITLE_SYS}, {"role": "user", "content": "\n".join(sample[:6])}]
        try:
            d = self.llm_bg.chat_json("summarizer", msgs, TITLE_SCHEMA, None, retries=1)
            title = clean_title(d.get("title", ""))
        except LLMError:
            title = ""
        fresh = self.mem.get_conv(cid)
        if not fresh or fresh["title_locked"]:
            return
        if title:
            self.mem.update_conv(cid, title=title, titled_at=n_user)
            self.pub(cid, "title", title=title)
        else:
            self.mem.update_conv(cid, titled_at=n_user)

    def _fold_chunk(self, summary: str, chunk: list[dict]) -> str:
        body = "\n".join(f"[#{m['id']} {'user' if m['role'] == 'user' else 'assistant'}] {clip(self._plain(m), 1200)}" for m in chunk)
        user = f"<previous_summary>\n{summary or '(none yet)'}\n</previous_summary>\n<new_messages>\n{body}\n</new_messages>"
        d = self.llm_bg.chat_json("summarizer", [{"role": "system", "content": SUMMARY_SYS}, {"role": "user", "content": user}], DIGEST_SCHEMA, None, retries=1)
        s = clip(d.get("summary", ""), 3200)
        if not s:
            raise LLMError("The summary came back empty")
        return s

    def _job_fold(self, cid: str) -> None:
        """Fold older messages into the rolling summary. Progress is saved per chunk, so an interrupted job resumes."""
        conv = self.mem.get_conv(cid)
        if not conv:
            return
        cc = self.cfg.chat
        msgs = [m for m in self.mem.msgs_range(cid, after=conv["summary_upto"], limit=500) if not m["meta"].get("error")]
        if len(msgs) <= cc.keep_recent:
            return
        old = msgs[:len(msgs) - cc.keep_recent]
        if sum(len(self._plain(m)) for m in old) < cc.summary_trigger and len(old) < 14:
            return
        summary = conv["summary"]
        chunk: list[dict] = []
        size = 0
        for m in old + [None]:
            t = clip(self._plain(m), 1200) if m else ""
            if chunk and (m is None or size + len(t) > 6000):
                summary = self._fold_chunk(summary, chunk)
                self.mem.update_conv(cid, summary=summary, summary_upto=chunk[-1]["id"])
                chunk, size = [], 0
            if m:
                chunk.append(m)
                size += len(t)

    def _job_extract(self, cid: str) -> None:
        conv = self.mem.get_conv(cid)
        if not conv:
            return
        new = [m for m in self.mem.msgs_range(cid, after=conv["extracted_upto"], limit=40) if not m["meta"].get("error")]
        if not new:
            return
        last_id = new[-1]["id"]
        users = [m["content"] for m in new if m["role"] == "user" and m["kind"] == "text"]
        joined = " ".join(users)
        sig = analyze(joined)
        worth = bool(users) and (sig.remember or sig.correction or len(joined) >= 80)
        if any(m["kind"] in ("run_result", "brief") for m in new) and users:
            worth = True
        if not worth:
            self.mem.update_conv(cid, extracted_upto=last_id)
            return
        known = self._recall(joined, 2500)
        known_ids = {f["id"]: f for f in known}
        body = "\n".join(f"{'user' if m['role'] == 'user' else 'assistant'}: {clip(self._plain(m), 900)}" for m in new[-10:] if m["kind"] != "notice")
        user = ("<known_facts>\n" + ("\n".join(f"[{f['id']}] {f['kind']}: {f['text']}" for f in known) or "(none)") + "\n</known_facts>\n"
                f"<new_messages>\n{clip_mid(body, 5000)}\n</new_messages>")
        d = self.llm_bg.chat_json("summarizer", [{"role": "system", "content": FACTS_SYS}, {"role": "user", "content": user}], FACTS_SCHEMA, None, retries=1)
        changed = False
        for f in (d.get("facts") or [])[:5]:
            text = " ".join(str(f.get("text", "")).split())
            kind = f.get("kind") if f.get("kind") in CHAT_KINDS else "fact"
            if not 15 <= len(text) <= 320 or SECRET_RE.search(text):
                continue
            for sid in f.get("supersedes") or []:
                old = known_ids.get(sid)
                if old and old["source"].startswith(("chat", "user")):
                    self.mem.forget(sid)
            if self.mem.add_fact(kind, text, f"chat:{cid}"):
                changed = True
        for fid in d.get("forget") or []:
            old = known_ids.get(fid)
            if old and (old["source"].startswith(("chat", "user")) or sig.correction):
                self.mem.forget(fid)
                changed = True
        self.mem.update_conv(cid, extracted_upto=last_id)
        if changed:
            self.pub(cid, "memory")