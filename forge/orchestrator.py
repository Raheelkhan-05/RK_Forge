# forge/orchestrator.py

"""Orchestrator: plans, executes subtasks through the agents, enforces gates, and commits atomically.

The controller (this file), not the model, decides: task order, retry limits, scope, acceptance, and whether the
real repository is ever touched. The model only proposes.
"""
from __future__ import annotations

import queue
import re
import sqlite3
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from typing import Callable

from .agents import Agents, fmt_task
from .config import Config
from .context import ContextBuilder
from .indexer import Index
from .llm import Cancelled, LLMError, OllamaLLM, est_tokens
from .telemetry import Tracer
from .memory import Memory
from .patching import apply_edits, parse_reply, CoderReply
from .verify import GateResult, Verifier, condense_output
from .workspace import ConflictError, TxError, Workspace, recover, safe_rmtree, undo_transaction

TERMINAL = {"done", "failed", "cancelled", "rejected", "undone", "conflict", "interrupted"}


class RunError(RuntimeError):
    pass


class RunCtl:
    """Thread-safe pause / cancel / ask-the-human channel for one run."""

    def __init__(self):
        self.cancelled = False
        self.running = threading.Event()
        self.running.set()
        self._answer_evt = threading.Event()
        self._answer: dict | None = None
        self.pending: dict | None = None

    def stop(self) -> bool:
        return self.cancelled

    def checkpoint(self) -> None:
        while not self.running.is_set():
            if self.cancelled:
                break
            self.running.wait(0.3)
        if self.cancelled:
            raise Cancelled()

    def wait_for(self, payload: dict) -> dict:
        self._answer = None
        self._answer_evt.clear()
        self.pending = payload
        try:
            while not self._answer_evt.wait(0.4):
                if self.cancelled:
                    raise Cancelled()
            return self._answer or {}
        finally:
            self.pending = None

    def respond(self, data: dict) -> bool:
        if self.pending is None:
            return False
        self._answer = data
        self._answer_evt.set()
        return True

    def cancel(self) -> None:
        self.cancelled = True
        self.running.set()
        self._answer_evt.set()


def gates_text(results: list[GateResult], only_blocking: bool = False, max_each: int = 1800) -> str:
    out = []
    for r in results:
        if only_blocking and not r.blocking:
            continue
        head = f"{r.name} ({r.project})"
        if r.skipped:
            out.append(f"[SKIP] {head}: {r.reason}")
        elif r.ok:
            out.append(f"[PASS] {head}" + (f" - {r.reason}" if r.reason else ""))
        elif r.preexisting:
            out.append(f"[FAIL, PRE-EXISTING] {head}: failed the same way before this change; not caused by it")
        else:
            out.append(f"[FAIL] {head}: {r.cmd}\n{condense_output(r.output, max_each)}")
    return "\n".join(out) or "(no gates ran)"


class Engine:
    """One per repository. Owns the memory, the persistent index and a single worker (one run at a time)."""

    def __init__(self, cfg: Config, llm_factory: Callable[[Callable[[], bool]], object] | None = None):
        self.cfg = cfg
        self.mem = Memory(cfg)
        self.ix = Index(cfg.forge_dir / "index.db", cfg.repo, cfg)
        self.lock = threading.RLock()
        self.ctls: dict[str, RunCtl] = {}
        self.live: dict[str, "RunExecutor"] = {}
        self.q: queue.Queue = queue.Queue()
        self.llm_factory = llm_factory or (lambda stop: OllamaLLM(cfg, stop))
        self.recovered = recover(cfg.forge_dir / "journal")
        for r in self.mem.list_runs(200):
            if r["status"] not in TERMINAL:
                self.mem.update_run(r["id"], status="interrupted")
        self.worker = threading.Thread(target=self._loop, daemon=True, name="forge-worker")
        self.worker.start()
        self.gc()

    # ------------------------------------------------------------------ public API
    def reindex(self) -> dict:
        with self.lock:
            stats = self.ix.refresh()
            self.mem.meta_set("project_card", self.ix.repo_map(5000))
            return {**stats, **self.ix.stats()}

    def submit(self, request: str, options: dict | None = None) -> str:
        rid = self.mem.create_run(request.strip(), options or {})
        self.ctls[rid] = RunCtl()
        self.mem.emit(rid, "status", status="queued")
        self.q.put((rid, False))
        return rid

    def resume(self, rid: str) -> None:
        run = self.mem.get_run(rid)
        if not run or run["status"] != "interrupted":
            raise ValueError("Only interrupted runs can be resumed")
        self.ctls[rid] = RunCtl()
        self.mem.update_run(rid, status="queued")
        self.q.put((rid, True))

    def control(self, rid: str, action: str, **data) -> bool:
        ctl = self.ctls.get(rid)
        if not ctl:
            return False
        if action == "pause":
            ctl.running.clear()
            self.mem.emit(rid, "status", status="paused")
            return True
        if action == "continue":
            ctl.running.set()
            self.mem.emit(rid, "status", status="running")
            return True
        if action == "cancel":
            ctl.cancel()
            return True
        return ctl.respond({"action": action, **data})

    def answer(self, rid: str, answers: list[str]) -> bool:
        ctl = self.ctls.get(rid)
        return bool(ctl and ctl.respond({"action": "answer", "answers": answers}))

    def pending(self, rid: str) -> dict | None:
        ctl = self.ctls.get(rid)
        return ctl.pending if ctl else None

    def undo(self, rid: str) -> None:
        with self.lock:
            undo_transaction(self.cfg.repo, self.cfg.forge_dir / "journal", rid)
            run = self.mem.get_run(rid)
            self.ix.refresh(run["files"] if run else None)
        self.mem.update_run(rid, status="undone")
        self.mem.emit(rid, "status", status="undone")

    def gc(self) -> None:
        """Delete staging copies of finished runs (failed ones are kept for inspection, newest 3)."""
        try:
            root = Workspace(self.cfg, "x").run_dir.parent
            if not root.is_dir():
                return
            failed = [r["id"] for r in self.mem.list_runs(200) if r["status"] in ("failed", "interrupted", "conflict")][:3]
            for d in root.iterdir():
                if d.name in failed or d.name in self.ctls:
                    continue
                safe_rmtree(d)
        except OSError:
            pass

    # ------------------------------------------------------------------ worker
    def _loop(self) -> None:
        while True:
            rid, resume = self.q.get()
            ctl = self.ctls.get(rid)
            if ctl is None or ctl.cancelled:
                self.mem.update_run(rid, status="cancelled", finished=time.time())
                continue
            try:
                ex = RunExecutor(self, rid)
                self.live[rid] = ex
                ex.execute(resume)
            except BaseException as e:  # noqa: BLE001
                self.mem.emit(rid, "error", message=f"Unexpected failure: {e}", detail=traceback.format_exc()[-1500:])
                self.mem.update_run(rid, status="failed", error=str(e), finished=time.time())
                self.mem.emit(rid, "status", status="failed")
            finally:
                self.live.pop(rid, None)


class RunExecutor:
    def __init__(self, eng: Engine, rid: str):
        self.eng, self.cfg, self.mem, self.rid = eng, eng.cfg, eng.mem, rid
        self.ctl = eng.ctls[rid]
        self.llm = eng.llm_factory(self.ctl.stop)
        run = self.mem.get_run(rid)
        self.request: str = run["request"]
        self.opts: dict = run["options"]
        self.tracer = Tracer(lambda scope, ev: self.mem.emit(scope, "span", persist=ev["phase"] != "update", **ev))
        self.llm.tracer = self.tracer
        self.tracer.begin(rid, "Task run", text=self.request[:120])
        self.max_attempts = int(self.opts.get("max_attempts") or self.cfg.limits.max_attempts)
        self.answers: list[str] = []
        self.done_lines: list[str] = []
        self.tasks: list[dict] = []
        self.states: dict[str, str] = {}
        self.replans = 0
        self.partial = False
        self.ws: Workspace | None = None
        self.six: Index | None = None
        self.ctx: ContextBuilder | None = None
        self.agents: Agents | None = None
        self.ver: Verifier | None = None
        self.phase = "running"

    # ------------------------------------------------------------------ plumbing
    def emit(self, type_: str, **kw) -> None:
        self.mem.emit(self.rid, type_, **kw)

    def status(self, s: str, **kw) -> None:
        self.phase = s if s not in ("awaiting_user", "awaiting_approval", "blocked") else self.phase
        self.mem.update_run(self.rid, status=s)
        self.emit("status", status=s, **kw)

    @contextmanager
    def stream(self, agent: str, task: str | None = None):
        buf: list[tuple[str, str]] = []
        last = [time.time()]

        def flush():
            if buf:
                for kind in ("thinking", "content"):
                    txt = "".join(t for k, t in buf if k == kind)
                    if txt:
                        self.mem.emit(self.rid, "stream", persist=False, agent=agent, kind=kind, text=txt, task=task)
                buf.clear()

        def on_token(kind: str, text: str):
            buf.append((kind, text))
            if time.time() - last[0] > 0.2:
                flush()
                last[0] = time.time()
        self.mem.emit(self.rid, "stream_start", persist=False, agent=agent, task=task)
        with self.tracer.span(f"{agent}{' · ' + task if task else ''}", "step", agent=agent, task=task):
            try:
                yield on_token
            finally:
                flush()
                self.mem.emit(self.rid, "stream_end", persist=False, agent=agent, task=task)

    def set_task(self, t: dict, status: str, attempts: int = 0, summary: str = "") -> None:
        self.states[t["id"]] = status
        idx = next((i for i, x in enumerate(self.tasks) if x["id"] == t["id"]), len(self.tasks))
        self.mem.upsert_task(self.rid, t["id"], idx, t["title"], status, attempts, t.get("files_to_modify", []) + t.get("files_to_create", []), summary)
        self.emit("task", id=t["id"], title=t["title"], status=status, attempts=attempts, summary=summary)

    def ask(self, questions: list[str], source: str, kind: str = "clarification", context: str = "") -> list[str]:
        payload = {"id": uuid.uuid4().hex[:8], "kind": kind, "source": source, "questions": questions, "context": context}
        self.mem.update_run(self.rid, status="awaiting_user")
        self.emit("status", status="awaiting_user")
        self.emit("question", **payload)
        data = self.ctl.wait_for(payload)
        self.status(self.phase)
        lines = []
        given = data.get("answers", [])
        for i, q in enumerate(questions):
            a = (given[i] if i < len(given) else "").strip()
            if a:
                lines.append(f"Q: {q}\nA: {a}")
                self.mem.add_fact("answer", f"{q} -> {a}", "user", run_id=self.rid)
            else:
                lines.append(f"Q: {q}\nA: (the user did not answer; choose the safest minimal behavior and state your assumption in SUMMARY)")
        self.emit("answered", count=len(lines))
        return lines

    # ------------------------------------------------------------------ main flow
    def execute(self, resume: bool = False) -> None:
        try:
            self.status("preparing")
            health = self.llm.health() if hasattr(self.llm, "health") else {"ok": True}
            if not health["ok"]:
                raise RunError(health["error"])
            self.prepare(resume)
            run = self.mem.get_run(self.rid)
            if resume and run and run["plan"]:
                self.tasks = run["plan"]["tasks"]
                self.states = {t["id"]: t["status"] for t in run["tasks"]}
                self.answers = [f"Q: {f['text'].split(' -> ')[0]}\nA: {f['text'].split(' -> ', 1)[1]}" for f in self.mem.list_facts() if f["run_id"] == self.rid and f["kind"] == "answer" and " -> " in f["text"]]
                self.done_lines = [f"{t['id']} {t['title']}: {t['summary']}" for t in run["tasks"] if t["status"] == "accepted"]
            else:
                self.plan_phase()
            self.status("running")
            self.exec_phase([t for t in self.tasks if self.states.get(t["id"]) != "accepted"])
            self.final_phase()
            self.approve_and_commit()
        except Cancelled:
            self.status("cancelled")
            self.mem.update_run(self.rid, finished=time.time())
            self.cleanup(keep=False)
        except (RunError, LLMError) as e:
            self.emit("error", message=str(e))
            self.mem.update_run(self.rid, status="failed", error=str(e), finished=time.time())
            self.emit("status", status="failed", reason=str(e))
            self.cleanup(keep=True)
        finally:
            self.tracer.end()
            if self.ver:
                self.ver.cleanup()
            if self.six:
                self.six.close()
            self.ctl.pending = None

    def cleanup(self, keep: bool) -> None:
        if self.ws and not keep:
            self.ws.cleanup()

    def prepare(self, resume: bool) -> None:
        with self.eng.lock:
            self.eng.reindex()
            self.ws = Workspace(self.cfg, self.rid)
            if resume and (self.ws.run_dir / "state.json").exists():
                self.ws.resume()
            else:
                self.emit("agent", agent="system", message="Creating isolated workspace (your real files stay untouched)")
                self.ws.create()
            dest = self.ws.run_dir / "index.db"
            if not dest.exists():
                dst = sqlite3.connect(dest)
                self.eng.ix.db.backup(dst)
                dst.close()
        self.six = Index(dest, self.ws.tree, self.cfg)
        self.six.refresh()
        self.ctx = ContextBuilder(self.six, self.ws, self.mem)
        self.agents = Agents(self.llm, self.cfg, self.six, self.ws, self.ctx, self.mem)
        self.ver = Verifier(self.cfg, self.six, self.ws, self.ctl.stop)
        st = self.six.stats()
        self.emit("workspace", files=st["files"], lines=st["lines"], projects=st["projects"], model={r: c.model for r, c in self.cfg.roles.items()})

    # ------------------------------------------------------------------ planning
    def plan_phase(self, followup: str | None = None) -> None:
        self.status("planning")
        feedback, q_rounds = "", 0
        request = self.request if not followup else f"{self.request}\n\nFollow-up requested by the user after reviewing the result (the current state already contains the earlier work):\n{followup}"
        existing = {t["id"] for t in self.tasks}
        new_tasks: list[dict] = []
        plan: dict = {}
        for _ in range(5):
            self.ctl.checkpoint()
            with self.stream("planner") as tok:
                plan = self.agents.plan(request, self.answers, self.emit, tok, feedback)
            qs = [q.strip() for q in plan.get("questions", []) if q.strip()]
            new_tasks, errs = self.agents.validate_plan(plan, existing)
            if qs and q_rounds < 2:
                q_rounds += 1
                self.answers += self.ask(qs, "planner", context=plan.get("understanding", ""))
                feedback = ""
                continue
            if errs:
                self.emit("plan_rejected", errors=errs)
                feedback = "Your previous plan was rejected for these reasons:\n- " + "\n- ".join(errs) + "\nReturn the corrected full plan."
                continue
            if not new_tasks:
                feedback = "The plan contained no subtasks. Investigate again and return at least one."
                continue
            break
        else:
            raise RunError("The planner could not produce a valid plan: " + "; ".join(errs[:4] or ["empty plan"]))
        self.tasks += new_tasks
        self.mem.update_run(self.rid, plan={"understanding": plan.get("understanding", ""), "tasks": self.tasks})
        for t in new_tasks:
            self.set_task(t, "pending")
        self.emit("plan", understanding=plan.get("understanding", ""), tasks=self.tasks)

    # ------------------------------------------------------------------ execution
    def exec_phase(self, queue_: list[dict]) -> None:
        queue_ = list(queue_)
        while queue_:
            self.ctl.checkpoint()
            t = queue_.pop(0)
            bad = [d for d in t.get("depends_on", []) if self.states.get(d) in ("failed", "skipped")]
            if bad:
                self.set_task(t, "skipped", 0, f"Skipped because {', '.join(bad)} did not complete")
                self.partial = True
                continue
            ok, failure = self.run_task(t)
            if ok:
                continue
            self.handle_failure(t, failure, queue_)

    def handle_failure(self, t: dict, failure: str, queue_: list[dict]) -> None:
        if self.replans < self.cfg.limits.max_replans:
            self.replans += 1
            try:
                with self.stream("planner", t["id"]) as tok:
                    plan = self.agents.replan(self.request, t, failure, self.done_lines, self.emit, tok)
                qs = [q for q in plan.get("questions", []) if q.strip()]
                if qs:
                    self.answers += self.ask(qs, "planner", context=f"While re-planning {t['id']}")
                new, errs = self.agents.validate_plan(plan, {x["id"] for x in self.tasks})
                if new and not errs:
                    self.set_task(t, "replaced", self.max_attempts, "Split into smaller steps: " + ", ".join(n["id"] for n in new))
                    idx = self.tasks.index(t) + 1 if t in self.tasks else len(self.tasks)
                    self.tasks[idx:idx] = new
                    self.mem.update_run(self.rid, plan={"tasks": self.tasks})
                    for n in new:
                        self.set_task(n, "pending")
                    queue_[0:0] = new
                    self.emit("plan_update", tasks=self.tasks)
                    return
            except LLMError:
                pass
        self.emit("status", status="blocked")
        self.mem.update_run(self.rid, status="blocked")
        payload = {"kind": "blocker", "task": t["id"], "title": t["title"], "failure": failure[-1800:], "options": ["retry", "skip", "abort"]}
        self.emit("blocker", **payload)
        d = self.ctl.wait_for(payload)
        self.status(self.phase)
        action = d.get("action", "abort")
        if action == "retry":
            hint = (d.get("text") or "").strip()
            if hint:
                self.answers.append(f"Hint from the user for {t['id']}: {hint}")
                self.mem.add_fact("answer", f"For '{t['title']}': {hint}", "user", run_id=self.rid)
            queue_.insert(0, t)
        elif action == "skip":
            self.set_task(t, "skipped", 0, "Skipped by user")
            self.partial = True
        else:
            raise RunError(f"Stopped by the user after {t['id']} could not be completed. Nothing was changed in your repository.")

    def run_task(self, t: dict, initial_feedback: str = "") -> tuple[bool, str]:
        ws = self.ws
        self.set_task(t, "running", 0)
        cp = ws.checkpoint()
        extras: list[str] = []
        feedback = initial_feedback
        failure = ""
        for attempt in range(1, self.max_attempts + 1):
            self.ctl.checkpoint()
            ws.restore(cp)
            self.set_task(t, "running", attempt)
            self.emit("attempt", task=t["id"], attempt=attempt, of=self.max_attempts)
            ok, info = self._attempt(t, cp, attempt, feedback, extras)
            if ok:
                self.done_lines.append(f"{t['id']} {t['title']}: {info}")
                self.set_task(t, "accepted", attempt, info)
                return True, ""
            feedback = failure = info
            self.emit("retry", task=t["id"], attempt=attempt, reason=info[:700])
        ws.restore(cp)
        self.set_task(t, "failed", self.max_attempts, failure[:300])
        return False, failure

    def _budget(self, t: dict, feedback: str) -> int:
        rc = self.cfg.roles["coder"]
        fixed = est_tokens(fmt_task(t)) + est_tokens(self.request[:1500]) + est_tokens(feedback) + sum(est_tokens(x) for x in self.done_lines[-12:])
        return max(1200, rc.prompt_budget() - est_tokens("x" * 3300) - fixed - 150)

    def _coder(self, t: dict, attempt: int, feedback: str, extras: list[str]) -> CoderReply | str:
        """One coder turn including doubt resolution. Returns a READY reply or a failure message."""
        L = self.cfg.limits
        rounds = 0
        while True:
            self.ctl.checkpoint()
            budget = self._budget(t, feedback)
            with self.tracer.span("Build context", "step", agent="context") as sp:
                bundle = self.ctx.build(t, self.request, self.answers, extras, budget)
                sp.set(response=f"{bundle.tokens} of {budget} tokens · {len(bundle.items)} items")
            self.emit("context", task=t["id"], tokens=bundle.tokens, budget=budget, items=bundle.items[:40], omitted=bundle.omitted[:15])
            msgs = self.agents.coder_messages(self.request, t, bundle.text, self.done_lines, feedback, attempt)
            with self.stream("coder", t["id"]) as tok:
                raw = self.llm.chat("coder", msgs, None, tok)
            reply = parse_reply(raw)
            self.emit("coder_reply", task=t["id"], status=reply.status, summary=reply.summary, edits=len(reply.edits),
                      asks=reply.asks, lookups=[f"{x.kind} {x.arg}" for x in reply.lookups], uses=[u[0] for u in reply.uses][:12])
            if reply.truncated:
                return "Your reply was cut off before it finished. Make fewer and smaller edits, and do not repeat unchanged code."
            lookups, asks = list(reply.lookups), list(reply.asks)
            if not lookups and not asks:
                if not reply.edits and not reply.deletes:
                    return "Your reply contained no edits. Either provide SEARCH/REPLACE edits or STATUS: NEED_INFO with what you need."
                bad = self.ctx.verify_uses(reply.uses, "\n".join(r for _, _, r in reply.edits))
                if not bad:
                    return reply
                self.emit("grounding", task=t["id"], unresolved=[f"{b.kind} {b.arg}" for b in bad])
                lookups = bad
            rounds += 1
            user_q: list[str] = []
            if rounds > L.max_lookup_rounds:
                user_q = asks + [f"I could not find {x.kind} '{x.arg}'" + (f" in {x.path}" if x.path else "") + ". Where is it defined, or what should be used?" for x in lookups]
            else:
                pending_q = list(asks)
                for lk in lookups:
                    text, found = self.ctx.resolve_lookup(lk)
                    if found:
                        if text not in extras:
                            extras.append(text)
                    else:
                        pending_q.append(f"Where is {lk.kind} '{lk.arg}'{' in ' + lk.path if lk.path else ''} defined or implemented in this repository?")
                for q in pending_q:
                    with self.stream("context", t["id"]) as tok:
                        r = self.agents.resolve_doubt(q, t, self.emit, tok)
                    self.emit("doubt", task=t["id"], question=q, answered=r["answered"], answer=r.get("answer", "")[:400], evidence=r.get("evidence", []))
                    if r["answered"]:
                        extras.append(f'<resolved question="{q}">\n{r["answer"]}\nevidence: {", ".join(r["evidence"])}\n</resolved>')
                    else:
                        user_q.append(q)
            if user_q:
                self.answers += self.ask(user_q, "coder", kind="doubt", context=f"{t['id']}: {t['title']}")

    def _attempt(self, t: dict, cp: dict, attempt: int, feedback: str, extras: list[str]) -> tuple[bool, str]:
        ws = self.ws
        allowed = set(t["files_to_modify"]) | set(t["files_to_create"])
        related = self.ctx.related(list(allowed)) if self.cfg.scope_policy == "related" else set()

        def scope_ok(path: str) -> str | None:
            if path in allowed or self.cfg.scope_policy == "free":
                return None
            if path in related:
                return None
            return f"{path} is outside this subtask's scope (allowed: {', '.join(sorted(allowed))}). Stay inside the allowed files."

        reply = self._coder(t, attempt, feedback, extras)
        res = None
        for fix in range(3):
            if isinstance(reply, str):
                return False, reply
            res = apply_edits(reply, ws.read, scope_ok)
            self.emit("apply", task=t["id"], ok=res.ok, errors=res.errors[:6], notes=res.notes[:6], files=list(res.files))
            if res.ok:
                break
            if fix == 2:
                return False, "Your edits could not be applied:\n" + "\n".join(res.errors[:5])
            repair = (feedback + "\n" if feedback else "") + "Your previous reply could not be applied:\n" + "\n".join(res.errors[:5]) \
                + "\nCopy the SEARCH text exactly from the file shown in the context. Your previous reply was:\n" + reply.raw[:2500]
            reply = self._coder(t, attempt, repair, extras)
        for path, text in res.files.items():
            ws.delete(path) if text is None else ws.write(path, text)
        changed = list(res.files)
        self.six.refresh(changed)
        live = [p for p in changed if ws.exists(p)]

        self.ctl.checkpoint()
        self.status("verifying")
        with self.tracer.span("Run checks", "step", agent="gates"):
            results = self.ver.run(self.cfg.verify.per_task, live, lambda ty, **kw: self.emit(ty, task=t["id"], **kw))
        self.status("running")
        blocking = [r for r in results if r.blocking]
        if blocking:
            return False, "Objective checks failed after your change:\n" + gates_text(results, only_blocking=True)

        self.ctl.checkpoint()
        self.status("reviewing")
        diff = ws.diff_since(cp, changed, context=8)
        usage = "\n".join(u for p in live for imp in self.six.importers_of(p)[:6] if imp not in changed and (u := self.ctx.usages(imp, p)))
        with self.stream("reviewer", t["id"]) as tok:
            rv = self.agents.review(self.request, t, diff, gates_text(results), res.notes, usage, self.emit, tok)
        self.status("running")
        crit = rv.get("criteria", [])
        issues = rv.get("issues", [])
        serious = [i for i in issues if i.get("severity") in ("blocker", "major")]
        accept = rv.get("verdict") == "accept" and all(c.get("met") for c in crit) and not serious
        self.emit("review", task=t["id"], verdict="accept" if accept else rv.get("verdict", "revise"), criteria=crit, issues=issues, feedback=rv.get("feedback_for_coder", ""))
        if accept:
            return True, reply.summary or t["title"]
        for q in rv.get("needs_info", [])[:2]:
            with self.stream("context", t["id"]) as tok:
                r = self.agents.resolve_doubt(q, t, self.emit, tok)
            if r["answered"]:
                extras.append(f'<resolved question="{q}">\n{r["answer"]}\nevidence: {", ".join(r["evidence"])}\n</resolved>')
            else:
                self.answers += self.ask([q], "reviewer", kind="doubt", context=f"{t['id']}: {t['title']}")
        parts = [rv.get("feedback_for_coder", "").strip()]
        parts += [f"- [{i.get('severity')}] {i.get('file', '')}: {i.get('description', '')} Fix: {i.get('fix', '')}" for i in issues if i.get("severity") != "minor"]
        parts += [f"- Criterion not met: {c.get('criterion')} ({c.get('evidence', '')})" for c in crit if not c.get("met")]
        return False, "Reviewer feedback:\n" + "\n".join(p for p in parts if p)

    # ------------------------------------------------------------------ whole-change verification
    def final_phase(self) -> None:
        for rnd in range(3):
            self.ctl.checkpoint()
            self.status("verifying", scope="final")
            changed = self.ws.changed_paths()
            if not changed:
                raise RunError("The agents finished without producing any change.")
            live = [p for p in changed if self.ws.exists(p)]
            results = self.ver.run(self.cfg.verify.final, live, lambda ty, **kw: self.emit(ty, task="final", **kw), "final")
            blocking = [r for r in results if r.blocking]
            problem = ""
            if blocking:
                problem = "Final verification failed:\n" + gates_text(results, only_blocking=True, max_each=2500)
            else:
                self.status("reviewing", scope="final")
                crit = [c for t in self.tasks if self.states.get(t["id"]) == "accepted" for c in t.get("acceptance_criteria", [])][:14]
                whole = {"id": "FINAL", "title": "Whole request", "description": self.request[:1500], "acceptance_criteria": crit or ["The request is fully implemented"],
                         "files_to_modify": changed[:12], "files_to_create": []}
                with self.stream("reviewer", "final") as tok:
                    rv = self.agents.review(self.request, whole, self.ws.diff(context=5), gates_text(results), ["Whole-request review across all subtasks."], "", self.emit, tok)
                serious = [i for i in rv.get("issues", []) if i.get("severity") in ("blocker", "major")]
                ok = rv.get("verdict") == "accept" and not serious and all(c.get("met") for c in rv.get("criteria", []))
                self.emit("review", task="final", verdict="accept" if ok else "revise", criteria=rv.get("criteria", []), issues=rv.get("issues", []), feedback=rv.get("feedback_for_coder", ""))
                if ok or rnd == 2:
                    if not ok:
                        self.emit("agent", agent="system", message="Final review still has concerns; they are shown on the approval screen")
                    self.final_results = results
                    return
                problem = "Whole-request review found problems:\n" + rv.get("feedback_for_coder", "") + "\n" + "\n".join(f"- {i.get('file', '')}: {i.get('description', '')}" for i in serious)
            if rnd == 2:
                raise RunError("Final verification still failing after repair attempts. Nothing was changed in your repository.\n" + problem[:1500])
            implicated = [p for p in live if p in problem] or live
            fix = {"id": f"FIX{rnd + 1}", "title": "Repair issues found in final verification", "description": problem[:1800],
                   "files_to_modify": implicated[: self.cfg.limits.max_files_per_task], "files_to_create": [], "files_to_read": [], "focus_symbols": [],
                   "acceptance_criteria": ["All verification gates pass", "The original request stays fully implemented"], "depends_on": []}
            self.tasks.append(fix)
            ok, why = self.run_task(fix, initial_feedback=problem)
            if not ok:
                raise RunError("Could not repair the final verification failure. Nothing was changed in your repository.\n" + why[:1200])

    # ------------------------------------------------------------------ approval + commit
    @staticmethod
    def _stats(diff: str) -> dict[str, tuple[int, int]]:
        out, cur = {}, None
        for ln in diff.split("\n"):
            if ln.startswith("+++ ") or ln.startswith("--- "):
                m = re.match(r"[+-]{3} (?:[ab]/)?(.+)", ln)
                if ln.startswith("+++ ") and m and m.group(1) != "/dev/null":
                    cur = m.group(1)
                elif ln.startswith("--- ") and m and m.group(1) != "/dev/null" and cur is None:
                    cur = m.group(1)
                if cur:
                    out.setdefault(cur, (0, 0))
            elif cur and ln.startswith("+"):
                out[cur] = (out[cur][0] + 1, out[cur][1])
            elif cur and ln.startswith("-"):
                out[cur] = (out[cur][0], out[cur][1] + 1)
        return out

    def approve_and_commit(self) -> None:
        ws = self.ws
        require = self.opts.get("require_approval", self.cfg.require_approval)
        while True:
            diff = ws.diff()
            files = ws.changed_paths()
            stats = self._stats(diff)
            ext = ws.external_changes()
            run_dir = self.cfg.forge_dir / "runs" / self.rid
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "changes.patch").write_text(diff, encoding="utf-8")
            self.mem.update_run(self.rid, files=files)
            if not require:
                break
            self.mem.update_run(self.rid, status="awaiting_approval")
            payload = {"kind": "approval", "files": [{"path": p, "added": stats.get(p, (0, 0))[0], "removed": stats.get(p, (0, 0))[1]} for p in files],
                       "partial": self.partial, "external": ext[:20], "gates": [r.public(600) for r in getattr(self, "final_results", [])]}
            self.emit("status", status="awaiting_approval")
            self.emit("approval", **payload)
            d = self.ctl.wait_for(payload)
            act = d.get("action")
            if act == "approve":
                if self.partial and not d.get("force"):
                    continue
                break
            if act == "reject":
                self.status("rejected")
                self.mem.update_run(self.rid, finished=time.time())
                self.cleanup(keep=False)
                return
            if act == "changes" and (d.get("text") or "").strip():
                self.plan_phase(followup=d["text"].strip())
                self.status("running")
                self.exec_phase([t for t in self.tasks if self.states.get(t["id"]) == "pending"])
                self.final_phase()
        self.status("committing")
        try:
            ws.commit()
        except ConflictError as e:
            self.emit("error", message=str(e) + ". Nothing was written. Re-run the request so it starts from your latest files.")
            self.mem.update_run(self.rid, status="conflict", error=str(e), finished=time.time())
            self.emit("status", status="conflict")
            return
        except TxError as e:
            self.emit("error", message=str(e))
            self.mem.update_run(self.rid, status="failed", error=str(e), finished=time.time())
            self.emit("status", status="failed")
            return
        with self.eng.lock:
            self.eng.ix.refresh(files)
            self.eng.mem.meta_set("project_card", self.eng.ix.repo_map(5000))
        self.remember(files)
        self.mem.update_run(self.rid, status="done", finished=time.time())
        self.emit("status", status="done")
        self.emit("committed", files=files)
        self.cleanup(keep=False)

    def remember(self, files: list[str]) -> None:
        try:
            lines = [f"{t['id']} {t['title']}: {t.get('summary', '')}" for t in self.mem.get_tasks(self.rid) if t["status"] == "accepted"]
            s = self.agents.summarize(self.request, lines, files, self.emit)
            hashes = lambda p: (self.eng.ix.db.execute("SELECT hash FROM files WHERE path=?", (p,)).fetchone() or [None])[0]  # noqa: E731
            for f in s.get("facts", []):
                paths = [p for p in f.get("paths", []) if hashes(p)]
                self.mem.add_fact(f.get("kind", "fact"), f["text"], f"run:{self.rid}", paths, {p: hashes(p) for p in paths}, self.rid)
            self.mem.add_fact("summary", f"{time.strftime('%Y-%m-%d')}: {self.request[:120]} - {s.get('summary', '')}", f"run:{self.rid}", files[:6], {p: hashes(p) for p in files[:6] if hashes(p)}, self.rid)
            self.mem.update_run(self.rid, summary=s.get("summary", ""))
            self.mem.export_markdown(hashes)
        except Exception:  # noqa: BLE001
            pass
