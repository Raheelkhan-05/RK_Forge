# forge/agents.py

"""The logical agents. All run on the same local model; they differ in prompt, tools and output contract."""
from __future__ import annotations

import posixpath
import re
from typing import Callable

from .config import Config
from .context import ContextBuilder
from .indexer import Index
from .llm import OllamaLLM, est_tokens
from .memory import Memory
from .workspace import TxError, Workspace, is_secret, norm_rel

Emit = Callable[..., None]

# ------------------------------------------------------------------------------------------------ schemas
S_STR = {"type": "string"}
S_STRS = {"type": "array", "items": S_STR}
PLAN_SCHEMA = {"type": "object", "properties": {
    "understanding": S_STR, "questions": S_STRS,
    "subtasks": {"type": "array", "items": {"type": "object", "properties": {
        "id": S_STR, "title": S_STR, "description": S_STR, "files_to_modify": S_STRS, "files_to_create": S_STRS,
        "files_to_read": S_STRS, "focus_symbols": S_STRS, "acceptance_criteria": S_STRS, "depends_on": S_STRS},
        "required": ["id", "title", "description", "files_to_modify", "acceptance_criteria"]}}},
    "required": ["understanding", "subtasks"]}
STEP_SCHEMA = {"type": "object", "properties": {
    "thought": S_STR,
    "action": {"type": "string", "enum": ["search", "find_symbol", "read_file", "grep", "importers", "list_dir", "finish"]},
    "arg": S_STR, "start": {"type": "integer"}, "end": {"type": "integer"},
    "answer": S_STR, "confidence": {"type": "string", "enum": ["high", "medium", "low"]}, "evidence": S_STRS},
    "required": ["thought", "action"]}
REVIEW_SCHEMA = {"type": "object", "properties": {
    "verdict": {"type": "string", "enum": ["accept", "revise", "blocked"]},
    "criteria": {"type": "array", "items": {"type": "object", "properties": {"criterion": S_STR, "met": {"type": "boolean"}, "evidence": S_STR}, "required": ["criterion", "met", "evidence"]}},
    "issues": {"type": "array", "items": {"type": "object", "properties": {
        "severity": {"type": "string", "enum": ["blocker", "major", "minor"]}, "file": S_STR, "description": S_STR, "fix": S_STR},
        "required": ["severity", "description"]}},
    "needs_info": S_STRS, "feedback_for_coder": S_STR},
    "required": ["verdict", "criteria", "issues", "feedback_for_coder"]}
SUMMARY_SCHEMA = {"type": "object", "properties": {
    "summary": S_STR,
    "facts": {"type": "array", "items": {"type": "object", "properties": {
        "kind": {"type": "string", "enum": ["fact", "decision", "convention", "gotcha"]}, "text": S_STR, "paths": S_STRS},
        "required": ["kind", "text"]}}}, "required": ["summary", "facts"]}

# ------------------------------------------------------------------------------------------------ prompts
PLANNER_SYS = """You are the Planning Agent of an automated software-engineering pipeline working on an existing repository that contains several sub-products (for example frontend, backend, admin, database).
You investigate the real code first, then break the request into subtasks that a separate Coding Agent will implement one at a time.

Principles:
- The number of subtasks must match the work. A one-line fix is ONE subtask. A cross-cutting feature may need many (database, backend, API client, frontend, admin, tests). Never pad and never merge unrelated work.
- Each subtask is the smallest change that can be verified on its own, touches at most {max_files} files, and names exact paths that exist in the repository (or new files in files_to_create).
- Order by dependency: database/schema -> backend -> API client -> UI -> tests. Use depends_on for real dependencies only.
- Never plan edits to code you have not seen. If something you need is missing, investigate more.
- acceptance_criteria must be observable in code, tests or build output ("saveGst() sends the gst field", not "works well").
- focus_symbols: names of the functions/components/classes that will change (helps the coder see the right code).
- If a business rule, naming decision or requirement is genuinely ambiguous and changes the implementation, list it in "questions" (short, specific, only blocking ones). Otherwise decide and proceed."""

INVESTIGATE_SYS = """You are a read-only code investigator. Each turn choose ONE action and I will run it and show you the result.
Actions: search(arg=keywords) | find_symbol(arg=name) | read_file(arg=path,start,end) | grep(arg=text) | importers(arg=path) | list_dir(arg=dir) | finish.
Rules: prefer find_symbol/search before read_file; read only the ranges you need; never guess a path, find it first. Use finish when you have enough evidence.
For finish: {goal_finish}"""

CODER_SYS = """You are the Coding Agent in an automated pipeline. You implement exactly ONE subtask in an existing codebase.

Hard rules:
1. Use ONLY code that is shown to you (<file>, <interface>, <usage>, <snippet>, <definition>, <lookup>, <answers_from_user>). Never invent functions, props, hooks, routes, columns, imports, packages or file paths.
2. If you need something that is not shown (a function's signature, a variable's meaning, a table column, an API response shape, a decision), do NOT guess. Reply with STATUS: NEED_INFO and list what you need. Asking is always better than guessing.
3. Change only what the subtask requires. Keep the existing style, naming, error handling and validation. Do not reformat or rename unrelated code. Never delete existing behavior unless the subtask says so.
4. Edit only files listed under "You may modify" or "You may create".
5. Write every line in full. Never write placeholders such as "...rest of code...".

Reply in EXACTLY this format (no other text):

STATUS: READY
SUMMARY: <one or two sentences: what you changed and why>
USES:
- <project symbol you call/import that is NOT defined in the code you wrote> @ <path where it is defined>
EDITS:
<<<<<<< SEARCH path/to/file.ext
<exact existing lines to replace, copied verbatim from the file; include 2-3 unchanged lines around the change so it is unique>
=======
<the new lines>
>>>>>>> REPLACE

To create a new file use an empty SEARCH section. To delete a file write a line: DELETE path/to/file.ext
USES lists only symbols defined in this repository (not libraries or language built-ins); write "- none" if there are none.

If you are unsure about anything:

STATUS: NEED_INFO
UNCERTAIN:
- LOOKUP symbol <name> in <path>      (ask to see a function/variable/component definition)
- LOOKUP file <path> <startLine>-<endLine>
- LOOKUP grep <text>
- ASK <a precise question that only a person can answer>
(No EDITS section when STATUS is NEED_INFO.)"""

REVIEWER_SYS = """You are the Reviewer Agent in an automated pipeline. You judge a code change that another agent made for one subtask. Be strict and concrete.

Check, in order:
1. For EACH acceptance criterion: is it actually satisfied by the diff? Quote the evidence (file and what the code does).
2. Invented or unverifiable things: identifiers, imports, props, routes, columns that the diff uses but that are not defined in the diff or shown in the code context.
3. Regressions: removed or weakened validation, error handling or existing behavior; callers (<usage> blocks) that would break.
4. Unrelated changes, reformatting, leftover debugging code, secrets.
5. The objective gate results (syntax, imports, lint, typecheck, tests, build). A failing gate that is not marked as pre-existing is a blocker.

verdict: "accept" only if every criterion is met and there are no blocker or major issues. "revise" otherwise, with feedback_for_coder that says exactly what to change. "blocked" only if the task cannot be completed without information nobody has provided (put it in needs_info).
feedback_for_coder must be specific enough that the coder can fix it without seeing your reasoning."""

SUMMARIZER_SYS = """You record durable project knowledge after a completed change. Output a 2-3 sentence summary and 0-5 facts worth remembering for FUTURE changes: where something lives, a convention that was followed or established, a decision and its reason, a pitfall discovered. Facts must be true of the code as it is now, specific, and include the file paths they concern. Do not record trivia or anything about this conversation."""


def fmt_task(t: dict) -> str:
    return (f'<subtask id="{t["id"]}" title="{t["title"]}">\n{t["description"]}\n'
            f'Acceptance criteria:\n' + "\n".join(f"- {c}" for c in t.get("acceptance_criteria", [])) +
            f'\nYou may modify: {", ".join(t.get("files_to_modify", [])) or "(none)"}'
            f'\nYou may create: {", ".join(t.get("files_to_create", [])) or "(none)"}'
            + (f'\nFocus on: {", ".join(t["focus_symbols"])}' if t.get("focus_symbols") else "") + "\n</subtask>")


class ToolBox:
    """Read-only investigation tools (index + staging workspace). Cannot modify anything."""

    def __init__(self, ix: Index, ws: Workspace, ctx: ContextBuilder):
        self.ix, self.ws, self.ctx = ix, ws, ctx

    def run(self, action: str, arg: str, start: int | None, end: int | None) -> str:
        arg = (arg or "").strip()
        try:
            if action == "search":
                hits = self.ix.search(arg, 8)
                return "\n".join(f"{h.path}:{h.start}-{h.end}  [{h.symbols}]" for h in hits) or "no matches"
            if action == "find_symbol":
                syms = self.ix.find_symbol(arg, limit=10)
                out = [f"{s.path}:{s.line}-{s.end_line}  {s.kind}  {s.sig}" for s in syms]
                refs = self.ix.refs_to(arg)
                if refs:
                    out.append("referenced in: " + ", ".join(sorted({p for p, _ in refs})[:8]))
                return "\n".join(out) or "symbol not found"
            if action == "read_file":
                path = norm_rel(arg)
                text = self.ws.read(path)
                if text is None:
                    near = [p for p in self.ix.all_paths() if p.endswith("/" + path.rsplit("/", 1)[-1])][:5]
                    return f"file not found: {path}. similar: {near}"
                lines = text.split("\n")
                if start is None and len(lines) > 160:
                    return f"{path} has {len(lines)} lines. Outline:\n{self.ctx.outline(path)}\nFirst 60 lines:\n" + "\n".join(f"{i}: {l}" for i, l in enumerate(lines[:60], 1))
                a = max(1, start or 1)
                b = min(len(lines), end or a + 119, a + 199)
                return "\n".join(f"{i}: {lines[i - 1]}" for i in range(a, b + 1))
            if action == "grep":
                return "\n".join(f"{p}:{i}: {l}" for p, i, l in self.ix.grep(arg, 20)) or "no matches"
            if action == "importers":
                path = norm_rel(arg)
                imps = [r for _, _, r, _ in self.ix.imports_of(path) if r]
                return f"imports: {imps}\nimported by: {self.ix.importers_of(path)}\ntests: {self.ix.tests_for(path)}"
            if action == "list_dir":
                d = arg.strip("/")
                kids = {}
                for p in self.ix.all_paths(indexed_only=False):
                    if d and not p.startswith(d + "/"):
                        continue
                    rest = p[len(d) + 1:] if d else p
                    name = rest.split("/")[0] + ("/" if "/" in rest else "")
                    kids[name] = kids.get(name, 0) + 1
                return "\n".join(f"{k}{'' if k.endswith('/') else ''}" + (f"  ({n} files)" if k.endswith("/") else "") for k, n in sorted(kids.items())[:80]) or "empty or unknown directory"
        except (TxError, ValueError) as e:
            return f"error: {e}"
        return "unknown action"


class Agents:
    def __init__(self, llm: OllamaLLM, cfg: Config, ix: Index, ws: Workspace, ctx: ContextBuilder, mem: Memory):
        self.llm, self.cfg, self.ix, self.ws, self.ctx, self.mem = llm, cfg, ix, ws, ctx, mem
        self.tools = ToolBox(ix, ws, ctx)

    # ------------------------------------------------------------------ investigation loop (planner + context agent)
    def investigate(self, role: str, goal: str, finish_hint: str, seed: str, max_steps: int, emit: Emit, on_token) -> tuple[list[str], dict | None]:
        budget_chars = int(self.cfg.roles[role].prompt_budget() * 3.0)
        obs: list[str] = []
        final = None
        sys_prompt = INVESTIGATE_SYS.format(goal_finish=finish_hint)
        seen_calls: set[tuple] = set()
        for step in range(1, max_steps + 1):
            hist = "\n\n".join(f"[{i}] {o}" for i, o in enumerate(obs, 1))
            if len(hist) > budget_chars:
                hist = "(older results trimmed)\n" + hist[-budget_chars:]
            msgs = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"{seed}\n\nGOAL: {goal}\n\nRESULTS SO FAR:\n{hist or '(none yet)'}\n\nChoose the next action ({step}/{max_steps})."}]
            d = self.llm.chat_json(role, msgs, STEP_SCHEMA, on_token)
            action = d.get("action", "finish")
            if action == "finish":
                final = d
                break
            key = (action, d.get("arg"), d.get("start"), d.get("end"))
            if key in seen_calls:
                obs.append(f"{action}({d.get('arg')}) was already run. Use the earlier result or try something different, or finish.")
                continue
            seen_calls.add(key)
            result = self.tools.run(action, d.get("arg", ""), d.get("start"), d.get("end"))
            emit("tool", agent=role, action=action, arg=d.get("arg", ""), thought=d.get("thought", "")[:300], result=result[:600])
            obs.append(f"{action}({d.get('arg', '')}{'' if d.get('start') is None else f', {d.get("start")}-{d.get("end")}'}):\n{result[:2400]}")
        return obs, final

    # ------------------------------------------------------------------ planner
    def plan(self, request: str, answers: list[str], emit: Emit, on_token, feedback: str = "") -> dict:
        card = self.mem.meta_get("project_card", "") or self.ix.repo_map(3500)
        facts = self.mem.recall(request, [], self.ctx._hash, 1800)
        seed = (f"<project_map>\n{card[:3500]}\n</project_map>\n"
                + ("<project_memory>\n" + "\n".join(f"- [{f['kind']}] {f['text']}" for f in facts) + "\n</project_memory>\n" if facts else "")
                + (f"<answers_from_user>\n" + "\n".join(f"- {a}" for a in answers) + "\n</answers_from_user>\n" if answers else "")
                + f"<request>\n{request}\n</request>")
        emit("agent", agent="planner", message="Investigating the codebase")
        obs, _ = self.investigate("planner", "Find every place in the code that must change to fulfil the request, and the code those places depend on or are used by.",
                                  "no extra fields needed.", seed, self.cfg.limits.planner_steps, emit, on_token)
        evidence = "\n\n".join(f"[{i}] {o}" for i, o in enumerate(obs, 1))
        evidence = evidence[-int(self.cfg.roles["planner"].prompt_budget() * 2.2):]
        sys_prompt = PLANNER_SYS.format(max_files=self.cfg.limits.max_files_per_task)
        user = f"{seed}\n\n<investigation>\n{evidence}\n</investigation>\n\n{feedback}\nProduce the plan as JSON."
        emit("agent", agent="planner", message="Writing the plan")
        return self.llm.chat_json("planner", [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}], PLAN_SCHEMA, on_token)

    def replan(self, request: str, task: dict, failure: str, done: list[str], emit: Emit, on_token) -> dict:
        sys_prompt = PLANNER_SYS.format(max_files=self.cfg.limits.max_files_per_task)
        user = (f"<request>\n{request}\n</request>\n<already_done>\n" + "\n".join(f"- {d}" for d in done) + "\n</already_done>\n"
                f"<failed_subtask>\n{fmt_task(task)}\n</failed_subtask>\n<why_it_failed>\n{failure[:2500]}\n</why_it_failed>\n"
                "Rewrite this failed subtask as one or more SMALLER, safer subtasks (new ids, only what is still missing). "
                "Keep paths exact. If the failure came from missing information, add it to questions. Return the plan JSON containing only the replacement subtasks.")
        emit("agent", agent="planner", message=f"Re-planning {task['id']} into smaller steps")
        return self.llm.chat_json("planner", [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}], PLAN_SCHEMA, on_token)

    def validate_plan(self, plan: dict, existing_ids: set[str] | None = None) -> tuple[list[dict], list[str]]:
        errs: list[str] = []
        raw = plan.get("subtasks", []) or []
        if len(raw) > self.cfg.limits.max_subtasks:
            errs.append(f"Too many subtasks ({len(raw)}); maximum is {self.cfg.limits.max_subtasks}. Merge related work.")
        out, ids = [], set(existing_ids or [])
        for i, t in enumerate(raw, 1):
            tid = (t.get("id") or f"T{i}").strip()
            if tid in ids:
                tid = f"{tid}_{i}"
            ids.add(tid)
            task = {"id": tid, "title": (t.get("title") or "").strip(), "description": (t.get("description") or "").strip(),
                    "acceptance_criteria": [c for c in t.get("acceptance_criteria", []) if c.strip()],
                    "depends_on": [d for d in t.get("depends_on", []) if d],
                    "focus_symbols": [s for s in t.get("focus_symbols", []) if s]}
            for key in ("files_to_modify", "files_to_create", "files_to_read"):
                paths = []
                for p in t.get(key, []) or []:
                    try:
                        np = norm_rel(p)
                    except TxError:
                        errs.append(f"{tid}: invalid path {p!r}")
                        continue
                    if is_secret(np):
                        errs.append(f"{tid}: {np} is a secret file and cannot be used")
                        continue
                    paths.append(np)
                task[key] = list(dict.fromkeys(paths))
            for p in list(task["files_to_modify"]):
                if not self.ws.exists(p):
                    near = [x for x in self.ix.all_paths() if x.endswith("/" + p.rsplit("/", 1)[-1])][:3]
                    errs.append(f"{tid}: files_to_modify '{p}' does not exist" + (f" (did you mean {near}?)" if near else " (new files belong in files_to_create)"))
            for p in task["files_to_create"]:
                if self.ws.exists(p):
                    errs.append(f"{tid}: files_to_create '{p}' already exists; put it in files_to_modify")
            for p in task["files_to_read"]:
                if not self.ws.exists(p):
                    task["files_to_read"].remove(p)
            if not task["title"] or not task["description"]:
                errs.append(f"{tid}: title and description are required")
            if not task["acceptance_criteria"]:
                errs.append(f"{tid}: at least one acceptance criterion is required")
            if len(task["files_to_modify"]) + len(task["files_to_create"]) > self.cfg.limits.max_files_per_task:
                errs.append(f"{tid}: touches {len(task['files_to_modify']) + len(task['files_to_create'])} files; split it (maximum {self.cfg.limits.max_files_per_task})")
            if not task["files_to_modify"] and not task["files_to_create"]:
                errs.append(f"{tid}: must modify or create at least one file")
            out.append(task)
        # dependency ordering
        by_id = {t["id"]: t for t in out}
        ordered, state = [], {}

        def visit(t):
            if state.get(t["id"]) == 2:
                return
            if state.get(t["id"]) == 1:
                errs.append(f"circular dependency at {t['id']}")
                return
            state[t["id"]] = 1
            for d in t["depends_on"]:
                if d in by_id:
                    visit(by_id[d])
            state[t["id"]] = 2
            ordered.append(t)
        for t in out:
            visit(t)
        for t in ordered:
            t["depends_on"] = [d for d in t["depends_on"] if d in by_id]
        return ordered, errs

    # ------------------------------------------------------------------ coder
    def coder_messages(self, request: str, task: dict, bundle_text: str, done: list[str], feedback: str, attempt: int) -> list[dict]:
        parts = [bundle_text,
                 f"<overall_request>\n{request[:1500]}\n</overall_request>"]
        if done:
            parts.append("<already_completed>\n" + "\n".join(f"- {d}" for d in done[-12:]) + "\n</already_completed>")
        if feedback:
            parts.append(f'<feedback attempt="{attempt}">\n{feedback}\n</feedback>')
        parts.append(fmt_task(task))
        parts.append("Implement the subtask now. If anything you need is not shown above, answer STATUS: NEED_INFO instead of guessing.")
        return [{"role": "system", "content": CODER_SYS}, {"role": "user", "content": "\n\n".join(parts)}]

    # ------------------------------------------------------------------ reviewer
    def review(self, request: str, task: dict, diff: str, gates_text: str, notes: list[str], usage_text: str, emit: Emit, on_token) -> dict:
        budget_chars = int(self.cfg.roles["reviewer"].prompt_budget() * 3.0)
        fixed = (f"<overall_request>\n{request[:1000]}\n</overall_request>\n{fmt_task(task)}\n<gate_results>\n{gates_text}\n</gate_results>\n"
                 + ("<notes>\n" + "\n".join(notes) + "\n</notes>\n" if notes else ""))
        room = max(2000, budget_chars - len(fixed) - 1500)
        d = diff if len(diff) <= room else diff[:room] + "\n[diff truncated]"
        usage = usage_text[:max(0, min(1800, room // 4))]
        user = fixed + (f"<callers>\n{usage}\n</callers>\n" if usage else "") + f"<diff>\n{d}\n</diff>\n\nReview this change."
        return self.llm.chat_json("reviewer", [{"role": "system", "content": REVIEWER_SYS}, {"role": "user", "content": user}], REVIEW_SCHEMA, on_token)

    # ------------------------------------------------------------------ doubt resolution (context agent)
    def resolve_doubt(self, question: str, task: dict, emit: Emit, on_token) -> dict:
        seed = f"{fmt_task(task)}\nA coding agent is blocked by this question:\n<question>\n{question}\n</question>"
        obs, final = self.investigate(
            "context", "Answer the question using ONLY evidence found in the code. If the code does not contain the answer, say so.",
            'answer = the answer; evidence = list of "path:startLine-endLine" proving it; confidence = high only when the code states it explicitly, low if you are guessing or the code does not say.',
            seed, self.cfg.limits.context_agent_steps, emit, on_token)
        if not final or not final.get("answer"):
            return {"answered": False, "answer": "", "evidence": []}
        good = []
        for ev in final.get("evidence", []):
            m = re.match(r"\s*([^\s:]+):(\d+)(?:-(\d+))?", ev)
            if not m:
                continue
            txt = self.ws.read(m.group(1))
            if txt is not None and int(m.group(2)) <= txt.count("\n") + 1:
                good.append(ev)
        ok = bool(good) and final.get("confidence") in ("high", "medium")
        return {"answered": ok, "answer": final["answer"], "evidence": good, "confidence": final.get("confidence")}

    # ------------------------------------------------------------------ summariser
    def summarize(self, request: str, task_lines: list[str], files: list[str], emit: Emit) -> dict:
        user = f"<request>\n{request[:1200]}\n</request>\n<completed_subtasks>\n" + "\n".join(task_lines) + "\n</completed_subtasks>\n<files_changed>\n" + "\n".join(files) + "\n</files_changed>"
        try:
            d = self.llm.chat_json("summarizer", [{"role": "system", "content": SUMMARIZER_SYS}, {"role": "user", "content": user}], SUMMARY_SCHEMA, None, retries=1)
        except Exception:  # noqa: BLE001
            return {"summary": "; ".join(task_lines)[:500], "facts": []}
        d["facts"] = [f for f in d.get("facts", [])[:5] if len(f.get("text", "")) > 15]
        return d
