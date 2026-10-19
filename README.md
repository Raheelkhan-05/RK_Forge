# Forge: a local, transactional, multi-agent coding system

Ollama + Qwen3.5, no paid APIs. One model plays eight roles; a controller (not the model) enforces the process.
You **talk first** (explain, compare, brainstorm, ask follow-ups for as long as you like), and a coding task starts only when your wording and a confirmation say so.

## Quick start (Windows)
```
pip install -r requirements.txt          (Python 3.12+)
ollama pull qwen3.5:9b                   (and/or qwen3.5:4b)
npm install                              (once, in frontend/, backend/, admin/ so lint/test/build gates can run)
python -m forge doctor --repo C:\path\to\project
start.bat C:\path\to\project             (opens http://127.0.0.1:8765)
```
Headless: `python -m forge run "your request" --repo . ` · Tests: `python -m pytest tests -q`

> Python 3.12+ is required (agents.py uses nested quotes inside f-strings, which 3.11 rejects).

## Chat mode

* **Chats** work like ChatGPT/Claude: a list on the left, search across titles and message text, rename, delete. A new chat is created when you send the first message and is **named automatically** from the first exchange (refined after the third message). A name you set yourself is never overwritten.
* **No fixed number of messages before a task.** Every message is read two ways: deterministic wording signals (`forge/intent.py`) and a small JSON "router" call. A controller combines them:
  * questions, "what if / should we / explain" wording, or a low-confidence reading → **discuss** (never a task)
  * a change request with missing pieces → **asks you specific questions**, starts nothing
  * a clear change request → a **task brief card** (title, full request, assumptions, still-open points). Press *Start task*, edit it first, say "go ahead", or say "not yet"
  * "do it / go ahead / start" said clearly, with high confidence, no hedging and no open questions → starts straight away (switch off in *Settings*)
  * "yes" / "go ahead" with nothing pending is never treated as a command to build something
  * *Make this a task* under any answer turns the discussion so far into a brief; *Run as task* (or `/task ...`) skips the discussion
* **Answers are grounded in your real code.** File and function names you mention are opened instantly from the index; broader questions trigger a short read-only investigation. The chat shows what was looked at.
* **Follow-ups at any depth.** Each reply is built from: recalled memory facts, a rolling summary of the older conversation, a digest of anything not yet summarised, older messages retrieved by full-text search (so "what was that job we named earlier?" works after 50 exchanges), the newest messages verbatim, the state of any running task, and the diff of the last finished task when you ask about changes. Everything is packed into the model's token budget.
* **Tasks live inside the chat.** A task card shows progress; questions, approvals and blockers appear in a panel above the message box; the finished result is written back into the conversation so you can ask "why did you change X?".

## Memory (facts, like Claude)

Durable facts (preferences, decisions, project knowledge, goals, conventions) are extracted from your chats in the background, shown in the **Memory** tab (add, search, forget), and recalled whenever they are relevant, in every chat and in the planner/coder prompts. A correction supersedes the old fact; "forget ..." removes it; `/remember ...` and `/forget ...` work as commands. Secrets (keys, tokens, passwords) are never saved. Facts derived from code are tied to file hashes and expire when the file changes.

## 9B or 4B?
Switch in **Settings** (or `profile = "fast"`). Weights are about 6.6 GB (9B) and 3.4 GB (4B) at Q4. With 4 GB VRAM the 4B
is the only one that mostly fits on the GPU; the 9B spills into system RAM and is slower but generally more reliable at code.
Forge uses one model and one context size for every role because Ollama keeps a single model resident (a different `num_ctx` also forces a reload).
Replies stream token by token; background work (title, summary, memory) pauses the moment you send a message.

## The task pipeline
1. **Planner** investigates with read-only tools, then emits a validated plan with as many subtasks as the work needs. Blocking ambiguities go to you first.
2. **Context agent** builds each coder prompt from a SQLite/FTS5 index (symbols, import graph, call sites, tests, remembered facts) within the token budget. Big files are windowed.
3. **Coder** edits with SEARCH/REPLACE blocks and must say `NEED_INFO` instead of guessing; every symbol it relies on is checked against the index.
4. **Gates** (objective): syntax, imports, lint, typecheck, related tests, build. A gate that already failed before the change is not blamed on the agent.
5. **Reviewer** checks each acceptance criterion against the diff and gate output; failures go back to the coder, then the step is re-planned smaller, then you decide.
6. **Whole-request review + final gates**, then your approval, then the commit.

## Atomicity (ACID-style)
* All edits happen in an isolated copy (`node_modules` linked, not copied). Failed attempts are rolled back to a checkpoint.
* Commit = journaled transaction: verify every target still equals the version the run started from, back up originals, write temp files, swap them in, fsync. Any error rolls **all** files back; a crash mid-commit is rolled back on next start.
* If you edited an affected file during the run, the commit is refused (nothing written). **Undo** restores the originals if those files were not edited since.
Honest limit: no filesystem offers a single atomic multi-file write; this gives all-or-nothing with crash recovery. Keep git as your outer safety net.

## Data (`.forge/`)
`memory.db` holds facts, **conversations and messages** (with a full-text index), run history and the event log. `.forge/memory/*.md` is a readable export of the facts.

## Safety
Binds to 127.0.0.1, rejects foreign Host headers and non-JSON writes. Model output is rendered as text only (a small markdown renderer builds DOM nodes; no `innerHTML`). Secret-like files (`.env*`, keys) are never read into prompts or written. Agents cannot run arbitrary commands: only your project's own lint/test/build commands run, inside the copy.

## What changed in this version
* new: `forge/chat.py`, `forge/intent.py`, `tests/test_chat.py`
* updated: `config.py` (chat + router roles, `[chat]` settings, fix: switching from a custom model back to a profile now works), `llm.py` (`chat_stream`), `memory.py` (conversations, messages, full-text search, thread-safe reads), `server.py` (chat endpoints, live event stream, memory add/edit), `ui/*` (chat-first interface)
* fixed: the message box cleared itself while typing. The old UI rebuilt the whole page (including the text box) every few seconds. The new composer is never re-rendered, and drafts are saved per chat.
* unchanged: orchestrator, agents, context, indexer, patching, verify, workspace, cli