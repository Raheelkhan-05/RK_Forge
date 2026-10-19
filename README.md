# Forge

A coding assistant that runs entirely on my own machine. It uses Ollama and Qwen3.5, so there are no API keys, no usage bills, and my code never leaves the laptop.

I built it for my own project, which is a monorepo with three parts (frontend, backend, admin) and somewhere around 30 to 50k lines. I wanted something that could change code across all three without me babysitting it, and I didn't want it to break things when it got confused. Most of Forge is built around that second part.

## What it does

You talk to it like you would talk to a person. Ask how something works, compare two ideas, think out loud. When you actually want a change made, it plans the work, makes the edits in a private copy of your repo, runs your project's own checks, has a second pass review the result, and only writes to your real files after you approve. If anything goes wrong along the way, your files stay exactly as they were.

## Features

### Chat first

- **Talk before building.** Nothing touches your code just because you mentioned an idea. Questions and "what if" messages stay as conversation.
- **It reads your intent.** Every message is checked two ways: simple wording rules (hedging, questions, "go ahead") and a small model call. A controller combines them. A task never starts from a half-formed idea.
- **Unclear requests get questions.** If a change request is missing something important, it asks specific questions instead of guessing.
- **Clear requests become a brief.** You get a card with a title, the full request, the assumptions it made and anything still open. Start it, edit it, or say "not yet".
- **Saying "go ahead" works.** If you clearly say "do it" and nothing is ambiguous, it starts straight away. You can turn this off in Settings so it always asks first.
- **Answers are based on your real code.** Name a file or a function and it opens it from the index. Broader questions trigger a short read-only look through the repo. The chat shows what it looked at.
- **Long chats don't fall apart.** Each reply is built from saved memory, a rolling summary of the older conversation, older messages found by full-text search, and the newest messages as they are. So "what did we name that job earlier?" still works fifty messages later.
- **Chats behave like you'd expect.** Chat list, search, rename, delete, automatic titles, and a draft that is saved per chat so a refresh never eats what you were typing.
- **Slash commands.** `/task ...` skips the discussion, `/remember ...` saves a note, `/forget ...` removes one.

### Memory

Forge keeps durable facts from your chats: preferences, decisions, project knowledge, goals and conventions. They are pulled out in the background, shown in the Memory tab (add, search, edit, forget) and brought back whenever they're relevant, in any chat and inside the planner and coder prompts.

- A correction replaces the old fact instead of piling up next to it.
- Anything that looks like a key, token or password is never saved.
- Facts learned from code are tied to the hash of the files they came from. When a file changes, the fact is treated as outdated and stops being used.
- Everything is also exported as readable markdown in `.forge/memory/`.

### The task pipeline

1. **Planner** investigates the repo with read-only tools, then writes a plan. The number of subtasks depends on the work. A one-line fix is one subtask, a feature that touches the database, backend and UI can be many. Plans are validated before anything runs (paths must exist, no circular dependencies, file limits per subtask).
2. **Context agent** builds the prompt for each step from the index: the files to change, the interfaces they import, real call sites, related tests, search hits and remembered facts. Big files are shown as windows with an outline of what was left out.
3. **Coder** edits with SEARCH/REPLACE blocks. If it isn't sure about a function, a variable or a table column, it has to say so instead of inventing one. Every symbol it relies on is checked against the index.
4. **Checks** run on the copy: syntax, imports, lint, typecheck, related tests and build. A check that was already failing before the change is reported but not blamed on the agent.
5. **Reviewer** goes through each acceptance criterion against the diff and the check output. Failures go back to the coder, then the step is split into smaller ones, and if that still fails it comes to you.
6. **Final pass.** A whole-request review and the full set of checks run, then you approve, then it is applied.

When something is unclear, questions go to the context agent first. It tries to answer from the code. Only if the code doesn't say, you get asked.

### Safety and atomic writes

- All edits happen in an isolated copy of the repo. `node_modules` is linked, not copied.
- Every retry starts from a checkpoint, so failed attempts leave nothing behind.
- Applying is a journaled all-or-nothing write. Before anything changes it checks that every target file is still what the run started from. Originals are backed up, new versions are written to temp files and swapped in. If any step fails, everything is rolled back. If the machine dies halfway, the rollback happens on the next start.
- If you edited an affected file while a task was running, the apply is refused and nothing is written.
- Undo restores the originals, as long as those files haven't been edited since.
- Files like `.env`, keys and certificates are never read into prompts and never written.
- The server only listens on `127.0.0.1`, rejects other Host headers and only accepts JSON for writes. Model output is shown as plain text, never as HTML.
- Agents can't run arbitrary commands. Only your project's own lint, test and build commands run, and only inside the copy.

One honest limit: no filesystem gives a single atomic multi-file write. This is all-or-nothing with crash recovery, which is close, but keep git as your outer safety net.

### Pipeline view

There's a Pipeline tab that shows what actually happens to a message, step by step. It was added because the whole thing felt slow and I wanted to know why.

- One node per stage (wording check, router, controller, context, chat model, and for tasks the planner, coder, checks and reviewer). The active one glows while it is running.
- A timeline with true start times for every step. Each model call is split into waiting for Ollama, loading the model, reading the prompt and writing the answer. Those numbers come from Ollama's own timings, nothing is estimated for the animation.
- A line at the top saying which step used the most time and what the main reason was.
- Click any step to see what was sent to the model, what came back, tokens per second and the full breakdown.
- The header shows how much of the model is sitting on the GPU, which turned out to matter a lot on my machine.

### Other things worth mentioning

- **Code index.** SQLite with full-text search. It knows symbols, the import graph, exports, API routes and Supabase table and RPC references. JS, TS, JSX and TSX are handled with a brace-aware scanner, Python with `ast`, SQL with a scanner that understands dollar quoting. Refreshes only re-read files that changed.
- **Pause, cancel, resume.** A running task can be paused or cancelled. If the server restarts mid-run, the task is marked interrupted and can be resumed.
- **Request changes after review.** Instead of applying or discarding, you can ask for a follow-up and it plans just that.
- **Light and dark theme**, and the layout works on a narrow window.

## Setup (Windows)

You need Python 3.12 or newer (`agents.py` uses nested quotes inside f-strings, which 3.11 rejects), Ollama, Git, and Node if your project has lint, test or build scripts.

```
pip install -r requirements.txt
ollama pull qwen3.5:9b
ollama pull qwen3.5:4b
```

In your target project, run `npm install` once in `frontend/`, `backend/` and `admin/` so the checks can run.

Then check everything is wired up:

```
python -m forge doctor --repo C:\path\to\project
```

Start it:

```
start.bat C:\path\to\project
```

That opens `http://127.0.0.1:8765`.

Other commands:

```
python -m forge serve --repo . --port 8765 --no-browser
python -m forge index --repo .
python -m forge run "your request" --repo . --auto-approve
```

`run` skips the UI and works from the terminal. Without `--auto-approve` it asks before applying.

## Choosing a model

There are two profiles, switchable in Settings or with `profile = "fast"` in the config.

| Profile | Model | Notes |
| --- | --- | --- |
| quality | `qwen3.5:9b` | Better at code, slower |
| fast | `qwen3.5:4b` | Fits a small GPU much better |

Forge uses one model and one context size for every role. Ollama keeps a single model loaded, and switching models or context size forces a reload, which is painful on limited memory. I run this on 16 GB of RAM with 4 GB of VRAM, so the 9B spills into system RAM and is noticeably slow. The 4B is the one that mostly fits on the GPU. If it feels slow, open the Pipeline tab and check the GPU percentage and the tokens per second before changing anything else.

## Configuration

Copy `config.example.toml` to `<your-project>/.forge/config.toml`. Settings changed in the UI are saved to `.forge/settings.json` and override the file.

A few settings that matter:

- `scope_policy`: `strict`, `related` or `free`. Controls which files a subtask is allowed to edit.
- `require_approval`: whether anything is written without you approving it.
- `[chat]`: `auto_start`, `research_steps` (how many read-only lookups a chat answer may make), `memory_chars`, `summary_trigger`, `keep_recent`.
- `[limits]`: `max_attempts` per subtask, `max_files_per_task`, `max_subtasks`, `max_replans`.
- `[verify.commands.<project>]`: set your own lint, test or build command for `frontend`, `backend` or `admin` instead of relying on auto-detection.
- `[aliases]`: only needed if your `@/` style import aliases aren't detected from tsconfig or vite.

## Where things live

```
forge/
  chat.py          chat service: routing, briefs, context assembly, background jobs
  intent.py        simple wording checks (question, hedge, go-ahead, confirm)
  orchestrator.py  runs a task: plan, execute, verify, review, approve, apply
  agents.py        planner, coder prompts, reviewer, context agent
  context.py       builds the prompt for the coder within the token budget
  indexer.py       code index (symbols, imports, search)
  patching.py      parses and applies SEARCH/REPLACE edits
  verify.py        syntax, imports, lint, typecheck, test, build checks
  workspace.py     isolated copy, journaled apply, rollback, undo
  memory.py        facts, chats, messages, run history (SQLite)
  llm.py           Ollama client (streaming, JSON output, cancel, retries)
  telemetry.py     real timings behind the Pipeline view
  server.py        local API and UI host
  cli.py           serve, doctor, index, run
ui/                plain HTML, CSS and JS, no build step
tests/             pytest
```

Runtime data goes into a `.forge/` folder inside the project Forge is working on: the memory database, the code index, settings, journals for rollback and run history.

## Tests

```
python -m pytest tests -q
```

The chat tests use a scripted stand-in for Ollama, so they don't need a model or a repo.

## Known limits

- It's only as good as a 4B or 9B local model. Expect to review the diffs, especially on bigger changes.
- Speed depends heavily on your GPU. On a small card the 9B is slow.
- JS and TS parsing is a scanner, not a full parser, so unusual syntax can occasionally confuse the index.
- The apply step is all-or-nothing with crash recovery, not a true atomic write.
- Windows is the platform I use and test on.