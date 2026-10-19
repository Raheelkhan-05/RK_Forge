# forge/cli.py

"""Command line: forge serve | doctor | index | run"""
from __future__ import annotations

import argparse
import shutil
import sys
import threading
import time
import webbrowser
from pathlib import Path

from .config import Config
from .llm import OllamaLLM


def doctor(cfg: Config) -> int:
    ok = True
    print(f"Repository : {cfg.repo}")
    print(f"Profile    : {cfg.profile}  ({cfg.roles['coder'].model}, context {cfg.roles['coder'].num_ctx})")
    h = OllamaLLM(cfg).health()
    print(f"Ollama     : {'OK' if h['ok'] else 'PROBLEM'} {cfg.host}")
    if h.get("error"):
        print(f"             {h['error']}")
        ok = False
    for name, tool in (("git", "git"), ("node", "node"), ("npm", "npm")):
        found = shutil.which(tool)
        print(f"{name:<11}: {found or 'not found'}")
        if name == "git" and not found:
            ok = False
    for proj in ("frontend", "backend", "admin"):
        p = cfg.repo / proj
        if p.is_dir():
            nm = (p / "node_modules").is_dir()
            print(f"{proj:<11}: found, node_modules {'present' if nm else 'MISSING (run npm install so lint/test/build gates can run)'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="forge", description="Local transactional multi-agent coding system")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("serve", "doctor", "index", "run"):
        sp = sub.add_parser(name)
        sp.add_argument("--repo", default=".", help="path to the project (the folder that contains frontend/backend/admin)")
        if name == "serve":
            sp.add_argument("--port", type=int, default=8765)
            sp.add_argument("--no-browser", action="store_true")
        if name == "run":
            sp.add_argument("request")
            sp.add_argument("--auto-approve", action="store_true")
    args = ap.parse_args(argv)
    cfg = Config.load(args.repo)
    if args.cmd == "doctor":
        sys.exit(doctor(cfg))
    from .orchestrator import Engine
    engine = Engine(cfg)
    if engine.recovered:
        print(f"Recovered from an interrupted commit: rolled back {', '.join(engine.recovered)}")
    if args.cmd == "index":
        print(engine.reindex())
        return
    if args.cmd == "run":
        rid = engine.submit(args.request, {"require_approval": not args.auto_approve})
        seen = 0
        while True:
            for ev in engine.mem.events_since(rid, seen):
                seen = ev["id"]
                p = ev["payload"]
                if ev["type"] in ("status", "plan", "task", "review", "error", "committed", "question", "approval"):
                    print(f"[{ev['type']}] {p.get('status') or p.get('title') or p.get('message') or p.get('verdict') or ''}")
                if ev["type"] == "question":
                    ans = [input(f"  {q}\n  > ") for q in p["questions"]]
                    engine.answer(rid, ans)
                if ev["type"] == "approval":
                    if input("Approve and commit? [y/N] ").lower().startswith("y"):
                        engine.control(rid, "approve", force=True)
                    else:
                        engine.control(rid, "reject")
            run = engine.mem.get_run(rid)
            if run["status"] in ("done", "failed", "cancelled", "rejected", "conflict"):
                print("Result:", run["status"], run.get("error") or "")
                return
            time.sleep(0.5)
    import uvicorn
    from .server import create_app
    engine.reindex()
    url = f"http://127.0.0.1:{args.port}"
    print(f"Forge is running at {url}  (repo: {cfg.repo})")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(engine, args.port), host="127.0.0.1", port=args.port, log_level="warning")
