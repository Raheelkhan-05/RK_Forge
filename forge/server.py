# forge/server.py

"""Local HTTP API + UI host. Binds to 127.0.0.1 only; rejects foreign Host headers and non-JSON writes (CSRF / DNS-rebinding guard)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .chat import ChatService
from .config import Config, PROFILES
from .llm import OllamaLLM
from .orchestrator import Engine, TERMINAL
from .workspace import ConflictError, TxError

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
MEMORY_KINDS = ("fact", "preference", "decision", "project", "goal", "convention")


def create_app(engine: Engine, port: int = 8765) -> FastAPI:
    app = FastAPI(title="Forge", docs_url=None, redoc_url=None)
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}", "127.0.0.1", "localhost", "testserver"}

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.headers.get("host", "") not in allowed_hosts:
            return JSONResponse({"error": "forbidden host"}, status_code=403)
        if request.method in ("POST", "PUT", "DELETE") and "application/json" not in request.headers.get("content-type", ""):
            return JSONResponse({"error": "JSON body required"}, status_code=415)
        return await call_next(request)

    mem, cfg = engine.mem, engine.cfg
    chat: ChatService = getattr(engine, "chat", None) or ChatService(engine)
    engine.chat = chat

    def need_run(rid: str) -> dict:
        run = mem.get_run(rid)
        if not run:
            raise HTTPException(404, "run not found")
        return run

    def need_chat(cid: str) -> dict:
        conv = mem.get_conv(cid)
        if not conv:
            raise HTTPException(404, "chat not found")
        return conv

    async def body_of(request: Request) -> dict:
        try:
            data = await request.json()
        except ValueError:
            raise HTTPException(400, "Invalid JSON body")
        return data if isinstance(data, dict) else {}

    # ------------------------------------------------------------------ status / settings
    @app.get("/api/status")
    def status():
        llm = OllamaLLM(cfg)
        with engine.lock:
            stats = engine.ix.stats()
        return {"repo": str(cfg.repo), "ollama": llm.health(), "index": stats, "profile": cfg.profile,
                "profiles": PROFILES, "roles": {r: {"model": c.model, "num_ctx": c.num_ctx, "think": c.think} for r, c in cfg.roles.items()},
                "require_approval": cfg.require_approval, "limits": {"max_attempts": cfg.limits.max_attempts, "max_subtasks": cfg.limits.max_subtasks},
                "chat": {"auto_start": cfg.chat.auto_start}, "recovered": engine.recovered}

    @app.post("/api/reindex")
    def reindex():
        return engine.reindex()

    @app.put("/api/settings")
    async def settings(request: Request):
        body = await body_of(request)
        patch: dict = {}
        replace: tuple[str, ...] = ()
        if body.get("profile") in PROFILES:
            patch["profile"] = body["profile"]
            patch["models"] = {}
            replace = ("models",)          # drop a previously saved custom model, otherwise it would override the profile
        if body.get("model"):
            patch["profile"] = "custom"
            patch["models"] = {"default": str(body["model"])}
            replace = ("models",)
        if "num_ctx" in body:
            patch["context"] = {"num_ctx": int(body["num_ctx"])}
        if "require_approval" in body:
            patch["require_approval"] = bool(body["require_approval"])
        roles: dict = {}
        if "think" in body:
            roles.update({r: {"think": bool(body["think"])} for r in ("planner", "reviewer", "coder", "context")})
        if "think_chat" in body:
            roles["chat"] = {"think": bool(body["think_chat"])}
        if roles:
            patch["role"] = roles
        if "auto_start" in body:
            patch["chat"] = {"auto_start": bool(body["auto_start"])}
        cfg.save_settings(patch, replace)
        fresh = Config.load(cfg.repo)
        cfg.profile, cfg.roles, cfg.require_approval, cfg.chat = fresh.profile, fresh.roles, fresh.require_approval, fresh.chat
        return status()

    # ------------------------------------------------------------------ chats
    @app.get("/api/chats")
    def chats(q: str = ""):
        return chat.list_chats(q.strip() or None)

    @app.post("/api/chats")
    async def new_chat(request: Request):
        body = await body_of(request)
        return chat.new_chat(str(body.get("title") or "New chat"))

    @app.get("/api/chats/{cid}")
    def chat_get(cid: str):
        snap = chat.snapshot(cid)
        if snap is None:
            raise HTTPException(404, "chat not found")
        return snap

    @app.get("/api/chats/{cid}/messages")
    def chat_older(cid: str, before: int, limit: int = 100):
        need_chat(cid)
        return chat.older(cid, before, limit)

    @app.put("/api/chats/{cid}")
    async def chat_rename(cid: str, request: Request):
        need_chat(cid)
        body = await body_of(request)
        try:
            return chat.rename(cid, str(body.get("title", "")))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.delete("/api/chats/{cid}")
    def chat_delete(cid: str):
        need_chat(cid)
        chat.delete(cid)
        return {"ok": True}

    @app.post("/api/chats/{cid}/messages")
    async def chat_post(cid: str, request: Request):
        need_chat(cid)
        body = await body_of(request)
        try:
            m = await asyncio.to_thread(chat.post, cid, str(body.get("text", "")), str(body.get("mode", "chat")))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"message": m}

    @app.post("/api/chats/{cid}/stop")
    def chat_stop(cid: str):
        need_chat(cid)
        chat.stop(cid)
        return {"ok": True}

    @app.post("/api/chats/{cid}/retry")
    async def chat_retry(cid: str, request: Request):
        need_chat(cid)
        body = await body_of(request)
        try:
            chat.retry(cid, int(body.get("mid", 0)))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True}

    @app.post("/api/chats/{cid}/brief")
    async def chat_brief(cid: str, request: Request):
        need_chat(cid)
        body = await body_of(request)
        try:
            return await asyncio.to_thread(chat.brief_action, cid, str(body.get("action", "")), body.get("request"))
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.get("/api/chats/{cid}/events")
    async def chat_events(cid: str, request: Request):
        need_chat(cid)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def listener(ev):
            try:
                loop.call_soon_threadsafe(q.put_nowait, ev)
            except RuntimeError:
                pass

        unsubscribe = chat.hub.subscribe(cid, listener)

        async def gen():
            try:
                snap = await asyncio.to_thread(chat.snapshot, cid)
                if snap is None:
                    return
                yield "data: " + json.dumps({"seq": snap["seq"], "type": "snapshot", "payload": snap}, default=str) + "\n\n"
                last = snap["seq"]
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    if ev["seq"] <= last:
                        continue
                    last = ev["seq"]
                    yield "data: " + json.dumps(ev, default=str) + "\n\n"
            finally:
                unsubscribe()

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------------ runs
    @app.get("/api/runs")
    def runs():
        return mem.list_runs()

    @app.post("/api/runs")
    async def create_run(request: Request):
        body = await body_of(request)
        text = (body.get("request") or "").strip()
        if len(text) < 5:
            raise HTTPException(400, "Describe what you want in a sentence or two.")
        if any(r["status"] not in TERMINAL and r["status"] != "queued" for r in mem.list_runs(20)) and not body.get("queue"):
            raise HTTPException(409, "Another run is in progress. Wait for it to finish or cancel it.")
        opts = {k: body[k] for k in ("require_approval", "max_attempts") if k in body}
        return {"id": engine.submit(text, opts)}

    @app.get("/api/runs/{rid}")
    def run(rid: str):
        r = need_run(rid)
        r["pending"] = engine.pending(rid)
        r["live"] = rid in engine.live
        return r

    @app.get("/api/runs/{rid}/events")
    async def events(rid: str, request: Request, after: int = 0):
        need_run(rid)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def listener(ev):
            if ev["run_id"] == rid:
                try:
                    loop.call_soon_threadsafe(q.put_nowait, ev)
                except RuntimeError:
                    pass

        mem.listeners.append(listener)

        async def gen():
            last = after
            try:
                for ev in mem.events_since(rid, after, 5000):
                    last = ev["id"]
                    yield f"id: {ev['id']}\ndata: {json.dumps(ev)}\n\n"
                yield "event: ready\ndata: {}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    if ev["id"] and ev["id"] <= last:
                        continue
                    if ev["id"]:
                        last = ev["id"]
                    yield (f"id: {ev['id']}\n" if ev["id"] else "") + f"data: {json.dumps(ev)}\n\n"
            finally:
                if listener in mem.listeners:
                    mem.listeners.remove(listener)

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/runs/{rid}/answer")
    async def answer(rid: str, request: Request):
        need_run(rid)
        body = await body_of(request)
        if not engine.answer(rid, [str(a) for a in body.get("answers", [])]):
            raise HTTPException(409, "This run is not waiting for an answer")
        return {"ok": True}

    @app.post("/api/runs/{rid}/control")
    async def control(rid: str, request: Request):
        need_run(rid)
        body = await body_of(request)
        action = body.get("action", "")
        if action == "resume":
            try:
                engine.resume(rid)
            except ValueError as e:
                raise HTTPException(409, str(e))
            return {"ok": True}
        if action not in ("pause", "continue", "cancel", "approve", "reject", "changes", "retry", "skip", "abort"):
            raise HTTPException(400, "unknown action")
        if not engine.control(rid, action, text=body.get("text", ""), force=bool(body.get("force"))):
            raise HTTPException(409, "Nothing is waiting for that action right now")
        return {"ok": True}

    @app.get("/api/runs/{rid}/diff")
    def diff(rid: str):
        need_run(rid)
        live = engine.live.get(rid)
        if live and live.ws:
            return PlainTextResponse(live.ws.diff(context=4))
        p = cfg.forge_dir / "runs" / rid / "changes.patch"
        return PlainTextResponse(p.read_text(encoding="utf-8") if p.is_file() else "")

    @app.post("/api/runs/{rid}/undo")
    def undo(rid: str):
        r = need_run(rid)
        if r["status"] != "done":
            raise HTTPException(409, "Only committed runs can be undone")
        try:
            engine.undo(rid)
        except ConflictError as e:
            raise HTTPException(409, f"Cannot undo safely: {e}")
        except (TxError, OSError) as e:
            raise HTTPException(500, str(e))
        return {"ok": True}

    # ------------------------------------------------------------------ memory
    def current_hash(p: str):
        return (engine.ix.db.execute("SELECT hash FROM files WHERE path=?", (p,)).fetchone() or [None])[0]

    @app.get("/api/memory")
    def memory():
        with engine.lock:
            return mem.list_facts(current_hash)

    @app.post("/api/memory")
    async def memory_add(request: Request):
        body = await body_of(request)
        text = " ".join(str(body.get("text", "")).split())
        kind = body.get("kind") if body.get("kind") in MEMORY_KINDS else "fact"
        if not 8 <= len(text) <= 400:
            raise HTTPException(400, "Write a note between 8 and 400 characters.")
        fid = mem.add_fact(kind, text, "user")
        return {"id": fid}

    @app.put("/api/memory/{fid}")
    async def memory_edit(fid: str, request: Request):
        body = await body_of(request)
        if not mem.update_fact(fid, str(body.get("text", ""))):
            raise HTTPException(404, "memory item not found")
        return {"ok": True}

    @app.delete("/api/memory/{fid}")
    def forget(fid: str):
        mem.forget(fid)
        return {"ok": True}

    # ------------------------------------------------------------------ UI
    @app.get("/")
    def index():
        return FileResponse(UI_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")
    return app