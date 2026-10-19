# forge/memory.py

"""Persistent memory (SQLite): facts, conversations + messages, run + task history, event log.

Facts that were derived from code carry the hash of the files they came from; once a file changes the fact is
treated as stale and is not injected again, so the agents never rely on outdated knowledge.
Facts learned in a chat (preferences, decisions, goals...) carry no hashes: they stay valid until the user
corrects or forgets them, and they are recalled into later chats and into the planner/coder prompts.

Conversations keep every message (with a full-text index), a rolling summary of the older part, and a small
JSON state (pending task brief, active run) so a chat can always be resumed exactly where it stopped.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from .config import Config
from .indexer import query_terms, split_ident

KIND_WEIGHT = {"answer": 3.0, "decision": 2.5, "preference": 2.5, "convention": 2.0, "gotcha": 2.0, "goal": 2.0,
               "project": 1.6, "fact": 1.2, "summary": 1.0}
CHAT_KINDS = ("preference", "decision", "project", "goal", "convention", "fact")
ALWAYS_ELIGIBLE = ("convention", "answer", "decision")   # recalled for code tasks even without a keyword match

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts(id TEXT PRIMARY KEY, kind TEXT, text TEXT, source TEXT, paths TEXT, hashes TEXT, run_id TEXT, created REAL, active INT DEFAULT 1);
CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, request TEXT, status TEXT, created REAL, finished REAL, summary TEXT, options TEXT, plan TEXT, error TEXT, files TEXT);
CREATE TABLE IF NOT EXISTS tasks(run_id TEXT, tid TEXT, idx INT, title TEXT, status TEXT, attempts INT, files TEXT, summary TEXT, PRIMARY KEY(run_id, tid));
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts REAL, type TEXT, payload TEXT);
CREATE INDEX IF NOT EXISTS ev_run ON events(run_id, id);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS conversations(id TEXT PRIMARY KEY, title TEXT, title_locked INT DEFAULT 0, created REAL, updated REAL,
  summary TEXT DEFAULT '', summary_upto INT DEFAULT 0, extracted_upto INT DEFAULT 0, titled_at INT DEFAULT 0, state TEXT DEFAULT '{}');
CREATE INDEX IF NOT EXISTS conv_updated ON conversations(updated);
CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT, conv_id TEXT, role TEXT, kind TEXT, content TEXT, meta TEXT, ts REAL);
CREATE INDEX IF NOT EXISTS msg_conv ON messages(conv_id, id);
CREATE VIRTUAL TABLE IF NOT EXISTS msg_fts USING fts5(body, conv_id UNINDEXED, tokenize='unicode61');
"""

CONV_COLS = "id,title,title_locked,created,updated,summary,summary_upto,extracted_upto,titled_at,state"
MSG_COLS = "id,conv_id,role,kind,content,meta,ts"


class _Rows:
    """Rows fetched while the lock was held, so a cursor is never read from two threads at once."""
    __slots__ = ("rows", "lastrowid", "rowcount", "_i")

    def __init__(self, rows, lastrowid, rowcount):
        self.rows, self.lastrowid, self.rowcount, self._i = rows, lastrowid, rowcount, 0

    def fetchone(self):
        if self._i < len(self.rows):
            self._i += 1
            return self.rows[self._i - 1]
        return None

    def fetchall(self):
        out = self.rows[self._i:]
        self._i = len(self.rows)
        return out

    def __iter__(self):
        return iter(self.fetchall())


def _fts_body(text: str) -> str:
    text = (text or "")[:6000]
    words: set[str] = set()
    for ident in _idents(text):
        words.update(split_ident(ident))
    return text + "\n" + " ".join(sorted(words))


def _idents(text: str) -> list[str]:
    import re
    return re.findall(r"[A-Za-z_$][\w$]{2,}", text)


def _fts_query(text: str, limit: int = 14) -> str:
    terms = query_terms(text, limit)
    return " OR ".join('"%s"' % t.replace('"', "") for t in terms)


class Memory:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        cfg.forge_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cfg.forge_dir / "memory.db", check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.executescript(SCHEMA)
        self.lock = threading.RLock()
        self.listeners: list[Callable[[dict], None]] = []

    # ---------------------------------------------------------------- generic
    def _x(self, sql: str, args=()) -> _Rows:
        with self.lock:
            cur = self.db.execute(sql, args)
            rows = cur.fetchall() if cur.description else []
            return _Rows(rows, cur.lastrowid, cur.rowcount)

    def meta_get(self, key: str, default=None):
        r = self._x("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r[0] if r else default

    def meta_set(self, key: str, value: str) -> None:
        self._x("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    # ---------------------------------------------------------------- facts
    def add_fact(self, kind: str, text: str, source: str, paths: list[str] | None = None,
                 hashes: dict[str, str] | None = None, run_id: str | None = None) -> str | None:
        text = " ".join(text.split()) if "\n" not in text.strip() else text.strip()
        if not text:
            return None
        with self.lock:
            dup = self._x("SELECT id FROM facts WHERE active=1 AND kind=? AND text=?", (kind, text)).fetchone()
            if dup:
                return dup[0]
            if kind in CHAT_KINDS:
                near = self._near_duplicate(kind, text)
                if near:
                    self._x("UPDATE facts SET created=? WHERE id=?", (time.time(), near))
                    return near
            fid = uuid.uuid4().hex[:10]
            self._x("INSERT INTO facts VALUES(?,?,?,?,?,?,?,?,1)",
                    (fid, kind, text, source, json.dumps(paths or []), json.dumps(hashes or {}), run_id, time.time()))
            return fid

    def _near_duplicate(self, kind: str, text: str) -> str | None:
        mine = set(query_terms(text, 40))
        if not mine:
            return None
        for fid, other in self._x("SELECT id,text FROM facts WHERE active=1 AND kind=?", (kind,)):
            theirs = set(query_terms(other, 40))
            if theirs and len(mine & theirs) / len(mine | theirs) >= 0.8:
                return fid
        return None

    def get_fact(self, fid: str) -> dict | None:
        r = self._x("SELECT id,kind,text,source,paths,hashes,run_id,created FROM facts WHERE id=? AND active=1", (fid,)).fetchone()
        return self._fact(r) if r else None

    @staticmethod
    def _fact(r) -> dict:
        return {"id": r[0], "kind": r[1], "text": r[2], "source": r[3], "paths": json.loads(r[4] or "[]"),
                "hashes": json.loads(r[5] or "{}"), "run_id": r[6], "created": r[7], "stale": False}

    def update_fact(self, fid: str, text: str, kind: str | None = None) -> bool:
        text = " ".join(text.split())
        if not text:
            return False
        if kind:
            return self._x("UPDATE facts SET text=?, kind=? WHERE id=? AND active=1", (text, kind, fid)).rowcount > 0
        return self._x("UPDATE facts SET text=? WHERE id=? AND active=1", (text, fid)).rowcount > 0

    def list_facts(self, include_stale_check: Callable[[str], str | None] | None = None) -> list[dict]:
        out = []
        for r in self._x("SELECT id,kind,text,source,paths,hashes,run_id,created FROM facts WHERE active=1 ORDER BY created DESC"):
            f = self._fact(r)
            if include_stale_check:
                f["stale"] = self._stale(f, include_stale_check)
            out.append(f)
        return out

    @staticmethod
    def _stale(fact: dict, current_hash: Callable[[str], str | None]) -> bool:
        return any(current_hash(p) != h for p, h in fact["hashes"].items())

    def forget(self, fact_id: str) -> None:
        self._x("UPDATE facts SET active=0 WHERE id=?", (fact_id,))

    def recall(self, query: str, paths: list[str], current_hash: Callable[[str], str | None], max_chars: int = 2400,
               always: tuple[str, ...] = (), limit: int = 14) -> list[dict]:
        """Facts relevant to `query` (and to the files being edited). Kinds in `always` are eligible without a keyword match."""
        terms = set(query_terms(query, 40))
        now = time.time()
        scored = []
        for f in self.list_facts(current_hash):
            if f["stale"]:
                continue
            overlap = len(terms & set(query_terms(f["text"], 40)))
            path_hit = len(set(paths) & set(f["paths"]))
            if overlap == 0 and path_hit == 0 and f["kind"] not in ALWAYS_ELIGIBLE and f["kind"] not in always:
                continue
            recency = 0.4 / (1.0 + max(0.0, now - f["created"]) / (30 * 86400))
            score = KIND_WEIGHT.get(f["kind"], 1.0) * 0.5 + overlap * 1.0 + path_hit * 2.5 + recency
            scored.append((score, f))
        scored.sort(key=lambda t: -t[0])
        out, used = [], 0
        for _, f in scored:
            if len(out) >= limit or used + len(f["text"]) > max_chars:
                continue
            out.append(f)
            used += len(f["text"]) + 4
        return out

    # ---------------------------------------------------------------- runs / tasks
    def create_run(self, request: str, options: dict) -> str:
        rid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:5]
        self._x("INSERT INTO runs(id,request,status,created,options) VALUES(?,?,?,?,?)",
                (rid, request, "queued", time.time(), json.dumps(options)))
        return rid

    def update_run(self, rid: str, **kw) -> None:
        for k, v in kw.items():
            if k in ("plan", "options", "files") and not isinstance(v, str):
                v = json.dumps(v)
            self._x(f"UPDATE runs SET {k}=? WHERE id=?", (v, rid))

    def get_run(self, rid: str) -> dict | None:
        r = self._x("SELECT id,request,status,created,finished,summary,options,plan,error,files FROM runs WHERE id=?", (rid,)).fetchone()
        if not r:
            return None
        return {"id": r[0], "request": r[1], "status": r[2], "created": r[3], "finished": r[4], "summary": r[5],
                "options": json.loads(r[6] or "{}"), "plan": json.loads(r[7]) if r[7] else None, "error": r[8],
                "files": json.loads(r[9]) if r[9] else [], "tasks": self.get_tasks(rid)}

    def run_status(self, rid: str | None) -> str | None:
        if not rid:
            return None
        r = self._x("SELECT status FROM runs WHERE id=?", (rid,)).fetchone()
        return r[0] if r else None

    def list_runs(self, limit: int = 60) -> list[dict]:
        return [{"id": r[0], "request": r[1], "status": r[2], "created": r[3], "finished": r[4]} for r in
                self._x("SELECT id,request,status,created,finished FROM runs ORDER BY created DESC LIMIT ?", (limit,))]

    def upsert_task(self, rid: str, tid: str, idx: int, title: str, status: str, attempts: int = 0,
                    files: list[str] | None = None, summary: str = "") -> None:
        self._x("INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(run_id,tid) DO UPDATE SET "
                "idx=excluded.idx,title=excluded.title,status=excluded.status,attempts=excluded.attempts,files=excluded.files,summary=excluded.summary",
                (rid, tid, idx, title, status, attempts, json.dumps(files or []), summary))

    def get_tasks(self, rid: str) -> list[dict]:
        return [{"id": r[0], "idx": r[1], "title": r[2], "status": r[3], "attempts": r[4], "files": json.loads(r[5] or "[]"), "summary": r[6]}
                for r in self._x("SELECT tid,idx,title,status,attempts,files,summary FROM tasks WHERE run_id=? ORDER BY idx", (rid,))]

    # ---------------------------------------------------------------- conversations
    @staticmethod
    def _conv(r) -> dict:
        try:
            state = json.loads(r[9] or "{}")
        except json.JSONDecodeError:
            state = {}
        return {"id": r[0], "title": r[1], "title_locked": bool(r[2]), "created": r[3], "updated": r[4], "summary": r[5] or "",
                "summary_upto": r[6] or 0, "extracted_upto": r[7] or 0, "titled_at": r[8] or 0, "state": state}

    def create_conv(self, title: str = "New chat") -> dict:
        cid = uuid.uuid4().hex[:12]
        now = time.time()
        self._x("INSERT INTO conversations(id,title,created,updated) VALUES(?,?,?,?)", (cid, title, now, now))
        return self.get_conv(cid)

    def get_conv(self, cid: str) -> dict | None:
        r = self._x(f"SELECT {CONV_COLS} FROM conversations WHERE id=?", (cid,)).fetchone()
        return self._conv(r) if r else None

    def list_convs(self, limit: int = 300) -> list[dict]:
        out = []
        for r in self._x(
                f"SELECT {CONV_COLS}, "
                "(SELECT content FROM messages m WHERE m.conv_id=c.id AND m.kind IN ('text','run_result') ORDER BY m.id DESC LIMIT 1), "
                "(SELECT COUNT(*) FROM messages m WHERE m.conv_id=c.id) "
                "FROM conversations c ORDER BY c.updated DESC LIMIT ?", (limit,)):
            c = self._conv(r)
            c["preview"] = " ".join((r[10] or "").split())[:140]
            c["count"] = r[11]
            out.append(c)
        return out

    def search_convs(self, q: str, limit: int = 60) -> list[dict]:
        ids = {r[0] for r in self._x("SELECT id FROM conversations WHERE title LIKE ?", (f"%{q}%",))}
        match = _fts_query(q)
        if match:
            try:
                ids |= {r[0] for r in self._x("SELECT DISTINCT conv_id FROM msg_fts WHERE msg_fts MATCH ? LIMIT 80", (match,))}
            except sqlite3.OperationalError:
                pass
        return [c for c in self.list_convs(500) if c["id"] in ids][:limit]

    def update_conv(self, cid: str, **kw) -> None:
        allowed = {"title", "title_locked", "summary", "summary_upto", "extracted_upto", "titled_at", "updated"}
        sets, args = [], []
        for k, v in kw.items():
            if k in allowed:
                sets.append(f"{k}=?")
                args.append(int(v) if k == "title_locked" else v)
        if sets:
            self._x(f"UPDATE conversations SET {', '.join(sets)} WHERE id=?", (*args, cid))

    def patch_state(self, cid: str, **kw) -> dict:
        """Merge keys into the conversation state; a value of None removes the key."""
        with self.lock:
            r = self._x("SELECT state FROM conversations WHERE id=?", (cid,)).fetchone()
            if not r:
                return {}
            try:
                st = json.loads(r[0] or "{}")
            except json.JSONDecodeError:
                st = {}
            for k, v in kw.items():
                if v is None:
                    st.pop(k, None)
                else:
                    st[k] = v
            self._x("UPDATE conversations SET state=? WHERE id=?", (json.dumps(st), cid))
            return st

    def delete_conv(self, cid: str) -> None:
        with self.lock:
            self._x("DELETE FROM msg_fts WHERE rowid IN (SELECT id FROM messages WHERE conv_id=?)", (cid,))
            self._x("DELETE FROM messages WHERE conv_id=?", (cid,))
            self._x("DELETE FROM conversations WHERE id=?", (cid,))

    # ---------------------------------------------------------------- messages
    @staticmethod
    def _msg(r) -> dict:
        try:
            meta = json.loads(r[5] or "{}")
        except json.JSONDecodeError:
            meta = {}
        return {"id": r[0], "conv_id": r[1], "role": r[2], "kind": r[3], "content": r[4], "meta": meta, "ts": r[6]}

    def add_msg(self, cid: str, role: str, kind: str, content: str, meta: dict | None = None) -> dict:
        ts = time.time()
        with self.lock:
            mid = self._x("INSERT INTO messages(conv_id,role,kind,content,meta,ts) VALUES(?,?,?,?,?,?)",
                          (cid, role, kind, content, json.dumps(meta or {}, default=str), ts)).lastrowid
            self._x("INSERT INTO msg_fts(rowid,body,conv_id) VALUES(?,?,?)", (mid, _fts_body(content), cid))
            self._x("UPDATE conversations SET updated=? WHERE id=?", (ts, cid))
        return {"id": mid, "conv_id": cid, "role": role, "kind": kind, "content": content, "meta": meta or {}, "ts": ts}

    def get_msg(self, mid: int) -> dict | None:
        r = self._x(f"SELECT {MSG_COLS} FROM messages WHERE id=?", (mid,)).fetchone()
        return self._msg(r) if r else None

    def update_msg(self, mid: int, content: str | None = None, meta: dict | None = None) -> dict | None:
        with self.lock:
            cur = self.get_msg(mid)
            if not cur:
                return None
            if content is not None and content != cur["content"]:
                self._x("UPDATE messages SET content=? WHERE id=?", (content, mid))
                self._x("DELETE FROM msg_fts WHERE rowid=?", (mid,))
                self._x("INSERT INTO msg_fts(rowid,body,conv_id) VALUES(?,?,?)", (mid, _fts_body(content), cur["conv_id"]))
            if meta is not None:
                self._x("UPDATE messages SET meta=? WHERE id=?", (json.dumps(meta, default=str), mid))
            return self.get_msg(mid)

    def delete_msg(self, mid: int) -> None:
        with self.lock:
            self._x("DELETE FROM msg_fts WHERE rowid=?", (mid,))
            self._x("DELETE FROM messages WHERE id=?", (mid,))

    def recent_msgs(self, cid: str, limit: int = 200, before: int | None = None) -> list[dict]:
        if before is None:
            rows = self._x(f"SELECT {MSG_COLS} FROM messages WHERE conv_id=? ORDER BY id DESC LIMIT ?", (cid, limit)).fetchall()
        else:
            rows = self._x(f"SELECT {MSG_COLS} FROM messages WHERE conv_id=? AND id<? ORDER BY id DESC LIMIT ?", (cid, before, limit)).fetchall()
        return [self._msg(r) for r in reversed(rows)]

    def msgs_range(self, cid: str, after: int = 0, before: int | None = None, limit: int = 400) -> list[dict]:
        if before is None:
            rows = self._x(f"SELECT {MSG_COLS} FROM messages WHERE conv_id=? AND id>? ORDER BY id LIMIT ?", (cid, after, limit))
        else:
            rows = self._x(f"SELECT {MSG_COLS} FROM messages WHERE conv_id=? AND id>? AND id<? ORDER BY id LIMIT ?", (cid, after, before, limit))
        return [self._msg(r) for r in rows]

    def count_msgs(self, cid: str, role: str | None = None, kind: str | None = None) -> int:
        q, args = "SELECT COUNT(*) FROM messages WHERE conv_id=?", [cid]
        if role:
            q += " AND role=?"
            args.append(role)
        if kind:
            q += " AND kind=?"
            args.append(kind)
        return self._x(q, args).fetchone()[0]

    def has_older(self, cid: str, before: int) -> bool:
        return self._x("SELECT 1 FROM messages WHERE conv_id=? AND id<? LIMIT 1", (cid, before)).fetchone() is not None

    def find_msgs(self, kind: str, limit: int = 400) -> list[dict]:
        return [self._msg(r) for r in self._x(f"SELECT {MSG_COLS} FROM messages WHERE kind=? ORDER BY id DESC LIMIT ?", (kind, limit))]

    def find_msgs_conv(self, cid: str, kind: str, limit: int = 50) -> list[dict]:
        rows = self._x(f"SELECT {MSG_COLS} FROM messages WHERE conv_id=? AND kind=? ORDER BY id DESC LIMIT ?", (cid, kind, limit)).fetchall()
        return [self._msg(r) for r in reversed(rows)]

    def search_msgs(self, cid: str, query: str, before_id: int, limit: int = 5) -> list[dict]:
        """Older messages of this chat that best match `query` (BM25), used to recall what was said long ago."""
        match = _fts_query(query)
        if not match:
            return []
        try:
            rows = self._x(
                "SELECT m.id,m.conv_id,m.role,m.kind,m.content,m.meta,m.ts, bm25(msg_fts) AS s FROM msg_fts "
                "JOIN messages m ON m.id = msg_fts.rowid "
                "WHERE msg_fts MATCH ? AND msg_fts.conv_id = ? AND m.id < ? AND m.kind IN ('text','brief','run_result') "
                "ORDER BY s LIMIT ?", (match, cid, before_id, limit)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [self._msg(r[:7]) for r in rows]

    # ---------------------------------------------------------------- events
    def emit(self, rid: str, type_: str, persist: bool = True, **payload) -> int:
        ts = time.time()
        eid = self._x("INSERT INTO events(run_id,ts,type,payload) VALUES(?,?,?,?)",
                      (rid, ts, type_, json.dumps(payload, default=str))).lastrowid if persist else 0
        ev = {"id": eid, "run_id": rid, "ts": ts, "type": type_, "payload": payload}
        for fn in list(self.listeners):
            try:
                fn(ev)
            except Exception:  # noqa: BLE001
                pass
        return eid

    def events_since(self, rid: str, after: int = 0, limit: int = 2000) -> list[dict]:
        return [{"id": r[0], "run_id": rid, "ts": r[1], "type": r[2], "payload": json.loads(r[3])} for r in
                self._x("SELECT id,ts,type,payload FROM events WHERE run_id=? AND id>? ORDER BY id LIMIT ?", (rid, after, limit))]

    # ---------------------------------------------------------------- human-readable export
    def export_markdown(self, current_hash: Callable[[str], str | None]) -> None:
        d = self.cfg.forge_dir / "memory"
        d.mkdir(exist_ok=True)
        facts = [f for f in self.list_facts(current_hash)]
        by_kind: dict[str, list[dict]] = {}
        for f in facts:
            by_kind.setdefault(f["kind"], []).append(f)
        names = {"decision": "decisions.md", "answer": "answers.md", "convention": "conventions.md", "gotcha": "gotchas.md",
                 "fact": "verified_facts.md", "summary": "run_summaries.md", "preference": "preferences.md",
                 "project": "project_facts.md", "goal": "goals.md"}
        for kind, fname in names.items():
            lines = [f"# {fname[:-3].replace('_', ' ').title()}", ""]
            for f in by_kind.get(kind, []):
                lines.append(f"- {f['text']}" + ("  _(stale: related files changed)_" if f["stale"] else "") + (f"  `{', '.join(f['paths'][:3])}`" if f["paths"] else ""))
            (d / fname).write_text("\n".join(lines) + "\n", encoding="utf-8")
        card = self.meta_get("project_card")
        if card:
            (d / "project_summary.md").write_text("# Project summary\n\n" + card + "\n", encoding="utf-8")