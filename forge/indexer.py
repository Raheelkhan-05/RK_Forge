# forge/indexer.py

"""Incremental code index (SQLite + FTS5) for large repos: symbols, import graph, exports, API/DB references.

Designed for 50k+ line monorepos: a refresh only re-parses files whose (size, mtime) or hash changed.
No tree-sitter dependency; JS/TS/JSX/TSX are handled with a brace-aware scanner, Python with `ast`, SQL with
dollar-quote aware scanning.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import posixpath
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .workspace import is_binary, is_secret, list_repo_files

LANG = {".py": "python", ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js", ".ts": "ts", ".tsx": "ts",
        ".sql": "sql", ".css": "css", ".scss": "css", ".html": "html", ".md": "md", ".json": "json",
        ".yml": "yaml", ".yaml": "yaml", ".toml": "toml", ".sh": "sh", ".env.example": "text", ".txt": "text",
        ".prisma": "text", ".graphql": "text"}
SKIP_NAMES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "bun.lockb", "npm-shrinkwrap.json"}
JS_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json", ".css", ".scss")
NODE_BUILTINS = set("assert async_hooks buffer child_process cluster console constants crypto dgram diagnostics_channel dns domain events fs http http2 https inspector module net os path perf_hooks process punycode querystring readline repl stream string_decoder sys timers tls trace_events tty url util v8 vm wasi worker_threads zlib".split())
STOP = set("a an the and or of to in on for with from by is are be this that it as at we you i not no add make update change fix use using new should must can will when then so if all any each into via only also than but do does".split())
KEYWORDS = {"if", "for", "while", "switch", "catch", "function", "return", "else", "do", "try", "with", "await", "typeof", "new"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, hash TEXT, size INT, mtime_ns INT, lang TEXT, project TEXT, lines INT, indexed INT);
CREATE TABLE IF NOT EXISTS symbols(id INTEGER PRIMARY KEY, path TEXT, name TEXT, kind TEXT, line INT, end_line INT, sig TEXT, parent TEXT, exported INT);
CREATE INDEX IF NOT EXISTS sym_name ON symbols(name COLLATE NOCASE); CREATE INDEX IF NOT EXISTS sym_path ON symbols(path);
CREATE TABLE IF NOT EXISTS imports(path TEXT, spec TEXT, names TEXT, resolved TEXT, kind TEXT);
CREATE INDEX IF NOT EXISTS imp_path ON imports(path); CREATE INDEX IF NOT EXISTS imp_res ON imports(resolved);
CREATE TABLE IF NOT EXISTS exports(path TEXT, name TEXT); CREATE INDEX IF NOT EXISTS exp_path ON exports(path);
CREATE TABLE IF NOT EXISTS export_meta(path TEXT PRIMARY KEY, star TEXT, dynamic INT);
CREATE TABLE IF NOT EXISTS refs(path TEXT, kind TEXT, name TEXT); CREATE INDEX IF NOT EXISTS ref_name ON refs(name);
CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, path TEXT, start_line INT, end_line INT, symbols TEXT);
CREATE INDEX IF NOT EXISTS chunk_path ON chunks(path);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(body, symbols, path, tokenize='unicode61');
"""


@dataclass
class Sym:
    path: str
    name: str
    kind: str
    line: int
    end_line: int
    sig: str = ""
    parent: str | None = None
    exported: bool = False


@dataclass
class Hit:
    path: str
    start: int
    end: int
    symbols: str
    score: float


def split_ident(s: str) -> list[str]:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    return [w.lower() for w in re.split(r"[^A-Za-z0-9]+", s) if w]


def query_terms(text: str, limit: int = 24) -> list[str]:
    terms: list[str] = []
    for w in re.findall(r"[A-Za-z_$][\w$./-]*", text):
        for part in ([w] if re.fullmatch(r"[a-z]+", w) else []) + split_ident(w):
            if len(part) > 2 and part not in STOP and part not in terms:
                terms.append(part)
    return terms[:limit]


# ------------------------------------------------------------------------------------------------ parsers
def _block_end(lines: list[str], start: int, cap: int = 1200) -> int:
    """0-based index of the last line of the declaration starting at `start` (brace/paren aware)."""
    depth, opened, i = 0, False, start
    in_block_comment = False
    tmpl = False
    while i < len(lines) and i - start < cap:
        ln = lines[i]
        j, quote = 0, None
        while j < len(ln):
            c = ln[j]
            if in_block_comment:
                if ln.startswith("*/", j):
                    in_block_comment, j = False, j + 1
            elif quote:
                if c == "\\":
                    j += 1
                elif c == quote:
                    quote = None
            elif tmpl:
                if c == "\\":
                    j += 1
                elif c == "`":
                    tmpl = False
            elif ln.startswith("//", j):
                break
            elif ln.startswith("/*", j):
                in_block_comment, j = True, j + 1
            elif c in "'\"":
                quote = c
            elif c == "`":
                tmpl = True
            elif c in "{([":
                depth += 1
                opened = True
            elif c in "})]":
                depth -= 1
            j += 1
        if opened and depth <= 0 and not in_block_comment and not tmpl:
            return i
        if not opened and i == start and ln.rstrip().endswith((";", ",")):
            return i
        if not opened and i - start >= 3:
            return start
        i += 1
    return min(i, len(lines) - 1) if opened else start


JS_DECL = [
    ("function", re.compile(r"^(?P<ind>\s*)(?P<exp>export\s+)?(?P<def>default\s+)?(?:async\s+)?function\s*\*?\s*(?P<name>[A-Za-z_$][\w$]*)\s*[(<]")),
    ("class", re.compile(r"^(?P<ind>\s*)(?P<exp>export\s+)?(?P<def>default\s+)?(?:abstract\s+)?class\s+(?P<name>[A-Za-z_$][\w$]*)")),
    ("function", re.compile(r"^(?P<ind>\s*)(?P<exp>export\s+)?(?P<def>default\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:\(|[A-Za-z_$][\w$]*\s*=>|function\b|(?:React\.)?(?:memo|forwardRef|lazy)\(|use[A-Z]\w*\(|create\w+\(|styled\b)")),
    ("const", re.compile(r"^(?P<ind>\s*)(?P<exp>export\s+)?(?P<def>default\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*[\[{]")),
    ("interface", re.compile(r"^(?P<ind>\s*)(?P<exp>export\s+)?(?P<def>default\s+)?(?:declare\s+)?interface\s+(?P<name>[A-Za-z_$][\w$]*)")),
    ("type", re.compile(r"^(?P<ind>\s*)(?P<exp>export\s+)?(?P<def>default\s+)?(?:declare\s+)?type\s+(?P<name>[A-Za-z_$][\w$]*)\s*(?:<[^=]*>)?\s*=")),
    ("enum", re.compile(r"^(?P<ind>\s*)(?P<exp>export\s+)?(?P<def>default\s+)?(?:declare\s+)?(?:const\s+)?enum\s+(?P<name>[A-Za-z_$][\w$]*)")),
]
JS_METHOD = re.compile(r"^(?P<ind>\s+)(?:(?:public|private|protected|static|async|get|set|readonly|override)\s+)*(?P<name>[A-Za-z_$#][\w$]*)\s*(?:<[^>]*>)?\s*\([^)]*\)\s*(?::\s*[^{=]+)?\{\s*$")
JS_ROUTE = re.compile(r"""\b(?:[\w$]*[rR]outer|app|route|server|api)\.(get|post|put|patch|delete|all)\(\s*['"`]([^'"`]+)['"`]""")
RE_RPC = re.compile(r"""\.rpc\(\s*['"]([\w.]+)['"]""")
RE_TABLE = re.compile(r"""\.from\(\s*['"](\w+)['"]""")
RE_API = re.compile(r"""['"`](/api/[\w\-/:.${}]*)""")


def _indent_width(ind: str) -> int:
    return len(ind.replace("\t", "    "))


def parse_js(text: str):
    lines = text.split("\n")
    syms: list[Sym] = []
    for i, ln in enumerate(lines):
        if not ln.strip() or ln.lstrip().startswith(("//", "*", "/*")):
            continue
        for kind, rx in JS_DECL:
            m = rx.match(ln)
            if not m:
                continue
            w = _indent_width(m.group("ind"))
            if w > 4 and kind in ("const", "class", "interface", "type", "enum"):
                break
            if w > 6:
                break
            end = _block_end(lines, i)
            exp = bool(m.group("exp"))
            syms.append(Sym("", m.group("name"), kind, i + 1, end + 1, ln.strip()[:140], None, exp or bool(m.group("def"))))
            break
        else:
            rm = JS_ROUTE.search(ln)
            if rm:
                syms.append(Sym("", f"{rm.group(1).upper()} {rm.group(2)}", "route", i + 1, _block_end(lines, i) + 1, ln.strip()[:140]))
    # parents + methods
    by_start = sorted(syms, key=lambda s: s.line)
    for s in by_start:
        for p in by_start:
            if p is not s and p.line < s.line <= p.end_line and p.kind != "route" and (p.end_line - p.line) >= (s.end_line - s.line):
                s.parent = p.name
    for p in [s for s in by_start if s.kind == "class"]:
        for i in range(p.line, min(p.end_line, len(lines))):
            m = JS_METHOD.match(lines[i])
            if m and m.group("name") not in KEYWORDS:
                syms.append(Sym("", m.group("name"), "method", i + 1, _block_end(lines, i) + 1, lines[i].strip()[:140], p.name))
    clean = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    clean = re.sub(r"(?m)^\s*//.*$", "", clean)
    imports = []
    for m in re.finditer(r"""(?:import|export)\s+(?P<type>type\s+)?(?P<clause>[^;'"`]*?)\s*from\s*['"](?P<spec>[^'"]+)['"]""", clean):
        names = []
        clause = m.group("clause")
        bm = re.search(r"\{([^}]*)\}", clause)
        if bm and not m.group("type"):
            for part in bm.group(1).split(","):
                nm = part.strip().split(" as ")[0].strip()
                nm = re.sub(r"^type\s+", "", nm) if part.strip().startswith("type ") else nm
                if nm and not part.strip().startswith("type "):
                    names.append(nm)
        imports.append((m.group("spec"), names))
    for m in re.finditer(r"""(?:^|[^\w$.])import\s*['"]([^'"]+)['"]""", clean):
        imports.append((m.group(1), []))
    for m in re.finditer(r"""(?:require|import)\(\s*['"]([^'"]+)['"]\s*\)""", clean):
        imports.append((m.group(1), []))
    names, star, dynamic = set(), [], False
    for m in re.finditer(r"^\s*export\s+(?P<def>default\s+)?(?:declare\s+)?(?:async\s+)?(?:function\s*\*?|class|abstract\s+class|const|let|var|interface|type|enum)\s+(?P<n>[A-Za-z_$][\w$]*)", clean, re.M):
        names.add(m.group("n"))
        if m.group("def"):
            names.add("default")
    if re.search(r"^\s*export\s+default\b", clean, re.M):
        names.add("default")
    for m in re.finditer(r"export\s*\{([^}]*)\}", clean):
        for part in m.group(1).split(","):
            part = part.strip()
            if part:
                names.add(part.split(" as ")[-1].strip())
    for m in re.finditer(r"""export\s*\*\s*(?:as\s+([\w$]+)\s+)?from\s*['"]([^'"]+)['"]""", clean):
        if m.group(1):
            names.add(m.group(1))
        else:
            star.append(m.group(2))
    for m in re.finditer(r"^\s*export\s+(?:const|let|var)\s*\{([^}]*)\}", clean, re.M):
        names.update(x.strip().split(":")[-1].strip() for x in m.group(1).split(",") if x.strip())
    mm = re.search(r"module\.exports\s*=\s*\{([^}]*)\}", clean)
    if mm:
        names.update(x.strip().split(":")[0].strip() for x in mm.group(1).split(",") if x.strip())
    elif re.search(r"module\.exports\s*=", clean):
        dynamic = True
    names.update(re.findall(r"(?:module\.)?exports\.([\w$]+)\s*=", clean))
    refs = [("rpc", n) for n in set(RE_RPC.findall(text))] + [("table", n) for n in set(RE_TABLE.findall(text))] \
        + [("api", re.sub(r"\$\{[^}]*\}", ":param", n)) for n in set(RE_API.findall(text))]
    return syms, imports, names, star, dynamic, refs


def parse_python(text: str):
    syms, imports, names, refs = [], [], set(), []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return syms, imports, names, [], False, refs

    def visit(body, parent=None):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = "class" if isinstance(n, ast.ClassDef) else ("method" if parent else "function")
                sig = f"{'class' if kind == 'class' else 'def'} {n.name}"
                if not isinstance(n, ast.ClassDef):
                    try:
                        sig += f"({ast.unparse(n.args)})"
                    except Exception:  # noqa: BLE001
                        sig += "(...)"
                syms.append(Sym("", n.name, kind, n.lineno, n.end_lineno or n.lineno, sig[:140], parent, parent is None))
                if parent is None:
                    names.add(n.name)
                for d in getattr(n, "decorator_list", []):
                    if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr in ("get", "post", "put", "patch", "delete") \
                            and d.args and isinstance(d.args[0], ast.Constant) and isinstance(d.args[0].value, str):
                        syms.append(Sym("", f"{d.func.attr.upper()} {d.args[0].value}", "route", n.lineno, n.end_lineno or n.lineno, sig[:140]))
                if isinstance(n, ast.ClassDef):
                    visit(n.body, n.name)
            elif isinstance(n, ast.Assign) and parent is None:
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
                        if t.id.isupper():
                            syms.append(Sym("", t.id, "const", n.lineno, n.end_lineno or n.lineno, "", None, True))
            elif isinstance(n, ast.Import):
                imports.extend((a.name, []) for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                imports.append(("." * n.level + (n.module or ""), [a.name for a in n.names if a.name != "*"]))
    visit(tree.body)
    return syms, imports, names, [], False, refs


SQL_CREATE = re.compile(r"(?i)^\s*create\s+(?:or\s+replace\s+)?(?:unique\s+)?(?P<kind>function|table|view|materialized\s+view|policy|trigger|index|type|extension)\s+(?:if\s+not\s+exists\s+)?(?P<name>(?:\"?[\w]+\"?\.)?\"?[\w ]+?\"?)\s*(?:\(|\s|$)")


def parse_sql(text: str):
    lines = text.split("\n")
    syms = []
    i = 0
    while i < len(lines):
        m = SQL_CREATE.match(lines[i])
        if not m:
            i += 1
            continue
        kind = re.sub(r"\s+", "-", m.group("kind").lower())
        name = m.group("name").strip().strip('"').split(".")[-1].strip('"')
        if kind == "policy":
            name = re.search(r'(?i)policy\s+"?([^"\n]+?)"?\s+on', lines[i])
            name = name.group(1) if name else m.group("name")
        j, dollars = i, 0
        while j < len(lines) and j - i < 800:
            dollars += len(re.findall(r"\$[A-Za-z_]*\$", lines[j]))
            if lines[j].rstrip().endswith(";") and dollars % 2 == 0:
                break
            j += 1
        syms.append(Sym("", name, "sql-" + kind, i + 1, min(j, len(lines) - 1) + 1, lines[i].strip()[:140], None, True))
        i = j + 1
    return syms, [], {s.name for s in syms}, [], False, []


# ------------------------------------------------------------------------------------------------ index
class Index:
    def __init__(self, db_path: Path, root: Path, cfg: Config):
        self.root, self.cfg = Path(root), cfg
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.db = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self._aliases: dict | None = None
        self._pdirs: list[str] | None = None
        self._pkg_cache: dict[str, set[str]] = {}

    def close(self):
        self.db.close()

    # ---------------------------------------------------------------- reading
    def read(self, rel: str) -> str | None:
        if is_secret(rel):
            return None
        p = self.root / rel
        try:
            data = p.read_bytes()
        except OSError:
            return None
        if is_binary(data):
            return None
        try:
            return data.decode("utf-8-sig").replace("\r\n", "\n")
        except UnicodeDecodeError:
            return None

    # ---------------------------------------------------------------- refresh
    def refresh(self, paths: list[str] | None = None) -> dict:
        """Full incremental refresh (paths=None) or targeted refresh after edits."""
        t0 = time.time()
        self._pdirs = None
        self._aliases = None
        full = paths is None
        all_files = list_repo_files(self.root, self.cfg.extra_ignore) if full else None
        known = {r[0]: r for r in self.db.execute("SELECT path,hash,size,mtime_ns FROM files")}
        changed = 0
        todo = all_files if full else list(paths)
        self.db.execute("BEGIN")
        try:
            if full:
                gone = set(known) - set(all_files)
                for p in gone:
                    self._drop(p)
            for rel in todo:
                real = self.root / rel
                if not real.is_file():
                    if not full:
                        self._drop(rel)
                    continue
                st = real.stat()
                old = known.get(rel)
                if full and old and old[2] == st.st_size and old[3] == st.st_mtime_ns:
                    continue
                if self._index_file(rel, st, old):
                    changed += 1
            self._assign_projects()
            self._resolve_all()
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"changed": changed, "files": self.db.execute("SELECT COUNT(*) FROM files").fetchone()[0],
                "seconds": round(time.time() - t0, 2)}

    def _drop(self, rel: str) -> None:
        d = self.db
        d.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE path=?)", (rel,))
        for t in ("files", "symbols", "imports", "exports", "export_meta", "refs", "chunks"):
            d.execute(f"DELETE FROM {t} WHERE path=?", (rel,))

    def _index_file(self, rel: str, st: os.stat_result, old) -> bool:
        ext = os.path.splitext(rel)[1].lower()
        name = rel.rsplit("/", 1)[-1]
        lang = LANG.get(ext)
        textual = lang is not None and name not in SKIP_NAMES and not name.endswith((".min.js", ".min.css", ".map")) \
            and st.st_size <= self.cfg.limits.max_file_bytes and not is_secret(rel)
        data = None
        h = old[1] if old else None
        if textual:
            try:
                data = (self.root / rel).read_bytes()
            except OSError:
                return False
            if is_binary(data):
                textual = False
            else:
                nh = hashlib.sha1(data).hexdigest()
                if old and nh == h:
                    self.db.execute("UPDATE files SET size=?, mtime_ns=? WHERE path=?", (st.st_size, st.st_mtime_ns, rel))
                    return False
                h = nh
        self._drop(rel)
        if not textual:
            self.db.execute("INSERT INTO files VALUES(?,?,?,?,?,?,?,0)", (rel, None, st.st_size, st.st_mtime_ns, None, None, 0))
            return True
        text = data.decode("utf-8-sig", "replace").replace("\r\n", "\n")
        lines = text.split("\n")
        self.db.execute("INSERT INTO files VALUES(?,?,?,?,?,?,?,1)", (rel, h, st.st_size, st.st_mtime_ns, lang, None, len(lines)))
        syms, imports, names, star, dynamic, refs = [], [], set(), [], False, []
        if lang in ("js", "ts"):
            syms, imports, names, star, dynamic, refs = parse_js(text)
        elif lang == "python":
            syms, imports, names, star, dynamic, refs = parse_python(text)
        elif lang == "sql":
            syms, imports, names, star, dynamic, refs = parse_sql(text)
        for s in syms:
            self.db.execute("INSERT INTO symbols(path,name,kind,line,end_line,sig,parent,exported) VALUES(?,?,?,?,?,?,?,?)",
                            (rel, s.name, s.kind, s.line, s.end_line, s.sig, s.parent, int(s.exported)))
        for spec, nm in imports:
            self.db.execute("INSERT INTO imports VALUES(?,?,?,?,?)", (rel, spec, json.dumps(nm), None, None))
        for n in names:
            self.db.execute("INSERT INTO exports VALUES(?,?)", (rel, n))
        if lang in ("js", "ts", "python"):
            self.db.execute("INSERT INTO export_meta VALUES(?,?,?)", (rel, json.dumps(star), int(dynamic)))
        for kind, n in refs:
            self.db.execute("INSERT INTO refs VALUES(?,?,?)", (rel, kind, n))
        self._chunk(rel, lines, syms)
        return True

    def _chunk(self, rel: str, lines: list[str], syms: list[Sym], win: int = 50, overlap: int = 8) -> None:
        n = len(lines)
        i = 0
        while i < n:
            j = min(n, i + win)
            body = "\n".join(lines[i:j])
            if body.strip():
                inside = [s.name for s in syms if s.line <= j and s.end_line >= i + 1 and s.kind != "route"][:8]
                words = set()
                for ident in re.findall(r"[A-Za-z_$][\w$]{2,}", body):
                    words.update(split_ident(ident))
                cid = self.db.execute("INSERT INTO chunks(path,start_line,end_line,symbols) VALUES(?,?,?,?)",
                                      (rel, i + 1, j, ",".join(inside))).lastrowid
                self.db.execute("INSERT INTO chunks_fts(rowid,body,symbols,path) VALUES(?,?,?,?)",
                                (cid, body + "\n" + " ".join(sorted(words)), " ".join(sum((split_ident(x) + [x] for x in inside), [])),
                                 " ".join(split_ident(rel))))
            if j >= n:
                break
            i = j - overlap

    # ---------------------------------------------------------------- projects & imports
    def project_dirs(self) -> list[str]:
        if self._pdirs is not None:
            return self._pdirs
        rows = self.db.execute("SELECT path FROM files WHERE path='package.json' OR path LIKE '%/package.json' OR path LIKE '%pyproject.toml' OR path LIKE '%requirements.txt'").fetchall()
        dirs = {posixpath.dirname(r[0]) for r in rows if r[0].count("/") <= 3}
        dirs.add("")
        self._pdirs = sorted(dirs)
        return self._pdirs

    def project_of(self, rel: str) -> str:
        best = ""
        for d in self.project_dirs():
            if d and (rel == d or rel.startswith(d + "/")) and len(d) > len(best):
                best = d
        return best

    def project_name(self, d: str) -> str:
        return d.rsplit("/", 1)[-1] if d else self.root.name

    def _assign_projects(self) -> None:
        dirs = [d for d in self.project_dirs() if d]
        for (p,) in self.db.execute("SELECT path FROM files WHERE project IS NULL").fetchall():
            best = ""
            for d in dirs:
                if (p.startswith(d + "/")) and len(d) > len(best):
                    best = d
            self.db.execute("UPDATE files SET project=? WHERE path=?", (best, p))

    def aliases(self) -> dict[str, dict[str, str]]:
        """Per-project import aliases: {project_dir: {"@/": "frontend/src/"}} (tsconfig/jsconfig paths, vite, defaults)."""
        if self._aliases is not None:
            return self._aliases
        out: dict[str, dict[str, str]] = {}
        for d in self.project_dirs():
            al: dict[str, str] = {}
            for cfgname in ("tsconfig.json", "jsconfig.json", "vite.config.js", "vite.config.ts", "vite.config.mjs"):
                rel = posixpath.join(d, cfgname) if d else cfgname
                text = self.read(rel)
                if not text:
                    continue
                if cfgname.endswith(".json"):
                    try:
                        data = json.loads(re.sub(r"(?m)^\s*//.*$|/\*.*?\*/", "", text, flags=re.S))
                    except json.JSONDecodeError:
                        continue
                    co = data.get("compilerOptions", {})
                    base = posixpath.normpath(posixpath.join(d, co.get("baseUrl", ".")))
                    for k, vs in co.get("paths", {}).items():
                        if vs:
                            al.setdefault(k.rstrip("*"), posixpath.normpath(posixpath.join(base, vs[0].rstrip("*"))).rstrip("/") + "/")
                else:
                    m = re.search(r"""['"]@['"]\s*:\s*[^,}]*?['"]([^'"]*src[^'"]*)['"]""", text)
                    if m:
                        al.setdefault("@/", posixpath.normpath(posixpath.join(d, m.group(1).lstrip("./"))).rstrip("/") + "/")
            src = posixpath.join(d, "src") if d else "src"
            if "@/" not in al and self.db.execute("SELECT 1 FROM files WHERE path LIKE ? LIMIT 1", (src + "/%",)).fetchone():
                al["@/"] = src + "/"
            out[d] = al
        self._aliases = out
        return out

    def _resolve(self, frm: str, spec: str, known: set[str], lang: str) -> tuple[str | None, str]:
        if lang == "python":
            if spec.startswith("."):
                lvl = len(spec) - len(spec.lstrip("."))
                base = posixpath.dirname(frm)
                for _ in range(lvl - 1):
                    base = posixpath.dirname(base)
                rest = spec.lstrip(".").replace(".", "/")
                cand = [posixpath.join(base, rest) + ".py", posixpath.join(base, rest, "__init__.py")] if rest else [posixpath.join(base, "__init__.py")]
                for c in cand:
                    if posixpath.normpath(c) in known:
                        return posixpath.normpath(c), "rel"
                return None, "rel"
            rest = spec.replace(".", "/")
            for d in self.project_dirs():
                for c in (posixpath.join(d, rest + ".py"), posixpath.join(d, rest, "__init__.py")):
                    if posixpath.normpath(c) in known:
                        return posixpath.normpath(c), "abs"
            return None, "bare"
        if spec.startswith(("./", "../")) or spec in (".", ".."):
            base = posixpath.normpath(posixpath.join(posixpath.dirname(frm), spec))
            kind = "rel"
        else:
            kind, base = "bare", None
            merged = {**self.cfg.aliases, **self.aliases().get(self.project_of(frm), {})}
            for pref, target in sorted(merged.items(), key=lambda kv: -len(kv[0])):
                if pref and spec.startswith(pref):
                    base, kind = posixpath.normpath(target + spec[len(pref):]), "alias"
                    break
            if base is None:
                return None, "bare"
        cands = [base] + [base + e for e in JS_EXTS] + [posixpath.join(base, "index" + e) for e in JS_EXTS]
        if base.endswith((".js", ".jsx", ".mjs")):
            stem = base.rsplit(".", 1)[0]
            cands += [stem + ".ts", stem + ".tsx"]
        for c in cands:
            if c in known:
                return c, kind
        return None, kind

    def _resolve_all(self) -> None:
        self._aliases = None
        self._pdirs = None
        known = {r[0] for r in self.db.execute("SELECT path FROM files")}
        langs = dict(self.db.execute("SELECT path, lang FROM files WHERE indexed=1"))
        rows = self.db.execute("SELECT rowid, path, spec, resolved, kind FROM imports").fetchall()
        for rid, path, spec, old_res, old_kind in rows:
            res, kind = self._resolve(path, spec, known, langs.get(path, "js"))
            if res != old_res or kind != old_kind:
                self.db.execute("UPDATE imports SET resolved=?, kind=? WHERE rowid=?", (res, kind, rid))

    # ---------------------------------------------------------------- queries
    def has_file(self, rel: str) -> bool:
        return self.db.execute("SELECT 1 FROM files WHERE path=?", (rel,)).fetchone() is not None

    def file_info(self, rel: str):
        return self.db.execute("SELECT path, lang, lines, project, indexed FROM files WHERE path=?", (rel,)).fetchone()

    def all_paths(self, indexed_only: bool = True) -> list[str]:
        q = "SELECT path FROM files" + (" WHERE indexed=1" if indexed_only else "")
        return [r[0] for r in self.db.execute(q + " ORDER BY path")]

    def symbols_in(self, rel: str) -> list[Sym]:
        return [Sym(rel, *r) for r in self.db.execute(
            "SELECT name,kind,line,end_line,sig,parent,exported FROM symbols WHERE path=? ORDER BY line", (rel,))]

    def find_symbol(self, name: str, path: str | None = None, limit: int = 12) -> list[Sym]:
        q = "SELECT path,name,kind,line,end_line,sig,parent,exported FROM symbols WHERE name = ? COLLATE NOCASE"
        args: list = [name]
        if path:
            q += " AND path = ?"
            args.append(path)
        q += " ORDER BY exported DESC, kind='method', path LIMIT ?"
        rows = self.db.execute(q, args + [limit]).fetchall()
        if not rows and not path:
            rows = self.db.execute("SELECT path,name,kind,line,end_line,sig,parent,exported FROM symbols WHERE name LIKE ? COLLATE NOCASE LIMIT ?",
                                   (f"%{name}%", limit)).fetchall()
        return [Sym(*r[:6], r[6], bool(r[7])) for r in rows]

    def refs_to(self, name: str, kind: str | None = None) -> list[tuple[str, str]]:
        q = "SELECT path, kind FROM refs WHERE name = ?" + (" AND kind = ?" if kind else "")
        return self.db.execute(q, [name] + ([kind] if kind else [])).fetchall()

    def imports_of(self, rel: str) -> list[tuple[str, list[str], str | None, str]]:
        return [(s, json.loads(n or "[]"), r, k) for s, n, r, k in
                self.db.execute("SELECT spec, names, resolved, kind FROM imports WHERE path=?", (rel,))]

    def importers_of(self, rel: str) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT DISTINCT path FROM imports WHERE resolved=?", (rel,))]

    def exports_of(self, rel: str) -> tuple[set[str], list[str], bool]:
        names = {r[0] for r in self.db.execute("SELECT name FROM exports WHERE path=?", (rel,))}
        m = self.db.execute("SELECT star, dynamic FROM export_meta WHERE path=?", (rel,)).fetchone()
        return names, (json.loads(m[0]) if m else []), bool(m[1]) if m else True

    def tests_for(self, rel: str) -> list[str]:
        stem = re.sub(r"\.(test|spec)$", "", rel.rsplit("/", 1)[-1].rsplit(".", 1)[0])
        out = []
        for (p,) in self.db.execute("SELECT path FROM files WHERE indexed=1 AND (path LIKE '%.test.%' OR path LIKE '%.spec.%' OR path LIKE '%test_%.py' OR path LIKE '%\\_test.py' ESCAPE '\\' OR path LIKE '%/__tests__/%' OR path LIKE 'tests/%' OR path LIKE '%/tests/%')"):
            base = p.rsplit("/", 1)[-1]
            if stem and stem.lower() in base.lower():
                out.append(p)
        for imp in self.importers_of(rel):
            if re.search(r"(\.test\.|\.spec\.|/__tests__/|/tests?/|test_)", imp) and imp not in out:
                out.append(imp)
        return sorted(set(out))

    def search(self, text: str, limit: int = 12, prefix: str | None = None) -> list[Hit]:
        terms = query_terms(text)
        if not terms:
            return []
        q = " OR ".join(f'"{t}"' for t in terms)
        sql = ("SELECT c.path, c.start_line, c.end_line, c.symbols, bm25(chunks_fts, 1.0, 6.0, 3.0) AS s "
               "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid WHERE chunks_fts MATCH ? ")
        args: list = [q]
        if prefix:
            sql += "AND c.path LIKE ? "
            args.append(prefix + "%")
        sql += "ORDER BY s LIMIT ?"
        try:
            rows = self.db.execute(sql, args + [limit * 3]).fetchall()
        except sqlite3.OperationalError:
            return []
        hits, seen = [], set()
        for p, a, b, sy, s in rows:
            key = (p, a // 40)
            if key in seen:
                continue
            seen.add(key)
            hits.append(Hit(p, a, b, sy, -s))
            if len(hits) >= limit:
                break
        return hits

    def grep(self, needle: str, limit: int = 20, regex: bool = False) -> list[tuple[str, int, str]]:
        """Literal (or regex) search over indexed text files; reads from disk so it is always current."""
        out = []
        try:
            rx = re.compile(needle if regex else re.escape(needle), re.I)
        except re.error:
            rx = re.compile(re.escape(needle), re.I)
        for p in self.all_paths():
            text = self.read(p)
            if not text or not rx.search(text):
                continue
            for i, ln in enumerate(text.split("\n"), 1):
                if rx.search(ln):
                    out.append((p, i, ln.strip()[:160]))
                    if len(out) >= limit:
                        return out
        return out

    def stats(self) -> dict:
        q = lambda s: self.db.execute(s).fetchone()[0]  # noqa: E731
        by_proj = {self.project_name(p) if p else "(root)": n for p, n in self.db.execute("SELECT project, SUM(lines) FROM files WHERE indexed=1 GROUP BY project")}
        return {"files": q("SELECT COUNT(*) FROM files WHERE indexed=1"), "lines": q("SELECT COALESCE(SUM(lines),0) FROM files"),
                "symbols": q("SELECT COUNT(*) FROM symbols"), "imports": q("SELECT COUNT(*) FROM imports"),
                "projects": [self.project_name(d) for d in self.project_dirs()], "lines_by_project": by_proj}

    # ---------------------------------------------------------------- import validation (anti-hallucination gate)
    def _deps_for(self, rel: str) -> set[str]:
        d = posixpath.dirname(rel)
        out: set[str] = set()
        while True:
            pj = posixpath.join(d, "package.json") if d else "package.json"
            if pj not in self._pkg_cache:
                deps: set[str] = set()
                text = self.read(pj)
                if text:
                    try:
                        data = json.loads(text)
                        for k in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
                            deps.update(data.get(k, {}).keys())
                    except json.JSONDecodeError:
                        pass
                    deps.add("__found__")
                self._pkg_cache[pj] = deps
            out |= self._pkg_cache[pj]
            if not d:
                return out
            d = posixpath.dirname(d)

    def check_imports(self, rel: str) -> list[str]:
        """Imports that cannot be satisfied: missing files, uninstalled packages, names a module does not export."""
        info = self.file_info(rel)
        if not info or not info[4] or info[1] not in ("js", "ts", "python"):
            return []
        problems = []
        self._pkg_cache = {}
        self._aliases = None
        known = {r[0] for r in self.db.execute("SELECT path FROM files")}
        for spec, names, _res, _kind in self.imports_of(rel):
            res, kind = self._resolve(rel, spec, known, info[1])
            if info[1] == "python":
                if kind == "rel" and res is None:
                    problems.append(f"{rel}: relative import '{spec}' does not match any file")
                continue
            if kind in ("rel", "alias") and res is None:
                problems.append(f"{rel}: import '{spec}' does not resolve to any file")
            elif kind == "bare":
                pkg = "/".join(spec.split("/")[:2]) if spec.startswith("@") else spec.split("/")[0]
                if spec.startswith(("node:", "virtual:", "http", "data:", "/", "~", "#")) or pkg in NODE_BUILTINS:
                    continue
                deps = self._deps_for(rel)
                if "__found__" in deps and pkg not in deps and not pkg.startswith("@types"):
                    problems.append(f"{rel}: package '{pkg}' is imported but is not listed in package.json")
            elif res and names and res.endswith((".js", ".jsx", ".mjs", ".ts", ".tsx", ".cjs")):
                exp, star, dyn = self.exports_of(res)
                if star or dyn:
                    continue
                text = self.read(res) or ""
                for nm in names:
                    if nm not in exp and not re.search(rf"^\s*export\b.*\b{re.escape(nm)}\b", text, re.M):
                        problems.append(f"{rel}: '{nm}' is imported from '{spec}' ({res}) but that file does not export it")
        return problems

    # ---------------------------------------------------------------- repo map
    def repo_map(self, max_chars: int = 6000) -> str:
        out = []
        for d in self.project_dirs():
            name = self.project_name(d)
            n = self.db.execute("SELECT COUNT(*), COALESCE(SUM(lines),0) FROM files WHERE indexed=1 AND project=?", (d,)).fetchone()
            if not n[0]:
                continue
            out.append(f"## {name}/  ({n[0]} files, {n[1]} lines)")
            rows = self.db.execute(
                "SELECT f.path, (SELECT COUNT(*) FROM imports i WHERE i.resolved=f.path) AS inb FROM files f "
                "WHERE f.indexed=1 AND f.project=? AND f.lang IN ('js','ts','python','sql') ORDER BY inb DESC, f.lines DESC LIMIT 14", (d,)).fetchall()
            for p, inb in rows:
                syms = [s.name for s in self.symbols_in(p) if s.exported and s.kind != "route"][:6]
                out.append(f"- {p}" + (f"  [{', '.join(syms)}]" if syms else "") + (f"  (imported by {inb})" if inb else ""))
        text = "\n".join(out)
        return text[:max_chars]
