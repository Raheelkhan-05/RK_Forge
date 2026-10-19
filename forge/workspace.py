# forge/workspace.py

"""Isolated staging workspace + journaled all-or-nothing commit.

Atomicity   Every edit lands in a private copy of the repo (the *stage*). The real files are touched only by
            `transact()`, which journals backups first and rolls everything back if any step fails.
Consistency Nothing reaches `transact()` unless the verification gates passed (enforced by the orchestrator).
Isolation   The real working tree is never modified while a run is in progress. At commit time every file we are
            about to change must still equal the version we started from, otherwise the commit is refused.
Durability  Temp files and journal entries are fsync'ed; an interrupted commit is rolled back on next start.
"""
from __future__ import annotations

import difflib
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Config, DEFAULT_IGNORE_DIRS, LINK_DIRS, SECRET_GLOBS


class TxError(RuntimeError):
    pass


class ConflictError(TxError):
    def __init__(self, paths: list[str]):
        super().__init__("These files changed on disk while the run was in progress: " + ", ".join(paths))
        self.paths = paths


# ---------------------------------------------------------------------------------------------- helpers
def norm_rel(rel: str) -> str:
    rel = rel.replace("\\", "/").strip()
    while rel.startswith("./"):
        rel = rel[2:]
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if not parts or rel.startswith("/") or ".." in parts or (len(parts[0]) == 2 and parts[0][1] == ":"):
        raise TxError(f"Unsafe path: {rel!r}")
    if parts[0] in (".git", ".forge"):
        raise TxError(f"Path not allowed: {rel!r}")
    return "/".join(parts)


def is_secret(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(name, g) for g in SECRET_GLOBS)


def is_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def sha(data: bytes | None) -> str:
    return hashlib.sha256(data or b"").hexdigest() if data is not None else "-"


def list_repo_files(root: Path, extra_ignore: list[str] | None = None) -> list[str]:
    """Relative posix paths of all non-ignored files. Uses git when available (honours .gitignore)."""
    extra = extra_ignore or []
    out: list[str] = []
    try:
        res = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                             capture_output=True, timeout=60)
        if res.returncode == 0 and res.stdout:
            for p in res.stdout.decode("utf-8", "replace").split("\0"):
                if p and not any(seg in DEFAULT_IGNORE_DIRS and seg != "vendor" for seg in p.split("/")[:-1]) \
                        and (root / p).is_file():
                    out.append(p)
            return sorted(set(out))
    except (OSError, subprocess.SubprocessError):
        pass
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in DEFAULT_IGNORE_DIRS and not d.startswith(".forge")]
        for fn in fns:
            rel = os.path.relpath(os.path.join(dp, fn), root).replace(os.sep, "/")
            if not any(fnmatch.fnmatch(rel, g) for g in extra):
                out.append(rel)
    return sorted(out)


def _is_link(p: str) -> bool:
    if os.path.islink(p):
        return True
    if sys.platform == "win32":
        try:
            return bool(os.lstat(p).st_file_attributes & 0x400)   # FILE_ATTRIBUTE_REPARSE_POINT (junction)
        except (OSError, AttributeError):
            return False
    return False


def link_dir(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst, target_is_directory=True)
        return True
    except OSError:
        if sys.platform == "win32":   # junctions need no admin rights
            r = subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)], capture_output=True)
            return r.returncode == 0
        return False


def safe_rmtree(path: Path) -> None:
    """Remove a tree WITHOUT ever following links (staging holds links to the real node_modules)."""
    if not path.exists() and not _is_link(str(path)):
        return
    for dp, dns, fns in os.walk(path, topdown=True, followlinks=False):
        for d in list(dns):
            full = os.path.join(dp, d)
            if _is_link(full):
                dns.remove(d)
                try:
                    os.rmdir(full) if sys.platform == "win32" else os.unlink(full)
                except OSError:
                    os.unlink(full)
    shutil.rmtree(path, ignore_errors=True)


def _fsync_dir(d: Path) -> None:
    if sys.platform == "win32":
        return
    try:
        fd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _write_durable(path: Path, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _replace(src: Path, dst: Path) -> None:
    for i in range(6):   # Windows: another process (watcher/AV) may briefly hold the file
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == 5:
                raise
            time.sleep(0.15 * (i + 1))


def _write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    _write_durable(tmp, json.dumps(obj, indent=1).encode("utf-8"))
    _replace(tmp, path)


# ---------------------------------------------------------------------------------------------- transaction
def transact(root: Path, journal_root: Path, tx_id: str, changes: dict[str, bytes | None],
             expect: dict[str, bytes | None], fail_after: int | None = None) -> dict:
    """Apply `changes` (path -> new bytes, None = delete) to `root` all-or-nothing.

    `expect[path]` is the content the file must currently have (None = must not exist); a mismatch aborts.
    `fail_after` is a test hook that raises after N applied files to prove rollback works.
    """
    rels = {norm_rel(p): v for p, v in changes.items()}
    exp = {norm_rel(p): v for p, v in expect.items()}
    conflicts, clashes = [], []
    for rel in rels:
        real = root / rel
        cur = real.read_bytes() if real.is_file() else None
        if real.is_dir():
            clashes.append(rel)
        if cur != exp.get(rel):
            conflicts.append(rel)
    if clashes:
        raise TxError("A directory exists where a file should be written: " + ", ".join(clashes))
    if conflicts:
        raise ConflictError(conflicts)

    jdir = journal_root / tx_id
    if jdir.exists():
        shutil.rmtree(jdir)
    (jdir / "b").mkdir(parents=True)
    entries, created_dirs = [], []
    for i, (rel, new) in enumerate(sorted(rels.items())):
        real = root / rel
        existed = real.is_file()
        if existed:
            _write_durable(jdir / "b" / str(i), real.read_bytes())
        entries.append({"i": i, "path": rel, "existed": existed, "action": "delete" if new is None else "write"})
        if new is not None:   # directories we will have to create
            d = real.parent
            missing = []
            while not d.exists():
                missing.append(d)
                d = d.parent
            for m in reversed(missing):
                if str(m) not in created_dirs:
                    created_dirs.append(str(m))
    manifest = {"id": tx_id, "state": "prepared", "root": str(root), "entries": entries,
                "created_dirs": created_dirs, "ts": time.time()}
    _write_json_atomic(jdir / "manifest.json", manifest)

    tmps: dict[str, Path] = {}
    try:
        for e in entries:
            new = rels[e["path"]]
            if new is None:
                continue
            real = root / e["path"]
            real.parent.mkdir(parents=True, exist_ok=True)
            tmp = real.parent / f".{real.name}.forge-{tx_id[:8]}.tmp"
            _write_durable(tmp, new)
            tmps[e["path"]] = tmp
        manifest["state"] = "applying"
        _write_json_atomic(jdir / "manifest.json", manifest)
        applied = 0
        for e in entries:
            real = root / e["path"]
            if e["action"] == "write":
                _replace(tmps[e["path"]], real)
            elif real.exists():
                os.remove(real)
            applied += 1
            if fail_after is not None and applied >= fail_after:
                raise TxError("simulated failure (test hook)")
        for d in {(root / e["path"]).parent for e in entries}:
            _fsync_dir(d)
    except BaseException as exc:
        _rollback(jdir, manifest)
        for t in tmps.values():
            t.unlink(missing_ok=True)
        raise TxError(f"Commit failed and was rolled back; no file was changed: {exc}") from exc
    manifest["state"] = "committed"
    _write_json_atomic(jdir / "manifest.json", manifest)
    return manifest


def _rollback(jdir: Path, manifest: dict) -> None:
    root = Path(manifest["root"])
    for e in reversed(manifest["entries"]):
        real = root / e["path"]
        tmp = real.parent / f".{real.name}.forge-{manifest['id'][:8]}.tmp"
        tmp.unlink(missing_ok=True)
        if e["existed"]:
            data = (jdir / "b" / str(e["i"])).read_bytes()
            if not real.is_file() or real.read_bytes() != data:
                real.parent.mkdir(parents=True, exist_ok=True)
                t2 = real.parent / f".{real.name}.forge-rb.tmp"
                _write_durable(t2, data)
                _replace(t2, real)
        elif real.exists():
            os.remove(real)
    for d in sorted(manifest.get("created_dirs", []), key=len, reverse=True):
        try:
            os.rmdir(d)
        except OSError:
            pass
    manifest["state"] = "rolled_back"
    _write_json_atomic(jdir / "manifest.json", manifest)


def recover(journal_root: Path) -> list[str]:
    """Roll back transactions that were interrupted (crash / power loss) before they were marked committed."""
    done = []
    if not journal_root.is_dir():
        return done
    for jdir in journal_root.iterdir():
        mf = jdir / "manifest.json"
        if not mf.is_file():
            continue
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if m.get("state") in ("prepared", "applying"):
            _rollback(jdir, m)
            done.append(m["id"])
    return done


def undo_transaction(root: Path, journal_root: Path, tx_id: str) -> None:
    """Revert a committed run, but only if every file still holds exactly what that run wrote."""
    jdir = journal_root / tx_id
    m = json.loads((jdir / "manifest.json").read_text(encoding="utf-8"))
    if m["state"] != "committed":
        raise TxError(f"Run is in state {m['state']!r}; nothing to undo")
    after = json.loads((jdir / "after.json").read_text(encoding="utf-8"))["hashes"]
    changes, expect, conflicts = {}, {}, []
    for e in m["entries"]:
        real = root / e["path"]
        cur = real.read_bytes() if real.is_file() else None
        if sha(cur) != after.get(e["path"]):
            conflicts.append(e["path"])
        changes[e["path"]] = (jdir / "b" / str(e["i"])).read_bytes() if e["existed"] else None
        expect[e["path"]] = cur
    if conflicts:
        raise ConflictError(conflicts)
    transact(root, journal_root, tx_id + "-undo", changes, expect)
    m["state"] = "undone"
    _write_json_atomic(jdir / "manifest.json", m)


# ---------------------------------------------------------------------------------------------- workspace
class Workspace:
    """A private copy of the repo in which agents edit freely. Real files are never touched until commit."""

    def __init__(self, cfg: Config, run_id: str):
        self.cfg, self.run_id, self.root = cfg, run_id, cfg.repo
        key = hashlib.sha1(str(cfg.repo).encode()).hexdigest()[:10]
        self.run_dir = Path(cfg.stage_root) / key / run_id
        self.tree = self.run_dir / "tree"
        self.journal_root = cfg.forge_dir / "journal"
        self.touched: dict[str, bytes | None] = {}     # path -> staged bytes (None = deleted)
        self.orig: dict[str, bytes | None] = {}        # path -> bytes at the start of the run
        self.sig: dict[str, tuple[int, int]] = {}      # real file (size, mtime_ns) at snapshot time
        self.links: list[str] = []

    # ---------------------------------------------------------------- lifecycle
    def create(self) -> None:
        safe_rmtree(self.run_dir)
        self.tree.mkdir(parents=True)
        files = list_repo_files(self.root, self.cfg.extra_ignore)
        for rel in files:
            src, dst = self.root / rel, self.tree / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dst)
                st = src.stat()
                self.sig[rel] = (st.st_size, st.st_mtime_ns)
            except OSError:
                continue
        self._link_heavy_dirs()
        self._save_state()

    def _link_heavy_dirs(self) -> None:
        def walk(d: Path, depth: int):
            if depth > 3:
                return
            try:
                entries = list(os.scandir(d))
            except OSError:
                return
            for e in entries:
                if not e.is_dir(follow_symlinks=False):
                    continue
                if e.name in LINK_DIRS:
                    rel = os.path.relpath(e.path, self.root).replace(os.sep, "/")
                    if link_dir(Path(e.path), self.tree / rel):
                        self.links.append(rel)
                elif e.name not in DEFAULT_IGNORE_DIRS and not e.name.startswith("."):
                    walk(Path(e.path), depth + 1)
        walk(self.root, 0)

    def resume(self) -> None:
        st = json.loads((self.run_dir / "state.json").read_text(encoding="utf-8"))
        self.sig = {k: tuple(v) for k, v in st["sig"].items()}
        self.links = st.get("links", [])
        for rel in st["touched"]:
            o = self.run_dir / "orig" / rel
            self.orig[rel] = o.read_bytes() if o.is_file() else None
            p = self.tree / rel
            self.touched[rel] = p.read_bytes() if p.is_file() else None

    def _save_state(self) -> None:
        _write_json_atomic(self.run_dir / "state.json", {
            "touched": sorted(self.touched), "sig": self.sig, "links": self.links})

    def cleanup(self) -> None:
        safe_rmtree(self.run_dir)

    # ---------------------------------------------------------------- reads
    def _p(self, rel: str) -> Path:
        return self.tree / norm_rel(rel)

    def exists(self, rel: str) -> bool:
        return self._p(rel).is_file()

    def read_bytes(self, rel: str) -> bytes | None:
        p = self._p(rel)
        return p.read_bytes() if p.is_file() else None

    def read(self, rel: str) -> str | None:
        """UTF-8 text with LF endings, or None when missing / binary / not decodable."""
        if is_secret(norm_rel(rel)):
            return None
        data = self.read_bytes(rel)
        if data is None or is_binary(data):
            return None
        try:
            return data.decode("utf-8-sig").replace("\r\n", "\n")
        except UnicodeDecodeError:
            return None

    # ---------------------------------------------------------------- writes (stage only)
    def _remember_orig(self, rel: str) -> None:
        if rel in self.orig:
            return
        data = self.read_bytes(rel)
        self.orig[rel] = data
        if data is not None:
            o = self.run_dir / "orig" / rel
            o.parent.mkdir(parents=True, exist_ok=True)
            o.write_bytes(data)

    def write(self, rel: str, text: str) -> None:
        rel = norm_rel(rel)
        if is_secret(rel):
            raise TxError(f"Refusing to write secret-like file {rel}")
        self._remember_orig(rel)
        old = self.orig[rel]
        crlf = old is not None and b"\r\n" in old
        bom = old is not None and old.startswith(b"\xef\xbb\xbf")
        data = text.replace("\r\n", "\n")
        if crlf:
            data = data.replace("\n", "\r\n")
        raw = (b"\xef\xbb\xbf" if bom else b"") + data.encode("utf-8")
        p = self.tree / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".forge.tmp")
        tmp.write_bytes(raw)
        os.replace(tmp, p)
        self.touched[rel] = raw
        self._save_state()

    def delete(self, rel: str) -> None:
        rel = norm_rel(rel)
        self._remember_orig(rel)
        (self.tree / rel).unlink(missing_ok=True)
        self.touched[rel] = None
        self._save_state()

    def checkpoint(self) -> dict[str, bytes | None]:
        return dict(self.touched)

    def restore(self, cp: dict[str, bytes | None]) -> None:
        """Return the stage to a checkpoint (used before every retry so failed attempts leave no residue)."""
        for rel in list(self.touched):
            target = cp[rel] if rel in cp else self.orig.get(rel)
            p = self.tree / rel
            if target is None:
                p.unlink(missing_ok=True)
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(target)
            if rel in cp:
                self.touched[rel] = target
            else:
                del self.touched[rel]
        self._save_state()

    # ---------------------------------------------------------------- inspection
    def changed_paths(self) -> list[str]:
        return sorted(p for p, v in self.touched.items() if v != self.orig.get(p))

    def diff(self, paths: list[str] | None = None, context: int = 3) -> str:
        chunks = []
        for rel in (paths or self.changed_paths()):
            a = self._txt(self.orig.get(rel))
            b = self._txt(self.touched.get(rel))
            if a == b:
                continue
            chunks.append("".join(difflib.unified_diff(
                a.splitlines(True), b.splitlines(True),
                fromfile="a/" + rel if self.orig.get(rel) is not None else "/dev/null",
                tofile="b/" + rel if self.touched.get(rel) is not None else "/dev/null", n=context)))
        return "\n".join(chunks)

    def diff_since(self, cp: dict[str, bytes | None], paths: list[str] | None = None, context: int = 3) -> str:
        """Unified diff of what changed after checkpoint `cp` (used to review one subtask in isolation)."""
        chunks = []
        for rel in (paths or sorted(set(self.touched) | set(cp))):
            before = cp[rel] if rel in cp else self.orig.get(rel)
            after = self.touched[rel] if rel in self.touched else self.orig.get(rel)
            a, b = self._txt(before), self._txt(after)
            if a == b:
                continue
            chunks.append("".join(difflib.unified_diff(
                a.splitlines(True), b.splitlines(True), fromfile="a/" + rel if before is not None else "/dev/null",
                tofile="b/" + rel if after is not None else "/dev/null", n=context)))
        return "\n".join(chunks)

    @staticmethod
    def _txt(b: bytes | None) -> str:
        return "" if b is None else b.decode("utf-8", "replace").replace("\r\n", "\n")

    def external_changes(self) -> list[str]:
        """Real files modified by someone else since the snapshot (excluding the ones we are changing)."""
        out = []
        for rel, (size, mt) in self.sig.items():
            if rel in self.touched:
                continue
            try:
                st = (self.root / rel).stat()
            except OSError:
                out.append(rel)
                continue
            if (st.st_size, st.st_mtime_ns) != (size, mt):
                out.append(rel)
        return out

    def pristine_copy(self, dest: Path) -> Path:
        """A copy of the stage with all agent changes reverted (for 'did this already fail before?' baselines)."""
        safe_rmtree(dest)
        shutil.copytree(self.tree, dest, symlinks=True)
        for rel, data in self.orig.items():
            p = dest / rel
            if data is None:
                p.unlink(missing_ok=True)
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
        return dest

    # ---------------------------------------------------------------- commit / undo
    def commit(self, fail_after: int | None = None) -> dict:
        changes = {p: self.touched[p] for p in self.changed_paths()}
        if not changes:
            raise TxError("Nothing to commit")
        expect = {p: self.orig.get(p) for p in changes}
        mf = transact(self.root, self.journal_root, self.run_id, changes, expect, fail_after)
        _write_json_atomic(self.journal_root / self.run_id / "after.json", {"hashes": {p: sha(v) for p, v in changes.items()}})
        return mf

    def undo(self) -> None:
        undo_transaction(self.root, self.journal_root, self.run_id)
