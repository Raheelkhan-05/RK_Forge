# forge/verify.py

"""Objective quality gates. Nothing is accepted on the model's word; these decide.

Gates run inside the staging workspace. Scoped (changed files only) where the tool supports it, otherwise the
project's own script. A failing gate is compared with the same gate on the untouched code, so failures that
already existed before the change are reported but never blamed on the agent.
"""
from __future__ import annotations

import ast
import json
import os
import posixpath
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

from .config import Config
from .indexer import Index
from .llm import Cancelled
from .workspace import Workspace

JS_EXT = (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx")
JSON_LOOSE = re.compile(r"(tsconfig|jsconfig).*\.json$|\.jsonc$|^\.eslintrc|^\.babelrc|settings\.json$")


@dataclass
class GateResult:
    name: str
    project: str
    ok: bool
    cmd: str = ""
    skipped: bool = False
    reason: str = ""
    exit_code: int | None = None
    seconds: float = 0.0
    output: str = ""
    preexisting: bool = False

    @property
    def blocking(self) -> bool:
        return not self.ok and not self.skipped and not self.preexisting

    def public(self, max_out: int = 4000) -> dict:
        d = asdict(self)
        d["output"] = self.output[-max_out:]
        d["blocking"] = self.blocking
        return d


def condense_output(text: str, max_chars: int = 3200) -> str:
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    if len(text) <= max_chars:
        return text.strip()
    lines = text.split("\n")
    keep = [i for i, ln in enumerate(lines) if re.search(r"(?i)error|fail|✗|×|exception|traceback|expected|received|cannot|unexpected|not found|undefined|warning:", ln)]
    picked: list[str] = []
    seen: set[int] = set()
    for i in keep:
        for j in range(max(0, i - 1), min(len(lines), i + 3)):
            if j not in seen:
                seen.add(j)
                picked.append(lines[j])
        if sum(len(x) + 1 for x in picked) > max_chars:
            break
    out = "\n".join(picked) if picked else "\n".join(lines[-40:])
    return out[:max_chars].strip()


def error_signature(text: str) -> set[str]:
    sig = set()
    for ln in re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text).split("\n"):
        if re.search(r"(?i)error|fail|✗|×|exception|cannot|unexpected", ln):
            sig.add(re.sub(r"\d+", "N", re.sub(r"\s+", " ", ln.strip()))[:200])
    return sig


def kill_tree(proc: subprocess.Popen) -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        proc.kill()


def run_cmd(cmd: list[str] | str, cwd: Path, timeout: int, should_stop: Callable[[], bool]) -> tuple[int, str, float]:
    env = dict(os.environ, CI="1", FORCE_COLOR="0", NO_COLOR="1")
    kw: dict = {"cwd": str(cwd), "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "env": env,
                "text": True, "encoding": "utf-8", "errors": "replace", "shell": isinstance(cmd, str)}
    if sys.platform == "win32":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    t0 = time.time()
    try:
        proc = subprocess.Popen(cmd, **kw)
    except OSError as e:
        return 127, f"Could not start command: {e}", 0.0
    out = ""
    while True:
        try:
            out, _ = proc.communicate(timeout=1)
            break
        except subprocess.TimeoutExpired:
            if should_stop():
                kill_tree(proc)
                raise Cancelled()
            if time.time() - t0 > timeout:
                kill_tree(proc)
                try:
                    out, _ = proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    out = ""
                return 124, (out or "") + f"\n[timed out after {timeout}s]", time.time() - t0
    return proc.returncode, out or "", time.time() - t0


class Verifier:
    def __init__(self, cfg: Config, index: Index, ws: Workspace, should_stop: Callable[[], bool] = lambda: False):
        self.cfg, self.ix, self.ws, self.should_stop = cfg, index, ws, should_stop
        self.tree = ws.tree
        self._baseline_dir: Path | None = None
        self._baseline_cache: dict[tuple[str, str], GateResult] = {}
        self.node = shutil.which("node")

    # ------------------------------------------------------------------ discovery
    def _pkg(self, d: str, root: Path | None = None) -> dict:
        p = (root or self.tree) / d / "package.json"
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _bin(self, d: str, rel: str, root: Path | None = None) -> Path | None:
        """Find node_modules/<rel> in the project dir or any ancestor (monorepo hoisting)."""
        base = root or self.tree
        cur = d
        while True:
            p = base / cur / "node_modules" / rel if cur else base / "node_modules" / rel
            if p.exists():
                return p
            if not cur:
                return None
            cur = posixpath.dirname(cur)

    def _script(self, d: str, names: tuple[str, ...], root: Path | None = None) -> str | None:
        scripts = self._pkg(d, root).get("scripts", {})
        for n in names:
            v = scripts.get(n)
            if v and "no test specified" not in v:
                return n
        return None

    def _custom(self, d: str, gate: str) -> str | None:
        name = self.ix.project_name(d)
        return self.cfg.verify.commands.get(name, {}).get(gate) or self.cfg.verify.commands.get(d, {}).get(gate)

    # ------------------------------------------------------------------ gates
    def run(self, gates: list[str], changed: list[str], emit: Callable[..., None] = lambda *a, **k: None,
            stage: str = "task") -> list[GateResult]:
        results: list[GateResult] = []
        projects: dict[str, list[str]] = {}
        for p in changed:
            projects.setdefault(self.ix.project_of(p), []).append(p)
        for gate in gates:
            fn = getattr(self, f"_gate_{gate}", None)
            if not fn:
                continue
            for d, files in sorted(projects.items()):
                if self.should_stop():
                    raise Cancelled()
                emit("gate_start", gate=gate, project=self.ix.project_name(d))
                res = fn(d, files)
                for r in (res if isinstance(res, list) else [res]):
                    if r.blocking and gate in ("lint", "typecheck", "test", "build"):
                        base = self._baseline(gate, d, files)
                        if base and not base.ok and not (error_signature(r.output) - error_signature(base.output)):
                            r.preexisting = True
                            r.reason = "This gate already failed the same way before the change"
                    results.append(r)
                    emit("gate_result", **r.public(1500))
        return results

    def _baseline(self, gate: str, d: str, files: list[str]) -> GateResult | None:
        key = (gate, d)
        if key in self._baseline_cache:
            return self._baseline_cache[key]
        if self._baseline_dir is None:
            self._baseline_dir = self.ws.pristine_copy(self.ws.run_dir / "baseline")
        fn = getattr(self, f"_gate_{gate}")
        res = fn(d, [f for f in files if (self._baseline_dir / f).exists()], root=self._baseline_dir)
        res = res[0] if isinstance(res, list) else res
        self._baseline_cache[key] = res
        return res

    def cleanup(self) -> None:
        if self._baseline_dir:
            from .workspace import safe_rmtree
            safe_rmtree(self._baseline_dir)

    def _exec(self, name: str, d: str, cmd: list[str] | str, root: Path | None = None) -> GateResult:
        cwd = (root or self.tree) / d if d else (root or self.tree)
        code, out, secs = run_cmd(cmd, cwd, self.cfg.limits.cmd_timeout, self.should_stop)
        return GateResult(name, self.ix.project_name(d), code == 0, cmd if isinstance(cmd, str) else " ".join(cmd),
                          exit_code=code, seconds=round(secs, 1), output=condense_output(out, 6000))

    def _skip(self, name: str, d: str, why: str) -> GateResult:
        return GateResult(name, self.ix.project_name(d), True, skipped=True, reason=why)

    def _gate_syntax(self, d: str, files: list[str], root: Path | None = None) -> list[GateResult]:
        root = root or self.tree
        problems, notes = [], []
        js_files = []
        for rel in files:
            p = root / rel
            if not p.is_file():
                continue
            name = rel.rsplit("/", 1)[-1]
            text = p.read_text(encoding="utf-8", errors="replace")
            if rel.endswith(".py"):
                try:
                    ast.parse(text)
                except SyntaxError as e:
                    problems.append(f"{rel}:{e.lineno}: {e.msg}")
            elif rel.endswith(".json") and not JSON_LOOSE.search(name):
                try:
                    json.loads(text)
                except json.JSONDecodeError as e:
                    problems.append(f"{rel}:{e.lineno}: invalid JSON ({e.msg})")
            elif rel.endswith(".sql"):
                if text.count("$$") % 2:
                    problems.append(f"{rel}: unbalanced $$ quoting")
                if text.count("(") != text.count(")"):
                    notes.append(f"{rel}: parentheses look unbalanced")
            elif rel.endswith(JS_EXT):
                js_files.append(rel)
        if js_files:
            esb = self._bin(d, "esbuild/bin/esbuild", root)
            if esb and self.node:
                outdir = tempfile.mkdtemp(prefix="forge-esb-")
                try:
                    args = [self.node, str(esb), *[str(root / f) for f in js_files], f"--outdir={outdir}", "--log-level=error",
                            "--loader:.js=jsx", "--loader:.mjs=jsx", "--loader:.cjs=jsx", "--loader:.ts=ts", "--loader:.tsx=tsx", "--color=false"]
                    code, out, _ = run_cmd(args, root, 120, self.should_stop)
                    if code != 0:
                        problems.append(condense_output(out.replace(str(root) + os.sep, ""), 2000))
                finally:
                    shutil.rmtree(outdir, ignore_errors=True)
            else:
                notes.append("esbuild not found in node_modules: JS/TS syntax is covered by the lint/typecheck/build gates only")
        r = GateResult("syntax", self.ix.project_name(d), not problems, reason="; ".join(notes), output="\n".join(problems))
        return [r]

    def _gate_imports(self, d: str, files: list[str], root: Path | None = None) -> GateResult:
        probs: list[str] = []
        for rel in files:
            if (self.tree / rel).is_file():
                probs += self.ix.check_imports(rel)
        return GateResult("imports", self.ix.project_name(d), not probs, output="\n".join(probs[:30]))

    def _gate_lint(self, d: str, files: list[str], root: Path | None = None):
        root = root or self.tree
        if (c := self._custom(d, "lint")):
            return self._exec("lint", d, c, root)
        lintable = [posixpath.relpath(f, d or ".") for f in files if f.endswith(JS_EXT) and (root / f).is_file()]
        eslint = self._bin(d, "eslint/bin/eslint.js", root)
        has_cfg = any(list((root / d).glob(g)) for g in ("eslint.config.*", ".eslintrc*")) or "eslintConfig" in self._pkg(d, root) \
            or any(list(root.glob(g)) for g in ("eslint.config.*", ".eslintrc*"))
        if lintable and eslint and has_cfg and self.node:
            return self._exec("lint", d, [self.node, str(eslint), "--no-error-on-unmatched-pattern", *lintable], root)
        if (s := self._script(d, ("lint",), root)) and lintable:
            return self._exec("lint", d, f"npm run {s} --silent", root)
        return self._skip("lint", d, "no linter configured")

    def _gate_typecheck(self, d: str, files: list[str], root: Path | None = None):
        root = root or self.tree
        if (c := self._custom(d, "typecheck")):
            return self._exec("typecheck", d, c, root)
        if not any(f.endswith((".ts", ".tsx")) for f in files):
            return self._skip("typecheck", d, "no TypeScript files changed")
        if (s := self._script(d, ("typecheck", "type-check", "tsc", "check-types"), root)):
            return self._exec("typecheck", d, f"npm run {s} --silent", root)
        tsc = self._bin(d, "typescript/bin/tsc", root)
        if tsc and self.node and (root / d / "tsconfig.json").is_file():
            return self._exec("typecheck", d, [self.node, str(tsc), "--noEmit", "-p", "."], root)
        return self._skip("typecheck", d, "TypeScript not installed for this project")

    def _gate_test(self, d: str, files: list[str], root: Path | None = None):
        root = root or self.tree
        if (c := self._custom(d, "test")):
            return self._exec("test", d, c, root)
        rel = [posixpath.relpath(f, d or ".") for f in files if (root / f).is_file()]
        pyf = [f for f in files if f.endswith(".py")]
        if pyf and (root / d / "tests").is_dir():
            tests = sorted({t for f in pyf for t in self.ix.tests_for(f)} | {f for f in pyf if "test" in f})
            tests = [posixpath.relpath(t, d or ".") for t in tests if (root / t).is_file()]
            return self._exec("test", d, [sys.executable, "-m", "pytest", "-x", "-q", *(tests or ["tests"])], root)
        vitest = self._bin(d, "vitest/vitest.mjs", root)
        jest = self._bin(d, "jest/bin/jest.js", root)
        jsrel = [r for r, f in zip(rel, files) if f.endswith(JS_EXT)]
        if jsrel and self.node:
            if vitest:
                return self._exec("test", d, [self.node, str(vitest), "related", "--run", "--passWithNoTests", *jsrel], root)
            if jest:
                return self._exec("test", d, [self.node, str(jest), "--findRelatedTests", "--passWithNoTests", *jsrel], root)
        if (s := self._script(d, ("test",), root)) and (jsrel or pyf):
            return self._exec("test", d, f"npm run {s} --silent", root)
        return self._skip("test", d, "no test runner configured")

    def _gate_build(self, d: str, files: list[str], root: Path | None = None):
        root = root or self.tree
        if (c := self._custom(d, "build")):
            return self._exec("build", d, c, root)
        if (s := self._script(d, ("build",), root)):
            return self._exec("build", d, f"npm run {s} --silent", root)
        return self._skip("build", d, "no build script")
