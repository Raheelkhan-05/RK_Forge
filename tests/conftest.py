# tests/conftest.py

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.config import Config  # noqa: E402
from forge.orchestrator import Engine  # noqa: E402


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "proj"
    (r / "pkg").mkdir(parents=True)
    (r / "tests").mkdir()
    (r / "pkg" / "__init__.py").write_text("")
    (r / "pkg" / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (r / "tests" / "test_calc.py").write_text("from pkg.calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    subprocess.run(["git", "init", "-q", str(r)], check=False)
    return r


@pytest.fixture
def cfg(repo, tmp_path):
    c = Config.load(repo)
    c.stage_root = tmp_path / "stages"
    c.limits.cmd_timeout = 120
    return c


class FakeLLM:
    """Scripted stand-in for Ollama. `script(role, kind, messages)` returns a dict (json) or str (chat)."""

    def __init__(self, script, stop=lambda: False):
        self.script, self.stop = script, stop
        self.calls = []

    def health(self):
        return {"ok": True, "installed": [], "loaded": [], "missing": [], "error": None}

    @staticmethod
    def kind(schema):
        if schema is None:
            return "text"
        props = schema.get("properties", {})
        return "plan" if "subtasks" in props else "step" if "action" in props else "review" if "verdict" in props else "summary"

    def chat(self, role, messages, schema=None, on_token=None):
        out = self.script(role, "text", messages)
        self.calls.append((role, "text", messages))
        if on_token:
            on_token("content", out[:20])
        return out

    def chat_json(self, role, messages, schema, on_token=None, retries=2):
        k = self.kind(schema)
        self.calls.append((role, k, messages))
        return self.script(role, k, messages)


def wait_status(engine, rid, wanted, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        s = engine.mem.get_run(rid)["status"]
        if s in wanted:
            return s
        time.sleep(0.1)
    raise AssertionError(f"timeout waiting for {wanted}; status={engine.mem.get_run(rid)['status']}; events={[e['type'] for e in engine.mem.events_since(rid)][-12:]}")


def make_engine(cfg, script):
    return Engine(cfg, llm_factory=lambda stop: FakeLLM(script, stop))
