# tests/test_workspace.py

import json
import pytest
from forge import workspace as W
from forge.workspace import ConflictError, TxError, Workspace


def test_stage_isolated_and_commit_atomic(cfg, repo):
    ws = Workspace(cfg, "r1")
    ws.create()
    ws.write("pkg/calc.py", "def add(a, b):\n    return a + b + 0\n")
    ws.write("pkg/new.py", "X = 1\n")
    assert "a + b + 0" not in (repo / "pkg" / "calc.py").read_text()      # real repo untouched
    assert not (repo / "pkg" / "new.py").exists()
    ws.commit()
    assert "a + b + 0" in (repo / "pkg" / "calc.py").read_text()
    assert (repo / "pkg" / "new.py").exists()
    ws.undo()
    assert (repo / "pkg" / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"
    assert not (repo / "pkg" / "new.py").exists()


def test_failure_mid_commit_rolls_back_everything(cfg, repo):
    before = (repo / "pkg" / "calc.py").read_text()
    ws = Workspace(cfg, "r2")
    ws.create()
    ws.write("pkg/calc.py", "CHANGED\n")
    ws.write("pkg/__init__.py", "CHANGED\n")
    ws.write("pkg/sub/deep.py", "NEW\n")
    with pytest.raises(TxError):
        ws.commit(fail_after=2)
    assert (repo / "pkg" / "calc.py").read_text() == before
    assert (repo / "pkg" / "__init__.py").read_text() == ""
    assert not (repo / "pkg" / "sub").exists()
    assert not list(repo.rglob("*.forge-*.tmp"))


def test_conflict_detected_when_user_edits_same_file(cfg, repo):
    ws = Workspace(cfg, "r3")
    ws.create()
    ws.write("pkg/calc.py", "def add(a, b):\n    return 0\n")
    (repo / "pkg" / "calc.py").write_text("# user edit\n")
    with pytest.raises(ConflictError):
        ws.commit()
    assert (repo / "pkg" / "calc.py").read_text() == "# user edit\n"


def test_crash_recovery_restores_originals(cfg, repo, monkeypatch):
    before = (repo / "pkg" / "calc.py").read_text()
    ws = Workspace(cfg, "r4")
    ws.create()
    ws.write("pkg/calc.py", "CHANGED\n")
    ws.write("pkg/__init__.py", "CHANGED\n")
    monkeypatch.setattr(W, "_rollback", lambda *a, **k: None)      # simulate power loss: no rollback happens
    with pytest.raises(TxError):
        ws.commit(fail_after=1)
    monkeypatch.undo()
    assert (repo / "pkg" / "calc.py").read_text() != before or (repo / "pkg" / "__init__.py").read_text() != ""
    assert "r4" in W.recover(cfg.forge_dir / "journal")
    assert (repo / "pkg" / "calc.py").read_text() == before
    assert (repo / "pkg" / "__init__.py").read_text() == ""


def test_checkpoint_restore_and_crlf_preserved(cfg, repo):
    (repo / "pkg" / "crlf.py").write_bytes(b"a = 1\r\nb = 2\r\n")
    ws = Workspace(cfg, "r5")
    ws.create()
    cp = ws.checkpoint()
    ws.write("pkg/crlf.py", "a = 1\nb = 3\n")
    assert ws.read_bytes("pkg/crlf.py") == b"a = 1\r\nb = 3\r\n"
    ws.write("pkg/other.py", "x\n")
    ws.restore(cp)
    assert ws.read_bytes("pkg/crlf.py") == b"a = 1\r\nb = 2\r\n"
    assert not ws.exists("pkg/other.py")


def test_unsafe_paths_and_secrets_refused(cfg):
    ws = Workspace(cfg, "r6")
    ws.create()
    for bad in ("../x.py", "/etc/passwd", ".git/config", ".forge/x"):
        with pytest.raises(TxError):
            ws.write(bad, "x")
    with pytest.raises(TxError):
        ws.write(".env", "SECRET=1")
