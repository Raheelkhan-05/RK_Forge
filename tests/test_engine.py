# tests/test_engine.py

import json
import time

from conftest import make_engine, wait_status

PLAN = {"understanding": "Add multiply with a test", "questions": [], "subtasks": [
    {"id": "T1", "title": "Add multiply()", "description": "Add multiply(a, b) to pkg/calc.py next to add()", "files_to_modify": ["pkg/calc.py"],
     "files_to_create": [], "files_to_read": [], "focus_symbols": ["add"], "acceptance_criteria": ["multiply(a, b) returns a * b"], "depends_on": []},
    {"id": "T2", "title": "Test multiply()", "description": "Add tests/test_multiply.py", "files_to_modify": [], "files_to_create": ["tests/test_multiply.py"],
     "files_to_read": ["pkg/calc.py"], "focus_symbols": [], "acceptance_criteria": ["a test asserts multiply(3, 4) == 12"], "depends_on": ["T1"]}]}

ACCEPT = {"verdict": "accept", "criteria": [{"criterion": "c", "met": True, "evidence": "diff"}], "issues": [], "feedback_for_coder": ""}


def edit(path, s, r):
    return f"<<<<<<< SEARCH {path}\n{s}\n=======\n{r}\n>>>>>>> REPLACE\n"


def base_script(coder_fn):
    def script(role, kind, msgs):
        if kind == "step":
            return {"thought": "enough", "action": "finish", "answer": "", "confidence": "low", "evidence": []}
        if kind == "plan":
            return PLAN
        if kind == "review":
            return ACCEPT
        if kind == "summary":
            return {"summary": "Added multiply", "facts": [{"kind": "fact", "text": "Arithmetic helpers live in pkg/calc.py", "paths": ["pkg/calc.py"]}]}
        return coder_fn(msgs)
    return script


def test_full_run_is_atomic_and_asks_for_info(cfg, repo):
    state = {"n": 0}

    def coder(msgs):
        user = msgs[-1]["content"]
        state["n"] += 1
        if "T1" in user.split("<subtask", 1)[-1][:40] and "<definition" not in user:
            return "STATUS: NEED_INFO\nUNCERTAIN:\n- LOOKUP symbol add in pkg/calc.py\n"      # asks before touching anything
        if "<subtask id=\"T1\"" in user:
            assert "def add(a, b)" in user and "<definition" in user                          # lookup result reached the coder
            return "STATUS: READY\nSUMMARY: added multiply\nUSES:\n- none\nEDITS:\n" + edit("pkg/calc.py", "def add(a, b):\n    return a + b", "def add(a, b):\n    return a + b\n\n\ndef multiply(a, b):\n    return a * b")
        return ("STATUS: READY\nSUMMARY: test\nUSES:\n- multiply @ pkg/calc.py\nEDITS:\n"
                + edit("tests/test_multiply.py", "", "from pkg.calc import multiply\n\n\ndef test_multiply():\n    assert multiply(3, 4) == 12"))

    eng = make_engine(cfg, base_script(coder))
    rid = eng.submit("add a multiply function", {"require_approval": True})
    wait_status(eng, rid, {"awaiting_approval"})
    # Isolation: nothing in the real repo changed while waiting for approval
    assert "multiply" not in (repo / "pkg" / "calc.py").read_text()
    assert not (repo / "tests" / "test_multiply.py").exists()
    run = eng.mem.get_run(rid)
    assert [t["status"] for t in run["tasks"]] == ["accepted", "accepted"]
    assert any(e["type"] == "gate_result" and e["payload"]["name"] == "test" and e["payload"]["ok"] for e in eng.mem.events_since(rid))
    eng.control(rid, "approve")
    wait_status(eng, rid, {"done"})
    assert "def multiply" in (repo / "pkg" / "calc.py").read_text() and (repo / "tests" / "test_multiply.py").exists()
    assert any(f["kind"] == "fact" for f in eng.mem.list_facts())
    eng.undo(rid)
    assert "multiply" not in (repo / "pkg" / "calc.py").read_text() and not (repo / "tests" / "test_multiply.py").exists()


def test_failed_gate_triggers_retry_and_never_leaks_bad_code(cfg, repo):
    attempts = {"n": 0}
    one = dict(PLAN, subtasks=PLAN["subtasks"][:1])

    def script(role, kind, msgs):
        if kind == "step":
            return {"thought": "x", "action": "finish"}
        if kind == "plan":
            return one
        if kind == "review":
            return ACCEPT
        if kind == "summary":
            return {"summary": "s", "facts": []}
        attempts["n"] += 1
        if attempts["n"] == 1:   # breaks the existing test
            return "STATUS: READY\nSUMMARY: bad\nUSES:\n- none\nEDITS:\n" + edit("pkg/calc.py", "return a + b", "return a - b")
        assert "Objective checks failed" in msgs[-1]["content"]          # real test output was fed back
        return "STATUS: READY\nSUMMARY: good\nUSES:\n- none\nEDITS:\n" + edit("pkg/calc.py", "return a + b", "return a + b  # ok\n\n\ndef multiply(a, b):\n    return a * b")

    eng = make_engine(cfg, script)
    rid = eng.submit("add multiply", {"require_approval": False})
    wait_status(eng, rid, {"done"})
    text = (repo / "pkg" / "calc.py").read_text()
    assert "a - b" not in text and "def multiply" in text and attempts["n"] == 2


def test_exhausted_attempts_leave_repo_untouched(cfg, repo):
    one = dict(PLAN, subtasks=PLAN["subtasks"][:1])
    before = (repo / "pkg" / "calc.py").read_text()

    def script(role, kind, msgs):
        if kind == "step":
            return {"thought": "x", "action": "finish"}
        if kind == "plan":
            return one
        if kind == "review":
            return ACCEPT
        if kind == "summary":
            return {"summary": "s", "facts": []}
        return "STATUS: READY\nSUMMARY: bad\nUSES:\n- none\nEDITS:\n" + edit("pkg/calc.py", "return a + b", "return a - b")

    cfg.limits.max_replans = 0
    eng = make_engine(cfg, script)
    rid = eng.submit("add multiply", {"require_approval": False})
    wait_status(eng, rid, {"blocked"})
    assert (repo / "pkg" / "calc.py").read_text() == before            # failed attempts never touch real files
    eng.control(rid, "abort")
    assert wait_status(eng, rid, {"failed"}) == "failed"
    assert (repo / "pkg" / "calc.py").read_text() == before


def test_unresolvable_doubt_goes_to_the_user(cfg, repo):
    one = dict(PLAN, subtasks=PLAN["subtasks"][:1])
    seen = {}

    def script(role, kind, msgs):
        if kind == "step":
            return {"thought": "x", "action": "finish", "answer": "", "confidence": "low", "evidence": []}
        if kind == "plan":
            return one
        if kind == "review":
            return ACCEPT
        if kind == "summary":
            return {"summary": "s", "facts": []}
        user = msgs[-1]["content"]
        if "<answers_from_user>" not in user:
            return "STATUS: NEED_INFO\nUNCERTAIN:\n- ASK Should multiply round the result to 2 decimals?\n"
        seen["answers"] = user
        return "STATUS: READY\nSUMMARY: done\nUSES:\n- none\nEDITS:\n" + edit("pkg/calc.py", "return a + b", "return a + b\n\n\ndef multiply(a, b):\n    return round(a * b, 2)")

    eng = make_engine(cfg, script)
    rid = eng.submit("add multiply", {"require_approval": False})
    wait_status(eng, rid, {"awaiting_user"})
    pend = eng.pending(rid)
    assert pend["kind"] == "doubt" and "round" in pend["questions"][0]
    assert "multiply" not in (repo / "pkg" / "calc.py").read_text()       # nothing applied before the question was answered
    eng.answer(rid, ["Yes, round to 2 decimals"])
    wait_status(eng, rid, {"done"})
    assert "Yes, round to 2 decimals" in seen["answers"] and "round(a * b, 2)" in (repo / "pkg" / "calc.py").read_text()
    assert any(f["kind"] == "answer" for f in eng.mem.list_facts())


def test_hallucinated_symbol_blocks_edit_until_resolved(cfg, repo):
    one = dict(PLAN, subtasks=PLAN["subtasks"][:1])
    n = {"c": 0}

    def script(role, kind, msgs):
        if kind == "step":
            return {"thought": "x", "action": "finish", "answer": "", "confidence": "low", "evidence": []}
        if kind == "plan":
            return one
        if kind == "review":
            return ACCEPT
        if kind == "summary":
            return {"summary": "s", "facts": []}
        n["c"] += 1
        if n["c"] == 1:   # claims to rely on a helper that does not exist
            return "STATUS: READY\nSUMMARY: x\nUSES:\n- computeTaxRate @ pkg/calc.py\nEDITS:\n" + edit("pkg/calc.py", "return a + b", "return a + b + computeTaxRate()")
        return "STATUS: READY\nSUMMARY: x\nUSES:\n- none\nEDITS:\n" + edit("pkg/calc.py", "return a + b", "return a + b\n\n\ndef multiply(a, b):\n    return a * b")

    cfg.limits.max_lookup_rounds = 1
    eng = make_engine(cfg, script)
    rid = eng.submit("add multiply", {"require_approval": False})
    wait_status(eng, rid, {"awaiting_user"})
    assert any(e["type"] == "grounding" for e in eng.mem.events_since(rid))
    eng.answer(rid, ["There is no computeTaxRate; do not use it"])
    wait_status(eng, rid, {"done"})
    assert "computeTaxRate" not in (repo / "pkg" / "calc.py").read_text()


def test_reject_and_cancel_leave_repo_untouched(cfg, repo):
    one = dict(PLAN, subtasks=PLAN["subtasks"][:1])

    def script(role, kind, msgs):
        if kind == "step":
            return {"thought": "x", "action": "finish"}
        if kind == "plan":
            return one
        if kind == "review":
            return ACCEPT
        if kind == "summary":
            return {"summary": "s", "facts": []}
        return "STATUS: READY\nSUMMARY: x\nUSES:\n- none\nEDITS:\n" + edit("pkg/calc.py", "return a + b", "return a + b\n\n\ndef multiply(a, b):\n    return a * b")

    before = (repo / "pkg" / "calc.py").read_text()
    eng = make_engine(cfg, script)
    rid = eng.submit("add multiply", {"require_approval": True})
    wait_status(eng, rid, {"awaiting_approval"})
    eng.control(rid, "reject")
    wait_status(eng, rid, {"rejected"})
    assert (repo / "pkg" / "calc.py").read_text() == before
