"""Chat layer tests. They use a scripted stand-in for Ollama and a small fake engine, so they need no model and no repo."""
import threading

from click import prompt
import pytest

from forge.chat import ChatService, compose_result, heuristic_title
from forge.config import Config
from forge.intent import analyze, code_names
from forge.llm import LLMError
from forge.memory import Memory


class FakeLLM:
    """Scripted Ollama. `script(role, kind, messages)` returns a dict (JSON calls) or a str (streamed reply)."""

    def __init__(self, script, stop=lambda: False):
        self.script, self.stop, self.calls = script, stop, []

    @staticmethod
    def kind(schema):
        p = schema.get("properties", {})
        if "intent" in p:
            return "route"
        if "request" in p:
            return "brief"
        if "forget" in p:
            return "facts"
        if "title" in p:
            return "title"
        if "summary" in p:
            return "digest"
        return "json"

    def chat_json(self, role, messages, schema, on_token=None, retries=2):
        k = self.kind(schema)
        self.calls.append((role, k, messages))
        return self.script(role, k, messages)

    def chat_stream(self, role, messages, on_token=None, think=None):
        self.calls.append((role, "stream", messages))
        out = self.script(role, "stream", messages)
        if on_token:
            for i in range(0, len(out), 7):
                on_token("content", out[i:i + 7])
        return out, False


class FakeEngine:
    def __init__(self, cfg, script):
        self.cfg, self.mem, self.lock, self.ix = cfg, Memory(cfg), threading.RLock(), None
        self.submitted = []
        self.llm_factory = lambda stop: FakeLLM(script, stop)

    def submit(self, request, options=None):
        rid = self.mem.create_run(request, options or {})
        self.submitted.append((rid, request, options or {}))
        return rid

    def pending(self, rid):
        return None


ROUTE_DISCUSS = {"intent": "discuss", "confidence": "high", "execute_now": False, "blocking_questions": []}
BRIEF = {"title": "Make GST optional", "request": "Make the GST field optional on the seller listing form and keep validation for other fields.",
         "keep_working": ["existing validation"], "assumptions": ["backend accepts null"], "open_questions": []}


def make(tmp_path, **parts):
    def script(role, kind, msgs):
        fn = parts.get(kind)
        if fn is None:
            return {"route": ROUTE_DISCUSS, "brief": BRIEF, "facts": {"facts": [], "forget": []}, "title": {"title": "Chat title"},
                    "digest": {"summary": "Summary of the earlier part."}, "stream": "Sure."}[kind]
        return fn(msgs) if callable(fn) else fn
    cfg = Config.load(tmp_path)
    cfg.stage_root = tmp_path / "stages"
    eng = FakeEngine(cfg, script)
    return ChatService(eng, start_worker=False), eng


def texts(eng, cid, kind=None):
    return [m for m in eng.mem.recent_msgs(cid, 500) if kind is None or m["kind"] == kind]


# ------------------------------------------------------------------ wording signals
def test_signals_confirm_decline_hedge_go():
    assert analyze("yes").confirm and analyze("ok, go ahead").confirm and analyze("sounds good, do it").confirm
    assert not analyze("yes but how would that affect the cache?").confirm
    assert analyze("not yet").decline and analyze("wait").decline and not analyze("no, change the title too").decline
    assert analyze("maybe we should make GST optional?").hedge
    assert analyze("Make GST optional, go ahead").go and not analyze("don't go ahead yet").go
    assert analyze("can you add a remember me checkbox").imperative
    assert not analyze("why does the cache expire").imperative
    assert analyze("remember that we use Supabase").remember


def test_code_names_ignore_plain_words():
    assert code_names("fix BrandsPage.jsx and useChat in src/api.js") == ["BrandsPage.jsx", "useChat", "src/api.js"]
    assert code_names("how does the search work here") == []


def test_heuristic_title():
    assert heuristic_title("hey can you explain how the renew button works?") == "How the renew button works"
    assert heuristic_title("hi") == "Hi"


# ------------------------------------------------------------------ memory
def test_memory_conversations_search_and_facts(tmp_path):
    mem = Memory(Config.load(tmp_path))
    c = mem.create_conv()
    a = mem.add_msg(c["id"], "user", "text", "We chose PelicanSync as the nightly job name")
    for i in range(5):
        mem.add_msg(c["id"], "assistant", "text", f"filler {i}")
    hits = mem.search_msgs(c["id"], "what was PelicanSync", before_id=10_000)
    assert hits and hits[0]["id"] == a["id"]
    assert mem.search_convs("PelicanSync")[0]["id"] == c["id"]
    f1 = mem.add_fact("decision", "The seller GST field is optional on the listing form", "chat:x")
    assert mem.add_fact("decision", "the seller GST field is optional on listing form", "chat:x") == f1       # near duplicate
    mem.add_fact("preference", "The user wants short explanations", "chat:x")
    got = mem.recall("unrelated words here", [], lambda p: None, always=("preference",))
    assert any(f["kind"] == "preference" for f in got)
    mem.delete_conv(c["id"])
    assert mem.get_conv(c["id"]) is None and mem.search_msgs(c["id"], "PelicanSync", 10_000) == []


# ------------------------------------------------------------------ routing safety
def test_discuss_reply_title_and_no_task(tmp_path):
    svc, eng = make(tmp_path, stream="The renew button resets validity because it recalculates the expiry.", title={"title": "Renew button validity"})
    svc.cfg.chat.llm_titles = True
    cid = svc.new_chat()["id"]
    svc.post(cid, "Why does the renew button reset validity?")
    svc.drain()
    msgs = texts(eng, cid, "text")
    assert [m["role"] for m in msgs] == ["user", "assistant"] and "recalculates" in msgs[1]["content"]
    assert eng.submitted == []
    assert eng.mem.get_conv(cid)["title"] == "Renew button validity"


def test_unclear_task_asks_questions_and_starts_nothing(tmp_path):
    svc, eng = make(tmp_path, route={"intent": "task", "confidence": "low", "execute_now": False,
                                     "blocking_questions": ["Which form should the field be optional on?"]})
    cid = svc.new_chat()["id"]
    svc.post(cid, "make that field optional")
    svc.drain()
    last = texts(eng, cid, "text")[-1]
    assert "Which form" in last["content"] and eng.submitted == []


def test_clear_task_proposes_brief_then_go_ahead_starts_it_once(tmp_path):
    svc, eng = make(tmp_path, route={"intent": "task", "confidence": "medium", "execute_now": False, "blocking_questions": []})
    cid = svc.new_chat()["id"]
    svc.post(cid, "Make GST optional on the seller listing form")
    svc.drain()
    briefs = texts(eng, cid, "brief")
    assert len(briefs) == 1 and briefs[0]["meta"]["status"] == "pending" and eng.submitted == []
    svc.post(cid, "go ahead")
    svc.drain()
    assert len(eng.submitted) == 1 and "GST" in eng.submitted[0][1] and eng.submitted[0][2]["conv_id"] == cid
    assert eng.mem.get_msg(briefs[0]["id"])["meta"]["status"] == "started"
    assert len(texts(eng, cid, "run")) == 1
    svc.post(cid, "go ahead")          # nothing is pending any more: it must not start a second run
    svc.drain()
    assert len(eng.submitted) == 1


def test_auto_start_only_on_explicit_go_and_never_when_hedging(tmp_path):
    route = {"intent": "task", "confidence": "high", "execute_now": True, "blocking_questions": []}
    svc, eng = make(tmp_path, route=route)
    cid = svc.new_chat()["id"]
    svc.post(cid, "Make GST optional on the seller form and go ahead")
    svc.drain()
    assert len(eng.submitted) == 1

    svc2, eng2 = make(tmp_path / "b", route=route)
    (tmp_path / "b").mkdir(exist_ok=True)
    cid2 = svc2.new_chat()["id"]
    svc2.post(cid2, "maybe we should make GST optional, what do you think?")
    svc2.drain()
    assert eng2.submitted == [] and len(texts(eng2, cid2, "brief")) == 1


def test_auto_start_can_be_switched_off(tmp_path):
    svc, eng = make(tmp_path, route={"intent": "task", "confidence": "high", "execute_now": True, "blocking_questions": []})
    svc.cfg.chat.auto_start = False
    cid = svc.new_chat()["id"]
    svc.post(cid, "Make GST optional on the seller form, do it")
    svc.drain()
    assert eng.submitted == [] and len(texts(eng, cid, "brief")) == 1


def test_decline_clears_pending_brief(tmp_path):
    svc, eng = make(tmp_path, route={"intent": "task", "confidence": "medium", "execute_now": False, "blocking_questions": []})
    cid = svc.new_chat()["id"]
    svc.post(cid, "Make GST optional on the seller listing form")
    svc.drain()
    svc.post(cid, "not yet")
    svc.drain()
    assert eng.mem.get_conv(cid)["state"].get("brief") is None and eng.submitted == []
    assert texts(eng, cid, "brief")[0]["meta"]["status"] == "dismissed"


def test_brief_buttons(tmp_path):
    svc, eng = make(tmp_path, route={"intent": "task", "confidence": "medium", "execute_now": False, "blocking_questions": []})
    cid = svc.new_chat()["id"]
    svc.post(cid, "Make GST optional on the seller listing form")
    svc.drain()
    svc.brief_action(cid, "start", request="Edited request text that is long enough.")
    assert len(eng.submitted) == 1 and "Edited request" in eng.submitted[0][1]
    with pytest.raises(ValueError):
        svc.brief_action(cid, "start")


def test_model_down_gives_retryable_notice(tmp_path):
    state = {"down": True}

    def route(msgs):
        if state["down"]:
            raise LLMError("Ollama request failed after 3 attempts: refused")
        return ROUTE_DISCUSS
    svc, eng = make(tmp_path, route=route, stream="Back online.")
    cid = svc.new_chat()["id"]
    m = svc.post(cid, "hello there, are you up?")
    svc.drain()
    notice = texts(eng, cid, "notice")[0]
    assert notice["meta"]["error"] and notice["meta"]["retry_of"] == m["id"]
    state["down"] = False
    svc.retry(cid, m["id"])
    svc.drain()
    assert texts(eng, cid, "notice") == [] and texts(eng, cid, "text")[-1]["content"] == "Back online."


# ------------------------------------------------------------------ context + memory
def test_long_chat_recalls_early_details(tmp_path):
    svc, eng = make(tmp_path, stream="PelicanSync was the nightly sync job.")
    cid = svc.new_chat()["id"]
    eng.mem.add_msg(cid, "user", "text", "We decided the nightly sync job will be called PelicanSync and run at 2am.")
    eng.mem.add_msg(cid, "assistant", "text", "Noted: PelicanSync nightly at 2am.")
    for i in range(45):
        eng.mem.add_msg(cid, "user" if i % 2 == 0 else "assistant", "text", f"unrelated filler number {i}. " + "lorem ipsum dolor " * 40)
    svc.post(cid, "Remind me, what was the name of the nightly sync job we picked?")
    svc.drain()
    stream_calls = [c for c in svc.llm_fg.calls if c[1] == "stream"]
    prompt = "\n".join(m["content"] for m in stream_calls[-1][2])
    assert "PelicanSync" in prompt
    total = sum(len(m["content"]) for m in stream_calls[-1][2])
    assert total < svc.cfg.roles["chat"].prompt_budget() * 3.3    # stays inside the token budget

def test_fix_and_decide_signals():
    assert analyze("fix it").fix_ref and analyze("please try again").fix_ref and analyze("fix yourself").fix_ref
    assert not analyze("fix the login bug").fix_ref
    assert analyze("you decide").decide and analyze("I don't know").decide and not analyze("use Supabase").decide


def test_fix_it_after_a_failed_task_retries_it_without_questions(tmp_path):
    svc, eng = make(tmp_path)
    cid = svc.new_chat()["id"]
    rid = eng.submit("Make GST optional on the seller listing form", {"conv_id": cid, "title": "Make GST optional"})
    eng.mem.add_msg(cid, "assistant", "run", "Started task: Make GST optional", {"run_id": rid, "title": "Make GST optional"})
    eng.mem.update_run(rid, status="failed", error="tests failed")
    eng.mem.emit(rid, "status", status="failed")
    assert len(texts(eng, cid, "run_result")) == 1
    svc.post(cid, "fix it")
    svc.drain()
    assert len(eng.submitted) == 2 and "tests failed" in eng.submitted[1][1]


def test_you_decide_answers_a_waiting_task_with_the_safest_choice(tmp_path):
    svc, eng = make(tmp_path)
    cid = svc.new_chat()["id"]
    given = []
    eng.pending = lambda rid: {"kind": "clarification", "questions": ["Which form?", "Which field?"]}
    eng.answer = lambda rid, answers: given.append(answers) or True
    eng.mem.patch_state(cid, active_run="r1")
    svc.post(cid, "I don't know, you decide")
    svc.drain()
    assert given == [["", ""]]
    assert "safest" in texts(eng, cid, "text")[-1]["content"]


def test_run_as_task_turns_open_questions_into_assumptions(tmp_path):
    brief = dict(BRIEF, open_questions=["Which framework is used?"])
    svc, eng = make(tmp_path, brief=brief)
    cid = svc.new_chat()["id"]
    svc.post(cid, "/task Make GST optional on the seller listing form")
    svc.drain()
    assert len(eng.submitted) == 1 and "safest option" in eng.submitted[0][1]


def test_no_change_result_is_not_reported_as_applied():
    assert "Nothing needed" in compose_result({"status": "done", "files": [], "summary": "Already optional.", "tasks": []})

def test_facts_are_extracted_superseded_and_recalled(tmp_path):
    seen = {}

    def facts(msgs):
        if "seller GST field is optional" in msgs[-1]["content"] and "[" in msgs[-1]["content"].split("<known_facts>")[1].split("</known_facts>")[0] and "required" in msgs[-1]["content"]:
            old = msgs[-1]["content"].split("[", 1)[1].split("]", 1)[0]
            return {"facts": [{"kind": "decision", "text": "The seller GST field is required again on the listing form", "supersedes": [old]}], "forget": []}
        return {"facts": [{"kind": "decision", "text": "The seller GST field is optional on the listing form", "supersedes": []}], "forget": []}

    def stream(msgs):
        seen["system"] = "\n".join(m["content"] for m in msgs)
        return "ok"
    svc, eng = make(tmp_path, facts=facts, stream=stream)
    cid = svc.new_chat()["id"]
    svc.post(cid, "We decided the seller GST field is optional on the listing form because many sellers are unregistered.")
    svc.drain()
    texts_ = [f["text"] for f in eng.mem.list_facts()]
    assert "The seller GST field is optional on the listing form" in texts_
    svc.post(cid, "What did we decide about the GST field?")
    svc.drain()
    assert "optional on the listing form" in seen["system"]
    svc.post(cid, "Actually the seller GST field is required again, the accountant said so, please remember that.")
    svc.drain()
    texts_ = [f["text"] for f in eng.mem.list_facts()]
    assert "The seller GST field is required again on the listing form" in texts_
    assert "The seller GST field is optional on the listing form" not in texts_


def test_secrets_are_never_saved(tmp_path):
    svc, eng = make(tmp_path, facts={"facts": [{"kind": "fact", "text": "The api_key = sk-abcdefghijklmnopqrstuvwx is used for billing", "supersedes": []}], "forget": []})
    cid = svc.new_chat()["id"]
    svc.post(cid, "remember that the api key is stored in the billing service config, please keep it safe")
    svc.drain()
    assert eng.mem.list_facts() == []


def test_slash_remember_and_forget(tmp_path):
    svc, eng = make(tmp_path)
    cid = svc.new_chat()["id"]
    svc.post(cid, "/remember The admin panel is deployed on Vercel")
    assert [f["text"] for f in eng.mem.list_facts()] == ["The admin panel is deployed on Vercel"]
    svc.post(cid, "/forget admin panel Vercel")
    assert eng.mem.list_facts() == []


def test_rolling_summary_folds_old_messages(tmp_path):
    svc, eng = make(tmp_path, digest={"summary": "User and assistant compared caching options."})
    svc.cfg.chat.summary_trigger = 500
    cid = svc.new_chat()["id"]
    ids = [eng.mem.add_msg(cid, "user" if i % 2 == 0 else "assistant", "text", f"message {i} " + "x" * 120)["id"] for i in range(20)]
    svc._job_fold(cid)
    conv = eng.mem.get_conv(cid)
    assert "caching" in conv["summary"] and conv["summary_upto"] == ids[len(ids) - svc.cfg.chat.keep_recent - 1]


# ------------------------------------------------------------------ run hand-off
def test_run_result_is_posted_once(tmp_path):
    svc, eng = make(tmp_path)
    cid = svc.new_chat()["id"]
    rid = eng.submit("Make GST optional", {"conv_id": cid, "title": "Make GST optional"})
    eng.mem.update_run(rid, status="done", summary="GST is optional now.", files=["a.js", "b.js"])
    eng.mem.emit(rid, "status", status="done")
    eng.mem.emit(rid, "status", status="done")
    results = texts(eng, cid, "run_result")
    assert len(results) == 1 and "2 files" in results[0]["content"] and "GST is optional" in results[0]["content"]


def test_reconcile_after_restart(tmp_path):
    svc, eng = make(tmp_path)
    cid = svc.new_chat()["id"]
    brief = {"title": "T", "request": "Do the thing in the code base please", "keep_working": [], "assumptions": [], "open_questions": []}
    rid = svc.start_task(cid, brief)
    eng.mem.update_run(rid, status="interrupted")
    ChatService(eng, start_worker=False)          # a fresh service after a restart
    assert len(texts(eng, cid, "run_result")) == 1


def test_compose_result_variants():
    assert "not changed" in compose_result({"status": "failed", "error": "tests failed", "files": [], "tasks": []})
    assert "cancelled" in compose_result({"status": "cancelled"})