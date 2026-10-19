# tests/test_patching.py

from forge.patching import apply_edits, parse_reply

SRC = "function a() {\n  return 1\n}\n\nfunction b() {\n  const x = 2\n  return x\n}\n"


def run(reply_text, files=None, scope=lambda p: None):
    files = files if files is not None else {"f.js": SRC}
    return apply_edits(parse_reply(reply_text), files.get, scope), files


def block(path, s, r):
    return f"<<<<<<< SEARCH {path}\n{s}\n=======\n{r}\n>>>>>>> REPLACE\n"


def test_exact_edit():
    res, _ = run("STATUS: READY\nEDITS:\n" + block("f.js", "  return 1", "  return 10"))
    assert res.ok and "return 10" in res.files["f.js"]


def test_ambiguous_rejected():
    res, _ = run(block("f.js", "  return", "  return 0"))
    assert not res.ok and "more than one" in res.errors[0]


def test_indentation_tolerant_and_reindented():
    res, _ = run(block("f.js", "const x = 2\nreturn x", "const x = 3\nreturn x"))
    assert res.ok and "  const x = 3\n  return x" in res.files["f.js"]


def test_missing_search_shows_closest_snippet():
    res, _ = run(block("f.js", "function b() {\n  const y = 22\n  return y\n}", "x"))
    assert not res.ok and "not found" in res.errors[0]


def test_lazy_placeholder_rejected():
    res, _ = run(block("f.js", "  return 1", "  // ... rest of code unchanged ..."))
    assert not res.ok and "placeholder" in res.errors[0]


def test_create_and_delete_and_scope():
    res, files = run(block("new.js", "", "export const z = 1") + "DELETE f.js\n", {"f.js": SRC})
    assert res.ok and res.files["new.js"] == "export const z = 1\n" and res.files["f.js"] is None
    res, _ = run(block("other.js", "", "x"), scope=lambda p: f"{p} out of scope")
    assert not res.ok


def test_create_over_existing_refused():
    res, _ = run(block("f.js", "", "x"))
    assert not res.ok


def test_parse_need_info():
    r = parse_reply("STATUS: NEED_INFO\nUNCERTAIN:\n- LOOKUP symbol saveGst in src/api.js\n- LOOKUP file src/a.js 10-40\n- LOOKUP grep rpc(\n- ASK Should GST be mandatory?\n")
    assert r.status == "NEED_INFO" and len(r.lookups) == 3 and r.asks == ["Should GST be mandatory?"]
    assert r.lookups[0].path == "src/api.js" and (r.lookups[1].start, r.lookups[1].end) == (10, 40)


def test_removal_leaves_no_blank_line():
    res, _ = run(block("f.js", "function a() {\n  return 1\n}\n", ""))
    assert res.ok and res.files["f.js"].startswith("\nfunction b") is False
