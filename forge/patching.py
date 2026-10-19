# forge/patching.py

"""Parse the coder's reply and apply SEARCH/REPLACE edits strictly (all-or-nothing per attempt)."""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Callable

from .workspace import norm_rel, TxError

EDIT_RE = re.compile(
    r"^<{5,9} SEARCH[ \t]+(?P<path>[^\n]+?)[ \t]*\n(?P<search>.*?)^={5,9}[ \t]*\n(?P<replace>.*?)^>{5,9} REPLACE[ \t]*$",
    re.S | re.M)
LAZY_RE = re.compile(
    r"(?i)(?:\.\.\.|…)\s*(?:rest|existing|remaining|other|unchanged|same|previous)\b"
    r"|\b(?:rest|remainder)\s+of\s+(?:the\s+)?(?:code|file|function|component|class|implementation|logic)\b"
    r"|\bexisting\s+code\s+(?:here|remains|unchanged)\b"
    r"|^\s*(?://|#|/\*|<!--)\s*(?:\.\.\.|…)\s*(?:\*/|-->)?\s*$")


@dataclass
class Lookup:
    kind: str                      # symbol | file | grep
    arg: str
    path: str | None = None
    start: int | None = None
    end: int | None = None


@dataclass
class CoderReply:
    status: str = "READY"
    summary: str = ""
    uses: list[tuple[str, str | None]] = field(default_factory=list)
    lookups: list[Lookup] = field(default_factory=list)
    asks: list[str] = field(default_factory=list)
    edits: list[tuple[str, str, str]] = field(default_factory=list)   # (path, search, replace)
    deletes: list[str] = field(default_factory=list)
    truncated: bool = False
    raw: str = ""


def _clean_path(p: str) -> str:
    p = p.strip().strip("`'\"*").strip()
    return p.replace("\\", "/")


def parse_reply(text: str) -> CoderReply:
    r = CoderReply(raw=text)
    r.truncated = "[[TRUNCATED]]" in text
    m = re.search(r"^\s*STATUS:\s*(READY|NEED_INFO)", text, re.M | re.I)
    r.status = m.group(1).upper() if m else "READY"
    m = re.search(r"^\s*SUMMARY:\s*(.+)$", text, re.M)
    r.summary = m.group(1).strip() if m else ""
    head = EDIT_RE.sub("", text)
    sec = None
    for line in head.splitlines():
        s = line.strip()
        up = s.upper().rstrip(":")
        if up in ("USES", "UNCERTAIN", "UNCERTAINTIES", "EDITS", "STATUS"):
            sec = "USES" if up == "USES" else ("UNC" if up.startswith("UNCERTAIN") else None)
            continue
        if s.upper().startswith("DELETE "):
            r.deletes.append(_clean_path(s[7:]))
            continue
        if not s.startswith(("-", "*")):
            continue
        item = s.lstrip("-* ").strip()
        if sec == "USES":
            if item.lower() in ("none", "n/a", ""):
                continue
            sym, _, path = item.partition("@")
            r.uses.append((sym.strip(" `"), _clean_path(path) if path.strip() else None))
        elif sec == "UNC":
            lm = re.match(r"(?i)LOOKUP\s+(symbol|file|grep)\s+(.+)$", item)
            if lm:
                kind, rest = lm.group(1).lower(), lm.group(2).strip()
                if kind == "symbol":
                    sm = re.match(r"(.+?)\s+in\s+(\S+)$", rest)
                    r.lookups.append(Lookup("symbol", sm.group(1).strip(" `"), _clean_path(sm.group(2))) if sm
                                     else Lookup("symbol", rest.strip(" `")))
                elif kind == "file":
                    fm = re.match(r"(\S+?)(?:\s+(\d+)\s*-\s*(\d+))?$", rest)
                    if fm:
                        r.lookups.append(Lookup("file", _clean_path(fm.group(1)), _clean_path(fm.group(1)),
                                                int(fm.group(2)) if fm.group(2) else None,
                                                int(fm.group(3)) if fm.group(3) else None))
                else:
                    r.lookups.append(Lookup("grep", rest.strip("`'\"")))
            else:
                q = re.sub(r"(?i)^ASK\s*:?\s*", "", item)
                if q:
                    r.asks.append(q)
    for em in EDIT_RE.finditer(text):
        search = em.group("search")[:-1] if em.group("search").endswith("\n") else em.group("search")
        replace = em.group("replace")[:-1] if em.group("replace").endswith("\n") else em.group("replace")
        r.edits.append((_clean_path(em.group("path")), search, replace))
    if r.lookups or r.asks:
        r.status = "NEED_INFO"
    return r


# ------------------------------------------------------------------------------------------ applying
@dataclass
class ApplyResult:
    files: dict[str, str | None] = field(default_factory=dict)   # path -> new text (None = delete)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _lines_with_offsets(text: str) -> tuple[list[str], list[int]]:
    lines = text.split("\n")
    offs, pos = [], 0
    for ln in lines:
        offs.append(pos)
        pos += len(ln) + 1
    return lines, offs


def _find_region(cur: str, search: str) -> tuple[int, int, str, str | None]:
    """Locate `search` inside `cur`. Returns (start, end, how, closest_snippet). start == -1 when not found,
    -2 when ambiguous."""
    n = cur.count(search)
    if n == 1:
        i = cur.index(search)
        return i, i + len(search), "exact", None
    if n > 1:
        return -2, -2, "ambiguous", None
    lines, offs = _lines_with_offsets(cur)
    s_lines = search.strip("\n").split("\n")
    k = len(s_lines)
    if k == 0 or not search.strip():
        return -1, -1, "empty", None

    def scan(norm: Callable[[str], str]):
        ns = [norm(x) for x in s_lines]
        hits = [i for i in range(len(lines) - k + 1) if all(norm(lines[i + j]) == ns[j] for j in range(k))]
        return hits

    for how, norm in (("trailing-whitespace", str.rstrip), ("indentation", str.strip)):
        hits = scan(norm)
        if len(hits) == 1:
            i = hits[0]
            end = offs[i + k - 1] + len(lines[i + k - 1])
            return offs[i], end, how, None
        if len(hits) > 1:
            return -2, -2, "ambiguous", None

    # fuzzy: best window by similarity; accept only if clearly the best and very similar
    target = "\n".join(x.strip() for x in s_lines)
    best, second, bi = 0.0, 0.0, -1
    for i in range(0, max(1, len(lines) - k + 1)):
        win = "\n".join(x.strip() for x in lines[i:i + k])
        sm = difflib.SequenceMatcher(None, win, target, autojunk=False)
        if sm.real_quick_ratio() < 0.8 or sm.quick_ratio() < 0.8:
            continue
        ratio = sm.ratio()
        if ratio > best:
            best, second, bi = ratio, best, i
        elif ratio > second:
            second = ratio
    if bi >= 0 and k >= 5 and best >= 0.97 and best - second >= 0.04:   # strict: a wrong match would overwrite the wrong code
        end = offs[bi + k - 1] + len(lines[bi + k - 1])
        return offs[bi], end, f"fuzzy({best:.2f})", None
    snippet = "\n".join(lines[bi:bi + k][:20]) if bi >= 0 and best >= 0.6 else None
    return -1, -1, "missing", snippet


def _reindent(replace: str, search_first: str, actual_first: str) -> str:
    s_ind = search_first[:len(search_first) - len(search_first.lstrip())]
    a_ind = actual_first[:len(actual_first) - len(actual_first.lstrip())]
    if s_ind == a_ind:
        return replace
    out = []
    for ln in replace.split("\n"):
        out.append(a_ind + ln[len(s_ind):] if ln.startswith(s_ind) else ln)
    return "\n".join(out)


def apply_edits(reply: CoderReply, read: Callable[[str], str | None],
                scope_ok: Callable[[str], str | None]) -> ApplyResult:
    """`read(path)` returns current text or None. `scope_ok(path)` returns an error string or None."""
    res = ApplyResult()
    current: dict[str, str | None] = {}

    def get(p: str):
        if p not in current:
            current[p] = read(p)
        return current[p]

    for path in reply.deletes:
        try:
            path = norm_rel(path)
        except TxError as e:
            res.errors.append(str(e))
            continue
        if (err := scope_ok(path)):
            res.errors.append(err)
        elif get(path) is None:
            res.errors.append(f"DELETE {path}: file does not exist")
        else:
            current[path] = None
            res.files[path] = None

    for idx, (path, search, replace) in enumerate(reply.edits, 1):
        tag = f"Edit {idx} ({path})"
        try:
            path = norm_rel(path)
        except TxError as e:
            res.errors.append(f"{tag}: {e}")
            continue
        if (err := scope_ok(path)):
            res.errors.append(f"{tag}: {err}")
            continue
        cur = get(path)
        lazy = [ln for ln in replace.split("\n") if LAZY_RE.search(ln) and ln not in (cur or "")]
        if lazy:
            res.errors.append(f"{tag}: the replacement contains an elision placeholder ({lazy[0].strip()[:60]!r}). "
                              "Write every line out in full; never abbreviate code.")
            continue
        if search.strip() == "":
            if cur not in (None, ""):
                res.errors.append(f"{tag}: the file already exists. Use SEARCH text that exists in the file instead of an empty SEARCH.")
                continue
            new = replace if replace.endswith("\n") or not replace else replace + "\n"
            current[path] = new
            res.files[path] = new
            res.notes.append(f"created {path} ({new.count(chr(10))} lines)")
            continue
        if cur is None:
            res.errors.append(f"{tag}: file does not exist. To create a file use an empty SEARCH section.")
            continue
        s, e, how, snippet = _find_region(cur, search)
        if s == -2:
            res.errors.append(f"{tag}: the SEARCH text matches more than one place. Include more surrounding lines so it is unique.")
            continue
        if s == -1:
            msg = f"{tag}: SEARCH text was not found. Copy it exactly from the current file."
            if snippet:
                msg += "\nClosest text in the file:\n" + snippet
            res.errors.append(msg)
            continue
        if how not in ("exact",):
            res.notes.append(f"{path}: SEARCH matched by {how}")
        if how in ("indentation",) or how.startswith("fuzzy"):
            first_actual = cur[s:e].split("\n")[0]
            replace = _reindent(replace, search.strip("\n").split("\n")[0], first_actual) if how == "indentation" else replace
        if replace == "" and (s == 0 or cur[s - 1] == "\n") and e < len(cur) and cur[e] == "\n":
            e += 1
        new = cur[:s] + replace + cur[e:]
        current[path] = new
        res.files[path] = new

    for path, new in list(res.files.items()):
        old = read(path)
        if new is not None and old and len(old.split("\n")) > 40 and len(new) < 0.5 * len(old):
            res.notes.append(f"{path}: file shrank from {len(old)} to {len(new)} characters (large deletion)")
    return res
