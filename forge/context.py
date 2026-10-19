# forge/context.py

"""Context builder: gives the coder exactly what it needs, packed into the model's token budget.

Strategy for large repos (50k+ lines): never dump files blindly. Files to modify are shown in full when they fit,
otherwise as a *window* (header + the most relevant symbols, with explicit elision markers and an outline of
everything that was left out). Around them: interfaces of what they import, real call sites of what they export,
related tests, full-text search hits, remembered decisions and the user's earlier answers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .indexer import Index, Sym, query_terms, split_ident
from .llm import est_tokens
from .memory import Memory
from .patching import Lookup
from .workspace import Workspace

COMMENT = {"python": "#", "yaml": "#", "sh": "#", "toml": "#", "sql": "--", "js": "//", "ts": "//", "css": "/*", "html": "<!--"}
BUILTINS = set("console require module exports process window document Promise Object Array JSON Math Date Error Map Set String Number Boolean "
               "useState useEffect useRef useMemo useCallback useContext useReducer print len range str int float dict list set tuple open "
               "self cls super fetch setTimeout setInterval clearTimeout parseInt parseFloat undefined null true false".split())


@dataclass
class Bundle:
    text: str = ""
    tokens: int = 0
    items: list[dict] = field(default_factory=list)
    shown: set[str] = field(default_factory=set)       # paths shown with real content
    omitted: list[str] = field(default_factory=list)


class ContextBuilder:
    def __init__(self, index: Index, ws: Workspace, memory: Memory):
        self.ix, self.ws, self.mem = index, ws, memory

    # ------------------------------------------------------------------ rendering helpers
    def _lang(self, path: str) -> str:
        info = self.ix.file_info(path)
        return (info[1] if info else None) or "text"

    def _marker(self, path: str, a: int, b: int, names: list[str]) -> str:
        c = COMMENT.get(self._lang(path), "#")
        note = f"... lines {a}-{b} omitted" + (f" ({', '.join(names[:5])})" if names else "") + " ..."
        return f"{c} {note}" + (" */" if c == "/*" else " -->" if c == "<!--" else "")

    def outline(self, path: str, limit: int = 60) -> str:
        syms = [s for s in self.ix.symbols_in(path) if s.kind != "const" or s.parent is None]
        lines = [f"L{s.line}-{s.end_line} {s.kind} {('%s.' % s.parent) if s.parent else ''}{s.name}" for s in syms[:limit]]
        if len(syms) > limit:
            lines.append(f"... {len(syms) - limit} more symbols")
        return "\n".join(lines)

    def render_file(self, path: str, budget: int, terms: list[str], focus: list[str]) -> tuple[str, str, int]:
        text = self.ws.read(path)
        if text is None:
            return "", "missing", 0
        n_lines = text.count("\n") + 1
        if est_tokens(text) <= budget:
            return f'<file path="{path}" lines="1-{n_lines}">\n{text}\n</file>', "full", est_tokens(text)
        lines = text.split("\n")
        syms = self.ix.symbols_in(path)
        focus_l = {f.lower() for f in focus}
        tset = set(terms)
        ranges: list[tuple[int, int]] = []
        used = 0
        first = min([s.line for s in syms if not s.parent] or [n_lines + 1])
        hdr = min(first - 1, 45, n_lines)
        if hdr > 0:
            ranges.append((1, hdr))
            used += est_tokens("\n".join(lines[:hdr]))
        scored = []
        for s in syms:
            toks = set(split_ident(s.name)) | {s.name.lower()}
            body = "\n".join(lines[s.line - 1:s.end_line]).lower()
            score = 3 * len(toks & tset) + (8 if (s.name.lower() in focus_l or (s.parent or "").lower() in focus_l) else 0) \
                + 0.3 * sum(1 for t in terms if t in body)
            if score > 0:
                scored.append((score, s))
        scored.sort(key=lambda t: (-t[0], t[1].end_line - t[1].line))
        if not scored:   # no symbol evidence: pick the lines that mention the terms
            for i, ln in enumerate(lines, 1):
                low = ln.lower()
                if any(t in low for t in terms[:12]):
                    scored.append((1.0, Sym(path, "", "line", max(1, i - 15), min(n_lines, i + 30))))
                if len(scored) > 6:
                    break
        chosen: list[Sym] = []
        for score, s in scored:
            a, b = s.line, s.end_line
            if b - a > 160:   # huge symbol: keep the neighbourhood of the first matching line
                hit = next((i for i in range(a, b + 1) if any(t in lines[i - 1].lower() for t in terms[:12])), a)
                a, b = max(a, hit - 40), min(b, hit + 80)
            cost = est_tokens("\n".join(lines[a - 1:b])) + 12
            if used + cost > budget:
                continue
            if any(c.line <= s.line and c.end_line >= s.end_line and c is not s for c in chosen):
                continue   # already inside a chosen parent
            ranges.append((a, b))
            chosen.append(s)
            used += cost
        ranges.sort()
        merged: list[list[int]] = []
        for a, b in ranges:
            if merged and a <= merged[-1][1] + 3:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        body, prev = [], 0
        for a, b in merged:
            if a > prev + 1:
                body.append(self._marker(path, prev + 1, a - 1, [s.name for s in syms if prev < s.line < a and not s.parent]))
            body.append("\n".join(lines[a - 1:b]))
            prev = b
        if prev < n_lines:
            body.append(self._marker(path, prev + 1, n_lines, [s.name for s in syms if s.line > prev and not s.parent]))
        shown = sum(b - a + 1 for a, b in merged)
        out = f'<file path="{path}" lines="partial: {shown} of {n_lines} shown">\n' + "\n".join(body) + "\n</file>"
        outline = self.outline(path)
        if outline:
            out += f'\n<outline path="{path}">\n{outline}\n</outline>'
        return out, "window", est_tokens(out)

    def interface(self, path: str, names: list[str] | None = None, limit: int = 14) -> str:
        syms = [s for s in self.ix.symbols_in(path) if (s.exported or (names and s.name in names)) and not s.parent and s.kind != "route"]
        if names:
            syms.sort(key=lambda s: s.name not in names)
        lines = [s.sig for s in syms[:limit] if s.sig] or [f"{s.kind} {s.name}" for s in syms[:limit]]
        if not lines:
            return ""
        return f'<interface path="{path}">\n' + "\n".join(lines) + "\n</interface>"

    def usages(self, importer: str, target: str, limit: int = 4) -> str:
        names = [n for n in self.ix.exports_of(target)[0] if n != "default" and len(n) > 2][:12]
        text = self.ws.read(importer)
        if not text:
            return ""
        stem = target.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        out = []
        for i, ln in enumerate(text.split("\n"), 1):
            if (names and any(re.search(rf"\b{re.escape(n)}\b", ln) for n in names)) or stem in ln:
                out.append(f"L{i}: {ln.strip()[:150]}")
                if len(out) >= limit:
                    break
        return f'<usage path="{importer}" of="{target}">\n' + "\n".join(out) + "\n</usage>" if out else ""

    def related(self, paths: list[str]) -> set[str]:
        out: set[str] = set()
        for p in paths:
            out.update(r for _, _, r, _ in self.ix.imports_of(p) if r)
            out.update(self.ix.importers_of(p))
            out.update(self.ix.tests_for(p))
        return out

    # ------------------------------------------------------------------ main entry
    def build(self, task: dict, request: str, answers: list[str], extras: list[str], budget: int) -> Bundle:
        b = Bundle()
        mods = [p for p in task.get("files_to_modify", []) if self.ws.exists(p)]
        reads = [p for p in task.get("files_to_read", []) if self.ws.exists(p) and p not in mods]
        focus = list(task.get("focus_symbols", []))
        spec = " ".join([task.get("title", ""), task.get("description", ""), " ".join(task.get("acceptance_criteria", [])), " ".join(focus)])
        terms = query_terms(spec + " " + request[:400], 30)
        parts: list[str] = []
        used = 0

        def add(text: str, path: str | None, mode: str, reason: str, tokens: int | None = None, shown: bool = True) -> bool:
            nonlocal used
            t = tokens if tokens is not None else est_tokens(text)
            if not text or used + t > budget:
                if path:
                    b.omitted.append(f"{path} ({reason})")
                return False
            parts.append(text)
            used += t
            b.items.append({"path": path, "mode": mode, "tokens": t, "reason": reason})
            if path and shown:
                b.shown.add(path)
            return True

        # 1. user answers + resolved lookups (highest value, small)
        if answers:
            add("<answers_from_user>\n" + "\n".join(f"- {a}" for a in answers) + "\n</answers_from_user>", None, "answers", "user decisions")
        for e in extras:
            add(e, None, "lookup", "resolved doubt")
        mem = self.mem.recall(spec, mods + reads, self._hash, max_chars=2200)
        if mem:
            add("<project_memory>\n" + "\n".join(f"- [{f['kind']}] {f['text']}" for f in mem) + "\n</project_memory>", None, "memory", f"{len(mem)} facts")

        # 2. files to modify (up to 62% of the remaining budget)
        remaining = budget - used
        if mods:
            per = int(remaining * 0.62 / len(mods))
            full_sizes = {p: est_tokens(self.ws.read(p) or "") for p in mods}
            if sum(full_sizes.values()) <= remaining * 0.62:
                per = max(full_sizes.values()) + 50
            for p in sorted(mods, key=lambda x: full_sizes[x]):
                text, mode, tok = self.render_file(p, per, terms, focus)
                add(text, p, mode, "file to modify", tok)

        # 3. planner-specified reads, then interfaces of imports, call sites, tests, search hits, siblings
        for p in reads:
            text, mode, tok = self.render_file(p, max(800, int((budget - used) * 0.25)), terms, focus)
            add(text, p, mode, "needed reference", tok)
        for p in mods:
            wanted = set(re.findall(r"[A-Za-z_$][\w$]*", self.ws.read(p) or ""))
            for spec_, names, res, _k in self.ix.imports_of(p):
                if res and res not in b.shown and res not in mods:
                    iface = self.interface(res, [n for n in names if n in wanted])
                    add(iface, res, "interface", f"imported by {p}", shown=False)
        for p in mods:
            for imp in self.ix.importers_of(p)[:8]:
                if imp not in mods:
                    add(self.usages(imp, p), imp, "usage", f"calls into {p}", shown=False)
        for p in mods:
            for t in self.ix.tests_for(p)[:2]:
                if t not in b.shown:
                    text, mode, tok = self.render_file(t, min(900, max(300, int((budget - used) * 0.2))), terms, focus)
                    add(text, t, mode, f"test for {p}", tok)
        hits = self.ix.search(spec, limit=10)
        for h in hits:
            if h.path in b.shown or h.path in mods or used > budget * 0.92:
                continue
            seg = self.ws.read(h.path)
            if not seg:
                continue
            snippet = "\n".join(seg.split("\n")[h.start - 1:h.end])
            add(f'<snippet path="{h.path}" lines="{h.start}-{h.end}">\n{snippet}\n</snippet>', h.path, "snippet", "search match", shown=False)
        for p in task.get("files_to_create", []):
            d = p.rsplit("/", 1)[0] if "/" in p else ""
            ext = p.rsplit(".", 1)[-1]
            sib = [x for x in self.ix.all_paths() if x.startswith(d + "/" if d else "") and x.endswith("." + ext) and x.count("/") == p.count("/")]
            if sib and used < budget * 0.85:
                text, mode, tok = self.render_file(sib[0], min(700, budget - used), terms, focus)
                add(text.replace("<file ", "<style_example ").replace("</file>", "</style_example>"), sib[0], "example", f"style reference for new file {p}", tok)

        b.text = "\n\n".join(parts)
        b.tokens = used
        return b

    def _hash(self, path: str) -> str | None:
        r = self.ix.db.execute("SELECT hash FROM files WHERE path=?", (path,)).fetchone()
        return r[0] if r else None

    # ------------------------------------------------------------------ doubt resolution (deterministic part)
    def resolve_lookup(self, lk: Lookup) -> tuple[str, bool]:
        if lk.kind == "symbol":
            syms = [s for s in self.ix.find_symbol(lk.arg, lk.path) if s.name.lower() == lk.arg.lower()] or self.ix.find_symbol(lk.arg, lk.path)
            if not syms:
                refs = self.ix.refs_to(lk.arg)
                if refs:
                    return f'<lookup symbol="{lk.arg}">\nNo definition found, but it is referenced in: ' + ", ".join(sorted({p for p, _ in refs})[:6]) + "\n</lookup>", True
                return f'<lookup symbol="{lk.arg}">\nNOT FOUND anywhere in the repository index.\n</lookup>', False
            out = []
            for s in syms[:4]:
                text = self.ws.read(s.path) or ""
                body = "\n".join(text.split("\n")[s.line - 1:min(s.end_line, s.line + 119)])
                out.append(f'<definition symbol="{s.name}" kind="{s.kind}" path="{s.path}" lines="{s.line}-{s.end_line}">\n{body}\n</definition>')
            extra = self.ix.refs_to(syms[0].name)
            if extra:
                out.append("referenced from: " + ", ".join(sorted({p for p, _ in extra})[:6]))
            return "\n".join(out), True
        if lk.kind == "file":
            text = self.ws.read(lk.arg)
            if text is None:
                near = [p for p in self.ix.all_paths() if p.endswith("/" + lk.arg.rsplit("/", 1)[-1])][:5]
                return f'<lookup file="{lk.arg}">\nNOT FOUND. Similar paths: {", ".join(near) or "none"}\n</lookup>', False
            lines = text.split("\n")
            a, b_ = (lk.start or 1), (lk.end or min(len(lines), (lk.start or 1) + 199))
            seg = "\n".join(lines[a - 1:b_][:250])
            return f'<file path="{lk.arg}" lines="{a}-{min(b_, len(lines))} of {len(lines)}">\n{seg}\n</file>', True
        hits = self.ix.grep(lk.arg, 15)
        if not hits:
            return f'<lookup grep="{lk.arg}">\nNo matches.\n</lookup>', False
        return f'<grep text="{lk.arg}">\n' + "\n".join(f"{p}:{i}: {ln}" for p, i, ln in hits) + "\n</grep>", True

    def verify_uses(self, uses: list[tuple[str, str | None]], new_text: str) -> list[Lookup]:
        """Symbols the coder says it relies on must exist. Returns lookups for the ones that cannot be verified."""
        bad: list[Lookup] = []
        for sym, path in uses:
            base = re.split(r"[.(\s]", sym)[-1] if sym.startswith("this.") else re.split(r"[(\s]", sym)[0]
            base = base.split(".")[-1] or sym
            if base in BUILTINS or len(base) <= 2:
                continue
            if re.search(rf"\b(?:function|const|let|var|def|class|interface|type)\s+{re.escape(base)}\b", new_text):
                continue
            if path:
                if not self.ws.exists(path):
                    bad.append(Lookup("file", path, path))
                    continue
                txt = self.ws.read(path) or ""
                if not re.search(rf"\b{re.escape(base)}\b", txt):
                    bad.append(Lookup("symbol", base, path))
                continue
            if not self.ix.find_symbol(base, limit=1) and not self.ix.refs_to(base):
                bad.append(Lookup("symbol", base))
        return bad
