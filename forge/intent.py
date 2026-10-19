"""Cheap, deterministic reading of the user's wording.

The language model makes the real intent call (discuss / task / confirm ...). These signals double-check it so
that a task is never started from hedging wording, a pure question, or a half-formed idea. Everything here is
pure, has no side effects and is unit-tested.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_VERBS = ("add|make|change|update|fix|implement|create|build|remove|delete|rename|refactor|replace|move|convert|"
          "optimi[sz]e|improve|migrate|integrate|wire|hook|hide|enable|disable|validate|extract|split|merge|redesign|"
          "style|align|resolve|patch|install|configure|connect|generate|insert|adjust|tweak|modify|rewrite|clean ?up|"
          "set ?up|apply|use|switch|reduce|increase|bump|upgrade|debug|handle|support")

IMPERATIVE = re.compile(
    r"(?i)^\s*(?:(?:please|pls|hey|hi|ok(?:ay)?|now|also|and|then|so|just)[\s,]+)*"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|i\s+(?:want|need|would\s+like)\s+(?:you\s+)?to\s+|let'?s\s+|we\s+(?:need|have)\s+to\s+)?"
    rf"(?:{_VERBS})\b")

QUESTION = re.compile(r"(?i)\?\s*$|^\s*(?:what|why|how|when|where|which|who|is|are|does|do|did|can|could|should|would|will)\b")

HEDGE = re.compile(
    r"(?i)\b(?:maybe|perhaps|what\s+if|should\s+(?:we|i)|not\s+sure|thinking\s+(?:about|of)|wondering|i\s+wonder|idea|brainstorm|"
    r"discuss|opinion|your\s+thoughts|what\s+do\s+you\s+think|pros\s+and\s+cons|compare|explain|why|how\s+(?:does|do|would|should|can|come)|"
    r"could\s+we|is\s+it\s+(?:better|possible|ok)|any\s+(?:other\s+)?(?:way|option)|let'?s\s+(?:talk|think|discuss)|"
    r"before\s+(?:we|you|i)|hold\s+on|wait|not\s+yet|don'?t\s+(?:start|do|run)|do\s+not\s+(?:start|do|run)|just\s+(?:asking|curious)|"
    r"soch\s+raha|kya\s+lagta|samjha|kaise|kyu(?:n)?|ruko|abhi\s+nahi)\b")

GO = re.compile(
    r"(?i)\b(?:go\s+ahead|go\s+for\s+it|do\s+it|do\s+that|do\s+this|start\s+(?:it|now|the\s+task|this\s+task|working|building|implementing)|"
    r"proceed|run\s+(?:it|the\s+task|this)|implement\s+(?:it|this|that)|make\s+it\s+happen|let'?s\s+(?:do\s+it|go|start|build\s+it)|"
    r"execute\s+(?:it|this)|ship\s+it|kar\s*(?:do|de|dena)|karo|shuru\s+kar(?:o|\s*do)?|start\s+kar(?:o|\s*do)?|chalu\s+kar(?:o|\s*do)?|laga\s+do)\b")
NEG_GO = re.compile(r"(?i)\b(?:don'?t|do\s+not|not|never|wait|hold)\b[^.!?\n]{0,24}\b(?:go\s+ahead|do\s+it|start|proceed|run|implement)\b")

_AFFIRM = (r"(?:ok(?:ay)?|yes|yep|yeah|yup|yea|sure|alright|fine|perfect|great|cool|correct|right|exactly|confirmed?|approved?|agreed?|"
           r"sounds\s+good|looks\s+good|lgtm|haan|han|ha|theek\s+hai|thik\s+hai|ji|bilkul|good)")
_GOWORDS = (r"(?:go\s+ahead|do\s+it|start(?:\s+(?:it|now|the\s+task))?|proceed|run\s+it|implement\s+it|build\s+it|kar\s*do|karo|"
            r"let'?s\s+(?:go|do\s+it))")
CONFIRM = re.compile(
    rf"(?i)^\s*(?:{_AFFIRM}\b[\s,.!:;-]*)*(?:(?:please|pls)\b[\s,.!-]*)?(?:{_GOWORDS}\b[\s,.!]*)?(?:(?:please|pls|thanks|thank\s+you|now)\b[\s,.!]*)*$")

DECLINE = re.compile(
    r"(?i)^\s*(?:no|nope|nah|not\s+yet|not\s+now|wait|hold\s+on|hold\s+up|stop|cancel(?:\s+it)?|never\s*mind|abort|discard(?:\s+it)?|"
    r"don'?t(?:\s+(?:start|do\s+it))?|nahi|ruko|mat\s+karo|rehne\s+do)\s*[.!]*\s*$")

REMEMBER = re.compile(r"(?i)^\s*(?:please\s+)?(?:remember|note\s+(?:that|this|down)|keep\s+in\s+mind|save\s+(?:this|that)|don'?t\s+forget)\b|\bremember\s+(?:that|this|to|my|our)\b")
CORRECTION = re.compile(
    r"(?i)\b(?:forget\s+(?:that|this|about|what)|ignore\s+(?:that|what\s+i\s+said)|that'?s\s+(?:wrong|not\s+right|incorrect|outdated)|"
    r"no\s+longer|not\s+anymore|changed\s+my\s+mind|instead\s+of)\b")
FIX_REF = re.compile(
    r"(?i)^\s*(?:(?:please|pls|ok(?:ay)?|now|so|then|hey)[\s,]+)*"
    r"(?:(?:fix|correct|repair|redo|retry|re-?run|try)\s+(?:it|that|this|yourself|again|the\s+(?:error|failure|problem|issue|task))"
    r"|try\s+again|do\s+(?:it|that)\s+again|run\s+(?:it|that)\s+again|self[- ]?correct)\b")
DECIDE = re.compile(
    r"(?i)\b(?:you\s+decide|decide\s+(?:it\s+)?(?:yourself|for\s+(?:me|us))|your\s+(?:call|choice)|use\s+your\s+(?:best\s+)?judge?ment|"
    r"whatever\s+(?:you\s+)?(?:think|prefer|want)|i\s+(?:don'?t|do\s+not)\s+know|not\s+sure|no\s+idea|tum\s+decide|aap\s+decide|khud\s+decide|jo\s+theek\s+lage)\b")
CHANGE_REF = re.compile(r"(?i)\b(?:diff|changed?|changes|what\s+did\s+you|which\s+files?|files?\s+(?:did|are)|modified|edited|your\s+(?:change|edit|implementation|fix)|that\s+task|the\s+task|last\s+task)\b")
MEMORY_REF = re.compile(
    r"(?i)\b(?:remember|memory|memories|what\s+do\s+you\s+know|do\s+you\s+recall|what\s+did\s+we\s+(?:decide|discuss|say|talk)|earlier|"
    r"previously|last\s+time|you\s+said|you\s+mentioned|we\s+(?:decided|discussed|agreed)|forgot)\b")

_EXT = r"(?:jsx?|tsx?|mjs|cjs|py|sql|css|scss|html|json|md|toml|ya?ml|sh)"
_TOKEN = re.compile(r"[A-Za-z_$][\w$./-]*")

CODEISH = re.compile(
    r"(?i)\b(?:code|codebase|files?|functions?|components?|page|route|api|endpoint|table|column|database|db|bug|error|login|"
    r"button|form|tests?|repo|project|backend|frontend|admin|class|method|query|schema|module|hook|state|cache|auth|config|"
    r"how\s+(?:does|do|is)|where\s+(?:is|do|does))\b")



@dataclass
class Signals:
    words: int = 0
    question: bool = False
    imperative: bool = False
    hedge: bool = False
    go: bool = False
    confirm: bool = False
    decline: bool = False
    remember: bool = False
    correction: bool = False
    change_ref: bool = False
    fix_ref: bool = False
    decide: bool = False
    memory_ref: bool = False

    def describe(self, pending: bool = False) -> str:
        yn = lambda b: "yes" if b else "no"  # noqa: E731
        return (f"question_form={yn(self.question)}; imperative_form={yn(self.imperative)}; hedging_words={yn(self.hedge)}; "
                f"explicit_go_words={yn(self.go)}; pending_brief={yn(pending)}; words={self.words}")


def analyze(text: str) -> Signals:
    t = (text or "").strip()
    s = Signals(words=len(re.findall(r"[\w']+", t)))
    s.question = bool(QUESTION.search(t))
    s.imperative = bool(IMPERATIVE.search(t))
    s.hedge = bool(HEDGE.search(t))
    s.go = bool(GO.search(t)) and not NEG_GO.search(t)
    s.decline = bool(DECLINE.match(t))
    pure_confirm = bool(t) and re.search(r"\w", t) is not None and bool(CONFIRM.match(t))
    short_go = s.go and s.words <= 10 and not s.question and not s.hedge
    s.confirm = not s.decline and (pure_confirm or short_go)
    s.remember = bool(REMEMBER.search(t))
    s.correction = bool(CORRECTION.search(t))
    s.change_ref = bool(CHANGE_REF.search(t))
    s.memory_ref = bool(MEMORY_REF.search(t))
    s.fix_ref = bool(FIX_REF.search(t)) and s.words <= 25
    s.decide = bool(DECIDE.search(t)) and s.words <= 20
    return s


def code_names(text: str, limit: int = 6) -> list[str]:
    """File names, paths and identifiers (camelCase, PascalCase, snake_case) mentioned in the text.

    Plain words never qualify, so ordinary English does not trigger code lookups.
    """
    out: list[str] = []
    for raw in _TOKEN.findall(text or ""):
        tok = raw.strip(".,;:/-")
        if len(tok) < 4 or tok.lower().startswith("http"):
            continue
        is_file = re.search(rf"\.{_EXT}$", tok, re.I) is not None
        is_path = "/" in tok and re.search(r"[A-Za-z]", tok) is not None
        is_ident = (re.search(r"[a-z][A-Z]", tok) is not None or re.match(r"[A-Z][a-z]+[A-Z]", tok) is not None
                    or ("_" in tok and re.search(r"[a-z]", tok) is not None))
        if (is_file or is_path or is_ident) and tok not in out:
            out.append(tok)
        if len(out) >= limit:
            break
    return out


def looks_like_code_talk(text: str) -> bool:
    """True when the message plausibly needs the router or the real code. Plain chatter returns False."""
    return bool(CODEISH.search(text or "")) or bool(code_names(text))