# forge/config.py

"""Configuration: model profiles, per-role settings, limits, chat behaviour, verification.

Resolution order (later wins):
  built-in defaults -> profile -> <repo>/.forge/config.toml -> <repo>/.forge/settings.json (edited from the UI)
"""
from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROLES = ("planner", "context", "coder", "reviewer", "summarizer", "chat", "router")

# One model for every role on purpose: Ollama keeps a single model resident, so mixing models
# makes it unload/reload weights between agents, which is very slow with 16 GB RAM / 4 GB VRAM.
# num_ctx must also be identical for every role, otherwise Ollama reloads the model when the context size changes.
PROFILES: dict[str, dict] = {
    "quality": {"model": "qwen3.5:9b", "num_ctx": 12288},
    "fast": {"model": "qwen3.5:4b", "num_ctx": 12288},
}

ROLE_DEFAULTS: dict[str, dict] = {
    "planner": {"temperature": 0.2, "num_predict": 3072, "think": False},
    "context": {"temperature": 0.1, "num_predict": 1536, "think": False},
    "coder": {"temperature": 0.1, "num_predict": 4096, "think": False},
    "reviewer": {"temperature": 0.1, "num_predict": 2048, "think": False},
    "summarizer": {"temperature": 0.2, "num_predict": 1024, "think": False},
    "chat": {"temperature": 0.4, "num_predict": 1800, "think": False},      # free conversation replies
    "router": {"temperature": 0.0, "num_predict": 600, "think": False},     # reads the wording: discuss, ask or start a task
}

SECRET_GLOBS = (".env", ".env.*", "*.pem", "*.key", "*.pfx", "*.p12", "id_rsa*", "*.keystore", "secrets.*", "credentials.*")

DEFAULT_IGNORE_DIRS = {
    ".git", "node_modules", ".venv", "venv", "env", "__pycache__", "dist", "build", ".next", ".nuxt",
    ".forge", "coverage", ".cache", ".turbo", ".vite", ".parcel-cache", ".pytest_cache", ".mypy_cache",
    ".idea", ".vscode", "out", "target", ".svelte-kit", ".expo", ".gradle", "vendor",
}
LINK_DIRS = ("node_modules", ".venv", "venv")  # linked (not copied) into the staging workspace


@dataclass
class RoleCfg:
    model: str = "qwen3.5:9b"
    num_ctx: int = 12288
    temperature: float = 0.1
    num_predict: int = 2048
    think: bool = False

    def prompt_budget(self, reserve: int = 600) -> int:
        """Tokens available for the prompt (context window minus generation room minus slack)."""
        return max(1024, self.num_ctx - self.num_predict - reserve)


@dataclass
class Limits:
    max_attempts: int = 3            # implementation attempts per subtask
    max_lookup_rounds: int = 3       # NEED_INFO resolution rounds per attempt before asking the user
    max_subtasks: int = 40
    max_files_per_task: int = 4
    planner_steps: int = 10          # investigation tool calls for the planner
    context_agent_steps: int = 6     # investigation tool calls when resolving a doubt
    max_replans: int = 2
    cmd_timeout: int = 900           # seconds per verification command
    llm_idle_timeout: int = 600      # seconds without a streamed token before giving up
    max_file_bytes: int = 1_000_000  # larger files are not indexed / editable


@dataclass
class ChatCfg:
    auto_start: bool = True          # start a task straight away when the user clearly says "do it" (otherwise a brief card asks first)
    research_steps: int = 2          # read-only code lookups a chat answer may make when it needs the real code
    memory_chars: int = 1600         # characters of recalled facts injected into each chat turn
    summary_trigger: int = 3000      # fold older messages into the running summary beyond this many characters
    keep_recent: int = 8             # newest messages that are never folded into the summary


@dataclass
class VerifyCfg:
    per_task: list[str] = field(default_factory=lambda: ["syntax", "imports", "lint", "typecheck", "test"])
    final: list[str] = field(default_factory=lambda: ["syntax", "imports", "lint", "typecheck", "test", "build"])
    # explicit overrides: {"frontend": {"lint": "npm run lint", "test": "npm test", "build": "npm run build"}}
    commands: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class Config:
    repo: Path
    host: str = "http://127.0.0.1:11434"
    profile: str = "quality"
    roles: dict[str, RoleCfg] = field(default_factory=dict)
    limits: Limits = field(default_factory=Limits)
    chat: ChatCfg = field(default_factory=ChatCfg)
    verify: VerifyCfg = field(default_factory=VerifyCfg)
    scope_policy: str = "related"    # strict | related | free
    require_approval: bool = True
    aliases: dict[str, str] = field(default_factory=dict)   # {"@/": "frontend/src/"}
    extra_ignore: list[str] = field(default_factory=list)
    stage_root: Path | None = None

    @property
    def forge_dir(self) -> Path:
        return self.repo / ".forge"

    def to_public(self) -> dict:
        d = asdict(self)
        d["repo"] = str(self.repo)
        d["stage_root"] = str(self.stage_root) if self.stage_root else None
        return d

    # ------------------------------------------------------------------ loading
    @staticmethod
    def load(repo: str | Path) -> "Config":
        repo = Path(repo).resolve()
        cfg = Config(repo=repo)
        raw: dict = {}
        toml_path = repo / ".forge" / "config.toml"
        if toml_path.is_file():
            raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        settings: dict = {}
        sp = repo / ".forge" / "settings.json"
        if sp.is_file():
            try:
                settings = json.loads(sp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                settings = {}
        cfg._apply(raw)
        cfg._apply(settings)
        cfg._build_roles(raw, settings)
        if cfg.stage_root is None:
            cfg.stage_root = Path.home() / ".forge" / "stages"
        return cfg

    def _apply(self, d: dict) -> None:
        if "ollama" in d:
            self.host = d["ollama"].get("host", self.host).rstrip("/")
            self.profile = d["ollama"].get("profile", self.profile)
        self.profile = d.get("profile", self.profile)
        for key in ("scope_policy", "require_approval", "extra_ignore"):
            if key in d:
                setattr(self, key, d[key])
        if "aliases" in d:
            self.aliases.update(d["aliases"])
        if "stage_root" in d:
            self.stage_root = Path(d["stage_root"]).expanduser()
        for k, v in d.get("limits", {}).items():
            if hasattr(self.limits, k):
                setattr(self.limits, k, type(getattr(self.limits, k))(v))
        for k, v in d.get("chat", {}).items():
            if hasattr(self.chat, k):
                setattr(self.chat, k, type(getattr(self.chat, k))(v))
        v = d.get("verify", {})
        for k in ("per_task", "final"):
            if k in v:
                setattr(self.verify, k, list(v[k]))
        if "commands" in v:
            self.verify.commands.update(v["commands"])

    def _build_roles(self, raw: dict, settings: dict) -> None:
        prof = PROFILES.get(self.profile, PROFILES["quality"])
        for role in ROLES:
            rc = RoleCfg(model=prof["model"], num_ctx=prof["num_ctx"], **ROLE_DEFAULTS[role])
            for src in (raw, settings):
                top = src.get("models", {})
                if "default" in top:
                    rc.model = top["default"]
                if role in top:
                    rc.model = top[role]
                if "num_ctx" in src.get("context", {}):
                    rc.num_ctx = int(src["context"]["num_ctx"])
                for k, v in src.get("role", {}).get(role, {}).items():
                    if hasattr(rc, k):
                        setattr(rc, k, type(getattr(rc, k))(v))
            self.roles[role] = rc

    def save_settings(self, patch: dict, replace: tuple[str, ...] = ()) -> None:
        """Persist UI-editable settings (profile, models, num_ctx, approval, chat, limits).

        Dict values are merged into the stored ones, except for keys listed in `replace`, which are overwritten
        (needed when switching from a custom model back to a profile, otherwise the old model name would stick).
        """
        self.forge_dir.mkdir(parents=True, exist_ok=True)
        p = self.forge_dir / "settings.json"
        cur = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
        for k, v in patch.items():
            if k not in replace and isinstance(v, dict) and isinstance(cur.get(k), dict):
                cur[k].update(v)
            else:
                cur[k] = v
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(cur, indent=2), encoding="utf-8")
        tmp.replace(p)