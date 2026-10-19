# tests/test_scale.py

import random
import time
from pathlib import Path

from forge.config import Config
from forge.context import ContextBuilder
from forge.indexer import Index
from forge.llm import est_tokens
from forge.memory import Memory
from forge.workspace import Workspace


def synth(root: Path, files_per_proj=120, funcs=14):
    random.seed(1)
    for proj in ("frontend", "backend", "admin"):
        (root / proj / "src").mkdir(parents=True)
        (root / proj / "package.json").write_text('{"name":"%s","dependencies":{"react":"18","express":"4"}}' % proj)
        for i in range(files_per_proj):
            body = [f"import {{ helper{(i + 1) % files_per_proj}_{j} }} from './mod{(i + 1) % files_per_proj}'" for j in range(1)]
            for j in range(funcs):
                body.append(f"export function helper{i}_{j}(a, b) {{\n  const total = a + b + {j}\n  if (total > 10) {{\n    return total * 2\n  }}\n  // sellerGst handling {i}\n  return total\n}}\n")
            (root / proj / "src" / f"mod{i}.js").write_text("\n".join(body))


def test_large_repo_index_and_context(tmp_path):
    repo = tmp_path / "big"
    synth(repo)
    total = sum(p.read_text().count("\n") for p in repo.rglob("*.js"))
    assert total > 40000
    cfg = Config.load(repo)
    cfg.stage_root = tmp_path / "st"
    ix = Index(repo / ".forge" / "index.db", repo, cfg)
    t = time.time(); ix.refresh(); first = time.time() - t
    t = time.time(); r = ix.refresh(); again = time.time() - t
    assert r["changed"] == 0 and again < first / 2
    assert first < 60
    ws = Workspace(cfg, "scale"); ws.create()
    six = Index(ws.run_dir / "index.db", ws.tree, cfg); six.refresh()
    b = ContextBuilder(six, ws, Memory(cfg)).build(
        {"title": "Handle sellerGst", "description": "update helper5_3 to handle sellerGst", "files_to_modify": ["frontend/src/mod5.js", "backend/src/mod9.js"],
         "files_to_read": [], "focus_symbols": ["helper5_3"], "acceptance_criteria": ["gst handled"]}, "update sellerGst handling", [], [], 3000)
    assert b.tokens <= 3000 and "helper5_3" in b.text and "mod5.js" in b.text
    ws.cleanup()
    print(f"\n{total} lines: first index {first:.1f}s, incremental {again:.2f}s, context {b.tokens} tokens")
