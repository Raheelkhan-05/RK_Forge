# forge/tune.py

"""Find the largest context size that keeps the whole model on the GPU, then save it.

    python -m forge tune --repo .

Close running Forge tasks first: the model is unloaded and reloaded a few times.
"""
from __future__ import annotations

import json
import time
import urllib.request

from .config import Config

CANDIDATES = (12288, 10240, 8192, 6144)      # biggest first; below 6144 the coder has too little room to be useful
PROMPT = "Summarise in one sentence: " + "The quick brown fox jumps over the lazy dog. " * 70


def _call(cfg: Config, path: str, body: dict | None = None, timeout: int = 600) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(cfg.host + path, data=data, method="POST" if body is not None else "GET",
                                 headers={"Content-Type": "application/json"} if body is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _gpu_share(cfg: Config, model: str) -> int | None:
    names = (model, model + ":latest")
    for m in _call(cfg, "/api/ps").get("models", []):
        if m.get("name") in names or m.get("model") in names:
            size = m.get("size") or 0
            return round(100 * (m.get("size_vram") or 0) / size) if size else None
    return None


def measure(cfg: Config, model: str, ctx: int) -> dict:
    try:
        _call(cfg, "/api/generate", {"model": model, "keep_alive": 0}, 60)     # unload so the new context size takes effect
    except Exception:  # noqa: BLE001
        pass
    time.sleep(1.0)
    rc = cfg.roles["chat"]
    opts = {"num_ctx": ctx, "num_predict": 48, "temperature": 0}
    if rc.num_gpu >= 0:
        opts["num_gpu"] = rc.num_gpu
    if rc.num_thread > 0:
        opts["num_thread"] = rc.num_thread
    try:
        r = _call(cfg, "/api/generate", {"model": model, "prompt": PROMPT, "stream": False, "think": False,
                                         "keep_alive": "10m", "options": opts})
    except Exception as e:  # noqa: BLE001
        return {"ctx": ctx, "error": str(e)[:160]}
    ns = lambda k: (r.get(k) or 0) / 1e9  # noqa: E731
    pre, gen = ns("prompt_eval_duration"), ns("eval_duration")
    return {"ctx": ctx, "gpu_pct": _gpu_share(cfg, model),
            "prefill_tps": round((r.get("prompt_eval_count") or 0) / pre, 1) if pre else 0.0,
            "gen_tps": round((r.get("eval_count") or 0) / gen, 1) if gen else 0.0,
            "load_s": round(ns("load_duration"), 1)}


def tune(cfg: Config, model: str | None = None, candidates=CANDIDATES, apply: bool = True) -> int:
    model = model or cfg.roles["chat"].model
    print(f"Testing {model}. This reloads the model a few times; close running Forge tasks first.\n")
    try:
        installed = [m["name"] for m in _call(cfg, "/api/tags").get("models", [])]
    except Exception as e:  # noqa: BLE001
        print(f"Cannot reach Ollama at {cfg.host}: {e}")
        return 1
    if model not in installed and f"{model}:latest" not in installed:
        print(f"{model} is not installed. Run: ollama pull {model}")
        return 1
    results = []
    for ctx in candidates:
        r = measure(cfg, model, ctx)
        results.append(r)
        if "error" in r:
            print(f"  context {ctx:>6}: failed ({r['error']})")
            continue
        print(f"  context {ctx:>6}: {r['gpu_pct']}% on GPU | reads prompt {r['prefill_tps']} tok/s | writes {r['gen_tps']} tok/s | load {r['load_s']}s")
        if (r["gpu_pct"] or 0) >= 98:
            break                                  # biggest context first, so this is the largest one that fits
    ok = [r for r in results if "error" not in r]
    if not ok:
        print("\nNo test succeeded. Check that Ollama is running and the model loads (ollama run " + model + ").")
        return 1
    full = [r for r in ok if (r["gpu_pct"] or 0) >= 98]
    best = full[0] if full else max(ok, key=lambda r: r["gen_tps"])
    print(f"\nBest: context {best['ctx']} ({best['gpu_pct']}% on GPU, {best['gen_tps']} tok/s).")
    if not full:
        print("Even the smallest context leaves part of the model on the CPU. Use the 4B profile, close apps that use the GPU "
              "(browser hardware acceleration, games), and set the Ollama variables shown below, then run this again.")
    if best["ctx"] < 8192:
        print("A context below 8192 leaves the coder little room for code. Prefer a smaller model over a smaller context.")
    if apply:
        cfg.save_settings({"context": {"num_ctx": best["ctx"]}})
        print("Saved to .forge/settings.json. Restart Forge for it to take effect.")
    print("\nAlso worth setting (Windows PowerShell), then quit Ollama from the tray and start it again, then run tune again:\n"
          "  setx OLLAMA_FLASH_ATTENTION 1\n  setx OLLAMA_KV_CACHE_TYPE q8_0\n  setx OLLAMA_NUM_PARALLEL 1")
    return 0