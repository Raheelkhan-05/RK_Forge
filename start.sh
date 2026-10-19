#!/usr/bin/env sh
# Usage: ./start.sh /path/to/your/project
export OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_MAX_LOADED_MODELS=1
python3 -m forge serve --repo "${1:-.}"
