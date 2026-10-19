@echo off
rem Usage: start.bat "C:\path\to\your\project"
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_KV_CACHE_TYPE=q8_0
set OLLAMA_MAX_LOADED_MODELS=1
python -m forge serve --repo %1
