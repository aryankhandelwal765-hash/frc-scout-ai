#!/usr/bin/env bash
# Serve a Qwen model natively on Apple silicon with MLX, for LLM_BACKEND=mlx.
#
#   ./mlx_serve.sh                       # Qwen3.6-35B-A3B 4-bit on :8080 (19 GB resident)
#   MLX_MODEL=mlx-community/Qwen3-8B-4bit ./mlx_serve.sh   # small, ~5 GB
#
# 35B-A3B is a mixture-of-experts model with ~3B active parameters, so it answers in about
# the time qwen3:8b takes on Ollama (strategy read ~45 s vs ~42 s, captain ~43 s vs ~60 s,
# same prompts) with noticeably better use of the vision data.
#
# Then run the API with:  LLM_BACKEND=mlx MLX_HOST=http://localhost:8080 ./start.sh
# Own venv (.venv-mlx): mlx-lm pulls a different transformers than the app's .venv.
#   python3.13 -m venv .venv-mlx && .venv-mlx/bin/pip install mlx-lm
set -euo pipefail
cd "$(dirname "$0")"
exec .venv-mlx/bin/mlx_lm.server \
  --model "${MLX_MODEL:-mlx-community/Qwen3.6-35B-A3B-4bit}" \
  --port "${MLX_PORT:-8080}" \
  --max-tokens "${MLX_MAX_TOKENS:-4096}"
