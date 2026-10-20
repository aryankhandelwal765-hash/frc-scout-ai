#!/usr/bin/env bash
#
# start.sh — launch the whole FRC Scout AI stack with one command:
#   • Ollama       (local LLM server, :11434)
#   • FastAPI      (backend API,      :8000)
#   • React/Vite   (frontend,         :5173)
#
# Usage:   ./start.sh
# Stop:    press Ctrl+C (stops everything this script started)
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Invoke uvicorn through the interpreter. The `uvicorn` console script is
# not always generated in .venv/bin, and calling it directly fails silently.
PYTHON="$ROOT/.venv/bin/python"
LOGDIR="$ROOT/.logs"
mkdir -p "$LOGDIR"
touch "$LOGDIR/api.log" "$LOGDIR/frontend.log" "$LOGDIR/ollama.log"

API_PORT=8000
WEB_PORT=5173
OLLAMA_PORT=11434

PIDS=()
OLLAMA_STARTED=0

up() { curl -s "http://localhost:$1$2" >/dev/null 2>&1; }

cleanup() {
  echo ""
  echo "Shutting down FRC Scout AI…"
  pkill -f "scout.api.main:app"  2>/dev/null
  pkill -f "$ROOT/frontend"          2>/dev/null
  for pid in "${PIDS[@]:-}"; do [ -n "$pid" ] && kill "$pid" 2>/dev/null; done
  [ "$OLLAMA_STARTED" = "1" ] && pkill -f "ollama serve" 2>/dev/null
  echo "Done."
  exit 0
}
trap cleanup INT TERM

# ── Prerequisite checks ─────────────────────────────────────────────────────────
[ -x "$PYTHON" ] || {
  echo "✗ Python venv not found at .venv/"
  echo "  Set it up:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
}
"$PYTHON" -c "import uvicorn" 2>/dev/null || {
  echo "✗ uvicorn not installed in .venv/"
  echo "  Install it:  .venv/bin/pip install -r requirements.txt"
  exit 1
}
command -v ollama >/dev/null 2>&1 || { echo "✗ ollama not installed — see https://ollama.com"; exit 1; }
command -v npm    >/dev/null 2>&1 || { echo "✗ npm/node not installed"; exit 1; }
if [ ! -d "$ROOT/frontend/node_modules" ]; then
  echo "▶ Installing frontend dependencies (first run)…"
  (cd "$ROOT/frontend" && npm install)
fi

# ── Ollama ──────────────────────────────────────────────────────────────────────
if up $OLLAMA_PORT /api/tags; then
  echo "✓ Ollama already running"
else
  echo "▶ Starting Ollama…"
  nohup ollama serve >"$LOGDIR/ollama.log" 2>&1 &
  PIDS+=("$!"); OLLAMA_STARTED=1
  for _ in $(seq 1 30); do up $OLLAMA_PORT /api/tags && break; sleep 1; done
  up $OLLAMA_PORT /api/tags && echo "✓ Ollama up" || echo "✗ Ollama failed — see $LOGDIR/ollama.log"
fi

# ── FastAPI backend ─────────────────────────────────────────────────────────────
if up $API_PORT /health; then
  echo "✓ API already running on :$API_PORT"
else
  echo "▶ Starting API…"
  nohup "$PYTHON" -m uvicorn scout.api.main:app --port $API_PORT >"$LOGDIR/api.log" 2>&1 &
  PIDS+=("$!")
  for _ in $(seq 1 30); do up $API_PORT /health && break; sleep 1; done
  up $API_PORT /health && echo "✓ API up" || echo "✗ API failed — see $LOGDIR/api.log"
fi

# ── React frontend ──────────────────────────────────────────────────────────────
if up $WEB_PORT /; then
  echo "✓ Frontend already running on :$WEB_PORT"
else
  echo "▶ Starting frontend…"
  nohup npm --prefix "$ROOT/frontend" run dev >"$LOGDIR/frontend.log" 2>&1 &
  PIDS+=("$!")
  for _ in $(seq 1 40); do up $WEB_PORT / && break; sleep 1; done
  up $WEB_PORT / && echo "✓ Frontend up" || echo "✗ Frontend failed — see $LOGDIR/frontend.log"
fi

echo ""
echo "────────────────────────────────────────────"
echo "  FRC Scout AI is running"
echo "    App:   http://localhost:$WEB_PORT"
echo "    API:   http://localhost:$API_PORT/docs"
echo "    Logs:  $LOGDIR/"
echo ""
echo "  Press Ctrl+C to stop everything."
echo "────────────────────────────────────────────"

# Stream backend + frontend logs; Ctrl+C runs cleanup().
tail -f "$LOGDIR/api.log" "$LOGDIR/frontend.log"
