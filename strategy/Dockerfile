# Multi-stage Dockerfile for FRC Scout AI.
#   Stage 1: build the React frontend to static assets.
#   Stage 2: install Python deps + serve API. FastAPI also serves the built
#            frontend as static files, so one container covers both.

# ── Stage 1: Frontend build ────────────────────────────────────────────────────
FROM node:24-alpine AS frontend-build
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci --silent

COPY frontend/ ./
RUN npm run build


# ── Stage 2: API + static frontend ─────────────────────────────────────────────
FROM python:3.13-slim AS runtime

# System deps for scientific Python (numpy/scipy/scikit-learn).
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache layer).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend + built frontend.
COPY scout/ ./scout/
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Cache DB and any per-user settings live in a volume so they survive rebuilds.
VOLUME ["/data"]
ENV CACHE_DB_PATH=/data/cache.db
ENV PYTHONUNBUFFERED=1

# Health check hits /health so orchestrators can detect a hung API.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "scout.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
