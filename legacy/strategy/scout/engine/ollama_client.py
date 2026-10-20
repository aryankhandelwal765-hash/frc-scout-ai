"""Ollama HTTP client. Streams ("thinking" | "content", text) events from a chat."""

import json
import os

import requests

# Base URL of the Ollama server. OLLAMA_HOST is the documented name; OLLAMA_BASE
# is accepted as a legacy alias so existing compose files keep working. Read
# from the environment rather than config so this module stays import-free.
OLLAMA_BASE = (
    os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_BASE") or "http://localhost:11434"
).rstrip("/")

# Default model for the strategy-captain synthesis layer. Kept in sync with
# config.SYNTHESIS_MODEL; duplicated here so this module has no config import.
DEFAULT_SYNTHESIS_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")


def list_models() -> list[str]:
    """Return available Ollama model names, empty list if Ollama isn't running."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        if r.ok:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def is_running() -> bool:
    try:
        requests.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
        return True
    except Exception:
        return False


def _stream_events(
    messages: list[dict],
    model: str,
    timeout: int = 600,
    think: bool | None = None,
    options: dict | None = None,
):
    """Yield (kind, text) events from an Ollama chat stream.

    kind is "thinking" or "content". Reasoning models (e.g. qwen3.6:35b-a3b)
    stream a long run of `thinking` chunks *before* the first `content` chunk;
    surfacing them lets callers show live progress during the reasoning phase
    instead of a frozen bar. Pass think=False to disable the reasoning phase on
    models that support it (much faster for short per-robot reads). `options`
    passes Ollama generation options (e.g. {"num_predict": 160} to cap length).
    """
    payload = {"model": model, "messages": messages, "stream": True}
    if think is not None:
        payload["think"] = think
    if options:
        payload["options"] = options
    with requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json=payload,
        stream=True,
        timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            msg = data.get("message", {})
            think = msg.get("thinking")
            if think:
                yield ("thinking", think)
            content = msg.get("content")
            if content:
                yield ("content", content)
            if data.get("done"):
                break


def stream_chat_verbose(
    model: str, system: str, user: str, think: bool | None = None, options: dict | None = None
):
    """Yield (kind, text) events ("thinking"/"content") for a single-turn chat."""
    yield from _stream_events(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model,
        think=think,
        options=options,
    )


def stream_chat_with_history_verbose(model: str, system: str, messages: list[dict]):
    """Yield (kind, text) events given a full message history list."""
    yield from _stream_events([{"role": "system", "content": system}] + messages, model)


def stream_chat(model: str, system: str, user: str):
    """Yield content text chunks (thinking omitted). Back-compat wrapper."""
    for kind, text in stream_chat_verbose(model, system, user):
        if kind == "content":
            yield text


def generate(
    model: str,
    system: str,
    user: str,
    timeout: int = 180,
    fmt: str | None = None,
    think: bool | None = None,
) -> str:
    """Non-streaming chat — return the full response as a single string.

    Convenience for callers (e.g. the synthesis layer, API endpoints) that want
    the complete strategy-captain read rather than a token stream. Pass
    fmt="json" for a structured (JSON) response, and think=False to skip the
    reasoning phase on thinking models (much faster for short structured calls).
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    if fmt is not None:
        payload["format"] = fmt
    if think is not None:
        payload["think"] = think
    resp = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


def stream_chat_with_history(model: str, system: str, messages: list[dict]):
    """Yield content text chunks given a full message history. Back-compat wrapper."""
    for kind, text in stream_chat_with_history_verbose(model, system, messages):
        if kind == "content":
            yield text
