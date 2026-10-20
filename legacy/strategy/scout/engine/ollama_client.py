"""LLM chat client. Streams ("thinking" | "content", text) events from a chat.

Two backends behind one interface, chosen by LLM_BACKEND:
  ollama (default)  Ollama's /api/chat. Used in Docker/prod and anywhere not on a Mac.
  mlx               an `mlx_lm.server` (OpenAI-style /v1/chat/completions), Apple-silicon native.
                    Start it from the separate .venv-mlx (see mlx_serve.sh); MLX_HOST points at it.
Callers pass the same model names either way; on MLX the server's loaded model is used.
"""

import json
import os

import requests

# Base URL of the Ollama server. OLLAMA_HOST is the documented name; OLLAMA_BASE
# is accepted as a legacy alias so existing compose files keep working. Read
# from the environment rather than config so this module stays import-free.
OLLAMA_BASE = (
    os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_BASE") or "http://localhost:11434"
).rstrip("/")

BACKEND = os.environ.get("LLM_BACKEND", "ollama").lower()
MLX_BASE = os.environ.get("MLX_HOST", "http://localhost:8080").rstrip("/")

# Default model for the strategy-captain synthesis layer. Kept in sync with
# config.SYNTHESIS_MODEL; duplicated here so this module has no config import.
DEFAULT_SYNTHESIS_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")


def list_models() -> list[str]:
    """Return available model names, empty list if the backend isn't running."""
    if BACKEND == "mlx":
        try:
            r = requests.get(f"{MLX_BASE}/v1/models", timeout=3)
            if r.ok:
                return [m["id"] for m in r.json().get("data", [])]
        except Exception:
            pass
        return []
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        if r.ok:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def is_running() -> bool:
    try:
        requests.get(
            f"{MLX_BASE}/v1/models" if BACKEND == "mlx" else f"{OLLAMA_BASE}/api/tags", timeout=2
        )
        return True
    except Exception:
        return False


def _mlx_payload(messages, think, options, stream):
    payload = {"model": "default_model", "messages": messages, "stream": stream}
    if think is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": bool(think)}
    opts = options or {}
    if opts.get("num_predict"):
        payload["max_tokens"] = opts["num_predict"]
    for key in ("temperature", "top_p"):
        if key in opts:
            payload[key] = opts[key]
    return payload


def _mlx_stream_events(messages, timeout, think, options):
    """SSE from mlx_lm.server; reasoning arrives in delta.reasoning, answer in delta.content."""
    with requests.post(
        f"{MLX_BASE}/v1/chat/completions",
        json=_mlx_payload(messages, think, options, True),
        stream=True,
        timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            text = line.decode() if isinstance(line, bytes) else line
            if not text.startswith("data:"):
                continue
            body = text[5:].strip()
            if body == "[DONE]":
                break
            choices = json.loads(body).get("choices") or [{}]
            delta = choices[0].get("delta", {})
            if delta.get("reasoning"):
                yield ("thinking", delta["reasoning"])
            if delta.get("content"):
                yield ("content", delta["content"])


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
    if BACKEND == "mlx":
        yield from _mlx_stream_events(messages, timeout, think, options)
        return
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
    if BACKEND == "mlx":
        return _mlx_generate(system, user, timeout, fmt, think)
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


JSON_HINT = "\n\nReply with ONLY one valid JSON object, no prose and no code fences."


def _mlx_generate(system, user, timeout, fmt, think):
    """MLX cannot force a JSON schema the way Ollama's format="json" does, so the reply
    shape is requested in the prompt, validated, and retried once."""
    sys_msg = system + (JSON_HINT if fmt == "json" else "")
    for _attempt in range(2 if fmt == "json" else 1):
        resp = requests.post(
            f"{MLX_BASE}/v1/chat/completions",
            json=_mlx_payload(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
                think,
                None,
                False,
            ),
            timeout=timeout,
        )
        resp.raise_for_status()
        text = (resp.json()["choices"][0]["message"].get("content") or "").strip()
        if fmt != "json":
            return text
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        try:
            json.loads(text)
            return text
        except ValueError:
            continue
    return text


def stream_chat_with_history(model: str, system: str, messages: list[dict]):
    """Yield content text chunks given a full message history. Back-compat wrapper."""
    for kind, text in stream_chat_with_history_verbose(model, system, messages):
        if kind == "content":
            yield text
