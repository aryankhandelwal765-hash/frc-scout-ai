"""ollama_client — the NDJSON event stream, with Ollama mocked out."""

import json

import pytest

from scout.engine import ollama_client as oc


class FakeResponse:
    """Stands in for a streaming requests.Response."""

    def __init__(self, lines, status_ok=True):
        self._lines = lines
        self._ok = status_ok

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if not self._ok:
            raise RuntimeError("500 Server Error")

    def iter_lines(self):
        for line in self._lines:
            yield line if isinstance(line, bytes) else line.encode()


def chunks(*msgs):
    return [json.dumps({"message": m}) for m in msgs]


@pytest.fixture
def post(monkeypatch):
    """Capture the payload and serve a scripted stream."""
    captured = {}

    def _install(lines, ok=True):
        def fake_post(url, json=None, stream=None, timeout=None):
            captured["url"] = url
            captured["payload"] = json
            return FakeResponse(lines, status_ok=ok)

        monkeypatch.setattr(oc.requests, "post", fake_post)
        return captured

    return _install


def test_separates_thinking_from_content(post):
    post(chunks({"thinking": "hmm"}, {"thinking": " more"}, {"content": "Answer"}))
    events = list(oc.stream_chat_verbose("qwen3:8b", "sys", "user"))
    assert events == [("thinking", "hmm"), ("thinking", " more"), ("content", "Answer")]


def test_blank_lines_are_skipped(post):
    post(["", *chunks({"content": "a"}), "", *chunks({"content": "b"})])
    assert [t for _k, t in oc.stream_chat_verbose("m", "s", "u")] == ["a", "b"]


def test_empty_content_chunks_are_not_yielded(post):
    """Ollama sends a final chunk with empty content; it is not a token."""
    post(chunks({"content": "real"}, {"content": ""}))
    assert list(oc.stream_chat_verbose("m", "s", "u")) == [("content", "real")]


def test_system_and_user_become_the_message_pair(post):
    captured = post(chunks({"content": "x"}))
    list(oc.stream_chat_verbose("qwen3:8b", "SYSTEM", "USER"))
    assert captured["payload"]["messages"] == [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "USER"},
    ]
    assert captured["payload"]["model"] == "qwen3:8b"
    assert captured["payload"]["stream"] is True


def test_think_false_is_forwarded(post):
    """Per-robot reads disable reasoning for speed; it must reach the payload."""
    captured = post(chunks({"content": "x"}))
    list(oc.stream_chat_verbose("m", "s", "u", think=False))
    assert captured["payload"]["think"] is False


def test_think_omitted_when_not_specified(post):
    captured = post(chunks({"content": "x"}))
    list(oc.stream_chat_verbose("m", "s", "u"))
    assert "think" not in captured["payload"]


def test_options_are_forwarded(post):
    captured = post(chunks({"content": "x"}))
    list(oc.stream_chat_verbose("m", "s", "u", options={"num_predict": 160}))
    assert captured["payload"]["options"] == {"num_predict": 160}


def test_http_error_propagates(post):
    post(chunks({"content": "x"}), ok=False)
    with pytest.raises(RuntimeError):
        list(oc.stream_chat_verbose("m", "s", "u"))


def test_base_url_comes_from_the_environment(monkeypatch):
    """OLLAMA_HOST must be honoured or the container cannot reach the service."""
    import importlib

    monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434")
    reloaded = importlib.reload(oc)
    try:
        assert reloaded.OLLAMA_BASE == "http://ollama:11434"
    finally:
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        importlib.reload(oc)


def test_default_model_is_the_small_one():
    """It used to default to qwen3.6:35b-a3b, which is ~10x slower."""
    assert oc.DEFAULT_SYNTHESIS_MODEL == "qwen3:8b"


def test_list_models_returns_empty_when_ollama_is_down(monkeypatch):
    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(oc.requests, "get", boom)
    assert oc.list_models() == []


# --- MLX backend -------------------------------------------------------------


def sse(*deltas):
    out = [f"data: {json.dumps({'choices': [{'delta': d}]})}" for d in deltas]
    return [*out, "data: [DONE]"]


@pytest.fixture
def mlx(monkeypatch):
    monkeypatch.setattr(oc, "BACKEND", "mlx")
    return monkeypatch


def test_mlx_separates_reasoning_from_content(mlx, post):
    post(sse({"reasoning": "hmm"}, {"content": "Answer"}))
    assert list(oc.stream_chat_verbose("m", "s", "u")) == [
        ("thinking", "hmm"),
        ("content", "Answer"),
    ]


def test_mlx_posts_to_the_openai_endpoint_and_maps_options(mlx, post):
    captured = post(sse({"content": "x"}))
    list(
        oc.stream_chat_verbose("qwen3:8b", "SYS", "USR", think=False, options={"num_predict": 160})
    )
    assert captured["url"].endswith("/v1/chat/completions")
    p = captured["payload"]
    assert p["chat_template_kwargs"] == {"enable_thinking": False}
    assert p["max_tokens"] == 160
    assert p["messages"][0] == {"role": "system", "content": "SYS"}
    assert "think" not in p


def test_mlx_generate_retries_invalid_json_once(mlx):
    replies = iter(["not json", '{"ok": true}'])

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": next(replies)}}]}

    mlx.setattr(oc.requests, "post", lambda *a, **k: R())
    assert json.loads(oc.generate("m", "s", "u", fmt="json")) == {"ok": True}


def test_mlx_list_models_empty_when_server_down(mlx):
    def boom(*a, **k):
        raise OSError("refused")

    mlx.setattr(oc.requests, "get", boom)
    assert oc.list_models() == []
    assert oc.is_running() is False
