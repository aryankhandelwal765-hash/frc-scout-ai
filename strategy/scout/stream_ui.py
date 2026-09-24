"""Streamlit helper: render an Ollama token stream with live progress.

Reasoning models (qwen3.6:35b-a3b) spend a long "thinking" phase before the
first answer token. This renders that honestly so the user can tell it's working:
a progress bar, a live elapsed-time clock (ticks during reasoning too), token
count, throughput, and a rough ETA once the answer starts.

The "how much is left" bar is necessarily an estimate for an LLM stream — it is
scaled against an expected answer length and labelled as an estimate.
"""

import time

import streamlit as st

# ~350-word target answer (see synthesis SYSTEM_PROMPT) ≈ this many content tokens.
DEFAULT_TARGET_TOKENS = 480


def render_stream_events(events, target_tokens: int = DEFAULT_TARGET_TOKENS,
                         thinking_label: str = "Reasoning") -> str:
    """Consume a stream of (kind, text) events and render live progress.

    Returns the full assembled content text. `events` comes from
    ollama_client.stream_chat_verbose / stream_chat_with_history_verbose.
    """
    start = time.time()
    bar = st.progress(0.0)
    status = st.empty()
    box = st.empty()

    full = ""
    n_think = 0
    n_content = 0

    for kind, chunk in events:
        elapsed = time.time() - start
        if kind == "thinking":
            n_think += 1
            # Reasoning length is unknown, so creep the bar to a 15% cap and let
            # the elapsed clock be the honest "it's alive" signal.
            bar.progress(min(0.15, n_think / 1200 * 0.15))
            status.markdown(
                f"🧠 **{thinking_label}…**  ·  ⏱ {elapsed:.0f}s elapsed  ·  "
                f"{n_think} reasoning tokens"
            )
        else:  # content
            n_content += 1
            full += chunk
            frac = 0.15 + min(n_content / target_tokens, 1.0) * 0.84
            tps = n_content / elapsed if elapsed > 0 else 0.0
            remaining = max(target_tokens - n_content, 0)
            eta = remaining / tps if tps > 0 else 0.0
            bar.progress(min(frac, 0.99))
            status.markdown(
                f"✍️ Writing  ·  ⏱ {elapsed:.0f}s elapsed  ·  {n_content} tokens  ·  "
                f"{tps:.1f} tok/s  ·  ~{eta:.0f}s left (est.)"
            )
            box.markdown(full + "▌")

    total = time.time() - start
    bar.progress(1.0)
    status.markdown(
        f"✅ Done in **{total:.0f}s**  ·  {n_content} tokens"
        + (f"  ·  {n_think} reasoning tokens" if n_think else "")
    )
    box.markdown(full)
    return full
