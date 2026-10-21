"""
Read a bumper number with a vision-language model served by `mlx_vlm.server`.

The VLM lives in its own venv (it pins a different transformers than the rest of the
project), so this talks to it over HTTP instead of importing it:

    vision/.venv-vlm/bin/python -m mlx_vlm.server --model mlx-community/Qwen3.6-35B-A3B-4bit \
        [--adapter-path vision/runs/vlm/v1] --port 8081

    VLM_HOST=http://localhost:8081   where the server is (default shown)

read_digits(crop_bgr) -> digits string, or "" when the model says UNKNOWN, the server is
unreachable, or the reply has no digits. Failure is silent on purpose: the caller falls back
to whatever other reader it has, and an unreachable server must not stop a batch.
"""

import base64
import json
import logging
import os
import re
import urllib.request

import cv2

logger = logging.getLogger(__name__)

VLM_HOST = os.environ.get("VLM_HOST", "http://localhost:8081")
TIMEOUT = float(os.environ.get("VLM_TIMEOUT", "60"))

# Must match vision/digits/eval_vlm.OPEN_PROMPT: the eval and the fine-tune use this wording.
PROMPT = (
    "This is a crop of a robot bumper from an FRC match video. Read the white team number "
    "printed on the bumper. Reply with ONLY the digits, or UNKNOWN if you cannot read them."
)


def _payload(crop_bgr):
    ok, buf = cv2.imencode(".png", crop_bgr)
    if not ok:
        return None
    uri = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()
    return {
        "model": "default_model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": uri}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        "max_tokens": 16,
        "temperature": 0.0,
    }


def read_digits(crop_bgr):
    body = _payload(crop_bgr)
    if body is None:
        return ""
    req = urllib.request.Request(
        VLM_HOST + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            text = json.load(resp)["choices"][0]["message"]["content"] or ""
    except Exception as e:  # noqa: BLE001 -- see module docstring
        logger.debug("vlm read failed: %r", e)
        return ""
    return re.sub(r"\D", "", text)
