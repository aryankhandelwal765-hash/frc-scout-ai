"""vlm_reader / read_numbers reader switch, with the VLM server mocked out."""

import io
import json

import numpy as np
import pytest
import read_numbers as rn
import vlm_reader


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def serve(monkeypatch, text):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data)
        return FakeResp(json.dumps({"choices": [{"message": {"content": text}}]}).encode())

    monkeypatch.setattr(vlm_reader.urllib.request, "urlopen", fake_urlopen)
    return seen


CROP = np.full((40, 120, 3), 200, np.uint8)


def test_digits_are_extracted_from_the_reply(monkeypatch):
    seen = serve(monkeypatch, " 4191\n")
    assert vlm_reader.read_digits(CROP) == "4191"
    assert seen["url"].endswith("/v1/chat/completions")
    part = seen["body"]["messages"][0]["content"][0]
    assert part["image_url"]["url"].startswith("data:image/png;base64,")


def test_unknown_reply_gives_empty(monkeypatch):
    serve(monkeypatch, "UNKNOWN")
    assert vlm_reader.read_digits(CROP) == ""


def test_unreachable_server_gives_empty_not_an_error(monkeypatch):
    def boom(*a, **k):
        raise OSError("refused")

    monkeypatch.setattr(vlm_reader.urllib.request, "urlopen", boom)
    assert vlm_reader.read_digits(CROP) == ""


def test_vlm_mode_snaps_to_a_candidate_without_loading_easyocr(monkeypatch):
    serve(monkeypatch, "4191")
    monkeypatch.setattr(rn, "BUMPER_READER", "vlm")
    monkeypatch.setattr(rn, "_ocr_digits", lambda img: pytest.fail("EasyOCR must not run"))
    frame = np.full((400, 400, 3), 128, np.uint8)
    team, conf = rn.read_bumper(frame, (50, 50, 350, 250), ["4191", "254", "1678"])
    assert team == "4191" and conf >= rn.MIN_SCORE
