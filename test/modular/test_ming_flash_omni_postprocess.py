"""Tests for MingFlashOmniModel.postprocess multi-modality encoding.

The model emits three output modalities across its graph walks — text
(thinker decode), audio (talker), image (imagegen) — so postprocess must encode
all three. Pure CPU: build a bare model via __new__ + a stub tokenizer.
"""

from __future__ import annotations

import io

import pytest
import torch
from PIL import Image

from mstar.model.ming_omni_flash.ming_omni_flash_model import MingFlashOmniModel


class _StubTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(65 + (i % 26)) for i in ids)


def _model() -> MingFlashOmniModel:
    inst = MingFlashOmniModel.__new__(MingFlashOmniModel)
    inst.tokenizer = _StubTokenizer()
    return inst


# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------


def test_postprocess_text_returns_utf8() -> None:
    out = _model().postprocess(torch.tensor([0, 1, 2]), "text")
    assert out == b"ABC"


def test_postprocess_empty_returns_empty_bytes() -> None:
    assert _model().postprocess(torch.tensor([], dtype=torch.long), "text") == b""


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------


def test_postprocess_audio_returns_raw_float32_pcm() -> None:
    wav = torch.tensor([0.0, 0.5, -0.5, 1.0], dtype=torch.float32)
    out = _model().postprocess(wav, "audio")
    # 4 float32 samples = 16 bytes; round-trips exactly.
    assert len(out) == 16
    import numpy as np

    assert np.frombuffer(out, dtype=np.float32).tolist() == [0.0, 0.5, -0.5, 1.0]


def test_postprocess_audio_empty() -> None:
    assert _model().postprocess(torch.empty(0), "audio") == b""


# ---------------------------------------------------------------------------
# image
# ---------------------------------------------------------------------------


def test_postprocess_image_chw_returns_png() -> None:
    img = torch.zeros(3, 8, 8)  # mid-gray after [-1,1]->[0,255] is 128
    out = _model().postprocess(img, "image")
    decoded = Image.open(io.BytesIO(out))
    assert decoded.format == "PNG"
    assert decoded.size == (8, 8)  # (W, H)
    assert decoded.mode == "RGB"
    # 0.0 in [-1,1] maps to (0+1)*127.5 = 127.5 -> round 128.
    px = decoded.getpixel((0, 0))
    assert px == (128, 128, 128)


def test_postprocess_image_bchw_takes_first() -> None:
    img = torch.ones(2, 3, 4, 4)  # 1.0 -> 255
    out = _model().postprocess(img, "image")
    decoded = Image.open(io.BytesIO(out))
    assert decoded.size == (4, 4)
    assert decoded.getpixel((0, 0)) == (255, 255, 255)


def test_postprocess_image_clamps_out_of_range() -> None:
    img = torch.full((3, 2, 2), 5.0)  # clamps to 1.0 -> 255
    out = _model().postprocess(img, "image")
    decoded = Image.open(io.BytesIO(out))
    assert decoded.getpixel((0, 0)) == (255, 255, 255)


def test_postprocess_image_single_channel_expands_to_rgb() -> None:
    img = torch.zeros(1, 4, 4)  # 1-channel -> repeated to RGB
    out = _model().postprocess(img, "image")
    decoded = Image.open(io.BytesIO(out))
    assert decoded.mode == "RGB"
    assert decoded.getpixel((0, 0)) == (128, 128, 128)


def test_postprocess_image_bad_shape_raises() -> None:
    with pytest.raises(ValueError, match="expected"):
        _model().postprocess(torch.zeros(5, 8, 8), "image")  # 5 channels


# ---------------------------------------------------------------------------
# unknown
# ---------------------------------------------------------------------------


def test_postprocess_unknown_modality_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported modality"):
        _model().postprocess(torch.zeros(3), "video")
