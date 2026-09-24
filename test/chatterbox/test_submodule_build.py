"""The submodules the model builds for the engine must carry the same computed
buffers (mel filters, STFT windows, RoPE tables, fades) as a module constructed
directly: the first GPU run showed HiFT's istft window as uninitialised memory
after a meta build."""

from __future__ import annotations

import os

import pytest
import torch

from mstar.model.chatterbox.chatterbox_model import ChatterboxModel
from mstar.model.chatterbox.components import S3Gen, S3Tokenizer, VoiceEncoder
from mstar.model.chatterbox.loader import resolve_snapshot

os.environ.setdefault("HF_HUB_OFFLINE", "1")


@pytest.fixture(scope="module")
def model():
    try:
        resolve_snapshot("ResembleAI/chatterbox")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"checkpoint not available: {exc}")
    return ChatterboxModel(model_path_hf="ResembleAI/chatterbox", variant="chatterbox")


def _same_buffers(built: torch.nn.Module, fresh: torch.nn.Module) -> None:
    built_bufs = dict(built.named_buffers())
    fresh_bufs = dict(fresh.named_buffers())
    assert built_bufs.keys() == fresh_bufs.keys()
    for name, ref in fresh_bufs.items():
        got = built_bufs[name].to(ref.dtype)
        assert torch.isfinite(got).all(), name
        # computed (non-persistent) buffers must be exactly what __init__ makes
        if name.rsplit(".", 1)[-1] in _non_persistent(fresh):
            assert torch.equal(got, ref), name


def _non_persistent(module: torch.nn.Module) -> set[str]:
    names = set()
    for sub in module.modules():
        names |= set(sub._non_persistent_buffers_set)
    return names


def test_s3gen_submodule_buffers_match_a_direct_build(model):
    sub = model._create_s3gen_submodule("cpu")
    _same_buffers(sub.s3gen, S3Gen(model.config.s3gen))
    # the istft window is finite and a proper Hann shape (peak 1, zero at the edge)
    window = sub.s3gen.vocoder.stft_window
    assert torch.isfinite(window).all() and window.max().item() == 1.0 and window[0].item() == 0.0


def test_voice_encoder_and_tokenizer_buffers_match_a_direct_build(model):
    sub = model._create_voice_encoder_submodule("cpu")
    _same_buffers(sub.encoder, VoiceEncoder(model.config.voice_encoder))
    _same_buffers(model._s3_tokenizer("cpu"), S3Tokenizer(model.config.s3_tokenizer))
