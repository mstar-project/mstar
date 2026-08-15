"""CPU unit tests for the Nemotron-Duplex NodeSubmodules — the per-node compute
wrappers — using lightweight fakes (no real weights / GPU).

Regression coverage for the E5 serving fixes:
  * nano ``check_stop`` must NOT stop on EOS (a normal per-frame token in duplex);
  * nano ``_fuse_frame`` must tolerate the empty terminal ``audio_frame`` chunk;
  * the codec keeps a per-request left-context and emits ONLY the new frames
    (a stateless full-window emit balloons the audio ~window/chunk×).
"""
from types import SimpleNamespace

import torch
from torch import nn

from mstar.model.nemotron_duplex.config import NemotronDuplexConfig
from mstar.model.nemotron_duplex.submodules import (
    AudioCodecDecoderSubmodule,
    NemotronHLLMSubmodule,
)


def _make_nano() -> NemotronHLLMSubmodule:
    cfg = NemotronDuplexConfig()
    lm = nn.Module()
    lm.embeddings = nn.Embedding(64, 8)   # covers the special ids (bos/pad/eos)
    lm.lm_head = nn.Linear(8, 64)
    lm.function_head = nn.Linear(8, 64)
    return NemotronHLLMSubmodule(language_model=lm, config=cfg)


def _fwd_info(iters: int, max_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        dynamic_loop_iter_counts={"decode_loop": iters}, max_tokens=max_tokens,
    )


def test_nano_check_stop_ignores_eos():
    """Text EOS is a normal per-frame token in duplex; the decode loop ends on
    audio-frame stream exhaustion, NOT on EOS — so check_stop must not stop on it."""
    nano = _make_nano()
    eos = {"new_token": [torch.tensor([nano.config.eos_token_id])]}
    assert nano.check_stop("r", _fwd_info(0, 2048), eos) == set()


def test_nano_check_stop_hits_max_tokens():
    nano = _make_nano()
    out = {"new_token": [torch.tensor([5])]}
    assert nano.check_stop("r", _fwd_info(max_tokens=8, iters=7), out) == {"decode_loop"}


def test_nano_fuse_frame_handles_empty_audio_frame():
    """The terminal audio_frame stream chunk (producer_done race) arrives with the
    key present but an empty tensor list; _fuse_frame must run a no-audio step
    instead of IndexError-ing on ``inputs["audio_frame"][0]``."""
    nano = _make_nano()
    empty = nano.prepare_inputs("decode", None, {"audio_frame": []})
    assert empty.input_embeds is not None
    assert empty.input_embeds.shape == (1, 8) and empty.input_seq_len == 1


def test_nano_fuse_frame_adds_audio_when_present():
    nano = _make_nano()
    got = nano.prepare_inputs("decode", None, {"audio_frame": [torch.ones(8)]})
    assert got.input_embeds.shape == (1, 8)
    # with a nonzero audio frame the fused embed differs from the no-audio (empty) step
    no_audio = nano.prepare_inputs("decode", None, {"audio_frame": []})
    assert not torch.allclose(got.input_embeds, no_audio.input_embeds)


class _FakeCodec(nn.Module):
    """Stand-in vocoder: emits SPF samples per code frame (value = frame index),
    so a chunk's emitted length and content are checkable."""

    SPF = 16

    def decode(self, codes, code_len):
        tf = codes.shape[1]
        wav = torch.repeat_interleave(torch.arange(tf, dtype=torch.float32) / 1000.0, self.SPF)
        return wav.view(1, 1, -1), torch.tensor([tf * self.SPF])


def _make_codec() -> AudioCodecDecoderSubmodule:
    return AudioCodecDecoderSubmodule(codec=_FakeCodec(), config=NemotronDuplexConfig())


def _run_codec(codec, rid, n_frames):
    codes = torch.zeros(n_frames, 4, dtype=torch.long)
    eng = SimpleNamespace(request_ids=[rid])
    return codec.forward("codec_chunk", eng, codes=codes)["audio_chunk"][0]


def test_codec_emits_only_new_frames_with_left_context():
    """First chunk emits all its frames; later chunks decode context+new but
    emit ONLY the new frames — so a 5+5 frame stream yields 10 frames of audio,
    not 5+10 (the overlap-re-emission balloon)."""
    codec = _make_codec()
    spf = _FakeCodec.SPF
    a = _run_codec(codec, "r", 5)
    assert a.shape[0] == 5 * spf                       # first chunk: all 5 frames
    b = _run_codec(codec, "r", 5)
    assert b.shape[0] == 5 * spf                       # second chunk: only the 5 NEW frames
    # context rolled forward, capped at codec_left_context_frames
    assert codec._ctx["r"].shape[0] == min(10, codec.config.eartts.codec_left_context_frames)


def test_codec_cleanup_clears_per_request_context():
    codec = _make_codec()
    _run_codec(codec, "r", 5)
    assert "r" in codec._ctx
    codec.cleanup_request("r")
    assert "r" not in codec._ctx


def test_codec_requests_are_isolated():
    codec = _make_codec()
    _run_codec(codec, "a", 5)
    _run_codec(codec, "b", 3)
    assert codec._ctx["a"].shape[0] == 5 and codec._ctx["b"].shape[0] == 3
