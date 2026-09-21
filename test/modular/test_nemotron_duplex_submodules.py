"""CPU unit tests for the Nemotron-Duplex NodeSubmodules — the per-node compute
wrappers — using lightweight fakes (no real weights / GPU).

Covers the engine contract of the nano node (host-only ``prepare_inputs``,
AddFusion in ``preprocess``, the step declaration per walk, the reference's
prompt priming, stop rule) and the codec's per-request left context.
"""
from types import SimpleNamespace

import torch
from torch import nn

from mstar.engine.resources import AttentionStep, KVStep, SamplerStep
from mstar.engine.resources.linear_attn.config import LinearAttnStep
from mstar.engine.resources.recurrent import RecurrentStep
from mstar.model.nemotron_duplex.config import (
    MAMBA,
    MAMBA_STATE,
    NANO_ATTN,
    NANO_KV,
    NANO_SAMPLER,
    NemotronDuplexConfig,
)
from mstar.model.nemotron_duplex.submodules import (
    AudioCodecDecoderSubmodule,
    NemotronHLLMSubmodule,
)

H = 8


def _make_nano() -> NemotronHLLMSubmodule:
    cfg = NemotronDuplexConfig()
    lm = nn.Module()
    lm.embeddings = nn.Embedding(64, H)   # covers the special ids (bos/pad/eos)
    lm.lm_head = nn.Linear(H, 64)
    lm.function_head = nn.Linear(H, 64)
    return NemotronHLLMSubmodule(language_model=lm, config=cfg)


def _fwd_info(iters: int, max_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        dynamic_loop_iter_counts={"decode_loop": iters}, max_tokens=max_tokens,
    )


_ENGINE = SimpleNamespace(request_ids=["r"], resources={}, per_request_states=None)


def _fuse(nano, walk, inputs):
    """prepare_inputs (host) -> preprocess (fusion) for one request."""
    inp = nano.prepare_inputs(walk, None, inputs, resources={})   # the engine passes ``resources=``
    return inp, nano.preprocess(walk, _ENGINE, [inp])


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


def test_nano_prepare_inputs_is_host_only_and_fusion_runs_in_preprocess():
    nano = _make_nano()
    inp, pre = _fuse(nano, "decode", {"audio_frame": [torch.ones(H)],
                                      "prev_text": [torch.tensor([3])], "prev_func": [torch.tensor([4])]})
    assert inp.input_embeds is None and inp.input_ids is None       # nothing embedded yet
    assert inp.input_seq_len == 1
    assert set(inp.tensor_inputs) == {"audio_frame", "prev_text", "prev_func"}
    assert pre["input_embeds"].shape == (1, H) and pre["seq_lens"] == [1]
    cfg, emb = nano.config, nano.embeddings
    expected = (emb(torch.tensor([3])) * cfg.agent_text_weight + torch.ones(1, H) * cfg.user_audio_weight
                + emb(torch.tensor([4])) * cfg.function_weight)
    assert torch.allclose(pre["input_embeds"], expected)


def test_nano_fuse_frame_handles_empty_audio_frame():
    """The terminal audio_frame stream chunk (producer_done race) arrives with the
    key present but an empty tensor list: run a no-audio step instead of
    IndexError-ing on ``inputs["audio_frame"][0]``; missing carry-ins default to BOS / PAD."""
    nano = _make_nano()
    inp, pre = _fuse(nano, "decode", {"audio_frame": []})
    assert "audio_frame" not in inp.tensor_inputs
    assert int(inp.tensor_inputs["prev_text"]) == nano.config.text_bos_id
    assert int(inp.tensor_inputs["prev_func"]) == nano.config.text_pad_id
    assert pre["input_embeds"].shape == (1, H)
    _, with_audio = _fuse(nano, "decode", {"audio_frame": [torch.ones(H)]})
    assert not torch.allclose(with_audio["input_embeds"], pre["input_embeds"])


def test_nano_prompt_priming_matches_reference():
    """System-prompt prefill fuses each prompt token with the agent channel (BOS
    on the first token, PAD after) and a PAD function token — the reference's
    ``_prime_prompt`` — instead of embedding the raw ids."""
    nano = _make_nano()
    cfg, emb = nano.config, nano.embeddings
    ids = torch.tensor([10, 11, 12, 13])
    inp, pre = _fuse(nano, "prefill_text", {"text_inputs": [ids]})
    assert inp.input_seq_len == 4 and pre["seq_lens"] == [4]
    agent = torch.tensor([cfg.text_bos_id] + [cfg.text_pad_id] * 3)
    expected = (emb(agent) * cfg.agent_text_weight + emb(ids) * cfg.user_audio_weight
                + emb(torch.tensor([cfg.text_pad_id])) * cfg.function_weight)
    assert torch.allclose(pre["input_embeds"], expected)


STATE_KEYS = {NANO_KV, NANO_ATTN, MAMBA_STATE, MAMBA}


def test_nano_declare_step_per_walk():
    """Every walk steps the KV + attention and the Mamba pool + resource; only
    decode steps the text sampler (the prompt region is never sampled). One
    'main' segment per request, spanning its token count — padding rows
    included (strict zip)."""
    nano = _make_nano()
    p_inp = nano.prepare_inputs("prefill_text", None, {"text_inputs": [torch.arange(5)]})
    step = nano.declare_step("prefill_text", ["a"], [p_inp])
    assert set(step.keys()) == STATE_KEYS
    assert isinstance(step.get(NANO_KV), KVStep) and isinstance(step.get(NANO_ATTN), AttentionStep)
    assert isinstance(step.get(MAMBA_STATE), RecurrentStep) and isinstance(step.get(MAMBA), LinearAttnStep)
    assert step.get(NANO_ATTN).causal is True
    assert [(s.request_id, s.label, s.span) for s in step.segments] == [("a", "main", 5)]

    d_inp = nano.prepare_inputs("decode", None, {"audio_frame": [torch.ones(H)]})
    step = nano.declare_step("decode", ["a", "b"], [d_inp, d_inp])
    assert set(step.keys()) == STATE_KEYS | {NANO_SAMPLER}
    assert isinstance(step.get(NANO_SAMPLER), SamplerStep)
    assert [(s.request_id, s.span) for s in step.segments] == [("a", 1), ("b", 1)]


def test_nano_captures_the_decode_step():
    """The frame-synchronous decode step is a CUDA graph: one config, decode
    only, whose dummy row is a fused frame (host-only prepare_inputs shape)
    and whose largest bucket fits the model's default recurrent-pool sizing
    (padding rows take a slot each during the step)."""
    from mstar.engine.cuda_graph_config import BatchedCudaGraphConfig
    from mstar.model.nemotron_duplex.nemotron_duplex_model import NemotronDuplexModel

    nano = _make_nano()
    configs = nano.get_cuda_graph_configs(torch.device("cpu"))
    assert len(configs) == 1
    cfg = configs[0]
    assert isinstance(cfg, BatchedCudaGraphConfig)
    assert cfg.capture_graph_walk == "decode" and cfg.replay_graph_walks == ["decode"]
    assert cfg.compile is False
    assert max(cfg.capture_batch_sizes) <= NemotronDuplexModel.DEFAULT_MAMBA_SLOTS
    row = cfg.single_request_inputs
    assert row.input_seq_len == 1 and row.kwargs["mode"] == "frame"
    assert row.tensor_inputs["audio_frame"].shape == (1, nano.config.nano.hidden_size)
    assert int(row.tensor_inputs["prev_text"]) == nano.config.text_bos_id
    # the dummy rows fuse like real frames: preprocess yields one embedding per row
    rows = cfg.get_node_inputs(3, 3)
    nano.embeddings = nn.Embedding(64, nano.config.nano.hidden_size)  # match the frame width
    pre = nano.preprocess("decode", _ENGINE, rows)
    assert pre["input_embeds"].shape == (3, nano.config.nano.hidden_size) and pre["seq_lens"] == [1, 1, 1]


def test_nano_postprocess_feeds_tokens_back():
    nano = _make_nano()
    out = {"new_token": [torch.tensor([7])], "new_func": [torch.tensor([12])]}
    nano.postprocess("r", None, out)
    assert out["prev_text"] is out["new_token"] and out["prev_func"] is out["new_func"]


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


def _ctx(codec, rid):
    return codec.request_state(rid).get(AudioCodecDecoderSubmodule.CONTEXT_KEY)


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
    assert _ctx(codec, "r").shape[0] == min(10, codec.config.eartts.codec_left_context_frames)


def test_codec_cleanup_clears_per_request_context():
    """The context lives in engine-owned per-request state, so the engine's
    ``cleanup_request`` drops it with everything else."""
    codec = _make_codec()
    _run_codec(codec, "r", 5)
    assert _ctx(codec, "r") is not None
    codec.cleanup_request("r")
    assert "r" not in codec.request_states


def test_codec_requests_are_isolated():
    codec = _make_codec()
    _run_codec(codec, "a", 5)
    _run_codec(codec, "b", 3)
    assert _ctx(codec, "a").shape[0] == 5 and _ctx(codec, "b").shape[0] == 3
