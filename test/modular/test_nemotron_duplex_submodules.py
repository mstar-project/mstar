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


def test_nano_prompt_rides_the_first_frame_step():
    """A session's first step with a system prompt is prompt tokens + first
    frame in one sequence: each prompt token fused with the agent channel (BOS
    on the first, PAD after) and a PAD function token -- the reference's
    ``_prime_prompt`` -- followed by the frame fused with prev PAD. Afterwards
    the loop-back carries the NO_PROMPT sentinel and the step is a plain frame."""
    from mstar.model.nemotron_duplex.submodules import NO_PROMPT

    nano = _make_nano()
    cfg, emb = nano.config, nano.embeddings
    ids = torch.tensor([10, 11, 12, 13])
    frame = torch.ones(H)
    inputs = {"text_inputs": [ids], "audio_frame": [frame],
              "prev_text": [torch.tensor([cfg.text_pad_id])], "prev_func": [torch.tensor([cfg.text_pad_id])]}
    inp, pre = _fuse(nano, "decode", inputs)
    assert inp.input_seq_len == 5 and pre["seq_lens"] == [5] and inp.kwargs["mode"] == "prompt_frame"
    agent = torch.tensor([cfg.text_bos_id] + [cfg.text_pad_id] * 3)
    pad = emb(torch.tensor([cfg.text_pad_id]))
    prompt_rows = emb(agent) * cfg.agent_text_weight + emb(ids) * cfg.user_audio_weight + pad * cfg.function_weight
    frame_row = pad * cfg.agent_text_weight + frame.view(1, -1) * cfg.user_audio_weight + pad * cfg.function_weight
    assert torch.allclose(pre["input_embeds"], torch.cat([prompt_rows, frame_row]))
    # sentinel (or absent) text_inputs -> an ordinary one-token frame step
    inputs["text_inputs"] = [torch.tensor([NO_PROMPT])]
    inp2, pre2 = _fuse(nano, "decode", inputs)
    assert inp2.input_seq_len == 1 and inp2.kwargs["mode"] == "frame" and pre2["input_embeds"].shape == (1, H)
    assert torch.allclose(pre2["input_embeds"], frame_row)


STATE_KEYS = {NANO_KV, NANO_ATTN, MAMBA_STATE, MAMBA}


def test_nano_declare_step_per_walk():
    """Every walk steps the KV + attention and the Mamba pool + resource; only
    decode steps the text sampler (the prompt region is never sampled). One
    'main' segment per request, spanning its token count — padding rows
    included (strict zip)."""
    nano = _make_nano()
    p_inp = nano.prepare_inputs("decode", None, {"text_inputs": [torch.arange(4)], "audio_frame": [torch.ones(H)]})
    step = nano.declare_step("decode", ["a"], [p_inp])
    assert set(step.keys()) == STATE_KEYS | {NANO_SAMPLER}
    assert isinstance(step.get(NANO_KV), KVStep) and isinstance(step.get(NANO_ATTN), AttentionStep)
    assert isinstance(step.get(MAMBA_STATE), RecurrentStep) and isinstance(step.get(MAMBA), LinearAttnStep)
    assert step.get(NANO_ATTN).causal is True
    assert [(s.request_id, s.label, s.span) for s in step.segments] == [("a", "main", 5)]
    assert step.cg_key_info is None                       # prompt + frame: no fixed-shape capture fits

    d_inp = nano.prepare_inputs("decode", None, {"audio_frame": [torch.ones(H)]})
    step = nano.declare_step("decode", ["a", "b"], [d_inp, d_inp])
    assert set(step.keys()) == STATE_KEYS | {NANO_SAMPLER}
    assert isinstance(step.get(NANO_SAMPLER), SamplerStep)
    assert [(s.request_id, s.span) for s in step.segments] == [("a", 1), ("b", 1)]
    assert step.cg_key_info == nano.DECODE_KEY
    # the engine asks before the inputs exist: the routing's step_metadata answers
    pending = {"a": SimpleNamespace(step_metadata={"prompt_pending": True}), "b": SimpleNamespace(step_metadata={})}
    assert nano.cg_key_info("decode", pending) is None
    assert nano.cg_key_info("decode", {"b": pending["b"]}) == nano.DECODE_KEY


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
    assert cfg.compile is False and cfg.additional_key_info == nano.DECODE_KEY
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
    """Stand-in vocoder: emits SPF samples per code frame, valued by the frame's
    first code (so a chunk's emitted length AND content are checkable), and
    counts its calls."""

    SPF = 16

    def __init__(self):
        super().__init__()
        self.calls = 0

    def decode(self, codes, code_len):
        self.calls += 1
        b, tf = codes.shape[0], codes.shape[1]
        per_frame = codes[:, :, 0].float() / 1000.0                     # (B, Tf)
        wav = torch.repeat_interleave(per_frame, self.SPF, dim=1)       # (B, Tf * SPF)
        return wav.view(b, 1, -1), torch.full((b,), tf * self.SPF)


def _make_codec() -> AudioCodecDecoderSubmodule:
    return AudioCodecDecoderSubmodule(codec=_FakeCodec(), config=NemotronDuplexConfig())


def _codes(n_frames, value=0):
    return torch.full((n_frames, 4), value, dtype=torch.long)


def _run_codec(codec, rid, n_frames, value=0):
    eng = SimpleNamespace(request_ids=[rid])
    return codec.forward("codec_chunk", eng, codes=_codes(n_frames, value))["audio_chunk"][0]


def _run_codec_batch(codec, rids, frames):
    eng = SimpleNamespace(request_ids=list(rids))
    inputs = [codec.prepare_inputs("codec_chunk", None, {"codec_tokens": [_codes(n, v)]})
              for n, v in frames]
    pre = codec.preprocess("codec_chunk", eng, inputs)
    return codec.forward_batched("codec_chunk", eng, **pre)


def test_codec_batches_requests_with_equal_windows():
    """Requests at the same window length share one decode; a newcomer with a
    shorter window gets its own; every request emits only its new frames with
    its own content."""
    codec = _make_codec()
    _run_codec(codec, "a", 5, 1)
    _run_codec(codec, "b", 5, 2)
    codec.codec.calls = 0
    out = _run_codec_batch(codec, ["a", "b", "c"], [(5, 3), (5, 4), (5, 5)])
    assert codec.codec.calls == 2                      # {a, b} at 10 frames, c at 5
    spf = _FakeCodec.SPF
    for rid, value in (("a", 3), ("b", 4), ("c", 5)):
        wav = out[rid]["audio_chunk"][0]
        assert wav.shape[0] == 5 * spf
        assert torch.all(wav == int(value / 1000.0 * 32767))    # the new frames' own content
    assert _ctx(codec, "c").shape[0] == 5 and _ctx(codec, "a").shape[0] == 10


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


def test_nano_frame_batch_fuses_rows_exactly_like_the_per_request_path():
    """The steady-state preprocess fuses the whole batch at once; each row must
    equal the per-request fusion (same weights, same terms, same order)."""
    from types import SimpleNamespace

    from mstar.model.nemotron_duplex.submodules import _MODE_FRAME
    from mstar.model.submodule_base import ARNodeInputs

    nano = _make_nano()
    torch.manual_seed(0)
    inputs = [
        ARNodeInputs(input_seq_len=1, kwargs={"mode": _MODE_FRAME}, tensor_inputs={
            "audio_frame": torch.randn(1, H), "prev_text": torch.tensor([3 + i]), "prev_func": torch.tensor([9 - i]),
        })
        for i in range(3)
    ]
    # a session past the end of its audio has no frame in the same batch
    inputs.append(ARNodeInputs(input_seq_len=1, kwargs={"mode": _MODE_FRAME},
                               tensor_inputs={"prev_text": torch.tensor([7]), "prev_func": torch.tensor([2])}))
    eng = SimpleNamespace(request_ids=["a", "b", "c", "d"], resources={}, per_request_states=None)
    pre = nano.preprocess("decode", eng, inputs)
    per_request = torch.cat([nano._fuse(inp, torch.device("cpu")) for inp in inputs])
    assert pre["seq_lens"] == [1, 1, 1, 1]
    torch.testing.assert_close(pre["input_embeds"], per_request)
