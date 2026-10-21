"""CPU unit tests for the Nemotron-Duplex NodeSubmodules — the per-node compute
wrappers — using lightweight fakes (no real weights / GPU).

Covers the engine contract of the nano node (host-only ``prepare_inputs``,
AddFusion in ``preprocess``, the step declaration per walk, the reference's
prompt priming, stop rule) and the codec's per-request left context.
"""
from types import SimpleNamespace

import pytest
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
    NO_PROMPT,
    AudioCodecDecoderSubmodule,
    ConformerEncoderSubmodule,
    NemotronHLLMSubmodule,
)
from mstar.model.submodule_base import BatchedModelOutput

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


def test_nano_never_stops_on_a_token():
    """Text EOS is a normal per-frame token in duplex; the decode loop ends on
    audio-frame stream exhaustion or the Loop's own max_iters, NOT on a token
    -- so the node offers check_stop nothing and stops on nothing."""
    nano = _make_nano()
    eos = {"new_token": [torch.tensor([nano.config.eos_token_id])]}
    assert nano.check_stop("r", _fwd_info(0, 2048), eos) == set()
    assert nano.check_stop("r", _fwd_info(max_tokens=8, iters=7), eos) == set()


def _decode_engine(rids):
    sampler = SimpleNamespace(sample=lambda rids, logits: logits.argmax(dim=-1))
    return SimpleNamespace(request_ids=rids, resources={NANO_SAMPLER: sampler}, per_request_states=None)


def test_nano_forward_hands_over_row_outputs_and_nothing_for_the_stop_check():
    """One batch tensor per output name, row i = request i (the engine clones
    it once and hands out views), the consumed-prompt sentinel on every row,
    and an empty stop-check set: no per-request device-to-host copy per step."""
    nano = _make_nano()
    nano.language_model.forward = lambda x, label=None: x
    torch.manual_seed(0)
    x = torch.randn(3, H)
    out = nano.forward_batched("decode", _decode_engine(["a", "b", "c"]), x, [1, 1, 1])
    assert isinstance(out, BatchedModelOutput) and out.check_stop_buffers == {}
    expect = nano.lm_head(x).argmax(dim=-1)
    torch.testing.assert_close(out.row_outputs["new_token"], expect)
    assert out.row_outputs["text_inputs"].tolist() == [NO_PROMPT] * 3
    if nano.config.use_function_head:
        torch.testing.assert_close(out.row_outputs["new_func"], nano.language_model.function_head(x).argmax(dim=-1))
    # the eager single-request contract is row 0 of the same tensors
    one = nano.forward("decode", _decode_engine(["a"]), x[:1], [1])
    assert one["new_token"][0].shape == (1,) and int(one["new_token"][0]) == int(expect[0])
    assert int(one["text_inputs"][0]) == NO_PROMPT


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
    assert nano.cg_key_info("decode", pending, per_request_input_metadata={}) is None
    assert nano.cg_key_info("decode", {"b": pending["b"]}) == nano.DECODE_KEY
    # the mark stays on the routing for the whole loop: once the request's first
    # step (prompt + frame) has run, its later steps take the capture
    nano.postprocess("a", None, {"new_token": [torch.tensor([7])]})
    assert nano.cg_key_info("decode", pending) == nano.DECODE_KEY


def test_nano_captures_the_decode_step():
    """The frame-synchronous decode step is a CUDA graph: one config, decode
    only, whose dummy row is a fused frame (host-only prepare_inputs shape)
    and whose largest bucket fits the model's default recurrent-pool sizing
    (padding rows take a slot each during the step)."""
    from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig
    from mstar.model.nemotron_duplex.nemotron_duplex_model import NemotronDuplexModel

    nano = _make_nano()
    configs = nano.get_accelerator_graph_configs(torch.device("cpu"))
    assert len(configs) == 1
    cfg = configs[0]
    assert isinstance(cfg, BatchedAcceleratorGraphConfig)
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


@pytest.mark.parametrize("done_before_last_pop", [True, False])
def test_nano_emits_one_token_per_frame_however_the_stream_ends(done_before_last_pop):
    """The audio stream's end can reach the LLM's buffer after its last frame was
    taken; the buffer then releases an empty final chunk, whose step only ends
    the loop. Either way the call gets one token per frame, as offline."""
    from mstar.streaming.chunk_policy import FixedChunkPolicy
    from mstar.streaming.stream_buffer import StreamBuffer

    nano = _make_nano()
    buf = StreamBuffer(request_id="r", edge_name="audio_frame", from_partition="Encoder",
                       policy=FixedChunkPolicy(chunk_size=1, continue_after_done=False))
    for k in range(3):
        buf.pre_read_register(f"f{k}")
        buf.put(f"f{k}", torch.randn(H))
    emitted, finals = [], 0
    while True:
        if len(emitted) == (2 if done_before_last_pop else 3):
            buf.signal_done()
        if not buf.has_chunk_ready():
            break
        chunk = buf.pop_chunk()
        data = chunk.data["data"]
        inputs = {"audio_frame": [] if data is None else [data], "prev_text": [torch.tensor([3])],
                  "prev_func": [torch.tensor([4])], "text_inputs": [torch.tensor([NO_PROMPT])]}
        ni = nano.prepare_inputs("decode", None, inputs, resources={}, is_final_stream_chunk=chunk.is_final)
        out = {"new_token": [torch.tensor([7])], "new_func": [torch.tensor([8])],
               "text_inputs": [torch.tensor([NO_PROMPT])]}
        nano.postprocess("r", None, out, inputs=ni)
        emitted += out.get("new_token", [])
        finals += chunk.is_final
        if chunk.is_final:
            break
    assert finals == 1 and len(emitted) == 3


@pytest.mark.parametrize("final", [True, False])
def test_nano_drops_the_token_of_an_empty_chunk_only_when_it_is_final(final):
    """The engine's final flag, not a missing frame, marks the step that only
    ends the loop."""
    nano = _make_nano()
    inputs = {"audio_frame": [], "prev_text": [torch.tensor([3])], "prev_func": [torch.tensor([4])],
              "text_inputs": [torch.tensor([NO_PROMPT])]}
    ni = nano.prepare_inputs("decode", None, inputs, resources={}, is_final_stream_chunk=final)
    out = {"new_token": [torch.tensor([7])], "new_func": [torch.tensor([8])]}
    nano.postprocess("r", None, out, inputs=ni)
    assert ("new_token" in out) is not final


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


def test_codec_prepare_inputs_flattens_stacked_row_views():
    """The talker emits one [1, num_q] row view per frame; the stream stacks a
    chunk of them to (T, 1, num_q). A whole window arrives as (1, T, num_q)."""
    codec = AudioCodecDecoderSubmodule(_FakeCodec(), NemotronDuplexConfig())
    q = codec.config.eartts.num_quantizers
    stacked = codec.prepare_inputs("codec_chunk", None, {"codec_tokens": [torch.arange(5 * q).view(5, 1, q)]})
    whole = codec.prepare_inputs("codec_chunk", None, {"codec_tokens": [torch.arange(5 * q).view(1, 5, q)]})
    single = codec.prepare_inputs("codec_chunk", None, {"codec_tokens": [torch.arange(q)]})
    assert stacked.input_seq_len == whole.input_seq_len == 5 and single.input_seq_len == 1
    torch.testing.assert_close(stacked.tensor_inputs["codes"], whole.tensor_inputs["codes"])
    assert stacked.tensor_inputs["codes"].shape == (5, q) and single.tensor_inputs["codes"].shape == (1, q)


def test_codec_runs_an_empty_final_chunk_as_a_flush():
    """A frame count divisible by the chunk size ends on an empty final chunk. The codec
    must run it, since a skipped rid in a shared batch never finishes its stream."""
    codec = _make_codec()
    for chunk in ({"codec_tokens": []}, {}):
        inp = codec.prepare_inputs("codec_chunk", None, chunk, is_final_stream_chunk=True)
        assert inp is not None and inp.kwargs["flush"] and not inp.tensor_inputs
    _run_codec(codec, "a", 5, 1)
    eng = SimpleNamespace(request_ids=["a", "b"])
    inputs = [codec.prepare_inputs("codec_chunk", None, {}, is_final_stream_chunk=True),
              codec.prepare_inputs("codec_chunk", None, {"codec_tokens": [_codes(5, 2)]})]
    out = codec.forward_batched("codec_chunk", eng, **codec.preprocess("codec_chunk", eng, inputs))
    assert out["a"]["audio_chunk"][0].numel() == 0     # the fake vocoder holds nothing back
    assert out["b"]["audio_chunk"][0].shape[0] == 5 * _FakeCodec.SPF


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
    lc = codec.config.eartts.codec_left_context_frames
    assert _ctx(codec, "c").shape[0] == min(5, lc) and _ctx(codec, "a").shape[0] == min(10, lc)


def _ctx(codec, rid):
    return codec.request_state(rid).get(AudioCodecDecoderSubmodule.CONTEXT_KEY)


def test_codec_emits_only_new_frames_with_left_context():
    """First chunk emits all its frames; later chunks decode context+new but
    emit ONLY the new frames — so a 5+5 frame stream yields 10 frames of audio,
    not 5+10 (the overlap-re-emission balloon)."""
    codec = _make_codec()
    spf = _FakeCodec.SPF
    assert codec._hold == 0                            # the fake vocoder has no overlap-add edge
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


@pytest.mark.parametrize("frames", [23, 25])
def test_codec_stream_is_the_full_history_decode_with_the_real_decoder(frames):
    """With the real (tiny-channel, real kernel / rates / block count) decoder,
    the chunked stream -- 3 frames of left context, right edge carried into the
    next chunk -- equals the one-shot decode of all frames, sample for sample.
    The final chunk flushes the held tail; 25 frames end on an empty one."""
    from mstar.model.nemotron_duplex.components.audio_codec import AudioCodec
    from mstar.model.nemotron_duplex.config import CodecConfig

    torch.manual_seed(0)
    ccfg = CodecConfig(base_hidden_size=8, latent_size=16, codebook_size=16, num_quantizers=4)
    real = AudioCodec(ccfg).eval()
    for prm in real.parameters():
        if prm.is_floating_point():
            prm.data.normal_(0, 0.05)
    cfg = NemotronDuplexConfig()
    codec = AudioCodecDecoderSubmodule(codec=real, config=cfg)
    assert cfg.eartts.codec_left_context_frames == 3 and codec._hold == ccfg.n_fft - ccfg.hop_length == 12
    chunk = 5
    codes = torch.randint(0, ccfg.codebook_size, (frames, ccfg.num_quantizers))
    with torch.no_grad():
        whole, _ = real.decode(codes.unsqueeze(0))
        whole = (whole[0, 0].clamp(-1, 1) * 32767).to(torch.int16)
        pieces = []
        starts = list(range(0, frames + (frames % chunk == 0), chunk))
        for start in starts:
            piece = codes[start:start + chunk]
            inp = codec.prepare_inputs("codec_chunk", None, {"codec_tokens": [piece] if len(piece) else []},
                                       is_final_stream_chunk=start == starts[-1])
            eng = SimpleNamespace(request_ids=["r"], resources={}, per_request_states=None)
            out = codec.forward_batched("codec_chunk", eng, **codec.preprocess("codec_chunk", eng, [inp]))
            pieces.append(out["r"]["audio_chunk"][0])
    stream = torch.cat(pieces)
    assert stream.shape[0] == whole.shape[0]
    torch.testing.assert_close(stream, whole, atol=1, rtol=0)   # int16: one quantization step


def test_codec_requests_are_isolated():
    codec = _make_codec()
    _run_codec(codec, "a", 5)
    _run_codec(codec, "b", 3)
    lc = codec.config.eartts.codec_left_context_frames
    assert _ctx(codec, "a").shape[0] == min(5, lc) and _ctx(codec, "b").shape[0] == min(3, lc)


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


def _small_perception():
    """A real Perception, shrunk, with random weights: the streamer runs its modules."""
    from mstar.model.nemotron_duplex.components.conformer import Perception
    from mstar.model.nemotron_duplex.config import ConformerSTTConfig

    torch.manual_seed(0)
    stt = ConformerSTTConfig(d_model=64, num_heads=4, num_layers=3, n_mels=32, feat_in=32,
                             subsampling_conv_channels=8, output_dim=48, att_context_size=(6, 0))
    p = Perception(stt)
    with torch.no_grad():
        for t in p.parameters():
            t.copy_(torch.randn_like(t) * 0.2)
        p.preprocessor.featurizer.window.copy_(torch.hann_window(stt.win_length, periodic=False))
        p.preprocessor.featurizer.fb.copy_(torch.rand_like(p.preprocessor.featurizer.fb))
    return p.eval()


def _make_encoder(max_session_samples=16000 * 10) -> ConformerEncoderSubmodule:
    return ConformerEncoderSubmodule(_small_perception(), nn.Identity(), nn.Identity(),
                                     config=NemotronDuplexConfig(), max_session_samples=max_session_samples)


def _prepare(enc, handle, chunk, final=False):
    inputs = {"audio_in": [chunk]} if chunk is not None else {"audio_in": []}
    return enc.prepare_inputs("encode", SimpleNamespace(rid_handle=handle), inputs, is_final_stream_chunk=final)


def _encode(enc, handle, chunk, final=False):
    """One encoder step for one call, as the engine runs one unbatched:
    prepare_inputs, preprocess, forward."""
    ni = _prepare(enc, handle, chunk, final)
    eng = SimpleNamespace(request_ids=[handle])
    with torch.no_grad():
        out = enc.forward("encode", eng, **enc.preprocess("encode", eng, [ni]))
    return out["audio_frame"]


def _encode_batch(enc, steps):
    """One batched encoder step, as the engine runs it: each ``(handle, chunk,
    final)`` row through prepare_inputs, then one preprocess and forward_batched
    over all of them. Returns each row's frames by handle."""
    nis = [_prepare(enc, h, c, f) for h, c, f in steps]
    eng = SimpleNamespace(request_ids=[h for h, _c, _f in steps])
    with torch.no_grad():
        out = enc.forward_batched("encode", eng, **enc.preprocess("encode", eng, nis))
    return {h: out.per_rid_outputs[h]["audio_frame"] for h, _c, _f in steps}


def _offline(enc, wav):
    with torch.no_grad():
        return enc.perception(wav.unsqueeze(0), torch.tensor([wav.numel()]))[0][0]


@pytest.mark.parametrize("end_rides_last_chunk", [False, True])
@pytest.mark.parametrize("n", [16000 * 2 + 777, 16000 * 2])
def test_encoder_streams_the_offline_frames(n, end_rides_last_chunk):
    """Chunks of any size, smaller than a frame included, then the end of input,
    on an empty final chunk or on the last chunk of audio (as an upload's usually
    is): the frames emitted in order are the offline encoding of the call, and
    the mel frames the offline featurizer's, the reflected right edge included
    (a length on the hop grid is the one whose last frame reads it)."""
    enc = _make_encoder()
    wav = torch.randn(n) * 0.1
    want = _offline(enc, wav)
    with torch.no_grad():
        mel = enc.perception.preprocessor(wav.unsqueeze(0), torch.tensor([n]))[0][0].T
    for handle, size in enumerate((320, 1280, 2560, 9000, 64000)):
        got = []
        for at in range(0, wav.numel(), size):
            last = at + size >= wav.numel()
            got += _encode(enc, handle, wav[at:at + size], final=last and end_rides_last_chunk)
        if not end_rides_last_chunk:
            got += _encode(enc, handle, None, final=True)
        assert len(got) == want.shape[0]
        assert torch.allclose(torch.stack(got), want, atol=1e-5), size
        st = enc.request_state(handle).get(enc.STATE_KEY)
        assert st.mels == mel.shape[0]
        assert torch.allclose(st.mel[:st.mels], mel, atol=1e-5), size


def test_encoder_batches_calls_at_different_points():
    """Two calls in one step, each with its own chunking, each get their own
    offline frames."""
    enc = _make_encoder()
    a, b = torch.randn(16000 * 2) * 0.1, torch.randn(16000 * 3) * 0.1
    got = {1: [], 2: []}
    pos = {1: 0, 2: 0}
    size = {1: 1280, 2: 4000}
    wav = {1: a, 2: b}
    while any(pos[h] < wav[h].numel() for h in (1, 2)):
        rids, chunks, finals = [], [], []
        for h in (1, 2):
            if pos[h] < wav[h].numel():
                ni = _prepare(enc, h, wav[h][pos[h]:pos[h] + size[h]])
                pos[h] += size[h]
                rids.append(h)
                chunks.append(ni.tensor_inputs["chunk"])
                finals.append(False)
        with torch.no_grad():
            out = enc.forward_batched("encode", SimpleNamespace(request_ids=rids), chunk=chunks, final=finals)
        for h in rids:
            got[h] += out.per_rid_outputs[h]["audio_frame"]
    for h in (1, 2):
        got[h] += _encode(enc, h, None, final=True)
        want = _offline(enc, wav[h])
        assert torch.allclose(torch.stack(got[h]), want, atol=1e-5)


@pytest.mark.parametrize("b_end_rides", [False, True])
@pytest.mark.parametrize("a_chunk", [1280, 9000])
@pytest.mark.parametrize("b_len", [16000 * 2, 16000 * 2 + 640, 16000 * 2 + 777])
def test_encoder_a_calls_last_frames_do_not_depend_on_its_batch(b_len, a_chunk, b_end_rides):
    """Call B ends in the same step as call A's longer chunk, A still mid-call:
    B's last frames, which read past its audio, are still its offline ones."""
    enc = _make_encoder()
    starts = list(range(0, b_len, 1280))
    steps = len(starts) + (0 if b_end_rides else 1)
    a, b = torch.randn(a_chunk * steps) * 0.1, torch.randn(b_len) * 0.1
    enc.max_session_samples = a.numel()
    got_b = []
    for k, pb in enumerate(starts):
        b_final = b_end_rides and k == len(starts) - 1
        rows = [(1, a[k * a_chunk:(k + 1) * a_chunk], False), (2, b[pb:pb + 1280], b_final)]
        got_b += _encode_batch(enc, rows)[2]
    if not b_end_rides:
        got_b += _encode_batch(enc, [(1, a[-a_chunk:], False), (2, None, True)])[2]
    want = _offline(enc, b)
    assert len(got_b) == want.shape[0]
    assert torch.allclose(torch.stack(got_b), want, atol=1e-5)


def test_encoder_runs_a_chunk_over_a_second_in_a_pass_of_its_own():
    """A streamer pass pads every call to its longest chunk: an upload or a large
    append runs alone, and only the calls with a second or less share a pass."""
    enc = _make_encoder()
    enc.forward_batched("encode", SimpleNamespace(request_ids=[]), chunk=[], final=[])   # builds the streamer
    steps = []
    step = enc._streamer.step
    enc._streamer.step = lambda sts, chunks, final: steps.append(
        sorted(0 if c is None else c.numel() for c in chunks)) or step(sts, chunks, final)
    wav = {1: torch.randn(16000 * 5) * 0.1, 2: torch.randn(1280) * 0.1, 3: torch.randn(16001) * 0.1,
           4: torch.randn(16000) * 0.1}
    ni = [_prepare(enc, h, w, final=True) for h, w in wav.items()]
    eng = SimpleNamespace(request_ids=list(wav))
    with torch.no_grad():
        out = enc.forward_batched("encode", eng, **enc.preprocess("encode", eng, ni))
    assert sorted(steps) == [[1280, 16000], [16001], [16000 * 5]]
    for h, w in wav.items():
        assert torch.allclose(torch.stack(out.per_rid_outputs[h]["audio_frame"]), _offline(enc, w), atol=1e-5)


def test_encoder_offers_nothing_to_the_stop_check():
    """The encoder has no check_stop and its stream ends with the client's: its
    frames are not copied to the host every step for one."""
    enc = _make_encoder()
    ni = _prepare(enc, 1, torch.randn(4000) * 0.1)
    eng = SimpleNamespace(request_ids=[1])
    with torch.no_grad():
        out = enc.forward_batched("encode", eng, **enc.preprocess("encode", eng, [ni]))
    assert isinstance(out, BatchedModelOutput) and out.check_stop_buffers == {}
    assert len(out.per_rid_outputs[1]["audio_frame"]) > 0


def test_encoder_projects_the_relative_positions_once_per_layer():
    """The positions a query can see are fixed, so each layer's projection of
    them is computed once, not every step."""
    enc = _make_encoder()
    calls = []
    for layer in enc.perception.encoder.layers:
        layer.self_attn.linear_pos.register_forward_hook(lambda *a: calls.append(1))
    wav = torch.randn(16000) * 0.1
    for at in range(0, wav.numel(), 1280):
        _encode(enc, 1, wav[at:at + 1280])
    assert len(calls) == len(enc.perception.encoder.layers)


def test_encoder_waits_for_enough_audio_to_featurize():
    enc = _make_encoder()
    assert _encode(enc, 1, torch.zeros(10)) == []


@pytest.mark.parametrize("n", [257, 1000, 1279, 1281])
def test_encoder_a_short_call_gets_the_offline_frames(n):
    """Offline encodes any call longer than n_fft/2 samples, under one frame too."""
    enc = _make_encoder()
    wav = torch.randn(n) * 0.1
    want = _offline(enc, wav)
    for handle, size in enumerate((100, n)):
        got = []
        for at in range(0, n, size):
            got += _encode(enc, handle, wav[at:at + size])
        got += _encode(enc, handle, None, final=True)
        assert len(got) == want.shape[0]
        assert torch.allclose(torch.stack(got), want, atol=1e-5), size


@pytest.mark.parametrize("chunks", [[], [torch.zeros(256)], [torch.zeros(100), torch.zeros(156)]])
def test_encoder_fails_a_call_that_ends_without_enough_audio(chunks):
    """No audio, or no more than n_fft/2 samples (which offline can't featurize):
    the call fails rather than reply to nothing."""
    enc = _make_encoder()
    for chunk in chunks:
        _encode(enc, 1, chunk)
    with pytest.raises(ValueError, match="ended with"):
        _encode(enc, 1, None, final=True)


def test_encoder_state_is_per_call_and_cleaned_up():
    enc = _make_encoder()
    _encode(enc, 1, torch.randn(4000) * 0.1)
    _encode(enc, 2, torch.randn(4000) * 0.1)
    enc.cleanup_request(1)
    assert 1 not in enc.request_states and 2 in enc.request_states
