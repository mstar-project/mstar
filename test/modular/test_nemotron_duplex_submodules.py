"""CPU unit tests for the Nemotron-Duplex NodeSubmodules — the per-node compute
wrappers — using lightweight fakes (no real weights / GPU).

Covers the engine contract of the nano node (host-only ``prepare_inputs``,
AddFusion in ``preprocess``, the step declaration per walk, the reference's
prompt priming, stop rule). The streaming encoder is the shared one
(``test_fastconformer``); the codec's stream is in ``test_nemotron_duplex_codec``.
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
from mstar.model.nemotron_duplex.submodules import NO_PROMPT, NemotronHLLMSubmodule
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
    assert max(cfg.capture_batch_sizes) <= NemotronDuplexModel.DEFAULT_SESSIONS
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


def _live_tensors() -> int:
    import gc

    gc.collect()
    return sum(type(o) is torch.Tensor for o in gc.get_objects())


def test_codec_inverse_stft_keeps_nothing_per_length():
    """The standalone stream re-decodes the whole code history every frame, so
    each frame is a new length: the inverse STFT must hold nothing per length."""
    from mstar.model.nemotron_duplex.components.audio_codec import _spec_to_wav

    n_fft, hop = 16, 4
    _spec_to_wav(torch.randn(1, n_fft // 2 + 1, 3, dtype=torch.complex64), n_fft, hop, n_fft)
    before = _live_tensors()
    for frames in range(4, 40):
        _spec_to_wav(torch.randn(1, n_fft // 2 + 1, frames, dtype=torch.complex64), n_fft, hop, n_fft)
    assert _live_tensors() == before


def test_codec_refuses_a_window_whose_overlap_add_has_a_gap():
    """The inverse STFT divides by the overlap-added squared window and zeroes
    it where that is ~0; a window and hop that leave a gap are refused when the
    codec is built, not met as silence mid-call."""
    from mstar.model.nemotron_duplex.components.audio_codec import check_window_envelope

    check_window_envelope(torch.hann_window(16), 4)
    with pytest.raises(ValueError, match="overlap-add"):
        check_window_envelope(torch.tensor([0.0, 0, 0, 1, 1, 0, 0, 0]), 4)


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
