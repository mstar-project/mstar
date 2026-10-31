"""CPU unit tests for the Nemotron-Duplex NodeSubmodules — the per-node compute
wrappers — using lightweight fakes (no real weights / GPU).

Covers the engine contract of the nano node (host-only ``prepare_inputs``,
AddFusion in ``preprocess``, the step declaration per walk, the reference's
prompt priming, stop rule) and the streaming encoder (the codec's stream is in
``test_nemotron_duplex_codec``).
"""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mstar.engine.resources import AttentionStep, KVStep, SamplerStep
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.bounded import BoundedKVSpec
from mstar.engine.resources.kv.bounded.manager import BoundedKVManager
from mstar.engine.resources.linear_attn.config import LinearAttnStep
from mstar.engine.resources.recurrent import RecurrentStateConfig, RecurrentStateSpec, RecurrentStep
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.step import StepContext
from mstar.model.nemotron_duplex.components.conformer import (
    encoder_kv_config,
    encoder_state_blocks,
    front_state_blocks,
)
from mstar.model.nemotron_duplex.config import (
    ENCODER_FRONT,
    ENCODER_KV,
    ENCODER_STATE,
    MAMBA,
    MAMBA_STATE,
    NANO_ATTN,
    NANO_KV,
    NANO_SAMPLER,
    NemotronDuplexConfig,
)
from mstar.model.nemotron_duplex.submodules import (
    NO_PROMPT,
    ConformerEncoderSubmodule,
    NemotronHLLMSubmodule,
)
from mstar.model.submodule_base import BatchedModelOutput, ModelInputsFromEngine

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


def _captured_on_cpu(config, label, resources=None, step_runner=None, bs=(1, 2, 4)):
    """The engine's piecewise runner for ``config``, captured on CPU: the real
    runner (bucket padding, its own static buffers per bucket, the region's
    admit / plan / commit), with a "graph" whose replay re-runs the region
    eagerly. ``calls`` counts its replays."""
    from unittest import mock

    from mstar.engine import accelerator_graph_runner as cgr
    from mstar.engine.accelerator_graph_backend import AcceleratorGraphPool

    class _Replay:
        def __init__(self, fn):
            self.fn = fn

        def replay(self):
            self.fn()

    config.capture_batch_sizes = list(bs)
    backend = SimpleNamespace(is_available=lambda: True, set_device=lambda: None, synchronize=lambda: None)
    runner = cgr.PiecewiseAcceleratorGraphRunner(
        label=label, config=config, resources=resources or {}, step_runner=step_runner or StepRunner({}),
        device=torch.device("cpu"), autocast_dtype=None, num_slots=1, node_name=label,
        memory_pool=AcceleratorGraphPool("cpu"), device_capability=SimpleNamespace(graph_backend=backend),
    )
    # a CPU device has no capture plan, so every bucket is warmed up and captured; the warm-up's
    # memory readings are the accelerator's, which a CPU device has none of
    no_memory_stats = dict(memory_allocated=lambda device=None: 0, max_memory_allocated=lambda device=None: 0,
                           reset_peak_memory_stats=lambda device=None: None, empty_cache=lambda: None)
    with mock.patch.object(cgr, "capture_into_graph", lambda fn, *a, **_: (_Replay(fn), fn())), \
            mock.patch.multiple(torch.accelerator, **no_memory_stats):
        runner.prepare_for_capture()
        runner.warmup_and_capture(None)
    assert runner.any_graphs and not runner.dropped_shapes
    runner.calls = 0
    run = runner.run

    def counted(*args, **kwargs):
        runner.calls += 1
        return run(*args, **kwargs)

    runner.run = counted
    return runner


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


def _small_perception(**featurizer):
    """A real Perception, shrunk, with random weights: the streamer runs its modules."""
    from mstar.model.nemotron_duplex.components.conformer import Perception
    from mstar.model.nemotron_duplex.config import ConformerSTTConfig

    torch.manual_seed(0)
    stt = ConformerSTTConfig(d_model=64, num_heads=4, num_layers=3, n_mels=32, feat_in=32,
                             subsampling_conv_channels=8, output_dim=48, att_context_size=(6, 0), **featurizer)
    p = Perception(stt)
    with torch.no_grad():
        for t in p.parameters():
            t.copy_(torch.randn_like(t) * 0.2)
        p.preprocessor.featurizer.window.copy_(torch.hann_window(stt.win_length, periodic=False))
        p.preprocessor.featurizer.fb.copy_(torch.rand_like(p.preprocessor.featurizer.fb))
    return p.eval()


class _Encoder:
    """The encoder node with its resources, driven as the engine drives an eager
    step: declare, admit and plan them, preprocess, forward, commit. The bounded
    KV runs its torch path on CPU, in fp32 (the engine's KV dtype here)."""

    def __init__(self, **featurizer):
        p = _small_perception(**featurizer)
        self.node = ConformerEncoderSubmodule(p, nn.Identity(), nn.Identity(), config=NemotronDuplexConfig())
        def pool(key, num_layers, blocks):
            spec = RecurrentStateSpec(key, {"conformer_encoder"},
                                      RecurrentStateConfig(num_layers=num_layers, blocks=blocks, max_slots=8))
            return RecurrentStatePool.build(spec, EngineResourceInfo(device=torch.device("cpu")))

        self.pool = pool(ENCODER_STATE, p.config.num_layers, encoder_state_blocks(p.config, torch.float32))
        self.front = pool(ENCODER_FRONT, 1, front_state_blocks(p.config))
        self.kv = BoundedKVManager.build(
            BoundedKVSpec(ENCODER_KV, {"conformer_encoder"}, encoder_kv_config(p.config, max_slots=8)),
            EngineResourceInfo(device=torch.device("cpu"), kv_dtype=torch.float32),
        )
        self.resources = {ENCODER_STATE: self.pool, ENCODER_FRONT: self.front, ENCODER_KV: self.kv}
        self.runner = StepRunner(self.resources)
        self.regions = {}       # the piecewise runners, captured on CPU, by label, when a test sets them

    def prepare(self, handle, chunk, final=False):
        inputs = {"audio_in": [chunk]} if chunk is not None else {"audio_in": []}
        return self.node.prepare_inputs("encode", SimpleNamespace(rid_handle=handle), inputs,
                                        is_final_stream_chunk=final)

    def declare(self, rids, inputs):
        """The node's declaration, with the leases the engine takes for it first."""
        from mstar.engine.engine import Engine

        engine = Engine.__new__(Engine)
        engine._submodules = {"conformer_encoder": SimpleNamespace(piecewise_runners=self.regions)}
        ctx = StepContext(request_ids=tuple(rids), graph_walk="encode", slot=0, capture=False)
        engine._maybe_lease_piecewise_regions("conformer_encoder", ctx, inputs)
        return self.node.declare_step("encode", rids, inputs, piecewise_leases=ctx.piecewise_leases)

    def step(self, rids, inputs) -> BatchedModelOutput:
        """One engine step over ``rids``: batched when the node can batch them,
        else each call's own step, as the engine's per-request path runs it."""
        if len(rids) > 1 and not self.node.can_batch(None, inputs):
            out = BatchedModelOutput(per_rid_outputs={})
            for rid, inp in zip(rids, inputs, strict=True):
                out.per_rid_outputs.update(self._step([rid], [inp]).per_rid_outputs)
            return out
        return self._step(rids, inputs)

    def _step(self, rids, inputs) -> BatchedModelOutput:
        for rid in rids:
            for resource in self.resources.values():
                resource.ingest_request(rid)
        step = self.declare(rids, inputs)
        if step is not None:
            step.set_ctx(StepContext(request_ids=tuple(rids), graph_walk="encode", slot=0, capture=False))
            assert self.runner.admit(step).ok
            self.runner.plan(step)
        engine_inputs = ModelInputsFromEngine(
            request_ids=list(rids), per_request_info={}, resources=dict(self.resources),
            piecewise_runners=dict(self.regions), step=step,
        )
        with torch.no_grad():
            out = self.node.forward_batched("encode", engine_inputs,
                                            **self.node.preprocess("encode", engine_inputs, inputs))
        if step is not None:
            self.runner.commit(step)
        return out

    def encode(self, handle, chunk, final=False):
        """One step for one call: what the stream hands the node, then its frames."""
        return self.step([handle], [self.prepare(handle, chunk, final)]).per_rid_outputs[handle]["audio_frame"]

    def offline(self, wav):
        with torch.no_grad():
            return self.node.perception(wav.unsqueeze(0), torch.tensor([wav.numel()]))[0][0]


def _encoder_regions(enc: _Encoder, *labels) -> dict:
    """The encoder's piecewise regions as the engine runs them, captured on CPU."""
    configs = enc.node.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.float32)
    return {label: _captured_on_cpu(configs[label], label, enc.resources, enc.runner) for label in labels}


def _encode_batch(enc, steps):
    """One batched encoder step, as the engine runs it: each ``(handle, chunk,
    final)`` row through prepare_inputs, then one step over all of them.
    Returns each row's frames by handle."""
    out = enc.step([h for h, _c, _f in steps], [enc.prepare(h, c, f) for h, c, f in steps]).per_rid_outputs
    return {h: out[h]["audio_frame"] for h, _c, _f in steps}


def _stream(enc, wavs, sizes):
    """Stream each call in its own chunk size, every call with audio left in
    each step, then end them all; the frames each call got."""
    got = {h: [] for h in wavs}
    pos = dict.fromkeys(wavs, 0)
    while any(pos[h] < wavs[h].numel() for h in wavs):
        rids, inputs = [], []
        for h in wavs:
            if pos[h] < wavs[h].numel():
                inputs.append(enc.prepare(h, wavs[h][pos[h]:pos[h] + sizes[h]]))
                pos[h] += sizes[h]
                rids.append(h)
        out = enc.step(rids, inputs).per_rid_outputs
        for h in rids:
            got[h] += out[h]["audio_frame"]
    for h in wavs:
        got[h] += enc.encode(h, None, final=True)
    return got


@pytest.mark.parametrize("end_rides_last_chunk", [False, True])
@pytest.mark.parametrize("n", [16000 * 2 + 777, 16000 * 2])
def test_encoder_streams_the_offline_frames(n, end_rides_last_chunk):
    """Chunks of any size, smaller than a frame included, then the end of input,
    on an empty final chunk or on the last chunk of audio (as an upload's usually
    is): the frames emitted in order are the offline encoding of the call, and
    the mel frames the offline featurizer's, the reflected right edge included
    (a length on the hop grid is the one whose last frame reads it), the newest
    of which the front end holds."""
    enc = _Encoder()
    wav = torch.randn(n) * 0.1
    want = enc.offline(wav)
    with torch.no_grad():
        mel = enc.node.perception.preprocessor(wav.unsqueeze(0), torch.tensor([n]))[0][0].T
    for handle, size in enumerate((320, 1280, 2560, 9000, 64000)):
        got = []
        for at in range(0, wav.numel(), size):
            last = at + size >= wav.numel()
            got += enc.encode(handle, wav[at:at + size], final=last and end_rides_last_chunk)
        if not end_rides_last_chunk:
            got += enc.encode(handle, None, final=True)
        assert len(got) == want.shape[0]
        assert torch.allclose(torch.stack(got), want, atol=1e-5), size
        st = enc.node.request_state(handle).get(enc.node.STATE_KEY)
        assert st.mels == mel.shape[0]
        held = enc.front.block("mel", 0)[enc.front._slots[handle]["main"].index]   # the newest, right-aligned
        assert torch.allclose(held, mel[-held.shape[0]:], atol=1e-5), size


def test_encoder_batches_calls_at_different_points():
    """Calls in one step, each with its own chunking (one frame a step, several,
    or none), each get their own offline frames."""
    enc = _Encoder()
    wavs = {1: torch.randn(16000 * 2) * 0.1, 2: torch.randn(16000 * 3) * 0.1, 3: torch.randn(9000) * 0.1}
    got = _stream(enc, wavs, {1: 1280, 2: 4000, 3: 320})
    for h, wav in wavs.items():
        assert torch.allclose(torch.stack(got[h]), enc.offline(wav), atol=1e-5), h


def test_encoder_live_steps_replay_the_regions_with_padding_rows():
    """Calls one frame of audio further on take the whole-step region; the
    others in the same step (a call's first frames, one under a frame a step)
    the eager front end and the layer region, both in buckets wider than the
    batch. Every call gets its offline frames: the padding rows, which address
    the pools' sinks, and the rows another path served leave them alone."""
    enc = _Encoder()
    enc.regions = _encoder_regions(enc, ConformerEncoderSubmodule.STEP_REGION, ConformerEncoderSubmodule.LAYERS_REGION)
    step, layers = enc.regions.values()
    wavs = {1: torch.randn(16000 * 2) * 0.1, 2: torch.randn(16000) * 0.1, 3: torch.randn(6000) * 0.1}
    got = _stream(enc, wavs, {1: 1280, 2: 1280, 3: 640})
    assert step.calls >= 20 and layers.calls >= 5
    for h, wav in wavs.items():
        assert torch.allclose(torch.stack(got[h]), enc.offline(wav), atol=1e-5), h


def test_encoder_layer_region_runs_only_the_calls_with_a_frame():
    """Calls one frame on take the whole-step region; the layer region then runs
    only the other calls that got a frame, not every call of the step."""
    enc = _Encoder()
    enc.regions = _encoder_regions(enc, ConformerEncoderSubmodule.STEP_REGION, ConformerEncoderSubmodule.LAYERS_REGION)
    layers = enc.regions[ConformerEncoderSubmodule.LAYERS_REGION]
    rows, run, step = [], layers.run, enc.step
    layers.run = lambda *a, request_ids, **k: rows.append(len(request_ids)) or run(*a, request_ids=request_ids, **k)
    enc.step = lambda rids, inputs: rows.append(f"step of {len(rids)}") or step(rids, inputs)
    wavs = {1: torch.randn(16000) * 0.1, 2: torch.randn(16000) * 0.1, 3: torch.randn(16000) * 0.1}
    got = _stream(enc, wavs, {1: 1280, 2: 1280, 3: 640})
    # once calls 1 and 2 are one frame on, the layer region runs call 3 alone
    assert rows[4:10] == ["step of 3", 1, "step of 3", "step of 3", 1, "step of 3"], rows
    for h, wav in wavs.items():
        assert torch.allclose(torch.stack(got[h]), enc.offline(wav), atol=1e-5), h


def test_encoder_paths_read_the_plan_current_when_they_use_it(monkeypatch):
    """The KV attends under a plan of exactly the rows it is given, padding
    included, with the eager path on this step's plan before a region plans its
    own (the pools' gather / scatter_ read the plan the pool holds when they
    run). A call with a chunk over a second runs in a step of its own."""
    enc = _Encoder()
    enc.regions = _encoder_regions(enc, ConformerEncoderSubmodule.STEP_REGION, ConformerEncoderSubmodule.LAYERS_REGION)
    attend = enc.kv.attend

    def attend_checked(layer_idx, q, *args, **kwargs):
        assert len(enc.kv.current.rows) == q.shape[0], "the KV attended under a plan for other rows"
        return attend(layer_idx, q, *args, **kwargs)

    monkeypatch.setattr(enc.kv, "attend", attend_checked)
    wavs = {1: torch.randn(16000 * 2) * 0.1, 2: torch.randn(16000) * 0.1, 3: torch.randn(6000) * 0.1,
            4: torch.randn(16000 * 3) * 0.1}
    got = _stream(enc, wavs, {1: 1280, 2: 1280, 3: 640, 4: 20000})
    assert all(region.calls for region in enc.regions.values())
    for h, wav in wavs.items():
        assert torch.allclose(torch.stack(got[h]), enc.offline(wav), atol=1e-5), h


def _kv_spans(step) -> list[int]:
    return [seg.span for seg in step.get(ENCODER_KV).segments]


def test_encoder_an_all_steady_step_leaves_its_resources_to_its_region():
    """When every call is one frame on and the whole-step region is captured
    for the batch, the region's own step owns the resources: the node declares
    nothing, so no second plan is built and thrown away. Otherwise it declares
    every call in each, and in the KV its new frames at its position."""
    enc = _Encoder()
    wavs = {1: torch.randn(1280 * 6) * 0.1, 2: torch.randn(1280 * 6) * 0.1}
    for at in (0, 1280):
        enc.step([1, 2], [enc.prepare(h, wavs[h][at:at + 1280]) for h in (1, 2)])
    nxt = [enc.prepare(h, wavs[h][2560:3840]) for h in (1, 2)]
    step = enc.declare([1, 2], nxt)                                           # no region captured
    assert set(step.keys()) == {ENCODER_FRONT, ENCODER_STATE, ENCODER_KV}
    assert _kv_spans(step) == [1, 1]
    states = [enc.node.request_state(h).get(enc.node.STATE_KEY) for h in (1, 2)]
    assert [step.get(ENCODER_KV).positions[h].written for h in (1, 2)] == [st.frames for st in states]
    # the call's own position, built when its frames last moved, not one per declaration
    assert all(step.get(ENCODER_KV).positions[h] is st.position for h, st in zip((1, 2), states, strict=True))
    enc.regions = _encoder_regions(enc, ConformerEncoderSubmodule.STEP_REGION)
    assert enc.declare([1, 2], nxt) is None
    for at in range(2560, 1280 * 6, 1280):
        out = enc.step([1, 2], [enc.prepare(h, wavs[h][at:at + 1280]) for h in (1, 2)]).per_rid_outputs
        assert all(len(out[h]["audio_frame"]) == 1 for h in (1, 2))
    assert enc.regions[ConformerEncoderSubmodule.STEP_REGION].calls == 4
    mixed = [enc.prepare(1, torch.randn(640) * 0.1), enc.prepare(2, torch.randn(1280) * 0.1)]
    step = enc.declare([1, 2], mixed)
    assert set(step.keys()) == {ENCODER_FRONT, ENCODER_STATE, ENCODER_KV}
    assert _kv_spans(step) == [mixed[0].kwargs["n"], 1]


def test_encoder_a_step_mixes_a_steady_call_with_eager_chunks_past_the_window():
    """One step holds a call one frame on, a call whose chunk adds more frames
    than the attention window (the eager layers, a step of T > 1 where each
    frame sees only the window before it, which the call one frame on then
    joins) and a call under a frame a step: each gets its offline frames."""
    enc = _Encoder()
    enc.regions = _encoder_regions(enc, ConformerEncoderSubmodule.STEP_REGION, ConformerEncoderSubmodule.LAYERS_REGION)
    window = enc.node.perception.config.att_context_size[0]
    wavs = {1: torch.randn(16000 * 3) * 0.1, 2: torch.randn(16000 * 3) * 0.1, 3: torch.randn(16000) * 0.1}
    sizes = {1: 1280, 2: 16000, 3: 640}
    assert 16000 // 1280 > window
    widths, front = [], enc.node.streamer.front

    def spy_front(*args):
        enc_in = front(*args)
        widths.append(0 if enc_in is None else enc_in.x.shape[1])
        return enc_in

    enc.node.streamer.front = spy_front
    got = _stream(enc, wavs, sizes)
    assert enc.regions[ConformerEncoderSubmodule.STEP_REGION].calls > 0 and max(widths) > window
    for h, wav in wavs.items():
        assert torch.allclose(torch.stack(got[h]), enc.offline(wav), atol=1e-5), h


def test_encoder_a_steady_call_joins_the_eager_layers_of_its_step():
    """A call one frame on whose window has wrapped, in a step where another
    call's chunk makes several frames: this step's KV plan, which the eager
    layers run on, writes both calls' frames, so the steady call can't also
    take the whole-step region (whose write of its frame would follow one from
    padding into the ring entry the region reads)."""
    enc = _Encoder()
    sub = ConformerEncoderSubmodule
    enc.regions = _encoder_regions(enc, sub.STEP_REGION, sub.LAYERS_REGION)
    window = enc.node.perception.config.att_context_size[0]
    at = 1280 * (window + 4)
    wavs = {1: torch.randn(at + 1280 * 3) * 0.1, 2: torch.randn(9000) * 0.1}
    got = {1: [], 2: []}
    for pos in range(0, at, 1280):
        got[1] += enc.encode(1, wavs[1][pos:pos + 1280])
    out = _encode_batch(enc, [(1, wavs[1][at:at + 1280], False), (2, wavs[2], False)])
    assert len(out[2]) > 1
    for h in (1, 2):
        got[h] += out[h]
    got[1] += enc.encode(1, wavs[1][at + 1280:], final=True)
    got[2] += enc.encode(2, None, final=True)
    for h in (1, 2):
        assert torch.allclose(torch.stack(got[h]), enc.offline(wavs[h]), atol=1e-5), h


def test_encoder_runs_a_steady_call_eagerly_when_its_region_has_no_graph():
    """The whole-step region leases the batch's bucket, but replays only the
    steady calls: when the runner dropped the capture for that sub-batch (here
    one call's), they take the eager path, which this step's plans cover."""
    enc = _Encoder()
    sub = ConformerEncoderSubmodule
    enc.regions = _encoder_regions(enc, sub.STEP_REGION, sub.LAYERS_REGION)
    step_region = enc.regions[sub.STEP_REGION]
    step_region._graphs = {k: v for k, v in step_region._graphs.items() if k.bs != 1}
    wavs = {1: torch.randn(1280 * 6) * 0.1, 2: torch.randn(1280 * 6) * 0.1}
    got = {1: [], 2: []}
    for at in (0, 1280, 2560):
        for h, frames in _encode_batch(enc, [(h, wavs[h][at:at + 1280], False) for h in (1, 2)]).items():
            got[h] += frames
    # call 1 is one frame on (steady), call 2 appends half a frame
    out = _encode_batch(enc, [(1, wavs[1][3840:5120], False), (2, wavs[2][3840:4480], False)])
    for h in (1, 2):
        got[h] += out[h]
    for h, end in ((1, 5120), (2, 4480)):
        got[h] += enc.encode(h, wavs[h][end:], final=True)
        assert torch.allclose(torch.stack(got[h]), enc.offline(wavs[h]), atol=1e-5), h


@pytest.mark.parametrize("b_end_rides", [False, True])
@pytest.mark.parametrize("a_chunk", [1280, 9000])
@pytest.mark.parametrize("b_len", [16000 * 2, 16000 * 2 + 640, 16000 * 2 + 777])
def test_encoder_a_calls_last_frames_do_not_depend_on_its_batch(b_len, a_chunk, b_end_rides):
    """Call B ends in the same step as call A's longer chunk, A still mid-call:
    B's last frames, which read past its audio, are still its offline ones."""
    enc = _Encoder()
    starts = list(range(0, b_len, 1280))
    steps = len(starts) + (0 if b_end_rides else 1)
    a, b = torch.randn(a_chunk * steps) * 0.1, torch.randn(b_len) * 0.1
    got_b = []
    for k, pb in enumerate(starts):
        b_final = b_end_rides and k == len(starts) - 1
        rows = [(1, a[k * a_chunk:(k + 1) * a_chunk], False), (2, b[pb:pb + 1280], b_final)]
        got_b += _encode_batch(enc, rows)[2]
    if not b_end_rides:
        got_b += _encode_batch(enc, [(1, a[-a_chunk:], False), (2, None, True)])[2]
    want = enc.offline(b)
    assert len(got_b) == want.shape[0]
    assert torch.allclose(torch.stack(got_b), want, atol=1e-5)


def test_encoder_keeps_a_chunk_over_a_second_out_of_a_batch():
    """A pass pads every call to its longest chunk, so a step holding an upload
    or a large append is not batched: can_batch refuses it, by the chunk length
    prepare_inputs reports, and the engine runs each call's step on its own.
    Calls with a second or less share one pass. The frames are the offline
    ones either way."""
    enc = _Encoder()
    streamer, fronts = enc.node.streamer, []
    front = streamer.front

    def spy_front(sts, chunks, final, *rest):
        fronts.append(sorted(c.numel() for c in chunks if c is not None))
        return front(sts, chunks, final, *rest)

    streamer.front = spy_front
    wav = {1: torch.randn(16000 * 5) * 0.1, 2: torch.randn(1280) * 0.1, 3: torch.randn(16001) * 0.1,
           4: torch.randn(16000) * 0.1}
    prepared = {h: enc.prepare(h, w, final=True) for h, w in wav.items()}
    assert {h: inp.input_seq_len for h, inp in prepared.items()} == {h: w.numel() for h, w in wav.items()}
    assert not enc.node.can_batch(None, list(prepared.values()))
    assert not enc.node.can_batch(None, [prepared[2], prepared[3]])
    assert enc.node.can_batch(None, [prepared[2], prepared[4]])
    assert enc.node.can_batch(None, [prepared[1]])                       # alone, it is its own step

    out = _encode_batch(enc, [(h, w, True) for h, w in wav.items()])
    assert sorted(fronts) == [[1280], [16000], [16001], [16000 * 5]]    # every call on its own
    for h, w in wav.items():
        assert torch.allclose(torch.stack(out[h]), enc.offline(w), atol=1e-5)

    fronts.clear()
    short = {5: torch.randn(1280) * 0.1, 6: torch.randn(16000) * 0.1}
    out = _encode_batch(enc, [(h, w, True) for h, w in short.items()])
    assert fronts == [[1280, 16000]]                                      # one pass
    for h, w in short.items():
        assert torch.allclose(torch.stack(out[h]), enc.offline(w), atol=1e-5)


def test_encoder_takes_no_batch_wider_than_its_graphs():
    """Its graphs are piecewise regions, which the engine's batch cap does not
    see: a wider step would run every call eagerly, so the node caps its batch
    at its widest capture and the scheduler splits the rest."""
    from mstar.engine.engine import Engine

    node = ConformerEncoderSubmodule(nn.Identity(), nn.Identity(), nn.Identity(), config=NemotronDuplexConfig())
    engine = Engine.__new__(Engine)
    engine._submodules = {"conformer_encoder": SimpleNamespace(submodule=node, accelerator_graph_runner=None,
                                                               capture_runner=None)}
    assert engine.get_max_batch_size("conformer_encoder", "encode") == max(node.CAPTURE_BATCH_SIZES)


@pytest.mark.parametrize("n_fft", [512, 1024])
def test_encoder_steady_steps_match_offline_for_the_featurizer_they_get(n_fft):
    """The whole-step region's slices assume the featurizer's half window spans
    more than one hop and at most two (512 / 160); with any other the encoder
    serves every step through the general path, which takes any geometry."""
    enc = _Encoder(n_fft=n_fft)
    regions = _encoder_regions(enc, ConformerEncoderSubmodule.STEP_REGION) if n_fft == 512 else {}
    assert (ConformerEncoderSubmodule.STEP_REGION in enc.node.get_piecewise_accelerator_graph_configs(
        torch.device("cpu"), torch.float32)) == (n_fft == 512)
    enc.regions = regions
    wav = torch.randn(16000 * 2) * 0.1
    got = torch.stack(_stream(enc, {1: wav}, {1: 1280})[1])
    assert torch.allclose(got, enc.offline(wav), atol=1e-5)
    if regions:
        assert regions[ConformerEncoderSubmodule.STEP_REGION].calls > 0


@pytest.mark.parametrize("n_fft", [512, 256])
@pytest.mark.parametrize("n", [1580, 1728])
def test_encoder_ends_a_call_with_its_last_frame_when_the_end_adds_no_mel_frame(n_fft, n):
    """With a half window under a hop, the end of input can make a last encoder
    frame final without adding a mel frame: the call still gets it."""
    enc = _Encoder(n_fft=n_fft, window_size=n_fft / 16000)
    wav = torch.randn(n) * 0.1
    got = []
    for at in range(0, n, 320):
        got += enc.encode(1, wav[at:at + 320])
    got += enc.encode(1, None, final=True)
    want = enc.offline(wav)
    assert len(got) == want.shape[0]
    assert torch.allclose(torch.stack(got), want, atol=1e-5)


def test_encoder_decides_a_calls_steadiness_once_per_step(monkeypatch):
    """Whether a call takes the whole-step region is decided once, where its
    inputs are prepared; the declaration and the forward read that decision."""
    enc = _Encoder()
    enc.regions = _encoder_regions(enc, ConformerEncoderSubmodule.STEP_REGION)
    wavs = {h: torch.randn(1280 * 4) * 0.1 for h in (1, 2)}

    def decided_already(*args):
        raise AssertionError("steadiness decided again after prepare_inputs")

    for at in range(0, 1280 * 4, 1280):
        inputs = [enc.prepare(h, wavs[h][at:at + 1280]) for h in wavs]
        with monkeypatch.context() as m:
            m.setattr(enc.node.streamer, "is_steady", decided_already)
            enc.step(list(wavs), inputs)
    assert enc.regions[ConformerEncoderSubmodule.STEP_REGION].calls == 2


def test_encoder_moves_a_steps_chunks_to_the_device_in_one_copy():
    """Each call's chunk arrives on the host; one copy per call would each be a
    blocking transfer from pageable memory, so the step copies them as one."""
    from torch.profiler import ProfilerActivity, profile

    chunks = [torch.randn(1280), None, torch.randn(640), torch.randn(1280)]
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        moved = ConformerEncoderSubmodule._to_device(chunks, torch.device("meta"))
    copies = [e for e in prof.events() if e.name == "aten::_to_copy"]
    assert len(copies) == 1
    assert [None if c is None else (c.device.type, c.shape) for c in moved] == [
        ("meta", (1280,)), None, ("meta", (640,)), ("meta", (1280,))]


def test_encoder_step_region_replays_three_calls_with_a_padding_row():
    """Three calls one frame on replay the whole-step region in a bucket of
    four: the padding row addresses both pools' sinks and the calls get their
    offline frames."""
    enc = _Encoder()
    enc.regions = _encoder_regions(enc, ConformerEncoderSubmodule.STEP_REGION)
    region = enc.regions[ConformerEncoderSubmodule.STEP_REGION]
    rows, run = [], region.run
    region.run = lambda *a, request_ids, **k: rows.append(len(request_ids)) or run(*a, request_ids=request_ids, **k)
    wavs = {h: torch.randn(16000) * 0.1 for h in (1, 2, 3)}
    got = _stream(enc, wavs, dict.fromkeys(wavs, 1280))
    assert rows and set(rows) == {3}
    for h, wav in wavs.items():
        assert torch.allclose(torch.stack(got[h]), enc.offline(wav), atol=1e-5), h


def test_encoder_offers_nothing_to_the_stop_check():
    """The encoder has no check_stop and its stream ends with the client's: its
    frames are not copied to the host every step for one."""
    enc = _Encoder()
    out = enc.step([1], [enc.prepare(1, torch.randn(4000) * 0.1)])
    assert isinstance(out, BatchedModelOutput) and out.check_stop_buffers == {}
    assert len(out.per_rid_outputs[1]["audio_frame"]) > 0


def test_encoder_projects_the_relative_positions_once_per_layer():
    """The positions a query can see are fixed, so each layer's projection of
    them is computed once, not every step."""
    enc = _Encoder()
    calls = []
    for layer in enc.node.perception.encoder.layers:
        layer.self_attn.linear_pos.register_forward_hook(lambda *a: calls.append(1))
    wav = torch.randn(16000) * 0.1
    for at in range(0, wav.numel(), 1280):
        enc.encode(1, wav[at:at + 1280])
    assert len(calls) == len(enc.node.perception.encoder.layers)


def test_encoder_waits_for_enough_audio_to_featurize():
    enc = _Encoder()
    assert enc.encode(1, torch.zeros(10)) == []


@pytest.mark.parametrize("n", [257, 1000, 1279, 1281])
def test_encoder_a_short_call_gets_the_offline_frames(n):
    """Offline encodes any call longer than n_fft/2 samples, under one frame too."""
    enc = _Encoder()
    wav = torch.randn(n) * 0.1
    want = enc.offline(wav)
    for handle, size in enumerate((100, n)):
        got = []
        for at in range(0, n, size):
            got += enc.encode(handle, wav[at:at + size])
        got += enc.encode(handle, None, final=True)
        assert len(got) == want.shape[0]
        assert torch.allclose(torch.stack(got), want, atol=1e-5), size


@pytest.mark.parametrize("chunks", [[], [torch.zeros(256)], [torch.zeros(100), torch.zeros(156)]])
def test_encoder_fails_a_call_that_ends_without_enough_audio(chunks):
    """No audio, or no more than n_fft/2 samples (which offline can't featurize):
    the call fails rather than reply to nothing."""
    enc = _Encoder()
    for chunk in chunks:
        enc.encode(1, chunk)
    with pytest.raises(ValueError, match="ended with"):
        enc.encode(1, None, final=True)


def test_encoder_state_is_per_call_and_cleaned_up():
    enc = _Encoder()
    free = enc.pool.num_free_slots
    enc.encode(1, torch.randn(4000) * 0.1)
    enc.encode(2, torch.randn(4000) * 0.1)
    assert enc.pool.num_free_slots == enc.front.num_free_slots == free - 2
    assert set(enc.kv._requests) == {1, 2}
    enc.node.cleanup_request(1)
    for resource in enc.resources.values():
        resource.remove_request(1)
    assert 1 not in enc.node.request_states and 2 in enc.node.request_states
    assert enc.pool.num_free_slots == enc.front.num_free_slots == free - 1
    assert set(enc.kv._requests) == {2}


# -- no cuFFT in a captured region ---------------------------------------------


def _without_fft(config):
    """``config`` with its region raising on any FFT op: torch.stft and
    torch.fft reach the dispatcher as ``aten._fft_*``."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class _NoFFT(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert not func.__name__.startswith("_fft_"), f"{func} in a captured region"
            return func(*args, **(kwargs or {}))

    fn = config.capture_fn

    def region(call):
        with _NoFFT():
            return fn(call)

    config.capture_fn = region
    return config


def test_encoder_regions_run_no_fft():
    """A captured cuFFT plan lives in torch's plan cache, whose LRU eviction
    destroys it under the graph; so the regions take their STFT as a matmul.
    The eager path, which the guard leaves alone, keeps torch.stft."""
    enc = _Encoder()
    configs = enc.node.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.float32)
    labels = (ConformerEncoderSubmodule.STEP_REGION, ConformerEncoderSubmodule.LAYERS_REGION)
    enc.regions = {label: _captured_on_cpu(_without_fft(configs[label]), label, enc.resources, enc.runner)
                   for label in labels}
    wavs = {1: torch.randn(16000) * 0.1, 2: torch.randn(16000 + 640) * 0.1}
    got = _stream(enc, wavs, {1: 1280, 2: 640})
    assert all(enc.regions[label].calls for label in labels)
    for h, wav in wavs.items():
        assert torch.allclose(torch.stack(got[h]), enc.offline(wav), atol=1e-5), h


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize(("n_fft", "win", "samples"), [(512, 400, 1632), (512, 400, 1700), (512, 512, 2000),
                                                        (256, 101, 900)])
def test_stft_as_a_matmul_is_torch_stft(n_fft, win, samples, autocast):
    """The matmul STFT is torch.stft's to fp32 rounding, measured against
    float64; bf16 autocast leaves it in full precision."""
    from mstar.model.nemotron_duplex.components.dft import stft_as_real, stft_basis

    torch.manual_seed(0)
    x, window = torch.randn(3, samples), torch.hann_window(win, periodic=False)

    def stft(signal):
        return torch.view_as_real(torch.stft(signal, n_fft, hop_length=160, win_length=win, center=False,
                                             window=window.to(signal.dtype), return_complex=True))

    exact = stft(x.double())
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        got = stft_as_real(x, stft_basis(window, n_fft), n_fft, 160)
    assert got.dtype == torch.float32 and got.shape == exact.shape
    fp32_fft = (stft(x) - exact).abs().max()
    assert (got - exact).abs().max() <= fp32_fft + 1e-7 * exact.abs().max()


@pytest.mark.parametrize("n", [16, 15])
def test_irfft_as_a_matmul_is_torch_irfft(n):
    """The matmul inverse DFT is torch.fft.irfft's to fp32 rounding, measured
    against float64, imaginary DC and Nyquist parts ignored as irfft ignores them."""
    from mstar.model.nemotron_duplex.components.dft import irfft_dim1

    torch.manual_seed(0)
    spec = torch.randn(4, n // 2 + 1, 50, dtype=torch.complex64)
    exact = torch.fft.irfft(spec.to(torch.complex128), n=n, dim=-2)
    got = irfft_dim1(spec, n)
    assert got.dtype == torch.float32 and got.shape == exact.shape
    fp32_fft = (torch.fft.irfft(spec, n=n, dim=-2) - exact).abs().max()
    assert (got - exact).abs().max() <= fp32_fft + 1e-7 * exact.abs().max()
