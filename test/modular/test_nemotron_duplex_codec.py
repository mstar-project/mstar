"""The duplex codec's stream: each call's decoder state is the ``codec_state``
Value, the talker's step decodes only the call's new frame, and a call's PCM
equals the full decode of its frames. The real decoder at tiny widths (its
kernel, rates and block count, so its receptive field, are the checkpoint's),
fp32, CPU."""
from types import SimpleNamespace

import pytest
import torch
from test_nemotron_duplex_talker_batch import CHAR_PAD, S2C, TINY, make_talker

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.step import BucketKey, SlotLease, StepContext, SubmoduleStep
from mstar.engine.state import Entry, TensorType, Value
from mstar.engine.state.adapters import StateSetup, _resolve, build_handle, build_specs, state_steps
from mstar.model.nemotron_duplex.components.audio_codec import AudioCodec, CodecStreamState, codec_entry
from mstar.model.nemotron_duplex.config import CODEC_STATE, TALKER_KV, TALKER_PREV, CodecConfig, NemotronDuplexConfig
from mstar.model.nemotron_duplex.submodules import EarTTSTalkerSubmodule
from mstar.model.submodule_base import ModelInputsFromEngine


def tiny_codec() -> AudioCodec:
    torch.manual_seed(0)
    real = AudioCodec(CodecConfig(base_hidden_size=8, latent_size=16, codebook_size=16, num_quantizers=4)).eval()
    for prm in real.parameters():
        if prm.is_floating_point():
            prm.data.normal_(0, 0.05)
    return real


def pcm(wav: torch.Tensor) -> torch.Tensor:
    return (wav.clamp(-1, 1) * 32767).to(torch.int16)


def full_decode(real: AudioCodec, codes: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return real.decode(codes.unsqueeze(0), by_matmul=True)[0][0, 0]


def codes_for(real: AudioCodec, frames: int) -> torch.Tensor:
    return torch.randint(0, real.config.codebook_size, (frames, real.config.num_quantizers))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("per_step", [1, 2, 3])
def test_decode_stream_is_the_full_decode(per_step, dtype):
    """From zeroed state (the causal zero padding), a stream of steps of any
    size gives the full decode's samples: each step's final samples (less the
    overlap-add's trimmed left edge at the first), then the last step's right
    edge. In fp32 to rounding; in bf16 (the served dtype; the overlap-add stays
    fp32) to one int16 step."""
    real = tiny_codec().to(dtype)
    codes = torch.stack([codes_for(real, 23), codes_for(real, 23)])
    tensors = codec_entry(real.config).tensors
    state = CodecStreamState(
        tails=[torch.zeros(2, *tensors[f"stage{i}"].shape[1:], dtype=dtype) for i in range(3)
               for _ in range(real.config.num_blocks)],
        overlap=torch.zeros(2, *tensors["overlap"].shape),
    )
    pieces = []
    with torch.no_grad():
        for at in range(0, 23, per_step):
            audio, state = real.decode_stream(codes[:, at:at + per_step], state)
            pieces.append(audio.final[:, real.stream_edge:] if at == 0 else audio.final)
    stream = torch.cat([*pieces, audio.end_after], dim=1)
    for row in range(2):
        whole = full_decode(real, codes[row]).float()
        assert stream[row].shape == whole.shape
        if dtype == torch.float32:
            torch.testing.assert_close(stream[row], whole, atol=1e-6, rtol=0)
        torch.testing.assert_close(pcm(stream[row]), pcm(whole), atol=1, rtol=0)


class _Talker:
    """The talker node with its codec and previous-codes Values, its backbone and
    sampler stubbed to emit given codes, driven as the engine drives a step:
    declare, admit and plan both Values' steps, open the handles, forward,
    check, commit, postprocess. ``captured`` runs every step as a replay does:
    at the bucket's width, padding rows included, on the pools' per-bucket
    addressing."""

    BUCKETS = (1, 2, 4)

    def __init__(self, captured: bool, max_slots: int = 8):
        cfg = NemotronDuplexConfig()
        cfg.eartts = TINY
        self.real = tiny_codec()
        # the forward takes its noise from the step below, so the memo is never read
        self.node = EarTTSTalkerSubmodule(make_talker(), cfg, codec=self.real, subword_to_char=S2C,
                                          char_pad_idx=CHAR_PAD, noise_rows=1)
        self.node.talker.backbone_pooled = lambda x: x
        self.node.talker.generate_step = lambda hidden, **kwargs: self.codes.unsqueeze(1)
        arrays = [
            Value(TALKER_PREV, nodes={"eartts_talker"}, layers={0},
                  entry=Entry({"codes": TensorType((TINY.num_quantizers,), dtype=torch.long)})),
            Value(CODEC_STATE, nodes={"eartts_talker"}, layers={0}, entry=codec_entry(self.real.config)),
        ]
        info = EngineResourceInfo(device=torch.device("cpu"), kv_dtype=torch.float32)
        settings = {array.name: {"max_slots": max_slots} for array in arrays}
        self.pools = {spec.resource_key: RecurrentStatePool.build(spec, info)
                      for spec in build_specs(arrays, kernel_settings=settings)}
        self.pool = self.pools[CODEC_STATE]
        resolved = _resolve(arrays, StateSetup())
        self.handles = {array.name: build_handle(array, self.pools, resolved) for array in arrays}
        self.node.bind_node_state({TALKER_KV: None, **self.handles})
        self.runner = StepRunner(self.pools)
        self.captured = captured
        self.opened = set()  # calls whose state is open: once per request, as the worker does

    def step(self, rids: list, inputs: list, codes: torch.Tensor) -> dict:
        n = len(rids)
        ctx = StepContext(request_ids=tuple(rids), graph_walk="talker_decode", slot=0, capture=False)
        rows = list(rids)
        if self.captured:
            bs = next(b for b in self.BUCKETS if b >= n)
            rows += [f"pad{i}" for i in range(bs - n)]
            ctx.slot_lease = SlotLease(slot=0, bucket=BucketKey("talker_decode", bs, 2 * bs, "decode"))
            ctx.set_padded_rids(tuple(rows))
            codes = torch.cat([codes, torch.zeros(bs - n, codes.shape[1], dtype=codes.dtype)])
            inputs = inputs + [inputs[0]] * (bs - n)
        for rid in rids:
            if rid not in self.opened:
                self.opened.add(rid)
                for pool in self.pools.values():
                    pool.ingest_request(rid)
        declared = self.node.declare_step("talker_decode", rows, inputs)
        # the talker's KV has no pool here: the Values' steps alone
        step = SubmoduleStep(segments=declared.segments, state={
            name: declared.state[name] for name in (TALKER_PREV, CODEC_STATE)})
        step.set_ctx(ctx)
        step = state_steps(step, self.handles, graph_walk="talker_decode")
        step.set_ctx(ctx)
        assert self.runner.admit(step).ok
        self.runner.plan(step)
        self.codes = codes
        m, e = len(rows), TINY
        engine_inputs = ModelInputsFromEngine(request_ids=rows, per_request_info={}, resources={}, step=step)
        step.begin_calls()
        with torch.no_grad():
            out = self.node.forward_batched(
                "talker_decode", engine_inputs, x=torch.zeros(2 * m, e.hidden_size), spans=[1] * m,
                noise_u=torch.zeros(m, e.inference_num_iter, 1, e.mog_num_predictions),
                noise_eps=torch.zeros(m, e.inference_num_iter, 1, e.code_dim))
        step.check_calls()
        self.runner.commit(step)
        step.commit_calls()
        step.end_calls()
        got = {}
        for i, rid in enumerate(rids):
            # the engine's per-row views of the batch outputs (padding rows dropped)
            row = {name: [t[i:i + 1]] for name, t in out.row_outputs.items()}
            self.node.postprocess(rid, None, row, inputs=inputs[i])
            got[rid] = row
        return got

    def end(self, rid):
        """The engine's teardown of a finished call."""
        self.opened.discard(rid)
        for pool in self.pools.values():
            pool.remove_request(rid)
        self.node.cleanup_request(rid)

    def run(self, calls: dict, starts: dict, end_on_empty: dict) -> dict:
        """Each call's frames, one a step from its start step, its token stream
        ending on an empty final chunk (whose PAD frame decodes codes of its
        own) or on its last token; every call with something to do in a step
        runs in that step. Each call's PCM, concatenated."""
        got = {rid: [] for rid in calls}
        last = max(starts[r] + len(c) + end_on_empty[r] for r, c in calls.items())
        for t in range(last):
            rids, inputs, codes = [], [], []
            for rid, call in calls.items():
                k, n = t - starts[rid], len(call)
                info = SimpleNamespace(rid_handle=rid, dynamic_loop_iter_counts={"talker_decode_loop": k})
                if 0 <= k < n:
                    inputs.append(self.node.prepare_inputs("talker_decode", info, {"new_token": [torch.tensor([5])]},
                                                           is_final_stream_chunk=k == n - 1 and not end_on_empty[rid]))
                    codes.append(call[k])
                elif k == n and end_on_empty[rid]:
                    inputs.append(self.node.prepare_inputs("talker_decode", info, {}, is_final_stream_chunk=True))
                    codes.append(codes_for(self.real, 1)[0])
                else:
                    continue
                rids.append(rid)
            if rids:
                out = self.step(rids, inputs, torch.stack(codes))
                for rid in rids:
                    got[rid] += out[rid].get("audio_chunk", [])
        return {rid: torch.cat([p.reshape(-1) for p in pieces]) for rid, pieces in got.items()}


@pytest.mark.parametrize("captured", [False, True])
def test_calls_at_different_points_stream_their_full_decode(captured):
    """Three calls starting at different steps, of different lengths, batched
    whenever they overlap (captured: in buckets of 1, 2, 4, so with padding rows
    that address the pool's sink), one ending on an empty final chunk and the
    others on their last token: each call's PCM is the full decode of its frames."""
    talker = _Talker(captured)
    calls = {"a": codes_for(talker.real, 23), "b": codes_for(talker.real, 17), "c": codes_for(talker.real, 9)}
    got = talker.run(calls, starts={"a": 0, "b": 2, "c": 5}, end_on_empty={"a": True, "b": False, "c": True})
    for rid, codes in calls.items():
        want = pcm(full_decode(talker.real, codes))
        assert got[rid].shape == want.shape, rid
        torch.testing.assert_close(got[rid], want, atol=1, rtol=0)   # int16: one quantization step


@pytest.mark.parametrize("captured", [False, True])
def test_a_reused_slot_starts_from_zeros(captured):
    """A call's slot goes back zeroed when the call ends, so the next call on it
    decodes from the causal zero padding, not from the last call's tails."""
    talker = _Talker(captured, max_slots=2)                 # the sink and one slot
    first = codes_for(talker.real, 11)
    talker.run({"a": first}, starts={"a": 0}, end_on_empty={"a": True})
    slot = talker.pool._slots["a"]["main"].index
    talker.end("a")
    second = codes_for(talker.real, 7)
    got = talker.run({"b": second}, starts={"b": 0}, end_on_empty={"b": False})["b"]
    assert talker.pool._slots["b"]["main"].index == slot
    torch.testing.assert_close(got, pcm(full_decode(talker.real, second)), atol=1, rtol=0)


def test_a_call_with_no_frame_emits_nothing():
    """A token stream that ends before its first token reaches the talker runs
    one PAD frame to end the loop, and the call gets no audio, as offline."""
    talker = _Talker(captured=False)
    info = SimpleNamespace(rid_handle="a", dynamic_loop_iter_counts={})
    inp = talker.node.prepare_inputs("talker_decode", info, {}, is_final_stream_chunk=True)
    assert talker.step(["a"], [inp], codes_for(talker.real, 1))["a"] == {}


def test_the_codec_decode_runs_no_fft():
    """A captured cuFFT plan lives in torch's plan cache, whose LRU eviction
    destroys it under the graph; so the decode's inverse DFT is a matmul."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class _NoFFT(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert not func.__name__.startswith("_fft_"), f"{func} in the codec decode"
            return func(*args, **(kwargs or {}))

    talker = _Talker(captured=True)
    codes = codes_for(talker.real, 3)
    with _NoFFT():
        got = talker.run({"a": codes}, starts={"a": 0}, end_on_empty={"a": False})["a"]
    torch.testing.assert_close(got, pcm(full_decode(talker.real, codes)), atol=1, rtol=0)
