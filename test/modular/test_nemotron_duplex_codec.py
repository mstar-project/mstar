"""The duplex codec's stream: each call's decoder state lives in the
``codec_state`` recurrent pool, a step decodes only the call's new frame, and a
call's PCM equals the full decode of its frames. The real decoder at tiny
widths (its kernel, rates and block count, so its receptive field, are the
checkpoint's), fp32, CPU."""
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.recurrent import RecurrentStateConfig, RecurrentStateSpec
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.step import StepContext
from mstar.model.nemotron_duplex.components.audio_codec import AudioCodec, CodecStreamState, codec_state_blocks
from mstar.model.nemotron_duplex.config import CODEC_STATE, CodecConfig, NemotronDuplexConfig
from mstar.model.nemotron_duplex.submodules import AudioCodecDecoderSubmodule
from mstar.model.submodule_base import ModelInputsFromEngine
from test.modular.test_nemotron_duplex_submodules import _captured_on_cpu, _without_fft


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
    blocks = codec_state_blocks(real.config, dtype)
    state = CodecStreamState(
        tails=[torch.zeros(2, *blocks[f"stage{i}"].shape[1:], dtype=dtype) for i in range(3)
               for _ in range(real.config.num_blocks)],
        overlap=torch.zeros(2, *blocks["overlap"].shape),
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


class _Codec:
    """The codec node with its pool, driven as the engine drives a step:
    leases, declare, admit and plan, preprocess, forward, commit."""

    def __init__(self, region: bool, max_slots: int = 8):
        self.real = tiny_codec()
        cfg = NemotronDuplexConfig()
        cfg.eartts.num_quantizers = self.real.config.num_quantizers
        self.node = AudioCodecDecoderSubmodule(codec=self.real, config=cfg)
        spec = RecurrentStateSpec(CODEC_STATE, {"audio_codec"}, RecurrentStateConfig(
            num_layers=1, blocks=codec_state_blocks(self.real.config, torch.float32), max_slots=max_slots))
        self.pool = RecurrentStatePool.build(spec, EngineResourceInfo(device=torch.device("cpu")))
        self.resources = {CODEC_STATE: self.pool}
        self.runner = StepRunner(self.resources)
        self.regions = {}
        if region:
            label = AudioCodecDecoderSubmodule.DECODE_REGION
            config = self.node.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.float32)[label]
            self.regions[label] = _captured_on_cpu(_without_fft(config), label, self.resources, self.runner)

    def prepare(self, rid, frame, final=False):
        chunk = {"codec_tokens": [frame.view(1, -1)] if frame is not None else []}
        return self.node.prepare_inputs("codec_chunk", SimpleNamespace(rid_handle=rid), chunk,
                                        is_final_stream_chunk=final)

    def declare(self, rids, inputs):
        from mstar.engine.engine import Engine

        engine = Engine.__new__(Engine)
        engine._submodules = {"audio_codec": SimpleNamespace(piecewise_runners=self.regions)}
        ctx = StepContext(request_ids=tuple(rids), graph_walk="codec_chunk", slot=0, capture=False)
        engine._maybe_lease_piecewise_regions("audio_codec", ctx, inputs)
        return self.node.declare_step("codec_chunk", rids, inputs, piecewise_leases=ctx.piecewise_leases)

    def step(self, rids, inputs) -> dict:
        for rid in rids:
            self.pool.ingest_request(rid)
        step = self.declare(rids, inputs)
        if step is not None:
            step.set_ctx(StepContext(request_ids=tuple(rids), graph_walk="codec_chunk", slot=0, capture=False))
            assert self.runner.admit(step).ok
            self.runner.plan(step)
        engine_inputs = ModelInputsFromEngine(request_ids=list(rids), per_request_info={},
                                              resources=dict(self.resources),
                                              piecewise_runners=dict(self.regions), step=step)
        with torch.no_grad():
            out = self.node.forward_batched("codec_chunk", engine_inputs,
                                            **self.node.preprocess("codec_chunk", engine_inputs, inputs))
        if step is not None:
            self.runner.commit(step)
        return out

    def end(self, rid):
        """The engine's teardown of a finished call."""
        self.pool.remove_request(rid)
        self.node.cleanup_request(rid)

    def run(self, calls: dict, starts: dict, end_on_empty: dict) -> dict:
        """Each call's frames, one a step from its start step, its stream ending
        on an empty final chunk or on its last frame; every call with something
        to do in a step runs in that step. Each call's PCM, concatenated."""
        got = {rid: [] for rid in calls}
        last = max(starts[r] + len(c) + end_on_empty[r] for r, c in calls.items())
        for t in range(last):
            rids, inputs = [], []
            for rid, codes in calls.items():
                k = t - starts[rid]
                n = len(codes)
                if 0 <= k < n:
                    inputs.append(self.prepare(rid, codes[k], final=k == n - 1 and not end_on_empty[rid]))
                elif k == n and end_on_empty[rid]:
                    inputs.append(self.prepare(rid, None, final=True))
                else:
                    continue
                rids.append(rid)
            if rids:
                out = self.step(rids, inputs)
                for rid in rids:
                    got[rid] += out[rid]["audio_chunk"]
        return {rid: torch.cat(pieces) for rid, pieces in got.items()}


@pytest.mark.parametrize("region", [False, True])
def test_calls_at_different_points_stream_their_full_decode(region):
    """Three calls starting at different steps, of different lengths, batched
    whenever they overlap (with the region: in buckets of 1, 2, 4, so with
    padding rows), one ending on an empty final chunk and the others on their
    last frame: each call's PCM is the full decode of its frames."""
    codec = _Codec(region)
    calls = {"a": codes_for(codec.real, 23), "b": codes_for(codec.real, 17), "c": codes_for(codec.real, 9)}
    got = codec.run(calls, starts={"a": 0, "b": 2, "c": 5}, end_on_empty={"a": True, "b": False, "c": False})
    if region:
        assert codec.regions[AudioCodecDecoderSubmodule.DECODE_REGION].calls >= 15
    for rid, codes in calls.items():
        want = pcm(full_decode(codec.real, codes))
        assert got[rid].shape == want.shape, rid
        torch.testing.assert_close(got[rid], want, atol=1, rtol=0)   # int16: one quantization step


@pytest.mark.parametrize("region", [False, True])
def test_a_reused_slot_starts_from_zeros(region):
    """A call's slot goes back zeroed when the call ends, so the next call on it
    decodes from the causal zero padding, not from the last call's tails."""
    codec = _Codec(region, max_slots=2)                 # the sink and one slot
    first = codes_for(codec.real, 11)
    codec.run({"a": first}, starts={"a": 0}, end_on_empty={"a": False})
    slot = codec.pool._slots["a"]["main"].index
    codec.end("a")
    second = codes_for(codec.real, 7)
    got = codec.run({"b": second}, starts={"b": 0}, end_on_empty={"b": True})["b"]
    assert codec.pool._slots["b"]["main"].index == slot
    torch.testing.assert_close(got, pcm(full_decode(codec.real, second)), atol=1, rtol=0)


def test_a_first_frame_reserves_its_slot_and_later_steps_leave_the_pool_to_the_region():
    """A call's first frame is declared by the node, so a full pool refuses the
    step at admit (backpressure) instead of failing in the region; once every
    call in a step holds a slot, the region's own step owns the pool."""
    codec = _Codec(region=True)
    a, b = codes_for(codec.real, 4), codes_for(codec.real, 4)
    assert set(codec.declare(["a"], [codec.prepare("a", a[0])]).keys()) == {CODEC_STATE}
    codec.step(["a"], [codec.prepare("a", a[0])])
    assert codec.declare(["a"], [codec.prepare("a", a[1])]) is None
    mixed = [codec.prepare("a", a[1]), codec.prepare("b", b[0])]
    assert set(codec.declare(["a", "b"], mixed).keys()) == {CODEC_STATE}
