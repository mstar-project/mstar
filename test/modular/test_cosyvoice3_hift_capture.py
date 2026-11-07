"""The CosyVoice3 streaming vocoder captured: one padded batch, each stream's tail and phase
carry in the recurrent state pool.

CPU tests use tiny random float64 weights and drive the real pool and the real
``PiecewiseAcceleratorGraphRunner``, with ``capture_into_graph`` replaced by one that "replays"
by running the region eagerly. The CUDA test captures for real.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mstar.engine import accelerator_graph_runner as cgr
from mstar.engine.accelerator_graph_backend import get_accelerator_graph_backend
from mstar.engine.accelerator_graph_runner import PiecewiseAcceleratorGraphRunner
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.runner import StepRunner
from mstar.model.cosyvoice3.components.hift import (
    CapturedStream,
    CausalHiFTGenerator,
    HiftStreamState,
    plan_stream_row,
    stream_vocode,
)
from mstar.model.cosyvoice3.config import HIFT_STATE, CausalHiFTConfig, CosyVoice3Config
from mstar.model.cosyvoice3.cosyvoice3_model import CosyVoice3Model, hift_state_specs
from mstar.model.cosyvoice3.submodules import HiFTSubmodule, vocode_region

CONTEXT = 12


@pytest.fixture(autouse=True)
def _threads():
    prior = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(prior)


def _config(**overrides) -> CosyVoice3Config:
    cfg = CosyVoice3Config(
        hift=CausalHiFTConfig(base_channels=16, f0_cond_channels=16, noise_seconds=12.0),
        hift_context_frames=CONTEXT,
        hift_capture_frame_buckets=(48, 96),
        hift_capture_batch_sizes=(1, 2, 4),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _hift(cfg: CosyVoice3Config, seed: int = 0) -> CausalHiFTGenerator:
    """Random weights whose F0 head predicts voiced frames, so the harmonics matter."""
    torch.manual_seed(seed)
    hift = CausalHiFTGenerator(cfg.hift)
    with torch.no_grad():
        for name, param in hift.named_parameters():
            if name.endswith("weight") and param.ndim >= 2:
                torch.nn.init.normal_(param, std=0.05)
            else:
                torch.nn.init.zeros_(param)
        hift.f0_predictor.classifier.bias.fill_(200.0)
        hift.f0_predictor.classifier.weight.mul_(20.0)
    return hift.double().eval()


def _mel(frames: int, seed: int, device="cpu", dtype=torch.float64) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(80, frames, generator=gen, dtype=torch.float64) * 0.5).to(device, dtype)


class _NoGraph:
    def __init__(self, run):
        self.replay = run


def _rig(cfg, hift, monkeypatch=None, device=torch.device("cpu"), dtype=torch.float64):
    """A node over its real pool, with every region's runner."""
    if monkeypatch is not None:
        monkeypatch.setattr(
            cgr, "capture_into_graph",
            lambda run, pool, dev, dtype, keep_graph=False, node_name=None, stream=None: (_NoGraph(run), run()),
        )
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
        for name in ("memory_allocated", "max_memory_allocated", "reset_peak_memory_stats", "empty_cache"):
            monkeypatch.setattr(torch.accelerator, name, lambda *args, **kwargs: 0)
    spec = hift_state_specs(cfg, dtype=dtype)[0]
    pool = build_resource(spec, EngineResourceInfo(device=device, kv_dtype=dtype))
    resources = {HIFT_STATE: pool}
    steps = StepRunner(resources, node_resources={"hift": [HIFT_STATE]})
    node = HiFTSubmodule(hift, cfg)
    node.bind_node_resources(resources)
    runners = {}
    memory_pool = None if monkeypatch is not None else get_accelerator_graph_backend(device).graph_pool_handle()
    for label, config in node.get_piecewise_accelerator_graph_configs(device, torch.float32).items():
        runner = PiecewiseAcceleratorGraphRunner(
            label=label, config=config, resources=resources, step_runner=steps, device=device,
            autocast_dtype=None, num_slots=1, memory_pool=memory_pool,
        )
        if monkeypatch is not None:
            runner._graph_backend = SimpleNamespace(synchronize=lambda: None)
            for shape in runner.prepare_for_capture():
                runner._capture_one(shape, 0, forwards=1)
        else:
            runner.warmup_and_capture(None)
            assert runner.any_graphs and not runner.dropped_shapes
        runners[label] = runner
    return SimpleNamespace(node=node, runners=runners, steps=steps, pool=pool)


def _open(rig, *rids):
    for rid in rids:
        rig.node.request_state(rid)
        rig.steps.ingest_request(rid, {})


def _spy(runners, monkeypatch) -> list[tuple[str, int]]:
    seen: list[tuple[str, int]] = []
    for label, runner in runners.items():
        real = runner.run

        def run(*args, _label=label, _real=real, **kwargs):
            seen.append((_label, kwargs["real_bs"]))
            return _real(*args, **kwargs)

        monkeypatch.setattr(runner, "run", run)
    return seen


def _eager(cfg, hift, mel, sizes) -> torch.Tensor:
    """The stream through ``stream_vocode``, as PCM16."""
    state = HiftStreamState(mel=mel.new_zeros(mel.shape[0], 0))
    pieces, at = [], 0
    for n in sizes:
        pieces.append(stream_vocode(hift, state, mel[:, at : at + n], at + n == mel.shape[-1], CONTEXT))
        at += n
    return (torch.cat(pieces).float().clamp(-1, 1) * 32767).to(torch.int16)


def _close(got, want, tol=2):
    assert got.shape == want.shape
    assert (got.long() - want.long()).abs().max() <= tol


# -- planning ------------------------------------------------------------------------------


def test_the_tail_holds_what_a_window_and_the_held_back_frames_need():
    cfg = _config()
    assert cfg.hift_tail_frames == CONTEXT + cfg.hift.f0_left_history + cfg.hift.stream_holdback_frames
    hift = CausalHiFTGenerator(cfg.hift)
    assert hift.f0_predictor.left_history == cfg.hift.f0_left_history
    assert hift.f0_predictor.look_right == cfg.hift.f0_look_right


def test_a_chunk_too_short_to_emit_waits_for_the_next():
    cfg = _config()
    stream = CapturedStream()
    tail = cfg.hift_tail_frames
    assert plan_stream_row(cfg.hift, stream, _mel(5, 0), False, CONTEXT, tail) is None
    row = plan_stream_row(cfg.hift, stream, _mel(20, 1), False, CONTEXT, tail)
    assert row.new_mel.shape[-1] == 25 and stream.pending is None


def test_a_finished_stream_plans_nothing():
    cfg = _config()
    stream = CapturedStream(done=True)
    assert plan_stream_row(cfg.hift, stream, _mel(5, 0), True, CONTEXT, cfg.hift_tail_frames) is None


# -- the node, through the real pool -------------------------------------------------------


@pytest.mark.parametrize("sizes", [[25, 40, 40, 15], [30, 30, 30, 30], [9, 9, 9, 93], [120]])
def test_a_captured_stream_equals_the_eager_one(monkeypatch, sizes):
    cfg = _config()
    hift = _hift(cfg)
    rig = _rig(cfg, hift, monkeypatch)
    _open(rig, 1)
    mel = _mel(120, 5)
    pieces, at = [], 0
    with torch.no_grad():
        for n in sizes:
            pieces += rig.node._run([1], [mel[:, at : at + n]], [at + n == 120], rig.runners)
            at += n
        want = _eager(cfg, hift, mel, sizes)
    _close(torch.cat(pieces), want)
    assert want.numel() == 120 * hift.upsample_factor


def test_rows_at_different_points_batch_without_touching_each_other(monkeypatch):
    cfg = _config()
    hift = _hift(cfg)
    rig = _rig(cfg, hift, monkeypatch)
    _open(rig, 1, 2)
    mels = {1: _mel(80, 8), 2: _mel(45, 9)}
    steps = [
        ([1, 2], [mels[1][:, :30], mels[2][:, :30]], [False, False]),
        ([1, 2], [mels[1][:, 30:60], mels[2][:, 30:]], [False, True]),
        ([1], [mels[1][:, 60:]], [True]),
    ]
    seen = _spy(rig.runners, monkeypatch)
    out: dict[int, list[torch.Tensor]] = {1: [], 2: []}
    with torch.no_grad():
        for rids, chunks, finals in steps:
            for rid, pcm in zip(rids, rig.node._run(rids, chunks, finals, rig.runners), strict=True):
                out[rid].append(pcm)
        _close(torch.cat(out[1]), _eager(cfg, hift, mels[1], [30, 30, 20]))
        _close(torch.cat(out[2]), _eager(cfg, hift, mels[2], [30, 15]))
    # the second step's first row runs 30 new frames behind 12 of context and 8 each of holdback and history
    assert seen == [(vocode_region(48), 2), (vocode_region(96), 2), (vocode_region(48), 1)]


def test_a_chunk_past_every_bucket_runs_as_several_steps(monkeypatch):
    cfg = _config()
    hift = _hift(cfg)
    rig = _rig(cfg, hift, monkeypatch)
    _open(rig, 3)
    seen = _spy(rig.runners, monkeypatch)
    mel = _mel(200, 10)
    with torch.no_grad():
        got = rig.node._run([3], [mel], [True], rig.runners)[0]
        want = _eager(cfg, hift, mel, [200])
    assert len(seen) > 1
    _close(got, want)


def test_a_step_with_nothing_to_emit_replays_nothing(monkeypatch):
    cfg = _config()
    rig = _rig(cfg, _hift(cfg), monkeypatch)
    _open(rig, 4)
    seen = _spy(rig.runners, monkeypatch)
    got = rig.node._run([4], [_mel(0, 11)], [False], rig.runners)[0]
    assert got.numel() == 0 and got.dtype == torch.int16 and seen == []


def test_a_replays_samples_outlive_the_next_replay(monkeypatch):
    cfg = _config()
    rig = _rig(cfg, _hift(cfg), monkeypatch)
    _open(rig, 5, 6)
    with torch.no_grad():
        first = rig.node._run([5], [_mel(30, 12)], [True], rig.runners)[0]
        kept = first.clone()
        rig.node._run([6], [_mel(30, 13)], [True], rig.runners)
    assert torch.equal(first, kept)


def test_a_released_slot_starts_the_next_stream_fresh(monkeypatch):
    cfg = _config()
    hift = _hift(cfg)
    rig = _rig(cfg, hift, monkeypatch)
    mel = _mel(60, 14)
    with torch.no_grad():
        _open(rig, 7)
        rig.node._run([7], [_mel(40, 15)], [False], rig.runners)
        rig.steps.remove_request(7)
        _open(rig, 8)
        got = rig.node._run([8], [mel], [True], rig.runners)[0]
        _close(got, _eager(cfg, hift, mel, [60]))


def test_no_regions_runs_eagerly_on_python_state():
    cfg = _config()
    hift = _hift(cfg)
    node = HiFTSubmodule(hift, cfg)
    node.request_state(9)
    mel = _mel(30, 16)
    with torch.no_grad():
        got = node._run([9], [mel], [True], {})[0]
        _close(got, _eager(cfg, hift, mel, [30]), tol=0)


def test_a_bucket_left_out_for_memory_says_so(monkeypatch):
    cfg = _config()
    rig = _rig(cfg, _hift(cfg), monkeypatch)
    _open(rig, 10)
    runners = {vocode_region(48): rig.runners[vocode_region(48)]}
    with pytest.raises(RuntimeError, match="left out for memory"):
        rig.node._run([10], [_mel(90, 17)], [False], runners)


# -- the declaration -----------------------------------------------------------------------


def test_regions_are_one_per_bucket_largest_first_within_the_node_cap():
    cfg = _config(hift_max_batch_size=2)
    node = HiFTSubmodule(_hift(cfg), cfg)
    regions = node.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.float32)
    assert list(regions) == [vocode_region(96), vocode_region(48)]
    assert all(region.capture_batch_sizes == [1, 2] for region in regions.values())


@pytest.mark.parametrize("overrides", [{"hift_capture_frame_buckets": ()}, {"hift_context_frames": 0}])
def test_what_cant_be_captured_declares_no_state(overrides):
    cfg = _config(**overrides)
    node = HiFTSubmodule(_hift(cfg), cfg)
    assert node.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.float32) == {}
    assert hift_state_specs(cfg) == []


def test_a_bucket_too_short_for_the_tail_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("mstar.model.cosyvoice3.cosyvoice3_model.resolve_snapshot", lambda *a: str(tmp_path))
    monkeypatch.setattr(CosyVoice3Model, "_build_text_tokenizer", lambda self: None)
    with pytest.raises(ValueError, match="hift_capture_frame_buckets"):
        CosyVoice3Model("x", hift_capture_frame_buckets=[64])


# -- real capture --------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU")
def test_a_captured_stream_on_cuda_equals_the_eager_one():
    device = torch.device("cuda", torch.cuda.current_device())
    cfg = _config()
    hift = _hift(cfg).float().to(device)
    hift.keep_f0_predictor_in_float64()
    rig = _rig(cfg, hift, device=device, dtype=torch.float32)
    mels = {
        1: _mel(70, 20, device, torch.float32),
        2: _mel(40, 21, device, torch.float32),
        3: _mel(25, 22, device, torch.float32),
    }
    _open(rig, 1, 2, 3)
    out: dict[int, list[torch.Tensor]] = {1: [], 2: [], 3: []}
    with torch.no_grad():
        # padded to batch 4, then batch 2, then 1, moving between the regions
        for rid, pcm in zip((1, 2, 3), rig.node._run(
            [1, 2, 3], [mels[1][:, :25], mels[2][:, :25], mels[3]], [False, False, True], rig.runners,
        ), strict=True):
            out[rid].append(pcm)
        for rid, pcm in zip((1, 2), rig.node._run(
            [1, 2], [mels[1][:, 25:55], mels[2][:, 25:]], [False, True], rig.runners,
        ), strict=True):
            out[rid].append(pcm)
        out[1] += rig.node._run([1], [mels[1][:, 55:]], [True], rig.runners)
        eager = {
            1: _eager(cfg, hift, mels[1], [25, 30, 15]),
            2: _eager(cfg, hift, mels[2], [25, 15]),
            3: _eager(cfg, hift, mels[3], [25]),
        }
    for rid, pieces in out.items():
        _close(torch.cat(pieces), eager[rid], tol=8)
