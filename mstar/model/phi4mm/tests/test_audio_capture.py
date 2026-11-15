"""The audio node's captured regions against its eager forward, on real
weights: steps mixing clip lengths, a clip past one window (its subsampling
past every mel bucket, so eager, its blocks captured), and clips on either
side of a bucket edge. A replay runs at the bucket's shapes, so it differs
from the eager step by bf16 rounding of other GEMM shapes only; replaying
the blocks without the step's key lengths, so a short clip attends its
padding, is the control that must diverge. Skipped without CUDA or the
cached checkpoint.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from mstar.model.phi4mm.audio import (
    AUDIO_EMBEDS,
    AUDIO_FEATURES,
    Phi4MMAudioEncoderSubmodule,
    audio_intake,
)
from mstar.model.phi4mm.components.audio import Phi4MMAudioConfig, Phi4MMAudioTower
from mstar.model.phi4mm.tests._paths import ORACLE, snapshot
from mstar.model.phi4mm.tests.test_audio_intake import _info
from mstar.model.submodule_base import ModelInputsFromEngine

SNAPSHOT = snapshot(full=True) if torch.cuda.is_available() else None
pytestmark = pytest.mark.skipif(SNAPSHOT is None, reason="needs CUDA and the cached checkpoint")

WALK = "prefill_audio"


def captured_node(batch_sizes: dict[str, list[int]] | None = None) -> SimpleNamespace:
    """The node on real bf16 weights with its regions captured by the
    engine's piecewise runner; ``batch_sizes`` narrows a region's ladder."""
    from mstar.engine.accelerator_graph_backend import get_accelerator_graph_backend
    from mstar.engine.accelerator_graph_runner import PiecewiseAcceleratorGraphRunner
    from mstar.engine.resources.runner import StepRunner

    device = torch.device("cuda", torch.cuda.current_device())
    config = Phi4MMAudioConfig.from_hf(json.loads((SNAPSHOT / "config.json").read_text()))
    node = Phi4MMAudioEncoderSubmodule(Phi4MMAudioTower.load(config, SNAPSHOT, device=device, dtype=torch.bfloat16))
    pool = get_accelerator_graph_backend(device).graph_pool_handle()
    runners = {}
    for label, region in node.get_piecewise_accelerator_graph_configs(device, torch.bfloat16).items():
        if batch_sizes is not None:
            if label not in batch_sizes:
                continue
            region.capture_batch_sizes = batch_sizes[label]
        runner = PiecewiseAcceleratorGraphRunner(
            label=label, config=region, resources={}, step_runner=StepRunner({}), device=device,
            autocast_dtype=None, num_slots=1, memory_pool=pool,
        )
        with torch.no_grad():
            runner.warmup_and_capture(None)
        assert runner.any_graphs and not runner.dropped_shapes
        runners[label] = runner
    return SimpleNamespace(node=node, runners=runners)


def clips_of(seconds: list[float], seed: int = 0) -> list[torch.Tensor]:
    """Clips ``seconds`` long: cut from the reference recordings' speech when
    ``PHI4MM_ORACLE`` is set, else a few tones under noise, whose bf16 error
    is several times speech's."""
    speech = None
    if ORACLE is not None:
        from mstar.model.components.audio_features import load_speech_file

        speech = load_speech_file(str(ORACLE / "media" / "speech_long.wav"), 16_000, 60.0)
    gen = torch.Generator().manual_seed(seed)
    out, at = [], 0
    for s in seconds:
        n = round(s * 16_000)
        if speech is not None and n <= speech.numel():
            at = at if at + n <= speech.numel() else 0
            wav, at = speech[at:at + n], at + n
        else:
            t = torch.arange(n) / 16_000
            tone = sum(torch.sin(2 * torch.pi * f * t) for f in (180.0, 420.0, 1300.0)) * 0.1
            wav = (tone + 0.05 * torch.randn(n, generator=gen)).to(torch.float32)
        out.append(audio_intake(wav, {"sample_rate": 16_000}).inputs[AUDIO_FEATURES])
    return out


def step(node, clips: list[torch.Tensor], runners: dict | None = None, edit_batch=None):
    """One step, a request per clip: its preprocess and forward, the regions
    replayed where ``runners`` hold a bucket. The rows of every request, the
    step's batch and its regions."""
    rids = [f"r{i}" for i in range(len(clips))]
    infos = {rid: _info(rid, WALK, [c.shape[0]]) for rid, c in zip(rids, clips, strict=True)}
    inputs = [node.prepare_inputs(WALK, infos[rid], {AUDIO_FEATURES: [c]}) for rid, c in zip(rids, clips, strict=True)]
    engine_inputs = ModelInputsFromEngine(request_ids=rids, per_request_info=infos, piecewise_runners=runners or {})
    with torch.no_grad():
        pre = node.preprocess(WALK, engine_inputs, inputs)
        if edit_batch is not None:
            pre["batch"] = edit_batch(pre["batch"])
        out = node.forward_batched(WALK, engine_inputs, **pre)
    rows = torch.cat([out[rid][AUDIO_EMBEDS][0] for rid in rids]).float().cpu()
    return SimpleNamespace(rows=rows, batch=pre["batch"], regions=pre["regions"])


def eager_on(node, batch) -> torch.Tensor:
    """The eager forward over a step's own (bucketed) batch."""
    with torch.no_grad():
        return node.tower.project(node.tower.encode(batch), "speech").float().cpu()


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


@pytest.fixture(scope="module")
def rig():
    rig = captured_node({
        "subsample_T2048": [1, 4],
        "subsample_T4096": [2],
        "blocks_W256": [4],
        "blocks_W500": [2, 4],
    })
    config = rig.node.config
    rig.fp32 = Phi4MMAudioEncoderSubmodule(
        Phi4MMAudioTower.load(config, SNAPSHOT, device=rig.node.get_device(), dtype=torch.float32),
    )
    return rig


@pytest.mark.parametrize(
    ("seconds", "regions"),
    [
        # 5-19 s: the 2048 mel bucket and the 256-frame window bucket, a batch of 4 for 3
        ([5.0, 12.3, 19.1], ("subsample_T2048", "blocks_W256")),
        # one clip padded to its buckets, unmasked when eager
        ([7.7], ("subsample_T2048", "blocks_W256")),
        # 46 s: past every mel bucket (eager), two windows at 500 (captured), unmasked
        ([46.07], (None, "blocks_W500")),
        # and with a short clip, masked
        ([46.07, 3.2], (None, "blocks_W500")),
        # 2048 mel frames exactly (256 encoder frames, the bucket's width) and one past it
        ([(2048 - 1) * 0.01 + 0.025, (2049 - 1) * 0.01 + 0.025], ("subsample_T4096", "blocks_W500")),
    ],
)
def test_replay_matches_eager(rig, seconds, regions):
    """A replay is the eager forward over the same bucketed batch, and as
    close to fp32 as the eager step at its own shape is: padding to a bucket
    changes bf16 rounding only."""
    clips = clips_of(seconds)
    live = step(rig.node, clips)
    replayed = step(rig.node, clips, rig.runners)
    fp32 = step(rig.fp32, clips).rows
    assert live.regions == (None, None) and tuple(replayed.regions) == regions
    same_shapes = rel(replayed.rows, eager_on(rig.node, replayed.batch))
    to_fp32, live_to_fp32 = rel(replayed.rows, fp32), rel(live.rows, fp32)
    print(f"\n{seconds} via {tuple(replayed.regions)}: replay vs eager on its batch {same_shapes:.1e}, "
          f"vs eager {rel(replayed.rows, live.rows):.1e}; vs fp32 replay {to_fp32:.1e}, eager {live_to_fp32:.1e}")
    assert replayed.rows.shape == live.rows.shape
    assert same_shapes < 1e-3
    assert to_fp32 < 1.5 * live_to_fp32 + 1e-2
    # eager steps run memory-efficient attention and captured ones cuDNN, which is
    # closer to fp32 at width 500 (~2e-2 against ~4.5e-2), so the two differ by a few %
    assert rel(replayed.rows, live.rows) < 6e-2


def _sdpa_ops(fn) -> set[str]:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        fn()
        torch.cuda.synchronize()
    return {e.key for e in prof.key_averages() if "dot_product" in e.key}


@pytest.mark.parametrize("seconds", [[46.07], [46.07, 3.2]], ids=["unmasked", "masked"])
def test_eager_blocks_run_memory_efficient_attention(rig, seconds):
    """An eager step's attention takes neither cuDNN, which builds a graph per new
    shape, nor math, which a non-contiguous bias would fall back to."""
    clips = clips_of(seconds)
    ops = _sdpa_ops(lambda: step(rig.node, clips))
    assert any("efficient_attention" in op for op in ops), ops
    assert not any("cudnn" in op or "math" in op for op in ops), ops


def test_a_replay_after_another_step_is_unchanged(rig):
    """No replay reads what the step before it left in the static buffers."""
    first, other = clips_of([5.0, 12.3, 19.1]), clips_of([7.7, 3.1], seed=1)
    a = step(rig.node, first, rig.runners)
    step(rig.node, other, rig.runners)
    again = step(rig.node, first, rig.runners)
    assert tuple(a.regions) == tuple(again.regions) == ("subsample_T2048", "blocks_W256")
    assert torch.equal(a.rows, again.rows)


def test_unmasked_replay_diverges(rig):
    """The control: the same replay with every window attending its whole
    bucket, padding included, moves the shorter clips far off."""
    clips = clips_of([5.0, 12.3, 19.1])
    live = step(rig.node, clips)

    def unmasked(batch):
        return batch._replace(key_lengths=torch.full_like(batch.key_lengths, batch.width))

    replayed = step(rig.node, clips, rig.runners, edit_batch=unmasked)
    assert replayed.regions.blocks == "blocks_W256"
    err = rel(replayed.rows, live.rows)
    print(f"\nunmasked replay vs eager rel {err:.2e}")
    assert err > 0.3
