"""Muse Glimmer's vision block loop replayed from its capture, against the same
loop run inline, on the real checkpoint. Needs CUDA and a cached
meta-models/Muse-Glimmer-30B (its metadata and vision shards); skips otherwise.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = "meta-models/Muse-Glimmer-30B"
DEVICE = torch.device("cuda:0")
WALK, RID = "prefill_vision", 1
# rel L2: the replay matches the inline loop bit for bit on an H200
TOLERANCE = 1e-3

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="captures CUDA graphs")


def _snapshot() -> str:
    from huggingface_hub import snapshot_download

    try:
        path = Path(snapshot_download(REPO, local_files_only=True))
        index = json.loads((path / "model.safetensors.index.json").read_text())
    except Exception:
        pytest.skip(f"{REPO} is not cached")
    shards = {f for key, f in index["weight_map"].items() if key.startswith("model.vision")}
    if not all((path / f).exists() for f in shards):
        pytest.skip(f"{REPO}'s vision shards are not cached")
    return str(path)


def test_the_replayed_block_loop_matches_the_inline_one():
    """The replay runs the inline loop's numerics: `embed` ends on `ln_pre`,
    which autocast returns in fp32, so the replay's buffers are fp32
    (`BLOCK_LOOP_DTYPE`; the runner refuses buffers that would round it)."""
    from mstar.engine.accelerator_graph_runner import PiecewiseAcceleratorGraphRunner, autocast_scope
    from mstar.engine.resources import StepContext, StepRunner
    from mstar.engine.resources.base import EngineResourceInfo, build_resource
    from mstar.model.components.block_loop_encoder import VISION_BLOCK_LOOP
    from mstar.model.muse.glimmer_model import MuseGlimmerModel

    bf16 = torch.bfloat16
    model = MuseGlimmerModel(_snapshot())
    node = model._create_submodule("vision_encoder", str(DEVICE), bf16, None)
    specs = [s for s in model.get_node_resources() if "vision_encoder" in s.nodes]
    resources = {s.resource_key: build_resource(s, EngineResourceInfo(device=DEVICE)) for s in specs}
    node.bind_node_resources(resources)
    step_runner = StepRunner(resources, node_resources={"vision_encoder": list(resources)})

    # 64 x 48 patches: four windows, two of them partial, in the 4096 bucket
    rows = torch.linspace(0, 1, 896)[:, None]
    cols = torch.linspace(0, 1, 672)[None, :]
    noise = torch.rand(3, 896, 672, generator=torch.Generator().manual_seed(0))
    image = (0.6 * torch.stack([rows * cols, rows.expand(-1, 672), cols.expand(896, -1)]) + 0.4 * noise)
    # the walk's media layout and pixels, as the API hands them over
    prompt = model.process_prompt("Describe this", ["image", "text"], ["text"], tensors={"image_inputs": [image]})
    fwd_info = SimpleNamespace(step_metadata=prompt.metadata)
    row = node.prepare_inputs(WALK, fwd_info, {"pixel_values": prompt.new_input_tensors["pixel_values"]})
    bucket = next(b for b in node.BLOCK_LOOP_PATCH_BUCKETS if b >= row.input_seq_len)

    config = node.get_piecewise_accelerator_graph_configs(DEVICE, bf16)[VISION_BLOCK_LOOP]
    runner = PiecewiseAcceleratorGraphRunner(
        label=VISION_BLOCK_LOOP, config=dataclasses.replace(config, total_tokens=[bucket]),
        resources=resources, step_runner=step_runner, device=DEVICE, autocast_dtype=bf16,
        num_slots=1, node_name="vision_encoder",
    )
    runner.size_captures()
    runner.warmup_and_capture({runner._bucket(s) for s in runner.prepare_for_capture()})
    # what `_replay_block_loop` asks, so the second encode below replays
    assert runner.can_run(1, row.input_seq_len) and runner.can_lease([row])

    def encode(runners: dict) -> torch.Tensor:
        step = None
        if not runners:
            # inline, the node's own step plans the attention
            step = node.declare_step(WALK, [RID], [row])
            step.set_ctx(StepContext(request_ids=(RID,), graph_walk=WALK, slot=0, capture=False))
            assert step_runner.admit(step).ok
            step_runner.plan(step)
        # a declared step keeps the forward inline; a leased replay has none
        engine_inputs = SimpleNamespace(request_ids=[RID], piecewise_runners=runners, step=step)
        with torch.no_grad(), autocast_scope(bf16, DEVICE.type):
            out = node.forward_batched(WALK, engine_inputs, **node.preprocess(WALK, None, [row]))
        if step is not None:
            step_runner.commit(step)
        return out[RID]["vision_embeds"][0].float()

    inline = encode({})
    replayed = encode({VISION_BLOCK_LOOP: runner})
    err = float((replayed - inline).norm() / inline.norm())
    assert err < TOLERANCE, err
