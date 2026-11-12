"""Sequence parallelism for LTX-2.5's DiT, on two CPU processes over gloo: the denoise step over
the model's own resources (its sequence shard and ragged attentions, FlashInfer's kernels as CPU
math), split over two ranks, against the same step on one rank attending through SDPA. Every
attention the shard reaches runs: the video self-attention's Ulysses exchange, the
cross-attentions cut to a rank's video tokens, and the audio queries over the video merged across
the ranks. The exchanges themselves are tested in ``test_sequence_parallel.py``.
"""

from __future__ import annotations

import dataclasses
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from flashinfer_reference import reference_flashinfer

sys.path.insert(0, ".")

from mstar.distributed.communication import CommGroup  # noqa: E402
from mstar.engine.resources.step import StepContext  # noqa: E402
from mstar.engine.state.adapters import node_handles, node_resources, state_setup, state_steps  # noqa: E402
from mstar.model.ltx2_5.components.transformer import LTX2DiT, build_rope  # noqa: E402
from mstar.model.ltx2_5.config import LTX25Config  # noqa: E402
from mstar.model.ltx2_5.ltx2_5_model import LTX25Model  # noqa: E402
from mstar.model.ltx2_5.submodules import LTXDenoiseSubmodule, LTXShape  # noqa: E402
from mstar.model.submodule_base import NodeInputs  # noqa: E402

WORLD = 2
CPU = torch.device("cpu")


def _group(rank: int) -> CommGroup:
    group = CommGroup(rank, rank, list(range(WORLD)))
    group.device_group = dist.group.WORLD
    group.initialized = True
    return group


def _run(fn, store_path, *args):
    mp.spawn(_entry, args=(fn, str(store_path), args), nprocs=WORLD)


def _entry(rank, fn, store_path, args):
    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=WORLD)
    torch.set_num_threads(1)
    try:
        fn(rank, *args)
    finally:
        dist.destroy_process_group()


def _toy_config() -> LTX25Config:
    # head dims FlashInfer has kernels for, so the CPU kernel attends them unpadded
    cfg = LTX25Config()
    toy = dataclasses.replace(
        cfg.transformer, num_layers=2, num_attention_heads=4, attention_head_dim=64,
        audio_num_attention_heads=4, audio_attention_head_dim=64, cross_attention_dim=256,
        audio_cross_attention_dim=256, in_channels=8, out_channels=8, audio_in_channels=8, audio_out_channels=8,
    )
    return dataclasses.replace(cfg, transformer=toy)


def _resources(cfg: LTX25Config, joint: JointGroups) -> dict:
    """The dit's resources from the model's own declaration, built and bound as the engine does."""
    model = LTX25Model(model_path_hf="test/ltx2_5", skip_weight_loading=True)
    model.set_config(cfg)
    specs = resolve_spec_dependencies(model.get_node_resources())
    for spec in specs.values():
        if isinstance(getattr(spec, "config", None), RaggedAttentionConfig):
            spec.config.dtype, spec.config.workspace_mb = torch.float32, 1
    info = EngineResourceInfo(device=CPU, joint_comm_group=joint, kv_dtype=torch.float32)
    res = {key: build_resource(spec, info) for key, spec in specs.items()}
    for key, spec in specs.items():
        res[key].bind_dependencies({dep: res[dep] for dep in spec.depends_on()})
    return res


def _opened_on_sdpa(dit, cfg, group, shape, rows):
    """``dit`` with its handles bound and open for one step on SDPA, as the engine would on a node set to it."""
    config = dataclasses.replace(LTX25Config(), transformer=cfg)
    model = LTX25Model(skip_weight_loading=True)
    model.set_config(config)
    on_sdpa = {spec.resource_key: {"backend": "sdpa"} for spec in node_resources(model)}
    setup, _ = state_setup({"resources": on_sdpa}, model.declare_state())
    handles = node_handles(model, "dit", {}, sp_group=group, setup=setup)
    for module in dit.modules():
        if hasattr(module, "bind_state"):
            module.bind_state(handles)
    sub = LTXDenoiseSubmodule(None, config, loop_name="denoise_loop")
    step = state_steps(sub.declare_step("gen", rows, [NodeInputs(resource_step_info=shape)] * len(rows)), handles)
    step.set_ctx(StepContext(request_ids=rows, graph_walk="gen", slot=0, capture=False))
    step.begin_calls()


@torch.inference_mode()
def _denoise(rank, frames, height, width):
    sys.modules["flashinfer"] = reference_flashinfer()
    cfg = _toy_config()
    torch.manual_seed(0)
    dit = LTX2DiT(cfg.transformer)
    for p in dit.parameters():
        torch.nn.init.normal_(p, std=0.05)
    shape = LTXShape(frames=frames, height=height, width=width, audio_frames=5, fps=24.0, text_len=7)
    # one rank: no resources, so every attention runs SDPA
    whole = LTXDenoiseSubmodule(dit, cfg, loop_name=DENOISE_LOOP, use_ragged_attention=True)
    split = LTXDenoiseSubmodule(dit, cfg, loop_name=DENOISE_LOOP, use_ragged_attention=True)
    sp = _group(rank)
    split.bind_node_resources(_resources(cfg, JointGroups(tp_group=CommGroup.trivial(), sp_group=sp, instance=sp)))
    rids = ["r0", "r1"]
    step = engine_step(split, "generate_av", rids, [NodeInputs(resource_step_info=shape)] * len(rids))
    step.set_ctx(StepContext(request_ids=rids, graph_walk="generate_av", slot=0, capture=False))
    StepRunner(split.node_resources).plan(step)
    tokens = frames * height * width
    rope = build_rope(cfg, frames, height, width, 24.0, 5, "cpu")
    _opened_on_sdpa(sharded, cfg, _group(rank), LTXShape(frames, height, width, 5, 24.0, 7), [0, 1])
    g = torch.Generator().manual_seed(1)
    inputs = (
        torch.randn(2, tokens, 8, generator=g), torch.randn(2, 5, 8, generator=g),
        torch.randn(2, 7, 64, generator=g), torch.randn(2, 7, 32, generator=g),
        torch.tensor([1000.0, 500.0]), rope,
    )
    want = whole.denoise(None, shape, **inputs)
    got = split.denoise(None, shape, **inputs)
    for key in want:
        assert got[key].shape == want[key].shape, key
        torch.testing.assert_close(got[key], want[key], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(2, 2, 3), (3, 1, 5)], ids=["even", "uneven"])
def test_sequence_parallel_denoise_step_matches_one_rank(tmp_path, shape):
    _run(_denoise, tmp_path / "store", *shape)
