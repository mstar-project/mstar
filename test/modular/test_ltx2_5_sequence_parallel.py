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

from mstar.distributed.communication import CommGroup, JointGroups, sp_seq_split  # noqa: E402
from mstar.engine.resources import RaggedAttentionConfig, StepContext, StepRunner  # noqa: E402
from mstar.engine.resources.base import EngineResourceInfo, build_resource  # noqa: E402
from mstar.engine.resources.spec import resolve_spec_dependencies  # noqa: E402
from mstar.engine.state.adapters import node_handles, node_resources  # noqa: E402
from mstar.model.components.diffusion.denoise_loop import StepKey  # noqa: E402
from mstar.model.ltx2_5.components.transformer import LTX2DiT  # noqa: E402
from mstar.model.ltx2_5.config import DENOISE_LOOP, VIDEO_SHARD, LTX25Config  # noqa: E402
from mstar.model.ltx2_5.ltx2_5_model import LTX25Model  # noqa: E402
from mstar.model.ltx2_5.submodules import LTXDenoiseSubmodule, LTXShape  # noqa: E402
from mstar.model.submodule_base import NodeInputs, engine_step  # noqa: E402

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


def _model(cfg: LTX25Config) -> LTX25Model:
    model = LTX25Model(model_path_hf="test/ltx2_5", skip_weight_loading=True)
    model.set_config(cfg)
    return model


def _resources(cfg: LTX25Config, joint: JointGroups) -> dict:
    """The dit's resources from the model's own declarations, built and bound as the engine does."""
    specs = resolve_spec_dependencies(node_resources(_model(cfg)))
    for spec in specs.values():
        if isinstance(getattr(spec, "config", None), RaggedAttentionConfig):
            spec.config.dtype, spec.config.workspace_mb = torch.float32, 1
    info = EngineResourceInfo(device=CPU, joint_comm_group=joint, kv_dtype=torch.float32)
    res = {
        key: build_resource(spec, dataclasses.replace(info, dependencies={k: specs[k] for k in spec.depends_on()}))
        for key, spec in specs.items()
    }
    for key, spec in specs.items():
        res[key].bind_dependencies({dep: res[dep] for dep in spec.depends_on()})
    return res


@torch.inference_mode()
def _denoise(rank, frames, height, width):
    sys.modules["flashinfer"] = reference_flashinfer()
    cfg = _toy_config()
    torch.manual_seed(0)
    dit = LTX2DiT(cfg.transformer)
    for p in dit.parameters():
        torch.nn.init.normal_(p, std=0.05)
    shape = StepKey(LTXShape(frames=frames, height=height, width=width, audio_frames=5, fps=24.0, text_len=7))
    t = cfg.transformer
    tokens = frames * height * width
    g = torch.Generator().manual_seed(1)
    sigma = torch.tensor([0.9, 0.5]).view(-1, 1, 1)
    inputs = dict(
        latents=torch.randn(2, tokens, t.in_channels, generator=g), timestep=sigma.flatten() * 1000,
        sigma=sigma, sigma_next=sigma * 0.5, audio_latents=torch.randn(2, 5, t.audio_in_channels, generator=g),
        text_video=torch.randn(2, 7, t.cross_attention_dim, generator=g),
        text_audio=torch.randn(2, 7, t.audio_cross_attention_dim, generator=g),
    )
    # one rank: no resources and no handles bound, so every attention runs SDPA
    want = LTXDenoiseSubmodule(dit, cfg, loop_name=DENOISE_LOOP).denoise(None, shape, **inputs)

    split = LTXDenoiseSubmodule(dit, cfg, loop_name=DENOISE_LOOP)
    sp = _group(rank)
    res = _resources(cfg, JointGroups(tp_group=CommGroup.trivial(), sp_group=sp, instance=sp))
    split.bind_node_resources(res)
    split.bind_node_state(node_handles(_model(cfg), "dit", res))
    rids = ["r0", "r1"]
    step = engine_step(split, "generate_av", rids, [NodeInputs(resource_step_info=shape)] * len(rids))
    step.set_ctx(StepContext(request_ids=rids, graph_walk="generate_av", slot=0, capture=False))
    StepRunner(res).plan(step)
    split_range = res[VIDEO_SHARD].split_range("video")
    assert split_range.counts == tuple(sp_seq_split(tokens, WORLD)) and split_range.rows == len(rids)
    step.begin_calls()
    got = split.denoise(None, shape, **inputs)
    step.end_calls()
    for key in want:
        assert got[key].shape == want[key].shape, key
        torch.testing.assert_close(got[key], want[key], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(2, 2, 3), (3, 1, 5)], ids=["even", "uneven"])
def test_sequence_parallel_denoise_step_matches_one_rank(tmp_path, shape):
    _run(_denoise, tmp_path / "store", *shape)
