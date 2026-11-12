"""Ulysses sequence parallelism for LTX-2.5's DiT, on two CPU processes over gloo: a
toy-size DiT sharded across two ranks against the same DiT on one. The exchange itself
is tested in ``test_sequence_parallel.py``.
"""

from __future__ import annotations

import dataclasses
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, ".")

from mstar.distributed.communication import CommGroup  # noqa: E402
from mstar.engine.resources.step import StepContext  # noqa: E402
from mstar.engine.state.adapters import node_handles, node_resources, state_setup, state_steps  # noqa: E402
from mstar.model.components.diffusion.denoise_loop import StepKey  # noqa: E402
from mstar.model.ltx2_5.components.transformer import LTX2DiT, build_rope  # noqa: E402
from mstar.model.ltx2_5.config import LTX25Config  # noqa: E402
from mstar.model.ltx2_5.ltx2_5_model import LTX25Model  # noqa: E402
from mstar.model.ltx2_5.submodules import LTXDenoiseSubmodule, LTXShape  # noqa: E402
from mstar.model.submodule_base import NodeInputs  # noqa: E402

WORLD = 2


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


def _toy_config():
    cfg = LTX25Config()
    return dataclasses.replace(
        cfg.transformer, num_layers=2, num_attention_heads=4, attention_head_dim=16,
        audio_num_attention_heads=4, audio_attention_head_dim=8, cross_attention_dim=64,
        audio_cross_attention_dim=32, in_channels=8, out_channels=8, audio_in_channels=8, audio_out_channels=8,
    )


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
    inputs = [NodeInputs(resource_step_info=StepKey(shape))] * len(rows)
    step = state_steps(sub.declare_step("gen", rows, inputs), handles)
    step.set_ctx(StepContext(request_ids=rows, graph_walk="gen", slot=0, capture=False))
    step.begin_calls()


@torch.inference_mode()
def _dit(rank, frames, height, width):
    cfg = _toy_config()
    torch.manual_seed(0)
    single = LTX2DiT(cfg)
    for p in single.parameters():
        torch.nn.init.normal_(p, std=0.05)
    sharded = LTX2DiT(cfg, sp_group=_group(rank))
    sharded.load_state_dict(single.state_dict())
    tokens = frames * height * width
    rope = build_rope(cfg, frames, height, width, 24.0, 5, "cpu")
    _opened_on_sdpa(sharded, cfg, _group(rank), LTXShape(frames, height, width, 5, 24.0, 7), [0, 1])
    g = torch.Generator().manual_seed(1)
    inputs = (
        torch.randn(2, tokens, 8, generator=g), torch.randn(2, 5, 8, generator=g),
        torch.randn(2, 7, 64, generator=g), torch.randn(2, 7, 32, generator=g),
        torch.tensor([1000.0, 500.0]), rope,
    )
    want_video, want_audio = single(*inputs)
    video, audio = sharded(*inputs)
    assert video.shape == want_video.shape and audio.shape == want_audio.shape
    torch.testing.assert_close(video, want_video, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(audio, want_audio, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [(2, 2, 3), (3, 1, 5)], ids=["even", "uneven"])
def test_sequence_parallel_dit_matches_one_rank(tmp_path, shape):
    _run(_dit, tmp_path / "store", *shape)
