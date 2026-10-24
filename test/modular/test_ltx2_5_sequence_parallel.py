"""Ulysses sequence parallelism for LTX-2.5's DiT, on two CPU processes over gloo: a
toy-size DiT sharded across two ranks against the same DiT on one. The exchange itself
is tested in ``test_sequence_parallel.py``.
"""

from __future__ import annotations

import dataclasses
import os
import socket
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, ".")

from mstar.distributed.communication import CommGroup  # noqa: E402
from mstar.model.components.diffusion.attention import sdpa_attention  # noqa: E402
from mstar.model.ltx2_5.components.transformer import LTX2Attends, LTX2DiT, build_rope  # noqa: E402
from mstar.model.ltx2_5.config import LTX25Config  # noqa: E402

WORLD = 2


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _group(rank: int) -> CommGroup:
    group = CommGroup(rank, rank, list(range(WORLD)))
    group.device_group = dist.group.WORLD
    group.initialized = True
    return group


def _run(fn, *args):
    mp.spawn(_entry, args=(fn, _free_port(), args), nprocs=WORLD)


def _entry(rank, fn, port, args):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
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
    attends = LTX2Attends(*([sdpa_attention] * 6))
    g = torch.Generator().manual_seed(1)
    inputs = (
        torch.randn(2, tokens, 8, generator=g), torch.randn(2, 5, 8, generator=g),
        torch.randn(2, 7, 64, generator=g), torch.randn(2, 7, 32, generator=g),
        torch.tensor([1000.0, 500.0]), rope, attends,
    )
    want_video, want_audio = single(*inputs)
    video, audio = sharded(*inputs)
    assert video.shape == want_video.shape and audio.shape == want_audio.shape
    torch.testing.assert_close(video, want_video, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(audio, want_audio, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [(2, 2, 3), (3, 1, 5)], ids=["even", "uneven"])
def test_sequence_parallel_dit_matches_one_rank(shape):
    _run(_dit, *shape)
