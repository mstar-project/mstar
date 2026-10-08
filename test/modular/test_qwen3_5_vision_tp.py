"""Sharding Qwen3.5's ViT across ranks.

The checkpoint stores attention's projection already fused as ``attn.qkv``, so
``QKVParallelLinear`` has to split it by block before it shards by head; a flat
narrow would hand rank 0 all of q. These load one synthetic checkpoint into a
single-rank tower and into each rank of a sharded one, then run the ranks in
lockstep threads whose comm group sums in Python, and compare the outputs.

Runs on CPU in CI: the fake group stands in for NCCL.
"""
from __future__ import annotations

import threading

import pytest
import torch
import torch.nn.functional as F

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed import QKVParallelLinear
from mstar.model.loader.base import load_weights_into
from mstar.model.qwen3_5.components.vision import (
    Qwen3_5VisionModel,
    vision_interpolation,
    vision_position_ids,
    vision_seq_lengths,
)
from mstar.model.qwen3_5.config import VISION_ATTN, Qwen3_5VisionConfig

CONFIG = Qwen3_5VisionConfig(
    depth=2, hidden_size=64, intermediate_size=96, num_heads=4,
    in_channels=3, patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
    num_position_embeddings=16, out_hidden_size=48,
)
GRID = [(1, 4, 6), (1, 2, 2)]  # two images, so two attending segments


class ThreadedCommGroup(CommGroup):
    """One rank of a group whose ranks are threads of this process."""

    def __init__(self, rank: int, shared: dict):
        super().__init__(rank, rank, list(range(shared["world"])))
        self.shared = shared

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        shared = self.shared
        shared["parts"][self.rank] = input_.clone()
        shared["barrier"].wait()
        total = torch.stack(shared["parts"]).sum(0)
        shared["barrier"].wait()  # nobody overwrites parts before all have read
        input_.copy_(total)
        return input_


class SdpaRagged:
    def run(self, q, k, v):
        outs, start = [], 0
        for n in vision_seq_lengths(GRID):
            s = slice(start, start + n)
            outs.append(F.scaled_dot_product_attention(
                q[s].transpose(0, 1), k[s].transpose(0, 1), v[s].transpose(0, 1),
            ).transpose(0, 1))
            start += n
        return torch.cat(outs)


def _checkpoint() -> dict[str, torch.Tensor]:
    """Unsharded weights under the names the checkpoint uses after the
    ``model.visual.`` prefix comes off (``qkv`` fused)."""
    torch.manual_seed(0)
    return {
        n: torch.randn_like(p) * 0.1
        for n, p in Qwen3_5VisionModel(CONFIG).named_parameters()
    }


def _tower(group: CommGroup, ckpt: dict) -> Qwen3_5VisionModel:
    tower = Qwen3_5VisionModel(CONFIG, group)
    loaded = load_weights_into(tower, ckpt.items())
    assert loaded == {n for n, _ in tower.named_parameters()}
    for block in tower.blocks:
        block.attn.bind_resources({VISION_ATTN: SdpaRagged()})
    return tower.eval()


@torch.no_grad()
def _run(tower: Qwen3_5VisionModel, pixels: torch.Tensor) -> torch.Tensor:
    merge = CONFIG.spatial_merge_size
    indices, weights = vision_interpolation(
        GRID, CONFIG.num_grid_per_side, merge, pixels.device,
    )
    return tower(
        pixels, indices, weights,
        vision_position_ids(GRID, merge, pixels.device),
    )


@pytest.mark.parametrize("tp", [2, 4])
def test_fused_qkv_shards_by_head_within_each_block(tp):
    hidden, heads = CONFIG.hidden_size, CONFIG.num_heads
    head_dim = hidden // heads
    fused = torch.arange(3 * hidden, dtype=torch.float32)[:, None].expand(-1, 2)
    q, k, v = fused.chunk(3)
    per = heads // tp * head_dim
    for rank in range(tp):
        layer = QKVParallelLinear(
            CommGroup(rank, rank, list(range(tp))), hidden_size=2,
            head_size=head_dim, total_num_heads=heads, bias=False,
        )
        layer.weight_loader(layer.weight, fused)
        sl = slice(rank * per, (rank + 1) * per)
        assert torch.equal(layer.weight, torch.cat([q[sl], k[sl], v[sl]]))


@pytest.mark.parametrize("tp", [2, 4])
def test_sharded_tower_matches_single_rank(tp):
    ckpt = _checkpoint()
    numel = CONFIG.in_channels * CONFIG.temporal_patch_size * CONFIG.patch_size ** 2
    pixels = torch.randn(sum(t * h * w for t, h, w in GRID), numel)
    ref = _run(_tower(CommGroup.trivial(), ckpt), pixels)

    shared = {"world": tp, "parts": [None] * tp, "barrier": threading.Barrier(tp)}
    towers = [_tower(ThreadedCommGroup(r, shared), ckpt) for r in range(tp)]
    assert towers[0].blocks[0].attn.num_heads == CONFIG.num_heads // tp
    outs: list[torch.Tensor | None] = [None] * tp
    errors: list[BaseException] = []

    def rank_main(r: int) -> None:
        try:
            outs[r] = _run(towers[r], pixels)
        except BaseException as e:  # surface it, and free the other ranks
            errors.append(e)
            shared["barrier"].abort()

    threads = [threading.Thread(target=rank_main, args=(r,)) for r in range(tp)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors
    assert ref.shape == (sum(h * w for _, h, w in GRID) // 4, CONFIG.out_hidden_size)
    for out in outs:
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
