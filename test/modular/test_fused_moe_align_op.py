"""Parity between the vendored CUDA ``moe_align_block_size`` op and the fallback.

Until the JIT build was fixed this op never actually compiled in-tree, so the
torch fallback was silently carrying every call and the CUDA path was
untested. These tests compare the two directly.

Only the prefix ``[0, num_tokens_post_pad)`` of ``sorted_ids`` is meaningful:
both implementations leave the tail of the worst-case buffer uninitialised,
and the Triton kernel never reads past that point.

If the CUDA op cannot be built on the machine running the tests, they skip
rather than fail -- the fallback is a supported configuration.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch
import triton

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _cuda_op_or_skip():
    from mstar.utils.fused_moe.align import _cuda_op_available

    if not _cuda_op_available():
        pytest.skip("vendored CUDA align op could not be built here")


def _routing(kind: str, m: int, num_experts: int, top_k: int) -> torch.Tensor:
    if kind == "skewed":  # every token to the same low experts
        return (
            torch.arange(top_k, dtype=torch.int32)
            .unsqueeze(0)
            .expand(m, top_k)
            .contiguous()
            .cuda()
        )
    if kind == "uniform":
        return (
            (torch.arange(m * top_k, dtype=torch.int32) % num_experts)
            .reshape(m, top_k)
            .contiguous()
            .cuda()
        )
    g = torch.Generator().manual_seed(7 + m)
    return (
        torch.stack([torch.randperm(num_experts, generator=g)[:top_k] for _ in range(m)])
        .to(torch.int32)
        .contiguous()
        .cuda()
    )


def _run_both(topk_ids: torch.Tensor, block_size: int, num_experts: int):
    """Call the CUDA op and the torch fallback into separate buffers."""
    from mstar.utils.fused_moe.align import _moe_align_block_size_torch

    n = topk_ids.numel()
    max_padded = n + num_experts * (block_size - 1)
    max_blocks = triton.cdiv(max_padded, block_size)

    def buffers():
        return (
            torch.zeros(max_padded, dtype=torch.int32, device="cuda"),
            torch.zeros(max_blocks, dtype=torch.int32, device="cuda"),
            torch.zeros(1, dtype=torch.int32, device="cuda"),
        )

    cuda_out = buffers()
    torch.ops._mstar_moe_C.moe_align_block_size(
        topk_ids, num_experts, block_size, *cuda_out
    )
    torch_out = buffers()
    _moe_align_block_size_torch(topk_ids, block_size, num_experts, *torch_out)
    torch.cuda.synchronize()
    return cuda_out, torch_out


@pytest.mark.parametrize("kind", ["uniform", "skewed", "random"])
@pytest.mark.parametrize("block_size", [16, 32, 64, 128])
@pytest.mark.parametrize("m", [1, 7, 64, 512])
def test_cuda_op_matches_torch_fallback(kind, block_size, m):
    _cuda_op_or_skip()
    num_experts, top_k = 128, 8
    topk_ids = _routing(kind, m, num_experts, top_k)

    (c_sorted, c_experts, c_npp), (t_sorted, t_experts, t_npp) = _run_both(
        topk_ids, block_size, num_experts
    )

    assert int(c_npp.item()) == int(t_npp.item()), "num_tokens_post_pad differs"
    npp = int(c_npp.item())
    nblocks = triton.cdiv(npp, block_size)

    assert torch.equal(c_experts[:nblocks], t_experts[:nblocks]), "expert_ids differ"

    # Placement of a slot *within* its expert's blocks is not part of the
    # contract. The CUDA op assigns positions with an atomic counter, so the
    # order inside a block -- and, when an expert owns more than one block,
    # the split between those blocks -- is arbitrary. That is harmless: the
    # Triton kernel writes each slot to its own row C[offs_token], so the
    # output is identical either way. What must agree is which expert each
    # slot is grouped under.
    n = topk_ids.numel()

    def slots_per_expert(sorted_ids, expert_ids):
        out: dict[int, list[int]] = {}
        for b in range(nblocks):
            blk = sorted_ids[b * block_size : (b + 1) * block_size]
            real = blk[blk < n]
            if real.numel():
                out.setdefault(int(expert_ids[b]), []).extend(real.tolist())
        return {e: sorted(v) for e, v in out.items()}

    assert slots_per_expert(c_sorted, c_experts) == slots_per_expert(t_sorted, t_experts), (
        "slot-to-expert grouping differs"
    )


def test_fused_experts_is_cuda_graph_capturable():
    """The whole dispatch must capture -- this is why the CUDA op matters.

    The torch fallback sizes a ``repeat_interleave`` from a device tensor,
    which forces a device-to-host sync and aborts capture with
    ``cudaErrorStreamCaptureInvalidated``. With the CUDA op built, the fused
    MoE goes into the graph as intended by ``engine/cuda_graph_runner.py``.
    """
    _cuda_op_or_skip()
    from mstar.utils.fused_moe import fused_experts

    torch.manual_seed(0)
    m, hidden, inter, E, top_k = 8, 256, 128, 32, 4
    hs = torch.randn(m, hidden, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(E, 2 * inter, hidden, device="cuda", dtype=torch.bfloat16) / hidden**0.5
    w2 = torch.randn(E, hidden, inter, device="cuda", dtype=torch.bfloat16) / inter**0.5
    probs = torch.softmax(torch.randn(m, E, device="cuda"), dim=-1)
    tw, ti = torch.topk(probs, top_k, dim=-1)
    tw = (tw / tw.sum(-1, keepdim=True)).to(torch.bfloat16)

    eager = fused_experts(hs, w1, w2, tw, ti)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fused_experts(hs, w1, w2, tw, ti)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        captured = fused_experts(hs, w1, w2, tw, ti)
    g.replay()
    torch.cuda.synchronize()

    rel = (captured.float() - eager.float()).abs().max() / eager.abs().float().max()
    assert rel < 2e-2, f"graph replay disagrees with eager: rel_err={rel:.3g}"


@pytest.mark.parametrize("block_size", [16, 64])
def test_cuda_op_output_is_well_formed(block_size):
    """Independent of the fallback: check the invariants the kernel relies on."""
    _cuda_op_or_skip()
    num_experts, top_k, m = 128, 8, 333
    topk_ids = _routing("random", m, num_experts, top_k)
    n = topk_ids.numel()

    (sorted_ids, expert_ids, npp_t), _ = _run_both(topk_ids, block_size, num_experts)
    npp = int(npp_t.item())

    assert npp % block_size == 0, "padded length must be a whole number of blocks"
    # Every real slot appears exactly once; the rest of the prefix is padding.
    real = sorted_ids[:npp][sorted_ids[:npp] < n]
    assert real.numel() == n
    assert torch.equal(real.sort().values, torch.arange(n, device="cuda", dtype=torch.int32))

    # Within a block, every real slot must belong to that block's expert.
    flat = topk_ids.reshape(-1)
    for b in range(triton.cdiv(npp, block_size)):
        blk = sorted_ids[b * block_size : (b + 1) * block_size]
        blk_real = blk[blk < n]
        if blk_real.numel() == 0:
            continue
        assert (flat[blk_real.long()] == expert_ids[b]).all(), f"block {b} mixes experts"
