"""Kimi K3.1's routed experts (MXFP4 x bf16, SiTU 4/25: ``fused_experts_w4a16``, what the B300 serves them with)
at a rank's full geometry on CPU (from the quant-w4 lane).

- the alignment buffers, the GEMM grids and the per-call caches at 8192 tokens x top-16 over 896 experts (TP8) and
  112 local experts with the other ranks' slots on the sentinel (EP8), against the int32 / grid limits the kernels
  index with;
- numerics at the full geometry (E=896, H=3584, I=384; E=112, I=3072) at a few tokens, and the token / slot index
  math at 8192 tokens x top-16 over the full expert count (small H / I), each in Triton's CPU interpreter (a
  subprocess: the interpreter is chosen at import) against the experts dequantized in fp32.

The interpreter runs in fp16: its bf16 tl.dot is wrong (bf16 has no numpy dtype), which is the interpreter's
limit, not the kernel's; the H200 / B300 runs cover bf16.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

INT32_MAX = 2**31 - 1
GEOMS = {"tp8": (896, 384), "ep8": (112, 3072)}  # (experts on the rank, expert width on the rank)
H, TOP_K, GLOBAL_E, PREFILL = 3584, 16, 896, 8192


def _ids(geom, m, seed=0):
    e, _ = GEOMS[geom]
    g = torch.Generator().manual_seed(seed)
    ids = torch.rand(m, GLOBAL_E, generator=g).topk(TOP_K, dim=-1).indices
    if geom == "ep8":
        ids = torch.where(ids < e, ids, torch.full_like(ids, e))
    return ids.to(torch.int32)


@pytest.mark.parametrize("geom", sorted(GEOMS))
@pytest.mark.parametrize("m", [1, 64, 896, 897, PREFILL])
def test_alignment_grids_and_caches_fit_their_index_types(geom, m):
    from mstar.utils.fused_moe.align import moe_align_block_size
    from mstar.utils.fused_moe.kernels import _grid_rows, get_default_config

    e, inter = GEOMS[geom]
    ids = _ids(geom, m)
    config = get_default_config(M=m, E=e, N=2 * inter, K=H, top_k=TOP_K)
    bm = config["BLOCK_SIZE_M"]
    sorted_ids, expert_ids, post_pad = moe_align_block_size(ids, bm, e)
    numel = ids.numel()
    valid = ids.reshape(-1) < e
    # every valid slot exactly once, padding = numel, blocks single-expert
    slots = sorted_ids[: post_pad.item()]
    real = slots[slots < numel].long()
    assert torch.equal(torch.sort(real).values, torch.nonzero(valid).flatten())
    assert post_pad.item() <= sorted_ids.numel() and post_pad.item() % bm == 0
    blocks = post_pad.item() // bm
    owner = ids.reshape(-1)[slots.clamp_max(numel - 1).long()].view(blocks, bm)
    is_real = (slots < numel).view(blocks, bm)
    assert bool(((owner == expert_ids[:blocks, None]) | ~is_real).all())
    em = _grid_rows(sorted_ids, ids, bm)
    assert em >= post_pad.item()
    for n in (2 * inter, H):  # gate/up, down
        assert triton_cdiv(em, bm) * triton_cdiv(n, config["BLOCK_SIZE_N"]) <= INT32_MAX
    # act_and_mul_kernel indexes a slot's row as pid * (2 I) in int32: within range at the prefill bound (EP8 at
    # 8192 tokens: 805M of 2^31; at 21K tokens it would wrap)
    assert numel * 2 * inter <= INT32_MAX


def triton_cdiv(a, b):
    return -(-a // b)


def test_per_call_caches_at_the_prefill_bound():
    """The three caches fused_experts_w4a16 allocates per call at 8192 tokens (bf16): TP8 1.24 GB, EP8 3.36 GB, of
    which EP8's gate/up and activation caches are sized for every slot though 1/8 are this rank's."""
    sizes = {}
    for geom, (_, inter) in GEOMS.items():
        m_topk = PREFILL * TOP_K
        sizes[geom] = 2 * (m_topk * 2 * inter + m_topk * inter + PREFILL * TOP_K * H)
    assert sizes["tp8"] / 2**30 == pytest.approx(1.16, abs=0.02)  # GiB
    assert sizes["ep8"] / 2**30 == pytest.approx(3.12, abs=0.02)


_CASE = Path(__file__).with_name("k31_experts_case.py")
# minutes each in the interpreter (TP8 at 2 tokens: 253 s on one core): opt in
interpreted = pytest.mark.skipif(not os.environ.get("MSTAR_INTERPRET_FULL_SHAPES"),
                                 reason="set MSTAR_INTERPRET_FULL_SHAPES=1 (minutes in Triton's CPU interpreter)")


def _interpret(*args, timeout=3600):
    root = Path(__file__).resolve().parents[4]
    env = {**os.environ, "TRITON_INTERPRET": "1", "OMP_NUM_THREADS": "2",
           "PYTHONPATH": os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")])}
    run = subprocess.run([sys.executable, str(_CASE), *args], env=env, capture_output=True, text=True,
                         check=False, timeout=timeout)
    assert run.returncode == 0, (run.stdout[-2000:], run.stderr[-4000:])
    return run.stdout


@interpreted
@pytest.mark.parametrize("geom, m", [("tp8", 2), ("ep8", 4)])
def test_situ_mxfp4_experts_at_the_full_geometry(geom, m):
    pytest.importorskip("triton")
    _interpret("--geom", geom, "--tokens", str(m))


@interpreted
@pytest.mark.parametrize("geom", sorted(GEOMS))
def test_slot_index_math_at_8192_tokens_over_every_expert(geom):
    pytest.importorskip("triton")
    _interpret("--geom", geom, "--tokens", str(PREFILL), "--h", "64", "--i", "32")
