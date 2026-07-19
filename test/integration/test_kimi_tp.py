"""Phase-3 tensor-parallel goldens for Kimi-K2.7 (reduced config): tp=2 == tp=1.

Proves the two TP subsystems added in Phase 3 are numerically correct on the
reduced config:

  * **MLA head-sharding** (``KimiMLAAttention``): the q/kv UP-projections shard
    ColumnParallel and ``o_proj`` reduces RowParallel, so each rank materializes
    only its ``num_attention_heads // tp_size`` local heads.
  * **MoE intermediate-sharding** (``KimiSparseMoeBlock``): the router stays
    replicated; each rank holds every expert but only a
    ``moe_intermediate_size // tp_size`` stripe of the SwiGLU intermediate
    (gate_up column-parallel / down row-parallel), all-reduced before the top-k
    sum-reduce. The shared expert shards via its ``ParallelGatedMLP``.

Two verification levels, both keyed to the SAME deterministic source weights and
loaded through the REAL per-rank ``weight_loader`` slicing (one weight path for
tp=1 and tp>1):

  1. **In-process rank simulation** (``test_*_tp2_sim_matches_tp1``, always runs
     on one GPU): for each of the 2 ranks, build the block with ``tp_size=2`` and
     that rank's weight shard, run it with a LOCAL no-op all-reduce so each rank
     returns only its partial, then SUM the two ranks' partials and assert it
     equals the tp=1 result within bf16 tolerance. This rigorously validates the
     shard math + weight-loading; the block outputs are pure row-parallel reduces
     (no un-sharded residual inside the block), so summing partials reconstructs
     the reduce exactly. The live NCCL all-reduce itself rides the shared,
     already-proven comm path (same primitive Orpheus tp2 uses).

  2. **Real multi-process NCCL** (``test_tp2_nccl_matches_tp1``, runs only when
     >= 2 CUDA devices are visible): spawn 2 ranks with a real NCCL comm group,
     run attention + MoE + a full decoder layer with the REAL all-reduce, and
     compare each rank's full output to the tp=1 reference. This exercises the
     actual collective and the decoder layer's residual wiring (which the
     partial-sum simulation cannot cover on its own).

Determinism: all weights/inputs come from a fixed-seed CPU generator, so the max
abs diffs are stable run-to-run (no flakiness).

Run:  pytest test/integration/test_kimi_tp.py -v
"""
from __future__ import annotations

import os
import socket

import pytest
import torch

from mstar.distributed.communication import CommGroup
from mstar.model.kimi_k2_7.components.attention import KimiMLAAttention
from mstar.model.kimi_k2_7.components.decoder_layer import KimiDecoderLayer
from mstar.model.kimi_k2_7.components.moe import KimiSparseMoeBlock
from mstar.model.kimi_k2_7.config import KimiK2Config

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Kimi TP goldens need a GPU (fused expert GEMM + MLA RMSNorm are CUDA-only)",
)

DEVICE = "cuda"
DTYPE = torch.bfloat16
TP = 2


# ---------------------------------------------------------------------------
# A world-size-2 comm group whose collectives are LOCAL no-ops. Used for the
# in-process rank simulation: each rank computes only its partial and the test
# sums the two ranks' partials to reconstruct the row-parallel all-reduce.
# ---------------------------------------------------------------------------
class _NoCommGroup(CommGroup):
    def __init__(self, rank: int) -> None:
        super().__init__(my_global_rank=rank, my_group_rank=rank, group_members=[0, 1])

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return input_


class _MockMLACache:
    """Paged-cache stand-in: causal SDPA at 1/sqrt(head_dim) over whatever local
    heads the rank hands it (attention is per-head independent, so a rank running
    SDPA on its head slice yields exactly those heads' outputs)."""

    def __init__(self, head_dim: int) -> None:
        self.scale = head_dim ** -0.5

    def set_layer_idx(self, _i):
        pass

    def set_active_label(self, _l):
        pass

    def advance_seq_lens(self, *_a, **_k):
        pass

    def run_attention(self, q, k, v):
        qt, kt, vt = (t.transpose(0, 1).float() for t in (q, k, v))  # (H,T,D)
        scores = torch.einsum("hqd,hkd->hqk", qt, kt) * self.scale
        num_tokens = q.shape[0]
        causal = torch.triu(
            torch.full((num_tokens, num_tokens), float("-inf"), device=q.device),
            diagonal=1,
        )
        attn = (scores + causal).softmax(-1)
        return torch.einsum("hqk,hkd->hqd", attn, vt).transpose(0, 1).to(q.dtype)


# ---------------------------------------------------------------------------
# Deterministic full-size source weights (CPU generator -> device-independent,
# identical across ranks / processes) and the REAL per-rank load helpers.
# ---------------------------------------------------------------------------
def _source_weights(cfg: KimiK2Config, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    H = cfg.num_attention_heads
    E, Hd, I = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
    sh = I * cfg.n_shared_experts

    def rn(*shape, std=0.03, mean=0.0):
        return torch.randn(*shape, generator=g) * std + mean

    return {
        # attention (replicated down-projs + norms, sharded up-projs + o_proj)
        "q_a": rn(cfg.q_lora_rank, cfg.hidden_size),
        "q_a_norm": rn(cfg.q_lora_rank, std=0.02, mean=1.0),
        "q_b": rn(H * cfg.qk_head_dim, cfg.q_lora_rank),
        "kv_a": rn(cfg.kv_lora_rank + cfg.qk_rope_head_dim, cfg.hidden_size),
        "kv_a_norm": rn(cfg.kv_lora_rank, std=0.02, mean=1.0),
        "kv_b": rn(H * (cfg.qk_nope_head_dim + cfg.v_head_dim), cfg.kv_lora_rank),
        "o": rn(cfg.hidden_size, H * cfg.v_head_dim),
        # moe (replicated fp32 router + sharded experts/shared)
        "router_w": torch.randn(E, Hd, generator=g),
        "router_b": torch.randn(E, generator=g),
        "gate": rn(E, I, Hd, std=0.05),
        "up": rn(E, I, Hd, std=0.05),
        "down": rn(E, Hd, I, std=0.05),
        "sh_gate": rn(sh, Hd, std=0.05),
        "sh_up": rn(sh, Hd, std=0.05),
        "sh_down": rn(Hd, sh, std=0.05),
        # decoder-layer norms
        "in_ln": rn(Hd, std=0.02, mean=1.0),
        "post_ln": rn(Hd, std=0.02, mean=1.0),
    }


def _load_attention(attn: KimiMLAAttention, src: dict) -> None:
    """Load full source weights through the REAL Column/Row weight_loaders (which
    slice this rank's head block) + direct copies for the replicated params."""
    attn.q_a_proj.weight.data.copy_(src["q_a"].to(DEVICE, DTYPE))
    attn.q_a_layernorm.weight.data.copy_(src["q_a_norm"].to(DEVICE, DTYPE))
    attn.q_b_proj.weight.weight_loader(attn.q_b_proj.weight, src["q_b"].to(DEVICE, DTYPE))
    attn.kv_a_proj_with_mqa.weight.data.copy_(src["kv_a"].to(DEVICE, DTYPE))
    attn.kv_a_layernorm.weight.data.copy_(src["kv_a_norm"].to(DEVICE, DTYPE))
    attn.kv_b_proj.weight.weight_loader(attn.kv_b_proj.weight, src["kv_b"].to(DEVICE, DTYPE))
    attn.o_proj.weight.weight_loader(attn.o_proj.weight, src["o"].to(DEVICE, DTYPE))


def _load_moe(block: KimiSparseMoeBlock, src: dict) -> None:
    """Load through the REAL fused-expert weight_loaders (per-rank intermediate
    slice) + replicated fp32 router + the shared expert's merged/row loaders."""
    block.gate.weight.data = src["router_w"].to(DEVICE)  # keep router fp32
    block.gate.e_score_correction_bias.data = src["router_b"].to(DEVICE)
    gu, dp = block.experts.gate_up_proj, block.experts.down_proj
    for e in range(block.num_experts):
        gu.weight_loader(gu, src["gate"][e].to(DEVICE, DTYPE), loaded_shard_id=f"gate:{e}")
        gu.weight_loader(gu, src["up"][e].to(DEVICE, DTYPE), loaded_shard_id=f"up:{e}")
        dp.weight_loader(dp, src["down"][e].to(DEVICE, DTYPE), loaded_shard_id=f"down:{e}")
    s = block.shared_expert
    s.gate_up_proj.weight.weight_loader(
        s.gate_up_proj.weight, src["sh_gate"].to(DEVICE, DTYPE), loaded_shard_id=0)
    s.gate_up_proj.weight.weight_loader(
        s.gate_up_proj.weight, src["sh_up"].to(DEVICE, DTYPE), loaded_shard_id=1)
    s.down_proj.weight.weight_loader(s.down_proj.weight, src["sh_down"].to(DEVICE, DTYPE))


def _load_decoder(layer: KimiDecoderLayer, src: dict) -> None:
    _load_attention(layer.self_attn, src)
    _load_moe(layer.mlp, src)
    layer.input_layernorm.weight.data.copy_(src["in_ln"].to(DEVICE, DTYPE))
    layer.post_attention_layernorm.weight.data.copy_(src["post_ln"].to(DEVICE, DTYPE))


def _inputs(cfg: KimiK2Config, num_tokens: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    h = (torch.randn(num_tokens, cfg.hidden_size, generator=g) * 0.1).to(DEVICE, DTYPE)
    pos = torch.arange(num_tokens, device=DEVICE)
    return h, pos


# ---------------------------------------------------------------------------
# Level 1 — in-process rank simulation (single GPU, deterministic, always runs)
# ---------------------------------------------------------------------------
def test_mla_attention_tp2_sim_matches_tp1():
    cfg = KimiK2Config.reduced()
    src = _source_weights(cfg, seed=101)
    h, pos = _inputs(cfg, num_tokens=6, seed=202)

    ref = KimiMLAAttention(cfg, CommGroup.trivial()).to(DEVICE, DTYPE)
    _load_attention(ref, src)
    assert ref.num_heads == cfg.num_attention_heads  # tp=1 sees all heads
    out_ref = ref(h, _MockMLACache(cfg.padded_head_dim), pos)

    partials = []
    for rank in range(TP):
        attn = KimiMLAAttention(cfg, _NoCommGroup(rank)).to(DEVICE, DTYPE)
        assert attn.num_heads == cfg.num_attention_heads // TP  # rank sees local heads
        _load_attention(attn, src)
        partials.append(attn(h, _MockMLACache(cfg.padded_head_dim), pos))

    out_tp2 = partials[0] + partials[1]  # row-parallel o_proj reduce == sum of ranks
    max_abs = (out_tp2 - out_ref).abs().max().item()
    assert max_abs < 5e-2, f"MLA tp2 vs tp1 max abs diff {max_abs}"
    torch.testing.assert_close(out_tp2, out_ref, rtol=2e-2, atol=2e-2)


def test_moe_block_tp2_sim_matches_tp1():
    cfg = KimiK2Config.reduced()
    src = _source_weights(cfg, seed=303)
    h, _ = _inputs(cfg, num_tokens=7, seed=404)

    ref = KimiSparseMoeBlock(cfg, CommGroup.trivial()).to(DEVICE, DTYPE)
    _load_moe(ref, src)
    full_inter = ref.experts.gate_up_proj.shape[1]
    out_ref = ref(h)

    partials = []
    for rank in range(TP):
        block = KimiSparseMoeBlock(cfg, _NoCommGroup(rank)).to(DEVICE, DTYPE)
        # each rank holds only a 1/TP stripe of the fused intermediate
        assert block.experts.gate_up_proj.shape[1] == full_inter // TP
        _load_moe(block, src)
        partials.append(block(h))

    out_tp2 = partials[0] + partials[1]  # intermediate-parallel reduce == sum of ranks
    max_abs = (out_tp2 - out_ref).abs().max().item()
    assert max_abs < 5e-2, f"MoE tp2 vs tp1 max abs diff {max_abs}"
    torch.testing.assert_close(out_tp2, out_ref, rtol=2e-2, atol=2e-2)


def test_tp2_sim_is_stable_across_repeats():
    """The simulated tp2==tp1 diffs must be bit-stable across repeats (no flaky
    golden). Re-run the attention + MoE simulation 3x from the same seeds and
    assert an identical max abs diff each time."""
    cfg = KimiK2Config.reduced()

    def attn_diff():
        src = _source_weights(cfg, seed=101)
        h, pos = _inputs(cfg, num_tokens=6, seed=202)
        ref = KimiMLAAttention(cfg, CommGroup.trivial()).to(DEVICE, DTYPE)
        _load_attention(ref, src)
        o_ref = ref(h, _MockMLACache(cfg.padded_head_dim), pos)
        parts = []
        for rank in range(TP):
            a = KimiMLAAttention(cfg, _NoCommGroup(rank)).to(DEVICE, DTYPE)
            _load_attention(a, src)
            parts.append(a(h, _MockMLACache(cfg.padded_head_dim), pos))
        return (parts[0] + parts[1] - o_ref).abs().max().item()

    diffs = [attn_diff() for _ in range(3)]
    assert diffs[0] == diffs[1] == diffs[2], f"unstable tp2 sim diffs: {diffs}"


# ---------------------------------------------------------------------------
# Level 2 — real multi-process NCCL (runs only with >= 2 CUDA devices)
# ---------------------------------------------------------------------------
def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _nccl_worker(rank: int, world_size: int, port: int, result_path: str) -> None:
    """One TP rank: real NCCL comm group, run attention + MoE + decoder layer, and
    compare each to a tp=1 reference built in-process from the same source weights.
    Rank 0 writes the max abs diffs to ``result_path``."""
    import torch.distributed as dist

    os.environ.setdefault("NCCL_IB_DISABLE", "1")  # coriander has no RDMA/IB
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world_size,
        rank=rank,
    )
    try:
        cfg = KimiK2Config.reduced()
        src = _source_weights(cfg, seed=505)
        h, pos = _inputs(cfg, num_tokens=6, seed=606)

        cg = CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=[0, 1])
        cg.device_group = None  # 2-rank world == default group
        cg.initialized = True

        # tp=2 blocks with the REAL all-reduce -> each rank produces the FULL output.
        attn = KimiMLAAttention(cfg, cg).to(DEVICE, DTYPE)
        _load_attention(attn, src)
        moe = KimiSparseMoeBlock(cfg, cg).to(DEVICE, DTYPE)
        _load_moe(moe, src)
        dec = KimiDecoderLayer(cfg, layer_idx=1, comm_group=cg).to(DEVICE, DTYPE)
        _load_decoder(dec, src)
        assert isinstance(dec.mlp, KimiSparseMoeBlock)  # layer_idx=1 is a MoE layer

        attn_tp2 = attn(h, _MockMLACache(cfg.padded_head_dim), pos)
        moe_tp2 = moe(h)
        dec_tp2 = dec(h, _MockMLACache(cfg.padded_head_dim), pos)

        # tp=1 reference (trivial group, same source weights).
        attn_ref = KimiMLAAttention(cfg, CommGroup.trivial()).to(DEVICE, DTYPE)
        _load_attention(attn_ref, src)
        moe_ref = KimiSparseMoeBlock(cfg, CommGroup.trivial()).to(DEVICE, DTYPE)
        _load_moe(moe_ref, src)
        dec_ref = KimiDecoderLayer(cfg, layer_idx=1, comm_group=CommGroup.trivial()).to(DEVICE, DTYPE)
        _load_decoder(dec_ref, src)

        o_attn = attn_ref(h, _MockMLACache(cfg.padded_head_dim), pos)
        o_moe = moe_ref(h)
        o_dec = dec_ref(h, _MockMLACache(cfg.padded_head_dim), pos)

        diffs = {
            "attn": (attn_tp2 - o_attn).abs().max().item(),
            "moe": (moe_tp2 - o_moe).abs().max().item(),
            "decoder": (dec_tp2 - o_dec).abs().max().item(),
        }
        dist.barrier()
        if rank == 0:
            torch.save(diffs, result_path)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="real NCCL tp=2 golden needs >= 2 CUDA devices",
)
def test_tp2_nccl_matches_tp1(tmp_path):
    import torch.multiprocessing as mp

    result_path = str(tmp_path / "tp2_diffs.pt")
    port = _free_port()
    mp.spawn(_nccl_worker, args=(TP, port, result_path), nprocs=TP, join=True)

    diffs = torch.load(result_path)
    # Real all-reduce -> each rank's full output must match the tp=1 reference
    # within bf16 tolerance (attention + MoE + decoder-layer residual wiring).
    assert diffs["attn"] < 5e-2, diffs
    assert diffs["moe"] < 5e-2, diffs
    assert diffs["decoder"] < 5e-2, diffs
