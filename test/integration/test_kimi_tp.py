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


class _NoCommGroup(CommGroup):
    def __init__(self, rank: int) -> None:
        super().__init__(my_global_rank=rank, my_group_rank=rank, group_members=[0, 1])

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return input_


class _MockMLACache:

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


def _source_weights(cfg: KimiK2Config, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    H = cfg.num_attention_heads
    E, Hd, I = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
    sh = I * cfg.n_shared_experts

    def rn(*shape, std=0.03, mean=0.0):
        return torch.randn(*shape, generator=g) * std + mean

    return {
        "q_a": rn(cfg.q_lora_rank, cfg.hidden_size),
        "q_a_norm": rn(cfg.q_lora_rank, std=0.02, mean=1.0),
        "q_b": rn(H * cfg.qk_head_dim, cfg.q_lora_rank),
        "kv_a": rn(cfg.kv_lora_rank + cfg.qk_rope_head_dim, cfg.hidden_size),
        "kv_a_norm": rn(cfg.kv_lora_rank, std=0.02, mean=1.0),
        "kv_b": rn(H * (cfg.qk_nope_head_dim + cfg.v_head_dim), cfg.kv_lora_rank),
        "o": rn(cfg.hidden_size, H * cfg.v_head_dim),
        "router_w": torch.randn(E, Hd, generator=g),
        "router_b": torch.randn(E, generator=g),
        "gate": rn(E, I, Hd, std=0.05),
        "up": rn(E, I, Hd, std=0.05),
        "down": rn(E, Hd, I, std=0.05),
        "sh_gate": rn(sh, Hd, std=0.05),
        "sh_up": rn(sh, Hd, std=0.05),
        "sh_down": rn(Hd, sh, std=0.05),
        "in_ln": rn(Hd, std=0.02, mean=1.0),
        "post_ln": rn(Hd, std=0.02, mean=1.0),
    }


def _load_attention(attn: KimiMLAAttention, src: dict) -> None:
    attn.q_a_proj.weight.data.copy_(src["q_a"].to(DEVICE, DTYPE))
    attn.q_a_layernorm.weight.data.copy_(src["q_a_norm"].to(DEVICE, DTYPE))
    attn.q_b_proj.weight.weight_loader(attn.q_b_proj.weight, src["q_b"].to(DEVICE, DTYPE))
    attn.kv_a_proj_with_mqa.weight.data.copy_(src["kv_a"].to(DEVICE, DTYPE))
    attn.kv_a_layernorm.weight.data.copy_(src["kv_a_norm"].to(DEVICE, DTYPE))
    attn.kv_b_proj.weight.weight_loader(attn.kv_b_proj.weight, src["kv_b"].to(DEVICE, DTYPE))
    attn.o_proj.weight.weight_loader(attn.o_proj.weight, src["o"].to(DEVICE, DTYPE))


def _load_moe(block: KimiSparseMoeBlock, src: dict) -> None:
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

def test_mla_attention_tp2_sim_matches_tp1():
    cfg = KimiK2Config.reduced()
    src = _source_weights(cfg, seed=101)
    h, pos = _inputs(cfg, num_tokens=6, seed=202)

    ref = KimiMLAAttention(cfg, CommGroup.trivial()).to(DEVICE, DTYPE)
    _load_attention(ref, src)
    assert ref.num_heads == cfg.num_attention_heads
    out_ref = ref(h, _MockMLACache(cfg.padded_head_dim), pos)

    partials = []
    for rank in range(TP):
        attn = KimiMLAAttention(cfg, _NoCommGroup(rank)).to(DEVICE, DTYPE)
        assert attn.num_heads == cfg.num_attention_heads // TP
        _load_attention(attn, src)
        partials.append(attn(h, _MockMLACache(cfg.padded_head_dim), pos))

    out_tp2 = partials[0] + partials[1]
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
        assert block.experts.gate_up_proj.shape[1] == full_inter // TP
        _load_moe(block, src)
        partials.append(block(h))

    out_tp2 = partials[0] + partials[1]
    max_abs = (out_tp2 - out_ref).abs().max().item()
    assert max_abs < 5e-2, f"MoE tp2 vs tp1 max abs diff {max_abs}"
    torch.testing.assert_close(out_tp2, out_ref, rtol=2e-2, atol=2e-2)


def test_tp2_sim_is_stable_across_repeats():
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

def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _nccl_worker(rank: int, world_size: int, port: int, result_path: str) -> None:
    import torch.distributed as dist

    os.environ.setdefault("NCCL_IB_DISABLE", "1")
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
        cg.device_group = None
        cg.initialized = True

        attn = KimiMLAAttention(cfg, cg).to(DEVICE, DTYPE)
        _load_attention(attn, src)
        moe = KimiSparseMoeBlock(cfg, cg).to(DEVICE, DTYPE)
        _load_moe(moe, src)
        dec = KimiDecoderLayer(cfg, layer_idx=1, comm_group=cg).to(DEVICE, DTYPE)
        _load_decoder(dec, src)
        assert isinstance(dec.mlp, KimiSparseMoeBlock)

        attn_tp2 = attn(h, _MockMLACache(cfg.padded_head_dim), pos)
        moe_tp2 = moe(h)
        dec_tp2 = dec(h, _MockMLACache(cfg.padded_head_dim), pos)

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
    assert diffs["attn"] < 5e-2, diffs
    assert diffs["moe"] < 5e-2, diffs
    assert diffs["decoder"] < 5e-2, diffs
