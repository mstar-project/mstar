"""Merge check: the absorbed-MLA and Marlin-MoE post-load hooks COMPOSE.

The MLA weight-absorption work and the Marlin W4A16 MoE work were developed in
separate silos; both hang a ``process_weights_after_loading`` finalizer off the
same generic walker (``mstar.model.components.quantization.process_weights_after_loading``,
called once in ``kimi_model.py::_create_submodule``). This test builds a tree
holding BOTH a weight-absorbed ``KimiMLAAttention`` and a Marlin-backed
``KimiSparseMoeBlock`` and runs that single walker pass, asserting it finalizes
BOTH: the attention's ``w_kc``/``w_vc``/``fused_qkv_a_proj`` and the MoE's Marlin
repack (+ freed packed params). This is the production-default combination
(``k27_code`` => ``mla_absorb=True`` + ``quant_kernel="auto"`` => Marlin on sm80+),
which neither silo validated together. A Marlin-MoE forward post-walker confirms
the block still runs in the combined module. (Reduced dims => the attention uses
the SDPA-over-latent path, not the dim-locked FlashInfer MLA kernel; the kernel +
Marlin combined forward at real dims is covered by the TP8 serve smoke.)

Run:  pytest test/integration/test_kimi_mla_absorb_marlin_merge.py -v
"""
import pytest
import torch
from torch import nn

from mstar.model.components.quantization import process_weights_after_loading
from mstar.model.kimi_k2_7._testing import fake_quantize_weight
from mstar.model.kimi_k2_7.config import KimiK2Config

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8),
    reason="the Marlin backend needs a CUDA GPU with sm80+",
)

DEVICE = "cuda"
GROUP_SIZE = 32
PACK_FACTOR = 8


def _quantize_stack(weight):
    """Per-expert fake-quantize a bf16 stack to packed int32 + bf16 scales."""
    E, N, K = weight.shape
    packed = torch.empty((E, N, K // PACK_FACTOR), dtype=torch.int32, device=DEVICE)
    scale = torch.empty((E, N, K // GROUP_SIZE), dtype=torch.bfloat16, device=DEVICE)
    for e in range(E):
        p, s, _d = fake_quantize_weight(
            weight[e], num_bits=4, group_size=GROUP_SIZE, symmetric=True,
            scale_dtype=torch.bfloat16,
        )
        packed[e], scale[e] = p.to(DEVICE), s.to(DEVICE)
    return packed, scale


def _build_marlin_moe(cfg):
    """A tp=1 Marlin-legal KimiSparseMoeBlock with synthetic packed experts
    (mirrors test_kimi_moe_marlin._build_block)."""
    from mstar.model.kimi_k2_7.components.moe import KimiSparseMoeBlock

    with torch.device("meta"):
        block = KimiSparseMoeBlock(cfg)
    block = block.to(torch.bfloat16)
    block.to_empty(device=DEVICE)
    for p in block.parameters():
        if p.dtype.is_floating_point:
            with torch.no_grad():
                p.copy_(torch.randn_like(p) * 0.1)
    E, H, I = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
    w1 = (torch.randn(E, 2 * I, H, device=DEVICE) * 0.3).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, device=DEVICE) * 0.3).to(torch.bfloat16)
    w1_packed, w1_scale = _quantize_stack(w1)
    w2_packed, w2_scale = _quantize_stack(w2)
    with torch.no_grad():
        block.experts.gate_up_proj_packed.data = w1_packed
        block.experts.gate_up_proj_scale.data = w1_scale
        block.experts.down_proj_packed.data = w2_packed
        block.experts.down_proj_scale.data = w2_scale
    return block


def _build_absorbed_attn(cfg):
    """A materialized weight-absorbed KimiMLAAttention (random weights on CUDA)."""
    from mstar.model.kimi_k2_7.components.attention import KimiMLAAttention

    torch.manual_seed(0)
    return KimiMLAAttention(cfg).to(device=DEVICE, dtype=torch.bfloat16)


def test_generic_walker_composes_absorbed_mla_and_marlin_moe():
    """One ``process_weights_after_loading`` pass finalizes BOTH the absorbed MLA
    projections and the Marlin MoE repack — the merged production default."""
    cfg = KimiK2Config.reduced_marlin()
    cfg.mla_absorb = True  # absorbed attention + Marlin experts in one config

    attn = _build_absorbed_attn(cfg)
    moe = _build_marlin_moe(cfg)
    root = nn.Module()
    root.add_module("attn", attn)
    root.add_module("moe", moe)

    # --- pre-walker: neither finalizer has run ---
    assert attn.mla_absorb
    assert attn.w_kc is None and attn.w_vc is None and attn.fused_qkv_a_proj_weight is None
    assert moe.experts.gate_up_proj_packed.numel() > 0
    assert not getattr(moe, "_use_marlin", False)

    # --- one generic walker pass (the exact call kimi_model._create_submodule makes) ---
    process_weights_after_loading(root, torch.device(DEVICE))

    # --- post-walker: absorbed-MLA finalizer ran ---
    assert attn.w_kc is not None, "walker did not build the absorbed w_kc"
    assert attn.w_vc is not None, "walker did not build the absorbed w_vc"
    assert attn.fused_qkv_a_proj_weight is not None, "walker did not build fused_qkv_a_proj"

    # --- post-walker: Marlin-MoE finalizer ran ---
    assert moe._use_marlin, "walker did not select the Marlin backend"
    assert moe.experts.gate_up_proj_packed.numel() == 0, "packed experts not freed after repack"

    # --- the Marlin MoE forward still runs in the combined module ---
    x = torch.randn(4, cfg.hidden_size, device=DEVICE, dtype=torch.bfloat16) * 0.1
    with torch.no_grad():
        out = moe(x)
    assert out.shape == x.shape and torch.isfinite(out).all()
