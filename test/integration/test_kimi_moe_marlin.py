import pytest
import torch

from mstar.model.kimi_k2_7._testing import fake_quantize_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="Marlin-backed KimiSparseMoeBlock golden needs a CUDA GPU with sm80+",
)

DEVICE = "cuda"
GROUP_SIZE = 32
PACK_FACTOR = 8


def _quantize_stack(weight):
    E, N, K = weight.shape
    packed = torch.empty((E, N, K // PACK_FACTOR), dtype=torch.int32, device=DEVICE)
    scale = torch.empty((E, N, K // GROUP_SIZE), dtype=torch.bfloat16, device=DEVICE)
    deq = torch.empty((E, N, K), dtype=torch.bfloat16, device=DEVICE)
    for e in range(E):
        p, s, d = fake_quantize_weight(
            weight[e], num_bits=4, group_size=GROUP_SIZE, symmetric=True,
            scale_dtype=torch.bfloat16,
        )
        packed[e], scale[e], deq[e] = p.to(DEVICE), s.to(DEVICE), d.to(DEVICE)
    return packed, scale, deq


def _build_block():
    from mstar.model.kimi_k2_7.components.moe import KimiSparseMoeBlock
    from mstar.model.kimi_k2_7.config import KimiK2Config

    torch.manual_seed(0)
    cfg = KimiK2Config.reduced_marlin()
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
    w1_packed, w1_scale, w1_deq = _quantize_stack(w1)
    w2_packed, w2_scale, w2_deq = _quantize_stack(w2)
    with torch.no_grad():
        block.experts.gate_up_proj_packed.data = w1_packed
        block.experts.gate_up_proj_scale.data = w1_scale
        block.experts.down_proj_packed.data = w2_packed
        block.experts.down_proj_scale.data = w2_scale
    return cfg, block, (w1_deq, w2_deq)


def test_marlin_block_matches_bf16_reference():
    from mstar.utils.fused_moe.runner import fused_experts

    cfg, block, (w1_deq, w2_deq) = _build_block()
    H = cfg.hidden_size
    x = (torch.randn(5, H, device=DEVICE) * 0.5).to(torch.bfloat16)

    # Build the bf16 reference before Marlin repack frees packed params.
    with torch.no_grad():
        topk_w, topk_ids = block.gate(x)
        routed_ref = fused_experts(x, w1_deq, w2_deq, topk_w.to(x.dtype), topk_ids)
        ref = (routed_ref + block.shared_expert(x)).view(x.shape)

    block.process_weights_after_loading(torch.device(DEVICE))
    assert block._use_marlin, "reduced_marlin config should select the Marlin backend"
    assert block.experts.gate_up_proj_packed.numel() == 0

    with torch.no_grad():
        out = block(x)

    assert out.shape == x.shape and out.dtype == torch.bfloat16
    cos = torch.nn.functional.cosine_similarity(
        out.flatten().float(), ref.flatten().float(), dim=0
    ).item()
    rel_l2 = ((out - ref).float().norm() / ref.float().norm()).item()
    assert cos > 0.999, f"cosine vs bf16 reference too low: {cos}"
    assert rel_l2 < 0.02, f"relative-L2 vs bf16 reference too high: {rel_l2}"


def test_forced_marlin_raises_on_illegal_shapes():
    from mstar.model.kimi_k2_7.components.moe import KimiSparseMoeBlock
    from mstar.model.kimi_k2_7.config import KimiK2Config

    cfg = KimiK2Config.reduced_quantized_inkernel()
    cfg.quant_kernel = "marlin"
    with torch.device("meta"):
        block = KimiSparseMoeBlock(cfg)
    block = block.to(torch.bfloat16)
    block.to_empty(device=DEVICE)
    with pytest.raises(RuntimeError, match="Marlin is ineligible"):
        block.process_weights_after_loading(torch.device(DEVICE))


def test_triton_backend_still_selected_when_forced():
    from mstar.model.kimi_k2_7.components.moe import KimiSparseMoeBlock
    from mstar.model.kimi_k2_7.config import KimiK2Config

    cfg = KimiK2Config.reduced_marlin()
    cfg.quant_kernel = "triton"
    with torch.device("meta"):
        block = KimiSparseMoeBlock(cfg)
    block = block.to(torch.bfloat16)
    block.to_empty(device=DEVICE)
    block.process_weights_after_loading(torch.device(DEVICE))
    assert not block._use_marlin
    # Packed params are retained for the Triton kernel (not freed).
    assert block.experts.gate_up_proj_packed.numel() > 0
