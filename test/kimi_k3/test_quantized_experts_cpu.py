"""Quantized (MXFP4) expert mode on CPU: the packed parameters load through the same
weight names the checkpoint uses, and the model reproduces a bf16 model whose experts
were dequantized from the same codes."""
import torch
from safetensors.torch import load_file

from mstar.model.kimi_k3.components.language_model import KimiK3ForCausalLM
from mstar.model.kimi_k3.config import KimiK3Config
from mstar.model.kimi_k3.reference.mxfp4 import dequant_mxfp4, quant_mxfp4


def _build(cfg, quantized):
    with torch.device("meta"):
        m = KimiK3ForCausalLM(cfg, quantized_experts=quantized)
    m = m.to(torch.float32)
    for n, p in m.named_parameters():
        if n.endswith(("_packed", "_scale")):
            p.data = p.data.to(torch.uint8)
    m.to_empty(device="cpu")
    return m.eval()


def test_quantized_mode_matches_dequantized_bf16(tiny_dir):
    cfg = KimiK3Config.from_hf_dir(tiny_dir).text
    sd = load_file(str(tiny_dir / "model.safetensors"))
    quant_weights, deq_weights = [], []
    for name, t in sd.items():
        if ".block_sparse_moe.experts." in name and name.endswith(".weight"):
            packed, scale = quant_mxfp4(t.float())
            base = name[: -len(".weight")]
            quant_weights += [(base + ".weight_packed", packed), (base + ".weight_scale", scale)]
            deq_weights.append((name, dequant_mxfp4(packed, scale, dtype=torch.float32)))
        else:
            quant_weights.append((name, t))
            deq_weights.append((name, t))
    mq = _build(cfg, quantized=True)
    md = _build(cfg, quantized=False)
    loaded_q = mq.load_weights(iter(quant_weights))
    loaded_d = md.load_weights(iter(deq_weights))
    assert not (set(dict(mq.named_parameters())) - loaded_q)
    assert not (set(dict(md.named_parameters())) - loaded_d)
    moe = mq.model.layers[1].block_sparse_moe
    latent, inter = cfg.routed_expert_hidden_size, cfg.moe_intermediate_size
    assert moe.experts.gate_up_packed.dtype == torch.uint8
    assert moe.experts.gate_up_packed.shape == (8, 2 * inter, latent // 2)
    assert moe.experts.down_scale.shape == (8, latent, inter // 32)
    torch.manual_seed(0)
    ids = torch.randint(0, 1000, (7,))
    with torch.no_grad():
        lq, _ = mq.forward_dense(ids)
        ld, _ = md.forward_dense(ids)
    torch.testing.assert_close(lq, ld, rtol=1e-5, atol=1e-4)
    # a packed expert loads through the packed/scale loaders exactly (round trip)
    w13, w2 = moe.dequantized_experts()
    e0 = dict(deq_weights)["language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight"]
    torch.testing.assert_close(w13[0, :inter].float(), e0, rtol=0, atol=0)
    # a blanket dtype sweep (what the engine does) keeps fp32 gates and uint8 experts
    mq.to(torch.bfloat16)
    assert mq.model.layers[0].self_attn.A_log.dtype == torch.float32
    assert moe.experts.gate_up_packed.dtype == torch.uint8 and moe.gate.e_score_correction_bias.dtype == torch.float32
