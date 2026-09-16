"""Quantized (MXFP4) expert mode on CPU: the packed parameters load through the same
weight names the checkpoint uses, and the model reproduces a bf16 model whose experts
were dequantized from the same codes; also with the packed experts placed whole on
expert-parallel ranks (the reference per-expert loop skipping the other ranks' assignments)."""
import threading

import torch
from safetensors.torch import load_file

from mstar.model.kimi_k3.components.language_model import KimiK3ForCausalLM
from mstar.model.kimi_k3.config import KimiK3Config
from mstar.model.kimi_k3.reference.mxfp4 import dequant_mxfp4, quant_mxfp4


def _build(cfg, quantized, group=None, ep=1):
    with torch.device("meta"):
        m = KimiK3ForCausalLM(cfg, comm_group=group, quantized_experts=quantized, moe_ep_size=ep)
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


def _quantized_stream(tiny_dir):
    sd = load_file(str(tiny_dir / "model.safetensors"))
    out = []
    for name, t in sd.items():
        if ".block_sparse_moe.experts." in name and name.endswith(".weight"):
            packed, scale = quant_mxfp4(t.float())
            base = name[: -len(".weight")]
            out += [(base + ".weight_packed", packed), (base + ".weight_scale", scale)]
        else:
            out.append((name, t))
    return out


def test_quantized_expert_parallel_matches_single_rank(tiny_dir):
    """EP2 over two threaded ranks (fake barrier collectives): each rank holds half of the
    packed experts, computes its partial through the reference loop and the all-reduce over the
    latent combines them; the logits match the single-rank quantized model."""
    from test_tp_cpu import FakeCommGroup, _Mailbox

    cfg = KimiK3Config.from_hf_dir(tiny_dir).text
    weights = _quantized_stream(tiny_dir)
    ref = _build(cfg, quantized=True)
    ref.load_weights(iter(weights))
    mb = _Mailbox(2)
    ranks = [_build(cfg, quantized=True, group=FakeCommGroup(r, mb), ep=2) for r in range(2)]
    for m in ranks:
        loaded = m.load_weights(iter(weights))
        assert not (set(dict(m.named_parameters())) - loaded)
    e, inter, latent = cfg.num_experts, cfg.moe_intermediate_size, cfg.routed_expert_hidden_size
    for r, m in enumerate(ranks):
        moe = m.model.layers[1].block_sparse_moe
        assert moe.experts.gate_up_packed.shape == (e // 2, 2 * inter, latent // 2)
        assert moe.sharding.expert_offset == r * (e // 2)
        ref_moe = ref.model.layers[1].block_sparse_moe
        lo = moe.sharding.expert_offset
        assert torch.equal(moe.experts.down_packed, ref_moe.experts.down_packed[lo:lo + e // 2])
    torch.manual_seed(0)
    ids = torch.randint(0, 1000, (7,))
    with torch.no_grad():
        ref_logits, _ = ref.forward_dense(ids)
    outs, errs = [None, None], [None, None]

    def run(r):
        try:
            with torch.no_grad():
                outs[r] = ranks[r].forward_dense(ids)[0]
        except BaseException as ex:
            errs[r] = ex
            mb.barrier.abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(x is None for x in errs), [repr(x) for x in errs if x is not None]
    assert torch.equal(outs[0], outs[1])
    torch.testing.assert_close(outs[0], ref_logits, rtol=1e-4, atol=1e-3)
    assert torch.equal(outs[0].argmax(-1), ref_logits.argmax(-1))
