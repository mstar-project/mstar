import pytest
import torch

from mstar.model.kimi_k2_7._testing import fake_quantize_weight
from mstar.model.kimi_k2_7.components.causal_lm import KimiForCausalLM
from mstar.model.kimi_k2_7.components.moe import KimiSparseMoeBlock
from mstar.model.kimi_k2_7.config import KimiK2Config
from mstar.model.loader import load_weights as driver_load_weights

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Kimi quant weight-loading golden needs a GPU (RMSNorm + fused expert GEMM)",
)

DEVICE = "cuda"


def _sdpa_causal(q, k, v, scale):
    qt, kt, vt = (t.transpose(0, 1).float() for t in (q, k, v))
    T = q.shape[0]
    causal = torch.triu(
        torch.full((T, T), float("-inf"), device=q.device), diagonal=1)
    attn = (torch.einsum("hqd,hkd->hqk", qt, kt) * scale + causal).softmax(-1)
    return torch.einsum("hqk,hkd->hqd", attn, vt).transpose(0, 1).to(q.dtype)


class _MockMLACache:
    def __init__(self, head_dim):
        self.scale = head_dim ** -0.5

    def set_layer_idx(self, _i):
        pass

    def advance_seq_lens(self, *_a, **_k):
        pass

    def run_attention(self, q, k, v):
        return _sdpa_causal(q, k, v, self.scale)

def _fill_layer(layer, cfg):
    a = layer.self_attn
    for lin in (a.q_a_proj, a.q_b_proj, a.kv_a_proj_with_mqa, a.kv_b_proj, a.o_proj):
        lin.weight.data.normal_(0, 0.03)
    for norm in (a.q_a_layernorm, a.kv_a_layernorm):
        norm.weight.data.normal_(1.0, 0.02)
    layer.input_layernorm.weight.data.normal_(1.0, 0.02)
    layer.post_attention_layernorm.weight.data.normal_(1.0, 0.02)
    mlp = layer.mlp
    if isinstance(mlp, KimiSparseMoeBlock):
        mlp.gate.weight.data.normal_(0, 1)
        mlp.gate.e_score_correction_bias.data = torch.randn(
            cfg.n_routed_experts, device=DEVICE, dtype=torch.float32)
        mlp.experts.gate_up_proj.data.normal_(0, 0.05)
        mlp.experts.down_proj.data.normal_(0, 0.05)
        mlp.shared_expert.gate_up_proj.weight.data.normal_(0, 0.05)
        mlp.shared_expert.down_proj.weight.data.normal_(0, 0.05)
    else:
        mlp.gate_up_proj.weight.data.normal_(0, 0.05)
        mlp.down_proj.weight.data.normal_(0, 0.05)


def _build_reference(cfg):
    model = KimiForCausalLM(cfg).to(device=DEVICE, dtype=torch.bfloat16)
    model.model.embed_tokens.weight.data.normal_(0, 0.05)
    model.model.norm.weight.data.normal_(1.0, 0.02)
    model.lm_head.weight.data.normal_(0, 0.02)
    for layer in model.model.layers:
        _fill_layer(layer, cfg)
    # Disable grad so in-place dequant write-back into leaf params is allowed.
    model.requires_grad_(False)
    return model.eval()

def _keep_bf16(key):
    if key == "lm_head.weight":
        return True
    return (
        key.endswith("norm.weight")            # all RMSNorms incl. *_layernorm
        or key.endswith("mlp.gate.weight")     # MoE router — never quantized
        or key.endswith("e_score_correction_bias")
        or key.endswith("embed_tokens.weight")
    )


def _emit(sd, key, view, quant_cfg):
    eligible = (
        not _keep_bf16(key)
        and view.dim() == 2
        and view.shape[-1] % quant_cfg.group_size == 0
    )
    if not eligible:
        sd[key] = view
        return
    # Match compressed-tensors checkpoints: bf16 scale drives the reference dequant.
    packed, scale, deq = fake_quantize_weight(
        view, num_bits=quant_cfg.num_bits, group_size=quant_cfg.group_size,
        symmetric=quant_cfg.symmetric, scale_dtype=torch.bfloat16,
    )
    base = key[: -len(".weight")]
    sd[base + ".weight_packed"] = packed
    sd[base + ".weight_scale"] = scale
    view.copy_(deq.to(view.dtype))  # reference now holds the dequantized weight


def _hf_quant_checkpoint(model, cfg, quant_cfg):
    inter = cfg.intermediate_size
    moe_inter = cfg.moe_intermediate_size
    shared_inter = cfg.moe_intermediate_size * cfg.n_shared_experts
    m = model.model
    sd = {}
    _emit(sd, "model.embed_tokens.weight", m.embed_tokens.weight, quant_cfg)
    for i, layer in enumerate(m.layers):
        p = f"model.layers.{i}."
        a = layer.self_attn
        _emit(sd, p + "self_attn.q_a_proj.weight", a.q_a_proj.weight, quant_cfg)
        _emit(sd, p + "self_attn.q_a_layernorm.weight", a.q_a_layernorm.weight, quant_cfg)
        _emit(sd, p + "self_attn.q_b_proj.weight", a.q_b_proj.weight, quant_cfg)
        _emit(sd, p + "self_attn.kv_a_proj_with_mqa.weight",
              a.kv_a_proj_with_mqa.weight, quant_cfg)
        _emit(sd, p + "self_attn.kv_a_layernorm.weight", a.kv_a_layernorm.weight, quant_cfg)
        _emit(sd, p + "self_attn.kv_b_proj.weight", a.kv_b_proj.weight, quant_cfg)
        _emit(sd, p + "self_attn.o_proj.weight", a.o_proj.weight, quant_cfg)
        _emit(sd, p + "input_layernorm.weight", layer.input_layernorm.weight, quant_cfg)
        _emit(sd, p + "post_attention_layernorm.weight",
              layer.post_attention_layernorm.weight, quant_cfg)
        mlp = layer.mlp
        if isinstance(mlp, KimiSparseMoeBlock):
            _emit(sd, p + "mlp.gate.weight", mlp.gate.weight, quant_cfg)
            _emit(sd, p + "mlp.gate.e_score_correction_bias",
                  mlp.gate.e_score_correction_bias, quant_cfg)
            gup, dwn = mlp.experts.gate_up_proj, mlp.experts.down_proj
            for e in range(cfg.n_routed_experts):
                _emit(sd, p + f"mlp.experts.{e}.gate_proj.weight",
                      gup[e, :moe_inter, :], quant_cfg)
                _emit(sd, p + f"mlp.experts.{e}.up_proj.weight",
                      gup[e, moe_inter:, :], quant_cfg)
                _emit(sd, p + f"mlp.experts.{e}.down_proj.weight", dwn[e], quant_cfg)
            sh = mlp.shared_expert
            _emit(sd, p + "mlp.shared_experts.gate_proj.weight",
                  sh.gate_up_proj.weight[:shared_inter], quant_cfg)
            _emit(sd, p + "mlp.shared_experts.up_proj.weight",
                  sh.gate_up_proj.weight[shared_inter:], quant_cfg)
            _emit(sd, p + "mlp.shared_experts.down_proj.weight", sh.down_proj.weight, quant_cfg)
        else:
            _emit(sd, p + "mlp.gate_proj.weight", mlp.gate_up_proj.weight[:inter], quant_cfg)
            _emit(sd, p + "mlp.up_proj.weight", mlp.gate_up_proj.weight[inter:], quant_cfg)
            _emit(sd, p + "mlp.down_proj.weight", mlp.down_proj.weight, quant_cfg)
    _emit(sd, "model.norm.weight", m.norm.weight, quant_cfg)
    _emit(sd, "lm_head.weight", model.lm_head.weight, quant_cfg)
    # Clone to cpu + break storage aliasing (safetensors rejects shared storage).
    return {k: v.detach().cpu().clone().contiguous() for k, v in sd.items()}


def _build_loaded(cfg, checkpoint_dir):
    with torch.device("meta"):
        model = KimiForCausalLM(cfg)
    model = model.to(torch.bfloat16)
    model.to_empty(device=DEVICE)
    loaded = driver_load_weights(model, checkpoint_dir, device=DEVICE)
    return model.eval(), loaded


def test_quant_weight_loading_roundtrip_and_forward(tmp_path):
    from safetensors.torch import save_file

    torch.manual_seed(0)
    cfg = KimiK2Config.reduced_quantized()
    assert cfg.quantization_config is not None
    ref = _build_reference(cfg)
    assert not isinstance(ref.model.layers[0].mlp, KimiSparseMoeBlock)
    assert isinstance(ref.model.layers[1].mlp, KimiSparseMoeBlock)

    sd = _hf_quant_checkpoint(ref, cfg, cfg.quantization_config)

    assert any(k.endswith("mlp.experts.0.gate_proj.weight_packed") for k in sd)
    assert any(k.endswith("self_attn.o_proj.weight_packed") for k in sd)
    assert "lm_head.weight" in sd and "lm_head.weight_packed" not in sd
    assert "model.embed_tokens.weight" in sd

    save_file(sd, str(tmp_path / "model.safetensors"))
    model, loaded = _build_loaded(cfg, tmp_path)

    all_params = set(dict(model.named_parameters()).keys())
    assert loaded == all_params, (
        f"unloaded: {all_params - loaded}; spurious: {loaded - all_params}")

    ref_sd = dict(ref.named_parameters())
    for name, param in model.named_parameters():
        assert torch.equal(param, ref_sd[name]), f"mismatch at {name}"

    bias = model.model.layers[1].mlp.gate.e_score_correction_bias
    assert bias.dtype == torch.float32
    assert {n for n, _ in model.named_buffers()} == set()

    T = 8
    ids = torch.randint(0, cfg.vocab_size, (T,), device=DEVICE)
    pos = torch.arange(T, device=DEVICE)
    with torch.no_grad():
        got = model(ids, _MockMLACache(cfg.padded_head_dim), pos)
        expected = ref(ids, _MockMLACache(cfg.padded_head_dim), pos)
    assert got.shape == (T, cfg.vocab_size)
    torch.testing.assert_close(got, expected, rtol=1e-3, atol=1e-3)
