import pytest
import torch

from mstar.model.kimi_k2_7.components.causal_lm import KimiForCausalLM
from mstar.model.kimi_k2_7.components.moe import KimiSparseMoeBlock
from mstar.model.kimi_k2_7.config import KimiK2Config
from mstar.model.loader import load_weights as driver_load_weights

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="M5 weight-loading golden needs a GPU (RMSNorm + fused expert GEMM)",
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
        # In place so the router weight keeps the model dtype (bf16).
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
    return model.eval()


def _hf_checkpoint(model, cfg):
    inter = cfg.intermediate_size
    moe_inter = cfg.moe_intermediate_size
    shared_inter = cfg.moe_intermediate_size * cfg.n_shared_experts
    m = model.model
    sd = {"model.embed_tokens.weight": m.embed_tokens.weight}
    for i, layer in enumerate(m.layers):
        p = f"model.layers.{i}."
        a = layer.self_attn
        sd[p + "self_attn.q_a_proj.weight"] = a.q_a_proj.weight
        sd[p + "self_attn.q_a_layernorm.weight"] = a.q_a_layernorm.weight
        sd[p + "self_attn.q_b_proj.weight"] = a.q_b_proj.weight
        sd[p + "self_attn.kv_a_proj_with_mqa.weight"] = a.kv_a_proj_with_mqa.weight
        sd[p + "self_attn.kv_a_layernorm.weight"] = a.kv_a_layernorm.weight
        sd[p + "self_attn.kv_b_proj.weight"] = a.kv_b_proj.weight
        sd[p + "self_attn.o_proj.weight"] = a.o_proj.weight
        sd[p + "input_layernorm.weight"] = layer.input_layernorm.weight
        sd[p + "post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight
        mlp = layer.mlp
        if isinstance(mlp, KimiSparseMoeBlock):
            sd[p + "mlp.gate.weight"] = mlp.gate.weight
            sd[p + "mlp.gate.e_score_correction_bias"] = mlp.gate.e_score_correction_bias
            gup, dwn = mlp.experts.gate_up_proj, mlp.experts.down_proj
            for e in range(cfg.n_routed_experts):
                sd[p + f"mlp.experts.{e}.gate_proj.weight"] = gup[e, :moe_inter, :]
                sd[p + f"mlp.experts.{e}.up_proj.weight"] = gup[e, moe_inter:, :]
                sd[p + f"mlp.experts.{e}.down_proj.weight"] = dwn[e]
            sh = mlp.shared_expert
            sd[p + "mlp.shared_experts.gate_proj.weight"] = sh.gate_up_proj.weight[:shared_inter]
            sd[p + "mlp.shared_experts.up_proj.weight"] = sh.gate_up_proj.weight[shared_inter:]
            sd[p + "mlp.shared_experts.down_proj.weight"] = sh.down_proj.weight
        else:
            sd[p + "mlp.gate_proj.weight"] = mlp.gate_up_proj.weight[:inter]
            sd[p + "mlp.up_proj.weight"] = mlp.gate_up_proj.weight[inter:]
            sd[p + "mlp.down_proj.weight"] = mlp.down_proj.weight
    sd["model.norm.weight"] = m.norm.weight
    sd["lm_head.weight"] = model.lm_head.weight
    # Clone to cpu + break storage aliasing (safetensors rejects shared storage).
    return {k: v.detach().cpu().clone().contiguous() for k, v in sd.items()}


def _build_loaded(cfg, checkpoint_dir):
    with torch.device("meta"):
        model = KimiForCausalLM(cfg)
    model = model.to(torch.bfloat16)
    model.to_empty(device=DEVICE)
    loaded = driver_load_weights(model, checkpoint_dir, device=DEVICE)
    return model.eval(), loaded


def test_weight_loading_roundtrip_and_forward(tmp_path):
    from safetensors.torch import save_file

    torch.manual_seed(0)
    cfg = KimiK2Config.reduced()
    ref = _build_reference(cfg)
    assert not isinstance(ref.model.layers[0].mlp, KimiSparseMoeBlock)
    assert isinstance(ref.model.layers[1].mlp, KimiSparseMoeBlock)

    save_file(_hf_checkpoint(ref, cfg), str(tmp_path / "model.safetensors"))
    model, loaded = _build_loaded(cfg, tmp_path)

    all_params = set(dict(model.named_parameters()).keys())
    assert loaded == all_params, (
        f"unloaded: {all_params - loaded}; spurious: {loaded - all_params}")

    ref_sd = dict(ref.named_parameters())
    for name, param in model.named_parameters():
        assert torch.equal(param, ref_sd[name]), f"mismatch at {name}"

    bias = model.model.layers[1].mlp.gate.e_score_correction_bias
    assert bias.dtype == torch.float32

    # Derived buffers must not survive meta -> to_empty as uninitialized memory.
    buffer_names = {n for n, _ in model.named_buffers()}
    assert buffer_names == set(), f"unexpected buffers survived the load path: {buffer_names}"

    inter = cfg.intermediate_size
    d_gup = model.model.layers[0].mlp.gate_up_proj.weight
    r_gup = ref.model.layers[0].mlp.gate_up_proj.weight
    assert torch.equal(d_gup[:inter], r_gup[:inter])   # gate half
    assert torch.equal(d_gup[inter:], r_gup[inter:])   # up half
    mi = cfg.moe_intermediate_size
    l_gup = model.model.layers[1].mlp.experts.gate_up_proj
    r_egup = ref.model.layers[1].mlp.experts.gate_up_proj
    r_edwn = ref.model.layers[1].mlp.experts.down_proj
    for e in range(cfg.n_routed_experts):
        assert torch.equal(l_gup[e, :mi], r_egup[e, :mi])   # gate:e
        assert torch.equal(l_gup[e, mi:], r_egup[e, mi:])   # up:e
    assert torch.equal(model.model.layers[1].mlp.experts.down_proj, r_edwn)

    T = 8
    ids = torch.randint(0, cfg.vocab_size, (T,), device=DEVICE)
    pos = torch.arange(T, device=DEVICE)
    with torch.no_grad():
        got = model(ids, _MockMLACache(cfg.padded_head_dim), pos)
        expected = ref(ids, _MockMLACache(cfg.padded_head_dim), pos)
    assert got.shape == (T, cfg.vocab_size)
    torch.testing.assert_close(got, expected, rtol=1e-3, atol=1e-3)
