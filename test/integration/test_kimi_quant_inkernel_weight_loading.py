import pytest
import torch

from mstar.distributed.communication import CommGroup
from mstar.model.kimi_k2_7._testing import fake_quantize_weight
from mstar.model.kimi_k2_7.components.causal_lm import KimiForCausalLM
from mstar.model.kimi_k2_7.components.moe import KimiSparseMoeBlock
from mstar.model.kimi_k2_7.config import KimiK2Config
from mstar.model.loader import load_weights as driver_load_weights

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Kimi packed-expert weight-loading golden needs a GPU (W4A16 fused expert GEMM)",
)

DEVICE = "cuda"
DTYPE = torch.bfloat16


def _sdpa_causal(q, k, v, scale):
    qt, kt, vt = (t.transpose(0, 1).float() for t in (q, k, v))
    T = q.shape[0]
    causal = torch.triu(torch.full((T, T), float("-inf"), device=q.device), diagonal=1)
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
        # Wide init exercises the top-nibble sign path.
        mlp.experts.gate_up_proj.data.normal_(0, 0.2)
        mlp.experts.down_proj.data.normal_(0, 0.2)
        mlp.shared_expert.gate_up_proj.weight.data.normal_(0, 0.05)
        mlp.shared_expert.down_proj.weight.data.normal_(0, 0.05)
    else:
        mlp.gate_up_proj.weight.data.normal_(0, 0.05)
        mlp.down_proj.weight.data.normal_(0, 0.05)


def _build_reference(cfg):
    model = KimiForCausalLM(cfg).to(device=DEVICE, dtype=DTYPE)
    model.model.embed_tokens.weight.data.normal_(0, 0.05)
    model.model.norm.weight.data.normal_(1.0, 0.02)
    model.lm_head.weight.data.normal_(0, 0.02)
    for layer in model.model.layers:
        _fill_layer(layer, cfg)
    model.requires_grad_(False)
    return model.eval()


def _keep_bf16(key):
    if key == "lm_head.weight":
        return True
    return (
        key.endswith("norm.weight")
        or key.endswith("mlp.gate.weight")
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
    packed, scale, deq = fake_quantize_weight(
        view, num_bits=quant_cfg.num_bits, group_size=quant_cfg.group_size,
        symmetric=quant_cfg.symmetric, scale_dtype=DTYPE,
    )
    base = key[: -len(".weight")]
    sd[base + ".weight_packed"] = packed
    sd[base + ".weight_scale"] = scale
    view.copy_(deq.to(view.dtype))


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
    return {k: v.detach().cpu().clone().contiguous() for k, v in sd.items()}


def _build_loaded(cfg, checkpoint_dir):
    with torch.device("meta"):
        model = KimiForCausalLM(cfg)
    model = model.to(DTYPE)
    model.to_empty(device=DEVICE)
    loaded = driver_load_weights(model, checkpoint_dir, device=DEVICE)
    return model.eval(), loaded

def test_inkernel_weight_loading_and_forward_vs_dequant_on_load(tmp_path):
    from safetensors.torch import save_file

    torch.manual_seed(0)
    cfg_a = KimiK2Config.reduced_quantized()
    cfg_b = KimiK2Config.reduced_quantized_inkernel()
    assert cfg_b.moe_in_kernel_dequant and cfg_b.quantization_config is not None

    ref = _build_reference(cfg_a)
    assert not isinstance(ref.model.layers[0].mlp, KimiSparseMoeBlock)
    assert isinstance(ref.model.layers[1].mlp, KimiSparseMoeBlock)

    sd = _hf_quant_checkpoint(ref, cfg_a, cfg_a.quantization_config)
    assert any(k.endswith("mlp.experts.0.gate_proj.weight_packed") for k in sd)
    assert any(k.endswith("mlp.experts.0.down_proj.weight_packed") for k in sd)
    save_file(sd, str(tmp_path / "model.safetensors"))

    model_a, _ = _build_loaded(cfg_a, tmp_path)
    model_b, loaded_b = _build_loaded(cfg_b, tmp_path)

    all_params_b = set(dict(model_b.named_parameters()).keys())
    assert loaded_b == all_params_b, (
        f"unloaded: {all_params_b - loaded_b}; spurious: {loaded_b - all_params_b}")
    moe_prefix = "model.layers.1.mlp.experts."
    for suffix in ("gate_up_proj_packed", "gate_up_proj_scale",
                   "down_proj_packed", "down_proj_scale"):
        assert moe_prefix + suffix in all_params_b, f"missing {suffix}"
    assert moe_prefix + "gate_up_proj" not in all_params_b  # bf16 fused param gone
    assert moe_prefix + "down_proj" not in all_params_b

    # PyTorch .to(dtype) skips integer tensors; packed params must stay int32.
    experts_b = model_b.model.layers[1].mlp.experts
    assert experts_b.gate_up_proj_packed.dtype == torch.int32
    assert experts_b.down_proj_packed.dtype == torch.int32
    assert experts_b.gate_up_proj_scale.dtype == DTYPE
    assert experts_b.down_proj_scale.dtype == DTYPE

    assert model_b.model.layers[1].mlp.gate.e_score_correction_bias.dtype == torch.float32
    assert {n for n, _ in model_b.named_buffers()} == set()

    a_sd = dict(model_a.named_parameters())
    b_sd = dict(model_b.named_parameters())
    shared_keys = set(a_sd) & set(b_sd)
    assert "model.layers.1.mlp.gate.weight" in shared_keys
    for name in shared_keys:
        assert torch.equal(a_sd[name], b_sd[name]), f"shared param mismatch at {name}"

    T = 8
    ids = torch.randint(0, cfg_b.vocab_size, (T,), device=DEVICE)
    pos = torch.arange(T, device=DEVICE)
    with torch.no_grad():
        got = model_b(ids, _MockMLACache(cfg_b.padded_head_dim), pos)
        expected = model_a(ids, _MockMLACache(cfg_a.padded_head_dim), pos)
    assert got.shape == (T, cfg_b.vocab_size)
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)

class _NoCommGroup(CommGroup):

    def __init__(self, rank: int) -> None:
        super().__init__(my_global_rank=rank, my_group_rank=rank, group_members=[0, 1])

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return input_


def _packed_source(cfg, seed):
    g = torch.Generator().manual_seed(seed)
    E, H, I = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
    sh = I * cfg.n_shared_experts
    qc = cfg.quantization_config

    def rn(*shape, std=0.05, mean=0.0):
        return torch.randn(*shape, generator=g) * std + mean

    def quant_stack(w):
        packs, scales = [], []
        for e in range(w.shape[0]):
            pk, sc, _ = fake_quantize_weight(
                w[e], num_bits=qc.num_bits, group_size=qc.group_size,
                symmetric=qc.symmetric, scale_dtype=DTYPE)
            packs.append(pk)
            scales.append(sc)
        return torch.stack(packs), torch.stack(scales)

    gate = rn(E, I, H, std=0.2)
    up = rn(E, I, H, std=0.2)
    down = rn(E, H, I, std=0.2)
    return {
        "router_w": torch.randn(E, H, generator=g),
        "router_b": torch.randn(E, generator=g),
        "gate_packed": quant_stack(gate),
        "up_packed": quant_stack(up),
        "down_packed": quant_stack(down),
        "sh_gate": rn(sh, H), "sh_up": rn(sh, H), "sh_down": rn(H, sh),
    }


def _load_moe_packed(block, src):
    block.gate.weight.data = src["router_w"].to(DEVICE)
    block.gate.e_score_correction_bias.data = src["router_b"].to(DEVICE)
    gup_p, gup_s = block.experts.gate_up_proj_packed, block.experts.gate_up_proj_scale
    dwn_p, dwn_s = block.experts.down_proj_packed, block.experts.down_proj_scale
    gate_pk, gate_sc = src["gate_packed"]
    up_pk, up_sc = src["up_packed"]
    down_pk, down_sc = src["down_packed"]
    for e in range(block.num_experts):
        gup_p.weight_loader(gup_p, gate_pk[e].to(DEVICE), loaded_shard_id=f"gate:{e}")
        gup_p.weight_loader(gup_p, up_pk[e].to(DEVICE), loaded_shard_id=f"up:{e}")
        gup_s.weight_loader(gup_s, gate_sc[e].to(DEVICE), loaded_shard_id=f"gate:{e}")
        gup_s.weight_loader(gup_s, up_sc[e].to(DEVICE), loaded_shard_id=f"up:{e}")
        dwn_p.weight_loader(dwn_p, down_pk[e].to(DEVICE), loaded_shard_id=f"down:{e}")
        dwn_s.weight_loader(dwn_s, down_sc[e].to(DEVICE), loaded_shard_id=f"down:{e}")
    s = block.shared_expert
    s.gate_up_proj.weight.weight_loader(
        s.gate_up_proj.weight, src["sh_gate"].to(DEVICE, DTYPE), loaded_shard_id=0)
    s.gate_up_proj.weight.weight_loader(
        s.gate_up_proj.weight, src["sh_up"].to(DEVICE, DTYPE), loaded_shard_id=1)
    s.down_proj.weight.weight_loader(s.down_proj.weight, src["sh_down"].to(DEVICE, DTYPE))


def test_packed_moe_block_tp2_sim_matches_tp1():
    cfg = KimiK2Config.reduced_quantized_inkernel()
    src = _packed_source(cfg, seed=707)
    g = torch.Generator().manual_seed(808)
    h = (torch.randn(7, cfg.hidden_size, generator=g) * 0.1).to(DEVICE, DTYPE)

    ref = KimiSparseMoeBlock(cfg, CommGroup.trivial()).to(DEVICE, DTYPE)
    _load_moe_packed(ref, src)
    full_inter = ref.experts.gate_up_proj_packed.shape[1]  # 2*moe_inter
    out_ref = ref(h)

    partials = []
    for rank in range(2):
        block = KimiSparseMoeBlock(cfg, _NoCommGroup(rank)).to(DEVICE, DTYPE)
        assert block.experts.gate_up_proj_packed.shape[1] == full_inter // 2
        assert block.experts.down_proj_packed.dtype == torch.int32
        _load_moe_packed(block, src)
        partials.append(block(h))

    out_tp2 = partials[0] + partials[1]
    max_abs = (out_tp2 - out_ref).abs().max().item()
    assert max_abs < 5e-2, f"packed MoE tp2 vs tp1 max abs diff {max_abs}"
    torch.testing.assert_close(out_tp2, out_ref, rtol=2e-2, atol=2e-2)
