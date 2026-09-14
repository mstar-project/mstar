"""Reference implementations and checkpoint fixtures the Kimi integration tests
check the served path against.

Unchanged by the resource refactor — this is plain ``F.linear`` algebra and a
safetensors writer. It lives here because three of the tests built the same
checkpoint and the same DeepSeek-MLA reference inline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mstar.model.kimi_k2_7.components.causal_lm import KimiForCausalLM
from mstar.model.kimi_k2_7.components.moe import KimiSparseMoeBlock
from mstar.model.kimi_k2_7.components.rope import (
    _yarn_find_correction_range,
    _yarn_linear_ramp_mask,
    rotate_gptj,
    yarn_get_mscale,
)
from mstar.model.kimi_k2_7.config import ATTN, KV_CACHE, ROPE
from mstar.model.kimi_k2_7.kimi_model import KimiK2Model

DEVICE = torch.device("cuda")


# ── reference math ──────────────────────────────────────────────────────

def ref_rmsnorm(x, weight, eps):
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x32.to(x.dtype)


def sdpa_causal(q, k, v, scale):
    qt, kt, vt = (t.transpose(0, 1).float() for t in (q, k, v))  # (H,T,D)
    t = q.shape[0]
    causal = torch.triu(
        torch.full((t, t), float("-inf"), device=q.device), diagonal=1
    )
    attn = (torch.einsum("hqd,hkd->hqk", qt, kt) * scale + causal).softmax(-1)
    return torch.einsum("hqk,hkd->hqd", attn, vt).transpose(0, 1).to(q.dtype)


def ref_yarn_rope(pos, q_pe, k_pe, cfg):
    r = cfg.rope_scaling
    rotary_dim, base, factor = cfg.qk_rope_head_dim, cfg.rope_theta, r["factor"]
    max_pos = r["original_max_position_embeddings"]
    beta_fast, beta_slow = r.get("beta_fast", 32), r.get("beta_slow", 1)
    mscale, mscale_all_dim = r.get("mscale", 1.0), r.get("mscale_all_dim", 0.0)
    pos_freqs = base ** (
        torch.arange(0, rotary_dim, 2, device=q_pe.device).float() / rotary_dim
    )
    ext, interp = 1.0 / pos_freqs, 1.0 / (factor * pos_freqs)
    low, high = _yarn_find_correction_range(
        beta_fast, beta_slow, rotary_dim, base, max_pos
    )
    mask = 1 - _yarn_linear_ramp_mask(
        low, high, rotary_dim // 2, torch.float
    ).to(q_pe.device)
    inv_freq = interp * (1 - mask) + ext * mask
    amp = yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim)
    freqs = torch.outer(pos.float(), inv_freq)
    cos = (freqs.cos() * amp).repeat_interleave(2, -1).unsqueeze(-2)
    sin = (freqs.sin() * amp).repeat_interleave(2, -1).unsqueeze(-2)
    qr = q_pe.float() * cos + rotate_gptj(q_pe.float()) * sin
    kr = k_pe.float() * cos + rotate_gptj(k_pe.float()) * sin
    return qr.to(q_pe.dtype), kr.to(k_pe.dtype)


def deepseek_scale(cfg):
    r = cfg.rope_scaling
    mscale = yarn_get_mscale(r["factor"], r.get("mscale_all_dim", 0.0))
    return cfg.qk_head_dim ** -0.5 * mscale * mscale


def ref_deepseek_mla(attn, cfg, h, pos):
    """One MLA layer, materialized: latent up-projected to full per-head K/V,
    attended at the DeepSeek scale, with no head-dim padding."""
    t, heads = h.shape[0], attn.num_heads
    d_nope, d_rope, d_v, latent = (
        cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
    )
    eps = cfg.rms_norm_eps
    q = ref_rmsnorm(F.linear(h, attn.q_a_proj.weight), attn.q_a_layernorm.weight, eps)
    q = F.linear(q, attn.q_b_proj.weight).view(t, heads, cfg.qk_head_dim)
    q_nope, q_pe = q.split([d_nope, d_rope], dim=-1)
    lat = F.linear(h, attn.kv_a_proj_with_mqa.weight)
    kv_a, k_pe = lat.split([latent, d_rope], dim=-1)
    kv = F.linear(
        ref_rmsnorm(kv_a, attn.kv_a_layernorm.weight, eps), attn.kv_b_proj.weight
    ).view(t, heads, d_nope + d_v)
    k_nope, v = kv.split([d_nope, d_v], dim=-1)
    k_pe = k_pe.view(t, 1, d_rope)
    q_pe, k_pe = ref_yarn_rope(pos, q_pe, k_pe, cfg)
    q = torch.cat([q_nope, q_pe], dim=-1)
    k = torch.cat([k_nope, k_pe.expand(t, heads, d_rope)], dim=-1)
    out = sdpa_causal(q, k, v, deepseek_scale(cfg)).reshape(t, heads * d_v)
    return F.linear(out, attn.o_proj.weight)


def ref_mla_latent_step(q_nope_new, q_pe_new, kv_c_all, k_pe_all, scale):
    """Absorbed MLA for one step: the new queries against the request's whole
    latent history, value being the ckv slice of the same latent."""
    sl = q_nope_new.shape[0]
    total = kv_c_all.shape[0]
    old_len = total - sl
    device = q_nope_new.device

    query = torch.cat([q_nope_new, q_pe_new], dim=-1)
    value = kv_c_all.squeeze(1)
    key = torch.cat([value, k_pe_all.squeeze(1)], dim=-1)

    scores = torch.einsum(
        "hqd,kd->hqk", query.transpose(0, 1).float(), key.float()
    ) * scale
    q_pos = old_len + torch.arange(sl, device=device)
    k_pos = torch.arange(total, device=device)
    mask = torch.where(
        k_pos[None, :] <= q_pos[:, None],
        0.0,
        torch.tensor(float("-inf"), device=device),
    )
    out = torch.einsum(
        "hqk,kd->hqd", (scores + mask).softmax(-1), value.float()
    )
    return out.transpose(0, 1).to(q_nope_new.dtype)


# ── checkpoint fixtures ─────────────────────────────────────────────────

def fill_layer(layer, cfg):
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
            cfg.n_routed_experts, device=DEVICE, dtype=torch.float32
        )
        mlp.experts.gate_up_proj.data.normal_(0, 0.05)
        mlp.experts.down_proj.data.normal_(0, 0.05)
        mlp.shared_expert.gate_up_proj.weight.data.normal_(0, 0.05)
        mlp.shared_expert.down_proj.weight.data.normal_(0, 0.05)
    else:
        mlp.gate_up_proj.weight.data.normal_(0, 0.05)
        mlp.down_proj.weight.data.normal_(0, 0.05)


def build_reference(cfg):
    model = KimiForCausalLM(cfg).to(device=DEVICE, dtype=torch.bfloat16)
    model.model.embed_tokens.weight.data.normal_(0, 0.05)
    model.model.norm.weight.data.normal_(1.0, 0.02)
    model.lm_head.weight.data.normal_(0, 0.02)
    for layer in model.model.layers:
        fill_layer(layer, cfg)
    return model.eval()


def hf_checkpoint(model, cfg):
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
    return {k: v.detach().cpu().clone().contiguous() for k, v in sd.items()}


def write_checkpoint(tmp_path, cfg, seed=0):
    """A reduced Kimi checkpoint on disk, plus the model it was built from."""
    from safetensors.torch import save_file

    torch.manual_seed(seed)
    ref = build_reference(cfg)
    save_file(hf_checkpoint(ref, cfg), str(tmp_path / "model.safetensors"))
    return ref


def make_model(cfg, checkpoint_dir) -> KimiK2Model:
    model = object.__new__(KimiK2Model)
    model.config = cfg
    model.model_path_hf = str(checkpoint_dir)
    model.cache_dir = None
    model._submodule_cache = {}
    model._config_variant = "reduced"
    model._tokenizer_mode = "byte"
    model._tokenizer = None
    return model


# ── fake resources ──────────────────────────────────────────────────────

class FakeResources(dict):
    """A resource set that attends the step's own K/V with causal SDPA.

    Stands in for the real backends in the layer tests, which are about the
    layer's algebra rather than the cache: the layer still writes through the
    KV resource and takes its positions off the position one, so the call path
    is the real one and only the kernels are replaced. One object serves all
    three roles; ``captured`` keeps the assembled tensors for inspection.
    """

    requires_kv_write = False
    default_label = "main"

    def __init__(self, scale: float, pos_ids: torch.Tensor):
        self.scale = scale
        self._pos_ids = pos_ids
        self.captured: dict = {}
        self.latent: torch.Tensor | None = None
        super().__init__({KV_CACHE: self, ATTN: self, ROPE: self})

    # cursors, held by both the KV and attention resources
    def set_default_label(self, label: str) -> None:
        pass

    def set_default_layer_idx(self, layer_idx: int) -> None:
        pass

    # KV side
    def layer_view(self, layer_idx: int | None = None):
        return None

    def write_kv(self, k, v, **kwargs) -> None:
        pass

    def write_latent(self, latent, **kwargs) -> None:
        self.latent = latent

    # position side
    def pos_ids(self, label: str) -> torch.Tensor:
        return self._pos_ids

    # attention side
    def run(self, q, label=None, kv_cache_layer=None, k=None, v=None, layer_idx=None):
        self.captured = {"q": q.clone(), "k": k.clone(), "v": v.clone()}
        return sdpa_causal(q, k, v, self.scale)

    def run_mla(self, q_nope, q_pe, latent_cache_layer=None, label=None):
        ckv = q_nope.shape[-1]
        latent = self.latent
        self.captured = {"q_nope": q_nope.clone(), "q_pe": q_pe.clone(),
                         "latent": latent.clone()}
        return ref_mla_latent_step(
            q_nope, q_pe,
            latent[:, :ckv].unsqueeze(1), latent[:, ckv:].unsqueeze(1),
            self.scale,
        )


def bind_fakes(module, scale: float, pos_ids: torch.Tensor, *, layer_idx: int = 0):
    """Bind one fake resource set to every layer under ``module`` and open a
    step on it, so the caller can invoke the layer with hidden states alone."""
    resources = FakeResources(scale, pos_ids)
    for sub in module.modules():
        bind = getattr(sub, "bind_resources", None)
        if bind is not None:
            bind(resources)
            sub.attend.bind_step("main")
            sub.attend.set_layer_idx(layer_idx)
    return resources
