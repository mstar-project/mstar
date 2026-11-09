"""Kimi K3 / K3.1 reference forward for the parity tests: plain torch over a state dict in the
checkpoint's own names (``model.layers.N...``, ``lm_head.weight``; ``language_model.``
stripped), so a pass checks the port's weight mapping as well as its math.

The math is ``modeling_kimi_linear.py`` from ``moonshotai/Kimi-K3`` (f831ab66): SiTU-GLU,
``KimiMoEGate`` / ``KimiSparseMoeBlock`` (latent MoE), ``KimiMLAAttention`` (NoPE, output gate),
``_apply_attn_res`` and the decoder loop; KDA is fla's ``chunk_kda`` with the checkpoint's flags
(q/k l2norm and the head_dim^-0.5 scale, beta sigmoid, the gate ``lb * sigmoid(exp(A) * (f +
dt_bias))``) written as a per-token recurrence. K3.1's indexer has no published modeling code;
it is DeepSeek-V3.2's here (the hypothesis the port makes too). Full-sequence causal forward
only: a decode step is checked against the forward of the whole prefix. Attention runs in chunks
of query rows and the experts over the tokens that picked them, so ~8K-token sequences fit a GPU.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from mstar.model.kimi_k3.config import INDEXER_K_NORM_EPS, KimiK3ModelConfig

# query rows a chunk of the attention and indexer math takes (bounds its scratch at long contexts)
_ROWS = 512


def rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """``KimiRMSNorm``: fp32 normalization, back to the input dtype, then the weight."""
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def situ(gate: torch.Tensor, up: torch.Tensor, beta: float, linear_beta: float | None) -> torch.Tensor:
    g, u = gate.float(), up.float()
    a = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * torch.tanh(u / linear_beta)
    return (a * u).to(gate.dtype)


def attn_res(prefix: torch.Tensor, blocks: list[torch.Tensor], norm_w, proj_w, eps) -> torch.Tensor:
    v = torch.stack(blocks + [prefix], dim=1).float()
    k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    scores = (k * (norm_w.float() * proj_w.squeeze(0).float())).sum(-1)
    return torch.matmul(scores.softmax(-1).unsqueeze(1), v).squeeze(1).to(prefix.dtype)


def gated_mlp(x, gate_w, up_w, down_w, cfg: KimiK3ModelConfig) -> torch.Tensor:
    return F.linear(situ(F.linear(x, gate_w), F.linear(x, up_w), cfg.situ_beta, cfg.situ_linear_beta), down_w)


def kda(sd: dict, p: str, x: torch.Tensor, cfg: KimiK3ModelConfig) -> torch.Tensor:
    t, h, d = x.shape[0], cfg.linear_num_heads, cfg.linear_head_dim
    k_conv = cfg.linear_conv_kernel_size

    def branch(name):
        y = F.linear(x, sd[f"{p}{name}_proj.weight"]).t().unsqueeze(0).float()  # (1, C, T)
        w = sd[f"{p}{name}_conv1d.weight"].float()
        y = F.conv1d(F.pad(y, (k_conv - 1, 0)), w, groups=w.shape[0])
        return F.silu(y)[0].t().reshape(t, h, d)  # fp32

    q, k, v = branch("q"), branch("k"), branch("v")
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * d ** -0.5
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    f = F.linear(F.linear(x, sd[f"{p}f_a_proj.weight"]), sd[f"{p}f_b_proj.weight"]).float().view(t, h, d)
    a_log = sd[f"{p}A_log"].float().reshape(-1)[:h]  # the checkpoint pads it to 128
    g = cfg.gate_lower_bound * torch.sigmoid(
        torch.exp(a_log).view(h, 1) * (f + sd[f"{p}dt_bias"].float().view(h, d)))
    beta = torch.sigmoid(F.linear(x, sd[f"{p}b_proj.weight"]).float())  # (T, H)
    state = torch.zeros(h, d, d, device=x.device)  # (H, K, V)
    out = torch.empty(t, h, d, device=x.device)
    for i in range(t):
        state = state * g[i].exp().unsqueeze(-1)
        delta = (v[i] - (state * k[i].unsqueeze(-1)).sum(-2)) * beta[i].unsqueeze(-1)
        state = state + k[i].unsqueeze(-1) * delta.unsqueeze(-2)
        out[i] = (state * q[i].unsqueeze(-1)).sum(-2)
    gate = F.linear(x, sd[f"{p}g_proj.weight"]).float().view(t, h, d)
    o = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)
    o = (sd[f"{p}o_norm.weight"].float() * o * torch.sigmoid(gate)).to(x.dtype)
    return F.linear(o.reshape(t, h * d), sd[f"{p}o_proj.weight"])


def dsa_mask(sd: dict, p: str, x: torch.Tensor, q_c: torch.Tensor, cfg: KimiK3ModelConfig,
             forced: dict[int, torch.Tensor] | None = None, trace: dict | None = None) -> torch.Tensor:
    """DeepSeek-V3.2's indexer: ``(T, T)`` bool, each query's top ``index_topk`` causal keys by
    ``sum_h w_h relu(q_h . k)`` (ties to the earlier key). ``forced`` maps a query position to the
    key positions to attend instead (-1 padding), to replay another run's picks; ``trace`` gets the
    fp32 scores under ``"scores"``."""
    t, h, d = x.shape[0], cfg.index_n_heads, cfg.index_head_dim
    k = F.layer_norm(F.linear(x, sd[f"{p}indexer.wk.weight"]), (d,), sd[f"{p}indexer.k_norm.weight"],
                     sd[f"{p}indexer.k_norm.bias"], INDEXER_K_NORM_EPS).float()
    q = F.linear(q_c, sd[f"{p}indexer.wq_b.weight"]).view(t, h, d).float()
    w = F.linear(x, sd[f"{p}indexer.weights_proj.weight"]).float() * d ** -0.5 * h ** -0.5
    scores = torch.empty(t, t, device=x.device)
    for r in range(0, t, _ROWS):  # query rows in chunks: (rows, heads, keys) fp32 at a time
        scores[r:r + _ROWS] = torch.einsum("th,ths->ts", w[r:r + _ROWS],
                                           torch.einsum("thd,sd->ths", q[r:r + _ROWS], k).relu())
    causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    if trace is not None:
        trace["scores"] = scores
    order = scores.sort(dim=1, descending=True, stable=True).indices[:, : cfg.index_topk]
    keep = torch.zeros(t, t, dtype=torch.bool, device=x.device)
    keep.scatter_(1, order, True)
    for pos, picks in (forced or {}).items():
        picks = picks[picks >= 0].to(device=x.device, dtype=torch.long)
        keep[pos] = False
        keep[pos, picks] = True
    return keep & causal


def mla(sd: dict, p: str, x: torch.Tensor, cfg: KimiK3ModelConfig, use_dsa: bool,
        forced: dict[int, torch.Tensor] | None = None, trace: dict | None = None) -> torch.Tensor:
    t, h = x.shape[0], cfg.num_attention_heads
    nope, rope, vd = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
    q_c = rms(F.linear(x, sd[f"{p}q_a_proj.weight"]), sd[f"{p}q_a_layernorm.weight"], cfg.rms_norm_eps)
    q = F.linear(q_c, sd[f"{p}q_b_proj.weight"]).view(t, h, nope + rope)
    k_pass, k_rot = F.linear(x, sd[f"{p}kv_a_proj_with_mqa.weight"]).split([cfg.kv_lora_rank, rope], -1)
    kv = F.linear(rms(k_pass, sd[f"{p}kv_a_layernorm.weight"], cfg.rms_norm_eps),
                  sd[f"{p}kv_b_proj.weight"]).view(t, h, nope + vd)
    k_nope, v = kv.split([nope, vd], -1)
    k = torch.cat([k_nope, k_rot.view(t, 1, rope).expand(t, h, rope)], -1)  # NoPE: unrotated
    allowed = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
    if use_dsa:
        allowed = dsa_mask(sd, p, x, q_c, cfg, forced, trace)
    out = torch.empty(t, h, vd, device=x.device)
    kf, vf = k.float(), v.float()
    for r in range(0, t, _ROWS):  # query rows in chunks: (heads, rows, keys) fp32 at a time
        scores = torch.einsum("qhd,khd->hqk", q[r:r + _ROWS].float(), kf) * cfg.softmax_scale
        probs = scores.masked_fill(~allowed[r:r + _ROWS], float("-inf")).softmax(-1)
        out[r:r + _ROWS] = torch.einsum("hqk,khd->qhd", probs, vf)
    out = out.to(x.dtype).reshape(t, h * vd)
    if cfg.mla_use_output_gate:
        out = out * torch.sigmoid(F.linear(x, sd[f"{p}g_proj.weight"]))
    return F.linear(out, sd[f"{p}o_proj.weight"])


def route(sd: dict, p: str, x: torch.Tensor, cfg: KimiK3ModelConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """``(T, top_k)`` expert ids and weights: noaux_tc, sigmoid scores, the top-k by score + bias, their unbiased
    scores renormalized (``moe_renormalize``) and times ``routed_scaling_factor``."""
    logits = F.linear(x.float(), sd[f"{p}gate.weight"].float())
    scores = logits.sigmoid()
    choice = scores + sd[f"{p}gate.e_score_correction_bias"].float()
    idx = choice.topk(cfg.num_experts_per_tok, dim=-1, sorted=False).indices
    w = scores.gather(1, idx)
    if cfg.moe_renormalize:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return idx, w * cfg.routed_scaling_factor


def latent_moe(sd: dict, p: str, x: torch.Tensor, cfg: KimiK3ModelConfig) -> torch.Tensor:
    idx, w = route(sd, p, x, cfg)
    z = F.linear(x, sd[f"{p}routed_expert_down_proj.weight"])
    y = torch.zeros(x.shape[0], z.shape[1], device=x.device)
    for e in idx.unique().tolist():  # each expert over the tokens that picked it
        tok, slot = torch.where(idx == e)
        ep = f"{p}experts.{e}."
        out = gated_mlp(z[tok], sd[f"{ep}w1.weight"], sd[f"{ep}w3.weight"], sd[f"{ep}w2.weight"], cfg)
        y.index_add_(0, tok, w[tok, slot, None] * out.float())
    y = y.to(x.dtype)
    if cfg.latent_moe_use_norm:
        y = rms(y, sd[f"{p}routed_expert_norm.weight"], cfg.rms_norm_eps)
    y = F.linear(y, sd[f"{p}routed_expert_up_proj.weight"])
    if cfg.num_shared_experts:
        s = f"{p}shared_experts."
        y = y + gated_mlp(x, sd[f"{s}gate_proj.weight"], sd[f"{s}up_proj.weight"], sd[f"{s}down_proj.weight"], cfg)
    return y


@torch.no_grad()
def reference_logits(sd: dict, input_ids: torch.Tensor, cfg: KimiK3ModelConfig, use_dsa: bool = False,
                     forced: dict[int, dict[int, torch.Tensor]] | None = None,
                     trace: dict | None = None, taps: dict | None = None, logits_from: int = 0) -> torch.Tensor:
    """``(T, vocab)`` logits of one sequence; ``use_dsa`` applies K3.1's top-k selection.
    ``forced`` (layer -> query position -> key positions) replays another run's DSA picks;
    ``trace`` collects each DSA layer's fp32 index scores under its layer index; ``taps`` each
    layer's output stream (the block's running sum) under its layer index. ``logits_from``: only
    positions from it on get logits (an 8K-token sequence's would be 5 GB in fp32)."""
    eps = cfg.rms_norm_eps
    prefix = sd["model.embed_tokens.weight"][input_ids]
    blocks: list[torch.Tensor] = []
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}."
        x = prefix
        if blocks:
            x = attn_res(prefix, blocks, sd[f"{p}self_attention_res_norm.weight"],
                         sd[f"{p}self_attention_res_proj.weight"], eps)
        cur = prefix
        if cfg.is_attn_res_block_start(i):
            blocks = blocks + [prefix]
            cur = None
        h = rms(x, sd[f"{p}input_layernorm.weight"], eps)
        if cfg.is_full_attention_layer(i):
            layer_trace = None if trace is None else trace.setdefault(i, {})
            a = mla(sd, f"{p}self_attn.", h, cfg, use_dsa and cfg.dsa, (forced or {}).get(i), layer_trace)
        else:
            a = kda(sd, f"{p}self_attn.", h, cfg)
        cur = a if cur is None else cur + a
        x = attn_res(cur, blocks, sd[f"{p}mlp_res_norm.weight"], sd[f"{p}mlp_res_proj.weight"], eps)
        h = rms(x, sd[f"{p}post_attention_layernorm.weight"], eps)
        if cfg.is_moe_layer(i):
            f = latent_moe(sd, f"{p}block_sparse_moe.", h, cfg)
        else:
            f = gated_mlp(h, sd[f"{p}mlp.gate_proj.weight"], sd[f"{p}mlp.up_proj.weight"],
                          sd[f"{p}mlp.down_proj.weight"], cfg)
        prefix = cur + f
        if taps is not None:
            taps[i] = prefix
    prefix, blocks = prefix[logits_from:], [b[logits_from:] for b in blocks]
    out = attn_res(prefix, blocks, sd["model.output_attn_res_norm.weight"],
                   sd["model.output_attn_res_proj.weight"], eps)
    return F.linear(rms(out, sd["model.norm.weight"], eps), sd["lm_head.weight"])


def random_checkpoint(cfg: KimiK3ModelConfig, seed: int = 0, dtype=torch.float32,
                      a_log_pad: int = 0, include_indexer: bool = True) -> dict[str, torch.Tensor]:
    """A random state dict in the checkpoint's names and dtypes (fp32 where it stores fp32),
    bf16-shaped experts; ``a_log_pad`` extra zero entries on ``A_log`` as the release stores."""
    g = torch.Generator().manual_seed(seed)

    def r(*shape, std=0.05):
        return (torch.randn(*shape, generator=g) * std).to(dtype)

    def norm(n):
        return (1.0 + torch.randn(n, generator=g) * 0.1).to(dtype)

    hdim, sd = cfg.hidden_size, {}
    sd["model.embed_tokens.weight"] = r(cfg.vocab_size, hdim, std=0.5)
    sd["lm_head.weight"] = r(cfg.vocab_size, hdim)
    sd["model.norm.weight"] = norm(hdim)
    sd["model.output_attn_res_norm.weight"] = norm(hdim)
    sd["model.output_attn_res_proj.weight"] = r(1, hdim, std=0.5)
    qkv = cfg.linear_num_heads * cfg.linear_head_dim
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}."
        for n in ("input_layernorm", "post_attention_layernorm", "self_attention_res_norm", "mlp_res_norm"):
            sd[f"{p}{n}.weight"] = norm(hdim)
        sd[f"{p}self_attention_res_proj.weight"] = r(1, hdim, std=0.5)
        sd[f"{p}mlp_res_proj.weight"] = r(1, hdim, std=0.5)
        a = f"{p}self_attn."
        if cfg.is_full_attention_layer(i):
            h = cfg.num_attention_heads
            sd[f"{a}q_a_proj.weight"] = r(cfg.q_lora_rank, hdim)
            sd[f"{a}q_a_layernorm.weight"] = norm(cfg.q_lora_rank)
            sd[f"{a}q_b_proj.weight"] = r(h * cfg.qk_head_dim, cfg.q_lora_rank, std=0.2)
            sd[f"{a}kv_a_proj_with_mqa.weight"] = r(cfg.cache_latent_dim, hdim, std=0.2)
            sd[f"{a}kv_a_layernorm.weight"] = norm(cfg.kv_lora_rank)
            sd[f"{a}kv_b_proj.weight"] = r(h * (cfg.qk_nope_head_dim + cfg.v_head_dim), cfg.kv_lora_rank, std=0.2)
            sd[f"{a}g_proj.weight"] = r(h * cfg.v_head_dim, hdim)
            sd[f"{a}o_proj.weight"] = r(hdim, h * cfg.v_head_dim)
            if cfg.dsa and include_indexer:
                ih, idim = cfg.index_n_heads, cfg.index_head_dim
                sd[f"{a}indexer.wq_b.weight"] = r(ih * idim, cfg.q_lora_rank, std=0.3)
                sd[f"{a}indexer.wk.weight"] = r(idim, hdim, std=0.3)
                sd[f"{a}indexer.weights_proj.weight"] = r(ih, hdim, std=0.3)
                sd[f"{a}indexer.k_norm.weight"] = norm(idim)
                sd[f"{a}indexer.k_norm.bias"] = r(idim, std=0.1)
        else:
            for b in ("q", "k", "v"):
                sd[f"{a}{b}_proj.weight"] = r(qkv, hdim, std=0.2)
                sd[f"{a}{b}_conv1d.weight"] = torch.randn(qkv, 1, cfg.linear_conv_kernel_size, generator=g) * 0.3
            sd[f"{a}f_a_proj.weight"] = r(cfg.linear_head_dim, hdim, std=0.2)
            sd[f"{a}f_b_proj.weight"] = r(qkv, cfg.linear_head_dim, std=0.2)
            sd[f"{a}dt_bias"] = torch.randn(qkv, generator=g) * 0.5
            sd[f"{a}A_log"] = torch.cat([torch.rand(cfg.linear_num_heads, generator=g), torch.zeros(a_log_pad)])
            sd[f"{a}b_proj.weight"] = r(cfg.linear_num_heads, hdim, std=0.2)
            sd[f"{a}g_proj.weight"] = r(qkv, hdim, std=0.2)
            sd[f"{a}o_norm.weight"] = 1.0 + torch.randn(cfg.linear_head_dim, generator=g) * 0.1
            sd[f"{a}o_proj.weight"] = r(hdim, qkv)
        if cfg.is_moe_layer(i):
            m, lat, inter = f"{p}block_sparse_moe.", cfg.routed_expert_hidden_size, cfg.moe_intermediate_size
            sd[f"{m}gate.weight"] = r(cfg.num_experts, hdim, std=0.3)
            sd[f"{m}gate.e_score_correction_bias"] = torch.randn(cfg.num_experts, generator=g) * 0.1
            sd[f"{m}routed_expert_down_proj.weight"] = r(lat, hdim, std=0.2)
            sd[f"{m}routed_expert_norm.weight"] = norm(lat)
            sd[f"{m}routed_expert_up_proj.weight"] = r(hdim, lat)
            for e in range(cfg.num_experts):
                sd[f"{m}experts.{e}.w1.weight"] = r(inter, lat, std=0.3)
                sd[f"{m}experts.{e}.w3.weight"] = r(inter, lat, std=0.3)
                sd[f"{m}experts.{e}.w2.weight"] = r(lat, inter, std=0.3)
            shared = inter * cfg.num_shared_experts
            sd[f"{m}shared_experts.gate_proj.weight"] = r(shared, hdim, std=0.2)
            sd[f"{m}shared_experts.up_proj.weight"] = r(shared, hdim, std=0.2)
            sd[f"{m}shared_experts.down_proj.weight"] = r(hdim, shared)
        else:
            sd[f"{p}mlp.gate_proj.weight"] = r(cfg.intermediate_size, hdim, std=0.2)
            sd[f"{p}mlp.up_proj.weight"] = r(cfg.intermediate_size, hdim, std=0.2)
            sd[f"{p}mlp.down_proj.weight"] = r(hdim, cfg.intermediate_size)
    return sd
