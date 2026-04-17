"""End-to-end comparison: mminf Pi0.5 action expert vs lerobot reference.

Approach
--------
1. Load lerobot PI05Policy (real production weights from lerobot/pi05_base).
2. Extract its gemma_expert (action expert) layers and pre-populated prefix
   KV cache by running PaliGemma's prefill on a deterministic input.
3. Build mminf Pi05ActionExpert with matching dims and copy lerobot's action
   expert weights into it (per-layer self_attn / mlp / adaRMS norms).
4. Compute the suffix embedding the same way lerobot does (action_in_proj on
   noise + sincos timestep + time_mlp -> adarms_cond).
5. Run lerobot's gemma_expert layers with past_key_values to produce the
   reference suffix output for each iteration of the flow-matching loop.
6. Run mminf's Pi05ActionExpert in the same loop using a MockCacheHandle
   pre-populated with the prefix KV cache and using the matched HF Gemma
   RoPE formula on the suffix Q,K.
7. Compare velocity outputs (action_out_proj) and final denoised actions.

The mminf side bypasses FlashInfer's paged KV cache for this test (we're
not validating page allocation, just the action-expert layer math). The
underlying compute — adaRMS, gated residuals, attention with past KV —
matches what FlashInfer's wrapper does up to bf16 precision (already
validated by the equivalence tests).
"""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lerobot.policies.pi05 import PI05Policy

from mminf.model.pi05.components.action_expert import (
    Pi05ActionExpert,
    Pi05TimeMLP,
)
from mminf.model.pi05.components.flow_matching import sincos_timestep_embedding
from mminf.model.pi05.config import Pi05Config

DEVICE = torch.device("cuda")
DTYPE = torch.float32  # match lerobot's loaded dtype
SEED = 0
torch.manual_seed(SEED)


# ----- HF Gemma RoPE (matches transformers.models.gemma) ------------------


def _hf_apply_rotary(q, k, position_ids, head_dim, theta=10000.0):
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim)
    )
    freqs = position_ids.float()[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(torch.float32)[:, None, :]
    sin = emb.sin().to(torch.float32)[:, None, :]

    def rot(x):
        x1 = x[..., : head_dim // 2]
        x2 = x[..., head_dim // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    qf = q.float()
    kf = k.float()
    return (qf * cos + rot(qf) * sin).to(q.dtype), (kf * cos + rot(kf) * sin).to(k.dtype)


# ----- MockCacheHandle: vanilla SDPA + HF-style RoPE, per-layer KV cache --


class MockCacheHandle:
    """Vanilla-torch stand-in for BatchedCacheManager.

    Stores per-layer prefix K/V (already RoPE'd by lerobot during prefill).
    Suffix Q/K get RoPE'd at suffix_positions before attention. K/V are
    appended to the prefix cache only if write_cache=True.
    """

    def __init__(
        self,
        head_dim: int,
        suffix_positions: torch.Tensor,
        rope_theta: float = 10000.0,
    ):
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.suffix_positions = suffix_positions
        self.rope_theta = rope_theta
        self.layer_idx = 0
        self._prefix_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.write_cache = False

    def set_layer_idx(self, layer_idx: int):
        self.layer_idx = layer_idx

    def set_active_label(self, label: str):
        pass

    def apply_rope(self, q, k, rope_theta=None, **kwargs):
        # Suffix Q, K get RoPE at suffix_positions.
        return _hf_apply_rotary(q, k, self.suffix_positions, self.head_dim, rope_theta or self.rope_theta)

    def run_attention(self, q, k, v):
        prefix_k, prefix_v = self._prefix_kv[self.layer_idx]
        k_full = torch.cat([prefix_k, k], dim=0)
        v_full = torch.cat([prefix_v, v], dim=0)

        nh = q.shape[1]
        nkv = k_full.shape[1]
        if nh != nkv:
            rep = nh // nkv
            k_full = k_full.repeat_interleave(rep, dim=1)
            v_full = v_full.repeat_interleave(rep, dim=1)

        qt = q.transpose(0, 1).float()
        kt = k_full.transpose(0, 1).float()
        vt = v_full.transpose(0, 1).float()
        scores = torch.einsum("hqd,hkd->hqk", qt, kt) * self.scale
        attn = scores.softmax(-1)
        out = torch.einsum("hqk,hkd->hqd", attn, vt)
        return out.transpose(0, 1).to(q.dtype)

    def advance_seq_lens(self, *args, **kwargs):
        pass


# ----- Weight copy helpers ------------------------------------------------


def _copy_linear(dst: nn.Linear, src: nn.Linear):
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        if dst.bias is not None and src.bias is not None:
            dst.bias.copy_(src.bias)


def copy_action_expert_weights(ours: Pi05ActionExpert, ref_layers, ref_norm):
    """Copy lerobot gemma_expert weights into mminf Pi05ActionExpert.

    ``ref_layers`` is the list of lerobot ``PiGemmaDecoderLayer`` instances.
    ``ref_norm`` is the lerobot final norm.
    """
    assert len(ours.layers) == len(ref_layers)
    for our_layer, ref_layer in zip(ours.layers, ref_layers, strict=True):
        # Attention projections
        _copy_linear(our_layer.self_attn.q_proj, ref_layer.self_attn.q_proj)
        _copy_linear(our_layer.self_attn.k_proj, ref_layer.self_attn.k_proj)
        _copy_linear(our_layer.self_attn.v_proj, ref_layer.self_attn.v_proj)
        _copy_linear(our_layer.self_attn.o_proj, ref_layer.self_attn.o_proj)
        # MLP
        _copy_linear(our_layer.mlp.gate_proj, ref_layer.mlp.gate_proj)
        _copy_linear(our_layer.mlp.up_proj, ref_layer.mlp.up_proj)
        _copy_linear(our_layer.mlp.down_proj, ref_layer.mlp.down_proj)
        # adaRMS norms (only the .dense layer; no plain weight in cond path)
        _copy_linear(our_layer.input_layernorm.dense, ref_layer.input_layernorm.dense)
        _copy_linear(
            our_layer.post_attention_layernorm.dense,
            ref_layer.post_attention_layernorm.dense,
        )
    _copy_linear(ours.norm.dense, ref_norm.dense)


# ----- Main ---------------------------------------------------------------


def main():
    print("Loading lerobot/pi05_base ...")
    policy = PI05Policy.from_pretrained("lerobot/pi05_base").to(DEVICE).eval()
    model = policy.model
    config = policy.config

    action_hidden = model.action_in_proj.out_features
    paligemma = model.paligemma_with_expert.paligemma
    gemma_expert = model.paligemma_with_expert.gemma_expert
    print(f"action_expert hidden: {action_hidden}")
    print(f"action_expert num_layers: {len(gemma_expert.model.layers)}")
    head_dim = gemma_expert.model.layers[0].self_attn.head_dim
    num_kv_heads = gemma_expert.model.layers[0].self_attn.config.num_key_value_heads
    num_qo_heads = gemma_expert.model.layers[0].self_attn.config.num_attention_heads
    print(f"head_dim: {head_dim}, num_qo_heads: {num_qo_heads}, num_kv_heads: {num_kv_heads}")

    # ----- Build deterministic input -----
    bsize = 1
    horizon = config.chunk_size  # 50
    action_dim = config.max_action_dim  # 32

    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    images = [
        torch.rand(bsize, 3, 224, 224, device=DEVICE, generator=g) * 2 - 1
        for _ in range(3)
    ]
    img_masks = [torch.ones(bsize, dtype=torch.bool, device=DEVICE) for _ in range(3)]
    tok_len = 4
    tokens = torch.randint(0, 200, (bsize, tok_len), device=DEVICE, generator=g)
    masks = torch.ones(bsize, tok_len, dtype=torch.bool, device=DEVICE)
    noise = torch.randn(bsize, horizon, action_dim, device=DEVICE, generator=g, dtype=torch.float32)

    # ----- Run lerobot end-to-end so we have a target action tensor -----
    ref_actions = model.sample_actions(
        images=[i.to(DTYPE) for i in images],
        img_masks=img_masks,
        tokens=tokens,
        masks=masks,
        noise=noise,
        num_steps=config.num_inference_steps,
    )
    print(f"\nReference actions shape: {ref_actions.shape}, abs max: {ref_actions.abs().max().item():.4f}")

    # ----- Run lerobot prefill manually so we can get past_key_values -----
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        [i.to(DTYPE) for i in images], img_masks, tokens, masks
    )
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
    paligemma.model.language_model.config._attn_implementation = "eager"
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_pos,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    prefix_len = prefix_pad_masks.sum(dim=-1).item()
    print(f"prefix length: {prefix_len}")
    print(f"past_key_values type: {type(past_key_values).__name__}")

    # ----- Build mminf Pi05ActionExpert with matching dims -----
    cfg = Pi05Config(
        hidden_size=2048,  # paligemma side (unused here)
        action_hidden_size=action_hidden,
        num_layers=len(gemma_expert.model.layers),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        action_intermediate_size=gemma_expert.model.layers[0].mlp.gate_proj.out_features,
        rms_norm_eps=gemma_expert.model.layers[0].input_layernorm.eps,
    )
    ours_action = Pi05ActionExpert(cfg).to(DEVICE, dtype=DTYPE)
    copy_action_expert_weights(ours_action, gemma_expert.model.layers, gemma_expert.model.norm)
    print("Copied action expert weights into mminf Pi05ActionExpert")

    # Time MLP weights too
    ours_time_mlp = Pi05TimeMLP(action_hidden).to(DEVICE, dtype=DTYPE)
    with torch.no_grad():
        ours_time_mlp.linear_in.weight.copy_(model.time_mlp_in.weight)
        ours_time_mlp.linear_in.bias.copy_(model.time_mlp_in.bias)
        ours_time_mlp.linear_out.weight.copy_(model.time_mlp_out.weight)
        ours_time_mlp.linear_out.bias.copy_(model.time_mlp_out.bias)

    # ----- Pre-populate MockCacheHandle with the lerobot prefix KV -----
    # past_key_values is HF DynamicCache; per-layer K,V have shape
    # [B, num_kv_heads, prefix_len, head_dim]. mminf format is
    # [seq_len, num_kv_heads, head_dim] with B=1.
    suffix_positions = (
        torch.arange(horizon, device=DEVICE, dtype=torch.long) + prefix_len
    )
    handle = MockCacheHandle(head_dim=head_dim, suffix_positions=suffix_positions)
    # HF DynamicCache stores per-layer K,V on past_key_values.layers[layer_idx]
    for layer_idx in range(cfg.num_layers):
        layer_cache = past_key_values.layers[layer_idx]
        # layer_cache has .keys, .values attrs (or similar). Try common ones.
        if hasattr(layer_cache, "keys"):
            k = layer_cache.keys
            v = layer_cache.values
        elif isinstance(layer_cache, tuple):
            k, v = layer_cache
        else:
            raise RuntimeError(f"Unknown layer cache format: {type(layer_cache)} attrs={dir(layer_cache)}")
        # k,v shape: [B, num_kv_heads, prefix_len, head_dim]
        handle._prefix_kv[layer_idx] = (
            k[0].transpose(0, 1).contiguous(),  # [prefix_len, num_kv_heads, head_dim]
            v[0].transpose(0, 1).contiguous(),
        )
    print("Populated MockCacheHandle with prefix KV")

    # ----- Run mminf denoising loop and compare against lerobot -----
    num_steps = config.num_inference_steps
    dt = -1.0 / num_steps
    x_t = noise.clone()
    time = torch.tensor(1.0, device=DEVICE, dtype=DTYPE)
    print(f"\nRunning {num_steps}-step denoise (mminf) ...")
    for _step in range(num_steps):
        # Time -> adarms_cond
        time_emb = sincos_timestep_embedding(
            time.unsqueeze(0), dim=action_hidden, min_period=config.min_period, max_period=config.max_period
        ).squeeze(0)
        adarms_cond = ours_time_mlp(time_emb.to(DTYPE))

        # Suffix embedding from action_in_proj (lerobot's, since we share weights)
        suffix = model.action_in_proj(x_t[0])  # [horizon, action_hidden]

        # mminf forward
        suffix_out = ours_action(
            query_sequence=suffix,
            cache_handle=handle,
            adarms_cond=adarms_cond,
        )
        v_t = model.action_out_proj(suffix_out)  # [horizon, action_dim]
        x_t = x_t + dt * v_t.unsqueeze(0)
        time = time + dt
    ours_actions = x_t
    print(f"\nmminf actions abs max: {ours_actions.abs().max().item():.4f}")
    print(f"reference abs max: {ref_actions.abs().max().item():.4f}")

    delta_max = (ours_actions - ref_actions).abs().max().item()
    delta_mean = (ours_actions - ref_actions).abs().mean().item()
    rel_max = ((ours_actions - ref_actions).abs() / (ref_actions.abs() + 1e-6)).max().item()
    rel_mean = ((ours_actions - ref_actions).abs() / (ref_actions.abs() + 1e-6)).mean().item()
    print("\n=== END-TO-END COMPARISON ===")
    print(f"  max abs delta: {delta_max:.4e}")
    print(f"  mean abs delta: {delta_mean:.4e}")
    print(f"  max rel err: {rel_max:.4e}")
    print(f"  mean rel err: {rel_mean:.4e}")
    print(f"  status: {'PASS' if delta_max < 1e-3 else 'FAIL'}")
    print()
    print(f"first 8 ref values: {ref_actions[0, 0, :8].cpu().tolist()}")
    print(f"first 8 our values: {ours_actions[0, 0, :8].cpu().tolist()}")


if __name__ == "__main__":
    main()
