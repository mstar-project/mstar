"""End-to-end numerical comparison of mminf's Pi0.5 action expert against
``lerobot/pi05_base`` (the production Pi0.5 release) on real weights.

This test is skipped automatically when:
  * CUDA is not available, or
  * the ``lerobot`` package is not installed, or
  * the ``lerobot/pi05_base`` checkpoint isn't already downloaded to the local
    HF cache (we don't trigger a 14 GB download from CI).

To run locally::

    pip install lerobot
    huggingface-cli download lerobot/pi05_base
    LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 \
        pytest test/integration/test_pi05_real_weights.py -v -s

What it does
------------
1. Loads ``lerobot/pi05_base`` (4.1B params, action expert is gemma_300m).
2. Builds a deterministic input (3 random images, random tokens, fixed RNG)
   and runs lerobot's ``PI05Pytorch.sample_actions`` to get the reference
   action trajectory of shape ``[1, 50, 32]``.
3. Runs lerobot's prefill manually to extract the per-layer prefix KV cache.
4. Builds an mminf ``Pi05ActionExpert`` with the matching dims (action_hidden
   = 1024) and copies lerobot's ``gemma_expert`` weights into it layer by
   layer (q/k/v/o projections, MLP, both adaRMS norms, plus the final norm).
5. Runs the same 10-step Euler flow-matching loop using the mminf action
   expert with a small ``MockCacheHandle`` that holds the prefix KV cache
   and uses HF Gemma's RoPE formula on the suffix.
6. Asserts that mminf's action trajectory matches lerobot's to within
   ``5e-4`` max absolute delta and ``1e-2`` mean relative error.

Why a mock cache handle instead of FlashInfer
---------------------------------------------
This test isolates the action expert's *math* (adaRMS, gated residuals, GQA
attention with past KV) from the FlashInfer paged KV cache infrastructure.
The FlashInfer wrapper has been validated against vanilla SDPA separately
in ``test_pi05_reference_equivalence.py``
(``test_flashinfer_paged_prefill_attention_matches_sdpa``), and the
``test_flashinfer_rope_matches_hf_gemma_formula`` test confirms that
``flashinfer.rope.apply_rope_pos_ids_inplace`` produces the same Q,K as
HF Gemma's ``apply_rotary_pos_emb``. So if the action expert math is right
on this test, plugging it into the FlashInfer cache_handle inside an
``AREngine`` is mechanical.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mminf.model.pi05.components.action_expert import (  # noqa: E402
    Pi05ActionExpert,
    Pi05TimeMLP,
)
from mminf.model.pi05.components.flow_matching import sincos_timestep_embedding  # noqa: E402
from mminf.model.pi05.components.paligemma import Pi05PaliGemmaExpert  # noqa: E402
from mminf.model.pi05.config import Pi05Config  # noqa: E402

PI05_REPO = "lerobot/pi05_base"


def _hf_cache_has_pi05() -> bool:
    """Check whether lerobot/pi05_base is in the local HF cache (avoid heavy network IO in CI)."""
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    candidate = cache_root / "models--lerobot--pi05_base"
    if candidate.exists():
        return True
    # also check the alternate HF_HUB_CACHE env
    alt = Path(os.environ.get("HF_HUB_CACHE", "")) / "models--lerobot--pi05_base"
    return bool(os.environ.get("HF_HUB_CACHE")) and alt.exists()


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(
        pytest.importorskip("lerobot", reason="lerobot package not installed") is None,
        reason="lerobot package not installed",
    ),
    pytest.mark.skipif(
        not _hf_cache_has_pi05(),
        reason=f"{PI05_REPO} not in local HF cache; run `huggingface-cli download {PI05_REPO}`",
    ),
]


def _hf_apply_rotary(q, k, position_ids, head_dim, theta=10000.0):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim)
    )
    freqs = position_ids.float()[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()[:, None, :]
    sin = emb.sin()[:, None, :]

    def rot(x):
        x1 = x[..., : head_dim // 2]
        x2 = x[..., head_dim // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    qf = q.float()
    kf = k.float()
    return (qf * cos + rot(qf) * sin).to(q.dtype), (kf * cos + rot(kf) * sin).to(k.dtype)


class _MockCacheHandle:
    """Vanilla-SDPA replacement for ``BatchedCacheManager``.

    Stores per-layer prefix K/V (already RoPE'd by lerobot during prefill).
    Suffix Q/K get HF-style RoPE applied at ``suffix_positions`` before the
    softmax-attention against the concatenated [prefix, suffix] KV.
    """

    def __init__(self, head_dim: int, suffix_positions: torch.Tensor, rope_theta: float = 10000.0):
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.suffix_positions = suffix_positions
        self.rope_theta = rope_theta
        self.layer_idx = 0
        self._prefix_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def set_layer_idx(self, layer_idx: int):
        self.layer_idx = layer_idx

    def set_active_label(self, label: str):  # noqa: ARG002
        pass

    def apply_rope(self, q, k, rope_theta=None, **_kwargs):
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


def _copy_linear(dst: nn.Linear, src: nn.Linear):
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        if dst.bias is not None and src.bias is not None:
            dst.bias.copy_(src.bias)


def _copy_action_expert(ours: Pi05ActionExpert, ref_layers, ref_norm):
    for our_layer, ref_layer in zip(ours.layers, ref_layers, strict=True):
        _copy_linear(our_layer.self_attn.q_proj, ref_layer.self_attn.q_proj)
        _copy_linear(our_layer.self_attn.k_proj, ref_layer.self_attn.k_proj)
        _copy_linear(our_layer.self_attn.v_proj, ref_layer.self_attn.v_proj)
        _copy_linear(our_layer.self_attn.o_proj, ref_layer.self_attn.o_proj)
        _copy_linear(our_layer.mlp.gate_proj, ref_layer.mlp.gate_proj)
        _copy_linear(our_layer.mlp.up_proj, ref_layer.mlp.up_proj)
        _copy_linear(our_layer.mlp.down_proj, ref_layer.mlp.down_proj)
        _copy_linear(our_layer.input_layernorm.dense, ref_layer.input_layernorm.dense)
        _copy_linear(
            our_layer.post_attention_layernorm.dense,
            ref_layer.post_attention_layernorm.dense,
        )
    _copy_linear(ours.norm.dense, ref_norm.dense)


def _copy_paligemma_expert(ours: Pi05PaliGemmaExpert, ref_layers, ref_norm):
    for our_layer, ref_layer in zip(ours.layers, ref_layers, strict=True):
        _copy_linear(our_layer.self_attn.q_proj, ref_layer.self_attn.q_proj)
        _copy_linear(our_layer.self_attn.k_proj, ref_layer.self_attn.k_proj)
        _copy_linear(our_layer.self_attn.v_proj, ref_layer.self_attn.v_proj)
        _copy_linear(our_layer.self_attn.o_proj, ref_layer.self_attn.o_proj)
        _copy_linear(our_layer.mlp.gate_proj, ref_layer.mlp.gate_proj)
        _copy_linear(our_layer.mlp.up_proj, ref_layer.mlp.up_proj)
        _copy_linear(our_layer.mlp.down_proj, ref_layer.mlp.down_proj)
        # Non-conditional Gemma RMSNorm: just .weight
        with torch.no_grad():
            our_layer.input_layernorm.weight.copy_(ref_layer.input_layernorm.weight)
            our_layer.post_attention_layernorm.weight.copy_(
                ref_layer.post_attention_layernorm.weight
            )
    with torch.no_grad():
        ours.norm.weight.copy_(ref_norm.weight)


class _PrefixCacheCapture:
    """MockCacheHandle that captures K,V written by mminf's PaliGemma forward.

    Mirrors the bidirectional prefill setup: applies HF Gemma RoPE on Q,K with
    consecutive positions [0, prefix_len), runs vanilla SDPA, and stores the
    per-layer post-RoPE K,V so they can be compared against lerobot's
    DynamicCache contents.
    """

    def __init__(self, head_dim: int, prefix_len: int):
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.positions = torch.arange(prefix_len, dtype=torch.long)
        self.layer_idx = 0
        self.captured_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def set_layer_idx(self, layer_idx: int):
        self.layer_idx = layer_idx

    def set_active_label(self, label: str):  # noqa: ARG002
        pass

    def apply_rope(self, q, k, rope_theta=None, **_kwargs):
        positions = self.positions.to(q.device)
        return _hf_apply_rotary(q, k, positions, self.head_dim, rope_theta or 10000.0)

    def run_attention(self, q, k, v):
        # Capture post-RoPE K, V (matches what HF DynamicCache stores).
        self.captured_kv[self.layer_idx] = (k.clone(), v.clone())
        # Bidirectional SDPA on the prefix
        nh = q.shape[1]
        nkv = k.shape[1]
        if nh != nkv:
            rep = nh // nkv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        qt = q.transpose(0, 1).float()
        kt = k.transpose(0, 1).float()
        vt = v.transpose(0, 1).float()
        scores = torch.einsum("hqd,hkd->hqk", qt, kt) * self.scale
        attn = scores.softmax(-1)
        out = torch.einsum("hqk,hkd->hqd", attn, vt)
        return out.transpose(0, 1).to(q.dtype)

    def advance_seq_lens(self, *args, **kwargs):
        pass


def test_pi05_action_expert_matches_lerobot_real_weights():
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    device = torch.device("cuda")
    dtype = torch.float32
    seed = 0
    torch.manual_seed(seed)

    policy = PI05Policy.from_pretrained(PI05_REPO).to(device).eval()
    model = policy.model
    config = policy.config

    action_hidden = model.action_in_proj.out_features
    paligemma = model.paligemma_with_expert.paligemma
    gemma_expert = model.paligemma_with_expert.gemma_expert
    head_dim = gemma_expert.model.layers[0].self_attn.head_dim
    num_qo_heads = gemma_expert.model.layers[0].self_attn.config.num_attention_heads
    num_kv_heads = gemma_expert.model.layers[0].self_attn.config.num_key_value_heads

    # Deterministic inputs
    bsize = 1
    horizon = config.chunk_size
    action_dim = config.max_action_dim
    g = torch.Generator(device=device).manual_seed(seed)
    images = [
        torch.rand(bsize, 3, 224, 224, device=device, generator=g) * 2 - 1
        for _ in range(3)
    ]
    img_masks = [torch.ones(bsize, dtype=torch.bool, device=device) for _ in range(3)]
    tokens = torch.randint(0, 200, (bsize, 4), device=device, generator=g)
    masks = torch.ones(bsize, 4, dtype=torch.bool, device=device)
    noise = torch.randn(bsize, horizon, action_dim, device=device, generator=g, dtype=torch.float32)

    # Reference: lerobot end-to-end action trajectory
    ref_actions = model.sample_actions(
        images=[i.to(dtype) for i in images],
        img_masks=img_masks,
        tokens=tokens,
        masks=masks,
        noise=noise,
        num_steps=config.num_inference_steps,
    )

    # Run lerobot prefill manually to capture per-layer prefix KV
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        [i.to(dtype) for i in images], img_masks, tokens, masks
    )
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
    prefix_len = int(prefix_pad_masks.sum(dim=-1).item())

    # Build mminf Pi05ActionExpert with matching dims
    cfg = Pi05Config(
        hidden_size=2048,
        action_hidden_size=action_hidden,
        num_layers=len(gemma_expert.model.layers),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        action_intermediate_size=gemma_expert.model.layers[0].mlp.gate_proj.out_features,
        rms_norm_eps=gemma_expert.model.layers[0].input_layernorm.eps,
    )
    ours_action = Pi05ActionExpert(cfg).to(device, dtype=dtype)
    _copy_action_expert(ours_action, gemma_expert.model.layers, gemma_expert.model.norm)

    ours_time_mlp = Pi05TimeMLP(action_hidden).to(device, dtype=dtype)
    with torch.no_grad():
        ours_time_mlp.linear_in.weight.copy_(model.time_mlp_in.weight)
        ours_time_mlp.linear_in.bias.copy_(model.time_mlp_in.bias)
        ours_time_mlp.linear_out.weight.copy_(model.time_mlp_out.weight)
        ours_time_mlp.linear_out.bias.copy_(model.time_mlp_out.bias)

    # Mock cache handle pre-populated with lerobot's prefix KV
    suffix_positions = torch.arange(horizon, device=device, dtype=torch.long) + prefix_len
    handle = _MockCacheHandle(head_dim=head_dim, suffix_positions=suffix_positions)
    for layer_idx in range(cfg.num_layers):
        layer_cache = past_key_values.layers[layer_idx]
        if hasattr(layer_cache, "keys"):
            k = layer_cache.keys
            v = layer_cache.values
        else:
            k, v = layer_cache
        handle._prefix_kv[layer_idx] = (
            k[0].transpose(0, 1).contiguous(),
            v[0].transpose(0, 1).contiguous(),
        )

    # Run mminf 10-step Euler flow-matching loop
    num_steps = config.num_inference_steps
    dt = -1.0 / num_steps
    x_t = noise.clone()
    time = torch.tensor(1.0, device=device, dtype=dtype)
    for _ in range(num_steps):
        time_emb = sincos_timestep_embedding(
            time.unsqueeze(0),
            dim=action_hidden,
            min_period=config.min_period,
            max_period=config.max_period,
        ).squeeze(0)
        adarms_cond = ours_time_mlp(time_emb.to(dtype))
        suffix = model.action_in_proj(x_t[0])
        suffix_out = ours_action(query_sequence=suffix, cache_handle=handle, adarms_cond=adarms_cond)
        v_t = model.action_out_proj(suffix_out)
        x_t = x_t + dt * v_t.unsqueeze(0)
        time = time + dt
    ours_actions = x_t

    max_delta = (ours_actions - ref_actions).abs().max().item()
    mean_delta = (ours_actions - ref_actions).abs().mean().item()
    mean_rel = ((ours_actions - ref_actions).abs() / (ref_actions.abs() + 1e-6)).mean().item()
    print(
        f"\nPi0.5 e2e (lerobot/pi05_base): max abs delta = {max_delta:.4e}, "
        f"mean abs delta = {mean_delta:.4e}, mean rel err = {mean_rel:.4e}"
    )
    # Observed on lerobot/pi05_base: max delta ~2e-4 on ref abs max ~0.44.
    assert max_delta < 5e-4, f"max delta {max_delta:.4e} too large"
    assert mean_rel < 1e-2, f"mean rel err {mean_rel:.4e} too large"


def test_pi05_paligemma_expert_matches_lerobot_real_weights():
    """Compare ``Pi05PaliGemmaExpert`` against lerobot's PaliGemma forward.

    The action-expert e2e test bypasses mminf's PaliGemma layers entirely
    (it uses lerobot's PaliGemma to compute the prefix KV cache, then runs
    only the action expert). This test closes that gap by validating that
    mminf's Pi05PaliGemmaExpert produces:
      * the same final hidden state, and
      * the same per-layer K, V tensors

    that lerobot writes during prefill, on the same prefix embeddings.
    """
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    device = torch.device("cuda")
    dtype = torch.float32
    seed = 1
    torch.manual_seed(seed)

    policy = PI05Policy.from_pretrained(PI05_REPO).to(device).eval()
    model = policy.model
    paligemma = model.paligemma_with_expert.paligemma
    lm = paligemma.model.language_model

    head_dim = lm.layers[0].self_attn.head_dim
    num_qo_heads = lm.layers[0].self_attn.config.num_attention_heads
    num_kv_heads = lm.layers[0].self_attn.config.num_key_value_heads
    pali_hidden = lm.layers[0].self_attn.config.hidden_size
    pali_intermediate = lm.layers[0].mlp.gate_proj.out_features
    rms_eps = lm.layers[0].input_layernorm.eps

    # Deterministic input prefix embeddings
    bsize = 1
    g = torch.Generator(device=device).manual_seed(seed)
    images = [torch.rand(bsize, 3, 224, 224, device=device, generator=g) * 2 - 1 for _ in range(3)]
    img_masks = [torch.ones(bsize, dtype=torch.bool, device=device) for _ in range(3)]
    tokens = torch.randint(0, 200, (bsize, 4), device=device, generator=g)
    masks = torch.ones(bsize, 4, dtype=torch.bool, device=device)

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        [i.to(dtype) for i in images], img_masks, tokens, masks
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
    paligemma.model.language_model.config._attn_implementation = "eager"
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)

    ref_outputs, ref_past_kv = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_pos,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    # PaliGemmaWithExpertModel returns [prefix_output, suffix_output]; we want
    # the prefix output (the only one populated when suffix is None).
    ref_hidden = ref_outputs[0]
    prefix_len = int(prefix_pad_masks.sum(dim=-1).item())

    # Build mminf Pi05PaliGemmaExpert with matching dims
    cfg = Pi05Config(
        hidden_size=pali_hidden,
        num_layers=len(lm.layers),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        pali_intermediate_size=pali_intermediate,
        rms_norm_eps=rms_eps,
    )
    ours = Pi05PaliGemmaExpert(cfg).to(device, dtype=dtype)
    _copy_paligemma_expert(ours, lm.layers, lm.norm)

    # Run mminf with a capturing mock cache handle
    handle = _PrefixCacheCapture(head_dim=head_dim, prefix_len=prefix_len)
    # Pi05PaliGemmaExpert expects [seq, hidden] (no batch dim) — strip B=1.
    ours_hidden = ours(
        query_sequence=prefix_embs[0],
        cache_handle=handle,
        write_cache=False,  # MockCacheHandle doesn't track seq_lens
    )

    # ----- Compare final hidden states -----
    # PaliGemma's layer outputs have very large magnitudes (~427 at layer 0,
    # growing to ~27000 by layer 17 — Gemma residual + non-normalized inputs).
    # The right metric is RELATIVE error: float32 mantissa precision is ~1e-7
    # but with 18 layers of compounding accumulation we expect ~1e-4 to 1e-3.
    ref_hidden_b1 = ref_hidden[0]  # [seq, hidden]
    h_max = (ours_hidden - ref_hidden_b1).abs().max().item()
    h_ref_abs = ref_hidden_b1.abs().max().item()
    h_rel_max = h_max / h_ref_abs
    print(
        f"\nPaliGemma prefix hidden state: max delta = {h_max:.4e}, "
        f"ref abs max = {h_ref_abs:.4f}, max rel = {h_rel_max:.4e}"
    )
    # Per-layer divergence is float32 precision (~1.5e-5 single-layer);
    # 18-layer compounding from reduction-order differences inside CUDA matmul
    # vs einsum kernels gives ~5e-3 by the final layer. The action expert
    # e2e test (above) further validates that this level of agreement is
    # sufficient for the downstream action trajectory to match to ~2e-4.
    assert h_rel_max < 1e-2, f"hidden state max rel err {h_rel_max:.4e} too large"

    # ----- Compare per-layer K, V against lerobot DynamicCache -----
    # Same reasoning: use relative tolerance against the per-layer K/V abs max.
    worst_layer_rel = 0.0
    for layer_idx in range(cfg.num_layers):
        our_k, our_v = handle.captured_kv[layer_idx]
        layer_cache = ref_past_kv.layers[layer_idx]
        if hasattr(layer_cache, "keys"):
            ref_k_full = layer_cache.keys
            ref_v_full = layer_cache.values
        else:
            ref_k_full, ref_v_full = layer_cache
        ref_k = ref_k_full[0].transpose(0, 1).contiguous()
        ref_v = ref_v_full[0].transpose(0, 1).contiguous()
        k_rel = (our_k - ref_k).abs().max().item() / max(ref_k.abs().max().item(), 1e-6)
        v_rel = (our_v - ref_v).abs().max().item() / max(ref_v.abs().max().item(), 1e-6)
        worst_layer_rel = max(worst_layer_rel, k_rel, v_rel)
    print(f"PaliGemma per-layer K,V: worst rel err = {worst_layer_rel:.4e}")
    # Same precision-accumulation reasoning as for the hidden state.
    assert worst_layer_rel < 1e-2, f"layer KV rel err {worst_layer_rel:.4e} too large"


def test_pi05_full_stack_matches_lerobot_real_weights():
    """End-to-end mminf vs lerobot using **only** mminf code for the math.

    Unlike the action-expert e2e test (which feeds lerobot's prefix KV cache
    into mminf's action expert), this test runs the entire pipeline through
    mminf:

      1. Pi05PaliGemmaExpert.forward over the prefix embeddings produces the
         per-layer K, V (captured by a mock cache handle).
      2. Pi05ActionExpert runs the 10-step Euler flow-matching loop, reading
         the prefix KV cache that mminf itself just wrote (not lerobot's).
      3. Final action trajectory is compared to lerobot's sample_actions
         output.

    This is the strictest validation: the only place lerobot is used is to
    embed images via SigLIP/PaliGemma's get_image_features (we share that)
    and to compute the reference action trajectory. Every transformer layer
    on the mminf side runs through mminf code with copied weights.
    """
    from lerobot.policies.pi05 import PI05Policy

    device = torch.device("cuda")
    dtype = torch.float32
    seed = 0
    torch.manual_seed(seed)

    policy = PI05Policy.from_pretrained(PI05_REPO).to(device).eval()
    model = policy.model
    config = policy.config
    paligemma = model.paligemma_with_expert.paligemma
    gemma_expert = model.paligemma_with_expert.gemma_expert
    lm = paligemma.model.language_model

    action_hidden = model.action_in_proj.out_features
    head_dim = lm.layers[0].self_attn.head_dim
    num_qo_heads = lm.layers[0].self_attn.config.num_attention_heads
    num_kv_heads = lm.layers[0].self_attn.config.num_key_value_heads
    pali_hidden = lm.layers[0].self_attn.config.hidden_size
    pali_intermediate = lm.layers[0].mlp.gate_proj.out_features
    rms_eps = lm.layers[0].input_layernorm.eps

    # Deterministic input
    bsize = 1
    horizon = config.chunk_size
    action_dim = config.max_action_dim
    g = torch.Generator(device=device).manual_seed(seed)
    images = [
        torch.rand(bsize, 3, 224, 224, device=device, generator=g) * 2 - 1
        for _ in range(3)
    ]
    img_masks = [torch.ones(bsize, dtype=torch.bool, device=device) for _ in range(3)]
    tokens = torch.randint(0, 200, (bsize, 4), device=device, generator=g)
    masks = torch.ones(bsize, 4, dtype=torch.bool, device=device)
    noise = torch.randn(bsize, horizon, action_dim, device=device, generator=g, dtype=torch.float32)

    # ----- Reference: lerobot end-to-end -----
    ref_actions = model.sample_actions(
        images=[i.to(dtype) for i in images],
        img_masks=img_masks,
        tokens=tokens,
        masks=masks,
        noise=noise,
        num_steps=config.num_inference_steps,
    )

    # ----- Build mminf PaliGemma expert and copy weights -----
    pali_cfg = Pi05Config(
        hidden_size=pali_hidden,
        num_layers=len(lm.layers),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        pali_intermediate_size=pali_intermediate,
        rms_norm_eps=rms_eps,
    )
    ours_pali = Pi05PaliGemmaExpert(pali_cfg).to(device, dtype=dtype)
    _copy_paligemma_expert(ours_pali, lm.layers, lm.norm)

    # ----- Build mminf action expert with matching dims -----
    ae_cfg = Pi05Config(
        hidden_size=pali_hidden,
        action_hidden_size=action_hidden,
        num_layers=len(gemma_expert.model.layers),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        action_intermediate_size=gemma_expert.model.layers[0].mlp.gate_proj.out_features,
        rms_norm_eps=gemma_expert.model.layers[0].input_layernorm.eps,
    )
    ours_action = Pi05ActionExpert(ae_cfg).to(device, dtype=dtype)
    _copy_action_expert(ours_action, gemma_expert.model.layers, gemma_expert.model.norm)

    ours_time_mlp = Pi05TimeMLP(action_hidden).to(device, dtype=dtype)
    with torch.no_grad():
        ours_time_mlp.linear_in.weight.copy_(model.time_mlp_in.weight)
        ours_time_mlp.linear_in.bias.copy_(model.time_mlp_in.bias)
        ours_time_mlp.linear_out.weight.copy_(model.time_mlp_out.weight)
        ours_time_mlp.linear_out.bias.copy_(model.time_mlp_out.bias)

    # ----- Embed prefix via lerobot's embed_prefix (SigLIP + lang embeddings) -----
    prefix_embs, prefix_pad_masks, _ = model.embed_prefix(
        [i.to(dtype) for i in images], img_masks, tokens, masks
    )
    prefix_len = int(prefix_pad_masks.sum(dim=-1).item())

    # ----- Step 1: run mminf PaliGemma to populate the per-layer KV cache -----
    prefill_handle = _PrefixCacheCapture(head_dim=head_dim, prefix_len=prefix_len)
    ours_pali(
        query_sequence=prefix_embs[0],
        cache_handle=prefill_handle,
        write_cache=False,
    )

    # ----- Step 2: feed mminf's captured KV cache into the action expert handle -----
    suffix_positions = torch.arange(horizon, device=device, dtype=torch.long) + prefix_len
    action_handle = _MockCacheHandle(head_dim=head_dim, suffix_positions=suffix_positions)
    for layer_idx in range(ae_cfg.num_layers):
        action_handle._prefix_kv[layer_idx] = prefill_handle.captured_kv[layer_idx]

    # ----- Step 3: run mminf 10-step Euler flow-matching loop -----
    num_steps = config.num_inference_steps
    dt = -1.0 / num_steps
    x_t = noise.clone()
    time = torch.tensor(1.0, device=device, dtype=dtype)
    for _ in range(num_steps):
        time_emb = sincos_timestep_embedding(
            time.unsqueeze(0),
            dim=action_hidden,
            min_period=config.min_period,
            max_period=config.max_period,
        ).squeeze(0)
        adarms_cond = ours_time_mlp(time_emb.to(dtype))
        suffix = model.action_in_proj(x_t[0])
        suffix_out = ours_action(query_sequence=suffix, cache_handle=action_handle, adarms_cond=adarms_cond)
        v_t = model.action_out_proj(suffix_out)
        x_t = x_t + dt * v_t.unsqueeze(0)
        time = time + dt
    ours_actions = x_t

    max_delta = (ours_actions - ref_actions).abs().max().item()
    mean_delta = (ours_actions - ref_actions).abs().mean().item()
    mean_rel = ((ours_actions - ref_actions).abs() / (ref_actions.abs() + 1e-6)).mean().item()
    print(
        f"\nPi0.5 FULL STACK e2e (lerobot/pi05_base): max abs delta = {max_delta:.4e}, "
        f"mean abs delta = {mean_delta:.4e}, mean rel err = {mean_rel:.4e}"
    )
    print(f"  ref abs max: {ref_actions.abs().max().item():.4f}")
    print(f"  first 4 ref:  {ref_actions[0, 0, :4].cpu().tolist()}")
    print(f"  first 4 ours: {ours_actions[0, 0, :4].cpu().tolist()}")
    # mminf's PaliGemma diverges from lerobot's by ~3e-3 in K/V (float32
    # accumulation). That error feeds into the action expert's attention,
    # so the final action trajectory shows somewhat larger drift than the
    # action-expert-only test (~2e-4). Allow up to 1e-2 max abs delta and
    # 5e-2 mean relative.
    assert max_delta < 1e-2, f"full-stack max delta {max_delta:.4e} too large"
    assert mean_rel < 5e-2, f"full-stack mean rel err {mean_rel:.4e} too large"
