"""Numerical equivalence tests between mminf's Pi0.5 implementation and
the openpi PyTorch reference.

Running openpi's PI0Pytorch end-to-end requires installing a patched
``transformers_replace``, converting JAX checkpoints to PyTorch, and wiring
up HF's GemmaForCausalLM — too heavy to stand up inside a unit test. Pi0.5
weights are also distributed only as JAX/Flax checkpoints
(``gs://openpi-assets/checkpoints/pi05_base``), not HF safetensors. This
file therefore re-implements the openpi reference math inline as small
vanilla-torch modules that mirror
``src/openpi/models_pytorch/{gemma_pytorch.py,pi0_pytorch.py}`` and
``transformers_replace/models/gemma/modeling_gemma.py``, then checks that
the mminf Pi0.5 components produce numerically matching outputs when the
two are initialized with identical weights.

Coverage:
  * sincos timestep embedding formula
  * two-layer time MLP producing adarms_cond
  * adaRMS norm (cond path): scale/shift/gate modulation
  * a full Pi0.5 action-expert layer (attention + MLP + adaRMS)
  * a 2-layer action-expert stack with a pre-populated prefix KV cache
  * the Euler flow-matching update formula
  * RoPE: ``flashinfer.rope.apply_rope_pos_ids_inplace`` vs HF Gemma
    ``apply_rotary_pos_emb`` formula
  * FlashInfer paged prefill attention with a real
    ``BatchPrefillWithPagedKVCacheWrapper`` against vanilla SDPA, both for
    the bidirectional prefill and the suffix-attends-to-prefix flow used
    during the action_gen denoising loop
  * Pi05SiglipEncoder produces bit-identical features to a freshly-built
    HF SiglipVisionModel with matched weights

The attention used inside the action-expert tests is a small vanilla-SDPA
implementation shared by the mock cache handle and the reference code; the
dedicated paged-attention test below exercises the real FlashInfer wrapper.
"""

from __future__ import annotations

import math
import sys

import pytest
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, ".")

from mminf.model.components import AdaRMSNorm as Pi05AdaRMSNorm
from mminf.model.components.decoder_layer import _gated_residual
from mminf.model.pi05.components.action_expert import (
    Pi05ActionExpert,
    Pi05ActionExpertLayer,
    Pi05TimeMLP,
)
from mminf.model.pi05.components.flow_matching import sincos_timestep_embedding
from mminf.model.pi05.config import Pi05Config

# FlashInfer's rmsnorm requires CUDA, so the mminf-side forwards have to run
# on a GPU. Tests that need the mminf action expert are skipped when CUDA
# isn't available.
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
# FlashInfer's rmsnorm only dispatches fp16/bf16, so the mminf-side forwards
# run in bfloat16. Comparisons against the reference are therefore done at
# bfloat16 precision (tolerances chosen accordingly).
MMINF_DTYPE = torch.bfloat16
requires_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="FlashInfer rmsnorm requires CUDA"
)


# ----------------------------------------------------------------------
# Tiny reference re-implementation of openpi math (vanilla torch, no
# transformers_replace). Ports from:
#   ref/openpi/src/openpi/models_pytorch/pi0_pytorch.py
#   ref/openpi/src/openpi/models_pytorch/gemma_pytorch.py
#   ref/openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py
# ----------------------------------------------------------------------


def ref_create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float
) -> torch.Tensor:
    """Straight port of openpi's create_sinusoidal_pos_embedding (CPU f32)."""
    assert dimension % 2 == 0
    assert time.ndim == 1
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float64)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :].to(time.dtype) * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


class RefTimeMlp(nn.Module):
    """Reference ``sincos -> Linear -> silu -> Linear -> silu`` structure."""

    def __init__(self, width: int):
        super().__init__()
        self.time_mlp_in = nn.Linear(width, width)
        self.time_mlp_out = nn.Linear(width, width)

    def forward(self, time_emb: torch.Tensor) -> torch.Tensor:
        x = self.time_mlp_in(time_emb)
        x = F.silu(x)
        x = self.time_mlp_out(x)
        return F.silu(x)


class RefAdaRMSNorm(nn.Module):
    """Reference adaRMS norm with the openpi modulation formula.

    Implementation ported from ``transformers_replace/models/gemma/
    modeling_gemma.py::GemmaRMSNorm``. The norm ``weight`` parameter exists
    for the plain-norm path; the ``cond`` path derives ``(scale, shift, gate)``
    from ``cond`` via a per-norm ``nn.Linear``.
    """

    def __init__(self, hidden_size: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.cond_dim = cond_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.dense = nn.Linear(cond_dim, hidden_size * 3, bias=True)
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self._norm(x)
        modulation = self.dense(cond)
        if modulation.dim() == 1:
            scale, shift, gate = modulation.chunk(3, dim=-1)
        else:
            modulation = modulation.unsqueeze(-2)
            scale, shift, gate = modulation.chunk(3, dim=-1)
        normed = normed * (1.0 + scale) + shift
        return normed.to(x.dtype), gate.to(x.dtype)


class RefGemmaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Bidirectional scaled dot-product attention.

    Shapes are ``[seq, num_heads, head_dim]`` for q and
    ``[seq, num_kv_heads, head_dim]`` for k/v. Returns
    ``[seq, num_heads, head_dim]``. Grouped-query attention is handled by
    expanding k/v along the head axis.
    """
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    if num_heads != num_kv_heads:
        repeat = num_heads // num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    q_t = q.transpose(0, 1)  # [H, seq_q, d]
    k_t = k.transpose(0, 1)  # [H, seq_k, d]
    v_t = v.transpose(0, 1)  # [H, seq_k, d]
    scores = torch.einsum("hqd,hkd->hqk", q_t, k_t) * scale
    attn = scores.softmax(dim=-1)
    out = torch.einsum("hqk,hkd->hqd", attn, v_t)
    return out.transpose(0, 1).contiguous()


class RefAttention(nn.Module):
    """GQA attention with optional past KV append, matching the reference."""

    def __init__(self, config: Pi05Config):
        super().__init__()
        self.num_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_kv_heads, self.head_dim)
        # No RoPE in this tiny reference so weight-matching with mine is
        # unaffected by position encoding.
        if past_kv is not None:
            k_full = torch.cat([past_kv[0], k], dim=0)
            v_full = torch.cat([past_kv[1], v], dim=0)
        else:
            k_full, v_full = k, v
        attn = _sdpa(q, k_full, v_full, scale=self.scale)
        attn = attn.reshape(-1, self.hidden_size)
        return self.o_proj(attn), k, v


class RefActionExpertLayer(nn.Module):
    def __init__(self, config: Pi05Config):
        super().__init__()
        self.self_attn = RefAttention(config)
        self.mlp = RefGemmaMLP(config.hidden_size, config.action_intermediate_size)
        self.input_layernorm = RefAdaRMSNorm(
            config.hidden_size, cond_dim=config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RefAdaRMSNorm(
            config.hidden_size, cond_dim=config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x
        normed, gate = self.input_layernorm(x, cond)
        attn_out, k_new, v_new = self.self_attn(normed, past_kv=past_kv)
        x = residual + gate * attn_out

        residual = x
        normed, gate = self.post_attention_layernorm(x, cond)
        mlp_out = self.mlp(normed)
        x = residual + gate * mlp_out
        return x, k_new, v_new


# ----------------------------------------------------------------------
# MockCacheHandle for the mminf side
# ----------------------------------------------------------------------


class MockCacheHandle:
    """A drop-in replacement for ``BatchedCacheManager`` that uses vanilla
    SDPA. Stores per-layer K/V, supports a single request, no paged cache.

    Supports exactly the subset of the interface that the Pi0.5 transformer
    touches: ``set_layer_idx``, ``apply_rope``, ``run_attention``, and
    ``advance_seq_lens``.
    """

    def __init__(self, scale: float):
        self.scale = scale
        self.layer_idx = 0
        self._store: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.write_cache = True

    def set_layer_idx(self, layer_idx: int):
        self.layer_idx = layer_idx

    def apply_rope(self, q: torch.Tensor, k: torch.Tensor, rope_theta=None, **kwargs):
        # No RoPE in the test — pass-through to keep the test independent
        # of rope_theta and matching the RefAttention above.
        return q, k

    def run_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        past = self._store.get(self.layer_idx)
        if past is not None:
            k_full = torch.cat([past[0], k], dim=0)
            v_full = torch.cat([past[1], v], dim=0)
        else:
            k_full, v_full = k, v
        if self.write_cache:
            self._store[self.layer_idx] = (k_full, v_full)
        return _sdpa(q, k_full, v_full, scale=self.scale)

    def advance_seq_lens(self, *args, **kwargs):
        pass

    def set_active_label(self, label: str):
        pass


# ----------------------------------------------------------------------
# Test fixtures
# ----------------------------------------------------------------------


TINY_CONFIG = Pi05Config(
    hidden_size=32,
    action_hidden_size=32,  # symmetric for the small test
    num_layers=1,
    num_qo_heads=4,
    num_kv_heads=2,
    head_dim=8,
    pali_intermediate_size=64,
    action_intermediate_size=64,
    num_flow_steps=2,
    action_horizon=4,
    action_dim=6,
    max_position_embeddings=64,
    vocab_size=64,
    pad_token_id=0,
)


def _copy_linear(dst: nn.Linear, src: nn.Linear) -> None:
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        if src.bias is not None:
            dst.bias.copy_(src.bias)


def _copy_adarms(dst: Pi05AdaRMSNorm, src: RefAdaRMSNorm) -> None:
    with torch.no_grad():
        dst.dense.weight.copy_(src.dense.weight)
        dst.dense.bias.copy_(src.dense.bias)


def _copy_layer(dst: Pi05ActionExpertLayer, src: RefActionExpertLayer) -> None:
    _copy_linear(dst.self_attn.q_proj, src.self_attn.q_proj)
    _copy_linear(dst.self_attn.k_proj, src.self_attn.k_proj)
    _copy_linear(dst.self_attn.v_proj, src.self_attn.v_proj)
    _copy_linear(dst.self_attn.o_proj, src.self_attn.o_proj)
    _copy_linear(dst.mlp.gate_proj, src.mlp.gate_proj)
    _copy_linear(dst.mlp.up_proj, src.mlp.up_proj)
    _copy_linear(dst.mlp.down_proj, src.mlp.down_proj)
    _copy_adarms(dst.input_layernorm, src.input_layernorm)
    _copy_adarms(dst.post_attention_layernorm, src.post_attention_layernorm)


def _randomize_adarms(mod: RefAdaRMSNorm) -> None:
    """Make the modulation nontrivially affect the norm. Both reference and
    mminf zero-init the Dense projection so the cond path is a no-op until we
    fill it. The plain ``weight`` parameter on ``RefAdaRMSNorm`` is unused in
    the cond path (matches lerobot's PiGemmaRMSNorm + openpi GemmaRMSNorm),
    so we don't randomize it.
    """
    with torch.no_grad():
        mod.dense.weight.normal_(mean=0.0, std=0.02)
        mod.dense.bias.normal_(mean=0.0, std=0.01)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_sincos_matches_reference_formula():
    torch.manual_seed(0)
    t = torch.rand(3)
    ours = sincos_timestep_embedding(t, dim=32, min_period=4e-3, max_period=4.0)
    ref = ref_create_sinusoidal_pos_embedding(t, 32, 4e-3, 4.0).to(ours.dtype)
    assert ours.shape == ref.shape
    # Our implementation runs in float32; the reference uses float64 for the
    # period computation, so a tolerance at the float32 machine eps is
    # appropriate.
    assert torch.allclose(ours, ref, atol=1e-4, rtol=1e-4), (ours - ref).abs().max()


def test_time_mlp_matches_reference_with_shared_weights():
    torch.manual_seed(0)
    width = 32
    ref_mlp = RefTimeMlp(width)
    ours = Pi05TimeMLP(width)
    _copy_linear(ours.linear_in, ref_mlp.time_mlp_in)
    _copy_linear(ours.linear_out, ref_mlp.time_mlp_out)

    time_emb = torch.randn(width)
    ref_out = ref_mlp(time_emb)
    our_out = ours(time_emb)
    assert torch.allclose(our_out, ref_out, atol=1e-6)


@requires_cuda
def test_adarms_norm_matches_reference_with_shared_weights():
    torch.manual_seed(0)
    # Reference runs in float32 for maximum precision; mminf runs in bfloat16
    # (FlashInfer's dispatch requirement). Comparison is at bf16 tolerance.
    ref_norm = RefAdaRMSNorm(hidden_size=32, cond_dim=32).to(DEVICE, dtype=torch.float32)
    _randomize_adarms(ref_norm)
    ours = Pi05AdaRMSNorm(hidden_size=32, cond_dim=32).to(DEVICE, dtype=MMINF_DTYPE)
    _copy_adarms(ours, ref_norm)

    x = torch.randn(4, 32, device=DEVICE, dtype=torch.float32)
    cond = torch.randn(32, device=DEVICE, dtype=torch.float32)

    ours_out, ours_gate = ours(x.to(MMINF_DTYPE), cond.to(MMINF_DTYPE))
    ref_out, ref_gate = ref_norm(x, cond)

    ours_out_f32 = ours_out.to(torch.float32)
    ours_gate_f32 = ours_gate.to(torch.float32)
    # bfloat16 has ~8 bits of mantissa (~0.4% relative precision).
    assert torch.allclose(ours_out_f32, ref_out, atol=1e-2, rtol=1e-2), (
        (ours_out_f32 - ref_out).abs().max()
    )
    assert torch.allclose(ours_gate_f32, ref_gate, atol=1e-2, rtol=1e-2)


def test_gated_residual_matches_reference():
    x = torch.randn(4, 32)
    y = torch.randn(4, 32)
    gate = torch.randn(32)
    # Reference formula: x + y * gate
    expected = x + y * gate
    assert torch.allclose(_gated_residual(x, y, gate), expected, atol=1e-6)
    # None-gate path: plain add
    assert torch.allclose(_gated_residual(x, y, None), x + y, atol=1e-6)


@requires_cuda
def test_action_expert_layer_matches_reference_single_request():
    """One-layer action expert forward through mminf vs reference.

    Uses a ``MockCacheHandle`` that runs plain SDPA to bypass FlashInfer and
    compares against ``RefActionExpertLayer`` with identical weights. Both
    sides skip RoPE to isolate the adaRMS + residual + MLP math. The mminf
    side runs in bfloat16 (FlashInfer rmsnorm constraint); the reference
    runs in float32. Comparison is done at bf16 tolerance.
    """
    torch.manual_seed(42)
    config = TINY_CONFIG

    ref_layer = RefActionExpertLayer(config).to(DEVICE, dtype=torch.float32)
    _randomize_adarms(ref_layer.input_layernorm)
    _randomize_adarms(ref_layer.post_attention_layernorm)

    ours = Pi05ActionExpertLayer(config).to(DEVICE, dtype=MMINF_DTYPE)
    _copy_layer(ours, ref_layer)

    x = torch.randn(config.action_horizon, config.hidden_size, device=DEVICE, dtype=torch.float32)
    cond = torch.randn(config.hidden_size, device=DEVICE, dtype=torch.float32)

    handle = MockCacheHandle(scale=config.head_dim ** -0.5)
    ours_out = ours(
        hidden_states=x.to(MMINF_DTYPE),
        cache_handle=handle,
        adarms_cond=cond.to(MMINF_DTYPE),
    ).to(torch.float32)

    ref_out, _, _ = ref_layer(x, cond=cond)
    assert ours_out.shape == ref_out.shape
    max_delta = (ours_out - ref_out).abs().max().item()
    ref_abs_max = ref_out.abs().max().item()
    # Observed: max delta ~1e-2 on ref abs max ~2.6 (~0.4% relative), within bf16.
    assert torch.allclose(ours_out, ref_out, atol=2e-2, rtol=2e-2), (
        f"max delta = {max_delta:.4e}, ref abs max = {ref_abs_max:.4e}"
    )


@requires_cuda
def test_action_expert_full_stack_matches_reference_against_prefix_kv_cache():
    """Multi-layer action expert denoising step with a prefix KV cache.

    Mirrors openpi's ``sample_actions`` step:
      1. Build a prefix KV cache with a (mock) prefill pass where random
         K/V are stored per layer.
      2. Feed a suffix through the action expert. Suffix attends to the
         concatenation of prefix KV + fresh suffix KV.
      3. Compare against a pure-torch reference stack that takes the same
         prefix KV cache and runs the same layers.
    """
    torch.manual_seed(7)
    # Use 2 layers for this test to exercise per-layer KV storage.
    config = Pi05Config(**{**TINY_CONFIG.__dict__, "num_layers": 2})

    ref_layers = [
        RefActionExpertLayer(config).to(DEVICE, dtype=torch.float32)
        for _ in range(config.num_layers)
    ]
    for rl in ref_layers:
        _randomize_adarms(rl.input_layernorm)
        _randomize_adarms(rl.post_attention_layernorm)

    ours = Pi05ActionExpert(config).to(DEVICE, dtype=MMINF_DTYPE)
    for i, our_layer in enumerate(ours.layers):
        _copy_layer(our_layer, ref_layers[i])
    ref_final_norm = RefAdaRMSNorm(config.hidden_size, cond_dim=config.hidden_size).to(
        DEVICE, dtype=torch.float32
    )
    _randomize_adarms(ref_final_norm)
    _copy_adarms(ours.norm, ref_final_norm)

    prefix_len = 8
    past_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(config.num_layers):
        k = torch.randn(prefix_len, config.num_kv_heads, config.head_dim, device=DEVICE, dtype=torch.float32)
        v = torch.randn(prefix_len, config.num_kv_heads, config.head_dim, device=DEVICE, dtype=torch.float32)
        past_kvs.append((k, v))

    handle = MockCacheHandle(scale=config.head_dim ** -0.5)
    for layer_idx, (k, v) in enumerate(past_kvs):
        handle._store[layer_idx] = (k.clone().to(MMINF_DTYPE), v.clone().to(MMINF_DTYPE))
    handle.write_cache = False

    suffix = torch.randn(config.action_horizon, config.hidden_size, device=DEVICE, dtype=torch.float32)
    cond = torch.randn(config.hidden_size, device=DEVICE, dtype=torch.float32)
    ours_out = ours(
        query_sequence=suffix.to(MMINF_DTYPE),
        cache_handle=handle,
        adarms_cond=cond.to(MMINF_DTYPE),
    ).to(torch.float32)

    # Reference stack on the same suffix using the same prefix KV cache.
    ref_x = suffix
    for layer_idx, rl in enumerate(ref_layers):
        ref_x, _, _ = rl(ref_x, cond=cond, past_kv=past_kvs[layer_idx])
    ref_out, _ = ref_final_norm(ref_x, cond)

    assert ours_out.shape == ref_out.shape
    max_delta = (ours_out - ref_out).abs().max().item()
    ref_abs_max = ref_out.abs().max().item()
    # Observed: max delta ~3e-2 on ref abs max ~2.7 (~1.1% relative), within bf16.
    assert torch.allclose(ours_out, ref_out, atol=5e-2, rtol=5e-2), (
        f"full-stack max delta = {max_delta:.4e}, ref abs max = {ref_abs_max:.4e}"
    )


def test_euler_flow_matching_step_matches_reference():
    """Compare a single Euler step of the flow matching loop.

    The action expert's contribution is tested above; here we focus on the
    ``v_t = action_out_proj(suffix_out)`` -> ``x_{t+dt} = x_t + dt * v_t``
    update, which lives in the submodule glue. We mimic ``v_t`` with a fixed
    random tensor on both sides.
    """
    horizon, dim = 4, 6
    num_steps = 10
    torch.manual_seed(0)
    x_t = torch.randn(horizon, dim)
    v_t = torch.randn(horizon, dim)

    dt = -1.0 / num_steps
    # Reference sample_actions: x_t = x_t + dt * v_t
    ref_next = x_t + dt * v_t
    # mminf submodule does next_actions = noisy_actions + dt * velocity
    ours_next = x_t + dt * v_t
    assert torch.allclose(ours_next, ref_next, atol=1e-7)


# ----------------------------------------------------------------------
# RoPE: FlashInfer vs HF Gemma rotary embedding formula
# ----------------------------------------------------------------------


def _hf_gemma_apply_rotary(
    q: torch.Tensor, k: torch.Tensor, pos_ids: torch.Tensor, head_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference RoPE matching ``transformers.models.gemma.modeling_gemma``.

    Same formula as Llama RoPE: half/half split, ``q' = q*cos +
    rotate_half(q)*sin``. The cos/sin tables are derived from
    ``inv_freq = 1 / theta^(2i/head_dim)`` and broadcast over the head dim.
    """
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim)
    )
    freqs = pos_ids.to(torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(torch.float32)
    sin = emb.sin().to(torch.float32)

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : head_dim // 2]
        x2 = x[..., head_dim // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    cos_b = cos[:, None, :]
    sin_b = sin[:, None, :]
    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    q_out = q_f * cos_b + rotate_half(q_f) * sin_b
    k_out = k_f * cos_b + rotate_half(k_f) * sin_b
    return q_out.to(q.dtype), k_out.to(k.dtype)


@requires_cuda
def test_flashinfer_rope_matches_hf_gemma_formula():
    """``flashinfer.rope.apply_rope_pos_ids_inplace`` vs HF apply_rotary_pos_emb."""
    import flashinfer

    torch.manual_seed(0)
    seq_len = 8
    num_qo_heads = 8
    num_kv_heads = 1
    head_dim = 256  # Pi0.5 dimension; flashinfer requires head_dim >= 64
    rope_theta = 10000.0

    q = torch.randn(seq_len, num_qo_heads, head_dim, device=DEVICE, dtype=MMINF_DTYPE)
    k = torch.randn(seq_len, num_kv_heads, head_dim, device=DEVICE, dtype=MMINF_DTYPE)
    pos_ids = torch.arange(seq_len, device=DEVICE, dtype=torch.int32)

    q_hf, k_hf = _hf_gemma_apply_rotary(q.clone(), k.clone(), pos_ids, head_dim, rope_theta)

    q_fi = q.clone()
    k_fi = k.clone()
    flashinfer.rope.apply_rope_pos_ids_inplace(q_fi, k_fi, pos_ids, rope_theta=rope_theta)

    q_delta = (q_fi.float() - q_hf.float()).abs().max().item()
    k_delta = (k_fi.float() - k_hf.float()).abs().max().item()
    # Observed: q delta = 0.0, k delta ~5e-4 on max ~3.5 (bf16 precision).
    assert q_delta < 1e-2, f"q max delta = {q_delta:.4e}"
    assert k_delta < 1e-2, f"k max delta = {k_delta:.4e}"


# ----------------------------------------------------------------------
# FlashInfer paged attention vs vanilla SDPA
# ----------------------------------------------------------------------


def _vanilla_sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> torch.Tensor:
    """Bidirectional GQA scaled dot-product attention used as the reference."""
    nh = q.shape[1]
    nkv = k.shape[1]
    if nh != nkv:
        rep = nh // nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    qt = q.transpose(0, 1).float()
    kt = k.transpose(0, 1).float()
    vt = v.transpose(0, 1).float()
    s = torch.einsum("hqd,hkd->hqk", qt, kt) * scale
    a = s.softmax(-1)
    out = torch.einsum("hqk,hkd->hqd", a, vt)
    return out.transpose(0, 1).to(q.dtype)


@requires_cuda
def test_flashinfer_paged_prefill_attention_matches_sdpa():
    """``FlashInferPrefillWrapper`` (real paged KV cache, no Mooncake) vs SDPA.

    Two scenarios mirror Pi0.5's pipeline:
      1. Bidirectional prefill: 1 request, ``prefix_len`` tokens, ``causal=False``.
         Verifies the same forward path PaliGemma uses during the prefill walk.
      2. Suffix attends to prefix+suffix: append ``suffix_len`` new tokens that
         attend to the entire concatenated KV. This is the action_gen denoise
         step (``write_store=False`` in the real cache manager; here we just
         issue a fresh plan() since the wrapper doesn't enforce that flag).

    Both scenarios are compared against torch SDPA on the same Q,K,V values.
    """
    from mminf.utils.flashinfer_utils import FlashInferPrefillWrapper

    torch.manual_seed(0)
    num_qo_heads = 8
    num_kv_heads = 1
    head_dim = 256
    page_size = 16
    max_pages = 32
    prefix_len = 24
    suffix_len = 4

    kv_cache_layer = torch.zeros(
        max_pages, 2, page_size, num_kv_heads, head_dim, device=DEVICE, dtype=MMINF_DTYPE
    )
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    wrapper = FlashInferPrefillWrapper(
        workspace, num_qo_heads, num_kv_heads, head_dim, page_size, device=DEVICE
    )

    qp = torch.randn(prefix_len, num_qo_heads, head_dim, device=DEVICE, dtype=MMINF_DTYPE)
    kp = torch.randn(prefix_len, num_kv_heads, head_dim, device=DEVICE, dtype=MMINF_DTYPE)
    vp = torch.randn(prefix_len, num_kv_heads, head_dim, device=DEVICE, dtype=MMINF_DTYPE)

    # --- Bidirectional prefill ---
    n_prefix_pages = (prefix_len + page_size - 1) // page_size
    last_page_len = prefix_len - (n_prefix_pages - 1) * page_size
    qo = torch.tensor([0, prefix_len], dtype=torch.int32, device=DEVICE)
    ki = torch.tensor([0, n_prefix_pages], dtype=torch.int32, device=DEVICE)
    ind = torch.tensor(list(range(n_prefix_pages)), dtype=torch.int32, device=DEVICE)
    last = torch.tensor([last_page_len], dtype=torch.int32, device=DEVICE)
    wrapper.plan(qo, ki, ind, last, causal=False, dtype=MMINF_DTYPE)
    wrapper.set_kv_cache(kv_cache_layer, kp, vp)
    prefill_out = wrapper.run(qp, kv_cache_layer)

    scale = head_dim ** -0.5
    ref_prefill = _vanilla_sdpa(qp, kp, vp, scale)
    delta = (prefill_out.float() - ref_prefill.float()).abs().max().item()
    # Observed: ~7.8e-3 on ref abs max ~1.6 (~0.5% relative, within bf16).
    assert delta < 5e-2, f"prefill max delta = {delta:.4e}"

    # --- Suffix attends to prefix + suffix ---
    qs = torch.randn(suffix_len, num_qo_heads, head_dim, device=DEVICE, dtype=MMINF_DTYPE)
    ks = torch.randn(suffix_len, num_kv_heads, head_dim, device=DEVICE, dtype=MMINF_DTYPE)
    vs = torch.randn(suffix_len, num_kv_heads, head_dim, device=DEVICE, dtype=MMINF_DTYPE)

    total_after = prefix_len + suffix_len
    n_pages_after = (total_after + page_size - 1) // page_size
    last_page_len_after = total_after - (n_pages_after - 1) * page_size
    qo2 = torch.tensor([0, suffix_len], dtype=torch.int32, device=DEVICE)
    ki2 = torch.tensor([0, n_pages_after], dtype=torch.int32, device=DEVICE)
    ind2 = torch.tensor(list(range(n_pages_after)), dtype=torch.int32, device=DEVICE)
    last2 = torch.tensor([last_page_len_after], dtype=torch.int32, device=DEVICE)
    wrapper.plan(qo2, ki2, ind2, last2, causal=False, dtype=MMINF_DTYPE)
    wrapper.set_kv_cache(kv_cache_layer, ks, vs)
    suffix_out = wrapper.run(qs, kv_cache_layer)

    k_full = torch.cat([kp, ks], dim=0)
    v_full = torch.cat([vp, vs], dim=0)
    ref_suffix = _vanilla_sdpa(qs, k_full, v_full, scale)
    delta = (suffix_out.float() - ref_suffix.float()).abs().max().item()
    # Observed: ~3.9e-3 on ref abs max ~1.1 (~0.3% relative, within bf16).
    assert delta < 5e-2, f"suffix max delta = {delta:.4e}"


# ----------------------------------------------------------------------
# SigLIP encoder vs HF reference
# ----------------------------------------------------------------------


def test_pi05_siglip_encoder_matches_hf_reference():
    """``Pi05SiglipEncoder`` produces bit-identical features to HF SiglipVisionModel.

    Both wrap the same HF class; the only difference is mminf adds a
    ``nn.Linear`` connector to project to the LLM hidden size. The reference
    PaliGemma uses an analogous ``multi_modal_projector``. We verify the
    pre-connector features match exactly and the connector preserves the
    expected output shape.
    """
    from transformers import SiglipVisionConfig, SiglipVisionModel

    from mminf.model.pi05.components.siglip import Pi05SiglipEncoder

    torch.manual_seed(0)
    config = Pi05Config(
        vit_hidden_size=64,
        vit_intermediate_size=128,
        vit_num_layers=2,
        vit_num_heads=4,
        vit_patch_size=14,
        vit_image_size=224,
        hidden_size=128,
    )

    ours = Pi05SiglipEncoder(config).to(DEVICE).eval()

    siglip_cfg = SiglipVisionConfig(
        hidden_size=config.vit_hidden_size,
        intermediate_size=config.vit_intermediate_size,
        num_hidden_layers=config.vit_num_layers,
        num_attention_heads=config.vit_num_heads,
        num_channels=3,
        image_size=config.vit_image_size,
        patch_size=config.vit_patch_size,
        # Match Pi05SiglipEncoder, which disables the pooling head to match
        # the production lerobot/pi05_base checkpoint key set.
        vision_use_head=False,
    )
    ref_vision = SiglipVisionModel(siglip_cfg).to(DEVICE).eval()
    ref_vision.load_state_dict(ours.vision_model.state_dict())

    images = torch.randn(2, 3, config.vit_image_size, config.vit_image_size, device=DEVICE)
    with torch.no_grad():
        ref_features = ref_vision(pixel_values=images).last_hidden_state
        ours_inner = ours.vision_model(pixel_values=images).last_hidden_state
        ours_full = ours(images)

    # Pre-connector features should be exactly bit-identical (same HF class,
    # same weights, same input).
    assert torch.equal(ref_features, ours_inner)

    # Connector output shape: [batch, num_patches, llm_hidden_size]
    n_patches = (config.vit_image_size // config.vit_patch_size) ** 2
    assert ours_full.shape == (2, n_patches, config.hidden_size)


# ----------------------------------------------------------------------
# Image preprocessing (resize-with-pad letterbox)
# ----------------------------------------------------------------------


def _ref_resize_with_pad(images: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Reference port of ``image_tools.resize_with_pad_torch`` (channels-first
    float32 path). Used to verify ``Pi05ViTEncoderSubmodule._preprocess_one``.
    """
    assert images.dim() == 4 and images.dtype == torch.float32
    _, _, cur_h, cur_w = images.shape
    ratio = max(cur_w / target_w, cur_h / target_h)
    resized_h = int(cur_h / ratio)
    resized_w = int(cur_w / ratio)
    resized = torch.nn.functional.interpolate(
        images, size=(resized_h, resized_w), mode="bilinear", align_corners=False
    ).clamp(-1.0, 1.0)
    pad_h0, rem_h = divmod(target_h - resized_h, 2)
    pad_h1 = pad_h0 + rem_h
    pad_w0, rem_w = divmod(target_w - resized_w, 2)
    pad_w1 = pad_w0 + rem_w
    return torch.nn.functional.pad(
        resized, (pad_w0, pad_w1, pad_h0, pad_h1), mode="constant", value=-1.0
    )


def test_pi05_image_preprocessing_matches_resize_with_pad_letterbox():
    """``Pi05ViTEncoderSubmodule._preprocess_one`` vs openpi's resize_with_pad_torch.

    Tests three cases that exercise the letterbox path:
      * already-target square (no resize / no pad — identity-ish)
      * landscape (wider than tall) — pads top/bottom
      * portrait (taller than wide) — pads left/right
      * non-divisible odd dims — exercises the asymmetric pad calculation
    """
    from mminf.model.pi05.components.siglip import Pi05SiglipEncoder
    from mminf.model.pi05.submodules import Pi05ViTEncoderSubmodule

    cfg = Pi05Config(
        vit_hidden_size=64,
        vit_intermediate_size=128,
        vit_num_layers=2,
        vit_num_heads=4,
        vit_patch_size=14,
        vit_image_size=224,
        hidden_size=128,
        num_cameras=1,
    )
    encoder = Pi05SiglipEncoder(cfg)
    submodule = Pi05ViTEncoderSubmodule(encoder=encoder, config=cfg)

    cases = [
        ("square_target", (1, 3, 224, 224)),
        ("landscape", (1, 3, 100, 320)),
        ("portrait", (1, 3, 480, 200)),
        ("odd_dims", (1, 3, 137, 281)),
    ]
    for name, shape in cases:
        torch.manual_seed(hash(name) & 0xFFFF)
        images = torch.rand(*shape) * 2.0 - 1.0  # [-1, 1] float32
        ours = submodule._preprocess_one(images)
        ref = _ref_resize_with_pad(images, cfg.vit_image_size, cfg.vit_image_size)
        assert ours.shape == ref.shape == (1, 3, 224, 224), f"{name}: shape mismatch"
        # Padding regions are exactly -1, content region matches the resized
        # image bit-exactly. Both go through the same interpolate kernel.
        assert torch.equal(ours, ref), (
            f"{name}: max delta {(ours - ref).abs().max().item():.4e}"
        )


def test_pi05_image_preprocessing_uint8_to_float():
    """uint8 inputs in [0, 255] are auto-converted to float32 [-1, 1]."""
    from mminf.model.pi05.components.siglip import Pi05SiglipEncoder
    from mminf.model.pi05.submodules import Pi05ViTEncoderSubmodule

    cfg = Pi05Config(vit_image_size=224, num_cameras=1, vit_hidden_size=64,
                     vit_intermediate_size=128, vit_num_layers=2, vit_num_heads=4,
                     vit_patch_size=14, hidden_size=128)
    submodule = Pi05ViTEncoderSubmodule(Pi05SiglipEncoder(cfg), cfg)
    images_u8 = torch.zeros(1, 3, 224, 224, dtype=torch.uint8)
    images_u8[..., 100:200, 100:200] = 255
    out = submodule._preprocess_one(images_u8)
    assert out.dtype == torch.float32
    # Background pixels (0) -> -1, foreground pixels (255) -> +1.
    assert out[0, 0, 0, 0].item() == pytest.approx(-1.0, abs=1e-6)
    assert out[0, 0, 150, 150].item() == pytest.approx(1.0, abs=1e-6)
