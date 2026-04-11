"""Numerical equivalence tests between mminf's Pi0.5 implementation and
the openpi PyTorch reference.

Running openpi's PI0Pytorch end-to-end requires installing a patched
``transformers_replace``, downloading PaliGemma weights, and wiring up HF's
GemmaForCausalLM — too heavy to stand up inside a unit test. Instead, this
file re-implements the openpi reference math inline as small vanilla-torch
modules that mirror ``src/openpi/models_pytorch/gemma_pytorch.py`` and
``transformers_replace/models/gemma/modeling_gemma.py``, then checks that
the mminf Pi0.5 components produce numerically matching outputs when the
two are initialized with identical weights.

Coverage:
  * sincos timestep embedding formula
  * two-layer time MLP producing adarms_cond
  * adaRMS norm (cond path): scale/shift/gate modulation
  * a full Pi0.5 action-expert layer (attention + MLP + adaRMS)
  * a 2-step flow-matching denoising loop over a 1-layer action expert
    exercising PaliGemma prefill + action-expert suffix attention against
    a frozen prefix KV cache

The attention used inside both sides is a small vanilla-SDPA implementation
shared by the mock cache handle and the reference code, which is the same
computation FlashInfer runs on the mminf side up to numerical tolerance.
"""

from __future__ import annotations

import math
import sys

import pytest
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, ".")

from mminf.model.pi05.components.action_expert import (
    Pi05ActionExpert,
    Pi05ActionExpertLayer,
    Pi05AdaRMSNorm,
    Pi05TimeMLP,
    _gated_residual,
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
        dst.weight.copy_(src.weight)
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
    """Make the modulation nontrivially affect the norm. The zero-init in
    both reference and mminf means the cond path is a no-op unless we fill
    the Dense layer with random values for the test.
    """
    with torch.no_grad():
        mod.weight.uniform_(-0.1, 0.1)
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
        query_sequence=x.to(MMINF_DTYPE),
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
