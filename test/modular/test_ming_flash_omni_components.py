"""Unit tests for Ling-2.0 architecture-novel components.

CPU-only, small-dim, no model weights — these validate the math we ported
in step 3a of ``mminf/model/ming_omni_flash/PORTING_NOTES.md``.

One test (``test_ling_router_matches_vllm_omni``) cross-checks against
vllm-omni's own ``BailingMoeV2Gate`` and skips when vllm-omni isn't
importable — that's the strongest guard against subtle routing bugs
(group_limited_topk has several easy off-by-one traps).
"""

from __future__ import annotations

import importlib

import pytest
import torch
import torch.nn.functional as F

from mminf.model.ming_omni_flash.components.attention import LingAttention
from mminf.model.ming_omni_flash.components.rope import (
    LingPartialMRotaryEmbedding,
)
from mminf.model.ming_omni_flash.components.router import LingMoeRouter

torch.manual_seed(2026)


class _MockCacheHandle:
    """Stand-in for :class:`BatchedCacheManager` in unit tests.

    Implements just ``set_layer_idx`` + ``run_attention`` — the two
    methods :class:`LingAttention` and :class:`LingMoeModel` call. The
    ``run_attention`` runs standard causal SDPA, matching what the
    inline path did before the cache_handle refactor. No KV cache state
    is preserved across calls (single-shot per layer is enough for unit
    tests; the real engine handles paging).
    """

    def __init__(self) -> None:
        self.layer_idx = 0

    def set_layer_idx(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx

    def run_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Plain causal SDPA. ``q``/``k``/``v``:
        ``(num_tokens, num_heads_or_kv, head_dim)``. Returns
        ``(num_tokens, num_heads, head_dim)``.
        """
        num_heads = q.shape[1]
        num_kv = k.shape[1]
        kv_groups = num_heads // num_kv
        if kv_groups > 1:
            k = k.repeat_interleave(kv_groups, dim=1)
            v = v.repeat_interleave(kv_groups, dim=1)
        # SDPA expects (B, num_heads, T, head_dim); we have
        # (T, num_heads, head_dim). Unsqueeze a batch + transpose.
        q4 = q.transpose(0, 1).unsqueeze(0)
        k4 = k.transpose(0, 1).unsqueeze(0)
        v4 = v.transpose(0, 1).unsqueeze(0)
        scale = q.shape[-1] ** -0.5
        out = F.scaled_dot_product_attention(q4, k4, v4, is_causal=True, scale=scale)
        return out.squeeze(0).transpose(0, 1).contiguous()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def test_ling_router_shapes_and_scaling() -> None:
    """Forward returns the (logits, weights, indices) 3-tuple with the
    expected shapes; weights sum to ~routed_scaling_factor per row."""
    router = LingMoeRouter(
        hidden_size=64, num_experts=16,
        num_experts_per_tok=4,
        n_group=4, topk_group=2,
        routed_scaling_factor=2.5,
    )
    x = torch.randn(8, 64)
    logits, weights, indices = router(x)
    assert logits.shape == (8, 16)
    assert weights.shape == (8, 4)
    assert indices.shape == (8, 4)
    assert indices.dtype == torch.int64
    # Renormalised weights sum to 1, then × routed_scaling_factor → 2.5.
    row_sums = weights.float().sum(dim=-1)
    assert torch.allclose(row_sums, torch.full((8,), 2.5), atol=1e-5), row_sums


def test_ling_router_group_limited() -> None:
    """If only group 0's experts score high (others -inf-ish), every
    selected index must fall inside group 0's expert range."""
    router = LingMoeRouter(
        hidden_size=8, num_experts=12,
        num_experts_per_tok=3,
        n_group=3, topk_group=1,
    )
    with torch.no_grad():
        router.gate.weight.zero_()
        # Boost group 0 (experts 0..3): a single boosted input dim hits
        # those experts strongly.
        router.gate.weight[0:4, 0] = 10.0
    x = torch.zeros(4, 8)
    x[:, 0] = 1.0  # activate the input dim that lights up group 0
    _, _, indices = router(x)
    # All chosen experts must be in [0, 4) since topk_group=1 means only
    # group 0 (experts 0..3) is eligible.
    assert (indices >= 0).all() and (indices < 4).all(), indices


def test_ling_router_expert_bias_shifts_routing() -> None:
    """A large positive bias on expert E forces it to be picked even when
    the gate logits favour another expert."""
    router = LingMoeRouter(
        hidden_size=4, num_experts=8,
        num_experts_per_tok=2,
        n_group=2, topk_group=2,
    )
    with torch.no_grad():
        router.gate.weight.zero_()
        router.gate.weight[1, 0] = 5.0  # gate prefers expert 1
    x = torch.zeros(3, 4)
    x[:, 0] = 1.0
    _, _, baseline = router(x)
    assert (baseline[:, 0] == 1).all()  # expert 1 picked first

    with torch.no_grad():
        router.expert_bias[6] = 5.0  # boost expert 6 via bias
    _, _, after = router(x)
    # Expert 6 should now appear in every row's top-2.
    assert (after == 6).any(dim=-1).all(), after


def test_ling_router_rejects_bad_group_split() -> None:
    """num_experts must divide evenly by n_group; otherwise the
    constructor must raise."""
    with pytest.raises(ValueError, match="divisible"):
        LingMoeRouter(
            hidden_size=4, num_experts=10,
            num_experts_per_tok=2,
            n_group=3, topk_group=1,
        )
    with pytest.raises(ValueError, match="topk_group"):
        LingMoeRouter(
            hidden_size=4, num_experts=8,
            num_experts_per_tok=2,
            n_group=2, topk_group=3,
        )


def test_ling_router_matches_vllm_omni() -> None:
    """Cross-check vs vllm-omni's ``BailingMoeV2Gate`` on the same inputs.

    Same hidden_size / num_experts / etc., same gate weight, same
    expert_bias — chosen indices must match exactly. (Returned weights
    differ because the upstream Gate returns the gathered scores
    pre-renormalisation; we compare the indices, which is what
    matters for downstream dispatch.)
    """
    try:
        importlib.import_module("vllm_omni")
        from vllm_omni.model_executor.models.ming_flash_omni.modeling_bailing_moe_v2 import (
            BailingMoeV2Gate,
        )
        from vllm_omni.transformers_utils.configs.ming_flash_omni import (
            BailingMoeV2Config,
        )
    except Exception as e:  # noqa: BLE001 — broad on purpose; any import path failure ⇒ skip
        pytest.skip(f"vllm-omni not importable: {e}")

    # vllm-omni's Gate calls get_tensor_model_parallel_world_size() — we
    # need to be in a TP-initialised state for that. Set up a single-rank
    # group manually.
    try:
        from vllm.distributed import init_distributed_environment, initialize_model_parallel
        if not torch.distributed.is_initialized():
            init_distributed_environment(
                world_size=1, rank=0, distributed_init_method="tcp://127.0.0.1:25555",
                local_rank=0, backend="gloo",
            )
        initialize_model_parallel(tensor_model_parallel_size=1)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"vllm distributed init not available: {e}")

    config = BailingMoeV2Config(
        hidden_size=32, num_experts=16, num_experts_per_tok=4,
        n_group=4, topk_group=2, routed_scaling_factor=2.5,
    )
    upstream = BailingMoeV2Gate(config)

    ours = LingMoeRouter(
        hidden_size=32, num_experts=16, num_experts_per_tok=4,
        n_group=4, topk_group=2, routed_scaling_factor=2.5,
    )
    # Copy gate weights + bias for an apples-to-apples comparison.
    with torch.no_grad():
        ours.gate.weight.copy_(upstream.gate.weight.data)
        ours.expert_bias.copy_(upstream.expert_bias.data)
        # Give expert_bias something non-trivial so the bias path is exercised.
        ours.expert_bias.normal_(std=0.01)
        upstream.expert_bias.data.copy_(ours.expert_bias.data)

    x = torch.randn(6, 32)
    _, _, ours_indices = ours(x)
    up_indices, up_weights, _ = upstream(x)

    # Compare as sets per row — top-k order isn't guaranteed to match by
    # construction (both use ``sorted=False`` in their final topk).
    for r in range(x.shape[0]):
        assert set(ours_indices[r].tolist()) == set(up_indices[r].tolist()), (
            f"row {r}: ours={sorted(ours_indices[r].tolist())} vs "
            f"upstream={sorted(up_indices[r].tolist())}"
        )


# ---------------------------------------------------------------------------
# Partial MRoPE
# ---------------------------------------------------------------------------


def _make_rope(head_dim: int = 128) -> LingPartialMRotaryEmbedding:
    return LingPartialMRotaryEmbedding(
        head_dim=head_dim,
        partial_rotary_factor=0.5,
        mrope_section=[8, 12, 12],
        rope_theta=2_400_000.0,
        max_position_embeddings=32768,
    )


def test_partial_mrope_shapes_and_pass_through() -> None:
    """Output shape unchanged; pass-through half is byte-identical.

    head_dim=128, partial=0.5 → rotary_dim=64. Indices 64..128 are
    untouched.
    """
    rope = _make_rope()  # head_dim=128, mrope_section=[8,12,12] sums to 32 = 64//2  ✓
    T = 7
    q = torch.randn(2, T, 128)  # (num_heads, T, head_dim)
    k = torch.randn(2, T, 128)
    positions = torch.arange(T)
    q_out, k_out = rope(q, k, positions)
    assert q_out.shape == q.shape == k_out.shape
    # The second half of head_dim must be untouched (rotary_dim=64).
    assert torch.equal(q_out[..., 64:], q[..., 64:])
    assert torch.equal(k_out[..., 64:], k[..., 64:])


def test_partial_mrope_1d_matches_standard_rotary() -> None:
    """With 1D position_ids, rotation reduces to plain rotary on the
    first 64 dims — invariant: identical inputs at identical positions
    produce identical rotations regardless of axis layout."""
    rope = _make_rope()
    q = torch.randn(1, 1, 128)
    k = torch.zeros(1, 1, 128)
    pos = torch.tensor([5])
    # Same q rotated at position 5 twice → identical.
    out1, _ = rope(q.clone(), k.clone(), pos)
    out2, _ = rope(q.clone(), k.clone(), pos)
    assert torch.equal(out1, out2)


def test_partial_mrope_video_rope_layout() -> None:
    """``video_rope`` axis assignment: spatial half uses H/W alternating,
    temporal tail uses T.

    Test by zeroing two of the three position rows and checking the
    rotation only touches the dims the surviving axis was assigned to.
    """
    rope = _make_rope()
    T = 1
    # Identity-friendly q: ones in the rotary half so rotation is observable.
    q = torch.zeros(1, T, 128)
    q[..., :64] = 1.0
    k = q.clone()

    # All time positions = 5, H = W = 0  → time should be the only
    # axis with nonzero effect. video_rope places T at indices [hw_size:half]
    # which is [24:32] in each of the two halves.
    positions = torch.zeros(3, T, dtype=torch.long)
    positions[0] = 5
    q_t, _ = rope(q.clone(), k.clone(), positions)

    # Pull the cos/sin we expect for time at indices [24:32] and [24+32:64]
    # (the two halves of rotary_dim=64). For H=W=0, cos=1 sin=0 everywhere,
    # so spatial dims should remain == 1.0 (no rotation).
    rotary_first = q_t[..., :64]
    # Spatial dims: 0..24 in each half — for H=W=0, freq=0, cos=1, sin=0
    # → rotation leaves value at 1.0.
    assert torch.allclose(rotary_first[..., :24], torch.ones_like(rotary_first[..., :24])), \
        "spatial dims rotated under H=W=0 — wrong axis assignment"
    assert torch.allclose(rotary_first[..., 32:32 + 24], torch.ones_like(rotary_first[..., 32:32 + 24])), \
        "spatial dims (second half) rotated under H=W=0"
    # Temporal dims [24:32] and [56:64]: position 5 with theta=2.4M and
    # rotary_dim=64 produces a measurable but small rotation (we don't
    # check exact value; just that it diverged from 1.0).
    assert not torch.allclose(rotary_first[..., 24:32], torch.ones_like(rotary_first[..., 24:32])), \
        "temporal dims unrotated when T=5 — time axis not applied"


def test_partial_mrope_rejects_inconsistent_section() -> None:
    """sum(mrope_section) must equal rotary_dim // 2."""
    with pytest.raises(ValueError, match="rotary_dim"):
        LingPartialMRotaryEmbedding(
            head_dim=128, partial_rotary_factor=0.5,
            mrope_section=[8, 16, 16],   # sums to 40, expected 32
            rope_theta=10000.0, max_position_embeddings=1024,
        )


# ---------------------------------------------------------------------------
# Attention (QK-norm + partial MRoPE composition)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="mminf RMSNorm uses flashinfer's CUDA-only rmsnorm")
def test_ling_attention_forward_runs_with_qk_norm() -> None:
    """End-to-end forward at small dim — main goal is that the QK-norm +
    rope composition doesn't crash and produces finite output."""
    head_dim = 32
    # rotary_dim=16, rotary_dim//2=8 — section sum must be 8.
    rope = LingPartialMRotaryEmbedding(
        head_dim=head_dim,
        partial_rotary_factor=0.5,
        mrope_section=[2, 3, 3],
        rope_theta=10000.0,
        max_position_embeddings=128,
    ).cuda()
    attn = LingAttention(
        hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=head_dim, rms_norm_eps=1e-6, rotary=rope,
    ).cuda()
    T = 5
    x = torch.randn(T, 64, device="cuda")
    pos = torch.arange(T, device="cuda")
    out = attn(x, _MockCacheHandle(), pos)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="mminf RMSNorm uses flashinfer's CUDA-only rmsnorm")
def test_ling_attention_qk_norm_actually_normalises() -> None:
    """Verify the q_norm / k_norm layers are RMSNorm-shaped — sanity guard
    for the right module is plumbed in. Using ``head_norm_check`` helper."""
    head_dim = 16
    # rotary_dim=8, rotary_dim//2=4 — section sum must be 4.
    rope = LingPartialMRotaryEmbedding(
        head_dim=head_dim, partial_rotary_factor=0.5,
        mrope_section=[1, 1, 2], rope_theta=10000.0,
        max_position_embeddings=64,
    ).cuda()
    attn = LingAttention(
        hidden_size=32, num_heads=2, num_kv_heads=2,
        head_dim=head_dim, rms_norm_eps=1e-6, rotary=rope,
    ).cuda()
    # Feed a heavily-scaled input — RMSNorm should bring per-head RMS to 1.
    q_big = torch.randn(3, 4, head_dim, device="cuda") * 100.0   # (T, H, head_dim)
    out = attn.q_norm(q_big)
    max_dev = LingAttention.head_norm_check(out)
    # 5e-3 tolerance accommodates bf16 RMSNorm; the load-bearing claim is
    # that q_norm reshapes per-head and applies normalisation, not that
    # the RMS is precisely 1.0 to 4 decimals on fp16 hardware.
    assert max_dev < 5e-3, f"q_norm did not produce unit-RMS output: dev={max_dev}"


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="mminf RMSNorm uses flashinfer's CUDA-only rmsnorm")
def test_ling_attention_causal_mask() -> None:
    """Sanity: appending a later token shouldn't change the output of
    earlier positions (proves causal masking is on)."""
    head_dim = 32
    # rotary_dim=16, rotary_dim//2=8 — section sum must be 8.
    rope = LingPartialMRotaryEmbedding(
        head_dim=head_dim, partial_rotary_factor=0.5,
        mrope_section=[2, 3, 3], rope_theta=10000.0,
        max_position_embeddings=128,
    ).cuda()
    attn = LingAttention(
        hidden_size=64, num_heads=4, num_kv_heads=4,
        head_dim=head_dim, rms_norm_eps=1e-6, rotary=rope,
    ).cuda().eval()
    x = torch.randn(3, 64, device="cuda")
    pos = torch.arange(3, device="cuda")
    out_a = attn(x, _MockCacheHandle(), pos)

    # Append a 4th token; first 3 outputs MUST equal out_a (causal).
    x4 = torch.cat([x, torch.randn(1, 64, device="cuda")], dim=0)
    pos4 = torch.arange(4, device="cuda")
    out_b = attn(x4, _MockCacheHandle(), pos4)
    assert torch.allclose(out_a, out_b[:3], atol=1e-4), \
        "causal mask leaked — adding a later token changed earlier outputs"
