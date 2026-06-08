"""Tests for the Ling-2.0 weight loader (TP-aware, step 3e).

Three pure-Python tests verify the new name remapper + QKV split +
per-expert StackedParamRules in isolation. Two CUDA/snapshot-gated
tests load the real released checkpoint and verify forward + per-param
shape — the strongest signal that the model code matches the upstream
architecture byte-for-byte.

Snapshot lookup mirrors the other ming tests: ``MING_FLASH_OMNI_DIR``
env var, then the default HF Hub cache layout.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from mminf.model.ming_omni_flash.components.model import LingMoeModel
from mminf.model.ming_omni_flash.loader import (
    _build_thinker_stacked_params,
    _remap_thinker_keys,
    _split_packed_qkv,
    load_thinker_weights,
)


def _find_local_snapshot() -> str | None:
    """Locate a Ming-flash-omni-2.0 snapshot on disk, or None."""
    override = os.environ.get("MING_FLASH_OMNI_DIR")
    if override and (Path(override) / "config.json").exists():
        return override

    hub_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = hub_root / "models--inclusionAI--Ming-flash-omni-2.0" / "snapshots"
    if not repo_dir.exists():
        return None
    for snap in sorted(repo_dir.iterdir()):
        if (snap / "config.json").exists():
            return str(snap)
    return None


# Real-config values for the released ckpt (so weight shapes line up).
def _real_thinker_dims(num_hidden_layers: int = 1) -> dict:
    return dict(
        vocab_size=157184,
        hidden_size=4096,
        intermediate_size=9216,
        moe_intermediate_size=1024,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=32,
        num_kv_heads=4,
        head_dim=128,
        rms_norm_eps=1e-6,
        rope_theta=2_400_000.0,
        max_position_embeddings=32768,
        partial_rotary_factor=0.5,
        mrope_section=[8, 12, 12],
        num_experts=256,
        num_experts_per_tok=8,
        num_shared_experts=1,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        first_k_dense_replace=1,
    )


# ---------------------------------------------------------------------------
# Pure-Python unit tests for the new loader helpers
# ---------------------------------------------------------------------------


def test_remap_thinker_keys_resolves_layer0_keys() -> None:
    """Every layer-0 LLM ckpt key remaps to a parameter that exists in
    a 1-layer dense-only LingMoeModel (after the synthetic q/k/v
    expansion from the QKV split; we test that separately)."""
    model = LingMoeModel(**_real_thinker_dims(num_hidden_layers=1))
    target_keys = set(model.state_dict().keys())

    # Direct-load keys (not QKV — that's split separately).
    direct_keys = {
        "model.lm_head.weight": "lm_head.weight",
        "model.model.word_embeddings.weight": "embed_tokens.weight",
        "model.model.norm.weight": "norm.weight",
        "model.model.layers.0.input_layernorm.weight":
            "layers.0.input_layernorm.weight",
        "model.model.layers.0.post_attention_layernorm.weight":
            "layers.0.post_attention_layernorm.weight",
        "model.model.layers.0.attention.dense.weight":
            "layers.0.self_attn.dense.weight",
        "model.model.layers.0.attention.q_norm.weight":
            "layers.0.self_attn.q_norm.weight",
        "model.model.layers.0.attention.k_norm.weight":
            "layers.0.self_attn.k_norm.weight",
    }
    for raw, expected in direct_keys.items():
        renamed = _remap_thinker_keys(raw)
        assert renamed == expected, f"{raw} → {renamed!r}, expected {expected!r}"
        assert renamed in target_keys, f"{renamed!r} not in model.state_dict()"


def test_remap_thinker_keys_handles_moe_layer() -> None:
    """MoE-layer renames + per-expert rewrite."""
    # Routers + shared expert.
    assert (
        _remap_thinker_keys("model.model.layers.5.mlp.gate.weight")
        == "layers.5.mlp.gate.gate.weight"
    )
    assert (
        _remap_thinker_keys("model.model.layers.5.mlp.image_gate.weight")
        == "layers.5.mlp.image_gate.gate.weight"
    )
    assert (
        _remap_thinker_keys("model.model.layers.5.mlp.audio_gate.expert_bias")
        == "layers.5.mlp.audio_gate.expert_bias"
    )
    assert (
        _remap_thinker_keys("model.model.layers.5.mlp.shared_experts.gate_proj.weight")
        == "layers.5.mlp.shared_expert.gate_proj.weight"
    )
    # Per-expert: rewritten with __expertN__ marker so StackedParamRule
    # suffix-match works downstream.
    assert (
        _remap_thinker_keys("model.model.layers.5.mlp.experts.42.gate_proj.weight")
        == "layers.5.mlp.experts.gate_proj.__expert42__.weight"
    )
    assert (
        _remap_thinker_keys("model.model.layers.5.mlp.experts.255.down_proj.weight")
        == "layers.5.mlp.experts.down_proj.__expert255__.weight"
    )


def test_remap_thinker_keys_drops_non_thinker_prefixes() -> None:
    """audio.* / vision.* keys aren't part of the thinker port; return None."""
    assert _remap_thinker_keys("audio.encoder.layers.0.weight") is None
    assert _remap_thinker_keys("vision.patch_embed.weight") is None


def test_build_stacked_params_covers_every_expert() -> None:
    """3 rules per expert × num_experts, plus dense MLP rules."""
    rules = _build_thinker_stacked_params(num_experts=8)
    # 3 × 8 expert rules + 2 dense-MLP rules = 26
    assert len(rules) == 3 * 8 + 2
    expert_shard_ids = {r.shard_id for r in rules if isinstance(r.shard_id, str) and ":" in r.shard_id}
    expected = set()
    for i in range(8):
        for kind in ("gate", "up", "down"):
            expected.add(f"{kind}:{i}")
    assert expert_shard_ids == expected


def test_split_packed_qkv_emits_three_synthetic_keys() -> None:
    """A single ``attention.query_key_value.weight`` becomes three
    synthetic keys with the expected row slicing."""
    # GQA shape: num_heads=4, num_kv_heads=2, head_dim=8 →
    # q_size=32, kv_size=16, total=64.
    packed = torch.arange(64 * 16, dtype=torch.float32).view(64, 16)
    stream = [(
        "layers.0.attention.query_key_value.weight", packed,
    ), (
        "layers.0.input_layernorm.weight", torch.ones(16),
    )]
    out = list(_split_packed_qkv(
        iter(stream),
        num_attention_heads=4, num_kv_heads=2, head_dim=8,
    ))
    # 3 synthetic + 1 passthrough = 4
    assert len(out) == 4
    names = [k for k, _ in out]
    assert names[:3] == [
        "layers.0.attention.q_proj.weight",
        "layers.0.attention.k_proj.weight",
        "layers.0.attention.v_proj.weight",
    ]
    # Row slicing: q=[0:32], k=[32:48], v=[48:64].
    assert torch.equal(out[0][1], packed[0:32, :])
    assert torch.equal(out[1][1], packed[32:48, :])
    assert torch.equal(out[2][1], packed[48:64, :])
    # Non-QKV key passes through unchanged.
    assert names[3] == "layers.0.input_layernorm.weight"


def test_split_packed_qkv_rejects_bad_shape() -> None:
    """Wrong first-dim raises a clear error."""
    bad = torch.zeros(50, 16)  # expected 64 for the dims below
    stream = [("layers.0.attention.query_key_value.weight", bad)]
    with pytest.raises(ValueError, match="expected first dim 64"):
        list(_split_packed_qkv(
            iter(stream),
            num_attention_heads=4, num_kv_heads=2, head_dim=8,
        ))


# ---------------------------------------------------------------------------
# Real-checkpoint smoke (CUDA + snapshot required)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def snapshot_dir() -> str:
    snap = _find_local_snapshot()
    if snap is None:
        pytest.skip(
            "Ming-flash-omni-2.0 snapshot not found. Set MING_FLASH_OMNI_DIR "
            "or download via `huggingface-cli download`."
        )
    return snap


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="real-ckpt smoke needs CUDA")
def test_load_layer0_real_weights_runs_forward(snapshot_dir: str) -> None:
    """Load embed + dense layer 0 + norm + lm_head from the real ckpt
    into a 1-layer LingMoeModel (TP=1, comm_group=None default); run a
    forward; verify shape + finite."""
    dims = _real_thinker_dims(num_hidden_layers=1)
    # Construct on meta + materialise on CUDA to avoid double allocation.
    with torch.device("meta"):
        model = LingMoeModel(**dims)
    model.to_empty(device="cuda")
    model.to(torch.bfloat16)

    load_thinker_weights(model, snapshot_dir, device="cuda", strict=True)
    model.eval()

    # Minimal mock cache handle — passthrough SDPA, same as step 3d tests.
    import torch.nn.functional as F

    class _Cache:
        def set_layer_idx(self, i):
            pass

        def run_attention(self, q, k, v):
            num_heads = q.shape[1]
            num_kv = k.shape[1]
            if num_heads // num_kv > 1:
                k = k.repeat_interleave(num_heads // num_kv, dim=1)
                v = v.repeat_interleave(num_heads // num_kv, dim=1)
            q4 = q.transpose(0, 1).unsqueeze(0)
            k4 = k.transpose(0, 1).unsqueeze(0)
            v4 = v.transpose(0, 1).unsqueeze(0)
            out = F.scaled_dot_product_attention(
                q4, k4, v4, is_causal=True, scale=q.shape[-1] ** -0.5,
            )
            return out.squeeze(0).transpose(0, 1).contiguous()

    input_ids = torch.tensor([100, 200, 300, 400], device="cuda")
    with torch.no_grad():
        out = model(_Cache(), input_ids=input_ids)

    assert out.shape == (4, dims["vocab_size"])
    assert torch.isfinite(out).all(), \
        f"Non-finite logits after 1-layer forward; max={out.abs().max().item()}"
    assert out.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="real-ckpt smoke needs CUDA")
def test_layer0_attention_weights_match_expected_shapes(snapshot_dir: str) -> None:
    """After load, every layer-0 attention param has the expected shape.

    With TP=1 these match the full per-rank-equals-total dims; the same
    test under TP>1 would expect num_heads / num_kv_heads divided by
    tp_size.
    """
    dims = _real_thinker_dims(num_hidden_layers=1)
    with torch.device("meta"):
        model = LingMoeModel(**dims)
    model.to_empty(device="cuda")
    model.to(torch.bfloat16)
    load_thinker_weights(model, snapshot_dir, device="cuda", strict=True)

    head_dim = dims["head_dim"]
    hidden = dims["hidden_size"]
    n_heads = dims["num_attention_heads"]
    n_kv = dims["num_kv_heads"]

    expected = {
        # QKVParallelLinear packs (q + 2*kv) * head_dim along dim 0.
        "layers.0.self_attn.qkv_proj.weight":
            ((n_heads + 2 * n_kv) * head_dim, hidden),
        # RowParallelLinear holds (output, input_per_partition); TP=1 →
        # input_per_partition = full.
        "layers.0.self_attn.dense.weight": (hidden, n_heads * head_dim),
        "layers.0.self_attn.q_norm.weight": (head_dim,),
        "layers.0.self_attn.k_norm.weight": (head_dim,),
        "layers.0.input_layernorm.weight": (hidden,),
        "layers.0.post_attention_layernorm.weight": (hidden,),
        "embed_tokens.weight": (dims["vocab_size"], hidden),
        "lm_head.weight": (dims["vocab_size"], hidden),
    }
    state = dict(model.state_dict())
    for name, shape in expected.items():
        assert name in state, f"{name} missing from loaded state_dict"
        assert tuple(state[name].shape) == shape, (
            f"{name}: expected {shape}, got {tuple(state[name].shape)}"
        )
        assert torch.isfinite(state[name]).all(), \
            f"{name} contains non-finite values after load"
