"""Mimi weight conversion and ring mask, and whole-sequence Mimi against NVIDIA's streaming Mimi.

The fixture-backed tests replay ``reference_mimi.pt`` from ``test/personaplex/reference_oracle.py --mode mimi``:
a shrunken random Mimi run one 1920-sample frame at a time by NVIDIA's code. Synthetic
weights only; the real ``tokenizer-e351c8d8-checkpoint125.safetensors`` is gated.
"""

import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from personaplex_fixture_utils import load_binary_fixture

from mstar.model.personaplex.components import mimi as mimi_mod
from mstar.model.personaplex.components.mimi import (
    CONVERT_LIST,
    MimiCodec,
    convert_mimi_state_dict,
    pad_to_frames,
    permute_for_rotate_half,
    ring_attention_mask,
)
from mstar.model.personaplex.components.temporal import apply_rope_adjacent

FIXTURE = Path(__file__).parent / "fixtures/personaplex/reference_mimi.pt"


def _rope_rotate_half(x, positions, theta=10000.0):
    """HF/NeoX rotation of (i, i + D/2), written independently of M* code."""
    D = x.shape[-1]
    inv = 1.0 / theta ** (torch.arange(0, D, 2, dtype=torch.float64) / D)
    ang = positions.double()[:, None, None] * inv
    x1, x2 = x.double()[..., : D // 2], x.double()[..., D // 2 :]
    return torch.cat((x1 * ang.cos() - x2 * ang.sin(), x2 * ang.cos() + x1 * ang.sin()), -1)


# --------------------------------------------------------------------------- conversion and ring mask (fixture-free)


def test_qk_permute_turns_adjacent_pair_rope_into_rotate_half():
    torch.manual_seed(0)
    heads, head_dim, dim, T = 2, 8, 16, 5
    w = torch.randn(heads * head_dim, dim)
    x = torch.randn(T, dim)
    positions = torch.arange(3, 3 + T)
    moshi = (x @ w.T).view(T, heads, head_dim)
    moshi, _ = apply_rope_adjacent(moshi, moshi, positions, 10000.0)
    hf = _rope_rotate_half((x @ permute_for_rotate_half(w, heads).T).view(T, heads, head_dim), positions)
    # The permutation sends pair (2i, 2i+1) to (i, i + D/2) within each head.
    order = torch.cat((torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)))
    torch.testing.assert_close(hf, moshi.double()[..., order], rtol=1e-6, atol=1e-6)
    # Attention scores are what matters, and they agree.
    torch.testing.assert_close(hf @ hf.transpose(-1, -2), (moshi @ moshi.transpose(-1, -2)).double(),
                               rtol=1e-5, atol=1e-5)


def _synthetic_moshi_state():
    d, h = 8, 2
    state = {
        "encoder.model.0.conv.conv.weight": torch.randn(4, 1, 7),
        "decoder.model.2.convtr.convtr.weight": torch.randn(4, 4, 4),
        "encoder_transformer.transformer.layers.0.self_attn.in_proj_weight": torch.randn(3 * d, d),
        "encoder_transformer.transformer.layers.0.self_attn.out_proj.weight": torch.randn(d, d),
        "encoder_transformer.transformer.layers.0.linear1.weight": torch.randn(16, d),
        "encoder_transformer.transformer.layers.0.norm1.weight": torch.randn(d),
        "encoder_transformer.transformer.layers.0.layer_scale_2.scale": torch.randn(d),
        "quantizer.rvq_first.vq.layers.0._codebook.embedding_sum": torch.randn(32, 4),
        "quantizer.rvq_rest.vq.layers.3._codebook._initialized": torch.ones(1),
        "quantizer.rvq_rest.input_proj.weight": torch.randn(4, d, 1),
    }
    return state, d, h


def test_conversion_renames_splits_and_is_strict():
    state, d, h = _synthetic_moshi_state()
    prefix = "encoder_transformer.layers.0."
    expected_names = {
        "encoder.layers.0.conv.weight", "decoder.layers.2.conv.weight",
        prefix + "self_attn.q_proj.weight", prefix + "self_attn.k_proj.weight", prefix + "self_attn.v_proj.weight",
        prefix + "self_attn.o_proj.weight", prefix + "mlp.fc1.weight", prefix + "input_layernorm.weight",
        prefix + "mlp_layer_scale.scale",
        "quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum",
        "quantizer.acoustic_residual_vector_quantizer.layers.3.codebook.initialized",
        "quantizer.acoustic_residual_vector_quantizer.input_proj.weight",
    }
    # Renames keep shapes; the fused in_proj becomes three [d, d] projections.
    split = {prefix + f"self_attn.{n}_proj.weight": torch.Size([d, d]) for n in "qkv"}
    expected = {name: split.get(name) for name in expected_names}
    for key, value in state.items():
        new = key
        for old, rep in CONVERT_LIST:
            new = new.replace(old, rep) if old in new else new
        if new in expected and expected[new] is None:
            expected[new] = value.shape
    assert None not in expected.values()
    out = convert_mimi_state_dict(state, expected, num_heads=h, num_kv_heads=h)
    assert set(out) == expected_names
    qkv = state[next(k for k in state if k.endswith("in_proj_weight"))]
    assert torch.equal(out[prefix + "self_attn.q_proj.weight"], permute_for_rotate_half(qkv[:d], h))
    assert torch.equal(out[prefix + "self_attn.k_proj.weight"], permute_for_rotate_half(qkv[d : 2 * d], h))
    assert torch.equal(out[prefix + "self_attn.v_proj.weight"], qkv[2 * d :])
    with pytest.raises(ValueError, match="extra keys"):
        convert_mimi_state_dict({**state, "encoder.model.9.conv.conv.bias": torch.zeros(1)}, expected,
                                num_heads=h, num_kv_heads=h)
    with pytest.raises(ValueError, match="missing keys"):
        convert_mimi_state_dict({k: v for k, v in state.items() if "linear1" not in k}, expected,
                                num_heads=h, num_kv_heads=h)
    with pytest.raises(ValueError, match="shape mismatch"):
        convert_mimi_state_dict(state, {**expected, prefix + "mlp.fc1.weight": torch.Size([1])},
                                num_heads=h, num_kv_heads=h)


def _simulated_ring_visibility(steps, capacity, chunk):
    """Transcribe RingKVCache.complete + the causal mask ([PP] modules/transformer.py:263-299, 425-434)."""
    visible = torch.zeros(steps, steps, dtype=torch.bool)
    slot_pos = [-1] * capacity  # true position stored in each slot
    end = 0
    for start in range(0, steps, chunk):
        for t in range(start, start + chunk):
            slot_pos[t % capacity] = t
        end += chunk
        end_index = end % capacity
        for slot in range(capacity):
            if slot >= end:
                continue  # invalid
            delta = slot - end_index
            label = end + delta if delta <= 0 else end + delta - capacity
            for q in range(start, start + chunk):
                if 0 <= label <= q:
                    assert label == slot_pos[slot] or label == end  # mislabelled slot is never visible
                    visible[q, slot_pos[slot]] = True
    return visible


@pytest.mark.parametrize(("steps", "capacity", "chunk"), [(12, 5, 1), (24, 8, 2), (40, 16, 1), (64, 10, 2)])
def test_ring_mask_equals_simulated_ring_cache(steps, capacity, chunk):
    mask = ring_attention_mask(steps, dtype=torch.float32, capacity=capacity, chunk=chunk)[0, 0]
    assert torch.equal(mask == 0, _simulated_ring_visibility(steps, capacity, chunk))


def test_ring_mask_for_mimi_first_narrows_at_query_248():
    visible = ring_attention_mask(320, dtype=torch.float32)[0, 0] == 0
    causal = torch.ones(320, 320).tril().bool()
    assert torch.equal(visible[:248], causal[:248])
    assert not torch.equal(visible[248], causal[248])
    assert int(visible[319].sum()) == 249 and int(visible[318].sum()) == 248


def test_padding_and_shape_validation():
    assert pad_to_frames(torch.zeros(1, 1, 1920)).shape[-1] == 1920
    padded = pad_to_frames(torch.ones(1, 1, 1921))
    assert padded.shape[-1] == 3840 and padded[..., 1921:].abs().sum() == 0
    codec = MimiCodec(torch.nn.Identity())
    with pytest.raises(ValueError, match="mono PCM"):
        codec.encode_latent(torch.zeros(1, 1, 1000))
    with pytest.raises(ValueError, match="8, N"):
        codec.decode(torch.zeros(1, 7, 3, dtype=torch.long))


# ------------------------------------------------------------------- whole-sequence vs NVIDIA streaming (fixture)


@contextmanager
def _without_triton_stub():
    """conftest's spec-less ``triton`` stub makes transformers' ``find_spec`` raise; hide it while importing."""
    stubs = {name: sys.modules.pop(name) for name in ("triton", "triton.language") if name in sys.modules}
    try:
        yield
    finally:
        sys.modules.update(stubs)


@pytest.fixture(scope="module")
def reference():
    with _without_triton_stub():
        pytest.importorskip("transformers", reason="the Mimi codec builds transformers.MimiModel")
    configured = os.environ.get("PERSONAPLEX_REFERENCE_MIMI")
    path = Path(configured) if configured else FIXTURE
    if not path.is_file() and not configured:
        pytest.skip("needs reference_mimi.pt from the NVIDIA oracle; see fixtures/personaplex/README.rst "
                    "or set PERSONAPLEX_REFERENCE_MIMI")
    data = load_binary_fixture(path)
    assert data["num_codebooks"] == 8
    return data


def _codec(data):
    kwargs = data["mimi_kwargs"]["tiny"]
    with _without_triton_stub():
        model = mimi_mod.build_hf_mimi(mimi_mod.mimi_config_kwargs(kwargs["seanet"], kwargs["transformer"],
                                                                   kwargs["quantizer"]))
    expected = {name: t.shape for name, t in model.state_dict().items()}
    heads = kwargs["transformer"]["num_heads"]
    model.load_state_dict(convert_mimi_state_dict(data["state_dict"], expected, num_heads=heads,
                                                  num_kv_heads=heads), strict=True)
    return MimiCodec(model)


def test_real_mimi_kwargs_map_to_hf_defaults(reference):
    with _without_triton_stub():
        from transformers import MimiConfig

    def rope_theta(config):
        # transformers 5 keeps the RoPE base in rope_parameters rather than rope_theta
        return config.get("rope_theta") or (config.get("rope_parameters") or {}).get("rope_theta")

    real = reference["mimi_kwargs"]["real"]
    mapped = mimi_mod.mimi_config_kwargs(real["seanet"], real["transformer"], real["quantizer"])
    defaults = MimiConfig().to_dict()
    defaults["rope_theta"] = rope_theta(defaults)
    differing = {k: (v, defaults.get(k)) for k, v in mapped.items()
                 if k not in ("frame_rate",) and defaults.get(k) != v}
    assert differing == {}
    config = MimiConfig(**mapped)
    assert config.frame_rate == 12.5 and rope_theta(config.to_dict()) == mapped["rope_theta"]


def test_conversion_covers_full_size_nvidia_mimi_keys(reference):
    real = reference["mimi_kwargs"]["real"]
    shapes = reference["real_state_dict_shapes"]
    assert sum(math.prod(s) for k, s in shapes.items() if "_codebook" not in k) > 70_000_000
    with _without_triton_stub():
        model = mimi_mod.build_hf_mimi(mimi_mod.mimi_config_kwargs(real["seanet"], real["transformer"],
                                                                   real["quantizer"]))
    expected = {name: t.shape for name, t in model.state_dict().items()}
    synthetic = {name: torch.zeros(shape) for name, shape in shapes.items()}
    heads = real["transformer"]["num_heads"]
    converted = convert_mimi_state_dict(synthetic, expected, num_heads=heads, num_kv_heads=heads)
    model.load_state_dict(converted, strict=True)


@torch.no_grad()
@pytest.mark.parametrize("case_index", [0, 1])
def test_mimi_whole_sequence_matches_nvidia_streaming(reference, case_index):
    case = reference["cases"][case_index]
    codec = _codec(reference)
    frames = case["frames"]
    latent = codec.encode_latent(case["audio"])
    torch.testing.assert_close(latent, case["latent"], rtol=1e-6, atol=1e-6)
    codes = codec.encode(case["audio"])
    assert codes.shape == (1, 8, frames)
    decided = case["margins"] > 1e-6  # fp32 argmin ties are not semantic
    assert decided.float().mean() > 0.99
    assert torch.equal(codes[0][decided], case["codes"][0][decided])
    for codes_key, pcm_key in (("codes", "pcm_from_codes"), ("random_codes", "pcm_from_random_codes")):
        pcm = codec.decode(case[codes_key])
        assert pcm.shape == (1, 1, frames * 1920)
        torch.testing.assert_close(pcm, case[pcm_key], rtol=1e-6, atol=1e-6)


@torch.no_grad()
def test_mimi_ring_mask_is_required_beyond_124_frames(reference, monkeypatch):
    case = reference["cases"][1]
    assert case["frames"] > 125
    codec = _codec(reference)
    ring = ring_attention_mask
    # An unbounded capacity turns the ring into plain causal attention (HF's public API).
    monkeypatch.setattr(mimi_mod, "ring_attention_mask", lambda steps, **kw: ring(steps, **{**kw, "capacity": 10**9}))
    per_frame = (codec.encode_latent(case["audio"]) - case["latent"]).abs().amax(dim=(0, 1))
    assert float(per_frame[:124].max()) < 1e-6
    assert float(per_frame[124:].max()) > 1e-5


@torch.no_grad()
def test_mimi_matches_offline_interleaved_streaming(reference):
    run = reference["interleaved"]
    codec = _codec(reference)
    assert torch.equal(codec.encode(run["audio"]), run["codes"])
    torch.testing.assert_close(codec.decode(run["agent_codes"]), run["pcm"], rtol=1e-6, atol=1e-6)
