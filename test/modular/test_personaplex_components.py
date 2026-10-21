"""T5/T6 fixture-free structure checks; T5/T13-LM compare against NVIDIA fixtures.

R-tier tests replay fixtures recorded by ``test/scratch/personaplex_reference_oracle.py``
from NVIDIA's own ``LMModel``/``LMGen`` with random weights (fixtures/personaplex/README.rst).
They establish algorithmic equivalence on synthetic weights, not checkpoint or GPU parity.
"""

import hashlib
import json
import math
import os
import platform
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from personaplex_fixture_utils import load_binary_fixture, verify_reference_fixture

from mstar.model.personaplex import config as cfg
from mstar.model.personaplex.components.delayed_cache import DelayedTokenCache
from mstar.model.personaplex.components.depformer import DepformerAttention, depth_attention_start
from mstar.model.personaplex.components.temporal import (
    ActivationGating,
    DenseTemporalCache,
    RMSNormF32,
    ScaledEmbedding,
    apply_rope_adjacent,
)
from mstar.model.personaplex.weights import (
    PersonaPlexLM,
    load_lm_weights,
    reference_lm_key,
    reference_manifest,
    remap_lm_key,
)

FIXTURES = Path(__file__).parent / "fixtures/personaplex"
TINY = cfg.LMArchConfig(
    dim=32, num_heads=4, num_layers=2, ffn_hidden=40, depformer_dim=16, depformer_heads=2,
    depformer_layers=2, depformer_ffn_hidden=24,
)


def _fixture(env: str, name: str) -> Path:
    configured = os.environ.get(env)
    path = Path(configured) if configured else FIXTURES / name
    if not path.is_file() and not configured:
        pytest.skip(f"needs {name} from the NVIDIA oracle; see fixtures/personaplex/README.rst or set {env}")
    return path


def _tiny_lm(seed: int = 0) -> PersonaPlexLM:
    torch.manual_seed(seed)
    lm = PersonaPlexLM(TINY).eval()
    with torch.no_grad():
        for name, param in lm.named_parameters():
            if name.endswith(".alpha"):
                param.copy_(1 + 0.25 * torch.randn_like(param))
    return lm


def _greedy(logits, _step=None):
    return logits.float().argmax(dim=-1)


def test_reference_fixture_loader_rejects_changed_tensor_and_source_pin():
    data = {"source": "NVIDIA/personaplex", "source_sha": cfg.REFERENCE_SHA,
            "payload": torch.tensor([1], dtype=torch.int32)}
    digest = {"payload": ["torch.int32", [1], hashlib.sha256(bytes([1, 0, 0, 0])).hexdigest()],
              "source": data["source"], "source_sha": data["source_sha"]}
    data["content_sha256"] = hashlib.sha256(
        json.dumps(digest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert verify_reference_fixture(data) is data
    with pytest.raises(ValueError, match="content SHA256"):
        verify_reference_fixture({**data, "payload": torch.tensor([2], dtype=torch.int32)})
    with pytest.raises(ValueError, match="source SHA"):
        verify_reference_fixture({**data, "source_sha": "0" * 40})


# --------------------------------------------------------------------------- T5 (fixture-free part)


def test_t5_full_size_native_lm_matches_published_parameter_count():
    with torch.device("meta"):
        lm = PersonaPlexLM(cfg.LMArchConfig())
    manifest = reference_manifest(lm)
    assert len(manifest) == 475
    # HF's published BF16 count for nvidia/personaplex-7b-v1 (worklog §2 R8).
    assert sum(math.prod(shape) for shape in manifest.values()) == 8_371_408_896
    assert manifest["depformer.layers.0.self_attn.in_proj_weight"] == [16 * 3 * 1024, 1024]
    assert manifest["depformer.layers.5.gating.15.linear_in.weight"] == [2 * 2816, 1024]
    assert manifest["transformer.layers.31.gating.linear_in.weight"] == [2 * 11264, 4096]
    assert manifest["out_norm.alpha"] == [1, 1, 4096]
    assert manifest["text_linear.weight"] == [32000, 4096]
    assert len([k for k in manifest if k.startswith("depformer_emb.")]) == 15


def test_t5_key_map_round_trips_and_routes_depth_keys():
    lm = PersonaPlexLM(TINY)
    for name, _ in lm.named_parameters():
        assert remap_lm_key(reference_lm_key(name)) == name
    assert remap_lm_key("linears.3.weight") == "depth.linears.3.weight"
    assert remap_lm_key("depformer_text_emb.weight") == "depth.depformer_text_emb.weight"
    assert remap_lm_key("transformer.layers.0.norm1.alpha") == "temporal.transformer.layers.0.norm1.alpha"
    assert remap_lm_key("emb.15.weight") == "temporal.emb.15.weight"


def test_t5_strict_loader_rejects_missing_extra_duplicate_and_misshapen_keys():
    pytest.importorskip("safetensors", reason="mstar.model.loader imports safetensors")
    source = _tiny_lm(1)
    target = PersonaPlexLM(TINY)
    state = {reference_lm_key(k): v for k, v in source.state_dict().items()}
    load_lm_weights(target, state.items())
    for name, value in target.state_dict().items():
        assert torch.equal(value, source.state_dict()[name]), name
    with pytest.raises(ValueError, match="1 missing"):
        load_lm_weights(target, [(k, v) for k, v in state.items() if k != "linears.15.weight"])
    with pytest.raises(ValueError, match="1 unexpected"):
        load_lm_weights(target, [*state.items(), ("linears.16.weight", state["linears.0.weight"])])
    with pytest.raises(ValueError, match="1 duplicate"):
        load_lm_weights(target, [*state.items(), ("emb.0.weight", state["emb.0.weight"])])
    with pytest.raises(ValueError, match="checkpoint shape"):
        load_lm_weights(target, [*[(k, v) for k, v in state.items() if k != "out_norm.alpha"],
                                 ("out_norm.alpha", torch.ones(TINY.dim))])


# --------------------------------------------------------------------------- T6 (fixture-free)


def test_t6_rope_rotates_adjacent_pairs_like_complex_multiplication():
    torch.manual_seed(0)
    q, k = torch.randn(2, 5, 3, 8), torch.randn(2, 5, 3, 8)
    positions = torch.arange(7, 12)
    qo, ko = apply_rope_adjacent(q, k, positions, 10000)
    # Independent formula: pair (2i, 2i+1) is a complex number rotated by pos * theta^(-2i/D).
    freqs = 10000.0 ** (-torch.arange(4, dtype=torch.float64) * 2 / 8)
    phase = torch.polar(torch.ones(5, 4, dtype=torch.float64), positions[:, None].double() * freqs)[:, None]
    for x, out in ((q, qo), (k, ko)):
        expected = torch.view_as_real(torch.view_as_complex(x.double().unflatten(-1, (4, 2))) * phase).flatten(-2)
        torch.testing.assert_close(out.double(), expected, rtol=1e-5, atol=1e-5)
    # Half-split (NeoX) RoPE rotates (i, i + D/2) instead; the layout is load-bearing (worklog D10).
    angle = (positions[:, None].double() * freqs)[:, None]
    x1, x2 = q.double()[..., :4], q.double()[..., 4:]
    neox = torch.cat((x1 * angle.cos() - x2 * angle.sin(), x2 * angle.cos() + x1 * angle.sin()), -1)
    assert not torch.allclose(qo.double(), neox, atol=1e-3)


def test_t6_rope_keeps_bfloat16_and_computes_in_fp32():
    q = torch.randn(1, 3, 2, 8, dtype=torch.bfloat16)
    qo, _ = apply_rope_adjacent(q, q, torch.arange(3), 10000)
    ref, _ = apply_rope_adjacent(q.float(), q.float(), torch.arange(3), 10000)
    assert qo.dtype == torch.bfloat16
    assert torch.equal(qo, ref.to(torch.bfloat16))


def test_t6_rms_norm_is_fp32_with_eps_1e8_and_broadcast_alpha():
    norm = RMSNormF32(6, cfg.NORM_EPS)
    assert norm.alpha.shape == (1, 1, 6) and norm.eps == 1e-8
    with torch.no_grad():
        norm.alpha.copy_(torch.arange(1, 7, dtype=torch.float32).view(1, 1, 6))
    x = torch.randn(4, 6, dtype=torch.bfloat16)
    y = norm(x)
    xf = x.double()
    expected = xf / torch.sqrt((xf**2).mean(-1, keepdim=True) + 1e-8) * torch.arange(1, 7).double()
    assert y.shape == x.shape and y.dtype == torch.bfloat16
    torch.testing.assert_close(y.double(), expected.to(torch.bfloat16).double(), rtol=1e-2, atol=1e-2)


def test_t6_gating_activates_first_half_and_embedding_minus_one_is_zero():
    gating = ActivationGating(4, 3)
    x = torch.randn(2, 4)
    h = x @ gating.linear_in.weight.T
    expected = (F.silu(h[:, :3]) * h[:, 3:]) @ gating.linear_out.weight.T
    torch.testing.assert_close(gating(x), expected)
    emb = ScaledEmbedding(5, 3)
    out = emb(torch.tensor([[-1, 0, 4]]))
    assert torch.equal(out[0, 0], torch.zeros(3))
    assert torch.equal(out[0, 1:], emb.weight[[0, 4]])


def test_t6_embed_codes_sums_sixteen_audio_streams_then_text():
    lm = _tiny_lm()
    codes = torch.randint(0, 2048, (2, 17, 3))
    codes[0, 5, 1] = -1
    codes[:, 0] = torch.randint(0, 32001, (2, 3))
    expected = sum(lm.temporal.emb[i](codes[:, i + 1]) for i in range(16)) + lm.temporal.text_emb(codes[:, 0])
    torch.testing.assert_close(lm.temporal.embed_codes(codes), expected)
    with pytest.raises(ValueError, match="17"):
        lm.temporal.embed_codes(codes[:, :16])


def test_t6_temporal_streaming_equals_one_shot_causal_prefill():
    lm = _tiny_lm()
    x = torch.randn(2, 6, TINY.dim)
    full = lm.temporal.forward_embeddings(x, DenseTemporalCache(TINY.num_layers, TINY.rope_theta))
    cache = DenseTemporalCache(TINY.num_layers, TINY.rope_theta)
    steps = []
    for t in range(6):
        steps.append(lm.temporal.forward_embeddings(x[:, t : t + 1], cache))
        cache.advance()
    for index in range(2):
        torch.testing.assert_close(torch.cat([s[index] for s in steps], 1), full[index], rtol=1e-5, atol=1e-5)
    # Causality: changing the last input leaves earlier outputs unchanged.
    y = x.clone()
    y[:, -1] += 1
    other = lm.temporal.forward_embeddings(y, DenseTemporalCache(TINY.num_layers, TINY.rope_theta))
    torch.testing.assert_close(other[0][:, :-1], full[0][:, :-1])
    assert not torch.allclose(other[0][:, -1], full[0][:, -1])


def test_t6_depth_attention_masks_step_zero_only_at_the_last_step():
    assert [depth_attention_start(step) for step in range(16)] == [0] * 15 + [1]


@torch.no_grad()
@pytest.mark.parametrize("step", [0, 7, 14, 15])
def test_t6_depth_attention_ignores_unwritten_and_expired_slots(step):
    """The fixed-capacity reference mask must hide dirty future slots and expired step 0."""
    torch.manual_seed(24)
    attention = DepformerAttention(dim=16, num_heads=2, steps=16).eval()
    x = torch.randn(2, 16)
    scratch = torch.randn(2, 2, 2, 16, 8)
    dirty = scratch.clone()
    dirty[:, :, :, step + 1:] = 1e4
    if step == 15:
        dirty[:, :, :, 0] = -1e4
    expected = attention(x, step, scratch)
    actual = attention(x, step, dirty)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _frame_inputs(batch=2, seed=3):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(batch, TINY.dim, generator=g)
    text = torch.randint(0, 32000, (batch,), generator=g)
    target = torch.randint(0, 2048, (batch, 16), generator=g)
    provided = torch.zeros(batch, 16, dtype=torch.bool)
    return text, hidden, target, provided


def test_t6_depformer_state_resets_every_frame_even_with_dirty_scratch():
    lm = _tiny_lm()
    args = _frame_inputs()
    tokens, logits = lm.depth.run_frame(*args, _greedy, return_logits=True)
    dirty = torch.randn_like(lm.depth.new_scratch(2, dtype=torch.float32, device=torch.device("cpu"))) * 100
    again, logits2 = lm.depth.run_frame(*args, _greedy, scratch=dirty, return_logits=True)
    assert tokens.shape == (2, 16) and logits.shape == (2, 16, 2048)
    assert torch.equal(tokens, again)
    torch.testing.assert_close(logits, logits2)


def test_t6_depformer_uses_step_weights_and_feeds_forced_tokens():
    lm = _tiny_lm()
    text, hidden, target, provided = _frame_inputs()
    _, base = lm.depth.run_frame(text, hidden, target, provided, _greedy, return_logits=True)
    # Perturb step 5's attention block in layer 0 and step 9's FFN in layer 1.
    with torch.no_grad():
        dim = TINY.depformer_dim
        lm.depth.depformer.layers[0].self_attn.in_proj_weight.view(16, 3 * dim, dim)[5] += 0.5
        lm.depth.depformer.layers[1].gating[9].linear_out.weight.add_(0.5)
    _, changed = lm.depth.run_frame(text, hidden, target, provided, _greedy, return_logits=True)
    delta = (changed - base).abs().amax(dim=(0, 2))
    assert torch.all(delta[:5] == 0) and torch.all(delta[5:] > 0)

    lm = _tiny_lm()
    all_forced = torch.ones_like(provided)
    tokens, forced_logits = lm.depth.run_frame(text, hidden, target, all_forced, _greedy, return_logits=True)
    # Step k's input is codebook k-1's forced token; replay it step by step.
    scratch = lm.depth.new_scratch(2, dtype=torch.float32, device=torch.device("cpu"))
    prev = text
    for step in range(16):
        torch.testing.assert_close(lm.depth.step(step, prev, hidden, scratch), forced_logits[:, step])
        prev = target[:, step]
    assert torch.equal(tokens, forced_logits.argmax(-1))


# --------------------------------------------------------------------------- T5 / T13-LM (NVIDIA fixtures)


@pytest.fixture(scope="module")
def reference_manifest_fixture():
    data = verify_reference_fixture(json.loads(
        _fixture("PERSONAPLEX_REFERENCE_MANIFEST", "reference_lm_manifest.json").read_text()))
    assert data["lm_kwargs"]["dep_q"] == 16 and data["buffers"] == {}
    return data


def test_t5_native_keys_and_shapes_equal_nvidia_lmmodel_manifest(reference_manifest_fixture):
    with torch.device("meta"):
        lm = PersonaPlexLM(cfg.LMArchConfig.from_reference_kwargs(reference_manifest_fixture["lm_kwargs"]))
    native = reference_manifest(lm)
    reference = reference_manifest_fixture["parameters"]
    assert sorted(set(reference) - set(native)) == [] and sorted(set(native) - set(reference)) == []
    assert {k: v for k, v in native.items() if reference[k] != v} == {}
    assert reference_manifest_fixture["num_parameters"] == 8_371_408_896


@pytest.fixture(scope="module")
def reference_lm():
    pytest.importorskip("safetensors", reason="mstar.model.loader imports safetensors")
    data = load_binary_fixture(_fixture("PERSONAPLEX_REFERENCE_LM", "reference_lm.pt"))
    assert data["use_sampling"] is False and data["dtype"] == "float32"
    assert {(c["batch_size"], c["mode"]) for c in data["cases"]} == {
        (b, m) for b in (1, 2) for m in ("token_priming", "embedding_replay")
    }
    lm = PersonaPlexLM(cfg.LMArchConfig.from_reference_kwargs(data["lm_kwargs"])).eval()
    load_lm_weights(lm, data["state_dict"].items())
    return data, lm


def _close(actual, expected, what):
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5, msg=lambda m: f"{what}: {m}")


@torch.no_grad()
def test_t13_temporal_reference_cache_shape_preserves_shared_prefix_bits(reference_lm):
    """Isolate temporal numerics using NVIDIA inputs and a matching runtime.

    This is a component check, not a free-running trajectory comparison. A
    shorter physical SDPA key tensor changes rounding despite the same mask.
    """
    data, lm = reference_lm
    if (data.get("torch_version") != str(torch.__version__)
            or data.get("platform") != platform.platform()
            or data.get("torch_build_config") != torch.__config__.show()):
        pytest.skip("bitwise component check needs fixtures from this platform and PyTorch build")
    previous_threads = torch.get_num_threads()
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_num_threads(data["num_threads"])
        torch.set_float32_matmul_precision(data["float32_matmul_precision"])
        case = next(c for c in data["cases"] if c["batch_size"] == 1 and c["mode"] == "token_priming")
        cache = DenseTemporalCache(lm.config.num_layers, lm.config.rope_theta)
        for record in case["records"][:4]:
            hidden, logits = lm.temporal.forward_embeddings(lm.temporal.embed_codes(record["input"]), cache)
            cache.advance()
            torch.testing.assert_close(hidden, record["transformer_out"], rtol=0, atol=0)
            torch.testing.assert_close(logits[:, None], record["text_logits"], rtol=0, atol=0)
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("mode", ["token_priming", "embedding_replay"])
@torch.no_grad()
def test_t13_lm_greedy_lmgen_replay_matches_nvidia(reference_lm, batch, mode):
    """Native cache + temporal + depformer, run greedy end to end, against NVIDIA's LMGen.

    Each forward compares the model input, hidden state, text logits, depth
    inputs, all 16 depth logits and tokens, and the gathered output. Inputs are
    produced by M*'s own cache from M*'s own samples, so an early token error
    would cascade rather than be masked.
    """
    data, lm = reference_lm
    case = next(c for c in data["cases"] if c["batch_size"] == batch and c["mode"] == mode)
    records, preset = case["records"], case["preset"]
    sine = torch.tensor(cfg.SINE_TOKENS).view(1, 8, 1).expand(batch, -1, -1)
    silence = torch.tensor(cfg.SILENCE_TOKENS).view(1, 8, 1).expand(batch, -1, -1)
    pad = torch.full((batch,), 3, dtype=torch.long)

    def voice_frame(index):
        return (torch.arange(batch * 8).view(batch, 8, 1) * 37 + 100 + index * 11) % 2048

    if preset is not None:
        for inputs, embedding in zip(preset["inputs"], preset["embeddings"], strict=True):
            _close(lm.temporal.embed_codes(inputs), embedding, "preset embed_codes")

    cache = DelayedTokenCache(batch)
    temporal = DenseTemporalCache(lm.config.num_layers, lm.config.rope_theta)
    frames: list[tuple[str, dict | int]] = []
    if mode == "token_priming":
        frames += [("tokens", dict(user=sine, agent=voice_frame(i), text=pad)) for i in range(4)]
    else:
        frames += [("embedding", i) for i in range(len(preset["embeddings"]))]
    for block in ("silence_before", "text", "silence_after"):
        for token in [3] * 6 if block != "text" else [7, 11, 19]:
            frames.append(("tokens", dict(user=sine, agent=silence, text=torch.full((batch,), token))))
    frames += [("tokens", dict(user=voice_frame(i + 4))) for i in range(data["user_frames"])]

    position = 0
    for kind, frame in frames:
        if kind == "embedding":
            cache.prepare_embedding()
            x = preset["embeddings"][frame]
        elif not cache.prepare(**frame):
            continue
        else:
            x = lm.temporal.embed_codes(cache.model_input)
        record = records[position]
        what = f"B={batch} {mode} forward={position} {record['phase']}"
        assert torch.equal(cache.model_input, record["input"]), what
        hidden, text_logits = lm.temporal.forward_embeddings(x, temporal)
        temporal.advance()
        _close(hidden, record["transformer_out"], f"{what} transformer_out")
        _close(text_logits[:, None], record["text_logits"], f"{what} text_logits")
        sampled_text = _greedy(text_logits[:, 0])
        provided, target = cache.target_provided, cache.target
        next_text = torch.where(provided[:, 0, 0], target[:, 0, 0], sampled_text)
        assert torch.equal(next_text, record["depth_text"]), what
        assert torch.equal(target[:, 1:, 0], record["depth_target"]), what
        assert torch.equal(provided[:, 1:, 0], record["depth_provided"]), what
        audio, audio_logits = lm.depth.run_frame(
            next_text, hidden[:, 0], target[:, 1:, 0], provided[:, 1:, 0], _greedy, return_logits=True
        )
        _close(audio_logits, record["audio_logits"], f"{what} depth logits")
        assert torch.equal(audio, record["audio_tokens"]), what
        output = cache.commit(sampled_text, audio)
        assert (output is None) == (record["output"] is None), what
        if output is not None:
            assert torch.equal(output, record["output"]), what
        if kind == "embedding" and frame == len(preset["embeddings"]) - 1:
            cache.load(preset["cache"])
        position += 1
    assert position == len(records)
    assert records[-1]["phase"] == f"user_{data['user_frames'] - 1}"
