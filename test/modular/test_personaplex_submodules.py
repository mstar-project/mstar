"""T12 and submodule checks for PersonaPlex (LM, MimiEncoder, MimiDecoder).

The LM's per-iteration logic is driven engine-free with the dense CPU attention
backend and greedy samplers; the paged-KV/FlashInfer path is only checked for
its wiring here (fake resources). R-tier tests replay the NVIDIA fixtures of
test_personaplex_components.py and test_personaplex_mimi.py.
"""

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from personaplex_fixture_utils import load_binary_fixture

from mstar.model.personaplex import config as cfg
from mstar.model.personaplex.components.delayed_cache import DelayedTokenCache
from mstar.model.personaplex.components.temporal import DenseTemporalCache, apply_rope_adjacent
from mstar.model.personaplex.personaplex_model import PersonaPlexModel
from mstar.model.personaplex.submodules import (
    LMSubmodule,
    MimiDecoderSubmodule,
    MimiEncoderSubmodule,
    ResourceTemporalBackend,
)
from mstar.model.personaplex.weights import PersonaPlexLM, build_lm, reference_lm_key

FIXTURES = Path(__file__).parent / "fixtures/personaplex"
TINY = cfg.LMArchConfig(
    dim=32, num_heads=4, num_layers=2, ffn_hidden=40, depformer_dim=16, depformer_heads=2,
    depformer_layers=2, depformer_ffn_hidden=24,
)


def _greedy_text(logits):
    return logits.float().argmax(-1)


def _greedy_depth(logits, _step):
    return logits.float().argmax(-1)


def _meta(n_emb, n_text, n_user):
    return {"num_voice_embeddings": n_emb, "num_text_tokens": n_text, "num_user_frames": n_user,
            "num_samples": n_user * 1920, "conditioning_frames": n_emb + 12 + n_text}


def _fixture(env, name):
    configured = os.environ.get(env)
    path = Path(configured) if configured else FIXTURES / name
    if not path.is_file() and not configured:
        pytest.skip(f"needs {name} from the NVIDIA oracle; see fixtures/personaplex/README.rst or set {env}")
    return load_binary_fixture(path)


class _Run:
    """Drive LMSubmodule frames the way the conditioning and generate walks do."""

    def __init__(self, sub, meta, voice, persona, user_codes):
        self.sub, self.meta = sub, meta
        self.backend = DenseTemporalCache(sub.lm.config.num_layers, sub.lm.config.rope_theta)
        self.static = {"voice_embeddings": [voice["embeddings"]], "voice_cache": [voice["cache"]],
                       "persona_tokens": [persona], "user_audio_codes": [user_codes]}

    def walk(self, walk, state, on_frame=None):
        outputs = []
        loop = cfg.CONDITIONING_LOOP if walk == cfg.CONDITIONING_WALK else cfg.DECODE_LOOP
        while True:
            inputs = {**self.static, "lm_state": [] if state is None else [state]}
            frame = self.sub.prepare_frame(walk, self.meta, inputs)
            assert frame is not None
            if on_frame:
                on_frame(frame)
            out = self.sub.run_frame(frame, self.backend, _greedy_text, _greedy_depth)
            self.backend.advance()
            outputs.append(out)
            state = out["lm_state"][0]
            info = SimpleNamespace(graph_walk=walk, step_metadata=self.meta)
            if self.sub.check_stop("r", info, out) == {loop}:
                break
            assert self.sub.check_stop("r", info, out) == set()
        # One more iteration would be an overshoot: vetoed, not run.
        assert self.sub.prepare_frame(walk, self.meta, {**self.static, "lm_state": [state]}) is None
        return outputs, state


# --------------------------------------------------------------------------- fixture-free


def test_lm_state_round_trips_and_rejects_corruption():
    cache = DelayedTokenCache(1)
    cache.prepare_embedding()
    cache.commit(torch.tensor([5]), torch.arange(16).view(1, 16))
    state = cache.state_tensor(7)
    assert state.shape == (1, 2 * 17 * 4 + 2) and state.dtype == torch.long
    again, step = DelayedTokenCache.from_state_tensor(state)
    assert step == 7 and again.offset == cache.offset
    assert torch.equal(again.cache, cache.cache) and torch.equal(again.provided, cache.provided)
    bad = state.clone()
    bad[0, 68] = 2
    with pytest.raises(ValueError, match="0/1"):
        DelayedTokenCache.from_state_tensor(bad)
    with pytest.raises(ValueError, match="int64"):
        DelayedTokenCache.from_state_tensor(state.int())
    cache.prepare(user=torch.zeros(1, 8, 1, dtype=torch.long))
    with pytest.raises(RuntimeError, match="commit"):
        cache.state_tensor(8)


def _tiny_submodule(seed=0):
    torch.manual_seed(seed)
    return LMSubmodule(PersonaPlexLM(TINY).eval())


def _random_inputs(n_emb=3, n_text=4, n_user=5, seed=1):
    g = torch.Generator().manual_seed(seed)
    voice = {"embeddings": torch.randn(n_emb, 1, 1, TINY.dim, generator=g),
             "cache": torch.randint(0, 2048, (1, 17, 4), generator=g)}
    persona = torch.randint(4, 32000, (n_text,), generator=g)
    codes = torch.randint(0, 2048, (1, 8, n_user), generator=g)
    return voice, persona, codes, _meta(n_emb, n_text, n_user)


@torch.no_grad()
def test_t12_conditioning_frames_are_fully_forced_and_generate_reads_the_mask_each_frame():
    sub = _tiny_submodule()
    voice, persona, codes, meta = _random_inputs()
    run = _Run(sub, meta, voice, persona, codes)
    masks = []
    original = sub.lm.depth.run_frame

    def spy(text, hidden, target, provided, sample, **kw):
        masks.append(provided.clone())
        return original(text, hidden, target, provided, sample, **kw)

    sub.lm.depth.run_frame = spy
    cond, state = run.walk(cfg.CONDITIONING_WALK, None)
    assert len(cond) == meta["conditioning_frames"] and masks == []  # no depformer while conditioning
    assert all(set(out) == {"lm_state"} for out in cond)
    gen, state = run.walk(cfg.GENERATE_WALK, state)
    assert len(gen) == meta["num_user_frames"] == len(masks)
    # First user step: agent acoustic (delay 1) still forced by the last conditioning frame,
    # agent semantic sampled (R4a, R15). Later steps sample all agent channels.
    acoustic = torch.zeros(16, dtype=torch.bool)
    acoustic[1:8] = True
    acoustic[8:] = True  # user channels are always provided
    assert torch.equal(masks[0][0], acoustic)
    for mask in masks[1:]:
        assert not mask[0, :8].any() and mask[0, 8:].all()
    assert DelayedTokenCache.from_state_tensor(state)[1] == meta["conditioning_frames"] + meta["num_user_frames"]


def test_t12_declare_step_declares_samplers_only_on_generate_walks():
    sub = _tiny_submodule()
    inputs = [SimpleNamespace(input_seq_len=1)]
    cond = sub.declare_step(cfg.CONDITIONING_WALK, ["r"], inputs)
    assert set(cond.steps) == {cfg.LM_KV, cfg.LM_ATTN, cfg.LM_POS}
    assert cond.steps[cfg.LM_ATTN].causal is True
    for walk in (cfg.GENERATE_WALK, cfg.GENERATE_AUDIO_WALK):
        gen = sub.declare_step(walk, ["r"], inputs)
        assert set(gen.steps) == {cfg.LM_KV, cfg.LM_ATTN, cfg.LM_POS, cfg.TEXT_SAMPLER, cfg.DEPTH_SAMPLER}
        assert [(s.request_id, s.label, s.span) for s in gen.segments] == [("r", "main", 1)]


@torch.no_grad()
def test_t12_conditioning_rejects_an_unforced_target():
    sub = _tiny_submodule()
    voice, persona, codes, meta = _random_inputs()
    frame = sub.prepare_frame(cfg.CONDITIONING_WALK, meta, {
        "voice_embeddings": [voice["embeddings"]], "voice_cache": [voice["cache"]],
        "persona_tokens": [persona], "lm_state": []})
    frame.cache.provided[:, 3, frame.cache.offset % cfg.CACHE_SIZE] = False
    with pytest.raises(RuntimeError, match="unprovided"):
        sub.run_frame(frame, DenseTemporalCache(2, 10000), _greedy_text, _greedy_depth)


@torch.no_grad()
def test_lm_frame_validation_and_engine_hooks():
    sub = _tiny_submodule()
    voice, persona, codes, meta = _random_inputs()
    with pytest.raises(ValueError, match="needs the conditioning lm_state"):
        sub.prepare_frame(cfg.GENERATE_WALK, meta, {"user_audio_codes": [codes], "lm_state": []})
    early = DelayedTokenCache(1).state_tensor(2)
    with pytest.raises(ValueError, match="before this walk's first frame"):
        sub.prepare_frame(cfg.GENERATE_WALK, meta, {"user_audio_codes": [codes], "lm_state": [early]})
    with pytest.raises(ValueError, match="disagree"):
        sub.prepare_frame(cfg.CONDITIONING_WALK, {**meta, "num_text_tokens": 9}, {
            "voice_embeddings": [voice["embeddings"]], "voice_cache": [voice["cache"]],
            "persona_tokens": [persona], "lm_state": []})
    with pytest.raises(ValueError, match="unknown"):
        sub.prepare_frame("decode", meta, {})
    fwd_info = SimpleNamespace(step_metadata=meta)
    inputs = {"voice_embeddings": [voice["embeddings"]], "voice_cache": [voice["cache"]],
              "persona_tokens": [persona], "lm_state": []}
    node_inputs = sub.prepare_inputs(cfg.CONDITIONING_WALK, fwd_info, inputs)
    assert node_inputs.input_seq_len == 1 and node_inputs.input_embeds.shape == (1, TINY.dim)
    assert sub.preprocess(cfg.CONDITIONING_WALK, None, [node_inputs]) == {"frame": node_inputs.kwargs["frame"]}
    with pytest.raises(NotImplementedError, match="batch size 1"):
        sub.preprocess(cfg.CONDITIONING_WALK, None, [node_inputs, node_inputs])
    with pytest.raises(RuntimeError, match="not bound"):
        sub.forward(cfg.CONDITIONING_WALK, None, node_inputs.kwargs["frame"])
    assert sub.can_batch(None, [node_inputs]) is False


class _FakeKV:
    def __init__(self):
        self.calls, self.keys, self.values = [], {}, {}
        self.default_label, self.layer = None, None

    def set_default_label(self, label):
        self.default_label = label

    def set_default_layer_idx(self, idx):
        self.layer = idx
        self.calls.append(("layer", idx))

    def write_kv(self, k, v):
        self.keys.setdefault(self.layer, []).append(k)
        self.values.setdefault(self.layer, []).append(v)

    def layer_view(self):
        return self.layer


class _FakeAttn:
    requires_kv_write = True

    def __init__(self, kv):
        self.kv = kv

    def set_default_label(self, label):
        pass

    def set_default_layer_idx(self, idx):
        pass

    def run(self, q, kv_cache_layer, k, v):
        keys = torch.stack(self.kv.keys[kv_cache_layer])  # [S, N, H, D]
        values = torch.stack(self.kv.values[kv_cache_layer])
        scores = torch.einsum("nhd,snhd->nhs", q, keys) / q.shape[-1] ** 0.5
        return torch.einsum("nhs,snhd->nhd", scores.softmax(-1), values)


class _FakePos:
    def __init__(self):
        self.calls, self.position = [], 0

    def apply_qk(self, q, k, label, **kwargs):
        self.calls.append((label, kwargs))
        q, k = apply_rope_adjacent(q[:, None], k[:, None], torch.tensor([self.position]), kwargs["rope_theta"])
        return q[:, 0], k[:, 0]


@torch.no_grad()
def test_resource_backend_wires_label_layers_and_interleaved_rope():
    sub = _tiny_submodule()
    kv, pos = _FakeKV(), _FakePos()
    sub.bind_node_resources({cfg.LM_KV: kv, cfg.LM_ATTN: _FakeAttn(kv), cfg.LM_POS: pos})
    assert isinstance(sub._backend, ResourceTemporalBackend)
    dense = DenseTemporalCache(TINY.num_layers, TINY.rope_theta)
    x = torch.randn(3, 1, TINY.dim)
    for t in range(3):
        pos.position = t
        sub._backend.begin()
        got, _ = sub.lm.temporal.forward_embeddings(x[t], sub._backend)
        want, _ = sub.lm.temporal.forward_embeddings(x[t], dense)
        dense.advance()
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    assert kv.default_label == "main"
    assert [c for c in kv.calls if c[0] == "layer"] == [("layer", i) for _ in range(3) for i in range(2)]
    assert {(label, kw["interleave"], kw["rope_theta"]) for label, kw in pos.calls} == {("main", True, 10000.0)}


def test_check_stop_ignores_other_walks_and_missing_state():
    sub = _tiny_submodule()
    meta = _meta(3, 4, 5)
    state = DelayedTokenCache(1).state_tensor(meta["conditioning_frames"])
    info = SimpleNamespace(graph_walk=cfg.CONDITIONING_WALK, step_metadata=meta)
    assert sub.check_stop("r", info, {"lm_state": [state]}) == {cfg.CONDITIONING_LOOP}
    assert sub.check_stop("r", info, {}) == set()
    gen = SimpleNamespace(graph_walk=cfg.GENERATE_AUDIO_WALK, step_metadata=meta)
    assert sub.check_stop("r", gen, {"lm_state": [state]}) == set()
    end = DelayedTokenCache(1).state_tensor(meta["conditioning_frames"] + 5)
    assert sub.check_stop("r", gen, {"lm_state": [end]}) == {cfg.DECODE_LOOP}


def test_build_lm_loads_strictly_in_bf16():
    source = _tiny_submodule(3).lm
    weights = [(reference_lm_key(k), v) for k, v in source.state_dict().items()]
    lm = build_lm(TINY, weights, dtype=torch.bfloat16)
    assert {p.dtype for p in lm.parameters()} == {torch.bfloat16} and not lm.training
    for name, value in lm.state_dict().items():
        assert torch.equal(value, source.state_dict()[name].to(torch.bfloat16)), name
    with pytest.raises(ValueError, match="1 missing"):
        build_lm(TINY, weights[1:])


def test_model_get_submodule_rejects_unknown_nodes_without_downloading(monkeypatch):
    model = PersonaPlexModel()
    monkeypatch.setattr(model, "_download", lambda name: pytest.fail("unexpected download"))
    with pytest.raises(ValueError, match="no node"):
        model.get_submodule("Talker")


# --------------------------------------------------------------------------- R tier: NVIDIA fixtures


@torch.no_grad()
def test_lm_submodule_conditioning_and_generate_match_nvidia_lmgen():
    """NVIDIA's embedding-replay B=1 case: preset, 6 silence, 3 text, 6 silence, 20 user frames."""
    pytest.importorskip("safetensors", reason="mstar.model.loader imports safetensors")
    data = _fixture("PERSONAPLEX_REFERENCE_LM", "reference_lm.pt")
    from mstar.model.personaplex.weights import load_lm_weights

    lm = PersonaPlexLM(cfg.LMArchConfig.from_reference_kwargs(data["lm_kwargs"])).eval()
    load_lm_weights(lm, data["state_dict"].items())
    case = next(c for c in data["cases"] if c["batch_size"] == 1 and c["mode"] == "embedding_replay")
    records, preset = case["records"], case["preset"]
    n_user = data["user_frames"]
    user = torch.cat([(torch.arange(8).view(1, 8, 1) * 37 + 100 + (i + 4) * 11) % 2048 for i in range(n_user)], -1)
    meta = _meta(len(preset["embeddings"]), 3, n_user)
    run = _Run(LMSubmodule(lm), meta, {"embeddings": preset["embeddings"], "cache": preset["cache"]},
               torch.tensor([7, 11, 19]), user)
    seen = []
    original = lm.temporal.forward_embeddings

    def record_forward(x, backend, *, text_logits=True):
        hidden, logits = original(x, backend, text_logits=text_logits)
        seen.append((hidden, logits))
        return hidden, logits

    lm.temporal.forward_embeddings = record_forward
    cond, state = run.walk(cfg.CONDITIONING_WALK, None)
    gen, state = run.walk(cfg.GENERATE_WALK, state)
    assert len(cond) + len(gen) == len(records) == len(seen)
    for index, (record, (hidden, logits)) in enumerate(zip(records, seen, strict=True)):
        what = f"forward {index} {record['phase']}"
        torch.testing.assert_close(hidden, record["transformer_out"][:, 0], rtol=1e-5, atol=1e-5, msg=what)
        if record["phase"].startswith("user_"):
            torch.testing.assert_close(logits, record["text_logits"][:, 0, 0], rtol=1e-5, atol=1e-5, msg=what)
        else:
            assert logits is None, what  # conditioning skips the text head
    for out, record in zip(gen, records[len(cond):], strict=True):
        assert torch.equal(out["agent_frame"][0], record["output"][:, 1:9]), record["phase"]
        assert torch.equal(out["text_frame"][0], record["output"][:, 0, 0]), record["phase"]


@contextmanager
def _without_triton_stub():
    stubs = {name: sys.modules.pop(name) for name in ("triton", "triton.language") if name in sys.modules}
    try:
        yield
    finally:
        sys.modules.update(stubs)


@torch.no_grad()
def test_mimi_submodules_match_nvidia_streaming_and_trim():
    with _without_triton_stub():
        pytest.importorskip("transformers", reason="the Mimi codec builds transformers.MimiModel")
    data = _fixture("PERSONAPLEX_REFERENCE_MIMI", "reference_mimi.pt")
    from mstar.model.personaplex.components import mimi as mimi_mod

    tiny = data["mimi_kwargs"]["tiny"]
    with _without_triton_stub():
        codec = mimi_mod.load_mimi_codec(
            data["state_dict"], mimi_mod.mimi_config_kwargs(tiny["seanet"], tiny["transformer"], tiny["quantizer"]))
    case = data["cases"][0]
    frames = case["frames"]
    encoder, decoder = MimiEncoderSubmodule(codec), MimiDecoderSubmodule(codec)
    meta = {"num_user_frames": frames, "num_samples": frames * 1920 - 700}
    info = SimpleNamespace(step_metadata=meta)
    audio = case["audio"].reshape(-1)
    prepared = encoder.prepare_inputs("encode_user_audio", info, {"audio_inputs": [audio]})
    assert torch.equal(prepared.tensor_inputs["pcm"], case["audio"])
    codes = encoder.forward("encode_user_audio", None, **prepared.tensor_inputs)["user_audio_codes"][0]
    assert torch.equal(codes, case["codes"])
    short = encoder.prepare_inputs("encode_user_audio", info, {"audio_inputs": [audio[:-700]]})
    assert torch.equal(short.tensor_inputs["pcm"][..., -700:], torch.zeros(1, 1, 700))
    with pytest.raises(ValueError, match="metadata expects"):
        encoder.prepare_inputs("encode_user_audio", info, {"audio_inputs": [audio[:1000]]})
    agent = [case["codes"][:, :, i : i + 1] for i in range(frames)]
    node = decoder.prepare_inputs("generate", info, {"agent_frame": agent})
    pcm = decoder.forward("generate", None, **node.tensor_inputs, **node.kwargs)["audio_output"][0]
    assert pcm.shape == (1, 1, meta["num_samples"])
    torch.testing.assert_close(pcm, case["pcm_from_codes"][..., : meta["num_samples"]], rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match="agent frames"):
        decoder.prepare_inputs("generate", info, {"agent_frame": agent[:-1]})


def _code_margins(codec, pcm):
    """Best-vs-second codebook distance per (codebook, frame) along HF's residual path."""
    quantizer = codec.model.quantizer
    latent = codec.encode_latent(pcm)
    margins = []
    for rvq, count in ((quantizer.semantic_residual_vector_quantizer, 1),
                       (quantizer.acoustic_residual_vector_quantizer, 7)):
        residual = rvq.input_proj(latent) if rvq.input_proj is not None else latent
        for layer in rvq.layers[:count]:
            embed = layer.codebook.embed
            dist = torch.cdist(residual[0].T[None].float(), embed[None].float())[0]
            top = dist.topk(2, largest=False).values
            margins.append(top[:, 1] - top[:, 0])
            residual = residual - embed[dist.argmin(-1)].T[None]
    return torch.stack(margins)


@torch.no_grad()
def test_offline_components_match_nvidia_with_reference_user_codes():
    """Compare M* stages with NVIDIA's offline.py, substituting NVIDIA's encoder codes.

    Tiny random LM and Mimi, greedy: warmup + reset, .pt voice replay, persona,
    131 user frames from stereo audio (channel 0 used, last frame zero-padded,
    > 124 frames so Mimi's ring mask matters), per-frame decode, trim to input.
    """
    pytest.importorskip("safetensors", reason="mstar.model.loader imports safetensors")
    with _without_triton_stub():
        pytest.importorskip("transformers", reason="the Mimi codec builds transformers.MimiModel")
    data = _fixture("PERSONAPLEX_REFERENCE_OFFLINE", "reference_offline.pt")
    from mstar.model.personaplex.components import mimi as mimi_mod
    from mstar.model.personaplex.weights import load_lm_weights

    lm = PersonaPlexLM(cfg.LMArchConfig.from_reference_kwargs(data["lm_kwargs"])).eval()
    load_lm_weights(lm, data["lm_state_dict"].items())
    tiny = data["mimi_kwargs"]["tiny"]
    with _without_triton_stub():
        mimi_config = mimi_mod.mimi_config_kwargs(tiny["seanet"], tiny["transformer"], tiny["quantizer"])
        codec = mimi_mod.load_mimi_codec(data["mimi_state_dict"], mimi_config)
    audio = data["user_audio"][0]  # PersonaPlexModel.load_audio keeps channel 0
    num_samples = audio.shape[-1]
    n_user = -(-num_samples // 1920)
    preset, persona = data["voice_preset"], torch.tensor(data["persona_tokens"])
    meta = _meta(preset["embeddings"].shape[0], len(persona), n_user)
    meta["num_samples"] = num_samples
    info = SimpleNamespace(step_metadata=meta)

    encoder = MimiEncoderSubmodule(codec)
    prepared = encoder.prepare_inputs(cfg.ENCODE_USER_AUDIO_WALK, info, {"audio_inputs": [audio]})
    codes = encoder.forward(cfg.ENCODE_USER_AUDIO_WALK, None, **prepared.tensor_inputs)["user_audio_codes"][0]
    reference_codes = data["user_codes"]
    # Codes may differ only where fp32 makes the nearest codebook entry a tie; a
    # residual quantizer then changes that frame's later codebooks too.
    margins = _code_margins(codec, prepared.tensor_inputs["pcm"])
    mismatch = (codes != reference_codes)[0]
    for frame in mismatch.any(0).nonzero().view(-1).tolist():
        first = int(mismatch[:, frame].nonzero()[0])
        assert margins[first, frame] < 1e-6, (frame, first, float(margins[first, frame]))
    assert mismatch.any(0).float().mean() < 0.02

    # This is component parity, not uninterrupted M* encoder-to-LM parity: the
    # near-tie above changes a code, so downstream stages use NVIDIA's codes.
    run = _Run(LMSubmodule(lm), meta, preset, persona, reference_codes)
    _, state = run.walk(cfg.CONDITIONING_WALK, None)
    generated, _ = run.walk(cfg.GENERATE_WALK, state)
    assert [int(out["text_frame"][0]) for out in generated] == data["text_tokens"]

    decoder = MimiDecoderSubmodule(codec)
    node = decoder.prepare_inputs(cfg.GENERATE_WALK, info, {"agent_frame": [o["agent_frame"][0] for o in generated]})
    pcm = decoder.forward(cfg.GENERATE_WALK, None, **node.tensor_inputs, **node.kwargs)["audio_output"][0]
    assert pcm.shape == (1, 1, num_samples)
    torch.testing.assert_close(pcm[0, 0], data["pcm"], rtol=1e-5, atol=1e-5)
