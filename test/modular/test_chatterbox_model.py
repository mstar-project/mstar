"""CPU contract tests for the Chatterbox model: registry, graph, partitions,
prompt processing, the conductor state machine and the T3 step declaration.
No weights are loaded; the model object is built without ``__init__``."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import StepContext, apply_yaml_overrides
from mstar.engine.resources.attn.config import AttentionSpec
from mstar.engine.resources.kv.config import KVReqConfig, KVSpec
from mstar.engine.resources.position.config import PositionSpec
from mstar.engine.resources.sampler.config import SamplerSpec
from mstar.model.chatterbox.chatterbox_model import (
    PREV_TOKEN,
    REF_AUDIO,
    SPEECH_TOKENS,
    TEXT_INPUTS,
    VOICE_KEY,
    ChatterboxModel,
    voice_key_for,
)
from mstar.model.chatterbox.config import (
    CFG_LABEL,
    COND_LABEL,
    T3_ATTN,
    T3_KV,
    T3_POS,
    T3_SAMPLER,
    UNCOND_LABEL,
    ChatterboxConfig,
    T3Config,
)
from mstar.model.chatterbox.submodules import (
    BuiltinT3Voice,
    S3GenSubmodule,
    T3Submodule,
    VoiceCache,
)
from mstar.model.registry import HF_MODELS, get_model_class
from mstar.model.submodule_base import ModelInputsFromEngine

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


class _TokenizerStub:
    def __init__(self, n=5):
        self.n = n
        self.last_text = None

    def __call__(self, text):
        self.last_text = text
        return torch.arange(1, self.n + 1, dtype=torch.long)


def _make_model(variant: str = "chatterbox", voices_dir=None) -> ChatterboxModel:
    model = object.__new__(ChatterboxModel)
    model.config = ChatterboxConfig.from_variant(variant)
    model.tokenizer = _TokenizerStub()
    model.voices_dir = voices_dir
    model.local_dir = "/nonexistent"
    model._submodule_cache = {}
    model._shared = {}
    return model


def _step_context(graph_walk: str, request_ids: list[str]) -> StepContext:
    return StepContext(
        request_ids=tuple(request_ids), graph_walk=graph_walk,
        slot=None, capture=False, plan_results={},
    )


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_variants_match_the_checkpoints():
    en = ChatterboxConfig.chatterbox()
    assert en.t3.cond_len == 34 and en.t3.backbone.kind == "llama"
    assert en.t3.backbone.num_hidden_layers == 30 and en.t3.speech_vocab_size == 8194
    assert en.t3.text_pos_table_size == 2050 and en.t3.speech_pos_table_size == 4100
    assert en.generation.cfg_weight == 0.5 and en.generation.min_p == 0.05

    tb = ChatterboxConfig.turbo()
    assert tb.is_turbo and tb.t3.backbone.is_gpt2
    assert tb.t3.cond_len == 1 + 375 and tb.t3.speech_vocab_size == 6563
    assert tb.t3.backbone.num_hidden_layers == 24 and tb.t3.speech_head_bias
    assert not tb.t3.duplicate_bos_in_prefill and tb.s3gen.meanflow
    assert tb.generation.cfg_weight == 0.0 and tb.generation.n_cfm_timesteps == 2
    assert tb.trailing_silence_tokens == 3 and tb.s3gen.hift.upsample_factor == 480

    assert ChatterboxConfig.from_model_path("ResembleAI/chatterbox-turbo").is_turbo
    assert not ChatterboxConfig.from_model_path("ResembleAI/chatterbox").is_turbo
    with pytest.raises(ValueError):
        ChatterboxConfig.from_variant("nano")
    assert T3Config.turbo().backbone.max_position_embeddings == 8196


# ---------------------------------------------------------------------------
# registry, graph, partitions, yaml
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key,repo,yaml_name", [
    ("chatterbox", "ResembleAI/chatterbox", "chatterbox.yaml"),
    ("chatterbox_turbo", "ResembleAI/chatterbox-turbo", "chatterbox_turbo.yaml"),
])
def test_registry_graph_and_yaml_are_consistent(key, repo, yaml_name):
    assert get_model_class(key) is ChatterboxModel
    assert HF_MODELS[key] == {"model_path_hf": repo}
    model = _make_model("turbo" if "turbo" in key else "chatterbox")

    walks = model.get_graph_walk_graphs()
    assert set(walks) == {"prefill", "prefill_voice", "decode", "s3gen_chunk", "s3gen_chunk_voice"}
    assert model.nodes == ["T3", "s3gen", "voice_encoder"]
    assert [p.name for p in model.get_partitions()] == ["T3", "S3Gen"]
    topo = model.get_partition_topology()
    assert topo.connections[0].edge_name == SPEECH_TOKENS
    policy = topo.connections[0].chunk_policy_factory()
    assert not policy.is_ready(model.config.t3.max_speech_tokens)  # flushes at producer done

    specs = model.get_node_resources()
    kv = next(s for s in specs if isinstance(s, KVSpec))
    assert kv.nodes == {"T3"} and kv.config.num_layers == model.config.t3.backbone.num_hidden_layers
    assert kv.config.head_dim == 64 and kv.config.num_kv_heads == 16
    pos = next(s for s in specs if isinstance(s, PositionSpec))
    if model.config.is_turbo:
        assert pos.config.llama31_params == {}
    else:
        assert pos.config.rope_scale == 8.0 and pos.config.old_context_len == 8192
    sampler = next(s for s in specs if isinstance(s, SamplerSpec))
    assert sampler.vocab_size == model.config.t3.speech_vocab_size
    assert next(s for s in specs if isinstance(s, AttentionSpec)).config.kv_cache == T3_KV

    serving = yaml.safe_load((CONFIGS / yaml_name).read_text(encoding="utf-8"))
    assert serving["model"] == key
    apply_yaml_overrides(specs, serving)
    assert kv.config.page_size == 64 and kv.config.max_num_pages == 1536

    worker_graphs = model.get_worker_graphs(str(CONFIGS / yaml_name))
    by_walk = {next(iter(wg.graph_walks)): wg for wg in worker_graphs}
    assert set(by_walk) == set(walks)
    assert by_walk["s3gen_chunk"].consumes_stream and by_walk["s3gen_chunk_voice"].consumes_stream
    assert not by_walk["prefill_voice"].consumes_stream
    assert all(wg.ranks == [0] for wg in worker_graphs)


def test_cli_adapter_and_benchmark_entries_are_registered():
    repo_root = str(Path(__file__).resolve().parents[2])
    sys.path.insert(0, repo_root)
    try:
        from benchmark.base import Chatterbox, ModelType, RequestType
        from mstar.api_server.openai.adapters import ADAPTER_REGISTRY, ChatterboxAdapter
        from mstar.cli.main import DEFAULT_CONFIGS, _next_steps

        assert DEFAULT_CONFIGS["chatterbox"] == "chatterbox.yaml"
        assert DEFAULT_CONFIGS["chatterbox_turbo"] == "chatterbox_turbo.yaml"
        assert isinstance(ADAPTER_REGISTRY["chatterbox"], ChatterboxAdapter)
        assert isinstance(ADAPTER_REGISTRY["chatterbox_turbo"], ChatterboxAdapter)
        assert 'voice="default"' in _next_steps("chatterbox", "0.0.0.0", 8000)
        bench = ModelType.CHATTERBOX.inst()
        assert isinstance(bench, Chatterbox)
        assert bench.get_supported_modalities() == {RequestType.T2S}
        assert bench.get_hf_url() == "ResembleAI/chatterbox"
    finally:
        sys.path.remove(repo_root)


def test_speech_adapter_maps_voice_and_reference_audio(tmp_path):
    from mstar.api_server.openai.adapters import ChatterboxAdapter

    req = SimpleNamespace(
        input="Hello", voice="default", temperature=0.7, top_p=0.9, seed=3,
        model_extra={"exaggeration": 0.8, "cfg_weight": 0.3, "speed": 1.2},
    )
    args = ChatterboxAdapter().speech_to_request(req, tmp_path)
    assert args.text == "Hello" and args.output_modalities == ["audio"]
    assert args.input_modalities == ["text"] and args.file_paths is None
    assert args.model_kwargs["voice"] == "default"
    assert args.model_kwargs["exaggeration"] == 0.8 and args.model_kwargs["cfg_weight"] == 0.3
    assert args.model_kwargs["temperature"] == 0.7 and args.model_kwargs["seed"] == 3

    clip = tmp_path / "ref.wav"
    clip.write_bytes(b"RIFF")
    req = SimpleNamespace(
        input="Hi", voice=None, temperature=None, top_p=None, seed=None,
        model_extra={"ref_audio": str(clip)},
    )
    args = ChatterboxAdapter().speech_to_request(req, tmp_path)
    assert args.input_modalities == ["text", "audio"]
    assert args.file_paths == {"audio": [str(clip)]}
    assert "ref_audio" not in args.model_kwargs


# ---------------------------------------------------------------------------
# prompt processing
# ---------------------------------------------------------------------------


def test_process_prompt_builtin_voice_emits_text_only():
    model = _make_model()
    out = model.process_prompt("hello", ["text"], ["audio"], voice="default")
    assert set(out) == {TEXT_INPUTS}
    assert out[TEXT_INPUTS][0].tolist() == [1, 2, 3, 4, 5]
    out = model.process_prompt("hello", ["text"], ["audio"])
    assert set(out) == {TEXT_INPUTS}


def test_process_prompt_reference_audio_is_keyed_and_capped():
    model = _make_model()
    wav = torch.rand(40 * 24000) - 0.5
    out = model.process_prompt(
        "hello", ["text", "audio"], ["audio"], tensors={"audio_inputs": [wav]},
    )
    assert set(out) == {TEXT_INPUTS, REF_AUDIO, VOICE_KEY}
    assert out[REF_AUDIO][0].numel() == 30 * 24000  # capped at 30 s
    assert out[VOICE_KEY][0].dtype == torch.long
    assert torch.equal(out[VOICE_KEY][0], voice_key_for(wav[: 30 * 24000]))
    assert not torch.equal(out[VOICE_KEY][0], voice_key_for(wav[1 : 30 * 24000 + 1]))


def test_process_prompt_turbo_rejects_short_reference():
    model = _make_model("turbo")
    with pytest.raises(ValueError, match="longer than 5 s"):
        model.process_prompt(
            "hello", ["text", "audio"], ["audio"],
            tensors={"audio_inputs": [torch.zeros(3 * 24000)]},
        )


@pytest.mark.parametrize("prompt,inputs,outputs,kwargs,message", [
    ("", ["text"], ["audio"], {}, "non-empty"),
    ("   ", ["text"], ["audio"], {}, "non-empty"),
    ("hi", ["image", "text"], ["audio"], {}, "optional reference audio"),
    ("hi", ["text"], ["text"], {}, "audio output only"),
    ("hi", ["text"], ["audio"], {"voice": "nobody"}, "no voices_dir"),
])
def test_process_prompt_rejects_bad_requests(prompt, inputs, outputs, kwargs, message):
    with pytest.raises(ValueError, match=message):
        _make_model().process_prompt(prompt, inputs, outputs, **kwargs)


def test_process_prompt_enforces_text_limit():
    model = _make_model()
    model.tokenizer = _TokenizerStub(n=model.config.max_text_tokens + 1)
    with pytest.raises(ValueError, match="Split it"):
        model.process_prompt("long", ["text"], ["audio"])


def test_preset_voice_lookup(tmp_path):
    model = _make_model(voices_dir=tmp_path)
    (tmp_path / "amy.wav").write_bytes(b"")
    with pytest.raises(ValueError, match=r"presets: \['amy'\]"):
        model._preset_voice_path("bob")
    assert model._preset_voice_path("amy") == tmp_path / "amy.wav"


# ---------------------------------------------------------------------------
# generation knobs and per-request resources
# ---------------------------------------------------------------------------


def test_generation_kwargs_defaults_and_turbo_guards():
    model = _make_model()
    knobs = model.resolve_generation_kwargs({})
    assert knobs["cfg_weight"] == 0.5 and knobs["exaggeration"] == 0.5
    assert knobs["min_p"] == 0.05 and knobs["repetition_penalty"] == 1.2
    assert knobs["max_new_tokens"] == 1000 and knobs["n_cfm_timesteps"] == 10
    assert model.resolve_generation_kwargs({"do_sample": False})["temperature"] == 0.0
    assert model.get_max_output_tokens(max_new_tokens=42) == 42
    with pytest.raises(ValueError, match="exceeds"):
        model.resolve_generation_kwargs({"max_new_tokens": 5000})

    configs = model.get_request_resource_configs({}, {"cfg_weight": 0.3, "temperature": 0.5, "seed": 1})
    assert configs[T3_SAMPLER].temperature == 0.5
    assert configs[T3_SAMPLER].repetition_penalty == 1.2
    assert isinstance(configs[T3_KV], KVReqConfig)
    assert configs[T3_KV].needed_labels == [COND_LABEL, UNCOND_LABEL]
    assert model.get_request_resource_configs({}, {"cfg_weight": 0})[T3_KV].needed_labels == [COND_LABEL]

    turbo = _make_model("turbo")
    knobs = turbo.resolve_generation_kwargs({"cfg_weight": 0.5, "exaggeration": 0.7, "min_p": 0.1})
    assert knobs["cfg_weight"] == 0.0 and knobs["exaggeration"] == 0.0 and knobs["min_p"] == 0.0
    assert knobs["top_k"] == 1000 and knobs["top_p"] == 0.95
    assert turbo.get_request_resource_configs({}, {})[T3_KV].needed_labels == [COND_LABEL]


# ---------------------------------------------------------------------------
# conductor state machine
# ---------------------------------------------------------------------------


def _pointers(*names):
    return {name: [SimpleNamespace(name=name)] for name in names}


def test_initial_args_builtin_voice():
    model = _make_model()
    signals = _pointers(TEXT_INPUTS)
    t3 = model.get_initial_forward_pass_args("T3", ["text"], ["audio"], signals, {"cfg_weight": 0.4})
    assert t3.full_metadata.graph_walk == "prefill" and t3.full_metadata.is_prefill
    assert [e.name for e in t3.inputs] == [TEXT_INPUTS]
    assert t3.unpersist_tensors == signals[TEXT_INPUTS]
    assert t3.step_metadata["cfg_weight"] == 0.4 and t3.step_metadata["is_prefill"] is True
    assert t3.full_metadata.kwargs["max_new_tokens"] == 1000

    s3 = model.get_initial_forward_pass_args("S3Gen", ["text"], ["audio"], signals, {})
    assert s3.full_metadata.graph_walk == "s3gen_chunk" and s3.inputs == []
    assert s3.step_metadata["n_cfm_timesteps"] == 10 and s3.step_metadata["watermark"] is True
    assert s3.request_done is False


def test_initial_args_uploaded_voice_routes_reference_to_both_partitions():
    model = _make_model()
    signals = _pointers(TEXT_INPUTS, REF_AUDIO, VOICE_KEY)
    t3 = model.get_initial_forward_pass_args("T3", ["text", "audio"], ["audio"], signals, None)
    assert t3.full_metadata.graph_walk == "prefill_voice"
    assert {(e.name, e.next_node) for e in t3.inputs} == {
        (TEXT_INPUTS, "T3"), (REF_AUDIO, "voice_encoder"), (VOICE_KEY, "voice_encoder"),
    }
    # the clip stays persisted: S3Gen reads it too
    assert t3.unpersist_tensors == signals[TEXT_INPUTS]

    s3 = model.get_initial_forward_pass_args("S3Gen", ["text", "audio"], ["audio"], signals, None)
    assert s3.full_metadata.graph_walk == "s3gen_chunk_voice"
    assert {(e.name, e.next_node) for e in s3.inputs} == {(REF_AUDIO, "s3gen"), (VOICE_KEY, "s3gen")}
    assert s3.inputs[0].tensor_info == signals[REF_AUDIO]


def test_t3_prefill_transitions_to_decode_then_done():
    model = _make_model()
    metadata = CurrentForwardConductorMetadata(
        input_modalities=["text"], output_modalities=["audio"],
        graph_walk="prefill_voice", is_prefill=True,
        kwargs={"cfg_weight": 0.5, "exaggeration": 0.5, "min_p": 0.05, "max_new_tokens": 10},
    )
    persist = {SPEECH_TOKENS: [SimpleNamespace(name=SPEECH_TOKENS)]}
    result = model.get_partition_forward_pass_args("T3", metadata, persist)
    assert result.full_metadata.graph_walk == "decode" and not result.full_metadata.is_prefill
    assert result.inputs[0].name == PREV_TOKEN and result.inputs[0].next_node == "T3"
    assert result.inputs[0].tensor_info == persist[SPEECH_TOKENS]
    assert result.unpersist_tensors == persist[SPEECH_TOKENS]
    assert result.step_metadata["is_prefill"] is False and result.step_metadata["max_new_tokens"] == 10
    assert result.request_done is False

    result = model.get_partition_forward_pass_args("T3", metadata, {})
    assert result.request_done is True


def test_s3gen_partition_reinjects_reference_each_chunk():
    model = _make_model()
    metadata = CurrentForwardConductorMetadata(
        input_modalities=["text", "audio"], output_modalities=["audio"],
        graph_walk="s3gen_chunk_voice", is_prefill=False,
        kwargs={"n_cfm_timesteps": 10, "watermark": False},
    )
    persist = _pointers(REF_AUDIO, VOICE_KEY, SPEECH_TOKENS)
    result = model.get_partition_forward_pass_args("S3Gen", metadata, persist)
    assert result.full_metadata.graph_walk == "s3gen_chunk_voice"
    assert [e.name for e in result.inputs] == [REF_AUDIO, VOICE_KEY]
    assert result.unpersist_tensors == [] and result.request_done is False
    assert result.step_metadata["watermark"] is False

    metadata.graph_walk = "s3gen_chunk"
    assert model.get_partition_forward_pass_args("S3Gen", metadata, persist).inputs == []


def test_postprocess_encodes_pcm16():
    model = _make_model()
    assert model.postprocess(torch.tensor([-1.0, 0.0, 1.0]), "audio") == \
        torch.tensor([-32767, 0, 32767], dtype=torch.int16).numpy().tobytes()
    assert model.postprocess(torch.zeros(0), "audio") == b""
    assert model.get_output_sample_rate() == 24000
    with pytest.raises(ValueError):
        model.postprocess(torch.zeros(1), "text")


# ---------------------------------------------------------------------------
# T3 submodule: step declaration, packing, batching, captures (no weights)
# ---------------------------------------------------------------------------


class _TinyT3(torch.nn.Module):
    """Stands in for T3Model: embedding tables and the methods the submodule calls."""

    def __init__(self, config: T3Config, dim: int = 8):
        super().__init__()
        self.config = config
        self.speech_emb = torch.nn.Embedding(config.speech_vocab_size, dim)
        self.speech_pos_emb = torch.nn.Embedding(config.speech_pos_table_size, dim)
        self.text_emb = torch.nn.Embedding(config.text_vocab_size, dim)

    def conditioning(self, speaker_emb, prompt_tokens, emotion_adv):
        return torch.zeros(1, self.config.cond_len, self.speech_emb.embedding_dim)

    def build_prefill_embeds(self, cond_emb, text_ids, *, uncond=False):
        text = torch.zeros_like(self.text_emb(text_ids)) if uncond else self.text_emb(text_ids)
        bos = self.speech_emb(torch.tensor([self.config.start_speech_token]))
        parts = [cond_emb, text, bos] + ([bos] if self.config.duplicate_bos_in_prefill else [])
        return torch.cat(parts)

    def embed_speech(self, ids, positions):
        return self.speech_emb(ids) + self.speech_pos_emb(positions)


def _t3_submodule(variant="chatterbox"):
    config = ChatterboxConfig.from_variant(variant)
    voice = BuiltinT3Voice(
        speaker_emb=torch.zeros(256), prompt_tokens=torch.zeros(3, dtype=torch.long),
    )
    return T3Submodule(_TinyT3(config.t3), config, builtin_voice=voice)


def _fwd_info(rid, cfg_weight=0.5, temperature=0.8, max_new=1000):
    return SimpleNamespace(
        request_id=rid,
        step_metadata={"cfg_weight": cfg_weight, "exaggeration": 0.5, "min_p": 0.05,
                       "max_new_tokens": max_new, "is_prefill": True},
        resource_configs={T3_SAMPLER: SimpleNamespace(temperature=temperature, ignore_eos=False)},
        max_tokens=4096, random_seed=0,
    )


def test_t3_prefill_inputs_and_cfg_step_declaration():
    sub = _t3_submodule()
    inputs = [
        sub.prepare_inputs("prefill", _fwd_info("a"), {TEXT_INPUTS: [torch.tensor([255, 5, 6, 0])]}),
        sub.prepare_inputs("prefill", _fwd_info("b"), {TEXT_INPUTS: [torch.tensor([255, 7, 0])]}),
    ]
    cond_len = sub.t3.cond_len
    assert inputs[0].input_seq_len == cond_len + 4 + 2  # cond | text | BOS BOS
    assert inputs[1].input_seq_len == cond_len + 3 + 2
    assert inputs[0].resource_step_info is True
    assert inputs[0].tensor_inputs["uncond_embeds"].shape == inputs[0].input_embeds.shape
    assert sub.request_state("a")["speech_step"] == 1

    batch = ExecutingBatch(
        node_name="T3", step_context=_step_context("prefill", ["a", "b"]),
        per_request_input_tensors={}, per_request_info={},
    )
    assert sub.can_batch(batch, inputs)
    packed = sub.preprocess("prefill", ModelInputsFromEngine(request_ids=["a", "b"], per_request_info={}), inputs)
    total = inputs[0].input_seq_len + inputs[1].input_seq_len
    assert packed["input_embeds"].shape[0] == 2 * total  # main rows then uncond rows
    assert packed["requires_cfg"] is True
    assert packed["cfg_weight"].tolist() == [[0.5], [0.5]]
    assert torch.allclose(packed["temperature"], torch.full((2, 1), 0.8))

    step = sub.declare_step("prefill", ["a", "b"], inputs)
    assert step.cg_key_info is True
    labels = [(s.request_id, s.label, s.span) for s in step.segments]
    assert labels == [
        ("a", COND_LABEL, inputs[0].input_seq_len), ("b", COND_LABEL, inputs[1].input_seq_len),
        ("a", UNCOND_LABEL, inputs[0].input_seq_len), ("b", UNCOND_LABEL, inputs[1].input_seq_len),
    ]
    assert step.steps[T3_KV].combined_labels == {(COND_LABEL, UNCOND_LABEL): CFG_LABEL}
    assert step.steps[T3_ATTN].causal is True
    assert set(step.steps) == {T3_KV, T3_ATTN, T3_POS, T3_SAMPLER}
    tracked = step.steps[T3_SAMPLER].prefill_tracked_tokens
    assert tracked["a"].tolist() == [sub.t3.start_speech_token]


def test_t3_decode_inputs_track_speech_positions_and_no_cfg_is_single_stream():
    sub = _t3_submodule()
    info = _fwd_info("a", cfg_weight=0.0)
    sub.prepare_inputs("prefill", info, {TEXT_INPUTS: [torch.tensor([255, 5, 0])]})
    d1 = sub.prepare_inputs("decode", info, {PREV_TOKEN: [torch.tensor([17])]})
    d2 = sub.prepare_inputs("decode", info, {PREV_TOKEN: [torch.tensor([18])]})
    assert d1.input_seq_len == 1 and d1.resource_step_info is False
    assert sub.request_state("a")["speech_step"] == 3
    expected = sub.model.speech_emb(torch.tensor([18])) + sub.model.speech_pos_emb(torch.tensor([2]))
    assert torch.allclose(d2.input_embeds, expected)

    step = sub.declare_step("decode", ["a"], [d1])
    assert step.cg_key_info is False and step.steps[T3_KV].combined_labels == {}
    assert [(s.label, s.span) for s in step.segments] == [(COND_LABEL, 1)]
    assert step.steps[T3_SAMPLER].prefill_tracked_tokens == {}

    packed = sub.preprocess("decode", ModelInputsFromEngine(request_ids=["a"], per_request_info={}), [d1])
    assert packed["input_embeds"].shape == (1, 8) and packed["requires_cfg"] is False


def test_t3_batches_only_one_guidance_mode_and_captures_both():
    sub = _t3_submodule()
    on = sub.prepare_inputs("prefill", _fwd_info("a", cfg_weight=0.5), {TEXT_INPUTS: [torch.tensor([255, 0])]})
    off = sub.prepare_inputs("prefill", _fwd_info("b", cfg_weight=0.0), {TEXT_INPUTS: [torch.tensor([255, 0])]})
    batch = ExecutingBatch(
        node_name="T3", step_context=_step_context("prefill", ["a", "b"]),
        per_request_input_tensors={}, per_request_info={},
    )
    assert not sub.can_batch(batch, [on, off])
    assert sub.can_batch(batch, [on, on])
    assert sub.cg_key_info("decode", {"a": _fwd_info("a", 0.5), "b": _fwd_info("b", 0.5)}) is True
    assert sub.cg_key_info("decode", {"a": _fwd_info("a", 0.5), "b": _fwd_info("b", 0.0)}) is None

    configs = sub.get_cuda_graph_configs(torch.device("cpu"))
    keys = {c.additional_key_info: c for c in configs}
    assert set(keys) == {True, False}
    assert keys[True].total_tokens_multiplier == 2 and keys[False].total_tokens_multiplier == 1
    assert keys[True].single_request_inputs.resource_step_info is True
    assert keys[True].capture_batch_sizes == [1, 2, 4, 8, 16, 32]
    assert all(c.capture_graph_walk == "decode" for c in configs)

    turbo = _t3_submodule("turbo")
    turbo_cfgs = turbo.get_cuda_graph_configs(torch.device("cpu"))
    assert [c.additional_key_info for c in turbo_cfgs] == [False]
    assert turbo.cg_key_info("decode", {"a": _fwd_info("a", 0.5)}) is False


def test_t3_min_p_mask_matches_hf_semantics():
    logits = torch.tensor([[2.0, 1.0, -3.0, 0.5]])
    temperature = torch.tensor([[0.8]])
    min_p = torch.tensor([[0.05]])
    probs = torch.softmax(logits / temperature, dim=-1)
    keep = probs >= min_p * probs.max()
    masked = T3Submodule._apply_min_p(logits, min_p, temperature)
    assert torch.equal(torch.isfinite(masked), keep)
    # min_p 0 keeps everything; greedy keeps the argmax
    assert torch.isfinite(T3Submodule._apply_min_p(logits, torch.zeros(1, 1), temperature)).all()
    greedy = T3Submodule._apply_min_p(logits, min_p, torch.zeros(1, 1))
    assert torch.isfinite(greedy).sum() == 1 and torch.isfinite(greedy[0, 0])


def test_t3_stop_on_eos_and_token_budget():
    sub = _t3_submodule()
    info = _fwd_info("a", max_new=3)
    sub.prepare_inputs("prefill", info, {TEXT_INPUTS: [torch.tensor([255, 0])]})
    out = {SPEECH_TOKENS: [torch.tensor([12])]}
    sub.postprocess("a", info, out)
    assert torch.equal(out[PREV_TOKEN][0], out[SPEECH_TOKENS][0])
    assert sub.check_stop("a", info, out) == set()
    eos = {SPEECH_TOKENS: [torch.tensor([sub.t3.stop_speech_token])]}
    sub.postprocess("a", info, eos)
    assert sub.check_stop("a", info, eos) == {"decode_loop"}
    info.resource_configs[T3_SAMPLER].ignore_eos = True
    assert sub.check_stop("a", info, eos) == set()
    sub.postprocess("a", info, out)
    assert sub.check_stop("a", info, out) == {"decode_loop"}  # 3 generated >= max_new 3
    sub.cleanup_request("a")
    assert "a" not in sub.request_states


# ---------------------------------------------------------------------------
# S3Gen submodule: token filtering, references, voice cache
# ---------------------------------------------------------------------------


class _FakeS3Gen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = []

    def tokens_to_mel(self, tokens, lens, ref, *, n_timesteps, generator=None, noise=None):
        self.calls.append(("mel", tokens.shape, int(lens[0]), n_timesteps, ref))
        return torch.zeros(1, 80, 2 * tokens.shape[1])

    def mel_to_wav(self, mel, *, generator=None):
        return torch.full((1, mel.shape[-1] * 480), 0.5)


def _s3_info(seed=7, watermark=False):
    return SimpleNamespace(
        request_id="r", random_seed=seed,
        step_metadata={"n_cfm_timesteps": 4, "watermark": watermark},
    )


def test_s3gen_filters_control_tokens_and_uses_builtin_voice():
    config = ChatterboxConfig.chatterbox()
    fake = _FakeS3Gen()
    sub = S3GenSubmodule(fake, s3_tokenizer=None, config=config, builtin_voice="builtin")
    tokens = torch.tensor([[6561], [10], [20], [6562]])  # BOS, speech, speech, EOS
    prepared = sub.prepare_inputs("s3gen_chunk", _s3_info(), {SPEECH_TOKENS: [tokens]})
    assert prepared.tensor_inputs[SPEECH_TOKENS].tolist() == [10, 20]
    assert prepared.kwargs["ref"] == "builtin" and prepared.kwargs["seed"] == 7
    assert prepared.kwargs["n_timesteps"] == 4 and prepared.kwargs["watermark"] is False

    out = sub.forward("s3gen_chunk", None, **prepared.tensor_inputs, **prepared.kwargs)
    pcm = out["audio_chunk"][0]
    assert pcm.dtype == torch.int16 and pcm.numel() == 2 * 2 * 480
    assert pcm[0].item() == int(0.5 * 32767)
    assert fake.calls[0][1:4] == ((1, 2), 2, 4)


def test_s3gen_turbo_appends_silence_and_empty_input_yields_no_audio():
    config = ChatterboxConfig.turbo()
    sub = S3GenSubmodule(_FakeS3Gen(), s3_tokenizer=None, config=config, builtin_voice="builtin")
    prepared = sub.prepare_inputs("s3gen_chunk", _s3_info(), {SPEECH_TOKENS: [torch.tensor([[5], [6562]])]})
    assert prepared.tensor_inputs[SPEECH_TOKENS].tolist() == [5, 4299, 4299, 4299]
    empty = sub.prepare_inputs("s3gen_chunk", _s3_info(), {SPEECH_TOKENS: [torch.tensor([[6562]])]})
    assert empty.tensor_inputs[SPEECH_TOKENS].numel() == 0
    out = sub.forward("s3gen_chunk", None, **empty.tensor_inputs, **empty.kwargs)
    assert out["audio_chunk"][0].numel() == 0


def test_s3gen_reference_is_cached_per_voice_key():
    config = ChatterboxConfig.chatterbox()
    sub = S3GenSubmodule(_FakeS3Gen(), s3_tokenizer=None, config=config, builtin_voice="builtin")
    calls = []
    sub.condition = lambda wav: calls.append(wav.numel()) or ("ref", wav.numel())  # noqa: E731
    inputs = {
        SPEECH_TOKENS: [torch.tensor([[1]])], REF_AUDIO: [torch.zeros(2400)], VOICE_KEY: [torch.tensor([99])],
    }
    first = sub.prepare_inputs("s3gen_chunk_voice", _s3_info(), inputs)
    second = sub.prepare_inputs("s3gen_chunk_voice", _s3_info(), inputs)
    assert first.kwargs["ref"] == ("ref", 2400) and second.kwargs["ref"] == first.kwargs["ref"]
    assert calls == [2400]


def test_voice_cache_is_lru():
    cache = VoiceCache(2)
    cache.put(1, "a")
    cache.put(2, "b")
    assert cache.get(1) == "a"
    cache.put(3, "c")
    assert cache.get(2) is None and cache.get(1) == "a" and cache.get(3) == "c"
