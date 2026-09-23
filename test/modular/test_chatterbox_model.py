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
from mstar.engine.cuda_graph_config import BatchedCudaGraphConfig, PackedCudaGraphConfig
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
from mstar.model.chatterbox.components.s3gen_graphs import SolveGraphs
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


def _sampler_spec(model: ChatterboxModel) -> SamplerSpec:
    return next(s for s in model.get_node_resources() if isinstance(s, SamplerSpec))


def _make_model(variant: str = "chatterbox", voices_dir=None) -> ChatterboxModel:
    model = object.__new__(ChatterboxModel)
    model.config = ChatterboxConfig.from_variant(variant)
    model._t3_dtype = torch.bfloat16
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
    assert policy.is_ready(model.config.stream_first_chunk_tokens)
    assert not policy.is_ready(model.config.stream_first_chunk_tokens - 1)
    model.config.stream_chunk_tokens = 0
    offline = topo.connections[0].chunk_policy_factory()
    assert not offline.is_ready(model.config.t3.max_speech_tokens)  # flushes at producer done

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
    # the file name other servers use for the same preset resolves too
    assert model._preset_voice_path("amy.wav") == tmp_path / "amy.wav"
    # but nothing outside the presets directory does
    (tmp_path.parent / "leak.wav").write_bytes(b"")
    with pytest.raises(ValueError, match="Unknown voice"):
        model._preset_voice_path("../leak")


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
    # signal-only after the first chunk: the node keeps the conditioned
    # reference, and the handshake the conductor issues after the final chunk
    # must not read a tensor the teardown may already have unlinked
    assert all(e.tensor_info == [] for e in result.inputs)
    assert result.unpersist_tensors == [] and result.request_done is False
    assert result.step_metadata["watermark"] is False
    # the first forward is the one that carries the clip
    initial = model.get_initial_forward_pass_args("S3Gen", ["text", "audio"], ["audio"], persist, {})
    assert [e.name for e in initial.inputs] == [REF_AUDIO, VOICE_KEY]
    assert all(len(e.tensor_info) == 1 for e in initial.inputs)

    metadata.graph_walk = "s3gen_chunk"
    assert model.get_partition_forward_pass_args("S3Gen", metadata, persist).inputs == []


def test_t3_dtype_switch():
    from mstar.model.chatterbox.chatterbox_model import _parse_dtype

    model = _make_model()
    assert model.get_autocast_dtype() == torch.bfloat16
    assert _parse_dtype("float32") == torch.float32 and _parse_dtype("bf16") == torch.bfloat16
    with pytest.raises(ValueError, match="t3_dtype"):
        _parse_dtype("int8")
    # float32 would size the KV cache and plan the paged attention in fp32,
    # which FlashInfer cannot run: refused up front with a clear message
    with pytest.raises(ValueError, match="cannot be served"):
        ChatterboxModel(model_path_hf="ResembleAI/chatterbox", variant="chatterbox", t3_dtype="float32")
    assert ChatterboxModel(
        model_path_hf="ResembleAI/chatterbox", variant="chatterbox", t3_dtype="float16",
    ).get_autocast_dtype() == torch.float16


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
    # sampling knobs are the sampler resource's, not forward inputs
    assert set(packed) == {"input_embeds", "requires_cfg", "cfg_weight"}

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
    batched = [c for c in configs if isinstance(c, BatchedCudaGraphConfig)]
    keys = {c.additional_key_info: c for c in batched}
    assert set(keys) == {True, False}
    assert keys[True].total_tokens_multiplier == 2 and keys[False].total_tokens_multiplier == 1
    assert keys[True].single_request_inputs.resource_step_info is True
    assert keys[True].capture_batch_sizes == [1, 2, 4, 8, 16, 32]
    assert all(c.capture_graph_walk == "decode" for c in batched)

    turbo = _t3_submodule("turbo")
    turbo_cfgs = [c for c in turbo.get_cuda_graph_configs(torch.device("cpu")) if isinstance(c, BatchedCudaGraphConfig)]
    assert [c.additional_key_info for c in turbo_cfgs] == [False]
    assert turbo.cg_key_info("decode", {"a": _fwd_info("a", 0.5)}) is False


def test_t3_prefill_is_captured_as_packed_graphs_for_both_guidance_modes():
    """Prefill replays packed captures (token buckets x small batch sizes) under
    both prefill walks; the stand-in inputs carry the guidance branch's second
    embedding row set like ``prepare_inputs`` does. The knob turns it off."""
    sub = _t3_submodule()
    packed = [c for c in sub.get_cuda_graph_configs(torch.device("cpu")) if isinstance(c, PackedCudaGraphConfig)]
    assert {c.additional_key_info for c in packed} == {True, False}
    for cfg in packed:
        assert cfg.capture_graph_walk == "prefill"
        assert cfg.replay_graph_walks == ["prefill", "prefill_voice"]
        assert cfg.capture_token_lengths == T3Submodule.PREFILL_TOKEN_BUCKETS
        assert cfg.capture_batch_sizes == T3Submodule.PREFILL_CAPTURE_BATCH_SIZES
        assert cfg.caps_eager_batch_size is False and cfg.compile is False
        stand_in = cfg.make_node_input(7)
        assert stand_in.input_seq_len == 7
        assert stand_in.input_embeds.shape == (7, sub.t3.hidden_size)
        assert stand_in.resource_step_info is cfg.additional_key_info
        assert ("uncond_embeds" in stand_in.tensor_inputs) is cfg.additional_key_info
        assert float(stand_in.tensor_inputs["cfg_weight"]) == (0.5 if cfg.additional_key_info else 0.0)
    prefill = ExecutingBatch(
        node_name="T3", step_context=_step_context("prefill_voice", ["a"]),
        per_request_input_tensors={}, per_request_info={},
    )
    inputs = [sub.prepare_inputs("prefill", _fwd_info("a"), {TEXT_INPUTS: [torch.tensor([255, 0])]})]
    assert sub.can_use_cuda_graphs(prefill, inputs)

    sub.config.t3_prefill_graphs = False
    assert not any(isinstance(c, PackedCudaGraphConfig) for c in sub.get_cuda_graph_configs(torch.device("cpu")))
    assert not sub.can_use_cuda_graphs(prefill, inputs)


def test_sampler_min_p_goes_through_the_resource():
    """The reference applies min_p after the penalty and temperature; that
    order only exists inside the sampler, so the knob rides the request's
    SamplingReqConfig and the T3 node declares the capability."""
    model = _make_model()
    assert model.get_request_resource_configs({}, {"seed": 1})[T3_SAMPLER].min_p == 0.05
    assert _sampler_spec(model).enable_min_p is True
    turbo = _make_model("turbo")
    assert turbo.get_request_resource_configs({}, {"min_p": 0.1, "seed": 1})[T3_SAMPLER].min_p == 0.0
    assert _sampler_spec(turbo).enable_min_p is False


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
    """Frame bookkeeping of the real S3Gen without its network: one token is
    two mel frames, one frame 480 samples; the mel carries the token id so the
    tests can see which frames a chunk re-synthesised."""

    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = []
        self.rows_calls = []
        self.buckets = []

    def tokens_to_mel(self, tokens, lens, ref, *, n_timesteps, generator=None, noise=None, finalize=True):
        self.calls.append(("mel", tokens.shape, int(lens[0]), n_timesteps, ref, finalize, noise is not None))
        frames = tokens.repeat_interleave(2, dim=1).float()[:, None]  # [1, 1, 2L]
        if not finalize:
            frames = frames[:, :, :-6]
        return frames.expand(-1, 80, -1).clone()

    def tokens_to_mel_rows(self, rows, *, n_timesteps, frame_bucket=0):
        self.buckets.append(frame_bucket)
        mels = [
            self.tokens_to_mel(
                row.tokens[None], torch.tensor([row.tokens.numel()]), row.ref, n_timesteps=n_timesteps,
                generator=row.generator, noise=row.noise, finalize=row.finalize,
            )
            for row in rows
        ]
        self.rows_calls.append(len(rows))  # kept apart so ``calls`` keeps its per-row order
        return mels

    def vocode(self, mel, *, generator=None, cache_source=None, fade_in=True):
        self.calls.append(("vocode", mel.shape[-1], cache_source is not None, fade_in))
        wav = mel[:, 0].repeat_interleave(480, dim=1) / 10000.0
        return wav, torch.ones(1, 1, wav.shape[-1])

    def mel_to_wav(self, mel, *, generator=None, fade_in=True):
        return self.vocode(mel, generator=generator, fade_in=fade_in)[0]


def _s3_info(seed=7, watermark=False, rid="r"):
    return SimpleNamespace(
        request_id=rid, random_seed=seed,
        step_metadata={"n_cfm_timesteps": 4, "watermark": watermark},
    )


def _s3_submodule(variant="chatterbox"):
    config = ChatterboxConfig.from_variant(variant)
    config.stream_context_tokens = 0  # these tests check the full-history bookkeeping
    fake = _FakeS3Gen()
    builtin = SimpleNamespace(num_prompt_tokens=4)
    return S3GenSubmodule(fake, s3_tokenizer=None, config=config, builtin_voice=builtin), fake


def _run(sub, tokens, is_final=None, rid="r"):
    inputs = {SPEECH_TOKENS: [tokens]} if tokens is not None else {SPEECH_TOKENS: []}
    kwargs = {} if is_final is None else {"is_final_stream_chunk": is_final}
    prepared = sub.prepare_inputs("s3gen_chunk", _s3_info(rid=rid), inputs, **kwargs)
    out = sub.forward("s3gen_chunk", None, **prepared.tensor_inputs, **prepared.kwargs)
    return prepared, out["audio_chunk"][0]


def test_s3gen_offline_chunk_filters_control_tokens_and_uses_builtin_voice():
    sub, fake = _s3_submodule()
    tokens = torch.tensor([[6561], [10], [20], [6562]])  # BOS, speech, speech, EOS
    prepared, pcm = _run(sub, tokens)  # no engine flag: EOS marks the end
    assert prepared.tensor_inputs[SPEECH_TOKENS].tolist() == [10, 20]
    assert prepared.kwargs["ref"] is sub.builtin_voice and prepared.kwargs["seed"] == 7
    assert prepared.kwargs["n_timesteps"] == 4 and prepared.kwargs["watermark"] is False
    assert prepared.kwargs["is_final"] is True
    assert pcm.dtype == torch.int16 and pcm.numel() == 2 * 2 * 480
    assert fake.calls[0][1:4] == ((1, 2), 2, 4) and fake.calls[0][5] is True


def test_s3gen_turbo_appends_silence_and_empty_input_yields_no_audio():
    sub, fake = _s3_submodule("turbo")
    prepared, pcm = _run(sub, torch.tensor([[5], [6562]]))
    assert prepared.tensor_inputs[SPEECH_TOKENS].tolist() == [5]
    assert fake.calls[0][1] == (1, 4)  # 5 + three silence tokens
    assert pcm.numel() == 4 * 2 * 480

    sub, fake = _s3_submodule("turbo")
    _, pcm = _run(sub, torch.tensor([[6562]]))
    assert pcm.numel() == 0 and fake.calls == []
    sub, fake = _s3_submodule("turbo")
    _, pcm = _run(sub, None)  # empty final flush
    assert pcm.numel() == 0


def test_s3gen_streaming_holds_back_lookahead_and_vocoder_tail():
    sub, fake = _s3_submodule()
    spf = 2 * 480  # samples per token
    tail = sub.cache_samples  # 8 frames x 480

    # chunk 1: 15 tokens, not final -> 12 tokens decoded, 8 frames held back
    _, pcm1 = _run(sub, torch.arange(1, 16)[:, None], is_final=False)
    assert pcm1.numel() == 12 * spf - tail
    mel_call, voc_call = fake.calls[-2], fake.calls[-1]
    assert mel_call[1] == (1, 15) and mel_call[5] is False and mel_call[6] is True  # noise field passed
    assert voc_call == ("vocode", 24, False, True)  # 24 new frames, no source cache, utterance fade-in
    stream = sub.request_state("r")["stream"]
    assert stream.token_offset == 12 and stream.noise is not None
    # prompt frames + every token the request may still produce
    assert stream.noise.shape[-1] == 2 * (4 + ChatterboxConfig.chatterbox().t3.max_speech_tokens)

    # chunk 2: 25 more tokens (40 total), still not final -> tokens 12..37 new
    _, pcm2 = _run(sub, torch.arange(16, 41)[:, None], is_final=False)
    assert pcm2.numel() == (37 - 12) * spf  # tail re-emitted, new tail withheld
    mel_call, voc_call = fake.calls[-2], fake.calls[-1]
    assert mel_call[1] == (1, 40) and mel_call[5] is False
    assert voc_call == ("vocode", 8 + 25 * 2, True, False)  # cache frames + new, source continued, no fade-in
    assert sub.request_state("r")["stream"].token_offset == 37

    # a small chunk still moves the look-ahead window: 2 more tokens decoded
    _, pcm3 = _run(sub, torch.tensor([[41], [42]]), is_final=False)
    assert pcm3.numel() == (39 - 37) * spf
    assert sub.request_state("r")["stream"].token_offset == 39

    # final flush with EOS: everything left (43 tokens total) is decoded
    _, pcm4 = _run(sub, torch.tensor([[43], [6562]]), is_final=True)
    assert pcm4.numel() == (43 - 39) * spf + tail
    assert fake.calls[-2][1] == (1, 43) and fake.calls[-2][5] is True
    assert sub.request_state("r")["stream"].done
    total = pcm1.numel() + pcm2.numel() + pcm3.numel() + pcm4.numel()
    assert total == 43 * spf

    # after the end, further chunks are ignored
    _, pcm5 = _run(sub, torch.tensor([[1]]), is_final=True)
    assert pcm5.numel() == 0


def test_s3gen_streaming_final_flush_without_new_tokens_releases_the_tail():
    sub, fake = _s3_submodule()
    _, pcm1 = _run(sub, torch.arange(1, 21)[:, None], is_final=False)  # 17 decoded
    _, pcm2 = _run(sub, torch.tensor([[6562]]), is_final=True)  # EOS only: 20 total, 3 new
    assert pcm1.numel() + pcm2.numel() == 20 * 2 * 480
    sub, fake = _s3_submodule()
    _run(sub, torch.arange(1, 21)[:, None], is_final=False)
    # engine says final but no new tokens arrive: the look-ahead tokens still finish
    _, pcm = _run(sub, None, is_final=True)
    assert pcm.numel() == 3 * 2 * 480 + sub.cache_samples


def test_s3gen_streaming_turbo_adds_silence_only_at_the_end():
    sub, fake = _s3_submodule("turbo")
    _run(sub, torch.arange(1, 21)[:, None], is_final=False)
    assert fake.calls[0][1] == (1, 20)
    _run(sub, torch.tensor([[21]]), is_final=True)
    assert fake.calls[-2][1] == (1, 24)  # 21 tokens + 3 silence, finalised once


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
    assert first.kwargs["is_final"] is False  # no flag, no EOS, tokens present


def test_voice_cache_is_lru():
    cache = VoiceCache(2)
    cache.put(1, "a")
    cache.put(2, "b")
    assert cache.get(1) == "a"
    cache.put(3, "c")
    assert cache.get(2) is None and cache.get(1) == "a" and cache.get(3) == "c"


def test_load_audio_decodes_with_soundfile_to_24k_mono(tmp_path):
    sf = pytest.importorskip("soundfile")
    sr = 16000
    t = torch.arange(sr) / sr
    stereo = torch.stack([torch.sin(2 * torch.pi * 440 * t), torch.zeros(sr)], dim=1)
    sf.write(str(tmp_path / "ref.wav"), stereo.numpy(), sr)

    loaded = _make_model().load_audio(str(tmp_path / "ref.wav"), "cpu")

    assert loaded.metadata == {"sample_rate": 24000, "num_channels": 1}
    assert loaded.data.dtype == torch.float32 and loaded.data.ndim == 1
    # one second of audio, resampled to 24 kHz and averaged to mono (half amplitude)
    assert abs(loaded.data.shape[0] - 24000) <= 1
    assert 0.4 < loaded.data.abs().max() < 0.55


def test_s3gen_batches_requests_into_one_flow_solve_and_matches_sequential():
    """Two streams advanced together give exactly the audio each gets alone,
    while their flow solves go through one rows call."""
    batched, fake_b = _s3_submodule()
    single, fake_s = _s3_submodule()
    chunks = {"a": [torch.arange(1, 21), torch.arange(21, 41), torch.tensor([41, 42])],
              "b": [torch.arange(101, 116), torch.arange(116, 131), torch.tensor([])]}
    finals = [False, False, True]

    expected = {rid: [] for rid in chunks}
    for rid, parts in chunks.items():
        for part, final in zip(parts, finals, strict=True):
            expected[rid].append(_run(single, part.long(), is_final=final, rid=rid)[1])

    got = {rid: [] for rid in chunks}
    for step, final in enumerate(finals):
        inputs = [
            batched.prepare_inputs(
                "s3gen_chunk", _s3_info(rid=rid), {SPEECH_TOKENS: [chunks[rid][step].long()]},
                is_final_stream_chunk=final,
            )
            for rid in chunks
        ]
        assert batched.can_batch(None, inputs)
        packed = batched.preprocess("s3gen_chunk", None, inputs)
        assert packed["request_id"] == ["a", "b"] and packed["is_final"] == [final, final]
        out = batched.forward_batched("s3gen_chunk", None, **packed)
        for rid in chunks:
            got[rid].append(out[rid]["audio_chunk"][0])

    for rid in chunks:
        for mine, theirs in zip(got[rid], expected[rid], strict=True):
            assert torch.equal(mine, theirs)
    # every step solves both streams in one padded batch ("b"'s empty final
    # flush still finalises its three withheld look-ahead tokens)
    assert fake_b.rows_calls == [2, 2, 2]
    assert set(fake_s.rows_calls) == {1}


def test_s3gen_batching_is_bounded_and_single_requests_keep_the_plain_path():
    sub, _ = _s3_submodule()
    one = [sub.prepare_inputs("s3gen_chunk", _s3_info(rid="a"), {SPEECH_TOKENS: [torch.tensor([1, 2])]})]
    assert not sub.can_batch(None, one)
    assert set(sub.preprocess("s3gen_chunk", None, one)) >= {SPEECH_TOKENS, "request_id", "ref", "is_final"}
    many = one * (sub.MAX_BATCH_SIZE + 1)
    assert not sub.can_batch(None, many) and sub.can_batch(None, one * 2)


def test_materialize_refuses_modules_with_computed_buffers():
    """A meta build + to_empty() would hand back uninitialised memory for a
    buffer computed in __init__ (the GPU HiFT window came out as garbage that
    way); such modules must be built for real."""
    from mstar.model.chatterbox.loader import materialize

    class WithWindow(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)
            self.register_buffer("window", torch.hann_window(16), persistent=False)

    with torch.device("meta"):
        bad = WithWindow()
        good = torch.nn.Linear(4, 4)
    with pytest.raises(ValueError, match="window"):
        materialize(bad, "cpu", torch.float32)
    assert materialize(good, "cpu", torch.float32).weight.device.type == "cpu"


def test_s3gen_context_window_bounds_the_solve_and_keeps_the_frame_bookkeeping():
    """With ``stream_context_tokens`` the flow sees only the prompt plus the
    last settled tokens, but the emitted audio must be the same as the full
    history run (the fake decoder's mel depends on nothing but the token ids,
    so any bookkeeping slip shows up as different or missing frames)."""
    full, fake_full = _s3_submodule()
    windowed, fake_win = _s3_submodule()
    windowed.context_tokens = 10
    chunks = [torch.arange(1, 21), torch.arange(21, 41), torch.arange(41, 61), torch.tensor([61, 62])]
    finals = [False, False, False, True]

    out_full = [_run(full, c, is_final=f)[1] for c, f in zip(chunks, finals, strict=True)]
    out_win = [_run(windowed, c, is_final=f)[1] for c, f in zip(chunks, finals, strict=True)]

    for a, b in zip(out_full, out_win, strict=True):
        assert torch.equal(a, b)
    full_lens = [c[1][1] for c in fake_full.calls if c[0] == "mel"]
    win_lens = [c[1][1] for c in fake_win.calls if c[0] == "mel"]
    # chunk 1 settles 17 tokens (20 - 3 look-ahead); chunk 2's solve keeps the
    # last 10 of them (tokens 7..39 = 33), chunk 3 likewise (27..59 = 33) and
    # the final one starts at 57 - 10 = 47 of 62 tokens
    assert full_lens == [20, 40, 60, 62]
    assert win_lens == [20, 33, 33, 15]


def test_streaming_and_compile_knobs_reach_the_config():
    model = ChatterboxModel(
        model_path_hf="ResembleAI/chatterbox", variant="chatterbox",
        stream_context_tokens=25, s3gen_compile=True,
    )
    assert model.config.stream_context_tokens == 25 and model.config.s3gen_compile is True
    assert _make_model().config.s3gen_compile is False


def test_graph_and_precision_knobs_reach_the_config():
    model = ChatterboxModel(
        model_path_hf="ResembleAI/chatterbox", variant="chatterbox",
        s3gen_graphs=True, s3gen_estimator_dtype="bfloat16", t3_prefill_graphs=False,
    )
    assert model.config.s3gen_graphs is True
    assert model.config.s3gen_frame_bucket == 64  # graphs need a bucket; 0 becomes the default one
    assert model._s3gen_estimator_dtype == torch.bfloat16
    assert model.config.t3_prefill_graphs is False
    with pytest.raises(ValueError, match="alternatives"):
        ChatterboxModel(
            model_path_hf="ResembleAI/chatterbox", variant="chatterbox", s3gen_graphs=True, s3gen_compile=True,
        )
    default = _make_model().config
    assert default.s3gen_graphs is False and default.s3gen_estimator_dtype == "float32"
    assert default.t3_prefill_graphs is True and default.s3gen_frame_bucket == 0


def test_graph_warmup_covers_the_built_in_voice_chunk_shapes():
    """One bucketed solve length per chunk size of the ramp: prompt frames plus
    2 x (left context + chunk + look-ahead), rounded up to the frame bucket."""
    model = _make_model()
    model.config.s3gen_frame_bucket = 64
    # 314 prompt frames; chunks 15 / 50 / 100 / 200 with 25 context and 3 look-ahead tokens
    assert ChatterboxModel._graph_warmup_frames(model, 314) == [448, 512, 576, 832]
    model.config.stream_chunk_tokens = 0  # whole-utterance mode: every utterance length is its own shape
    assert ChatterboxModel._graph_warmup_frames(model, 314) == []


# ---------------------------------------------------------------------------
# SolveGraphs: shape keys, row padding, reuse and eviction (no CUDA: a stand-in
# graph re-runs the solve on the static buffers when replayed)
# ---------------------------------------------------------------------------


class _FakeGraph:
    def __init__(self, run, output):
        self.run, self.output = run, output

    def replay(self):
        self.output.copy_(self.run())


def _fake_solve(mu, mask, spks, cond, noise, n_timesteps):
    return (2 * mu + noise + spks[:, :, None] + cond) * mask * n_timesteps


def _solve_graphs_on_cpu(solve=_fake_solve, **kwargs) -> SolveGraphs:
    graphs = SolveGraphs(solve, **kwargs)
    graphs._capturable = lambda mu: True

    def record(run, device):
        del device
        out = run()
        return _FakeGraph(run, out), out

    graphs._record = record
    return graphs


def _solve_inputs(batch: int, frames: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "mu": torch.randn(batch, 3, frames, generator=g), "mask": torch.ones(batch, 1, frames),
        "spks": torch.randn(batch, 3, generator=g), "cond": torch.randn(batch, 3, frames, generator=g),
        "noise": torch.randn(batch, 3, frames, generator=g),
    }


def test_solve_graphs_pad_rows_reuse_shapes_and_match_eager():
    seen = []

    def solve(mu, mask, spks, cond, noise, n_timesteps):
        seen.append(mu.shape[0])
        return _fake_solve(mu, mask, spks, cond, noise, n_timesteps)

    graphs = _solve_graphs_on_cpu(solve, rows=(1, 2, 4), max_graphs=8)
    a = _solve_inputs(3, 16, 1)
    assert torch.equal(graphs(**a, n_timesteps=2), _fake_solve(**a, n_timesteps=2))
    assert graphs.keys() == [(4, 16, 2)] and graphs.captures == 1 and seen[-1] == 4  # 3 rows padded to 4
    b = _solve_inputs(4, 16, 2)
    assert torch.equal(graphs(**b, n_timesteps=2), _fake_solve(**b, n_timesteps=2))
    assert graphs.captures == 1 and graphs.replays == 2  # same shape, replayed
    c = _solve_inputs(5, 16, 3)  # more rows than captured: eager
    assert torch.equal(graphs(**c, n_timesteps=2), _fake_solve(**c, n_timesteps=2))
    assert graphs.captures == 1 and graphs.replays == 2
    graphs(**_solve_inputs(1, 32, 4), n_timesteps=2)
    graphs(**_solve_inputs(1, 16, 5), n_timesteps=3)
    assert graphs.captures == 3 and set(graphs.keys()) == {(4, 16, 2), (1, 32, 2), (1, 16, 3)}


def test_solve_graphs_fall_back_to_eager_when_a_capture_fails(caplog):
    graphs = _solve_graphs_on_cpu(rows=(1,))

    def failing(run, device):
        raise RuntimeError("operation not permitted when stream is capturing")

    graphs._record = failing
    a = _solve_inputs(1, 8, 0)
    with caplog.at_level("WARNING"):
        assert torch.equal(graphs(**a, n_timesteps=2), _fake_solve(**a, n_timesteps=2))
    assert graphs.disabled and graphs.captures == 0 and "eagerly" in caplog.text
    assert graphs.warmup([8, 16], n_timesteps=2, example=_solve_inputs(1, 16, 0)) == 0
    assert torch.equal(graphs(**a, n_timesteps=2), _fake_solve(**a, n_timesteps=2))  # still served


def test_solve_graphs_drop_the_least_recently_used_shape():
    graphs = _solve_graphs_on_cpu(rows=(1,), max_graphs=2)
    for frames in (8, 16):
        graphs(**_solve_inputs(1, frames, 0), n_timesteps=1)
    graphs(**_solve_inputs(1, 8, 0), n_timesteps=1)  # 8 is recent again, 16 is the oldest
    graphs(**_solve_inputs(1, 24, 0), n_timesteps=1)
    assert graphs.keys() == [(1, 8, 1), (1, 24, 1)]


def test_solve_graphs_warmup_captures_the_grid_once():
    graphs = _solve_graphs_on_cpu(rows=(1, 2))
    example = _solve_inputs(1, 64, 0)
    assert graphs.warmup([32, 64], n_timesteps=2, example=example) == 4
    assert set(graphs.keys()) == {(1, 32, 2), (2, 32, 2), (1, 64, 2), (2, 64, 2)}
    assert graphs.warmup([32], n_timesteps=2, example=example) == 0
    out = graphs(**_solve_inputs(2, 32, 1), n_timesteps=2)
    assert out.shape == (2, 3, 32) and graphs.captures == 4


def test_s3gen_solves_go_through_the_solver_when_enabled():
    """``S3Gen.enable_graphs`` routes both the single and the batched path
    through the solver; without it the decoder is called directly."""
    from mstar.model.chatterbox.components.s3gen import S3Gen

    s3gen = object.__new__(S3Gen)
    s3gen.config = SimpleNamespace(meanflow=False)
    calls = []
    s3gen.decoder = SimpleNamespace(
        solve=lambda *a: calls.append(("decoder", a[-1])) or a[0],
        solve_meanflow=lambda *a: calls.append(("meanflow", a[-1])) or a[0],
    )
    s3gen.solver = None
    mu = torch.zeros(1, 2, 4)
    S3Gen._run_solve(s3gen, mu, mu, mu, mu, mu, 3)
    assert calls == [("decoder", 3)]
    s3gen.solver = lambda *a: calls.append(("graphs", a[-1])) or a[0]
    S3Gen._run_solve(s3gen, mu, mu, mu, mu, mu, 5)
    assert calls[-1] == ("graphs", 5)


def test_s3gen_node_advertises_its_batch_size_to_the_scheduler():
    """The micro-scheduler only groups requests up to ``max_batch_size``; the
    base default is one, which would silently disable the batched flow solve."""
    sub, _ = _s3_submodule()
    assert sub.max_batch_size("s3gen_chunk") == sub.MAX_BATCH_SIZE == 8
    assert sub.max_batch_size("s3gen_chunk_voice") == 8


def test_chunk_policy_growth_knobs_reach_the_ramp():
    model = ChatterboxModel(
        model_path_hf="ResembleAI/chatterbox", variant="chatterbox",
        stream_chunk_tokens=25, stream_chunk_growth=2.0, stream_max_chunk_tokens=100,
    )
    policy = model._chunk_policy()
    sizes = []
    for buffered in (15, 25, 50, 100, 100):
        assert policy.is_ready(buffered)
        sizes.append(policy.next_chunk_size(buffered))
        policy.register_chunk(buffered)
    assert sizes == [15, 25, 50, 100, 100]
    # the shipped default: 15, then 50, 100, 200, 200 ... with a 25-token context window
    default = _make_model()
    assert default.config.stream_context_tokens == 25
    policy = default._chunk_policy()
    sizes = []
    for buffered in (15, 50, 100, 200, 200):
        assert policy.is_ready(buffered)
        sizes.append(policy.next_chunk_size(buffered))
        policy.register_chunk(buffered)
    assert sizes == [15, 50, 100, 200, 200]


def test_s3gen_keeps_the_reference_for_chunks_without_the_clip():
    """The clip rides with the first chunk only; later chunks arrive with
    signal-only voice edges and must reuse the conditioned reference rather
    than fall back to the built-in voice."""
    config = ChatterboxConfig.chatterbox()
    sub = S3GenSubmodule(_FakeS3Gen(), s3_tokenizer=None, config=config, builtin_voice="builtin")
    sub.condition = lambda wav: ("ref", wav.numel())  # noqa: E731
    wav = torch.randn(2400)
    key = torch.tensor([12345])
    first = {SPEECH_TOKENS: [torch.tensor([1, 2, 3])], REF_AUDIO: [wav], VOICE_KEY: [key]}
    later = {SPEECH_TOKENS: [torch.tensor([4, 5, 6])], REF_AUDIO: [], VOICE_KEY: []}

    a = sub.prepare_inputs("s3gen_chunk_voice", _s3_info(rid="r"), first, is_final_stream_chunk=False)
    b = sub.prepare_inputs("s3gen_chunk_voice", _s3_info(rid="r"), later, is_final_stream_chunk=True)
    assert a.kwargs["ref"] == ("ref", 2400) and b.kwargs["ref"] == ("ref", 2400)
    # another request without any clip still gets the built-in voice
    c = sub.prepare_inputs("s3gen_chunk", _s3_info(rid="other"), {SPEECH_TOKENS: [torch.tensor([7])]})
    assert c.kwargs["ref"] == "builtin"
