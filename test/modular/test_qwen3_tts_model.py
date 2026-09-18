import importlib.util
import json
import subprocess
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
from mstar.engine.resources.attn.wrappers import (
    FlashInferDecodeWrapper,
    FlashInferPrefillWrapper,
)
from mstar.engine.resources.kv.config import KVSpec
from mstar.engine.resources.sampler.config import SamplerSpec
from mstar.model.qwen3_tts.components.talker import (
    Qwen3TTSCodePredictor,
    Qwen3TTSTalkerModel,
)
from mstar.model.qwen3_tts.config import (
    CODE_PRED_SAMPLER,
    TALKER_ATTN,
    TALKER_SAMPLER,
    Qwen3TTSCodecConfig,
    Qwen3TTSCodePredictorConfig,
    Qwen3TTSModelConfig,
    Qwen3TTSTalkerConfig,
)
from mstar.model.qwen3_tts.qwen3_tts_model import Qwen3TTSModel
from mstar.model.qwen3_tts.submodules import CodecSubmodule, TalkerSubmodule
from mstar.model.registry import HF_MODELS, get_model_class
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine
from mstar.streaming.chunk_policy import ScheduledLeftContextChunkPolicy
from mstar.streaming.stream_buffer import StreamBuffer

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "qwen3tts.yaml"


ASSISTANT_PREFIX = [151644, 77091, 198]
ASSISTANT_SUFFIX = [151645, 198, 151644, 77091, 198]
USER_PREFIX = [151644, 872, 198]
USER_SUFFIX = [151645, 198]


class _TokenizerStub:
    """Tokenizes the two reference templates: fixed ChatML wrappers, one id per word."""

    def __init__(self):
        self.texts = []

    @property
    def last_text(self):
        return self.texts[-1] if self.texts else None

    def __call__(self, text, **kwargs):
        self.texts.append(text)
        assert kwargs == {"return_tensors": "pt", "padding": True}
        if text.endswith("<|im_end|>\n<|im_start|>assistant\n"):
            # the assistant turn to synthesize
            body = text[len("<|im_start|>assistant\n"):-len("<|im_end|>\n<|im_start|>assistant\n")]
            prefix, suffix = ASSISTANT_PREFIX, ASSISTANT_SUFFIX
        elif text.startswith("<|im_start|>assistant\n"):
            # the reference transcript turn (voice clone)
            body = text[len("<|im_start|>assistant\n"):-len("<|im_end|>\n")]
            prefix, suffix = ASSISTANT_PREFIX, USER_SUFFIX
        else:
            assert text.startswith("<|im_start|>user\n")
            body = text[len("<|im_start|>user\n"):-len("<|im_end|>\n")]
            prefix, suffix = USER_PREFIX, USER_SUFFIX
        words = [1000 + i for i, _ in enumerate(body.split())]
        return {"input_ids": torch.tensor([prefix + words + suffix])}


def _make_model() -> Qwen3TTSModel:
    model = object.__new__(Qwen3TTSModel)
    model.config = Qwen3TTSModelConfig()
    model.tokenizer = _TokenizerStub()
    model._submodule_cache = {}
    return model



def _step_context(graph_walk: str, request_ids: list[str]) -> StepContext:
    """Minimal eager StepContext; ExecutingBatch reads its graph_walk off this."""
    return StepContext(
        request_ids=tuple(request_ids),
        graph_walk=graph_walk,
        slot=None,
        capture=False,
        plan_results={},
    )

def test_qwen3_tts_config_reads_checkpoint_json(tmp_path):
    (tmp_path / "speech_tokenizer").mkdir()
    (tmp_path / "config.json").write_text(json.dumps({
        "tts_model_type": "custom_voice",
        "talker_config": {
            "num_hidden_layers": 30,
            "num_code_groups": 16,
            "spk_id": {"test_voice": 42},
            "code_predictor_config": {"num_hidden_layers": 6},
        },
    }))
    (tmp_path / "generation_config.json").write_text(json.dumps({
        "temperature": 0.7,
        "max_new_tokens": 123,
    }))
    (tmp_path / "speech_tokenizer" / "config.json").write_text(json.dumps({
        "output_sample_rate": 22050,
        "decoder_config": {"num_quantizers": 16, "codebook_size": 1024},
    }))

    config = Qwen3TTSModelConfig.from_pretrained(tmp_path)

    assert config.talker.num_hidden_layers == 30
    assert config.talker.code_predictor.num_hidden_layers == 6
    assert config.talker.spk_id == {"test_voice": 42}
    assert config.generation.temperature == 0.7
    assert config.generation.min_new_tokens == 2
    assert config.generation.max_new_tokens == 123
    assert config.codec.output_sample_rate == 22050
    assert config.codec.codebook_size == 1024


def test_qwen3_tts_model_loads_tokenizer_with_correct_regex(
    tmp_path, monkeypatch
):
    (tmp_path / "speech_tokenizer").mkdir()
    (tmp_path / "config.json").write_text(json.dumps({
        "tts_model_type": "custom_voice",
        "talker_config": {},
    }))
    (tmp_path / "generation_config.json").write_text("{}")
    (tmp_path / "speech_tokenizer" / "config.json").write_text("{}")
    captured = {}

    def from_pretrained(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return _TokenizerStub()

    monkeypatch.setattr(
        "mstar.model.qwen3_tts.qwen3_tts_model.AutoTokenizer.from_pretrained",
        from_pretrained,
    )
    Qwen3TTSModel(model_path_hf=str(tmp_path))

    assert captured["path"] == str(tmp_path)
    assert captured["fix_mistral_regex"] is True


def test_qwen3_tts_declares_talker_and_codec_graphs():
    model = _make_model()

    assert set(model.get_graph_walk_graphs()) == {
        "talker_prefill",
        "talker_decode",
        "codec_chunk",
    }
    assert [part.name for part in model.get_partitions()] == ["Talker", "Codec"]
    topology = model.get_partition_topology()
    assert topology.partitions == ["Talker", "Codec"]
    assert len(topology.connections) == 1
    assert topology.connections[0].edge_name == "codec_tokens"


def test_qwen3_tts_registry_engines_cache_and_yaml_are_consistent():
    model = _make_model()

    assert get_model_class("qwen3_tts") is Qwen3TTSModel
    assert HF_MODELS["qwen3_tts"] == {
        "model_path_hf": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    }
    specs = model.get_node_resources()
    kv_specs = [spec for spec in specs if isinstance(spec, KVSpec)]
    assert len(kv_specs) == 1
    kv_spec = kv_specs[0]
    assert kv_spec.nodes == {"Talker"}
    kv = kv_spec.config
    assert kv.num_layers == model.config.talker.num_hidden_layers
    assert kv.num_kv_heads == model.config.talker.num_key_value_heads
    assert kv.num_qo_heads == model.config.talker.num_attention_heads
    assert kv.head_dim == model.config.talker.head_dim
    serving_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert serving_config["resources"][TALKER_ATTN]["flashinfer_backend"] == "fa2"

    # the model default is "auto"; the deployment pins FA2 because the image
    # cannot build the FA3 JIT kernels, so the override has to reach attention
    attn = next(spec for spec in specs if isinstance(spec, AttentionSpec))
    assert attn.config.flashinfer_backend == "auto"
    apply_yaml_overrides(specs, serving_config)
    assert attn.config.flashinfer_backend == "fa2"

    worker_graphs = model.get_worker_graphs(str(CONFIG_PATH))
    by_walk = {
        next(iter(worker_graph.graph_walks)): worker_graph
        for worker_graph in worker_graphs
    }
    assert set(by_walk) == {
        "talker_prefill",
        "talker_decode",
        "codec_chunk",
    }
    assert all(worker_graph.ranks == [0] for worker_graph in worker_graphs)
    assert by_walk["codec_chunk"].consumes_stream is True


QWEN3_TTS_VARIANTS = {
    "qwen3_tts_1p7b": ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "qwen3tts_1p7b.yaml"),
    "qwen3_tts_voicedesign": ("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "qwen3tts_voicedesign.yaml"),
    "qwen3_tts_base": ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "qwen3tts_base.yaml"),
}


def test_qwen3_tts_1p7b_variants_share_class_configs_and_adapter():
    from mstar.api_server.openai.adapters import Qwen3TTSAdapter, get_adapter
    from mstar.cli.main import DEFAULT_CONFIGS

    base_yaml = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    for key, (hf_id, yaml_name) in QWEN3_TTS_VARIANTS.items():
        assert get_model_class(key) is Qwen3TTSModel
        assert HF_MODELS[key] == {"model_path_hf": hf_id}
        assert DEFAULT_CONFIGS[key] == yaml_name
        deployment = yaml.safe_load(
            (CONFIG_PATH.parent / yaml_name).read_text(encoding="utf-8")
        )
        assert deployment["model"] == key
        assert deployment["resources"] == base_yaml["resources"]
        ranks = {name: group["ranks"] for group in deployment["node_groups"] for name in group["node_names"]}
        base_ranks = {name: group["ranks"] for group in base_yaml["node_groups"] for name in group["node_names"]}
        assert ranks["Talker"] == base_ranks["Talker"] and ranks["Codec"] == base_ranks["Codec"]
        if key == "qwen3_tts_base":
            assert ranks["RefEncoder"] == ranks["Talker"]   # the clone prefill runs on the Talker's GPU
        else:
            assert deployment["node_groups"] == base_yaml["node_groups"]
        assert isinstance(get_adapter(key), Qwen3TTSAdapter)
    assert isinstance(get_adapter("qwen3_tts"), Qwen3TTSAdapter)


def test_qwen3_tts_cli_and_benchmark_entries_are_registered():
    repo_root = str(Path(__file__).resolve().parents[2])
    sys.path.insert(0, repo_root)
    from benchmark.base import ModelType, Qwen3TTS, RequestType
    from mstar.cli.main import DEFAULT_CONFIGS, _next_steps

    assert DEFAULT_CONFIGS["qwen3_tts"] == "qwen3tts.yaml"
    benchmark_model = ModelType.QWEN3TTS.inst()
    assert isinstance(benchmark_model, Qwen3TTS)
    assert benchmark_model.get_hf_url() == (
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    )
    assert benchmark_model.get_supported_modalities() == {RequestType.T2S}
    assert 'voice="Vivian"' in _next_steps("qwen3_tts", "0.0.0.0", 8000)
    sys.path.remove(repo_root)


def test_qwen3_tts_decoder_import_does_not_probe_sox():
    if importlib.util.find_spec("qwen_tts") is None:
        pytest.skip("qwen-tts optional dependency is not installed")
    script = """
import sys
import importlib.util
from mstar.model.qwen3_tts.qwen3_tts_model import _load_qwen3_tts_codec_classes
config_cls, decoder_cls, encoder_cls = _load_qwen3_tts_codec_classes()
print(config_cls.__name__, decoder_cls.__name__, encoder_cls.__name__)
print('sox_loaded=' + str('sox' in sys.modules))
print('public_qwen_tts_loaded=' + str(any(
    name == 'qwen_tts' or name.startswith('qwen_tts.') for name in sys.modules
)))
public_spec = importlib.util.find_spec('qwen_tts')
print('public_qwen_tts_origin=' + str(public_spec.origin))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Qwen3TTSTokenizerV2DecoderConfig Qwen3TTSTokenizerV2Decoder Qwen3TTSTokenizerV2Encoder" in (
        result.stdout
    )
    assert "sox_loaded=False" in result.stdout
    assert "public_qwen_tts_loaded=False" in result.stdout
    assert "qwen_tts/__init__.py" in result.stdout
    assert "SoX could not be found" not in result.stderr


def test_flashinfer_wrappers_forward_explicit_kernel_backend(monkeypatch):
    captured = {}

    class _PrefillWrapper:
        def __init__(self, *args, **kwargs):
            captured["prefill"] = kwargs["backend"]

    class _DecodeWrapper:
        def __init__(self, *args, **kwargs):
            captured["decode"] = kwargs["backend"]

    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(
            BatchPrefillWithPagedKVCacheWrapper=_PrefillWrapper,
            BatchDecodeWithPagedKVCacheWrapper=_DecodeWrapper,
        ),
    )
    common = {
        "workspace_buffer": torch.empty(1),
        "num_qo_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 8,
        "page_size": 16,
        "device": torch.device("cpu"),
        "backend": "fa2",
    }

    FlashInferPrefillWrapper(**common)
    FlashInferDecodeWrapper(**common)

    assert captured == {"prefill": "fa2", "decode": "fa2"}


def test_qwen3_tts_process_prompt_matches_official_template():
    model = _make_model()

    tensors = model.process_prompt(
        "你好",
        input_modalities=["text"],
        output_modalities=["audio"],
        voice="Vivian",
        language="Chinese",
    )

    assert model.tokenizer.last_text == (
        "<|im_start|>assistant\n你好<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert tensors["text_inputs"][0].tolist() == ASSISTANT_PREFIX + [1000] + ASSISTANT_SUFFIX
    # CustomVoice default: whole text in the prefill (stream_text = 0); no
    # reference transcript or frames.
    assert tensors["prompt_layout"][0].tolist() == [0, 1, 0, 0, 0]
    assert "ref_frames" not in tensors
    assert tensors["speaker_id"][0].item() == 3065
    assert tensors["language_id"][0].item() == 2055


def _variant_model(tts_model_type: str, tts_model_size: str = "1b7") -> Qwen3TTSModel:
    model = _make_model()
    talker = Qwen3TTSTalkerConfig(hidden_size=2048, intermediate_size=6144)
    if tts_model_type != "custom_voice":
        talker.spk_id = {}
        talker.spk_is_dialect = {}
    model.config = Qwen3TTSModelConfig(
        tts_model_type=tts_model_type, tts_model_size=tts_model_size, talker=talker
    )
    return model


def test_qwen3_tts_1p7b_custom_voice_prepends_instruction_turn():
    model = _variant_model("custom_voice")
    assert model.config.supports_instruct and not model.config.requires_instruct

    tensors = model.process_prompt(
        "hello big world",
        input_modalities=["text"],
        output_modalities=["audio"],
        voice="Ryan",
        instructions="speak slowly",
        non_streaming_mode=False,
    )

    instruct_ids = USER_PREFIX + [1000, 1001] + USER_SUFFIX
    assistant_ids = ASSISTANT_PREFIX + [1000, 1001, 1002] + ASSISTANT_SUFFIX
    assert model.tokenizer.texts[-1] == "<|im_start|>user\nspeak slowly<|im_end|>\n"
    assert tensors["text_inputs"][0].tolist() == instruct_ids + assistant_ids
    assert tensors["prompt_layout"][0].tolist() == [len(instruct_ids), 3, 1, 0, 0]
    assert tensors["speaker_id"][0].item() == 3061


def test_qwen3_tts_voice_design_requires_instruct_and_has_no_speakers():
    model = _variant_model("voice_design")
    assert model.config.default_speaker is None
    assert model.config.requires_instruct

    tensors = model.process_prompt(
        "hello",
        input_modalities=["text"],
        output_modalities=["audio"],
        instruct="A deep, calm male voice",
    )
    assert tensors["speaker_id"][0].item() == -1
    assert tensors["prompt_layout"][0].tolist() == [3 + 5 + 2, 1, 0, 0, 0]

    with pytest.raises(ValueError, match="requires an 'instruct'"):
        model.process_prompt("hello", input_modalities=["text"], output_modalities=["audio"])
    with pytest.raises(ValueError, match="no built-in speakers"):
        model.process_prompt(
            "hello", input_modalities=["text"], output_modalities=["audio"],
            voice="vivian", instruct="x",
        )


def test_qwen3_tts_base_config_declares_speaker_encoder():
    model = _variant_model("base")
    assert model.config.supports_reference_audio
    assert model.config.speaker_encoder is not None
    assert model.config.speaker_encoder.enc_dim == 2048
    # Base feeds text one token per frame by default (reference default).
    assert model.config.default_non_streaming_mode is False


def test_qwen3_tts_base_process_prompt_builds_in_context_clone():
    model = _variant_model("base")
    clip = torch.zeros(24000 + 1)  # 1 s + 1 sample -> 13 codec frames at 1920 samples/frame
    tensors = model.process_prompt(
        "hello big world",
        input_modalities=["audio", "text"],
        output_modalities=["audio"],
        tensors={"audio_inputs": [clip]},
        ref_text="the reference says",
        language="English",
    )
    ref_ids = ASSISTANT_PREFIX + [1000, 1001, 1002] + USER_SUFFIX  # same <|im_end|>\n tail
    assistant_ids = ASSISTANT_PREFIX + [1000, 1001, 1002] + ASSISTANT_SUFFIX
    assert model.tokenizer.texts[-1] == "<|im_start|>assistant\nthe reference says<|im_end|>\n"
    assert tensors["text_inputs"][0].tolist() == assistant_ids + ref_ids
    # [instruct_len, text_len, stream_text (Base default: streaming), ref_text_len, ref_frames]
    assert tensors["prompt_layout"][0].tolist() == [0, 3, 1, len(ref_ids), 13]
    assert tensors["ref_frames"][0].item() == 13
    assert tensors["speaker_id"][0].item() == -1

    xvec = model.process_prompt(
        "hello", input_modalities=["audio", "text"], output_modalities=["audio"],
        tensors={"audio_inputs": [clip]}, x_vector_only_mode=True,
    )
    assert xvec["prompt_layout"][0].tolist() == [0, 1, 1, 0, 0]
    assert xvec["ref_frames"][0].item() == 0

    with pytest.raises(ValueError, match="ref_text"):
        model.process_prompt(
            "hello", input_modalities=["audio", "text"], output_modalities=["audio"],
            tensors={"audio_inputs": [clip]},
        )
    with pytest.raises(ValueError, match="exactly one reference clip"):
        model.process_prompt("hello", input_modalities=["text"], output_modalities=["audio"])
    with pytest.raises(ValueError, match="no built-in speakers"):
        model.process_prompt(
            "hello", input_modalities=["audio", "text"], output_modalities=["audio"],
            tensors={"audio_inputs": [clip]}, ref_text="x", voice="vivian",
        )
    # Other variants refuse reference audio instead of ignoring it.
    with pytest.raises(ValueError, match="does not take reference audio"):
        _variant_model("custom_voice").process_prompt(
            "hello", input_modalities=["audio", "text"], output_modalities=["audio"],
            tensors={"audio_inputs": [clip]},
        )


def test_qwen3_tts_base_declares_clone_walks_and_routes_reference_audio():
    model = _variant_model("base")
    walks = model.get_graph_walk_graphs()
    assert {"talker_prefill_clone", "codec_chunk_clone"} <= set(walks)
    assert "RefEncoder" in model.nodes
    partitions = {part.name: part for part in model.get_partitions()}
    assert "talker_prefill_clone" in partitions["Talker"].graph_walks
    assert "codec_chunk_clone" in partitions["Codec"].graph_walks
    # Non-Base variants do not even declare the clone walks (config-driven).
    assert "talker_prefill_clone" not in _variant_model("voice_design").get_graph_walk_graphs()

    pointers = {
        name: [SimpleNamespace(name=name)]
        for name in (*Qwen3TTSModel.PREFILL_INPUTS, "audio_inputs", "ref_frames")
    }
    talker = model.get_initial_forward_pass_args(
        "Talker", input_modalities=["audio", "text"], output_modalities=["audio"],
        input_signals=pointers,
    )
    assert talker.full_metadata.graph_walk == "talker_prefill_clone"
    routes = {(edge.name, edge.next_node) for edge in talker.inputs}
    assert ("audio_inputs", "RefEncoder") in routes and ("prompt_layout", "RefEncoder") in routes
    assert ("text_inputs", "Talker") in routes
    codec = model.get_initial_forward_pass_args(
        "Codec", input_modalities=["audio", "text"], output_modalities=["audio"],
        input_signals=pointers,
    )
    assert codec.full_metadata.graph_walk == "codec_chunk_clone"
    # The very first codec chunk already needs the reference frame count: it
    # rides the initial inputs (and stays persisted for every later chunk).
    assert [edge.name for edge in codec.inputs] == ["ref_frames"]
    assert codec.inputs[0].tensor_info == pointers["ref_frames"]
    assert codec.unpersist_tensors == []
    rearmed = model.get_partition_forward_pass_args(
        "Codec", codec.full_metadata, persist_signals={"ref_frames": pointers["ref_frames"]},
    )
    assert rearmed.full_metadata.graph_walk == "codec_chunk_clone"
    assert [edge.name for edge in rearmed.inputs] == ["ref_frames"]
    assert rearmed.inputs[0].tensor_info == pointers["ref_frames"]

    # The Base deployment maps the extra node and walks.
    deployment = yaml.safe_load((CONFIG_PATH.parent / "qwen3tts_base.yaml").read_text(encoding="utf-8"))
    groups = {name: group for group in deployment["node_groups"] for name in group["node_names"]}
    assert "RefEncoder" in groups
    assert "talker_prefill_clone" in groups["RefEncoder"]["graph_walks"]
    assert "codec_chunk_clone" in groups["Codec"]["graph_walks"]
    by_walk = {}
    for worker_graph in model.get_worker_graphs(str(CONFIG_PATH.parent / "qwen3tts_base.yaml")):
        by_walk.setdefault(next(iter(worker_graph.graph_walks)), worker_graph)
    assert {"talker_prefill_clone", "codec_chunk_clone"} <= set(by_walk)


def test_qwen3_tts_config_rejects_unknown_variant():
    with pytest.raises(ValueError, match="tts_model_type"):
        Qwen3TTSModelConfig(tts_model_type="duplex")


def test_qwen3_tts_validates_speaker_dialect_after_language_override():
    model = _make_model()

    tensors = model.process_prompt(
        "你好",
        input_modalities=["text"],
        output_modalities=["audio"],
        voice="Eric",
        language="auto",
    )
    assert tensors["language_id"][0].item() == 2062

    model.config.talker.spk_is_dialect["vivian"] = "missing_dialect"
    with pytest.raises(ValueError, match="missing_dialect"):
        model.process_prompt(
            "你好",
            input_modalities=["text"],
            output_modalities=["audio"],
            voice="Vivian",
            language="auto",
        )


@pytest.mark.parametrize(
    ("prompt", "inputs", "outputs", "kwargs", "message"),
    [
        ("", ["text"], ["audio"], {}, "non-empty"),
        ("hello", ["audio"], ["audio"], {}, "does not take reference audio"),
        ("hello", ["video", "text"], ["audio"], {}, "text input only"),
        ("hello", ["text"], ["text"], {}, "audio output only"),
        ("hello", ["text"], ["audio", "text"], {}, "audio output only"),
        ("hello", ["text"], ["audio"], {"voice": "unknown"}, "speaker"),
        (
            "hello",
            ["text"],
            ["audio"],
            {"language": "unknown"},
            "language",
        ),
        (
            "hello",
            ["text"],
            ["audio"],
            {"instruct": "speak slowly"},
            "does not support instructions",
        ),
    ],
)
def test_qwen3_tts_rejects_unsupported_requests(
    prompt, inputs, outputs, kwargs, message
):
    with pytest.raises(ValueError, match=message):
        _make_model().process_prompt(
            prompt,
            input_modalities=inputs,
            output_modalities=outputs,
            **kwargs,
        )


def test_qwen3_tts_initial_partition_args_route_expected_inputs():
    model = _make_model()
    pointers = {
        name: [SimpleNamespace(name=name)]
        for name in Qwen3TTSModel.PREFILL_INPUTS
    }
    assert Qwen3TTSModel.PREFILL_INPUTS == (
        "text_inputs", "prompt_layout", "speaker_id", "language_id",
    )

    talker = model.get_initial_forward_pass_args(
        "Talker",
        input_modalities=["text"],
        output_modalities=["audio"],
        input_signals=pointers,
        model_kwargs={"max_new_tokens": 12, "subtalker_top_k": 7},
    )
    assert talker.full_metadata.graph_walk == "talker_prefill"
    assert [edge.name for edge in talker.inputs] == list(pointers)
    assert talker.full_metadata.kwargs["talker_max_tokens"] == 12
    # Residual-group sampling rides the code predictor's own per-request
    # sampler config, not step metadata.
    assert "subtalker_sampling" not in talker.step_metadata
    configs = model.get_request_resource_configs({}, {"subtalker_top_k": 7})
    assert configs[CODE_PRED_SAMPLER].top_k == 7

    codec = model.get_initial_forward_pass_args(
        "Codec",
        input_modalities=["text"],
        output_modalities=["audio"],
        input_signals=pointers,
    )
    assert codec.full_metadata.graph_walk == "codec_chunk"
    assert codec.inputs == []
    assert codec.request_done is False


def test_qwen3_tts_talker_prefill_transitions_to_decode():
    model = _make_model()
    metadata = CurrentForwardConductorMetadata(
        input_modalities=["text"],
        output_modalities=["audio"],
        graph_walk="talker_prefill",
        is_prefill=True,
        kwargs={
            "talker_max_tokens": 100,
        },
    )

    result = model.get_partition_forward_pass_args(
        partition_name="Talker",
        partition_metadata=metadata,
        persist_signals={"talker_input_embeds": []},
    )

    assert result.full_metadata.graph_walk == "talker_decode"
    assert result.full_metadata.is_prefill is False
    assert result.inputs[0].name == "talker_input_embeds"
    assert result.request_done is False


def test_qwen3_tts_talker_decode_marks_partition_done():
    model = _make_model()
    metadata = CurrentForwardConductorMetadata(
        input_modalities=["text"],
        output_modalities=["audio"],
        graph_walk="talker_decode",
        is_prefill=False,
    )

    result = model.get_partition_forward_pass_args(
        partition_name="Talker",
        partition_metadata=metadata,
        persist_signals={},
    )

    assert result.request_done is True


def test_qwen3_tts_postprocess_encodes_pcm16():
    model = _make_model()

    output = model.postprocess(
        torch.tensor([-1.0, 0.0, 1.0]),
        modality="audio",
    )

    expected = torch.tensor([-32767, 0, 32767], dtype=torch.int16)
    assert output == expected.numpy().tobytes()


def _tiny_model_config() -> Qwen3TTSModelConfig:
    code_predictor = Qwen3TTSCodePredictorConfig(
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        hidden_size=16,
        intermediate_size=32,
        head_dim=16,
        vocab_size=32,
        num_code_groups=4,
    )
    talker = Qwen3TTSTalkerConfig(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        hidden_size=16,
        intermediate_size=32,
        head_dim=8,
        vocab_size=64,
        text_hidden_size=16,
        text_vocab_size=128,
        num_code_groups=4,
        codec_pad_id=33,
        codec_bos_id=34,
        codec_eos_token_id=35,
        codec_think_id=36,
        codec_nothink_id=37,
        codec_think_bos_id=38,
        codec_think_eos_id=39,
        code_predictor=code_predictor,
    )
    return Qwen3TTSModelConfig(
        tts_pad_token_id=120,
        tts_bos_token_id=121,
        tts_eos_token_id=122,
        talker=talker,
        codec=Qwen3TTSCodecConfig(
            num_quantizers=4,
            chunk_schedule=(1,),
            chunk_frames=3,
            left_context_frames=2,
            upsample_rates=(2,),
            upsampling_ratios=(2,),
            decode_upsample_rate=4,
        ),
    )


def test_qwen3_tts_talker_builds_official_streaming_prefill():
    config = _tiny_model_config()
    talker = Qwen3TTSTalkerModel(config)
    predictor = Qwen3TTSCodePredictor(config)
    submodule = TalkerSubmodule(talker, predictor, config)
    submodule.CHATML_ASSISTANT_PREFIX_TOKEN_IDS = (1, 2, 3)
    submodule.CHATML_ASSISTANT_SUFFIX_TOKEN_IDS = (8, 9, 10, 11, 12)

    # 3 prefix + 4 text + 5 suffix tokens, streaming text layout.
    embeds = submodule._build_prefill(
        request_id="request",
        text_ids=torch.arange(1, 13),
        prompt_layout=torch.tensor([0, 4, 1]),
        speaker_id=40,
        language_id=-1,
    )

    # role(3) + [nothink, think_bos, think_eos, speaker, pad](5) + first text token
    assert embeds.shape == (9, 16)
    state = submodule.request_state("request")
    assert state["trailing_text_hidden"].shape == (4, 16)
    assert state["tts_pad_embed"].shape == (16,)
    assert state["generation_step"] == 0


def test_qwen3_tts_talker_builds_official_non_streaming_prefill():
    config = _tiny_model_config()
    submodule = TalkerSubmodule(
        Qwen3TTSTalkerModel(config), Qwen3TTSCodePredictor(config), config
    )
    submodule.CHATML_ASSISTANT_PREFIX_TOKEN_IDS = (1, 2, 3)
    submodule.CHATML_ASSISTANT_SUFFIX_TOKEN_IDS = (8, 9, 10, 11, 12)

    embeds = submodule._build_prefill(
        request_id="request",
        text_ids=torch.arange(1, 13),
        prompt_layout=torch.tensor([0, 4, 0]),
        speaker_id=40,
        language_id=41,
    )

    # role(3) + [think, think_bos, lang, think_eos, speaker, pad](6)
    # + (4 text + tts_eos) over codec pads (5) + (tts_pad + codec_bos)(1)
    assert embeds.shape == (15, 16)
    state = submodule.request_state("request")
    # Nothing streams: every decode frame adds the TTS PAD embedding.
    assert state["trailing_text_hidden"].shape == (0, 16)
    prepared = submodule.prepare_inputs(
        "talker_decode",
        SimpleNamespace(request_id="request"),
        {"talker_input_embeds": [torch.zeros(1, 16)]},
    )
    assert torch.equal(prepared.input_embeds[0], state["tts_pad_embed"])


def test_qwen3_tts_talker_prefill_prepends_instruction_without_speaker():
    config = _tiny_model_config()
    submodule = TalkerSubmodule(
        Qwen3TTSTalkerModel(config), Qwen3TTSCodePredictor(config), config
    )
    submodule.CHATML_ASSISTANT_PREFIX_TOKEN_IDS = (1, 2, 3)
    submodule.CHATML_ASSISTANT_SUFFIX_TOKEN_IDS = (8, 9, 10, 11, 12)
    instruct = torch.tensor([20, 21, 22, 23, 24, 25])
    text_ids = torch.cat([instruct, torch.arange(1, 13)])

    embeds = submodule._build_prefill(
        request_id="request",
        text_ids=text_ids,
        prompt_layout=torch.tensor([6, 4, 1]),
        speaker_id=-1,
        language_id=-1,
    )

    # instruct(6) + role(3) + [nothink, think_bos, think_eos, pad](4) + first text
    assert embeds.shape == (14, 16)
    # A layout whose text span disagrees with the token stream is rejected
    # even when the ChatML wrapper itself still lines up.
    with pytest.raises(ValueError, match="prompt layout disagrees"):
        submodule._build_prefill(
            request_id="request",
            text_ids=text_ids,
            prompt_layout=torch.tensor([6, 3, 1]),
            speaker_id=-1,
            language_id=-1,
        )


def test_qwen3_tts_talker_builds_in_context_clone_prefill():
    config = _tiny_model_config()
    submodule = TalkerSubmodule(
        Qwen3TTSTalkerModel(config), Qwen3TTSCodePredictor(config), config
    )
    submodule.CHATML_ASSISTANT_PREFIX_TOKEN_IDS = (1, 2, 3)
    submodule.CHATML_ASSISTANT_SUFFIX_TOKEN_IDS = (8, 9, 10, 11, 12)
    assistant = torch.arange(1, 13)                      # 4 text tokens
    reference = torch.tensor([1, 2, 3, 50, 51, 8, 9])     # 2 transcript tokens + <|im_end|>\n
    text_ids = torch.cat([assistant, reference])
    ref_codes = torch.randint(0, 32, (5, config.talker.num_code_groups))
    speaker_embed = torch.randn(config.talker.hidden_size)

    def build(layout):
        return submodule._build_prefill(
            request_id="clone", text_ids=text_ids, prompt_layout=torch.tensor(layout),
            speaker_id=-1, language_id=-1, speaker_embed=speaker_embed, ref_codes=ref_codes,
        )

    # Streaming text (Base default): text (2 ref + 4 + eos = 7) longer than
    # codec (bos + 5 frames = 6) -> 6 in the prefill, 1 trailing.
    embeds = build([0, 4, 1, 7, 5])
    # role(3) + [nothink, think_bos, think_eos, xvec, pad](5) + icl(6)
    assert embeds.shape == (3 + 5 + 6, 16)
    state = submodule.request_state("clone")
    assert state["trailing_text_hidden"].shape == (1, 16)
    assert torch.equal(state["reference_frames"], ref_codes)
    # The x-vector occupies the speaker slot right after the three think tags.
    tags = embeds[3:3 + 5]
    assert torch.allclose(tags[3], speaker_embed.to(tags.dtype) + state["tts_pad_embed"], atol=1e-5)

    # Non-streaming: (text + eos) over codec pads, then (bos + frames) + tts pad.
    embeds = build([0, 4, 0, 7, 5])
    assert embeds.shape == (3 + 5 + 7 + 6, 16)
    assert submodule.request_state("clone")["trailing_text_hidden"].shape == (0, 16)

    # Text shorter than the codec span: padded with TTS PAD, nothing trails.
    short = torch.cat([torch.tensor([1, 2, 3, 4, 8, 9, 10, 11, 12]), reference])
    embeds = submodule._build_prefill(
        request_id="clone", text_ids=short, prompt_layout=torch.tensor([0, 1, 1, 7, 5]),
        speaker_id=-1, language_id=-1, speaker_embed=speaker_embed, ref_codes=ref_codes,
    )
    assert embeds.shape == (3 + 5 + 6, 16)
    assert submodule.request_state("clone")["trailing_text_hidden"].shape == (0, 16)

    # x-vector only: standard layout with the x-vector in the speaker slot.
    embeds = submodule._build_prefill(
        request_id="xvec", text_ids=assistant, prompt_layout=torch.tensor([0, 4, 1, 0, 0]),
        speaker_id=-1, language_id=-1, speaker_embed=speaker_embed, ref_codes=None,
    )
    assert embeds.shape == (3 + 5 + 1, 16)
    assert "reference_frames" not in submodule.request_state("xvec")

    with pytest.raises(ValueError, match="reference frames"):
        build([0, 4, 1, 7, 9])


def test_qwen3_tts_clone_prefill_streams_reference_frames_first():
    config = _tiny_model_config()
    submodule = TalkerSubmodule(
        Qwen3TTSTalkerModel(config), Qwen3TTSCodePredictor(config), config
    )
    reference = torch.arange(8).view(2, 4)
    submodule.request_state("clone").add("reference_frames", reference)
    frame = torch.tensor([9, 9, 9, 9])
    items = submodule._codec_stream_items("talker_prefill_clone", "clone", frame)
    assert [item.tolist() for item in items] == [[0, 1, 2, 3], [4, 5, 6, 7], [9, 9, 9, 9]]
    # Only the clone prefill leads with the reference; decode never does.
    assert submodule._codec_stream_items("talker_decode", "clone", frame) == [frame]
    assert submodule._codec_stream_items("talker_prefill_clone", "other", frame) == [frame]


def test_qwen3_tts_talker_rejects_changed_chatml_layout():
    config = _tiny_model_config()
    submodule = TalkerSubmodule(
        Qwen3TTSTalkerModel(config),
        Qwen3TTSCodePredictor(config),
        config,
    )
    submodule.CHATML_ASSISTANT_PREFIX_TOKEN_IDS = (1, 2, 3)
    submodule.CHATML_ASSISTANT_SUFFIX_TOKEN_IDS = (8, 9, 10, 11, 12)
    text_ids = torch.arange(1, 13)
    text_ids[-1] = 7

    with pytest.raises(ValueError, match="ChatML assistant suffix changed"):
        submodule._build_prefill(
            request_id="request",
            text_ids=text_ids,
            prompt_layout=torch.tensor([0, 4, 1]),
            speaker_id=40,
            language_id=-1,
        )


def test_qwen3_tts_prefill_frame_counts_toward_generation_limit():
    config = _tiny_model_config()
    submodule = TalkerSubmodule(
        Qwen3TTSTalkerModel(config),
        Qwen3TTSCodePredictor(config),
        config,
    )
    submodule.request_state("request").add("generated_frames", 0)
    outputs = {"new_token": [torch.tensor(1)]}
    request_info = SimpleNamespace(
        step_metadata={"talker_max_tokens": 1},
        resource_configs={},
        max_tokens=8192,
    )

    submodule.postprocess("request", request_info, outputs)

    assert submodule.check_stop("request", request_info, outputs) == {
        "talker_decode_loop"
    }


def test_qwen3_tts_stops_on_eos_from_routed_codec_tokens():
    config = _tiny_model_config()
    submodule = TalkerSubmodule(
        Qwen3TTSTalkerModel(config),
        Qwen3TTSCodePredictor(config),
        config,
    )
    submodule.request_state("request").add("generated_frames", 0)
    outputs = {"codec_tokens": [torch.tensor([
        config.talker.codec_eos_token_id, 1, 2, 3
    ])]}
    request_info = SimpleNamespace(
        step_metadata={"talker_max_tokens": 100},
        resource_configs={TALKER_SAMPLER: SimpleNamespace(ignore_eos=False)},
        max_tokens=8192,
    )

    submodule.postprocess("request", request_info, outputs)

    assert outputs["layer0_codes"][0].item() == config.talker.codec_eos_token_id
    assert submodule.check_stop("request", request_info, outputs) == {
        "talker_decode_loop"
    }


def test_qwen3_tts_honors_ignore_eos_for_fixed_length_benchmarks():
    config = _tiny_model_config()
    submodule = TalkerSubmodule(
        Qwen3TTSTalkerModel(config),
        Qwen3TTSCodePredictor(config),
        config,
    )
    submodule.request_state("request").add("generated_frames", 0)
    outputs = {"new_token": [torch.tensor(
        config.talker.codec_eos_token_id
    )]}
    request_info = SimpleNamespace(
        step_metadata={"talker_max_tokens": 100},
        resource_configs={TALKER_SAMPLER: SimpleNamespace(ignore_eos=True)},
        max_tokens=8192,
    )

    submodule.postprocess("request", request_info, outputs)

    assert submodule.check_stop("request", request_info, outputs) == set()


def test_qwen3_tts_suppresses_eos_for_official_minimum_frames():
    config = _tiny_model_config()
    submodule = TalkerSubmodule(
        Qwen3TTSTalkerModel(config),
        Qwen3TTSCodePredictor(config),
        config,
    )
    inputs = [
        ARNodeInputs(tensor_inputs={
            "suppress_eos": torch.tensor([True]),
        }),
        ARNodeInputs(tensor_inputs={
            "suppress_eos": torch.tensor([False]),
        }),
    ]
    suppress_eos = submodule._get_batch_suppress_eos(inputs)
    eos = config.talker.codec_eos_token_id
    logits = submodule._mask_invalid_logits(
        torch.zeros(2, config.talker.vocab_size), suppress_eos
    )

    assert suppress_eos.tolist() == [True, False]
    assert torch.isneginf(logits[0, eos])
    assert logits[1, eos].item() == 0.0


def test_qwen3_tts_eos_suppression_ignores_graph_dummy_request_ids():
    config = _tiny_model_config()
    submodule = TalkerSubmodule(
        Qwen3TTSTalkerModel(config),
        Qwen3TTSCodePredictor(config),
        config,
    )
    real_state = submodule.request_state("real")
    real_state.add_all(
        generation_step=0,
        generated_frames=config.generation.min_new_tokens,
        trailing_text_hidden=torch.zeros(1, config.talker.hidden_size),
        tts_pad_embed=torch.zeros(config.talker.hidden_size),
    )
    prepared = submodule.prepare_inputs(
        "talker_decode",
        SimpleNamespace(request_id="real"),
        {"talker_input_embeds": [torch.zeros(1, config.talker.hidden_size)]},
    )
    packed = submodule.preprocess(
        "talker_decode",
        ModelInputsFromEngine(
            request_ids=["__graph_dummy__"],
            per_request_info={},
        ),
        [prepared],
    )

    eos = config.talker.codec_eos_token_id
    assert prepared.tensor_inputs["suppress_eos"].item() is False
    assert packed["suppress_eos"].item() is False
    logits = submodule._mask_invalid_logits(
        torch.zeros(1, config.talker.vocab_size), packed["suppress_eos"]
    )
    assert logits[0, eos].item() == 0.0
    assert "__graph_dummy__" not in submodule.request_states


def test_qwen3_tts_talker_batches_and_captures_decode():
    config = _tiny_model_config()
    submodule = TalkerSubmodule(
        Qwen3TTSTalkerModel(config),
        Qwen3TTSCodePredictor(config),
        config,
    )
    info = {
        request_id: SimpleNamespace(step_metadata={})
        for request_id in ("a", "b")
    }
    batch = ExecutingBatch(
        node_name="Talker",
        step_context=_step_context("talker_decode", ["a", "b"]),
        per_request_input_tensors={},
        per_request_info=info,
    )
    model_inputs = [
        ARNodeInputs(
            input_embeds=torch.zeros(1, 16),
            input_seq_len=1,
            tensor_inputs={"suppress_eos": torch.tensor([True])},
        )
        for _ in range(2)
    ]

    assert submodule.disable_torch_compile is True
    assert submodule.can_batch(batch, model_inputs)
    assert submodule.can_use_cuda_graphs(batch, model_inputs)
    packed = submodule.preprocess(
        "talker_decode",
        ModelInputsFromEngine(
            request_ids=["a", "b"],
            per_request_info=info,
        ),
        model_inputs,
    )
    assert packed["input_embeds"].shape == (2, 16)
    assert packed["last_token_indices"].tolist() == [0, 1]
    assert packed["suppress_eos"].tolist() == [True, True]
    graph_config = submodule.get_cuda_graph_configs(torch.device("cpu"))[0]
    assert graph_config.capture_graph_walk == "talker_decode"
    assert graph_config.capture_batch_sizes == [1, 2, 4, 8, 16, 32]
    assert graph_config.single_request_inputs.tensor_inputs[
        "suppress_eos"
    ].item() is True
    # Residual sampling params live in per-request sampler buffers, so requests
    # that disagree about them still batch AND still replay the decode graph.
    # (They used to fall out of both.)
    info["b"].step_metadata["subtalker_sampling"] = {"temperature": 0.7}
    assert submodule.can_batch(batch, model_inputs)
    assert submodule.can_use_cuda_graphs(batch, model_inputs)


def test_qwen3_tts_code_predictor_projects_wider_talker_inputs():
    """1.7B: Talker width 2048 vs predictor width 1024 -> biased projection on
    every depth input; 0.6B (equal widths) -> identity, no extra parameters."""
    narrow = _tiny_model_config()
    assert isinstance(
        Qwen3TTSCodePredictor(narrow).small_to_mtp_projection, torch.nn.Identity
    )

    wide = _tiny_model_config()
    wide.talker.hidden_size = 32
    predictor = Qwen3TTSCodePredictor(wide)
    projection = predictor.small_to_mtp_projection
    assert isinstance(projection, torch.nn.Linear)
    assert projection.weight.shape == (16, 32)
    assert projection.bias.shape == (16,)
    # Residual embedding tables stay in the Talker width: their sum feeds the
    # next Talker step, only the predictor input is projected.
    assert predictor.model.codec_embedding[0].weight.shape == (32, 32)
    assert {"small_to_mtp_projection.weight", "small_to_mtp_projection.bias"} <= set(
        dict(predictor.named_parameters())
    )

    for layer in predictor.model.layers:
        layer.input_layernorm = torch.nn.Identity()
        layer.post_attention_layernorm = torch.nn.Identity()
        layer.self_attn.q_norm = torch.nn.Identity()
        layer.self_attn.k_norm = torch.nn.Identity()
    predictor.model.norm = torch.nn.Identity()
    import mstar.model.qwen3_tts.components.talker as talker_module
    original_rope = talker_module.apply_rope_pos_ids
    original_attn = talker_module.decode_attn_nhd
    talker_module.apply_rope_pos_ids = lambda q, k, pos, theta: (q, k)
    talker_module.decode_attn_nhd = lambda q, k_cache, v_cache, n: (
        torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k_cache[:, :n].transpose(1, 2),
            v_cache[:, :n].transpose(1, 2), enable_gqa=True,
        ).transpose(1, 2)
    )
    try:
        cp = wide.talker.code_predictor
        out = predictor.forward_depth_unrolled(
            inputs_embeds=torch.randn(2, 1, 32),
            position_ids=torch.zeros(2, 1, dtype=torch.long),
            kv_cache=torch.zeros(
                cp.num_hidden_layers, 2, 2, wide.talker.num_code_groups,
                cp.num_key_value_heads, cp.head_dim,
            ),
            cache_pos=0,
        )
    finally:
        talker_module.apply_rope_pos_ids = original_rope
        talker_module.decode_attn_nhd = original_attn
    assert out.shape == (2, 1, 16)


def test_qwen3_tts_code_predictor_uses_decode_attn_nhd(monkeypatch):
    config = _tiny_model_config()
    predictor = Qwen3TTSCodePredictor(config)
    # This CPU-only contract test targets the decode-attention call. FlashInfer
    # RMSNorm and the Triton attention kernel are CUDA-only, so replace them
    # without changing attention geometry.
    for layer in predictor.model.layers:
        layer.input_layernorm = torch.nn.Identity()
        layer.post_attention_layernorm = torch.nn.Identity()
        layer.self_attn.q_norm = torch.nn.Identity()
        layer.self_attn.k_norm = torch.nn.Identity()
    predictor.model.norm = torch.nn.Identity()
    rope_calls = []

    def capture_rope(q, k, position_ids, rope_theta):
        rope_calls.append((q.shape, k.shape, position_ids.dtype, rope_theta))
        return q, k

    monkeypatch.setattr(
        "mstar.model.qwen3_tts.components.talker.apply_rope_pos_ids",
        capture_rope,
    )
    calls = []

    def capture_decode_attn(q, k_cache, v_cache, cache_len):
        calls.append((q.shape, k_cache.shape, v_cache.shape, cache_len))
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k_cache[:, :cache_len].transpose(1, 2),
            v_cache[:, :cache_len].transpose(1, 2),
            is_causal=False,
            enable_gqa=True,
        ).transpose(1, 2)

    monkeypatch.setattr(
        "mstar.model.qwen3_tts.components.talker.decode_attn_nhd",
        capture_decode_attn,
    )
    output = predictor.forward_depth_unrolled(
        inputs_embeds=torch.randn(1, 1, config.talker.hidden_size),
        position_ids=torch.zeros(1, 1, dtype=torch.long),
        kv_cache=torch.empty(
            config.talker.code_predictor.num_hidden_layers,
            1,
            2,
            config.talker.num_code_groups,
            config.talker.code_predictor.num_key_value_heads,
            config.talker.code_predictor.head_dim,
        ),
        cache_pos=0,
    )

    query_shape, key_shape, value_shape, cache_len = calls[0]
    assert output.shape == (1, 1, config.talker.hidden_size)
    assert query_shape[2] == config.talker.code_predictor.num_attention_heads
    assert key_shape[2] == config.talker.code_predictor.num_key_value_heads
    assert value_shape[2] == config.talker.code_predictor.num_key_value_heads
    assert key_shape[1] == config.talker.num_code_groups
    assert cache_len == 1
    assert len(calls) == config.talker.code_predictor.num_hidden_layers
    assert len(rope_calls) == config.talker.code_predictor.num_hidden_layers
    assert rope_calls[0][2] == torch.int32


def test_qwen3_tts_per_request_sampler_configs_drive_code_predictor():
    """Residual groups are configured through the code predictor's own sampler
    resource, which the engine opens per-request buffers for."""
    model = _make_model()
    generation = model.config.generation

    default = model.get_request_resource_configs({})[CODE_PRED_SAMPLER]
    assert default.temperature == generation.subtalker_temperature
    assert default.top_k == generation.subtalker_top_k
    assert default.top_p == generation.subtalker_top_p
    # No penalty on the depth loop.
    assert default.repetition_penalty == 1

    overridden = model.get_request_resource_configs(
        {},
        {"subtalker_temperature": 0.5, "subtalker_top_k": 3, "subtalker_top_p": 0.25},
    )[CODE_PRED_SAMPLER]
    assert (overridden.temperature, overridden.top_k, overridden.top_p) == (0.5, 3, 0.25)

    # do_sample=False is expressed as temperature 0 (encoded as greedy downstream).
    greedy = model.get_request_resource_configs(
        {}, {"subtalker_dosample": False}
    )[CODE_PRED_SAMPLER]
    assert greedy.temperature == 0.0

    # Two samplers, one per head; only the Talker's carries a vocab for the
    # repetition penalty (see get_node_resources).
    configs = model.get_request_resource_configs({})
    assert set(configs) == {TALKER_SAMPLER, CODE_PRED_SAMPLER}
    specs = {
        spec.resource_key: spec for spec in model.get_node_resources()
        if isinstance(spec, SamplerSpec)
    }
    assert specs[TALKER_SAMPLER].vocab_size is not None
    assert specs[CODE_PRED_SAMPLER].enable_repetion_penalty is False

    # The conductor seeds each stream; they are seeded independently.
    configs[TALKER_SAMPLER].apply_conductor_config(seed=99)
    assert configs[TALKER_SAMPLER].seed == 99
    assert configs[CODE_PRED_SAMPLER].seed != 99


class _FakeCodecDecoder(torch.nn.Module):
    def __init__(self, upsample: int):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.upsample = upsample

    def forward(self, codes):
        length = codes.shape[-1] * self.upsample
        return torch.zeros(codes.shape[0], 1, length, dtype=torch.float32)


def test_qwen3_tts_codec_trims_reported_context_audio():
    config = _tiny_model_config()
    submodule = CodecSubmodule(_FakeCodecDecoder(4), config)
    assert submodule.windows == [1, 4] and submodule.max_window == 4

    # The stream buffer reports how many leading frames are repeated context;
    # the first window has none, later ones up to left_context (2). The
    # geometry travels with the pass's inputs.
    first = {"audio_chunk": [torch.arange(20)]}
    submodule.postprocess("request", None, first, inputs=_geometry(frames=5, context=0))
    assert first["audio_chunk"][0].tolist() == list(range(20))

    second = {"audio_chunk": [torch.arange(20)]}
    submodule.postprocess("request", None, second, inputs=_geometry(frames=5, context=2))
    assert second["audio_chunk"][0].tolist() == list(range(8, 20))

    # Padding frames of a bucket never reach the client.
    padded = {"audio_chunk": [torch.arange(16)]}
    submodule.postprocess("request", None, padded, inputs=_geometry(frames=3, context=1))
    assert padded["audio_chunk"][0].tolist() == list(range(4, 12))


def _geometry(frames: int, context: int) -> ARNodeInputs:
    return ARNodeInputs(kwargs={"frames": frames, "context": context})


def test_qwen3_tts_codec_postprocess_uses_its_own_pass_geometry():
    """Speculative scheduling prepares a request's next window before the
    current one is postprocessed; the trim must follow the pass, not the
    request's latest state."""
    config = _tiny_model_config()   # upsample 4, windows [1, 4]
    submodule = CodecSubmodule(_FakeCodecDecoder(4), config)
    meta = lambda context: SimpleNamespace(  # noqa: E731
        request_id="request",
        step_metadata={"stream_chunks": {"codec_tokens": {"context_items": context, "is_final": False}}},
    )
    first = submodule.prepare_inputs("codec_chunk", meta(0), {"codec_tokens": [torch.ones(1, 4, dtype=torch.long)]})
    second = submodule.prepare_inputs("codec_chunk", meta(1), {"codec_tokens": [torch.ones(4, 4, dtype=torch.long)]})
    assert (first.kwargs, second.kwargs) == ({"frames": 1, "context": 0}, {"frames": 4, "context": 1})

    out_first = {"audio_chunk": [torch.arange(4)]}
    submodule.postprocess("request", None, out_first, inputs=first)
    assert out_first["audio_chunk"][0].tolist() == [0, 1, 2, 3]
    out_second = {"audio_chunk": [torch.arange(16)]}
    submodule.postprocess("request", None, out_second, inputs=second)
    assert out_second["audio_chunk"][0].tolist() == list(range(4, 16))


def test_qwen3_tts_codec_trims_reference_audio_from_clone_streams():
    config = _tiny_model_config()   # upsample 4 samples per frame, chunk 3, left context 2
    submodule = CodecSubmodule(_FakeCodecDecoder(4), config)
    codes = torch.ones(3, 4, dtype=torch.long)
    submodule.prepare_inputs(
        "codec_chunk_clone", SimpleNamespace(request_id="clone"),
        {"codec_tokens": [codes], "ref_frames": [torch.tensor([4])]},
    )
    state = submodule.request_state("clone")
    assert state["skip_samples"] == 16

    # First chunk: 3 frames = 12 samples, all reference -> nothing emitted.
    first = {"audio_chunk": [torch.arange(20)]}
    submodule.postprocess("clone", None, first, inputs=_geometry(frames=3, context=0))
    assert first["audio_chunk"][0].numel() == 0
    assert state["skip_samples"] == 4
    # Second chunk: 2 context + 3 new frames; 4 more samples belong to the reference.
    second = {"audio_chunk": [torch.arange(20)]}
    submodule.postprocess("clone", None, second, inputs=_geometry(frames=5, context=2))
    assert second["audio_chunk"][0].tolist() == list(range(12, 20))
    assert state["skip_samples"] == 0


def test_qwen3_tts_codec_filters_eos_and_pads_to_capture_shape():
    config = _tiny_model_config()
    submodule = CodecSubmodule(_FakeCodecDecoder(4), config)
    eos = config.talker.codec_eos_token_id
    codes = torch.tensor([
        [1, 2, 3, 4],
        [eos, 0, 0, 0],
        [5, 6, 7, 8],
    ])

    fwd_info = SimpleNamespace(
        request_id="request",
        step_metadata={"stream_chunks": {"codec_tokens": {
            "start_offset": 1, "context_items": 1, "num_items": 3, "is_final": False,
        }}},
    )
    prepared = submodule.prepare_inputs("codec_chunk", fwd_info, {"codec_tokens": [codes]})

    # Three items (one of them EOS) pad up to the smallest captured window (4).
    packed = prepared.tensor_inputs["codec_tokens"]
    assert packed.shape == (4, 4)
    assert packed[:, :2].t().tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert packed[:, 2:].count_nonzero().item() == 0
    assert prepared.kwargs == {"frames": 2, "context": 1}
    assert submodule.request_state("request")["codec_bucket"] == 4

    # A single frame lands in the first ramp bucket; too many frames is an error.
    one = submodule.prepare_inputs(
        "codec_chunk", SimpleNamespace(request_id="one"), {"codec_tokens": [codes[:1]]},
    )
    assert one.tensor_inputs["codec_tokens"].shape == (4, 1)
    with pytest.raises(ValueError, match="maximum is 4"):
        submodule.prepare_inputs(
            "codec_chunk", SimpleNamespace(request_id="big"),
            {"codec_tokens": [torch.ones(5, 4, dtype=torch.long)]},
        )


def test_qwen3_tts_streaming_policy_ramps_and_flushes_only_new_tail_audio():
    config = _tiny_model_config()
    stream = StreamBuffer(
        request_id="request",
        edge_name="codec_tokens",
        from_partition="Talker",
        policy=ScheduledLeftContextChunkPolicy(
            schedule=config.codec.chunk_schedule,
            chunk=config.codec.chunk_frames,
            left_context=config.codec.left_context_frames,
        ),
    )
    chunks = []
    for i in range(5):
        tensor_id = str(i)
        stream.pre_read_register(tensor_id)
        stream.put(tensor_id, torch.tensor([i]))
        while stream.has_chunk_ready():
            chunks.append(stream.pop_chunk())
    # First audio after a single frame, then 1 context + 3 new frames.
    assert [c.data["data"].flatten().tolist() for c in chunks] == [[0], [0, 1, 2, 3]]
    assert [c.context_items for c in chunks] == [0, 1]

    stream.signal_done()
    assert stream.has_chunk_ready()
    tail = stream.pop_chunk()
    assert tail.data["data"].flatten().tolist() == [2, 3, 4]
    assert tail.context_items == 2
    assert tail.is_final is True

    # The codec trims exactly the context frames the buffer reported (tail: 1 new frame).
    codec = CodecSubmodule(_FakeCodecDecoder(4), config)
    fwd_info = SimpleNamespace(
        request_id="request",
        step_metadata={"stream_chunks": {"codec_tokens": {
            "start_offset": tail.start_offset, "context_items": tail.context_items, "is_final": True,
        }}},
    )
    tail_codes = tail.data["data"].view(3, 1).expand(3, 4)
    prepared = codec.prepare_inputs("codec_chunk", fwd_info, {"codec_tokens": [tail_codes]})
    assert prepared.tensor_inputs["codec_tokens"].shape == (4, 4)   # 3 frames padded to the 4-frame bucket
    outputs = {"audio_chunk": [torch.arange(16)]}
    codec.postprocess("request", None, outputs, inputs=prepared)
    assert outputs["audio_chunk"][0].tolist() == list(range(8, 12))


def test_qwen3_tts_codec_batches_and_declares_cuda_graphs():
    config = _tiny_model_config()
    submodule = CodecSubmodule(_FakeCodecDecoder(4), config)
    model_inputs = [
        ARNodeInputs(tensor_inputs={
            "codec_tokens": torch.zeros(4, 4, dtype=torch.long)
        })
        for _ in range(2)
    ]
    batch = ExecutingBatch(
        node_name="Codec",
        step_context=_step_context("codec_chunk", ["a", "b"]),
        per_request_input_tensors={},
        per_request_info={},
    )

    assert submodule.can_batch(batch, model_inputs)
    assert submodule.can_use_cuda_graphs(batch, model_inputs)
    packed = submodule.preprocess(
        "codec_chunk",
        ModelInputsFromEngine(request_ids=["a", "b"], per_request_info={}),
        model_inputs,
    )
    assert packed["codec_tokens"].shape == (2, 4, 4)
    # One capture per window of the chunk ramp, keyed by the window, replayed
    # by both codec walks.
    graph_configs = submodule.get_cuda_graph_configs(torch.device("cpu"))
    assert [c.additional_key_info for c in graph_configs] == [1, 4]
    for graph_config in graph_configs:
        assert graph_config.capture_graph_walk == "codec_chunk"
        # CustomVoice has no clone walk; a Base config would add codec_chunk_clone.
        assert set(graph_config.replay_graph_walks) == {"codec_chunk"}
        assert graph_config.capture_batch_sizes == [1, 2, 4, 8, 16]
        assert graph_config.single_request_inputs.tensor_inputs["codec_tokens"].shape == (
            4, graph_config.additional_key_info,
        )
    base_config = _tiny_model_config()
    base_config.tts_model_type = "base"
    base_codec = CodecSubmodule(_FakeCodecDecoder(4), base_config)
    assert set(base_codec.get_cuda_graph_configs(torch.device("cpu"))[0].replay_graph_walks) == {
        "codec_chunk", "codec_chunk_clone",
    }
    assert submodule.max_batch_size("codec_chunk") == 16
    # The batch's capture key is the bucket its requests pad to: read off the
    # stream metadata when present (before prepare_inputs), else off the state.
    def meta(num_items):
        return SimpleNamespace(step_metadata={"stream_chunks": {"codec_tokens": {
            "num_items": num_items, "context_items": 0, "start_offset": 0, "is_final": False,
        }}})

    assert submodule.cg_key_info("codec_chunk", {"a": meta(3), "b": meta(4)}) == 4
    # Requests at different points of the ramp share a batch: the key (and the
    # padding in preprocess) is the widest window among them.
    assert submodule.cg_key_info("codec_chunk", {"a": meta(1), "b": meta(4)}) == 4
    for rid in ("a", "b"):
        submodule.request_state(rid).add("codec_bucket", 4)
    assert submodule.cg_key_info("codec_chunk", {"a": None, "b": None}) == 4
    submodule.request_state("b").add("codec_bucket", 1)
    assert submodule.cg_key_info("codec_chunk", {"a": None, "b": None}) == 4

    mixed = model_inputs + [ARNodeInputs(tensor_inputs={"codec_tokens": torch.ones(4, 1, dtype=torch.long)})]
    assert submodule.can_batch(batch, mixed)
    assert submodule.can_use_cuda_graphs(batch, mixed)
    packed = submodule.preprocess(
        "codec_chunk", ModelInputsFromEngine(request_ids=["a", "b", "c"], per_request_info={}), mixed,
    )
    assert packed["codec_tokens"].shape == (3, 4, 4)
    assert packed["codec_tokens"][2].tolist() == [[1, 0, 0, 0]] * 4   # 1-frame window padded on the right
    oversized = model_inputs * 9
    assert len(oversized) == 18
    assert not submodule.can_batch(batch, oversized)


def test_qwen3_tts_codec_single_forward_yields_one_request_samples():
    """The eager single-request path must hand postprocess a ``[samples]``
    tensor: slicing a ``[1, samples]`` batch on its first axis emitted empty
    chunks whenever a window carried context (the truncation seen at c=8)."""
    config = _tiny_model_config()   # upsample 4, windows [1, 4]
    submodule = CodecSubmodule(_FakeCodecDecoder(4), config)
    engine_inputs = ModelInputsFromEngine(request_ids=["r"], per_request_info={})
    window = ARNodeInputs(
        tensor_inputs={"codec_tokens": torch.ones(4, 4, dtype=torch.long)},
        kwargs={"frames": 4, "context": 2},
    )
    packed = submodule.preprocess("codec_chunk", engine_inputs, [window])
    out = submodule.forward("codec_chunk", engine_inputs, **packed)
    assert out["audio_chunk"][0].shape == (16,)
    submodule.postprocess("r", None, out, inputs=window)
    assert out["audio_chunk"][0].shape == (8,)   # frames 2..4 of 4

    # A batch-shaped chunk is flattened rather than sliced on the batch axis;
    # too few samples for the window is an error, never a silent cut.
    batched = {"audio_chunk": [torch.arange(16).view(1, 16)]}
    submodule.postprocess("r", None, batched, inputs=window)
    assert batched["audio_chunk"][0].tolist() == list(range(8, 16))
    with pytest.raises(ValueError, match="samples"):
        submodule.postprocess("r", None, {"audio_chunk": [torch.arange(8)]}, inputs=window)


class _FakeCodecEncoder(torch.nn.Module):
    """Stands in for the Mimi encoder: deterministic codes, one frame per 4 samples."""

    def __init__(self, num_quantizers: int):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.num_quantizers = num_quantizers
        self.calls = 0

    def encode(self, input_values, return_dict=True):
        del return_dict
        self.calls += 1
        frames = input_values.shape[-1] // 4 + 2   # the real encoder pads a little
        codes = torch.arange(frames).repeat(self.num_quantizers + 1, 1).unsqueeze(0)
        return SimpleNamespace(audio_codes=codes)


def test_qwen3_tts_ref_encoder_emits_xvector_and_reference_frames():
    from mstar.model.qwen3_tts.components.speaker_encoder import (
        Qwen3TTSMelFrontEnd,
        Qwen3TTSSpeakerEncoder,
    )
    from mstar.model.qwen3_tts.config import Qwen3TTSSpeakerEncoderConfig
    from mstar.model.qwen3_tts.submodules import RefEncoderSubmodule

    config = _tiny_model_config()
    speaker_config = Qwen3TTSSpeakerEncoderConfig(
        enc_dim=config.talker.hidden_size, enc_channels=(16, 16, 16, 16, 48),
        enc_se_channels=8, enc_attention_channels=8,
    )
    encoder = _FakeCodecEncoder(config.codec.num_quantizers)
    submodule = RefEncoderSubmodule(
        Qwen3TTSSpeakerEncoder(speaker_config), Qwen3TTSMelFrontEnd(speaker_config), encoder, config,
    )
    clip = torch.randn(2, 4000) * 0.1   # stereo, averaged to mono
    prepared = submodule.prepare_inputs(
        "talker_prefill_clone", SimpleNamespace(request_id="clone"),
        {"audio_inputs": [clip], "prompt_layout": [torch.tensor([0, 2, 1, 5, 7])]},
    )
    assert prepared.tensor_inputs["waveform"].shape == (4000,)
    assert prepared.kwargs["ref_frames"] == 7
    engine_inputs = ModelInputsFromEngine(request_ids=["clone"], per_request_info={})
    out = submodule.forward("talker_prefill_clone", engine_inputs, **submodule.preprocess(
        "talker_prefill_clone", engine_inputs, [prepared]))
    assert out["speaker_embed"][0].shape == (config.talker.hidden_size,)
    assert out["ref_codes"][0].shape == (7, config.codec.num_quantizers)
    assert out["ref_codes"][0][:, 0].tolist() == list(range(7))
    assert encoder.calls == 1

    # x-vector only: the codec encoder is skipped and a placeholder frame rides the edge.
    prepared = submodule.prepare_inputs(
        "talker_prefill_clone", SimpleNamespace(request_id="xvec"),
        {"audio_inputs": [clip[0]], "prompt_layout": [torch.tensor([0, 2, 1, 0, 0])]},
    )
    out = submodule.forward("talker_prefill_clone", engine_inputs, **submodule.preprocess(
        "talker_prefill_clone", engine_inputs, [prepared]))
    assert out["ref_codes"][0].shape == (1, config.codec.num_quantizers)
    assert encoder.calls == 1


def test_qwen3_tts_ref_encoder_memoises_conditioning_by_clip_content():
    from mstar.model.qwen3_tts.components.speaker_encoder import (
        Qwen3TTSMelFrontEnd,
        Qwen3TTSSpeakerEncoder,
    )
    from mstar.model.qwen3_tts.config import Qwen3TTSSpeakerEncoderConfig
    from mstar.model.qwen3_tts.submodules import RefEncoderSubmodule

    config = _tiny_model_config()
    speaker_config = Qwen3TTSSpeakerEncoderConfig(
        enc_dim=config.talker.hidden_size, enc_channels=(16, 16, 16, 16, 48),
        enc_se_channels=8, enc_attention_channels=8,
    )
    encoder = _FakeCodecEncoder(config.codec.num_quantizers)
    submodule = RefEncoderSubmodule(
        Qwen3TTSSpeakerEncoder(speaker_config), Qwen3TTSMelFrontEnd(speaker_config), encoder, config,
    )
    engine_inputs = ModelInputsFromEngine(request_ids=["clone"], per_request_info={})

    def encode(rid: str, clip: torch.Tensor, ref_frames: int):
        prepared = submodule.prepare_inputs(
            "talker_prefill_clone", SimpleNamespace(request_id=rid),
            {"audio_inputs": [clip], "prompt_layout": [torch.tensor([0, 2, 1, ref_frames, ref_frames])]},
        )
        return submodule.forward("talker_prefill_clone", engine_inputs, **submodule.preprocess(
            "talker_prefill_clone", engine_inputs, [prepared]))

    clip_a = torch.randn(4000) * 0.1
    first = encode("a1", clip_a, 7)
    assert encoder.calls == 1
    # The same samples again (a fresh tensor, as an upload produces) cost no encoder pass
    # and yield the same conditioning.
    again = encode("a2", clip_a.clone(), 7)
    assert encoder.calls == 1
    assert again["speaker_embed"][0] is first["speaker_embed"][0]
    assert torch.equal(again["ref_codes"][0], first["ref_codes"][0])
    # Other content, or the same clip used x-vector-only, is a different entry.
    encode("b1", torch.randn(4000) * 0.1, 7)
    assert encoder.calls == 2
    xvec = encode("a3", clip_a, 0)
    assert encoder.calls == 2 and xvec["ref_codes"][0].shape == (1, config.codec.num_quantizers)
    assert len(submodule._conditioning) == 3
    # The memo is bounded, oldest first.
    submodule.CONDITIONING_CACHE_SIZE = 2
    encode("c1", torch.randn(4000) * 0.1, 7)
    assert encoder.calls == 3 and len(submodule._conditioning) == 2
    encode("a4", clip_a, 7)   # evicted, so encoded again
    assert encoder.calls == 4
