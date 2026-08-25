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
from mstar.engine.resources import StepContext
from mstar.model.qwen3_tts.components.talker import (
    Qwen3TTSCodePredictor,
    Qwen3TTSTalkerModel,
)
from mstar.model.qwen3_tts.config import (
    Qwen3TTSCodecConfig,
    Qwen3TTSCodePredictorConfig,
    Qwen3TTSModelConfig,
    Qwen3TTSTalkerConfig,
)
from mstar.model.qwen3_tts.qwen3_tts_model import Qwen3TTSModel
from mstar.model.qwen3_tts.submodules import CodecSubmodule, TalkerSubmodule
from mstar.model.registry import HF_MODELS, MODEL_REGISTRY
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine
from mstar.streaming.chunk_policy import LeftContextChunkPolicy
from mstar.streaming.stream_buffer import StreamBuffer
from mstar.utils.flashinfer_utils import (
    FlashInferDecodeWrapper,
    FlashInferPrefillWrapper,
)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "qwen3tts.yaml"


class _TokenizerStub:
    def __init__(self):
        self.last_text = None

    def __call__(self, text, **kwargs):
        self.last_text = text
        assert kwargs == {"return_tensors": "pt", "padding": True}
        return {"input_ids": torch.tensor([[1, 2, 3]])}


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

    assert MODEL_REGISTRY["qwen3_tts"] is Qwen3TTSModel
    assert HF_MODELS["qwen3_tts"] == {
        "model_path_hf": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    }
    kv_configs = model.get_kv_cache_config()
    assert len(kv_configs) == 1
    kv = kv_configs[0]
    assert kv.nodes == ["Talker"]
    assert kv.num_layers == model.config.talker.num_hidden_layers
    assert kv.num_kv_heads == model.config.talker.num_key_value_heads
    assert kv.num_qo_heads == model.config.talker.num_attention_heads
    assert kv.head_dim == model.config.talker.head_dim
    assert kv.flashinfer_backend == "auto"
    serving_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert serving_config["kv_cache"]["flashinfer_backend"] == "fa2"

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
from mstar.model.qwen3_tts.qwen3_tts_model import _load_qwen3_tts_decoder_classes
config_cls, decoder_cls = _load_qwen3_tts_decoder_classes()
print(config_cls.__name__, decoder_cls.__name__)
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
    assert "Qwen3TTSTokenizerV2DecoderConfig Qwen3TTSTokenizerV2Decoder" in (
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
    assert tensors["text_inputs"][0].tolist() == [1, 2, 3]
    assert tensors["speaker_id"][0].item() == 3065
    assert tensors["language_id"][0].item() == 2055


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
        ("hello", ["audio"], ["audio"], {}, "text input only"),
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
        for name in ("text_inputs", "speaker_id", "language_id")
    }

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
    # Residual-group sampling rides the aux sampling config, not step metadata.
    assert "subtalker_sampling" not in talker.step_metadata
    aux = model.get_aux_sampling_configs("Talker", {"subtalker_top_k": 7})
    assert aux["code_predictor"].top_k == 7

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
        num_attention_heads=2,
        num_key_value_heads=1,
        hidden_size=16,
        intermediate_size=32,
        head_dim=8,
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

    embeds = submodule._build_prefill(
        request_id="request",
        text_ids=torch.arange(1, 13),
        speaker_id=40,
        language_id=-1,
    )

    assert embeds.shape == (9, 16)
    state = submodule.request_state("request")
    assert state["trailing_text_hidden"].shape == (4, 16)
    assert state["tts_pad_embed"].shape == (16,)
    assert state["generation_step"] == 0


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
        sampling_config={
            "Talker": SimpleNamespace(ignore_eos=False),
        },
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
        sampling_config={
            "Talker": SimpleNamespace(ignore_eos=True),
        },
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
    mask = submodule._get_batch_suppress_mask(inputs)
    eos = config.talker.codec_eos_token_id

    assert mask.shape == (2, config.talker.vocab_size)
    assert mask[0, eos].item() is True
    assert mask[1, eos].item() is False


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
    cache_manager = SimpleNamespace(
        set_active_label=lambda label: None,
        plan_attention=lambda **kwargs: None,
        plan_rope=lambda **kwargs: None,
    )

    packed = submodule.preprocess(
        "talker_decode",
        ModelInputsFromEngine(
            request_ids=["__graph_dummy__"],
            per_request_info={},
            cache_manager=cache_manager,
        ),
        [prepared],
    )

    eos = config.talker.codec_eos_token_id
    assert prepared.tensor_inputs["suppress_eos"].item() is False
    assert packed["suppress_mask"][0, eos].item() is False
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
        request_ids=["a", "b"],
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
    cache_manager = SimpleNamespace(
        set_active_label=lambda label: None,
        plan_attention=lambda **kwargs: None,
        plan_rope=lambda **kwargs: None,
    )
    packed = submodule.preprocess(
        "talker_decode",
        ModelInputsFromEngine(
            request_ids=["a", "b"],
            per_request_info=info,
            cache_manager=cache_manager,
        ),
        model_inputs,
    )
    assert packed["input_embeds"].shape == (2, 16)
    assert packed["last_token_indices"].tolist() == [0, 1]
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


def test_qwen3_tts_code_predictor_uses_native_gqa(monkeypatch):
    config = _tiny_model_config()
    predictor = Qwen3TTSCodePredictor(config)
    # This CPU-only contract test targets the SDPA call. FlashInfer RMSNorm is
    # CUDA-only, so replace normalization without changing attention geometry.
    for layer in predictor.model.layers:
        layer.input_layernorm = torch.nn.Identity()
        layer.post_attention_layernorm = torch.nn.Identity()
        layer.self_attn.q_norm = torch.nn.Identity()
        layer.self_attn.k_norm = torch.nn.Identity()
    predictor.model.norm = torch.nn.Identity()
    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    calls = []

    def capture_sdpa(query, key, value, **kwargs):
        calls.append((query.shape, key.shape, value.shape, kwargs))
        return original_sdpa(query, key, value, **kwargs)

    monkeypatch.setattr(
        torch.nn.functional,
        "scaled_dot_product_attention",
        capture_sdpa,
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

    query_shape, key_shape, value_shape, kwargs = calls[0]
    assert output.shape == (1, 1, config.talker.hidden_size)
    assert query_shape[1] == config.talker.code_predictor.num_attention_heads
    assert key_shape[1] == config.talker.code_predictor.num_key_value_heads
    assert value_shape[1] == config.talker.code_predictor.num_key_value_heads
    assert kwargs["enable_gqa"] is True


def test_qwen3_tts_aux_sampling_config_drives_code_predictor():
    """Residual groups are configured through the ``code_predictor`` aux config,
    which the engine turns into its own per-request sampler buffers."""
    model = _make_model()
    generation = model.config.generation

    default = model.get_aux_sampling_configs("Talker")["code_predictor"]
    assert default.temperature == generation.subtalker_temperature
    assert default.top_k == generation.subtalker_top_k
    assert default.top_p == generation.subtalker_top_p
    # No penalty on the depth loop => no seen-token mask buffers for this label.
    assert default.vocab_size is None

    overridden = model.get_aux_sampling_configs(
        "Talker",
        {"subtalker_temperature": 0.5, "subtalker_top_k": 3, "subtalker_top_p": 0.25},
    )["code_predictor"]
    assert (overridden.temperature, overridden.top_k, overridden.top_p) == (0.5, 3, 0.25)

    # do_sample=False is expressed as temperature 0 (encoded as greedy downstream).
    greedy = model.get_aux_sampling_configs(
        "Talker", {"subtalker_dosample": False}
    )["code_predictor"]
    assert greedy.temperature == 0.0

    # Only the Talker owns an aux sampler.
    assert model.get_aux_sampling_configs("Codec") == {}

    # The bundle the engines actually consume.
    multi = model.resolve_sampling_configs("Talker", {})
    assert multi.main.vocab_size is not None
    assert set(multi.aux) == {"code_predictor"}
    multi.set_seed(99)
    assert multi.main.seed == 99
    # Aux stream is seeded independently of the Talker's.
    assert multi.aux["code_predictor"].seed != 99


class _FakeCodecDecoder(torch.nn.Module):
    def __init__(self, upsample: int):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.upsample = upsample

    def forward(self, codes):
        length = codes.shape[-1] * self.upsample
        return torch.zeros(codes.shape[0], 1, length, dtype=torch.float32)


def test_qwen3_tts_codec_trims_overlap_after_first_chunk():
    config = _tiny_model_config()
    submodule = CodecSubmodule(_FakeCodecDecoder(4), config)
    state = submodule.request_state("request")
    state.add("latest_codec_frames", 5)

    first = {"audio_chunk": [torch.arange(20)]}
    submodule.postprocess("request", None, first)
    assert first["audio_chunk"][0].tolist() == list(range(20))

    second = {"audio_chunk": [torch.arange(20)]}
    submodule.postprocess("request", None, second)
    assert second["audio_chunk"][0].tolist() == list(range(8, 20))


def test_qwen3_tts_codec_filters_eos_and_pads_to_capture_shape():
    config = _tiny_model_config()
    submodule = CodecSubmodule(_FakeCodecDecoder(4), config)
    eos = config.talker.codec_eos_token_id
    codes = torch.tensor([
        [1, 2, 3, 4],
        [eos, 0, 0, 0],
        [5, 6, 7, 8],
    ])

    prepared = submodule.prepare_inputs(
        "codec_chunk",
        SimpleNamespace(request_id="request"),
        {"codec_tokens": [codes]},
    )

    packed = prepared.tensor_inputs["codec_tokens"]
    assert packed.shape == (4, 5)
    assert packed[:, :2].t().tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert packed[:, 2:].count_nonzero().item() == 0
    assert submodule.request_state("request")["latest_codec_frames"] == 2


def test_qwen3_tts_streaming_policy_flushes_only_new_tail_audio():
    config = _tiny_model_config()
    stream = StreamBuffer(
        request_id="request",
        edge_name="codec_tokens",
        from_partition="Talker",
        policy=LeftContextChunkPolicy(
            chunk=config.codec.chunk_frames,
            left_context=config.codec.left_context_frames,
        ),
    )
    for i in range(5):
        tensor_id = str(i)
        stream.pre_read_register(tensor_id)
        stream.put(tensor_id, torch.tensor([i]))
        if i == 2:
            first = stream.pop_chunk()
            assert first.data["data"].flatten().tolist() == [0, 1, 2]

    stream.signal_done()
    assert stream.has_chunk_ready()
    tail = stream.pop_chunk()
    assert tail.data["data"].flatten().tolist() == [1, 2, 3, 4]
    assert tail.is_final is True

    codec = CodecSubmodule(_FakeCodecDecoder(4), config)
    state = codec.request_state("request")
    state.add_all(latest_codec_frames=4, codec_chunk_emitted=True)
    outputs = {"audio_chunk": [torch.arange(16)]}
    codec.postprocess("request", None, outputs)
    assert outputs["audio_chunk"][0].tolist() == list(range(8, 16))


def test_qwen3_tts_codec_batches_and_declares_cuda_graphs():
    config = _tiny_model_config()
    submodule = CodecSubmodule(_FakeCodecDecoder(4), config)
    model_inputs = [
        ARNodeInputs(tensor_inputs={
            "codec_tokens": torch.zeros(4, 5, dtype=torch.long)
        })
        for _ in range(2)
    ]
    batch = ExecutingBatch(
        node_name="Codec",
        step_context=_step_context("codec_chunk", ["a", "b"]),
        request_ids=["a", "b"],
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
    assert packed["codec_tokens"].shape == (2, 4, 5)
    graph_config = submodule.get_cuda_graph_configs(torch.device("cpu"))[0]
    assert graph_config.capture_graph_walk == "codec_chunk"
    assert submodule.max_batch_size("codec_chunk") == 8
    assert graph_config.capture_batch_sizes == [1, 2, 4, 8]
    assert graph_config.single_request_inputs.tensor_inputs[
        "codec_tokens"
    ].shape == (4, 5)

    oversized = model_inputs * 5
    assert len(oversized) == 10
    assert not submodule.can_batch(batch, oversized)
