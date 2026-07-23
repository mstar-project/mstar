import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.engine.base import EngineType, NodeBatch
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
    assert model.get_node_engine_types() == {
        "Talker": EngineType.KV_CACHE,
        "Codec": EngineType.STATELESS,
    }
    kv_configs = model.get_kv_cache_config()
    assert len(kv_configs) == 1
    kv = kv_configs[0]
    assert kv.nodes == ["Talker"]
    assert kv.num_layers == model.config.talker.num_hidden_layers
    assert kv.num_kv_heads == model.config.talker.num_key_value_heads
    assert kv.num_qo_heads == model.config.talker.num_attention_heads
    assert kv.head_dim == model.config.talker.head_dim
    assert kv.flashinfer_backend == "fa2"

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
    assert talker.step_metadata["subtalker_sampling"]["top_k"] == 7

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
            "subtalker_sampling": {},
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


def test_qwen3_tts_suppresses_eos_for_official_minimum_frames():
    config = _tiny_model_config()
    submodule = TalkerSubmodule(
        Qwen3TTSTalkerModel(config),
        Qwen3TTSCodePredictor(config),
        config,
    )
    submodule.request_state("new").add("generated_frames", 0)
    submodule.request_state("ready").add(
        "generated_frames", config.generation.min_new_tokens
    )

    mask = submodule._get_batch_suppress_mask(["new", "ready"])
    eos = config.talker.codec_eos_token_id

    assert mask.shape == (2, config.talker.vocab_size)
    assert mask[0, eos].item() is True
    assert mask[1, eos].item() is False


def test_qwen3_tts_talker_batches_and_captures_decode():
    config = _tiny_model_config()
    submodule = TalkerSubmodule(
        Qwen3TTSTalkerModel(config),
        Qwen3TTSCodePredictor(config),
        config,
    )
    expected_sampling = {
        "do_sample": config.generation.subtalker_dosample,
        "temperature": config.generation.subtalker_temperature,
        "top_k": config.generation.subtalker_top_k,
        "top_p": config.generation.subtalker_top_p,
    }
    info = {
        request_id: SimpleNamespace(
            step_metadata={"subtalker_sampling": expected_sampling.copy()}
        )
        for request_id in ("a", "b")
    }
    batch = NodeBatch(
        node_name="Talker",
        graph_walk="talker_decode",
        request_ids=["a", "b"],
        per_request_input_tensors={},
        per_request_info=info,
    )
    model_inputs = [
        ARNodeInputs(input_embeds=torch.zeros(1, 16), input_seq_len=1)
        for _ in range(2)
    ]

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
    piecewise = submodule.get_piecewise_cuda_graph_configs(
        torch.device("cpu"), torch.float32
    )["code_predictor_loop"]
    assert piecewise.seq_len == 1
    assert piecewise.uses_kv_cache is False
    assert piecewise.capture_batch_sizes == [1, 2, 4, 8, 16, 32]
    capture_shape = piecewise.get_capture_shapes([2])[0]
    static_inputs = piecewise.make_static_inputs(capture_shape)
    assert static_inputs["last_hidden"].shape == (2, 16)
    assert static_inputs["uniforms"].shape == (2, 3)

    info["b"].step_metadata["subtalker_sampling"]["temperature"] = 0.7
    assert not submodule.can_batch(batch, model_inputs)
    assert not submodule.can_use_cuda_graphs(batch, model_inputs)


def test_qwen3_tts_piecewise_sampling_is_tensor_only():
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    tokens = TalkerSubmodule._sample_from_uniform(
        logits=logits,
        uniform=torch.tensor([0.3, 0.7]),
        temperature=torch.ones(2),
        top_k=torch.tensor([1, 1]),
        top_p=torch.ones(2),
        do_sample=torch.tensor([True, False]),
    )
    assert tokens.tolist() == [2, 0]


def test_qwen3_tts_uses_piecewise_runner_when_available():
    config = _tiny_model_config()
    submodule = TalkerSubmodule(
        Qwen3TTSTalkerModel(config),
        Qwen3TTSCodePredictor(config),
        config,
    )

    class _Sampler:
        @staticmethod
        def _broadcast_tokens(tensor):
            return tensor

    class _Runner:
        called = False

        @staticmethod
        def can_run(batch_size):
            return batch_size == 1

        def run(self, static_inputs, real_bs):
            self.called = True
            assert real_bs == 1
            assert static_inputs["uniforms"].shape == (1, 3)
            return {
                "all_codes": torch.tensor([[1, 2, 3, 4]]),
                "codec_embed_sum": torch.zeros(1, 16),
            }

    runner = _Runner()
    result = submodule._run_code_predictor_piecewise(
        engine_inputs=ModelInputsFromEngine(
            request_ids=["request"],
            per_request_info={
                "request": SimpleNamespace(random_seed=123)
            },
            sampler=_Sampler(),
            piecewise_runners={"code_predictor_loop": runner},
        ),
        last_hidden=torch.zeros(1, 16),
        layer0_codes=torch.tensor([1]),
        sampling={
            "do_sample": True,
            "temperature": 0.9,
            "top_k": 10,
            "top_p": 0.8,
        },
    )

    assert runner.called
    assert result is not None
    assert result[0].tolist() == [[1, 2, 3, 4]]
    assert result[1].shape == (1, 16)


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
    batch = NodeBatch(
        node_name="Codec",
        graph_walk="codec_chunk",
        request_ids=["a", "b"],
        per_request_input_tensors={},
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
    assert graph_config.capture_batch_sizes == [1, 2, 4, 8, 16]
    assert graph_config.single_request_inputs.tensor_inputs[
        "codec_tokens"
    ].shape == (4, 5)
