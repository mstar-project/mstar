"""Parakeet-TDT and Nemotron ASR on tiny fake snapshots: config parsing, the
language prompt, request limits, the graph and the adapter. Transcripts are
checked against HF on a GPU with real weights."""

import json

import pytest
import torch

from mstar.api_server.openai.adapters import TransducerAdapter
from mstar.api_server.openai.protocol import TranscriptionRequest
from mstar.engine.resources.recurrent import RecurrentStateSpec
from mstar.graph.base import GraphNode, Loop, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.parakeet.config import (
    DECODE_LOOP,
    DECODE_WALK,
    NUM_FRAMES,
    PREFILL_WALK,
    TRANSDUCER_STATE,
    ParakeetModelConfig,
)
from mstar.model.parakeet.parakeet_model import ParakeetModel

ENCODER = dict(
    hidden_size=32, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64, conv_kernel_size=9,
    num_mel_bins=128, subsampling_factor=8, subsampling_conv_channels=8, subsampling_conv_kernel_size=3,
    subsampling_conv_stride=2, attention_bias=False, convolution_bias=False, scale_input=False,
)


def _snapshot(path, nemotron: bool):
    vocab = {"<unk>": 0, "▁hi": 1, "!": 2}
    tokenizer = {
        "model": {"type": "BPE", "vocab": vocab, "merges": [], "byte_fallback": True},
        "added_tokens": [{"id": 0, "content": "<unk>", "special": True}],
    }
    if nemotron:
        encoder = {**ENCODER, "model_type": "nemotron_asr_streaming_encoder", "sliding_window": 57,
                   "default_num_lookahead_tokens": 3, "supported_num_lookahead_tokens": [3, 0, 6, 13]}
        config = {"model_type": "nemotron3_5_asr", "vocab_size": 4, "blank_token_id": 3, "decoder_hidden_size": 16,
                  "num_decoder_layers": 1, "num_prompts": 8, "prompt_intermediate_size": 16,
                  "default_prompt_id": 7, "durations": [], "encoder_config": encoder}
        processor = {"prompt_dictionary": {"en": 0, "en-US": 0, "de": 2, "de-DE": 2, "ja-JA": 5, "ja-JP": 5,
                                           "zh-CN": 3, "zh-TW": 4, "auto": 7}}
    else:
        encoder = {**ENCODER, "model_type": "parakeet_encoder"}
        config = {"model_type": "parakeet_tdt", "vocab_size": 4, "blank_token_id": 3, "decoder_hidden_size": 16,
                  "num_decoder_layers": 1, "durations": [0, 1, 2, 3, 4], "encoder_config": encoder}
        processor = {}
    (path / "config.json").write_text(json.dumps(config))
    (path / "generation_config.json").write_text(json.dumps({"decoder_start_token_id": 3}))
    (path / "processor_config.json").write_text(json.dumps(processor))
    (path / "tokenizer.json").write_text(json.dumps(tokenizer))
    return path


def test_parakeet_tdt_config(tmp_path):
    cfg = ParakeetModelConfig.from_pretrained(_snapshot(tmp_path, nemotron=False))
    assert cfg.transducer.durations == [0, 1, 2, 3, 4] and cfg.transducer.start_token_id == 3
    assert cfg.normalize_features and not cfg.encoder.causal and not cfg.has_prompt


def test_nemotron_config_and_prompt(tmp_path):
    snap = _snapshot(tmp_path, nemotron=True)
    cfg = ParakeetModelConfig.from_pretrained(snap)
    # whole clips: the widest lookahead the checkpoint supports
    assert cfg.encoder.causal and cfg.encoder.left_context == 56 and cfg.encoder.lookahead == 13
    assert not cfg.normalize_features and cfg.has_prompt
    assert cfg.prompt_id(None) == cfg.prompt_id("auto") == cfg.prompt_id("AUTO") == 7
    assert cfg.prompt_id("de-DE") == cfg.prompt_id("de_de") == cfg.prompt_id("DE") == 2
    # a bare code finds its locale when they agree on one slot, and a locale
    # the table lacks finds its language
    assert cfg.prompt_id("ja") == cfg.prompt_id("ja-jp") == 5
    assert cfg.prompt_id("de-AT") == 2 and cfg.prompt_id("en-AU") == 0
    for bad in ("tlh", "zh", "zh-HK", 5):  # zh: two locales, two slots
        with pytest.raises(ValueError):
            cfg.prompt_id(bad)
    assert ParakeetModelConfig.from_pretrained(snap, lookahead=3).encoder.lookahead == 3
    with pytest.raises(ValueError):
        ParakeetModelConfig.from_pretrained(snap, lookahead=5)


def test_unsupported_checkpoint(tmp_path):
    snap = _snapshot(tmp_path, nemotron=False)
    config = json.loads((snap / "config.json").read_text())
    (snap / "config.json").write_text(json.dumps({**config, "model_type": "parakeet_ctc"}))
    with pytest.raises(ValueError):
        ParakeetModelConfig.from_pretrained(snap)


@pytest.mark.parametrize("nemotron", [False, True])
def test_graph_and_process_prompt(tmp_path, nemotron):
    model = ParakeetModel(str(_snapshot(tmp_path, nemotron)))
    assert model.nodes == ["audio_encoder", "transducer"]
    (pool,) = model.get_node_resources()
    assert isinstance(pool, RecurrentStateSpec) and pool.resource_key == TRANSDUCER_STATE
    assert pool.nodes == {"transducer"} and pool.config.usable_slots == model.MAX_CONCURRENT_REQUESTS
    assert pool.config.blocks.keys() == {"h", "c", "dec_out", "cursor", "symbols"}
    out = model.process_prompt(None, ["audio"], ["text"], tensors={"audio_inputs": [torch.zeros(8000)]},
                               language="de")
    tensors = out.new_input_tensors
    assert tensors["audio"][0].shape == (8000,)
    # the encoder frames of 50 mel frames, as the encoder node will count them
    assert out.metadata == {NUM_FRAMES: model.config.encoder.output_lengths(8000 // 160)}
    assert NUM_FRAMES in model.INTERNAL_KWARGS
    assert ("prompt_ids" in tensors) == nemotron
    if nemotron:
        assert tensors["prompt_ids"][0].tolist() == [2]
    with pytest.raises(ValueError):
        model.process_prompt(None, ["audio"], ["text"], tensors={"audio_inputs": [torch.zeros(611 * 16_000)]})
    # Parakeet-TDT refuses a language it can't detect, Nemotron one its prompt
    # table lacks, and both a language that isn't a string (/generate's JSON)
    for bad in ("tlh", 5, 0, False):
        with pytest.raises(ValueError):
            model.process_prompt(None, ["audio"], ["text"], tensors={"audio_inputs": [torch.zeros(8000)]},
                                 language=bad)
    # the blank and special pieces render as nothing
    assert model.postprocess(torch.tensor([1, 2, 3, 0]), "text") == b" hi!"


@pytest.mark.parametrize("detects_language", [True, False], ids=["parakeet", "nemotron"])
def test_adapter(detects_language):
    adapter = TransducerAdapter(detects_language=detects_language)
    # greedy decoding: a retry at a higher temperature would repeat the text
    assert adapter.compression_ratio_threshold is None
    args = adapter.transcription_to_request(TranscriptionRequest(model="parakeet", language="en", prompt="x"), "a")
    assert args.model_kwargs == {"language": "en", "temperature": 0.0}
    with pytest.raises(ValueError):
        adapter.transcription_to_request(TranscriptionRequest(model="parakeet", timestamp_granularities=["word"]), "")
    # Whisper's task: a translation would come back as a same-language transcript
    with pytest.raises(ValueError, match="only transcribes"):
        adapter.transcription_to_request(TranscriptionRequest(model="parakeet", task="translate"), "")
    mk = adapter.transcription_to_request(TranscriptionRequest(model="parakeet", task="transcribe"), "").model_kwargs
    assert "task" not in mk
    # streamed text drops the space SentencePiece starts it with, as the final text does
    assert adapter.stream_text(" Hello") == "Hello" and adapter.stream_text(" ") == ""
    # Parakeet's languages are known up front, so a stream is refused before it starts;
    # Nemotron's locales come from the checkpoint and are checked at preprocessing
    for language in ("de-AT", "PT_br", "auto"):
        adapter.transcription_to_request(TranscriptionRequest(model="parakeet", language=language), "a")
    unknown = TranscriptionRequest(model="parakeet", language="ja")
    if detects_language:
        with pytest.raises(ValueError, match="unsupported language 'ja'"):
            adapter.transcription_to_request(unknown, "a")
    else:
        assert adapter.transcription_to_request(unknown, "a").model_kwargs["language"] == "ja"


def _pointer(uuid: int) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=[7, 16], dtype=torch.float32, nbytes=448, address=0, stride=[16, 1], uuid=uuid,
        source_session_id="host:0", source_entity="worker",
    )


@pytest.mark.parametrize("nemotron", [False, True])
def test_prefill_then_a_decode_loop(tmp_path, nemotron):
    """prefill persists the projected frames; decode is a loop over the
    transducer that the Loop re-injects them into, with a self loop-back
    clock that makes the node a speculation target; then the request is done."""
    model = ParakeetModel(str(_snapshot(tmp_path, nemotron)))
    walks = model.get_graph_walk_graphs()
    assert walks.keys() == {PREFILL_WALK, DECODE_WALK}
    transducer = walks[PREFILL_WALK].get_nodes()["transducer"]
    assert transducer.input_names == ({"encoder_states", "prompt_ids"} if nemotron else {"encoder_states"})
    [frames] = transducer.outputs
    assert (frames.name, frames.next_node, frames.persist) == ("frames", EMPTY_DESTINATION, True)

    loop = walks[DECODE_WALK]
    assert isinstance(loop, Loop) and loop.name == DECODE_LOOP and loop.outputs == []
    node = loop.section
    assert isinstance(node, GraphNode) and node.name == "transducer" and node.enable_async_scheduling
    assert node.input_names == {"frames", "clock"}
    assert {(e.name, e.next_node) for e in node.outputs} == {("clock", "transducer"), ("tokens", EMIT_TO_CLIENT)}
    # the frames come from outside the loop, so it saves and re-injects them; the clock loops back
    assert loop._external_inputs == {("frames", "transducer")}
    assert loop._loop_back_inputs == {("clock", "transducer")}
    # a ceiling only: the clip's own cap, and its stop check, end the loop first
    assert loop.max_iters == model.config.transducer.decode_iterations(
        model.config.encoder.output_lengths(610 * 16_000 // 160))
    assert model.get_loop_caps(**{NUM_FRAMES: 7}) == {DECODE_LOOP: model.config.transducer.decode_iterations(7)}

    audio, prompt = _pointer(1), _pointer(2)
    first = model.get_initial_forward_pass_args(
        "default", ["audio"], ["text"], {"audio": [audio], "prompt_ids": [prompt]}, model_kwargs={NUM_FRAMES: 7},
    )
    assert first.full_metadata.graph_walk == PREFILL_WALK and first.full_metadata.is_prefill
    assert [e.next_node for e in first.inputs] == ["audio_encoder"] + (["transducer"] if nemotron else [])
    persisted = _pointer(3)
    decode = model.get_partition_forward_pass_args("default", first.full_metadata, {"frames": [persisted]})
    assert decode.full_metadata.graph_walk == DECODE_WALK and not decode.full_metadata.is_prefill
    assert not decode.request_done
    edges = {e.name: e for e in decode.inputs}
    assert edges.keys() == {"frames", "clock"} and all(e.next_node == "transducer" for e in decode.inputs)
    assert edges["frames"].tensor_info == [persisted] and edges["clock"].tensor_info == []
    # the conductor lets go of the frames; the loop holds its own reference
    assert decode.unpersist_tensors == [persisted]
    assert decode.step_metadata == {NUM_FRAMES: 7}
    done = model.get_partition_forward_pass_args("default", decode.full_metadata, {})
    assert done.request_done and done.inputs == []


def test_startup_refuses_more_requests_than_slots(tmp_path):
    model = ParakeetModel(str(_snapshot(tmp_path, nemotron=False)))
    model.validate_config_yaml({"max_concurrent_requests": 128}, "a.yaml")
    model.validate_config_yaml(
        {"max_concurrent_requests": 128, "resources": {TRANSDUCER_STATE: {"max_slots": 129}}}, "a.yaml",
    )
    with pytest.raises(ValueError, match="need 129 slots"):
        model.validate_config_yaml(
            {"max_concurrent_requests": 128, "resources": {TRANSDUCER_STATE: {"max_slots": 128}}}, "a.yaml",
        )
    with pytest.raises(ValueError, match="need 130 slots"):
        model.validate_config_yaml({"max_concurrent_requests": 129}, "a.yaml")
    with pytest.raises(ValueError, match="positive int"):
        model.validate_config_yaml({"max_concurrent_requests": 0}, "a.yaml")
    # slot 0 is the sink, so one slot leaves none to admit into, with or without a cap
    with pytest.raises(ValueError, match="max_slots must be an int >= 2"):
        model.validate_config_yaml({"resources": {TRANSDUCER_STATE: {"max_slots": 1}}}, "a.yaml")
