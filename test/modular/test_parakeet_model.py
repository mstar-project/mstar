"""Parakeet-TDT and Nemotron ASR on tiny fake snapshots: config parsing, the
language prompt, request limits, the graph and the adapter. Transcripts are
checked against HF on a GPU with real weights."""

import json

import pytest
import torch

from mstar.api_server.openai.adapters import TransducerAdapter
from mstar.api_server.openai.protocol import TranscriptionRequest
from mstar.engine.resources.recurrent import RecurrentStateSpec
from mstar.graph.base import GraphNode, Loop, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.parakeet.config import (
    DECODE_LOOP,
    NUM_FRAMES,
    PROMPT_ID,
    TRANSDUCER_STATE,
    WALK,
    ParakeetModelConfig,
)
from mstar.model.parakeet.parakeet_model import ParakeetModel
from mstar.validation.resources import pool_problems

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
    assert tensors.keys() == {"audio"} and tensors["audio"][0].shape == (8000,)
    # the encoder frames of 50 mel frames, as the encoder node will count them, and
    # Nemotron's prompt, both host values for the decode steps
    frames = model.config.encoder.output_lengths(8000 // 160)
    assert out.metadata == ({NUM_FRAMES: frames, PROMPT_ID: 2} if nemotron else {NUM_FRAMES: frames})
    assert set(model.INTERNAL_KWARGS) == {NUM_FRAMES, PROMPT_ID}
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
def test_one_walk_the_encoder_then_a_decode_loop(tmp_path, nemotron):
    """One walk: the encoder, then a loop over the transducer that re-injects
    the encoder output every iteration. No loop-back edge, so the worker
    speculates a request's first iteration but no next one; then the request
    is done."""
    model = ParakeetModel(str(_snapshot(tmp_path, nemotron)))
    walks = model.get_graph_walk_graphs()
    assert walks.keys() == {WALK}
    walk = walks[WALK]
    assert isinstance(walk, Sequential)
    encoder, loop = walk.sections
    assert encoder.name == "audio_encoder"
    assert [(e.name, e.next_node) for e in encoder.outputs] == [("encoder_states", "transducer")]
    assert isinstance(loop, Loop) and loop.name == DECODE_LOOP and loop.outputs == []
    node = loop.section
    assert isinstance(node, GraphNode) and node.name == "transducer" and node.enable_async_scheduling
    assert node.input_names == {"encoder_states"}
    assert [(e.name, e.next_node) for e in node.outputs] == [("tokens", EMIT_TO_CLIENT)]
    # the encoder output comes from outside the loop, so it saves and re-injects it
    assert loop._external_inputs == {("encoder_states", "transducer")}
    assert loop._loop_back_inputs == set()
    # a ceiling only: the clip's own cap, and its stop check, end the loop first
    assert loop.max_iters == model.config.transducer.decode_iterations(
        model.config.encoder.output_lengths(610 * 16_000 // 160))
    assert model.get_loop_caps(**{NUM_FRAMES: 7}) == {DECODE_LOOP: model.config.transducer.decode_iterations(7)}

    audio = _pointer(1)
    kwargs = {NUM_FRAMES: 7, PROMPT_ID: 3} if nemotron else {NUM_FRAMES: 7}
    first = model.get_initial_forward_pass_args("default", ["audio"], ["text"], {"audio": [audio]}, model_kwargs=kwargs)
    assert first.full_metadata.graph_walk == WALK
    [edge] = first.inputs
    assert (edge.name, edge.next_node, edge.tensor_info) == ("audio", "audio_encoder", [audio])
    assert first.unpersist_tensors == [audio]
    assert first.step_metadata == {NUM_FRAMES: 7, PROMPT_ID: 3 if nemotron else None}
    done = model.get_partition_forward_pass_args("default", first.full_metadata, {})
    assert done.request_done and done.inputs == []


def test_startup_refuses_more_requests_than_slots(tmp_path):
    model = ParakeetModel(str(_snapshot(tmp_path, nemotron=False)))
    assert pool_problems(model, {"max_concurrent_requests": 128}) == []
    assert pool_problems(
        model, {"max_concurrent_requests": 128, "resources": {TRANSDUCER_STATE: {"max_slots": 129}}},
    ) == []
    [problem] = pool_problems(
        model, {"max_concurrent_requests": 128, "resources": {TRANSDUCER_STATE: {"max_slots": 128}}},
    )
    assert "need 129 slots" in problem
    [problem] = pool_problems(model, {"max_concurrent_requests": 129})
    assert "need 130 slots" in problem
    [problem] = pool_problems(model, {"max_concurrent_requests": 0})
    assert "positive int" in problem
    # slot 0 is the sink, so one slot leaves none to admit into, with or without a cap
    [problem] = pool_problems(model, {"resources": {TRANSDUCER_STATE: {"max_slots": 1}}})
    assert "max_slots must be an int >= 2" in problem
