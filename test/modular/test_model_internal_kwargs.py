"""Keys only a model's process_prompt sets are never the client's.

The data worker lays process_prompt's metadata over the request's own
model_kwargs, and the conductor reads some of those keys back (a prompt
layout). A client value for such a key must reach neither process_prompt nor
the conductor.
"""

from __future__ import annotations

import queue
import threading

import pytest
import torch

from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.api_server.request_types import PreprocessInput
from mstar.model.base import ProcessPromptOutput
from mstar.model.qwen3_tts.config import Qwen3TTSModelConfig, Qwen3TTSTalkerConfig
from mstar.model.qwen3_tts.qwen3_tts_model import Qwen3TTSModel
from mstar.testing_utils.model import StubModel

ASSISTANT_PREFIX = [151644, 77091, 198]
ASSISTANT_SUFFIX = [151645, 198, 151644, 77091, 198]
REF_SUFFIX = [151645, 198]


class _TokenizerStub:
    """The reference ChatML wrappers around one id per word."""

    def __call__(self, text, **kwargs):
        if text.endswith("<|im_end|>\n<|im_start|>assistant\n"):
            body = text[len("<|im_start|>assistant\n"):-len("<|im_end|>\n<|im_start|>assistant\n")]
            suffix = ASSISTANT_SUFFIX
        else:
            body = text[len("<|im_start|>assistant\n"):-len("<|im_end|>\n")]
            suffix = REF_SUFFIX
        words = [1000 + i for i, _ in enumerate(body.split())]
        return {"input_ids": torch.tensor([ASSISTANT_PREFIX + words + suffix])}


def _qwen3_tts(tts_model_type: str) -> Qwen3TTSModel:
    model = object.__new__(Qwen3TTSModel)
    talker = Qwen3TTSTalkerConfig(hidden_size=2048, intermediate_size=6144)
    if tts_model_type != "custom_voice":
        talker.spk_id = {}
        talker.spk_is_dialect = {}
    model.config = Qwen3TTSModelConfig(tts_model_type=tts_model_type, tts_model_size="1b7", talker=talker)
    model.tokenizer = _TokenizerStub()
    model._submodule_cache = {}
    return model


def test_qwen3_tts_reports_only_keys_the_data_worker_strips():
    """The conductor reads process_prompt's metadata back, so none of it may
    be a key a client could also send."""
    model = _qwen3_tts("base")
    clip = {"tensors": {"audio_inputs": [torch.zeros(24000 + 1)]}, "ref_text": "the reference says"}
    result = model.process_prompt("hello", ["audio", "text"], ["audio"], **clip)
    assert set(result.metadata) - {"state_layout", "stream_lead_items"} <= set(model.INTERNAL_KWARGS)


class _RecordingCommunicator:
    def __init__(self):
        self.sent = []

    def get_all_new_messages(self):
        return []

    def send(self, entity, msg):
        self.sent.append(msg)


class _StubTensorManager:
    def store_and_return_tensor_info(self, request_id, tensors, **kwargs):
        return {}

    def register_for_send(self, request_id, tensor_infos, **kwargs):
        pass

    def set_persist(self, uuid, persist):
        pass


class _Internal(StubModel):
    """Reports a key of its own to the conductor, and only on some prompts."""

    INTERNAL_KWARGS = ("prompt_layout",)

    def process_prompt(self, *args, **kwargs):
        self.handed = kwargs
        return ProcessPromptOutput({"text_inputs": [torch.tensor([1, 2, 3])]}, {"from_process_prompt": 1})


def test_a_clients_value_for_a_model_internal_key_never_reaches_the_model_or_conductor():
    model = _Internal()
    worker = PreprocessWorkerThread(
        in_queue=queue.Queue(), result_tensor_queue=queue.Queue(),
        out_queue=queue.Queue(), profile_queue=queue.Queue(),
        cleanup_request_queue=queue.Queue(), abort_request_queue=queue.Queue(),
        reads_done_queue=queue.Queue(), discard_tensor_queue=queue.Queue(),
        stop_event=threading.Event(), communicator=_RecordingCommunicator(),
        tensor_manager=_StubTensorManager(), model=model,
    )

    worker._process_input(PreprocessInput(
        request_id="r0", text="hello", file_paths=None,
        input_modalities=["text"], output_modalities=["text"],
        model_kwargs={"prompt_layout": 999, "voice": "tara"},
    ))
    kwargs = worker.communicator.sent[-1].body.model_kwargs

    assert "prompt_layout" not in model.handed, "process_prompt was handed the client's value"
    assert "prompt_layout" not in kwargs, (
        "a key only the model reports reached the conductor with the client's value"
    )
    assert kwargs["voice"] == "tara", "the strip dropped the request's own kwargs"
    assert kwargs["from_process_prompt"] == 1, "the model's own metadata never arrived"


@pytest.mark.parametrize("kwargs, field", [
    ({"pad_duration": "soon"}, "pad_duration"),
    # the pad is allocated on the API server, after generation
    ({"pad_duration": 1e5}, "pad_duration"),
    ({"fade_duration": float("nan")}, "fade_duration"),
    # bool("false") is True
    ({"postprocess_output": "false"}, "postprocess_output"),
    # a non-finite unmask knob
    ({"position_temperature": float("inf")}, "position_temperature"),
    ({"layer_penalty_factor": "steep"}, "layer_penalty_factor"),
    # parsed at the conductor, where the raise hangs the client
    ({"num_step": "many"}, "num_step"),
])
def test_omnivoice_refuses_a_bad_request_value_before_it_generates(kwargs, field):
    """``postprocess`` shapes the waveform after the whole unmask loop, which
    reads its knobs at every step; a bad value in either would fail a request
    whose generation already ran (or, read at the conductor, hang its client)."""
    from mstar.model.omnivoice.config import OmniVoiceConfig
    from mstar.model.omnivoice.omnivoice_model import OmniVoiceModel

    model = object.__new__(OmniVoiceModel)
    model.config = OmniVoiceConfig()
    with pytest.raises(ValueError, match=field):
        model.process_prompt("hello", ["text"], ["audio"], **kwargs)


def test_omnivoice_output_shaping_takes_json_booleans():
    """``postprocess_output: 0`` (or 1), as a JSON client sends a boolean."""
    from mstar.model.omnivoice.omnivoice_model import output_shaping

    assert output_shaping({"postprocess_output": 0})["enabled"] is False
    assert output_shaping({"postprocess_output": 1})["enabled"] is True
