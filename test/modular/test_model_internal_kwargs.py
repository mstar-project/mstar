"""Keys only a model's process_prompt sets are never the client's.

The data worker lays process_prompt's metadata over the request's own
model_kwargs, and the conductor reads some of those keys back (Qwen3-TTS's
codec stream lead primes the Codec worker's stream). A client value for such a
key must reach neither process_prompt nor the conductor.
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
from mstar.streaming.topology import PartitionTopology

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


@pytest.mark.parametrize(
    "variant, mode, lead",
    [("custom_voice", None, 0), ("base", "x_vector", 0), ("base", "in_context", 13)],
)
def test_qwen3_tts_a_client_cannot_set_the_codec_stream_lead(variant, mode, lead):
    """A client value past the codec's left context would raise in the worker's
    ingest, outside any one request's failure path."""
    model = _qwen3_tts(variant)
    modalities = ["text"] if mode is None else ["audio", "text"]
    # 1 s + 1 sample -> 13 codec frames, fewer than the left context
    request = {} if mode is None else {"tensors": {"audio_inputs": [torch.zeros(24000 + 1)]}}
    if mode == "x_vector":
        request["x_vector_only_mode"] = True
    elif mode == "in_context":
        request["ref_text"] = "the reference says"
    client = {Qwen3TTSModel.CODEC_STREAM_LEAD_KEY: 999}
    result = model.process_prompt("hello", modalities, ["audio"], **request, **client)
    codec = model.get_initial_forward_pass_args(
        "Codec", modalities, ["audio"], {}, model_kwargs={**client, **result.metadata}
    )
    assert codec.stream_lead_items == ({"codec_tokens": lead} if lead else {})
    # every key the conductor reads back is one the data worker strips from the client
    assert set(result.metadata) <= set(model.INTERNAL_KWARGS)


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


class _Internal:
    """Reports a key of its own to the conductor, and only on some prompts."""

    INTERNAL_KWARGS = ("stream_lead",)

    def get_partition_topology(self):
        return PartitionTopology(partitions=["default"])

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
        model_kwargs={"stream_lead": 999, "voice": "tara"},
    ))
    kwargs = worker.communicator.sent[-1].body.model_kwargs

    assert "stream_lead" not in model.handed, "process_prompt was handed the client's value"
    assert "stream_lead" not in kwargs, (
        "a key only the model reports reached the conductor with the client's value"
    )
    assert kwargs["voice"] == "tara", "the strip dropped the request's own kwargs"
    assert kwargs["from_process_prompt"] == 1, "the model's own metadata never arrived"
