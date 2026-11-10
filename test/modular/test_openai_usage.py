"""What the OpenAI ``usage`` and ``finish_reason`` are built from: the data
worker counts the prompt and each text chunk's tokens, the conductor returns
the prompt count and the output budget on ``RequestComplete``, and
``RequestUsage`` says whether the reply used its whole budget."""
import dataclasses
import queue
import threading

import pytest

torch = pytest.importorskip("torch")

from test_conductor_status import _conductor, _done, _to_api_server  # noqa: E402  (sibling helpers)
from test_ingest_unpersist import _info, _msg, _run  # noqa: E402
from test_inline_tokens import _run_until, _thread  # noqa: E402
from test_kv_layout_intake import PROMPT, _Comm, _Model  # noqa: E402
from test_kv_prefix_request_keys import _StubTensorManager  # noqa: E402

import mstar.communication.wire_types  # noqa: E402,F401  (registers the wire tags)
from mstar.api_server.data_worker import InlineTokens, PreprocessWorkerThread  # noqa: E402
from mstar.api_server.request_types import (  # noqa: E402
    APIServerMessage,
    PreprocessInput,
    RequestComplete,
    RequestUsage,
)
from mstar.communication import wire  # noqa: E402
from mstar.testing_utils.model import StubModel  # noqa: E402
from mstar.utils.ipc_format import ConductorMessageType, NewRequestConductor  # noqa: E402


@pytest.mark.parametrize("completion,cap,finish", [
    (2, 2, "length"), (3, 2, "length"), (1, 2, "stop"), (5, None, "stop"),
], ids=["at-cap", "past-cap", "below-cap", "no-cap"])
def test_a_reply_that_used_its_whole_budget_ended_on_length(completion, cap, finish):
    usage = RequestUsage(prompt_tokens=4, completion_tokens=completion, max_output_tokens=cap)
    assert usage.finish_reason == finish


def test_the_prompt_is_counted_from_its_ids_by_default():
    count = StubModel().count_prompt_tokens
    assert count({"text_inputs": [torch.arange(5), torch.arange(3).reshape(1, 3)]}) == 8
    # embeddings are not ids, and a prompt with no ids has none
    assert count({"text_inputs": [torch.arange(5), torch.zeros(2, 4)]}) == 5
    assert count({"image_inputs": [torch.zeros(3, 2, 2)]}) == 0


def test_the_data_worker_sends_the_prompt_s_length_with_the_request():
    worker = PreprocessWorkerThread(
        *(queue.Queue() for _ in range(8)), stop_event=threading.Event(), communicator=_Comm(),
        tensor_manager=_StubTensorManager(), model=_Model(layout=None),
        model_config={"model": "stub", "node_groups": [{"node_names": ["llm", "talker"], "ranks": [0]}]},
    )
    worker._process_input(PreprocessInput(
        request_id="r", text="hello", file_paths=None, input_modalities=["text"],
        output_modalities=["text"], model_kwargs={},
    ))
    # through the wire codec, as it travels
    sent = worker.communicator.sent[-1].body
    assert isinstance(sent, NewRequestConductor) and sent.prompt_tokens == PROMPT


class _AnyText(StubModel):
    def postprocess(self, output, modality, request_kwargs=None):
        return b"x"


def test_each_text_chunk_carries_its_token_count():
    worker, stop = _thread()
    worker.result_tensor_queue.put(InlineTokens(["a", "b"], [11, 12], "text"))
    assert [c.num_tokens for c in _run_until(worker, stop, 2)] == [1, 1]
    # a chunk read as a tensor counts its ids; a float text tensor (Whisper's alignment) has none
    worker, _ = _thread()
    worker.model = _AnyText()
    for i, tensor in enumerate((torch.tensor([[5, 6, 7]]), torch.rand(2, 3))):
        worker.tensor_uuid_to_metadata_per_request = {"a": {i: {}}}
        worker._postprocess_inline("a", i, "text", i, tensor, None)
    assert [worker.out_queue.get_nowait().num_tokens for _ in range(2)] == [3, 0]


def test_the_conductor_returns_the_prompt_count_and_the_budget_it_ran_with(tmp_path, monkeypatch):
    c = _conductor(tmp_path, monkeypatch, status_interval_s=None)
    _run(c, [
        [_msg(ConductorMessageType.NEW_REQUEST, NewRequestConductor(
            request_id="r1", initial_signals={"x": [_info(1)]}, initial_input_modalities=["text"],
            initial_output_modalities=["text"], input_metadata={}, model_kwargs={"max_output_tokens": 5},
            prompt_tokens=7,
        ))],
        [_done(c, "r1", "enc"), _done(c, "r1", "llm")],
    ])
    (done,) = _to_api_server(c, "request_complete")
    assert (done.prompt_tokens, done.max_output_tokens) == (7, 5)
    # and it decodes as itself with the two
    frame = wire.encode(APIServerMessage("request_complete", done))
    assert dataclasses.astuple(wire.decode(frame).body) == dataclasses.astuple(done)
    assert isinstance(wire.decode(frame).body, RequestComplete)
