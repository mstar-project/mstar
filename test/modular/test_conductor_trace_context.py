"""The API request span's context crosses preprocessing and conductor intake."""

from __future__ import annotations

import logging
import queue
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, ".")

from test_kv_layout_intake import _Comm, _Model, _StubTensorManager  # noqa: E402

import mstar.communication.wire_types  # noqa: F401, E402 (registers typed wire tags)
import mstar.conductor.conductor as conductor_mod  # noqa: E402
from mstar.api_server.data_worker import PreprocessWorkerThread  # noqa: E402
from mstar.api_server.request_types import PreprocessInput  # noqa: E402
from mstar.communication import wire  # noqa: E402
from mstar.conductor.conductor import Conductor  # noqa: E402
from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.observability.tracing import trace_id_from_parent  # noqa: E402
from mstar.utils.ipc_format import ConductorMessage, NewRequest, NewRequestConductor  # noqa: E402
from mstar.utils.logging_config import REQUEST_LOGGER  # noqa: E402

PARENT = "00-0123456789abcdef0123456789abcdef-aaaaaaaaaaaaaaaa-01"


@pytest.mark.parametrize("parent", [None, PARENT])
def test_data_worker_and_conductor_forward_only_active_context(
    parent: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    model = _Model()
    worker_comm = _Comm()
    worker = PreprocessWorkerThread(
        *(queue.Queue() for _ in range(8)), stop_event=threading.Event(),
        communicator=worker_comm, tensor_manager=_StubTensorManager(), model=model,
        model_config={
            "model": "stub", "resources": {"kv": {"page_size": 16}},
            "node_groups": [{"node_names": ["llm", "talker"], "ranks": [0]}],
        },
    )
    worker._process_input(PreprocessInput(
        request_id="client#1", client_request_id="client", text="hello", file_paths=None,
        input_modalities=["text"], output_modalities=["text"], model_kwargs={},
        traceparent=parent,
    ))
    conductor_message = worker_comm.sent[-1]
    assert isinstance(conductor_message, ConductorMessage)
    assert isinstance(conductor_message.body, NewRequestConductor)
    assert conductor_message.body.traceparent == parent

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({
        "max_seq_len": 4096, "node_groups": [{"node_names": ["llm", "talker"], "ranks": [0]}],
    }))
    sent = []
    comm = SimpleNamespace(send=lambda entity, msg: sent.append((entity, wire.decode(wire.encode(msg)))))
    monkeypatch.setattr(Conductor, "_launch_workers", lambda self: None)
    monkeypatch.setattr(conductor_mod, "make_communicator", lambda **kwargs: comm)
    # The shared local extension was built from another worktree; this test
    # exercises conductor message routing, not the graph runtime binary.
    monkeypatch.setattr("mstar.validation.knobs.require_rust_extension", lambda: None)
    conductor = Conductor(model, str(cfg), socket_path_prefix=str(tmp_path / "s"))
    with caplog.at_level(logging.DEBUG, logger=REQUEST_LOGGER):
        conductor._ingest_request(conductor_message.body)
    assert conductor.requests["client#1"].traceparent == parent
    requests = [msg.body for _, msg in sent if isinstance(msg.body, NewRequest)]
    assert requests and all(msg.traceparent == parent for msg in requests)
    if parent is None:
        assert "trace_id=" not in caplog.text
    else:
        trace_id = trace_id_from_parent(parent)
        assert trace_id is not None and f"trace_id={trace_id}" in caplog.text


def test_typed_wire_omits_context_when_tracing_is_off() -> None:
    import msgpack

    base = dict(
        request_id="r", initial_signals={}, initial_input_modalities=["text"],
        initial_output_modalities=["text"], input_metadata={}, model_kwargs={},
    )
    worker_request = NewRequest(
        request_id="r", partition_worker_graph_ids=[], worker_graph_to_workers={},
        initial_inputs=[], loop_caps={}, request_info=CurrentForwardPassInfo(
            request_id="r", graph_walk="prefill", fwd_index=0, random_seed=1, max_tokens=4,
        ),
    )
    for message in (NewRequestConductor(**base), worker_request):
        _, fields = msgpack.unpackb(wire.encode(message), raw=False)
        assert "traceparent" not in fields
    for traced in (NewRequestConductor(**base, traceparent=PARENT),
                   NewRequest(**{**vars(worker_request), "traceparent": PARENT})):
        _, fields = msgpack.unpackb(wire.encode(traced), raw=False)
        assert fields["traceparent"] == PARENT
        assert wire.decode(wire.encode(traced)).traceparent == PARENT
