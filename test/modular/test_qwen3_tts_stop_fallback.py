"""The Talker's per-request stop check on a decode step.

A decode step's stop buffer reaches the host as rows under the forward's name,
``new_token``. When ``check_stop_batched`` raises or declines,
``Engine.check_stop_for_batch`` hands those rows to ``check_stop`` one request
at a time, and an EOS row must still stop its loop and keep its frame off the
codec stream: EOS is past the codec's codebooks. Drives the real Talker
forward and checks, the engine's collect, postprocess and stop check, and
``Worker._postprocess_batch`` on CPU, with the model's frame and the graph
runtime stubbed.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from test_qwen3_tts_model import _talker_with_frame, _tiny_model_config
from test_withheld_outputs import _Communicator, _RecordingRuntime

from mstar.communication.tensors import SharedMemoryCommunicationManager
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources import StepContext
from mstar.model.qwen3_tts.submodules import TALKER_SAMPLER
from mstar.model.submodule_base import HostRows
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.worker import Worker

NODE, WALK, LOOP = "Talker", "talker_decode", "talker_decode_loop"
SIGNALS = ["talker_input_embeds", "codec_tokens"]
ROWS = 8
EOS = _tiny_model_config().talker.codec_eos_token_id


class _Runtime(_RecordingRuntime):
    supports_inline_emit = False


def _info(max_tokens=8192, ignore_eos=False):
    return SimpleNamespace(
        max_tokens=max_tokens, resource_configs={TALKER_SAMPLER: SimpleNamespace(ignore_eos=ignore_eos)},
    )


def _talker(tokens, padded_to, batched):
    """A Talker whose decode frame carries ``tokens`` as group 0, padded the
    way a replay pads, and whose batched stop check does as ``batched`` says."""
    sub, frame = _talker_with_frame(_tiny_model_config(), padded_to or len(tokens))
    frame["new_token"][:len(tokens)] = torch.tensor(tokens)
    frame["codec_tokens"][:, 0] = frame["new_token"]
    if batched == "raises":
        def check_stop_batched(*args, **kwargs):
            raise RuntimeError("a row the stop check cannot read")
        sub.check_stop_batched = check_stop_batched
    elif batched == "declines":
        sub.check_stop_batched = lambda *args, **kwargs: None
    return sub


def _engine(sub):
    engine = Engine.__new__(Engine)
    engine._submodules = {NODE: SimpleNamespace(submodule=sub, piecewise_runners={})}
    return engine


def _decode_step(engine, sub, rids, staged):
    """Forward, collect and postprocess as the engine runs a decode step.
    ``staged``: the gpu thread's host copy of the stop rows (all rows of the
    forward, padding included), which the worker reads in place of its own."""
    batch = ExecutingBatch(
        node_name=NODE, per_request_info={rid: _info() for rid in rids},
        step_context=StepContext(request_ids=tuple(rids), graph_walk=WALK, slot=0, capture=False),
        inputs=[None] * len(rids),
    )
    raw = sub.forward_batched(
        WALK, SimpleNamespace(request_ids=rids), input_embeds=None, last_token_indices=None, suppress_eos=None,
    )
    outputs = engine._collect_outputs(
        engine._submodules[NODE], None, raw, [None] * len(rids), batch.per_request_info, list(rids),
        tuple(rids), WALK,
    )
    engine._postprocess_batch(batch, outputs)
    if staged:
        outputs.host_rows = HostRows(tuple(rids), {name: t.clone() for name, t in raw.check_stop_buffers.items()})
        batch.completion_event = object()  # waited on before the stop check; a no-op on CPU
    return batch, outputs


def _worker(engine, tmp_path):
    w = Worker.__new__(Worker)
    w._phase_period = 0
    w.enable_nvtx = w.enable_prof = False
    w.device = torch.device("cpu")
    w.device_capability = SimpleNamespace(is_available=False, is_accelerator=False)
    w._last_active = {}
    w._innermost_loop_by_node = {}
    w._walk_drivers = {}
    w._graph_runtime = _Runtime()
    w.scheduler = SimpleNamespace(failed_rids=set())
    w.tensor_manager = SharedMemoryCommunicationManager(
        my_entity_id="worker_0", hostname="localhost", device="cpu",
        communicator=_Communicator(), shm_dir=str(tmp_path),
    )
    w.engine_manager = SimpleNamespace(get_engine=lambda node: SimpleNamespace(
        check_stop_for_batch=engine.check_stop_for_batch,
        extend_prefix_chains=lambda batch, outputs, **kwargs: None,
        materialize_publish=lambda batch, rids: None,
    ))
    w.request_state = SimpleNamespace(
        per_request_info={}, buffer_publish_info=lambda *args: None,
        get_fwd_info=lambda rid, partition: None, get_pending_publish_info=lambda rid, partition: {},
    )
    return w


def _route(w, batch, outputs, rids):
    pending = SimpleNamespace(
        batch=ScheduledBatch(node_name=NODE, graph_walk=WALK, request_to_worker_graph={rid: 0 for rid in rids},
                             output_signals=SIGNALS),
        node_batch=batch, node_name=NODE, partition="default", graph_walk=WALK, loop_name=None,
        speculative_new_iter=False,
    )
    Worker._postprocess_batch(w, pending, outputs)


def _routed(w, signal):
    """The rows the step routed a tensor of ``signal`` for."""
    (route,) = w._graph_runtime.routes
    counts = iter(route.num_tensors)
    return [rid for rid in route.wg_ids.keys for name in SIGNALS if next(counts) and name == signal]


def _warnings(caplog):
    return [r for r in caplog.records if r.name == "mstar.engine.engine" and r.levelno == logging.WARNING]


@pytest.mark.parametrize("batched", ["answers", "raises", "declines"])
@pytest.mark.parametrize("staged", [False, True], ids=["copied", "staged"])
@pytest.mark.parametrize("padded_to", [None, 16], ids=["unpadded", "padded"])
@pytest.mark.parametrize("eos_row", range(ROWS))
def test_an_eos_row_stops_and_keeps_its_frame(tmp_path, caplog, eos_row, padded_to, staged, batched):
    rids = list(range(ROWS))
    tokens = [EOS if rid == eos_row else 3 + rid for rid in rids]
    sub = _talker(tokens, padded_to, batched)
    engine = _engine(sub)
    batch, outputs = _decode_step(engine, sub, rids, staged)
    w = _worker(engine, tmp_path)
    with caplog.at_level(logging.WARNING, logger="mstar.engine.engine"):
        _route(w, batch, outputs, rids)
    assert _routed(w, "codec_tokens") == [rid for rid in rids if rid != eos_row]
    assert _routed(w, "talker_input_embeds") == rids
    assert w._graph_runtime.stops == [{eos_row: [LOOP]}]
    # a raise is a bug, so it is logged where a server at INFO shows it, once
    warned = _warnings(caplog)
    assert len(warned) == (batched == "raises")
    if warned:
        assert "a row the stop check cannot read" in warned[0].getMessage()


@pytest.mark.parametrize("staged", [False, True], ids=["copied", "staged"])
def test_the_per_request_path_reads_each_requests_own_row(tmp_path, staged):
    """The forward ran rows 0..3; row 1 left the request list before the stop
    check, which then reads rows by request, not by position."""
    sub = _talker([3, EOS, EOS, 4], None, "raises")
    engine = _engine(sub)
    batch, outputs = _decode_step(engine, sub, [0, 1, 2, 3], staged)
    outputs.per_rid_outputs.pop(1)
    batch.drop_rids({1})
    w = _worker(engine, tmp_path)
    _route(w, batch, outputs, [0, 2, 3])
    assert _routed(w, "codec_tokens") == [0, 3]
    assert w._graph_runtime.stops == [{2: [LOOP]}]


def test_the_per_request_answer_is_the_batched_one():
    """EOS, EOS under ``ignore_eos`` (withheld, the loop runs on), the frame
    limit, and a running row, off the same host rows either way."""
    rids = ["eos", "ignored", "limit", "running"]
    infos = {"eos": _info(), "ignored": _info(ignore_eos=True), "limit": _info(max_tokens=2), "running": _info()}
    answers = []
    for batched in ("answers", "raises"):
        sub = _talker([0] * len(rids), None, batched)
        for rid in rids:
            sub.request_state(rid).add("generated_frames", 2)
        tokens = torch.tensor([EOS, EOS, 3, 4])
        batch = ExecutingBatch(
            node_name=NODE, per_request_info=infos,
            step_context=StepContext(request_ids=tuple(rids), graph_walk=WALK, slot=0, capture=False),
        )
        host = {"new_token": tokens}
        stops = _engine(sub).check_stop_for_batch(
            batch, Worker._rows_to_per_rid(host, rids), host_rows=HostRows(tuple(rids), host),
        )
        answers.append((stops, batch.withheld_outputs))
    assert answers[0] == answers[1] == (
        {"eos": {LOOP}, "limit": {LOOP}},
        {"eos": ("codec_tokens",), "ignored": ("codec_tokens",)},
    )
