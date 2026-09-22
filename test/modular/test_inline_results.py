"""Small client outputs travel inline in one message per step (``InlineResults``): the worker
picks the tensors that have host copies under the size limit, the API server rebuilds them and
emits the chunks without a transport read, and the pending-output accounting still holds."""
from __future__ import annotations

import queue
import sys
import threading
from types import SimpleNamespace

sys.path.insert(0, ".")

import torch

from mstar.api_server.data_worker import InlineResult, PreprocessWorkerThread
from mstar.api_server.request_types import INLINE_MAX_BYTES, InlineResults, ResultTensors
from mstar.communication.tensors import _deserialize_tensor, _serialize_tensor
from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.worker.worker import inline_output_bytes


def _loop_idx():
    return NestedLoopIndices(loop_name_order=[], loop_indices={}, wg_fwd_pass_idx=0)


def _info(uuid, tensor):
    return TensorPointerInfo(dims=list(tensor.shape), dtype=tensor.dtype, nbytes=tensor.nbytes, address=0,
                             stride=list(tensor.stride()), uuid=uuid, source_session_id="s", source_entity="w")


def test_inline_bytes_take_small_host_tensors_only():
    tok = torch.tensor([42], dtype=torch.long)
    big = torch.zeros(INLINE_MAX_BYTES + 1, dtype=torch.uint8)  # one byte over the limit
    infos = {"new_token": [_info("u-tok", tok)], "blob": [_info("u-big", big)], "other": [_info("u-x", tok)]}
    host = {"new_token": [tok], "blob": [big], "other": [None]}
    out = inline_output_bytes(host, infos)
    assert set(out) == {"u-tok"}, "the big tensor and the tensor without a host copy stay on the transport"
    assert torch.equal(_deserialize_tensor(out["u-tok"], "cpu", infos["new_token"][0]), tok)
    assert inline_output_bytes(None, infos) == {} and inline_output_bytes({"new_token": tok}, infos) == {}
    # a mismatch between stored infos and host copies is left alone rather than mis-paired
    assert inline_output_bytes({"new_token": [tok, tok]}, infos) == {}


def test_serialized_bytes_round_trip_for_the_dtypes_a_step_emits():
    for t in (torch.tensor([7], dtype=torch.long), torch.tensor([[1, 2, 3]], dtype=torch.int32),
              torch.tensor([0.5], dtype=torch.bfloat16), torch.tensor(3, dtype=torch.long)):
        info = _info("u", t)
        back = _deserialize_tensor(_serialize_tensor(t), "cpu", info)
        assert back.shape == t.shape and back.dtype == t.dtype and torch.equal(back, t)


class _Model:
    def postprocess(self, output, modality, **kwargs):
        return bytes(f"tok{int(output.reshape(-1)[0])};", "utf-8")


def _thread():
    q = {name: queue.Queue() for name in ("in", "res", "out", "prof", "cleanup", "abort", "reads", "discard")}
    th = PreprocessWorkerThread(
        in_queue=q["in"], result_tensor_queue=q["res"], out_queue=q["out"], profile_queue=q["prof"],
        cleanup_request_queue=q["cleanup"], abort_request_queue=q["abort"], reads_done_queue=q["reads"],
        discard_tensor_queue=q["discard"], stop_event=threading.Event(),
        communicator=SimpleNamespace(send=lambda *a, **k: None), tensor_manager=SimpleNamespace(), model=_Model(),
    )
    return th, q


def test_inline_result_is_delivered_without_a_transport_read():
    th, q = _thread()
    tok = torch.tensor([42], dtype=torch.long)
    info = _info("u-1", tok)
    res = ResultTensors(request_id="r1", modality="text", loop_indices=_loop_idx(),
                        graph_edge=GraphEdge(next_node="EMIT_TO_CLIENT", name="new_token", tensor_info=[info]))
    th._deliver_inline_result(InlineResult(res, {"u-1": _serialize_tensor(tok)}))
    chunk = q["out"].get_nowait()
    assert (chunk.request_id, chunk.modality, chunk.data) == ("r1", "text", b"tok42;")
    # a request being drained gets nothing (and nothing is acked, there is no transport state)
    th._draining_rids.add("r2")
    res2 = ResultTensors(request_id="r2", modality="text", loop_indices=_loop_idx(),
                         graph_edge=GraphEdge(next_node="EMIT_TO_CLIENT", name="new_token", tensor_info=[info]))
    th._deliver_inline_result(InlineResult(res2, {"u-1": _serialize_tensor(tok)}))
    assert q["out"].empty()


def test_main_thread_accounting_counts_inline_outputs_like_read_ones():
    from mstar.api_server.data_worker import PreprocessWorker

    pw = PreprocessWorker.__new__(PreprocessWorker)  # no threads, no sockets: just the bookkeeping
    pw.output_loop_idxs = {"r1": {}}
    pw.per_request_reading_tensors = {"r1": 0}
    pw.result_tensor_input_queue = queue.Queue()
    tok = torch.tensor([5], dtype=torch.long)
    info = _info("u-9", tok)
    res = ResultTensors(request_id="r1", modality="text", loop_indices=_loop_idx(),
                        graph_edge=GraphEdge(next_node="EMIT_TO_CLIENT", name="new_token", tensor_info=[info]))
    batch = InlineResults(results=[res], data={"u-9": _serialize_tensor(tok), "u-other": b"x"})
    pw.new_inline_result(res, batch.data)
    assert pw.per_request_reading_tensors["r1"] == 1 and pw.has_pending_tensors("r1")
    item = pw.result_tensor_input_queue.get_nowait()
    assert isinstance(item, InlineResult) and set(item.data) == {"u-9"}
    # a request the API server no longer tracks is dropped quietly
    pw.new_inline_result(ResultTensors(request_id="gone", modality="text", loop_indices=_loop_idx(),
                                       graph_edge=res.graph_edge), batch.data)
    assert pw.result_tensor_input_queue.empty()


def test_host_bytes_from_a_shared_buffer_match_the_direct_path():
    from mstar.worker.worker import host_tensor_bytes

    buf = torch.arange(40, dtype=torch.int32)
    views = [buf[3:4], buf[7:9], buf[20:40].view(4, 5)]
    blobs: dict[int, bytes] = {}
    for v in views:
        assert host_tensor_bytes(v, blobs) == host_tensor_bytes(v) == v.reshape(-1).view(torch.uint8).numpy().tobytes()
    assert len(blobs) == 1  # one conversion for the shared storage
    strided = torch.arange(12, dtype=torch.int32).view(3, 4)[:, 1]
    assert host_tensor_bytes(strided, blobs) == strided.contiguous().view(torch.uint8).numpy().tobytes()


def test_tensor_ids_are_unique_plain_strings(monkeypatch):
    from mstar.communication import tensors

    ids = [tensors.new_tensor_id() for _ in range(1000)]
    assert len(set(ids)) == 1000 and all(isinstance(i, str) and "/" not in i and " " not in i for i in ids)
    # a forked child (new pid) starts its own prefix instead of continuing the parent's sequence
    monkeypatch.setattr(tensors.os, "getpid", lambda: 1 << 30)
    child = tensors.new_tensor_id()
    assert child.startswith(f"{1 << 30:x}-") and child not in ids
