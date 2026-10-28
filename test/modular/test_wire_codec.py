"""Round-trip every message that crosses a communicator edge.

The codec is type-driven, so a field whose declared type lies about its value
(``TensorPointerInfo.dtype``) or whose shape msgpack cannot express (tuple
dict keys, sets, abstract field types) is exactly where it breaks. Each of
those has a test.
"""
import datetime
import os
import sys

sys.path.insert(0, ".")

import msgpack
import numpy as np
import pytest
import torch

import mstar.communication.wire_types  # noqa: F401  (registers the tags)
from mstar.api_server.request_types import APIServerMessage, RequestComplete, ResultTensors
from mstar.communication.wire import _TYPE_TO_TAG, WireError, decode, encode
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.profile.format import GraphTiming, RxInfo, TxInfo
from mstar.utils.ipc_format import (
    ConductorMessage,
    ConductorMessageType,
    FailRequests,
    InputSignals,
    NewRequest,
    ScheduleTPNode,
    StopLoops,
    WorkerGraphsDone,
    WorkerMessage,
    WorkerMessageType,
)


@pytest.fixture(autouse=True)
def _no_unpickling(monkeypatch):
    """Every round trip in this module must decode without pickle."""
    import pickle

    def refuse(*args, **kwargs):
        raise AssertionError("the wire codec unpickled a value")

    for name in ("loads", "load", "Unpickler"):
        monkeypatch.setattr(pickle, name, refuse)


def _info(uuid="u-1", dtype=torch.bfloat16) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=[8, 16], dtype=dtype, nbytes=256, address=0xDEAD, stride=(16, 1),
        uuid=uuid, source_session_id="host:1", source_entity="w0",
        offset=32, source_tp_size=2, source_tp_rank=1,
        shm_segment="seg-0", shm_offset=64,
    )


def _fwd_info() -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id="r0", fwd_index=3, random_seed=7, max_tokens=128,
        graph_walk="decode", partition_name="p0",
        step_metadata={"is_prefill": False, "chunk": 2},
    )


def _edge(**kw) -> GraphEdge:
    base = dict(
        next_node="llm", name="tok", tensor_info=[_info()], persist=True,
        conductor_new_token=True, is_streaming=False, output_modality="text",
    )
    base.update(kw)
    return GraphEdge(**base)


def roundtrip(msg):
    out = decode(encode(msg))
    assert type(out) is type(msg)
    return out


# ----------------------------------------------------------------------
# the awkward field shapes
# ----------------------------------------------------------------------


def test_torch_dtype_survives_and_is_usable():
    """Declared ``str``, holds a ``torch.dtype``, and the receiver feeds it to
    ``torch.empty`` — so it has to come back as a real dtype, not its name."""
    msg = WorkerMessage(
        message_type=WorkerMessageType.INPUT_SIGNALS,
        body=InputSignals(request_id="r0", inputs=[_edge()],
                          request_info=_fwd_info(), partition_name="p0"),
    )
    info = roundtrip(msg).body.inputs[0].tensor_info[0]
    assert isinstance(info.dtype, torch.dtype)
    assert info.dtype is torch.bfloat16
    assert torch.empty(info.dims, dtype=info.dtype).shape == torch.Size([8, 16])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16,
                                   torch.int64, torch.uint8, torch.bool])
def test_every_dtype_we_ship(dtype):
    msg = WorkerMessage(
        message_type=WorkerMessageType.INPUT_SIGNALS,
        body=InputSignals(request_id="r", inputs=[_edge(tensor_info=[_info(dtype=dtype)])],
                          request_info=_fwd_info(), partition_name="p"),
    )
    assert roundtrip(msg).body.inputs[0].tensor_info[0].dtype is dtype


def test_tuple_keyed_dict():
    """``graph_timings`` is keyed by (node, walk); msgpack has no tuple keys."""
    timings = {
        ("llm", "decode"): GraphTiming(
            node="llm", graph_walk="decode", exec_count=4, total_time=1.5,
            forward_time=1.0, preprocess_time=0.2, postprocess_time=0.3),
    }
    msg = ConductorMessage(
        message_type=ConductorMessageType.WORKER_GRAPHS_DONE,
        body=WorkerGraphsDone(request_id="r0", worker_graph_ids=["wg0"],
                              is_first_tp_rank=True, graph_timings=timings),
    )
    got = roundtrip(msg).body.graph_timings
    assert set(got) == {("llm", "decode")}
    assert got[("llm", "decode")].exec_count == 4
    assert got[("llm", "decode")].total_time == pytest.approx(1.5)


def test_sets_come_back_as_sets():
    msg = WorkerMessage(
        message_type=WorkerMessageType.STOP_LOOPS,
        body=StopLoops(
            request_id="r0", loop_names={"decode_loop", "outer"},
            partition_name="p0",
            loop_stop_times={"decode_loop": NestedLoopIndices(
                loop_name_order=["outer", "decode_loop"],
                loop_indices={"outer": 1, "decode_loop": 7},
                wg_fwd_pass_idx=2)},
        ),
    )
    body = roundtrip(msg).body
    assert body.loop_names == {"decode_loop", "outer"}
    assert isinstance(body.loop_names, set)
    snap = body.loop_stop_times["decode_loop"]
    assert snap.loop_name_order == ["outer", "decode_loop"]
    assert snap.loop_indices == {"outer": 1, "decode_loop": 7}


def test_polymorphic_body_dispatch():
    """``WorkerMessage.body`` is declared as the abstract ``MessageBody``, so
    the concrete class has to be tagged."""
    for body, mtype in [
        (ScheduleTPNode(node_name="llm", graph_walk="decode",
                        request_ids=["a", "b"], spec_seq=3),
         WorkerMessageType.SCHEDULE_TP),
        (NewRequest(request_id="r", partition_worker_graph_ids=["wg0"],
                    worker_graph_to_workers={"wg0": ["w0", "w1"]},
                    initial_inputs=[_edge()], request_info=_fwd_info(), loop_caps={}),
         WorkerMessageType.NEW_REQUEST),
    ]:
        got = roundtrip(WorkerMessage(message_type=mtype, body=body))
        assert type(got.body) is type(body)
        assert got.message_type is mtype


def test_unregistered_payloads_still_round_trip():
    """Supported unregistered values retain their types in a data envelope,
    including tuple-ness, which plain msgpack would flatten."""
    for payload in [
        "first",
        ("done", 42),
        {"op": "execute", "rids": [1, 2, 3], "nested": {"f": 1.5}},
        None,
        [1, ("a", "b")],
    ]:
        assert decode(encode(payload)) == payload


def test_unknown_tag_is_a_clear_error():
    import msgpack
    with pytest.raises(WireError, match="unknown wire message tag"):
        decode(msgpack.packb(["nope", {}]))


# ----------------------------------------------------------------------
# representative full messages
# ----------------------------------------------------------------------


def test_worker_graphs_done_full():
    msg = ConductorMessage(
        message_type=ConductorMessageType.WORKER_GRAPHS_DONE,
        body=WorkerGraphsDone(
            request_id="r0", worker_graph_ids=["wg0", "wg1"], is_first_tp_rank=False,
            persist_signals={"kv": [_info("u-a"), _info("u-b")]},
            new_token_counts={"tok": 5},
            resource_publish_info={},
            partition_name="p0", partition_done=True,
            stream_tokens_consumed={"audio": 12},
            output_signal_names=["tok", "tok"],
            rx_info=[RxInfo(edge_name="e", source_entity="w1", dest_entity="w0",
                            count=2, num_bytes=99, time=0.5)],
            tx_info=[TxInfo(edge_name="e", source_entity="w0", count=1,
                            num_bytes=50, time=0.25)],
        ),
    )
    body = roundtrip(msg).body
    assert body.worker_graph_ids == ["wg0", "wg1"]
    assert body.is_first_tp_rank is False
    assert [i.uuid for i in body.persist_signals["kv"]] == ["u-a", "u-b"]
    assert body.persist_signals["kv"][0].dtype is torch.bfloat16
    assert body.stream_tokens_consumed == {"audio": 12}
    assert body.rx_info[0].num_bytes == 99
    assert body.tx_info[0].time == pytest.approx(0.25)
    assert body.partition_done is True
    assert body.output_signal_names == ["tok", "tok"]


def test_api_server_request_complete_carries_chunk_counts():
    msg = APIServerMessage(
        message_type="request_complete",
        body=RequestComplete(
            request_id="r0", output_chunk_counts={"new_token": 5, "audio_chunk": 2},
            conductor_ingest_time=1.0, conductor_finish_time=2.0,
        ),
    )
    body = roundtrip(msg).body
    assert body.output_chunk_counts == {"new_token": 5, "audio_chunk": 2}


def test_api_server_result_tensors():
    msg = APIServerMessage(
        message_type="result_tensors",
        body=ResultTensors(
            request_id="r0", modality="audio", graph_edge=_edge(name="chunk"),
            loop_indices=NestedLoopIndices(
                loop_name_order=["l"], loop_indices={"l": 3}, wg_fwd_pass_idx=0),
            metadata={"sr": 24000},
        ),
    )
    body = roundtrip(msg).body
    assert body.modality == "audio"
    assert body.graph_edge.name == "chunk"
    assert body.metadata == {"sr": 24000}


def test_fail_requests_and_empty_collections():
    msg = ConductorMessage(
        message_type=ConductorMessageType.FAIL_REQUESTS,
        body=FailRequests(errors={"r0": "oom", "r1": "admit rejected"}),
    )
    assert roundtrip(msg).body.errors == {"r0": "oom", "r1": "admit rejected"}

    # a body where every optional collection is left at its default
    msg = ConductorMessage(
        message_type=ConductorMessageType.WORKER_GRAPHS_DONE,
        body=WorkerGraphsDone(request_id="r", worker_graph_ids=[], is_first_tp_rank=True),
    )
    body = roundtrip(msg).body
    assert body.persist_signals == {} and body.rx_info == []


def test_every_registered_tag_is_unique():
    tags = list(_TYPE_TO_TAG.values())
    assert len(tags) == len(set(tags))


# ----------------------------------------------------------------------
# reached-through-a-loose-field types
# ----------------------------------------------------------------------


def test_dataclass_reached_through_an_untyped_field_round_trips():
    """Regression: CudaIpcKVTransferInfo travels inside step_metadata (a bare
    ``dict``), so no annotation names it. Enumerating such types by hand is a
    standing trap — one gets added and it fails at runtime on whichever path
    first carries it — so the codec tags them self-describingly instead."""
    from mstar.engine.resources.kv.transfer import CudaIpcKVTransferInfo

    probe = CudaIpcKVTransferInfo(
        cuda_share=(1, 2), size=(3, 4), stride=(4, 1), offset=8,
        dtype="float16", requires_grad=False, layout=None,
    )
    fi = _fwd_info()
    fi.step_metadata = {"probe": probe, "n": 3, "nested": [probe, "x"]}
    msg = WorkerMessage(
        message_type=WorkerMessageType.INPUT_SIGNALS,
        body=InputSignals(request_id="r", partition_name="p",
                          request_info=fi, inputs=[]),
    )
    md = roundtrip(msg).body.request_info.step_metadata
    assert isinstance(md["probe"], CudaIpcKVTransferInfo)
    assert md["probe"].dtype == "float16"
    assert md["probe"].offset == 8
    assert md["n"] == 3
    assert isinstance(md["nested"][0], CudaIpcKVTransferInfo)
    assert md["nested"][1] == "x"


def test_required_field_that_is_none_survives():
    """Regression: omitting None keeps frames small, but only a field WITH a
    default can be reconstructed from its absence. A required field that is
    legitimately None has to go on the wire or the reconstruct raises."""
    from mstar.engine.resources.kv.transfer import CudaIpcKVTransferInfo

    probe = CudaIpcKVTransferInfo(
        cuda_share=(1,), size=(1,), stride=(1,), offset=0,
        dtype="f32", requires_grad=False, layout=None,  # required, and None
    )
    fi = _fwd_info()
    fi.step_metadata = {"probe": probe}
    msg = WorkerMessage(
        message_type=WorkerMessageType.INPUT_SIGNALS,
        body=InputSignals(request_id="r", partition_name="p",
                          request_info=fi, inputs=[]),
    )
    assert roundtrip(msg).body.request_info.step_metadata["probe"].layout is None


def test_none_in_a_field_with_a_non_none_default_survives():
    """Regression: an omitted None decoded to the field's default, so
    SamplingReqConfig(top_k=None) came back with top_k=0."""
    from mstar.engine.resources.sampler.config import SamplingReqConfig

    cfg = SamplingReqConfig(top_k=None, temperature=None)
    fi = _fwd_info()
    fi.resource_configs = {"sampler": cfg}
    msg = WorkerMessage(
        message_type=WorkerMessageType.INPUT_SIGNALS,
        body=InputSignals(request_id="r", partition_name="p",
                          request_info=fi, inputs=[]),
    )
    out = roundtrip(msg).body.request_info.resource_configs["sampler"]
    assert out.top_k is None
    assert out.temperature is None
    assert out.top_p == 1


def test_non_mstar_types_are_refused_rather_than_imported_from_the_wire():
    from dataclasses import dataclass

    import mstar.communication.wire as _wire

    @dataclass
    class Outsider:
        x: int

    Outsider.__module__ = "somewhere.else"
    with pytest.raises(_wire.WireError, match="not an mstar type"):
        _wire._encode_tagged(Outsider(x=1))


# --- splicing: a field encoded alone, spliced into a frame built elsewhere ---



# --- loosely-typed fields ------------------------------------------------------

def _conductor_request(model_kwargs, input_metadata=None):
    from mstar.utils.ipc_format import NewRequestConductor

    return ConductorMessage(
        message_type=ConductorMessageType.NEW_REQUEST,
        body=NewRequestConductor(
            request_id="r", initial_signals={}, initial_input_modalities=[],
            initial_output_modalities=[],
            input_metadata=input_metadata or {}, model_kwargs=model_kwargs,
        ),
    )


def test_untyped_model_kwargs_round_trip_exactly():
    """``model_kwargs`` holds whatever the client and model put there. A
    tuple, an int key, a set, a dtype or an array has to come back as itself
    -- and must not fail the send, which would drop the request silently."""
    import numpy as np

    kwargs = {
        "size": (512, 768),
        "ids": {3: "three"},
        "tags": {"a", "b"},
        "dtype": torch.float16,
        "mask": np.arange(4),
        "plain": {"k": [1, 2.5, "x", None, True]},
    }
    got = decode(encode(_conductor_request(kwargs))).body.model_kwargs

    assert got["size"] == (512, 768) and isinstance(got["size"], tuple)
    assert got["ids"] == {3: "three"}
    assert got["tags"] == {"a", "b"}
    assert got["dtype"] is torch.float16
    assert np.array_equal(got["mask"], np.arange(4))
    assert got["plain"] == kwargs["plain"]


def test_json_shaped_untyped_values_stay_plain_msgpack():
    """Only what msgpack cannot carry exactly needs extensions: plain data stays
    readable by a non-Python peer."""
    import msgpack

    from mstar.communication.wire import DATA_EXT

    def body_of(frame):
        # [envelope tag, {..., "body": [body tag, {fields}]}]
        _tag, envelope = msgpack.unpackb(frame, raw=False, strict_map_key=False)
        return envelope["body"][1]

    body = body_of(encode(_conductor_request({"steps": 30, "cfg": [1.5, 2.0]})))
    assert body["model_kwargs"] == {"steps": 30, "cfg": [1.5, 2.0]}

    body = body_of(encode(_conductor_request({"size": (1, 2)})))
    assert isinstance(body["model_kwargs"]["size"], msgpack.ExtType)
    assert body["model_kwargs"]["size"].code == DATA_EXT


def test_one_undecodable_frame_does_not_lose_the_batch():
    """A drained batch can carry control frames for other requests; one bad
    payload must cost only itself."""
    from mstar.communication.codec import WireCodec, decode_each

    good = WireCodec.encode(_conductor_request({}))
    got = decode_each(WireCodec, [good, b"\xc1not msgpack", good], "test")
    assert len(got) == 2


# --- union branches -----------------------------------------------------------

def _round_trip(value, hint):
    import msgpack

    from mstar.communication import wire
    raw = msgpack.unpackb(
        msgpack.packb(wire._encoder(hint)(value), use_bin_type=True),
        raw=False, strict_map_key=False,
    )
    return wire._decoder(hint)(raw)


@pytest.mark.parametrize("value, hint", [
    # The first branch used to claim everything: a plain type passed the list
    # through, so the tuple came back a list -- and as a key, raised
    # unhashable.
    ({3: 1, ("a", "b"): 2}, dict[int | tuple[str, str], int]),
    ((1, 2), int | tuple[int, int]),
    (7, int | tuple[int, int]),
    # And a container iterated a string: "ab" came back ('a', 'b').
    ("ab", tuple[str, str] | str),
    (("a", "b"), tuple[str, str] | str),
    ([1, 2], int | list[int]),
    ({"k": 1}, list[int] | dict[str, int]),
    ({1, 2}, str | set[int]),
    (b"x", str | bytes),
    (True, bool | int),
    (5, bool | int),
    # Lenient where Python is: an int is a fine float.
    (2, float | str),
    ([(1, "a")], list[tuple[int, str]] | str),
])
def test_a_union_decodes_to_the_branch_that_was_sent(value, hint):
    got = _round_trip(value, hint)
    assert got == value
    assert type(got) is type(value)


def test_the_wrong_arity_is_not_a_match():
    # (1, 2, 3) is no tuple[int, int]; the list branch is.
    got = _round_trip([1, 2, 3], tuple[int, int] | list[int])
    assert got == [1, 2, 3] and type(got) is list


def test_outside_a_union_nothing_is_checked():
    """Strictness is for choosing a branch. A plain field keeps passing its
    value through, so a sender that was loose about types -- None in a field
    declared int -- still round-trips as it did."""
    from mstar.communication import wire
    assert _round_trip(None, int) is None
    assert wire._decoder(int)("not an int") == "not an int"


# --- fields encoded on their own -------------------------------------------

def test_a_field_encoded_alone_matches_the_field_encoded_in_place():
    """``encode_field`` is what lets a Rust sender stay ignorant of the type.

    It encodes what it owns and splices these in as opaque values, so it never
    has to know CurrentForwardPassInfo or grow a case for every new
    PublishedInfo subclass. That only holds if encoding the field on its own
    gives exactly what encoding the whole message would have put there.
    """
    import msgpack

    from mstar.communication.wire import encode_field
    from mstar.utils.ipc_format import InputSignals

    info = CurrentForwardPassInfo(
        request_id="r1", graph_walk="decode", fwd_index=3,
        random_seed=7, max_tokens=16, partition_name="default",
    )
    msg = InputSignals(request_id="r1", inputs=[], request_info=info)

    in_place = msgpack.unpackb(encode(msg), raw=False,
                               strict_map_key=False)[1]["request_info"]
    alone = msgpack.unpackb(
        encode_field(info, CurrentForwardPassInfo), raw=False,
        strict_map_key=False,
    )
    assert alone == in_place
    # And the spliced frame decodes back to an equal message.
    assert decode(encode(msg)).request_info == info


def test_a_polymorphic_field_keeps_its_tag_when_encoded_alone():
    """``resource_publish_info`` is dict[str, PublishedInfo] -- abstract, so
    each value rides as [tag, payload]. A Rust sender splices it verbatim
    rather than learning the subclasses."""
    import msgpack

    from mstar.communication.wire import encode_field
    from mstar.engine.resources.base import PublishedInfo
    from mstar.utils.ipc_format import WorkerGraphsDone

    published = {
        tag: cls for cls, tag in _TYPE_TO_TAG.items()
        if isinstance(cls, type) and issubclass(cls, PublishedInfo)
    }
    tag, cls = next(iter(sorted(published.items())))
    value = {"kv": cls()}

    msg = WorkerGraphsDone(
        request_id="r1", worker_graph_ids=[0], is_first_tp_rank=True,
        resource_publish_info=value,
    )
    in_place = msgpack.unpackb(encode(msg), raw=False,
                               strict_map_key=False)[1]["resource_publish_info"]
    alone = msgpack.unpackb(
        encode_field(value, dict[str, PublishedInfo]), raw=False,
        strict_map_key=False,
    )
    assert alone == in_place
    assert alone["kv"][0] == tag, "the subclass tag has to survive"


@pytest.mark.parametrize("value", [
    torch.Size([2, 3]),
    torch.Size([]),
    torch.Size([-1, 3]),
    frozenset({1, (2, 3)}),
    {(1, "x"): {"nested": [(1, 2), {3, 4}]}},
    torch.int64,
])
def test_dynamic_containers_preserve_their_types(value):
    got = decode(encode(value))
    assert type(got) is type(value)
    assert got == value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA IPC needs a GPU")
def test_real_cuda_ipc_transfer_descriptor_round_trips():
    import subprocess

    from mstar.engine.resources.kv.cache import KVCache
    from mstar.engine.resources.kv.config import PagedKVConfig
    from mstar.engine.resources.kv.transfer import CudaIpcKVTransferEngine

    config = PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=8,
        max_seq_len=16, max_num_pages=4, page_size=8,
    )
    cache = KVCache(config, torch.device("cuda:0"))
    cache.chunk_view(layer_idx=0, page_idx=1, token_start=0, token_end=8).fill_(3.0)
    expected = cache.chunk_view(layer_idx=0, page_idx=1, token_start=0, token_end=8).float().sum().item()
    transfer = CudaIpcKVTransferEngine(cache)
    try:
        info = transfer.get_kv_transfer_info()
        frame = encode(_conductor_request({"transfer": info}))
        assert decode(frame).body.model_kwargs["transfer"] == info
        # A real CUDA IPC open needs a second process.
        pythonpath = os.pathsep.join([".", os.environ.get("PYTHONPATH", "")])
        reader = subprocess.run(
            [sys.executable, "-c", _CUDA_IPC_READER], input=frame, capture_output=True,
            check=False, timeout=120, env={**os.environ, "PYTHONPATH": pythonpath},
        )
        assert reader.returncode == 0, reader.stderr.decode()[-2000:]
        assert float(reader.stdout.decode().split()[-1]) == expected
    finally:
        transfer.shutdown()


_CUDA_IPC_READER = """
import sys, torch
import mstar.communication.wire_types
from mstar.communication.wire import decode
from mstar.engine.resources.kv.cache import KVCache
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.transfer import CudaIpcKVTransferEngine, KVReadInfo

info = decode(sys.stdin.buffer.read()).body.model_kwargs["transfer"]
config = PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=16, max_num_pages=4, page_size=8)
cache = KVCache(config, torch.device("cuda:0"))
engine = CudaIpcKVTransferEngine(cache)
read = KVReadInfo(layer_idx=0, local_page_idx=2, remote_page_idx=1, token_start=0, token_end=8)
engine.read_batched_async(info, [read])
engine.shutdown()
torch.cuda.synchronize()
print(cache.chunk_view(layer_idx=0, page_idx=2, token_start=0, token_end=8).float().sum().item())
"""


@pytest.mark.parametrize("dtype", ["int16", ">f4", "complex128", "bool", "S3", "U3", "datetime64[ns]"])
@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
def test_numpy_arrays_preserve_dtype_shape_and_values(dtype, shape):
    import numpy as np

    value = np.ones(shape, dtype=dtype)
    got = decode(encode(value))
    assert type(got) is np.ndarray
    assert got.dtype == value.dtype
    assert got.shape == value.shape
    np.testing.assert_array_equal(got, value)


def test_noncontiguous_numpy_array_and_numpy_scalar():
    import numpy as np

    value = np.arange(24).reshape(4, 6).T[::2]
    np.testing.assert_array_equal(decode(encode(value)), value)
    value = np.float32(1.5)
    got = decode(encode(value))
    assert type(got) is np.float32
    assert got == value


@pytest.mark.parametrize("dtype", [
    [("x", "i4"), ("y", "f8")],                           # structured
    [("x", np.dtype("i4", metadata={"unit": "m"}))],      # nested field metadata
    np.dtype("f4", metadata={"unit": "m"}),                # metadata
    "V0",                                                  # zero itemsize
])
def test_numpy_dtypes_that_cannot_round_trip_are_refused_at_send(dtype):
    """A refused value fails at the sender; one the receiver can't rebuild would
    instead drop the whole frame and leave its request waiting."""
    with pytest.raises(WireError):
        encode(np.zeros(2, dtype=np.dtype(dtype)))


# --- values a /generate/ws client or warmup YAML can produce ------------------

@pytest.mark.parametrize("value", [
    msgpack.Timestamp(1_700_000_000, 123),
    msgpack.ExtType(5, b"client bytes"),
    datetime.date(2024, 1, 2),
    datetime.datetime(2024, 1, 2, 3, 4, 5, 6),
    datetime.datetime(2024, 1, 2, 3, 4, 5, tzinfo=datetime.timezone(datetime.timedelta(hours=-7))),
])
def test_client_and_yaml_values_round_trip(value):
    got = decode(encode(_conductor_request({"v": value}))).body.model_kwargs["v"]
    assert type(got) is type(value)
    assert got == value


@pytest.mark.parametrize("code", [80, 81])
def test_client_extensions_with_wire_codes_stay_opaque(code):
    """A client's own ext 80 or 81 is data; it is never read as legacy pickle or
    as one of the wire's own data extensions."""
    value = msgpack.ExtType(code, b"client bytes")
    assert decode(encode(_conductor_request({"v": value}))).body.model_kwargs["v"] == value


def test_warmup_yaml_dates_round_trip():
    import yaml

    kwargs = yaml.safe_load("day: 2024-01-02\nat: 2024-01-02 03:04:05+07:00\n")
    assert decode(encode(_conductor_request(kwargs))).body.model_kwargs == kwargs


def test_datetime_with_zone_rules_is_refused():
    from zoneinfo import ZoneInfo

    value = datetime.datetime(2024, 1, 2, tzinfo=ZoneInfo("America/Los_Angeles"))
    with pytest.raises(WireError, match="fixed-offset"):
        encode(_conductor_request({"v": value}))


def test_kv_transfer_descriptors_round_trip():
    from mstar.engine.resources.kv.config import KVLayout
    from mstar.engine.resources.kv.transfer import (
        MooncakeKVTransferInfo,
        ShmKVSnapshotChunk,
        ShmKVTransferInfo,
    )

    layout = next(iter(KVLayout))
    for info in (
        ShmKVTransferInfo(path="/dev/shm/kv", page_indices=(1, 2), layout=layout,
                          chunks=(ShmKVSnapshotChunk(path="/dev/shm/c0", start_len=0, end_len=8, page_indices=(1,)),)),
        MooncakeKVTransferInfo(entity_id="w0", session_id="h:1", data_ptr=0xDEAD, layout=layout),
    ):
        assert decode(encode(_conductor_request({"kv": info}))).body.model_kwargs["kv"] == info


def test_auto_tagged_messages_stay_self_describing(monkeypatch):
    """The first send of an unregistered mstar dataclass auto-registers its tag;
    a later send must still decode on a peer that never saw the class."""
    import dataclasses

    from mstar.communication import wire
    from mstar.engine.resources.kv.transfer import KVReadInfo

    msg = KVReadInfo(**{f.name: 0 for f in dataclasses.fields(KVReadInfo)})
    encode(msg)
    frame = encode(msg)
    assert msgpack.unpackb(frame, raw=False, strict_map_key=False)[0] == wire.DATA
    monkeypatch.delitem(wire._TAG_TO_TYPE, wire._TYPE_TO_TAG[KVReadInfo])
    monkeypatch.delitem(wire._TYPE_TO_TAG, KVReadInfo)
    assert decode(frame) == msg


def test_per_pass_kv_publish_info_takes_the_typed_path():
    """The conductor re-sends KV publish info in every INPUT_SIGNALS; its
    transfer descriptor must travel as [tag, fields], not a data extension."""
    from mstar.conductor.request_info import CurrentForwardPassInfo
    from mstar.engine.resources.kv.config import KVLayout
    from mstar.engine.resources.kv.manager import KVSequenceInfo, PublishedKVInfo
    from mstar.engine.resources.kv.transfer import MooncakeKVTransferInfo, ShmKVTransferInfo
    from mstar.utils.ipc_format import InputSignals, WorkerMessage, WorkerMessageType

    def extensions(value):
        if isinstance(value, msgpack.ExtType):
            return 1
        if isinstance(value, list):
            return sum(extensions(v) for v in value)
        if isinstance(value, dict):
            return sum(extensions(k) + extensions(v) for k, v in value.items())
        return 0

    layout = next(iter(KVLayout))
    for info in (
        ShmKVTransferInfo(path="/dev/shm/kv", page_indices=(1, 2), layout=layout),
        MooncakeKVTransferInfo(entity_id="w0", session_id="h:1", data_ptr=0xDEAD, layout=layout),
        None,
    ):
        seq = KVSequenceInfo(seq_len=4, latest_kv_transfer_info=info, page_indices=[1, 2])
        msg = WorkerMessage(
            message_type=WorkerMessageType.INPUT_SIGNALS,
            body=InputSignals(request_id="r", inputs=[], request_info=CurrentForwardPassInfo(
                request_id="r", graph_walk="decode", fwd_index=1, random_seed=0, max_tokens=8,
                resource_publish_info={"kv": PublishedKVInfo(info={0: {"kv": seq}})},
            )),
        )
        frame = encode(msg)
        assert extensions(msgpack.unpackb(frame, raw=False, strict_map_key=False)) == 0
        got = decode(frame).body.request_info.resource_publish_info["kv"].info[0]["kv"]
        assert got == seq
