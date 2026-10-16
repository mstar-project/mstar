"""Control-plane decoding must not invoke callables supplied by a frame."""

import collections
import pickle
import socket
import time

import msgpack
import pytest

import mstar.communication.wire_types  # noqa: F401 (registers the tags)
from mstar.communication import wire
from mstar.communication.codec import WireCodec, decode_each, default_codec
from mstar.utils.ipc_format import ConductorMessage, ConductorMessageType, NewRequestConductor

_CALLS = []


def _record_call():
    _CALLS.append("called")
    return "unexpected result"


class _CallablePayload:
    def __reduce__(self):
        return _record_call, ()


def _request():
    return ConductorMessage(
        message_type=ConductorMessageType.NEW_REQUEST,
        body=NewRequestConductor(
            request_id="r", initial_signals={}, initial_input_modalities=["text"],
            initial_output_modalities=["text"], input_metadata={}, model_kwargs={},
        ),
    )


def _frame(path):
    payload = pickle.dumps(_CallablePayload())
    if path == "opaque":
        return WireCodec, msgpack.packb([wire.OPAQUE, payload], use_bin_type=True)
    raw = msgpack.unpackb(WireCodec.encode(_request()), raw=False)
    value = msgpack.ExtType(wire.PICKLED_EXT, payload)
    raw[1]["body"][1]["model_kwargs"] = {"nested": [value]} if path == "nested" else value
    return WireCodec, msgpack.packb(raw, use_bin_type=True)


@pytest.mark.parametrize("path", ["opaque", "metadata", "nested"])
def test_callable_payload_is_rejected_without_execution(path):
    codec, frame = _frame(path)
    _CALLS.clear()
    try:
        with pytest.raises(wire.WireError):
            codec.decode(frame)
    finally:
        assert _CALLS == []


@pytest.mark.parametrize("path", ["opaque", "metadata", "nested"])
def test_rejected_frame_does_not_drop_healthy_neighbors(path):
    _, rejected = _frame(path)
    good = WireCodec.encode(_request())
    _CALLS.clear()
    decoded = decode_each(WireCodec, [good, rejected, good], "test")
    assert _CALLS == []
    assert [message.body.request_id for message in decoded] == ["r", "r"]


def test_pickle_codec_is_rejected_at_startup(monkeypatch):
    monkeypatch.setenv("MSTAR_WIRE_CODEC", "pickle")
    with pytest.raises(ValueError, match="MSTAR_WIRE_CODEC must be msgpack"):
        default_codec()


def test_unsupported_objects_do_not_call_reduce_at_send_time():
    class Unsupported:
        def __reduce__(self):
            pytest.fail("the encoder must not invoke a pickle reducer")

    with pytest.raises(wire.WireError, match="unsupported wire value type"):
        WireCodec.encode(Unsupported())


def test_tensors_in_control_metadata_point_to_tensor_edges():
    import torch

    message = _request()
    message.body.model_kwargs = {"x": torch.ones(2)}
    with pytest.raises(wire.WireError, match="model_kwargs: x: .*PREPROCESS_ONLY_KWARGS"):
        WireCodec.encode(message)


@pytest.mark.parametrize("value", [
    collections.namedtuple("Point", "x y")(1, 2),
    collections.OrderedDict(a=1),
    collections.defaultdict(list, a=[1]),
    type("Items", (list,), {})([1]),
])
def test_container_subclasses_are_refused_at_send(value):
    message = _request()
    message.body.model_kwargs = {"v": value}
    with pytest.raises(wire.WireError, match="model_kwargs: v: unsupported wire value type"):
        WireCodec.encode(message)


def test_object_arrays_are_rejected_on_encode_and_decode():
    import numpy as np

    with pytest.raises(wire.WireError, match="object arrays"):
        WireCodec.encode(np.array([_CallablePayload()], dtype=object))
    payload = msgpack.packb(["numpy_array", ["|O", [1], b"\0" * 8]], use_bin_type=True)
    frame = msgpack.packb([wire.DATA, msgpack.ExtType(wire.DATA_EXT, payload)], use_bin_type=True)
    with pytest.raises(wire.WireError, match="object arrays"):
        WireCodec.decode(frame)


@pytest.mark.parametrize("shape, data", [([2], b"\0" * 4), ([-1], b""), ([True], b"\0" * 4)])
def test_numpy_extension_validates_shape_and_data_size(shape, data):
    payload = msgpack.packb(["numpy_array", ["<i4", shape, data]], use_bin_type=True)
    frame = msgpack.packb([wire.DATA, msgpack.ExtType(wire.DATA_EXT, payload)], use_bin_type=True)
    with pytest.raises(wire.WireError, match="NumPy shape"):
        WireCodec.decode(frame)


@pytest.mark.parametrize("dtype_str", ["i4,f8", "(2,)i4", "V0", 7])
def test_receiver_refuses_dtypes_the_sender_never_produces(dtype_str):
    payload = msgpack.packb(["numpy_array", [dtype_str, [2], b"\0" * 8]], use_bin_type=True)
    frame = msgpack.packb([wire.DATA, msgpack.ExtType(wire.DATA_EXT, payload)], use_bin_type=True)
    with pytest.raises(wire.WireError):
        wire.decode(frame)


@pytest.mark.parametrize("shape", [[True], ["2"], 2])
def test_rejects_malformed_torch_size(shape):
    payload = msgpack.packb(["torch_size", shape], use_bin_type=True)
    frame = msgpack.packb([wire.DATA, msgpack.ExtType(wire.DATA_EXT, payload)], use_bin_type=True)
    with pytest.raises(wire.WireError):
        wire.decode(frame)


@pytest.mark.parametrize("kind, value", [
    ("msgpack_timestamp", ["1", 0]),
    ("msgpack_timestamp", [1, -1]),
    ("msgpack_ext", [300, b"x"]),
    ("msgpack_ext", [5, "not bytes"]),
    ("datetime", 7),
    ("date", "not a date"),
])
def test_rejects_malformed_client_value_kinds(kind, value):
    payload = msgpack.packb([kind, value], use_bin_type=True)
    frame = msgpack.packb([wire.DATA, msgpack.ExtType(wire.DATA_EXT, payload)], use_bin_type=True)
    with pytest.raises(wire.WireError):
        wire.decode(frame)


@pytest.mark.parametrize("tag", [
    "auto:mstar.communication.wire:encode",  # an mstar function, not a dataclass
    "auto:os:system",                        # a module outside mstar
    "auto:mstar.graph.base:torch._guards.CompileId",  # a non-mstar class via an import
])
def test_hostile_auto_tags_cannot_resolve_callables(tag):
    with pytest.raises(wire.WireError):
        wire._decode_tagged([tag, {}])


def _data_frame(kind, value):
    payload = msgpack.packb([kind, value], use_bin_type=True)
    return msgpack.packb([wire.DATA, msgpack.ExtType(wire.DATA_EXT, payload)], use_bin_type=True)


def test_legacy_pickle_nested_inside_a_data_extension_is_rejected():
    legacy = msgpack.ExtType(wire.PICKLED_EXT, pickle.dumps(_CallablePayload()))
    _CALLS.clear()
    with pytest.raises(wire.WireError, match="pickle"):
        wire.decode(_data_frame("tuple", [1, legacy]))
    assert _CALLS == []


def test_auto_tagged_dataclass_with_missing_fields_is_refused():
    tag = "auto:mstar.engine.resources.kv.transfer:KVReadInfo"
    with pytest.raises(wire.WireError, match="malformed"):
        wire.decode(_data_frame("dataclass", [tag, {"layer_idx": 0}]))


def test_deeply_nested_extensions_drop_only_their_frame():
    value = msgpack.ExtType(wire.DATA_EXT, msgpack.packb(["tuple", []], use_bin_type=True))
    for _ in range(5000):
        value = msgpack.ExtType(wire.DATA_EXT, msgpack.packb(["tuple", [value]], use_bin_type=True))
    deep = msgpack.packb([wire.DATA, value], use_bin_type=True)
    with pytest.raises(wire.WireError):
        wire.decode(deep)
    good = WireCodec.encode(_request())
    assert [m.body.request_id for m in decode_each(WireCodec, [good, deep, good], "test")] == ["r", "r"]


@pytest.mark.parametrize("transport", ["python", "rust"])
@pytest.mark.parametrize("protocol", ["IPC", "TCP"])
@pytest.mark.parametrize("path", ["opaque", "metadata", "nested"])
def test_socket_receiver_rejects_callable_and_delivers_healthy_messages(
    transport, protocol, path, tmp_path, monkeypatch,
):
    import zmq

    from mstar.communication.communicator import BaseCommunicator, ZMQCommunicator

    monkeypatch.setenv("MSTAR_ZMQ_TRANSPORT", protocol)
    monkeypatch.setenv("MSTAR_ZMQ_TCP_HOST", "127.0.0.1")
    if transport == "rust":
        pytest.importorskip("mstar_rust")
        from mstar.communication.rust_communicator import RustZMQCommunicator

        cls = RustZMQCommunicator
    else:
        cls = ZMQCommunicator
    if protocol == "TCP":
        with socket.socket() as port_probe:
            port_probe.bind(("127.0.0.1", 0))
            port = port_probe.getsockname()[1]
        monkeypatch.setattr(BaseCommunicator, "_tcp_port", staticmethod(lambda _: port))
    receiver = cls("receiver", push_ids=[], ipc_socket_path_prefix=str(tmp_path), codec=WireCodec)
    sender = zmq.Context.instance().socket(zmq.PUSH)
    sender.setsockopt(zmq.SNDTIMEO, 1000)
    sender.connect(receiver._endpoint("receiver"))
    _, rejected = _frame(path)
    good = WireCodec.encode(_request())
    _CALLS.clear()
    try:
        for frame in (good, rejected, good):
            sender.send(frame)
        decoded = []
        deadline = time.monotonic() + 3
        while len(decoded) < 2 and time.monotonic() < deadline:
            decoded.extend(receiver.get_all_new_messages(blocking=True, timeout_s=0.1))
        assert _CALLS == []
        assert [message.body.request_id for message in decoded] == ["r", "r"]
    finally:
        sender.close(linger=0)
        if transport == "python":
            receiver.pull_socket.close(linger=0)
