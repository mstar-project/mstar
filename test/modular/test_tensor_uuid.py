"""Tensor uuids are ints, and the entity prefix is what keeps them unique.

A bare per-process counter would not be safe: the *receiving* worker keys its
own maps by the sender's uuid, so two senders minting the same value hand back
the wrong tensor — silently.
"""
import sys

sys.path.insert(0, ".")

import pytest
import torch

import mstar.communication.wire_types  # noqa: F401  (registers the tags)
from mstar.communication import wire
from mstar.communication.tensor_uuid import (
    COUNTER_BITS,
    COUNTER_MASK,
    TensorUuidMinter,
    counter_of,
    entity_index,
    owner_index,
)
from mstar.graph.base import TensorPointerInfo
from mstar.utils.ipc_format import (
    TensorReceived,
    UnpersistTensors,
    WorkerMessage,
    WorkerMessageType,
)


def test_entity_prefix_makes_uuids_globally_unique():
    """The property the whole scheme exists for."""
    minters = {
        eid: TensorUuidMinter(eid)
        for eid in ("worker_0", "worker_1", "worker_7", "api_server_preprocess_worker")
    }
    seen = set()
    for _ in range(50):
        for eid, m in minters.items():
            u = m.mint()
            assert u not in seen, f"{eid} minted a uuid another entity already used"
            seen.add(u)


def test_uuid_decomposes():
    m = TensorUuidMinter("worker_5")
    u = m.mint()
    assert owner_index(u) == 6  # worker_N -> N + 1; index 0 is the data worker
    assert counter_of(u) == 1
    assert counter_of(m.mint()) == 2


def test_counter_starts_at_one_so_zero_is_never_a_valid_uuid():
    """0 is a natural 'unset' value in Rust and on the wire; keeping it out of
    the value space means an uninitialised handle cannot alias a real tensor."""
    for eid in ("api_server_preprocess_worker", "worker_0"):
        assert TensorUuidMinter(eid).mint() != 0


def test_non_production_entities_get_distinct_process_local_indices():
    """Tests and fixtures use synthetic entity ids. They must still be unique
    within the process — a hash of the name could collide, and a uuid
    collision corrupts data silently."""
    from mstar.communication.tensor_uuid import _DYNAMIC_BASE

    a, b = entity_index("some_test_entity"), entity_index("another_test_entity")
    assert a != b
    assert a >= _DYNAMIC_BASE and b >= _DYNAMIC_BASE
    # stable within the process
    assert entity_index("some_test_entity") == a
    # and disjoint from the production range
    assert entity_index("worker_0") < _DYNAMIC_BASE
    assert entity_index("api_server_preprocess_worker") < _DYNAMIC_BASE


def test_synthetic_and_production_entities_do_not_collide():
    minters = [TensorUuidMinter(e) for e in
               ("worker_0", "worker_1", "api_server_preprocess_worker", "w0", "f_0")]
    seen = set()
    for _ in range(20):
        for m in minters:
            u = m.mint()
            assert u not in seen
            seen.add(u)


def test_rank_beyond_the_stable_range_is_rejected():
    with pytest.raises(ValueError, match="exceeds the stable entity range"):
        entity_index("worker_999999")


def test_counter_exhaustion_raises():
    m = TensorUuidMinter("worker_0")
    m._counter = COUNTER_MASK
    with pytest.raises(RuntimeError, match="exhausted"):
        m.mint()


def test_uuid_fits_in_u64():
    m = TensorUuidMinter("worker_1")
    u = m.mint()
    assert 0 < u < (1 << 64)
    assert u >> COUNTER_BITS == owner_index(u)


# ----------------------------------------------------------------------
# the wire
# ----------------------------------------------------------------------


def _roundtrip(msg):
    return wire.decode(wire.encode(msg))


def test_int_uuids_survive_the_wire_in_tensor_pointer_info():
    m = TensorUuidMinter("worker_2")
    u = m.mint()
    info = TensorPointerInfo(
        dims=[4, 8], dtype=torch.float16, nbytes=64, address=1, stride=(8, 1),
        uuid=u, source_session_id="h:1", source_entity="worker_2",
    )
    from mstar.conductor.request_info import CurrentForwardPassInfo
    from mstar.graph.base import GraphEdge
    from mstar.utils.ipc_format import InputSignals

    msg = WorkerMessage(
        message_type=WorkerMessageType.INPUT_SIGNALS,
        body=InputSignals(
            request_id="r", partition_name="p",
            request_info=CurrentForwardPassInfo(
                request_id="r", fwd_index=0, random_seed=0, max_tokens=1,
                graph_walk="w", partition_name="p"),
            inputs=[GraphEdge(next_node="n", name="s", tensor_info=[info])],
        ),
    )
    got = _roundtrip(msg).body.inputs[0].tensor_info[0]
    assert got.uuid == u
    assert isinstance(got.uuid, int)


def test_int_keyed_uuid_maps_survive_the_wire():
    """msgpack has no int map keys in the strict sense; these go over as
    list-of-pairs and have to come back as an int-keyed dict."""
    m = TensorUuidMinter("worker_4")
    u1, u2 = m.mint(), m.mint()

    got = _roundtrip(WorkerMessage(
        message_type=WorkerMessageType.TENSOR_RECEIVED,
        body=TensorReceived(request_id="r", successful_tensors={u1: 2, u2: 1},
                            failed_tensor_ids=[u2]),
    )).body
    assert got.successful_tensors == {u1: 2, u2: 1}
    assert all(isinstance(k, int) for k in got.successful_tensors)
    assert got.failed_tensor_ids == [u2]

    got = _roundtrip(WorkerMessage(
        message_type=WorkerMessageType.UNPERSIST_TENSORS,
        body=UnpersistTensors(request_id="r", uuid_to_ref_count={u1: 3}),
    )).body
    assert got.uuid_to_ref_count == {u1: 3}


def test_int_uuids_are_smaller_on_the_wire_than_uuid4_strings():
    """The secondary reason for the change: 36 bytes per tensor per frame."""
    m = TensorUuidMinter("worker_0")
    infos = [
        TensorPointerInfo(
            dims=[8], dtype=torch.float32, nbytes=32, address=0, stride=(1,),
            uuid=m.mint(), source_session_id="h:1", source_entity="worker_0",
        )
        for _ in range(8)
    ]
    msg = WorkerMessage(
        message_type=WorkerMessageType.UNPERSIST_TENSORS,
        body=UnpersistTensors(request_id="r",
                              uuid_to_ref_count={i.uuid: 1 for i in infos}),
    )
    int_bytes = len(wire.encode(msg))
    # the same map keyed by uuid4 strings, for reference
    import uuid as _uuid
    str_msg = wire.encode(WorkerMessage(
        message_type=WorkerMessageType.UNPERSIST_TENSORS,
        body=UnpersistTensors(request_id="r",
                              uuid_to_ref_count={str(_uuid.uuid4()): 1 for _ in infos}),
    ))
    assert int_bytes < len(str_msg)
