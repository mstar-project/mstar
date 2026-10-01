"""The Rust TensorBookkeeping must be a drop-in for the Python one.

Both back the same ``TensorStore``, so every case runs against both and the
answers have to agree. A divergence here is not a crash: a refcount that drops
early frees a tensor a peer is still reading, and one that never drops leaks
until the request ends.
"""
import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.communication.tensor_store import PythonTensorBookkeeping, TensorStore
from mstar.graph.base import TensorPointerInfo

# Probe the EXTENSION, not the module that wraps it: the wrapper imports
# fine either way now (the mstar_rust import inside it is lazy), so probing
# the wrapper would fail rather than skip where the extension is not built.
pytest.importorskip(
    "mstar_rust",
    reason="mstar_rust not built (maturin develop --release in rust/)",
)
from mstar.communication.tensor_store import RustTensorBookkeeping


@pytest.fixture(params=["python", "rust"])
def bk(request):
    return (
        PythonTensorBookkeeping()
        if request.param == "python"
        else RustTensorBookkeeping()
    )


def _info(uuid: int, **over) -> TensorPointerInfo:
    base = dict(
        dims=[4, 8], dtype=torch.float16, nbytes=64, address=0x1000,
        stride=(8, 1), uuid=uuid, source_session_id="host:1",
        source_entity="worker_0",
    )
    base.update(over)
    return TensorPointerInfo(**base)


def _tracked(bk, uuid: int) -> bool:
    # A record and its descriptor come and go together.
    return bk.get_info(uuid) is not None


# --- reference state ---------------------------------------------------------

def test_a_fresh_tensor_is_collectable(bk):
    bk.put_tensor(1, _info(1))
    assert _tracked(bk, 1)
    assert bk.can_gc(1), "nothing holds it yet"


def test_an_untracked_uuid_is_never_collectable(bk):
    # Distinct from "tracked with no refs": an already-collected tensor must
    # not report collectable a second time and get freed twice.
    assert not bk.can_gc(99)
    assert not _tracked(bk, 99)
    assert not bk.is_registered(99)
    assert bk.get_info(99) is None


def test_mutating_an_untracked_uuid_is_a_no_op(bk):
    # A late TENSOR_RECEIVED can arrive for a tensor already collected.
    bk.increment_ref(42, 3)
    bk.dereference(42, 1)
    bk.set_persist(42, True)
    bk.set_mem_registered(42, True)
    assert not _tracked(bk, 42)


def test_references_gate_collection(bk):
    bk.put_tensor(1, _info(1))
    bk.increment_ref(1, 2)
    assert not bk.can_gc(1)
    bk.dereference(1, 1)
    assert not bk.can_gc(1)
    bk.dereference(1, 1)
    assert bk.can_gc(1)


def test_persist_holds_a_tensor_with_no_references(bk):
    bk.put_tensor(1, _info(1))
    bk.set_persist(1, True)
    assert not bk.can_gc(1), "the conductor still needs it"
    bk.set_persist(1, False)
    assert bk.can_gc(1)


def test_dereference_accepts_a_negative_delta(bk):
    # set_output_ref_counts corrects downward from the safety hold this way.
    bk.put_tensor(1, _info(1))
    bk.dereference(1, -2)
    assert not bk.can_gc(1), "a negative dereference is an increment"


def test_put_tensor_resets_reference_state(bk):
    # put_tensor on a live uuid means a NEW tensor; stale refs must not carry
    # over and pin it forever.
    bk.put_tensor(1, _info(1))
    bk.increment_ref(1, 5)
    bk.set_persist(1, True)
    bk.put_tensor(1, _info(1))
    assert bk.can_gc(1)


def test_mem_registered_does_not_hold_a_reference(bk):
    bk.put_tensor(1, _info(1))
    bk.set_mem_registered(1, True)
    assert bk.is_registered(1)
    assert bk.can_gc(1)


def test_forget_drops_the_descriptor_too(bk):
    bk.put_tensor(1, _info(1))
    bk.forget_tensor(1)
    assert not _tracked(bk, 1)
    assert bk.get_info(1) is None




# --- descriptors -------------------------------------------------------------

def test_a_descriptor_round_trips(bk):
    info = _info(
        7, dims=[2, 3], dtype=torch.bfloat16, nbytes=12, address=0xDEAD,
        stride=(3, 1), offset=16, source_tp_size=4, source_tp_rank=2,
        shm_segment="seg-3", shm_offset=64,
        _source_node_name="prefill", _source_graph_walk="decode",
    )
    bk.put_tensor(7, info)
    got = bk.get_info(7)

    assert got.uuid == 7
    assert list(got.dims) == [2, 3]
    assert got.dtype is torch.bfloat16, "dtype must survive as a torch.dtype"
    assert got.nbytes == 12
    assert got.address == 0xDEAD
    assert tuple(got.stride) == (3, 1)
    assert got.offset == 16
    assert got.source_tp_size == 4 and got.source_tp_rank == 2
    assert got.source_session_id == "host:1"
    assert got.source_entity == "worker_0"
    assert got.shm_segment == "seg-3" and got.shm_offset == 64
    assert got._source_node_name == "prefill"
    assert got._source_graph_walk == "decode"


def test_an_absent_shm_segment_stays_none(bk):
    # None means "not the arena transport"; an empty string would read as a
    # segment named "".
    bk.put_tensor(1, _info(1))
    got = bk.get_info(1)
    assert got.shm_segment is None
    assert got._source_node_name is None




# --- batch forms -------------------------------------------------------------

def test_batch_forms_match_the_single_ones(bk):
    bk.put_tensor_batch([1, 2], [_info(1), _info(2)])
    assert _tracked(bk, 1) and _tracked(bk, 2)

    bk.increment_ref_batch([1, 2], [1, 3])
    assert not bk.can_gc(1) and not bk.can_gc(2)
    bk.dereference_batch([1, 2], [1, 3])
    assert bk.can_gc(1) and bk.can_gc(2)



def test_dereference_batch_uniform_reports_what_it_freed(bk):
    """The whole point of the batched form: the caller learns which tensors
    became collectable without asking per uuid, and gets mem_registered with
    them -- the flag a teardown still needs after the record is gone."""
    bk.put_tensor_batch([1, 2, 3], [_info(1), _info(2), _info(3)])
    bk.increment_ref_batch([1, 2, 3], [1, 2, 1])
    bk.set_mem_registered(3, True)

    collectable, registered = bk.dereference_batch_uniform([1, 2, 3], 1)
    assert collectable == [1, 3], "2 still holds a reference"
    assert registered == [False, True]
    # No cleanup: the records are still there, exactly as can_gc reports them.
    assert _tracked(bk, 1) and bk.can_gc(1)


def test_dereference_batch_uniform_skips_persisted_and_untracked(bk):
    bk.put_tensor_batch([1, 2], [_info(1), _info(2)])
    bk.increment_ref_batch([1, 2], [1, 1])
    bk.set_persist(2, True)

    collectable, _ = bk.dereference_batch_uniform([1, 2, 99], 1)
    assert collectable == [1], "the conductor still needs 2, and 99 is not ours"


def test_dereference_batch_uniform_cleans_up_in_the_same_pass(bk):
    """``cleanup`` is what saves the second crossing to forget each one."""
    bk.put_tensor_batch([1, 2], [_info(1), _info(2)])
    bk.increment_ref_batch([1, 2], [1, 2])

    collectable, _ = bk.dereference_batch_uniform([1, 2], 1, True)
    assert collectable == [1]
    assert not _tracked(bk, 1), "the freed record should be gone"
    assert bk.get_info(1) is None, "and its descriptor with it"
    assert _tracked(bk, 2), "the held one must survive"




def test_mismatched_batch_lengths_are_rejected(bk):
    # Silent truncation would drop tensors from the store while the caller
    # believes they are tracked.
    with pytest.raises((ValueError, RuntimeError)):
        bk.put_tensor_batch([1, 2], [_info(1)])
    with pytest.raises((ValueError, RuntimeError)):
        bk.increment_ref_batch([1, 2], [1])


def test_a_negative_increment_is_rejected(bk):
    # Python asserts; the Rust side raises. Either way it must not silently
    # decrement -- that would mean a caller lost track of the direction.
    bk.put_tensor(1, _info(1))
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        bk.increment_ref(1, -1)


# --- through TensorStore -----------------------------------------------------

def test_tensor_store_works_over_either_backend(bk):
    store = TensorStore(bookkeeping=bk)
    tensor = torch.randn(4, 8)
    store.put_tensor(request_id=5, uuid=1, tensor=tensor, info=_info(1))

    assert store.check_uuid_presence(1)
    assert torch.equal(store.get_tensor(1), tensor)
    assert store.get_info(1).uuid == 1
    assert store.can_gc(1)

    store.increment_ref(1, 2)
    assert not store.can_gc(1)
    store.dereference(1, 2)
    assert store.can_gc(1)

    assert store.get_all_uuids(5) == [1]
    store.remove_tensor(1)
    assert store.get_all_uuids(5) == []
    assert not store.check_uuid_presence(1)
    assert store.get_info(1) is None, "teardown must drop the descriptor too"


def test_the_store_does_not_forget_twice(bk):
    """``dereference_batch_uniform(cleanup=True)`` already dropped the record,
    so the teardown that follows must not cross again to drop it. Harmless if
    it does -- forget is idempotent -- but it is the crossing the batched form
    exists to remove."""
    store = TensorStore(bookkeeping=bk)
    store.put_tensor(request_id=5, uuid=1, tensor=torch.randn(4, 8), info=_info(1))
    store.increment_ref(1, 1)

    forgets = []
    real_forget = bk.forget_tensor
    bk.forget_tensor = lambda uuid: (forgets.append(uuid), real_forget(uuid))[1]

    collectable, _ = store.dereference_batch_uniform([1], cleanup=True)
    assert collectable == [1]
    store.remove_tensor(1)  # what the caller's per-uuid teardown ends in

    assert forgets == [], "the bookkeeper was asked to forget it twice"
    assert not store.check_uuid_presence(1)
    assert store.get_info(1) is None
