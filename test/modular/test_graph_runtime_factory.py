"""``worker._make_graph_runtime``: the flag actually selects a backend.

The factory had no coverage. Every worker test injects a runtime directly,
so nothing checked that MSTAR_RUST_GRAPH=1 reaches the Rust one -- which is
how the Rust bookkeeper shipped never being instantiated at all. A flag that
silently does nothing reads exactly like a flag that works.

The guards matter as much as the selection. The Rust runtime holds an Arc to
the transport and a SHARE of the bookkeeper, so a mismatched communicator or
bookkeeper is not a degraded mode -- it is a runtime that cannot send, or two
views of the refcounts that drift apart with nothing raising.

The flag is 0, 1 or AUTO. 0 is the default for now; AUTO is Rust where the
extension imports and Python where it does not, so it is a property of the
machine, and both halves of it need a case.
"""
import sys

import pytest

sys.path.insert(0, ".")

import torch  # noqa: F401  (import order: torch before mstar internals)

from mstar.communication.tensor_store import (
    PythonTensorBookkeeping,
    TensorStore,
)
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphNode
from mstar.graph.runtime.python import PythonGraphRuntime
from mstar.model.base import WorkerGraph
from mstar.worker.worker import _make_graph_runtime

WG_ID = 0
NODE = "prefill"


def _kwargs(communicator=None, bookkeeping=None):
    section = GraphNode(name=NODE, input_names={"prompt"}, outputs=[])
    store = TensorStore(bookkeeping=bookkeeping or PythonTensorBookkeeping())
    return dict(
        my_worker_id="worker0",
        my_worker_graphs=[WorkerGraph(
            section=section, graph_walks={"prefill"}, ranks=[0],
            worker_graph_id=WG_ID,
        )],
        all_wg_ids_to_graph_walks={WG_ID: {"prefill"}},
        all_wg_ids_to_dyn_loops={WG_ID: set()},
        all_wg_ids_to_nodes={WG_ID: {NODE}},
        node_to_partition={NODE: "default"},
        sharding_config=ShardingConfig(
            groups=[], tp_enabled_nodes=set(), shard_dim={},
        ),
        tensor_manager=type("_TM", (), {"tensor_store": store})(),
        communicator=communicator,
    )


def test_the_default_is_the_python_runtime(monkeypatch):
    """Rust is opt-in until it has run in production: an unset flag must not
    move a worker onto it, even where the extension is built."""
    monkeypatch.delenv("MSTAR_RUST_GRAPH", raising=False)
    assert isinstance(_make_graph_runtime(**_kwargs()), PythonGraphRuntime)


def test_auto_takes_rust_where_it_is_built(monkeypatch):
    """AUTO means "whatever this machine has"."""
    pytest.importorskip("mstar_rust")
    monkeypatch.setenv("MSTAR_RUST_GRAPH", "AUTO")
    # AUTO reads the transport flag too, so the ambient value would decide it.
    monkeypatch.delenv("MSTAR_RUST_ZMQ", raising=False)

    from mstar.communication.rust_communicator import RustZMQCommunicator
    from mstar.communication.tensor_store import RustTensorBookkeeping
    from mstar.graph.runtime.rust import RustGraphRuntime

    runtime = _make_graph_runtime(**_kwargs(
        communicator=RustZMQCommunicator.__new__(RustZMQCommunicator),
        bookkeeping=RustTensorBookkeeping(),
    ))
    assert isinstance(runtime, RustGraphRuntime)


def test_auto_falls_back_to_python_without_the_extension(monkeypatch):
    """The other half of AUTO, and the one that keeps mstar runnable on a
    machine that never ran maturin. A None entry in sys.modules makes the
    import raise, which is what the probe sees when it is really absent."""
    monkeypatch.setenv("MSTAR_RUST_GRAPH", "AUTO")
    monkeypatch.setitem(sys.modules, "mstar.graph.runtime.rust", None)
    assert isinstance(_make_graph_runtime(**_kwargs()), PythonGraphRuntime)


def test_zero_is_the_python_runtime(monkeypatch):
    """Explicit 0 pins Python even where the extension is built, so AUTO
    cannot quietly move a worker onto a backend nobody asked for."""
    monkeypatch.setenv("MSTAR_RUST_GRAPH", "0")
    assert isinstance(_make_graph_runtime(**_kwargs()), PythonGraphRuntime)


@pytest.mark.parametrize("value", ["auto", "Auto", "1 ", "true", "yes", ""])
def test_anything_other_than_0_1_or_auto_is_refused(monkeypatch, value):
    """The three spellings are exact. A typo that fell back to a default would
    hide a build that did not take -- which is the whole reason the flag is
    read rather than inferred."""
    monkeypatch.setenv("MSTAR_RUST_GRAPH", value)
    with pytest.raises(ValueError, match="must be 0, 1, or AUTO"):
        _make_graph_runtime(**_kwargs())


def test_auto_does_not_pick_rust_against_a_pinned_pyzmq_transport(monkeypatch):
    """MSTAR_RUST_ZMQ=0 on its own has to keep working.

    It is documented as safe to set per process, and the Rust runtime refuses
    a pyzmq communicator -- so AUTO selecting Rust there is a worker that
    cannot start at all. AUTO means "whatever this machine can run", and this
    machine cannot run that pair.
    """
    pytest.importorskip("mstar_rust")
    monkeypatch.setenv("MSTAR_RUST_GRAPH", "AUTO")
    monkeypatch.setenv("MSTAR_RUST_ZMQ", "0")
    assert isinstance(_make_graph_runtime(**_kwargs()), PythonGraphRuntime)


def test_rust_refuses_the_pyzmq_communicator(monkeypatch):
    """MSTAR_RUST_ZMQ=0 builds a pyzmq communicator, which the Rust runtime
    cannot share. Caught on the communicator's TYPE rather than on the flag:
    the flag says what the worker was asked to build, the type says what it
    actually built, and only the second one can be shared."""
    monkeypatch.setenv("MSTAR_RUST_GRAPH", "1")
    monkeypatch.setenv("MSTAR_RUST_ZMQ", "0")
    with pytest.raises(ValueError, match="needs the Rust communicator"):
        _make_graph_runtime(**_kwargs())


def test_rust_refuses_a_communicator_that_is_not_the_rust_one(monkeypatch):
    pytest.importorskip("mstar_rust")
    monkeypatch.setenv("MSTAR_RUST_GRAPH", "1")
    monkeypatch.setenv("MSTAR_RUST_ZMQ", "1")
    with pytest.raises(ValueError, match="built a"):
        _make_graph_runtime(**_kwargs(communicator=object()))


def test_rust_refuses_a_python_bookkeeper(monkeypatch):
    """The runtime takes a share of the bookkeeper, so a Python one cannot be
    handed over -- and a copy would let the two views of the refcounts drift
    apart silently."""
    pytest.importorskip("mstar_rust")
    monkeypatch.setenv("MSTAR_RUST_GRAPH", "1")
    monkeypatch.setenv("MSTAR_RUST_ZMQ", "1")

    from mstar.communication.rust_communicator import RustZMQCommunicator

    comm = RustZMQCommunicator.__new__(RustZMQCommunicator)
    with pytest.raises(ValueError, match="Rust TensorBookkeeping"):
        _make_graph_runtime(**_kwargs(
            communicator=comm, bookkeeping=PythonTensorBookkeeping(),
        ))


def test_one_selects_the_rust_runtime(monkeypatch):
    """The positive case, which is the one that was never true."""
    pytest.importorskip("mstar_rust")
    monkeypatch.setenv("MSTAR_RUST_GRAPH", "1")
    monkeypatch.setenv("MSTAR_RUST_ZMQ", "1")

    from mstar.communication.rust_communicator import RustZMQCommunicator
    from mstar.communication.tensor_store import RustTensorBookkeeping
    from mstar.graph.runtime.rust import RustGraphRuntime

    comm = RustZMQCommunicator.__new__(RustZMQCommunicator)
    bookkeeping = RustTensorBookkeeping()
    runtime = _make_graph_runtime(**_kwargs(
        communicator=comm, bookkeeping=bookkeeping,
    ))
    assert isinstance(runtime, RustGraphRuntime)
    # The same object, not a rebuilt one: the store and the runtime have to
    # see each other's refcount moves.
    assert runtime._bookkeeping is bookkeeping
