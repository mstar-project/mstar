"""``worker._make_graph_runtime``: every worker gets the Rust runtime.

Every worker test injects a runtime directly, so without this nothing checks
that the factory builds one -- which is how the Rust bookkeeper once shipped
never being instantiated at all.

The guards matter as much as the build. The runtime holds an Arc to the
transport and a SHARE of the bookkeeper, so a mismatched communicator or
bookkeeper is not a degraded mode -- it is a runtime that cannot send, or two
views of the refcounts that drift apart with nothing raising.
"""
import sys

import pytest

sys.path.insert(0, ".")

import torch  # noqa: F401  (import order: torch before mstar internals)

from mstar.communication.codec import WireCodec
from mstar.communication.rust_communicator import RustZMQCommunicator
from mstar.communication.tensor_store import (
    PythonTensorBookkeeping,
    RustTensorBookkeeping,
    TensorStore,
)
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphNode
from mstar.graph.runtime.rust import RustGraphRuntime
from mstar.graph.runtime.utils import require_rust_extension
from mstar.model.base import WorkerGraph
from mstar.worker.worker import _make_graph_runtime

WG_ID = 0
NODE = "prefill"


def _kwargs(communicator=None, bookkeeping=None):
    section = GraphNode(name=NODE, input_names={"prompt"}, outputs=[])
    store = TensorStore(bookkeeping=bookkeeping or RustTensorBookkeeping())
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


def _rust_communicator(codec=WireCodec):
    comm = RustZMQCommunicator.__new__(RustZMQCommunicator)
    comm.codec = codec
    return comm


@pytest.fixture(autouse=True)
def _ambient_transport_flag_unset(monkeypatch):
    monkeypatch.delenv("MSTAR_RUST_ZMQ", raising=False)


def test_the_factory_builds_the_rust_runtime():
    bookkeeping = RustTensorBookkeeping()
    runtime = _make_graph_runtime(**_kwargs(
        communicator=_rust_communicator(), bookkeeping=bookkeeping,
    ))
    assert isinstance(runtime, RustGraphRuntime)
    # The same object, not a rebuilt one: the store and the runtime have to
    # see each other's refcount moves.
    assert runtime._bookkeeping is bookkeeping


@pytest.mark.parametrize("value", ["1", "AUTO", "auto"])
def test_a_transport_flag_that_allows_rust_builds_it(monkeypatch, value):
    monkeypatch.setenv("MSTAR_RUST_ZMQ", value)
    assert isinstance(_make_graph_runtime(**_kwargs(communicator=_rust_communicator())), RustGraphRuntime)


def test_a_worker_pinned_to_pyzmq_is_refused(monkeypatch):
    """MSTAR_RUST_ZMQ=0 contradicts the only runtime, and says so rather than
    leaving the worker to trip over the communicator it built."""
    monkeypatch.setenv("MSTAR_RUST_ZMQ", "0")
    with pytest.raises(ValueError, match="MSTAR_RUST_ZMQ=0 pins the pyzmq communicator"):
        _make_graph_runtime(**_kwargs())


def test_the_pyzmq_communicator_is_refused():
    """Caught on the communicator's TYPE as well as on the flag: the flag says
    what the worker was asked to build, the type says what it actually built
    (AUTO falls back to pyzmq on a stale extension), and only the second one
    can be shared."""
    with pytest.raises(ValueError, match="needs the Rust communicator"):
        _make_graph_runtime(**_kwargs())
    with pytest.raises(ValueError, match="built a object"):
        _make_graph_runtime(**_kwargs(communicator=object()))


def test_a_codec_other_than_msgpack_is_refused():
    with pytest.raises(ValueError, match="needs the msgpack codec"):
        _make_graph_runtime(**_kwargs(communicator=_rust_communicator(codec=object)))


def test_a_python_bookkeeper_is_refused():
    """The runtime takes a share of the bookkeeper, so a Python one cannot be
    handed over -- and a copy would let the two views of the refcounts drift
    apart silently."""
    with pytest.raises(ValueError, match="Rust TensorBookkeeping"):
        _make_graph_runtime(**_kwargs(
            communicator=_rust_communicator(), bookkeeping=PythonTensorBookkeeping(),
        ))


def test_a_missing_extension_says_how_to_build_it(monkeypatch):
    """A None entry in sys.modules makes the import raise, which is what a
    machine that never ran maturin sees."""
    kwargs = _kwargs(communicator=object())
    monkeypatch.setitem(sys.modules, "mstar_rust", None)
    with pytest.raises(ImportError, match="maturin develop --release -m rust/Cargo.toml"):
        require_rust_extension()
    with pytest.raises(ImportError, match="needs the mstar_rust extension"):
        _make_graph_runtime(**kwargs)


def test_importing_the_worker_does_not_need_the_extension():
    """Tooling imports mstar.worker on machines that never built it."""
    import subprocess

    script = (
        "import sys; sys.modules['mstar_rust'] = None; import mstar.worker.worker; "
        "assert 'mstar.graph.runtime.rust' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
