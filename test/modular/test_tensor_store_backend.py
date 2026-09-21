"""Which TensorBookkeeping a TensorStore gets, and that the package imports
without the Rust extension.

Both are silent failures: the wrong backend makes MSTAR_RUST_GRAPH=1 refuse
to start (the runtime holds a SHARE of this object, so the two must match),
and an unguarded extension import makes the whole package unimportable
wherever it is not built.
"""
import importlib
import sys

sys.path.insert(0, ".")

import pytest

from mstar.communication.tensor_store import TensorStore


def test_the_default_backend_is_python(monkeypatch):
    monkeypatch.delenv("MSTAR_RUST_GRAPH", raising=False)
    assert type(TensorStore().bookkeeping).__name__ == "PythonTensorBookkeeping"


def test_the_rust_graph_flag_selects_the_rust_backend(monkeypatch):
    pytest.importorskip("mstar_rust", reason="extension not built")
    monkeypatch.setenv("MSTAR_RUST_GRAPH", "1")
    bk = TensorStore().bookkeeping
    assert type(bk).__name__ == "RustTensorBookkeeping"
    # The worker's guard keys off this: the runtime takes a share of it.
    assert hasattr(bk, "_rust")


def test_an_unrelated_flag_does_not_select_it(monkeypatch):
    # The bookkeeper is not worth taking on its own -- a descriptor is copied
    # in and rebuilt out, which the Python runtime pays for and gains nothing
    # from. It follows MSTAR_RUST_GRAPH and nothing else.
    monkeypatch.delenv("MSTAR_RUST_GRAPH", raising=False)
    monkeypatch.setenv("MSTAR_SHM_ARENA", "1")
    monkeypatch.setenv("MSTAR_RUST_ZMQ", "1")
    assert type(TensorStore().bookkeeping).__name__ == "PythonTensorBookkeeping"


def test_the_package_imports_without_the_extension():
    """tensor_store is imported by effectively everything, so a top-level
    `from mstar_rust import ...` there makes mstar unimportable on any machine
    that has not run maturin."""
    class _Block:
        def find_spec(self, name, path=None, target=None):
            if name == "mstar_rust":
                raise ModuleNotFoundError(
                    "No module named 'mstar_rust'", name="mstar_rust"
                )

    blocker = _Block()
    sys.meta_path.insert(0, blocker)
    saved = {k: v for k, v in sys.modules.items() if k.startswith("mstar")}
    try:
        for k in list(sys.modules):
            if k.startswith("mstar"):
                del sys.modules[k]
        importlib.import_module("mstar.communication.tensors")
        store = importlib.import_module("mstar.communication.tensor_store")
        assert type(store.TensorStore().bookkeeping).__name__ == (
            "PythonTensorBookkeeping"
        )
    finally:
        sys.meta_path.remove(blocker)
        for k in list(sys.modules):
            if k.startswith("mstar"):
                del sys.modules[k]
        sys.modules.update(saved)
