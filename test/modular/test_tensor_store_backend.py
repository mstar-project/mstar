"""Which TensorBookkeeping a TensorStore gets, and that the package imports
without the Rust extension.

Both are silent failures: the wrong backend makes a worker refuse to start
(the runtime holds a SHARE of this object, so the two must match), and an
unguarded extension import makes the whole package unimportable wherever it is
not built.
"""
import importlib
import sys

sys.path.insert(0, ".")

import pytest

from mstar.communication import tensor_store
from mstar.communication.tensor_store import TensorStore


def test_the_default_backend_is_the_rust_one():
    """The runtime takes a SHARE of this object; the worker's guard keys off
    ``_rust``."""
    pytest.importorskip("mstar_rust", reason="extension not built")
    bk = TensorStore().bookkeeping
    assert type(bk).__name__ == "RustTensorBookkeeping"
    assert hasattr(bk, "_rust")


def test_the_python_backend_is_the_fallback(monkeypatch):
    """The data worker may run without the extension."""
    def no_extension():
        raise ImportError("No module named 'mstar_rust'")

    monkeypatch.setattr(tensor_store, "RustTensorBookkeeping", no_extension)
    assert type(TensorStore().bookkeeping).__name__ == "PythonTensorBookkeeping"


def test_a_rust_bookkeeper_can_be_passed_in():
    pytest.importorskip("mstar_rust", reason="extension not built")
    from mstar.communication.tensor_store import RustTensorBookkeeping

    store = TensorStore(RustTensorBookkeeping())
    assert type(store.bookkeeping).__name__ == "RustTensorBookkeeping"


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
