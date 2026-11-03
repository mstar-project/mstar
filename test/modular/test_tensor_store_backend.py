"""Which TensorBookkeeping a TensorStore gets: the wrong one makes a worker
refuse to start, since the runtime holds a SHARE of this object."""
import sys

sys.path.insert(0, ".")

import pytest

from mstar.communication.tensor_store import TensorStore


def test_the_default_backend_is_the_rust_one():
    """The runtime takes a SHARE of this object; the worker's guard keys off
    ``_rust``."""
    pytest.importorskip("mstar_rust", reason="extension not built")
    bk = TensorStore().bookkeeping
    assert type(bk).__name__ == "RustTensorBookkeeping"
    assert hasattr(bk, "_rust")


def test_a_rust_bookkeeper_can_be_passed_in():
    pytest.importorskip("mstar_rust", reason="extension not built")
    from mstar.communication.tensor_store import RustTensorBookkeeping

    store = TensorStore(RustTensorBookkeeping())
    assert type(store.bookkeeping).__name__ == "RustTensorBookkeeping"
