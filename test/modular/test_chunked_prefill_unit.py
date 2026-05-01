"""Unit tests for chunked prefill primitives. CPU-only, no model weights."""
from __future__ import annotations

from mminf.model.submodule_base import NodeSubmodule


class _DummySubmodule(NodeSubmodule):
    """Concrete NodeSubmodule with the bare minimum to instantiate."""
    def prepare_inputs(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError


def test_supports_chunked_prefill_default_false():
    sub = _DummySubmodule()
    assert sub.supports_chunked_prefill() is False
