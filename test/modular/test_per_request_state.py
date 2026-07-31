"""Engine-level per-request state: ``PerRequestState`` routes tensors vs
kwargs, ``NodeSubmodule`` owns a store the engine cleans via
``cleanup_request``, and the injected view aliases the submodule's own."""

import pytest
import torch

from mstar.model.submodule_base import (
    ModelInputsFromEngine,
    NodeSubmodule,
    PerRequestState,
)


class _Sub(NodeSubmodule):
    def forward(self, *args, **kwargs):
        return {}


def test_add_routes_by_value_kind():
    st = PerRequestState()
    st.add("mask", torch.ones(2))
    st.add("gs", 6.0)
    st.add_all(cond={"n": 3}, scheduler=object())
    assert "mask" in st.tensors and "mask" not in st.kwargs
    assert "gs" in st.kwargs and "cond" in st.kwargs and "scheduler" in st.kwargs
    assert st["gs"] == 6.0 and st["cond"]["n"] == 3
    assert torch.equal(st["mask"], torch.ones(2))


def test_add_same_key_switches_store():
    st = PerRequestState()
    st.add("x", torch.zeros(1))
    st.add("x", 5)
    assert st["x"] == 5 and "x" not in st.tensors
    st.add("x", torch.ones(1))
    assert "x" not in st.kwargs and torch.equal(st["x"], torch.ones(1))


def test_get_remove_contains():
    st = PerRequestState()
    st.add_all(a=1, b=torch.zeros(1))
    assert st.get("missing") is None and st.get("missing", 7) == 7
    assert "a" in st and "b" in st and "missing" not in st
    st.remove("a")
    st.remove(["b", "missing"])
    assert "a" not in st and "b" not in st
    with pytest.raises(KeyError):
        _ = st["a"]


def test_none_values_live_in_kwargs():
    st = PerRequestState()
    st.add("uncond", None)
    assert "uncond" in st and st["uncond"] is None


def test_submodule_store_lifecycle():
    sub = _Sub()
    st = sub.request_state("r1")
    assert sub.request_state("r1") is st  # created once
    st.add("gs", 6.0)
    sub.cleanup_request("r1")
    assert "r1" not in sub.request_states
    sub.cleanup_request("r1")  # idempotent


def test_engine_injection_aliases_store():
    sub = _Sub()
    ei = ModelInputsFromEngine(
        request_ids=["r1"],
        per_request_info={},
        per_request_states={"r1": sub.request_state("r1")},
    )
    ei.per_request_states["r1"].add("gs", 7.0)
    assert sub.request_states["r1"]["gs"] == 7.0
    # Default: paths that don't inject leave the field None.
    assert ModelInputsFromEngine(request_ids=[], per_request_info={}).per_request_states is None
