"""Signature-conformance test for ``get_piecewise_cuda_graph_configs`` overrides.

``build_piecewise_runners`` calls this method OUTSIDE the per-label try/except,
so an override whose signature drifts from the base raises ``TypeError`` at
engine warmup and kills the worker — but no modular test builds every model's
piecewise runners, so the suite stays green on a branch that can't serve. This
CPU-only test binds the engine's call args against each override's signature to
catch that mismatch cheaply.
"""
import importlib
import inspect
import pkgutil

import pytest

import mstar.model
from mstar.model.submodule_base import NodeSubmodule


def _all_subclasses(cls: type) -> set[type]:
    subs = set(cls.__subclasses__())
    for sub in list(subs):
        subs |= _all_subclasses(sub)
    return subs


def _overriding_subclasses() -> list[type]:
    # Walk the whole mstar.model package so any override is discovered, not just
    # a hand-maintained list — importing each module also registers its
    # NodeSubmodule subclasses.
    for info in pkgutil.walk_packages(mstar.model.__path__, mstar.model.__name__ + "."):
        importlib.import_module(info.name)
    base_fn = NodeSubmodule.get_piecewise_cuda_graph_configs
    return [
        cls for cls in _all_subclasses(NodeSubmodule)
        if cls.get_piecewise_cuda_graph_configs is not base_fn
    ]


@pytest.mark.parametrize("cls", _overriding_subclasses(), ids=lambda c: c.__name__)
def test_piecewise_config_accepts_engine_call(cls):
    """Each override must accept the exact args the engine passes."""
    import torch

    sig = inspect.signature(cls.get_piecewise_cuda_graph_configs)
    # Mirror build_piecewise_runners' call site (self is bound at call time).
    sig.bind(
        object(),  # self placeholder
        torch.device("cpu"), torch.bfloat16, 1,
    )
