"""How a verifying node holds its drafter: the `Drafter` and `DrafterDecl` protocols a drafter
subclasses, the share the engine binds once at load (`NodeSubmodule.bind_draft`), and what the engine
refuses of a built submodule (`submodule_problems`)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mstar.engine.resources.draft.drafter import (  # noqa: E402
    ChunkBoundary,
    Drafter,
    DrafterDecl,
    DrafterNeeds,
    ObserveInputs,
    VerifyResult,
)
from mstar.engine.resources.draft.node import DrafterShare  # noqa: E402
from mstar.engine.resources.draft.startup import submodule_problems  # noqa: E402

NEEDS = DrafterNeeds(chunk_boundary=ChunkBoundary.NONE)


class _Proposes(nn.Module, Drafter):
    needs = NEEDS

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 4)

    def propose(self, inputs):
        return torch.zeros(1, 1, dtype=torch.long)


class _NoPropose(nn.Module, Drafter):
    needs = NEEDS

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 4)


class _NoResources(DrafterDecl):
    needs = NEEDS


def test_a_drafter_that_doesnt_propose_cant_be_built_even_on_the_meta_device():
    _Proposes()
    with pytest.raises(TypeError, match="propose"):
        _NoPropose()
    with torch.device("meta"), pytest.raises(TypeError, match="propose"):
        _NoPropose()
    with pytest.raises(TypeError, match="node_resources"):
        _NoResources()


def test_a_hosted_drafter_decl_that_doesnt_build_cant_be_made():
    """Without the abstract methods a missing one would inherit the protocol's ``...`` body and
    hand the target None for its drafter or its steps."""
    from mstar.model.components.hosted_drafter import HostedDrafterDecl

    class _NoBuild(HostedDrafterDecl):
        needs, num_draft_tokens = NEEDS, 1

        def node_resources(self, nodes):
            return []

        def prefill_steps(self):
            return {}

        def verify_steps(self):
            return {}

    with pytest.raises(TypeError, match="build"):
        _NoBuild()
    with torch.device("meta"), pytest.raises(TypeError, match="build"):
        _NoBuild()


def test_a_drafter_observes_nothing_unless_it_says_otherwise():
    result = VerifyResult(torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, dtype=torch.long))
    assert _Proposes().observe(ObserveInputs(torch.zeros(1, 2, dtype=torch.long), result)) is None


def _share(k=3):
    return DrafterShare("decode", "sampler", k, frozenset({"prefill"}), frozenset(), frozenset(), {}, {})


def test_the_share_is_bound_once_and_answers_for_the_verify_walk():
    from spec_toy import ToyLM

    sub = ToyLM(3)
    assert sub.drafter_share is None and not sub.is_verify_walk("decode")
    sub.bind_draft(_share())
    assert sub.is_verify_walk("decode") and not sub.is_verify_walk("prefill") and sub.drafter_share.block == 4
    with pytest.raises(RuntimeError, match="twice"):
        sub.bind_draft(_share())


def test_a_class_level_drafter_is_refused_since_it_hides_a_module_one():
    from mstar.model.submodule_base import ARNodeSubmodule

    with pytest.raises(TypeError, match="set `drafter` in __init__"):
        type("Shadowing", (ARNodeSubmodule,), {"drafter": None})


def test_the_engine_refuses_a_drafter_it_cant_hold():
    decl = SimpleNamespace(node="LLM")
    assert submodule_problems(decl, SimpleNamespace(drafter=_Proposes())) == []
    assert "must set `drafter`" in submodule_problems(decl, SimpleNamespace(drafter=None))[0]

    class _Duck:
        needs = NEEDS

        def propose(self, inputs):
            ...

    assert "is not a Drafter: subclass it" in submodule_problems(decl, SimpleNamespace(drafter=_Duck()))[0]

    class _FullDuck(_Duck):
        # every member the protocol has, observe with its old signature: still not a Drafter
        def observe(self, block, result):
            ...

        def prefill(self, inputs):
            ...

    assert isinstance(_FullDuck(), Drafter)
    assert "is not a Drafter: subclass it" in submodule_problems(decl, SimpleNamespace(drafter=_FullDuck()))[0]

    class _Binds(Drafter):
        needs = NEEDS

        def propose(self, inputs):
            ...

        def bind_resources(self, resources):
            ...

    (problem,) = submodule_problems(decl, SimpleNamespace(drafter=_Binds()))
    assert "is not an nn.Module, so bind_node_resources never reaches it" in problem


def test_a_node_holding_a_drafter_the_model_declares_no_verify_node_for_is_refused(monkeypatch):
    from spec_toy import ToyModel, engine

    from mstar.engine.resources.kv import manager as manager_mod

    class _NoTransfer:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(manager_mod, "KVTransferManager", _NoTransfer)
    monkeypatch.setattr(ToyModel, "verify_nodes", lambda self: [])
    with pytest.raises(RuntimeError, match="declares no VerifyNode"):
        engine(3)


def test_a_hosted_drafter_must_say_what_its_prefill_keeps():
    """A hosted drafter runs its own stack over each prompt, so prefill is abstract on it; a plain
    drafter's defaults to nothing."""
    from mstar.model.components.hosted_drafter import HostedDrafter

    class _NoPrefill(HostedDrafter):
        needs = NEEDS

        def propose(self, inputs):
            ...

    with pytest.raises(TypeError, match="prefill"):
        _NoPrefill()
    assert _Proposes().prefill(None) is None
