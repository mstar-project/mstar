"""Per-node weight residency: what `residency: on_demand` buys and costs.

The point of the policy is that components never co-reside, so a model whose
parts sum past device memory still runs — the peak becomes the largest single
node instead of the sum. These tests pin the two halves of that: the config
surface, and the evict/load swap itself.

Everything here runs on CPU: `_ensure_resident` moves to whatever `_device` is.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.engine import Engine
from mstar.worker.engine_manager import parse_on_demand_nodes


class _FakeSubmodule:
    """Records every device it was moved to."""

    def __init__(self) -> None:
        self.moves: list[str] = []

    def to(self, device=None, **kwargs):
        target = device if device is not None else kwargs.get("device")
        self.moves.append(str(target))
        return self


def _engine_with(nodes: dict[str, _FakeSubmodule], on_demand: set[str]) -> Engine:
    engine = Engine(autocast_dtype=torch.bfloat16)
    engine._device = torch.device("cpu")
    engine._submodules = {
        name: SimpleNamespace(submodule=sub) for name, sub in nodes.items()
    }
    engine._on_demand = set(on_demand)
    return engine


# --- config surface --------------------------------------------------------

def test_default_is_resident():
    cfg = {"node_groups": [{"node_names": ["dit", "vae"], "ranks": [0]}]}
    assert parse_on_demand_nodes(cfg, {"dit", "vae"}) == set()


def test_on_demand_selected_per_group():
    cfg = {"node_groups": [
        {"node_names": ["text_encoder"], "ranks": [0], "residency": "on_demand"},
        {"node_names": ["dit"], "ranks": [0], "residency": "resident"},
        {"node_names": ["vae"], "ranks": [0]},
    ]}
    assert parse_on_demand_nodes(cfg, {"text_encoder", "dit", "vae"}) == {"text_encoder"}


def test_only_nodes_on_this_worker_are_returned():
    """A group may place nodes this worker doesn't own; those are not ours."""
    cfg = {"node_groups": [
        {"node_names": ["text_encoder", "dit"], "ranks": [0], "residency": "on_demand"},
    ]}
    assert parse_on_demand_nodes(cfg, {"dit"}) == {"dit"}


def test_misspelled_policy_raises():
    """A misspelling is never silently ignored — same contract as the resource
    overrides, where an unknown key fails loading."""
    cfg = {"node_groups": [
        {"node_names": ["dit"], "ranks": [0], "residency": "on-demand"},  # hyphen
    ]}
    with pytest.raises(ValueError, match="residency must be one of"):
        parse_on_demand_nodes(cfg, {"dit"})


# --- the swap --------------------------------------------------------------

def test_resident_node_is_never_moved():
    dit = _FakeSubmodule()
    engine = _engine_with({"dit": dit}, on_demand=set())
    engine._ensure_resident("dit")
    engine._ensure_resident("dit")
    assert dit.moves == []
    assert engine._resident_on_demand is None


def test_on_demand_node_is_loaded_once_then_reused():
    enc = _FakeSubmodule()
    engine = _engine_with({"enc": enc}, on_demand={"enc"})
    engine._ensure_resident("enc")
    engine._ensure_resident("enc")  # idempotent: no second transfer
    assert enc.moves == ["cpu"]
    assert engine._resident_on_demand == "enc"


def test_second_on_demand_node_evicts_the_first():
    """The eviction is the whole point: two large components must never both
    hold the device, or the peak is their sum and the model does not fit."""
    enc, dit = _FakeSubmodule(), _FakeSubmodule()
    engine = _engine_with({"enc": enc, "dit": dit}, on_demand={"enc", "dit"})

    engine._ensure_resident("enc")
    engine._ensure_resident("dit")

    assert enc.moves == ["cpu", "cpu"]   # loaded, then evicted back to host
    assert dit.moves == ["cpu"]          # loaded
    assert engine._resident_on_demand == "dit"


def test_mixing_resident_and_on_demand_leaves_resident_alone():
    enc, vae = _FakeSubmodule(), _FakeSubmodule()
    engine = _engine_with({"enc": enc, "vae": vae}, on_demand={"enc"})
    engine._ensure_resident("vae")   # resident: untouched
    engine._ensure_resident("enc")
    engine._ensure_resident("vae")   # still untouched, and enc stays loaded
    assert vae.moves == []
    assert engine._resident_on_demand == "enc"


def test_failed_move_does_not_leave_a_stale_claim():
    """If the incoming move raises, no node may still be recorded as resident —
    otherwise the next call skips the eviction and two nodes co-reside."""
    class _Exploding(_FakeSubmodule):
        def to(self, device=None, **kwargs):
            raise RuntimeError("CUDA out of memory")

    enc, bad = _FakeSubmodule(), _Exploding()
    engine = _engine_with({"enc": enc, "bad": bad}, on_demand={"enc", "bad"})
    engine._ensure_resident("enc")
    with pytest.raises(RuntimeError, match="out of memory"):
        engine._ensure_resident("bad")
    assert engine._resident_on_demand is None
    assert enc.moves == ["cpu", "cpu"]   # it was still evicted


# --- the shipped config ----------------------------------------------------

def test_bagel_residency_config_pages_the_encoders():
    """configs/bagel_residency.yaml is the validation deployment for this
    feature: encoders paged, LLM resident (it runs every decode step and is the
    node that most wants CUDA-graph capture)."""
    import pathlib

    import yaml

    cfg = yaml.safe_load(pathlib.Path("configs/bagel_residency.yaml").read_text())
    nodes = {"vit_encoder", "vae_encoder", "vae_decoder", "LLM"}
    assert parse_on_demand_nodes(cfg, nodes) == {"vit_encoder", "vae_encoder", "vae_decoder"}


# --- the reload policy -----------------------------------------------------

def test_parse_residency_reports_policy_per_node():
    from mstar.worker.engine_manager import parse_residency

    cfg = {"node_groups": [
        {"node_names": ["enc"], "ranks": [0], "residency": "reload"},
        {"node_names": ["vae"], "ranks": [0], "residency": "on_demand"},
        {"node_names": ["dit"], "ranks": [0]},
    ]}
    got = parse_residency(cfg, {"enc", "vae", "dit"})
    # resident nodes are omitted: absent means default
    assert got == {"enc": "reload", "vae": "on_demand"}


def test_reload_node_drops_storage_and_rebuilds_from_the_factory():
    """`reload` is the policy that actually frees memory: it evicts to `meta`
    (no storage) rather than to the host, and the next execution rebuilds from
    the checkpoint. On a unified-memory part that distinction is the whole
    ballgame — host and device are one pool, so moving weights host-side frees
    nothing."""
    built: list[str] = []

    class _Rebuilt(_FakeSubmodule):
        def requires_grad_(self, flag):
            return self

        def bind_node_resources(self, resources):
            return None

        forward = staticmethod(lambda *a, **k: None)
        forward_batched = staticmethod(lambda *a, **k: None)

    def factory(name):
        built.append(name)
        return _Rebuilt()

    enc, dit = _FakeSubmodule(), _FakeSubmodule()
    engine = _engine_with({"enc": enc, "dit": dit}, on_demand=set())
    engine._reload = {"enc", "dit"}
    engine._submodule_factory = factory
    engine._submodules["enc"].resources = {}
    engine._submodules["dit"].resources = {}
    engine._submodules["enc"].forward = None
    engine._submodules["enc"].forward_batched = None
    engine._submodules["dit"].forward = None
    engine._submodules["dit"].forward_batched = None

    engine._ensure_resident("enc")
    assert built == ["enc"]              # rebuilt from the checkpoint
    engine._ensure_resident("dit")
    assert built == ["enc", "dit"]
    # the outgoing node's storage was dropped, not parked on the host
    assert engine._submodules["enc"].submodule.moves[-1] == "meta"


def test_reload_without_a_factory_is_refused_at_load():
    """Failing here is much better than failing on the first request, when the
    weights are already gone and there is nothing to rebuild them with."""
    engine = Engine(autocast_dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="submodule_factory"):
        engine.load_model(
            submodules={"enc": _FakeSubmodule()},
            specs=[],
            parallel_groups=SimpleNamespace(
                all_in_same_group=lambda *_: True,
                get_joint_group_for_node=lambda *_: None,
            ),
            device=torch.device("cpu"),
            transfer_engine_info=None,
            reload_nodes={"enc"},
            submodule_factory=None,
        )


def test_reload_eviction_releases_the_models_cached_submodule():
    """Regression: every model here memoises get_submodule, so a `reload`
    rebuild used to hand back the very object whose storage had just been
    dropped — the next .to(device) then died with "Cannot copy out of meta
    tensor". Eviction must tell the model to forget the node first."""
    released: list[str] = []

    class _Rebuilt(_FakeSubmodule):
        def requires_grad_(self, flag):
            return self

        def bind_node_resources(self, resources):
            return None

        forward = staticmethod(lambda *a, **k: None)
        forward_batched = staticmethod(lambda *a, **k: None)

    enc, dit = _FakeSubmodule(), _FakeSubmodule()
    engine = _engine_with({"enc": enc, "dit": dit}, on_demand=set())
    engine._reload = {"enc", "dit"}
    engine._submodule_factory = lambda name: _Rebuilt()
    engine._submodule_release = released.append
    for n in ("enc", "dit"):
        engine._submodules[n].resources = {}
        engine._submodules[n].forward = None
        engine._submodules[n].forward_batched = None

    engine._ensure_resident("enc")
    engine._ensure_resident("dit")          # evicts enc
    assert released == ["enc"], "eviction did not release the cached submodule"


def test_model_release_submodule_drops_the_conventional_cache():
    """The base implementation covers every model in this package, all of which
    keep a `_submodule_cache` dict."""
    from mstar.model.base import Model

    class _M(Model):
        def __init__(self):
            self._submodule_cache = {"enc": object(), "dit": object()}
        # abstract methods are irrelevant here
        get_node_resources = get_graph_walk_graphs = None
        get_initial_forward_pass_args = process_prompt = None
        postprocess = get_submodule = get_partition_forward_pass_args = None

    m = _M.__new__(_M)
    m._submodule_cache = {"enc": object(), "dit": object()}
    Model.release_submodule(m, "enc")
    assert set(m._submodule_cache) == {"dit"}
    Model.release_submodule(m, "absent")   # tolerates an unknown node


def test_startup_release_is_what_makes_the_first_reload_genuine():
    """Regression, second order: eviction releasing the cache is not enough.

    A reload node is built once at startup (to bind resources) and immediately
    dropped to meta. The FIRST execution has nothing to evict, so if startup
    does not also release the cache, get_submodule's memo hands the factory that
    same meta object and .to(device) dies — the identical failure the eviction
    fix was supposed to cure, one step earlier.
    """
    import inspect

    from mstar.worker import engine_manager

    src = inspect.getsource(engine_manager.EngineManager.build)
    reload_block = src[src.index("for name in reload_nodes:"):]
    assert 'to("meta")' in reload_block
    assert "release_submodule" in reload_block, (
        "startup drops the weights but never tells the model to forget them"
    )
