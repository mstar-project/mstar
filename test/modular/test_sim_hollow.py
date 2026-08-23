"""Unit tests for hollow mode's fake engine.

Hollow mode only earns its keep if the real worker cannot tell the
difference structurally: the worker stores outputs as tensors and counts
tokens with ``numel()``, and it asks the engine — not the model — whether a
request should stop. These pin those contracts.
"""

import os

import pytest
import torch

from mstar.engine.base import NodeBatch
from mstar.sim.hollow import DEFAULT_STEP_S, SimEngine, hollow_enabled


class _Info:
    def __init__(self, max_tokens=None):
        self.max_tokens = max_tokens
        self.requires_cfg = False


class _Node:
    def __init__(self, name, out_names):
        self.name = name
        self.outputs = [type("E", (), {"name": n})() for n in out_names]


class _Section:
    def __init__(self, nodes):
        self._nodes = {n.name: n for n in nodes}

    def get_nodes(self):
        return self._nodes

    def get_loops(self):
        return {"decode_loop": object()}


class _Model:
    def get_graph_walk_graphs(self):
        return {"decode": _Section([_Node("LLM", ["text_inputs", "new_token"])])}


def _batch(rids, node="LLM", walk="decode", max_tokens=None):
    return NodeBatch(
        node_name=node, graph_walk=walk, request_ids=list(rids),
        per_request_input_tensors={r: {} for r in rids},
        per_request_info={r: _Info(max_tokens) for r in rids},
    )


@pytest.fixture
def engine():
    e = SimEngine(model=_Model())
    for rid in ("r0", "r1"):
        e.add_request(rid)
    return e


def test_env_flag_controls_activation(monkeypatch):
    monkeypatch.delenv("MSTAR_HOLLOW", raising=False)
    assert not hollow_enabled()
    monkeypatch.setenv("MSTAR_HOLLOW", "1")
    assert hollow_enabled()
    monkeypatch.setenv("MSTAR_HOLLOW", "0")
    assert not hollow_enabled()


def test_outputs_are_real_tensors_the_worker_can_count(engine):
    out = engine.execute_batch(_batch(["r0", "r1"]))
    for rid in ("r0", "r1"):
        assert rid in out.per_request_output_tensors
        for tensors in out.per_request_output_tensors[rid].values():
            assert tensors and isinstance(tensors[0], torch.Tensor)
            # The worker's token accounting is tensor.numel(); a zero-element
            # tensor would silently count as no token.
            assert tensors[0].numel() == 1


def test_output_names_come_from_the_models_graph(engine):
    out = engine.execute_batch(_batch(["r0"]))
    assert set(out.per_request_output_tensors["r0"]) == {"text_inputs", "new_token"}


def test_unknown_node_still_produces_a_routable_output():
    e = SimEngine(model=_Model())
    e.add_request("r0")
    out = e.execute_batch(_batch(["r0"], node="mystery"))
    assert out.per_request_output_tensors["r0"]


def test_stop_fires_at_the_token_budget(engine):
    b = _batch(["r0"], max_tokens=3)
    for _ in range(2):
        out = engine.execute_batch(b)
        assert not engine.check_stop_for_batch(b, out).stops
    out = engine.execute_batch(b)
    stops = engine.check_stop_for_batch(b, out).stops
    assert "r0" in stops and "decode_loop" in stops["r0"]


def test_no_budget_means_no_synthetic_stop(engine):
    b = _batch(["r0"])
    for _ in range(5):
        out = engine.execute_batch(b)
    assert not engine.check_stop_for_batch(b, out).stops


def test_step_counts_are_per_request(engine):
    engine.execute_batch(_batch(["r0", "r1"]))
    engine.execute_batch(_batch(["r0"]))
    assert engine._steps["r0"] == 2
    assert engine._steps["r1"] == 1


def test_removing_a_request_drops_its_state(engine):
    engine.execute_batch(_batch(["r0"]))
    engine.remove_request("r0")
    assert "r0" not in engine._steps


def test_delay_falls_back_when_no_cost_table(engine):
    assert engine._delay_for(_batch(["r0"])) == DEFAULT_STEP_S


def test_delay_uses_the_cost_table_when_present(tmp_path):
    from mstar.sim.stepdb import StepDB, StepKey, StepSample

    path = os.path.join(tmp_path, "s.db")
    db = StepDB(path, gpu_name="cpu")
    db.add(StepSample(
        StepKey(model="m", node="LLM", graph_walk="decode",
                padded_bs=1, padded_num_tokens=1),
        kv_len_total=0, gpu_s=0.05,
    ))
    db.close()

    e = SimEngine(model=_Model(), stepdb_path=path)
    e.add_request("r0")
    assert e._delay_for(_batch(["r0"])) == pytest.approx(0.05)
    e.shutdown()


def test_install_redirects_every_engine_factory():
    from mstar.sim.hollow import install
    from mstar.worker import engine_manager as em

    kv_before = dict(em.ENGINE_TYPE_FACTORIES)
    stateless_before = dict(em.STATELESS_FLAVOR_FACTORIES)
    try:
        install(_Model())
        assert set(em.ENGINE_TYPE_FACTORIES) == set(kv_before)
        assert set(em.STATELESS_FLAVOR_FACTORIES) == set(stateless_before)
        made = em.ENGINE_TYPE_FACTORIES["kv_cache"](None, False, False)
        assert isinstance(made, SimEngine)
        flavor = next(iter(em.STATELESS_FLAVOR_FACTORIES))
        assert isinstance(em.STATELESS_FLAVOR_FACTORIES[flavor](None, False, False), SimEngine)
    finally:
        em.ENGINE_TYPE_FACTORIES = kv_before
        em.STATELESS_FLAVOR_FACTORIES = stateless_before
