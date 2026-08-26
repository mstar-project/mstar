"""The simulator follows the model's walk sequence, not a naming convention.

These build a small model whose walks are named so that no heuristic could
guess the order, then assert the simulator executes exactly the chain the
model's transition functions declare — including per-partition progress,
persisted signals flowing between partitions, and a partition that opts out
of a request entirely.
"""

import os
import tempfile

import pytest

from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    PartitionDefinition,
)
from mstar.graph.base import GraphEdge, GraphNode
from mstar.model.base import ForwardPassArgs
from mstar.sim.deployment import Deployment, SimWorkerGraph
from mstar.sim.des import SimRequest, Simulator
from mstar.sim.stepdb import StepDB, StepKey, StepSample

EMIT = "emit_to_client"


def _node(name, inputs, outputs):
    return GraphNode(
        name=name,
        input_names=list(inputs),
        outputs=[GraphEdge(next_node=d, name=n) for n, d in outputs],
    )


def _wg(wg_id, section, walk, ranks=(0,)):
    return SimWorkerGraph(
        wg_id=wg_id, section=section, graph_walks={walk}, ranks=list(ranks),
        tp_size=1, sp_size=1, tp_comm_size=1, instance_ranks=[list(ranks)],
        group_id=0, consumes_stream=False,
        node_names=list(section.get_nodes().keys()),
    )


class _ChainModel:
    """Walks named to defeat any prefill/decode heuristic.

    The declared order is zeta → alpha → omega, which no name-based rule
    would produce, and only ``omega`` ends the request.
    """

    ORDER = {"zeta": "alpha", "alpha": "omega"}

    def get_initial_forward_pass_args(
        self, partition_name, input_modalities, output_modalities,
        input_signals, model_kwargs=None,
    ):
        return ForwardPassArgs(
            full_metadata=CurrentForwardConductorMetadata(
                graph_walk="zeta", is_prefill=True,
                input_modalities=list(input_modalities),
                output_modalities=list(output_modalities),
            ),
            inputs=[GraphEdge(next_node="Nz", name="seed")],
            unpersist_tensors=[],
        )

    def get_partition_forward_pass_args(
        self, partition_name, partition_metadata, persist_signals,
        incoming_connections=None,
    ):
        current = partition_metadata.graph_walk
        nxt = self.ORDER.get(current)
        if nxt is None:
            return ForwardPassArgs(
                full_metadata=partition_metadata, inputs=[],
                unpersist_tensors=[], request_done=True,
            )
        partition_metadata.graph_walk = nxt
        partition_metadata.is_prefill = False
        seed = {"alpha": "Na", "omega": "No"}[nxt]
        return ForwardPassArgs(
            full_metadata=partition_metadata,
            inputs=[GraphEdge(next_node=seed, name="seed")],
            unpersist_tensors=[],
        )


def _chain_deployment(model=None):
    walks = {
        "zeta": _wg("wz", _node("Nz", ["seed"], [("out", "")]), "zeta"),
        "alpha": _wg("wa", _node("Na", ["seed"], [("out", "")]), "alpha"),
        "omega": _wg("wo", _node("No", ["seed"], [("out", EMIT)]), "omega"),
    }
    return Deployment(
        model_key="x", model=model or _ChainModel(), config_path="-", config={},
        walk_to_wgs={w: [wg] for w, wg in walks.items()},
        ranks=[0],
        node_engine_types={"Nz": "stateless", "Na": "stateless", "No": "stateless"},
        node_to_ranks={"Nz": [0], "Na": [0], "No": [0]},
        partitions=[PartitionDefinition(
            name="default", graph_walks={"zeta", "alpha", "omega"},
            initial_walk="zeta",
        )],
        partition_topology=None,
        max_concurrent_requests=None,
        max_output_tokens=4096,
    )


def _db(dep, gpu_s=0.001):
    path = os.path.join(tempfile.mkdtemp(), "s.db")
    db = StepDB(path, gpu_name="T")
    db.add_many([
        StepSample(
            StepKey("x", node, walk, padded_bs=bs, padded_num_tokens=tok),
            kv_len_total=0, gpu_s=gpu_s,
        )
        for walk, wgs in dep.walk_to_wgs.items() for wg in wgs
        for node in wg.node_names for bs in (1, 2, 4) for tok in (1, 2, 4, 32, 64)
    ])
    return db


def _run(dep, **req_kw):
    db = _db(dep)
    sim = Simulator(dep, db)
    req = SimRequest(
        rid="r0", arrival_s=0.0, target_output_tokens=req_kw.pop("target", 50),
        prompt_tokens=4, **req_kw,
    )
    sim.submit(req)
    sim.run(max_events=200000)
    db.close()
    return sim, req


def test_follows_the_models_declared_chain_not_walk_names():
    sim, req = _run(_chain_deployment())
    ran = [k[1] for k in sim.step_counts_by_key]
    assert set(ran) == {"zeta", "alpha", "omega"}
    assert req.done


def test_request_ends_when_the_model_says_done():
    sim, req = _run(_chain_deployment())
    # omega ends it, so nothing runs twice despite a generous token budget.
    assert sim.step_counts_by_key[("No", "omega")] == 1
    assert req.finish_s is not None


class _IdlePartitionModel(_ChainModel):
    """A second partition that plays no part in this request."""

    def get_initial_forward_pass_args(self, partition_name, *a, **k):
        if partition_name == "idle":
            return ForwardPassArgs(
                full_metadata=CurrentForwardConductorMetadata(
                    graph_walk="", is_prefill=False,
                ),
                inputs=[], unpersist_tensors=[], request_done=True,
            )
        return super().get_initial_forward_pass_args(partition_name, *a, **k)


def test_a_partition_can_opt_out_of_a_request():
    dep = _chain_deployment(model=_IdlePartitionModel())
    dep.partitions = list(dep.partitions) + [
        PartitionDefinition(name="idle", graph_walks=set(), initial_walk=None)
    ]
    sim, req = _run(dep)
    assert req.done
    assert req.partition_states["idle"].is_done
    assert set(k[1] for k in sim.step_counts_by_key) == {"zeta", "alpha", "omega"}


def test_request_finishes_only_when_every_partition_is_done():
    dep = _chain_deployment(model=_IdlePartitionModel())
    dep.partitions = list(dep.partitions) + [
        PartitionDefinition(name="idle", graph_walks=set(), initial_walk=None)
    ]
    db = _db(dep)
    sim = Simulator(dep, db)
    req = SimRequest(rid="r0", arrival_s=0.0, target_output_tokens=50, prompt_tokens=4)
    sim.submit(req)
    sim.run(max_events=200000)
    assert all(ps.is_done for ps in req.partition_states.values())
    db.close()


class _PersistModel(_ChainModel):
    """Reads back a signal the previous pass persisted."""

    seen: list = []

    def get_partition_forward_pass_args(
        self, partition_name, partition_metadata, persist_signals,
        incoming_connections=None,
    ):
        _PersistModel.seen.append(sorted(persist_signals))
        return super().get_partition_forward_pass_args(
            partition_name, partition_metadata, persist_signals,
            incoming_connections,
        )


def test_persisted_edges_reach_the_next_transition():
    # A model whose next walk depends on what the last one persisted cannot
    # advance unless the simulator absorbs persist signals the way the
    # conductor does.
    _PersistModel.seen = []
    walks = {
        "zeta": _wg("wz", GraphNode(
            name="Nz", input_names=["seed"],
            outputs=[GraphEdge(next_node="", name="carried", persist=True)],
        ), "zeta"),
        "alpha": _wg("wa", _node("Na", ["seed"], [("out", "")]), "alpha"),
        "omega": _wg("wo", _node("No", ["seed"], [("out", EMIT)]), "omega"),
    }
    dep = _chain_deployment(model=_PersistModel())
    dep.walk_to_wgs = {w: [wg] for w, wg in walks.items()}
    _run(dep)
    assert any("carried" in names for names in _PersistModel.seen)


def test_input_signals_are_seeded_from_the_requests_modalities():
    from mstar.sim.request_inputs import InputSpec, build_input_signals

    sig = build_input_signals(InputSpec(input_modalities=["image", "text"]))
    assert "text_inputs" in sig and "image_inputs" in sig
    assert "audio_features" not in sig
    # Several models read different names for the same modality.
    assert {"pixel_values", "image_grid_thw"} <= set(sig)


def test_media_contributes_prefill_tokens():
    from mstar.sim.request_inputs import InputSpec, token_count_for

    text_only = token_count_for(InputSpec(prompt_tokens=32))
    with_image = token_count_for(
        InputSpec(input_modalities=["image", "text"], prompt_tokens=32,
                  image_size=(256, 256))
    )
    assert with_image > text_only


def test_transition_failure_is_reported_not_fatal():
    class _Broken(_ChainModel):
        def get_partition_forward_pass_args(self, *a, **k):
            raise KeyError("some_signal_the_model_wanted")

    sim, req = _run(_chain_deployment(model=_Broken()))
    # The run completes and names the failure rather than dying mid-sweep.
    assert req.done
    assert any("advance" in k for k in sim.model_errors)


@pytest.mark.parametrize("target,expect_more_than", [(4, 1), (40, 2)])
def test_token_budget_caps_a_repeating_walk(target, expect_more_than):
    class _Loops(_ChainModel):
        ORDER = {"zeta": "alpha", "alpha": "alpha"}

    dep = _chain_deployment(model=_Loops())
    dep.node_engine_types["Na"] = "kv_cache"
    sim, req = _run(dep, target=target)
    assert sim.step_counts_by_key[("Na", "alpha")] >= expect_more_than
    assert req.decode_steps <= target + 2


def test_a_partition_with_work_in_flight_is_not_ended():
    """A consumer draining its last chunk must not be cut off.

    When a producer finishes it signals its consumers, and the consumer is
    asked whether it is done. If that question is answered while the
    consumer still has a step on the GPU, the step's output — for a codec,
    the request's only audio — is discarded.
    """
    dep = _chain_deployment()
    db = _db(dep)
    sim = Simulator(dep, db)
    req = SimRequest(rid="r0", arrival_s=0.0, target_output_tokens=50, prompt_tokens=4)
    sim.submit(req)
    sim.run(max_events=200)

    req.partition_inflight["default"] = 1
    before = req.partition_states["default"].is_done
    sim._advance_partition(req, "default")
    assert req.partition_states["default"].is_done == before
    db.close()


def test_emitted_chunks_wait_for_the_gpu_step_that_made_them():
    # Routing runs on the CPU lane and can finish before the GPU does; a
    # client-visible chunk must still not be reported early.
    dep = _chain_deployment()
    db = _db(dep, gpu_s=0.050)
    sim = Simulator(dep, db)
    req = SimRequest(rid="r0", arrival_s=0.0, target_output_tokens=50, prompt_tokens=4)
    sim.submit(req)
    sim.run(max_events=200000)
    assert req.chunks
    _, first = req.chunks[0]
    assert first >= 0.050
    db.close()
