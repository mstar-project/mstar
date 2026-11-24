"""The compiled Praxis loop routes only the expert's final action to the head."""

import pytest
import torch

pytest.importorskip("mstar_rust", reason="the graph runtime needs the Rust extension")

from test_runtime_lockstep import WORKER, _build, _finish_teardown, _info  # noqa: E402

from mstar.graph.runtime.base import RouteInput  # noqa: E402
from mstar.model.praxis1.praxis1_model import Praxis1Model  # noqa: E402
from mstar.utils.containers import ParallelList  # noqa: E402


@pytest.mark.parametrize("walk", [Praxis1Model.CHUNK_WALK, Praxis1Model.CHUNK_WARM_WALK])
@pytest.mark.parametrize("num_steps", [1, 2])
def test_praxis_loop_head_input_is_final_expert_action(walk: str, num_steps: int) -> None:
    model = Praxis1Model(
        "runwayml/praxis-1-droid", skip_weight_loading=True,
        num_steps=num_steps, max_context_chunks=1, max_text_tokens=8,
    )
    graph = model.get_graph_walk_graphs()[walk]
    runtime, _book, store = _build(graph)
    rid = runtime.add_request(
        "praxis-loop", "default", "decode", [0], ParallelList([0], [[WORKER]]),
        {"praxis_denoise": num_steps},
    )

    for step in range(num_steps):
        for node_offset, node_name in ((0, "backbone"), (10, "action_expert")):
            names = [edge.name for edge in graph.get_nodes()[node_name].outputs]
            uuids = [100 + 100 * step + node_offset + i for i in range(len(names))]
            for uuid in uuids:
                store.put_tensor(rid, uuid, torch.tensor([uuid]), _info(uuid))
                store.increment_ref(uuid)
            result = runtime.complete_and_route_batch(RouteInput(
                partition="default", graph_walk="decode", node_name=node_name,
                output_signals=names, wg_ids=ParallelList([rid], [0]),
                tensors=uuids, num_tensors=[1] * len(names),
            ))
            _finish_teardown(store, result.freed_inputs)
            if node_name == "action_expert":
                final_action_uuid = uuids[names.index("action_latents")]

    popped = runtime.pop_rids("action_head", "decode", [rid], check_ready=True)
    assert popped is not None
    assert [(edge.signal, edge.uuids) for edge in popped.input_edges.edge_tuples()] == [
        ("action_latents", [final_action_uuid]),
    ]
    runtime.remove_request(rid)
