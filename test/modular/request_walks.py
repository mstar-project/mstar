"""A request through an AR model's walks: the graph runtime takes its stops, the model its walks."""
from __future__ import annotations

from collections.abc import Callable

import torch
from test_graph_runtime import StubTensorManager, _sharding_config

from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.graph.runtime.python import PythonGraphRuntime
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import Model, WorkerGraph
from mstar.utils.containers import ParallelList

PARTITION = "default"

# (walk, inputs, decode iteration) -> (this request's outputs, the loops check_stop named)
Step = Callable[[str, dict[str, list[torch.Tensor]], int], tuple[dict, set[str]]]


def _edges(section) -> list[GraphEdge]:
    """The outputs of the walk's last node, the one ``step`` stands for."""
    while not isinstance(section, GraphNode):
        section = section.section if isinstance(section, Loop) else section.sections[-1]
    return section.outputs


def run_walks(model: Model, step: Step, prompt: torch.Tensor, max_tokens: int,
              max_steps: int = 64, node: str = "LLM", signals: dict | None = None) -> list[int]:
    """Run the request to completion; returns the token ids it sent to the client.

    ``step`` stands for ``node``, the last node of every walk (an encoder before it
    does not run). ``signals`` are the request's inputs: ``prompt`` as its
    ``text_inputs`` unless given."""
    # a stop counts only if the runtime accepts it for the walk; the conductor also ends
    # the request once its counted tokens reach max_tokens
    walks = model.get_graph_walk_graphs()
    names = sorted(walks)
    ids = list(range(len(names)))
    runtime = PythonGraphRuntime(
        my_worker_id="w0",
        my_worker_graphs=[
            WorkerGraph(section=walks[w], graph_walks={w}, ranks=[0], worker_graph_id=i)
            for i, w in zip(ids, names, strict=True)
        ],
        all_wg_ids_to_graph_walks={i: {w} for i, w in zip(ids, names, strict=True)},
        all_wg_ids_to_dyn_loops={
            i: {walks[w].name} if isinstance(walks[w], Loop) else set()
            for i, w in zip(ids, names, strict=True)
        },
        all_wg_ids_to_nodes={i: set(walks[w].get_nodes()) for i, w in zip(ids, names, strict=True)},
        node_to_partition={name: PARTITION for walk in walks.values() for name in walk.get_nodes()},
        sharding_config=_sharding_config(),
        tensor_manager=StubTensorManager(),
    )
    args = model.get_initial_forward_pass_args(
        PARTITION, ["text"], ["text"], {"text_inputs": [prompt]} if signals is None else signals,
    )
    metadata = args.full_metadata
    caps = model.get_loop_caps(max_output_tokens=max_tokens)
    rid = runtime.add_request(
        request_id="r", partition=PARTITION, graph_walk=metadata.graph_walk,
        partition_worker_graph_ids=ids, worker_graph_to_workers=ParallelList(ids, [["w0"]] * len(ids)),
        loop_caps=caps,
    )
    emitted: list[int] = []
    counted = 0  # the conductor's count: tokens on conductor_new_token edges
    steps = 0
    while True:
        walk = metadata.graph_walk
        runtime.set_walk(rid, PARTITION, walk)
        edges = _edges(walks[walk])
        inputs = {edge.name: list(edge.tensor_info) for edge in args.inputs}
        persisted: dict[str, list[torch.Tensor]] = {}
        iteration = 0
        while True:
            steps += 1
            assert steps <= max_steps, f"no stop after {max_steps} steps: {emitted}"
            out, stops = step(walk, inputs, iteration)
            for edge in edges:
                tokens = out.get(edge.name)
                if edge.next_node == EMIT_TO_CLIENT and tokens:
                    emitted += tokens[0].reshape(-1).tolist()
                if edge.conductor_new_token and tokens:
                    counted += tokens[0].numel()
                if edge.persist and tokens:
                    persisted[edge.name] = tokens
            runtime.clear_pending_loop_stops()
            stopped = runtime.stop_loops_batched(
                partition=PARTITION, graph_walk=walk, last_node_run=node,
                loop_names=ParallelList([rid], [sorted(stops)]),
            ) if stops else []
            loop = walks[walk]
            # a loop runs the smaller of its max_iters and the request's cap
            if not isinstance(loop, Loop) or rid in stopped or iteration + 1 >= min(
                    loop.max_iters, caps.get(loop.name, loop.max_iters)):
                break
            inputs = {edge.name: out[edge.name] for edge in edges if edge.next_node == node}
            iteration += 1
        args = model.get_partition_forward_pass_args(PARTITION, metadata, persisted)
        metadata = args.full_metadata
        if args.request_done or counted >= max_tokens:
            return emitted
