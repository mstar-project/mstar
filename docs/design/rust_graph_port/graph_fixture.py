"""Synthetic worker graphs + sharding config for the port benchmarks.

Decode-loop shape (Orpheus/Qwen3): Loop[ llm -> sampler ], sampler loops back
to llm and emits a token. `width` adds extra fan-out nodes inside the loop to
model wider graphs (cosmos3/bagel).
"""
from mstar.distributed.base import ShardingConfig, ShardingGroup
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT

WALK = "decode"


def _tinfo(uuid: str, dim0: int = 8) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=[dim0, 128], dtype="bfloat16", nbytes=dim0 * 128 * 2,
        address=0, stride=[128, 1], uuid=uuid,
        source_session_id="h:1", source_entity="worker0",
    )


def make_section(width: int = 0, max_iters: int = 4096) -> Loop:
    """Loop[llm -> sampler (-> side_i)*width]; sampler feeds llm + client."""
    side_names = [f"side{i}" for i in range(width)]
    llm = GraphNode(
        name="llm",
        input_names={"input_ids", "position"},
        outputs=[GraphEdge(next_node="sampler", name="logits")],
    )
    sampler = GraphNode(
        name="sampler",
        input_names={"logits"},
        outputs=[
            GraphEdge(next_node="llm", name="input_ids"),
            GraphEdge(next_node="llm", name="position"),
            GraphEdge(next_node=EMIT_TO_CLIENT, name="token",
                      persist=True, conductor_new_token=True, output_modality="text"),
            *[GraphEdge(next_node=n, name=f"logits{i}") for i, n in enumerate(side_names)],
        ],
    )
    sides = [
        GraphNode(name=n, input_names={f"logits{i}"},
                  outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name=f"aux{i}", persist=True)])
        for i, n in enumerate(side_names)
    ]
    body = Sequential([llm, sampler, *sides])
    return Loop(
        section=body, max_iters=max_iters, name="decode_loop",
        outputs=[GraphEdge(next_node="done", name="token")],
    )


def make_nested_section(outer_iters: int = 3, inner_iters: int = 2) -> Loop:
    """Loop_outer[ pre -> Loop_inner[ step ] -> post ].

    `step` loops back to itself inside the inner loop; the inner loop's output
    feeds `post`, whose output loops back to `pre`. Exercises: nested advance,
    inner reset on outer advance, cascade on inner termination, and the
    outer loop's own declared output.
    """
    step = GraphNode(
        name="step", input_names={"seed", "acc"},
        outputs=[GraphEdge(next_node="step", name="acc")],
    )
    inner = Loop(
        section=step, max_iters=inner_iters, name="inner",
        outputs=[GraphEdge(next_node="post", name="acc")],
        accumulated_outputs=[GraphEdge(next_node="post", name="trace")],
    )
    pre = GraphNode(
        name="pre", input_names={"x"},
        outputs=[GraphEdge(next_node="step", name="seed"),
                 GraphEdge(next_node="step", name="acc")],
    )
    post = GraphNode(
        name="post", input_names={"acc", "trace"},
        outputs=[GraphEdge(next_node="pre", name="x"),
                 GraphEdge(next_node=EMIT_TO_CLIENT, name="chunk", persist=True)],
    )
    return Loop(
        section=Sequential([pre, inner, post]), max_iters=outer_iters, name="outer",
        outputs=[GraphEdge(next_node="done", name="chunk")],
    )


def node_names(width: int) -> list[str]:
    return ["llm", "sampler", *[f"side{i}" for i in range(width)]]


def make_sharding_config(width: int, worker_id: str, tp_size: int = 1) -> ShardingConfig:
    names = node_names(width)
    workers = [f"w{i}" for i in range(tp_size)]
    groups = [ShardingGroup(nodes=set(names), tp_size=tp_size, graph_walks={WALK})]
    cfg = ShardingConfig(groups=groups, tp_enabled_nodes=set(names) if tp_size > 1 else set(),
                         shard_dim={})
    from mstar.graph.base import NodeAndGraphWalk
    cfg.setup({NodeAndGraphWalk(n, WALK): workers for n in names})
    grp = cfg.get_sharding_group("llm", WALK)
    grp.register_workers(workers, my_tp_rank=workers.index(worker_id))
    return cfg


def sampler_output_edges(width: int, rid: str) -> list[GraphEdge]:
    """What `mark_node_complete("sampler")` hands back, tensor_info filled."""
    edges = [
        GraphEdge(next_node="llm", name="input_ids", tensor_info=[_tinfo(f"{rid}-ids")]),
        GraphEdge(next_node="llm", name="position", tensor_info=[_tinfo(f"{rid}-pos")]),
        GraphEdge(next_node=EMIT_TO_CLIENT, name="token", persist=True,
                  conductor_new_token=True, output_modality="text",
                  tensor_info=[_tinfo(f"{rid}-tok")]),
    ]
    edges += [
        GraphEdge(next_node=f"side{i}", name=f"logits{i}", tensor_info=[_tinfo(f"{rid}-l{i}")])
        for i in range(width)
    ]
    return edges
