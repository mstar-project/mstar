"""Smoke tests for WorkerGraphIO execution.

Originally hand-authored by the refactor lead at
~/Downloads/disaggregation_research/multimodal_inference/smoke_test_graph.py;
moved here so pytest picks them up.
"""
from mminf.graph.base import GraphEdge, GraphNode, Loop, Sequential
from mminf.graph.graph_io import WorkerGraphIO


def run_graph(
    io_manager: WorkerGraphIO,
    initial_edges: list[GraphEdge],
    on_node_complete=None,
    max_steps: int = 50,
) -> list[GraphEdge]:
    """Drive the execution loop until the registry reports done.

    Mirrors the responsibilities of the real worker at small scale: ingest,
    pick a ready node, mark complete, route outputs.
    """
    final_outputs: list[GraphEdge] = []
    pending: list[GraphEdge] = list(initial_edges)
    step = 0

    while not io_manager.wg_state_registry.is_done and step < max_steps:
        unrouted = []
        for edge in pending:
            if not io_manager.ingest_input(edge):
                unrouted.append(edge)
        pending = unrouted

        ready = list(io_manager.ready_node_names)
        if not ready:
            break

        node_name = ready[0]
        io_manager.ready_node_names.discard(node_name)
        step += 1

        if on_node_complete:
            on_node_complete(node_name, io_manager)

        completion = io_manager.mark_node_complete(node_name)
        for edge in completion.output_edges:
            if (edge.name, edge.next_node) in completion.filtered_signals:
                continue
            if edge.next_node in io_manager.nodes:
                pending.append(edge)
            else:
                final_outputs.append(edge)

    return final_outputs


def test_simple_pipeline():
    graph = Sequential(sections=[
        GraphNode(
            name="encode",
            input_names={"prompt"},
            outputs=[GraphEdge(name="hidden", next_node="decode")],
        ),
        GraphNode(
            name="decode",
            input_names={"hidden"},
            outputs=[GraphEdge(name="output", next_node="EMIT_TO_CLIENT")],
        ),
    ])
    io = WorkerGraphIO(graph)
    for _ in range(2):
        result = run_graph(io, [GraphEdge(name="prompt", next_node="encode")])
        assert len(result) == 1
        assert result[0].name == "output"
        io.clear()


def test_diffusion_loop_fixed_iters():
    graph = Sequential(sections=[
        Loop(
            name="diffusion",
            section=GraphNode(
                name="unet",
                input_names={"latents", "t"},
                outputs=[
                    GraphEdge(name="latents", next_node="unet"),
                    GraphEdge(name="t", next_node="unet"),
                ],
            ),
            outputs=[GraphEdge(name="latents", next_node="vae_dec")],
            max_iters=3,
        ),
        GraphNode(
            name="vae_dec",
            input_names={"latents"},
            outputs=[GraphEdge(name="image", next_node="EMIT_TO_CLIENT", output_modality="image")],
        ),
    ])
    io = WorkerGraphIO(graph)
    for _ in range(2):
        result = run_graph(io, [
            GraphEdge(name="latents", next_node="unet"),
            GraphEdge(name="t", next_node="unet"),
        ])
        assert len(result) == 1
        assert result[0].name == "image"
        io.clear()


def test_ar_generation_with_dynamic_finish():
    graph = Sequential(sections=[
        GraphNode(
            name="prefill",
            input_names={"prompt"},
            outputs=[
                GraphEdge(name="token", next_node="ar_decode"),
                GraphEdge(name="kv_cache", next_node="ar_decode"),
            ],
        ),
        Loop(
            name="ar_loop",
            section=GraphNode(
                name="ar_decode",
                input_names={"token", "kv_cache"},
                outputs=[
                    GraphEdge(name="token", next_node="ar_decode"),
                    GraphEdge(name="kv_cache", next_node="ar_decode"),
                ],
            ),
            outputs=[GraphEdge(name="tokens", next_node="EMIT_TO_CLIENT")],
            max_iters=1000,
        ),
    ])
    io = WorkerGraphIO(graph)
    decode_count = [0]

    def on_step(node_name, io_manager):
        if node_name != "ar_decode":
            return
        decode_count[0] += 1
        if decode_count[0] == 4:
            io_manager.register_loop_finish_signal("ar_loop")

    for _ in range(2):
        decode_count[0] = 0
        run_graph(io, [GraphEdge(name="prompt", next_node="prefill")],
                  on_node_complete=on_step)
        assert decode_count[0] == 4
        io.clear()


def test_nested_loops():
    graph = Sequential(sections=[
        Loop(
            name="refine_loop",
            section=Sequential(sections=[
                Loop(
                    name="denoise_loop",
                    section=GraphNode(
                        name="denoiser",
                        input_names={"latents"},
                        outputs=[GraphEdge(name="latents", next_node="denoiser")],
                    ),
                    outputs=[GraphEdge(name="latents", next_node="refiner")],
                    max_iters=3,
                ),
                GraphNode(
                    name="refiner",
                    input_names={"latents"},
                    outputs=[GraphEdge(name="latents", next_node="denoiser")],
                ),
            ]),
            outputs=[GraphEdge(name="latents", next_node="decoder")],
            max_iters=2,
        ),
        GraphNode(
            name="decoder",
            input_names={"latents"},
            outputs=[GraphEdge(name="image", next_node="EMIT_TO_CLIENT")],
        ),
    ])
    io = WorkerGraphIO(graph)
    log = []

    def record(node_name, io_manager):
        log.append(node_name)

    result = run_graph(io, [GraphEdge(name="latents", next_node="denoiser")],
                       on_node_complete=record)

    assert len(result) == 1 and result[0].name == "image"
    assert log.count("denoiser") == 6
    assert log.count("refiner") == 2
    assert log[-1] == "decoder"


def test_eos_clears_ready_signals():
    graph = Sequential(sections=[
        GraphNode(
            name="prefill",
            input_names={"prompt"},
            outputs=[
                GraphEdge(name="token", next_node="ar_decode"),
                GraphEdge(name="kv_cache", next_node="ar_decode"),
            ],
        ),
        Loop(
            name="ar_loop",
            section=GraphNode(
                name="ar_decode",
                input_names={"token", "kv_cache"},
                outputs=[
                    GraphEdge(name="token", next_node="ar_decode"),
                    GraphEdge(name="kv_cache", next_node="ar_decode"),
                ],
            ),
            outputs=[],
            max_iters=1000,
        ),
    ])
    io = WorkerGraphIO(graph)
    count = [0]

    def on_step(node_name, io_manager):
        if node_name != "ar_decode":
            return
        count[0] += 1
        if count[0] == 3:
            io_manager.register_loop_finish_signal("ar_loop")

    result = run_graph(io, [GraphEdge(name="prompt", next_node="prefill")],
                       on_node_complete=on_step)
    assert len(result) == 0
    ar_decode = io.nodes["ar_decode"]
    assert not ar_decode.ready_signals.ready_names
    assert not ar_decode.ready_next_iter.ready_names
