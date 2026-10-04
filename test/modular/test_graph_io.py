"""Regression tests for bug fixes in graph/base.py + graph_io.py.

Each test pins down a specific behavior that the prior implementation got
wrong; if any of these regress, the underlying bugs have re-surfaced.
"""

from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential
from mstar.graph.graph_io import WorkerGraphIO


def _ar_loop_graph(max_iters: int = 5):
    return Sequential(sections=[
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
            max_iters=max_iters,
        ),
    ])


def _drive(io: WorkerGraphIO, initial: list[GraphEdge], on_step=None,
           max_steps: int = 50):
    pending = list(initial)
    step = 0
    while not io.wg_state_registry.is_done and step < max_steps:
        remaining = []
        for edge in pending:
            if not io.ingest_input(edge):
                remaining.append(edge)
        pending = remaining

        ready = list(io.ready_node_names)
        if not ready:
            return step
        name = ready[0]
        io.ready_node_names.discard(name)
        step += 1
        if on_step:
            on_step(name, io)
        completion = io.mark_node_complete(name)
        for edge in completion.output_edges:
            if (edge.name, edge.next_node) in completion.filtered_signals:
                continue
            if edge.next_node in io.nodes:
                pending.append(edge)
    return step


def test_clear_resets_finish_signal():
    """After register_loop_finish_signal + clear(), the loop must run its
    full max_iters on the next forward pass — the finish signal does not
    persist."""
    io = WorkerGraphIO(_ar_loop_graph(max_iters=10))
    decode_count = [0]

    def on_step(name, io_mgr):
        if name != "ar_decode":
            return
        decode_count[0] += 1
        if decode_count[0] == 3:
            io_mgr.register_loop_finish_signal("ar_loop")

    _drive(io, [GraphEdge(name="prompt", next_node="prefill")], on_step=on_step)
    assert decode_count[0] == 3
    assert io.loops["ar_loop"]._finish_signal is True

    io.clear()
    assert io.loops["ar_loop"]._finish_signal is False
    assert io.loops["ar_loop"].is_done is False
    assert io.loops["ar_loop"].curr_iter == 0

    def count_only(name, _io):
        if name == "ar_decode":
            decode_count[0] += 1

    decode_count[0] = 0
    _drive(io, [GraphEdge(name="prompt", next_node="prefill")], on_step=count_only)
    # No finish signal this run; should run the full max_iters=10.
    assert decode_count[0] == 10


def test_top_level_node_ready_signals_clear_on_complete():
    """A top-level (non-loop) GraphNode's ready_signals must be cleared
    after mark_node_complete, so a future ingest doesn't fall through to
    ready_next_iter and leak across forward passes."""
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
    io.ingest_input(GraphEdge(name="prompt", next_node="encode"))
    assert io.nodes["encode"].ready_signals.ready_names == {"prompt"}

    io.ready_node_names.discard("encode")
    io.mark_node_complete("encode")

    # After completion, ready_signals must be empty so the same name can be
    # ingested for the next forward pass without falling through to
    # ready_next_iter (which has no semantic meaning for a top-level node).
    assert io.nodes["encode"].ready_signals.ready_names == set()
    assert io.nodes["encode"].ready_signals.is_ready is False


def test_speculation_strict_mode_partial_spec_not_ready():
    """A speculative ingest that covers only some inputs must NOT mark the
    node as ready, even when ready_signals already has the rest. The gate
    is on speculative_signals alone."""
    io = WorkerGraphIO(_ar_loop_graph())

    # Simulate the running iter having ingested both inputs (via prefill →
    # ar_decode, then we manually mark the slot full).
    ar_decode = io.nodes["ar_decode"]
    ar_decode.ready_signals.update(
        GraphEdge(name="token", next_node="ar_decode")
    )
    ar_decode.ready_signals.update(
        GraphEdge(name="kv_cache", next_node="ar_decode")
    )
    assert ar_decode.ready_signals.is_ready

    # Now speculatively ingest ONLY one of the two anticipated outputs.
    ready = io.ingest_for_speculation(
        [GraphEdge(name="token", next_node="ar_decode")], "ar_decode"
    )
    # Strict gate: speculative_signals alone doesn't cover input_names, so
    # the node must not be returned as ready despite the union with
    # ready_signals being a full set.
    assert ready == []

    ready = io.ingest_for_speculation(
        [GraphEdge(name="kv_cache", next_node="ar_decode")], "ar_decode"
    )
    assert len(ready) == 1
    assert ready[0].node_name == "ar_decode"
    assert ready[0].is_new_loop_iter is True


def test_triple_deliver_returns_false():
    """ingest_input called three times for the same edge name returns False
    on the third call — only ready_signals + ready_next_iter slots exist.
    Soft rejection (rather than raising) lets streaming back-pressure work:
    the worker's StreamBuffer re-queues the rejected edge until the consumer
    catches up. Same return semantics is used for the cross-walk persist case
    (edge name not in destination node's input_names)."""
    graph = Sequential(sections=[
        GraphNode(
            name="x",
            input_names={"a"},
            outputs=[],
        ),
    ])
    io = WorkerGraphIO(graph)
    edge = GraphEdge(name="a", next_node="x")
    assert io.ingest_input(edge) is True
    assert io.ingest_input(edge) is True
    assert io.ingest_input(edge) is False


def test_ingest_input_rejects_unknown_input_name():
    """An edge whose name is not in the destination node's input_names
    returns False (does NOT raise). Mirrors the Q3-Omni ``talker_input_embeds
    → Talker`` cross-walk persist edge that lands on a Talker node in the
    current walk that doesn't take that input."""
    graph = Sequential(sections=[
        GraphNode(
            name="x",
            input_names={"a"},
            outputs=[],
        ),
    ])
    io = WorkerGraphIO(graph)
    assert io.ingest_input(GraphEdge(name="not_an_input", next_node="x")) is False
    # The node's ready state must be untouched.
    assert io.nodes["x"].ready_signals.ready_names == set()


def test_parallel_preserves_outside_fed_member_inputs():
    """Parallel members fed from OUTSIDE the Parallel keep their inputs in
    ext_inputs. The prior implementation classified every member input as
    internal (its destination is always a member node), which emptied the
    IO of a decoder fan-out fed by an upstream loop."""
    from mstar.graph.base import Parallel

    par = Parallel([
        GraphNode(
            name="vae_decoder",
            input_names={"latents"},
            outputs=[GraphEdge(name="video_output", next_node="EMIT_TO_CLIENT")],
        ),
        GraphNode(
            name="audio_decoder",
            input_names={"sound_latents"},
            outputs=[GraphEdge(name="audio_output", next_node="EMIT_TO_CLIENT")],
        ),
    ])
    io = par.get_inputs_outputs()
    assert io.ext_inputs == {
        ("latents", "vae_decoder"), ("sound_latents", "audio_decoder")
    }
    assert {(e.name, e.next_node) for e in io.ext_outputs} == {
        ("video_output", "EMIT_TO_CLIENT"), ("audio_output", "EMIT_TO_CLIENT"),
    }
    assert io.loop_back == set()


def test_parallel_sibling_produced_edge_is_internal():
    """An edge produced by one member and consumed by another is internal:
    absent from both ext_inputs and ext_outputs. Inputs no sibling produces
    stay external."""
    from mstar.graph.base import Parallel

    par = Parallel([
        GraphNode(
            name="producer",
            input_names={"seed"},
            outputs=[GraphEdge(name="feat", next_node="consumer")],
        ),
        GraphNode(
            name="consumer",
            input_names={"feat", "cond"},
            outputs=[GraphEdge(name="out", next_node="EMIT_TO_CLIENT")],
        ),
    ])
    io = par.get_inputs_outputs()
    assert io.ext_inputs == {("seed", "producer"), ("cond", "consumer")}
    assert {(e.name, e.next_node) for e in io.ext_outputs} == {
        ("out", "EMIT_TO_CLIENT")
    }
    assert io.loop_back == set()


def test_cfg_branch_loop_io_artifacts():
    """A CFG-style loop — parallel denoise branches plus a combine node that
    feeds them back — derives no external inputs, classifies every branch
    feed as loop-back, and keeps the loop's declared output. This is BAGEL's
    image_gen_cfg shape; these artifacts are identical before and after the
    Parallel member-input fix (the branch feeds are produced by combine_cfg,
    so the enclosing Sequential reclassifies them as loop-back either way)."""
    from mstar.graph.base import Parallel

    branches = [
        GraphNode(
            name=n,
            input_names={"latents", "time_index"},
            outputs=[GraphEdge(name=v, next_node="combine_cfg")],
        )
        for n, v in [
            ("LLM", "v_main"),
            ("LLM_cfg_text", "v_cfg_text"),
            ("LLM_cfg_img", "v_cfg_img"),
        ]
    ]
    combine = GraphNode(
        name="combine_cfg",
        input_names={"v_main", "v_cfg_text", "v_cfg_img", "latents", "time_index"},
        outputs=[
            GraphEdge(name="latents", next_node="LLM"),
            GraphEdge(name="time_index", next_node="LLM"),
            GraphEdge(name="latents", next_node="LLM_cfg_text"),
            GraphEdge(name="time_index", next_node="LLM_cfg_text"),
            GraphEdge(name="latents", next_node="LLM_cfg_img"),
            GraphEdge(name="time_index", next_node="LLM_cfg_img"),
            GraphEdge(name="latents", next_node="combine_cfg"),
            GraphEdge(name="time_index", next_node="combine_cfg"),
        ],
    )
    loop = Loop(
        name="cfg_loop",
        section=Sequential([Parallel(branches), combine]),
        max_iters=10,
        outputs=[GraphEdge(name="latents", next_node="vae_decoder")],
    )
    assert loop._external_inputs == set()
    assert loop._loop_back_inputs == {
        ("latents", "LLM"), ("time_index", "LLM"),
        ("latents", "LLM_cfg_text"), ("time_index", "LLM_cfg_text"),
        ("latents", "LLM_cfg_img"), ("time_index", "LLM_cfg_img"),
        ("latents", "combine_cfg"), ("time_index", "combine_cfg"),
    }
    assert [(e.name, e.next_node) for e in loop.outputs] == [
        ("latents", "vae_decoder")
    ]


def test_streaming_inputs_propagate_to_ready_signals():
    """When _register_streaming is called on a GraphNode, the existing
    ReadySignals instances must see the new streaming names. Regression
    test for the rebound-set bug."""
    node = GraphNode(
        name="x",
        input_names={"a", "b"},
        outputs=[],
    )
    # Initially no streaming inputs known.
    assert node.ready_signals.streaming_inputs == set()

    # Simulate what _divide_into_worker_graphs does.
    node._register_streaming({"a"})

    # The ReadySignals objects captured the set BY REFERENCE in __post_init__,
    # so this should now see "a" without any further plumbing.
    assert node.ready_signals.streaming_inputs == {"a"}
    assert node.ready_next_iter.streaming_inputs == {"a"}
    assert node.speculative_signals.streaming_inputs == {"a"}
    assert node.consumes_stream is True


def test_queue_add_suppressed_in_flight_is_restored_when_the_flag_clears():
    """A node marked in-flight must not be queued by an arriving input, and must
    be queued once the flag clears -- otherwise the wake-up is lost and the
    request stalls.

    ``register_ingested_input`` only runs at ingest time and skips the add while
    ``_speculatively_scheduled`` is set, so clearing the flag has to re-evaluate
    membership. A loop member's add walks up to the root registry, which is where
    ``requeue_if_ready`` has to look.
    """
    io = WorkerGraphIO(_ar_loop_graph())
    registry = io.wg_state_registry
    decode = io.get_node("ar_decode")

    # Mark the node in flight, then deliver everything it needs.
    decode._speculatively_scheduled = True
    io.ingest_input(GraphEdge(name="token", next_node="ar_decode"))
    io.ingest_input(GraphEdge(name="kv_cache", next_node="ar_decode"))

    assert decode.ready_signals.is_ready, "inputs should still be recorded as ready"
    assert "ar_decode" not in registry.ready_names, \
        "an in-flight node must not be queued while the flag is set"

    # Clearing the flag must hand the node back to the scheduler.
    decode._speculatively_scheduled = False
    registry.requeue_if_ready("ar_decode")
    assert "ar_decode" in registry.ready_names, \
        "clearing the in-flight flag lost the queue add -- the request would stall"


def test_requeue_if_ready_is_a_noop_while_still_in_flight():
    io = WorkerGraphIO(_ar_loop_graph())
    registry = io.wg_state_registry
    decode = io.get_node("ar_decode")
    decode._speculatively_scheduled = True
    io.ingest_input(GraphEdge(name="token", next_node="ar_decode"))
    io.ingest_input(GraphEdge(name="kv_cache", next_node="ar_decode"))
    registry.requeue_if_ready("ar_decode")
    assert "ar_decode" not in registry.ready_names


def test_runtime_clear_flag_requeues_through_the_real_attribute_chain():
    """Exercise ``PythonGraphRuntime.set_speculatively_scheduled`` itself.

    The method reaches the registry as ``queues[rid].wg_state_registry``; writing
    it against ``WorkerGraphIO`` instead raises AttributeError only at runtime,
    which no parity test catches (they skip when the rid is absent from the
    queues). So drive the real method over a real WorkerGraphIO.
    """
    from types import SimpleNamespace

    from mstar.graph.runtime.python import PythonGraphRuntime

    io = WorkerGraphIO(_ar_loop_graph())
    decode = io.get_node("ar_decode")
    decode._speculatively_scheduled = True
    io.ingest_input(GraphEdge(name="token", next_node="ar_decode"))
    io.ingest_input(GraphEdge(name="kv_cache", next_node="ar_decode"))
    assert "ar_decode" not in io.wg_state_registry.ready_names

    stub = SimpleNamespace(_queues={7: SimpleNamespace(per_request_queues={42: io})})
    PythonGraphRuntime.set_speculatively_scheduled(stub, "ar_decode", 7, [42], False)

    assert decode._speculatively_scheduled is False
    assert "ar_decode" in io.wg_state_registry.ready_names, \
        "clearing the flag through the runtime did not re-queue the node"
