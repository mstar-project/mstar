"""A node that consumes a stream keeps what its next chunk needs on loop-back edges of a loop around it.

Such a loop runs one iteration per chunk and ends on the stream's last one, its section may hold the consumer and a
node after it, a loop-back edge may carry a host tensor, and a consumer that declines the empty last chunk still ends
its loop and emits what the loop accumulated. A model that moves its per-request state onto that shape (CosyVoice3's
flow and HiFT, Cosmos3's decoder) has nothing else to keep it, so each test runs on both graph runtimes.
"""

from __future__ import annotations

import itertools

import pytest
import torch
from test_graph_runtime import _build_on_real_manager, _complete, _ingest, _ready

from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential, TensorPointerInfo
from mstar.graph.graph_io import WorkerGraphIO
from mstar.graph.runtime.base import ColumnarEdgeSpecs
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.utils.containers import ParallelList

CHUNKS = 3


def _token2wav(clip: bool = True) -> Loop:
    """The flow consumes the token stream and loops its voice back; HiFT vocodes the flow's mel and loops its own
    stream back. With ``clip`` the flow also reads the reference clip, from outside the loop."""
    flow = GraphNode(
        name="flow", input_names={"speech_tokens", "voice"} | ({"ref_audio"} if clip else set()),
        outputs=[GraphEdge(name="voice", next_node="flow"), GraphEdge(name="mel", next_node="hift")],
    )
    flow._register_streaming({"speech_tokens"})
    hift = GraphNode(
        name="hift", input_names={"mel", "stream"},
        outputs=[
            GraphEdge(name="stream", next_node="hift"),
            GraphEdge(name="audio_chunk", next_node=EMIT_TO_CLIENT, output_modality="audio"),
        ],
    )
    return Loop(name="t2w", section=Sequential([flow, hift]), max_iters=8, outputs=[])


def _decoder() -> Loop:
    """A decoder that consumes window latents, loops its tail back, and emits its frames once, at the end."""
    decoder = GraphNode(
        name="decoder", input_names={"window_latents", "decode_tail"},
        outputs=[GraphEdge(name="decode_tail", next_node="decoder"), GraphEdge(name="frames", next_node="decoder")],
    )
    decoder._register_streaming({"window_latents"})
    return Loop(
        name="decode", section=decoder, max_iters=8, outputs=[],
        accumulated_outputs=[GraphEdge(name="frames", next_node=EMIT_TO_CLIENT, output_modality="video")],
    )


class _Python:
    """Drives `WorkerGraphIO` as the worker does, routing each completion's edges back in."""

    def __init__(self, loop: Loop, nodes, tmp_path):
        del nodes, tmp_path
        self.io = WorkerGraphIO(loop)
        self._uuids = itertools.count(1)

    def ingest(self, name: str, node: str) -> None:
        assert self.io.ingest_input(GraphEdge(name=name, next_node=node)), f"{node} refused {name}"

    def chunk(self, name: str, node: str) -> None:
        assert self.io.ingest_input(GraphEdge(name=name, next_node=node), can_buffer=False), f"{node} refused {name}"

    def ready(self) -> set[str]:
        return set(self.io.ready_node_names)

    def inputs(self, node: str) -> dict[str, list[int]]:
        return {
            name: [info.uuid for info in edge.tensor_info]
            for name, edge in self.io.get_node(node).ready_signals.ready_inputs.items()
        }

    def complete(self, node: str, outputs: dict[str, list[torch.Tensor]]) -> tuple[dict[str, list[int]], list[int]]:
        """The uuids ``node`` wrote by signal, and those this completion emits to the client."""
        written = {name: [next(self._uuids) for _ in tensors] for name, tensors in outputs.items()}
        for edge in self.io.get_node(node).outputs:
            edge.tensor_info = [
                TensorPointerInfo(dims=[1], dtype=torch.float32, nbytes=4, address=0, stride=[1], uuid=uuid,
                                  source_session_id="s", source_entity="worker_0")
                for uuid in written.get(edge.name, [])
            ]
        # as the worker runs it: off the ready set, and in flight until its outputs are routed
        self.io.ready_node_names.discard(node)
        self.io.get_node(node)._in_flight = True
        emitted = []
        for edge in self.io.mark_node_complete(node).output_edges:
            if edge.next_node == EMIT_TO_CLIENT:
                emitted += [info.uuid for info in edge.tensor_info]
            elif edge.next_node in self.io.nodes:
                self.io.ingest_input(edge)
        self.io.get_node(node)._in_flight = False
        return written, emitted

    def stop(self, loop: str, node: str) -> None:
        del node
        self.io.register_loop_finish_signal(loop)

    def done(self) -> bool:
        return self.io.wg_state_registry.is_done


class _Rust:
    """Drives `RustGraphRuntime` over a real CPU tensor manager, as the worker does."""

    def __init__(self, loop: Loop, nodes, tmp_path):
        self.tm, self.runtime, self.rid = _build_on_real_manager(loop, nodes, tmp_path, loops={loop.name})

    def ingest(self, name: str, node: str) -> None:
        assert _ingest(self.runtime, self.rid, [GraphEdge(name=name, next_node=node)]) == [], f"{node} refused {name}"

    def chunk(self, name: str, node: str) -> None:
        """A stream chunk, as `InputIngest.poll_stream_buffers` offers one."""
        block = ColumnarEdgeSpecs.empty()
        block.add(self.rid, name, [], False, next_node=node)
        refused = self.runtime.ingest_inputs_batch(block, can_buffer=False, is_streaming=True)
        assert refused == [], f"{node} refused {name}"

    def ready(self) -> set[str]:
        return {node for node, _ in _ready(self.runtime)}

    def inputs(self, node: str) -> dict[str, list[int]]:
        edges = self.runtime.pop_rids(node, "w", [self.rid]).input_edges
        uuids = iter(edges.uuids)
        return {
            edges.signal_names[k]: list(itertools.islice(uuids, n))
            for k, n in zip(edges.signal_name_idxs, edges.tensors_per_edge, strict=True)
        }

    def complete(self, node: str, outputs: dict[str, list[torch.Tensor]]) -> tuple[dict[str, list[int]], list[int]]:
        # the step's consumed inputs are released before its outputs are routed, as `StepExecutor.postprocess` does
        self.tm.cleanup_collectable(*self.runtime.cleanup_consumed_inputs(node, [self.rid], [0]))
        signals = list(outputs)
        stored, out = _complete(self.tm, self.runtime, self.rid, node, outputs, signals)
        uuids = iter(stored.flat_uuids)
        written = {name: list(itertools.islice(uuids, n)) for name, n in zip(signals, stored.num_tensors, strict=True)}
        return written, list(out.register_uuids)

    def stop(self, loop: str, node: str) -> None:
        self.runtime.stop_loops_batched(
            partition="default", graph_walk="w", last_node_run=node, loop_names=ParallelList([self.rid], [[loop]]),
        )

    def done(self) -> bool:
        return self.runtime.is_partition_idle(self.rid, "default")


RUNTIMES = pytest.mark.parametrize("runtime", [_Python, _Rust], ids=["python", "rust"])


def _enter_token2wav(runtime) -> None:
    """The walk's entry edges, as the conductor sends them: the clip, and an empty seed for each loop-back."""
    runtime.ingest("ref_audio", "flow")
    runtime.ingest("voice", "flow")
    runtime.ingest("stream", "hift")


@RUNTIMES
def test_a_two_node_loop_runs_once_per_chunk(runtime, tmp_path):
    graph = runtime(_token2wav(), {"flow", "hift"}, tmp_path)
    _enter_token2wav(graph)
    runs = []
    for chunk in range(CHUNKS):
        assert graph.ready() == set(), "the flow waits for its chunk"
        graph.chunk("speech_tokens", "flow")
        assert graph.ready() == {"flow"}
        graph.complete("flow", {"voice": [torch.ones(3)], "mel": [torch.ones(2)]})
        if chunk == CHUNKS - 1:
            # the worker ends the consumer's loop when it takes the stream's last chunk
            graph.stop("t2w", "flow")
        assert graph.ready() == {"hift"}, "HiFT runs the chunk the flow just took, the last one included"
        graph.complete("hift", {"stream": [torch.ones(4)], "audio_chunk": [torch.ones(5)]})
        runs.append(chunk)
    assert runs == list(range(CHUNKS))
    assert graph.ready() == set() and graph.done(), "the loop should end with the chunk the stop came on"


@RUNTIMES
def test_a_consumer_that_reads_nothing_from_outside_its_loop_takes_every_chunk(runtime, tmp_path):
    graph = runtime(_token2wav(clip=False), {"flow", "hift"}, tmp_path)
    graph.ingest("voice", "flow")
    graph.ingest("stream", "hift")
    for _ in range(CHUNKS):
        # refused unless the voice the last chunk looped back counts once the loop advances
        graph.chunk("speech_tokens", "flow")
        assert graph.ready() == {"flow"}
        graph.complete("flow", {"voice": [torch.ones(3)], "mel": [torch.ones(2)]})
        graph.complete("hift", {"stream": [torch.ones(4)], "audio_chunk": [torch.ones(5)]})


@RUNTIMES
def test_a_loop_back_edge_carries_a_host_tensor(runtime, tmp_path):
    graph = runtime(_token2wav(), {"flow", "hift"}, tmp_path)
    _enter_token2wav(graph)
    graph.chunk("speech_tokens", "flow")
    written, _ = graph.complete("flow", {"voice": [torch.ones(3)], "mel": [torch.ones(2)]})
    graph.complete("hift", {"stream": [torch.ones(4)], "audio_chunk": [torch.ones(5)]})
    graph.chunk("speech_tokens", "flow")
    assert graph.inputs("flow")["voice"] == written["voice"], "the next chunk should read the voice the last one wrote"


@RUNTIMES
def test_a_declined_last_chunk_ends_the_loop_and_emits_its_outputs(runtime, tmp_path):
    graph = runtime(_decoder(), {"decoder"}, tmp_path)
    graph.ingest("decode_tail", "decoder")
    frames = []
    for _ in range(CHUNKS - 1):
        graph.chunk("window_latents", "decoder")
        written, emitted = graph.complete("decoder", {"decode_tail": [torch.ones(3)], "frames": [torch.ones(6)]})
        assert emitted == [], "frames wait for the loop's end"
        frames += written["frames"]
    graph.chunk("window_latents", "decoder")
    graph.stop("decode", "decoder")
    # declined: the node completes with nothing on its outputs
    _, emitted = graph.complete("decoder", {"decode_tail": [], "frames": []})
    assert emitted == frames, "the loop should emit every window's frames once it ends"
    assert graph.done()
