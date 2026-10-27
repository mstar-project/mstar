"""Outputs a request gets once encode on the data worker's pool: each is released
to its request when its own encode finishes, in the request's output order, while
streamed outputs are postprocessed inline as before. Which outputs those are comes
from the model's graph."""
from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import pytest
import torch

from mstar.api_server.data_worker import (
    PreprocessWorkerThread,
    RequestOutputState,
    encode_workers,
    one_shot_outputs,
)
from mstar.conductor.request_info import PartitionDefinition
from mstar.graph.base import GraphEdge, GraphNode, Loop, Parallel, Sequential
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.components.diffusion.image_io import png_workers
from mstar.streaming.topology import Connection, PartitionTopology, StreamingGraphEdge


class _Model:
    """postprocess blocks on the gate named by the output's value, so a test
    decides when each encode finishes."""

    def __init__(self):
        self.gates: dict[int, threading.Event] = {}
        self.threads: dict[int, str] = {}

    def gate(self, value: int) -> threading.Event:
        return self.gates.setdefault(value, threading.Event())

    def postprocess(self, output, modality, request_kwargs=None):
        value = int(output.flatten()[0])
        self.threads[value] = threading.current_thread().name
        if not self.gate(value).wait(timeout=10):
            raise TimeoutError(value)
        if value < 0:
            raise ValueError("bad pixels")
        return f"{modality}:{value}".encode()

    def get_partition_topology(self):
        return PartitionTopology(partitions=["default"])

    def get_output_sample_rate(self, modality="audio"):
        return 24000

    def get_output_audio_channels(self, modality="audio"):
        return 2


class _TensorManager:
    def __init__(self):
        self.ready: dict = {}
        self.tensors: dict[str, torch.Tensor] = {}
        self.dereferenced: list[str] = []

    def get_ready_tensors(self):
        ready, self.ready = self.ready, {}
        return ready

    def take_failed_reads(self):
        return []

    def get_tensor(self, uuid):
        return self.tensors[uuid]

    def dereference(self, uuid, n=1):
        self.dereferenced.append(uuid)

    def has_inflight_reads(self, request_id):
        return False

    def cleanup_request(self, request_id):
        pass

    def force_cleanup_request(self, request_id):
        pass


class _Comm:
    def get_all_new_messages(self):
        return []

    def send(self, entity, msg):
        pass


# what the fixture's model gets once per request
_POOLED = frozenset({("gen", "image"), ("gen", "audio")})


def _thread(pooled_outputs=_POOLED) -> PreprocessWorkerThread:
    return PreprocessWorkerThread(
        in_queue=queue.Queue(), result_tensor_queue=queue.Queue(), out_queue=queue.Queue(),
        profile_queue=queue.Queue(), cleanup_request_queue=queue.Queue(),
        abort_request_queue=queue.Queue(), reads_done_queue=queue.Queue(),
        discard_tensor_queue=queue.Queue(), stop_event=threading.Event(), communicator=_Comm(),
        tensor_manager=_TensorManager(), model=_Model(), pooled_outputs=pooled_outputs,
    )


@pytest.fixture
def worker():
    wt = _thread()
    yield wt
    for gate in wt.model.gates.values():
        gate.set()
    wt._encode_pool.shutdown(wait=True)


def _output(wt, request_id: str, modality: str, values: list[int], walk: str = "gen"):
    """``values`` arrive as ``request_id``'s next outputs from ``walk``, one tensor each."""
    state = wt.request_output_state.setdefault(request_id, RequestOutputState())
    infos = []
    for value in values:
        uuid = f"{request_id}/{state.next_sequence}"
        wt.tensor_manager.tensors[uuid] = torch.tensor([value])
        wt.tensor_uuid_to_metadata_per_request.setdefault(request_id, {})[uuid] = {"v": value}
        state.order[uuid] = (state.next_sequence, None)
        state.next_sequence += 1
        infos.append(SimpleNamespace(uuid=uuid, _source_graph_walk=walk))
    edge = SimpleNamespace(name=f"{modality}_output", tensor_info=infos)
    wt.tensor_manager.ready.setdefault(request_id, []).append(edge)


def _drain(wt) -> list:
    chunks = []
    while not wt.out_queue.empty():
        chunks.append(wt.out_queue.get_nowait())
    return chunks


def _pass_until(wt, n: int, timeout_s: float = 5.0) -> list:
    """Run the thread's read and release steps until ``n`` chunks are out."""
    chunks, deadline = [], threading.Event()
    timer = threading.Timer(timeout_s, deadline.set)
    timer.start()
    try:
        while len(chunks) < n and not deadline.is_set():
            wt._process_read_tensors()
            wt._release_encoded()
            chunks += _drain(wt)
    finally:
        timer.cancel()
    return chunks


@pytest.mark.parametrize(
    ("cpus", "workers"), [(1, 1), (4, 1), (7, 1), (8, 1), (15, 1), (16, 2), (24, 3), (32, 4), (40, 4), (256, 4)],
)
def test_the_pool_times_a_pngs_own_threads_fits_the_cpus(cpus, workers):
    assert encode_workers(cpus) == workers
    assert encode_workers(cpus) * png_workers(cpus) <= cpus


def test_an_image_is_encoded_off_the_thread_and_released_when_done(worker):
    _output(worker, "a", "image", [1])
    worker._process_read_tensors()
    assert _drain(worker) == [] and worker.tensor_manager.dereferenced == []
    worker.model.gate(1).set()
    (chunk,) = _pass_until(worker, 1)
    assert (chunk.request_id, chunk.modality, chunk.data, chunk.metadata) == ("a", "image", b"image:1", {"v": 1})
    assert worker.model.threads[1].startswith("mstar-encode")
    # the transport keeps its reference until the encode is released
    assert worker.tensor_manager.dereferenced == ["a/0"]
    assert worker.request_output_state["a"].pending == {}


def test_pooled_audio_carries_what_a_client_needs_to_play_it(worker):
    _output(worker, "a", "audio", [1])
    worker.model.gate(1).set()
    (chunk,) = _pass_until(worker, 1)
    assert worker.model.threads[1].startswith("mstar-encode")
    assert chunk.metadata == {"v": 1, "sample_rate": 24000, "num_channels": 2}


def test_an_output_its_walk_streams_stays_inline(worker):
    _output(worker, "a", "image", [1], walk="stream")
    worker.model.gate(1).set()
    worker._process_read_tensors()
    assert worker.model.threads[1] == threading.current_thread().name
    assert [c.data for c in _drain(worker)] == [b"image:1"]


def test_a_model_with_nothing_to_pool_starts_no_pool():
    wt = _thread(pooled_outputs=frozenset())
    assert wt._encode_pool is None
    _output(wt, "a", "image", [1])
    wt.model.gate(1).set()
    wt._process_read_tensors()
    assert wt.model.threads[1] == threading.current_thread().name
    assert [c.data for c in _drain(wt)] == [b"image:1"]


def test_a_finished_image_does_not_wait_for_another_requests_encode(worker):
    _output(worker, "slow", "image", [1])
    _output(worker, "fast", "image", [2])
    worker.model.gate(2).set()
    (chunk,) = _pass_until(worker, 1)
    assert chunk.request_id == "fast"
    worker.model.gate(1).set()
    (chunk,) = _pass_until(worker, 1)
    assert chunk.request_id == "slow"


def test_one_requests_images_keep_their_order(worker):
    _output(worker, "a", "image", [1, 2, 3])
    worker._process_read_tensors()
    worker.model.gate(3).set()
    worker.model.gate(2).set()
    assert _pass_until(worker, 1, timeout_s=0.3) == [], "a later image overtook the first"
    worker.model.gate(1).set()
    assert [c.data for c in _pass_until(worker, 3)] == [b"image:1", b"image:2", b"image:3"]


def test_a_streamed_chunk_after_an_image_waits_for_it_and_stays_inline(worker):
    _output(worker, "a", "image", [1])
    _output(worker, "a", "text", [2])
    worker.model.gate(2).set()
    worker._process_read_tensors()
    assert worker.model.threads[2] == threading.current_thread().name
    assert _drain(worker) == []
    worker.model.gate(1).set()
    assert [c.modality for c in _pass_until(worker, 2)] == ["image", "text"]


def test_a_failed_encode_answers_in_its_slot(worker):
    _output(worker, "a", "image", [-1])
    _output(worker, "a", "text", [2])
    worker.model.gate(2).set()
    worker.model.gate(-1).set()
    chunks = _pass_until(worker, 2)
    assert [c.modality for c in chunks] == ["error", "text"]
    assert chunks[0].metadata == {"status": 400}


def test_an_encode_finishing_after_teardown_delivers_nothing(worker):
    _output(worker, "a", "image", [1])
    worker._process_read_tensors()
    worker._drop_request_state("a")
    # the rid comes back as a new request before the old encode finishes
    worker.request_output_state["a"] = RequestOutputState()
    worker.model.gate(1).set()
    worker._encode_pool.shutdown(wait=True)
    worker._release_encoded()
    assert _drain(worker) == []
    assert worker.request_output_state["a"].next_emit == 0
    assert worker.tensor_manager.dereferenced == ["a/0"]


def test_a_thread_built_without_init_encodes_inline():
    wt = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    assert wt._encode_pool is None and wt._release_encoded() is False


# -- which outputs a request gets once, from the model's graph --------------

def _emit(name: str, modality: str) -> GraphEdge:
    return GraphEdge(next_node=EMIT_TO_CLIENT, name=name, output_modality=modality)


class _Graphs:
    """The three model methods one_shot_outputs reads."""

    def __init__(self, walks, partitions=None, connections=()):
        self.walks, self.partitions, self.connections = walks, partitions, list(connections)

    def get_graph_walk_graphs(self):
        return self.walks()

    def get_partitions(self):
        return self.partitions or [PartitionDefinition(name="default", graph_walks=set(self.walks()))]

    def get_partition_topology(self):
        return PartitionTopology(partitions=[p.name for p in self.get_partitions()], connections=self.connections)


def _denoise(*outputs: GraphEdge) -> Loop:
    return Loop(
        section=GraphNode(name="dit", input_names=["latents"], outputs=[GraphEdge(next_node="dit", name="latents")]),
        max_iters=4, outputs=[GraphEdge(next_node="vae", name="latents"), *outputs],
    )


def test_a_decoded_image_or_video_after_its_denoise_loop_is_pooled():
    model = _Graphs(lambda: {
        "encode_text": GraphNode(
            name="text_encoder", input_names=["text_inputs"],
            outputs=[GraphEdge(next_node=EMPTY_DESTINATION, name="text_embeds", persist=True)],
        ),
        "image_gen": Sequential([
            _denoise(), GraphNode(name="vae", input_names=["latents"], outputs=[_emit("image_output", "image")]),
        ]),
        "av_gen": Sequential([_denoise(), Parallel([
            GraphNode(name="vae", input_names=["latents"], outputs=[_emit("video_output", "video")]),
            GraphNode(name="audio_vae", input_names=["sound"], outputs=[_emit("audio_output", "audio")]),
        ])]),
    })
    assert one_shot_outputs(model) == {("image_gen", "image"), ("av_gen", "video"), ("av_gen", "audio")}


def test_a_loops_own_output_leaves_once():
    action = GraphEdge(next_node=EMIT_TO_CLIENT, name="latents", output_modality="action")
    model = _Graphs(lambda: {"action_gen": _denoise(action)})
    assert one_shot_outputs(model) == {("action_gen", "action")}


def test_the_prefills_first_token_streams_with_the_decode_loop_after_it():
    model = _Graphs(lambda: {
        "prefill": GraphNode(name="LLM", input_names=["text_inputs"], outputs=[_emit("new_token", "text")]),
        "decode": Loop(
            section=GraphNode(name="LLM", input_names=["text_inputs"], outputs=[
                GraphEdge(next_node="LLM", name="text_inputs"), _emit("text_inputs", "text"),
            ]),
            max_iters=8, outputs=[],
        ),
    })
    assert one_shot_outputs(model) == frozenset()


def test_a_modality_streamed_by_one_walk_stays_inline_in_the_others():
    # a rollout streams its video per iteration, so the batched rollout and the
    # one-pass walk stay inline too; the scalar the scorer emits once is pooled
    def walks():
        def rollout(per_iteration: bool) -> Loop:
            loopback = GraphEdge(next_node="predictor", name="hidden")
            return Loop(
                section=GraphNode(name="predictor", input_names=["hidden"], outputs=[
                    loopback, *([_emit("predicted", "video")] if per_iteration else []),
                ]),
                max_iters=4, outputs=[],
                accumulated_outputs=[] if per_iteration else [_emit("hidden", "video")],
            )
        return {
            "one_pass": GraphNode(name="predictor", input_names=["frames"], outputs=[_emit("predicted", "video")]),
            "rollout": rollout(per_iteration=False),
            "rollout_streaming": rollout(per_iteration=True),
            "score": GraphNode(name="scorer", input_names=["hidden"], outputs=[_emit("best", "scalar")]),
        }
    assert one_shot_outputs(_Graphs(walks)) == {("score", "scalar")}


def test_a_partition_a_stream_feeds_streams_what_it_emits():
    # the windowed walk streams latents to a decoder partition, which runs once
    # per window; the whole-clip walk of the producer partition still emits once
    def walks():
        return {
            "video_gen": Sequential([
                _denoise(), GraphNode(name="vae", input_names=["latents"], outputs=[_emit("video_output", "video")]),
            ]),
            "video_gen_windowed": Loop(
                section=GraphNode(name="dit", input_names=["latents"], outputs=[
                    GraphEdge(next_node="dit", name="latents"),
                    StreamingGraphEdge(next_node="vae_ar", name="window", target_partition="decoder"),
                ]),
                max_iters=8, outputs=[],
            ),
            "decode_window": GraphNode(name="vae_ar", input_names=["window"], outputs=[_emit("video_output", "video")]),
        }
    model = _Graphs(
        walks,
        partitions=[
            PartitionDefinition(name="default", graph_walks={"video_gen", "video_gen_windowed"}),
            PartitionDefinition(
                name="decoder", graph_walks={"decode_window"}, initial_walk="decode_window",
                producer_partitions=["default"],
            ),
        ],
        connections=[Connection(
            from_partition="default", to_partition="decoder", edge_name="window", chunk_policy_factory=lambda: None,
        )],
    )
    assert one_shot_outputs(model) == {("video_gen", "video")}
