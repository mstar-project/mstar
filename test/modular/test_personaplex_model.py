"""Dummy-mode tests for the PersonaPlex model contract.

No weights, GPU, network or gated assets. The model is built through its real
constructor in dummy mode; the SentencePiece tokenizer is replaced by a stub.
"""

import sys
import types
from itertools import count
from pathlib import Path
from queue import SimpleQueue

import pytest
import torch

from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.api_server.request_types import ResultTensors
from mstar.communication.tensor_store import PythonTensorBookkeeping, TensorStore
from mstar.communication.tensors import SharedMemoryCommunicationManager, TensorCommunicationManager
from mstar.graph.base import GraphEdge, Loop, TensorPointerInfo
from mstar.graph.graph_io import WorkerGraphIO
from mstar.graph.runtime.base import ColumnarEdgeSpecs, RouteInput
from mstar.graph.runtime.python import PythonGraphRuntime
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.personaplex import config as cfg
from mstar.model.personaplex.personaplex_model import GENERATE_AUDIO_WALK, PersonaPlexModel
from mstar.model.personaplex.voice import load_voice_preset, resolve_voice_path
from mstar.utils.containers import ParallelList

CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs" / "personaplex.yaml")
LM_WALKS = (cfg.CONDITIONING_WALK, cfg.GENERATE_WALK, GENERATE_AUDIO_WALK)


class StubTokenizer:
    def encode(self, text):
        return [4 + (ord(ch) % 900) for ch in text]

    def id_to_piece(self, token):
        return f"▁w{token}"


def _write_voice(path: Path, frames: int = 5, **overrides) -> Path:
    state = {
        "embeddings": torch.randn(frames - 1, 1, 1, cfg.TEMPORAL_DIM),
        "cache": torch.randint(0, 2048, (1, 17, 4), dtype=torch.long),
    }
    state.update(overrides)
    torch.save({k: v for k, v in state.items() if v is not None}, path)
    return path


def _model(tmp_path=None, voice_frames: int = 5) -> PersonaPlexModel:
    voice_dir = None
    if tmp_path is not None:
        voice_dir = tmp_path / "voices"
        voice_dir.mkdir(exist_ok=True)
        _write_voice(voice_dir / "NATF2.pt", frames=voice_frames)
    model = PersonaPlexModel(voice_dir=None if voice_dir is None else str(voice_dir), skip_weight_loading=True)
    model.tokenizer = StubTokenizer()
    return model


def _info(name: str, dims: list[int]) -> TensorPointerInfo:
    return TensorPointerInfo(dims=dims, dtype="torch.int64", nbytes=8, address=0, stride=[1], uuid=f"uuid-{name}",
                             source_session_id="test:0", source_entity="test")


def _signals(n_emb=4, n_text=9, samples=1920 * 20 + 7):
    return {
        "audio_inputs": [_info("audio", [samples])],
        "voice_embeddings": [_info("voice_embeddings", [n_emb, 1, 1, 4096])],
        "voice_cache": [_info("voice_cache", [1, 17, 4])],
        "persona_tokens": [_info("persona_tokens", [n_text])],
    }


# --------------------------------------------------------------------------- walks and schedule


def test_walks_instantiate_with_one_lm_per_walk_and_async_off():
    walks = _model().get_graph_walk_graphs()
    assert set(walks) == {cfg.ENCODE_USER_AUDIO_WALK, *LM_WALKS, cfg.CODEC_WALK}
    assert set(walks[cfg.ENCODE_USER_AUDIO_WALK].get_nodes()) == {"MimiEncoder"}
    for walk in LM_WALKS:
        assert set(walks[walk].get_nodes()) == {"LM"}
        assert walks[walk].get_nodes()["LM"].enable_async_scheduling is False
    decoder = walks[cfg.CODEC_WALK]
    assert set(decoder.get_nodes()) == {"MimiDecoder"}
    # The decoder updates its per-request context in the forward, so no step is prepared ahead.
    assert decoder.consumes_stream is True and decoder.enable_async_scheduling is False
    assert _model().nodes == ["LM", "MimiDecoder", "MimiEncoder"]


def test_lm_loop_backs_are_consumed_and_outputs_routed():
    walks = _model().get_graph_walk_graphs()
    for walk, loop_backs in ((cfg.CONDITIONING_WALK, {"lm_state"}),
                             (cfg.GENERATE_WALK, {"lm_state", "agent_frame", "text_frame"})):
        lm = walks[walk].get_nodes()["LM"]
        assert {e.name for e in lm.outputs if e.next_node == "LM"} == loop_backs
        assert loop_backs <= set(lm.input_names)  # consumed, never rejected
    cond = walks[cfg.CONDITIONING_WALK]
    assert isinstance(cond, Loop) and cond.max_iters == cfg.MAX_TEMPORAL_FORWARDS
    assert [(e.name, e.next_node, e.persist) for e in cond.outputs] == [("lm_state", EMPTY_DESTINATION, True)]
    encode = walks[cfg.ENCODE_USER_AUDIO_WALK]
    assert [(e.name, e.persist) for e in encode.outputs] == [("user_audio_codes", True)]
    for walk in (cfg.GENERATE_WALK, GENERATE_AUDIO_WALK):
        loop = walks[walk]
        assert isinstance(loop, Loop) and loop.outputs == [] and loop.accumulated_outputs == []
        lm = loop.get_nodes()["LM"]
        streamed = [e for e in lm.outputs if e.is_streaming]
        assert [(e.name, e.next_node, e.target_partition) for e in streamed] == [
            ("agent_frame", "MimiDecoder", cfg.CODEC_PARTITION)
        ]
        text_edges = [e for e in lm.outputs if e.next_node == EMIT_TO_CLIENT]
        assert [(e.name, e.output_modality) for e in text_edges] == (
            [("text_frame", "text")] if walk == cfg.GENERATE_WALK else []
        )
    assert [(e.name, e.next_node, e.output_modality) for e in walks[cfg.CODEC_WALK].outputs] == [
        ("audio_output", EMIT_TO_CLIENT, "audio")
    ]


def test_partitions_stream_agent_frames_to_the_codec():
    model = _model()
    partitions = {p.name: p for p in model.get_partitions()}
    assert set(partitions) == {cfg.LM_PARTITION, cfg.CODEC_PARTITION}
    lm, codec = partitions[cfg.LM_PARTITION], partitions[cfg.CODEC_PARTITION]
    assert lm.graph_walks == {cfg.ENCODE_USER_AUDIO_WALK, *LM_WALKS} and lm.initial_walk == cfg.ENCODE_USER_AUDIO_WALK
    assert lm.producer_partitions == []
    assert codec.graph_walks == {cfg.CODEC_WALK} and codec.producer_partitions == [cfg.LM_PARTITION]
    (connection,) = model.get_partition_topology().connections
    assert (connection.from_partition, connection.to_partition, connection.edge_name) == (
        cfg.LM_PARTITION, cfg.CODEC_PARTITION, "agent_frame")
    policy = connection.chunk_policy_factory()
    assert type(policy).__name__ == "ScheduledLeftContextChunkPolicy"
    assert (policy._schedule, policy._chunk, policy._left_context) == (
        cfg.STREAM_CHUNK_SCHEDULE, cfg.STREAM_CHUNK_FRAMES, 0)


def test_codec_partition_carries_the_sample_budget_and_never_finishes_itself():
    """The codec ends by its stream's final chunk; it must not read the LM's schedule."""
    model = _model()
    signals = {"audio_inputs": _signals()["audio_inputs"]}  # nothing the LM partition needs
    args = model.get_initial_forward_pass_args(cfg.CODEC_PARTITION, ["audio"], ["audio", "text"], signals)
    assert args.full_metadata.graph_walk == cfg.CODEC_WALK and not args.request_done
    assert args.inputs == [] and args.unpersist_tensors == []
    assert args.step_metadata == {"num_samples": 1920 * 20 + 7, "num_user_frames": 21}
    again = model.get_partition_forward_pass_args(cfg.CODEC_PARTITION, args.full_metadata, {}, incoming_connections=[])
    assert not again.request_done and again.inputs == [] and again.step_metadata == args.step_metadata
    with pytest.raises(ValueError, match="audio_inputs"):
        model.get_initial_forward_pass_args(cfg.CODEC_PARTITION, ["audio"], ["audio"], {})


def _step(model, metadata, persist):
    return model.get_partition_forward_pass_args("default", metadata, persist, incoming_connections=[])


def test_schedule_encode_condition_generate_done_with_persisted_state():
    model = _model()
    signals = _signals()
    args = model.get_initial_forward_pass_args("default", ["audio"], ["audio", "text"], signals)
    meta = args.full_metadata
    assert meta.graph_walk == cfg.ENCODE_USER_AUDIO_WALK and meta.is_prefill
    assert meta.kwargs["walk_schedule"] == [cfg.ENCODE_USER_AUDIO_WALK, cfg.CONDITIONING_WALK, cfg.GENERATE_WALK]
    assert [(e.next_node, e.name) for e in args.inputs] == [("MimiEncoder", "audio_inputs")]
    assert args.step_metadata == {"num_voice_embeddings": 4, "num_text_tokens": 9, "num_user_frames": 21,
                                  "num_samples": 1920 * 20 + 7, "conditioning_frames": 4 + 12 + 9}
    assert {i.uuid for i in args.unpersist_tensors} == {"uuid-audio"}

    persist = {name: signals[name] for name in ("voice_embeddings", "voice_cache", "persona_tokens")}
    persist["user_audio_codes"] = [_info("codes", [1, 8, 21])]
    cond = _step(model, meta, persist)
    assert cond.full_metadata.graph_walk == cfg.CONDITIONING_WALK and not cond.request_done
    by_name = {e.name: e for e in cond.inputs}
    assert set(by_name) == {"voice_embeddings", "voice_cache", "persona_tokens", "lm_state"}
    assert all(e.next_node == "LM" for e in cond.inputs) and by_name["lm_state"].tensor_info == []
    assert {i.uuid for i in cond.unpersist_tensors} == {"uuid-voice_embeddings", "uuid-voice_cache",
                                                       "uuid-persona_tokens"}

    persist = {"user_audio_codes": persist["user_audio_codes"], "lm_state": [_info("lm_state", [1, 137])]}
    gen = _step(model, cond.full_metadata, persist)
    assert gen.full_metadata.graph_walk == cfg.GENERATE_WALK
    by_name = {e.name: e for e in gen.inputs}
    assert set(by_name) == {"lm_state", "user_audio_codes", "agent_frame", "text_frame"}
    assert by_name["lm_state"].tensor_info == persist["lm_state"]
    assert by_name["agent_frame"].tensor_info == [] and by_name["text_frame"].tensor_info == []
    assert {i.uuid for i in gen.unpersist_tensors} == {"uuid-lm_state", "uuid-codes"}

    done = _step(model, gen.full_metadata, {})
    assert done.request_done and done.inputs == []


def test_audio_only_requests_use_the_walk_without_text_emission():
    args = _model().get_initial_forward_pass_args("default", ["audio"], ["audio"], _signals())
    assert args.full_metadata.kwargs["walk_schedule"][-1] == GENERATE_AUDIO_WALK


@pytest.mark.parametrize("with_text,reverse_ready", [(False, False), (True, False), (True, True)],
                         ids=["audio-only", "audio-plus-text", "audio-read-first"])
def test_data_worker_emits_requested_modalities(with_text, reverse_ready):
    """Exercise transfer setup and the real edge-name to ResultChunk boundary.

    The graph runtime sends ``ResultTensors.modality = edge.output_modality``;
    ``_read_result_tensor`` renames the edge to ``f"{modality}_output"``, so
    the producer name ``text_frame`` never reaches postprocessing.
    """
    model = _model()
    walks = model.get_graph_walk_graphs()
    walk = walks[cfg.GENERATE_WALK if with_text else GENERATE_AUDIO_WALK]
    edges = [edge for edge in walk.get_nodes()["LM"].outputs if edge.next_node == EMIT_TO_CLIENT]
    edges += [walks[cfg.CODEC_WALK].outputs[0]]
    edges = [edge.clone() for edge in edges]
    assert [(e.name, e.output_modality) for e in edges] == (
        [("text_frame", "text"), ("audio_output", "audio")] if with_text else [("audio_output", "audio")]
    )
    tensors = {}
    for edge in edges:
        info = _info(edge.name, [1])
        edge.tensor_info = [info]
        tensors[info.uuid] = torch.tensor([4]) if edge.output_modality == "text" else torch.tensor([0.5])

    class ReadyTensors:
        def __init__(self):
            self.released = []
            self.ready = []

        def start_read_tensors(self, request_id, graph_edges):
            assert request_id == "request"
            self.ready.extend(graph_edges)

        def get_ready_tensors(self):
            return {"request": list(reversed(self.ready)) if reverse_ready else self.ready}

        def take_failed_reads(self):
            return []

        def get_tensor(self, uuid):
            return tensors[uuid]

        def dereference(self, uuid):
            self.released.append(uuid)

    worker = types.SimpleNamespace(
        tensor_manager=ReadyTensors(), model=model, out_queue=SimpleQueue(),
        request_model_kwargs={},
        tensor_uuid_to_metadata_per_request={},
        request_output_state={}, enable_nvtx=False,
    )
    worker._queue_completed_output = types.MethodType(PreprocessWorkerThread._queue_completed_output, worker)
    worker._fail_request = lambda _rid, exc, _stage, **_kw: pytest.fail(f"data worker failed: {exc}")
    for edge in edges:
        PreprocessWorkerThread._read_result_tensor(worker, ResultTensors(
            request_id="request", modality=edge.output_modality,
            graph_edge=edge, loop_indices=None, metadata={},
        ))
    assert [edge.name for edge in edges] == (["text_output", "audio_output"] if with_text else ["audio_output"])
    assert PreprocessWorkerThread._process_read_tensors(worker)
    chunks = [worker.out_queue.get_nowait() for _ in edges]
    assert [chunk.modality for chunk in chunks] == (["text", "audio"] if with_text else ["audio"])
    assert all(chunk.request_id == "request" for chunk in chunks)
    if with_text:
        assert chunks[0].data == b" w4"
    assert chunks[-1].data == (torch.tensor([0.5]) * 32767).to(torch.int16).numpy().tobytes()
    assert chunks[-1].metadata == {"sample_rate": 24000, "num_channels": 1}
    assert set(worker.tensor_manager.released) == set(tensors)


def test_conductor_backstops_budget_and_missing_inputs():
    model = _model()
    with pytest.raises(ValueError, match="context budget"):
        model.get_initial_forward_pass_args("default", ["audio"], ["audio"],
                                            _signals(n_emb=50, n_text=20, samples=1920 * 2918))
    model.get_initial_forward_pass_args("default", ["audio"], ["audio"],
                                        _signals(n_emb=50, n_text=20, samples=1920 * 2917))
    signals = _signals()
    del signals["voice_cache"]
    with pytest.raises(ValueError, match="voice_cache"):
        model.get_initial_forward_pass_args("default", ["audio"], ["audio"], signals)


def test_worker_graphs_from_yaml_stream_agent_frames():
    model = _model()
    graphs = model.get_worker_graphs(CONFIG_PATH)
    assert {walk for wg in graphs for walk in wg.graph_walks} == {cfg.ENCODE_USER_AUDIO_WALK, *LM_WALKS, cfg.CODEC_WALK}
    loops = [loop for wg in graphs if cfg.GENERATE_WALK in wg.graph_walks
             for loop in wg.section.get_loops().values()]
    assert [loop.accumulated_outputs for loop in loops] == [[]]
    assert all(wg.ranks == [0] for wg in graphs)


@pytest.fixture
def triton_int64(monkeypatch):
    """The Mac conftest's fake triton lacks ``int64``, which an upstream kernel module
    annotates with when the worker graph path imports it. Provide only that annotation."""
    language = sys.modules.get("triton.language")
    if language is not None and not hasattr(language, "int64"):
        monkeypatch.setattr(language, "int64", int, raising=False)


def _personaplex_runtime(tensor_manager):
    model = _model()
    graphs = model.get_worker_graphs(CONFIG_PATH)
    walks = {wg.worker_graph_id: wg.graph_walks for wg in graphs}
    loops = {wg.worker_graph_id: set(wg.section.get_loops()) for wg in graphs}
    nodes = {wg.worker_graph_id: set(wg.section.get_nodes()) for wg in graphs}
    runtime = PythonGraphRuntime(
        my_worker_id="worker_0", my_worker_graphs=graphs,
        all_wg_ids_to_graph_walks=walks, all_wg_ids_to_dyn_loops=loops,
        all_wg_ids_to_nodes=nodes,
        node_to_partition={"MimiEncoder": cfg.LM_PARTITION, "LM": cfg.LM_PARTITION, "MimiDecoder": cfg.CODEC_PARTITION},
        sharding_config=model.get_sharding_config(CONFIG_PATH), tensor_manager=tensor_manager,
    )
    generate_ids = [wg.worker_graph_id for wg in graphs if cfg.GENERATE_WALK in wg.graph_walks]
    mapping = ParallelList([wg.worker_graph_id for wg in graphs], [["worker_0"] for _ in graphs])
    # What the conductor sends when a request names no caps: every loop at its max_iters.
    caps = {name: loop.max_iters for wg in graphs for name, loop in wg.section.get_loops().items()}
    rid = runtime.add_request("request", cfg.LM_PARTITION, cfg.GENERATE_WALK, generate_ids, mapping, caps)
    return runtime, rid


def test_python_graph_runtime_admits_personaplex_generate_walk(triton_int64):
    """The new runtime must accept the split PersonaPlex worker graphs."""
    runtime, rid = _personaplex_runtime(_Comm())
    assert runtime.get_rid_string(rid) == "request"
    assert {"lm_state", "agent_frame", "text_frame"} <= set(
        runtime.get_output_signals("LM", cfg.GENERATE_WALK))
    assert runtime.get_output_signals("MimiDecoder", cfg.CODEC_WALK) == ["audio_output"]
    runtime.remove_request(rid)


def test_python_runtime_streams_agent_frames_and_emits_text(tmp_path, triton_int64):
    """Do not defer text backed by the worker's reused stop-check host buffer; stream each agent frame.

    Real Python graph routing and SHM: emulate the GPU worker's borrowed
    CPU copy, then reuse it on the next step. Agent codes leave every step on
    the stream to the codec partition, in order.
    """
    manager = SharedMemoryCommunicationManager(
        my_entity_id="worker_0", hostname="localhost", device="cpu",
        communicator=types.SimpleNamespace(send=lambda *a, **kw: None), shm_dir=str(tmp_path),
    )
    runtime, rid = _personaplex_runtime(manager)
    manager.register_request(rid, runtime.get_sharding_config(rid))
    receiver = SharedMemoryCommunicationManager(
        my_entity_id="api_server", hostname="localhost", device="cpu",
        communicator=types.SimpleNamespace(send=lambda *a, **kw: None), shm_dir=str(tmp_path),
    )
    receiver.register_request(rid, runtime.get_sharding_config(rid))
    host_text = torch.empty(1, dtype=torch.int64)
    received_text, emitted_text = [], []
    registered = set()
    block = ColumnarEdgeSpecs.empty()
    for name in ("lm_state", "user_audio_codes", "agent_frame", "text_frame"):
        block.add_edge(rid, GraphEdge(next_node="LM", name=name))
    assert runtime.ingest_inputs_batch(block) == []
    lm_wg = runtime.get_worker_graph_id_for_node("LM", cfg.GENERATE_WALK)
    outputs = runtime.get_output_signals("LM", cfg.GENERATE_WALK)
    agent_uuids, text_uuids, streamed = [], [], []
    for frame in range(2):
        if frame == 1:
            runtime.stop_loops_batched(
                cfg.LM_PARTITION, cfg.GENERATE_WALK, "LM", ParallelList([rid], [[cfg.DECODE_LOOP]]),
            )
        host_text.fill_(frame + 4)
        stored = manager.store_and_return_tensor_info_batch(
            [rid], {rid: {
                "lm_state": [torch.tensor([frame])],
                "agent_frame": [torch.full((8,), frame)],
                "text_frame": [torch.tensor([frame + 4])],
            }}, outputs, cpu_tensors={rid: {"text_frame": [host_text]}},
        )
        manager.increment_ref_batch_uniform(stored.flat_uuids, 1)
        agent_uuids.append(stored.flat_uuids[outputs.index("agent_frame")])
        text_uuids.append(stored.flat_uuids[outputs.index("text_frame")])
        routed = runtime.complete_and_route_batch(RouteInput(
            partition=cfg.LM_PARTITION, graph_walk=cfg.GENERATE_WALK, node_name="LM",
            output_signals=outputs, wg_ids=ParallelList([rid], [lm_wg]),
            tensors=stored.flat_uuids, num_tensors=stored.num_tensors,
        ))
        registered.update(routed.register_uuids)
        if routed.register_uuids:
            manager.register_for_send_uuids(ParallelList([rid], [routed.register_uuids]))
        completion = runtime._peek_completion(routed.completion_id).routing[rid]
        stream_edges = completion.streaming_local + [e for es in completion.streaming_to_workers.values() for e in es]
        streamed.extend((e.name, e.next_node, [info.uuid for info in e.tensor_info]) for e in stream_edges)
        for edge in completion.emit_to_client:
            emitted_text.append((edge.name, [info.uuid for info in edge.tensor_info]))
            receiver.start_read_tensors(rid, [edge], graph_walk=cfg.GENERATE_WALK)
            receiver.get_ready_tensors(graph_walk=cfg.GENERATE_WALK)
            received_text.extend(int(receiver.get_tensor(info.uuid).item()) for info in edge.tensor_info)
    assert received_text == [4, 5], "deferred SHM sends must not read a later step's borrowed host copy"
    assert emitted_text == [("text_frame", [uuid]) for uuid in text_uuids]
    assert streamed == [("agent_frame", "MimiDecoder", [uuid]) for uuid in agent_uuids]
    # The agent frames ride the stream; nothing routes them to a decoder in this walk.
    completion = runtime._peek_completion(routed.completion_id).routing[rid]
    assert "agent_frame" not in {edge.name for edge in completion.routed_to_this_worker_graph}


# --------------------------------------------------------------------------- reference counts through a walk


class _Comm:
    """Use the real Python tensor bookkeeper with the current graph-I/O calls."""

    def __init__(self):
        self.tensor_store = TensorStore(bookkeeping=PythonTensorBookkeeping())

    def increment_ref(self, uuid, n=1):
        self.tensor_store.increment_ref(uuid, n=n)

    def dereference(self, uuid, n=1):
        self.tensor_store.dereference(uuid, n=n)

    def dereference_batch_uniform(self, uuids, n=1):
        self.tensor_store.dereference_batch_uniform(uuids, n=n)

    def dereference_batch(self, uuids, counts):
        self.tensor_store.dereference_batch(uuids, counts)

    set_output_ref_counts = TensorCommunicationManager.set_output_ref_counts


def _drive(walk, initial, *, finish_after=None, rid=1):
    """Run a walk on one worker the way worker.py's post-forward path does.

    Real: WorkerGraphIO, Loop caching, NodeInputs.clear, TensorStore and
    set_output_ref_counts. The single-worker routing and safety hold are
    transcribed from PythonGraphRuntime.complete_and_route_batch.
    """
    comm = _Comm()
    io = WorkerGraphIO(walk)
    io.register_communication_info(comm, rid)
    uid, produced, runs = count(), {}, {}
    out = {"client": [], "streamed": [], "rejected": [], "persisted": []}

    def tensor(name, it):
        uuid = next(uid)
        produced[uuid] = (name, it)
        info = TensorPointerInfo(dims=[1], dtype=torch.int64, nbytes=8, address=0, stride=[1], uuid=uuid,
                                 source_session_id="s", source_entity="w0")
        comm.tensor_store.put_tensor(rid, uuid, torch.tensor([it]), info)
        return info

    pending = []
    for edge in initial:
        if edge.tensor_info == ["external"]:
            info = tensor(edge.name, -1)
            comm.increment_ref(info.uuid)  # held by the conductor's persisted forward-pass arg
            edge.tensor_info = [info]
        pending.append(edge)
    while not io.wg_state_registry.is_done:
        pending = [e for e in pending if not io.ingest_input(e)]
        ready = sorted(io.ready_node_names)
        assert ready, "walk stalled"
        name = ready[0]
        io.ready_node_names.discard(name)
        node = io.nodes[name]
        it = runs.get(name, 0)
        runs[name] = it + 1
        node.reset_outputs()
        hold = set()
        for out_name in dict.fromkeys(e.name for e in node.outputs):
            info = tensor(out_name, it)
            comm.increment_ref(info.uuid)
            hold.add(info.uuid)
            for e in node.outputs:
                if e.name == out_name:
                    e.tensor_info = [info]
        if name == "LM" and finish_after is not None and runs["LM"] == finish_after:
            io.register_loop_finish_signal(next(iter(io.loops)))
        node.ready_signals.clear()
        completion = io.mark_node_complete(name)
        outs = [e.clone() for e in completion.output_edges]
        routed = []
        for e in outs:
            if e.next_node in io.nodes:
                if not io.ingest_input(e):
                    out["rejected"].append((e.name, it))
                routed.append(e)
            elif e.next_node == EMIT_TO_CLIENT:
                routed.append(e)
                out["client"].append((e.name, [produced[i.uuid] for i in e.tensor_info]))
            elif e.is_streaming:
                # One reader in the consumer partition, which releases it after reading.
                routed.append(e)
                out["streamed"].append((e.name, e.next_node, [produced[i.uuid] for i in e.tensor_info]))
            elif e.persist:
                out["persisted"].append((e.name, [produced[i.uuid] for i in e.tensor_info]))
        comm.set_output_ref_counts(hold, routed, completion.transferred_refs)
        for e in outs:
            if e.next_node == EMIT_TO_CLIENT or e.is_streaming:
                for i in e.tensor_info:
                    comm.dereference(i.uuid)
    refs = {produced[u]: comm.tensor_store.bookkeeping._ref_info[u].ref_cnt
            for u in produced if produced[u][1] >= 0}
    return out, runs, refs


def _generate_inputs():
    return [GraphEdge(next_node="LM", name="user_audio_codes", tensor_info=["external"]),
            GraphEdge(next_node="LM", name="lm_state", tensor_info=["external"]),
            GraphEdge(next_node="LM", name="agent_frame"), GraphEdge(next_node="LM", name="text_frame")]


@pytest.mark.parametrize("finish_after", [1, 7, 25])
def test_generate_streams_frames_in_order_and_completes(finish_after):
    walk = _model().get_graph_walk_graphs()[cfg.GENERATE_WALK]
    out, runs, refs = _drive(walk, _generate_inputs(), finish_after=finish_after)
    assert runs == {"LM": finish_after}
    assert out["streamed"] == [("agent_frame", "MimiDecoder", [("agent_frame", i)]) for i in range(finish_after)]
    assert out["client"] == [("text_frame", [("text_frame", i)]) for i in range(finish_after)]
    assert out["rejected"] == []
    assert all(r >= 0 for r in refs.values())
    # Every tensor of every non-final iteration is released.
    assert {k: r for k, r in refs.items() if r and k[1] != finish_after - 1} == {}
    assert refs[("lm_state", finish_after - 1)] == 0


def test_generate_runs_to_max_iters_without_a_finish_signal():
    model = _model()
    walk = model._generate_walk(with_text=False)
    walk.max_iters = 5
    out, runs, _ = _drive(walk, _generate_inputs())
    assert runs == {"LM": 5}
    assert [frames for *_, frames in out["streamed"]] == [[("agent_frame", i)] for i in range(5)]
    assert out["client"] == []


def test_final_iteration_streamed_tensors_are_released():
    walk = _model().get_graph_walk_graphs()[cfg.GENERATE_WALK]
    _, _, refs = _drive(walk, _generate_inputs(), finish_after=7)
    assert refs[("agent_frame", 6)] == 0 and refs[("text_frame", 6)] == 0


def test_conditioning_persists_only_the_final_lm_state():
    walk = _model().get_graph_walk_graphs()[cfg.CONDITIONING_WALK]
    initial = [GraphEdge(next_node="LM", name=n, tensor_info=["external"])
               for n in ("voice_embeddings", "voice_cache", "persona_tokens")]
    initial.append(GraphEdge(next_node="LM", name="lm_state"))
    out, runs, refs = _drive(walk, initial, finish_after=25)
    assert runs == {"LM": 25}
    assert out["persisted"] == [("lm_state", [("lm_state", 24)])]
    assert out["rejected"] == [] and out["client"] == []
    assert {k: r for k, r in refs.items() if r and k != ("lm_state", 24)} == {}


# --------------------------------------------------------------------------- request validation


_UNSET = object()


def _fake_torchcodec(monkeypatch, samples: torch.Tensor, *, header_seconds=_UNSET, open_error=None):
    """torchcodec stand-in; ``seen["decoded_until"]`` records how far it was asked to decode."""
    seen = {}

    class AudioDecoder:
        def __init__(self, path, **kwargs):
            if open_error is not None:
                raise open_error
            seen.update(path=path, **kwargs)
            header = samples.shape[-1] / 24000 if header_seconds is _UNSET else header_seconds
            self.metadata = types.SimpleNamespace(duration_seconds_from_header=header)

        def get_samples_played_in_range(self, start_seconds=0.0, stop_seconds=None):
            seen["decoded_until"] = stop_seconds
            stop = samples.shape[-1] if stop_seconds is None else round(stop_seconds * 24000)
            return types.SimpleNamespace(data=samples[..., round(start_seconds * 24000):stop])

    decoders = types.ModuleType("torchcodec.decoders")
    decoders.AudioDecoder = AudioDecoder
    monkeypatch.setitem(sys.modules, "torchcodec", types.ModuleType("torchcodec"))
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", decoders)
    return seen


def test_load_audio_keeps_channel_zero_at_24khz(monkeypatch):
    stereo = torch.stack((torch.full((5000,), 0.25), torch.full((5000,), -0.75)))
    seen = _fake_torchcodec(monkeypatch, stereo)
    loaded = _model().load_audio("in.wav", "cpu")
    assert {k: seen[k] for k in ("path", "sample_rate")} == {"path": "in.wav", "sample_rate": 24000}
    assert "num_channels" not in seen  # native channels requested, not a downmix
    assert torch.equal(loaded.data, stereo[0]) and loaded.data.dtype == torch.float32
    assert loaded.metadata == {"sample_rate": 24000, "num_channels": 1, "source_channels": 2, "num_samples": 5000}


def test_load_audio_reports_undecodable_upload_as_client_error(monkeypatch):
    error = RuntimeError("Could not open input file: /tmp/mstar_uploads/x.wav Invalid data found")
    _fake_torchcodec(monkeypatch, torch.zeros(1, 10), open_error=error)
    with pytest.raises(ValueError, match="cannot decode the uploaded audio") as info:
        _model().load_audio("/tmp/mstar_uploads/x.wav", "cpu")
    assert "/tmp/mstar_uploads" not in str(info.value)


def test_load_audio_rejects_a_long_header_before_decoding(monkeypatch):
    seen = _fake_torchcodec(monkeypatch, torch.zeros(1, 10), header_seconds=2 * 3600.0)
    with pytest.raises(ValueError, match=r"longer than 239\.9 s.*\(7200\.0 s\)"):
        _model().load_audio("long.mp3", "cpu")
    assert "decoded_until" not in seen


def test_load_audio_bounds_decoding_when_the_header_understates_length(monkeypatch):
    limit = cfg.MAX_TEMPORAL_FORWARDS * cfg.FRAME_SIZE
    seen = _fake_torchcodec(monkeypatch, torch.zeros(1, limit + 5 * cfg.FRAME_SIZE), header_seconds=None)
    with pytest.raises(ValueError, match="longer than 239.9 s"):
        _model().load_audio("lying-header.mp3", "cpu")
    assert seen["decoded_until"] == pytest.approx((limit + cfg.FRAME_SIZE) / 24000)


def test_load_audio_accepts_exactly_the_sample_limit(monkeypatch):
    limit = cfg.MAX_TEMPORAL_FORWARDS * cfg.FRAME_SIZE
    _fake_torchcodec(monkeypatch, torch.zeros(1, limit))
    assert _model().load_audio("max.wav", "cpu").data.numel() == limit


def test_load_audio_missing_decoder_runtime_stays_a_server_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "torchcodec", types.ModuleType("torchcodec"))
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", types.ModuleType("torchcodec.decoders"))
    with pytest.raises(ImportError):
        _model().load_audio("in.wav", "cpu")


def _request(model, *, samples=1920 * 3, prompt=None, outputs=("audio", "text"), **kwargs):
    kwargs.setdefault("voice", "NATF2")
    return model.process_prompt(prompt, ["audio"], list(outputs), {"audio_inputs": [torch.zeros(samples)]},
                                **kwargs).new_input_tensors


def test_process_prompt_builds_conditioning_tensors(tmp_path):
    model = _model(tmp_path, voice_frames=5)
    out = _request(model, text_prompt="Be kind.")
    assert set(out) == {"voice_embeddings", "voice_cache", "persona_tokens"}
    assert out["voice_embeddings"][0].shape == (4, 1, 1, 4096) and out["voice_cache"][0].shape == (1, 17, 4)
    expected = StubTokenizer().encode("<system> Be kind. <system>")
    assert out["persona_tokens"][0].tolist() == expected
    default = _request(model)["persona_tokens"][0].tolist()
    assert default == StubTokenizer().encode(f"<system> {cfg.DEFAULT_TEXT_PROMPT} <system>")
    assert _request(model, prompt="<system> Hi <system>")["persona_tokens"][0].tolist() == \
        StubTokenizer().encode("<system> Hi <system>")


def test_kv_layout_and_loop_caps_match_the_frames_each_loop_runs(tmp_path):
    """One KV entry per LM iteration; each loop is capped at the frames its stop check runs."""
    from mstar.model.personaplex.personaplex_model import LOOP_FRAMES_KWARG

    model = _model(tmp_path, voice_frames=5)
    out = model.process_prompt(None, ["audio"], ["audio"], {"audio_inputs": [torch.zeros(1920 * 3 + 5)]},
                               voice="NATF2", text_prompt="Be kind.")
    tokens = len(StubTokenizer().encode("<system> Be kind. <system>"))
    frames = out.metadata[LOOP_FRAMES_KWARG]
    assert frames == {cfg.CONDITIONING_LOOP: 4 + 2 * cfg.SILENCE_FRAMES + tokens, cfg.DECODE_LOOP: 4}
    layout = out.metadata["kv_layout"][cfg.LM_KV][cfg.KV_LABEL]
    assert layout.spans == () and layout.growth == ((cfg.CONDITIONING_LOOP, 1), (cfg.DECODE_LOOP, 1))
    assert model.get_loop_caps(**out.metadata) == frames
    assert model.get_loop_caps() == {}  # before process_prompt: each loop's max_iters
    assert LOOP_FRAMES_KWARG in model.INTERNAL_KWARGS  # the data worker drops a client's value


def test_kv_declarations_pass_the_worker_startup_checks():
    from mstar.validation.resources import check_resources

    model = _model()
    specs = model.get_node_resources()
    assert check_resources(model, specs, {}) == []
    growth = model.kv_labels()[cfg.LM_KV][cfg.KV_LABEL].growth
    assert set(growth) == {(cfg.CONDITIONING_WALK, cfg.CONDITIONING_LOOP), (cfg.GENERATE_WALK, cfg.DECODE_LOOP),
                           (cfg.GENERATE_AUDIO_WALK, cfg.DECODE_LOOP)}


@pytest.mark.parametrize(("kwargs", "match"), [
    (dict(text_prompt="   "), "non-empty"),
    (dict(prompt="a", text_prompt="b"), "not both"),
    (dict(samples=0), "empty"),
    (dict(outputs=("text",)), "outputs audio"),
    (dict(outputs=("audio", "image")), "outputs audio"),
    (dict(voice=None), "voice"),
    (dict(voice="../NATF2"), "not a path"),
    (dict(voice="missing"), r"available presets: \['NATF2'\]"),
    (dict(temp_text=-1.0), "temp_text"),
    (dict(temp_audio=True), "temp_audio"),
    # float() overflows on these; an OverflowError would be a 500 at the data worker
    (dict(temp_text=10**400), "temp_text"),
    (dict(temp_audio=-(10**400)), "temp_audio"),
    (dict(topk_audio=0), "topk_audio"),
    (dict(greedy="yes"), "greedy"),
    (dict(samples=1920 * 2966), "context budget"),
])
def test_process_prompt_rejects_bad_requests(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _request(_model(tmp_path), **kwargs)


def test_process_prompt_accepts_the_budget_boundary(tmp_path):
    model = _model(tmp_path, voice_frames=5)
    text = len(StubTokenizer().encode(f"<system> {cfg.DEFAULT_TEXT_PROMPT} <system>"))
    frames = cfg.MAX_TEMPORAL_FORWARDS - 4 - 12 - text
    _request(model, samples=1920 * frames)
    with pytest.raises(ValueError, match="context budget"):
        _request(model, samples=1920 * frames + 1)


def test_process_prompt_rejects_missing_or_multiple_audio(tmp_path):
    model = _model(tmp_path)
    for tensors in ({}, {"audio_inputs": [torch.zeros(10), torch.zeros(10)]}):
        with pytest.raises(ValueError, match="exactly one user audio"):
            model.process_prompt(None, ["audio"], ["audio"], tensors, voice="NATF2")
    with pytest.raises(ValueError, match="voice_dir"):
        _request(_model(), voice="NATF2")


class _Evil:
    def __reduce__(self):
        return (print, ("unpickled!",))


@pytest.mark.parametrize(("overrides", "match"), [
    (dict(embeddings=None), "exactly 'embeddings' and 'cache'"),
    (dict(extra=torch.zeros(1)), "exactly 'embeddings' and 'cache'"),
    (dict(embeddings=torch.zeros(3, 1, 1, 8)), r"\[N, 1, 1, 4096\]"),
    (dict(embeddings=torch.zeros(0, 1, 1, 4096)), "non-empty"),
    (dict(embeddings=torch.full((2, 1, 1, 4096), float("nan"))), "finite"),
    (dict(cache=torch.zeros(1, 17, 3, dtype=torch.long)), r"\[1, 17, 4\]"),
    (dict(cache=torch.zeros(1, 17, 4, dtype=torch.int32)), "int64"),
    (dict(cache=torch.full((1, 17, 4), 2049, dtype=torch.long)), "outside"),
    (dict(cache=torch.full((1, 17, 4), -3, dtype=torch.long)), "outside"),
    (dict(cache=_Evil()), "cannot load"),
])
def test_voice_preset_validation(tmp_path, overrides, match):
    path = _write_voice(tmp_path / "bad.pt", **overrides)
    with pytest.raises(ValueError, match=match):
        load_voice_preset(path)


def test_voice_preset_accepts_sentinels_and_initial_tokens(tmp_path):
    cache = torch.full((1, 17, 4), 2048, dtype=torch.long)
    cache[0, 0] = 32000
    cache[0, :, 3] = -2
    preset = load_voice_preset(_write_voice(tmp_path / "ok.pt", cache=cache))
    assert preset.num_embeddings == 4
    assert resolve_voice_path(tmp_path, "ok") == (tmp_path / "ok.pt").resolve()
    assert resolve_voice_path(tmp_path, "ok.pt") == (tmp_path / "ok.pt").resolve()


# --------------------------------------------------------------------------- dummy mode and I/O


def test_dummy_mode_builds_without_submodules_or_network():
    model = PersonaPlexModel(skip_weight_loading=True)
    assert model.tokenizer is None and model.model_path_hf == "nvidia/personaplex-7b-v1"
    assert all(model.get_submodule(node) is None for node in model.nodes)
    specs = {spec.resource_key: spec for spec in model.get_node_resources()}
    assert set(specs) == {cfg.LM_KV, cfg.LM_ATTN, cfg.LM_POS, cfg.TEXT_SAMPLER, cfg.DEPTH_SAMPLER}
    assert all(spec.nodes == {"LM"} for spec in specs.values())
    kv = specs[cfg.LM_KV].config
    assert (kv.num_layers, kv.num_kv_heads, kv.num_qo_heads, kv.head_dim, kv.max_seq_len) == (32, 32, 32, 128, 2999)
    assert kv.max_num_pages == 128
    pos = specs[cfg.LM_POS].config
    assert pos.interleave is True and pos.rope_theta == 10000.0 and pos.kv_cache == cfg.LM_KV
    assert specs[cfg.TEXT_SAMPLER].vocab_size == 32000 and specs[cfg.DEPTH_SAMPLER].vocab_size == 2048
    assert model.get_max_output_tokens() == 2999 and model.get_autocast_dtype() is torch.bfloat16


def test_client_output_cap_is_ignored_because_the_input_fixes_the_length():
    model = PersonaPlexModel(skip_weight_loading=True)
    assert model.get_max_output_tokens(max_output_tokens=1) == cfg.MAX_TEMPORAL_FORWARDS


@pytest.mark.parametrize("value", [0, -1, 2.0, True, "4"])
def test_max_batch_size_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match="max_batch_size"):
        PersonaPlexModel(skip_weight_loading=True, max_batch_size=value)


def test_max_batch_size_reaches_the_lm_submodule(monkeypatch):
    from mstar.model.personaplex import weights

    model = PersonaPlexModel(max_batch_size=3)
    monkeypatch.setattr(model, "_download", lambda name: name)
    monkeypatch.setattr(weights, "iter_lm_checkpoint", lambda path, device: iter(()))
    monkeypatch.setattr(weights, "build_lm", lambda config, tensors, device: torch.nn.Linear(1, 1))
    lm = model.get_submodule(cfg.LM_NODE)
    assert lm.max_batch_size(cfg.GENERATE_WALK) == 3
    assert PersonaPlexModel().max_batch_size == cfg.DEFAULT_MAX_BATCH_SIZE


def test_sampling_configs_default_to_reference_and_greedy_is_temperature_zero():
    model = _model()
    configs = model.get_request_resource_configs({}, {})
    text, depth = configs[cfg.TEXT_SAMPLER], configs[cfg.DEPTH_SAMPLER]
    assert (text.temperature, text.top_k, depth.temperature, depth.top_k) == (0.7, 25, 0.8, 250)
    greedy = model.get_request_resource_configs({}, {"greedy": True, "temp_text": 0.9})
    assert greedy[cfg.TEXT_SAMPLER].temperature == 0.0 and greedy[cfg.DEPTH_SAMPLER].temperature == 0.0


@pytest.mark.parametrize("name", ["temp_text", "temp_audio", "topk_text", "topk_audio", "greedy"])
def test_null_sampling_options_mean_unset(tmp_path, name):
    model = _model(tmp_path)
    _request(model, **{name: None})  # validation accepts null
    configs = model.get_request_resource_configs({}, {name: None})
    text, depth = configs[cfg.TEXT_SAMPLER], configs[cfg.DEPTH_SAMPLER]
    assert (text.temperature, text.top_k, depth.temperature, depth.top_k) == (0.7, 25, 0.8, 250)


def test_postprocess_text_pieces_and_pcm16():
    model = _model()
    assert model.postprocess(torch.tensor([42]), "text") == b" w42"
    assert [model.postprocess(torch.tensor([t]), "text") for t in range(4)] == [b""] * 4
    with pytest.raises(ValueError, match="one text token"):
        model.postprocess(torch.tensor([1, 2]), "text")
    pcm = model.postprocess(torch.tensor([[[0.0, 0.5, -1.5]]]), "audio")
    assert pcm == torch.tensor([0, 16383, -32767], dtype=torch.int16).numpy().tobytes()
    with pytest.raises(ValueError, match="modality"):
        model.postprocess(torch.zeros(1), "image")
    assert model.get_output_sample_rate() == 24000


def test_registry_hf_entry_and_yaml_agree():
    import yaml

    from mstar.model.registry import HF_MODELS, MODEL_REGISTRY, get_model_class

    assert get_model_class("personaplex") is PersonaPlexModel
    assert MODEL_REGISTRY["personaplex"] == ("mstar.model.personaplex.personaplex_model", "PersonaPlexModel")
    assert HF_MODELS["personaplex"] == {"model_path_hf": "nvidia/personaplex-7b-v1"}
    with open(CONFIG_PATH) as f:
        deployment = yaml.safe_load(f)
    assert deployment["model"] == "personaplex" and deployment["max_seq_len"] == cfg.MAX_TEMPORAL_FORWARDS
    assert set(deployment["model_kwargs"]) == {"voice_dir", "max_batch_size"}
    model = PersonaPlexModel(**deployment["model_kwargs"], skip_weight_loading=True)
    assert model.max_batch_size == deployment["model_kwargs"]["max_batch_size"] == cfg.DEFAULT_MAX_BATCH_SIZE
    assert {n for g in deployment["node_groups"] for n in g["node_names"]} == set(_model().nodes)


# --------------------------------------------------------------------------- voice presets from the repo


def _voices_archive(tmp_path: Path, members: dict[str, bytes]) -> Path:
    import io
    import tarfile

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir(exist_ok=True)
    archive = snapshot / "voices.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return archive


def test_voice_archive_extracts_once_beside_itself(tmp_path):
    from mstar.model.personaplex.voice import extract_voice_presets

    archive = _voices_archive(tmp_path, {"voices/NATF2.pt": b"a", "voices/NATM1.pt": b"b"})
    target = extract_voice_presets(archive)
    assert target == archive.parent / "voices" and sorted(p.name for p in target.iterdir()) == ["NATF2.pt", "NATM1.pt"]
    (target / "NATF2.pt").write_bytes(b"kept")  # a second call must not re-extract over it
    assert extract_voice_presets(archive) == target and (target / "NATF2.pt").read_bytes() == b"kept"
    assert [p.name for p in archive.parent.iterdir() if p.name.startswith(".voices-")] == []


def test_voice_archive_without_voices_dir_fails_and_leaves_nothing(tmp_path):
    from mstar.model.personaplex.voice import extract_voice_presets

    archive = _voices_archive(tmp_path, {"other/NATF2.pt": b"a"})
    with pytest.raises(RuntimeError, match="no voices/ directory"):
        extract_voice_presets(archive)
    assert sorted(p.name for p in archive.parent.iterdir()) == ["voices.tgz"]


def test_voice_archive_refuses_members_outside_the_tree(tmp_path):
    import tarfile

    from mstar.model.personaplex.voice import extract_voice_presets

    archive = _voices_archive(tmp_path, {"voices/NATF2.pt": b"a", "../escaped.pt": b"x"})
    with pytest.raises(tarfile.FilterError):
        extract_voice_presets(archive)
    assert not (tmp_path / "escaped.pt").exists() and not (archive.parent / "voices").exists()


def _served_model(monkeypatch, tmp_path, **kwargs):
    archive = _voices_archive(tmp_path, {"voices/NATF2.pt": b"a", "voices/NATM1.pt": b"b"})
    downloads = []

    def download(self, filename):
        downloads.append(filename)
        return str(archive)

    monkeypatch.setattr(PersonaPlexModel, "_download", download)
    return PersonaPlexModel(**kwargs), downloads, archive


def test_unset_voice_dir_fetches_the_repo_archive_once(monkeypatch, tmp_path):
    model, downloads, archive = _served_model(monkeypatch, tmp_path)
    assert model._voice_dir() == str(archive.parent / "voices")
    assert model._voice_dir() == str(archive.parent / "voices") and downloads == ["voices.tgz"]


def test_configured_or_dummy_voice_dirs_never_download(monkeypatch, tmp_path):
    model, downloads, _ = _served_model(monkeypatch, tmp_path, voice_dir="/srv/voices")
    assert model._voice_dir() == "/srv/voices"
    dummy, dummy_downloads, _ = _served_model(monkeypatch, tmp_path, skip_weight_loading=True)
    assert dummy._voice_dir() is None and dummy.get_voices() is None
    assert downloads == [] and dummy_downloads == []
    with pytest.raises(ValueError, match="no voice_dir"):
        _request(dummy)


def test_warmup_fetches_tokenizer_and_voices_except_in_dummy_mode(monkeypatch, tmp_path):
    model, downloads, _ = _served_model(monkeypatch, tmp_path)
    tokenizers = []
    monkeypatch.setattr(PersonaPlexModel, "_ensure_tokenizer", lambda self: tokenizers.append(1))
    model.warmup_preprocess()
    assert tokenizers == [1] and downloads == ["voices.tgz"]
    PersonaPlexModel(skip_weight_loading=True).warmup_preprocess()
    assert tokenizers == [1] and downloads == ["voices.tgz"]


def test_get_voices_lists_presets_without_downloading(monkeypatch, tmp_path):
    model, downloads, archive = _served_model(monkeypatch, tmp_path)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    assert model.get_voices() is None  # not cached yet: no fixed list, and no download
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: str(archive))
    model._voice_dir()
    assert model.get_voices() == ["NATF2", "NATM1"] and downloads == ["voices.tgz"]
    assert _model(tmp_path).get_voices() == ["NATF2"]


def test_cli_and_benchmark_entries_are_registered():
    repo_root = str(Path(__file__).resolve().parents[2])
    sys.path.insert(0, repo_root)
    try:
        from benchmark.base import ModelType, PersonaPlex, RequestType
        from mstar.cli.main import DEFAULT_CONFIGS, _next_steps
    finally:
        sys.path.remove(repo_root)

    assert DEFAULT_CONFIGS["personaplex"] == "personaplex.yaml"
    assert (Path(repo_root) / "configs" / DEFAULT_CONFIGS["personaplex"]).is_file()
    benchmark_model = ModelType.PERSONAPLEX.inst()
    assert isinstance(benchmark_model, PersonaPlex) and benchmark_model.get_hf_url() == "nvidia/personaplex-7b-v1"
    assert benchmark_model.get_supported_modalities() == {RequestType.A2S}
    assert 'voice="NATF2"' in _next_steps("personaplex", "0.0.0.0", 8000)


# --------------------------------------------------------------------------- KV pool sizing


def test_shipped_yaml_fits_its_request_cap_in_the_kv_pool(caplog):
    import yaml

    with open(CONFIG_PATH) as f:
        deployment = yaml.safe_load(f)
    model = _model()
    assert deployment["max_concurrent_requests"] == 5
    assert deployment["resources"]["lm_kv"] == {"page_size": 128, "max_num_pages": 128}
    model.validate_config_yaml(deployment, CONFIG_PATH)  # 1 sink + 5 x 24 = 121 <= 128
    with pytest.raises(ValueError, match=r"6 concurrent requests .* need 145 pages of resources\.lm_kv, which has 128"):
        model.validate_config_yaml({**deployment, "max_concurrent_requests": 6}, CONFIG_PATH)
    model.validate_config_yaml({k: v for k, v in deployment.items() if k != "max_concurrent_requests"}, CONFIG_PATH)
    assert "sets no max_concurrent_requests" in caplog.text


def test_kv_fit_counts_full_length_requests_and_the_sink_page():
    model = _model()
    assert model._request_kv_pages(128) == 24 and model._request_kv_pages(64) == 47
    assert model._request_kv_pages(2999) == 1 and model._request_kv_pages(1) == 2999
    fits = {"max_concurrent_requests": 2, "resources": {"lm_kv": {"page_size": 64, "max_num_pages": 95}}}
    model.validate_config_yaml(fits, "x.yaml")
    fits["resources"]["lm_kv"]["max_num_pages"] = 94
    with pytest.raises(ValueError, match="need 95 pages"):
        model.validate_config_yaml(fits, "x.yaml")
    # without a resources override the declared default pool is checked
    model.validate_config_yaml({"max_concurrent_requests": 5}, "x.yaml")
    with pytest.raises(ValueError, match="which has 128"):
        model.validate_config_yaml({"max_concurrent_requests": 6}, "x.yaml")


def test_yaml_kv_pool_reaches_the_resource_spec():
    import yaml

    from mstar.engine.resources.spec import apply_yaml_overrides

    with open(CONFIG_PATH) as f:
        deployment = yaml.safe_load(f)
    deployment["resources"]["lm_kv"]["max_num_pages"] = 64  # the YAML value, not the default, must land
    specs = _model().get_node_resources()
    apply_yaml_overrides(specs, deployment)
    kv = next(s for s in specs if s.resource_key == cfg.LM_KV).config
    assert (kv.page_size, kv.max_num_pages) == (128, 64)
