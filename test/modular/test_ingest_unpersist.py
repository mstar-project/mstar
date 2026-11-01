"""The prompt tensors a model declares released by its first walk are released
at ingest, the same as every later walk's declaration. The preprocess worker
persists every input signal until the conductor unpersists it, so without this
pixel values, audio features and prompt tokens stay held for the whole request.

Drives the real ``Conductor.__init__`` and ``Conductor.run`` with worker launch
and ZMQ stubbed: the stub communicator records every send and replays one
scripted batch of inbound messages per run-loop iteration.
"""
from collections import deque
from types import SimpleNamespace

import pytest
import yaml

torch = pytest.importorskip("torch")

import mstar.conductor.conductor as conductor_mod  # noqa: E402
from mstar.conductor.conductor import Conductor, _walk_io  # noqa: E402
from mstar.conductor.request_info import (  # noqa: E402
    FINAL_PREFILL,
    CurrentForwardConductorMetadata,
    PartitionDefinition,
    PartitionState,
    WalkStep,
)
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig  # noqa: E402
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential, TensorPointerInfo  # noqa: E402
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION  # noqa: E402
from mstar.model.base import ForwardPassArgs, KVLabel, LabelLayout, Model, Span  # noqa: E402
from mstar.model.submodule_base import ChunkingPolicy, chunk_tick_name  # noqa: E402
from mstar.streaming.chunk_policy import FixedChunkPolicy  # noqa: E402
from mstar.streaming.topology import Connection, PartitionTopology, StreamingGraphEdge  # noqa: E402
from mstar.utils.ipc_format import (  # noqa: E402
    ConductorMessage,
    ConductorMessageType,
    NewRequestConductor,
    SetupDone,
    WorkerGraphsDone,
)

PREPROCESS = "api_server_preprocess_worker"


class _StubComm:
    def __init__(self):
        self.sent = []
        self.batches = deque()

    def send(self, entity_id, msg):
        self.sent.append((entity_id, msg))

    def get_all_new_messages(self):
        return self.batches.popleft() if self.batches else []


class _ScriptDoneError(Exception):
    pass


class _ToyModel(Model):
    """Graph, partitions and walk transitions come from the test."""

    def __init__(self, walks, initial_args, next_args=None, partitions=None):
        self._walks = walks
        self._initial_args = initial_args
        self._next_args = next_args
        self._partitions = partitions

    def get_node_resources(self):
        return []

    def get_graph_walk_graphs(self):
        return self._walks

    def get_partitions(self):
        return self._partitions or super().get_partitions()

    def get_partition_topology(self):
        if not self._partitions:
            return super().get_partition_topology()
        return PartitionTopology(partitions=[p.name for p in self._partitions], connections=[])

    def get_initial_forward_pass_args(self, partition_name, input_modalities, output_modalities,
                                      input_signals, model_kwargs=None):
        return self._initial_args(partition_name, input_signals)

    def get_partition_forward_pass_args(self, partition_name, partition_metadata, persist_signals,
                                        incoming_connections=None):
        return self._next_args(partition_metadata, persist_signals)

    def process_prompt(self, *a, **k):
        return {}

    def postprocess(self, *a, **k):
        return b""

    def get_submodule(self, *a, **k):
        return None


def _conductor(model, node_groups, tmp_path, monkeypatch) -> Conductor:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"max_seq_len": 1024, "node_groups": node_groups}))
    comm = _StubComm()
    monkeypatch.setattr(Conductor, "_launch_workers", lambda self: None)
    monkeypatch.setattr(conductor_mod, "make_communicator", lambda **kw: comm)
    c = Conductor(model, str(path), socket_path_prefix=str(tmp_path / "s"))
    c.comm = comm
    return c


def _run(c: Conductor, batches):
    """Run the real ``Conductor.run`` over ``batches``, one per iteration."""
    setup = [_msg(ConductorMessageType.SETUP_DONE, SetupDone(w)) for w in c.worker_ids]
    c.comm.batches.extend([setup, *batches])

    def poll():
        if not c.comm.batches:
            raise _ScriptDoneError

    c._poll_worker_liveness = poll
    with pytest.raises(_ScriptDoneError):
        c.run()


def _msg(kind, body):
    return ConductorMessage(message_type=kind, body=body)


def _new_request(rid, signals):
    return _msg(ConductorMessageType.NEW_REQUEST, NewRequestConductor(
        request_id=rid, initial_signals=signals, initial_input_modalities=["text"],
        initial_output_modalities=["text"], input_metadata={}, model_kwargs={},
    ))


def _info(uuid, entity=PREPROCESS, node=None, walk=None):
    return TensorPointerInfo(
        dims=[4], dtype=torch.int64, nbytes=32, address=0, stride=[1], uuid=uuid,
        source_session_id="s", source_entity=entity, _source_node_name=node, _source_graph_walk=walk,
    )


def _edge(node, name, infos):
    edge = GraphEdge(next_node=node, name=name)
    edge.tensor_info = list(infos)
    return edge


def _fwd(walk, inputs=(), unpersist=(), done=False):
    return ForwardPassArgs(
        full_metadata=CurrentForwardConductorMetadata(graph_walk=walk, is_prefill=False),
        inputs=list(inputs), unpersist_tensors=list(unpersist), request_done=done,
    )


def _sends(c, kinds=("NEW_REQUEST", "UNPERSIST_TENSORS")):
    """(kind, entity, body) for each send of one of ``kinds``, in send order."""
    out = []
    for entity, msg in c.comm.sent:
        kind = getattr(msg.message_type, "name", msg.message_type)
        if kind in kinds:
            out.append((kind, entity, msg.body))
    return out


def _releases(c):
    return [(entity, dict(body.uuid_to_ref_count))
            for kind, entity, body in _sends(c) if kind == "UNPERSIST_TENSORS"]


PROMPT, TOKEN = 101, 202


def test_first_walk_release_reaches_the_preprocess_worker_after_dispatch(tmp_path, monkeypatch):
    walks = {
        "prefill": GraphNode(name="llm", input_names=["text_inputs"], outputs=[
            GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text", persist=True),
        ]),
        "decode": Loop(name="decode_loop", max_iters=8, outputs=[], section=GraphNode(
            name="llm", input_names=["text_inputs"], outputs=[
                GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text"),
                GraphEdge(next_node="llm", name="text_inputs"),
            ])),
    }

    def initial(partition, signals):
        edge = _edge("llm", "text_inputs", signals["text_inputs"])
        return _fwd("prefill", [edge], unpersist=edge.tensor_info)

    def next_args(metadata, persist_signals):
        if metadata.graph_walk == "prefill":
            edge = _edge("llm", "text_inputs", persist_signals.get("new_token", []))
            return _fwd("decode", [edge], unpersist=edge.tensor_info)
        return _fwd("decode", done=True)

    c = _conductor(_ToyModel(walks, initial, next_args),
                   [{"node_names": ["llm"], "ranks": [0]}], tmp_path, monkeypatch)
    prefill_wg = c.node_walk_to_wg[("llm", "prefill")].worker_graph_id
    token = _info(TOKEN, entity="worker_0", node="llm", walk="prefill")
    _run(c, [
        [_new_request("r", {"text_inputs": [_info(PROMPT)]})],
        [_msg(ConductorMessageType.WORKER_GRAPHS_DONE, WorkerGraphsDone(
            request_id="r", worker_graph_ids=[prefill_wg], is_first_tp_rank=True,
            partition_name="default", persist_signals={"new_token": [token]},
        ))],
    ])

    assert [(kind, entity) for kind, entity, _ in _sends(c)] == [
        ("NEW_REQUEST", "worker_0"),
        ("UNPERSIST_TENSORS", PREPROCESS),  # at ingest, after the NewRequest
        ("UNPERSIST_TENSORS", "worker_0"),  # the prefill -> decode transition
    ]
    assert _releases(c) == [(PREPROCESS, {PROMPT: 1}), ("worker_0", {TOKEN: 1})]
    request = c.requests["r"]
    assert request.persist_signal_ref_cnt == {}
    assert all(not infos for infos in request.persist_signals.values())


def test_ingest_releases_the_union_of_every_partitions_declaration(tmp_path, monkeypatch):
    shared, only_a, done_partition, never_sent = 1, 2, 3, 4
    walks = {
        "a": GraphNode(name="na", input_names=["x", "y", "z"], outputs=[]),
        "b": GraphNode(name="nb", input_names=["x"], outputs=[]),
        "c": GraphNode(name="nc", input_names=["z"], outputs=[]),
    }
    partitions = [PartitionDefinition(name=p.upper(), graph_walks={p}, initial_walk=p) for p in "abc"]

    def initial(partition, signals):
        if partition == "A":
            inputs = [_edge("na", name, signals[name]) for name in ("x", "y", "z")]
            return _fwd("a", inputs, unpersist=signals["x"] + signals["w"])
        if partition == "B":
            # reads x too, and declares y, which only A sent
            return _fwd("b", [_edge("nb", "x", signals["x"])], unpersist=signals["y"])
        # inactive for this request, but declares z, which A sent
        return _fwd("c", unpersist=signals["z"], done=True)

    c = _conductor(
        _ToyModel(walks, initial, partitions=partitions),
        [{"node_names": ["na", "nc"], "ranks": [0]}, {"node_names": ["nb"], "ranks": [1]}],
        tmp_path, monkeypatch,
    )
    signals = {
        "x": [_info(shared)], "y": [_info(only_a)], "z": [_info(done_partition)], "w": [_info(never_sent)],
    }
    _run(c, [[_new_request("r", signals)]])

    sends = _sends(c)
    assert sorted((entity, body.request_info.partition_name)
                  for kind, entity, body in sends if kind == "NEW_REQUEST") == [
        ("worker_0", "A"), ("worker_0", "C"), ("worker_1", "B"),
    ]
    # one release, after every NewRequest, with x counted once per worker that reads it
    assert _releases(c) == [(PREPROCESS, {shared: 2, only_a: 1, done_partition: 1})]
    assert [kind for kind, _, _ in sends] == ["NEW_REQUEST"] * 3 + ["UNPERSIST_TENSORS"]
    # Never sent to a worker, so no reader count: it stays persisted (and visible to
    # later walks) until the request's teardown, like an untracked uuid on any pass.
    request = c.requests["r"]
    assert [i.uuid for i in request.persist_signals["w"]] == [never_sent]
    assert not any(request.persist_signals[name] for name in ("x", "y", "z"))


# ---------------------------------------------------------------------------
# A model's walk plan: what the conductor sends and releases at each step
# ---------------------------------------------------------------------------


def plan_step(model, plan, k, signals) -> ForwardPassArgs:
    """What the conductor sends a request entering ``plan[k]``: its edges and
    step metadata, by the conductor's own `_plan_step`."""
    c = Conductor.__new__(Conductor)
    c._walk_inputs, _, _ = _walk_io(model)
    c._growth_walks = frozenset(
        walk for labels in model.kv_labels().values() for label in labels.values() for walk, _ in label.growth
    )
    pstate = PartitionState("p", CurrentForwardConductorMetadata(graph_walk="", is_prefill=True), plan=list(plan))
    return c._plan_step(SimpleNamespace(persist_signals=signals), pstate, k)


class _PlanModel(_ToyModel):
    """Walks, partitions and streams from the test; the plan is all it decides."""

    def __init__(self, walks, plans, groups, connections=(), growth=False, chunked=()):
        super().__init__(walks, None, partitions=[
            PartitionDefinition(name=name, graph_walks=set(walks_)) for name, walks_ in groups.items()
        ])
        self._plans, self._connections, self._growth, self._chunked = plans, list(connections), growth, chunked

    def walk_plan(self, partition_name, input_modalities, output_modalities, input_signals, model_kwargs):
        return list(self._plans[partition_name])

    def get_partition_topology(self):
        return PartitionTopology(partitions=[p.name for p in self._partitions], connections=self._connections)

    def get_node_resources(self):
        if not self._growth:
            return []
        return [KVSpec(resource_key="kv", nodes={"llm"}, config=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=1024, max_num_pages=64, page_size=16,
        ))]

    def kv_labels(self):
        return {"kv": {"main": KVLabel(growth=(("decode", "decode_loop"),))}} if self._growth else {}

    def get_chunking_policies(self):
        return {"llm": (ChunkingPolicy(walks=self._chunked, chunkable=True, max_batch_tokens=64),)} \
            if self._chunked else {}


def _decoder_walks(prefills=("prefill",)):
    """Prefill walks that persist the sampled token, and a decode loop."""
    walks = {walk: GraphNode(name="llm", input_names=["text_inputs", *extra], outputs=[
        GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text", persist=True),
    ]) for walk, extra in prefills}
    walks["decode"] = Loop(name="decode_loop", max_iters=8, outputs=[], section=GraphNode(
        name="llm", input_names=["text_inputs"], outputs=[GraphEdge(next_node="llm", name="text_inputs")],
    ))
    return walks


DECODE = WalkStep("decode", bind={"text_inputs": "new_token"})
PROMPT_KV = {"kv_layout": {"kv": {"main": LabelLayout(
    spans=(Span("ids", "prefill", 4, 4, "text_inputs"),), growth=(("decode_loop", 1),),
).row()}}}
REF, TAIL, LANG, EMB = 303, 404, 505, 606


def _p2():
    model = _PlanModel(_decoder_walks([("prefill", ())]), {"default": [WalkStep("prefill"), DECODE]},
                       {"default": ["prefill", "decode"]}, growth=True)
    return model, {"text_inputs": [PROMPT]}, PROMPT_KV, [
        ("default", "prefill", {"new_token": TOKEN}, False), ("default", "decode", {}, False),
    ], [
        ("NEW_REQUEST", "default", "prefill", {("llm", "text_inputs"): [PROMPT]}, True),
        ("UNPERSIST_TENSORS", PREPROCESS, {PROMPT: 1}),
        ("INPUT_SIGNALS", "default", "decode", {("llm", "text_inputs"): [TOKEN]}, False),
        ("UNPERSIST_TENSORS", "worker_0", {TOKEN: 1}),
    ]


def _chunked_prefill():
    model, signals, kwargs, script, sends = _p2()
    model._chunked = ("prefill",)
    tick = chunk_tick_name("llm", "prefill")
    sends[0] = ("NEW_REQUEST", "default", "prefill", {("llm", "text_inputs"): [PROMPT], ("llm", tick): []}, True)
    return model, signals, kwargs, script, sends


def _rebound_name():
    """Whisper's detect -> prompt -> decode: each prefill persists ``new_token``,
    and the prompt walk's own re-persist means it consumes the language token."""
    model = _PlanModel(
        _decoder_walks([("detect", ()), ("prompt", ("tail",))]),
        {"default": [WalkStep("detect"), WalkStep("prompt", bind={"text_inputs": "new_token"}), DECODE]},
        {"default": ["detect", "prompt", "decode"]}, growth=True,
    )
    return model, {"text_inputs": [PROMPT], "tail": [_info(TAIL)]}, PROMPT_KV, [
        ("default", "detect", {"new_token": LANG}, False), ("default", "prompt", {"new_token": TOKEN}, False),
        ("default", "decode", {}, False),
    ], [
        # the tail is read later; only the step before decode samples a real token
        ("NEW_REQUEST", "default", "detect", {("llm", "text_inputs"): [PROMPT]}, False),
        ("UNPERSIST_TENSORS", PREPROCESS, {PROMPT: 1}),
        ("INPUT_SIGNALS", "default", "prompt", {("llm", "text_inputs"): [LANG], ("llm", "tail"): [TAIL]}, True),
        ("UNPERSIST_TENSORS", PREPROCESS, {TAIL: 1}),
        ("UNPERSIST_TENSORS", "worker_0", {LANG: 1}),
        ("INPUT_SIGNALS", "default", "decode", {("llm", "text_inputs"): [TOKEN]}, False),
        ("UNPERSIST_TENSORS", "worker_0", {TOKEN: 1}),
    ]


def _diffusion(refine=False):
    """An encode walk, then a denoise loop whose loop-back seed arrives empty.
    With LTX-2.5's refine, a second denoise walk reads the conditioning again,
    so the first one's send keeps it."""
    def denoise():
        return Sequential([
            Loop(name="denoise", max_iters=4, outputs=[GraphEdge(next_node="vae", name="latents")],
                 section=GraphNode(name="dit", input_names=["emb", "latents"],
                                   outputs=[GraphEdge(next_node="dit", name="latents")])),
            GraphNode(name="vae", input_names=["latents"], outputs=[
                GraphEdge(next_node=EMIT_TO_CLIENT, name="image", output_modality="image"),
            ]),
        ])

    walks = {
        "encode": GraphNode(name="enc", input_names=["text_inputs"], outputs=[
            GraphEdge(next_node=EMPTY_DESTINATION, name="emb", persist=True),
        ]),
        "generate": denoise(),
        "refine": denoise(),
    }
    gens = ["generate", "refine"] if refine else ["generate"]
    md = {"steps": 4}
    model = _PlanModel(walks, {"default": [WalkStep(w, step_metadata=md) for w in ["encode", *gens]]},
                       {"default": list(walks)})
    seeds = {("dit", "emb"): [EMB], ("dit", "latents"): []}
    return model, {"text_inputs": [PROMPT]}, {}, [
        ("default", "encode", {"emb": EMB}, False), *(("default", w, {}, False) for w in gens),
    ], [
        ("NEW_REQUEST", "default", "encode", {("enc", "text_inputs"): [PROMPT]}, False),
        ("UNPERSIST_TENSORS", PREPROCESS, {PROMPT: 1}),
        *(("INPUT_SIGNALS", "default", w, seeds, False) for w in gens),
        ("UNPERSIST_TENSORS", "worker_0", {EMB: len(gens)}),
    ]


def _stream_consumer(consumer_active=True):
    """Chatterbox's S3Gen: a stream consumer that also reads the reference clip
    its producer reads. The clip goes to both on entry and is released once
    both have read it; each later chunk's pass carries a signal-only edge."""
    walks = {
        "a": GraphNode(name="na", input_names=["text_inputs", "ref"], outputs=[
            StreamingGraphEdge(next_node="nb", name="tok", target_partition="B"),
        ]),
        "b": GraphNode(name="nb", input_names=["tok", "ref"], outputs=[
            GraphEdge(next_node=EMIT_TO_CLIENT, name="audio", output_modality="audio"),
        ]),
    }
    model = _PlanModel(walks, {"A": [WalkStep("a")], "B": [WalkStep("b")] if consumer_active else []},
                       {"A": ["a"], "B": ["b"]},
                       connections=[Connection("A", "B", "tok", lambda: FixedChunkPolicy(chunk_size=1))])
    if not consumer_active:
        return model, {"text_inputs": [PROMPT], "ref": [_info(REF)]}, {}, [("A", "a", {}, True)], [
            ("NEW_REQUEST", "A", "a", {("na", "ref"): [REF], ("na", "text_inputs"): [PROMPT]}, False),
            ("NEW_REQUEST", "B", "", {}, False),
            ("UNPERSIST_TENSORS", PREPROCESS, {PROMPT: 1, REF: 1}),
        ]
    return model, {"text_inputs": [PROMPT], "ref": [_info(REF)]}, {}, [
        ("B", "b", {}, False), ("A", "a", {}, True), ("B", "b", {}, True),
    ], [
        ("NEW_REQUEST", "A", "a", {("na", "ref"): [REF], ("na", "text_inputs"): [PROMPT]}, False),
        ("NEW_REQUEST", "B", "b", {("nb", "ref"): [REF]}, False),
        ("UNPERSIST_TENSORS", PREPROCESS, {PROMPT: 1, REF: 2}),
        ("INPUT_SIGNALS", "B", "b", {("nb", "ref"): []}, False),
        ("INPUT_SIGNALS", "B", "b", {}, False),  # producer_done
    ]


def _stream_moved_walks():
    """Qwen3-Omni's Talker: the stream moves it through its prefill walks; the
    conductor sends only the decode seed, after the last prefill reports."""
    walks = {
        "think": GraphNode(name="na", input_names=["text_inputs"], outputs=[
            StreamingGraphEdge(next_node="nb", name="states", target_partition="T"),
        ]),
        "t_prefill": GraphNode(name="nb", input_names=["states"], outputs=[]),
        "t_last": GraphNode(name="nb", input_names=["states"], outputs=[
            GraphEdge(next_node=EMPTY_DESTINATION, name="embeds", persist=True),
        ]),
        "t_decode": Loop(name="talk_loop", max_iters=8, outputs=[], section=GraphNode(
            name="nb", input_names=["embeds"], outputs=[GraphEdge(next_node="nb", name="embeds")],
        )),
    }
    model = _PlanModel(
        walks, {"A": [WalkStep("think")], "T": [WalkStep(w) for w in ("t_prefill", "t_last", "t_decode")]},
        {"A": ["think"], "T": ["t_prefill", "t_last", "t_decode"]},
        connections=[Connection("A", "T", "states", lambda: FixedChunkPolicy(chunk_size=1),
                                consumer_walk=lambda ctx: "t_prefill")],
    )
    return model, {"text_inputs": [PROMPT]}, {}, [
        ("T", "t_prefill", {}, False), ("T", "t_prefill", {}, False), ("A", "think", {}, True),
        ("T", "t_last", {"embeds": EMB}, False), ("T", "t_decode", {}, False),
    ], [
        ("NEW_REQUEST", "A", "think", {("na", "text_inputs"): [PROMPT]}, False),
        ("NEW_REQUEST", "T", "t_prefill", {}, False),
        ("UNPERSIST_TENSORS", PREPROCESS, {PROMPT: 1}),
        ("INPUT_SIGNALS", "T", "t_prefill", {}, False),  # producer_done, the walk still the stream's
        ("INPUT_SIGNALS", "T", "t_decode", {("nb", "embeds"): [EMB]}, False),
        ("UNPERSIST_TENSORS", "worker_1", {EMB: 1}),
    ]


_PLANS = {
    "prefill then decode": _p2,
    "chunked prefill seeds its tick": _chunked_prefill,
    "a re-persisted name": _rebound_name,
    "diffusion's empty seed": _diffusion,
    "a later reader keeps it": lambda: _diffusion(refine=True),
    "stream consumer re-sends": _stream_consumer,
    "inactive consumer": lambda: _stream_consumer(consumer_active=False),
    "walks the stream moves": _stream_moved_walks,
}


@pytest.mark.parametrize("case", list(_PLANS))
def test_a_walk_plan_sends_each_steps_inputs_and_releases_their_last_read(tmp_path, monkeypatch, case):
    model, signals, kwargs, script, expected = _PLANS[case]()
    walks = model.get_graph_walk_graphs()
    # each partition on a rank of its own
    groups = [sorted({n for w in p.graph_walks for n in walks[w].get_nodes()}) for p in model.get_partitions()]
    c = _conductor(model, [{"node_names": g, "ranks": [i]} for i, g in enumerate(groups)], tmp_path, monkeypatch)
    worker = {n: f"worker_{i}" for i, g in enumerate(groups) for n in g}
    signals = {name: [i if isinstance(i, TensorPointerInfo) else _info(i) for i in infos]
               for name, infos in signals.items()}
    reports = []
    for partition, walk, persist, done in script:
        node = next(iter(walks[walk].get_nodes()))
        wg = c.node_walk_to_wg[(node, walk)].worker_graph_id
        reports.append([_msg(ConductorMessageType.WORKER_GRAPHS_DONE, WorkerGraphsDone(
            request_id="r", worker_graph_ids=[wg], is_first_tp_rank=True, partition_name=partition,
            partition_done=done, graph_walk=walk,
            persist_signals={name: [_info(u, entity=worker[node], node=node, walk=walk)]
                             for name, u in persist.items()},
        ))])
    _run(c, [[_msg(ConductorMessageType.NEW_REQUEST, NewRequestConductor(
        request_id="r", initial_signals=signals, initial_input_modalities=["text"],
        initial_output_modalities=["text"], input_metadata={}, model_kwargs=kwargs,
    ))], *reports])

    got = []
    for kind, entity, body in _sends(c, ("NEW_REQUEST", "INPUT_SIGNALS", "UNPERSIST_TENSORS")):
        if kind == "UNPERSIST_TENSORS":
            got.append((kind, entity, dict(body.uuid_to_ref_count)))
            continue
        info = body.request_info
        edges = body.initial_inputs if kind == "NEW_REQUEST" else body.inputs
        got.append((kind, info.partition_name, info.graph_walk,
                    {(e.next_node, e.name): [i.uuid for i in e.tensor_info] for e in edges},
                    bool(info.step_metadata.get(FINAL_PREFILL))))
    assert got == expected
    assert any(getattr(msg, "message_type", None) == "request_complete" for _, msg in c.comm.sent), \
        "the plan's last step did not end the request"
