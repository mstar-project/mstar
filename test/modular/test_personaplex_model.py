"""Dummy-mode tests for the PersonaPlex model contract (T8-T10, T14).

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
from mstar.communication.tensors import TensorCommunicationManager, TensorStore
from mstar.graph.base import GraphEdge, Loop, Sequential, TensorPointerInfo
from mstar.graph.graph_io import WorkerGraphIO
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.personaplex import config as cfg
from mstar.model.personaplex.personaplex_model import GENERATE_AUDIO_WALK, PersonaPlexModel
from mstar.model.personaplex.voice import load_voice_preset, resolve_voice_path

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


# --------------------------------------------------------------------------- T9 walks and schedule


def test_t9_walks_instantiate_with_one_lm_per_walk_and_async_off():
    walks = _model().get_graph_walk_graphs()
    assert set(walks) == {cfg.ENCODE_USER_AUDIO_WALK, *LM_WALKS}
    assert set(walks[cfg.ENCODE_USER_AUDIO_WALK].get_nodes()) == {"MimiEncoder"}
    assert set(walks[cfg.CONDITIONING_WALK].get_nodes()) == {"LM"}
    for walk in (cfg.GENERATE_WALK, GENERATE_AUDIO_WALK):
        assert set(walks[walk].get_nodes()) == {"LM", "MimiDecoder"}
    for walk in LM_WALKS:
        assert walks[walk].get_nodes()["LM"].enable_async_scheduling is False
    assert _model().nodes == ["LM", "MimiDecoder", "MimiEncoder"]


def test_t9_lm_loop_backs_are_consumed_and_outputs_routed():
    walks = _model().get_graph_walk_graphs()
    for walk, loop_backs in ((cfg.CONDITIONING_WALK, {"lm_state"}),
                             (cfg.GENERATE_WALK, {"lm_state", "agent_frame", "text_frame"})):
        lm = walks[walk].get_nodes()["LM"]
        assert {e.name for e in lm.outputs if e.next_node == "LM"} == loop_backs
        assert loop_backs <= set(lm.input_names)  # consumed, never rejected (§3 M3)
    cond = walks[cfg.CONDITIONING_WALK]
    assert isinstance(cond, Loop) and cond.max_iters == cfg.MAX_TEMPORAL_FORWARDS
    assert [(e.name, e.next_node, e.persist) for e in cond.outputs] == [("lm_state", EMPTY_DESTINATION, True)]
    encode = walks[cfg.ENCODE_USER_AUDIO_WALK]
    assert [(e.name, e.persist) for e in encode.outputs] == [("user_audio_codes", True)]
    for walk, accumulated in ((cfg.GENERATE_WALK, [("agent_frame", "MimiDecoder"), ("text_frame", EMIT_TO_CLIENT)]),
                              (GENERATE_AUDIO_WALK, [("agent_frame", "MimiDecoder")])):
        seq = walks[walk]
        assert isinstance(seq, Sequential)
        loop, decoder = seq.sections
        assert loop.outputs == [] and [(e.name, e.next_node) for e in loop.accumulated_outputs] == accumulated
        assert [(e.name, e.next_node, e.output_modality) for e in decoder.outputs] == [
            ("audio_output", EMIT_TO_CLIENT, "audio")
        ]


def _step(model, metadata, persist):
    return model.get_partition_forward_pass_args("default", metadata, persist)


def test_t9_schedule_encode_condition_generate_done_with_persisted_state():
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


def test_t9_audio_only_requests_use_the_walk_without_text_emission():
    args = _model().get_initial_forward_pass_args("default", ["audio"], ["audio"], _signals())
    assert args.full_metadata.kwargs["walk_schedule"][-1] == GENERATE_AUDIO_WALK


@pytest.mark.parametrize("with_text", [False, True], ids=["audio-only", "audio-plus-text"])
def test_t9_data_worker_emits_requested_modalities(with_text):
    """Exercise transfer setup and the real edge-name to ResultChunk boundary.

    The worker sends ``ResultTensors.modality = edge.output_modality``
    (worker.py:1055-1059) and ``_read_result_tensor`` renames the edge to
    ``f"{modality}_output"`` (data_worker.py:419), so the producer name
    ``text_frame`` never reaches postprocessing.
    """
    model = _model()
    walk = model.get_graph_walk_graphs()[cfg.GENERATE_WALK if with_text else GENERATE_AUDIO_WALK]
    edges = ([walk.sections[0].accumulated_outputs[1]] if with_text else []) + [walk.sections[1].outputs[0]]
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
            return {"request": self.ready}

        def get_tensor(self, request_id, uuid):
            assert request_id == "request"
            return tensors[uuid]

        def dereference(self, request_id, uuid):
            self.released.append((request_id, uuid))

    worker = types.SimpleNamespace(
        tensor_manager=ReadyTensors(), model=model, out_queue=SimpleQueue(),
        request_model_kwargs={},
        tensor_uuid_to_metadata_per_request={},
    )
    worker._fail_request = lambda _rid, exc, _stage: pytest.fail(f"data worker failed: {exc}")
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
    assert set(worker.tensor_manager.released) == {("request", uuid) for uuid in tensors}


def test_t9_conductor_backstops_budget_and_missing_inputs():
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


def test_t9_worker_graphs_from_yaml_keep_accumulated_edges():
    model = _model()
    graphs = model.get_worker_graphs(CONFIG_PATH)
    assert {walk for wg in graphs for walk in wg.graph_walks} == {cfg.ENCODE_USER_AUDIO_WALK, *LM_WALKS}
    loops = [loop for wg in graphs if cfg.GENERATE_WALK in wg.graph_walks
             for loop in wg.section.get_loops().values()]
    assert [[e.name for e in loop.accumulated_outputs] for loop in loops] == [["agent_frame", "text_frame"]]
    assert all(wg.ranks == [0] for wg in graphs)


# --------------------------------------------------------------------------- T8 accumulated outputs


class _Comm:
    """Refcount stand-in exposing what the real Loop/NodeInputs/fanout code touches."""

    def __init__(self):
        self.tensor_store = TensorStore()

    def increment_ref(self, rid, uuid, n=1):
        self.tensor_store.increment_ref(rid, uuid, n=n)

    def dereference(self, rid, uuid, n=1):
        self.tensor_store.dereference(rid, uuid, n=n)

    set_output_ref_counts = TensorCommunicationManager.set_output_ref_counts


def _drive(walk, initial, *, finish_after=None, rid="r0"):
    """Run a walk on one worker the way worker.py's post-forward path does.

    Real: WorkerGraphIO, Loop caching, NodeInputs.clear, TensorStore,
    set_output_ref_counts. Transcribed (single worker): safety hold of 1 per
    produced tensor (worker.py:2213-2230), input cleanup before
    mark_node_complete (2045-2057), local ingest or client emission
    (node_manager_utils.py:392-531), fanout adjustment (2252-2270), and the
    API server's receipt ack for emitted tensors (worker.py:513-518).
    """
    comm = _Comm()
    io = WorkerGraphIO(walk)
    io.register_communication_info(comm, rid)
    uid, produced, runs = count(), {}, {}
    out = {"client": [], "decoder_frames": None, "rejected": [], "persisted": []}

    def tensor(name, it):
        uuid = f"{name}@{it}#{next(uid)}"
        comm.tensor_store.put_tensor(rid, uuid, torch.tensor([it]))
        produced[uuid] = (name, it)
        return TensorPointerInfo(dims=[1], dtype="int64", nbytes=8, address=0, stride=[1], uuid=uuid,
                                 source_session_id="s", source_entity="w0")

    pending = []
    for edge in initial:
        if edge.tensor_info == ["external"]:
            info = tensor(edge.name, -1)
            comm.increment_ref(rid, info.uuid)  # held by the conductor's persisted forward-pass arg
            edge.tensor_info = [info]
        pending.append(edge)
    while not io.wg_state_registry.is_done:
        pending = [e for e in pending if not io.ingest_input(e)]
        ready = sorted(io.ready_node_names)
        assert ready, "walk stalled"
        name = ready[0]
        io.ready_node_names.discard(name)
        node = io.nodes[name]
        if name == "MimiDecoder":
            frames = node.ready_signals.ready_inputs["agent_frame"].tensor_info
            out["decoder_frames"] = [produced[i.uuid] for i in frames]
        it = runs.get(name, 0)
        runs[name] = it + 1
        node.reset_outputs()
        hold = set()
        for out_name in dict.fromkeys(e.name for e in node.outputs):
            info = tensor(out_name, it)
            comm.increment_ref(rid, info.uuid)
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
            elif e.persist:
                out["persisted"].append((e.name, [produced[i.uuid] for i in e.tensor_info]))
        comm.set_output_ref_counts(rid, hold, routed)
        for e in outs:
            if e.next_node == EMIT_TO_CLIENT:
                for i in e.tensor_info:
                    comm.dereference(rid, i.uuid)
    refs = {produced[u]: comm.tensor_store.per_req_tensors[rid][u].ref_cnt for u in produced if produced[u][1] >= 0}
    return out, runs, refs


def _generate_inputs():
    return [GraphEdge(next_node="LM", name="user_audio_codes", tensor_info=["external"]),
            GraphEdge(next_node="LM", name="lm_state", tensor_info=["external"]),
            GraphEdge(next_node="LM", name="agent_frame"), GraphEdge(next_node="LM", name="text_frame")]


@pytest.mark.parametrize("finish_after", [1, 7, 25])
def test_t8_generate_accumulates_frames_in_order_and_completes(finish_after):
    walk = _model().get_graph_walk_graphs()[cfg.GENERATE_WALK]
    out, runs, refs = _drive(walk, _generate_inputs(), finish_after=finish_after)
    assert runs == {"LM": finish_after, "MimiDecoder": 1}
    assert out["decoder_frames"] == [("agent_frame", i) for i in range(finish_after)]
    assert out["client"] == [("text_frame", [("text_frame", i) for i in range(finish_after)]),
                             ("audio_output", [("audio_output", 0)])]
    assert out["rejected"] == []
    assert all(r >= 0 for r in refs.values())
    # Every tensor of every non-final iteration, and the decoder output, is released.
    assert {k: r for k, r in refs.items() if r and k[1] != finish_after - 1} == {}
    assert refs[("lm_state", finish_after - 1)] == 0 and refs[("audio_output", 0)] == 0


def test_t8_generate_runs_to_max_iters_without_a_finish_signal():
    model = _model()
    walk = model._generate_walk(with_text=False)
    walk.sections[0].max_iters = 5
    out, runs, _ = _drive(walk, _generate_inputs())
    assert runs == {"LM": 5, "MimiDecoder": 1}
    assert out["decoder_frames"] == [("agent_frame", i) for i in range(5)]
    assert [name for name, _ in out["client"]] == ["audio_output"]


@pytest.mark.xfail(strict=True, reason=(
    "M* loop-output accounting: the final iteration's loop-output tensor keeps one reference "
    "(worklog §11 3.1-c; same for ordinary Loop.outputs). Flips when graph/worker accounting changes."
))
def test_t8_final_iteration_accumulated_tensors_are_released():
    walk = _model().get_graph_walk_graphs()[cfg.GENERATE_WALK]
    _, _, refs = _drive(walk, _generate_inputs(), finish_after=7)
    assert refs[("agent_frame", 6)] == 0 and refs[("text_frame", 6)] == 0


def test_t8_conditioning_persists_only_the_final_lm_state():
    walk = _model().get_graph_walk_graphs()[cfg.CONDITIONING_WALK]
    initial = [GraphEdge(next_node="LM", name=n, tensor_info=["external"])
               for n in ("voice_embeddings", "voice_cache", "persona_tokens")]
    initial.append(GraphEdge(next_node="LM", name="lm_state"))
    out, runs, refs = _drive(walk, initial, finish_after=25)
    assert runs == {"LM": 25}
    assert out["persisted"] == [("lm_state", [("lm_state", 24)])]
    assert out["rejected"] == [] and out["client"] == []
    assert {k: r for k, r in refs.items() if r and k != ("lm_state", 24)} == {}


# --------------------------------------------------------------------------- T10 request validation


def _fake_torchcodec(monkeypatch, samples: torch.Tensor):
    seen = {}

    class AudioDecoder:
        def __init__(self, path, **kwargs):
            seen.update(path=path, **kwargs)

        def get_all_samples(self):
            return types.SimpleNamespace(data=samples)

    decoders = types.ModuleType("torchcodec.decoders")
    decoders.AudioDecoder = AudioDecoder
    monkeypatch.setitem(sys.modules, "torchcodec", types.ModuleType("torchcodec"))
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", decoders)
    return seen


def test_t10_load_audio_keeps_channel_zero_at_24khz(monkeypatch):
    stereo = torch.stack((torch.full((5000,), 0.25), torch.full((5000,), -0.75)))
    seen = _fake_torchcodec(monkeypatch, stereo)
    loaded = _model().load_audio("in.wav", "cpu")
    assert seen == {"path": "in.wav", "sample_rate": 24000}  # native channels requested
    assert torch.equal(loaded.data, stereo[0]) and loaded.data.dtype == torch.float32
    assert loaded.metadata == {"sample_rate": 24000, "num_channels": 1, "source_channels": 2, "num_samples": 5000}


def _request(model, *, samples=1920 * 3, prompt=None, outputs=("audio", "text"), **kwargs):
    kwargs.setdefault("voice", "NATF2")
    return model.process_prompt(prompt, ["audio"], list(outputs), {"audio_inputs": [torch.zeros(samples)]},
                                **kwargs)


def test_t10_process_prompt_builds_conditioning_tensors(tmp_path):
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
    (dict(topk_audio=0), "topk_audio"),
    (dict(greedy="yes"), "greedy"),
    (dict(samples=1920 * 2966), "context budget"),
])
def test_t10_process_prompt_rejects_bad_requests(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _request(_model(tmp_path), **kwargs)


def test_t10_process_prompt_accepts_the_budget_boundary(tmp_path):
    model = _model(tmp_path, voice_frames=5)
    text = len(StubTokenizer().encode(f"<system> {cfg.DEFAULT_TEXT_PROMPT} <system>"))
    frames = cfg.MAX_TEMPORAL_FORWARDS - 4 - 12 - text
    _request(model, samples=1920 * frames)
    with pytest.raises(ValueError, match="context budget"):
        _request(model, samples=1920 * frames + 1)


def test_t10_process_prompt_rejects_missing_or_multiple_audio(tmp_path):
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
def test_t10_voice_preset_validation(tmp_path, overrides, match):
    path = _write_voice(tmp_path / "bad.pt", **overrides)
    with pytest.raises(ValueError, match=match):
        load_voice_preset(path)


def test_t10_voice_preset_accepts_sentinels_and_initial_tokens(tmp_path):
    cache = torch.full((1, 17, 4), 2048, dtype=torch.long)
    cache[0, 0] = 32000
    cache[0, :, 3] = -2
    preset = load_voice_preset(_write_voice(tmp_path / "ok.pt", cache=cache))
    assert preset.num_embeddings == 4
    assert resolve_voice_path(tmp_path, "ok") == (tmp_path / "ok.pt").resolve()
    assert resolve_voice_path(tmp_path, "ok.pt") == (tmp_path / "ok.pt").resolve()


# --------------------------------------------------------------------------- T14 dummy mode and I/O


def test_t14_dummy_mode_builds_without_submodules_or_network():
    model = PersonaPlexModel(skip_weight_loading=True)
    assert model.tokenizer is None and model.model_path_hf == "nvidia/personaplex-7b-v1"
    assert all(model.get_submodule(node) is None for node in model.nodes)
    specs = {spec.resource_key: spec for spec in model.get_node_resources()}
    assert set(specs) == {cfg.LM_KV, cfg.LM_ATTN, cfg.LM_POS, cfg.TEXT_SAMPLER, cfg.DEPTH_SAMPLER}
    assert all(spec.nodes == {"LM"} for spec in specs.values())
    kv = specs[cfg.LM_KV].config
    assert (kv.num_layers, kv.num_kv_heads, kv.num_qo_heads, kv.head_dim, kv.max_seq_len) == (32, 32, 32, 128, 2999)
    pos = specs[cfg.LM_POS].config
    assert pos.interleave is True and pos.rope_theta == 10000.0 and pos.kv_cache == cfg.LM_KV
    assert specs[cfg.TEXT_SAMPLER].vocab_size == 32000 and specs[cfg.DEPTH_SAMPLER].vocab_size == 2048
    assert model.get_max_output_tokens() == 2999 and model.get_autocast_dtype() is None


def test_t14_sampling_configs_default_to_reference_and_greedy_is_temperature_zero():
    model = _model()
    configs = model.get_request_resource_configs({}, {})
    text, depth = configs[cfg.TEXT_SAMPLER], configs[cfg.DEPTH_SAMPLER]
    assert (text.temperature, text.top_k, depth.temperature, depth.top_k) == (0.7, 25, 0.8, 250)
    greedy = model.get_request_resource_configs({}, {"greedy": True, "temp_text": 0.9})
    assert greedy[cfg.TEXT_SAMPLER].temperature == 0.0 and greedy[cfg.DEPTH_SAMPLER].temperature == 0.0


def test_t14_postprocess_text_pieces_and_pcm16():
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
    assert set(deployment["model_kwargs"]) == {"voice_dir"}
    assert {n for g in deployment["node_groups"] for n in g["node_names"]} == set(_model().nodes)
