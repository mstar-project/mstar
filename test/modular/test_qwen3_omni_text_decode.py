"""A Qwen3-Omni request with no audio output decodes on its own walk: its token loops back
on the device, reaches the client inline and sends nothing to the Talker. A request with
audio output keeps ``thinker_decode``. The Thinker declares the loop-back and the stop; the
hooks behind them are ``ARNodeSubmodule``'s. A captured step of either walk embeds its ids
and builds its rotary tables in the graph."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from rust_runtime import build_runtime
from test_graph_runtime import _sharding_config
from torch.overrides import TorchFunctionMode

pytest.importorskip("flashinfer")

from mstar.conductor.request_info import CurrentForwardConductorMetadata  # noqa: E402
from mstar.engine.resources.sampler.resource import SamplerResource  # noqa: E402
from mstar.graph.base import Loop  # noqa: E402
from mstar.graph.special_destinations import EMIT_TO_CLIENT  # noqa: E402
from mstar.model.base import WorkerGraph  # noqa: E402
from mstar.model.components.mrope import mrope_position_ids  # noqa: E402
from mstar.model.qwen3_omni.components.rope import compute_3d_cos_sin, compute_rope_freqs  # noqa: E402
from mstar.model.qwen3_omni.config import (  # noqa: E402
    THINKER_ATTN,
    THINKER_MIXED,
    THINKER_POS,
    THINKER_SAMPLER,
    THINKER_TEXT_DECODE,
)
from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel  # noqa: E402
from mstar.model.qwen3_omni.submodules import ThinkerSubmodule  # noqa: E402
from mstar.model.submodule_base import (  # noqa: E402
    ARNodeInputs,
    BatchedModelOutput,
    DeviceLoopback,
    HostRows,
    ModelInputsFromEngine,
)
from mstar.utils.containers import ParallelList  # noqa: E402
from mstar.utils.h2d import PinnedStager  # noqa: E402

EOS = 7
HIDDEN = 4
DECODE_WALKS = ("thinker_decode", THINKER_TEXT_DECODE)
PREFILL_WALKS = ("prefill_text", "prefill_audio", "prefill_vision")
_T = torch.Tensor
_HOST_READS = {_T.item, _T.tolist, _T.cpu, _T.numpy, _T.__int__, _T.__float__, _T.__bool__, _T.__index__}


class _HostReads(TorchFunctionMode):
    """Records every host read of a tensor's values made inside the block."""

    def __init__(self):
        super().__init__()
        self.reads: list[str] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func in _HOST_READS:
            self.reads.append(func.__name__)
        return func(*args, **(kwargs or {}))


class _Body:
    """The Thinker body: hands its input back, and its layer-0 and layer-N rows unless told not to keep them."""

    mask_padding = False

    def __init__(self):
        self.model = SimpleNamespace(embed_tokens=torch.nn.Embedding.from_pretrained(
            torch.arange(32.0).unsqueeze(-1).expand(-1, HIDDEN).clone(),
        ))
        self.calls = []

    def __call__(self, input_embeds, keep_states=True, **kwargs):
        self.calls.append({"keep_states": keep_states, **kwargs})
        return input_embeds, input_embeds if keep_states else None, input_embeds if keep_states else None

    def lm_head(self, hidden):
        return hidden

    def maybe_log_expert_load(self, synthetic=False):
        pass


@pytest.fixture
def thinker():
    sub = ThinkerSubmodule(_Body(), SimpleNamespace(
        im_end_token_id=EOS, thinker_head_dim=128, thinker_hidden_size=HIDDEN,
        thinker_text=SimpleNamespace(rope_theta=1e6),
        vision=SimpleNamespace(deepstack_visual_indexes=[]),
    ))
    sub.bind_node_resources({THINKER_SAMPLER: SimpleNamespace(has_slot_masters=True)})
    sub.get_device = lambda: torch.device("cpu")
    return sub


def _info(walk, ignore_eos=False):
    return SimpleNamespace(
        graph_walk=walk, max_tokens=100, step_metadata={"audio_output": False},
        resource_configs={THINKER_SAMPLER: SimpleNamespace(ignore_eos=ignore_eos)},
    )


def _model():
    model = object.__new__(Qwen3OmniModel)
    model.config = SimpleNamespace(thinker_text=SimpleNamespace(max_position_embeddings=64))
    return model


def _metadata(audio_output: bool, walk="prefill_text") -> CurrentForwardConductorMetadata:
    return CurrentForwardConductorMetadata(
        input_modalities=["text"], output_modalities=["text"], graph_walk=walk, is_prefill=True,
        kwargs={"prefill_schedule": [("prefill_text", {})], "prefill_step": 0, "audio_output": audio_output},
    )


def _engine_inputs(request_ids, tokens, captured=False):
    sampler = SimpleNamespace(
        loopback_tokens=lambda rids: torch.tensor(tokens),
        overwrite_rows=lambda rows, *updates: SamplerResource.overwrite_rows(sampler, rows, *updates),
        sample=lambda request_ids, logits: logits[:, 0].long(),
        _row_stager=PinnedStager(torch.long, numel=64),
    )
    return ModelInputsFromEngine(
        request_ids=request_ids, per_request_info={}, captured=captured,
        resources={THINKER_SAMPLER: sampler, THINKER_ATTN: SimpleNamespace(select_last_hidden=lambda h: h[-2:])},
    )


# ---- what the walk declares ----

def test_the_walk_loops_its_token_and_emits_it_with_no_edge_to_the_talker():
    model = _model()
    walk = model.get_graph_walk_graphs()[THINKER_TEXT_DECODE]
    assert walk.name == "thinker_decode_loop", "the loop's cap and stop are the decode walk's"
    assert [(edge.next_node, edge.name) for edge in walk.section.outputs] == [
        ("Thinker", "text_inputs"), (EMIT_TO_CLIENT, "text_inputs"),
    ]
    assert THINKER_TEXT_DECODE in model.get_combined_graph_walks()["Thinker"][THINKER_MIXED]


def test_the_thinker_declares_its_loop_back_and_its_stop(thinker):
    assert thinker.device_loopback == DeviceLoopback(
        walk=THINKER_TEXT_DECODE, signal="text_inputs", sampler=THINKER_SAMPLER, buffer="new_token",
    )
    stop = thinker.token_stop
    assert (stop.loop, stop.buffer, stop.sampler) == ("thinker_decode_loop", "new_token", THINKER_SAMPLER)
    assert stop.stop_ids == frozenset({EOS})
    assert stop.walk == frozenset(DECODE_WALKS), "both decode walks end the loop, and no prefill"
    assert not stop.prefill_walks, "a stop at prefill ending the partition isn't this model's behaviour"
    for hook in (
        "uniform_row_inputs", "device_loopback_signals", "inline_client_signals",
        "check_stop", "check_stop_batched", "loopback_inputs",
    ):
        assert hook not in vars(ThinkerSubmodule), f"{hook} follows from the declarations"


def test_every_config_that_splits_the_thinker_by_walk_places_the_text_walk():
    root = Path(__file__).resolve().parents[2]
    split = [
        group for path in root.glob("configs/qwen3omni*.yaml")
        for group in yaml.safe_load(path.read_text())["node_groups"]
        if "thinker_decode" in group.get("graph_walks", ())
    ]
    assert split, "the prefill/decode configs split the Thinker by walk"
    assert all(THINKER_TEXT_DECODE in group["graph_walks"] for group in split)


@pytest.mark.parametrize(("audio_output", "walk"), [(False, THINKER_TEXT_DECODE), (True, "thinker_decode")])
def test_prefill_hands_over_to_the_requests_decode_walk(audio_output, walk):
    metadata = _metadata(audio_output)
    args = _model()._get_thinker_forward(metadata, {"new_token": []})
    assert metadata.graph_walk == walk and not metadata.is_prefill
    assert args.inputs[0].name == "text_inputs"
    done = _model()._get_thinker_forward(metadata, {})
    assert done.request_done


def test_only_the_text_walk_keeps_its_token_on_the_device(thinker):
    assert thinker.device_loopback_signals(THINKER_TEXT_DECODE) == frozenset({"text_inputs"})
    assert thinker.device_loopback_signals("thinker_decode") == frozenset()
    assert thinker.inline_client_signals(THINKER_TEXT_DECODE) == {"text_inputs": "new_token"}
    assert thinker.inline_client_signals("thinker_decode") == {}
    assert thinker.uniform_row_inputs(THINKER_TEXT_DECODE, has_loopback=True).input_seq_len == 1
    assert thinker.uniform_row_inputs(THINKER_TEXT_DECODE, has_loopback=False) is None
    assert thinker.uniform_row_inputs("thinker_decode", has_loopback=True) is None


def test_the_tensor_path_stays_with_loop_back_off(thinker, monkeypatch):
    monkeypatch.setenv("MSTAR_DEVICE_LOOPBACK", "0")
    assert thinker.device_loopback_signals(THINKER_TEXT_DECODE) == frozenset()
    assert thinker.uniform_row_inputs(THINKER_TEXT_DECODE, has_loopback=True) is None
    row = thinker.prepare_inputs(THINKER_TEXT_DECODE, None, {"text_inputs": [torch.tensor([5])]}, has_loopback=True)
    assert row.input_ids.tolist() == [5], "a row brings its token when nothing keeps it"
    assert thinker.inline_client_signals(THINKER_TEXT_DECODE) == {"text_inputs": "new_token"}


def test_an_eager_only_sampler_keeps_the_tensor_path(thinker):
    thinker.bind_node_resources({THINKER_SAMPLER: SimpleNamespace(has_slot_masters=False)})
    assert thinker.device_loopback_signals(THINKER_TEXT_DECODE) == frozenset()
    assert thinker.uniform_row_inputs(THINKER_TEXT_DECODE, has_loopback=True) is None


@pytest.mark.parametrize("walk", DECODE_WALKS)
def test_a_decode_step_is_declared_input_free(thinker, walk):
    assert thinker.step_is_input_free(walk)
    template = ARNodeInputs(input_seq_len=1, input_ids=torch.zeros(1, dtype=torch.long))
    row = ARNodeInputs(input_seq_len=1)
    declared, real = (
        thinker.declare_step(walk, ["a", "b"], [one, one], per_request_info={}) for one in (template, row)
    )
    assert [(s.request_id, s.span) for s in declared.segments] == [(s.request_id, s.span) for s in real.segments]
    assert declared.cg_key_info is real.cg_key_info is None
    for step in (declared, real):
        sampler = step.steps[THINKER_SAMPLER]
        assert sampler.kept_rids is None and not sampler.prefill_tracked_tokens
        assert step.steps[THINKER_POS].advance is None


@pytest.mark.parametrize("walk", [*PREFILL_WALKS, THINKER_MIXED])
def test_a_step_that_reads_its_inputs_is_not_input_free(thinker, walk):
    assert not thinker.step_is_input_free(walk)


# ---- the loop-back row ----

def test_a_loop_back_row_brings_no_token_and_a_routed_one_does(thinker):
    row = thinker.prepare_inputs(THINKER_TEXT_DECODE, None, {}, has_loopback=True)
    assert row.input_ids is None and row.input_embeds is None and row.input_seq_len == 1


@pytest.mark.parametrize("walk", DECODE_WALKS)
@pytest.mark.parametrize("token", [torch.tensor(9), torch.tensor([9])])
def test_a_routed_decode_row_is_its_token_and_nothing_else(thinker, walk, token):
    row = thinker.prepare_inputs(walk, None, {"text_inputs": [token]})
    assert row.input_ids.tolist() == [9] and row.input_seq_len == 1
    assert row.input_embeds is None and row.custom_pos_ids is None, "the graph builds both"


def test_both_decode_captures_take_ids():
    thinker = ThinkerSubmodule(_Body(), SimpleNamespace(
        im_end_token_id=EOS, thinker_head_dim=128, thinker_hidden_size=HIDDEN,
        thinker_text=SimpleNamespace(rope_theta=1e6),
        vision=SimpleNamespace(deepstack_visual_indexes=[]),
    ))
    configs = {
        config.capture_graph_walk: config for config in thinker.get_accelerator_graph_configs(torch.device("cpu"))
    }
    for walk in DECODE_WALKS:
        row = configs[walk].single_request_inputs
        assert row.input_ids.shape == (1,) and row.input_embeds is None and row.custom_pos_ids is None


def test_a_decode_step_takes_each_rows_last_token_and_a_routed_row_keeps_its_own(thinker):
    thinker.node_resources[THINKER_POS] = SimpleNamespace(pos_ids=lambda label: torch.tensor([4.0, 5.0, 6.0]))
    row = ARNodeInputs(input_seq_len=1)
    routed = ARNodeInputs(input_seq_len=1, input_ids=torch.tensor([7]))
    out = thinker.preprocess(THINKER_TEXT_DECODE, _engine_inputs(["a", "b", "c"], [11, 12, 13]), [row, routed, row])
    assert out["input_ids"].tolist() == [11, 7, 13]
    assert out["cos_3d"].shape[0] == 3, "an eager step stages its rotary tables"


def test_a_padding_row_of_a_capture_keeps_what_its_slot_reads(thinker):
    rows = [ARNodeInputs(input_seq_len=1), ARNodeInputs(input_seq_len=0, input_ids=torch.tensor([0]))]
    out = thinker.preprocess(THINKER_TEXT_DECODE, _engine_inputs(["a", "b"], [11, 12], captured=True), rows)
    assert out["input_ids"].tolist() == [11, 12]


def test_a_step_of_routed_rows_alone_takes_their_tokens(thinker):
    thinker.node_resources[THINKER_POS] = SimpleNamespace(pos_ids=lambda label: torch.tensor([4.0, 5.0]))
    rows = [ARNodeInputs(input_seq_len=1, input_ids=torch.tensor([t])) for t in (3, 4)]
    out = thinker.preprocess(THINKER_TEXT_DECODE, _engine_inputs(["a", "b"], [11, 12]), rows)
    assert out["input_ids"].tolist() == [3, 4]


@pytest.mark.parametrize("walk", DECODE_WALKS)
def test_a_captured_decode_step_builds_its_rotary_tables_in_the_graph(thinker, walk):
    planned = torch.tensor([4.0, 5.0, 6.0, 99.0])
    thinker.node_resources[THINKER_POS] = SimpleNamespace(pos_ids=lambda label: planned)
    engine_inputs = _engine_inputs(["a", "b", "c"], [1, 2, 3], captured=True)
    rows = [ARNodeInputs(input_seq_len=1, input_ids=torch.tensor([t])) for t in (1, 2, 3)]
    out = thinker.preprocess(walk, engine_inputs, rows)
    assert set(out) == {"input_ids"} and out["input_ids"].tolist() == [1, 2, 3]

    tokens = thinker.forward_batched(walk, engine_inputs, **out)
    call = thinker.model.calls[-1]
    assert call["keep_states"] is (walk == "thinker_decode"), "only the Talker reads the states"
    want = compute_3d_cos_sin(
        planned[:3].unsqueeze(0).expand(3, -1), thinker._get_inv_freq(torch.device("cpu")),
        mrope_section=thinker.MROPE_SECTION, target_dtype=torch.float32,
    )
    assert torch.equal(call["cos_sin_3d"][0], want[0]) and torch.equal(call["cos_sin_3d"][1], want[1])
    assert isinstance(tokens, BatchedModelOutput)


def test_a_captured_audio_decode_step_keeps_the_talkers_states_and_mask(thinker):
    thinker.node_resources[THINKER_POS] = SimpleNamespace(pos_ids=lambda label: torch.tensor([4.0, 5.0]))
    engine_inputs = _engine_inputs(["a", "b"], [0, 0], captured=True)
    rows = [ARNodeInputs(input_seq_len=1, input_ids=torch.tensor([t])) for t in (3, 8)]
    out = thinker.preprocess("thinker_decode", engine_inputs, rows)
    out = thinker.forward_batched("thinker_decode", engine_inputs, **out)
    assert set(out.row_outputs) == {"thinker_states", "new_token", "thinker_mask"}
    assert out.row_outputs["thinker_states"][:, 0].tolist() == [3.0, 8.0], "the embeds the graph built, as layer 0"
    assert not out.row_outputs["thinker_mask"].any() and out.row_outputs["thinker_mask"].shape == (2,)
    assert set(out.check_stop_buffers) == {"new_token"}


def test_an_eager_audio_decode_step_stages_its_tables_and_keeps_the_talkers_states(thinker):
    thinker.node_resources[THINKER_POS] = SimpleNamespace(pos_ids=lambda label: torch.tensor([4.0]))
    rows = [ARNodeInputs(input_seq_len=1, input_ids=torch.tensor([3]))]
    single = _engine_inputs(["a"], [0])
    out = thinker.preprocess("thinker_decode", single, rows)
    assert out["cos_3d"].shape[0] == 1
    outputs = thinker.forward("thinker_decode", single, **out)
    assert set(outputs) == {"new_token", "thinker_states", "thinker_mask"}
    assert outputs["thinker_states"][0][:, 0].tolist() == [3.0]
    assert outputs["thinker_mask"][0].tolist() == [False]


@pytest.mark.parametrize("walk", DECODE_WALKS)
def test_a_decode_step_reads_nothing_on_the_host(thinker, walk):
    thinker.node_resources[THINKER_POS] = SimpleNamespace(pos_ids=lambda label: torch.tensor([4.0, 5.0, 6.0]))
    rows = [
        ARNodeInputs(input_seq_len=1),
        thinker.prepare_inputs(THINKER_TEXT_DECODE, None, {"text_inputs": [torch.tensor([7])]}, has_loopback=False),
        thinker.prepare_inputs(THINKER_TEXT_DECODE, None, {}, has_loopback=True),
    ] if walk == THINKER_TEXT_DECODE else [
        thinker.prepare_inputs(walk, None, {"text_inputs": [torch.tensor([t])]}) for t in (7, 8, 9)
    ]
    for captured in (False, True):
        engine_inputs = _engine_inputs(["a", "b", "c"], [11, 12, 13], captured=captured)
        with _HostReads() as reads:
            out = thinker.preprocess(walk, engine_inputs, rows)
            thinker.forward_batched(walk, engine_inputs, **out)
        assert reads.reads == []


def test_the_rope_from_the_plan_is_the_rope_custom_positions_gave():
    """The tables a captured step builds from planned positions, against the old custom ids."""
    inv_freq = compute_rope_freqs(128, rope_theta=1e6)
    planned = torch.tensor([3.0, 40.0, 500.0])
    tables = dict(mrope_section=(24, 20, 20), target_dtype=torch.bfloat16)
    from_plan = compute_3d_cos_sin(planned.unsqueeze(0).expand(3, -1), inv_freq, **tables)
    custom = [ARNodeInputs(input_seq_len=1, custom_pos_ids=torch.full((3, 1), float(p))) for p in planned]
    from_rows = compute_3d_cos_sin(mrope_position_ids(None, custom), inv_freq, **tables)
    assert torch.equal(from_plan[0], from_rows[0]) and torch.equal(from_plan[1], from_rows[1])


def test_a_text_decode_output_is_addressed_by_row(thinker):
    assert "filter_batched_output" not in vars(ThinkerSubmodule), (
        "a per-row filter would keep the step out of the engine's rows-only path"
    )
    engine_inputs = _engine_inputs(["a", "b"], [1, 2], captured=True)
    thinker.node_resources[THINKER_POS] = SimpleNamespace(pos_ids=lambda label: torch.tensor([1.0, 2.0]))
    out = thinker.forward_batched(THINKER_TEXT_DECODE, engine_inputs, input_ids=torch.tensor([1, 2]))
    assert set(out.row_outputs) == {"new_token"} and not out.per_rid_outputs and not out.packed_outputs
    assert set(out.check_stop_buffers) == {"new_token", "text_inputs"}
    assert out.check_stop_buffers["new_token"] is out.check_stop_buffers["text_inputs"], "one host copy a step"


# ---- the stop, on both walks ----

def test_either_decode_walk_ends_the_loop_and_both_checks_agree(thinker):
    # rows in forward order, plus a padding row a replay leaves at the end
    tokens = torch.tensor([EOS, EOS, 3, 3, EOS, EOS, 0])
    rids = ("text", "audio", "text-word", "audio-word", "text-ignored", "audio-ignored")
    infos = {
        "text": _info(THINKER_TEXT_DECODE),
        "audio": _info("thinker_decode"),
        "text-word": _info(THINKER_TEXT_DECODE),
        "audio-word": _info("thinker_decode"),
        "text-ignored": _info(THINKER_TEXT_DECODE, ignore_eos=True),
        "audio-ignored": _info("thinker_decode", ignore_eos=True),
    }
    batched = thinker.check_stop_batched(list(reversed(rids)), infos, HostRows(rids, {"new_token": tokens}))
    per_request = {}
    for i, rid in enumerate(rids):
        got = thinker.check_stop(rid, infos[rid], {"new_token": [tokens[i:i + 1]]})
        if got:
            per_request[rid] = got
    assert batched == per_request == {"text": {"thinker_decode_loop"}, "audio": {"thinker_decode_loop"}}
    assert thinker.check_stop_batched(["text"], infos, HostRows(("text",), {})) is None, "no such buffer staged"
    assert thinker.check_stop("text", infos["text"], {}) == set(), "a row with no token decides nothing"


def _runtime(model: Qwen3OmniModel, walk: str):
    """The graph runtime, with the request on ``walk``."""
    walks = model.get_graph_walk_graphs()
    names = [*PREFILL_WALKS, *DECODE_WALKS]
    ids = list(range(len(names)))
    runtime = build_runtime(
        my_worker_id="w0",
        my_worker_graphs=[
            WorkerGraph(section=walks[w], graph_walks={w}, ranks=[0], worker_graph_id=i)
            for i, w in zip(ids, names, strict=True)
        ],
        all_wg_ids_to_graph_walks={i: {w} for i, w in zip(ids, names, strict=True)},
        all_wg_ids_to_dyn_loops={
            i: {walks[w].name} if isinstance(walks[w], Loop) else set() for i, w in zip(ids, names, strict=True)
        },
        all_wg_ids_to_nodes=dict.fromkeys(ids, {"Thinker"}),
        node_to_partition={"Thinker": "Thinker"},
        sharding_config=_sharding_config(),
    )
    rid = runtime.add_request(
        request_id="r", partition="Thinker", graph_walk=walk, partition_worker_graph_ids=ids,
        worker_graph_to_workers=ParallelList(ids, [["w0"]] * len(ids)),
        loop_caps=model.get_loop_caps(max_output_tokens=16),
    )
    return runtime, rid


@pytest.mark.parametrize("walk", DECODE_WALKS)
def test_the_runtime_stops_the_loop_on_either_decode_walk(thinker, walk):
    named = thinker.check_stop_batched(["r"], {"r": _info(walk)}, HostRows(("r",), {"new_token": torch.tensor([EOS])}))
    assert named == {"r": {"thinker_decode_loop"}}

    runtime, rid = _runtime(_model(), walk)
    stopped = runtime.stop_loops_batched(
        partition="Thinker", graph_walk=walk, last_node_run="Thinker",
        loop_names=ParallelList([rid], [sorted(named["r"])]),
    )
    assert stopped == [rid]


@pytest.mark.parametrize("walk", PREFILL_WALKS)
def test_a_prefill_rows_stop_id_names_no_loop(thinker, walk):
    """A prefill row samples the stop id often (a prompt section ends a user turn) and is not in
    the loop yet. Naming the loop there would reach the runtime's model-bug path."""
    token = torch.tensor([EOS])
    assert thinker.check_stop_batched(["r"], {"r": _info(walk)}, HostRows(("r",), {"new_token": token})) == {}
    assert thinker.check_stop("r", _info(walk), {"new_token": [token]}) == set()


# ---- a mixed step ----

def test_a_mixed_step_embeds_its_loop_back_rows_beside_the_prompts(thinker):
    prompt = ARNodeInputs(input_seq_len=2, input_embeds=torch.full((2, HIDDEN), 20.0))
    row = ARNodeInputs(input_seq_len=1)
    routed = thinker.prepare_inputs(THINKER_TEXT_DECODE, None, {"text_inputs": [torch.tensor(9)]})
    thinker.node_resources[THINKER_POS] = SimpleNamespace(
        pos_ids=lambda label: torch.tensor([0.0, 1.0, 2.0, 3.0]),
    )
    out = thinker.preprocess(THINKER_MIXED, _engine_inputs(["a", "b", "c"], [5, 6, 7]), [prompt, row, routed])
    assert out["input_embeds"][:, 0].tolist() == [20.0, 20.0, 6.0, 9.0]
    assert out["seq_lens"] == [2, 1, 1]
    assert out["cos_3d"].shape[0] == 4


def test_a_mixed_step_embeds_an_audio_decode_row_at_its_planned_position(thinker):
    prompt = ARNodeInputs(input_seq_len=2, input_embeds=torch.full((2, HIDDEN), 20.0))
    row = thinker.prepare_inputs("thinker_decode", None, {"text_inputs": [torch.tensor(9)]})
    planned = torch.tensor([30.0, 31.0, 12.0])
    thinker.node_resources[THINKER_POS] = SimpleNamespace(pos_ids=lambda label: planned)
    # no loop-back row in this step, so nothing reads the sampler's slot masters
    engine_inputs = _engine_inputs(["a", "b"], [5, 6])
    engine_inputs.resources[THINKER_SAMPLER].loopback_tokens = None
    out = thinker.preprocess(THINKER_MIXED, engine_inputs, [prompt, row])
    assert out["input_embeds"][:, 0].tolist() == [20.0, 20.0, 9.0]
    want = compute_3d_cos_sin(
        planned.unsqueeze(0).expand(3, -1), thinker._get_inv_freq(torch.device("cpu")),
        mrope_section=thinker.MROPE_SECTION, target_dtype=torch.float32,
    )
    assert torch.equal(out["cos_3d"], want[0]) and torch.equal(out["sin_3d"], want[1])


def test_a_mixed_step_keeps_a_decode_rows_token_and_drops_a_mid_prompt_one(thinker):
    rows = [
        ARNodeInputs(input_seq_len=1, graph_walk=THINKER_TEXT_DECODE),
        ARNodeInputs(input_seq_len=1, graph_walk="thinker_decode"),
        ARNodeInputs(input_seq_len=4, graph_walk="prefill_text", input_ids=torch.arange(4)),
    ]
    step = thinker.declare_step(THINKER_MIXED, ["text", "audio", "prompt"], rows, per_request_info={})
    assert step.steps[THINKER_SAMPLER].kept_rids == frozenset({"text", "audio"})
    assert set(step.steps[THINKER_SAMPLER].prefill_tracked_tokens) == {"prompt"}


def test_a_mixed_steps_text_decode_rows_get_their_stop_and_inline_rows_and_no_talker_states(thinker):
    rids = ["text", "prompt"]
    inputs = [
        ARNodeInputs(input_seq_len=1, graph_walk=THINKER_TEXT_DECODE),
        ARNodeInputs(input_seq_len=2, graph_walk="prefill_text", tensor_inputs={
            "masks_for_talker": torch.ones(2, 2, dtype=torch.bool),
        }),
    ]
    infos = {
        "text": _info(THINKER_TEXT_DECODE),
        "prompt": SimpleNamespace(**{**vars(_info("prefill_text")), "step_metadata": {"audio_output": True}}),
    }
    embeds = torch.stack([torch.full((HIDDEN,), float(t)) for t in (EOS, 1, 2)])
    out = thinker.forward_batched(THINKER_MIXED, _engine_inputs(rids, [EOS, 0]), input_embeds=embeds)

    assert set(out.check_stop_buffers) == {"new_token", "text_inputs"}
    assert set(out.row_outputs) == {"new_token"}
    host = HostRows(tuple(rids), {"new_token": torch.tensor([EOS, EOS])})
    assert thinker.check_stop_batched(rids, infos, host) == {"text": {"thinker_decode_loop"}}

    unpacked = thinker.unpack_packed_outputs(out.packed_outputs, rids, [1, 2], inputs, infos)
    assert set(unpacked) == {"prompt"}, "a text-only row has no Talker to send states to"
    assert unpacked["prompt"]["thinker_states"][0].shape == (2, 2 * HIDDEN)
