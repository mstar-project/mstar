"""A Qwen3-Omni request with no audio output decodes on its own walk: its token loops back
on the device, reaches the client inline and sends nothing to the Talker. A request with
audio output keeps ``thinker_decode``."""
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("flashinfer")

from mstar.conductor.request_info import CurrentForwardConductorMetadata  # noqa: E402
from mstar.engine.resources.sampler.resource import SamplerResource  # noqa: E402
from mstar.graph.special_destinations import EMIT_TO_CLIENT  # noqa: E402
from mstar.model.components.mrope import mrope_position_ids  # noqa: E402
from mstar.model.qwen3_omni.components.rope import compute_3d_cos_sin, compute_rope_freqs  # noqa: E402
from mstar.model.qwen3_omni.config import (  # noqa: E402
    THINKER_MIXED,
    THINKER_POS,
    THINKER_SAMPLER,
    THINKER_TEXT_DECODE,
)
from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel  # noqa: E402
from mstar.model.qwen3_omni.submodules import ThinkerSubmodule  # noqa: E402
from mstar.model.submodule_base import ARNodeInputs, BatchedModelOutput, HostRows, ModelInputsFromEngine  # noqa: E402

EOS = 7


def _info(walk, decoded=0, max_tokens=100, ignore_eos=False):
    return SimpleNamespace(
        graph_walk=walk, max_tokens=max_tokens, dynamic_loop_iter_counts={"thinker_decode_loop": decoded},
        resource_configs={THINKER_SAMPLER: SimpleNamespace(ignore_eos=ignore_eos)},
    )


@pytest.fixture
def thinker():
    sub = object.__new__(ThinkerSubmodule)
    sub.__dict__.update(
        config=SimpleNamespace(im_end_token_id=EOS, thinker_head_dim=128, thinker_text=SimpleNamespace(rope_theta=1e6)),
        _inv_freq=None,
        node_resources={THINKER_SAMPLER: SimpleNamespace(has_slot_masters=True)},
        model=SimpleNamespace(
            mask_padding=False,
            model=SimpleNamespace(embed_tokens=torch.nn.Embedding.from_pretrained(torch.arange(32.0).unsqueeze(-1))),
            maybe_log_expert_load=lambda synthetic=False: None,
        ),
    )
    sub.get_device = lambda: torch.device("cpu")
    return sub


def _model():
    model = object.__new__(Qwen3OmniModel)
    model.config = SimpleNamespace(thinker_text=SimpleNamespace(max_position_embeddings=64))
    return model


def _metadata(audio_output: bool, walk="prefill_text") -> CurrentForwardConductorMetadata:
    return CurrentForwardConductorMetadata(
        input_modalities=["text"], output_modalities=["text"], graph_walk=walk, is_prefill=True,
        kwargs={"prefill_schedule": [("prefill_text", {})], "prefill_step": 0, "audio_output": audio_output},
    )


def test_the_walk_loops_its_token_and_emits_it_with_no_edge_to_the_talker():
    model = _model()
    walk = model.get_graph_walk_graphs()[THINKER_TEXT_DECODE]
    assert walk.name == "thinker_decode_loop", "the loop's cap and stop check are the decode walk's"
    assert [(edge.next_node, edge.name) for edge in walk.section.outputs] == [
        ("Thinker", "text_inputs"), (EMIT_TO_CLIENT, "text_inputs"),
    ]
    assert THINKER_TEXT_DECODE in model.get_combined_graph_walks()["Thinker"][THINKER_MIXED]


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
    row = thinker.prepare_inputs(
        THINKER_TEXT_DECODE, None, {"text_inputs": [torch.tensor([5])]}, has_loopback=True,
    )
    assert row.input_ids.tolist() == [5], "a row brings its token when nothing keeps it"


def test_only_decode_steps_are_declared_input_free(thinker):
    assert thinker.step_is_input_free(THINKER_TEXT_DECODE) and thinker.step_is_input_free("thinker_decode")
    assert not thinker.step_is_input_free("prefill_text") and not thinker.step_is_input_free(THINKER_MIXED)


def test_a_loop_back_row_brings_no_token_and_a_routed_one_does(thinker):
    row = thinker.prepare_inputs(THINKER_TEXT_DECODE, None, {}, has_loopback=True)
    assert row.input_ids is None and row.input_embeds is None and row.input_seq_len == 1
    routed = thinker.prepare_inputs(THINKER_TEXT_DECODE, None, {"text_inputs": [torch.tensor(9)]})
    assert routed.input_ids.tolist() == [9] and routed.input_embeds.shape == (1, 1)


def _engine_inputs(request_ids, tokens):
    sampler = SimpleNamespace(
        loopback_tokens=lambda rids: torch.tensor(tokens),
        overwrite_rows=lambda rows, *updates: SamplerResource.overwrite_rows(sampler, rows, *updates),
        _row_stager=None,
    )
    return ModelInputsFromEngine(
        request_ids=request_ids, per_request_info={}, resources={THINKER_SAMPLER: sampler},
    )


def test_a_decode_step_takes_each_rows_last_token_and_a_routed_row_keeps_its_own(thinker, monkeypatch):
    thinker.node_resources[THINKER_POS] = SimpleNamespace(pos_ids=lambda label: torch.tensor([4.0, 5.0, 6.0]))
    row = ARNodeInputs(input_seq_len=1)
    routed = ARNodeInputs(input_seq_len=1, input_ids=torch.tensor([7]))
    out = thinker.preprocess(THINKER_TEXT_DECODE, _engine_inputs(["a", "b", "c"], [11, 12, 13]), [row, routed, row])
    assert out["input_ids"].tolist() == [11, 7, 13]
    assert out["cos_3d"].shape[0] == 3, "an eager step stages its rotary tables"


def test_a_mixed_step_embeds_its_loop_back_rows_beside_the_prompts(thinker):
    prompt = ARNodeInputs(input_seq_len=2, input_embeds=torch.tensor([[1.0], [2.0]]))
    row = ARNodeInputs(input_seq_len=1)
    embeds = thinker._row_embeds(_engine_inputs(["a", "b"], [5, 6]), [prompt, row])
    assert torch.cat(embeds).squeeze(-1).tolist() == [1.0, 2.0, 6.0]


def test_the_rope_from_the_plan_is_the_rope_custom_positions_gave():
    """The tables a captured step builds from planned positions, against the old custom ids."""
    inv_freq = compute_rope_freqs(128, rope_theta=1e6)
    planned = torch.tensor([3.0, 40.0, 500.0])
    from_plan = compute_3d_cos_sin(planned.unsqueeze(0).expand(3, -1), inv_freq, (24, 20, 20), target_dtype=torch.bfloat16)
    custom = [ARNodeInputs(input_seq_len=1, custom_pos_ids=torch.full((3, 1), float(p))) for p in planned]
    from_rows = compute_3d_cos_sin(mrope_position_ids(None, custom), inv_freq, (24, 20, 20), target_dtype=torch.bfloat16)
    assert torch.equal(from_plan[0], from_rows[0]) and torch.equal(from_plan[1], from_rows[1])


def test_the_batched_stop_check_agrees_with_the_per_request_one(thinker):
    # rows in forward order, plus a padding row a replay leaves at the end
    tokens = torch.tensor([EOS, 3, EOS, 3, EOS, 0])
    rids = ("eos", "word", "ignored", "budget", "prefill")
    infos = {
        "eos": _info(THINKER_TEXT_DECODE),
        "word": _info(THINKER_TEXT_DECODE),
        "ignored": _info("thinker_decode", ignore_eos=True),
        "budget": _info(THINKER_TEXT_DECODE, decoded=9, max_tokens=10),
        "prefill": _info("prefill_text"),
    }
    batched = thinker.check_stop_batched(list(reversed(rids)), infos, HostRows(rids, {"new_token": tokens}))
    per_request = {}
    for i, rid in enumerate(rids):
        got = thinker.check_stop(rid, infos[rid], {"new_token": [tokens[i:i + 1]]})
        if got:
            per_request[rid] = got
    assert batched == per_request == {"eos": {"thinker_decode_loop"}, "budget": {"thinker_decode_loop"}}
    assert thinker.check_stop_batched(["word"], infos, HostRows(("word",), {})) is None
    assert thinker.check_stop_batched(["eos"], infos, HostRows(rids, {"new_token": tokens}), "prefill_text") == {}


def test_a_step_keeps_a_decode_rows_token_and_drops_a_mid_prompt_one(thinker):
    rows = [
        ARNodeInputs(input_seq_len=1, graph_walk=THINKER_TEXT_DECODE),
        ARNodeInputs(input_seq_len=4, graph_walk="prefill_text", input_ids=torch.arange(4)),
    ]
    step = thinker.declare_step(THINKER_MIXED, ["decode", "prompt"], rows, per_request_info={})
    sampler = next(s for s in step.steps.values() if hasattr(s, "kept_rids"))
    assert sampler.kept_rids == frozenset({"decode"})


def test_a_text_decode_output_is_addressed_by_row(thinker):
    assert BatchedModelOutput(row_outputs={"new_token": torch.zeros(1)}).per_rid_outputs == {}
    assert "filter_batched_output" not in vars(ThinkerSubmodule), (
        "a per-row filter would keep the step out of the engine's rows-only path"
    )
