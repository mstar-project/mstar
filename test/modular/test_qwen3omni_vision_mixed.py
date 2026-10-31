"""Qwen3-Omni vision prefill rows ride in the Thinker's mixed steps: deepstack
is packed over every row (zeros beside non-vision rows), each row advances
MRoPE positions by its own rule, and such a step replays captures that take
deepstack. A decode row's Talker mask is a tensor of its own."""
from types import SimpleNamespace

import pytest
import torch

from mstar.model.qwen3_omni.config import THINKER_MIXED, THINKER_POS
from mstar.model.qwen3_omni.submodules import ThinkerSubmodule
from mstar.model.submodule_base import ARNodeInputs

HIDDEN = 4


@pytest.fixture
def thinker():
    thinker = object.__new__(ThinkerSubmodule)
    thinker.__dict__["config"] = SimpleNamespace(
        vision=SimpleNamespace(deepstack_visual_indexes=[1, 2]), thinker_hidden_size=HIDDEN,
    )
    return thinker


def _row(n, walk, deepstack=None, advance=None):
    tensor_inputs = {}
    if deepstack is not None:
        tensor_inputs = {"deepstack": deepstack, "mrope_pos_advance": advance}
    return ARNodeInputs(input_seq_len=n, graph_walk=walk, tensor_inputs=tensor_inputs)


def test_deepstack_is_packed_over_every_row(thinker):
    vision = [torch.full((3, HIDDEN), 1.0), torch.full((3, HIDDEN), 2.0)]
    rows = [_row(1, "thinker_decode"), _row(3, "prefill_vision", vision, 9), _row(2, "prefill_text")]

    out = thinker._deepstack_inputs(rows, torch.float, torch.device("cpu"))

    assert sorted(out) == ["deepstack_0", "deepstack_1"]
    assert out["deepstack_0"][:, 0].tolist() == [0, 1, 1, 1, 0, 0]
    assert out["deepstack_1"][:, 0].tolist() == [0, 2, 2, 2, 0, 0]


def test_each_row_of_a_mixed_step_advances_positions_by_its_own_rule(thinker):
    vision = [torch.zeros(3, HIDDEN)] * 2
    rows = [_row(1, "thinker_decode"), _row(3, "prefill_vision", vision, 9), _row(2, "prefill_text")]

    step = thinker.declare_step(THINKER_MIXED, [0, 1, 2], rows)

    assert step.steps[THINKER_POS].advance == (1, 9, 2)


@pytest.mark.parametrize("step_walk,walks,key", [
    (THINKER_MIXED, ["thinker_decode", "prefill_vision"], "deepstack"),
    (THINKER_MIXED, ["thinker_decode", "prefill_text"], None),
    ("prefill_vision", ["prefill_vision", "prefill_vision"], "deepstack"),
])
def test_a_step_with_a_vision_row_replays_the_deepstack_captures(thinker, step_walk, walks, key):
    info = {i: SimpleNamespace(graph_walk=w, step_metadata={}) for i, w in enumerate(walks)}

    assert thinker.cg_key_info(step_walk, info) == key


def test_one_vision_capture_serves_vision_and_mixed_steps(thinker):
    configs = thinker.get_accelerator_graph_configs(torch.device("cpu"))
    vision = [c for c in configs if "prefill_vision" in c.replay_graph_walks]

    assert len(vision) == 1
    assert vision[0].replay_graph_walks == ["prefill_vision", THINKER_MIXED]
    assert vision[0].additional_key_info == "deepstack"


@pytest.mark.parametrize("step_walk,walks", [
    (THINKER_MIXED, ["thinker_decode", "prefill_vision"]),
    (THINKER_MIXED, ["thinker_decode", "prefill_text"]),
    ("prefill_vision", ["prefill_vision", "prefill_vision"]),
])
def test_declare_step_stamps_the_key_cg_key_info_leased(thinker, step_walk, walks):
    vision = [torch.zeros(1, HIDDEN)] * 2
    rows = [_row(1, w, vision if w == "prefill_vision" else None, 4) for w in walks]
    info = {i: SimpleNamespace(graph_walk=w, step_metadata={}) for i, w in enumerate(walks)}

    step = thinker.declare_step(step_walk, list(range(len(rows))), rows)

    assert step.cg_key_info == thinker.cg_key_info(step_walk, info)


class _DecodeModel:
    """The Thinker body at decode: one hidden per row, a token per row."""

    def __call__(self, input_embeds, **kwargs):
        return input_embeds, input_embeds, input_embeds

    def lm_head(self, hidden):
        return hidden


class _Greedy:
    def sample(self, request_ids, logits):
        return logits.argmax(-1)


def _decode_rows_outputs(thinker, step_walk, walks):
    """What postprocess leaves for each decode row of a step: from a decode
    step (row views of one clone) or a mixed one (cut from the pack)."""
    from mstar.model.qwen3_omni.config import THINKER_ATTN, THINKER_SAMPLER

    rids = [str(i) for i in range(len(walks))]
    if step_walk == "thinker_decode":
        thinker.__dict__["model"] = _DecodeModel()
        engine_inputs = SimpleNamespace(
            resources={THINKER_ATTN: None, THINKER_SAMPLER: _Greedy()}, request_ids=rids,
        )
        out = thinker.forward_batched(step_walk, engine_inputs, input_embeds=torch.randn(len(rids), HIDDEN))
        # what the engine hands each row: views of one clone of the batch tensor
        rows = out.clone_row_outputs(len(rids))
        per_row = [{name: [rows[name].split(1)[i]] for name in rows} for i in range(len(rids))]
    else:
        lens = [1 if w == "thinker_decode" else 3 for w in walks]
        inputs = [
            ARNodeInputs(input_seq_len=n, graph_walk=w, tensor_inputs={} if w == "thinker_decode" else {
                "masks_for_talker": torch.ones(2, n),
            })
            for n, w in zip(lens, walks, strict=True)
        ]
        packed = {"__batched_thinker_states__": torch.randn(sum(lens), 2 * HIDDEN)}
        unpacked = thinker.unpack_packed_outputs(packed, rids, lens, inputs, {})
        per_row = [unpacked[rid] for rid in rids]
    for rid, walk, outputs in zip(rids, walks, per_row, strict=True):
        thinker.postprocess(rid, SimpleNamespace(graph_walk=walk, step_metadata={}), outputs)
    return [o for o, w in zip(per_row, walks, strict=True) if w == "thinker_decode"]


@pytest.mark.parametrize("step_walk,walks", [
    ("thinker_decode", ["thinker_decode"] * 3),
    (THINKER_MIXED, ["thinker_decode", "prefill_text", "thinker_decode", "thinker_decode"]),
])
def test_every_decode_row_gets_its_own_talker_mask(thinker, step_walk, walks):
    """A decode row's Talker mask is ``[False]`` in storage of its own, in a
    decode step or a mixed one: the RDMA/TCP store registers each output
    tensor's address for remote reads, and one tensor shared by every row
    would be registered once per row at the same address."""
    masks = [o["thinker_mask"][0] for o in _decode_rows_outputs(thinker, step_walk, walks)]

    assert all(m.dtype == torch.bool and m.shape == (1,) and not m.any() for m in masks)
    assert len({m.data_ptr() for m in masks}) == len(masks)


def test_a_decode_step_s_thinker_states_are_the_rows_the_engine_handed_out(thinker):
    """A decode step's ``thinker_states`` row output is already the Talker's
    layer-0 row, so postprocess leaves the engine's row view as it is: a
    rebound slice would fail the row-output fast store's identity check."""
    from mstar.model.qwen3_omni.config import THINKER_ATTN, THINKER_SAMPLER

    rids = ["0", "1", "2"]
    thinker.__dict__["model"] = _DecodeModel()
    engine_inputs = SimpleNamespace(resources={THINKER_ATTN: None, THINKER_SAMPLER: _Greedy()}, request_ids=rids)
    out = thinker.forward_batched("thinker_decode", engine_inputs, input_embeds=torch.randn(len(rids), HIDDEN))
    rows = out.clone_row_outputs(len(rids))
    views = rows["thinker_states"].split(1)
    for i, rid in enumerate(rids):
        outputs = {name: [rows[name].split(1)[i]] for name in rows}
        outputs["thinker_states"] = [views[i]]
        thinker.postprocess(rid, SimpleNamespace(graph_walk="thinker_decode", step_metadata={}), outputs)
        assert outputs["thinker_states"][0] is views[i]
        assert outputs["thinker_states"][0].shape == (1, HIDDEN)
    # a lone decode row runs the unbatched forward: the same layer-0 row
    info = SimpleNamespace(graph_walk="thinker_decode", step_metadata={})
    single = SimpleNamespace(
        resources={THINKER_SAMPLER: _Greedy()}, request_ids=["0"], single_request_info=info,
    )
    outputs = thinker.forward("thinker_decode", single, input_embeds=torch.randn(1, HIDDEN))
    thinker.postprocess("0", info, outputs)
    assert outputs["thinker_states"][0].shape == (1, HIDDEN)
