"""The Thinker's postprocess picks the rows it streams to the Talker without reading the device: a decode row is its
layer-0 half, and a prefill row's kept rows come from a mask built on the host from its prompt ids. Runs on CPU
through the real prepare_inputs, forward_batched, Engine._collect_outputs and Engine._postprocess_batch, against the
boolean-index selection it replaced; meta tensors stand in for the device where the test is that nothing reads it."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mstar.engine.engine import Engine
from mstar.engine.resources.position.config import PositionConfig
from mstar.engine.resources.position.manager import RopeManager
from mstar.model.qwen3_omni.config import (
    AUDIO_PART,
    IMAGE_PART,
    TEXT_PART,
    THINKER_ATTN,
    THINKER_KV,
    THINKER_MIXED,
    THINKER_POS,
    THINKER_SAMPLER,
)
from mstar.model.qwen3_omni.submodules import ThinkerSubmodule
from mstar.model.submodule_base import ModelInputsFromEngine

HIDDEN = 8
VOCAB = 16
IM_START, SYSTEM, ASSISTANT, USER = 1, 2, 3, 9
NODE = "Thinker"


class _Layers(nn.Module):
    """Layer 0 is the input and layer N the input plus 100, so a selected row says which half it came from."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        with torch.no_grad():
            self.model.embed_tokens.weight.copy_(torch.arange(VOCAB * HIDDEN, dtype=torch.float).view(VOCAB, HIDDEN))
            self.lm_head.weight.zero_()

    def forward(self, input_embeds, **kwargs):
        return input_embeds, input_embeds, input_embeds + 100


class _Sampler:
    def sample(self, logits):
        return logits.argmax(-1)


class _Attn:
    def __init__(self, seq_lens):
        self.seq_lens = seq_lens

    def select_last_hidden(self, hidden):
        return hidden[torch.tensor(self.seq_lens).cumsum(0) - 1]


@pytest.fixture
def thinker():
    config = SimpleNamespace(
        im_start_token_id=IM_START, system_token_id=SYSTEM, assistant_token_id=ASSISTANT, thinker_hidden_size=HIDDEN,
        thinker=SimpleNamespace(
            audio_start_token_id=4, audio_end_token_id=5, vision_start_token_id=6, vision_end_token_id=7,
            position_id_per_seconds=25,
        ),
        vision=SimpleNamespace(spatial_merge_size=2, deepstack_visual_indexes=[1, 2]),
    )
    thinker = ThinkerSubmodule(_Layers(), config)
    thinker.node_resources = {THINKER_POS: RopeManager(
        config=PositionConfig(kv_cache=THINKER_KV), device=torch.device("cpu"), dtype=torch.float32,
    )}
    return thinker


def _info(rid, walk, order=None, audio=True, decode_iter=1):
    metadata = {"audio_output": audio, **({"prefill_order": order} if order else {})}
    return SimpleNamespace(
        rid_handle=rid, graph_walk=walk, step_metadata=metadata,
        dynamic_loop_iter_counts={"thinker_decode_loop": decode_iter},
    )


def _reference(ts, masks, first_embed=None):
    """The selection this replaced: a boolean index, which needs the kept row count from the device."""
    multimodal = masks[0].bool()
    inclusion = (masks[1].bool() & ~multimodal) | multimodal
    selected = torch.where(multimodal.unsqueeze(-1), ts[..., HIDDEN:], ts[..., :HIDDEN])
    states, mask = selected[inclusion], multimodal[inclusion]
    if first_embed is not None:
        states = torch.cat([states, first_embed.to(states.dtype)])
        mask = torch.cat([mask, mask.new_zeros(1)])
    return states, mask


# a system turn the Talker drops, a user turn it keeps, an earlier assistant turn it drops, then the generation prompt
CHAT = [
    IM_START, SYSTEM, 10, 11, IM_START, USER, 12, 13, IM_START, ASSISTANT, 14, IM_START, USER, 15, IM_START, ASSISTANT,
]


def _requests(thinker):
    """A prefill with audio, an image and text around them, a text prefill, and a decode row, as one mixed step."""
    media = {
        "text_inputs": [torch.tensor([IM_START, SYSTEM, 10, IM_START, USER]), torch.tensor([12, 13]),
                        torch.tensor([14, IM_START, ASSISTANT])],
        "audio_embeds": [torch.randn(3, HIDDEN)],
        "vision_embeds": [torch.randn(6, HIDDEN)], "deepstack": [torch.randn(6, HIDDEN)] * 2,
        "image_grid_thw": [torch.tensor([1, 4, 6])],
    }
    order = (TEXT_PART, AUDIO_PART, TEXT_PART, IMAGE_PART, TEXT_PART)
    infos = {
        "a": _info("a", "prefill_audio_vision", order),
        "t": _info("t", "prefill_text"),
        "d": _info("d", "thinker_decode"),
    }
    inputs = [
        thinker.prepare_inputs("prefill_audio_vision", infos["a"], media),
        thinker.prepare_inputs("prefill_text", infos["t"], {"text_inputs": [torch.tensor(CHAT)]}),
        thinker.prepare_inputs("thinker_decode", infos["d"], {"text_inputs": [torch.tensor([5])]}),
    ]
    return ["a", "t", "d"], infos, inputs


def _step(thinker, rids, infos, inputs):
    """The engine's real path from the forward to the per-request outputs the worker routes, keeping the forward's
    per-request rows and masks as they were before postprocess."""
    lens = [inp.input_seq_len for inp in inputs]
    engine_inputs = ModelInputsFromEngine(
        request_ids=list(rids), per_request_info={},
        resources={THINKER_ATTN: _Attn(lens), THINKER_SAMPLER: _Sampler()},
    )
    raw = thinker.forward_batched(
        THINKER_MIXED, engine_inputs, input_embeds=torch.cat([inp.input_embeds for inp in inputs]),
    )
    engine = Engine.__new__(Engine)
    mgmt = SimpleNamespace(submodule=thinker, accelerator_graph_runner=None)
    engine._submodules = {NODE: mgmt}
    out = engine._collect_outputs(
        mgmt, None, raw, inputs, infos, request_ids=list(rids), step_request_ids=tuple(rids), graph_walk=THINKER_MIXED,
    )
    before = {rid: dict(out.per_rid_outputs[rid]) for rid in rids}
    failures = {}
    batch = SimpleNamespace(
        node_name=NODE, request_ids=list(rids), inputs=inputs, per_request_info_wrapped=infos, failed_requests=failures,
        register_failure=failures.__setitem__,
    )
    engine._postprocess_batch(batch, out)
    assert failures == {}
    return out, before


def test_every_row_streams_what_the_boolean_index_picked(thinker):
    rids, infos, inputs = _requests(thinker)
    out, before = _step(thinker, rids, infos, inputs)

    for rid in rids:
        row = out.per_rid_outputs[rid]
        first = None if rid == "d" else before[rid]["first_token_embed"][0]
        states, mask = _reference(before[rid]["thinker_states"][0], before[rid]["thinker_mask"][0], first)
        assert torch.equal(row["thinker_states"][0], states), rid
        assert torch.equal(row["thinker_mask"][0], mask), rid
        assert row["text_inputs"] is row["new_token"]
    # the system turn and the earlier assistant turn are gone, and the media rows went through whole
    assert out.per_rid_outputs["t"]["thinker_states"][0].shape[0] == 4 + 5 + 1
    assert out.per_rid_outputs["a"]["thinker_mask"][0].sum() == 3 + 6


def test_the_rows_are_picked_without_reading_the_device(thinker):
    """The forward's outputs on the meta device have no values to read, so a boolean index, ``.item()`` or ``if`` on
    one raises; the host's keep mask is all the selection may read."""
    rids, infos, inputs = _requests(thinker)
    _, before = _step(thinker, rids, infos, inputs)

    for rid, inp in zip(rids, inputs, strict=True):
        ts, masks = before[rid]["thinker_states"][0], before[rid]["thinker_mask"][0]
        outputs = {
            "new_token": [before[rid]["new_token"][0].to("meta")],
            "thinker_states": [ts.to("meta")],
            "thinker_mask": [masks if rid == "d" else masks.to("meta")],
        }
        if rid != "d":
            outputs["first_token_embed"] = [before[rid]["first_token_embed"][0].to("meta")]
        thinker.postprocess(rid, infos[rid], outputs, inputs=inp)

        states, mask = _reference(ts, masks, None if rid == "d" else before[rid]["first_token_embed"][0])
        assert outputs["thinker_states"][0].shape == states.shape, rid
        assert outputs["thinker_mask"][0].shape == mask.shape, rid
    with pytest.raises((NotImplementedError, RuntimeError)):
        _reference(before["t"]["thinker_states"][0].to("meta"), before["t"]["thinker_mask"][0].to("meta"))


def test_a_decode_row_streams_a_view_of_its_layer_0_half(thinker):
    rids, infos, inputs = _requests(thinker)
    out, before = _step(thinker, rids, infos, inputs)

    states = out.per_rid_outputs["d"]["thinker_states"][0]
    assert states.shape == (1, HIDDEN)
    assert states.data_ptr() == before["d"]["thinker_states"][0].data_ptr()
    assert out.per_rid_outputs["d"]["thinker_mask"][0].tolist() == [False]


@pytest.mark.parametrize("start, end", [(3, len(CHAT)), (0, 3), (5, 12)])
def test_a_cut_prefill_keeps_the_rows_its_cut_of_the_masks_names(thinker, start, end):
    info = _info("t", "prefill_text")
    whole = thinker.prepare_inputs("prefill_text", info, {"text_inputs": [torch.tensor(CHAT)]})
    cut = thinker.split_inputs("prefill_text", info, whole, start, end)
    assert torch.equal(cut.tensor_inputs["talker_keep"], whole.tensor_inputs["talker_keep"][start:end])

    _, before = _step(thinker, ["t"], {"t": info}, [cut])
    ts, masks = before["t"]["thinker_states"][0], before["t"]["thinker_mask"][0]
    outputs = {k: list(v) for k, v in before["t"].items()}
    thinker.postprocess("t", info, outputs, inputs=cut)

    states, mask = _reference(ts, masks, before["t"]["first_token_embed"][0])
    assert torch.equal(outputs["thinker_states"][0], states)
    assert torch.equal(outputs["thinker_mask"][0], mask)


def test_the_host_mask_is_the_text_inclusion_row_the_talker_masks_carry(thinker):
    whole = thinker.prepare_inputs("prefill_text", _info("t", "prefill_text"), {"text_inputs": [torch.tensor(CHAT)]})

    keep = whole.tensor_inputs["talker_keep"]
    assert keep.device.type == "cpu" and keep.dtype == torch.bool
    assert torch.equal(whole.tensor_inputs["masks_for_talker"][1], keep)
    assert keep.tolist() == [False] * 4 + [True] * 4 + [False] * 3 + [True] * 5
