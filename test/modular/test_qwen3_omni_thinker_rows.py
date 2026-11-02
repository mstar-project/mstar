"""The Thinker's batched forward returns rows in batch order, and the engine maps
them to request ids outside the compiled region, so a new id never guards a frame.
Runs on CPU with stub layers, through the real ``forward_batched`` and the real
``Engine._collect_outputs``; dynamo's guards are the eager backend's, no compiler."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch._dynamo.convert_frame import register_bytecode_hook

from mstar.engine.engine import Engine
from mstar.model.qwen3_omni.config import THINKER_ATTN, THINKER_SAMPLER
from mstar.model.qwen3_omni.submodules import ThinkerSubmodule
from mstar.model.submodule_base import ARNodeInputs, BatchedModelOutput, ModelInputsFromEngine

HIDDEN = 4
VOCAB = 16
DECODE_MASK = torch.tensor([[False], [True]])


class _Head(nn.Module):
    """Highest for the token the row's first column names."""

    def forward(self, hidden):
        return -(torch.arange(VOCAB, dtype=hidden.dtype) - hidden[..., :1]) ** 2


class _Layers(nn.Module):
    """Layer 0 is the input and layer N the input plus 100, so each row of the
    states says which row it came from."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.lm_head = _Head()
        with torch.no_grad():
            self.model.embed_tokens.weight.copy_(torch.arange(VOCAB * HIDDEN, dtype=torch.float).view(VOCAB, HIDDEN))

    def forward(self, input_embeds, **kwargs):
        return input_embeds, input_embeds, input_embeds + 100


class _Sampler:
    # the engine's sampler is kept out of dynamo too, which is where the frame breaks
    @torch.compiler.disable
    def sample(self, logits):
        return logits.argmax(-1)


class _Attn:
    def __init__(self, seq_lens):
        self.seq_lens = seq_lens

    def select_last_hidden(self, hidden):
        return hidden[torch.tensor(self.seq_lens).cumsum(0) - 1]


def _thinker():
    return ThinkerSubmodule(_Layers(), SimpleNamespace(
        vision=SimpleNamespace(deepstack_visual_indexes=[1, 2]), thinker_hidden_size=HIDDEN,
    ))


def _rows(tokens):
    """One input row per token, whose ``lm_head`` output samples that token back."""
    return torch.tensor(tokens, dtype=torch.float).unsqueeze(-1).repeat(1, HIDDEN)


def _engine_inputs(rids, seq_lens=None):
    return ModelInputsFromEngine(
        request_ids=list(rids), per_request_info={},
        resources={THINKER_ATTN: _Attn(seq_lens or [1] * len(rids)), THINKER_SAMPLER: _Sampler()},
    )


def _collect(thinker, raw, rids, inputs, audio=None, lease=None, dummy_rids=None):
    """The engine's real mapping from what the forward returned to per-rid dicts."""
    audio = audio or {}
    info = {rid: SimpleNamespace(step_metadata={"audio_output": audio.get(rid, True)}) for rid in rids}
    mgmt = SimpleNamespace(
        submodule=thinker,
        accelerator_graph_runner=SimpleNamespace(slot_for=lambda _: SimpleNamespace(dummy_rids=dummy_rids)),
    )
    return Engine.__new__(Engine)._collect_outputs(
        mgmt, lease, raw, inputs, info, request_ids=list(rids), step_request_ids=tuple(rids),
        graph_walk="unused",
    )


def _decode_inputs(n):
    return [ARNodeInputs(input_seq_len=1, tensor_inputs={"masks_for_talker": DECODE_MASK}) for _ in range(n)]


def _prefill_inputs(lens):
    return [
        ARNodeInputs(input_seq_len=n, tensor_inputs={"masks_for_talker": torch.ones(2, n, dtype=torch.bool)})
        for n in lens
    ]


def test_decode_rows_reach_their_requests_in_order():
    thinker = _thinker()
    rids, tokens = [7, 3, 5], [1, 2, 3]
    embeds = _rows(tokens)

    raw = thinker.forward_batched("thinker_decode", _engine_inputs(rids), input_embeds=embeds)
    out = _collect(thinker, raw, rids, _decode_inputs(3))

    for i, rid in enumerate(rids):
        row = out.per_rid_outputs[rid]
        assert row["new_token"][0].tolist() == [tokens[i]]
        assert torch.equal(row["thinker_states"][0], torch.cat([embeds[i:i + 1], embeds[i:i + 1] + 100], dim=-1))
        assert torch.equal(row["thinker_mask"][0], DECODE_MASK)
    assert out.row_request_ids == tuple(rids)


def test_prefill_rows_and_packed_states_reach_their_requests_in_order():
    thinker = _thinker()
    rids, lens, tokens = [9, 2, 6], [2, 3, 1], [4, 5, 6]
    embeds = _rows([0, tokens[0], 0, 0, tokens[1], tokens[2]])
    inputs = _prefill_inputs(lens)

    raw = thinker.forward_batched("prefill_text", _engine_inputs(rids, lens), input_embeds=embeds)
    out = _collect(thinker, raw, rids, inputs)

    states = torch.cat([embeds, embeds + 100], dim=-1)
    start = 0
    for i, (rid, n) in enumerate(zip(rids, lens, strict=True)):
        row = out.per_rid_outputs[rid]
        assert row["new_token"][0].tolist() == [tokens[i]]
        assert torch.equal(row["first_token_embed"][0], thinker.model.model.embed_tokens(torch.tensor([tokens[i]])))
        assert torch.equal(row["thinker_states"][0], states[start:start + n])
        assert row["thinker_mask"][0] is inputs[i].tensor_inputs["masks_for_talker"]
        start += n


@pytest.mark.parametrize("walk", ["thinker_decode", "prefill_text"])
def test_a_text_only_request_gets_its_token_and_nothing_for_the_talker(walk):
    thinker = _thinker()
    rids = [1, 2]
    embeds, inputs = (
        (_rows([1, 2]), _decode_inputs(2)) if walk == "thinker_decode" else (_rows([3, 4]), _prefill_inputs([1, 1]))
    )

    raw = thinker.forward_batched(walk, _engine_inputs(rids), input_embeds=embeds)
    out = _collect(thinker, raw, rids, inputs, audio={1: False})

    assert set(out.per_rid_outputs[1]) == {"new_token"}
    assert "thinker_states" in out.per_rid_outputs[2] and "thinker_mask" in out.per_rid_outputs[2]


@pytest.mark.parametrize("walk", ["thinker_decode", "prefill_text"])
def test_a_captured_replay_hands_real_rows_to_real_ids_and_drops_the_padding(walk):
    thinker = _thinker()
    dummy_rids = [-2, -3, -4, -5]
    rids = [8, 1]
    embeds = _rows([1, 2, 3, 4])
    # capture ran on the padding rows; the slot's outputs are what every replay returns
    slot_output = thinker.forward_batched(walk, _engine_inputs(dummy_rids), input_embeds=embeds)
    inputs = _decode_inputs(4) if walk == "thinker_decode" else _prefill_inputs([1, 1, 1, 1])

    out = _collect(thinker, slot_output, rids, inputs, lease=object(), dummy_rids=dummy_rids)

    assert set(out.per_rid_outputs) == set(rids)
    assert out.per_rid_outputs[8]["new_token"][0].tolist() == [1]
    assert out.per_rid_outputs[1]["new_token"][0].tolist() == [2]
    assert out.per_rid_outputs[1]["thinker_states"][0].shape[0] == 1
    # a replay writes the same buffers next step: what a request holds is its own copy
    assert not any(
        t.untyped_storage().data_ptr() == slot_output.row_outputs["new_token"].untyped_storage().data_ptr()
        for t in out.per_rid_outputs[8]["new_token"]
    )


def test_a_request_that_leaves_mid_batch_takes_only_its_own_row():
    thinker = _thinker()
    rids, tokens = [4, 5, 6], [1, 2, 3]
    raw = thinker.forward_batched("thinker_decode", _engine_inputs(rids), input_embeds=_rows(tokens))
    out = _collect(thinker, raw, rids, _decode_inputs(3))

    # the worker drops a removed or stopped request from the per-rid outputs
    assert out.pop(5) is not None

    assert out.pop(5) is None
    assert out.per_rid_outputs[4]["new_token"][0].tolist() == [1]
    assert out.per_rid_outputs[6]["new_token"][0].tolist() == [3]


def _eager_forward(thinker, walk, rid, embeds, audio=True):
    """What the engine holds after the unbatched forward, the path of a step no capture fits."""
    inputs = _engine_inputs([rid])
    inputs.per_request_info = {rid: SimpleNamespace(step_metadata={"audio_output": audio})}
    return BatchedModelOutput.coerce({rid: thinker.forward(walk, inputs, input_embeds=embeds)})


def test_the_eager_decode_forward_gets_its_mask_from_the_same_place():
    thinker = _thinker()
    embeds = _rows([3])

    out = _collect(thinker, _eager_forward(thinker, "thinker_decode", 9, embeds), [9], _decode_inputs(1))

    row = out.per_rid_outputs[9]
    assert row["new_token"][0].tolist() == [3]
    assert torch.equal(row["thinker_states"][0], torch.cat([embeds, embeds + 100], dim=-1))
    assert torch.equal(row["thinker_mask"][0], DECODE_MASK)


def test_the_eager_prefill_forward_sends_its_first_token_embed_and_masks():
    thinker = _thinker()
    inputs = _prefill_inputs([2])

    out = _collect(thinker, _eager_forward(thinker, "prefill_text", 9, _rows([0, 4])), [9], inputs)

    row = out.per_rid_outputs[9]
    assert row["new_token"][0].tolist() == [4]
    assert row["thinker_states"][0].shape[0] == 2
    assert row["thinker_mask"][0] is inputs[0].tensor_inputs["masks_for_talker"]
    assert "first_token_embed" in row


def test_the_eager_forward_of_a_text_only_request_sends_the_talker_nothing():
    thinker = _thinker()

    raw = _eager_forward(thinker, "prefill_text", 9, _rows([4]), audio=False)
    out = _collect(thinker, raw, [9], _prefill_inputs([1]), audio={9: False})

    assert set(out.per_rid_outputs[9]) == {"new_token"}


class _Compiles:
    """Counts the frames dynamo compiles while it is open."""

    def __enter__(self):
        self.frames = []
        self._handle = register_bytecode_hook(lambda code, _: self.frames.append(code.co_name))
        return self.frames

    def __exit__(self, *_):
        self._handle.remove()


@pytest.mark.parametrize("walk", ["thinker_decode", "prefill_text"])
def test_new_request_ids_do_not_recompile_the_batched_forward(walk):
    thinker = _thinker()
    compiled = torch.compile(thinker.forward_batched, backend="eager")
    embeds = _rows([1, 2])

    with _Compiles() as frames:
        first = compiled(walk, _engine_inputs([-2, -3]), input_embeds=embeds)
    assert frames, "the forward was not compiled, so this test shows nothing"
    with _Compiles() as frames:
        second = compiled(walk, _engine_inputs([0, 1]), input_embeds=embeds)
    assert frames == []
    assert torch.equal(first.row_outputs["new_token"], second.row_outputs["new_token"])


def test_new_request_ids_do_not_recompile_the_eager_forward():
    thinker = _thinker()
    compiled = torch.compile(thinker.forward, backend="eager")
    embeds = _rows([1])

    def inputs(rid):
        out = _engine_inputs([rid])
        out.per_request_info = {rid: SimpleNamespace(step_metadata={"audio_output": True})}
        return out

    with _Compiles() as frames:
        compiled("thinker_decode", inputs(0), input_embeds=embeds)
    assert frames, "the forward was not compiled, so this test shows nothing"
    with _Compiles() as frames:
        compiled("thinker_decode", inputs(1), input_embeds=embeds)
    assert frames == []


def test_a_frame_that_reads_the_ids_does_recompile():
    """The guard these tests are about, so the two above can fail."""
    sampler = _Sampler()

    def forward(engine_inputs, logits):
        tokens = sampler.sample(logits)
        return {rid: tokens[i:i + 1] for i, rid in enumerate(engine_inputs.request_ids)}

    compiled = torch.compile(forward, backend="eager")
    logits = torch.zeros(2, VOCAB)

    compiled(_engine_inputs([-2, -3]), logits)
    with _Compiles() as frames:
        compiled(_engine_inputs([0, 1]), logits)
    assert frames
