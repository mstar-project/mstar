"""Higgs LLM decode loop: one token signal kept on the device and sent to the
client inline, and a batched stop check that agrees with the per-request one."""

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.state import Sampler
from mstar.engine.state.calls import SamplerHandle
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.higgs_audio.config import SAMPLER, HiggsAudioModelConfig
from mstar.model.higgs_audio.higgs_audio_model import HiggsAudioModel
from mstar.model.higgs_audio.submodules import HiggsAudioLLMSubmodule
from mstar.model.submodule_base import END_PARTITION, ARNodeInputs, HostRows


def _submodule(slot_masters=True):
    sub = HiggsAudioLLMSubmodule(torch.nn.Linear(1, 1), HiggsAudioModelConfig())
    sub.node_resources = {SAMPLER: SimpleNamespace(has_slot_masters=slot_masters)}
    return sub


def _info(walk="decode", decoded=0, max_tokens=8, ignore_eos=False):
    return SimpleNamespace(
        graph_walk=walk,
        dynamic_loop_iter_counts={"decode_loop": decoded},
        max_tokens=max_tokens,
        resource_configs={SAMPLER: SimpleNamespace(ignore_eos=ignore_eos)},
        step_metadata={},
    )


def test_decode_loop_routes_one_signal_to_itself_and_the_client():
    model = object.__new__(HiggsAudioModel)
    model.config = HiggsAudioModelConfig()
    outputs = model.get_graph_walk_graphs()["decode"].section.outputs
    assert {(edge.next_node, edge.name) for edge in outputs} == {
        ("LLM", "text_inputs"), (EMIT_TO_CLIENT, "text_inputs"),
    }


def test_decode_keeps_its_token_on_the_device_only_with_slot_masters():
    sub = _submodule()
    assert sub.device_loopback_signals("decode") == {"text_inputs"}
    assert sub.device_loopback_signals("prefill_audio") == frozenset()
    assert sub.inline_client_signals("decode") == {"text_inputs": "new_token"}
    assert sub.inline_client_signals("prefill_audio") == {}
    row = sub.prepare_inputs("decode", _info(), {}, has_loopback=True)
    assert row.input_ids is None and row.input_seq_len == 1
    assert sub.uniform_row_inputs("decode", has_loopback=True).input_seq_len == 1
    assert sub.uniform_row_inputs("decode", has_loopback=False) is None

    eager = _submodule(slot_masters=False)
    assert eager.device_loopback_signals("decode") == frozenset()
    assert eager.uniform_row_inputs("decode", has_loopback=True) is None


def test_batched_stop_check_agrees_with_the_per_request_one():
    sub = _submodule()
    eos = next(iter(sub.config.stop_token_ids))
    infos = {
        "a": _info(),
        "b": _info(),                      # stop token
        "c": _info(decoded=7),             # at its budget: the loop's cap ends it
        "d": _info(ignore_eos=True),       # stop token ignored
    }
    tokens = torch.tensor([[11], [eos], [11], [eos], [11]])  # a padded replay's extra row last
    want = {
        rid: stops for i, rid in enumerate(infos)
        if (stops := sub.check_stop(rid, infos[rid], {"new_token": [tokens[i]]}))
    }
    assert want == {"b": {"decode_loop"}}
    rows = HostRows(("a", "b", "c", "d"), {"new_token": tokens})
    assert sub.check_stop_batched(["d", "c", "b", "a"], infos, rows) == want
    # a prefill row's stop token ends the partition: there is no loop yet
    prefill = {rid: SimpleNamespace(**{**vars(info), "graph_walk": "prefill_audio"}) for rid, info in infos.items()}
    assert sub.check_stop("b", prefill["b"], {"new_token": [tokens[1]]}) == {END_PARTITION}
    assert sub.check_stop_batched(list(prefill), prefill, rows) == {"b": {END_PARTITION}}


class _SlotMasters:
    """A row's token is 100 plus its row, and the overlay is the sampler's own."""

    overwrite_rows = SamplerResource.overwrite_rows
    _row_stager = None

    def loopback_tokens(self, rids):
        return 100 + torch.arange(len(rids))


def _routed(token: int) -> ARNodeInputs:
    return ARNodeInputs(input_ids=torch.tensor([token]), input_seq_len=1)


LOOPED = ARNodeInputs(input_seq_len=1)


@pytest.mark.parametrize("inputs, rows", [
    ([_routed(7), LOOPED], 2),
    ([LOOPED, _routed(7), LOOPED, _routed(8)], 4),
    ([LOOPED, _routed(7), _routed(0), _routed(0)], 4),    # padding rows copied from a template that carries ids
    ([LOOPED, _routed(7)], 4),    # padding rows the slot keeps past the inputs
    ([LOOPED, ARNodeInputs(input_ids=torch.tensor([9]), input_seq_len=0)], 2),    # a row with nothing to run
], ids=["fresh-row-first", "mixed", "padded-with-ids", "padded-past-the-inputs", "empty-row"])
def test_a_decode_step_mixing_routed_and_loop_back_rows_gives_the_ids_loopback_inputs_gave(inputs, rows):
    sub = _submodule()
    rids = list(range(rows))
    want = sub.loopback_inputs(SimpleNamespace(resources={SAMPLER: _SlotMasters()}, request_ids=rids), inputs)
    sub.sampler = SamplerHandle(Sampler(SAMPLER, nodes={"LLM"}, vocab_size=8), _SlotMasters())
    sub.sampler.begin_step(SimpleNamespace(ctx=SimpleNamespace(padded_request_ids=rids)))
    got = sub.preprocess("decode", SimpleNamespace(request_ids=rids), inputs)
    assert torch.equal(got["input_ids"], want["input_ids"]), "the splice should build the rows the overlay wrote"
