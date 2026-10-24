"""Higgs LLM decode loop: one token signal kept on the device and sent to the
client inline, and a batched stop check that agrees with the per-request one."""

from types import SimpleNamespace

import torch

from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.higgs_audio.config import SAMPLER, HiggsAudioModelConfig
from mstar.model.higgs_audio.higgs_audio_model import HiggsAudioModel
from mstar.model.higgs_audio.submodules import HiggsAudioLLMSubmodule
from mstar.model.submodule_base import HostRows


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
    assert sub.reads_device_loopback
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
        "c": _info(decoded=7),             # out of budget
        "d": _info(ignore_eos=True),       # stop token ignored
    }
    tokens = torch.tensor([[11], [eos], [11], [eos], [11]])  # a padded replay's extra row last
    want = {
        rid: stops for i, rid in enumerate(infos)
        if (stops := sub.check_stop(rid, infos[rid], {"new_token": [tokens[i]]}))
    }
    assert want == {"b": {"decode_loop"}, "c": {"decode_loop"}}
    rows = HostRows(("a", "b", "c", "d"), {"new_token": tokens})
    assert sub.check_stop_batched(["d", "c", "b", "a"], infos, rows, step_walk="decode") == want
    assert sub.check_stop_batched(list(infos), infos, rows, step_walk=None) == want
    # only decode rows end the loop
    assert sub.check_stop_batched(list(infos), infos, rows, step_walk="prefill_audio") == {}
    prefill = {**infos, "b": _info(walk="prefill_audio")}
    assert sub.check_stop_batched(list(prefill), prefill, rows, step_walk=None) == {"c": {"decode_loop"}}
