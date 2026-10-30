"""Whisper's decoder node on the shared TokenStepNode: HF generate's
token suppression, applied to the logits the sampler gets; the batched stop
check, which only a decode step's token answers; and the walk and loop names
of Whisper's graph."""
from types import SimpleNamespace

import torch
from torch import nn

from mstar.model.submodule_base import HostRows
from mstar.model.whisper.config import PREFILL_WALK, WhisperModelConfig
from mstar.model.whisper.submodules import WhisperDecoderSubmodule


def _node():
    config = WhisperModelConfig(eos_token_id=3, suppress_tokens=[1, 5], begin_suppress_tokens=[2, 3])
    return WhisperDecoderSubmodule(nn.Identity(), config)


class _Decoder(nn.Module):
    """Zero logits over 8 tokens, whatever the input."""

    def embed(self, input_ids, position_ids):
        return torch.zeros(input_ids.shape[0], 4)

    def forward(self, input_embeds, *, label):
        return input_embeds

    def lm_head(self, hidden):
        return torch.zeros(hidden.shape[0], 8)


class _Sampler:
    def __init__(self):
        self.logits = []

    def sample(self, request_ids, logits):
        self.logits.append(logits.clone())
        return logits.argmax(-1)


def test_forward_suppresses_before_sampling():
    """A prefill and a decode step through ``forward``: the sampler sees the
    begin-suppressed tokens masked only for the first token. (The context is
    written in ``preprocess``, outside the captured prefill.)"""
    node, sampler = _node(), _Sampler()
    node.decoder = _Decoder()
    engine_inputs = SimpleNamespace(request_ids=["a"], resources={
        "attn": SimpleNamespace(select_last_hidden=lambda hidden: hidden[-1:]),
        "positions": SimpleNamespace(pos_ids=lambda label: None),
        "sampler": sampler,
    })
    node.forward("prefill", engine_inputs, input_ids=torch.tensor([9, 9, 9, 9]))
    node.forward("decode", engine_inputs, input_ids=torch.tensor([9]))
    first, later = sampler.logits
    assert first.shape == later.shape == (1, 8)
    assert torch.isinf(first[0, [1, 2, 3, 5]]).all() and not torch.isinf(first[0, [0, 4, 6, 7]]).any()
    assert torch.isinf(later[0, [1, 5]]).all() and not torch.isinf(later[0, [0, 2, 3, 4, 6, 7]]).any()


def test_names_match_whispers_graph():
    from mstar.model.whisper.whisper_model import WhisperModel

    model = object.__new__(WhisperModel)
    model.config = WhisperModelConfig()
    walks = model.get_graph_walk_graphs()
    node = _node()
    assert {PREFILL_WALK, node.DECODE_WALK} <= set(walks)
    assert walks[node.DECODE_WALK].name == node.DECODE_LOOP


def test_batched_stop_check_answers_decode_steps_only():
    """As the per-request check: EOS ends the loop on a decode step (a row at
    its token cap is the loop cap's to end), while a prefill walk's EOS runs
    outside the loop and ends nothing."""
    node = _node()
    infos = {
        rid: SimpleNamespace(graph_walk=walk, resource_configs={"sampler": SimpleNamespace(ignore_eos=False)},
                             dynamic_loop_iter_counts={"decode_loop": n}, max_tokens=10,
                             step_metadata={}, rid_handle=i)
        for i, (rid, walk, n) in enumerate((("eos", "decode", 0), ("limit", "decode", 9), ("go", "decode", 0),
                                            ("prefill_eos", "prefill", 0)))
    }
    rows = HostRows(request_ids=("go", "limit", "eos", "prefill_eos"),
                    buffers={"new_token": torch.tensor([[7], [7], [3], [3]])})
    assert node.check_stop_batched(["eos", "limit", "go", "prefill_eos"], infos, rows) == {
        "eos": {"decode_loop"},
    }
