"""Whisper's decoder node on the shared encoder-decoder base: HF generate's
token suppression, applied to the logits the sampler gets; the batch caps and
captures; the stop check; and the walk and loop names of Whisper's graph."""
from types import SimpleNamespace

import torch
from torch import nn

from mstar.model.submodule_base import HostRows
from mstar.model.whisper.submodules import WhisperDecoderSubmodule


def _node():
    config = SimpleNamespace(eos_token_id=3, suppress_tokens=[1, 5], begin_suppress_tokens=[2, 3])
    return WhisperDecoderSubmodule(nn.Identity(), config)


def test_suppresses_like_hf_generate():
    """The suppressed set always; the begin-suppressed set (blank, EOS) only
    for the first token, which prefill samples."""
    node = _node()
    first = node.process_logits(torch.zeros(2, 8), "prefill")
    later = node.process_logits(torch.zeros(2, 8), "decode")
    assert torch.isinf(first[:, [1, 2, 3, 5]]).all() and not torch.isinf(first[:, [0, 4, 6, 7]]).any()
    assert torch.isinf(later[:, [1, 5]]).all() and not torch.isinf(later[:, [0, 2, 3, 4, 6, 7]]).any()


class _Decoder(nn.Module):
    """Zero logits over 8 tokens, whatever the input; notes the context write."""

    def __init__(self):
        super().__init__()
        self.wrote_context = False

    def write_cross_kv(self, encoder_states):
        self.wrote_context = True

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
    """A prefill and a decode step through ``forward``: the context written
    once, and the sampler sees the begin-suppressed tokens masked only for the
    first token."""
    node, sampler = _node(), _Sampler()
    node.decoder = _Decoder()
    engine_inputs = SimpleNamespace(request_ids=["a"], resources={
        "attn": SimpleNamespace(select_last_hidden=lambda hidden: hidden[-1:]),
        "positions": SimpleNamespace(pos_ids=lambda label: None),
        "sampler": sampler,
    })
    node.forward("prefill", engine_inputs, input_ids=torch.tensor([9, 9, 9, 9]), encoder_states=torch.zeros(5, 4))
    assert node.decoder.wrote_context
    node.decoder.wrote_context = False
    node.forward("decode", engine_inputs, input_ids=torch.tensor([9]))
    assert not node.decoder.wrote_context
    first, later = sampler.logits
    assert first.shape == later.shape == (1, 8)
    assert torch.isinf(first[0, [1, 2, 3, 5]]).all() and not torch.isinf(first[0, [0, 4, 6, 7]]).any()
    assert torch.isinf(later[0, [1, 5]]).all() and not torch.isinf(later[0, [0, 2, 3, 4, 6, 7]]).any()


def test_names_match_whispers_graph():
    from mstar.model.whisper.config import WhisperModelConfig
    from mstar.model.whisper.whisper_model import WhisperModel

    model = object.__new__(WhisperModel)
    model.config = WhisperModelConfig()
    walks = model.get_graph_walk_graphs()
    node = _node()
    assert set(walks) == {node.PREFILL_WALK, node.DECODE_WALK}
    assert walks[node.DECODE_WALK].name == node.DECODE_LOOP


def test_caps_captures_and_stops():
    node = _node()
    assert node.max_batch_size("prefill") == 16 and node.max_batch_size("decode") is None
    (capture,) = node.get_accelerator_graph_configs(torch.device("cpu"))
    assert capture.capture_graph_walk == "decode" and capture.capture_batch_sizes == [1, 2, 4, 8, 16]
    infos = {
        rid: SimpleNamespace(resource_configs={"sampler": SimpleNamespace(ignore_eos=False)},
                             dynamic_loop_iter_counts={"decode_loop": n}, max_tokens=10)
        for rid, n in (("eos", 0), ("limit", 9), ("go", 0))
    }
    rows = HostRows(request_ids=("go", "limit", "eos"), buffers={"new_token": torch.tensor([[7], [7], [3]])})
    # step_walk=None: the engine's call for a step that mixes walks
    assert node.check_stop_batched(["eos", "limit", "go"], infos, rows, step_walk=None) == {
        "eos": {"decode_loop"}, "limit": {"decode_loop"},
    }
    assert node.check_stop("go", infos["go"], {"new_token": [torch.tensor([7])]}) == set()
