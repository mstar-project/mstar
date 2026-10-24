"""Orpheus's LM head computes only the custom-token rows, and its stop check
runs batched off the step's host rows.

CPU-only: the sliced head equals the full head with the text rows masked and
hands the sampler full-width logits, so token ids stay in full-vocab space;
``check_stop_batched`` agrees with ``check_stop``.
"""
from types import SimpleNamespace

import torch
from torch import nn

from mstar.model.orpheus.config import ATTN, SAMPLER, OrpheusModelConfig
from mstar.model.orpheus.submodules import OrpheusLLMSubmodule
from mstar.model.submodule_base import HostRows

V, H, LO = 96, 16, 60


def _config() -> OrpheusModelConfig:
    return OrpheusModelConfig(vocab_size=V, custom_token_base_id=LO, stop_token_id=LO + 2)


class _FakeLM(nn.Module):
    def __init__(self, tp_size: int = 1):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(V, H)
        self.lm_head = nn.Linear(H, V, bias=False)
        self.lm_head.tp_size = tp_size

    def forward(self, emb, *, label):
        return emb


class _GreedySampler:
    def sample(self, request_ids, logits):
        self.logits = logits
        return logits.argmax(-1)


def _forward(tp_size: int = 1):
    torch.manual_seed(3)
    lm = _FakeLM(tp_size)
    sub = OrpheusLLMSubmodule(lm, _config())
    sampler = _GreedySampler()
    ids = torch.tensor([1, 7, LO + 4, V - 1])
    engine_inputs = SimpleNamespace(resources={SAMPLER: sampler, ATTN: None}, request_ids=list("abcd"))
    with torch.no_grad():
        out = sub.forward("decode", engine_inputs, text_inputs=ids)["new_token"]
        full = lm.lm_head(lm.model.embed_tokens(ids))
    return sub, sampler.logits, out, full


def test_real_config_ids_fall_in_the_custom_rows():
    cfg = OrpheusModelConfig()
    assert cfg.stop_token_id >= cfg.custom_token_base_id
    # every SNAC code position (7 per frame, 4096 codes each) after the 10 markers
    assert cfg.custom_token_base_id + 10 + cfg.tokens_per_frame * 4096 <= cfg.vocab_size


def test_head_logits_are_the_full_head_with_text_rows_masked():
    sub, logits, out, full = _forward()
    assert sub._audio_vocab_start == LO and logits.shape == (4, V)
    masked = full.clone()
    masked[:, :LO] = float("-inf")
    torch.testing.assert_close(logits, masked)
    assert torch.equal(out, masked.argmax(-1)) and (out >= LO).all()


def test_full_head_under_tensor_parallel():
    sub, logits, _, full = _forward(tp_size=2)
    assert sub._audio_vocab_start is None
    torch.testing.assert_close(logits, full)


def test_batched_stop_check_agrees_with_the_per_request_one():
    sub = OrpheusLLMSubmodule(_FakeLM(), _config())
    eos = LO + 2

    def info(ignore_eos: bool, done: int, max_tokens: int = 100):
        return SimpleNamespace(
            resource_configs={SAMPLER: SimpleNamespace(ignore_eos=ignore_eos)},
            dynamic_loop_iter_counts={"decode_loop": done}, max_tokens=max_tokens,
        )

    cases = {"eos": (eos, info(False, 3)), "ignored": (eos, info(True, 3)),
             "budget": (LO + 9, info(False, 9, max_tokens=10)), "running": (LO + 9, info(False, 3))}
    infos = {rid: i for rid, (_, i) in cases.items()}
    expected = {
        rid: stop for rid, (tok, i) in cases.items()
        if (stop := sub.check_stop(rid, i, {"new_token": [torch.tensor(tok)]}))
    }
    assert expected == {"eos": {"decode_loop"}, "budget": {"decode_loop"}}
    order = list(reversed(cases))
    rows = HostRows(tuple(order), {"new_token": torch.tensor([cases[r][0] for r in order] + [eos])})
    assert sub.check_stop_batched(list(cases), infos, rows, "decode") == expected
