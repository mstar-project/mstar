"""Orpheus's LM head computes only the custom-token rows, and its stop check
runs batched off the step's host rows.

CPU-only: the windowed head equals the full head with the text rows masked,
under any TP size, and hands the sampler full-width logits, so token ids stay
in full-vocab space; ``check_stop_batched`` agrees with ``check_stop``.
"""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mstar.model.components.distributed import ColumnParallelLinear
from mstar.model.orpheus.config import ATTN, SAMPLER, OrpheusModelConfig
from mstar.model.orpheus.submodules import OrpheusLLMSubmodule
from mstar.model.submodule_base import HostRows

V, H, LO = 96, 16, 60


def _config() -> OrpheusModelConfig:
    return OrpheusModelConfig(vocab_size=V, custom_token_base_id=LO, stop_token_id=LO + 2)


class _Group:
    def __init__(self, rank: int, world_size: int):
        self.rank, self.world_size = rank, world_size


def _head(weight: torch.Tensor, rank: int = 0, tp_size: int = 1) -> ColumnParallelLinear:
    """This rank's shard of the Orpheus head, its output not gathered."""
    head = ColumnParallelLinear(_Group(rank, tp_size), H, V, gather_output=False, output_start=LO)
    head.weight.weight_loader(head.weight, weight)
    return head


class _FakeLM(nn.Module):
    def __init__(self, rank: int = 0, tp_size: int = 1):
        super().__init__()
        torch.manual_seed(3)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(V, H)
        self.full_head = nn.Linear(H, V, bias=False)
        self.lm_head = _head(self.full_head.weight.data, rank, tp_size)

    def forward(self, emb, *, label):
        return emb


class _GreedySampler:
    def sample(self, request_ids, logits):
        self.logits = logits
        return logits.argmax(-1)


IDS = torch.tensor([1, 7, LO + 4, V - 1])


def _forward(rank: int = 0, tp_size: int = 1):
    lm = _FakeLM(rank, tp_size)
    sub = OrpheusLLMSubmodule(lm, _config())
    sampler = _GreedySampler()
    engine_inputs = SimpleNamespace(resources={SAMPLER: sampler, ATTN: None}, request_ids=list("abcd"))
    with torch.no_grad():
        out = sub.forward("decode", engine_inputs, text_inputs=IDS)["new_token"]
        full = lm.full_head(lm.model.embed_tokens(IDS))
    return sampler.logits, out, full


def test_real_config_ids_fall_in_the_custom_rows():
    cfg = OrpheusModelConfig()
    assert cfg.stop_token_id >= cfg.custom_token_base_id
    # every SNAC code position (7 per frame, 4096 codes each) after the 10 markers
    assert cfg.custom_token_base_id + 10 + cfg.tokens_per_frame * 4096 <= cfg.vocab_size


def test_head_logits_are_the_full_head_with_text_rows_masked():
    logits, out, full = _forward()
    assert logits.shape == (4, V)
    masked = full.clone()
    masked[:, :LO] = float("-inf")
    torch.testing.assert_close(logits, masked)
    assert torch.equal(out, masked.argmax(-1)) and (out >= LO).all()


@pytest.mark.parametrize("tp_size", [2, 4])
def test_tensor_parallel_shards_gather_to_the_single_rank_head(tp_size):
    """Each rank windows its own shard (a rank entirely below the window
    computes nothing), so the gathered logits are the TP1 ones."""
    single, _, _ = _forward()
    shards = [_forward(rank, tp_size)[0] for rank in range(tp_size)]
    torch.testing.assert_close(torch.cat(shards, dim=-1), single)


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
    # a row at its budget goes on: the loop's cap ends it
    assert expected == {"eos": {"decode_loop"}}
    order = list(reversed(cases))
    rows = HostRows(tuple(order), {"new_token": torch.tensor([cases[r][0] for r in order] + [eos])})
    assert sub.check_stop_batched(list(cases), infos, rows) == expected
