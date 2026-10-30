"""The MTP step's graph configs and acceptance log (CPU, reduced)."""
from __future__ import annotations

import logging
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _cpu_rmsnorm(x, weight, eps=1e-6):
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * weight.float()).to(x.dtype)


def _cpu_flashinfer() -> types.ModuleType:
    fi = types.ModuleType("flashinfer")
    fi.norm = types.SimpleNamespace(rmsnorm=_cpu_rmsnorm)
    return fi


@pytest.fixture(autouse=True)
def _force_cpu_flashinfer(monkeypatch):
    monkeypatch.setitem(sys.modules, "flashinfer", _cpu_flashinfer())


from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig  # noqa: E402
from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM  # noqa: E402
from mstar.model.glm52.config import Glm52ModelConfig  # noqa: E402
from mstar.model.glm52.submodules import Glm52LLMSubmodule  # noqa: E402

CPU = torch.device("cpu")


def _mtp_cfg(k: int) -> Glm52ModelConfig:
    cfg = Glm52ModelConfig.reduced()
    cfg.num_hidden_layers = 4  # the MTP position lands FULL (4 = offset-1 + freq)
    cfg.mtp_num_draft_tokens = k
    cfg.mla_absorb = True
    return cfg


@pytest.mark.parametrize("k", [1, 3])
def test_mtp_decode_captures_one_graph_over_the_block(k):
    """Under MTP the whole verify step is one decode capture of k+1 rows per request (one
    id in), next to the same prefill captures as k=0, and no piecewise regions."""
    cfg = _mtp_cfg(k)
    sub = Glm52LLMSubmodule(Glm52ForCausalLM(cfg), cfg)
    configs = sub.get_accelerator_graph_configs(CPU)
    (decode,) = [c for c in configs if isinstance(c, BatchedAcceleratorGraphConfig)]
    assert decode.capture_graph_walk == "decode"
    assert decode.single_request_inputs.input_seq_len == k + 1
    assert decode.single_request_inputs.input_ids.numel() == 1
    assert decode.get_total_tokens(4) == [4 * (k + 1)]
    assert decode.capture_batch_sizes == Glm52LLMSubmodule.MTP_CAPTURE_BATCH_SIZES
    cfg.mtp_num_draft_tokens = 0
    plain = Glm52LLMSubmodule(Glm52ForCausalLM(cfg), cfg).get_accelerator_graph_configs(CPU)
    assert len(plain) == len(configs)
    assert sub.get_piecewise_accelerator_graph_configs(CPU, torch.bfloat16, tp_world_size=1) == {}


def test_mtp_acceptance_log_per_position(caplog):
    """The 512-step acceptance line must carry the conditional per-position profile (the
    datum that separates "first draft mediocre" from "chained drafts collapse").
    """
    k = 3

    def _ns() -> SimpleNamespace:
        return SimpleNamespace(
            config=SimpleNamespace(mtp_num_draft_tokens=k),
            _MTP_STAT_LOG_EVERY=Glm52LLMSubmodule._MTP_STAT_LOG_EVERY,
            _mtp_stat_steps=512,
            _mtp_stat_logged=0,
            # 512 steps halving at each position: reached = [512, 256, 128, 64].
            _mtp_stat_acc_hist=[256, 128, 64, 64],
            _mtp_stat_emitted=256 * 1 + 128 * 2 + 64 * 3 + 64 * 4,
        )

    ns = _ns()
    with caplog.at_level(logging.INFO, logger="mstar.model.glm52.submodules"):
        Glm52LLMSubmodule._maybe_log_mtp_acceptance(ns)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("emitted/step" in m for m in msgs)
    (pos_line,) = [m for m in msgs if "by position" in m]
    assert "[256, 128, 64, 64]" in pos_line
    assert "0.50 0.50 0.50" in pos_line
    assert ns._mtp_stat_logged == 512
    # below the threshold nothing is logged
    caplog.clear()
    quiet = _ns()
    quiet._mtp_stat_steps = 100
    with caplog.at_level(logging.INFO, logger="mstar.model.glm52.submodules"):
        Glm52LLMSubmodule._maybe_log_mtp_acceptance(quiet)
    assert not caplog.records


class _ChainMTP:
    """A stand-in MTP layer: a pass returns (head input, layer output) as two distinct
    functions of the stream it was given, and a block-row pass records that stream."""

    def __init__(self):
        self.prev_hidden: list[torch.Tensor] = []

    def forward_block_row(self, embeds, prev_hidden, position_ids, row, block, rows):
        self.prev_hidden.append(prev_hidden.clone())
        out = prev_hidden + embeds + 1.0
        return out * 2.0, out  # (shared_head-normed, under the norm)

    def __call__(self, embeds, paired, position_ids):  # the seed pass over the block
        out = paired + embeds
        return out * 2.0, out


def test_the_draft_chain_and_the_seed_carry_the_post_norm_stream():
    """A draft's MTP pass reads the previous pass's shared_head-normed output, the stream
    the head reads, as d1's reads the trunk's post-final-norm state; the seed left for the
    next step's first chained draft is that stream too. Chaining the output under the norm
    cut GLM-5.2's second and third drafts' acceptance on 8xH200."""
    k, n, hidden, vocab = 3, 2, 8, 16
    block = k + 1
    mtp = _ChainMTP()
    embed = lambda ids: ids.float()[:, None].expand(-1, hidden)  # noqa: E731
    ns = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(embed_tokens=embed), mtp=mtp),
                         mtp_k=k, mtp_block=block)
    seed_h = torch.randn(n, hidden)
    ns._mtp_read_seed = lambda slots: (seed_h, torch.tensor([3, 5]))
    ns._draft_tokens = lambda h_head, prev: h_head.sum(-1).abs().long() % vocab
    ns._hidden = lambda ids, pos, *dsa: torch.randn(ids.shape[0], hidden)
    ns._dsa_use = lambda *dsa: nullcontext()
    ns.lm_head = lambda h: torch.randn(h.shape[0], vocab)
    ns._broadcast = lambda t: t
    ns._stage_verdict = lambda *a: None
    ns._mtp_seed = lambda slots, h_head, prev: None
    Glm52LLMSubmodule._mtp_decode(ns, SimpleNamespace(request_ids=["r0", "r1"]), torch.tensor([1, 2]),
                                  torch.arange(n * block), torch.tensor([1, 2]), torch.tensor([7]))
    # k - 1 chained passes: the first reads the seed, the second the first pass's head input
    assert len(mtp.prev_hidden) == k - 1
    assert torch.equal(mtp.prev_hidden[0], seed_h)
    assert torch.equal(mtp.prev_hidden[1], (seed_h + embed(torch.tensor([3, 5])) + 1.0) * 2.0)

    store = SimpleNamespace(_mtp_seed_hidden=torch.zeros(4, hidden), _mtp_seed_draft=torch.zeros(4, dtype=torch.long),
                            _draft_tokens=ns._draft_tokens)
    slots, h_head = torch.tensor([1, 2]), torch.randn(n, hidden)
    Glm52LLMSubmodule._mtp_seed(store, slots, h_head, torch.tensor([0, 0]))
    assert torch.equal(store._mtp_seed_hidden[slots], h_head)
