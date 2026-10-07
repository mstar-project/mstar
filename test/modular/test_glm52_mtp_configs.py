"""The MTP step's graph configs and acceptance log (CPU, reduced)."""
from __future__ import annotations

import logging
import sys
import types
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


try:
    import flashinfer  # noqa: F401  (the real one stays for later test files)
except ImportError:
    sys.modules["flashinfer"] = _cpu_flashinfer()


@pytest.fixture(autouse=True)
def _force_cpu_flashinfer(monkeypatch):
    monkeypatch.setitem(sys.modules, "flashinfer", _cpu_flashinfer())


from mstar.engine.cuda_graph_config import BatchedCudaGraphConfig  # noqa: E402
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
    configs = sub.get_cuda_graph_configs(CPU)
    (decode,) = [c for c in configs if isinstance(c, BatchedCudaGraphConfig)]
    assert decode.capture_graph_walk == "decode"
    assert decode.single_request_inputs.input_seq_len == k + 1
    assert decode.single_request_inputs.input_ids.numel() == 1
    assert decode.get_total_tokens(4) == [4 * (k + 1)]
    assert decode.capture_batch_sizes == Glm52LLMSubmodule.MTP_CAPTURE_BATCH_SIZES
    cfg.mtp_num_draft_tokens = 0
    plain = Glm52LLMSubmodule(Glm52ForCausalLM(cfg), cfg).get_cuda_graph_configs(CPU)
    assert len(plain) == len(configs)
    assert sub.get_piecewise_cuda_graph_configs(CPU, torch.bfloat16, tp_world_size=1) == {}


def test_mtp_acceptance_log_per_position(caplog):
    """The 512-step acceptance line must carry the conditional per-position profile (the
    datum that separates "first draft mediocre" from "chained drafts collapse").
    """
    k = 3

    def _ns(pair_postnorm: bool) -> SimpleNamespace:
        return SimpleNamespace(
            config=SimpleNamespace(mtp_num_draft_tokens=k),
            _MTP_STAT_LOG_EVERY=Glm52LLMSubmodule._MTP_STAT_LOG_EVERY,
            _mtp_stat_steps=512,
            _mtp_stat_logged=0,
            # 512 steps halving at each position: reached = [512, 256, 128, 64].
            _mtp_stat_acc_hist=[256, 128, 64, 64],
            _mtp_stat_emitted=256 * 1 + 128 * 2 + 64 * 3 + 64 * 4,
            _mtp_pair_postnorm=pair_postnorm,
        )

    ns = _ns(False)
    with caplog.at_level(logging.INFO, logger="mstar.model.glm52.submodules"):
        Glm52LLMSubmodule._maybe_log_mtp_acceptance(ns)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("emitted/step" in m for m in msgs)
    (pos_line,) = [m for m in msgs if "by position" in m]
    assert "[256, 128, 64, 64]" in pos_line
    assert "0.50 0.50 0.50" in pos_line
    assert ns._mtp_stat_logged == 512
    # The line must name which trunk-pairing mode produced it: a profile whose
    # mode you have to infer from the launch environment is one you cannot
    # trust afterwards, and a mislabelled one inverts the comparison silently.
    assert "pre-final-norm" in pos_line and "POST" not in pos_line

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="mstar.model.glm52.submodules"):
        Glm52LLMSubmodule._maybe_log_mtp_acceptance(_ns(True))
    (post_line,) = [
        r.getMessage() for r in caplog.records if "by position" in r.getMessage()
    ]
    assert "POST-final-norm" in post_line
    # below the threshold nothing is logged
    caplog.clear()
    quiet = _ns(True)
    quiet._mtp_stat_steps = 100
    with caplog.at_level(logging.INFO, logger="mstar.model.glm52.submodules"):
        Glm52LLMSubmodule._maybe_log_mtp_acceptance(quiet)
    assert not caplog.records
