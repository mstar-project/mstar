"""The MTP step's graph configs and acceptance log (CPU, reduced)."""
from __future__ import annotations

import sys
import types
from pathlib import Path

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
