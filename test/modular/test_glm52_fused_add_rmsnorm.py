"""GLM-5.2 with each all-reduce fused into the next add + RMSNorm: the layers carry
(partial, residual) and emit what the unfused layers emit."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_glm52_engine_cycle import (  # noqa: E402
    _cfg,
    _drive,
    _Driver,
    _force_cpu_flashinfer,  # noqa: F401 — autouse here too
    _fwd_info,
)
from test_glm52_moe import _RecordingGroup  # noqa: E402

from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM  # noqa: E402
from mstar.model.glm52.components.language_model import build_dense_mlp  # noqa: E402
from mstar.model.glm52.components.moe import Glm52SparseMoeBlock  # noqa: E402
from mstar.model.glm52.config import Glm52ModelConfig  # noqa: E402
from mstar.model.glm52.quantization import process_weights_after_loading  # noqa: E402
from mstar.model.glm52.submodules import Glm52LLMSubmodule  # noqa: E402


def _fused(cfg: Glm52ModelConfig) -> Glm52ModelConfig:
    cfg = copy.deepcopy(cfg)
    cfg.fused_add_rmsnorm = True
    return cfg


class _FusingGroup(_RecordingGroup):
    """Records each fused all-reduce + add + RMSNorm; one process, so the reduce is identity."""

    def __init__(self):
        super().__init__()
        self.fused = 0

    def allreduce_add_rmsnorm(self, x, residual, weight, eps, weight_bias=0.0):
        self.fused += 1
        r = x + residual
        v = r.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
        return (v * (weight.float() + weight_bias)).to(r.dtype), r


def _model(cfg, comm_group=None, seed=0) -> Glm52ForCausalLM:
    """test_glm52_engine_cycle's model, on a given group: same draws, same weights."""
    torch.manual_seed(seed)
    model = Glm52ForCausalLM(cfg, comm_group=comm_group)
    for name, p in model.named_parameters():
        if "norm" in name:
            p.data.normal_(1.0, 0.02)
        elif name.endswith("gate.weight") or "e_score_correction_bias" in name:
            p.data.normal_(0, 1.0)
        else:
            p.data.normal_(0, 0.05)
    process_weights_after_loading(model, torch.device("cpu"))
    return model.eval()


def _run(cfg, comm_group, prompts):
    """Each request's emitted tokens and every hidden state lm_head saw."""
    model = _model(cfg, comm_group)
    seen = []
    model.lm_head.register_forward_hook(lambda m, args, out: seen.append(args[0].clone()))
    tokens = {}
    for rid, prompt in prompts.items():
        sub = Glm52LLMSubmodule(model, cfg)
        tokens[rid] = _drive(_Driver(sub, cfg, [rid]), prompt, _fwd_info(rid, 10, True))
    return model, tokens, seen


@pytest.mark.parametrize("mla_absorb", [True, False], ids=["absorbed", "naive"])
@pytest.mark.parametrize("last_layer_rows", [False, True])
def test_fused_wiring_emits_what_the_unfused_does(mla_absorb, last_layer_rows):
    cfg = _cfg(mla_absorb)
    cfg.prefill_last_layer_rows = last_layer_rows
    prompts = {"a": torch.arange(5, dtype=torch.long) + 3,
               "b": torch.arange(3, dtype=torch.long) + 9}
    plain, plain_tokens, plain_seen = _run(cfg, None, prompts)
    fused, fused_tokens, fused_seen = _run(_fused(cfg), None, prompts)
    assert not plain.model.fused_add_rmsnorm and fused.model.fused_add_rmsnorm
    assert len(set(plain_tokens["a"].tolist())) > 3  # a real stream
    # one rank: the same adds and norms in the same order, so the same bits
    for rid in prompts:
        assert torch.equal(fused_tokens[rid], plain_tokens[rid]), rid
    assert len(fused_seen) == len(plain_seen)
    for got, want in zip(fused_seen, plain_seen, strict=True):
        assert torch.equal(got, want)


def test_fused_wiring_leaves_the_row_parallel_outputs_partial():
    cfg = Glm52ModelConfig.reduced()
    cfg.num_hidden_layers = cfg.first_k_dense_replace + 1
    model = Glm52ForCausalLM(_fused(cfg))
    kinds = set()
    for layer in model.model.layers:
        assert layer.fused_add_rmsnorm and not layer.self_attn.o_proj.reduce_results
        mlp = layer.mlp
        kinds.add(type(mlp))
        if isinstance(mlp, Glm52SparseMoeBlock):
            assert not mlp.reduce_results
        else:
            assert not mlp.down_proj.reduce_results
    assert len(kinds) == 2  # dense and MoE layers
    plain = Glm52ForCausalLM(cfg)
    assert all(layer.self_attn.o_proj.reduce_results for layer in plain.model.layers)


def test_moe_partial_is_what_the_fused_allreduce_block_reduces(monkeypatch):
    """reduce_results=False: no collective in the block, and the returned partial is the
    routed + shared sum a moe_fused_allreduce block hands its one all-reduce."""
    monkeypatch.delenv("MSTAR_GLM52_MOE_FUSED_ALLREDUCE", raising=False)
    torch.manual_seed(3)
    cfg = Glm52ModelConfig.reduced()
    cfg.moe_fused_allreduce = True
    g_reduce, g_partial = _RecordingGroup(), _RecordingGroup()
    reducing = Glm52SparseMoeBlock(cfg, comm_group=g_reduce)
    partial = Glm52SparseMoeBlock(cfg, comm_group=g_partial, reduce_results=False)
    assert not partial.shared_expert.down_proj.reduce_results
    for p in reducing.parameters():
        p.data.normal_(0, 0.05)
    partial.load_state_dict(reducing.state_dict())
    x = torch.randn(5, cfg.hidden_size) * 0.1
    reducing(x)
    out = partial(x)
    assert g_partial.reduced == [] and len(g_reduce.reduced) == 1
    assert torch.equal(out, g_reduce.reduced[0].view_as(out))


def test_dense_mlp_partial_skips_the_reduce():
    cfg = Glm52ModelConfig.reduced()
    group = _RecordingGroup()
    mlp = build_dense_mlp(cfg, comm_group=group, reduce_results=False)
    for p in mlp.parameters():
        p.data.normal_(0, 0.05)
    mlp(torch.randn(3, cfg.hidden_size))
    assert group.reduced == []


def test_norms_reduce_through_the_group_under_tp():
    cfg = _fused(_cfg(True))
    group = _FusingGroup()
    _, tokens, seen = _run(cfg, group, {"a": torch.arange(5, dtype=torch.long) + 3})
    # per forward: each layer's second norm, each later layer's first, and the final one
    assert group.fused == 2 * cfg.num_hidden_layers * len(seen) > 0
