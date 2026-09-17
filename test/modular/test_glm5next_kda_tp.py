"""KDA head-sharding across TP, simulated on CPU."""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.model.glm5_next.components.attention import Glm5NextKdaAttention
from mstar.model.glm5_next.config import Glm5NextModelConfig
from mstar.model.glm5_next.kda import Glm5NextKdaConfig, Glm5NextLinearAttention
from mstar.model.glm5_next.kda_state import CONV, RECURRENT, kda_slot_state_config
from mstar.model.glm5_next.weight_loader import build_glm5_next_read_plan

TP = 2


class _FakeGroup:
    """A comm group whose all-reduce is the identity: the test sums the
    ranks' partials itself, which is what NCCL would do."""

    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size

    def all_reduce(self, x):
        return x


def _cfg() -> Glm5NextKdaConfig:
    return Glm5NextKdaConfig(hidden_size=32, linear_num_heads=4, linear_head_dim=8)


def _full_layer(seed: int) -> Glm5NextLinearAttention:
    torch.manual_seed(seed)
    layer = Glm5NextLinearAttention(_cfg(), dtype=torch.float32)
    for p in layer.parameters():
        torch.nn.init.normal_(p, std=0.3)
    return layer


# checkpoint-shaped tensors of the full layer, keyed like the loader sees
# them after remapping (conv split back into its q|k|v thirds)
def _checkpoint(full: Glm5NextLinearAttention) -> dict[str, tuple[torch.Tensor, str | None]]:
    q = full.qkv_dim
    w = full.conv1d.weight.detach()
    return {
        "q_proj.weight": (full.q_proj.weight.detach(), None),
        "k_proj.weight": (full.k_proj.weight.detach(), None),
        "v_proj.weight": (full.v_proj.weight.detach(), None),
        "forget_gate.f_a_proj.weight": (full.forget_gate.f_a_proj.weight.detach(), None),
        "forget_gate.f_b_proj.weight": (full.forget_gate.f_b_proj.weight.detach(), None),
        "forget_gate.dt_bias": (full.forget_gate.dt_bias.detach(), None),
        "forget_gate.A_log": (full.forget_gate.A_log.detach(), None),
        "b_proj.weight": (full.b_proj.weight.detach(), None),
        "g_a_proj.weight": (full.g_a_proj.weight.detach(), None),
        "g_b_proj.weight": (full.g_b_proj.weight.detach(), None),
        "o_norm.weight": (full.o_norm.weight.detach(), None),
        "o_proj.weight": (full.o_proj.weight.detach(), None),
        "conv1d.weight/q": (w[0:q], "q"),
        "conv1d.weight/k": (w[q:2 * q], "k"),
        "conv1d.weight/v": (w[2 * q:3 * q], "v"),
    }


def _slice_for_rank(name: str, tensor: torch.Tensor, rank: int, full: Glm5NextLinearAttention) -> torch.Tensor:
    """What the read plan hands rank ``rank``: the head block of a sharded
    tensor, the whole tensor otherwise."""
    qkv = full.qkv_dim // TP
    heads = full.num_heads // TP
    base = name.split("/", maxsplit=1)[0]
    if base in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "forget_gate.f_b_proj.weight",
                "forget_gate.dt_bias", "g_b_proj.weight", "conv1d.weight"):
        return tensor[rank * qkv:(rank + 1) * qkv]
    if base in ("forget_gate.A_log", "b_proj.weight"):
        return tensor[rank * heads:(rank + 1) * heads]
    if base == "o_proj.weight":
        return tensor[:, rank * qkv:(rank + 1) * qkv]
    return tensor


def _load(layer: Glm5NextKdaAttention, ckpt, rank: int, full, presliced: bool) -> None:
    params = dict(layer.named_parameters())
    for name, (tensor, shard_id) in ckpt.items():
        base = name.split("/")[0]
        param = params[base]
        if presliced:
            tensor = _slice_for_rank(name, tensor, rank, full)
        loader = getattr(param, "weight_loader", None)
        if loader is not None:
            loader(param, tensor, shard_id) if shard_id is not None else loader(param, tensor)
        else:
            param.data.copy_(tensor)


def _ranks(full, presliced: bool) -> list[Glm5NextKdaAttention]:
    ranks = []
    for r in range(TP):
        layer = Glm5NextKdaAttention(_cfg(), comm_group=_FakeGroup(r, TP), dtype=torch.float32)
        _load(layer, _checkpoint(full), r, full, presliced)
        ranks.append(layer)
    return ranks


@pytest.mark.parametrize("presliced", [False, True])
@torch.no_grad()
def test_sharded_ranks_reproduce_the_full_layer(presliced):
    full = _full_layer(0)
    ranks = _ranks(full, presliced)
    assert all(r.num_heads == full.num_heads // TP for r in ranks)
    assert all(r.qkv_dim == full.qkv_dim // TP for r in ranks)

    torch.manual_seed(1)
    x = torch.randn(2, 11, full.hidden_size)
    out_full, s_full, c_full = full.prefill(x)
    parts = [r.prefill(x) for r in ranks]
    out_tp = sum(p[0] for p in parts)
    torch.testing.assert_close(out_tp, out_full, atol=1e-5, rtol=1e-5)
    heads = full.num_heads // TP
    for r, (_, s_r, c_r) in enumerate(parts):
        torch.testing.assert_close(s_r, s_full[:, r * heads:(r + 1) * heads], atol=1e-6, rtol=1e-6)
        # conv channels are q|k|v blocks of this rank's heads
        q = full.qkv_dim
        ql = q // TP
        expect = torch.cat(
            [c_full[:, b * q + r * ql: b * q + (r + 1) * ql] for b in range(3)], dim=1
        )
        torch.testing.assert_close(c_r, expect)

    # one decode step on the states the prefill left, in place
    tok = torch.randn(2, 1, full.hidden_size)
    y_full = full.decode_step(tok, s_full, c_full)
    y_tp = sum(r.decode_step(tok, s_r, c_r) for r, (_, s_r, c_r) in zip(ranks, parts, strict=True))
    torch.testing.assert_close(y_tp, y_full, atol=1e-5, rtol=1e-5)


def test_loaders_refuse_a_wrong_width():
    full = _full_layer(2)
    layer = Glm5NextKdaAttention(_cfg(), comm_group=_FakeGroup(0, TP), dtype=torch.float32)
    bad = torch.zeros(full.qkv_dim + 1, full.hidden_size)
    with pytest.raises(ValueError, match="rows"):
        layer.q_proj.weight.weight_loader(layer.q_proj.weight, bad)
    with pytest.raises(ValueError, match="cols"):
        layer.o_proj.weight.weight_loader(layer.o_proj.weight, torch.zeros(full.hidden_size, 3))
    with pytest.raises(ValueError, match="expected"):
        layer.conv1d.weight.weight_loader(
            layer.conv1d.weight, torch.zeros(5, 1, full.conv_kernel_size), "q",
        )


def test_slot_state_config_shards_the_same_axes():
    cfg = Glm5NextModelConfig.reduced()
    sc = kda_slot_state_config(cfg, max_slots=4, conv_dtype=torch.float32)
    full_rec = sc.tensors[RECURRENT].shape
    full_conv = sc.tensors[CONV].shape
    sc.shard(TP)
    assert sc.tensors[RECURRENT].shape == (full_rec[0], full_rec[1] // TP, full_rec[2], full_rec[3])
    assert sc.tensors[CONV].shape == (full_conv[0], full_conv[1] // TP, full_conv[2])
    layer = Glm5NextKdaAttention(cfg, comm_group=_FakeGroup(0, TP), dtype=torch.float32)
    assert sc.tensors[CONV].shape[1] == layer.conv_dim
    assert sc.tensors[RECURRENT].shape[1] == layer.num_heads


def test_read_plan_slices_the_kda_tensors():
    cfg = Glm5NextModelConfig.reduced()
    kda_layer = cfg.kda_layer_indices[0]
    mla_layer = cfg.full_attn_layer_indices[0]
    pre = "model.language_model.layers"
    keys = [
        f"{pre}.{kda_layer}.self_attn.q_proj.weight",
        f"{pre}.{kda_layer}.self_attn.q_conv1d.weight",
        f"{pre}.{kda_layer}.self_attn.dt_bias",
        f"{pre}.{kda_layer}.self_attn.A_log",
        f"{pre}.{kda_layer}.self_attn.b_proj.weight",
        f"{pre}.{kda_layer}.self_attn.o_proj.weight",
        f"{pre}.{kda_layer}.self_attn.f_a_proj.weight",
        f"{pre}.{kda_layer}.self_attn.o_norm.weight",
        f"{pre}.{mla_layer}.self_attn.o_proj.weight",
    ]
    qkv = cfg.linear_qkv_dim // TP
    heads = cfg.linear_num_heads // TP
    _, specs = build_glm5_next_read_plan(keys, cfg, tp_rank=1, tp_size=TP)
    assert specs[keys[0]] == (0, qkv, 2 * qkv)
    assert specs[keys[1]] == (0, qkv, 2 * qkv)
    assert specs[keys[2]] == (0, qkv, 2 * qkv)
    assert specs[keys[3]] == (0, heads, 2 * heads)
    assert specs[keys[4]] == (0, heads, 2 * heads)
    assert specs[keys[5]] == (1, qkv, 2 * qkv)
    # replicated tensors and the MLA layer's o_proj are not sliced here
    assert keys[6] not in specs and keys[7] not in specs and keys[8] not in specs
    _, specs1 = build_glm5_next_read_plan(keys, cfg, tp_rank=0, tp_size=1)
    assert specs1 == {}
