"""The TP fast read path: sliced safetensors reads + the read plan.

The generic loader has every rank read the full checkpoint and keep its
slice (8x the bytes at TP8). The fast path excludes never-loaded keys and
reads only this rank's shard of routed-expert tensors. These tests pin:
the iterator's get_slice path, the plan's exclusions and specs, the
shape-driven expert loaders, and bit-exact end-to-end parity between the
fast path and the generic driver.
"""
import json

import torch
from safetensors.torch import save_file

from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM
from mstar.model.glm52.components.moe import (
    _down_fp8_loader,
    _gate_up_fp8_loader,
)
from mstar.model.glm52.config import Glm52ModelConfig
from mstar.model.glm52.weight_loader import build_glm52_read_plan
from mstar.model.loader.iterators import iter_safetensors_shards

BLOCK = (16, 16)


def _write_sharded(tmp_path, tensors):
    save_file(tensors, str(tmp_path / "model-00001-of-00001.safetensors"))
    index = {"weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))


def test_iterator_slice_spec_reads_exact_shards(tmp_path):
    torch.manual_seed(0)
    a = torch.randn(8, 6)
    b = torch.randn(6, 8)
    c = torch.randn(4)
    _write_sharded(tmp_path, {"a": a, "b": b, "c": c})

    specs = {"a": (0, 2, 5), "b": (1, 4, 8)}
    out = dict(iter_safetensors_shards(tmp_path, keys={"a", "b", "c"},
                                       slice_spec=specs.get))
    assert torch.equal(out["a"], a[2:5])
    assert torch.equal(out["b"], b[:, 4:8])
    assert torch.equal(out["c"], c)  # no spec -> full read


def test_read_plan_excludes_and_slices():
    cfg = Glm52ModelConfig.reduced_fp8(block=BLOCK)  # moe_inter 64, 2 layers
    keys = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.indexer.wk.weight",       # layer 0 FULL
        "model.layers.1.self_attn.indexer.wk.weight",       # layer 1 SHARED
        "model.layers.1.mlp.experts.2.gate_proj.weight",
        "model.layers.1.mlp.experts.2.gate_proj.weight_scale_inv",
        "model.layers.1.mlp.experts.2.down_proj.weight",
        "model.layers.1.mlp.experts.2.down_proj.weight_scale_inv",
        "model.layers.1.mlp.shared_experts.gate_proj.weight",
        "model.layers.2.enorm.weight",                      # MTP layer
        "model.layers.2.mlp.experts.0.up_proj.weight",      # MTP expert
    ]
    plan_keys, specs = build_glm52_read_plan(keys, cfg, tp_rank=1, tp_size=2)

    assert "model.layers.2.enorm.weight" not in plan_keys
    assert "model.layers.2.mlp.experts.0.up_proj.weight" not in plan_keys
    assert "model.layers.1.self_attn.indexer.wk.weight" not in plan_keys  # SHARED
    assert "model.layers.0.self_attn.indexer.wk.weight" in plan_keys      # FULL
    assert "model.layers.1.mlp.shared_experts.gate_proj.weight" in plan_keys
    assert "model.layers.1.mlp.shared_experts.gate_proj.weight" not in specs

    shard = cfg.moe_intermediate_size // 2  # 32
    assert specs["model.layers.1.mlp.experts.2.gate_proj.weight"] == (0, 32, 64)
    srows = shard // BLOCK[0]  # 2
    assert specs["model.layers.1.mlp.experts.2.gate_proj.weight_scale_inv"] == (
        0, srows, 2 * srows)
    assert specs["model.layers.1.mlp.experts.2.down_proj.weight"] == (1, 32, 64)
    scols = shard // BLOCK[1]
    assert specs["model.layers.1.mlp.experts.2.down_proj.weight_scale_inv"] == (
        1, scols, 2 * scols)


def test_expert_loaders_accept_full_and_presliced():
    full_inter, tp = 64, 2
    shard = full_inter // tp
    hidden = 32
    param = torch.nn.Parameter(
        torch.zeros(4, 2 * shard, hidden, dtype=torch.uint8), requires_grad=False)
    full = torch.arange(full_inter * hidden, dtype=torch.uint8).view(full_inter, hidden)

    _gate_up_fp8_loader(1, tp, full_inter, 1, param, full, "gate:3")
    from_full = param.data[3, :shard].clone()
    param.data.zero_()
    _gate_up_fp8_loader(1, tp, full_inter, 1, param, full[shard:], "gate:3")
    assert torch.equal(param.data[3, :shard], from_full)
    assert torch.equal(from_full, full[shard:])

    try:
        _gate_up_fp8_loader(1, tp, full_inter, 1, param, full[:10], "gate:3")
        raise AssertionError("wrong-shape tensor must be rejected")
    except ValueError:
        pass

    dparam = torch.nn.Parameter(
        torch.zeros(4, hidden, shard, dtype=torch.uint8), requires_grad=False)
    dfull = torch.arange(hidden * full_inter, dtype=torch.uint8).view(hidden, full_inter)
    _down_fp8_loader(1, tp, full_inter, 1, dparam, dfull, "down:2")
    d_from_full = dparam.data[2].clone()
    dparam.data.zero_()
    _down_fp8_loader(1, tp, full_inter, 1, dparam, dfull[:, shard:], "down:2")
    assert torch.equal(dparam.data[2], d_from_full)


def _fabricate_checkpoint_files(tmp_path, cfg, include_mtp=False):
    """On-disk fp8 checkpoint reusing the in-memory fabricator's layout."""
    from test.modular.test_glm52_moe import _fabricate_checkpoint

    state, refs = _fabricate_checkpoint(cfg, include_mtp=include_mtp)
    tensors = {}
    for name, tensor in state:
        tensors[name] = tensor.contiguous()
    _write_sharded(tmp_path, tensors)
    return refs


def _mtp_fp8_config():
    """reduced_fp8 with the MTP position landing FULL: 4 trunk layers put
    layer 4 on the IndexShare grid (offset-1 + freq, the same geometry as
    the real 78 = 2 + 19·4), and drafting on builds the mtp submodule."""
    cfg = Glm52ModelConfig.reduced_fp8(block=BLOCK)
    cfg.num_hidden_layers = 4
    cfg.mtp_num_draft_tokens = 2
    return cfg


def test_read_plan_includes_mtp_when_enabled():
    """The 2026-08-09 0.00-acceptance bug: the plan dropped every layer-78
    key regardless of drafting (1569/118629 keys on the real checkpoint),
    so the MTP module served ``to_empty`` memory. ``load_mtp=True`` must
    read the MTP layer like trunk: glue + decoder + FULL indexer, with
    its routed experts sliced by the same specs."""
    cfg = _mtp_fp8_config()
    keys = [
        "model.layers.3.self_attn.q_a_proj.weight",
        "model.layers.4.enorm.weight",
        "model.layers.4.eh_proj.weight",
        "model.layers.4.self_attn.indexer.wk.weight",   # layer 4 FULL
        "model.layers.4.mlp.experts.0.up_proj.weight",
        "model.layers.4.mlp.experts.0.up_proj.weight_scale_inv",
    ]
    # Default stays M1 flag-off behavior: the MTP layer is never read.
    plan_keys, _ = build_glm52_read_plan(keys, cfg, tp_rank=1, tp_size=2)
    assert "model.layers.3.self_attn.q_a_proj.weight" in plan_keys
    assert not any(".layers.4." in k for k in plan_keys)

    plan_keys, specs = build_glm52_read_plan(
        keys, cfg, tp_rank=1, tp_size=2, load_mtp=True)
    assert "model.layers.4.enorm.weight" in plan_keys
    assert "model.layers.4.eh_proj.weight" in plan_keys
    assert "model.layers.4.self_attn.indexer.wk.weight" in plan_keys
    shard = cfg.moe_intermediate_size // 2
    assert specs["model.layers.4.mlp.experts.0.up_proj.weight"] == (
        0, shard, 2 * shard)
    srows = shard // BLOCK[0]
    assert specs["model.layers.4.mlp.experts.0.up_proj.weight_scale_inv"] == (
        0, srows, 2 * srows)
    assert "model.layers.4.enorm.weight" not in specs


def test_fast_path_mtp_guard_and_bitwise(tmp_path):
    torch.manual_seed(4)
    cfg = _mtp_fp8_config()
    refs = _fabricate_checkpoint_files(tmp_path, cfg, include_mtp=True)
    assert refs
    with open(tmp_path / "model.safetensors.index.json") as f:
        ckpt_keys = list(json.load(f)["weight_map"])

    # A plan built without load_mtp while drafting is on is exactly the
    # bench failure: mtp.* params never see the checkpoint. The load must
    # refuse loudly instead of serving garbage drafts.
    model = Glm52ForCausalLM(cfg)
    keys, specs = build_glm52_read_plan(ckpt_keys, cfg, 0, 1)
    stream = iter_safetensors_shards(tmp_path, keys=keys, slice_spec=specs.get)
    try:
        model.load_weights(stream)
        raise AssertionError("mtp-less weight stream must be refused")
    except RuntimeError as e:
        assert "mtp." in str(e)

    # With load_mtp, fast path == generic driver, bit for bit, mtp included.
    def load(use_plan):
        torch.manual_seed(7)  # identical init for unloaded-buffer parity
        model = Glm52ForCausalLM(cfg)
        if use_plan:
            keys, specs = build_glm52_read_plan(
                ckpt_keys, cfg, 0, 1, load_mtp=True)
            model.load_weights(iter_safetensors_shards(
                tmp_path, keys=keys, slice_spec=specs.get))
        else:
            from mstar.model.glm52.weight_loader import load_weights as generic
            generic(model, tmp_path)
        return model

    fast, ref = load(True), load(False)
    mtp_params = 0
    for (name, p_fast), (_, p_ref) in zip(
        fast.named_parameters(), ref.named_parameters(), strict=True,
    ):
        assert torch.equal(p_fast, p_ref), f"fast path diverged at {name}"
        mtp_params += name.startswith("mtp.")
    assert mtp_params  # the comparison actually covered the draft module


def test_fast_path_matches_generic_driver_bitwise(tmp_path):
    torch.manual_seed(3)
    cfg = Glm52ModelConfig.reduced_fp8(block=BLOCK)
    refs = _fabricate_checkpoint_files(tmp_path, cfg)
    assert refs

    def load(use_plan):
        torch.manual_seed(7)  # identical init for unloaded-buffer parity
        model = Glm52ForCausalLM(cfg)
        if use_plan:
            with open(tmp_path / "model.safetensors.index.json") as f:
                ckpt_keys = list(json.load(f)["weight_map"])
            keys, specs = build_glm52_read_plan(ckpt_keys, cfg, 0, 1)
            weights = iter_safetensors_shards(
                tmp_path, keys=keys, slice_spec=specs.get)
            model.load_weights(weights)
        else:
            from mstar.model.glm52.weight_loader import load_weights as generic
            generic(model, tmp_path)
        return model

    fast, ref = load(True), load(False)
    for (name, p_fast), (_, p_ref) in zip(
        fast.named_parameters(), ref.named_parameters(), strict=True,
    ):
        assert torch.equal(p_fast, p_ref), f"fast path diverged at {name}"
