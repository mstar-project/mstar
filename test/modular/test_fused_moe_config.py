"""Tuned-config tables (lookup and the shipped files) and precomputed alignment for the fused MoE."""

from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.utils.fused_moe import kernels

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused MoE requires CUDA")
LAUNCH_KEYS = {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "num_warps", "num_stages"}
# Compiler options Triton's HIP backend takes. A CUDA launch rejects them.
AMD_KEYS = {"waves_per_eu", "matrix_instr_nonkdim", "kpack"}


@pytest.fixture(autouse=True)
def _fresh_tables():
    kernels.tuned_configs.cache_clear()
    yield
    kernels.tuned_configs.cache_clear()


@pytest.fixture
def tuned_table(tmp_path, monkeypatch):
    monkeypatch.setattr(kernels, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(kernels, "device_config_name", lambda: "Some_GPU")
    return kernels.config_path(8, 64, 32, "Some_GPU")


def test_get_config_falls_back_to_default(tuned_table):
    up, down = kernels.get_config(M=4, E=8, N=64, K=32, top_k=2)
    assert up == down == kernels.get_default_config(4, 8, 64, 32, 2)


def test_get_config_uses_nearest_batch_and_down_overrides(tuned_table):
    tile = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3}
    tuned_table.write_text(json.dumps({
        "1": {**tile, "down": {"BLOCK_SIZE_K": 64, "BLOCK_SIZE_M": 64}},
        "64": {**tile, "BLOCK_SIZE_M": 64},
    }))
    up, down = kernels.get_config(M=2, E=8, N=64, K=32, top_k=2)
    assert up == tile
    # The down GEMM shares the up GEMM's alignment, so BLOCK_SIZE_M is not overridable.
    assert down == {**tile, "BLOCK_SIZE_K": 64}
    assert kernels.get_config(M=50, E=8, N=64, K=32, top_k=2)[0]["BLOCK_SIZE_M"] == 64


def test_get_config_ties_go_to_the_smaller_batch(tuned_table):
    tile = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1}
    tuned_table.write_text(json.dumps({"3": {**tile, "num_warps": 8}, "1": {**tile, "num_warps": 4}}))
    assert kernels.get_config(M=2, E=8, N=64, K=32, top_k=2)[0]["num_warps"] == 4


def test_no_gpu_means_no_table(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert kernels.device_config_name() is None
    assert kernels.tuned_configs(128, 1536, 2048) is None


@pytest.mark.parametrize("gpu", ["AMD Instinct MI300X", "AMD Instinct MI325X"])
def test_qwen3_omni_tables_load_by_device_name(monkeypatch, gpu):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: gpu)
    for shape in [(128, 1536, 2048), (128, 768, 1024)]:  # the Thinker's and Talker's w1
        assert kernels.tuned_configs(*shape) is not None, (gpu, shape)


@pytest.mark.parametrize("path", sorted(kernels.CONFIG_DIR.glob("*.json")), ids=lambda p: p.name)
def test_shipped_tables_are_well_formed(path, monkeypatch):
    raw = json.loads(path.read_text())
    device = path.name.split("device_name=")[1].removesuffix(".json")
    allowed = LAUNCH_KEYS | AMD_KEYS if device.startswith("AMD_") else LAUNCH_KEYS
    assert raw and all(m.isdigit() for m in raw)
    for m, entry in raw.items():
        up = {k: v for k, v in entry.items() if k != "down"}
        down = {**up, **entry.get("down", {})}
        assert {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M"} <= set(up), (m, entry)
        assert set(up) <= allowed and set(down) <= allowed, (m, entry)
        assert "BLOCK_SIZE_M" not in entry.get("down", {}), (m, entry)
        for cfg in (up, down):  # compiles but computes garbage on gfx942 with Triton 3.7
            assert not (cfg.get("kpack") == 2 and cfg["BLOCK_SIZE_K"] < 32), (m, entry)
    E, N, K = (int(part.split("=")[1]) for part in path.name.split(",")[:3])
    assert kernels.config_path(E, N, K, device) == path
    monkeypatch.setattr(kernels, "device_config_name", lambda: device)
    assert kernels.tuned_configs(E, N, K) == {int(m): entry for m, entry in raw.items()}


@CUDA
@pytest.mark.parametrize("num_tokens", [1, 3, 16, 64])
def test_fused_experts_matches_naive_and_precomputed_alignment(num_tokens):
    from mstar.model.components.moe import dispatch_experts_fused
    from mstar.utils.fused_moe import MoeAlignment, fused_experts, moe_block_m
    from mstar.utils.fused_moe.align import moe_align_block_size

    # The Command A+ expert shape, which has a tuned table on GB200.
    experts, hidden, inter, top_k = 128, 4096, 1024, 8
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(experts, 2 * inter, hidden, device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(experts, hidden, inter, device="cuda", dtype=torch.bfloat16) * 0.02
    weights, ids = torch.rand(num_tokens, experts, device="cuda").topk(top_k, dim=-1)
    weights = (weights / weights.sum(-1, keepdim=True)).to(torch.bfloat16)

    out = fused_experts(x, w1, w2, weights, ids)
    naive = dispatch_experts_fused(x, w1, w2, experts, ids, weights)
    torch.testing.assert_close(out, naive, atol=2e-2, rtol=2e-2)

    block_m = moe_block_m(num_tokens, w1, top_k)
    alignment = MoeAlignment(*moe_align_block_size(ids.to(torch.int32), block_m, experts), block_m)
    precomputed = fused_experts(x, w1, w2, weights, ids, alignment=alignment)
    torch.testing.assert_close(precomputed, out, atol=0, rtol=0)

    with pytest.raises(AssertionError, match="BLOCK_SIZE_M"):
        fused_experts(x, w1, w2, weights, ids, alignment=alignment._replace(block_m=block_m * 2))
