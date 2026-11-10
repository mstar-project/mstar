"""Fused-MoE backend choice, without a GPU.

AUTO is Triton everywhere. The Triton backend runs fused_experts, which takes
the GPU's moe_gemm table (test_fused_moe_config.py covers the lookup) and the
kernel's own defaults without one. The AITER backend refuses unsupported
platforms and sizes up front.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from mstar import platforms
from mstar.platforms import CpuPlatform, CudaPlatform, Platform, RocmPlatform
from mstar.utils.fused_moe.backend import (
    FusedMoEBackend,
    MoEBackend,
    TritonFusedMoE,
    make_moe_backend,
)

THINKER = 768  # the Thinker's per-expert intermediate size
TALKER_TP2 = 192  # the Talker's at TP2


@pytest.fixture
def on_platform(monkeypatch):
    def use(platform: Platform) -> Platform:
        monkeypatch.setattr(platforms, "resolve_platform", lambda: platform)
        return platform

    return use


@pytest.mark.parametrize("kind", ["auto", "triton", MoEBackend.AUTO, MoEBackend.TRITON])
@pytest.mark.parametrize("platform", [CudaPlatform(), RocmPlatform(), CpuPlatform()], ids=repr)
def test_auto_and_triton_build_the_triton_backend(on_platform, kind, platform):
    on_platform(platform)
    assert type(make_moe_backend(kind, THINKER)) is TritonFusedMoE


def test_unknown_backend_is_an_error(on_platform):
    on_platform(CudaPlatform())
    with pytest.raises(ValueError):
        make_moe_backend("cutlass", THINKER)


def test_aiter_refuses_non_amd_platforms(on_platform):
    on_platform(CudaPlatform())
    with pytest.raises(ValueError, match="AMD GPUs only"):
        make_moe_backend("aiter", THINKER)


def test_aiter_refuses_intermediate_sizes_its_kernels_cannot_tile(on_platform, monkeypatch):
    on_platform(RocmPlatform())
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object())
    with pytest.raises(ValueError, match="multiple of 128"):
        make_moe_backend("aiter", TALKER_TP2)


def test_aiter_without_the_package_is_an_error(on_platform, monkeypatch):
    on_platform(RocmPlatform())
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "aiter" else real(name, *a))
    with pytest.raises(ValueError, match="needs the aiter package"):
        make_moe_backend("aiter", THINKER)


class _RecordingBackend(FusedMoEBackend):
    def __init__(self, intermediate_size):
        super().__init__(intermediate_size)
        self.prepared = []

    def prepare_weights(self, w1, w2):
        self.prepared.append((w1, w2))


@pytest.mark.parametrize("block_name", [
    "SparseMoeBlock", "SparseMoeBlockWithSharedExpert",
    "ParallelSparseMoeBlock", "ParallelSparseMoeBlockWithSharedExpert",
    "ExpertParallelSparseMoeBlock",
])
def test_blocks_build_their_backend_and_prepare_hands_it_the_expert_weights(
    on_platform, monkeypatch, block_name,
):
    from mstar.model.components import GatedMLP
    from mstar.model.components import moe as moe_module

    on_platform(CpuPlatform())
    built = []

    def fake_make(kind, intermediate_size):
        built.append((kind, intermediate_size))
        return _RecordingBackend(intermediate_size)

    monkeypatch.setattr(moe_module, "make_moe_backend", fake_make)
    kwargs = dict(hidden_size=64, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
                  moe_backend="triton")
    if "Shared" in block_name:
        kwargs["shared_expert"] = GatedMLP(hidden_size=64, intermediate_size=32, activation="silu")
    with torch.device("meta"):
        block = getattr(moe_module, block_name)(**kwargs)
    assert built == [("triton", 32)]

    container = torch.nn.Sequential(block)
    moe_module.prepare_moe_weights(container)
    assert block.experts_backend.prepared == [(block.experts.gate_up_proj, block.experts.down_proj)]


@pytest.mark.parametrize("parallel", ["tp", "ep"])
def test_instantiate_moe_block_passes_the_backend_choice(on_platform, monkeypatch, parallel):
    from mstar.model.components import MoeParallelConfig, instantiate_moe_block
    from mstar.model.components import moe as moe_module

    on_platform(CpuPlatform())
    built = []

    def fake_make(kind, intermediate_size):
        built.append(kind)
        return _RecordingBackend(intermediate_size)

    monkeypatch.setattr(moe_module, "make_moe_backend", fake_make)
    with torch.device("meta"):
        instantiate_moe_block(
            hidden_size=64, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
            moe_parallel=MoeParallelConfig(parallel=parallel), moe_backend="aiter",
        )
    assert built == ["aiter"]


@pytest.mark.parametrize("skip_invalid", [False, True])
def test_triton_hands_skip_invalid_to_fused_experts(on_platform, monkeypatch, skip_invalid):
    from mstar.utils.fused_moe import runner

    on_platform(CudaPlatform())
    seen = {}
    monkeypatch.setattr(runner, "fused_experts", lambda *args, **kwargs: seen.update(kwargs))
    make_moe_backend("triton", THINKER)(torch.zeros(3, 2048), None, None, None, None, skip_invalid=skip_invalid)
    assert seen["skip_invalid"] is skip_invalid


def test_aiter_refuses_to_skip_slots(on_platform, monkeypatch):
    on_platform(RocmPlatform())
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object())
    backend = make_moe_backend("aiter", THINKER)
    with pytest.raises(NotImplementedError, match="use 'triton'"):
        backend(torch.zeros(1, 2048), None, None, torch.zeros(1, 8), torch.zeros(1, 8), skip_invalid=True)


def test_qwen3_omni_text_configs_default_to_auto():
    from mstar.model.qwen3_omni.config import TalkerTextConfig, ThinkerTextConfig

    assert ThinkerTextConfig.from_dict({"head_dim": 128}).moe_backend == "auto"
    assert TalkerTextConfig.from_dict({"moe_backend": "aiter"}).moe_backend == "aiter"
