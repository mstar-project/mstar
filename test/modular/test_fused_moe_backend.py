"""Fused-MoE backend choice and launch-config tables, without a GPU.

AUTO is Triton everywhere. Without a tuned table NVIDIA runs the kernel's own
default config (same as before backends existed) and AMD adds num_stages=2.
Tuned tables are found by GPU name and nearest token count. The AITER backend
refuses unsupported platforms and sizes up front.
"""

from __future__ import annotations

import importlib.util
import json

import pytest
import torch

from mstar import platforms
from mstar.platforms import CpuPlatform, CudaPlatform, Platform, RocmPlatform
from mstar.utils.fused_moe import backend as backend_module
from mstar.utils.fused_moe import config as config_module
from mstar.utils.fused_moe.backend import (
    FusedMoEBackend,
    MoEBackend,
    TritonFusedMoE,
    make_moe_backend,
)
from mstar.utils.fused_moe.config import (
    CONFIG_DIR,
    MoEProblem,
    config_file_name,
    load_tuned_configs,
    nearest_config,
)
from mstar.utils.fused_moe.kernels import get_default_config

THINKER = MoEProblem(num_experts=128, intermediate_size=768, hidden_size=2048, top_k=8)
TALKER_TP2 = MoEProblem(num_experts=128, intermediate_size=192, hidden_size=1024, top_k=6)
LAUNCH_KEYS = {
    "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "num_warps", "num_stages",
    "waves_per_eu", "matrix_instr_nonkdim", "kpack",
}


@pytest.fixture
def on_platform(monkeypatch):
    def use(platform: Platform, device_name: str | None = None) -> Platform:
        monkeypatch.setattr(platforms, "resolve_platform", lambda: platform)
        monkeypatch.setattr(backend_module, "device_config_name", lambda: device_name)
        backend_module._tuned_configs.cache_clear()
        return platform

    yield use
    backend_module._tuned_configs.cache_clear()


def _default(m: int, p: MoEProblem = THINKER) -> dict:
    return get_default_config(M=m, E=p.num_experts, N=2 * p.intermediate_size, K=p.hidden_size, top_k=p.top_k)


@pytest.mark.parametrize("kind", ["auto", "triton", MoEBackend.AUTO, MoEBackend.TRITON])
@pytest.mark.parametrize("platform", [CudaPlatform(), RocmPlatform(), CpuPlatform()], ids=repr)
def test_auto_and_triton_build_the_triton_backend(on_platform, kind, platform):
    on_platform(platform)
    assert type(make_moe_backend(kind, THINKER)) is TritonFusedMoE


def test_unknown_backend_is_an_error(on_platform):
    on_platform(CudaPlatform())
    with pytest.raises(ValueError):
        make_moe_backend("cutlass", THINKER)


@pytest.mark.parametrize("m", [1, 8, 128, 129, 4096])
def test_untuned_size_on_nvidia_runs_the_kernel_default(on_platform, m):
    on_platform(CudaPlatform(), device_name="NVIDIA_H100_80GB_HBM3")
    backend = make_moe_backend("auto", THINKER)
    assert backend.config(m) is None  # fused_experts then picks get_default_config itself


@pytest.mark.parametrize("m", [1, 8, 128, 129, 4096])
def test_untuned_size_on_amd_pins_two_stages(on_platform, m):
    on_platform(RocmPlatform(), device_name="AMD_Instinct_MI300X")
    backend = make_moe_backend("auto", TALKER_TP2)  # no table ships for these sizes
    assert backend.config(m) == {**_default(m, TALKER_TP2), "num_stages": 2}


def test_tuned_table_is_found_by_device_name_and_nearest_count(on_platform, monkeypatch, tmp_path):
    table = {
        "triton_version": "3.7.1",
        "1": {"BLOCK_SIZE_M": 16, "num_warps": 1},
        "16": {"BLOCK_SIZE_M": 16, "num_warps": 2},
        "64": {"BLOCK_SIZE_M": 32, "num_warps": 4},
    }
    (tmp_path / config_file_name(THINKER, "Some_GPU")).write_text(json.dumps(table))
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(
        backend_module, "load_tuned_configs",
        lambda problem, name: load_tuned_configs(problem, name, config_dir=tmp_path),
    )
    on_platform(CudaPlatform(), device_name="Some_GPU")
    backend = make_moe_backend("triton", THINKER)
    assert backend.config(1)["num_warps"] == 1
    assert backend.config(8)["num_warps"] == 1  # |8 - 1| < |8 - 16|
    assert backend.config(9)["num_warps"] == 2
    assert backend.config(40)["num_warps"] == 2  # tie between 16 and 64 goes to the smaller
    assert backend.config(1000)["num_warps"] == 4

    on_platform(CudaPlatform(), device_name="Other_GPU")
    assert make_moe_backend("triton", THINKER).config(1) is None


def test_nearest_config_ties_go_to_the_smaller_count():
    table = {1: {"a": 1}, 3: {"a": 3}}
    assert nearest_config(table, 2) == {"a": 1}
    assert nearest_config(table, 0) == {"a": 1}
    assert nearest_config(table, 10) == {"a": 3}


def test_no_gpu_means_no_table():
    assert load_tuned_configs(THINKER, None) is None


def test_file_name_names_the_whole_problem():
    assert config_file_name(THINKER, "AMD_Instinct_MI300X") == (
        "E=128,N=768,K=2048,topk=8,device_name=AMD_Instinct_MI300X.json"
    )


def _shipped():
    return sorted(CONFIG_DIR.glob("*.json"))


@pytest.mark.parametrize("path", _shipped(), ids=lambda p: p.name)
def test_shipped_tables_are_well_formed(path):
    raw = json.loads(path.read_text())
    entries = {k: v for k, v in raw.items() if k.isdigit()}
    assert entries and set(raw) - set(entries) <= {"triton_version"}
    for m, cfg in entries.items():
        assert set(cfg) <= LAUNCH_KEYS, (m, cfg)
        assert {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M"} <= set(cfg)
        assert cfg.get("num_warps", 4) <= 16  # 64-wide waves, so 1024 threads max
        # compiles but computes garbage on gfx942 with Triton 3.7
        assert not (cfg.get("kpack") == 2 and cfg["BLOCK_SIZE_K"] < 32), (m, cfg)
    name = path.name
    problem = MoEProblem(*(int(part.split("=")[1]) for part in name.split(",")[:4]))
    device = name.split("device_name=")[1].removesuffix(".json")
    assert config_file_name(problem, device) == name
    assert load_tuned_configs(problem, device) == {int(m): c for m, c in entries.items()}


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
    def __init__(self, problem):
        super().__init__(problem)
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

    def fake_make(kind, problem):
        built.append((kind, problem))
        return _RecordingBackend(problem)

    monkeypatch.setattr(moe_module, "make_moe_backend", fake_make)
    kwargs = dict(hidden_size=64, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
                  moe_backend="triton")
    if "Shared" in block_name:
        kwargs["shared_expert"] = GatedMLP(hidden_size=64, intermediate_size=32, activation="silu")
    with torch.device("meta"):
        block = getattr(moe_module, block_name)(**kwargs)
    assert built == [("triton", MoEProblem(4, 32, 64, 2))]

    container = torch.nn.Sequential(block)
    moe_module.prepare_moe_weights(container)
    assert block.experts_backend.prepared == [(block.experts.gate_up_proj, block.experts.down_proj)]


@pytest.mark.parametrize("parallel", ["tp", "ep"])
def test_instantiate_moe_block_passes_the_backend_choice(on_platform, monkeypatch, parallel):
    from mstar.model.components import MoeParallelConfig, instantiate_moe_block
    from mstar.model.components import moe as moe_module

    on_platform(CpuPlatform())
    built = []

    def fake_make(kind, problem):
        built.append(kind)
        return _RecordingBackend(problem)

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
