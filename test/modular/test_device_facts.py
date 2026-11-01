"""``Platform.device_facts``: what torch reports about the GPU, completed from
the table of known products, with unknown fields None. ``torch.cuda`` is
stubbed, so no GPU is needed."""

from types import SimpleNamespace

import pytest
import torch

from mstar import platforms
from mstar.platforms import CpuPlatform, CudaPlatform, RocmPlatform, XpuPlatform, current_platform
from mstar.platforms import facts as facts_module
from mstar.platforms.facts import DeviceFacts, MiB, facts_for, normalize_product


@pytest.mark.parametrize(
    ("name", "product"),
    [
        ("AMD Instinct MI300X", "mi300x"),
        ("AMD Instinct MI300X VF", "mi300x"),
        ("AMD Instinct MI325X", "mi325x"),
        ("NVIDIA H100 80GB HBM3", "h100-80gb-hbm3"),
        ("NVIDIA A100-SXM4-80GB", "a100-sxm4-80gb"),
        ("", None),
        (None, None),
    ],
)
def test_product_names_are_normalized(name, product):
    assert normalize_product(name) == product


def test_an_mi300x_slice_is_an_mi300x():
    facts = facts_for("rocm", name="AMD Instinct MI300X VF", arch="gfx942", compute_units=304)
    assert (facts.product, facts.arch, facts.xcds, facts.compute_units) == ("mi300x", "gfx942", 8, 304)
    assert (facts.lds_bytes, facts.mall_bytes, facts.hbm_gbps) == (64 * 1024, 256 * MiB, 5300.0)
    assert facts.peak_tflops["bf16"] == pytest.approx(1307.4)


def test_mi325x_shares_the_arch_but_not_the_bandwidth():
    mi300x = facts_for("rocm", name="AMD Instinct MI300X", arch="gfx942")
    mi325x = facts_for("rocm", name="AMD Instinct MI325X", arch="gfx942")
    assert mi300x.arch == mi325x.arch and mi300x.peak_tflops == mi325x.peak_tflops
    assert (mi300x.hbm_gbps, mi325x.hbm_gbps) == (5300.0, 6000.0)


def test_gfx950_doubles_lds():
    facts = facts_for("rocm", name="AMD Instinct MI355X", arch="gfx950")
    assert (facts.compute_units, facts.lds_bytes, facts.hbm_gbps) == (256, 160 * 1024, 8000.0)


def test_what_the_device_reports_wins_over_the_table():
    facts = facts_for("rocm", name="AMD Instinct MI300X", arch="gfx942", compute_units=228, l2_bytes=2 * MiB)
    assert (facts.compute_units, facts.l2_bytes) == (228, 2 * MiB)


def test_an_unknown_product_has_no_roofline():
    facts = facts_for("cuda", name="NVIDIA Future GPU", arch="sm_130", compute_units=200, memory_bytes=1 << 40)
    assert (facts.product, facts.compute_units, facts.memory_bytes) == ("future-gpu", 200, 1 << 40)
    assert facts.hbm_gbps is None and dict(facts.peak_tflops) == {}
    assert facts.floor_us(1e9) is None


def test_a_product_of_another_arch_is_not_matched():
    assert facts_for("rocm", name="AMD Instinct MI300X", arch="gfx950").hbm_gbps is None


def test_the_floor_is_the_higher_of_bandwidth_and_throughput():
    facts = facts_for("rocm", name="AMD Instinct MI300X", arch="gfx942")
    # 5.3 GB at 5.3 TB/s is 1 ms; 1.3 TFLOP at 1307.4 TFLOPS is 1 ms too
    assert facts.floor_us(5.3e9) == pytest.approx(1000.0)
    assert facts.floor_us(5.3e6, flops=1307.4e9) == pytest.approx(1000.0)
    assert facts.floor_us(5.3e9, flops=1.0) == pytest.approx(1000.0)
    assert facts.floor_us(1.0, flops=1.0, dtype="fp4") is None


@pytest.mark.parametrize("platform", [CpuPlatform(), XpuPlatform()], ids=repr)
def test_a_platform_without_gpu_facts_says_only_its_name(platform):
    assert platform.device_facts() == DeviceFacts(platform=platform.name)


@pytest.fixture
def gpu(monkeypatch):
    """A GPU whose properties torch reports as given."""

    def use(platform, **props):
        monkeypatch.setattr(platforms, "resolve_platform", lambda: platform)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device=None: SimpleNamespace(**props))
        facts_module._gpu_facts.cache_clear()

    yield use
    facts_module._gpu_facts.cache_clear()


def test_rocm_reads_the_device_then_the_table(gpu):
    gpu(
        RocmPlatform(), name="AMD Instinct MI300X", gcnArchName="gfx942:sramecc+:xnack-",
        multi_processor_count=304, total_memory=192 * 1024 ** 3, L2_cache_size=4 * MiB, warp_size=64,
    )
    facts = current_platform.device_facts()
    assert (facts.platform, facts.arch, facts.product, facts.warp_size) == ("rocm", "gfx942", "mi300x", 64)
    assert (facts.memory_bytes, facts.l2_bytes, facts.hbm_gbps) == (192 * 1024 ** 3, 4 * MiB, 5300.0)
    assert current_platform.device_facts(0) is facts, "cached per device"


def test_cuda_reads_the_capability_as_its_arch(gpu, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    gpu(
        CudaPlatform(), name="NVIDIA H100 80GB HBM3", multi_processor_count=132,
        total_memory=80 * 1024 ** 3, shared_memory_per_multiprocessor=233472, warp_size=32,
    )
    facts = current_platform.device_facts("cuda:0")
    assert (facts.arch, facts.product, facts.compute_units, facts.hbm_gbps) == ("sm_90", "h100-80gb-hbm3", 132, 3350.0)
    assert facts.l2_bytes == 50 * MiB, "from the table when the device does not say"
