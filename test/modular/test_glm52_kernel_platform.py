"""Which GPUs the GLM-5.2 kernels' paths apply to. The GPU checks go through
``current_platform``: an AMD GPU reports a CUDA-style capability (an MI300X is 9.4), which a raw
capability read takes for Hopper."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")

def test_an_amd_gpu_gets_neither_pdl_nor_the_sm90_gemm(monkeypatch):
    from mstar.utils import dense_fp8, mla_prep

    rocm = SimpleNamespace(cuda_capability=lambda device=None: None,
                           has_cuda_capability=lambda major, minor=0, device=None: False)
    monkeypatch.setattr(dense_fp8, "current_platform", rocm)
    monkeypatch.setattr(mla_prep, "current_platform", rocm)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    device = torch.device("cuda", 0)
    monkeypatch.setattr(mla_prep, "_PDL", {})

    assert not mla_prep._pdl(device)
    assert not dense_fp8._probe_fi(device)


# What dense_fp8 picked before its choice moved into the fp8_linear table: see the file's
# "made_by" for how it was recorded
GOLDEN = Path(__file__).with_name("fp8_linear_dispatch_942c1bcc8.json")


def _pick(dense_fp8, device, n, k, glu, m):
    """``_choose``'s pick in the golden file's encoding: the w8a16 tile it launches, as
    ``_w8a16`` resolves a row without a config."""
    impl, config = dense_fp8._choose(device, n, k, bool(glu), m)
    if impl != "w8a16":
        return impl
    tile = dict(config) if config else dense_fp8._w8a16_tiles(m, n, k, bool(glu))
    keys = ("DOT", "BN", "BK", "SPLIT", "STAGES", "num_warps", "TB")
    return "w8a16:" + "/".join(str(int(tile[key])) for key in keys if key in tile)


def test_each_gpu_picks_what_dense_fp8_picked_before_the_table(monkeypatch):
    """Every dense fp8 linear shape GLM-5.2 builds at TP8 and TP4 (and GLM-5.3's MLA and MLP
    shapes, should it turn dense_fp8 on), at every token boundary, on an H200 with and without
    flashinfer's and cuBLAS's GEMMs, an H200 NVL and an H100 (no table rows): the kernel and
    tile are the ones the dispatch before the fp8_linear table picked, except where the H200's rows
    for another model's identical GEMM now decide (the golden file's made_by says which)."""
    from mstar.kernels import DeviceKey, registry
    from mstar.utils import dense_fp8

    golden = json.loads(GOLDEN.read_text())
    monkeypatch.delenv("MSTAR_KERNEL_TABLES", raising=False)
    device = torch.device("cuda", 0)
    wrong = []
    for scen, (gpu, fi_ok, cublas_ok) in golden["scenarios"].items():
        monkeypatch.setattr(registry, "device_key", lambda dev, gpu=gpu: DeviceKey(gpu, "sm_90"))
        monkeypatch.setitem(dense_fp8._FI_OK, device, fi_ok)
        monkeypatch.setitem(dense_fp8._W8A8_OK, device, cublas_ok)
        registry.reset_lookups()
        for name, (n, k, glu) in golden["shapes"].items():
            want = [pick for count, pick in golden["picks"][scen][name] for _ in range(count)]
            got = [_pick(dense_fp8, device, n, k, glu, m) for m in golden["tokens"]]
            wrong += [(scen, name, m, w, g) for m, w, g in zip(golden["tokens"], want, got, strict=True) if w != g]
    registry.reset_lookups()
    assert not wrong, wrong[:10]


def test_an_unusable_cublas_block_gemm_falls_back(monkeypatch):
    """torch 2.10-2.12 have the blockwise scaled_mm API, but it raises off SM90 or on a cuBLASLt
    before 12.9 (the cu128 builds): the probe says so, and the large batches take w8a16."""
    from mstar.utils import dense_fp8

    def raises(*args, **kwargs):
        raise NotImplementedError("blockwise scaling needs cuBLASLt >= 12.9")

    monkeypatch.setattr(dense_fp8, "_W8A8_API", True)
    monkeypatch.setattr(dense_fp8, "_w8a8", raises)
    assert dense_fp8._probe_w8a8(torch.device("cpu")) is False
    taken = []
    monkeypatch.setattr(dense_fp8, "_w8a16", lambda *args: taken.append(1) or None)
    monkeypatch.setattr(dense_fp8, "_W8A8_OK", {torch.device("cpu"): False})
    dense_fp8._linear(torch.zeros(600, 256, dtype=torch.bfloat16),
                      torch.zeros(256, 256, dtype=torch.uint8), torch.ones(2, 2), (128, 128), False)
    assert taken == [1]
