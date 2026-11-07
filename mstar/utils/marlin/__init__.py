"""vLLM's Marlin W4A16 GEMM for the 4-bit linear formats (``model.components.int4_formats``), vendored.

The CUDA kernel (``csrc/``: vllm-project/vllm @ 4ba787ec9, Apache-2.0, see ``csrc/NOTICE``) is JIT-built on first
use against the local CUDA toolkit, one extension per weight variant (int4 symmetric / with zero points / MXFP4,
times the group size), so a model compiles only the 15 kernel instantiations it runs (two to four minutes cold; the
build is cached under torch's extensions directory, keyed by the sources and the GPU architecture). Without
``nvcc`` or a GPU the build fails once, is logged, and the formats keep their Triton kernels.

The weight layout Marlin reads is built here in torch at load (``prepare``), ported from vLLM's pure-torch
reference of its repack kernel (``marlin_utils_test.marlin_weights``) and its scale / zero-point permutations
(``marlin_utils``, Apache-2.0, Copyright contributors to the vLLM project): the CPU-testable half of the port.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import time

import torch

logger = logging.getLogger(__name__)

# the weight kinds, as int4_formats numbers them (KIND_INT4_SYM, KIND_INT4_ZP, KIND_E2M1)
KIND_SYM, KIND_ZP, KIND_E2M1 = 0, 1, 2
GROUPS = {KIND_SYM: (32, 64, 128), KIND_ZP: (32, 64, 128), KIND_E2M1: (32,)}
_B_TYPE = {KIND_SYM: "kU4B8", KIND_ZP: "kU4", KIND_E2M1: "kFE2M1f"}
_TILE = 16

_CSRC = os.path.join(os.path.dirname(__file__), "csrc")
_HEADERS = ("marlin_template.h", "dequant.h", "marlin_mma.h", "marlin.cuh", "marlin_dtypes.cuh", "kernel.h",
            "core/scalar_type.hpp")
_HOST = "mstar_marlin.cu"


def shape_reason(kind: int, group: int, n: int, k: int) -> str | None:
    """Why Marlin can't run an ``[n, k]`` weight of this kind and group, or None."""
    if kind not in GROUPS or group not in GROUPS[kind]:
        return f"Marlin is built for groups {GROUPS.get(kind)} of this kind, not {group}"
    if n % 64 or k % 128:
        return f"Marlin needs N a multiple of 64 and K of 128; this weight is {n} x {k}"
    if k <= group:
        return "Marlin's grouped kernels need at least two groups along K"
    return None


def disabled() -> bool:
    return os.environ.get("MSTAR_W4A16_MARLIN", "1") in ("0", "false", "off")


# ---------------------------------------------------------------- the layout (torch, at load)
@functools.lru_cache(maxsize=None)
def _weight_perm() -> torch.Tensor:
    """vLLM's ``get_weight_perm(4)``: where each code of a 16 x 64 tile goes so one thread's 32-bit load is its
    MMA fragment, and the nibble interleave its dequantizer expects."""
    perm = []
    for i in range(32):
        col = i // 4
        perm1 = [16 * row + col + 8 * block for block in (0, 1)
                 for row in (2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1)]
        for j in range(4):
            perm.extend(p + 256 * j for p in perm1)
    perm = torch.tensor(perm).view(-1, 8)[:, torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])]
    return perm.reshape(-1)


@functools.lru_cache(maxsize=None)
def _scale_perms() -> tuple[torch.Tensor, torch.Tensor]:
    scale_perm = [i + 8 * j for i in range(8) for j in range(8)]
    single = [2 * i + j for i in range(4) for j in (0, 1, 8, 9, 16, 17, 24, 25)]
    return torch.tensor(scale_perm), torch.tensor(single)


def _pack8(codes: torch.Tensor) -> torch.Tensor:
    """Codes 0..15 ``[R, C]`` to int32 ``[R, C / 8]``, code 8i + j in bits 4j of word i (vLLM's ``pack_cols``)."""
    r, c = codes.shape
    v = codes.to(torch.int64).view(r, c // 8, 8) << (4 * torch.arange(8, device=codes.device))
    return v.sum(-1).to(torch.int32)  # disjoint bits: the sum is the OR; int64 -> int32 wraps the top bit


def repack_codes(codes_kn: torch.Tensor) -> torch.Tensor:
    """4-bit codes ``[K, N]`` as Marlin's B operand, int32 ``[K / 16, N * 2]`` (``marlin_weights``)."""
    k, n = codes_kn.shape
    q = codes_kn.reshape(k // _TILE, _TILE, n // _TILE, _TILE).permute(0, 2, 1, 3).reshape(k // _TILE, n * _TILE)
    perm = _weight_perm().to(q.device)
    q = q.reshape(-1, perm.numel())[:, perm].reshape(k // _TILE, n * _TILE)
    return _pack8(q).contiguous()


def permute_scales(s_kn: torch.Tensor) -> torch.Tensor:
    """Group scales ``[K / group, N]`` (at least two groups) in Marlin's order (``marlin_permute_scales``)."""
    scale_perm, _ = _scale_perms()
    g, n = s_kn.shape
    return s_kn.reshape(-1, scale_perm.numel())[:, scale_perm.to(s_kn.device)].reshape(g, n).contiguous()


def zero_points(z_kn: torch.Tensor) -> torch.Tensor:
    """Zero points 0..15 ``[K / group, N]`` as Marlin's int32 ``[K / group, N / 8]`` (``marlin_zero_points``)."""
    scale_perm, _ = _scale_perms()
    g, n = z_kn.shape
    z = z_kn.reshape(-1, scale_perm.numel())[:, scale_perm.to(z_kn.device)]
    z = z.reshape(-1, 8)[:, torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=z.device)].reshape(g, n)
    return _pack8(z).contiguous()


def prepare(kind: int, codes_nk: torch.Tensor, scale_ng: torch.Tensor, zero_ng: torch.Tensor | None = None,
            dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    """Marlin's buffers for a weight given as codes ``[N, K]`` (0..15: int4 ``q + 8`` or ``q + zero``, or e2m1
    codes), its group scales ``[N, K / group]`` (e8m0 bytes for MXFP4) and, for zero points, ``[N, K / group]``
    ints: ``marlin_weight``, ``marlin_scale`` (``dtype``, the activations'; uint8 e8m0 for MXFP4) and, with zero
    points, ``marlin_zero``."""
    out = {"marlin_weight": repack_codes(codes_nk.t().contiguous())}
    s = scale_ng.t().contiguous()
    if kind == KIND_E2M1:
        s = permute_scales(s.view(torch.uint8))
        # the fp4 dequantizer reads a word's four scales in 0, 2, 1, 3 order (mxfp4_marlin_process_scales)
        out["marlin_scale"] = s.view(-1, 4)[:, torch.tensor([0, 2, 1, 3], device=s.device)].reshape(s.shape)
    else:
        out["marlin_scale"] = permute_scales(s.to(dtype))
    if kind == KIND_ZP:
        out["marlin_zero"] = zero_points(zero_ng.t().contiguous())
    return out


# ---------------------------------------------------------------- the CUDA op (JIT, per variant)
_INSTANCE = ("template __global__ void Marlin<vllm::kBFloat16.id(), vllm::{b}.id(), vllm::kBFloat16.id(), "
             "vllm::{s}.id(), {threads}, {m}, {n}, {k}, {m8}, 4, {gb}, false>( MARLIN_KERNEL_PARAMS );")
_SELECT = ("a_type == vllm::kBFloat16 && b_type == vllm::{b} && c_type == vllm::kBFloat16 && s_type == vllm::{s} "
           "&& threads == {threads} && thread_m_blocks == {m} && thread_n_blocks == {n} && thread_k_blocks == {k} "
           "&& m_block_size_8 == {m8} && stages == 4 && group_blocks == {gb} && is_zp_float == false")
_KERNEL = ("Marlin<vllm::kBFloat16.id(), vllm::{b}.id(), vllm::kBFloat16.id(), vllm::{s}.id(), {threads}, {m}, "
           "{n}, {k}, {m8}, 4, {gb}, false>")


def _instances(kind: int, group: int) -> dict[float, list[dict]]:
    """generate_kernels.py's configurations for bf16 activations, one weight kind and group: per m-block count
    (0.5 is 8 rows), the small-batch thread tiles (k, n, threads) up to one m block, the large-batch ones above."""
    small = ((128, 128, 256), (64, 128, 128), (128, 64, 128))
    large = ((64, 256, 256), (64, 128, 128), (128, 64, 128))
    b, s = _B_TYPE[kind], "kFE8M0fnu" if kind == KIND_E2M1 else "kBFloat16"
    out = {}
    for m in (0.5, 1, 2, 3, 4):
        out[m] = [dict(b=b, s=s, threads=t, m=max(int(m), 1), n=tn // 16, k=tk // 16,
                       m8="true" if m == 0.5 else "false", gb=group // 16)
                  for tk, tn, t in (small if m <= 1 else large)]
    return out


def _sources(kind: int, group: int, arch: str) -> tuple[str, str, list[str]]:
    """(extension name, op library, sources): the vendored files and this variant's generated instantiations
    under the extensions root, named by their content and the architecture (another checkout of the same
    sources reuses the build)."""
    from torch.utils.cpp_extension import _get_build_directory

    files = {}
    for rel in (*_HEADERS, _HOST):
        with open(os.path.join(_CSRC, rel), "rb") as f:
            files[rel] = f.read()
    instances = _instances(kind, group)
    head = '#include "kernel.h"\n#include "marlin_template.h"\n\nnamespace marlin {\n\n'
    for i, cfgs in enumerate(instances.values()):
        body = "\n\n".join(_INSTANCE.format(**c) for c in cfgs)
        files[f"kernels_{i}.cu"] = (head + body + "\n\n}\n").encode()
    select = "".join(f"{'if' if not j else 'else if'} ({_SELECT.format(**c)})\n  kernel = {_KERNEL.format(**c)};\n"
                     for j, c in enumerate(c for cfgs in instances.values() for c in cfgs))
    files["kernel_selector.h"] = select.encode()
    digest = hashlib.sha256(arch.encode() + b"".join(k.encode() + v for k, v in sorted(files.items())))
    tag = f"{_B_TYPE[kind][1:].lower()}_g{group}"
    lib = f"_mstar_marlin_{tag}"
    name = f"{lib}_{digest.hexdigest()[:12]}"
    src_dir = os.path.join(os.path.dirname(_get_build_directory(name, verbose=False)), "_mstar_marlin_src",
                           digest.hexdigest()[:12])
    for rel, data in files.items():
        path = os.path.join(src_dir, rel)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)  # a racing rank sees the whole file or none
    sources = [os.path.join(src_dir, _HOST)] + [os.path.join(src_dir, f"kernels_{i}.cu") for i in range(len(instances))]
    return name, lib, sources


def _arch(device: torch.device | None, capability: tuple[int, int] | None) -> str:
    major, minor = capability or torch.cuda.get_device_capability(device)
    return f"{major}{minor}"


@functools.lru_cache(maxsize=None)
def build(kind: int, group: int, arch: str) -> str | None:
    """JIT-build (or load the cached build of) one variant for ``arch`` (e.g. "90"); its op library's name, or
    None if it can't be built here (logged once)."""
    from torch.utils.cpp_extension import load

    from mstar.utils.fused_moe.align import _clear_stale_build_lock

    last_error: Exception | None = None
    t0 = time.time()
    for attempt in range(2):
        try:
            name, lib, sources = _sources(kind, group, arch)
            if attempt == 0:
                _clear_stale_build_lock(name)
            else:
                time.sleep(2.0)
            load(name=name, sources=sources, extra_include_paths=[os.path.dirname(sources[0])],
                 extra_cflags=["-O3", "-std=c++17"],
                 extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", f"-DMSTAR_MARLIN_LIB={lib}",
                                    f"-gencode=arch=compute_{arch},code=sm_{arch}"],
                 is_python_module=False, verbose=False)
            _ = getattr(torch.ops, lib).gemm
            logger.info("w4a16: Marlin %s ready (%.0f s)", lib, time.time() - t0)
            return lib
        except Exception as e:  # noqa: BLE001  (no nvcc, no device, a toolchain mismatch)
            last_error = e
    logger.warning("w4a16: could not build vendored Marlin for kind %d group %d (%s); the Triton kernels run "
                   "instead", kind, group, str(last_error)[:500])
    return None


def available(kind: int, group: int, device: torch.device) -> str | None:
    """The built op library for a weight on ``device``, or None (not CUDA, before Ampere, disabled, no build)."""
    if disabled() or device.type != "cuda" or torch.cuda.get_device_capability(device) < (8, 0):
        return None
    if kind not in GROUPS or group not in GROUPS[kind]:
        return None
    return build(kind, group, _arch(device, None))


_WORKSPACES: dict[tuple[int, int], torch.Tensor] = {}


def _workspace(device: torch.device) -> torch.Tensor:
    """Marlin's locks for this device and stream: int32 zeros, four per SM; each call leaves them zero."""
    key = (device.index or 0, torch.cuda.current_stream(device).cuda_stream)
    ws = _WORKSPACES.get(key)
    if ws is None:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        ws = _WORKSPACES[key] = torch.zeros(sms * 4, dtype=torch.int32, device=device)
    return ws


def gemm(lib: str, x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor | None,
         kind: int, n: int) -> torch.Tensor:
    """``x [M, K] @ W.T`` from ``prepare``'s buffers through the variant ``lib`` (``build``'s result)."""
    if kind == KIND_E2M1:
        scale = scale.view(torch.float8_e8m0fnu)
    return getattr(torch.ops, lib).gemm(x, weight, scale, zero, _workspace(x.device), kind, n, x.shape[1])
