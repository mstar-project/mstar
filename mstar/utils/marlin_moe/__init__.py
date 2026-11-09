"""vLLM's Marlin MoE GEMM (``marlin_moe_wna16``) for the 4-bit routed experts (``moe.QuantExperts``), vendored.

The CUDA kernel (``csrc/``: vllm-project/vllm @ 4ba787ec9, Apache-2.0, see ``csrc/NOTICE``) is JIT-built on first
use against the local CUDA toolkit, one extension per weight variant (MXFP4, int4 symmetric / with zero points
times the group size), so a model compiles only the 15 kernel instantiations it runs; the build is cached under
torch's extensions directory, keyed by the sources and the GPU architecture. Without ``nvcc`` or a GPU the build
fails once, is logged, and the experts keep the Triton kernel (``fused_moe.w4a16``).

The weight layout Marlin reads is built here in torch at load (``prepare_experts``): per expert, vLLM's dense
Marlin layout (its pure-torch reference of the repack kernel, ``marlin_utils_test.marlin_weights``, and its
scale / zero-point permutations, ``marlin_utils`` / ``marlin_utils_fp4``; Apache-2.0, Copyright contributors to
the vLLM project), batched over experts and chunked so a layer's conversion holds a few hundred MB at a time.

A call (``fused_experts_marlin``) runs the plan of vLLM's ``fused_marlin_moe``: moe_align at a block size picked
from the tokens per expert, the gate/up grouped GEMM, the gated activation as a separate step
(``kernels.act_and_mul_triton``: SwiGLU or Kimi's SiTU, which Marlin's epilogue doesn't compute; with
``skip_invalid``, only this rank's slots), the down grouped GEMM times the routing weights, the top-k sum.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import math
import os
import time

import torch

logger = logging.getLogger(__name__)

# the weight kinds, as int4_formats numbers them (KIND_INT4_SYM, KIND_INT4_ZP, KIND_E2M1)
KIND_SYM, KIND_ZP, KIND_E2M1 = 0, 1, 2
GROUPS = {KIND_SYM: (32, 64, 128), KIND_ZP: (32, 64, 128), KIND_E2M1: (32,)}
_B_TYPE = {KIND_SYM: "kU4B8", KIND_ZP: "kU4", KIND_E2M1: "kFE2M1f"}
_TILE = 16
_BLOCK_SIZES = (8, 16, 32, 48, 64)

_CSRC = os.path.join(os.path.dirname(__file__), "csrc")
_MOE_DIR = "moe/marlin_moe_wna16"
_HEADERS = ("quantization/marlin/marlin.cuh", "quantization/marlin/marlin_dtypes.cuh",
            "quantization/marlin/dequant.h", "quantization/marlin/marlin_mma.h", "core/scalar_type.hpp",
            f"{_MOE_DIR}/kernel.h", f"{_MOE_DIR}/marlin_template.h")
_HOST = f"{_MOE_DIR}/mstar_marlin_moe.cu"


def disabled() -> bool:
    return os.environ.get("MSTAR_MOE_MARLIN", "1") in ("0", "false", "off")


def shape_reason(kind: int, group: int, two_inter: int, hidden: int) -> str | None:
    """Why Marlin MoE can't run experts ``(E, 2I, H)`` of this kind and group, or None."""
    if kind not in GROUPS or group not in GROUPS[kind]:
        return f"Marlin MoE is built for groups {GROUPS.get(kind)} of this kind, not {group}"
    inter = two_inter // 2
    for n, k in ((two_inter, hidden), (hidden, inter)):
        if n % 64 or k % 64 or (n % 128 and k % 128):
            return f"Marlin MoE needs N, K in 64s (and one of them in 128s); a GEMM here is {n} x {k}"
        if k <= group:
            return "Marlin MoE's grouped kernels need at least two groups along K"
    return None


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
def _scale_perm() -> torch.Tensor:
    return torch.tensor([i + 8 * j for i in range(8) for j in range(8)])


def _pack8(codes: torch.Tensor) -> torch.Tensor:
    """Codes 0..15 ``[..., C]`` (uint8) to int32 ``[..., C / 8]``, code 8i + j in bits 4j of word i (vLLM's
    ``pack_cols``), through bytes: no wider intermediate than the codes."""
    q = codes.to(torch.uint8)
    pairs = (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous()
    return pairs.view(torch.int32)


def repack_experts(codes_ekn: torch.Tensor) -> torch.Tensor:
    """4-bit codes ``[E, K, N]`` as Marlin's B operand per expert, int32 ``[E, K / 16, N * 2]``."""
    e, k, n = codes_ekn.shape
    q = codes_ekn.reshape(e, k // _TILE, _TILE, n // _TILE, _TILE).permute(0, 1, 3, 2, 4)
    q = q.reshape(e, k // _TILE, n * _TILE)
    perm = _weight_perm().to(q.device)
    q = q.reshape(e, -1, perm.numel())[..., perm].reshape(e, k // _TILE, n * _TILE)
    return _pack8(q)


def permute_scales(s_egn: torch.Tensor) -> torch.Tensor:
    """Group scales ``[E, K / group, N]`` (at least two groups) in Marlin's order (``marlin_permute_scales``)."""
    perm = _scale_perm().to(s_egn.device)
    e, g, n = s_egn.shape
    return s_egn.reshape(e, -1, perm.numel())[..., perm].reshape(e, g, n).contiguous()


def zero_points(z_egn: torch.Tensor) -> torch.Tensor:
    """Zero points 0..15 ``[E, K / group, N]`` as Marlin's int32 ``[E, K / group, N / 8]``."""
    perm = _scale_perm().to(z_egn.device)
    e, g, n = z_egn.shape
    z = z_egn.reshape(e, -1, perm.numel())[..., perm]
    z = z.reshape(e, -1, 8)[..., torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=z.device)].reshape(e, g, n)
    return _pack8(z)


def _codes(packed: torch.Tensor) -> torch.Tensor:
    """``[..., K / 2]`` bytes (two codes each, the even K first) -> ``[..., K]`` codes."""
    b = packed.view(torch.uint8)
    return torch.stack([b & 0xF, b >> 4], dim=-1).reshape(*b.shape[:-1], -1)


def prepare_experts(kind: int, packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor | None = None,
                    dtype: torch.dtype = torch.bfloat16, chunk: int = 32) -> dict[str, torch.Tensor]:
    """Marlin's buffers for stacked experts: ``packed (E, N, K/2)`` bytes or ``(E, N, K/8)`` int32 (the
    checkpoint's codes: int4 ``q + 8`` / ``q + zero``, or e2m1), ``scale (E, N, K/group)`` (e8m0 bytes for MXFP4),
    ``zero (E, N/8, K/group)`` packed int4 for zero points: ``weight (E, K/16, 2N)`` int32, ``scale (E, K/group,
    N)`` (``dtype``, the activations'; uint8 e8m0 for MXFP4), ``zero (E, K/group, N/8)`` int32. ``chunk`` experts at
    a time."""
    e, n = packed.shape[0], packed.shape[1]
    weight, scales, zeros = [], [], []
    for first in range(0, e, chunk):
        sl = slice(first, min(e, first + chunk))
        codes = _codes(packed[sl])                                         # (c, N, K)
        weight.append(repack_experts(codes.transpose(1, 2)))               # (c, K/16, 2N)
        s = scale[sl].transpose(1, 2)                                      # (c, K/g, N)
        if kind == KIND_E2M1:
            s = permute_scales(s.view(torch.uint8))
            # the fp4 dequantizer reads a word's four scales in 0, 2, 1, 3 order (mxfp4_marlin_process_scales)
            s = s.reshape(-1, 4)[:, torch.tensor([0, 2, 1, 3], device=s.device)].reshape(s.shape)
        else:
            s = permute_scales(s.to(dtype))
        scales.append(s)
        if kind == KIND_ZP:
            words = zero[sl].view(torch.int32)                              # (c, N/8, K/g): 8 rows a word
            z = torch.stack([(words >> (4 * j)) & 0xF for j in range(8)], dim=2)  # (c, N/8, 8, K/g)
            zeros.append(zero_points(z.reshape(z.shape[0], n, -1).transpose(1, 2).contiguous()))
    out = {"weight": torch.cat(weight), "scale": torch.cat(scales)}
    if kind == KIND_ZP:
        out["zero"] = torch.cat(zeros)
    return out


# ---------------------------------------------------------------- reading the layout back (the CPU reference)
@functools.lru_cache(maxsize=None)
def _tables() -> dict[str, torch.Tensor]:
    """Where ``prepare_experts`` put each value of an expert (the dense Marlin layout's tables): ``weight [16 *
    64]`` ``(word << 5) | shift`` of code (k, n) in a row of 16-row tiles, word offset ``(n // 64) * 128 + word``,
    indexed ``(k % 16) * 64 + n % 64``; ``scale [64]`` / ``scale_e2m1 [64]`` the column of scale (g, n) at
    ``(n // 64) * 64 + table[n % 64]``; ``zero [64]`` zero point (g, n)'s word ``(n // 64) * 8 + word`` and shift."""
    inv = torch.argsort(_weight_perm())
    kk, nn = torch.arange(16)[:, None], torch.arange(64)[None, :]
    j = inv[(nn // 16) * 256 + kk * 16 + nn % 16]
    sinv = torch.argsort(_scale_perm())
    shuffle = torch.tensor([0, 2, 1, 3])
    il_inv = torch.argsort(torch.tensor([0, 2, 4, 6, 1, 3, 5, 7]))
    return {"weight": (((j // 8) << 5) | (4 * (j % 8))).reshape(-1),
            "scale": sinv, "scale_e2m1": (sinv // 4) * 4 + shuffle[sinv % 4],
            "zero": ((sinv // 8) << 5) | (4 * il_inv[sinv % 8])}


def dequantize_experts(kind: int, group: int, weight: torch.Tensor, scale: torch.Tensor,
                       zero: torch.Tensor | None = None) -> torch.Tensor:
    """fp32 ``(E, N, K)`` from ``prepare_experts``' buffers: its inverse, the CPU reference."""
    w = weight.view(torch.int32)
    e, k, n = w.shape[0], w.shape[1] * _TILE, w.shape[2] // 2
    t = {name: v.to(w.device) for name, v in _tables().items()}
    kk, nv = torch.arange(k, device=w.device)[None, :], torch.arange(n, device=w.device)[:, None]
    entry = t["weight"][(kk % 16) * 64 + nv % 64]                       # (N, K)
    words = w[:, kk // 16, (nv // 64) * 128 + (entry >> 5)]             # (E, N, K)
    codes = (words >> (entry & 31)) & 0xF
    cols = ((nv // 64) * 64 + t["scale_e2m1" if kind == KIND_E2M1 else "scale"][nv % 64])[:, 0]
    s = scale.view(torch.uint8) if kind == KIND_E2M1 else scale
    s = s.transpose(1, 2)[:, cols]                                     # (E, N, K/g)
    if kind == KIND_E2M1:
        table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
                              -6.0], device=w.device)
        values, s = table[codes.long()], torch.exp2(s.float() - 127.0)
    else:
        if kind == KIND_ZP:
            ze = t["zero"][nv[:, 0] % 64]
            zw = zero.view(torch.int32).transpose(1, 2)[:, (nv[:, 0] // 64) * 8 + (ze >> 5)]   # (E, N, K/g)
            z = ((zw >> (ze & 31)[None, :, None]) & 0xF).float()
        else:
            z = torch.full_like(s, 8.0, dtype=torch.float32)
        values = codes.float().view(e, n, -1, group) - z[..., None]
        s = s.float()
    return (values.reshape(e, n, -1, group) * s[..., None]).reshape(e, n, k)


# ---------------------------------------------------------------- the CUDA op (JIT, per variant)
_INSTANCE = ("template __global__ void Marlin<vllm::kBFloat16.id(), vllm::{b}.id(), vllm::kBFloat16.id(), "
             "vllm::{s}.id(), {threads}, {m}, {n}, {k}, {m8}, 4, {gb}, false>( MARLIN_KERNEL_PARAMS );")
_SELECT = ("a_type == vllm::kBFloat16 && b_type == vllm::{b} && c_type == vllm::kBFloat16 && s_type == vllm::{s} "
           "&& threads == {threads} && thread_m_blocks == {m} && thread_n_blocks == {n} && thread_k_blocks == {k} "
           "&& m_block_size_8 == {m8} && stages == 4 && group_blocks == {gb} && is_zp_float == false")
_KERNEL = ("Marlin<vllm::kBFloat16.id(), vllm::{b}.id(), vllm::kBFloat16.id(), vllm::{s}.id(), {threads}, {m}, "
           "{n}, {k}, {m8}, 4, {gb}, false>")


def _instances(kind: int, group: int) -> dict[float, list[dict]]:
    """generate_kernels.py's configurations (vLLM runs it for the MoE kernels too) for bf16 activations, one weight
    kind and group: per m-block count (0.5 is the 8-row block), the small-batch thread tiles (k, n, threads) up to
    one m block, the large-batch ones above."""
    small = ((128, 128, 256), (64, 128, 128), (128, 64, 128))
    large = ((64, 256, 256), (64, 128, 128), (128, 64, 128))
    b, s = _B_TYPE[kind], "kFE8M0fnu" if kind == KIND_E2M1 else "kBFloat16"
    return {m: [dict(b=b, s=s, threads=t, m=max(int(m), 1), n=tn // 16, k=tk // 16,
                     m8="true" if m == 0.5 else "false", gb=group // 16)
                for tk, tn, t in (small if m <= 1 else large)]
            for m in (0.5, 1, 2, 3, 4)}


def _sources(kind: int, group: int, arch: str) -> tuple[str, str, str, list[str]]:
    """(extension name, op library, csrc root, sources): the vendored files and this variant's generated
    instantiations under the extensions root, named by their content and the architecture."""
    from torch.utils.cpp_extension import _get_build_directory

    files = {}
    for rel in (*_HEADERS, _HOST):
        with open(os.path.join(_CSRC, rel), "rb") as f:
            files[rel] = f.read()
    instances = _instances(kind, group)
    head = '#include "kernel.h"\n#include "marlin_template.h"\n\nnamespace MARLIN_NAMESPACE_NAME {\n\n'
    for i, cfgs in enumerate(instances.values()):
        body = "\n\n".join(_INSTANCE.format(**c) for c in cfgs)
        files[f"{_MOE_DIR}/kernels_{i}.cu"] = (head + body + "\n\n}\n").encode()
    select = "".join(f"{'if' if not j else 'else if'} ({_SELECT.format(**c)})\n  kernel = {_KERNEL.format(**c)};\n"
                     for j, c in enumerate(c for cfgs in instances.values() for c in cfgs))
    files[f"{_MOE_DIR}/kernel_selector.h"] = select.encode()
    digest = hashlib.sha256(arch.encode() + b"".join(k.encode() + v for k, v in sorted(files.items())))
    lib = f"_mstar_marlin_moe_{_B_TYPE[kind][1:].lower()}_g{group}"
    name = f"{lib}_{digest.hexdigest()[:12]}"
    root = os.path.join(os.path.dirname(_get_build_directory(name, verbose=False)), "_mstar_marlin_moe_src",
                        digest.hexdigest()[:12])
    for rel, data in files.items():
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)  # a racing rank sees the whole file or none
    sources = [os.path.join(root, _HOST)] + [os.path.join(root, _MOE_DIR, f"kernels_{i}.cu")
                                             for i in range(len(instances))]
    return name, lib, root, sources


@functools.lru_cache(maxsize=None)
def _nvcc_version() -> tuple[int, int]:
    import re
    import subprocess

    from torch.utils.cpp_extension import CUDA_HOME

    try:
        out = subprocess.run([os.path.join(CUDA_HOME or "", "bin", "nvcc"), "--version"], capture_output=True,
                             text=True, check=False).stdout
    except OSError:
        return (0, 0)
    m = re.search(r"release (\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _cuda_flags(lib: str, arch: str) -> list[str]:
    flags = ["-O3", "-std=c++17", "--expt-relaxed-constexpr", f"-DMSTAR_MARLIN_MOE_LIB={lib}",
             f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    # nvcc 13 makes a __global__ template's host stub static by default; the selector (one file) takes the
    # address of instantiations made in others (the B300's nvcc 13.2: "undefined hidden symbol" at link)
    if _nvcc_version() >= (12, 8):
        flags.append("-static-global-template-stub=false")
    return flags


@functools.lru_cache(maxsize=None)
def build(kind: int, group: int, arch: str) -> str | None:
    """JIT-build (or load the cached build of) one variant for ``arch`` (e.g. "90", "103"); its op library's name,
    or None if it can't be built here (logged once)."""
    from torch.utils.cpp_extension import load

    from mstar.utils.fused_moe.align import _clear_stale_build_lock

    last_error: Exception | None = None
    t0 = time.time()
    for attempt in range(2):
        try:
            name, lib, root, sources = _sources(kind, group, arch)
            if attempt == 0:
                _clear_stale_build_lock(name)
            else:
                time.sleep(2.0)
            load(name=name, sources=sources, extra_include_paths=[root, os.path.join(root, _MOE_DIR)],
                 extra_cflags=["-O3", "-std=c++17"],
                 extra_cuda_cflags=_cuda_flags(lib, arch),
                 is_python_module=False, verbose=False)
            _ = getattr(torch.ops, lib).gemm
            logger.info("MoE: Marlin %s ready (%.0f s)", lib, time.time() - t0)
            return lib
        except Exception as e:  # noqa: BLE001  (no nvcc, no device, a toolchain mismatch)
            last_error = e
    logger.warning("MoE: could not build vendored Marlin MoE for kind %d group %d (%s); the Triton experts kernel "
                   "runs instead", kind, group, str(last_error)[:500])
    return None


def available(kind: int, group: int, device: torch.device) -> str | None:
    """The built op library for experts on ``device``, or None (not CUDA, before Ampere, disabled, no build)."""
    if disabled() or device.type != "cuda" or torch.cuda.get_device_capability(device) < (8, 0):
        return None
    if kind not in GROUPS or group not in GROUPS[kind]:
        return None
    major, minor = torch.cuda.get_device_capability(device)
    return build(kind, group, f"{major}{minor}")


_WORKSPACES: dict[tuple[int, int], torch.Tensor] = {}


def _workspace(device: torch.device) -> torch.Tensor:
    """Marlin's locks for this device and stream: int32 zeros, four per SM (the most a call uses); each call
    leaves them zero."""
    key = (device.index or 0, torch.cuda.current_stream(device).cuda_stream)
    ws = _WORKSPACES.get(key)
    if ws is None:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        ws = _WORKSPACES[key] = torch.zeros(sms * 4, dtype=torch.int32, device=device)
    return ws


def block_size(tokens: int, experts: int, top_k: int, global_experts: int | None = None) -> int:
    """vLLM's choice of ``moe_block_size``: the smallest of 8 / 16 / 32 / 48 / 64 that a typical expert's tokens
    fill to at least 90%, with the tokens scaled to this rank's share of the experts under expert parallelism."""
    if global_experts:
        tokens = math.ceil(tokens * experts / global_experts)
    for bs in _BLOCK_SIZES:
        if tokens * top_k / experts / bs < 0.9:
            return bs
    return _BLOCK_SIZES[-1]


def fused_experts_marlin_impl(
    hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor,
    z1: torch.Tensor | None, z2: torch.Tensor | None, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
    kind: int, group: int, activation: str = "silu", skip_invalid: bool = False, global_experts: int = 0,
) -> torch.Tensor:
    """``(tokens, hidden)``: the routed experts' weighted sum from ``prepare_experts``' buffers (``w1`` gate/up
    ``(E, H/16, 4I)``, ``w2`` down ``(E, I/16, 2H)``). With ``skip_invalid``, slots routed to the sentinel id ``E``
    run nothing; ``global_experts`` (expert parallelism: the experts over every rank) scales the token count the
    block size is picked from."""
    from mstar.utils.fused_moe.align import moe_align_block_size
    from mstar.utils.fused_moe.kernels import act_and_mul_triton, moe_sum_reduce_triton

    assert hidden_states.is_contiguous() and hidden_states.dim() == 2
    m, hidden = hidden_states.shape
    e = w1.shape[0]
    two_inter, inter = w1.shape[2] // 2, w2.shape[1] * _TILE
    assert w1.shape[1] * _TILE == hidden and w2.shape[2] // 2 == hidden, (w1.shape, w2.shape, hidden)
    top_k = topk_ids.shape[1]
    lib = available(kind, group, hidden_states.device)
    assert lib, "fused_experts_marlin: the Marlin MoE op is not built for this variant / device"
    topk_ids = topk_ids.to(torch.int32).contiguous()
    weights = topk_weights.to(torch.float32).contiguous()
    bs = block_size(m, e, top_k, global_experts or None)
    sorted_ids, expert_ids, post_pad = moe_align_block_size(topk_ids, bs, e)
    ws = _workspace(hidden_states.device)
    op = getattr(torch.ops, lib).gemm
    device, dtype = hidden_states.device, hidden_states.dtype
    m_topk = m * top_k
    cache1 = torch.empty((m_topk, two_inter), device=device, dtype=dtype)
    op(hidden_states, cache1, w1, s1.view(torch.float8_e8m0fnu) if kind == KIND_E2M1 else s1, z1, ws, sorted_ids,
       expert_ids, post_pad, weights, bs, top_k, False, kind, m, two_inter, hidden, False, True)
    cache2 = torch.empty((m_topk, inter), device=device, dtype=dtype)
    act_and_mul_triton(cache1, cache2, activation=activation, rows=sorted_ids if skip_invalid else None)
    # skipped slots are never written by the down GEMM, so they must start at zero
    cache3 = (torch.zeros if skip_invalid else torch.empty)((m_topk, hidden), device=device, dtype=dtype)
    op(cache2, cache3, w2, s2.view(torch.float8_e8m0fnu) if kind == KIND_E2M1 else s2, z2, ws, sorted_ids,
       expert_ids, post_pad, weights, bs, 1, True, kind, m_topk, hidden, inter, False, True)
    output = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(cache3.view(m, top_k, hidden), output, routed_scaling_factor=1.0)
    return output


@torch.library.custom_op("mstar::fused_experts_marlin", mutates_args=())
def fused_experts_marlin(
    hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor,
    z1: torch.Tensor | None, z2: torch.Tensor | None, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
    kind: int, group: int, activation: str, skip_invalid: bool, global_experts: int,
) -> torch.Tensor:
    """One opaque op to dynamo (the launch path's host logic would break the graph); see the impl."""
    return fused_experts_marlin_impl(hidden_states, w1, w2, s1, s2, z1, z2, topk_weights, topk_ids, kind, group,
                                     activation, skip_invalid, global_experts)


@fused_experts_marlin.register_kernel("cpu")
def _(hidden_states, w1, w2, s1, s2, z1, z2, topk_weights, topk_ids, kind, group, activation, skip_invalid,
      global_experts):
    """The reference off CUDA (tests): every slot's expert dequantized from Marlin's copy, in fp32."""
    from mstar.utils.situ import SiTU

    e = w1.shape[0]
    d1, d2 = dequantize_experts(kind, group, w1, s1, z1), dequantize_experts(kind, group, w2, s2, z2)
    situ = SiTU.parse(activation)
    out = torch.zeros(hidden_states.shape, dtype=torch.float32)
    x = hidden_states.float()
    for i in topk_ids.unique().tolist():
        if i >= e:
            continue
        rows, slot = (topk_ids == i).nonzero(as_tuple=True)
        gate, up = (x[rows] @ d1[i].T).chunk(2, -1)
        h = situ(gate, up) if situ is not None else torch.nn.functional.silu(gate) * up
        out.index_add_(0, rows, (h.float() @ d2[i].T) * topk_weights[rows, slot, None].float())
    return out.to(hidden_states.dtype)


@fused_experts_marlin.register_fake
def _(hidden_states, w1, w2, s1, s2, z1, z2, topk_weights, topk_ids, kind, group, activation, skip_invalid,
      global_experts):
    return torch.empty_like(hidden_states)


# ---------------------------------------------------------------- a QuantExperts block's experts on Marlin
class MarlinExperts:
    """A ``moe.QuantExperts`` block's w4a16 / mxfp4 experts on Marlin MoE (its ``_experts_kernel``): converted at
    load (``convert``), then called as the block calls its experts kernel."""

    def __init__(self, block, kind: int, group: int, held: dict[str, torch.Tensor | None], global_experts: int):
        self.block, self.kind, self.group, self.held, self.global_experts = block, kind, group, held, global_experts

    @staticmethod
    def _source(block, proj: str):
        b = block._expert_buffers(proj, slice(None))
        packed, scale, zero = b["weight_packed"], b["weight_scale"], b.get("weight_zero_point")
        return packed, scale.view(torch.uint8) if scale.element_size() == 1 else scale, zero

    @classmethod
    @torch.no_grad()
    def convert(cls, block, device: torch.device, probe_tokens: int = 16, tol: float = 2e-2) -> MarlinExperts | None:
        """Marlin's copy of the block's experts, checked against the Triton kernel on random tokens before the
        checkpoint's copy is freed (one copy: a Kimi K3.1 rank's experts would not fit twice). None (the block
        keeps its storage and the Triton kernel) where Marlin MoE isn't built, doesn't take the shape, or the
        probe disagrees."""
        from mstar.model.components.int4_formats import KIND_E2M1 as E2M1
        from mstar.model.components.int4_formats import KIND_INT4_SYM, KIND_INT4_ZP
        from mstar.utils.fused_moe.w4a16 import fused_experts_w4a16
        from mstar.utils.situ import activation_spec

        fmt = block.experts_packed
        e, two_inter, hidden = block._experts_dims()
        w1, s1, z1 = cls._source(block, "gate_up_proj")
        w2, s2, z2 = cls._source(block, "down_proj")
        kind = E2M1 if fmt.name == "mxfp4" else KIND_INT4_SYM if z1 is None else KIND_INT4_ZP
        reason = shape_reason(kind, fmt.group, two_inter, hidden)
        if reason is not None or available(kind, fmt.group, device) is None:
            logger.info("MoE: %s experts stay on the Triton kernel: %s", fmt.name,
                        reason or "Marlin MoE is not available here")
            return None
        p1 = prepare_experts(kind, w1, s1, z1)
        p2 = prepare_experts(kind, w2, s2, z2)
        held = {"w1": p1["weight"], "w2": p2["weight"], "s1": p1["scale"], "s2": p2["scale"],
                "z1": p1.get("zero"), "z2": p2.get("zero")}
        global_experts = getattr(block, "num_experts", e)
        out = cls(block, kind, fmt.group, held, global_experts if global_experts != e else 0)
        # the numerics probe: the same tokens through both kernels
        act = activation_spec(block._experts_activation)
        g = torch.Generator(device=device).manual_seed(0)
        x = torch.randn(probe_tokens, hidden, generator=g, device=device).to(torch.bfloat16)
        top_k = min(getattr(block, "num_experts_per_tok", 8) or 8, e)
        ids = torch.rand(probe_tokens, e, generator=g, device=device).topk(top_k, dim=-1).indices.to(torch.int32)
        weights = torch.rand(probe_tokens, top_k, generator=g, device=device).softmax(-1)
        want = fused_experts_w4a16(x, w1.view(torch.uint8), w2.view(torch.uint8), s1, s2, z1, z2, weights, ids, kind,
                                   fmt.group, act, False)
        got = out(x, weights, ids, False)
        err = ((got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-12)).item()
        if not (err < tol and torch.isfinite(got).all()):
            logger.warning("MoE: Marlin MoE disagrees with the Triton experts kernel (rel L2 %.3g over %d tokens); "
                           "these experts stay on Triton", err, probe_tokens)
            return None
        # one copy: the checkpoint layout's storage is no longer read
        for name in block._experts_views:
            param = getattr(block.experts, name)
            param.data = torch.empty(0, dtype=param.dtype, device=param.device)
        return out

    def __call__(self, flat: torch.Tensor, routing_weights: torch.Tensor, selected_experts: torch.Tensor,
                 skip_invalid: bool) -> torch.Tensor:
        from mstar.utils.situ import activation_spec

        h = self.held
        return fused_experts_marlin(flat, h["w1"], h["w2"], h["s1"], h["s2"], h["z1"], h["z2"], routing_weights,
                                    selected_experts, self.kind, self.group,
                                    activation_spec(self.block._experts_activation), skip_invalid,
                                    self.global_experts)
