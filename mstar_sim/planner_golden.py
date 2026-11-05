"""FlashInfer's own planner over a fixed grid of cases: the golden the stand-in's port is held to.

The stand-in sizes its attention workspaces with a port of FlashInfer's host planner
(``fake_libs/flashinfer/_paged.py``, ``mla.py``). This tool runs the real one, from the wheel of
the stand-in's ``__version__``, on every named case in :data:`CASES` and on each emulated GPU's
device properties, and writes what it answered to ``tests/planner_golden.json``:
``tests/test_device.py`` replays every case through the port on a CPU box.

It needs a CUDA toolkit (nvcc 12.8 or newer, and a host compiler it supports), so it runs on a dev
box, not in CI: regenerate the golden whenever the stand-in's FlashInfer version changes::

    python -m mstar_sim.planner_golden [--wheel <flashinfer_python-*.whl>]

How it runs the planner without a GPU: FlashInfer's planners (``scheduler.cuh``) are compiled
with ``-cudart none`` into one library, linked with ``-Bsymbolic`` against a stand-in CUDA
runtime of its own (so a CUDA torch's ``libcudart``, loaded globally, can't answer for it). That
runtime answers device attributes from the case's GPU, copies nothing, and answers the decode
planner's occupancy query with NVIDIA's own calculator (``cuda_occupancy.h``) for the case's
register count: a case input, since the stand-in's decode sizing takes none (0, the bound with
no register limit). A planner's answer is the smallest float and int workspace it plans in, and
the overflow it raises one byte short of each; or the error it raises however large they are.
"""

from __future__ import annotations

import argparse
import ctypes
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "tests" / "planner_golden.json"
# one emulated GPU per distinct set of the properties a planner reads (the H200 is an H100's)
GPUS = ("h100", "h100-pcie", "b200", "a100", "l40s", "4090")

STUB = r"""
#include <cstring>
#include <cuda_runtime_api.h>
#include <cuda_occupancy.h>

static int g_sm, g_major, g_minor, g_threads, g_smem, g_optin, g_regs_sm, g_regs;

extern "C" void golden_set_device(int sm, int major, int minor, int threads, int smem, int optin,
                                  int regs_sm, int regs) {
  g_sm = sm; g_major = major; g_minor = minor; g_threads = threads; g_smem = smem; g_optin = optin;
  g_regs_sm = regs_sm; g_regs = regs;
}

extern "C" {
cudaError_t cudaGetDevice(int* device) { *device = 0; return cudaSuccess; }
cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int) {
  switch (attr) {
    case cudaDevAttrMultiProcessorCount: *value = g_sm; return cudaSuccess;
    case cudaDevAttrComputeCapabilityMajor: *value = g_major; return cudaSuccess;
    case cudaDevAttrComputeCapabilityMinor: *value = g_minor; return cudaSuccess;
    case cudaDevAttrMaxSharedMemoryPerBlockOptin: *value = g_optin; return cudaSuccess;
    default: return cudaErrorInvalidValue;
  }
}
const char* cudaGetErrorString(cudaError_t error) { return error == cudaSuccess ? "no error" : "stub error"; }
cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) { return cudaSuccess; }
cudaError_t cudaLaunchKernel(const void*, dim3, dim3, void**, size_t, cudaStream_t) {
  return cudaErrorNotSupported;
}
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* blocks, const void*, int block_size,
                                                                   size_t dynamic_smem, unsigned int) {
  cudaOccDeviceProp prop;
  prop.computeMajor = g_major;
  prop.computeMinor = g_minor;
  prop.maxThreadsPerBlock = 1024;
  prop.maxThreadsPerMultiprocessor = g_threads;
  prop.regsPerBlock = 65536;
  prop.regsPerMultiprocessor = g_regs_sm;
  prop.warpSize = 32;
  prop.sharedMemPerBlock = 49152;
  prop.sharedMemPerMultiprocessor = g_smem;
  prop.numSms = g_sm;
  prop.sharedMemPerBlockOptin = g_optin;
  prop.reservedSharedMemPerBlock = 1024;
  cudaOccFuncAttributes attributes;  // a kernel as a fresh process plans it: no opt-in yet
  attributes.maxThreadsPerBlock = 1024;
  attributes.numRegs = g_regs;
  attributes.sharedSizeBytes = 0;
  attributes.maxDynamicSharedSizeBytes = 49152;
  cudaOccDeviceState state;
  cudaOccResult result;
  if (cudaOccMaxActiveBlocksPerMultiprocessor(&result, &prop, &attributes, &state, block_size, dynamic_smem)
      != CUDA_OCC_SUCCESS) {
    return cudaErrorInvalidValue;
  }
  *blocks = result.activeBlocksPerMultiprocessor;
  return cudaSuccess;
}
// what an nvcc-built object registers with the runtime: nothing here needs it
void** __cudaRegisterFatBinary(void*) { static void* handle; return &handle; }
void __cudaRegisterFatBinaryEnd(void**) {}
void __cudaUnregisterFatBinary(void**) {}
void __cudaRegisterFunction(void**, const char*, char*, const char*, int, void*, void*, void*, void*, int*) {}
void __cudaRegisterVar(void**, char*, char*, const char*, int, size_t, int, int) {}
char __cudaInitModule(void**) { return 0; }
unsigned __cudaPushCallConfiguration(dim3, dim3, size_t, void*) { return 0; }
cudaError_t __cudaPopCallConfiguration(dim3*, dim3*, size_t*, void*) { return cudaSuccess; }
}
"""

PLANNER = r"""
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <string>

using namespace flashinfer;
static std::string g_error;

extern "C" const char* golden_error() { return g_error.c_str(); }

#define GUARDED(call)                                                    \
  try {                                                                  \
    cudaError_t status = call;                                           \
    if (status != cudaSuccess) {                                         \
      g_error = std::string("cuda error ") + std::to_string((int)status); \
      return 2;                                                          \
    }                                                                    \
    return 0;                                                            \
  } catch (const std::exception& e) {                                    \
    g_error = e.what();                                                  \
    return 1;                                                            \
  }

extern "C" int golden_fa2(char* f, size_t fs, char* i, char* pl, size_t is, int32_t* qo, int32_t* kv,
                          uint32_t rows, uint32_t batch, uint32_t qo_heads, uint32_t kv_heads, uint32_t hd_qk,
                          uint32_t hd_vo, uint32_t page, int graph, int32_t window, int32_t fixed, int disable,
                          int64_t colocated, int64_t uniform, uint32_t kv_bytes) {
  PrefillPlanInfo info;
  GUARDED(PrefillPlan<int32_t>(f, fs, i, pl, is, info, qo, kv, rows, batch, qo_heads, kv_heads, hd_qk, hd_vo, page,
                               graph, 2, window, fixed, disable, colocated, uniform, nullptr, kv_bytes))
}

extern "C" int golden_fa3(char* f, size_t fs, char* i, char* pl, size_t is, int32_t* qo, int32_t* kv,
                          int32_t* kv_len, uint32_t rows, uint32_t batch, uint32_t qo_heads, uint32_t kv_heads,
                          uint32_t hd_qk, uint32_t hd_vo, uint32_t page, int causal, int graph) {
  PrefillPlanSM90Info info;
  GUARDED(PrefillSM90Plan<int32_t>(f, fs, i, pl, is, info, qo, kv, kv_len, rows, batch, qo_heads, kv_heads, hd_qk,
                                   hd_vo, page, causal, graph, 2, nullptr))
}

extern "C" int golden_mla(char* f, size_t fs, char* i, char* pl, size_t is, int32_t* qo, int32_t* kv,
                          int32_t* kv_len, uint32_t batch, uint32_t heads, uint32_t hd_o, int causal) {
  MLAPlanInfo info;
  size_t staged = 0;
  GUARDED(MLAPlan<int32_t>(f, fs, i, pl, is, info, staged, qo, kv, kv_len, batch, heads, hd_o, causal, nullptr))
}

template <uint32_t GROUP, uint32_t HEAD_DIM, typename DTypeKV>
int decode(char* f, size_t fs, char* i, char* pl, size_t is, int32_t* kv, uint32_t batch, uint32_t qo_heads,
           uint32_t page, int graph) {
  using Params = BatchDecodeParams<__nv_bfloat16, DTypeKV, __nv_bfloat16, int32_t>;
  using Variant = DefaultAttention<false, false, false, false>;
  DecodePlanInfo info;
  auto estimate = BatchDecodeWithPagedKVCacheWorkEstimationDispatched<GROUP, HEAD_DIM, PosEncodingMode::kNone,
                                                                     Variant, Params>;
  GUARDED((DecodePlan<HEAD_DIM, PosEncodingMode::kNone, Variant, Params>(f, fs, i, pl, is, info, kv, batch, qo_heads,
                                                                          page, graph, nullptr, estimate)))
}

extern "C" int golden_decode(char* f, size_t fs, char* i, char* pl, size_t is, int32_t* kv, uint32_t batch,
                             uint32_t qo_heads, uint32_t kv_heads, uint32_t page, int graph, uint32_t head_dim,
                             uint32_t kv_bytes) {
  uint32_t group = qo_heads / kv_heads;
%(dispatch)s
  g_error = "no decode instance for this group, head dim and KV size";
  return 3;
}
"""

DECODE_GROUPS = (1, 2, 3, 4, 6, 8, 16)
DECODE_HEAD_DIMS = (64, 128, 256)
DECODE_KV = {2: "__nv_bfloat16", 1: "__nv_fp8_e4m3"}


def _dispatch() -> str:
    return "\n".join(
        f"  if (group == {g} && head_dim == {d} && kv_bytes == {b}) "
        f"return decode<{g}, {d}, {t}>(f, fs, i, pl, is, kv, batch, qo_heads, page, graph);"
        for g, d, (b, t) in itertools.product(DECODE_GROUPS, DECODE_HEAD_DIMS, DECODE_KV.items()))


# --------------------------------------------------------------------------- cases

def _uniform(rows: int, n: int) -> list[int]:
    return [i * n for i in range(rows + 1)]


def _indptr(lengths) -> list[int]:
    out = [0]
    for n in lengths:
        out.append(out[-1] + n)
    return out


def fa2(name, pins, qo_lens, kv_pages, *, heads=(32, 8), hd=(128, 128), page=16, graph=False, rows=None,
        window=-1, fixed=-1, disable=False, uniform=0, kv_bytes=2, colocated=0, qo=None):
    """A batch prefill (paged and ragged; ragged is page 1) plan of ``qo_lens`` queries over
    ``kv_pages`` pages a request. ``rows`` is the graph's row budget (``total_num_rows``)."""
    qo = qo or _indptr(qo_lens)
    return {"name": f"fa2/{name}", "pins": pins, "planner": "fa2", "qo": qo, "kv": _indptr(kv_pages),
            "rows": rows or qo[-1], "heads": list(heads), "hd": list(hd), "page": page, "graph": graph,
            "window": window, "fixed": fixed, "disable": disable, "uniform": uniform, "kv_bytes": kv_bytes,
            "colocated": colocated}


def fa3(name, pins, qo_lens, kv_lens, *, heads=(32, 8), hd=(128, 128), page=16, graph=False, rows=None):
    qo = _indptr(qo_lens)
    return {"name": f"fa3/{name}", "pins": pins, "planner": "fa3", "qo": qo,
            "kv": _indptr([-(-n // page) for n in kv_lens]), "kv_len": list(kv_lens), "rows": rows or qo[-1],
            "heads": list(heads), "hd": list(hd), "page": page, "graph": graph}


def decode(name, pins, kv_pages, *, group=4, kv_heads=8, head_dim=128, page=16, graph=False, kv_bytes=2, regs=0):
    return {"name": f"decode/{name}", "pins": pins, "planner": "decode", "kv": _indptr(kv_pages),
            "heads": [group * kv_heads, kv_heads], "head_dim": head_dim, "page": page, "graph": graph,
            "kv_bytes": kv_bytes, "regs": regs}


def mla(name, pins, qo_lens, kv_lens, *, heads=16, hd_o=512):
    return {"name": f"mla/{name}", "pins": pins, "planner": "mla", "qo": _indptr(qo_lens),
            "kv": _indptr([-(-n // 64) for n in kv_lens]), "kv_len": list(kv_lens), "heads": heads, "hd_o": hd_o}


def _pairwise(axes: dict) -> list[dict]:
    """A small set of combinations of ``axes`` covering every pair of two axes' values
    (greedy: each pick covers the most pairs not yet covered)."""
    names = list(axes)
    uncovered = {(a, x, b, y) for a, b in itertools.combinations(names, 2) for x in axes[a] for y in axes[b]}
    picks = []
    while uncovered:
        best = max(itertools.product(*axes.values()), key=lambda combo: sum(
            (a, combo[names.index(a)], b, combo[names.index(b)]) in uncovered
            for a, b in itertools.combinations(names, 2)))
        picks.append(dict(zip(names, best, strict=True)))
        uncovered -= {(a, best[names.index(a)], b, best[names.index(b)]) for a, b in itertools.combinations(names, 2)}
    return picks


_FA2_SHAPES = {  # (queries, pages) a request
    "one-row": ([1], [1]),
    "long-prefill": ([4000], [250]),
    "mixed": ([1, 17, 64, 300], [3, 10, 40, 100]),
    "decode-rows": ([1] * 64, [512] * 64),
    "past-grid": ([1] * 512, [1] * 512),
}


def _fa2_cases() -> list[dict]:
    cases = [
        fa2("one-row", "a single query plans with no split", [1], [1]),
        fa2("long-prefill-splits", "a long request over few heads splits its KV", [4000], [250], heads=(8, 8)),
        fa2("past-the-grid", "a batch past the grid doesn't split", [1] * 512, [1] * 512),
        fa2("empty-request", "a request with no queries plans", [0, 5], [0, 3]),
        fa2("negative-length", "a negative query length is refused", None, [2, 2], qo=[0, 8, 4]),
        fa2("graph-rows", "under a graph, tiles are sized for the row budget", [3, 9], [4, 4], graph=True, rows=64),
        fa2("graph-uniform", "under a graph, uniform_q_len sizes tiles by the uniform length", [4] * 8, [6] * 8,
            graph=True, uniform=4),
        fa2("graph-uniform-mismatch", "a request that breaks uniform_q_len is refused", [4, 5], [6, 6], graph=True,
            uniform=4),
        fa2("uniform-without-graph", "uniform_q_len is ignored outside a graph", [4, 5], [6, 6], uniform=4),
        fa2("window", "a window bounds the KV the split partitions", [16, 16], [200, 200], window=128),
        fa2("window-page1", "a window over single-token pages", [16, 16], [3000, 3000], window=128, page=1),
        fa2("fixed-split", "a fixed split size chunks the KV", [8, 8], [64, 64], fixed=256),
        fa2("fixed-split-graph-refused", "a fixed split past the graph's padded batch is refused", [64] * 8,
            [400] * 8, graph=True, fixed=16),
        fa2("disable-split", "disable_split_kv keeps no partials", [4000], [250], heads=(8, 8), disable=True),
        fa2("disable-beats-fixed", "disable_split_kv wins over a fixed split", [4000], [250], heads=(8, 8),
            disable=True, fixed=256),
        fa2("colocated", "colocated decode CTAs shrink the grid prefill splits for", [4000], [250], heads=(8, 8),
            colocated=128),
        fa2("group-7", "a group size that doesn't divide the tile", [100, 7], [20, 20], heads=(28, 4)),
        fa2("heads-40", "a KV head count that doesn't divide the grid", [100, 7], [20, 20], heads=(40, 40)),
        fa2("hd256-short", "vo 256 at a short packed length takes tile 16", [1, 1], [50, 50], heads=(16, 16),
            hd=(256, 256)),
        fa2("hd256-long", "vo 256 at a long packed length takes tile 64", [500], [50], hd=(256, 256)),
        fa2("hd448-256-bf16-short", "the shared-memory probe: (448, 256) at 2-byte KV takes tile 64 on 99 KiB parts",
            [1, 1], [50, 50], heads=(16, 16), hd=(448, 256)),
        fa2("hd448-256-fp8-short", "the shared-memory probe: (448, 256) at 1-byte KV fits tile 16 everywhere",
            [1, 1], [50, 50], heads=(16, 16), hd=(448, 256), kv_bytes=1),
        fa2("hd512-short", "vo >= 512 at a short packed length takes tile 16", [2, 2], [50, 50], heads=(16, 16),
            hd=(512, 512)),
        fa2("hd512-long", "vo >= 512 at a long packed length takes tile 32", [300], [50], hd=(512, 512)),
        fa2("hd576-256", "qk >= 512 takes tile 16 at any length", [300], [50], hd=(576, 256)),
        fa2("hd192-128", "unequal head dims under 256 size tmp_v by vo", [300, 5], [50, 50], hd=(192, 128)),
    ]
    axes = {"shape": list(_FA2_SHAPES), "heads": [(32, 8), (28, 4), (40, 40)], "hd": [(128, 128), (256, 256)],
            "page": [1, 16], "graph": [False, True], "window": [-1, 128], "split": ["auto", "fixed", "off"]}
    for i, combo in enumerate(_pairwise(axes)):
        qo_lens, pages = _FA2_SHAPES[combo["shape"]]
        rows = sum(qo_lens) + 16 * combo["graph"]
        label = "-".join(f"{k}{v}" if not isinstance(v, tuple) else f"{k}{v[0]}x{v[1]}" for k, v in combo.items())
        cases.append(fa2(f"pairwise-{i:02d}", label, qo_lens, pages, heads=combo["heads"], hd=combo["hd"],
                         page=combo["page"], graph=combo["graph"], rows=rows, window=combo["window"],
                         fixed=256 if combo["split"] == "fixed" else -1, disable=combo["split"] == "off"))
    return cases


def _fa3_cases() -> list[dict]:
    return [
        fa3("eager", "one work a query tile of one head", [100, 30], [500, 30]),
        fa3("graph", "under a graph, works are sized for the row budget", [100, 30], [500, 30], graph=True, rows=256),
        fa3("hd64-eager", "vo 64 takes tile 192", [500, 30], [500, 30], hd=(64, 64)),
        fa3("hd64-graph", "vo 64 under a graph", [500, 30], [500, 30], hd=(64, 64), graph=True, rows=1024),
        fa3("many-tiles-eager", "past 4096 tiles a head, the heads share a schedule", [600_000], [1000]),
        fa3("many-tiles-graph", "past 4096 tiles a head under a graph", [500_000], [16], graph=True),
        fa3("many-heads-graph", "many heads under a graph", [20_000], [16], heads=(80, 8), graph=True),
        fa3("hd256", "vo 256 takes tile 128", [100, 30], [500, 30], hd=(256, 256)),
    ]


def _decode_cases() -> list[dict]:
    cases = [decode(f"g{g}-hd{d}-kv{b}", "a batch that splits, per kernel shape", [37, 5], group=g, head_dim=d,
                    kv_bytes=b)
             for d in DECODE_HEAD_DIMS for g in DECODE_GROUPS for b in DECODE_KV]
    shapes = {"no-split": [2] * 1024, "short": [1] * 8, "mixed": [1, 40, 7, 300, 2]}
    for (shape, pages), graph, page in itertools.product(shapes.items(), (False, True), (1, 16)):
        cases.append(decode(f"{shape}-graph{int(graph)}-page{page}", f"batch shape {shape}", pages, graph=graph,
                            page=page))
    cases.append(decode("kv-heads-1", "one KV head", [37, 5], group=8, kv_heads=1))
    for regs in (0, 32, 33, 40, 48, 49, 64, 72, 96, 128, 168, 256, 257):  # either side of the rounding, and the cap
        cases.append(decode(f"regs{regs}-g2-hd128", "registers bound the graph's grid (Qwen3-ASR's shape)",
                            [37, 5, 300], group=2, head_dim=128, regs=regs, graph=True))
        cases.append(decode(f"regs{regs}-g1-hd64", "registers against shared memory (Whisper's shape)",
                            [37, 5, 300], group=1, head_dim=64, regs=regs, graph=True))
    cases.append(decode("regs48-g6-hd256", "6 warps a block: registers go by sub-partition, not the whole SM",
                        [37, 5, 300], group=6, head_dim=256, regs=48, graph=True))
    cases.append(decode("regs168-g16-hd256", "a block past the registers one block may hold: no grid", [37, 5, 300],
                        group=16, head_dim=256, regs=168, graph=True))
    return cases


def _mla_cases() -> list[dict]:
    return [
        mla("decode", "one query a request: no clusters", [1] * 32, [1000] * 32),
        mla("prefill", "long queries take clusters of 2", [200, 300], [400, 300]),
        mla("one", "a single request", [1], [10]),
        mla("heads-128", "many heads pack past 64 a request", [1] * 8, [500] * 8, heads=128),
        mla("hd256", "another output head dim", [1] * 8, [500] * 8, hd_o=256),
        mla("mixed", "mixed lengths", [1, 70, 5], [100, 900, 5]),
    ]


CASES = [*_fa2_cases(), *_fa3_cases(), *_decode_cases(), *_mla_cases()]


# --------------------------------------------------------------------------- build and run

def _flashinfer_version() -> str:
    text = (HERE / "fake_libs" / "flashinfer" / "__init__.py").read_text()
    return re.search(r'^__version__ = "([^"]+)"', text, re.M).group(1)


def _headers(wheel: str | None, version: str, scratch: Path) -> Path:
    """The wheel's ``data`` directory (its include and cccl trees)."""
    if wheel is None:
        subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "-d", str(scratch),
                        f"flashinfer-python=={version}"], check=True)
        wheel = str(next(scratch.glob("flashinfer_python-*.whl")))
    with zipfile.ZipFile(wheel) as z:
        z.extractall(scratch / "wheel", [n for n in z.namelist()
                                         if n.startswith(("flashinfer/data/include/", "flashinfer/data/cccl/"))])
    return scratch / "wheel" / "flashinfer" / "data"


def build(data: Path, scratch: Path) -> ctypes.CDLL:
    """FlashInfer's planners and the stand-in runtime, as one library (serial builds: nvcc on
    the decode kernels takes over a GB)."""
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    cuda_include = Path(nvcc).resolve().parent.parent / "include"
    (scratch / "planner.cu").write_text(PLANNER % {"dispatch": _dispatch()})
    (scratch / "stub.cpp").write_text(STUB)
    include = [f"-I{data / 'include'}", f"-I{data / 'cccl' / 'libcudacxx' / 'include'}",
               f"-I{data / 'cccl' / 'cub'}", f"-I{data / 'cccl' / 'thrust'}"]
    # PTX only (no ptxas): the kernels are never run, only their addresses taken
    subprocess.run([nvcc, "-std=c++17", "--expt-relaxed-constexpr", "-x", "cu", "-gencode",
                    "arch=compute_80,code=compute_80", "-cudart", "none", "-Xcompiler", "-fPIC", *include, "-c",
                    str(scratch / "planner.cu"), "-o", str(scratch / "planner.o")], check=True)
    subprocess.run(["g++", "-std=c++17", "-fPIC", f"-I{cuda_include}", "-c", str(scratch / "stub.cpp"),
                    "-o", str(scratch / "stub.o")], check=True)
    library = scratch / "golden.so"
    subprocess.run(["g++", "-shared", "-o", str(library), str(scratch / "planner.o"), str(scratch / "stub.o"),
                    "-Wl,-Bsymbolic", "-Wl,--no-undefined"], check=True)
    lib = ctypes.CDLL(str(library))
    lib.golden_error.restype = ctypes.c_char_p
    return lib


_MESSAGE = re.compile(r"^Error in function '.*?' at .*?:\d+: (.*)$", re.S)
_BASE = 1 << 40  # where the device buffers seem to be: aligned, never dereferenced


def _message(raw: bytes) -> str:
    text = raw.decode()
    match = _MESSAGE.match(text)
    return match.group(1) if match else text


def _ints(values):
    return (ctypes.c_int32 * max(len(values), 1))(*values)


def _call(lib, case: dict, float_bytes: int, int_bytes: int, pinned) -> tuple[int, str]:
    """Plan ``case`` in workspaces of these sizes: (status, error message)."""
    f, i = ctypes.c_void_p(_BASE), ctypes.c_void_p(_BASE * 2)
    head = (f, ctypes.c_size_t(float_bytes), i, pinned, ctypes.c_size_t(int_bytes))
    qo = _ints(case.get("qo", []))
    kv = _ints(case["kv"])
    kind = case["planner"]
    if kind == "fa2":
        status = lib.golden_fa2(*head, qo, kv, *map(ctypes.c_uint32, (
            case["rows"], len(case["qo"]) - 1, *case["heads"], *case["hd"], case["page"])),
            ctypes.c_int(case["graph"]), ctypes.c_int32(case["window"]), ctypes.c_int32(case["fixed"]),
            ctypes.c_int(case["disable"]), ctypes.c_int64(case["colocated"]), ctypes.c_int64(case["uniform"]),
            ctypes.c_uint32(case["kv_bytes"]))
    elif kind == "fa3":
        status = lib.golden_fa3(*head, qo, kv, _ints(case["kv_len"]), *map(ctypes.c_uint32, (
            case["rows"], len(case["qo"]) - 1, *case["heads"], *case["hd"], case["page"])), ctypes.c_int(1),
            ctypes.c_int(case["graph"]))
    elif kind == "mla":
        status = lib.golden_mla(*head, qo, kv, _ints(case["kv_len"]), *map(ctypes.c_uint32, (
            len(case["qo"]) - 1, case["heads"], case["hd_o"])), ctypes.c_int(1))
    else:
        status = lib.golden_decode(*head, kv, *map(ctypes.c_uint32, (
            len(case["kv"]) - 1, *case["heads"], case["page"])), ctypes.c_int(case["graph"]),
            ctypes.c_uint32(case["head_dim"]), ctypes.c_uint32(case["kv_bytes"]))
    if status > 1:
        raise RuntimeError(f"{case['name']}: {lib.golden_error().decode()}")
    return status, _message(lib.golden_error()) if status else ""


def _smallest(fits, high: int) -> int:
    """The smallest size in [0, high] for which ``fits`` holds (it holds at ``high``)."""
    low = -1
    while high - low > 1:
        mid = (low + high) // 2
        if fits(mid):
            high = mid
        else:
            low = mid
    return high


def answer(lib, case: dict) -> dict:
    """What FlashInfer's planner answers for ``case`` on the device set last."""
    room = 1 << 26
    pinned = ctypes.create_string_buffer(room)
    big_float = 1 << 40
    status, message = _call(lib, case, big_float, room, pinned)
    if status and message.startswith("Buffer overflow"):
        raise RuntimeError(f"{case['name']}: needs more than {room} int workspace bytes: {message}")
    if status:
        return {"error": message}
    float_bytes = _smallest(lambda n: _call(lib, case, n, room, pinned)[0] == 0, big_float)
    int_bytes = _smallest(lambda n: _call(lib, case, big_float, n, pinned)[0] == 0, room)
    out = {"float": float_bytes, "int": int_bytes}
    if float_bytes:
        out["float_short"] = _call(lib, case, float_bytes - 1, room, pinned)[1]
    if int_bytes:
        out["int_short"] = _call(lib, case, big_float, int_bytes - 1, pinned)[1]
    return out


def device_properties() -> dict[str, dict]:
    """The device properties a planner reads, per emulated GPU, as the emulator reports them."""
    from mstar_sim import cuda, gpus

    keys = ("multi_processor_count", "major", "minor", "max_threads_per_multi_processor",
            "shared_memory_per_multiprocessor", "shared_memory_per_block", "shared_memory_per_block_optin",
            "regs_per_multiprocessor")
    out = {}
    for alias in GPUS:
        props = cuda.properties(gpus.lookup(alias))
        out[alias] = {k: getattr(props, k) for k in keys}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m mstar_sim.planner_golden", description=__doc__.split("\n\n")[0])
    ap.add_argument("--wheel", help="the flashinfer-python wheel (default: pip download the stand-in's version)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    version = _flashinfer_version()
    devices = device_properties()
    with tempfile.TemporaryDirectory(prefix="planner_golden-") as tmp:
        scratch = Path(tmp)
        lib = build(_headers(a.wheel, version, scratch), scratch)
        answers = {}
        for alias, p in devices.items():
            for case in CASES:
                lib.golden_set_device(p["multi_processor_count"], p["major"], p["minor"],
                                      p["max_threads_per_multi_processor"], p["shared_memory_per_multiprocessor"],
                                      p["shared_memory_per_block_optin"], p["regs_per_multiprocessor"],
                                      case.get("regs", 0))
                answers.setdefault(case["name"], {})[alias] = answer(lib, case)
    nvcc = subprocess.run([shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc", "--version"], capture_output=True,
                          text=True, check=True).stdout.strip().splitlines()[-1]
    golden = {"flashinfer": version, "generator": f"python -m mstar_sim.planner_golden ({nvcc})",
              "devices": devices, "cases": [{**case, "answers": answers[case["name"]]} for case in CASES]}
    cases = golden.pop("cases")  # a case a line, so a regeneration's diff names what moved
    head = json.dumps(golden, indent=1)[:-2]
    Path(a.out).write_text(head + ',\n "cases": [\n' + ",\n".join(json.dumps(c) for c in cases) + "\n ]\n}\n")
    print(f"{len(CASES)} cases x {len(devices)} GPUs -> {a.out}")


if __name__ == "__main__":
    os.environ.setdefault("MSTAR_SIM", "0")
    main()
