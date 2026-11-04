"""Measured kernel times from NVIDIA AIConfigurator's per-op tables.

AIConfigurator (github.com/ai-dynamo/aiconfigurator, Apache-2.0) ships latencies
measured on real GPUs under ``src/aiconfigurator/systems/data/<gpu>/``: GEMMs
over (m, n, k), context and generation attention over (heads, sequence, batch)
per KV geometry, GDN and causal-conv kernels over (batch, sequence), NCCL and
custom all-reduce over message size. :mod:`mstar_sim.cost` asks this module
first and keeps its roofline for whatever the tables don't cover (a ``None``
answer).

Every family is answered the way AIConfigurator's ``perf_interp`` engine does,
through its speed of light ``SOL = max(flops / peak, bytes / bandwidth)``
(datasheet peaks, no efficiency factors) and the utilization ``util = SOL /
latency`` of measured points:

- an exact hit returns the measured latency (kernel-heuristic cliffs, such as
  a GEMM's tile change between m = 1024 and 1025, survive as collected);
- inside the grid each axis brackets the query and blends the two neighbours'
  util with weights in log space, and the result ``SOL(query) / util`` is kept
  between the two neighbours' latencies;
- outside the grid the boundary point's util is held and SOL carries the
  growth;
- nothing goes under the smallest latency measured in the slice (the family's
  launch-and-tail floor: ~4 us for a small GEMM on H100, ~11 us for decode
  attention).

A GEMM shape (n, k) that was never collected borrows util from the nearest
collected shapes in log2 space (inverse-distance weights, within 2 octaves; a
shape past every collected one on a single axis, such as an LM head's vocab,
anchors on the frontier shapes). Unlike AIConfigurator, which blends raw
latency (sqrt-latency along context sequence) between two brackets, this
blends util there too: one rule inside and outside the grid.

The newest version of each table is read once per GPU and cached as a small
``.npz`` under ``~/.cache/mstar_sim`` keyed by the files' mtimes, so a process
pays neither pandas' import nor the parquet scan; each family is built on its
first query.
"""

from __future__ import annotations

import bisect
import glob
import hashlib
import json
import math
import os
import re
import threading
from collections.abc import Callable

import numpy as np

from mstar_sim import clock, gpus

# the AIConfigurator commit whose table layout this reads (Apache-2.0 data)
AIC_REPO = "https://github.com/ai-dynamo/aiconfigurator"
AIC_COMMIT = "6cd6f9686ab7de6e8fb28171d6e350ee0aa4ed39"
AIC_CHECKOUT = os.path.expanduser("~/.cache/mstar_sim/aiconfigurator")
DEFAULT_DATA = os.path.join(AIC_CHECKOUT, "src/aiconfigurator/systems/data")
FETCH = f"git clone {AIC_REPO} {AIC_CHECKOUT} && git -C {AIC_CHECKOUT} checkout {AIC_COMMIT}"
_FORMAT = 2

# mstar_sim GPU -> AIConfigurator system directory
SYSTEMS = {
    "NVIDIA H100 80GB HBM3": "h100_sxm",
    "NVIDIA H100 PCIe": "h100_pcie",
    "NVIDIA H200": "h200_sxm",
    "NVIDIA B200": "b200_sxm",
    "NVIDIA A100-SXM4-80GB": "a100_sxm",
    "NVIDIA L40S": "l40s",
}
# Per-direction GPU-to-GPU bandwidth (NVLink, else PCIe), bytes/s: the collectives' SOL.
LINK_BW = {
    "NVIDIA H100 80GB HBM3": 450e9, "NVIDIA H200": 450e9, "NVIDIA B200": 900e9,
    "NVIDIA A100-SXM4-80GB": 300e9, "NVIDIA H100 PCIe": 64e9, "NVIDIA L40S": 32e9,
}
# FlashInfer first, then the closest kernels AIConfigurator collected for SGLang
_ATTENTION_LANES = ("flashinfer", "fa3", "trtllm_mha", "flash_attention", "triton")
_GDN_KERNELS = {
    "chunk_gated_delta_rule": "gdn",
    "fused_recurrent_gated_delta_rule_packed_decode": "gdn",
    "fused_recurrent_gated_delta_rule": "gdn",
    "causal_conv1d_fn": "conv",
    "causal_conv1d_update": "conv",
}
_SITE_GATE = 2.0  # octaves (log2 distance) a GEMM shape may borrow util across
_SITE_NEIGHBOURS = 4
_MS = 1e-3


def data_dir() -> str:
    return os.environ.get("MSTAR_SIM_AIC_DATA", DEFAULT_DATA)


class Tables:
    """The measured tables of one GPU; ``None`` from a query means not covered."""

    def __init__(self, gpu: gpus.GPU, root: str | None = None) -> None:
        self.gpu = gpu
        self.root = root or data_dir()
        self.system = SYSTEMS.get(gpu.name)
        self._raw: dict[str, np.ndarray] | None = None
        self._built: dict[str, object] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------- queries

    def gemm_seconds(self, m: int, n: int, k: int, dtype: str = "bf16", batch: int = 1) -> float | None:
        """``batch`` GEMMs of [m, k] x [k, n] in one launch (``dtype`` "bf16" or "fp8")."""
        sites = self._family("gemm").get(dtype)
        if sites is None or min(m, n, k) <= 0:
            return None
        el = 1 if dtype == "fp8" else 2
        peak = self.gpu.peak(dtype)
        bw = self.gpu.hbm_bw

        def sol(mm, nn, kk):
            return max(2.0 * mm * nn * kk / peak, el * (mm * nn + mm * kk + nn * kk) / bw)

        rows = m * batch  # the util of one GEMM as tall as the batch
        want = batch * sol(m, n, k)
        own = sites.curves.get((n, k))
        if own is not None:
            curve, floor = own
            lat = _resolve(curve, (rows,), lambda p: sol(p[0], n, k))
            if lat is None:
                return None
            return max(floor, lat * want / sol(rows, n, k))
        utils, weights, floors = [], [], []
        for d, (sn, sk) in sites.neighbours(n, k):
            curve, floor = sites.curves[(sn, sk)]
            lat = _resolve(curve, (rows,), lambda p, sn=sn, sk=sk: sol(p[0], sn, sk))
            if lat:
                utils.append(sol(rows, sn, sk) / lat)
                weights.append(1.0 / max(d, 1e-6) ** 2)
                floors.append(floor)
        if not utils:
            return None
        util = sum(u * w for u, w in zip(utils, weights, strict=True)) / sum(weights)
        return max(min(floors), want / util)

    def attention_seconds(
        self, phase: str, batch: int, q_tokens: int, kv_tokens: int, pairs: int, heads: int,
        kv_heads: int, head_dim: int, kv_dtype: str = "bf16",
    ) -> float | None:
        """One attention kernel over ``batch`` requests of one phase.

        ``phase`` is "generation" (one query a request; ``kv_tokens`` = Σ kv
        lengths) or "context" (``q_tokens`` = Σ new tokens, ``kv_tokens`` = Σ
        (prefix + new), ``pairs`` = Σ visible (query, key) pairs, e.g.
        e·p + e(e+1)/2 causal). Sums, not means, so ragged lengths cost what
        they do."""
        if batch <= 0 or q_tokens <= 0:
            return 0.0
        slices = self._family("attention")
        found = slices.get((phase, kv_dtype, kv_heads, head_dim))
        if found is None:
            return None
        grid, floor = found
        kv_el = 1 if kv_dtype == "fp8" else 2
        peak, bw = self.gpu.peak("bf16"), self.gpu.hbm_bw

        def sol(q, kv, pr):
            return max(4.0 * pr * heads * head_dim / peak,
                       (4 * q * heads * head_dim + 2 * kv * kv_heads * head_dim * kv_el) / bw)

        if phase == "generation":
            s = kv_tokens / batch
            point = (heads, batch, s)

            def grid_sol(p):
                return max(4.0 * p[1] * p[2] * p[0] * head_dim / peak,
                           (4 * p[1] * p[0] * head_dim + 2 * p[1] * p[2] * kv_heads * head_dim * kv_el) / bw)
        else:
            # the causal prefill of equal requests with as many pairs: s(s+1)/2 = pairs/batch
            s = (math.sqrt(1.0 + 8.0 * pairs / batch) - 1.0) / 2.0
            point = (heads, max(s, 1.0), batch)

            def grid_sol(p):
                h, ss, b = p
                return max(4.0 * b * ss * (ss + 1) / 2 * h * head_dim / peak,
                           (4 * b * ss * h * head_dim + 2 * b * ss * kv_heads * head_dim * kv_el) / bw)
        lat = _resolve(grid, point, grid_sol)
        if lat is None:
            return None
        return max(floor, lat * sol(q_tokens, kv_tokens, pairs) / grid_sol(point))

    def gdn_seconds(
        self, kernel: str, batch: int, tokens: int, *, k_heads: int = 0, v_heads: int = 0,
        head_k_dim: int = 128, head_v_dim: int = 128, channels: int = 0,
    ) -> float | None:
        """A gated-delta-rule kernel (``chunk_gated_delta_rule``,
        ``fused_recurrent_gated_delta_rule_packed_decode``: heads and head dims)
        or a causal conv (``causal_conv1d_fn``, ``causal_conv1d_update``:
        ``channels``) over ``batch`` requests and ``tokens`` tokens in all."""
        kind = _GDN_KERNELS.get(kernel)
        if kind is None or batch <= 0 or tokens <= 0:
            return None if kind is None else 0.0
        geometry = (channels,) if kind == "conv" else (k_heads, v_heads, head_k_dim, head_v_dim)
        slices = self._family("gdn")
        found = slices.get((kernel, *geometry))
        if found is None and kernel.endswith("_packed_decode"):  # older collections: unpacked decode
            found = slices.get((kernel[: -len("_packed_decode")], *geometry))
        if found is None:
            return None
        grid, floor = found
        sol = _gdn_sol(self.gpu, kind, geometry)
        point = (batch, tokens / batch)
        lat = _resolve(grid, point, sol)
        return None if lat is None else max(floor, lat)

    def collective_seconds(self, op: str, num_gpus: int, nbytes: int, graph: bool = True) -> float | None:
        """``op`` ("all_reduce", "all_gather", "reduce_scatter", "alltoall") of a
        ``nbytes`` 16-bit message over ``num_gpus`` GPUs of one node. An
        all-reduce takes the faster of NCCL and SGLang's custom all-reduce
        (its graph-captured or eager timing, per ``graph``)."""
        if num_gpus <= 1 or nbytes <= 0:
            return 0.0
        found = self._family("comm")
        link = LINK_BW.get(self.gpu.name)
        if link is None:
            return None
        factor = 2.0 * (num_gpus - 1) / num_gpus if op == "all_reduce" else (num_gpus - 1) / num_gpus
        elements = nbytes / 2

        def sol(p):
            return factor * p[1] * 2 / link

        best = None
        for source in (("nccl", op), ("custom", op, "graph" if graph else "eager")):
            grid = found.get(source)
            if grid is None:
                continue
            lat = _resolve(grid[0], (num_gpus, elements), sol)
            if lat is not None:
                lat = max(grid[1], lat)
                best = lat if best is None else min(best, lat)
        return best

    # ------------------------------------------------------------- loading

    def _family(self, name: str):
        built = self._built.get(name)
        if built is None:
            # emulator overhead, not the simulated host's time: it must not open a device gap
            with self._lock, clock.computing():
                built = self._built.get(name)
                if built is None:
                    raw = self._load()
                    built = self._built[name] = _BUILD[name](raw) if raw else {}
        return built

    def _load(self) -> dict[str, np.ndarray] | None:
        if self._raw is None:
            self._raw = _load_raw(self.root, self.system) if self.system else {}
        return self._raw


# --------------------------------------------------------------- resolver


Grid = tuple  # (sorted keys, children): a child is a Grid or a latency in seconds


def _resolve(node: Grid, query: tuple, sol: Callable[[tuple], float]) -> float | None:
    """Latency at ``query`` (one coordinate per axis) from a ragged grid; see the module doc."""
    return _at(node, 0, list(query), sol)


def _at(node, depth: int, point: list, sol) -> float | None:
    if depth == len(point):
        return node
    keys, kids = node
    c = point[depth]
    i = bisect.bisect_left(keys, c)
    if i < len(keys) and keys[i] == c:
        return _at(kids[i], depth + 1, point, sol)

    def branch(j):
        p = point.copy()
        p[depth] = keys[j]
        lat = _at(kids[j], depth + 1, p, sol)
        return None if lat is None else (lat, sol(p))

    if i == 0 or i == len(keys):  # outside: hold the boundary util
        found = branch(0 if i == 0 else len(keys) - 1)
        return None if found is None else found[0] * sol(point) / found[1]
    lo, hi = branch(i - 1), branch(i)
    if lo is None or hi is None:  # a ragged corner: hold the survivor's util
        found = lo or hi
        return None if found is None else found[0] * sol(point) / found[1]
    w = math.log(c / keys[i - 1]) / math.log(keys[i] / keys[i - 1])
    util = (1 - w) * lo[1] / lo[0] + w * hi[1] / hi[0]
    lat = sol(point) / util
    return min(max(lat, min(lo[0], hi[0])), max(lo[0], hi[0]))


def _grid(rows: dict[tuple, float]) -> Grid:
    """A nested grid from {coordinates: latency}; the first row of a point wins."""
    groups: dict = {}
    for coords, lat in rows.items():
        groups.setdefault(coords[0], {})[coords[1:]] = lat
    keys = sorted(groups)
    if not next(iter(rows))[1:]:
        return keys, [groups[k][()] for k in keys]
    return keys, [_grid(groups[k]) for k in keys]


def _slices(keys: list[tuple], coords: list[tuple], lat: np.ndarray) -> dict:
    """{slice key: (grid, floor)} from parallel rows of slice keys and grid coordinates."""
    rows: dict[tuple, dict] = {}
    for key, c, t in zip(keys, coords, lat.tolist(), strict=True):
        rows.setdefault(key, {}).setdefault(c, t * _MS)
    return {key: (_grid(r), min(r.values())) for key, r in rows.items()}


class _Sites:
    """GEMM shapes (n, k), each with its measured m-curve."""

    def __init__(self, curves: dict) -> None:
        self.curves = curves
        self.logs = [(math.log2(n), math.log2(k), (n, k)) for n, k in curves]
        self.max_n = max(n for n, _ in curves)
        self.max_k = max(k for _, k in curves)

    def neighbours(self, n: int, k: int) -> list[tuple[float, tuple]]:
        ln, lk = math.log2(n), math.log2(k)
        over_n, over_k = n > self.max_n, k > self.max_k
        ranked = []
        for sn, sk, site in self.logs:
            dn, dk = sn - ln, sk - lk
            if over_n != over_k:  # past every site on one axis: gate on the other only
                gate = abs(dk) if over_n else abs(dn)
            else:
                gate = math.hypot(dn, dk)
            if gate <= _SITE_GATE:
                ranked.append((math.hypot(dn, dk), site))
        ranked.sort()
        return ranked[:_SITE_NEIGHBOURS]


def _build_gemm(raw) -> dict:
    out = {}
    for dtype in ("bf16", "fp8"):
        sel = raw["gemm.dtype"] == dtype
        if not sel.any():
            continue
        m, n, k = (raw[f"gemm.{a}"][sel].tolist() for a in "mnk")
        lat = raw["gemm.latency"][sel]
        curves = _slices(list(zip(n, k, strict=True)), [(x,) for x in m], lat)
        out[dtype] = _Sites(curves)
    return out


def _build_attention(raw) -> dict:
    out = {}
    for phase in ("generation", "context"):
        p = f"{phase}."
        if p + "latency" not in raw:
            continue
        lanes = raw[p + "lane"].tolist()
        geometry = list(zip(raw[p + "kv_dtype"].tolist(), raw[p + "kv_heads"].tolist(),
                            raw[p + "head_dim"].tolist(), strict=True))
        # the first lane (in _ATTENTION_LANES order, unknown ones last) holding a geometry serves it
        rank = {lane: i for i, lane in enumerate(_ATTENTION_LANES)}
        serving: dict = {}
        for lane, g in set(zip(lanes, geometry, strict=True)):
            if g not in serving or rank.get(lane, len(rank)) < rank.get(serving[g], len(rank)):
                serving[g] = lane
        keep = [i for i, (lane, g) in enumerate(zip(lanes, geometry, strict=True)) if serving[g] == lane]
        heads, batch, seq = ([col[i] for i in keep] for col in (raw[p + a].tolist() for a in ("heads", "batch", "seq")))
        axes = (heads, batch, seq) if phase == "generation" else (heads, seq, batch)
        sliced = _slices([(phase, *geometry[i]) for i in keep], list(zip(*axes, strict=True)),
                         raw[p + "latency"][keep])
        out.update(sliced)
    return out


def _build_gdn(raw) -> dict:
    if "gdn.latency" not in raw:
        return {}
    kernels = raw["gdn.kernel"].tolist()
    nk, nv, dk, dv = (raw[f"gdn.{a}"].tolist() for a in ("nk", "nv", "dk", "dv"))
    keys = [
        (kern, 2 * a * c + b * d) if _GDN_KERNELS.get(kern) == "conv" else (kern, a, b, c, d)
        for kern, a, b, c, d in zip(kernels, nk, nv, dk, dv, strict=True)
    ]
    coords = list(zip(raw["gdn.batch"].tolist(), raw["gdn.seq"].tolist(), strict=True))
    return _slices(keys, coords, raw["gdn.latency"])


def _build_comm(raw) -> dict:
    out = {}
    if "nccl.latency" in raw:
        keys = [("nccl", op) for op in raw["nccl.op"].tolist()]
        coords = list(zip(raw["nccl.gpus"].tolist(), raw["nccl.size"].tolist(), strict=True))
        out.update(_slices(keys, coords, raw["nccl.latency"]))
    if "custom.latency" in raw:
        keys = [("custom", "all_reduce", b) for b in raw["custom.backend"].tolist()]
        coords = list(zip(raw["custom.gpus"].tolist(), raw["custom.size"].tolist(), strict=True))
        out.update(_slices(keys, coords, raw["custom.latency"]))
    return out


_BUILD = {"gemm": _build_gemm, "attention": _build_attention, "gdn": _build_gdn, "comm": _build_comm}


def _gdn_sol(gpu: gpus.GPU, kind: str, geometry: tuple) -> Callable[[tuple], float]:
    peak, bw = gpu.peak("bf16"), gpu.hbm_bw
    if kind == "conv":
        (ch,) = geometry

        def sol(p):
            b, s = p
            return max(8.0 * b * s * ch / peak, (4 * b * s * ch + 12 * b * ch) / bw)
    else:
        nk, nv, dk, dv = geometry

        def sol(p):
            b, s = p
            return max(6.0 * b * s * nv * dk * dv / peak,
                       (4 * b * s * (nk * dk + nv * dv) + 8 * b * nv * dk * dv) / bw)
    return sol


# --------------------------------------------------------------- the files


def _version(path: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", os.path.basename(os.path.dirname(path))))


def _newest(root: str, system: str, family: str, frameworks: tuple, name: str) -> str | None:
    for fw in frameworks:
        found = glob.glob(os.path.join(root, system, family, fw, "*", name))
        if found:
            return max(found, key=_version)
    return None


def _sources(root: str, system: str) -> dict[str, str]:
    paths = {
        "gemm": _newest(root, system, "gemm", ("sglang", "vllm", "trtllm"), "gemm_perf.parquet"),
        "generation": _newest(root, system, "attention", ("sglang", "vllm", "trtllm"),
                              "generation_attention_perf.parquet"),
        "context": _newest(root, system, "attention", ("sglang", "vllm", "trtllm"),
                           "context_attention_perf.parquet"),
        "gdn": _newest(root, system, "linear_attention", ("sglang", "vllm", "trtllm"), "gdn_perf.parquet"),
        "nccl": _newest(root, system, "comm", ("nccl",), "nccl_perf.parquet"),
        "custom": _newest(root, system, "comm", ("sglang",), "custom_allreduce_perf.parquet"),
    }
    return {k: v for k, v in paths.items() if v}


def _load_raw(root: str, system: str) -> dict[str, np.ndarray]:
    sources = _sources(root, system)
    if not sources:
        return {}
    stamp = json.dumps([_FORMAT, sorted((k, p, os.stat(p).st_mtime_ns, os.stat(p).st_size)
                                        for k, p in sources.items())])
    digest = hashlib.sha1(stamp.encode()).hexdigest()[:16]
    cache = os.path.join(os.path.expanduser("~"), ".cache", "mstar_sim", f"aic-{system}-{digest}.npz")
    try:
        with np.load(cache) as f:
            raw = {k: f[k] for k in f.files}
    except (OSError, ValueError):
        raw = None
    if raw is None:
        raw = _read_parquet(sources)
        _save(cache, raw)
    for name in [k for k in raw if k.endswith(".vocab")]:  # back to strings
        column = name[: -len(".vocab")]
        raw[column] = raw[name][raw[column]]
    return raw


def _save(cache: str, raw: dict[str, np.ndarray]) -> None:
    try:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        tmp = f"{cache}.{os.getpid()}.tmp.npz"
        np.savez(tmp, **raw)
        os.replace(tmp, cache)
    except OSError:
        pass


def _read_parquet(sources: dict[str, str]) -> dict[str, np.ndarray]:
    import pandas as pd

    raw: dict[str, np.ndarray] = {}

    def put(prefix: str, frame, columns: dict[str, str]) -> None:
        for name, column in columns.items():
            values = frame[column].to_numpy()
            if np.issubdtype(values.dtype, np.integer):
                values = values.astype(np.int64 if values.max(initial=0) >= 2**31 else np.int32)
            elif not np.issubdtype(values.dtype, np.floating):  # strings: codes into a vocabulary
                vocab, values = np.unique(values.astype(str), return_inverse=True)
                raw[f"{prefix}.{name}.vocab"] = vocab
                values = values.astype(np.int16)
            raw[f"{prefix}.{name}"] = values

    if "gemm" in sources:
        d = pd.read_parquet(sources["gemm"])
        bf16 = d[d.gemm_dtype == "bfloat16"]
        if "sglang_torch_linear" in set(bf16.kernel_source):
            bf16 = bf16[bf16.kernel_source == "sglang_torch_linear"]
        fp8 = d[d.gemm_dtype == "fp8"]
        d = pd.concat([bf16.assign(dtype="bf16"), fp8.assign(dtype="fp8")])
        put("gemm", d, {"dtype": "dtype", "m": "m", "n": "n", "k": "k", "latency": "latency"})
    for phase in ("generation", "context"):
        if phase not in sources:
            continue
        d = pd.read_parquet(sources[phase])
        if "window_size" in d:
            d = d[d.window_size == 0]
        d = d[d.attn_dtype == "bfloat16"]
        kv = d.kv_cache_dtype.map({"bfloat16": "bf16", "fp8": "fp8"})
        d = d.assign(kv=kv, seq=d.isl + d.step, lane=d.get("kernel_source", "default"))
        d = d[d.kv.notna()]
        put(phase, d, {"lane": "lane", "kv_dtype": "kv", "kv_heads": "num_key_value_heads",
                       "head_dim": "head_dim", "heads": "num_heads", "batch": "batch_size",
                       "seq": "seq", "latency": "latency"})
    if "gdn" in sources:
        d = pd.read_parquet(sources["gdn"])
        put("gdn", d, {"kernel": "kernel_source", "nk": "num_k_heads", "nv": "num_v_heads",
                       "dk": "head_k_dim", "dv": "head_v_dim", "batch": "batch_size",
                       "seq": "seq_len", "latency": "latency"})
    if "nccl" in sources:
        d = pd.read_parquet(sources["nccl"])
        d = d[d.nccl_dtype == "half"]
        put("nccl", d, {"op": "op_name", "gpus": "num_gpus", "size": "message_size", "latency": "latency"})
    if "custom" in sources:
        d = pd.read_parquet(sources["custom"])
        d = d.assign(mode=d.backend.str.replace("sglang_", "", regex=False))
        put("custom", d, {"backend": "mode", "gpus": "num_gpus", "size": "message_size",
                          "latency": "latency"})
    return raw
