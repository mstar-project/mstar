"""``ragged_attention``: which kernel a cacheless self-attention runs, per head geometry and
longest query segment (``mstar.engine.resources.attn.ragged.kernels``).

A bucket is a segment length. Each candidate runs through the resource's own graph-mode
wrapper, planned with 1 and 8 equal segments of that length and timed from a CUDA graph; its
score is the geometric mean of the two. Its output is checked against an fp32 reference (math
SDPA per segment, TF32 off) first. The candidates are what a captured resource can replay:
torch's cuDNN SDPA (``cudnn``), FlashAttention-3 on Hopper (``fa3``, where ``flash_attn_3``
imports), and FlashInfer's FA2 and cuDNN backends. FlashInfer's
``cutlass`` backend is not one: its plan allocates fresh buffers, so a graph-mode wrapper can't
be planned twice.

Run as a module for a wider table at one size, including kernels the resource doesn't run
captured (FlashInfer cutlass, torch's FA2, FA3's varlen kernel), burst and sustained::

    python -m benchmark.kernels.ragged_attention --heads 24 --seq 4608 --segments 1 8
"""
from __future__ import annotations

import argparse
import itertools
import math
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmark.kernels.sweep import BucketResult
from benchmark.kernels.timing import time_graph
from mstar.engine.resources.attn.ragged.cudnn_wrappers import CudnnRaggedPrefillWrapper
from mstar.engine.resources.attn.ragged.fa3_wrappers import FlashAttn3RaggedPrefillWrapper
from mstar.engine.resources.attn.ragged.kernels import RAGGED_ATTENTION
from mstar.engine.resources.attn.ragged.wrappers import FlashInferRaggedPrefillWrapper
from mstar.engine.resources.attn.workspace import Workspace
from mstar.kernels import Choice, TableEntry

DEFAULT_BUCKETS = (4608,)
SEGMENTS = (1, 8)
CANDIDATES = (
    Choice("cudnn"),
    Choice("fa3"),
    Choice("flashinfer", {"backend": "fa2"}),
    Choice("flashinfer", {"backend": "cudnn"}),
)
# what a resource without a table row runs: FlashInfer's own `auto`
DEFAULT = Choice("flashinfer")
# the fp32 scores the reference holds at once
REF_SCORE_BYTES = 8 << 30


class Problem:
    """``n`` segments of ``seq`` tokens packed ``[n * seq, H, D]``, and their fp32 reference."""

    def __init__(self, n: int, seq: int, heads: int, head_dim: int, dtype: torch.dtype, device):
        self.n, self.seq, self.heads, self.head_dim = n, seq, heads, head_dim
        self.cu = torch.arange(0, (n + 1) * seq, seq, dtype=torch.int32)
        gen = torch.Generator(device=device).manual_seed(0)
        self.q, self.k, self.v = (torch.randn(n * seq, heads, head_dim, device=device, dtype=dtype, generator=gen)
                                  for _ in range(3))
        self._ref: torch.Tensor | None = None

    @property
    def ref(self) -> torch.Tensor:
        if self._ref is None:
            tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            # math SDPA holds fp32 scores for every head it is given, [H, S, S]: a few heads at a
            # time keeps that near REF_SCORE_BYTES, so the reference fits at 32k-64k tokens
            step = max(1, REF_SCORE_BYTES // (4 * self.seq * self.seq))
            try:
                q, k, v = (t.float().view(self.n, self.seq, self.heads, self.head_dim).transpose(1, 2)
                           for t in (self.q, self.k, self.v))
                with sdpa_kernel(SDPBackend.MATH):
                    out = torch.cat([
                        torch.cat([F.scaled_dot_product_attention(q[i:i + 1, h:h + step], k[i:i + 1, h:h + step],
                                                                  v[i:i + 1, h:h + step])
                                   for h in range(0, self.heads, step)], dim=1)
                        for i in range(self.n)])
            finally:
                torch.backends.cuda.matmul.allow_tf32 = tf32
            self._ref = out.transpose(1, 2).reshape(self.n * self.seq, self.heads, self.head_dim)
        return self._ref

    @property
    def tflop(self) -> float:
        return 4 * self.n * self.seq * self.seq * self.head_dim * self.heads / 1e12


def wrapper_for(choice: Choice, p: Problem, graph: bool = True):
    """The resource's wrapper for ``choice``, planned on ``p``'s layout."""
    kw = dict(num_qo_heads=p.heads, num_kv_heads=p.heads, head_dim=p.head_dim, device=p.q.device,
              sm_scale=p.head_dim ** -0.5, q_data_type=p.q.dtype)
    if graph:
        kw.update(accelerator_graph=True, max_num_segments=p.n, max_total_tokens=p.n * p.seq)
    if choice.impl == "cudnn":
        w = CudnnRaggedPrefillWrapper(**kw)
    elif choice.impl == "fa3":
        w = FlashAttn3RaggedPrefillWrapper(**kw)
    else:
        ws = Workspace(torch.empty(512 << 20, dtype=torch.uint8, device=p.q.device), "sweep")
        w = FlashInferRaggedPrefillWrapper(ws, backend=(choice.config or {}).get("backend", "auto"), **kw)
    w.plan(p.cu.pin_memory())
    return w


def rel_l2(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float((out.float() - ref).norm() / ref.norm())


class RaggedAttentionSweep:
    default_buckets = DEFAULT_BUCKETS

    def __init__(self, shape, dtype, device, quick: bool = False):
        self.shape, self.dtype, self.device = dict(shape), dtype, device
        if shape["num_qo_heads"] != shape["num_kv_heads"]:
            raise SystemExit("ragged_attention sweeps MHA geometries only")
        self.target = RAGGED_ATTENTION.target(dtype, shape, device)
        self.unavailable = RAGGED_ATTENTION.unavailable(self.target)

    def _score(self, choice: Choice, problems: list[Problem]) -> tuple[float, float]:
        """(geomean us over the problems, worst rel L2); NaN us when it fails the check or won't run."""
        impl = RAGGED_ATTENTION.impl(choice.impl)
        if choice.impl in self.unavailable:
            return math.nan, math.nan
        times, errs = [], []
        try:
            for p in problems:
                w = wrapper_for(choice, p)
                out = w.run(p.q, p.k, p.v)
                torch.cuda.synchronize()
                errs.append(rel_l2(out, p.ref))
                if impl.tolerance.error(out, p.ref) > 1:
                    return math.nan, max(errs)
                times.append(time_graph(lambda w=w, p=p: w.run(p.q, p.k, p.v)))
        except Exception as exc:  # noqa: BLE001
            print(f"#   {choice.describe()}: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}", flush=True)
            return math.nan, math.nan
        return math.exp(sum(math.log(t) for t in times) / len(times)), max(errs)

    def run_bucket(self, m: int, current: Choice | None) -> BucketResult:
        problems = [Problem(n, m, self.shape["num_qo_heads"], self.shape["head_dim"], self.dtype, self.device)
                    for n in SEGMENTS]
        current = current or DEFAULT
        # Choice holds a dict, so the choices are keyed by how they print
        choices = {c.describe(): c for c in (*CANDIDATES, DEFAULT, current)}
        scores = {name: self._score(c, problems) for name, c in choices.items()}
        for name, (us, err) in scores.items():
            print(f"#   {m}: {name:40s} {us:9.1f} us (geomean over {SEGMENTS} segments), rel L2 {err:.5f}", flush=True)
        passed = {c.describe(): scores[c.describe()][0] for c in CANDIDATES if not math.isnan(scores[c.describe()][0])}
        best = min(passed, key=passed.get) if passed else None
        impl_us = {(c.impl if c.config is None else f"{c.impl}_{c.config['backend']}"): passed[c.describe()]
                   for c in CANDIDATES if c.describe() in passed}
        return BucketResult(bucket=m, current=current, current_us=scores[current.describe()][0],
                            default_us=scores[DEFAULT.describe()][0], best=choices[best] if best else None,
                            best_us=passed[best] if best else math.nan, impl_us=impl_us,
                            timed=len(passed), rejected=len(CANDIDATES) - len(passed))

    def check(self, entry: TableEntry) -> None:
        refused = sorted(set(entry.impls()) & set(self.unavailable))
        if refused:
            raise ValueError(f"rows name {refused}, which can't run here: {self.unavailable}")


# --- the wider table ------------------------------------------------------------------------------


def _sustained(launch: Callable[[], object], calls: int = 10, seconds: float = 1.5) -> tuple[float, float]:
    """(burst, sustained) us per call: the fastest replay, and the median of the last third of
    ``seconds`` of back-to-back replays, once the clock has settled under the power cap."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            launch()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            launch()
    graph.replay()
    torch.cuda.synchronize()
    times, end = [], time.time() + seconds
    while time.time() < end:
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        graph.replay()
        e1.record()
        e1.synchronize()
        times.append(e0.elapsed_time(e1) * 1000 / calls)
    tail = sorted(times[len(times) * 2 // 3:])
    return min(times), tail[len(tail) // 2]


def _kernels(p: Problem) -> dict[str, Callable[[], Callable[[], torch.Tensor]]]:
    def via(choice: Choice, graph: bool = True):
        def make():
            w = wrapper_for(choice, p, graph)
            return lambda: w.run(p.q, p.k, p.v)
        return make

    def torch_flash():
        cu = p.cu.to(p.q.device)
        return lambda: torch.ops.aten._flash_attention_forward(
            p.q, p.k, p.v, cu, cu, p.seq, p.seq, 0.0, False, False, scale=p.head_dim ** -0.5)[0]

    def fa3_varlen():
        from flash_attn_interface import flash_attn_varlen_func

        cu = p.cu.to(p.q.device)
        return lambda: flash_attn_varlen_func(p.q, p.k, p.v, cu, cu, p.seq, p.seq)

    return {
        "flashinfer fa2 (graph wrapper)": via(Choice("flashinfer", {"backend": "fa2"})),
        "flashinfer auto (graph wrapper)": via(Choice("flashinfer")),
        "flashinfer auto (eager wrapper)": via(Choice("flashinfer"), graph=False),
        "flashinfer fa2 (eager wrapper)": via(Choice("flashinfer", {"backend": "fa2"}), graph=False),
        "flashinfer cudnn (graph wrapper)": via(Choice("flashinfer", {"backend": "cudnn"})),
        "flashinfer cutlass (eager wrapper)": via(Choice("flashinfer", {"backend": "cutlass"}), graph=False),
        "torch cudnn sdpa (cudnn wrapper)": via(Choice("cudnn")),
        "flash_attn_3 (fa3 wrapper)": via(Choice("fa3")),
        "flash_attn_3 varlen": fa3_varlen,
        "torch flash varlen (FA2, sm_100 build)": torch_flash,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--seq", type=int, nargs="+", required=True, help="segment lengths")
    ap.add_argument("--segments", type=int, nargs="+", default=list(SEGMENTS))
    ap.add_argument("--seconds", type=float, default=1.5)
    ap.add_argument("--only", default="", help="comma-separated substrings of kernel names")
    a = ap.parse_args()
    dev = torch.device("cuda")
    print("| heads | seq x segments | kernel | burst us | sustained us | TFLOPS sustained | rel L2 vs fp32 |")
    print("|---|---|---|---|---|---|---|")
    for seq, n in itertools.product(a.seq, a.segments):
        p = Problem(n, seq, a.heads, a.head_dim, torch.bfloat16, dev)
        floor = rel_l2(p.ref.bfloat16(), p.ref)
        print(f"| {a.heads} | {seq} x {n} | fp32 reference rounded to bf16 | | | | {floor:.5f} |", flush=True)
        for name, make in _kernels(p).items():
            if a.only and not any(s in name for s in a.only.split(",")):
                continue
            try:
                launch = make()
                err = rel_l2(launch(), p.ref)
                burst, sustained = _sustained(launch, seconds=a.seconds)
                print(f"| {a.heads} | {seq} x {n} | {name} | {burst:.1f} | {sustained:.1f} | "
                      f"{p.tflop / sustained * 1e6:.0f} | {err:.5f} |", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"| {a.heads} | {seq} x {n} | {name} | failed: {type(exc).__name__}: "
                      f"{str(exc).splitlines()[0][:120]} | | | |", flush=True)
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
