"""``fp8_linear``: dense_fp8's block-fp8 linear (128 x 128 weight scales), swept per token count.

Each bucket times every kernel that runs on this GPU (w8a16 at its built-in tile and at candidate tiles up to
``W8A16_SWEEP_MAX`` tokens, flashinfer's swapAB GEMM and cuBLAS's block-scaled GEMM on SM90, flashinfer's groupwise
GEMM on sm_100 / sm_103 for a shape in whole 128-row blocks) inside a CUDA graph that calls
the linear once per weight copy in a rotation, so the weights come from HBM rather than L2, as in a decode step.
A candidate competes only when its output is within ``MAX_REL_L2`` of the bf16 product of the dequantized weight
(the kernels that quantize activations to e4m3 per 128 columns sit a few percent away; a broken tile far more).

This is the sweep the registry's boot log names for a shape without rows:

    python -m benchmark.kernels.sweep fp8_linear --dtype float8_e4m3fn --shape N=5120,K=2048,glu=0 --out DIR

``benchmark/quantization/fp8_rows.py`` collects a run's missing shapes, or a model's per-rank ones, into one command.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Mapping

import torch
import torch.nn.functional as F

from benchmark.kernels.sweep import BucketResult
from benchmark.kernels.timing import time_graph
from mstar.kernels import Choice, TableEntry
from mstar.utils import dense_fp8
from mstar.utils.fp8_block import FP8_DTYPE

DEFAULT_BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 2048)
W8A16_SWEEP_MAX = 64
MAX_REL_L2 = 0.06
# enough weight copies that a rotation outgrows any GPU's L2
ROTATION_BYTES = 256 << 20
BLOCK = (128, 128)


def w8a16_candidates(m: int, k: int, quick: bool = False) -> Iterator[dict]:
    """w8a16 tiles worth timing at ``m`` tokens: the FMA tile (one accumulator a token) up to 4, the dot above."""
    if m <= 4:
        grid = itertools.product((False,), (8, 16) if quick else (8, 16, 32), (256, 512), (1, 2, 4), (3,), (2, 4))
    else:
        grid = itertools.product((True,), (16, 32) if quick else (16, 32, 64), (64, 128), (1, 2, 4), (3, 4), (4,))
    for dot, bn, bk, split, stages, warps in grid:
        if k % (split * bk) == 0:
            yield dict(DOT=dot, BN=bn, BK=bk, SPLIT=split, STAGES=stages, num_warps=warps)


class Fp8LinearSweep:
    op = "fp8_linear"
    default_buckets = DEFAULT_BUCKETS

    def __init__(self, shape: Mapping[str, int], dtype: torch.dtype, device: torch.device, quick: bool = False):
        if dtype != FP8_DTYPE:
            raise ValueError(f"fp8_linear's weights are float8_e4m3fn; got {dtype}")
        self.n, self.k, self.glu = int(shape["N"]), int(shape["K"]), bool(shape["glu"])
        self.device, self.quick = device, quick
        dense_fp8.reserve(device)
        rows = 2 * self.n if self.glu else self.n
        copies = max(2, min(16, ROTATION_BYTES // (rows * self.k)))
        torch.manual_seed(0)
        self.weights = []
        padded = -(-rows // 128) * 128  # a partial last row block (GLM's 2624-row q_a + kv_a) has its own scale
        for _ in range(copies):
            w = F.pad(torch.randn(rows, self.k, device=device) * 0.02, (0, 0, 0, padded - rows))
            tiles = w.reshape(padded // 128, 128, self.k // 128, 128)
            s = (tiles.abs().amax(dim=(1, 3)) / torch.finfo(FP8_DTYPE).max).clamp_min(1e-12)
            q = (tiles / s[:, None, :, None]).to(FP8_DTYPE).reshape(padded, self.k)[:rows]
            self.weights.append((q.view(torch.uint8).contiguous(), s.contiguous()))
        q, s = self.weights[0]
        scale = s.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)
        self.ref_w = (q.view(FP8_DTYPE).float() * scale).bfloat16()

    def check(self, entry: TableEntry) -> None:
        """Rows name the op's kernels; the registry's check_rows already refused anything else."""

    def _runs(self, choice: Choice) -> bool:
        whole = (self.n * (2 if self.glu else 1)) % 128 == 0
        return (choice.impl == "w8a16" or (choice.impl == "flashinfer" and dense_fp8._FI_OK.get(self.device))
                or (choice.impl == "cublas" and dense_fp8._W8A8_OK.get(self.device))
                or (choice.impl == "groupwise" and dense_fp8._BW_OK.get(self.device) and whole))

    def run_bucket(self, m: int, current: Choice | None) -> BucketResult:
        """``current`` is the table's row for ``m``; None, the built-in dispatch."""
        default = Choice(dense_fp8._builtin(self.device, self.n, self.k, self.glu, m))
        current = default if current is None else current
        x = torch.randn(m, self.k, device=self.device).bfloat16()
        want = F.linear(x, self.ref_w).float()
        if self.glu:
            g, u = want.chunk(2, dim=-1)
            want = F.silu(g) * u

        def call(choice: Choice, q, s):
            orig = dense_fp8._choose
            dense_fp8._choose = lambda *a, **k: (choice.impl, None if choice.config is None else dict(choice.config))
            try:
                return dense_fp8._linear(x, q, s, BLOCK, self.glu)
            finally:
                dense_fp8._choose = orig

        def us_of(choice: Choice) -> float:
            if not self._runs(choice):
                return float("nan")
            try:
                out = call(choice, *self.weights[0]).float()
                if not torch.isfinite(out).all() or (out - want).norm() / want.norm() > MAX_REL_L2:
                    return float("nan")
                return time_graph(lambda: [call(choice, q, s) for q, s in self.weights], replays=10) / len(
                    self.weights)
            except Exception:  # noqa: BLE001 - a tile the kernel's asserts or Triton reject
                return float("nan")

        result = BucketResult(bucket=m, current=current, current_us=us_of(current), default_us=us_of(default),
                              best=None, best_us=float("nan"))
        options = [Choice("w8a16"), Choice("flashinfer"), Choice("cublas"), Choice("groupwise")]
        if m <= W8A16_SWEEP_MAX:
            options += [Choice("w8a16", cfg) for cfg in w8a16_candidates(m, self.k, self.quick)]
        for choice in options:
            us = us_of(choice)
            if math.isnan(us):  # refused, failed or off the reference
                result.rejected += 1
                continue
            result.timed += 1
            if us < result.impl_us.get(choice.impl, float("inf")):
                result.impl_us[choice.impl] = us
            if result.best is None or us < result.best_us:
                result.best, result.best_us = choice, us
        return result
