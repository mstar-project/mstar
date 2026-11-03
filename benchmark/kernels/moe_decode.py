"""``moe_decode``: the small-batch fp8 MoE's launch tiles (``mstar.utils.fused_moe.decode``), per
token count.

Each bucket times ``decode.experts`` (routed plus shared experts) inside a CUDA graph of one call
per routing in a rotation, each with its own shared-expert weights, so expert weights come from
HBM rather than L2, as in a decode step. It times every per-pair (gate/up, down) tile pair and
the grouped kernels, and a bucket takes the faster path. A candidate, and the row that runs now,
competes only when its output is within the implementation's tolerance of the kernels' at their
built-in tiles (the reference), which allows the grouped kernels' other summation order.

An entry runs per-pair kernels up to some token count and grouped ones above it, so its rows are
the per-pair winners up to the first bucket the grouped kernels win, then the grouped ``"*"``
row. A sweep whose buckets never reach that crossover writes nothing: extend ``--buckets``.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import replace
from functools import partial

import torch

from benchmark.kernels.sweep import BucketResult, merge_runs
from benchmark.kernels.timing import time_graph
from mstar.kernels import Choice, TableEntry
from mstar.kernels.table import Rows
from mstar.utils.fused_moe import decode
from mstar.utils.fused_moe.decode import MOE_DECODE

DEFAULT_BUCKETS = tuple(range(1, 17))
ROTATIONS = 10


def _random_fp8(rows: int, cols: int, block=(128, 128)):
    w = torch.randn(rows, cols, device="cuda") * 0.02
    bo, bi = block
    nb, kb = -(-rows // bo), -(-cols // bi)
    padded = torch.zeros(nb * bo, kb * bi, device="cuda")
    padded[:rows, :cols] = w
    b = padded.view(nb, bo, kb, bi)
    s = b.abs().amax(dim=(1, 3)) / 448.0
    s = torch.where(s == 0, torch.ones_like(s), s)
    q = (b / s[:, None, :, None]).to(torch.float8_e4m3fn).view(nb * bo, kb * bi)[:rows, :cols]
    return q.contiguous().view(torch.uint8), s.contiguous()


def candidates(quick: bool = False):
    gate_up = [dict(bj=bj, bk=bk, bjs=bjs, bks=bks, warps=w) for bj, bk, bjs, bks, w in
               itertools.product((8, 16), (256, 512, 1024), (16, 32), (128, 256), (2, 4))]
    down = [dict(bn=bn, bc=bc, warps=w) for bn, bc, w in itertools.product((16, 32), (128, 256), (4, 8))]
    if quick:
        gate_up = [g for g in gate_up if g["warps"] == 4]
    return itertools.product(gate_up, down)


def as_row(tuning: decode._Tuning, tokens: int) -> Choice:
    """What ``tuning`` launches at ``tokens``, as a table row."""
    if tokens > tuning.pair_max_tokens:
        return Choice("triton", {"path": "grouped", "router": dict(tuning.router)})
    gate_up, down = decode._tiles(tokens, tuning)
    return Choice("triton", {"path": "pair", "gate_up": dict(gate_up), "down": dict(down),
                             "router": dict(tuning.router)})


def as_tuning(choice: Choice, builtin: decode._Tuning) -> decode._Tuning:
    """A row as a tuning that launches it at any token count; no config is the built-in table."""
    if choice.config is None:
        return builtin
    config = choice.config
    router = dict(config["router"])
    if config["path"] == "grouped":
        return replace(builtin, pair_max_tokens=0, router=router)
    return replace(builtin, pair_max_tokens=64, gate_up=((None, dict(config["gate_up"])),),
                   down=((None, dict(config["down"])),), router=router)


class MoeDecodeSweep:
    op = "moe_decode"
    default_buckets = DEFAULT_BUCKETS

    def __init__(self, shape: Mapping[str, int], dtype: torch.dtype, device: torch.device,
                 quick: bool = False):
        if dtype != torch.float8_e4m3fn:
            raise ValueError(f"moe_decode's experts are float8_e4m3fn; got {dtype}")
        self.H, self.E, self.I, self.K = (shape[k] for k in ("H", "E", "I", "K"))
        self.device, self.quick = device, quick
        torch.manual_seed(0)
        w13, s13 = zip(*(_random_fp8(2 * self.I, self.H) for _ in range(self.E)), strict=True)
        w2, s2 = zip(*(_random_fp8(self.H, self.I) for _ in range(self.E)), strict=True)
        self.w13, self.s13, self.w2, self.s2 = (torch.stack(t) for t in (w13, s13, w2, s2))
        self.shared = [((torch.randn(2 * self.I, self.H, device="cuda") * 0.02).bfloat16(),
                        (torch.randn(self.H, self.I, device="cuda") * 0.02).bfloat16()) for _ in range(ROTATIONS)]
        self.builtin = decode._tuning(device)  # the built-in table for this GPU

    def rows(self, chosen: list[tuple[int, Choice]]) -> Rows:
        """The per-pair winners up to the first bucket the grouped kernels win, then the grouped
        ``"*"`` row; per-pair winners past that bucket are reported and dropped. Every row carries
        the built-in router config, which this sweep doesn't time."""
        router = dict(self.builtin.router)
        chosen = [(b, Choice(c.impl, {**c.config, "router": router})) for b, c in chosen]
        first = next((i for i, (_, c) in enumerate(chosen) if c.config["path"] == "grouped"), None)
        if first is None:
            raise ValueError(f"the grouped kernels won none of buckets {[b for b, _ in chosen]}, so the "
                             "count where they take over is unknown; sweep larger counts (--buckets)")
        if first == 0:
            raise ValueError(f"the grouped kernels won the smallest bucket, {chosen[0][0]}; sweep "
                             "smaller counts (--buckets) to find the per-pair range")
        later = [b for b, c in chosen[first:] if c.config["path"] == "pair"]
        if later:
            print(f"# buckets {later}: per-pair won after the grouped kernels took over at "
                  f"{chosen[first][0]}; they run grouped", flush=True)
        return (*merge_runs(chosen[:first]), (None, chosen[first][1]))

    def check(self, entry: TableEntry) -> None:
        """Refuse an entry the kernels would refuse."""
        decode._from_entry(entry)

    def run_bucket(self, m: int, current: Choice | None) -> BucketResult:
        """``current`` is the table's row for ``m``; None, the built-in table's tiles."""
        # Written, a row carries its tiles, so the built-in table's are spelled out.
        current, default = (as_row(self.builtin, m) if current is None else current), as_row(self.builtin, m)
        x = torch.randn(m, self.H, device="cuda").bfloat16()
        routes = [(torch.rand(m, self.K, device="cuda"),
                   torch.stack([torch.randperm(self.E, device="cuda")[:self.K] for _ in range(m)]))
                  for _ in range(ROTATIONS)]
        orig = decode._tuning
        tolerance = MOE_DECODE.impl("triton").tolerance

        def timed(tuning) -> tuple[float, torch.Tensor] | None:
            decode._tuning = lambda device, shape=None: tuning
            try:
                calls = [partial(decode.experts, x, self.w13, self.s13, self.w2, self.s2, self.shared[i][0],
                                 self.shared[i][1], tw, ids, block_size=(128, 128))
                         for i, (tw, ids) in enumerate(routes)]
                out = calls[0]().clone()
                return time_graph(lambda: [c() for c in calls], replays=20) / ROTATIONS, out
            except Exception:  # noqa: BLE001 - a tile Triton or the kernel's asserts reject
                return None
            finally:
                decode._tuning = orig

        reference = timed(self.builtin)  # the op's reference: the kernels at their built-in tiles
        ref = None if reference is None else reference[1]

        def us_of(tuning) -> float:
            r = timed(tuning)
            if r is None or ref is None or tolerance.error(r[1], ref) > 1.0:
                return float("nan")
            return r[0]

        result = BucketResult(bucket=m, current=current, current_us=us_of(as_tuning(current, self.builtin)),
                              default_us=us_of(as_tuning(default, self.builtin)), best=None,
                              best_us=float("nan"))
        router = dict(self.builtin.router)
        grouped = Choice("triton", {"path": "grouped", "router": router})
        options = [grouped] + [Choice("triton", {"path": "pair", "gate_up": gate_up, "down": down, "router": router})
                               for gate_up, down in candidates(self.quick)]
        for choice in options:
            us = us_of(as_tuning(choice, self.builtin))
            if math.isnan(us):
                result.rejected += 1
                continue
            result.timed += 1
            if result.best is None or us < result.best_us:
                result.best, result.best_us = choice, us
        if result.best is not None:
            result.impl_us["triton"] = result.best_us
        return result
