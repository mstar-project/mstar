"""Multi-rank harness for the fused TP all-reduce + residual + RMSNorm on ROCm.

One process per GPU sets up the TP group like a worker does (RCCL default
group, gloo twin, TCPStore) and calls ``CommGroup.allreduce_add_rmsnorm``. That
runs AITER's one-stage kernel where ``AiterARFusion`` accepts the input and
RCCL + norm otherwise. Results are compared with a reference built from all
ranks' inputs, and every call runs twice and must match bitwise. Races in this
kernel family have shown up as a few wrong rows that change between runs.

Modes (``--mode``):
  parity   eager, over token counts, hidden sizes and gemma on/off, plus the
           workspace's memory per rank
  capture  chains of calls captured in one graph and replayed on new inputs,
           with captured inputs either copied into AITER's pool or registered
  timing   time per call inside a graph, fused (copy-in and registered) vs
           RCCL + a compiled add+norm
  stages   AITER's one- and two-stage kernels called directly, outside what
           mstar fuses. Works with any importable ``aiter`` version.
  drift    a residual stream over 2 x ``--layers`` steps, each adding the
           all-reduce of new partial sums like a decoder layer. Runs AITER's
           kernel and RCCL + add next to an fp64 stream and reports how far
           each moves and whether it shrinks. The kernel build sets the rounding.

pytest runs parity and capture at 2, 4 and 8 ranks if that many GPUs are
visible. From the command line, for example::

    python test/rocm/test_ar_fusion_tp.py --world 4 --mode parity --hidden 5120

Each rank needs its own GPU. Two processes on one GPU get wrong sums and
illegal memory accesses from AITER's IPC kernels.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HIDDENS = (2048, 4096, 5120, 8192)
TOKENS = (1, 2, 3, 4, 5, 7, 8, 12, 13, 16, 24, 25, 26, 31, 32, 48, 63, 64,
          79, 80, 81, 96, 127, 128, 256, 512, 1000, 1024, 2048, 4096)
CAPTURE_TOKENS = (1, 2, 3, 7, 8, 12, 25, 31, 64, 80)
TIMING_TOKENS = (1, 4, 8, 16, 25, 32, 64, 80, 128, 512)


@dataclass
class Options:
    mode: str = "parity"
    hiddens: tuple[int, ...] = HIDDENS
    tokens: tuple[int, ...] = TOKENS
    dtype: str = "bfloat16"
    max_tokens: int = 2048
    chain: int = 16
    replays: int = 8
    repeats: int = 2
    timing_calls: int = 50
    timing_replays: int = 20
    stages: tuple[int, ...] = (1, 2)
    stage_max_size: int = 64 << 20
    layers: int = 64
    timeout: float = 1800.0


@dataclass
class Row:
    mode: str
    rank: int
    world: int
    hidden: int
    tokens: int
    gemma: bool = False
    path: str = ""
    max_err_norm: float = 0.0
    max_err_res: float = 0.0
    bad_rows: int = 0
    deterministic: bool = True
    inputs_untouched: bool = True
    us_per_call: float | None = None
    mem_mib: float | None = None
    note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.bad_rows == 0 and self.deterministic and self.inputs_untouched


# --- per-rank setup -----------------------------------------------------------


class _Ranks:
    """This process's view of the group: devices, groups, the reference."""

    def __init__(self, rank: int, world: int, port: int, opts: Options):
        self.rank, self.world, self.opts = rank, world, opts
        # all ranks are local, and the hostname can resolve to an unroutable
        # link-local address on these nodes
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        torch.cuda.set_device(rank)
        self.device = torch.device("cuda", rank)
        self.dtype = getattr(torch, opts.dtype)
        dist.init_process_group(
            "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
            world_size=world, device_id=self.device,
        )
        self.cpu_group = dist.new_group(
            list(range(world)), backend="gloo", use_local_synchronization=True,
        )

    def comm_group(self):
        from mstar.distributed.communication import CommGroup

        group = CommGroup(self.rank, self.rank, list(range(self.world)))
        group.device_group = dist.group.WORLD
        group.cpu_group = self.cpu_group
        group.initialized = True
        return group

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        """Every rank's ``x``, stacked on dim 0."""
        out = torch.empty((self.world, *x.shape), dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x.contiguous())
        return out

    def inputs(self, tokens: int, hidden: int, gemma: bool, seed: int):
        """``x`` differs per rank, residual and weight are the same on all."""
        shared = torch.Generator(device=self.device).manual_seed(seed)
        own = torch.Generator(device=self.device).manual_seed(seed * 1009 + self.rank + 1)
        x = torch.randn(tokens, hidden, generator=own, device=self.device).to(self.dtype)
        res = torch.randn(tokens, hidden, generator=shared, device=self.device).to(self.dtype)
        w = 0.1 * torch.randn(hidden, generator=shared, device=self.device)
        w = (w if gemma else 1.0 + w).to(self.dtype)
        return x, res, w

    def reference(self, x, res, w, eps, weight_bias, path="fused"):
        """Returns ``((norm, residual), (norm_tol, residual_tol))`` built from
        every rank's ``x``. Same tolerances as the startup check, plus some
        extra for RCCL, whose ring rounds each partial sum."""
        from mstar.distributed.ar_fusion import _reference

        xs = self.gather(x)
        want, (tol_n, tol_r) = _reference(xs, res, w, weight_bias, eps)
        if path != "fused" and self.world > 2:
            r = want[1].float()
            scale = torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + eps)
            extra = 2.0 ** -7 * (self.world - 1) * xs.float().abs().sum(0)
            tol_r = tol_r + extra
            tol_n = tol_n + scale * (w.float() + weight_bias).abs() * extra
        return want, (tol_n, tol_r)

    def compare(self, row: Row, got, reference) -> None:
        (normed_ref, r_ref), (tol_n, tol_r) = reference
        normed, r = got
        err_n = (normed.float() - normed_ref.float()).abs().nan_to_num(float("inf"))
        err_r = (r.float() - r_ref.float()).abs().nan_to_num(float("inf"))
        bad = (err_n > tol_n).any(-1) | (err_r > tol_r).any(-1)
        row.max_err_norm = max(row.max_err_norm, err_n.max().item())
        row.max_err_res = max(row.max_err_res, err_r.max().item())
        row.bad_rows += int(bad.sum().item())
        # which way the residual's rounding differences go (truncation shrinks)
        differs = r.float() != r_ref.float()
        shrunk = (r.float().abs() < r_ref.float().abs()) & differs
        row.extra["differ"] = row.extra.get("differ", 0) + int(differs.sum().item())
        row.extra["shrunk"] = row.extra.get("shrunk", 0) + int(shrunk.sum().item())
        row.extra["elements"] = row.extra.get("elements", 0) + r.numel()


# --- modes --------------------------------------------------------------------


def _memory(ranks: _Ranks, fn):
    torch.cuda.synchronize(ranks.device)
    free0 = torch.cuda.mem_get_info(ranks.device)[0]
    out = fn()
    torch.cuda.synchronize(ranks.device)
    free1 = torch.cuda.mem_get_info(ranks.device)[0]
    return out, (free0 - free1) / 2**20


def _parity(ranks: _Ranks) -> list[Row]:
    rows = []
    for hidden in ranks.opts.hiddens:
        group = ranks.comm_group()
        fused, mem = _memory(ranks, lambda group=group, hidden=hidden: (
            group.init_allreduce_fusion(hidden, ranks.dtype, ranks.opts.max_tokens)
        ))
        fusion = group._ar_fusion
        rows.append(Row("memory", ranks.rank, ranks.world, hidden, 0,
                        path="fused" if fused else "none", mem_mib=mem,
                        note=f"fused up to {getattr(fusion, 'max_tokens', 0)} tokens"))
        for tokens in ranks.opts.tokens:
            for gemma in (False, True):
                eps, wb = (1e-6, 1.0) if gemma else (1e-5, 0.0)
                x, res, w = ranks.inputs(tokens, hidden, gemma, seed=hidden + tokens)
                x0, res0 = x.clone(), res.clone()
                row = Row("parity", ranks.rank, ranks.world, hidden, tokens, gemma)
                fused_here = fusion is not None and fusion.accepts(x, res, wb)
                row.path = "fused" if fused_here else "rccl+norm"
                # RCCL reduces the partial sum in place (the caller is done
                # with it), so each call gets a copy. The fused op is out of
                # place and gets the original.
                outs = [group.allreduce_add_rmsnorm(
                            x if fused_here else x.clone(), res, w, eps, wb)
                        for _ in range(ranks.opts.repeats)]
                want = ranks.reference(x, res, w, eps, wb, row.path)
                for got in outs:
                    ranks.compare(row, got, want)
                row.deterministic = all(
                    torch.equal(a, b) for got in outs[1:] for a, b in zip(got, outs[0], strict=True)
                )
                row.inputs_untouched = torch.equal(x, x0) and torch.equal(res, res0)
                rows.append(row)
        if fusion is not None:
            fusion.close()
    return rows


def _capture(ranks: _Ranks) -> list[Row]:
    from mstar.distributed.ar_fusion import GraphCaptureScope

    rows = []
    pool = torch.cuda.graphs.graph_pool_handle()
    for hidden in ranks.opts.hiddens:
        group = ranks.comm_group()
        group.init_allreduce_fusion(hidden, ranks.dtype, ranks.opts.max_tokens)
        fusion = group._ar_fusion
        if fusion is None:
            rows.append(Row("capture", ranks.rank, ranks.world, hidden, 0,
                            path="none", note="no fusion here"))
            continue
        for tokens in ranks.opts.tokens:
            if not 0 < tokens <= fusion.max_tokens:
                continue
            for registered in (False, True):
                for gemma in (False, True):
                    rows.append(_capture_case(
                        ranks, group, pool, hidden, tokens, gemma, registered,
                        GraphCaptureScope,
                    ))
        fusion.close()
    return rows


def _capture_case(ranks, group, pool, hidden, tokens, gemma, registered, plain_scope):
    eps, wb = (1e-6, 1.0) if gemma else (1e-5, 0.0)
    scale = (ranks.rank + 1) / ranks.world
    x0, res0, w = ranks.inputs(tokens, hidden, gemma, seed=7 * hidden + tokens)

    def chain():
        steps, x, r = [], x0, res0
        for _ in range(ranks.opts.chain):
            normed, r_next = group.allreduce_add_rmsnorm(x, r, w, eps, wb)
            steps.append((x, r, normed, r_next))
            x, r = normed * scale, r_next
        return steps

    row = Row("capture", ranks.rank, ranks.world, hidden, tokens, gemma,
              path="fused-registered" if registered else "fused-copy-in")
    for _ in range(2):
        chain()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    scope = group.graph_capture() if registered else plain_scope()
    with scope:
        with torch.cuda.graph(graph, pool=pool):
            steps = chain()
        scope.captured = True
    if not scope.usable:
        row.bad_rows, row.note = -1, "capture scope rejected the graph"
        return row
    previous = None
    for rep in range(ranks.opts.replays):
        fresh, fresh_res, _ = ranks.inputs(tokens, hidden, gemma, seed=1000 * rep + tokens)
        x0.copy_(fresh)
        res0.copy_(fresh_res)
        graph.replay()
        torch.cuda.synchronize()
        for x, r, normed, r_next in steps:
            ranks.compare(row, (normed, r_next), ranks.reference(x, r, w, eps, wb))
        last = [t.clone() for t in steps[-1][2:]]
        if rep == 0:
            previous = (fresh.clone(), fresh_res.clone(), last)
    # replaying the first inputs again must give the same outputs bit for bit
    x0.copy_(previous[0])
    res0.copy_(previous[1])
    graph.replay()
    torch.cuda.synchronize()
    row.deterministic = all(
        torch.equal(a, b) for a, b in zip(steps[-1][2:], previous[2], strict=True)
    )
    del graph
    return row


def _add_rmsnorm(x, residual, w, eps: float, weight_bias: float):
    r = x + residual
    v = r.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return (v * (w.float() + weight_bias)).to(r.dtype), r


def _timing(ranks: _Ranks) -> list[Row]:
    rows = []
    # one compile per input size, like the engine does per bucket
    torch._dynamo.config.recompile_limit = 1024
    torch._dynamo.config.accumulated_recompile_limit = 4096
    compiled = torch.compile(_add_rmsnorm, dynamic=False)
    pool = torch.cuda.graphs.graph_pool_handle()
    for hidden in ranks.opts.hiddens:
        group = ranks.comm_group()
        group.init_allreduce_fusion(hidden, ranks.dtype, ranks.opts.max_tokens)
        fusion = group._ar_fusion
        for tokens in ranks.opts.tokens:
            x, res, w = ranks.inputs(tokens, hidden, False, seed=hidden + tokens)

            # in place, like CommGroup's fallback. The values grow over
            # replays, which doesn't change the timing.
            def unfused(y=x.clone(), res=res, w=w):
                dist.all_reduce(y)
                return compiled(y, res, w, 1e-6, 0.0)

            variants = {"rccl+compiled-norm": (unfused, False)}
            if fusion is not None and fusion.accepts(x, res, 0.0):
                def fused(x=x, res=res, w=w, group=group):
                    return group.allreduce_add_rmsnorm(x, res, w, 1e-6, 0.0)

                variants["fused-copy-in"] = (fused, False)
                variants["fused-registered"] = (fused, True)
            for name, (fn, registered) in variants.items():
                row = Row("timing", ranks.rank, ranks.world, hidden, tokens, path=name)
                row.us_per_call = _time_in_graph(ranks, group, pool, fn, registered)
                rows.append(row)
        if fusion is not None:
            fusion.close()
    return rows


def _time_in_graph(ranks, group, pool, fn, registered) -> float:
    from mstar.distributed.ar_fusion import GraphCaptureScope

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    scope = group.graph_capture() if registered else GraphCaptureScope()
    with scope:
        with torch.cuda.graph(graph, pool=pool):
            keep = [fn() for _ in range(ranks.opts.timing_calls)]
        scope.captured = True
    assert scope.usable
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier(group=ranks.cpu_group)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ranks.opts.timing_replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    del keep, graph
    return 1000 * start.elapsed_time(end) / (ranks.opts.timing_replays * ranks.opts.timing_calls)


def _stages(ranks: _Ranks) -> list[Row]:
    import aiter
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    version = getattr(aiter, "__version__", None)
    if version is None:
        try:
            from aiter._version import __version__ as version
        except ImportError:
            version = "unknown"
    ca = CustomAllreduce(ranks.cpu_group, ranks.device, max_size=ranks.opts.stage_max_size)
    rows = []
    for hidden in ranks.opts.hiddens:
        for tokens in ranks.opts.tokens:
            if tokens * hidden * 2 > ranks.opts.stage_max_size // 2:
                continue
            for stage in ranks.opts.stages:
                for gemma in (False, True):
                    eps, wb = (1e-6, 1.0) if gemma else (1e-5, 0.0)
                    x, res, w = ranks.inputs(tokens, hidden, gemma, seed=3 * hidden + tokens)
                    row = Row("stages", ranks.rank, ranks.world, hidden, tokens, gemma,
                              path=f"aiter-{stage}stage", note=f"aiter {version}")
                    want = ranks.reference(x, res, w, eps, wb)
                    outs = []
                    for _ in range(ranks.opts.repeats):
                        outs.append(ca.fused_ar_rms(
                            x, res, w=w, eps=eps, registered=False,
                            use_1stage=stage == 1, gemma_norm=gemma,
                        ))
                    torch.cuda.synchronize()
                    for got in outs:
                        ranks.compare(row, got, want)
                    row.deterministic = all(
                        torch.equal(a, b) for got in outs[1:]
                        for a, b in zip(got, outs[0], strict=True)
                    )
                    rows.append(row)
    ca.close()
    return rows


def _drift(ranks: _Ranks) -> list[Row]:
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    ca = CustomAllreduce(ranks.cpu_group, ranks.device, max_size=ranks.opts.stage_max_size)
    rows = []
    for hidden in ranks.opts.hiddens:
        for tokens in ranks.opts.tokens:
            if tokens * hidden * 2 > ranks.opts.stage_max_size:
                continue
            shared = torch.Generator(device=ranks.device).manual_seed(hidden + tokens)
            own = torch.Generator(device=ranks.device).manual_seed(7919 * (hidden + tokens) + ranks.rank)
            r0 = torch.randn(tokens, hidden, generator=shared, device=ranks.device).to(ranks.dtype)
            w = torch.ones(hidden, dtype=ranks.dtype, device=ranks.device)
            r64, r_fused, r_rccl = r0.double(), r0.clone(), r0.clone()
            out_fused = None
            for _ in range(2 * ranks.opts.layers):
                x = (torch.randn(tokens, hidden, generator=own, device=ranks.device)
                     * (0.5 / ranks.world ** 0.5)).to(ranks.dtype)
                r64 = r64 + ranks.gather(x).double().sum(0)
                out_fused, r_fused = ca.fused_ar_rms(
                    x, r_fused, w=w, eps=1e-6, registered=False, use_1stage=True,
                    gemma_norm=False,
                )
                y = x.clone()
                dist.all_reduce(y)
                r_rccl = r_rccl + y
            torch.cuda.synchronize()
            norm64 = r64 * torch.rsqrt(r64.pow(2).mean(-1, keepdim=True) + 1e-6)
            ref_sq = r64.pow(2).sum()
            for path, r, out in (("aiter-1stage", r_fused, out_fused), ("rccl+add", r_rccl, None)):
                r = r.double()
                row = Row("drift", ranks.rank, ranks.world, hidden, tokens, path=path)
                row.max_err_res = ((r - r64).norm() / r64.norm()).item()
                shrink = ((r * r64).sum() / ref_sq).item() - 1.0
                if out is not None:
                    row.max_err_norm = ((out.double() - norm64).norm() / norm64.norm()).item()
                row.note = (f"{2 * ranks.opts.layers} steps: residual drift "
                            f"{row.max_err_res:.4f}, projection {shrink:+.4f}")
                rows.append(row)
    ca.close()
    return rows


MODES = {"parity": _parity, "capture": _capture, "timing": _timing, "stages": _stages,
         "drift": _drift}


def _entry(rank: int, world: int, port: int, opts: Options, out_dir: str) -> None:
    ranks = _Ranks(rank, world, port, opts)
    rows = MODES[opts.mode](ranks)
    with open(Path(out_dir) / f"rank{rank}.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(asdict(row)) + "\n")
    dist.barrier(group=ranks.cpu_group)
    dist.destroy_process_group()


# --- launching ----------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def launch(world: int, opts: Options) -> list[Row]:
    """Runs ``opts.mode`` on ``world`` ranks and returns every rank's rows."""
    with tempfile.TemporaryDirectory(prefix="arfusion-") as out_dir:
        ctx = mp.start_processes(
            _entry, args=(world, _free_port(), opts, out_dir), nprocs=world,
            start_method="spawn", join=False,
        )
        deadline = time.monotonic() + opts.timeout
        while not ctx.join(timeout=5):
            if time.monotonic() > deadline:
                for p in ctx.processes:
                    p.kill()
                raise TimeoutError(f"{opts.mode} at world {world} hung for {opts.timeout:.0f}s")
        rows = []
        for rank in range(world):
            for line in (Path(out_dir) / f"rank{rank}.jsonl").read_text().splitlines():
                rows.append(Row(**json.loads(line)))
        return rows


def merge(rows: list[Row]) -> list[Row]:
    """One row per case, worst over ranks (for timing, the slowest rank)."""
    merged: dict[tuple, Row] = {}
    for row in rows:
        key = (row.mode, row.hidden, row.tokens, row.gemma, row.path)
        m = merged.get(key)
        if m is None:
            merged[key] = Row(**{**asdict(row), "rank": -1})
            continue
        m.max_err_norm = max(m.max_err_norm, row.max_err_norm)
        m.max_err_res = max(m.max_err_res, row.max_err_res)
        m.bad_rows += row.bad_rows
        m.deterministic &= row.deterministic
        m.inputs_untouched &= row.inputs_untouched
        if row.us_per_call is not None:
            m.us_per_call = max(m.us_per_call or 0.0, row.us_per_call)
        if row.mem_mib is not None:
            m.mem_mib = max(m.mem_mib or 0.0, row.mem_mib)
        for k, v in row.extra.items():
            m.extra[k] = m.extra.get(k, 0) + v
    return list(merged.values())


def format_table(rows: list[Row]) -> str:
    lines = ["mode     world hidden tokens gemma path                 "
             "err_norm  err_res bad det untouched  differ shrunk     us   MiB note"]
    for r in rows:
        us = f"{r.us_per_call:6.2f}" if r.us_per_call is not None else "     -"
        mib = f"{r.mem_mib:5.1f}" if r.mem_mib is not None else "    -"
        n = r.extra.get("elements", 0)
        differ = f"{r.extra.get('differ', 0) / n:7.3f}" if n else "      -"
        shrunk = (f"{r.extra['shrunk'] / r.extra['differ']:6.2f}"
                  if r.extra.get("differ") else "     -")
        lines.append(
            f"{r.mode:8} {r.world:5} {r.hidden:6} {r.tokens:6} {int(r.gemma):5} {r.path:20} "
            f"{r.max_err_norm:8.3g} {r.max_err_res:8.3g} {r.bad_rows:3} {int(r.deterministic):3} "
            f"{int(r.inputs_untouched):9} {differ} {shrunk} {us} {mib} {r.note}"
        )
    return "\n".join(lines)


# --- pytest -------------------------------------------------------------------


def _gpus_for(world: int) -> None:
    from mstar.platforms import current_platform

    if not current_platform.is_rocm():
        pytest.skip("AITER's custom all-reduce is the ROCm strategy")
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs, {torch.cuda.device_count()} visible")


@pytest.mark.parametrize("world", [2, 4, 8])
def test_fused_matches_rccl_and_a_separate_norm(world):
    _gpus_for(world)
    rows = merge(launch(world, Options(mode="parity")))
    print(format_table(rows))
    assert any(r.path == "fused" for r in rows), "nothing ran fused"
    assert [r for r in rows if not r.ok] == []


@pytest.mark.parametrize("world", [2, 4, 8])
def test_fused_inside_a_captured_graph(world):
    _gpus_for(world)
    rows = merge(launch(world, Options(mode="capture", tokens=CAPTURE_TOKENS)))
    print(format_table(rows))
    assert any(r.path.startswith("fused") for r in rows)
    assert [r for r in rows if not r.ok] == []


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--world", type=int, required=True)
    p.add_argument("--mode", choices=sorted(MODES), default="parity")
    p.add_argument("--hidden", type=int, nargs="+", default=None)
    p.add_argument("--tokens", type=int, nargs="+", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--chain", type=int, default=16)
    p.add_argument("--replays", type=int, default=8)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--stages", type=int, nargs="+", default=[1, 2])
    p.add_argument("--layers", type=int, default=64, help="drift: decoder layers (2 steps each)")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--out", type=Path, default=None, help="write merged rows as JSON")
    args = p.parse_args(argv)
    default_tokens = {"capture": CAPTURE_TOKENS, "timing": TIMING_TOKENS,
                      "drift": (8, 64)}.get(args.mode, TOKENS)
    opts = Options(
        mode=args.mode, hiddens=tuple(args.hidden or HIDDENS),
        tokens=tuple(args.tokens or default_tokens), dtype=args.dtype,
        max_tokens=args.max_tokens, chain=args.chain,
        replays=args.replays, repeats=args.repeats, stages=tuple(args.stages),
        layers=args.layers,
        timeout=args.timeout,
    )
    t0 = time.monotonic()
    rows = merge(launch(args.world, opts))
    print(format_table(rows))
    bad = [r for r in rows if not r.ok]
    print(f"{len(rows)} cases, {len(bad)} failing, {time.monotonic() - t0:.0f}s")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps([asdict(r) for r in rows], indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
