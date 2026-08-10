"""Join measured per-kernel times with analytic work -> headroom report.

Buckets (per step, additive where stated):
    observed              what the profiler measured across all kernels
  - kernel_quality_gap    observed - speed_of_light, per modeled kernel
  = speed_of_light        roofline time for the work actually performed
  - movement_elimination  SOL of pure data-movement kernels (floor ~0: a fused
                          pipeline never materializes them)
  - fusion_in_graph       per compiled subgraph: sum of member SOLs minus the
                          perfectly-fused subgraph floor
  - harness_artifacts     kernels only present due to test scaffolding
  = estimated_floor       what remains; library kernels we can't model
                          (attention etc.) pass through at observed time

Graph-break traffic is reported separately (informational, overlaps with the
buckets above): bytes returned by each compiled subgraph must round-trip HBM
because compilation stops there.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .costmodel import (
    LAUNCH_FLOOR_MS,
    MOVEMENT_EAGER_OPS,
    extern_sol_ms,
    is_movement_kernel,
    kernel_kind,
    sol_ms,
    triton_flops_estimate,
)
from .eager_attr import EagerOpAttribution
from .hardware import GpuSpec, resolve_gpu
from .inductor_dump import Subgraph

SCHEMA_VERSION = 1

DEFINITIONS = {
    "observed_ms": "measured device time per step, summed over all kernels",
    "speed_of_light_ms": "max(flops/peak, bytes/HBM bandwidth) for the work each kernel actually did; "
                         "clamped to observed (never claims a modeled kernel is slower than reality)",
    "kernel_quality_gap_ms": "observed - speed_of_light: recoverable by a better implementation of the SAME kernels",
    "movement_elimination_ms": "SOL of pure data-movement kernels (copies/casts/fills/cats): recoverable by "
                               "fusing into producers/consumers or removing layout round-trips",
    "fusion_in_graph_ms": "sum of per-kernel SOLs inside a compiled subgraph minus the perfectly-fused "
                          "subgraph floor max(sum flops/peak, external io bytes/bw): upper bound on further fusion",
    "harness_artifacts_ms": "kernels induced by test scaffolding (e.g. deterministic-mode asserts); absent in prod",
    "estimated_floor_ms": "speed_of_light - movement - fusion - artifacts; unmodeled library kernels pass through",
    "graph_break_traffic_ms": "informational: output buffers of each compiled subgraph x2 (write+re-read) / HBM bw; "
                              "the price of each compile boundary, overlaps other buckets",
    "opportunity_ms": "per kernel: quality gap + (movement ? SOL : 0) + (artifact ? SOL : 0) - what fixing "
                      "this kernel could recover",
    "launch_floor": f"per-kernel SOL is floored at {LAUNCH_FLOOR_MS * 1e3:.1f} us (launch/teardown); "
                    "kernels at the floor are latency-bound - eliminate launches, don't tune the kernel",
}


@dataclass
class KernelRow:
    key: str                       # full profiler kernel name
    kind: str                      # triton / gemm / attention / eager / conv / memcpy
    klass: str                     # quality | movement | library | artifact | unmodeled
    ms_step: float
    calls_step: float
    flops_call: float | None = None
    bytes_call: float | None = None
    sol_step: float | None = None
    opportunity_ms: float = 0.0
    dtype: str = "bf16"
    source: list[tuple[str, str]] = field(default_factory=list)   # (loc, code/frame)
    notes: list[str] = field(default_factory=list)

    @property
    def per_call_ms(self) -> float:
        return self.ms_step / self.calls_step if self.calls_step else self.ms_step

    @property
    def ai(self) -> float | None:
        if self.flops_call and self.bytes_call:
            return self.flops_call / self.bytes_call
        return None


@dataclass
class SubgraphSummary:
    name: str
    mult: float                    # invocations per step
    sum_sol_call_ms: float         # non-movement member SOLs, one invocation
    fused_floor_call_ms: float
    fusion_gap_step_ms: float
    boundary_bytes: int
    boundary_ms_step: float
    externs: list[dict] = field(default_factory=list)
    tritons: list[dict] = field(default_factory=list)


@dataclass
class Report:
    gpu_name: str
    spec: GpuSpec | None
    meta: dict
    rows: list[KernelRow]
    subgraphs: list[SubgraphSummary]
    buckets: dict[str, float]
    caveats: list[str]

    def to_json(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "gpu_name": self.gpu_name,
            "gpu_spec_matched": self.spec.name if self.spec else None,
            "meta": self.meta,
            "buckets_ms_per_step": self.buckets,
            "kernels": [
                {
                    "kernel": r.key, "kind": r.kind, "class": r.klass,
                    "observed_ms_step": round(r.ms_step, 4),
                    "calls_per_step": r.calls_step,
                    "flops_per_call": r.flops_call, "bytes_per_call": r.bytes_call,
                    "arithmetic_intensity": round(r.ai, 3) if r.ai else None,
                    "sol_ms_step": round(r.sol_step, 4) if r.sol_step is not None else None,
                    "opportunity_ms_step": round(r.opportunity_ms, 4),
                    "source": [{"loc": loc, "code": code} for loc, code in r.source],
                    "notes": r.notes,
                }
                for r in self.rows
            ],
            "subgraphs": [
                {
                    "name": s.name, "invocations_per_step": s.mult,
                    "sum_member_sol_call_ms": round(s.sum_sol_call_ms, 4),
                    "fused_floor_call_ms": round(s.fused_floor_call_ms, 4),
                    "fusion_gap_ms_step": round(s.fusion_gap_step_ms, 4),
                    "boundary_bytes": s.boundary_bytes,
                    "boundary_traffic_ms_step": round(s.boundary_ms_step, 4),
                    "extern_gemms": s.externs,
                    "triton_kernels": s.tritons,
                }
                for s in self.subgraphs
            ],
            "caveats": self.caveats,
            "definitions": DEFINITIONS,
        }


def build_report(
    subgraphs: list[Subgraph],
    ktime_us: dict[str, tuple[float, int]],
    replays: int,
    gpu_name: str,
    eager_attr: dict[str, list[EagerOpAttribution]] | None = None,
    meta: dict | None = None,
) -> Report:
    spec = resolve_gpu(gpu_name)
    eager_attr = eager_attr or {}
    caveats: list[str] = []
    if spec is None:
        caveats.append(f"GPU '{gpu_name}' not in gpu_spec.json: no speed-of-light column, all gaps read 0.")

    # index triton kernel definitions by name (a name can recur across subgraphs)
    defs: dict[str, list[tuple[int, object]]] = {}
    for si, sg in enumerate(subgraphs):
        for name, k in sg.triton.items():
            defs.setdefault(name, []).append((si, k))

    rows: list[KernelRow] = []
    rows_by_def_name: dict[str, KernelRow] = {}
    for key, (us, cnt) in ktime_us.items():
        ms_step = us / 1e3 / replays
        if ms_step <= 0:
            continue
        row = KernelRow(key=key, kind=kernel_kind(key), klass="unmodeled",
                        ms_step=ms_step, calls_step=cnt / replays)

        if row.kind == "triton":
            matched = [d for name, ds in defs.items() if name in key for d in ds]
            names = {name for name in defs if name in key}
            if matched:
                ks = [k for _, k in matched]
                row.bytes_call = statistics.mean(k.traffic_bytes for k in ks)
                row.flops_call = statistics.mean(triton_flops_estimate(k) for k in ks)
                out_dtypes = [t.dtype for k in ks for t in k.outputs.values()]
                row.dtype = out_dtypes[0] if out_dtypes else "bf16"
                row.klass = "movement" if all(is_movement_kernel(k) for k in ks) else "quality"
                seen = {}
                for k in ks:
                    seen.update(k.source_lines)
                row.source = list(seen.items())
                if len({k.traffic_bytes for k in ks}) > 1:
                    row.notes.append(f"{len(ks)} same-named defs with different shapes; using mean bytes")
                for name in names:
                    rows_by_def_name[name] = row
        elif row.kind in ("gemm", "conv"):
            row.klass = "library"          # modeled at the aggregate level below
        elif row.kind == "attention":
            row.klass = "library"
            row.notes.append("library attention kernel; no analytic floor computed")
        elif row.kind in ("eager", "memcpy"):
            attrs = eager_attr.get(key, [])
            if attrs:
                top = attrs[0]
                row.source = [(f"[{top.op}]", " <- ".join(top.stack) or "(no project frame)")]
                if top.op in ("aten::_assert_async", "aten::_assert_scalar"):
                    row.klass = "artifact"
                elif top.op in MOVEMENT_EAGER_OPS:
                    row.klass = "movement"
                    row.bytes_call = top.traffic_bytes()
                    if row.bytes_call is None:
                        row.notes.append("no input shapes recorded; SOL falls back to observed")
                    else:
                        row.notes.append("bytes assume bf16 elements (profiler records shapes, not dtypes)")
                else:
                    row.klass = "quality"
                    row.bytes_call = top.traffic_bytes()
            else:
                row.notes.append("no warmup attribution matched this kernel name")

        s = sol_ms(row.flops_call, row.bytes_call, spec, row.dtype)
        if s is not None:
            if s < LAUNCH_FLOOR_MS:
                s = LAUNCH_FLOOR_MS
                row.notes.append("latency-bound (~us-scale): recover by eliminating launches "
                                 "(fusion/batching), not by a faster kernel")
            row.sol_step = min(s * row.calls_step, row.ms_step)
        elif row.klass in ("movement", "artifact"):
            row.sol_step = row.ms_step     # unmodeled but eliminable: whole cost is the opportunity
        rows.append(row)

    # ---- per-subgraph: invocation multiplier, GEMM aggregate, fused floor ----
    gemm_sol_step_total = 0.0
    sub_summaries: list[SubgraphSummary] = []
    bw = spec.mem_bandwidth_gbps * 1e9 if spec else None
    def infer_mult(sg: Subgraph) -> float | None:
        unique = [
            rows_by_def_name[name].calls_step
            for name in sg.triton
            if name in rows_by_def_name and len(defs.get(name, [])) == 1
        ]
        if unique:
            return statistics.median(unique)
        # every member kernel name recurs in another subgraph: assume the
        # measured calls split evenly across the same-named definitions
        shared = [
            rows_by_def_name[name].calls_step / len(defs[name])
            for name in sg.triton
            if name in rows_by_def_name
        ]
        return statistics.median(shared) if shared else None

    direct_mults = [infer_mult(sg) for sg in subgraphs]
    known = [m for m in direct_mults if m is not None]
    fallback_mult = statistics.median(known) if known else 1.0
    for si, sg in enumerate(subgraphs):
        mult = direct_mults[si]
        if mult is None:
            mult = fallback_mult
            caveats.append(f"{sg.name}: no kernel matched profiler data; guessed {mult:.0f} invocations/step")

        externs = []
        ext_sol_call = 0.0
        ext_flops = 0
        for e in sg.externs:
            es = extern_sol_ms(e, spec)
            shape = e.gemm_shape()
            externs.append({
                "op": e.op, "bmnk": shape, "flops": e.flops, "dtype": e.dtype,
                "sol_ms_call": round(es, 5) if es is not None else None,
                "source": [{"loc": loc, "code": code} for loc, code in e.source_lines.items()],
            })
            if es is not None:
                ext_sol_call += max(es, LAUNCH_FLOOR_MS)
                ext_flops += e.flops or 0
        gemm_sol_step_total += ext_sol_call * mult

        tri_sol_call = 0.0
        tri_flops = 0
        for name, k in sg.triton.items():
            if is_movement_kernel(k):
                continue                    # claimed by the movement bucket
            row = rows_by_def_name.get(name)
            s = sol_ms(triton_flops_estimate(k), k.traffic_bytes, spec,
                       row.dtype if row else "bf16")
            if s is not None:
                tri_sol_call += max(s, LAUNCH_FLOOR_MS)
                tri_flops += triton_flops_estimate(k)

        tritons = []
        for name, k in sg.triton.items():
            row = rows_by_def_name.get(name)
            s = sol_ms(triton_flops_estimate(k), k.traffic_bytes, spec, row.dtype if row else "bf16")
            tritons.append({
                "kernel": name, "movement": is_movement_kernel(k),
                "bytes_per_call": k.traffic_bytes,
                "sol_ms_call": round(s, 5) if s is not None else None,
                "aten": sorted(k.aten_ops),
                "source": [{"loc": loc, "code": code} for loc, code in k.source_lines.items()],
            })

        io_bytes = sg.input_bytes + sg.output_bytes
        floor_call = 0.0
        if spec:
            peak = spec.peak_tflops("bf16") or 0
            floor_call = max(
                (ext_flops + tri_flops) / (peak * 1e12) * 1e3 if peak else 0.0,
                io_bytes / bw * 1e3 if bw else 0.0,
            )
        sum_call = ext_sol_call + tri_sol_call
        fusion_gap_step = max(0.0, sum_call - floor_call) * mult if spec else 0.0
        boundary_ms = (2 * sg.output_bytes / bw * 1e3 * mult) if bw else 0.0
        sub_summaries.append(SubgraphSummary(
            name=sg.name, mult=mult, sum_sol_call_ms=sum_call,
            fused_floor_call_ms=floor_call, fusion_gap_step_ms=fusion_gap_step,
            boundary_bytes=sg.output_bytes, boundary_ms_step=boundary_ms,
            externs=externs, tritons=tritons,
        ))

    # distribute the GEMM aggregate SOL across gemm rows proportionally to time
    gemm_rows = [r for r in rows if r.kind in ("gemm", "conv")]
    gemm_obs = sum(r.ms_step for r in gemm_rows)
    if gemm_rows and gemm_obs > 0:
        ratio = min(gemm_sol_step_total / gemm_obs, 1.0)
        for r in gemm_rows:
            r.sol_step = r.ms_step * ratio
            r.klass = "quality"
        if gemm_sol_step_total > gemm_obs:
            caveats.append("GEMM analytic SOL exceeds measured GEMM time; clamped (shape inference or "
                           "multiplier is off, treat GEMM gap as ~0)")

    # ---- buckets ----
    observed = sum(r.ms_step for r in rows)
    sol_total = sum(r.sol_step if r.sol_step is not None else r.ms_step for r in rows)
    movement = sum(r.sol_step or 0.0 for r in rows if r.klass == "movement")
    artifacts = sum(r.sol_step or 0.0 for r in rows if r.klass == "artifact")
    fusion = sum(s.fusion_gap_step_ms for s in sub_summaries)
    unmodeled = sum(r.ms_step for r in rows if r.sol_step is None)
    buckets = {
        "observed": observed,
        "kernel_quality_gap": observed - sol_total,
        "speed_of_light": sol_total,
        "movement_elimination": movement,
        "fusion_in_graph": fusion,
        "harness_artifacts": artifacts,
        "estimated_floor": max(sol_total - movement - fusion - artifacts, 0.0),
        "unmodeled_passthrough": unmodeled,
        "graph_break_traffic": sum(s.boundary_ms_step for s in sub_summaries),
    }

    for r in rows:
        gap = r.ms_step - r.sol_step if r.sol_step is not None else 0.0
        r.opportunity_ms = gap + (r.sol_step or 0.0 if r.klass in ("movement", "artifact") else 0.0)
    rows.sort(key=lambda r: -r.opportunity_ms)

    caveats.append("triton flops are estimates (~1 flop/elem/arith-node); fine because fused elementwise "
                   "chains are bandwidth-bound at any realistic intensity")
    caveats.append("fused-floor / fusion_in_graph is an upper bound: it assumes perfect fusion of a whole "
                   "compiled subgraph incl. GEMM epilogues")
    return Report(gpu_name=gpu_name, spec=spec, meta=meta or {}, rows=rows,
                  subgraphs=sub_summaries, buckets=buckets, caveats=caveats)


def render_text(rep: Report, top: int = 40) -> str:
    out: list[str] = []
    w = out.append
    b = rep.buckets
    obs = b["observed"] or 1.0
    spec_name = rep.spec.name if rep.spec else "UNKNOWN (no spec matched)"
    w("=" * 100)
    w(f"KERNEL HEADROOM REPORT   gpu={rep.gpu_name} -> spec {spec_name}")
    if rep.spec:
        w(f"ceilings: bf16 {rep.spec.bf16_tflops:.0f} TFLOP/s dense, HBM {rep.spec.mem_bandwidth_gbps:.0f} GB/s, "
          f"ridge {rep.spec.ridge_flops_per_byte('bf16'):.0f} flop/byte")
    for k, v in rep.meta.items():
        w(f"meta: {k} = {v}")
    w("=" * 100)
    w("")
    w("WATERFALL (ms/step, additive top to bottom):")
    w(f"  observed                {b['observed']:8.3f}   (100.0%)")
    w(f"  - kernel_quality_gap    {b['kernel_quality_gap']:8.3f}   ({b['kernel_quality_gap'] / obs * 100:5.1f}%)  "
      f"<- better kernels, same work")
    w(f"  = speed_of_light        {b['speed_of_light']:8.3f}   ({b['speed_of_light'] / obs * 100:5.1f}%)")
    w(f"  - movement_elimination  {b['movement_elimination']:8.3f}   ({b['movement_elimination'] / obs * 100:5.1f}%)  "
      f"<- fuse/remove pure data movement")
    w(f"  - fusion_in_graph       {b['fusion_in_graph']:8.3f}   ({b['fusion_in_graph'] / obs * 100:5.1f}%)  "
      f"<- fuse remaining kernels per subgraph (upper bound)")
    w(f"  - harness_artifacts     {b['harness_artifacts']:8.3f}   ({b['harness_artifacts'] / obs * 100:5.1f}%)")
    w(f"  = estimated_floor       {b['estimated_floor']:8.3f}   ({b['estimated_floor'] / obs * 100:5.1f}%)")
    w(f"  [unmodeled passthrough  {b['unmodeled_passthrough']:8.3f}   ({b['unmodeled_passthrough'] / obs * 100:5.1f}%)"
      f"  - library kernels kept at observed]")
    w(f"  [graph_break_traffic    {b['graph_break_traffic']:8.3f}   informational, overlaps buckets above]")
    w("")
    w(f"optimality (speed_of_light/observed): {b['speed_of_light'] / obs:.2f}   "
      f"floor ratio (estimated_floor/observed): {b['estimated_floor'] / obs:.2f}")
    w("")
    w("=" * 100)
    w(f"TOP OPPORTUNITIES (rank by recoverable ms/step; {len(rep.rows)} kernels total)")
    w("=" * 100)
    for i, r in enumerate(rep.rows[:top], 1):
        sol = f"{r.sol_step:.3f}" if r.sol_step is not None else "  n/a"
        byts = f"{r.bytes_call / 1e6:.2f}MB/call" if r.bytes_call else ""
        w(f"\n#{i:<3} opportunity {r.opportunity_ms:.3f} ms/step   observed {r.ms_step:.3f}  sol {sol}  "
          f"x{r.calls_step:.0f}/step   [{r.kind}/{r.klass}] {byts}")
        w(f"     kernel: {r.key}")
        for loc, code in r.source[:4]:
            w(f"     source: {loc:<26} {code}")
        for note in r.notes:
            w(f"     note:   {note}")
    rest = rep.rows[top:]
    if rest:
        w(f"\n( +{len(rest)} more kernels, {sum(r.opportunity_ms for r in rest):.3f} ms/step "
          f"opportunity, {sum(r.ms_step for r in rest):.3f} ms/step observed )")
    w("")
    w("=" * 100)
    w("PER-SUBGRAPH FUSION VIEW (per compiled Inductor graph)")
    w("=" * 100)
    for s in rep.subgraphs:
        w(f"\n{s.name}   x{s.mult:.0f}/step   boundary out {s.boundary_bytes / 1e6:.2f} MB "
          f"(~{s.boundary_ms_step:.3f} ms/step round-trip)")
        w(f"   sum of member SOLs {s.sum_sol_call_ms:.4f} ms/call vs perfectly-fused floor "
          f"{s.fused_floor_call_ms:.4f} ms/call  -> fusion gap {s.fusion_gap_step_ms:.3f} ms/step")
        for t in s.tritons:
            tag = "movement" if t["movement"] else "compute "
            sol = f"{t['sol_ms_call']:.5f} ms/call" if t["sol_ms_call"] is not None else "n/a"
            w(f"   [{tag}] {t['kernel']}   {t['bytes_per_call'] / 1e6:.3f} MB/call  sol {sol}")
            for src in t["source"][:3]:
                w(f"        {src['loc']:<26} {src['code']}")
        for e in s.externs:
            src = e["source"][0]["loc"] + " " + e["source"][0]["code"] if e["source"] else ""
            bmnk = e["bmnk"]
            shape = f"b{bmnk[0]} m{bmnk[1]} n{bmnk[2]} k{bmnk[3]}" if bmnk else "?"
            sol = f"{e['sol_ms_call']:.4f} ms/call" if e["sol_ms_call"] is not None else "n/a"
            w(f"   [gemm] {e['op']:<7} {shape:<28} sol {sol}   {src}")
    w("")
    w("CAVEATS:")
    for c in rep.caveats:
        w(f"  - {c}")
    return "\n".join(out)
