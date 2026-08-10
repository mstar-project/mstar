# Op-/kernel-level optimization feedback for Cosmos3

Two profilers that report where the recoverable optimization headroom is, using
VibeSim-style roofline/optimality reasoning:

- **`cosmos3_optimality_report.py`** — the **hierarchical report**, in M*'s model
  abstraction: `Walk > component/node > op-class > kernel`. It profiles each
  component of a Walk (the served DiT denoise step and the Wan-VAE decode),
  weights by multiplicity (dit × `num_inference_steps`, vae × 1) so the ranking
  reflects real per-image priority, and tags each op-class's headroom with a
  named bucket (fusion / redundant / occupancy / at-roofline). **Start here** to
  decide *what* to optimize.
- **`cosmos3_compiled_kernel_analysis.py`** — the **kernel-level drill-down** for
  one component (the served DiT): every kernel backed to its `transformer.py`
  source line (Inductor provenance + extern-call parsing), the full per-step
  device breakdown incl. non-fused kernels, and the per-op-class optimality
  ladder + top-kernels-with-source. Use this once the report points you at a
  component.

Both share the same method (measured device time from CUDA-graph replay + an
analytic/FLOP-counted roofline + live-measured peaks) and the same caveats below.

## Run (kernel-level drill-down)

```bash
cd <mstar repo>
COSMOS3_NANO_DIR=/path/to/Cosmos3-Nano \
  CUDA_VISIBLE_DEVICES=<free gpu> python perf_testing/cosmos3_compiled_kernel_analysis.py
```

- Needs the `mstar` conda env (torch + flashinfer + diffusers), a CUDA GPU with
  ~24 GB free, and a local Cosmos3-Nano checkpoint (HF `nvidia/Cosmos3-Nano`).
- `COSMOS3_NANO_DIR` defaults to a local path in the script — override via env.
- Takes ~2–3 min (it forces a cold recompile so Inductor re-emits provenance).

## Run (hierarchical report)

```bash
COSMOS3_NANO_DIR=/path/to/Cosmos3-Nano \
  CUDA_VISIBLE_DEVICES=<free gpu> python perf_testing/cosmos3_optimality_report.py
```

Example (Cosmos3-Nano, 256×256, one B200) — the top of the report:

```
WALK TOTAL:  ~1085 ms   |   recoverable ~859 ms (79%)
  component        per-call ms    xN   walk ms  share  recoverable ms
  dit                   21.4      50      1072    99%             850
  vae_decoder           13.2       1        13     1%               9
COMPONENT: dit  ...
   op-class                measured roofline  gap  walk-headroom  bucket
   norm/adaLN/pointwise      15.98    0.089  179x           794   FUSION — unfused/eager pointwise ...
   attention                  0.66    0.018   37x            32   KERNEL/launch ...
   GEMM                       4.33    4.644    1x             0   at roofline — no headroom
```

`walk-headroom` = per-call headroom × multiplicity = recoverable ms per generated
image. It makes the DiT's per-step cost dominate (×50), which is the real priority.

## How to read the hierarchical report

Read it **top-down, biggest-number-first**. The mental model:

> **component share → op-class walk-headroom → bucket (the fix) → top kernels (the place).**

Rank at every level by the recoverable-ms column; the bucket names the fix; drill
to kernels/source for the exact edit.

**1. WALK TOTAL + component table — "where do I even look".**

| column | meaning |
| --- | --- |
| per-call ms | device time for one invocation of the component |
| xN | how many times it runs per generated image (`num_inference_steps` for dit, 1 for vae) |
| walk ms | per-call × N — the component's real cost per image |
| share | % of the whole walk |
| recoverable ms | headroom at walk scale — **the number to rank by** |

Read: the DiT is 99% of the walk and holds ~850 of the ~859 ms recoverable →
optimize the DiT. (A naive per-call view would wrongly make the vae's 13 ms look
comparable to the dit's 21 ms; the ×N weighting corrects that.)

**2. Component's op-class table — "what kind of fix".** Sorted by walk-headroom.

| column | meaning |
| --- | --- |
| measured | per-call device time for that op-class |
| roofline | hardware-necessary time `max(FLOPs/peak, bytes/peak)` |
| gap | measured/roofline — how far from optimal (higher = more waste) |
| walk-headroom | (measured − roofline) × N — recoverable ms per image |
| bucket | the **named fix** (fusion / redundant / occupancy / at-roofline) |

Read: `norm/adaLN/pointwise` at 179× / 794 ms, bucket **FUSION** → fuse the eager
pointwise. `GEMM` at gap 1×, 0 headroom → **leave it alone.**

**3. Op-class top kernels — "where in source".** The specific kernels carrying the
headroom. DiT Triton kernels also print their `transformer.py:line` + source text;
eager/library kernels have no Inductor source — switch to
`cosmos3_compiled_kernel_analysis.py` (or the capture-time module-hook profiler)
to pin the exact line.

**Reading caveats (so you don't over-trust a number):**
- **gap ≈ 1 means "at roofline, leave it"** — the peak benches slightly under-read
  (compute ~1600 vs B200's ~2250 TFLOP/s), so 0.9–1.0× is *done*, not sub-optimal.
- The `overhead` row and part of `norm/adaLN/pointwise` include `_assert_async`
  **determinism-artifact** kernels (harness runs with deterministic algorithms on),
  absent in production — that headroom is slightly inflated.
- Headroom is **per-op-class, not per-kernel** — a kernel inherits its class's numbers.
- Roofline is analytic *necessary* work (dit) / FLOP-counted (vae); treat gaps as
  **ROI signals** (which class to attack), not exact bounds.

The saved file under `perf_testing/results/` is byte-identical to the console output.

### Recording the output

By **default** the report is tee'd (printed live **and** saved) to
`perf_testing/results/cosmos3_feedback_<timestamp>.txt` — that directory is
git-ignored. Control it with env vars:

```bash
# default: auto-saved under perf_testing/results/
CUDA_VISIBLE_DEVICES=<gpu> python perf_testing/cosmos3_compiled_kernel_analysis.py

# choose the output path
COSMOS3_FEEDBACK_OUT=report.txt python perf_testing/cosmos3_compiled_kernel_analysis.py

# disable file recording (console only)
COSMOS3_FEEDBACK_OUT= python perf_testing/cosmos3_compiled_kernel_analysis.py

# also keep the raw Inductor dump (output_code.py, fx_graph_readable.py, ...)
COSMOS3_KEEP_TRACE=1 python perf_testing/cosmos3_compiled_kernel_analysis.py
```

## What it prints

1. **Per compiled subgraph**: the fused **Triton** kernels, each mapped to the
   source line(s) it fuses (Inductor `output_code.py` provenance joined to
   `fx_graph_readable.py`), plus the **non-fused `extern_kernels` (cuBLAS) calls**.
2. **A full measured per-step device breakdown** across ALL kernels — Triton
   (fused) / cuBLAS GEMM / FlashInfer attention / eager-uncompiled / … — from
   replaying the captured graph under `torch.profiler`.
3. **Fusion-boundary hot spots**: the source lines the most fused kernels touch.
4. **Optimality ladder (VibeSim-style)**: per op-class, the measured device time
   (R0) vs the hardware-necessary roofline (R6 = `max(FLOPs/peak_compute,
   bytes/peak_bw)` over the class's analytic necessary work), the `gap`
   (measured/roofline), and the recoverable `headroom` (measured − roofline) —
   ranked, so you pick the next kernel-level optimization by ROI. Peaks are
   measured on the live GPU (a large GEMM and a DtoD copy).

## How it handles the compile + CUDA-graph gotchas

- **compile silently no-ops on a cached eager submodule** → evicts
  `model._submodule_cache['dit']` before rebuilding with compile on.
- **CUDA-graph replay loses CPU↔kernel correlation** → attributes source at
  *compile* time (Inductor provenance) and measures time at *replay* time, then
  joins by kernel name.
- **Inductor serves compiles from disk cache and skips the debug dump** →
  `force_disable_caches` for a cold recompile.

## Reading the numbers

- The offline test scaffolding (`tests/test_engine_cache`) runs with
  `use_deterministic_algorithms(True)`, so a few `_assert_async` kernels appear
  that are absent in production. Treat absolute ms as indicative; the **relative
  split** and the **kernel→source mapping** are the deliverable.
- Library kernels (cuBLAS `nvjet`, FlashInfer) are not Triton, so they show in
  the full breakdown but not in the fused-kernel source map.

## Headline finding (Cosmos3-Nano, 256×256, one B200)

`torch.compile` fuses the pointwise/norm/RoPE tail into ~27 Triton kernels but,
because `fullgraph=False` breaks at the FlashInfer attention, it captures only
~3% of the step. The dominant cost is **uncompiled eager pointwise** (~70% —
adaLN modulation / gated residuals around the attention breaks), then cuBLAS
GEMMs (~20%). The optimality ladder makes the ROI explicit: of a ~21 ms/step,
only ~5 ms is hardware-necessary (R6); **~79% (~17 ms) is recoverable**, almost
all of it in the `norm/adaLN/pointwise` class (gap ~185×). GEMMs sit at their
bandwidth roofline (no headroom). The top lever is extending the compiled region
past the attention breaks (or a fused adaLN kernel), not the GEMMs.

Caveats: absolute ms carry the determinism-artifact noise noted above, and the
roofline uses live-measured peaks + analytic necessary work (adaLN/gating counted
as memory-bound pointwise), so treat gaps as order-of-magnitude ROI signals.
