# Op-/kernel-level feedback for the served Cosmos3 DiT

`cosmos3_compiled_kernel_analysis.py` profiles the Cosmos3-Nano DiT **as it is
actually served** — `torch.compile` (`compile_denoise=True`) + CUDA-graph capture
via the real `CudaGraphRunner` — and backtracks every kernel to its `transformer.py`
source line, so you can see where the denoise step spends time and what to optimize.

## Run

```bash
cd <mstar repo>
COSMOS3_NANO_DIR=/path/to/Cosmos3-Nano \
  CUDA_VISIBLE_DEVICES=<free gpu> python perf_testing/cosmos3_compiled_kernel_analysis.py
```

- Needs the `mstar` conda env (torch + flashinfer + diffusers), a CUDA GPU with
  ~24 GB free, and a local Cosmos3-Nano checkpoint (HF `nvidia/Cosmos3-Nano`).
- `COSMOS3_NANO_DIR` defaults to a local path in the script — override via env.
- Takes ~2–3 min (it forces a cold recompile so Inductor re-emits provenance).

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
