# Kernel→source feedback for the served Cosmos3 DiT

`cosmos3_compiled_kernel_analysis.py` profiles the Cosmos3-Nano DiT **as it is
actually served** — `torch.compile` (`compile_denoise=True`) + CUDA-graph capture
via the real `CudaGraphRunner` — and maps **every kernel** back to its source line,
ranked by execution time. It's a flat, per-kernel view: no op-family or component
aggregation, just "which kernel, how long, and which source line it came from."

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

## What it prints

1. **Per compiled subgraph** — each fused **Triton** kernel with the source line(s)
   it fuses (Inductor `output_code.py` provenance joined to `fx_graph_readable.py`)
   and its fused ATen ops, plus the **non-fused `extern_kernels` (cuBLAS/cuTLASS)
   calls** with their call sites.
2. **`EVERY KERNEL by execution time → source`** — the flat, ranked list: for each
   kernel (Triton, cuBLAS GEMM, FlashInfer, eager), its ms/step, call count, full
   name, and source mapping.

## How each kernel maps to source

| kernel type | source mapping |
| --- | --- |
| **Triton (fused)** | exact `transformer.py:line` + the line text, from Inductor provenance |
| **cuBLAS/cuTLASS GEMM/conv** | the `extern_kernels` call sites (library kernel — no per-kernel line) |
| **FlashInfer attention** | labeled (library kernel) |
| **eager `at::native`** | not in the Inductor graph; the kernel name carries the aten op — use the capture-time module-hook method to pin the exact line |

## How it handles the compile + CUDA-graph gotchas

- **compile silently no-ops on a cached eager submodule** → evicts
  `model._submodule_cache['dit']` before rebuilding with compile on.
- **CUDA-graph replay loses CPU↔kernel correlation** → attributes source at
  *compile* time (Inductor provenance) and measures time at *replay* time, then
  joins by kernel name.
- **Inductor serves compiles from disk cache and skips the debug dump** →
  `force_disable_caches` for a cold recompile.

## Recording the output

By **default** the report is tee'd (printed live **and** saved) to
`perf_testing/results/cosmos3_feedback_<timestamp>.txt` — that directory is
git-ignored. Control it with env vars:

```bash
COSMOS3_FEEDBACK_OUT=report.txt python perf_testing/cosmos3_compiled_kernel_analysis.py  # custom path
COSMOS3_FEEDBACK_OUT= python perf_testing/cosmos3_compiled_kernel_analysis.py             # console only
COSMOS3_KEEP_TRACE=1 python perf_testing/cosmos3_compiled_kernel_analysis.py              # keep Inductor dump
```

## Caveats

- The offline test scaffolding (`tests/test_engine_cache`) runs with
  `use_deterministic_algorithms(True)`, so a few `_assert_async` kernels appear
  that are absent in production. Treat absolute ms as indicative; the **relative
  ranking** and the **kernel→source mapping** are the deliverable.
- Library kernels (cuBLAS `nvjet`, cuTLASS, FlashInfer) are not Triton, so they
  map to call sites, not a single fused source line.
- eager kernels (the biggest ones, around the attention graph break) aren't in the
  Inductor graph — their source needs the capture-time module-hook profiler.
