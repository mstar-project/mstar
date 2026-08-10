# Kernel-level headroom feedback for served models

Two pieces:

- **`kernel_feedback/`** — a **model-agnostic** analysis package: parse a
  TorchInductor debug dump into exact per-kernel work (tensor traffic, GEMM
  FLOPs, source provenance), join with measured per-kernel times, score each
  kernel against a GPU spec (speed-of-light roofline), and bucket the gap into
  *kernel-quality* / *data-movement-elimination* / *fusion* headroom.
- **`cosmos3_compiled_kernel_analysis.py`** — the Cosmos3 driver: profiles the
  Cosmos3-Nano DiT **as it is actually served** (`compile_denoise=True` + the
  real `CudaGraphRunner` capture) and feeds the package. Other models need only
  a driver like this one; the package does the rest.

## Run

```bash
cd <mstar repo>
COSMOS3_NANO_DIR=/path/to/Cosmos3-Nano \
  CUDA_VISIBLE_DEVICES=<free gpu> python perf_testing/cosmos3_compiled_kernel_analysis.py
```

- Needs the `mstar` conda env (torch + flashinfer + diffusers), a CUDA GPU with
  ~24 GB free, and a local Cosmos3-Nano checkpoint (HF `nvidia/Cosmos3-Nano`).
- Takes ~2–3 min (it forces a cold recompile so Inductor re-emits provenance).

## What the report says

The point is to rank kernels by **recoverable ms**, not observed ms — a big
kernel that already runs at the hardware roofline is not an opportunity; a
mid-sized copy that shouldn't exist at all is.

1. **Waterfall** (ms/step, additive):
   `observed → kernel_quality_gap → speed_of_light → movement_elimination →
   fusion_in_graph → harness_artifacts → estimated_floor`
   - `kernel_quality_gap` — observed − roofline `max(flops/peak, bytes/HBM_bw)`
     for the work each kernel actually did: a better implementation of the
     *same* kernels.
   - `movement_elimination` — SOL of pure data-movement kernels (copies, casts,
     fills, cats): their necessary floor is ~0, a fused pipeline never
     materializes them.
   - `fusion_in_graph` — per compiled subgraph, sum of member SOLs minus the
     perfectly-fused floor `max(Σflops/peak, external-IO-bytes/bw)` (upper bound).
   - `graph_break_traffic` (informational) — bytes each compiled subgraph
     returns ×2 (write + re-read) / HBM bw: the price of each compile boundary.
2. **Top opportunities** — every kernel ranked by recoverable ms/step, with its
   analytic bytes/FLOPs, SOL, class, and the M\* source line(s) it comes from.
3. **Per-subgraph fusion view** — each Inductor graph's kernels + extern GEMM
   shapes (`b/m/n/k`, per-call SOL), boundary traffic, and its fusion gap.

Output goes to `perf_testing/results/cosmos3_feedback_<ts>.txt` **and a
machine-readable `.json`** (same basename; `schema_version`, buckets, all
kernel rows, embedded definitions) — the JSON is the input for cross-config
comparison later. `COSMOS3_FEEDBACK_OUT=path` overrides,
`COSMOS3_FEEDBACK_OUT=` prints to stdout, `COSMOS3_KEEP_TRACE=1` keeps the
Inductor dump.

## How each kernel gets its work + source attribution

| kernel type | work (flops/bytes) | source mapping |
| --- | --- | --- |
| **Triton (fused)** | exact tensor shapes/dtypes from the Inductor "Graph fragment" provenance | exact `file.py:line` + line text |
| **cuBLAS/`extern_kernels`** | exact `2·b·m·n·k` from resolved call-site shapes; SOL distributed over measured GEMM kernels proportionally | call-site line via ordered fx-node pairing |
| **eager glue** (around graph breaks) | shapes from a one-off **eager step** under `torch.profiler(with_stack, record_shapes)`; bytes assume bf16 | Python stack frames (project files) |
| **FlashInfer attention** | unmodeled — passes through at observed time | labeled library kernel |

Compile/CUDA-graph gotchas handled: submodule-cache eviction so compile isn't a
no-op, `force_disable_caches` so the provenance dump is emitted, times measured
at *replay*, work attributed at *compile*, joined by kernel name (replay loses
CPU↔kernel correlation), and eager attribution taken from a separate eager step
so the profiler never sees the multi-minute compilation.

## Hardware model

`kernel_feedback/gpu_spec.json` (21 GPUs, copied from VibeSim's catalog: dense
TFLOPS by dtype, HBM bandwidth, interconnect). Resolution is by name/alias from
`torch.cuda.get_device_name()`; an unknown GPU degrades to "no SOL column"
with a caveat rather than a fabricated ceiling. Because SOL/floor are pure
functions of `(flops, bytes, spec)`, the same measured run can be re-scored
against a different GPU's spec from the JSON alone.

## Caveats

- Absolute ms come from the offline test scaffolding
  (`use_deterministic_algorithms(True)`); induced `_assert_async` kernels are
  bucketed as `harness_artifacts`. Relative ranking + attribution are the point.
- Triton FLOPs are estimates (~1 flop/elem/arith-node) — irrelevant for SOL
  since fused elementwise chains are bandwidth-bound at any realistic intensity.
- `fusion_in_graph` and eager-movement byte estimates are upper bounds; each
  carries a note in the report.

## Legacy

`offline_homogenous.sh` — end-to-end batch-size sweeps over `benchmark/run_benchmark.sh`.
