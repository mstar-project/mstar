# Kernel sweeps

Tuned kernel configs are data: one JSON table per (op, GPU) under `mstar/kernels/tables/<op>/`.
`python -m benchmark.kernels.sweep` measures an op's candidates at given shapes on the local GPU
and writes the winners there, so a kernel gets faster on a GPU without a change to model code.

## Tune a model's shapes

1. Serve the model. Each shape it runs without a tuned entry is logged once per process, with
   the command that tunes it:

   ```
   kernel table: no moe_gemm bfloat16 E=128,N=1536,K=2048,top_k=8 (triton) entry for NVIDIA H100 80GB HBM3; default configs. To tune it: python -m benchmark.kernels.sweep moe_gemm --dtype bfloat16 --shape E=128,N=1536,K=2048,top_k=8
   ```

2. Run it on one idle GPU of that model, with `--out` to a scratch directory and a `--note`
   saying what the shape is (model, node, TP):

   ```
   python -m benchmark.kernels.sweep moe_gemm --dtype bfloat16 \
       --shape E=128,N=1536,K=2048,top_k=8 --note "Qwen3-Omni thinker, TP1" --out ~/kernel-tables
   ```

   For each bucket (token count, for `moe_gemm`) it prints what runs now (`current`: the table's
   row, or the default), the op's default, and the best candidate, in microseconds per call, timed
   from a CUDA graph. A candidate whose output differs from the default config's beyond accumulation
   order is dropped before it is timed. A bucket keeps its current row unless the best candidate
   beats it by `--min-gain` (2%), so re-sweeping a GPU rewrites only rows that measurably improve.

3. Serve with `MSTAR_KERNEL_TABLES=~/kernel-tables` and A/B it against the same build without
   (see `.claude/skills/benchmarking`). Then `--write` puts the entry into the package's table,
   with the GPU, date, versions, commit and command it came from, to commit with the numbers.

`--list` prints every op and the shapes each table file covers.

`moe_gemm` takes `--dtype float8_e4m3fn` for fp8 block-scale experts (`fused_experts_fp8`, W8A8
with 128 x 128 weight scale blocks): its rows keep `BLOCK_SIZE_K` at the block's 128, which the
kernel's per-tile rescale needs, and the server logs that command for an fp8 shape it runs untuned.

## How a config is chosen

- A table file is named for one GPU model (its `torch.cuda.get_device_name()`, non-alphanumerics
  as `_`) or for an architecture (`sm_90`, `gfx942`). A lookup searches `MSTAR_KERNEL_TABLES`'
  directories, then the package's, the GPU's file before its architecture's, and takes the first
  entry with the same dtype, shape and backend.
- Rows are keyed by the op's bucket; the nearest one applies (the smaller on a tie).
- The component that launches the op resolves its config once per shape and process, at load or
  during capture, never per step. Inside a compiled forward the lookup is a constant to dynamo.
- Every rank of a node reads the same tables for the same GPU model, so the ranks of a lockstep
  instance pick the same config. Set `MSTAR_KERNEL_TABLES` the same way on every node.

## Add an op

1. Declare a `KernelOp` next to its kernels (`mstar/kernels/registry.py` has the fields) and
   `register` it; add its module to `_OP_MODULES` there. Name the static dimensions that select a
   config (`shape`) and the dynamic one rows are keyed by (`bucket`), and give each backend a check
   that says why it can't run a (device, dtype, shape).
2. Where the op launches, look up its entry once per shape (`op.lookup(...)`) and keep the default
   where there is none. If that code runs inside a compiled forward, do as
   `mstar/utils/fused_moe/kernels.py`'s `get_config` does: resolve the entry's rows in a function
   marked `torch.compiler.assume_constant_result`, with only static arguments (the weights'
   shapes, never the token count), and pick the row in traced code. The config then reaches the
   launch as a constant, which Triton launch options (`num_warps`) must be, and a symbolic token
   count only adds guards. A graph break between the lookup and the launch turns the config into
   data, so keep that path break-free, and test it the way the engine compiles (`dynamic=False`
   and `dynamic=None`, with `mstar.engine` imported for its dynamo settings).
3. Write `benchmark/kernels/<op>.py` with a class like `MoeGemmSweep` (candidates, a correctness
   check against a reference, timing in a CUDA graph) and add it to `OP_SWEEPS` in `sweep.py`.
