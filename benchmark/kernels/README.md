# Kernel sweeps

Launch configs that depend on the GPU and the shapes, such as the fused MoE's tiles, are data:
one JSON table per (op, GPU) under `mstar/kernels/tables/<op>/`, rows keyed by token count.
`python -m benchmark.kernels.sweep` times candidate configs on the local GPU at given shapes and
token counts, checks each against a reference output, and writes the winners there, so an op
gets faster on a GPU without a change to model code.

## Tune a model's shapes

1. Serve the model. Each shape it runs without a table entry is logged once per process, with
   the command that tunes it:

   ```
   kernel table: no moe_gemm bfloat16 E=128,N=1536,K=2048,top_k=8 entry for NVIDIA H100 80GB HBM3; built-in configs. To tune it: python -m benchmark.kernels.sweep moe_gemm --dtype bfloat16 --shape E=128,N=1536,K=2048,top_k=8
   ```

2. Run it on one idle GPU of that model, with `--out` to a scratch directory and a `--note`
   saying what the shape is (model, node, TP):

   ```
   python -m benchmark.kernels.sweep moe_gemm --dtype bfloat16 \
       --shape E=128,N=1536,K=2048,top_k=8 --note "Qwen3-Omni thinker, TP1" --out ~/kernel-tables
   ```

   For each bucket (token count) it prints what runs now (`current`: the table's row, or the
   built-in config), the built-in config, the best candidate, and the chosen row, in
   microseconds per call, timed from a CUDA graph with cold weights. A candidate whose output is
   outside its implementation's tolerance of the reference output is dropped before it is timed,
   so it is never written, and neither is a current row that fails the same check. A bucket
   keeps its current row unless the best candidate beats it by `--min-gain` (2%), so re-sweeping
   a GPU rewrites only rows that measurably improve. `--fresh` decides from scratch, against the
   built-in config. Before writing, the sweep checks the entry the way the kernel's lookup does,
   so it never writes one a server would refuse.

3. Serve with `MSTAR_KERNEL_TABLES=~/kernel-tables` and A/B it against the same build without
   (see `.claude/skills/benchmarking`). Then `--write` puts the entry into the package's table,
   with the GPU, date, versions, commit and command it came from, to commit with the numbers.

`--list` prints every op and the shapes each table file covers.

## How a config is found

- A table file is named for one GPU model (its `torch.cuda.get_device_name()`, non-alphanumerics
  as `_`) or for an architecture (`sm_90`, `gfx942`). A lookup searches `MSTAR_KERNEL_TABLES`'
  directories, then the package's, the GPU's file before its architecture's, and takes the first
  entry for the dtype and shape. Without one, the kernel keeps its built-in config.
- A row names an implementation and its config, and takes the token counts up to its key; the
  `"*"` row, always last, takes every count above. Rows are checked when a table is read: a
  malformed table raises rather than silently stop applying.
- The code that launches an op looks its shape up once per process, never per step.
- Every rank of a node reads the same tables for the same GPU model, so the ranks of a lockstep
  instance pick the same configs. Set `MSTAR_KERNEL_TABLES` the same way on every node.

## Add an op

1. Declare a `KernelOp` next to its kernels (`mstar/kernels/registry.py` has the fields) and
   `register` it; add its module to `_OP_MODULES` there. Name the static dimensions its configs
   are keyed by (`shape`) and the dynamic one rows are keyed by (`bucket`), and list its
   implementations with their capability checks and sweep tolerances.
2. Where the op launches, look its entry up once (`op.lookup(...)`) and keep the built-in config
   where there is none. If that code runs inside a compiled forward, do as
   `mstar/utils/fused_moe/kernels.py`'s `get_config` does: read the rows in a function marked
   `torch.compiler.assume_constant_result`, with only static arguments (the weights' shapes,
   never the token count), and pick the row in traced code. The config then reaches the launch as
   a constant, which Triton launch options (`num_warps`) must be, and a symbolic token count only
   adds guards. Test it the way the engine compiles (`dynamic=False` and `dynamic=None`, with
   `mstar.engine` imported for its dynamo settings).
3. Write `benchmark/kernels/<op>.py` with a sweep class and add it to `OP_SWEEPS` in
   `sweep.py`. Its `run_bucket(m, current)` checks and times the candidates at `m` tokens and
   returns a `BucketResult`; its `check(entry)` raises where the kernel's lookup would refuse the
   entry; an op whose rows need more than merging runs of equal winners (`moe_decode`) also
   assembles them (`rows`).

## The ops

- `moe_gemm`: `fused_experts`' bf16 grouped GEMMs (Qwen3-Omni, Command A+): tiles per token
  count.
- `moe_decode`: the small-batch fp8 MoE's per-pair kernels (`--dtype float8_e4m3fn --shape
  H=...,E=...,I=...,K=...`: hidden, experts, one rank's intermediate, top-k). Its rows are the
  per-pair tiles by token count, then the grouped `"*"` row.
