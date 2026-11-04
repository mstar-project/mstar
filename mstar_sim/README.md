# mstar_sim: GPU-time simulation for mstar

Run mstar on a box with no GPU and get GPU timing. `mstar_sim` is an add-on: it
changes no mstar file. mstar believes it is on CUDA (an H100 by default) and
runs its real code: API server, conductor, worker, micro-scheduler, async
worker, engine, resources, FlashInfer attention wrappers, CUDA-graph capture
and replay, the sampler. No kernel is computed. Each is charged its time on
the simulated GPU, taken from NVIDIA AIConfigurator's measured per-GPU tables
where they cover it and a roofline elsewhere, on per-stream timelines that the
host waits on exactly where CUDA would make it wait.

```bash
# serve on emulated GPUs; any mstar command works
python -m mstar_sim --gpu h100 serve qwen3_5_0.8b
python -m mstar_sim --gpu b200 serve qwen3_5_27b            # its TP4 config: four emulated B200s

# a full-size model with no weights: headers over sparse files, ~20 MB for 27B
python -m mstar_sim.checkpoint Qwen/Qwen3.5-27B ~/ckpt/q27

# any Python command (tests, benchmarks) with every process on the emulated GPU
python -m mstar_sim --gpu h100 -- -m pytest test/modular/test_x.py
```

Then drive it with mstar's own harnesses (`benchmark.runner`, `benchmark/worker_phases`). To run your own code in every process the launcher starts (fault injection, tracing, a per-rank override), put a `sitecustomize.py` on `PYTHONPATH`: the emulator's own `sitecustomize` runs it after installing.

## How it works

| Piece | What | File |
|---|---|---|
| Emulated device | `SimTensor` reports `cuda:N`. Small tensors (up to 4096 elements of any dtype, or 1M integers: positions, page tables, lengths, what mstar plans with) keep real values; every other tensor is `meta` (shape and dtype, no data). A host read of a never-computed value reads zeros, so a sampled token is token 0: benchmark with fixed output lengths (`ignore_eos`). | `device.py` |
| Clock | Per-stream timelines on the host clock. A kernel starts at `max(launch, stream free)` and lasts its predicted time; events, `synchronize`, `.item()`/`.tolist()` and D2H copies wait for the device. The emulator's own CPU time is not counted: the clock stands still while it runs, so a launch is stamped when mstar made the call. A CUDA graph records its kernels' costs at capture and replays as one launch (a kernel priced from its arguments' values, such as attention over the latest plan, is priced again at each replay). | `clock.py`, `cuda.py` |
| Kernel times | A profile DB you collect, then AIConfigurator's tables (GEMM, prefill and decode attention, GDN, NCCL and custom all-reduce, per GPU), interpolated in utilization space; a roofline (FLOPs from `torch.utils.flop_counter`, bytes from tensor sizes, datasheet peaks) for anything uncovered. Attention is priced from each request's real query and KV lengths. | `cost.py`, `tables.py`, `gpus.py` |
| Memory | The simulated card's capacity, accounted as torch's caching allocator does: allocated bytes by storage; reserved bytes per pool, the default one and each graph's private pool, at its high-water mark until `empty_cache` (or until the graph goes); `mem_get_info` and `memory_snapshot` from those. A reservation past the card raises `torch.cuda.OutOfMemoryError`. So the KV pool mstar sizes from `mem_get_info`, and what a capture takes, are what that card would see. | `device.py`, `cuda.py` |
| Autocast | The engine runs forwards under CUDA autocast, which never sees the emulated tensors' payloads; its cast policy is applied by hand, so matmuls run (and are priced) in the autocast dtype. | `device.py` |
| Weights | Read from safetensors headers only; a model built on the CPU in a worker allocates nothing. | `patches.py` |
| Kernel stand-ins | A `flashinfer` package, mstar's GDN and causal-conv kernels, Triton launches: each is one priced kernel that returns its outputs' shapes (`clock.kernel_fn`). | `fake_libs/`, `kernels.py`, `triton_launch.py` |
| Multi-GPU | One process per emulated GPU, as on a node. NCCL groups become gloo; a collective over GPU tensors moves no data and is charged its NCCL time on every rank. | `dist.py` |
| Bootstrap | A `sitecustomize` the launcher puts on `PYTHONPATH` installs the emulator in every process. | `_site/`, `install.py`, `__main__.py` |

Prior art it follows: SGLang's simulator (real scheduler, predicted forward, fake tokens),
AIConfigurator (measured op tables, utilization-space interpolation), Vidur (per-request
attention features), Maya (one interposed process per device, symmetric collectives).

## Fast mode, for bug repros

`--fast` (`MSTAR_SIM_FAST=1`) keeps everything a repro of host control flow depends on: the scheduler, admission, KV and lease bookkeeping, TP agreement, CUDA-graph capture and replay, refusal of host reads during capture, and small-int values. It drops all timing: no kernel is priced or queued, and no host read waits. The clock is the real, monotonic one, so mstar's own timeouts still fire. Timed runs are unchanged without the flag.

Boot is mostly mstar sizing each capture bucket with one eager forward. `--max-capture-tokens N` drops large token buckets. `--max-capture-batch N` does the same for batch buckets, but it also caps the served batch at N (mstar caps eager batches at the largest captured one).

## Repro tips

- **Models by name, no weights.** `python -m mstar_sim.checkpoint <repo> --hf-home <dir>` writes a shapes-only copy into a Hugging Face cache (headers over sparse files, about 20 MB for 27B). Run with `HF_HOME=<dir> HF_HUB_OFFLINE=1`, and `from_pretrained`, `snapshot_download` and `hf_hub_download` find it.
- **GPU-only tests.** `python -m mstar_sim --fast -- -m pytest -p mstar_sim.pytest_plugin --gpu-only <paths>` runs the tests a CPU box skips for want of a GPU. It reports value checks the emulator can't answer as expected failures, and lists its own gaps at the end.
- **Timing-shaped bugs** (queues that fill while the GPU is busy): drop `--fast`, and slow the GPU with `--time-scale K`. Under `--fast` every CUDA event has completed by the time anyone asks, so code that branches on `event.query()` or a pending copy only takes its "still pending" path in a timed run.
- **Hangs.** `kill -USR1 <server pid>` prints every thread's stack in that process and in all its children (the workers).
- **Your own probes** (a `sitecustomize.py` on `PYTHONPATH`):
  - They also run in the `python -m mstar_sim` process itself, where the tree isn't on `sys.path` yet. Guard their mstar imports, or put the tree first on `PYTHONPATH`.
  - They run after the emulator installs, by which time mstar modules are already imported. Patch what's in `sys.modules` as well as hooking imports.
  - A probe run against several commits (buggy and fixed) has to tolerate what an older tree lacks: check for the attribute (`getattr(cls, name, None)`) and skip it if missing, and catch `ModuleNotFoundError` for a module only one tree has.
  - A `builtins.__import__` hook that imports inside itself recurses; guard it with a flag. `sys.modules` can hold a module that is still initialising (`partially initialized module`): patch it once its attribute exists, not on first sight.
- **Modules loaded by file** (`spec_from_file_location`, as Qwen3-TTS loads `qwen_tts`) bypass import hooks. Patch the dependency's own module instead (e.g. `transformers.models.mimi`).
- **`-- script.py` with multiprocessing.** The script needs an `if __name__ == "__main__":` guard. Spawned children re-import it, and without the guard the conductor dies ("is not going to be frozen") and the launcher hangs.
- **A small `/dev/shm`.** `--shm-mb N` makes mstar's tmpfs sizing see an N MiB `/dev/shm` (`statvfs`). Free space is N less what `mstar_*` files there hold, a dead run's leftovers included. Writes past N still succeed.
- **Serving a config.** It's `python -m mstar_sim … serve <model> --config <yaml>`; the CLI takes a model name, not a YAML path.
- **Assets the model `torch.load`s** (a vocoder's `hift.pt`): the checkpoint keeps them shape-only, a few KB each, after downloading each one once. Large non-torch assets (`.onnx`) are skipped, so a model that runs them needs the real file next to the checkpoint.
- **Small pools.** A config with a small cross-attention or KV pool can fail capture planning, because each bucket's plan needs room for its whole batch. Cap the buckets (`--max-capture-batch 8`) rather than shrinking the pool below that.

## Measured kernel times

By default GEMMs, attention, GDN and causal-conv kernels cost the latencies NVIDIA
AIConfigurator measured on the real GPU (Apache-2.0 data for H100/H200/B200/A100/L40S:
SGLang's newest GEMM, context/generation attention and GDN tables, NCCL and custom
all-reduce), interpolated in utilization space (`util = speed of light / latency`): an
exact hit is the measured time, a point between grid points blends its neighbours' util
per axis in log space, a point past the grid holds the boundary util, and nothing goes
under the family's smallest measured kernel. Whatever the tables don't cover (another
head_dim, fp32, MoE, MLA, a GPU without data) costs the roofline. See `tables.py`.

```bash
# once: the commit whose table layout tables.py reads (tables.AIC_COMMIT)
git clone https://github.com/ai-dynamo/aiconfigurator ~/.cache/mstar_sim/aiconfigurator
git -C ~/.cache/mstar_sim/aiconfigurator checkout 6cd6f9686ab7de6e8fb28171d6e350ee0aa4ed39
python -m mstar_sim --cost tables ...     # default when the data exists; --cost roofline turns it off
```

Elsewhere, point `MSTAR_SIM_AIC_DATA` at the checkout's `src/aiconfigurator/systems/data`. Without
the data the launcher says so and every kernel costs the roofline.

The tables are read once per GPU into `~/.cache/mstar_sim/aic-<gpu>-<hash>.npz` (keyed
by the parquet files' mtimes); a process then builds each family on first use (~0.1 s,
not charged as device time).

For kernels the tables miss, or for this workload's exact build, record what a simulated
run launches and measure it once on the real GPU (ServingStudio-style):

```bash
python -m mstar_sim --gpu h100 --trace trace.jsonl serve ...      # CPU box: run the workload
python -m mstar_sim.profile --trace trace.jsonl --out h100.json    # GPU box: time each kernel
python -m mstar_sim --gpu h100 --profile-db h100.json serve ...    # CPU box: measured times
```

An op in the DB costs its measured time; the same op at another token count costs its
roofline scaled by the measured/roofline ratio; anything else costs the tables' time, else
the roofline.

## Limits

- **Host time is real time on this box.** mstar's own CPU work (scheduling, planning, Python) runs
  at this CPU's speed, and the emulator's time is hidden only while the device is busy: a step
  whose emulation outlasts its device time reads host-bound. A different CPU or a loaded box moves
  host-bound results; device time is unaffected (the report's `device_busy_s` against `wall_s`
  says which bound a run was).
- `--eager-nodes A,B` runs just those nodes eagerly. Use it for a node whose capture needs a real
  Triton build.
- `--no-graphs` changes admission: without capture buffers the sampler never asks admission for
  room, so admission paths that depend on it only run with graphs on.
- A value that was never computed reads as zeros: data-dependent control flow on device values
  (beyond small integer tensors) takes the zero branch.
- MoE, MLA, grouped and fp4 GEMMs cost the roofline (~30% error in the literature, against
  ~5-10% for measured tables).
- Collectives assume symmetric ranks (no waiting on the slowest rank).
- Nothing checks numerics: a port's wrong math (a transposed weight, a bad mask) still serves.
- A FlashInfer API the stand-in lacks raises `NotImplementedError("mstar_sim: flashinfer.<name>
  has no stand-in yet")` when called.
