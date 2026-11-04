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

Then drive it with mstar's own harnesses (`benchmark.runner`, `benchmark/worker_phases`).

## How it works

| Piece | What | File |
|---|---|---|
| Emulated device | `SimTensor` reports `cuda:N`. Small integer tensors (positions, page tables, lengths: what mstar plans with) keep real values; every other tensor is `meta` (shape and dtype, no data). A host read of a never-computed value reads zeros, so a sampled token is token 0: benchmark with fixed output lengths (`ignore_eos`). | `device.py` |
| Clock | Per-stream timelines on the host clock. A kernel starts at `max(launch, stream free)` and lasts its predicted time; events, `synchronize`, `.item()`/`.tolist()` and D2H copies wait for the device. The emulator's own CPU time is not counted: the clock stands still while it runs, so a launch is stamped when mstar made the call. A CUDA graph records its kernels' costs at capture and replays as one launch (a kernel priced from its arguments' values, such as attention over the latest plan, is priced again at each replay). | `clock.py`, `cuda.py` |
| Kernel times | A profile DB you collect, then AIConfigurator's tables (GEMM, prefill and decode attention, GDN, NCCL and custom all-reduce, per GPU), interpolated in utilization space; a roofline (FLOPs from `torch.utils.flop_counter`, bytes from tensor sizes, datasheet peaks) for anything uncovered. Attention is priced from each request's real query and KV lengths. | `cost.py`, `tables.py`, `gpus.py` |
| Memory | The simulated card's capacity; allocations counted by storage, so the KV pool mstar sizes from `mem_get_info` is what that card would hold. | `cuda.py` |
| Autocast | The engine runs forwards under CUDA autocast, which never sees the emulated tensors' payloads; its cast policy is applied by hand, so matmuls run (and are priced) in the autocast dtype. | `device.py` |
| Weights | Read from safetensors headers only; a model built on the CPU in a worker allocates nothing. | `patches.py` |
| Kernel stand-ins | A `flashinfer` package, mstar's GDN and causal-conv kernels, Triton launches: each is one priced kernel that returns its outputs' shapes (`clock.kernel_fn`). | `fake_libs/`, `kernels.py`, `triton_launch.py` |
| Multi-GPU | One process per emulated GPU, as on a node. NCCL groups become gloo; a collective over GPU tensors moves no data and is charged its NCCL time on every rank. | `dist.py` |
| Bootstrap | A `sitecustomize` the launcher puts on `PYTHONPATH` installs the emulator in every process. | `_site/`, `install.py`, `__main__.py` |

Prior art it follows: SGLang's simulator (real scheduler, predicted forward, fake tokens),
AIConfigurator (measured op tables, utilization-space interpolation), Vidur (per-request
attention features), Maya (one interposed process per device, symmetric collectives).

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
- A value that was never computed reads as zeros: data-dependent control flow on device values
  (beyond small integer tensors) takes the zero branch.
- MoE, MLA, grouped and fp4 GEMMs cost the roofline (~30% error in the literature, against
  ~5-10% for measured tables).
- Collectives assume symmetric ranks (no waiting on the slowest rank).
- Nothing checks numerics: a port's wrong math (a transposed weight, a bad mask) still serves.
- A FlashInfer API the stand-in lacks raises `NotImplementedError("mstar_sim: flashinfer.<name>
  has no stand-in yet")` when called.
