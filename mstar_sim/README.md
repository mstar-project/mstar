# mstar_sim: run mstar without a GPU, for correctness testing

Run mstar on a box with no GPU and test that it behaves correctly. **Use it for
correctness, not performance: kernel times come from numbers measured on real GPUs
where tables cover a kernel and a roofline elsewhere, the end-to-end timing hasn't
been checked against a real GPU, and the host overhead here is too different from a
real run, so don't quote its throughput or latency numbers, or compare two changes
by them.** It computes no kernels, so it can't check model output either. What it
does find is bugs in mstar's own code: crashes, hangs, lifecycle and admission
errors, bad indices, out-of-memory, API mistakes.

`mstar_sim` is an add-on: it
changes no mstar file. mstar believes it is on CUDA (an H100 by default) and
runs its real code: API server, conductor, worker, micro-scheduler, async
worker, engine, resources, FlashInfer attention wrappers, CUDA-graph capture
and replay, the sampler. No kernel is computed. Each is charged its time on
the simulated GPU, taken from NVIDIA AIConfigurator's measured per-GPU tables
where they cover it and a roofline elsewhere, on per-stream timelines that the
host waits on exactly where CUDA would make it wait. That timing exists to give
mstar's host code realistic orderings (queues that fill while the GPU is busy), not
to predict performance.

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
| Clock | One clock per process, which every time source mstar reads (`time.*`, `threading`, `queue`) shows: mstar's own host work at its real cost, device time at its predicted cost, the emulator's time at zero. Per-stream timelines on that clock: a kernel starts at `max(launch, stream free)` and lasts its predicted time; events, `synchronize`, `.item()`/`.tolist()` and D2H copies wait for the device. A wait for the device doesn't sleep: once every thread of the process is in a wait the emulator knows, the clock jumps to the device's time. A CUDA graph is a tape recorded at capture: its kernels' costs and its work on the values the emulator keeps. Capture runs nothing; each replay runs that work in order, as one launch (a kernel priced from its arguments' values, such as attention over the latest plan, is priced again at each replay). | `clock.py`, `waits.py`, `cuda.py` |
| Kernel times | A profile DB you collect, then AIConfigurator's tables (GEMM, prefill and decode attention, GDN, NCCL and custom all-reduce, per GPU), interpolated in utilization space; a roofline (FLOPs from `torch.utils.flop_counter`, bytes from tensor sizes, datasheet peaks) for anything uncovered. Attention is priced from each request's real query and KV lengths. | `cost.py`, `tables.py`, `gpus.py` |
| Memory | The simulated card's capacity, accounted as torch's caching allocator does: allocated bytes by storage; reserved bytes per pool, the default one and each graph's private pool, at its high-water mark until `empty_cache` (or until the graph goes); `mem_get_info` and `memory_snapshot` from those. A block is rounded to 512 B and a pool grows by the allocator's segment (2 MiB, 20 MiB, or the block rounded to 2 MiB). A reservation past the card raises `torch.cuda.OutOfMemoryError`. It is a lower bound, per process: the CUDA context, cuBLAS and NCCL workspaces and fragmentation are not counted, and each process sees the whole card, so workers sharing a GPU (MPS) each size from all of it. | `memory.py`, `cuda.py` |
| Autocast | The engine runs forwards under CUDA autocast, which never sees the emulated tensors' payloads; its cast policy is applied by hand, so matmuls run (and are priced) in the autocast dtype. | `device.py` |
| Weights | Small tensors (under 65,536 elements: norms, fp8 scales) are read for real; the rest from a copy's safetensors headers or value-less torch files, or a large `.onnx` model's holes. A model built on the CPU in a worker allocates nothing. | `patches.py`, `checkpoint.py` |
| Kernel stand-ins | A `flashinfer` package, mstar's GDN and causal-conv kernels, Triton launches, the ops of a CUDA extension `cpp_extension.load` builds (their `TORCH_LIBRARY` schemas): each is one priced kernel that returns its outputs' shapes (`clock.kernel_fn`). A JIT-built op's outputs are never computed, where a CPU box's torch fallback would compute them (the fused MoE's sorted ids, expert ids and padded count). FlashInfer's planners lay out their workspaces by a port of FlashInfer's own sizing (0.7.2's `scheduler.cuh`), so a workspace too small for a plan raises its overflow. `tests/planner_golden.json` holds what the real planner answered on 140 named cases, on the device properties it stores (the emulated GPUs' when it was generated), and a test holds the port to it; a new FlashInfer means regenerating it on a box with a CUDA toolkit (`python -m mstar_sim.planner_golden`). | `fake_libs/`, `kernels.py`, `triton_launch.py`, `cpp_extension.py` |
| Multi-GPU | One process per emulated GPU, as on a node. NCCL groups become gloo. Ranks meet at each collective over GPU tensors and at each barrier, as NCCL's do, and a rank whose collective doesn't pair with its peers' (skipped, reordered, another size) aborts, printing every rank's; a captured graph meets once per group at each replay, on its collectives over that group's ranks. Data moves only if every rank holds values; each collective is charged its NCCL time on every rank. | `dist.py` |
| Bootstrap | A `sitecustomize` the launcher puts on `PYTHONPATH` installs the emulator in every process. The launcher also puts the driver's `nvidia-cuda-mps-control` on `PATH`, a stand-in that runs mstar's MPS daemon lifecycle (`mps: true`). | `_site/`, `_bin/`, `install.py`, `__main__.py` |

Prior art it follows: SGLang's simulator (real scheduler, predicted forward, fake tokens),
AIConfigurator (measured op tables, utilization-space interpolation), Vidur (per-request
attention features), Maya (one interposed process per device, symmetric collectives).

## Fast mode, for bug repros

`--fast` (`MSTAR_SIM_FAST=1`) keeps everything a repro of host control flow depends on: the scheduler, admission, KV and lease bookkeeping, TP agreement, CUDA-graph capture and replay, refusal of host reads during capture, and small tensors' values. It drops all timing: no kernel is priced or queued, and no host read waits. The clock is the real, monotonic one, so mstar's own timeouts still fire. Timed runs are unchanged without the flag.

Boot is mostly mstar sizing each capture bucket with one eager forward. `--max-capture-tokens N` drops large token buckets. `--max-capture-batch N` does the same for batch buckets, but it also caps the served batch at N (mstar caps eager batches at the largest captured one).

## Repro tips

- **Models by name, no weights.** `python -m mstar_sim.checkpoint <repo> --hf-home <dir>` writes a shapes-only copy into a Hugging Face cache (headers over sparse files that hold only the small tensors' bytes, about 20 MB for 27B). Run with `HF_HOME=<dir> HF_HUB_OFFLINE=1`, and `from_pretrained`, `snapshot_download` and `hf_hub_download` find it. Give it the Hub id: a config-only snapshot in another cache has nothing to copy, and mstar refuses a local path for `model_path_hf`, so `--hf-home` with `HF_HOME` is the way in.
- **GPU-only tests.** `python -m mstar_sim --fast -- -m pytest -p mstar_sim.pytest_plugin --gpu-only <paths>` runs the tests a CPU box skips for want of a GPU. It reports value checks the emulator can't answer as expected failures, whether they failed or passed (a test whose own code read a never-computed value: such a check can't answer), and lists its own gaps at the end. The plugin's docstring says what a pass does and doesn't show.
- **Timing-shaped bugs** (queues that fill while the GPU is busy): drop `--fast`, and slow the GPU with `--time-scale K`. Under `--fast` every CUDA event has completed by the time anyone asks, so code that branches on `event.query()` only takes its "still pending" path in a timed run. A copy's value lands when it is launched, in either mode.
- **Hangs.** `kill -USR1 <server pid>` prints every thread's stack in that process and in all its children (the workers).
- **Your own probes** (a `sitecustomize.py` on `PYTHONPATH`):
  - They also run in the `python -m mstar_sim` process itself, where the tree isn't on `sys.path` yet. Guard their mstar imports, or put the tree first on `PYTHONPATH`.
  - They run after the emulator installs, by which time mstar modules are already imported. Patch what's in `sys.modules` as well as hooking imports.
  - A probe run against several commits (buggy and fixed) has to tolerate what an older tree lacks: check for the attribute (`getattr(cls, name, None)`) and skip it if missing, and catch `ModuleNotFoundError` for a module only one tree has.
  - A `builtins.__import__` hook that imports inside itself recurses; guard it with a flag. `sys.modules` can hold a module that is still initialising (`partially initialized module`): patch it once its attribute exists, not on first sight.
- **Holding one request's step, jitter, a mid-step abort.** Stage them in your `sitecustomize`, not with an emulator knob. The GPU thread calls `Engine.prepare_inputs(self, batch)` just before each forward, so wrap it:
  ```python
  import os, time
  try:
      from mstar.engine.engine import Engine
  except ImportError:  # the launcher process has no mstar on its path
      Engine = None
  if Engine is not None:
      prepare = Engine.prepare_inputs
      def held(self, batch):
          # node X, and request N by its wire id (batch.request_ids are worker-local, recycled handles)
          if batch.node_name == "LLM" and any(i.request_id == "N" for i in batch.per_request_info.values()):
              while not os.path.exists("/tmp/gate"):  # or time.sleep(T)
                  time.sleep(0.05)
          return prepare(self, batch)
      Engine.prepare_inputs = held
  ```
  - A mid-step abort is the same recipe: hold the step, send the abort, then open the gate.
  - For jitter, set `mstar_sim.clock.TIME_SCALE` to a random factor in that wrapper, once per step (timed mode), or sleep a random time under `--fast`.
  - Under TP, hold and jitter every rank of the instance alike: match on `batch.node_name` on all ranks, and draw the factor from a seed the instance shares plus the step count. A rank held alone leaves its peers waiting at their next collective until it aborts at the group's timeout, which is a GPU hang, not the race you meant to stage.
- **KV state over time.** A daemon thread in your `sitecustomize` gives the timeline: find the `KVManager` instances once (`gc.get_objects()`), then every 0.5 s print what you want as JSON when it changes, e.g. `kv.page_counts()` and `getattr(kv, "_unused", None)`. Read fields with `getattr(kv, name, None)`, since they move between commits.
- **Modules loaded by file** (`spec_from_file_location`, as Qwen3-TTS loads `qwen_tts`) bypass import hooks. Patch the dependency's own module instead (e.g. `transformers.models.mimi`).
- **`-- script.py` with multiprocessing.** The script needs an `if __name__ == "__main__":` guard. Spawned children re-import it, and without the guard the conductor dies ("is not going to be frozen") and the launcher hangs.
- **A small `/dev/shm`.** `--shm-mb N` makes mstar's Python tmpfs sizing see an N MiB `/dev/shm` (`os.statvfs`). Free space is N less what `mstar_*` files there hold, a dead run's leftovers included. Writes past N still succeed, and the Rust arena's growth check (`fstatvfs` in `rust/src/shm.rs`) still sees the real tmpfs, so a growth failure past N doesn't reproduce.
- **Serving a config.** It's `python -m mstar_sim … serve <model> --config <yaml>`; the CLI takes a model name, not a YAML path. `serve` has no per-request timeout flag; the API server's entrypoint does: `python -m mstar_sim -- -m mstar.api_server.entrypoint --config <yaml> --tensor-comm-protocol SHM --timeout 30`.
- **Weights outside safetensors** (a vocoder's `hift.pt`, Kokoro's `.pth`, CosyVoice3's `speech_tokenizer_v3.onnx`): the checkpoint keeps them shape-only: a torch file is downloaded once and saved again in a few KB; an ONNX model over 50 MiB is read by ranges, only its graph and small tensors (a few MB), and its large weights' data is never fetched. An `.onnx` under that is copied whole. Other large files are skipped.
- **What a copy can't serve.** A tensor at or over `SMALL_WEIGHT` (65,536 elements) is value-less in the copy, so a load-time check of its values against the model's own fails: Fish S2 Pro's codec RoPE tables (262,144 and 1,048,576 elements) are refused at boot. In an ONNX model over 50 MiB such a tensor is a hole that reads as zeros, unmarked: fine for what mstar reads into torch modules, but a model that runs the file (onnxruntime) runs it on zeros with no error, and needs the real file next to the checkpoint. The manifest's notes name each one. A tokenizer loaded at the first request can ask the cache for files its repo doesn't have (Wan2.2's `AutoTokenizer` asks for `tokenizer/config.json`): an online run records the miss in the cache (`.no_exist`), a copy doesn't, so offline every request fails after a ready boot. Load it once with the Hub reachable into the same `HF_HOME` (`HF_HOME=<dir> python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Wan-AI/Wan2.2-TI2V-5B-Diffusers', subfolder='tokenizer')"`).
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

- mstar's own CPU work (scheduling, planning, Python) runs at this CPU's speed, so a different CPU
  moves host-bound results. A loaded box moves them less: time the box took from the emulator
  isn't counted, but time it took from mstar's own code is. Device time is unaffected.
- A thread in a wait the emulator doesn't know (a plain lock, `Thread.join`, a socket) counts as
  running, so its process sleeps for the device instead of jumping: `--report` counts that time
  (`slept_s`) and names the call sites (`jump_blockers`).
- Each process has its own clock: clients and the API server read real time less their own blocked
  and emulator time, which isn't the workers' time.
- `--eager-nodes A,B` runs just those nodes eagerly. Use it for a node whose capture needs a real
  Triton build.
- `--no-graphs` changes admission: without capture buffers the sampler never asks admission for
  room, so admission paths that depend on it only run with graphs on.
- A value that was never computed reads as zeros: data-dependent control flow on device values
  (beyond small tensors) takes the zero branch. A small tensor counts as never computed once a
  stand-in wrote it (a Triton launch: any tensor it was handed) and after an op reads such a
  value; a real write of all of it ends that, and a real write of part doesn't. An allocation
  nothing wrote also reads as zeros, and doesn't count.
- Exporting a kept tensor for IPC moves its storage into shared memory, so its `data_ptr()`
  changes at the export, which a GPU's doesn't.
- MoE, MLA, grouped and fp4 GEMMs cost the roofline (~30% error in the literature, against
  ~5-10% for measured tables). Those figures are for single kernels; end-to-end numbers
  here are not validated and not accurate (see the top of this file).
- Collective timing assumes symmetric ranks: no rank's device waits on a slower one's.
- A host blocks at an eager collective's call until its peers come, where a GPU's host only
  enqueues it. Device times are unchanged, but the host waits for its slowest peer to launch, and
  host-side gloo traffic (an `agree` on a CPU group) or another stream's collectives, interleaved
  with device collectives in a different order on different ranks, hang here where a GPU wouldn't.
  `send`/`recv` (and `isend`/`irecv`) aren't met: a send posts its descriptor, then its data if it
  holds values, so what pairs on a gloo group pairs here. A receive keeps that data only if it
  holds values too, and aborts on a size or dtype that doesn't pair, or a sender that never comes,
  as a collective does. A receive from any source raises, as NCCL's does.
- A skipped eager collective shifts its group's pairing, which shows at the next collective of
  another shape: between identical ones (per-layer all-reduces outside a graph) it is silent
  until then, as NCCL's corruption is. Inside a graph, the replay's sequence hash catches it.
- A peer that never comes aborts an eager collective at the group's timeout (gloo's 30 min, or
  mstar's `dist_timeout_s` while loading), as NCCL's watchdog does; a replay hangs, as an
  unwatched one does on a GPU. `kill -USR1` prints the peers' stacks.
- `torch.compile` runs its function eagerly: Dynamo's guards, recompiles and graph breaks on
  mstar's compiled paths aren't exercised. A test that calls `torch._dynamo.explain` itself
  traces, but CPU torch's Dynamo can't see the emulated GPU, which the plugin reports as a gap.
- Nothing checks numerics: a port's wrong math (a transposed weight, a bad mask) still serves.
- A daemon thread inside a torch call as the process exits aborts or hangs it ("terminate called
  without an active exception"), on plain torch too: Python ends the thread through torch's C++
  frames. Here that includes the thread behind an unwaited `irecv` (mstar sends no point-to-point).
  One parked in Python exits cleanly. A tensor gloo still holds when the process exits right after
  the collective that used it does the same: the clone an `async_op` broadcast or reduce sends, or
  an mstar tensor on the data-moving path dropped right after a synchronous collective. The
  emulator holds only its own tensors of a synchronous call until gloo lets go.
- The FlashInfer stand-in sizes the CUDA-core decode kernel's grid without its register count
  (a fact of the compiled kernel, which varies with the box's nvcc), so a mixed wrapper's decode
  workspace can come out larger than a GPU's, never smaller. A decode kernel past 48 KiB of shared
  memory (head dim 512 at group 1 or 16 with 2-byte KV) has no stand-in: its grid on a GPU depends on whether a run came first. MLA plans as fa2/fa3's planner does on any GPU, and a backend other than
  fa2 and fa3 asks for no workspace. `auto` chooses between fa2 and fa3 as FlashInfer does; on
  Blackwell a ragged `auto` stays fa2 here, where FlashInfer can upgrade it to cudnn or cutlass.
  Decode `auto` never moves to cuDNN here, where FlashInfer's can on SM100/SM103: tensor-core
  decode of 2 to 4 queries a row stays on fa2, and CUDA-core decode of 2 or more is refused,
  which FlashInfer serves on cuDNN at head dim 128 or 256.
- A FlashInfer API the stand-in lacks raises `NotImplementedError("mstar_sim: flashinfer.<name>
  has no stand-in yet")` when called.
