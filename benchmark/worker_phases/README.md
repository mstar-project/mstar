# worker_phases

Per-phase wall-clock breakdown of the worker loop, for answering "where does a step actually spend its time".

## Run it

Terminal: the server. The command is passed verbatim; the wrapper only
adds `MSTAR_PHASE_TIMING` to its environment.

```
python -m benchmark.worker_phases.server \
    --command "mstar serve orpheus --config configs/orpheus_tp2.yaml --gpus 0,1 --port 8100" \
    --period 100 --server-log /tmp/orpheus.log
```

It prints `SERVER READY` when `/health` answers -- then, terminal 2:

```
benchmark/worker_phases/clients/orpheus_t2s.sh
```

Tables go to stdout (or `--out FILE`); the server's own output goes to
`--server-log`, which keeps the terminal readable.

To re-read a log captured earlier:

```
python -c "
from benchmark.worker_phases.phases import parse_log, segment, render
print(render(segment(parse_log(open('/tmp/orpheus.log').read()))))"
```

## Reading the table

One table per SEGMENT per worker:

```
=== worker_0  iters 400-1800  records=15  batch=15.94 (range 15.6-16.0) ===
phase                                      mean_ms   p50_ms   p95_ms  samples
iter_total                                   3.629    3.680    5.120     1500
worker.postprocess.route                     0.050    0.060    0.110     1500
```

* `mean_ms` is n-weighted across records and exact.
* `p50_ms`/`p95_ms` are median-of-p50 and worst-p95. Records arrive
  pre-summarised, so true percentiles cannot be recovered -- these are
  indicative, and `mean_ms` is the number to compare.
* `samples` is how many individual timings went in.

A phase whose p50 << mean << p95 is bimodal: the mean is mixing two kinds of
iteration (e.g. prefill and decode, or two nodes). Don't try to recover the
two modes from mean/p50/p95 algebra; doing so once "found" a 76 ms cost that
measured 0.15 ms when timed directly. Add a span keyed by whatever
distinguishes them (node, walk) instead.

### Which phases are work and which are waits

The names say neither which thread a phase runs on nor whether it is work.

| phase | thread | what it is |
| --- | --- | --- |
| `worker.postprocess.event_sync` | main | **The only real GPU wait.** Blocks on the step's completion event |
| `await_gpu` | main | Waits for the GPU *thread's* host work (`prepare_inputs` plus kernel launch). Not GPU execution |
| `worker.gpu_thread.exec` | GPU | CPU time spent enqueuing kernels, not GPU time |
| `worker.gpu_thread.prepare_inputs` | GPU | Host work, but it absorbs any host sync inside a submodule's `prepare_inputs` |
| `worker.gpu_thread.await_plan` | GPU | Wait on the plan thread |
| `worker.plan_thread.await_commit` | plan | Wait on the previous step's commit event |
| `worker.plan_thread.pre_plan` | plan | Work |
| `submit_spec` | main | Includes a deliberate wait for the GPU thread to reach the forward launch |
| `follow_await` | main | A TP follower waiting for its leader's decision |

Consequences:

* **Never add a wait to a work total.** `pre_plan + await_plan + await_commit`
  is not "planning cost": the last two can be the same interval, waited on from
  two threads. Summing them once turned a 2 ms cost into 5.5 ms and a 3% win
  into a projected 2x.
* Waits GROW when the CPU side gets faster, because the CPU arrives at them
  sooner. A larger wait is not automatically worse.
* `postprocess_batch` contains several of the `postprocess.*` phases, so it
  does not sum with them.
* **GPU busy time is not in this table**, and `iter_total - event_sync` is not
  GPU idle time. A non-zero `event_sync` means the CPU did wait on the GPU; a
  small one means the GPU-bound component is marginal, not absent. For the
  actual GPU utilization, take an nsys capture, merge the intervals from
  `SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL`, and compare their union
  to the capture's wall time.
* **A host sync is charged to the span that encloses it, not to
  `event_sync`.** A `.tolist()` in `prepare_inputs` reports as expensive
  `prepare_inputs`, and `event_sync` gets *smaller* because the wait moved
  upstream. Watching `event_sync` will not find syncs.

### Adding spans

* Phase names must match `[\w.]+`. The parser silently drops anything else,
  so `foo[bar/baz]` vanishes without an error. Use dots.
* Instrumentation is not free. Per-step f-string span names have cost 7% of
  throughput, and a three-span probe cost 2%, so don't compare throughput across
  differently instrumented builds.

## Why segments

A closed-loop run ramps up, holds at max concurrency, then drains stragglers.
Those are three different regimes, and one mean across all of them describes
none of them. Each record carries `bs=`, the mean in-flight requests over its
window; a record whose `bs` differs from the segment's FIRST record by more
than `--bs-tolerance` starts a new segment. Anchoring on the first record
rather than the running mean is what stops a slow ramp -- where every step is
under the tolerance -- from collapsing into one segment. The drain lands in its own
table instead of being averaged into steady state, with no per-workload
guessing about how much to trim.

Knobs:

| flag | default | meaning |
| --- | --- | --- |
| `--period` | 100 | iterations per flush (`MSTAR_PHASE_TIMING`) |
| `--every` | 20 | emit a table every N records |
| `--skip-warmup` | 3 | drop leading records per worker (cold caches, ramp) |
| `--bs-tolerance` | 0.15 | fractional `bs` change that starts a segment |
| `--min-records` | 3 | drop segments shorter than this |
| `--phases` | all | only show phases containing one of these |

Take the LONGEST segment at the expected batch size as steady state. Check
`bs` against the client concurrency: the worker may build smaller batches
than the client could fill, which can be a performance finding in itself.

## Caveats

* This harness only measures time *inside the worker*. TTFT and ITL are not
  here and cannot be derived from these numbers -- they come from the
  runner's closed loop, measured client-side against a live server.
* Only iterations that run a batch are counted; an idle worker waiting for
  work records nothing.
* Per-worker. Under TP the leader and follower have genuinely different phase
  sets (`speculate`/`schedule_yield_away` vs `follow_await`), so compare each
  role with itself.
* `torch.compile` picks kernels by measured time at compile, so two server
  processes running identical code can differ. For an A/B, run several server
  restarts per side rather than trusting one.

## Tests

`pytest benchmark/worker_phases/test_phases.py` -- parsing and segmentation
are pure functions over a string, so the part that decides which numbers get
averaged together is checked without a GPU.
