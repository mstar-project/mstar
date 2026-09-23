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

Phases that are WAITS, not work -- a larger number is not automatically worse:

* `await_gpu`, `postprocess.event_sync` -- waiting on the GPU. These GROW when
  the CPU side gets faster, because the CPU arrives at the sync sooner.
* `follow_await` -- a TP follower waiting for its leader's decision.
* `submit_spec` -- includes a deliberate wait for the GPU thread to reach the
  forward launch.
* `postprocess_batch` contains several of the `postprocess.*` phases, so it
  does not sum with them.

## Why segments

A closed-loop run ramps up, holds at max concurrency, then drains stragglers.
Those are three different regimes, and one mean across all of them describes
none of them. Each record carries `bs=`, the mean in-flight requests over its
window; a record whose `bs` differs from the running mean by more than
`--bs-tolerance` starts a new segment. The drain therefore lands in its own
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

Take the LONGEST segment at the expected batch size as steady state.

## Caveats

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
