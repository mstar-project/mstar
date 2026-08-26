# M\*-Sim — performance prediction for mstar deployments

Predict end-to-end serving performance for a deployment — given the model, the
GPU config, and the `node_groups` placement — without running the server.

The guiding rule is **semantics are imported, costs are measured**. Everything
that decides *what runs when* is mstar's own code, executed: walk graphs,
`WorkerGraphIO` readiness and loop iteration, `node_groups` placement, the
model's streaming `ChunkPolicy` objects. Everything that decides *how long it
takes* is a real measurement of a real step on a real GPU. Nothing is derived
from a roofline, and no scheduling rule is restated here in a second place
where it could drift from the one the server uses.

## The four commands

```
        (a real, profiled run)                    (no GPU needed)
   ┌──────────────────────────┐          ┌───────────────────────────┐
   │ mstar serve … --log-stats│          │ mstar predict   step costs│
   │   MSTAR_STEP_LOG=…       │          │ mstar simulate  a workload│
   └────────────┬─────────────┘          │ mstar validate  vs measured│
                │ step logs + profiles   └─────────────▲─────────────┘
                ├──► mstar harvest   ──►  stepdb ──────┤
                └──► mstar calibrate ──►  timing.json ─┘
```

### 1. Capture a profiled run

```bash
MSTAR_STEP_LOG=/tmp/cap/steps/run.jsonl \
mstar serve orpheus --log-stats --log-stats-json /tmp/cap/profiles/run.jsonl
# …drive it with the benchmark harness or your own client…
```

`--log-stats` turns on profiling, which is what enables the CUDA event pair
that measures true GPU time per step. `MSTAR_STEP_LOG` adds one JSON record
per executed batch — the executed shape (real *and* padded), the mode
(captured-graph replay / eager / sequential), the KV context, the GPU time,
and the engine's CPU phases. `--log-stats-json` writes per-request profiles as
JSON Lines instead of only the human report.

Cover the batch sizes you care about: the cost table can only answer for
shapes it has seen. Driving the workload at several concurrency levels is the
cheapest way to fill it.

### 2. Build the cost table

```bash
mstar harvest /tmp/cap/steps --db orpheus.db --model orpheus
mstar predict --db orpheus.db list
mstar predict --db orpheus.db sweep --node LLM --walk decode --kv 4096
```

Observations of the same padded shape and KV bucket collapse to their
**median** — step times have a long right tail (first replay of a bucket pays
warmup) and a mean would track it.

### 3. Measure the overheads that are not engine steps

```bash
mstar calibrate --profiles /tmp/cap/profiles --steps /tmp/cap/steps \
                --out timing.json
```

Conductor hop, api-server preprocess, client delivery, and the fixed per-step
worker overhead. Each comes from a specific pair of checkpoints, so an
implausible number points at one identifiable stage. Without this file the
simulator uses documented placeholders and **says so in every report**.

### 4. Predict

```bash
mstar simulate --config configs/orpheus.yaml --db orpheus.db \
               --timing timing.json \
               --requests 32 --mode closed_loop --concurrency 4 \
               --prompt-tokens 41 --output-tokens 740
```

`--output-tokens` is **autoregressive steps**, not client-visible chunks. For
a codec model these differ by the codec's tokens-per-frame (Orpheus emits one
audio chunk per 7 decode steps), so read the ratio off a measured run rather
than assuming.

### 5. Score it against reality

```bash
mstar validate --config configs/orpheus.yaml --db orpheus.db \
               --profiles /tmp/cap/profiles --steps /tmp/cap/steps \
               --timing timing.json \
               --requests 12 --mode closed_loop --concurrency 4 \
               --prompt-tokens 41 --output-tokens 740
```

Three gates, tightest first, because a loose end-to-end match can hide two
errors that cancel:

* **V1 semantics** — per-(node, walk) step counts. If the simulator did not do
  the same *work*, no timing comparison below it means anything.
* **V2 step costs** — table lookups vs the measured times they were built
  from. A self-consistency check on bucketing and aggregation; buckets with
  fewer than five samples are reported but not judged.
* **V3 end-to-end** — TTFT and E2E distributions, compared as distributions.
  Per-request pairing is not attempted: DP replica choice and batch
  composition make an individual request's fate incomparable between runs.

## What is modeled, and how

### Worker timing

Each worker is two serial resources, a GPU lane and a CPU lane, wired to
reproduce the one-deep speculation pipeline in `worker.py:1084-1098`:

```
CPU:  [ build N ][ launch N + post N ][ build N+1 ][ launch N+1 + post N+1 ]
GPU:            [====== step N ======][===== step N+1 =====]
```

The build of the next batch overlaps the current GPU step, and the launch and
postprocess of a step overlap *its own* GPU execution. That is what makes the
steady-state cadence `max(GPU, CPU)` rather than their sum. Getting this
ordering wrong is not a small error — it turns every predicted step into
`gpu + cpu` and inflates every latency downstream, which is why the code says
so at the site.

### Batching

All ready requests for one `(node, graph_walk)`, with the node chosen
round-robin by least-recently-run — the `MicroScheduler` rule. The batch is
then padded up to the CUDA-graph bucket the engine would actually replay
into, because the GPU pays for the padded shape.

### Streaming

Consumer cadence comes from the model's own `ChunkPolicy` via its
`PartitionTopology` — a fresh policy instance per request, matching the
lifetime the real `StreamBuffer` gives it. A sliding-window codec therefore
runs once per *stride* tokens, not once per token.

### Admission and arrival

The api server preprocesses on one thread, so concurrent arrivals serialize
through it. That stagger is modeled because it is load-bearing: it offsets
each request's streaming buffer by a token or two, which is what stops a
codec consumer from batching every request in lockstep with the backbone.

### EOS

A simulator has no token values to inspect, so a request stops at the
workload's target length — exactly what a measured run with a pinned
`max_tokens` produces. The stop is delivered through
`register_loop_finish_signal`, the worker's own path, so loop teardown and
completion accounting stay identical.

## Coverage flags

Every lookup returns a coverage flag, OR'd across the whole run and printed in
every report:

| flag | meaning |
| --- | --- |
| `exact` | measured at this shape and KV bucket |
| `interpolated` | between two measured KV points |
| `extrapolated` | outside the measured KV range |
| `missing` | never measured — the step was priced at zero |

A `missing` step means the report is not a prediction. This is deliberately
loud: the failure mode worth engineering against is a plausible-looking number
nobody checked.

## Hollow mode

```bash
MSTAR_HOLLOW=1 MSTAR_HOLLOW_DB=orpheus.db mstar serve orpheus
```

Runs the real conductor, workers, micro-scheduler, graph routing, and ZMQ with
a fake engine that returns correctly-shaped tensors after the modeled delay.
No GPU, no weights.

Use it as a **drift gate**: the DES re-implements the worker's pipeline, hollow
mode does not, so running both on one workload and comparing step counts
catches a divergence introduced by either. Do *not* use it to measure CPU
overhead — the fake engine replaces `prepare_inputs`, the attention plan, and
sampling, which are the terms that matter most. Those come from instrumented
real-GPU runs.

## Limits worth knowing before trusting a number

* **One GPU model per table.** Rows are keyed by device name; there is no
  cross-GPU scaling. Predicting for hardware you do not have means profiling
  on it first.
* **Model-version-scoped rows.** A step's cost bakes in the submodule code and
  the capture bucket lists. Re-harvest after a model change.
* **Host-specific CPU terms.** Calibration measures the machine it ran on.
* **Model coverage.** Placement, graph, engine types and streaming policies
  load for every registered model, and the cost table and instrumentation are
  model-agnostic. But *walk sequencing* is still a name heuristic rather than
  the model's own transition function, so only prefill→decode pipelines
  (orpheus, qwen3_tts, whisper) simulate end to end. For bagel, qwen3_omni,
  pi05, wan22, vjepa2 and cosmos3 the run completes having executed only the
  first leg — silent in the metrics, visible in `step_counts_by_key`. See the
  table in `des.py`'s module docstring.
* **Codec batch aggregation** is the largest known semantic residual among the
  models that do work: the simulator batches a streaming consumer somewhat
  more than the real system does.
* **TTFT reads slightly low**, because scheduler and queueing overhead outside
  the measured terms is not modeled. Under-prediction is the expected
  direction; do not tune the cost model to close it.
