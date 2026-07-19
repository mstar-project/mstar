# Soak / large-tensor stress harness (client half)

The client half of the two SHM-arena stress runs:

- **(a) soak** — multi-hour mixed-concurrency run with heterogeneous tensor
  sizes, watching for drift in throughput, failures, and client-side
  backpressure.
- **(b) large-tensor stress** — heavy generation (esp. cosmos3 video) driven
  toward the arena cap, watching failure rate vs. the old file transport.

It fires a **weighted mixture** of request types at a **Poisson** arrival rate,
capped at `max_in_flight` concurrent requests, for a fixed wall-clock
`duration_s`, and reports **rolling moving-average** metrics plus failure /
timeout / admission-delay signals. It reuses the existing
`benchmark.request` adapters (`OurSystem` → `/generate`) and `RequestMetrics`,
and `benchmark.dataset` for prompts/inputs.

## Run

```bash
PYTHONPATH=. python -m benchmark.long.soak \
    --config benchmark/long/configs/qwen3omni.yaml \
    --url http://127.0.0.1:8000 \
    --duration-s 7200 \
    --metrics-jsonl soak_qwen.jsonl
```

Any scalar top-level knob can be overridden on the CLI (`--rate`,
`--max-in-flight`, `--duration-s`, `--request-timeout-s`, `--window-s`,
`--report-interval-s`, `--pool-size`, `--seed`, `--system`, `--cache-dir`) so a
single YAML sweeps over load points.

## Config schema

```yaml
model: qwen3omni            # ModelType value, or any name (generic passthrough)
system: ours               # ours (/generate) | ours_openai (/v1/*)
rate: 8.0                  # Poisson arrivals/sec
max_in_flight: 32          # admission cap = server-facing concurrency
duration_s: 7200
request_timeout_s: 300     # per-request; exceeding it counts as a timeout
pool_size: 64              # prompts/inputs materialised per entry (once, up front)
window_s: 60               # moving-average window
report_interval_s: 15
cache_dir: ./.soak_cache
seed: 0
requests:                  # weights MUST sum to 1.0
  - req_type: text_to_speech
    dataset: seed_tts
    weight: 0.25
    model_kwargs: {}       # merged into the per-request payload (e.g. width/height/size)
    dataset_kwargs: {}     # dataset overrides (locale, data dirs, prompt file, …)
    label: t2s             # optional; defaults to "<req_type>:<dataset>"
```

Datasets: `text`, `seed_tts`, `libri`, `food101`, `ucf101`, `video_mme`,
`vbench` (same classes `benchmark.runner` uses).

Generation resolution goes in `model_kwargs`, passed straight through `/generate`:
- **BAGEL** reads `width` / `height` (`mstar/model/bagel/bagel_model.py`).
- **cosmos3** reads `size` (`"WxH"`), `num_frames`, `num_inference_steps`,
  `fps`, `guidance_scale` (`mstar/model/cosmos3/cosmos3_model.py`).

## Time-varying load (`rate_profile`)

A constant Poisson rate settles into a steady-state occupancy that hides the
arena behaviors most worth watching: whether fragmentation and `live_slots`
**reset to baseline on drain** (vs ratchet across cycles), whether grown segments
stay pinned while idle, and whether spill/backpressure **recovers** once load
drops. A cycling load makes the troughs diagnostic. Omit `rate_profile` for a
flat `rate`; otherwise (non-homogeneous Poisson, λ sampled per arrival):

```yaml
rate: 8.0                     # base/constant rate; still the fallback
rate_profile:
  shape: square               # sine | square | triangle | ramp | piecewise
  min: 2                      # trough λ (req/s)
  max: 40                     # peak λ (drive this above what max_in_flight sustains)
  period_s: 900               # one full cycle (sine/square/triangle)
  duty: 0.5                   # square only: fraction of the period at `max`
```

- **square** — burst/quiet; the most diagnostic (troughs should drain the arena).
- **sine** / **triangle** — smooth diurnal-like oscillation for multi-hour drift.
- **ramp** — one-shot `min`→`max` across the whole `duration_s`; sweeps the
  saturation knee (watch where spill/backpressure first appears).
- **piecewise** — `points: [[0, 2], [600, 40], [1200, 2]]`, linear-interpolated
  (elapsed_s, λ), ends held. Full control of an arbitrary load shape.

Push `max` above what `max_in_flight` can sustain so the peaks actually saturate
(admission delay + server-side spill/backpressure), then confirm the troughs
recover — that recover/drain transition is the point of varying the load.

## Metrics (all rolling over `window_s`)

Printed each `report_interval_s` and appended to `--metrics-jsonl`:

- **text** → `tok/s` (windowed system throughput) + per-request tok/s + TTFT.
- **audio** → `audio-seconds/second` (realtime factor; >1 = keeping up) +
  per-request RTF (`e2e/audio_dur`).
- **image / video** → e2e p50/p95 (throughput is naturally low; latency is the SLO).
- run-wide: `launched / admitted / in_flight / backlog / ok / failed / timed_out`.
- **admission delay** p50/p95 — how long an arrival waited for an in-flight
  slot. Climbing while the server's own queue stays flat is the client-visible
  backpressure signature (arena reviewer's "backpressure waits over time").

Arrivals and admission are decoupled: the arrival process stays a true Poisson
stream; the `max_in_flight` cap shows up as admission delay, not as a distorted
arrival rate. A separate backlog cap bounds client memory if the server falls
behind (arrivals pause with a logged warning rather than OOMing).

## Server side — `server_monitor.py`

Wrap the normal server launch; the monitor tees stderr through unchanged and
parses the `ARENA stats: {...}` lines that `--log-stats` emits (make sure the
server is started with `--log-stats`):

```bash
python -m benchmark.long.server_monitor \
    --stats-jsonl shm_server.jsonl --shm-size-gb 64 -- \
    bash test/qwen3-omni/launch_server.sh
```

Everything after `--` runs verbatim. Per parsed sample it derives and logs, per
producer entity (`api_server` data worker + each `worker_N`):

- every field the server logs, passed through (`segments`, `free_bytes`,
  `largest_free_block`, `pinned_bytes`, `live_slots`, `spill_files`, …);
- **`free / total`** (occupancy);
- **`largest_free_block / free`** (fragmentation gauge — collapsing toward 0
  while `free/total` stays high is the fragmentation signature, and triggers a
  `--frag-warn` warning);
- **node aggregates**: `Σ total_bytes` (/dev/shm), `Σ pinned_bytes` across all
  entities, with an >80%-of-`--shm-size-gb` warning.
- stress canaries: `spill_files > 0` (arena saturated → file transport, logged
  once per entity) and `live_slots` climbing while requests finish (reclaim
  leaking).
- **between-sample events**: the stats dict is logged only once per
  `MSTAR_SHM_ARENA_STATS_INTERVAL_S` (default **60 s**), but the arena also logs
  at the moment they happen — `fragmentation`, `at_capacity`, `pin_budget_reached`,
  `ttl_reclaim` (with a running total), `grew`, `over_80pct_shm`,
  `register_failed`. The monitor counts these per entity, timestamps each into
  the JSONL (`{"kind": "event", ...}`), and tallies them in the report + peaks
  summary, so a sub-minute spike isn't invisible until the next stats sample.

> **Tip:** for a soak, export `MSTAR_SHM_ARENA_STATS_INTERVAL_S=10` (or lower)
> before launching so the occupancy/fragmentation/pinned *time series* is
> finer-grained than the 60 s default. The event lines above fire regardless of
> this interval. JSONL rows are tagged `"kind": "stats"` vs `"kind": "event"`
> for downstream filtering.

`--stats-jsonl` shares the `t_wall` field with the client's `--metrics-jsonl`,
so the two series line up on one time axis for run (a) and run (b).

## SHM & pinned-RAM sizing (the maximums multiply by entity count)

The arena is created **once per producer entity** (`mstar_arena_{entity_id}`,
[`arena.py`](../../mstar/communication/arena.py)), so its caps are **per
process**, not per node:

- **/dev/shm** per entity ≤ `MSTAR_SHM_ARENA_MAX_SEGMENTS × MSTAR_SHM_ARENA_SEGMENT_MB`.
  Node total = that **× (num workers + 1 api-server data worker)**. With the
  defaults (32 × 256 MiB = 8 GiB/entity) and, say, 8 workers, the node can reach
  **72 GiB** of /dev/shm — which must fit tmpfs (defaults to ~50% of RAM; check
  `df -h /dev/shm`). Size `MAX_SEGMENTS`/`SEGMENT_MB` against
  `/dev/shm ÷ (num_entities)`.
- **Pinned host RAM** is a separate axis and also per process: each process pins
  its own arena segments **plus the peer segments it consumes**, capped by
  `MSTAR_SHM_ARENA_PIN_MAX_MB` each — so `pinned_bytes` for a worker can exceed
  its own arena's `total_bytes` (it includes peer pins), and the node pinned-RAM
  ceiling is ≈ `PIN_MAX_MB × num_entities`.

`server_monitor.py` reports both per-entity and node-aggregate SHM/pinned so
these multipliers are visible during a run.

> **Note (see the arena review):** poll the arena `stats()` from the **same
> thread that owns sends**, or first make `SegmentedShmArena.reserve` take
> `&self` — as written, `reserve` holds a `&mut` borrow across its GIL-released
> growth, so a cross-thread `stats()` poll can raise `RuntimeError: Already
> borrowed` under exactly the load you're measuring.

## New `RequestType`

This harness adds `RequestType.T2V` (`text_to_video`) to `benchmark/base.py` so
cosmos3 text→video generation flows through the unified `/generate` path with
`output_modalities=video` (the enum previously had only `V2V`, which requires a
video input).
