# Waypoint Optimization Backlog

An item may remain deferred only when it records evidence, expected benefit,
dependency, proposed benchmark, and completion criterion. These items are outside
the scripted streaming MVP unless a gate promotes one.

## STREAM-001: Streaming Viability Baseline and Threshold

- **Status:** Captured baseline complete on 2026-09-11; threshold decision deferred.
- **Evidence:** Normal server startup captured all four graph-enabled paths for
  each run. The 16-step baseline and matched slow-consumer stream passed at both
  resolutions with identical payload hashes. Raw artifacts:
  `baselines/streaming-2026-09-11-360p.json` and
  `baselines/streaming-2026-09-11-720p.json`.
- **Expected benefit:** Quantifies user-visible startup latency, sustained delivery,
  stalls, backpressure, and memory behavior before setting a release threshold.
- **Dependency:** Satisfied: Phases 1-7 pass with all four graph-enabled paths
  captured.
- **Proposed benchmark:** Measure time to first frame, generated-media-time divided
  by wall time, p50/p95 inter-chunk gap, jitter, stall count/duration, slow-consumer
  backpressure, peak GPU memory, and peak host PSS for 360p and 720p.
- **Completion criterion:** Baseline satisfied by the results below. A later
  decision defines thresholds; no threshold is inferred from one run.

| Variant | TTFF | Sustained media/wall | Gap p50 / p95 | Baseline stalls | Peak host PSS | GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| 360p | 0.148 s | 3.268x | 0.020 / 0.027 s | 0 | 3291.5 MiB | 4648 MiB |
| 720p | 0.465 s | 0.570x | 0.112 / 0.142 s | 0 | 3687.3 MiB | 5756 MiB |

The slow consumer paused 0.25 seconds between 15 reads. It added 3.717 seconds
at 360p and 3.462 seconds at 720p for 3.750 seconds deliberately injected, with
0 MiB GPU growth and byte-identical output. Its host peak changed by +60.4 MiB at
360p and +7.1 MiB at 720p. These are observations, not release limits.

## INTERACTIVE-001: Interactive Sessions

- **Evidence:** The MVP input is a complete action script and has no session
  lifetime or reconnect contract.
- **Expected benefit:** Enables long-lived controllable worlds rather than fixed
  offline scripts.
- **Dependency:** Stable scripted cleanup, world-slot reuse, and streaming protocol.
- **Proposed benchmark:** Reconnect, cancellation, idle timeout, and one-hour
  session soak with deterministic action traces.
- **Completion criterion:** A documented session state machine passes lifecycle and
  soak tests without state leakage.

## ACTION-INGRESS-001: Live Action Ingress

- **Evidence:** Actions are validated as a fixed list before execution; no
  bidirectional live ingress or timing policy exists.
- **Expected benefit:** Allows real-time control while frames are generated.
- **Dependency:** `INTERACTIVE-001` and an explicit late/missing-action policy.
- **Proposed benchmark:** Timestamped actions under latency, reordering, loss, and
  backpressure with output/action correlation checks.
- **Completion criterion:** Every generated latent consumes exactly one documented
  live action under normal and degraded transport tests.

## BATCH-001: True Request Batching

- **Evidence:** MVP graph slots isolate worlds but do not establish a shared batched
  DiT/AE execution path.
- **Expected benefit:** Higher throughput under concurrent scripted requests.
- **Dependency:** Correct interleaved worlds and measurements showing launch or
  occupancy headroom.
- **Proposed benchmark:** Throughput, tail latency, graph memory, and parity at batch
  sizes 1, 2, 4, and 8 for each resolution.
- **Completion criterion:** A selected batch policy improves throughput without
  parity drift or unacceptable p95 latency.

## QUANT-001: Weight Quantization

- **Evidence:** The parity baseline uses checkpoint-native BF16; no quantized error
  or speed/memory data exists.
- **Expected benefit:** Lower GPU memory and potentially higher DiT throughput.
- **Dependency:** Stable BF16 end-to-end baseline and quality evaluation corpus.
- **Proposed benchmark:** Layer/rollout error, pixel metrics, action consistency,
  memory, and media-time/wall-time for candidate formats.
- **Completion criterion:** A format meets an explicitly recorded quality bound and
  materially improves memory or throughput.

## DECODER-PLACEMENT-001: Decoder on Another GPU

- **Evidence:** Decoder order is stateful and the MVP keeps DiT and decoder in one
  worker group; transfer and scheduling costs are unmeasured.
- **Expected benefit:** Overlap decode with DiT work and reduce rank-0 pressure.
- **Dependency:** Typed frame streaming, decoder graph state transfer, and correct
  multi-worker loop ordering.
- **Proposed benchmark:** Compare colocated and split placement for throughput,
  inter-chunk gaps, transfer time, and memory.
- **Completion criterion:** Split placement is parity-preserving and wins a recorded
  performance target without ordering failures.

## PRIME-GRAPH-001: DiT Prime Capture

- **Evidence:** MVP deliberately compiles but does not capture the one-time DiT
  prime/cache pass.
- **Expected benefit:** Lower request startup latency.
- **Dependency:** Required steady graphs and stable prime inputs/state addresses.
- **Proposed benchmark:** Admission-to-first-frame latency and graph memory with and
  without prime capture across repeated world-slot reuse.
- **Completion criterion:** Capture reduces p50/p95 startup latency without state
  leakage or disproportionate graph memory.

## ENCODED-VIDEO-001: Encoded Video Output

- **Evidence:** Per-step encoded fragments are not independently playable and the
  MVP protocol intentionally emits raw RGB frames.
- **Expected benefit:** Lower network bandwidth and direct media playback.
- **Dependency:** Session-aware muxing, cancellation/finalization semantics, and
  separate API design from the MVP `video_frame` modality.
- **Proposed benchmark:** End-to-end latency, bandwidth, seek/playability, encoder
  load, and cancellation integrity for candidate codecs/containers.
- **Completion criterion:** A complete playable stream meets a separately recorded
  latency/bandwidth target and never exposes broken fragments.

## NOISE-001: GPU or Stateless Noise

- **Evidence:** Current deterministic parity supplies CPU FP32 noise then casts;
  device/stateless generation would change reproducibility and possibly bytes.
- **Expected benefit:** Avoid host generation/copy and simplify graph inputs.
- **Dependency:** A documented seed mapping and a new numerical baseline.
- **Proposed benchmark:** Generation/copy time, replay behavior, determinism across
  world slots, and rollout parity/quality.
- **Completion criterion:** Deterministic request-to-noise mapping and a measured
  performance win pass long interleaved rollouts.

## GRAPH-MEM-001: CUDA Graph Memory Reduction

- **Evidence:** Four graph-enabled paths across two resolutions can duplicate pools
  and staging buffers. Full-server peaks are now measured at 4648 MiB for 360p and
  5756 MiB for 720p, but per-bucket pool and staging attribution remains open.
- **Expected benefit:** More world slots or lower deployment GPU requirements.
- **Dependency:** Complete optional-capture implementation and memory attribution.
- **Proposed benchmark:** Per-bucket private-pool/staging bytes, peak allocated and
  reserved memory, and reuse across sequential/interleaved requests.
- **Completion criterion:** A change reduces measured graph memory without capture
  fallback, address instability, or parity drift.

## HOTSPOT-001: Measured Runtime Hotspots

- **Evidence:** Mask rebuild was historically measured at 14.31 ms/frame. Final
  full-size captured traces now prove host-sync-free steady DiT replay, while the
  720p streaming baseline remains below real-time at 0.570x sustained media time;
  detailed hotspot attribution is still open.
- **Expected benefit:** Direct optimization effort toward the dominant final-path
  cost.
- **Dependency:** `STREAM-001` traces with synchronized attribution outside timed
  replay.
- **Proposed benchmark:** GPU/CPU trace of startup and steady state, ranked by frame
  time and memory traffic for both resolutions.
- **Completion criterion:** Each promoted hotspot gets its own evidence-backed item;
  close this placeholder when the final trace has no untracked material hotspot.
