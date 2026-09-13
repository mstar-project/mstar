# Waypoint MVP Port Plan

This is the active completion plan for the scripted Waypoint MVP. Historical
experiments and measurements remain in `WAYPOINT_PROGRESS.md`; a historical
"done" label there is evidence about that experiment, not an MVP release gate.

## MVP Contract

- Serve both `waypoint-1.5-1b-360p` and `waypoint-1.5-1b-720p`.
- Use reference-compatible arithmetic by default. Exact-table arithmetic remains
  an explicitly selected experimental mode.
- Attempt CUDA graphs by default for encoder prime, steady DiT rollout, decoder
  initialization, and steady decoder execution. Allow explicit graph-free
  execution and eager fallback when capture fails.
- Accept exactly one validated action per generated latent step. The internal idle
  prime action does not consume action zero.
- Prime encoder, DiT cache, and decoder state without emitting reconstructed seed
  frames.
- Stream contiguous RGB24 frames only. Exactly `4 * num_steps` frames are emitted,
  starting at frame index zero.

## Phases

| Phase | Scope | Completion gate |
|---|---|---|
| 1 | Documentation and baseline | Current diff/test baseline recorded; plan, decisions, validation, and backlog exist. |
| 2 | Startup and configuration | Local/HF sources resolve before allocation; downloads are selective; config/checkpoint facts and TAEHV dependency fail clearly when invalid. |
| 3 | Request and numerical correctness | Positive steps, exact action count, internal prime semantics, and live parity through pixels are tested. |
| 4 | DiT and attention execution | Runtime tables are post-load; optional full-graph DiT compilation is independent of optional CUDA graph capture; masks are planned/staged once; captured and eager paths are supported. |
| 5 | Encoder and decoder graphs | Tensor-only fixed-shape state; encoder prime and decoder init/steady paths support optional capture with eager fallback, equivalence, reuse, and cleanup tests. |
| 6 | Streaming frame protocol | `video_frame` events and SDK `VideoFrameChunk` validate typed metadata and expose a zero-copy `[N,H,W,3]` NumPy view; non-streaming use is rejected. |
| 7 | End-to-end MVP gate | Registry and `EngineManager` runs pass for local/HF sources, 360p/720p, sequential/interleaved worlds, cleanup, bounded memory, and exact frame counts. |
| 8 | Post-MVP streaming viability | Only after Phase 7: collect latency, throughput, gap/jitter/stall, backpressure, and memory baselines under `STREAM-001`. |

## Execution Rules

1. Preserve unrelated working-tree changes. Do not stage, commit, or revert.
2. Treat artifact resolution and manifest validation as pre-allocation contracts;
   tensor/runtime architecture and request validation must pass before admission.
3. Keep CUDA graph capture optional. `cuda_graph=False` skips capture, and failed
   capture attempts fall back to eager execution without changing numerical mode.
4. Validate numerical parity live in one process where reproducibility permits.
   Stored cross-process artifacts establish a measured floor, not a bit-exact
   oracle.
5. Update `VALIDATION.md` with the command, environment, result, and evidence for
   every completed gate. Record any deferral in `OPTIMIZATION_BACKLOG.md` with all
   required fields.

## Parallel Ownership

Startup/configuration, ring/Flex/DiT, and server/SDK protocol can proceed in
parallel when their file sets do not overlap. Functional TAEHV state, generic
required-capture support, and isolated tests form the second wave. One integration
owner then changes Waypoint model wiring and YAML. Numerical, graph, server, and
SDK gates run only after integration.

## Current State

Phases 1-8 are complete. Local and registry-selected Hub startup, live
same-process numerical parity, all required CUDA captures, fixed-address mask and
TAEHV state, typed SDK streaming at both resolutions, two-world interleaving,
cleanup/reuse, and bounded server memory have passed. `STREAM-001` now records the
first captured 360p/720p latency, pacing, backpressure, and memory measurements;
as decided in WP-007, they establish a baseline and do not set a release threshold.
