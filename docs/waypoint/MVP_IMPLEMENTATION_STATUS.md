# Waypoint MVP Implementation Status

This is the live implementation report for the current completion pass. It is
separate from `WAYPOINT_PROGRESS.md`, which remains the historical investigation
log. Durable decisions, acceptance gates, and deferred work remain in the other
files in this directory.

## Current Snapshot

Last updated: 2026-09-11.

The branch was fetched and rebased onto `origin/main` at `9ef65097`. Its four
Waypoint commits now sit directly above the four new upstream packaging, ragged
attention, API-validation, and sampler commits. The pre-rebase tracked patch ID
and all 25 restored untracked file blobs matched the safety stash; there are no
unmerged entries, conflict artifacts, or staged files.

| Area | Implementation | Current evidence | Remaining gate |
|---|---|---|---|
| Startup/config | Passed | Both artifact paths, the TAEHV runtime, and the DiT manifest validate before device allocation; tensor completeness and TAEHV architecture validate during loading before admission; variant-specific Hub mapping prevents cross-variant weights; Python 3.12/uv resolves and builds the pin; registry-selected 360p Hub startup passed | None for scripted MVP |
| Request semantics | Passed | Positive bounded `num_steps`, exact validated action count, required seed, internal idle prime, action zero preserved; full 360p/720p eight-step streams emitted exactly 32 generated frames from index zero | None for scripted MVP |
| DiT execution | Integrated | Runtime tables materialize after load; `compile_dit` independently selects compiled or eager denoise/cache regions; optional 128-token and 512-token rollout graphs captured on H100 | Record a full server graph-off run |
| Mask planning | Passed | One fixed-address local/global block mask per slot; immutable device visibility tables remove per-step allocations; multi-wrap/dilation/world parity passes; full-size 360p and 720p profiles each found 16/16 steady DiT graph replays and zero blocking CUDA calls | None for scripted MVP |
| Encoder/decoder | Integrated | Pure tensor encoder; nine explicit histories; real-weight FP32 parity; optional encoder/init/steady graphs captured and served at both resolutions on H100; graph-free declarations use eager forwards | Record a full server graph-off run |
| Optional capture policy | Integrated | `cuda_graph` controls declaration independently of `compile_dit`; all Waypoint buckets use normal eager fallback on capture failure; all four graph-enabled buckets previously captured at both resolutions | Record graph-off and injected capture-failure server runs |
| Frame protocol | Passed | Python streaming-only `video_frame`, canonical RGB24 metadata, immutable zero-copy SDK view, named-byte upload, explicit stream errors, ordered async-read delivery, and live 360p/720p SDK streams pass | Rust frontend support is deferred and outside the scripted MVP |
| End to end | Passed | Local 720p and registry-Hub 360p normal `EngineManager` paths captured all four buckets; both resolutions passed sequential and full-size two-world interleaved deterministic streams, exact counts, cleanup, repeated slot reuse, and bounded server memory | None for scripted MVP |
| Streaming viability | Baseline complete | `STREAM-001` records captured 16-step baseline and matched slow-consumer runs for both resolutions, including typed-stream correctness and server process-group memory | A later decision may define a viability threshold from these measurements |

## Latest Validation

- After reverting the out-of-scope Waypoint Rust frontend changes, the Python
  frame protocol, SDK, and API guard selection passed: **45 passed, 2 existing
  FastAPI deprecation warnings**.
- Consolidated Waypoint, startup, frame/SDK, graph-policy, attention/ring, GPU,
  and live-parity CPU selection: **385 passed, 41 skipped, 4 warnings**.
- On H100 (`CUDA_VISIBLE_DEVICES=2`), the reduced random-weight graph suite passed
  **10 tests** covering capture/replay, planned masks, interleaving, and
  world-slot reuse.
- The 720p full-checkpoint same-process parity suite passed **14 tests**. Its
  41-frame reference-compatible rollout was bit exact through tables,
  conditioning, all DiT stages, five-pass output, ring writes, and rollout
  latents. The touched suite was rerun after the rebase: **14 passed in 56.98s**.
- The dedicated 360p same-process gate passed on the published 360p checkpoint:
  **1 passed in 37.79s** on H100. Zero-tolerance comparisons covered all three
  derived tables, five conditioner rows, all 30 DiT stages at frozen and
  committing sigmas, 201 passes across 41 latent frames, every ring write, the
  full-size BF16 functional encoder, all nine explicit decoder histories, and
  164 decoded 640x360 RGB frames. The retained report is
  `baselines/reference-parity-2026-09-11-360p.json`.
- The real-checkpoint pixel suite passed **5 tests**, including TAEHV seed
  encode/decode and reference pixel/order checks.
- A normal local 720p registry and `EngineManager` server run captured all four
  graph-enabled buckets under the earlier required-capture policy. Two sequential eight-step SDK requests each emitted eight
  typed chunks and exactly 32 1280x720 RGB24 frames; all 88,473,600 output bytes
  were identical across slot reuse. The SDK's NDJSON reader now uses 1 MiB input
  chunks, avoiding quadratic buffering of each 14.7 MB base64 response line.
- The distinct 360p checkpoint was selectively downloaded from its published
  Hub repository: only `config.yaml` and `model.safetensors`. Its local SHA-256
  matched Hub metadata, and the variant-specific manifest preflight passed.
- A normal 360p registry and `EngineManager` run through `MStarClient` captured
  all four graph-enabled paths under the earlier required-capture policy. Two sequential one-step requests each returned one
  typed 2,764,800-byte chunk containing four 640x360 RGB24 frames from index
  zero; the repeat was byte-identical. Startup took 223.5 seconds and requests
  took 29.7 and 6.2 seconds.
- Real pinned TAEHV test on a reduced spatial grid: functional encoder, decoder
  initialization, decoder steady output, and all nine histories are bit-identical
  to the upstream streaming scheduler in FP32.
- `ruff check` over all changed implementation and focused test files: passed.
- `python3 -m compileall` over the changed Python surfaces: passed.
- `git diff --check`: passed.
- An optional full `test/modular` run required redirecting FlashInfer's cache
  to `/tmp`; it reached 31% but then stopped producing output for several
  minutes and was interrupted. The focused 16-file gate above was rerun cleanly
  afterward, so this incomplete broad run is not counted as validation.
- The server log exposed a spurious `rollout_loop` stop signal during the prime
  walk. The graph runtime ignored it, but it was a model lifecycle bug; the stop
  hook now returns no signal outside rollout and its regression test passes.
- SDK error/upload regressions plus frame protocol tests pass: **26 passed**.
- Historical Rust validation remains recorded in `VALIDATION.md`, but the
  Waypoint-specific Rust guard and tests were reverted after the Rust frontend
  was removed from the scripted MVP scope.
- `uv 0.11.13` with Python 3.12 resolved the index-safe `.[waypoint]`
  dependencies in dry-run mode. A separate `--no-deps` install then built the
  pinned TAEHV revision as a real 21,947-byte `taehv.py` package exporting
  `TAEHV`; this closes the installer-floor check without downloading a second
  CUDA/PyTorch stack or placing a PyPI-incompatible direct URL in `m-star`
  metadata.
- Async result reads now retain notification sequence and loop snapshots, buffer
  out-of-order completions, and assign frame indices only in emission order. A
  deliberately reversed-completion regression test passes.
- Flex mask planning now reads immutable per-geometry device lookup tables; a
  repeat-plan test forbids fresh `torch.tensor` staging and checks all mask/table
  addresses remain stable.
- Latest combined protocol, shell, ring, and Flex resource gate: **199 passed,
  3 skipped**; Ruff and Python compilation passed.
- A registry-selected 360p Hub deployment captured all required buckets, served
  distinct deterministic solo baselines, and reproduced them byte-for-byte over
  two concurrent two-world waves with actual A/B/A DiT scheduling. Every request
  executed eight rollout steps and emitted 32 frames. A three-wave memory run
  measured -2.0 MiB quiescent server PSS growth and 0 MiB GPU growth after warmup.
- A local 720p deployment repeated the full-size two-world gate over three
  concurrent waves. All 48 chunks matched their distinct solo baselines, worker
  execution interleaved, every request cleaned up, and measured quiescent growth
  after warmup was -49.5 MiB host PSS and 0 MiB GPU memory.
- Reproducible Nsight validation now scopes CUDA calls to the nested steady DiT
  `engine.forward` ranges. Full 360p and 720p eight-step traces each reported
  **16 forwards, 16 graph replays, 0 synchronization or blocking calls**.
- The post-MVP streaming harness passed 16-step captured runs at both resolutions.
  Baseline TTFF / sustained media ratio / p50-p95 gap were **0.148s / 3.268x /
  0.020-0.027s** at 360p and **0.465s / 0.570x / 0.112-0.142s** at 720p, with
  no baseline stalls. GPU memory was flat under slow-reader backpressure, payload
  hashes matched, and the raw JSON artifacts are retained under
  `docs/waypoint/baselines/`.

## Completed This Pass

- Startup/configuration, graph/mask, frame-protocol, and end-to-end audits
  completed in parallel, followed by coordinator integration review.
- Mandatory DiT compilation, TAEHV dependency preflight, exact action mapping,
  real-weight functional AE parity, and the planned-mask CUDA test path were
  tightened during those audits.
- Rebased the four committed Waypoint changes onto `origin/main` at `9ef65097`
  and restored the full tracked/untracked working tree without content loss.
- Integrated upstream's PyPI packaging: Waypoint's extra is index-safe, both
  alias packages forward it, the CLI resolves its packaged default config, and
  pinned TAEHV remains an explicit separate install with actionable errors.
- Preserved upstream malformed-`model_kwargs` handling. The Python frontend
  rejects `video_frame` input as HTTP 400; Rust frontend parity is deferred.
- No changes were staged or committed.
- Closed the separate 360p numerical gate with the native 360p Hub weights.
  Inputs use the canonical seed/action script and seeded CPU-fp32 noise recipe,
  resized/generated directly at 360p. The existing stored oracle remains a 720p
  artifact and was deliberately not reused as a 360p numerical target.

## Release Boundary

The scripted streaming MVP gate and its post-MVP measurement phase are complete:
both supported resolutions start through the normal registry and `EngineManager`,
all four graph-enabled paths have captured successfully, server output is consumed
as typed SDK frames, and interleaved world-slot reuse remains deterministic and
memory-bounded. `STREAM-001` supplies the first captured baseline without inventing
a release threshold. CUDA graph capture is now optional; graph-off and injected
capture-failure full-server runs remain to be recorded.
