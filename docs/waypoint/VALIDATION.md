# Waypoint Validation Ledger

This ledger distinguishes historical evidence from the active MVP gate. Add the
exact command, environment, artifact location, and result when closing a row.

## Active Gates

| ID | Gate | State | Required evidence |
|---|---|---|---|
| CFG-001 | 360p and 720p supported config facts | Passed | CPU tests cover variant geometry, scheduler, FPS, temporal assumptions, explicit model kwargs, and registry construction; both variants started on H100 from their own manifests. |
| CKPT-001 | Local checkpoint resolution | Passed | Valid, missing, partial, cross-variant, and incompatible local checkpoint tests plus both published manifests. |
| CKPT-002 | Hub checkpoint resolution | Passed | Mocked and real selective downloads resolve only native safetensors plus `config.yaml`; registry-selected 360p Hub startup completed without a local model or AE override. |
| CKPT-003 | TAEHV resolution and dependency pin | Passed | Local/HF single-file tests, actual local weights, missing/empty-runtime preflight, index-safe Python 3.12 dependency resolution, and a separate real pinned-revision TAEHV build/install pass. |
| NUM-001 | Live reference-compatible parity | Passed at 360p and 720p | Native-checkpoint same-process tables, conditioner, every DiT stage, five passes, ring writes, 41 rollout latents, functional TAEHV state, and pixels passed with zero tolerance on H100 at both resolutions. |
| REQ-001 | Request and prime semantics | Passed | CPU tests reject non-positive steps and wrong action counts and prove idle prime preserves action zero; live sequential and interleaved 360p/720p runs emitted no seed frames and exact generated counts. |
| GRAPH-001 | Optional DiT compile | Passed | Post-load table materialization, `fullgraph=True` construction, and bounded compiled/eager equivalence are tested; both full-size variants compiled on H100. |
| GRAPH-002 | Optional capture and eager fallback | CPU mode-selection coverage added; existing capture path passed on H100 | `cuda_graph=False` declares no buckets; attempted captures are optional and use the engine's eager fallback on failure. All four graph-enabled buckets previously captured at both resolutions. A full server graph-off run remains to be recorded. |
| MASK-001 | Planned masks and replay sync | Passed | Tests cover one staged local/global mask per geometry and graph slot; full 360p and 720p profiles each prove 16/16 graph replay and zero blocking CUDA calls inside steady DiT forwards. |
| AE-001 | Functional TAEHV execution paths | Passed | Nine-history state, fixed graph interfaces, isolation, cleanup, real-weight parity, optional capture declarations, and full-size eight-step streams pass at both resolutions. |
| FRAME-001 | Typed RGB protocol | Passed | Server/SDK tests cover metadata, contiguous RGB24 bytes, zero-copy NumPy shape, indexing, errors, non-streaming rejection, ordered delivery, and live 360p/720p typed consumption. |
| E2E-360 | 360p normal serving path | Passed | Registry-selected Hub source + `EngineManager`, all buckets, SDK stream, exact counts, deterministic concurrent worlds, cleanup, slot reuse, and bounded memory passed. |
| E2E-720 | 720p normal serving path | Passed | Local source + `EngineManager`, all buckets, typed SDK stream, exact counts, byte-identical sequential reuse, full-size two-world interleaving, cleanup, bounded memory, and full eight-step profiler soak passed. |
| WORLD-001 | World isolation and reuse | Passed | Full server two-world DiT execution interleaved at both resolutions, reproduced distinct solo baselines byte-for-byte, cleaned every request, reused slots, and had -2.0/-49.5 MiB host PSS and 0/0 MiB GPU quiescent growth after warmup at 360p/720p. |
| STREAM-001 | Post-MVP streaming baseline | Passed without a release threshold | Captured 16-step baseline and slow-consumer runs at both resolutions record TTFF, sustained media/wall ratio, p50/p95 gaps, jitter, stalls, backpressure, host PSS, and GPU memory in retained JSON artifacts. |

## Historical Evidence

The earlier investigation reported the following. These results are retained as
diagnostic evidence and must not be read as completion of the active graph or
end-to-end gates.

- A 41-frame same-process run with `reference_compat=True` reported bit-exact DiT
  state and ring bookkeeping. Cross-process reference runs were not bit
  reproducible, with pixel drift reported as high as 72/255.
- CPU TAEHV component comparisons reported bit-identical FP32 and BF16 encode and
  decode results against the reference implementation.
- Earlier GPU tests reported full-graph compilation and stable DiT capture, while
  compiled versus eager output was bounded rather than bit-exact due to BF16
  fusion behavior.
- Mask reconstruction was measured at 14.31 ms per 720p frame in the eager test,
  motivating planned per-frame/per-slot local and global masks.

## Validation Record

| Date | Command/environment | Result | Scope |
|---|---|---|---|
| 2026-09-11 | Historical baseline copied from `WAYPOINT_PROGRESS.md` | 242 CPU tests reported green before subsequent uncommitted work | Not a current-tree result |
| 2026-09-11 | `pytest -q test/modular/test_waypoint_checkpoint.py` | 21 passed | Config and local/mocked-HF resolver contracts |
| 2026-09-11 | `pytest -q test/modular/test_waypoint_weight_loader.py` | 61 passed | Existing synthetic checkpoint loading |
| 2026-09-11 | `PYTHONPATH=. pytest -q test/modular/test_waypoint_reference_compat.py` | 11 passed | Default/experimental numerical modes |
| 2026-09-11 | Focused Waypoint execution-mode and capture-runner selection (7 files) | 255 passed, 9 skipped | Independent `cuda_graph`, `compile_dit`, and `reference_compat` modes; optional declarations and generic eager fallback. CUDA-only cases skipped because CUDA was unavailable in the sandbox. |
| 2026-09-11 | `PYTHONPATH=. pytest -q test/modular/test_waypoint_dit.py test/modular/test_waypoint_components.py` | 56 passed, 2 warnings | Existing CPU DiT/component contracts |
| 2026-09-11 | Direct resolver call on local Waypoint and TAEHV checkpoints | Historically accepted 720p weights for both variants; superseded | Exposed the stale shared-weight assumption. Exact variant geometry validation now rejects this pairing. |
| 2026-09-11 | `ruff check` on changed Waypoint Python/tests | Passed | Static checks |
| 2026-09-11 | `PYTHONPATH=. pytest -q test/modular/test_waypoint_checkpoint.py` | 27 passed | Config, resolver, dependency preflight, normal registry/engine construction, and startup ordering |
| 2026-09-11 | `PYTHONPATH=. pytest -q test/modular/test_waypoint_shell.py` | 62 passed | Request/prime, graph declarations, functional AE state, and YAML contracts |
| 2026-09-11 | `PYTHONPATH=. pytest -q test/modular/test_video_frame_protocol.py test/modular/test_cuda_graph_capture.py` | 22 passed, 9 skipped, 2 warnings | Typed frame protocol and required-capture CPU contracts; nine real CUDA cases skipped because CUDA was unavailable |
| 2026-09-11 | `pytest -q test/modular/test_waypoint_taehv_equivalence.py` with local pinned weights | 1 passed | Bit-exact FP32 encoder, decoder init/steady pixels, and all nine histories against upstream `StreamingTAEHV` |
| 2026-09-11 | Consolidated 16-file Waypoint/startup/protocol/graph/resource test selection | 345 passed, 41 skipped, 4 warnings | All then-current CPU-verifiable integration contracts; CUDA and live-reference cases skipped |
| 2026-09-11 | `FLASHINFER_WORKSPACE_BASE=/tmp/waypoint-flashinfer pytest -q test/modular` | Interrupted after 31% when no output was produced for several minutes | Optional broad regression run; earlier output included failures that could not be attributed because the run did not finish, so this is not passing evidence |
| 2026-09-11 | `ruff check ...`; `python3 -m compileall -q ...`; `git diff --check` | Passed | Changed Python/static formatting and syntax checks |
| 2026-09-11 | Default-sandbox torch/NVML probe | CUDA unavailable, zero devices; `nvidia-smi` driver failure | Sandbox-only limitation; later escalated runs reached physical GPU 2. |
| 2026-09-11 | `cargo check --locked` in `rust/server` | Toolchain blocked: Cargo 1.75 cannot parse lockfile v4 | Rust route source and tests added; build requires a current Cargo/rustfmt environment |
| 2026-09-11 | `pytest -q test/rust/test_rust_frontend.py` | Suite skipped because built Rust frontend is unavailable | Native route behavior still needs execution after a toolchain build |
| 2026-09-11 | `CUDA_VISIBLE_DEVICES=2 pytest -q test/modular/test_waypoint_gpu.py` | 10 passed in 84.44s | Required capture/replay, planned masks, long rollout, two-world interleaving, cleanup, and slot reuse on H100. |
| 2026-09-11 | `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. WAYPOINT_GPU_TESTS=1 pytest -q test/modular/test_waypoint_gpu.py` after optional-capture change | 10 passed in 20.70s | Graph-enabled capture/replay, compiled/eager DiT equivalence, planned masks, interleaving, cleanup, and slot reuse on H100. |
| 2026-09-11 | `CUDA_VISIBLE_DEVICES=2 pytest -q test/modular/test_waypoint_reference_equivalence.py` | 14 passed in 54.85s | Full-checkpoint same-process parity, including the 41-frame zero-difference gate through planned masks and ring writes. |
| 2026-09-11 | `CUDA_VISIBLE_DEVICES=2 pytest -q test/modular/test_waypoint_pixel_equivalence.py` | 5 passed in 32.89s | Real TAEHV seed encode/decode, output order, VAE output, and reference pixels. |
| 2026-09-11 | `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. WAYPOINT_360P_PARITY_REPORT=/tmp/waypoint-360p-reference-parity.json pytest -q -s test/modular/test_waypoint_360p_reference_equivalence.py` | 1 passed, 3 expected dtype-preservation warnings in 37.79s; every asserted maximum difference was zero | Native revision `35acd20...`; 3 derived tables, 5 conditioner rows, 30 stages at 2 sigmas, 201 passes, 41 latents, 24 ring layers per frame, full-size BF16 encoder, 9 decoder histories per frame, and 164 RGB frames on physical H100 GPU 2. Live same-process/eager scope with a shared corrected masked-attention kernel; the 720p stored oracle was not used. Raw report `/tmp/waypoint-360p-reference-parity.json`; retained report `baselines/reference-parity-2026-09-11-360p.json`. |
| 2026-09-11 | `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. pytest -q test/modular/test_waypoint_reference_equivalence.py` after adding the variant-specific checkpoint argument | 14 passed, 3 expected dtype-preservation warnings in 56.98s | Post-rebase/touched-file 720p numerical regression; the default checkpoint behavior remains unchanged. |
| 2026-09-11 | Normal 720p `serve_rollout.py --steps 1`; log `/tmp/waypoint_server_720.log` | Passed: four required captures; two byte-identical 1280x720 RGB24 requests | Local registry/`EngineManager`, exact four-frame output, cleanup and sole-world reuse. This predated typed-SDK harness conversion. |
| 2026-09-11 | Hugging Face metadata query + selective `snapshot_download` for `Overworld/Waypoint-1.5-1B-360P` | Passed; revision `35acd20e649fe79c1c1002df456696408547202d`, safetensors SHA-256 `a2cdccb5...c8101d9` | Confirms distinct 360p weights and downloads only `config.yaml` plus `model.safetensors`. |
| 2026-09-11 | `CUDA_VISIBLE_DEVICES=2 serve_rollout.py --variant 360p --steps 1`; log `/tmp/waypoint_server_360_sdk.log` | Passed: startup 223.5s; requests 29.7s/6.2s; byte-identical 2,764,800-byte chunks | Normal registry/`EngineManager`, four required captures, typed SDK, four 640x360 frames from index zero, cleanup and reuse. |
| 2026-09-11 | Startup/shell/SDK/frame selection after variant and SDK fixes | 136 passed, 2 warnings | Variant-specific repositories/manifests, prime stop guard, SDK uploads/errors, harness geometry, and frame protocol. |
| 2026-09-11 | Rust 1.98.1: locked `cargo check`, release build, and `cargo test` with target/cache under `/tmp` | Passed; 4 Rust unit tests | Native server compiles against the repository lockfile without changing the system Rust installation. |
| 2026-09-11 | `MSTAR_SERVER_BIN=/tmp/waypoint-rust-target/release/mstar-server pytest -q test/rust/test_rust_frontend.py` outside socket sandbox | Historical: 16 passed in 4.32s | This validated the former Rust raw-frame guards. Those Waypoint-specific changes were later reverted when `--rust-frontend` was removed from the MVP scope. |
| 2026-09-11 | Python 3.12 + uv 0.11.13: `uv pip install --dry-run --torch-backend=auto -e '.[waypoint]'` under the original direct-URL metadata | Resolved 88 packages; packaging layout later superseded | Proved dependency compatibility, but the TAEHV URL was subsequently moved out of project metadata because PyPI rejects direct-URL requirements. The current extra retains the index-hosted dependencies, including `tensordict==0.10.0`. |
| 2026-09-11 | Python 3.12 + uv: install pinned TAEHV URL with `--no-deps --target /tmp/waypoint-taehv-install` | Built/installed `taehv==0.1.0`; module exports `TAEHV` | Verifies Metadata-Version 2.4 packaging and non-empty artifact at the pinned upstream revision. |
| 2026-09-11 | `CUDA_VISIBLE_DEVICES=2 serve_rollout.py --variant 720p --steps 8` with local checkpoints; log `/tmp/waypoint_server_720_sdk_soak_fixed.log` | Passed: startup 43.1s; 8 typed chunks/32 frames per request; 88,473,600 bytes byte-identical across reuse | Full local 720p registry/`EngineManager` SDK soak and all four required buckets. |
| 2026-09-11 | Exact 360p Hub memory command below; log `/tmp/waypoint_server_360_hub_interleaved.log` | Serving checks passed; measured quiescent host growth -2.0 MiB and GPU growth 0 MiB; superseded parser falsely reported zero schedules | Registry-selected Hub source, exact typed streams, deterministic two-world waves, cleanup/reuse, and bounded server memory. Parser was corrected and rerun below. |
| 2026-09-11 | `CUDA_VISIBLE_DEVICES=2 serve_rollout.py --variant 360p --source hub --steps 8 --worlds 2 --concurrent-waves 2`; log `/tmp/waypoint_server_360_hub_interleaved_pass8.log` | Passed: exact 8 DiT executions per request, observed A/B/A interleaving, byte-exact solo/concurrent output, all cleanup markers | Corrected full registry-Hub 360p two-world gate. |
| 2026-09-11 | Exact unfiltered Nsight CUDA/NVTX commands below, `MSTAR_ENGINE_STEP_SYNC=0`, two sequential 8-step SDK requests per variant; `/tmp/waypoint-mask-{360,720}-full.nsys-rep` | Each variant: 16 steady forwards, exactly 16 graph replays, 0 synchronization/blocking calls | `check_nsys_replay.py` scopes API inspection to nested steady DiT `engine.forward` ranges and rejects anything other than one graph launch per forward, synchronize, blocking memcpy, or synchronous malloc/free. |
| 2026-09-11 | `PYTHONPATH=. pytest -q test/modular/test_waypoint_profiler.py`; exact report checks below | 4 tests passed; both real profiles passed | Reproducible profiler-query contract plus real 360p/720p evidence. |
| 2026-09-11 | Final consolidated 16-file Waypoint/startup/protocol/graph/resource selection | 385 passed, 41 skipped, 4 warnings in 41.19s | Current-tree CPU-verifiable MVP contracts; count includes 40 tests added since the earlier 345-pass row. |
| 2026-09-11 | `PYTHONPATH=. pytest -q test/modular/test_waypoint_streaming_benchmark.py test/modular/test_waypoint_profiler.py` | 15 passed | Streaming metric/orchestration and profiler-query contracts. |
| 2026-09-11 | Exact 360p benchmark command below | Passed; TTFF 0.148s, sustained 3.268x, p50/p95 0.020/0.027s, 0 baseline stalls, 3291.5 MiB peak PSS, 4648 MiB GPU | Registry-Hub captured baseline; 64 typed frames; slow-reader payload hash matched and GPU delta was 0 MiB. Artifact `baselines/streaming-2026-09-11-360p.json`. |
| 2026-09-11 | Exact 720p benchmark command below | Passed; TTFF 0.465s, sustained 0.570x, p50/p95 0.112/0.142s, 0 baseline stalls, 3687.3 MiB peak PSS, 5756 MiB GPU | Local captured baseline; 64 typed frames; slow-reader payload hash matched and GPU delta was 0 MiB. Artifact `baselines/streaming-2026-09-11-720p.json`. |
| 2026-09-11 | Exact 720p two-world command below; log `/tmp/waypoint_server_720_interleaved_memory.log` | Passed; startup 135.3s, exact 8 DiT executions/request, observed interleaving, deterministic output, all cleanup, -49.5 MiB host PSS/0 MiB GPU quiescent growth | Full local 720p three-wave world isolation, reuse, and bounded-memory gate. |
| 2026-09-11 | `git fetch --prune origin`; stash tracked/untracked work; `git rebase origin/main`; restore with `git stash apply` | Passed; branch is four commits above `origin/main` at `9ef65097`, with no unmerged entries | Range-diff found three patch-identical commits and one expected Bagel import-context merge; stable patch ID and all 25 untracked blobs matched the safety stash. |
| 2026-09-11 | Upstream/resource overlap and Waypoint CPU audit selections | 66 passed, 70 skipped; 279 passed, 12 skipped; 27 passed, 29 skipped | One TAEHV guidance assertion failed in the broad run, was corrected, and passed in the post-fix gate. |
| 2026-09-11 | Isolated sdist/wheel build after PyPI integration fixes | Passed; 125 `Requires-Dist` entries, zero direct URLs, Waypoint extra and default config present | Confirms the separately installed TAEHV pin does not make published metadata invalid. |
| 2026-09-11 | Post-rebase Python API/SDK/worker and native Rust checks | Historical: 51 Python tests and 16 Rust wire tests passed; `cargo check --locked` passed | Python raw-frame handling remains in scope. The Waypoint-specific Rust changes covered by this run were later reverted. Socket tests ran outside the restricted sandbox. |
| 2026-09-11 | `pytest -q test/modular/test_waypoint_packaging.py test/modular/test_waypoint_checkpoint.py test/modular/test_video_frame_protocol.py` | 73 passed, 2 warnings | Post-fix CLI/alias/metadata/pinned-TAEHV contracts and the complete Python raw-frame protocol gate. |
| 2026-09-12 | `PYTHONPATH=. pytest -q test/modular/test_video_frame_protocol.py test/modular/test_client_sdk.py test/modular/test_api_completion_guard.py` after reverting Waypoint-specific Rust frontend changes | 45 passed, 2 existing FastAPI deprecation warnings | Confirms the supported Python server/SDK frame path is unaffected; `rust/server/src/main.rs` and `test/rust/test_rust_frontend.py` have no remaining Waypoint diff. |

## GPU Reproduction Commands

All commands ran from the repository root on physical H100 GPU 2. The multiline
forms below are the exact invocations represented by the compact ledger rows.

```bash
env CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
  WAYPOINT_360P_PARITY_REPORT=/tmp/waypoint-360p-reference-parity.json \
  pytest -q -s test/modular/test_waypoint_360p_reference_equivalence.py

env CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 test/waypoint/serve_rollout.py \
  --variant 360p --source hub --cache-dir /tmp/waypoint-hf-cache \
  --steps 8 --worlds 2 --concurrent-waves 3 --measure-memory --physical-gpu 2 \
  --startup-timeout 1200 --request-timeout 600 \
  --log /tmp/waypoint_server_360_hub_interleaved.log

env CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 test/waypoint/serve_rollout.py \
  --variant 360p --source hub --cache-dir /tmp/waypoint-hf-cache \
  --steps 8 --worlds 2 --concurrent-waves 2 \
  --startup-timeout 1200 --request-timeout 600 \
  --log /tmp/waypoint_server_360_hub_interleaved_pass8.log

env CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 test/waypoint/serve_rollout.py \
  --variant 720p \
  --checkpoint-dir /mnt/storage/garv901/waypoint-1.5-1B/checkpoints/Waypoint-1.5-1B \
  --ae-path /mnt/storage/garv901/waypoint-1.5-1B/checkpoints/taehv1_5 \
  --seed-image /mnt/storage/garv901/waypoint-1.5-1B/checkpoints/seed/default.jpg \
  --steps 8 --worlds 2 --concurrent-waves 3 --measure-memory --physical-gpu 2 \
  --startup-timeout 1200 --request-timeout 900 \
  --log /tmp/waypoint_server_720_interleaved_memory.log
```

The profiler commands differed only in variant/source and output prefix:

```bash
env CUDA_VISIBLE_DEVICES=2 MSTAR_ENGINE_STEP_SYNC=0 PYTHONPATH=. nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --trace-fork-before-exec=true --cuda-graph-trace=graph --force-overwrite=true \
  -o /tmp/waypoint-mask-360-full python3 test/waypoint/serve_rollout.py \
  --variant 360p \
  --checkpoint-dir /tmp/waypoint-hf-cache/models--Overworld--Waypoint-1.5-1B-360P/snapshots/35acd20e649fe79c1c1002df456696408547202d \
  --ae-path ../../checkpoints/taehv1_5 --seed-image ../../checkpoints/seed/default.jpg \
  --steps 8 --enable-nvtx --startup-timeout 1200 --request-timeout 600 \
  --log /tmp/waypoint_mask_360_full_server.log

env CUDA_VISIBLE_DEVICES=2 MSTAR_ENGINE_STEP_SYNC=0 PYTHONPATH=. nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --trace-fork-before-exec=true --cuda-graph-trace=graph --force-overwrite=true \
  -o /tmp/waypoint-mask-720-full python3 test/waypoint/serve_rollout.py \
  --variant 720p \
  --checkpoint-dir /mnt/storage/garv901/waypoint-1.5-1B/checkpoints/Waypoint-1.5-1B \
  --ae-path /mnt/storage/garv901/waypoint-1.5-1B/checkpoints/taehv1_5 \
  --seed-image /mnt/storage/garv901/waypoint-1.5-1B/checkpoints/seed/default.jpg \
  --steps 8 --enable-nvtx --startup-timeout 1200 --request-timeout 900 \
  --log /tmp/waypoint_mask_720_full_server.log

nsys export -t sqlite -f true -o /tmp/waypoint-mask-360-full.sqlite \
  /tmp/waypoint-mask-360-full.nsys-rep
nsys export -t sqlite -f true -o /tmp/waypoint-mask-720-full.sqlite \
  /tmp/waypoint-mask-720-full.nsys-rep
python3 test/waypoint/check_nsys_replay.py \
  /tmp/waypoint-mask-360-full.sqlite --expected-forwards 16
python3 test/waypoint/check_nsys_replay.py \
  /tmp/waypoint-mask-720-full.sqlite --expected-forwards 16
```

Trace SHA-256 values are
`b82080765b36ca0da72fcd665153230869c60a83d51de11b33c13b910e159da9`
(360p) and
`a97777e4c375bce8b54a6b702d9ac01111f3db62eedf71a5c75d750d1af307ff`
(720p).

```bash
env CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 test/waypoint/benchmark_streaming.py \
  --variant 360p --source hub --cache-dir /tmp/waypoint-hf-cache \
  --physical-gpu 2 --steps 16 --warmup-steps 1 --slow-consumer-delay 0.25 \
  --startup-timeout 1200 --request-timeout 900 \
  --artifact /tmp/waypoint-streaming-360p.json \
  --log /tmp/waypoint-streaming-360p-server.log

env CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 test/waypoint/benchmark_streaming.py \
  --variant 720p \
  --checkpoint-dir /mnt/storage/garv901/waypoint-1.5-1B/checkpoints/Waypoint-1.5-1B \
  --ae-path /mnt/storage/garv901/waypoint-1.5-1B/checkpoints/taehv1_5 \
  --seed-image /mnt/storage/garv901/waypoint-1.5-1B/checkpoints/seed/default.jpg \
  --physical-gpu 2 --steps 16 --warmup-steps 1 --slow-consumer-delay 0.25 \
  --startup-timeout 1200 --request-timeout 900 \
  --artifact /tmp/waypoint-streaming-720p.json \
  --log /tmp/waypoint-streaming-720p-server.log
```

`uv run` could not be used for the initial CPU pass because the then-current extra
needed network access to resolve the pinned TAEHV source archive. A later clean
Python 3.12/uv resolution and separate real pinned artifact build closed CKPT-003;
the archive is now intentionally installed outside the index-safe project metadata.

## End-to-End Acceptance Checklist

- Both local checkpoint paths and the registry hub ID start through normal model
  construction without downloading unrelated repository assets.
- Graph-enabled runs either capture their declared buckets or fall back eagerly;
  graph-disabled runs declare no buckets.
- 360p and 720p requests each emit contiguous RGB24 chunks with complete metadata.
- The first emitted index is zero and the total is exactly `4 * num_steps`.
- Sequential and interleaved worlds preserve deterministic action mapping.
- Teardown releases request state and repeated world-slot reuse remains bounded.
