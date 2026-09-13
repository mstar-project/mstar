# Waypoint Decision Record

## WP-001: Reference-Compatible Numerics by Default

- **Status:** Accepted, 2026-09-11
- **Decision:** `reference_compat=True` is the serving default. Setting it to
  `False` selects experimental exact-table arithmetic.
- **Evidence:** A live same-process 41-frame run previously measured zero
  difference for reference-compatible tables, conditioner, DiT stages, five-pass
  output, ring writes, and rollout latents. Exact-table arithmetic intentionally
  diverges because the released reference BF16-round-trips derived FP32 tables and
  builds its sigma table with a batch-5 operation.
- **Consequence:** Release validation targets reference-compatible mode. Exact
  mode remains useful for investigation but cannot satisfy the reference parity
  gate.

## WP-002: Resolve and Validate Before Allocation

- **Status:** Accepted, 2026-09-11
- **Decision:** Resolve local paths before considering a string to be a Hugging
  Face ID. Download only root native safetensors plus `config.yaml`; resolve
  `taehv1_5.pth` separately. Validate the manifest's architecture, supported
  geometry, scheduler, temporal compression, and FPS before device allocation;
  validate tensor completeness and the pinned TAEHV runtime architecture while
  loading, before request admission.
- **Reason:** The upstream repository also contains redundant transformer and VAE
  weights. Downloading the whole snapshot wastes several GiB and allows partial or
  incompatible snapshots to fail late.

## WP-003: Optional CUDA Graph Acceleration

- **Status:** Supersedes the required-capture policy, 2026-09-11
- **Decision:** `cuda_graph=True` attempts capture for encoder prime, steady DiT
  rollout, decoder initialization, and steady decoder execution. Capture failure
  falls back to eager execution; `cuda_graph=False` declares no capture buckets.
  `compile_dit` independently controls the two outer DiT regions.
- **Reason:** CUDA graphs are an acceleration mechanism, not part of the model's
  numerical contract. The engine already supports eager fallback, and Waypoint's
  ring, mask planning, and functional AE state have eager execution paths.
- **Constraint:** The masked FlexAttention primitive remains compiled because bare
  eager `flex_attention` ignores this BlockMask's block-index visibility data.
- **Exception:** The one-time DiT prime/cache pass is compiled with
  `fullgraph=True` only when `compile_dit=True`, and remains uncaptured.

## WP-004: Internal Prime Is Not User Output

- **Status:** Accepted; end-to-end validation passed
- **Decision:** Prime with an internal idle action, preserve user action zero for
  the first generated latent, initialize all model state, and emit no reconstructed
  seed frames.
- **Consequence:** A request with `num_steps=N` supplies exactly `N` actions and
  emits exactly `4*N` RGB frames indexed from zero.

## WP-005: Typed Streaming RGB Frames

- **Status:** Accepted; live 360p and 720p validation passed
- **Decision:** Waypoint emits only the `video_frame` modality in streaming mode.
  Payloads are contiguous RGB24 with width, height, FPS, pixel format, frame index,
  and frame count metadata.
- **Consequence:** The OpenAI encoded-video endpoint is not a Waypoint transport.
  Non-streaming `video_frame` requests fail before execution. The scripted MVP
  supports only the Python frontend; `--rust-frontend` support is deferred.

## WP-006: TAEHV Source Pin and Installer Floor

- **Status:** Accepted
- **Decision:** TAEHV is installed separately from the index-safe `waypoint` extra
  and pinned to upstream commit `7dc60ec6601af2e668e31bc70acc4cb3665e4c22`.
  Direct URLs cannot appear in metadata published to PyPI. Supported installation
  uses `uv>=0.4.0` or `pip>=24.3`; absence of TAEHV must produce an actionable
  error containing its exact pinned archive command.
- **Reason:** Old pip releases misread the source package's Metadata-Version 2.4
  metadata and can install an empty `UNKNOWN` wheel.

## WP-007: Streaming Benchmark Follows the MVP

- **Status:** Accepted; first captured baseline recorded
- **Decision:** The first trustworthy captured run establishes a baseline. It does
  not invent a release threshold. A later decision record sets a viability
  threshold from the `STREAM-001` measurements.
- **Evidence:** Captured 16-step baseline and matched slow-consumer runs passed at
  360p and 720p. The durable JSON artifacts and summarized measurements are in
  `OPTIMIZATION_BACKLOG.md`.
