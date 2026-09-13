# Waypoint Port Progress

The active plan and durable records are in `docs/waypoint/`. This file preserves
the earlier investigation log, including failed approaches and measured results.
Its old phase numbers and "done" labels describe those experiments; they do not
close the current scripted-streaming MVP gates.

## Current MVP Status

| Phase | Scope | State |
|---|---|---|
| 1 | Documentation and baseline | Recorded; durable plan/decision/validation/backlog records added |
| 2 | Startup and configuration | Passed, including clean install and registry-selected Hub startup |
| 3 | Request and numerical correctness | Passed, including 41-step same-process parity |
| 4 | DiT and attention execution | Passed, including required full-size capture and host-sync-free steady replay |
| 5 | Encoder and decoder CUDA graphs | Passed, including real-weight parity and full-size eight-step streams |
| 6 | Streaming frame protocol | Passed through Python/Rust server and typed SDK at both resolutions |
| 7 | End-to-end 360p/720p MVP gate | Passed: local/Hub, sequential/interleaved, cleanup/reuse, bounded memory |
| 8 | Streaming viability harness | Baseline passed at 360p/720p; threshold remains a later decision |

The earlier baseline was **242 CPU tests green** at `HEAD` `0b88001a`. It is
historical, not a claim about the current dirty tree. Current commands and results
belong in `docs/waypoint/VALIDATION.md`.

## Historical Investigation Log

### Wave 1 — three parallel streams, no file overlap

| stream | phase | owns |
|---|---|---|
| A | 7 | `test/modular/test_waypoint_gpu.py` (new only) |
| B | 8 | `pyproject.toml`, `test/waypoint/record_oracle.py`, checkpoint download |
| C | 10 + 11a | `configs/waypoint.yaml`, `submodules.py`, `waypoint_model.py`, `test_waypoint_shell.py` |

Split this way because the three touch disjoint files.

Checkpoints download **outside the worktree** (`../checkpoints/`) so they cannot
enter the diff.

### Wave 2 — phase 9, launched once B's oracle landed

| stream | phase | owns |
|---|---|---|
| D | 9 | `test/modular/test_waypoint_reference_equivalence.py` (new only) |

Runs alongside A and C, which still own their own files. D compares **eager to
eager**: the oracle was recorded with the 4+1 driver unrolled to capture per-pass
outputs, so it carries the reference's arithmetic but not its kernel selection.

### Box constraints

Only **GPU 2** is usable. 0, 1 and 3 each hold ~72–75 GB of another job, so anything
CUDA must run under `CUDA_VISIBLE_DEVICES=2` and no test may assume a device count.
torch is 2.9.1+cu128; 4.4 TB free on `/mnt/storage`. A and D now share GPU 2 — the
oracle run peaked at 18 GiB, so there is headroom, but a CUDA OOM in either is
contention before it is a bug.

### Reported results

**Phase 7.** `test/modular/test_waypoint_gpu.py`, 10 tests green (~22 s warm, 93 s
cold), prose 26.9%. A.1 **failed as shipped** and is fixed (below). A.3/A.4/A.5 pass
bit-exact. A.2 does not — bounded instead, at 4 bf16 ulp of frame peak against a
worst measured 2.35 over 120 comparisons; localized to inductor holding bf16
pointwise intermediates in fp32 across a fusion, which moves compiled *toward* an
fp32 reference (0.0168) and away from eager (0.0252), and does not compound across
20 frames. Flex compiled-vs-eager and the GEMMs are 0.0. Also fixed the
`compile_dit` and `369 keys` docstrings the two agents flagged.

**Phase 8.** `tensordict==0.10.0` and `taehv 0.1.0` installed (torch untouched at
2.9.1+cu128); both declared under a new `[waypoint]` extra. Checkpoints at
`../checkpoints/{Waypoint-1.5-1B,taehv1_5}` (11 GB / 22 MB). `build_waypoint_dit`
loads the real `model.safetensors` clean — 174 params, 1.282 B, the 2 fp32
`NoiseConditioner` params intact. Oracle recorded to `../oracle` (9.2 GB, 41
frames, 62 s): `test/waypoint/record_oracle.py`, world_engine only.

**Phase 8, re-recorded.** The first cut ran eager and was wrong: eager
`flex_attention` ignores the `BlockMask` block index lists, so every pass attended
over unwritten ring slots (rel 0.68 against the compiled reference). State now
comes from `engine._denoise_pass`/`_cache_pass` verbatim. It cannot be instrumented
— making the per-pass output an extra output of that region, or splitting it into
five, moves the latent ~1 bf16 ULP and compounds — so `dit_out` comes from frozen
shadow passes, and every run checks they leave the state alone.

Two findings Phase 9 depends on. **The reference is not bit-reproducible across
processes**: two processes running it alone disagree by 1 ULP at layer 0, compounding
to 11.6 on peak 20.4 by layer 23; latent drifts 0.03 → 0.38 over 6 frames. Not
cudagraphs; `max_autotune` is one source, not the only one. So nothing supports a
bit-exact assertion — `../oracle/repro/` is a second independent recording so the
floor is measurable from artifacts. **And a port that runs its denoise passes as
separate compiled regions cannot match the reference's latent**, for the same
fusion reason. The port is not such a port — `compile_regions()` compiles the same
two outer regions the reference does — but any future refactor that splits them
inherits a ~1 bf16 ULP floor at frame 1.

**Why phase 9 survives this.** Its bit-exactness is an *in-process* claim: the
parity file drives the reference live and uses the oracle only for inputs (noise,
controls, seed latent) and kernel-independent bookkeeping (`written`, live buckets).
Where it starts both sides from an oracle ring snapshot, both get the same bytes, so
cross-process drift cannot enter. Verified by reading the file, not assumed. What
the finding *does* constrain is **L4**: the oracle's stored `latent` and `pixels`
are not bit-exact targets — pixels drift up to 72/255 across processes — so L4 must
either drive the reference live the same way, or assert against the floor measured
from `../oracle/repro/`.

**The parity claim, stated exactly.** Port == reference, bit-exact, *when both run
in one process with the attention kernel shared and `reference_compat=True`*. That
is narrower than "the port matches the reference", and it is the strongest claim the
reference's own non-determinism permits.

**Phase 9.** `test/modular/test_waypoint_reference_equivalence.py`, 10 tests, prose
26.8%. Result: **the port is not bit-equivalent to the reference, because the port
is more numerically correct than the reference.** As served, every layer diverges —
L1 frame 0 maxabs 9.06e-01 (rel 1.38e-01), first divergent stage `cond`, then
`rope`, then the blocks; L2 diverges at pass 0; L3 at frame 0. `written` masks are
**equal at every frame**, so ring bookkeeping is correct and only arithmetic
differs. The control is what makes this conclusive rather than a guess: injecting
the reference's three tables plus its cached LUT gives **0/30 divergent stages** and
a **41-frame rollout bit-exact** on latent, ring bytes and `written`, at every frame
and layer. Every tolerance in the file is exactly 0.0, justified by that control.

**`WaypointConfig.reference_compat`** (the phase-9 implementation). At the time of
this experiment its default was `False`; WP-001 has since superseded that choice
and makes it `True` for serving. On, it bf16-round-trips the three tables where they are built and serves
the reference's batch-5 sigma LUT; off, the exact path is unchanged. Threads through
5 lines of `dit.py`; `submodules.py` and `waypoint_model.py` needed nothing.
**Flag on ⇒ 0.0 everywhere**: three tables, the LUT at all 5 sigmas, 30 stages,
both forwards, all 5 pass outputs, and the 41-frame rollout on latent, ring bytes
and `written` at every frame × layer. The port builds its own batch-5 table from its
own quantized `freq` and matches the reference bit-for-bit, so the flag is
self-contained rather than borrowing reference state — the hand-injection helper is
deleted. Parity file 5 failed/5 passed → **14 passed**; 277 passed across the
waypoint + ring + flex + GPU set; ruff clean.

The five failing tests became `[exact]` / `[reference_compat]` pairs. The exact side
asserts *characterised* divergence, not magnitudes — the strongest being
`torch.equal(bf16_roundtrip(port_table), reference_table)`, a zero-tolerance
identity that survives an oracle re-record and goes red if the divergence ever stops
being `NoCastModule`'s cast. Nothing got weaker; `written`-mask equality and that
identity are new. Verified independently: no `allclose`/`atol`/`rtol`/`approx` in
either parity file.

**Phase 10 + 11a.** `components/taehv.py` (port of `ae.py`'s
`ChunkedStreamingTAEHV` + `load_taehv`, all `taehv` imports deferred), the two VAE
submodules, the rewired walks, and `configs/waypoint.yaml`. Walks are now
`prime: vae_encoder → dit → vae_decoder → EMIT` and
`rollout: Loop { dit → vae_decoder → EMIT }`; the DiT node's contract is unchanged
and `get_node_resources` stays DiT-only. Streaming state is one
`ChunkedStreamingTAEHV` per request per AE node in `PerRequestState.kwargs`, dropped
by the engine's own `cleanup_request` — no cleanup code on either node. `taehv`
landed mid-task, so the port was checked against `world_engine/src/ae.py` directly:
**encode and decode bit-identical in fp32 and bf16**, including the moved
`.div(255)`. Tests still run on a fake `taehv`, and one pins that the tree imports
with the package absent. 253 passed, 3 skipped; ruff clean.

Config is one `node_groups` entry with `[vae_encoder, dit, vae_decoder]` on rank 0,
not wan22's split — a worker boundary inside the rollout loop would put a process
hop between the DiT and an order-dependent decoder.

## Problems hit

- **Wave 1 was killed mid-run and produced nothing.** The parent process exited
  while all three streams were still working; none had written a file. Verified on
  disk: no `test_waypoint_gpu.py`, no `test/waypoint/`, no `configs/waypoint.yaml`,
  no `../checkpoints/`, and neither `tensordict` nor `taehv` installed. All three
  resumed from their saved transcripts with the GPU constraint added — their
  exploration survived, their output did not.
- **`pip install taehv` from the pinned URL silently installs nothing.** pip here
  is 22.0.2 (Ubuntu system pip) and cannot parse `Metadata-Version: 2.4`, which
  modern setuptools emits for taehv's PEP 639 `license = "MIT"`. It falls back to
  project name `unknown`, rejects the URL on a name mismatch, and installing the
  extracted directory instead produces an empty `UNKNOWN-0.0.0` wheel with no
  `taehv.py` in it — and uninstalls any other `UNKNOWN-0.0.0` on the box on the
  way past. Fixed by building the wheel directly with the system setuptools
  (`setuptools.build_meta.build_wheel`) and installing that. uv, which the
  reference uses, does not hit this.
- **`fullgraph=True` did not hold on the shipped code.** `_denoise_pass` had
  `zip(sigmas, sigmas.diff(), strict=False)`; Dynamo rejects a ragged `zip` under
  `fullgraph` regardless of `strict`, with `UserError: zip() has one argument of
  len differing from others`. Localized by monkeypatching a zip-free pass, which
  compiled clean. Fixed to `zip(sigmas[:-1], sigmas.diff(), strict=True)` — same
  four iterations, same values, bit-identical in eager. `capture_scalar_outputs`
  is **not** needed and the test asserts it stays unset. The oracle's own unrolled
  driver still uses `strict=False`, which is fine: it is never compiled.
- **Gate B costs ~6× the old estimate.** 120 eager `BlockMask` rebuilds at real
  720P = **14.31 ms/frame**, of which the block-alignment `torch.equal` sync is
  4.38 ms — against 40.8 ms/frame steady state, so it is ~35% of frame time. The
  `(layer, frame)` cache would cut it to 24 rebuilds = 2.86 ms, **saving 11.45
  ms/frame**. The plan gated this fix on the measurement; the measurement says do
  it. Deferred until stream D finishes so it cannot contaminate parity debugging.
- **`compile_dit`'s docstring was wrong** — it claimed compilation is "a
  throughput knob only" that "does NOT govern attention correctness." It is a
  **capture prerequisite**: eager capture dies with
  `cudaErrorStreamCaptureInvalidated` at `make_block_mask`'s device-to-host sync.
  Corrected in `config.py`, and shorter than what it replaced.
- **The reference's own buffer handling is lossy.** `NoCastModule._apply`
  (`world_engine/src/model/nn.py:11-24`) casts fp32→bf16 and then casts *the result*
  back to fp32. Parameters recover because `load_state_dict` refills them
  afterwards; **non-persistent derived buffers never do**. So the served reference
  runs on bf16-quantized `denoise_step_emb.freq` (1.80e-03), `rope_angles.xy`
  (1.88e-02) and `rope_angles.inv_t` (1.78e-04). Not cosmetic: `freq` is multiplied
  by `sigma*1000`, so 1.8e-3 relative is a RoPE phase error up to ~1.8 rad. The
  reference warns about it itself. Verified in the source before acting on it.
- **The oracle was recorded under eager attention and is partly unusable.**
  `record_oracle.py` calls `engine.model(...)` directly (lines 264, 270, 354),
  bypassing the reference's two `@torch.compile` regions. Eager `flex_attention`
  ignores a `BlockMask`'s block index lists and attends **unwritten ring slots** —
  a property this port already pins. Oracle vs reference-compiled is maxabs 5.6406
  on peak 8.25 (rel 0.684). So `dit_out`, `committed_kv`, and the `latent`/`pixels`
  that decode from them are **not valid parity targets**; inputs, `ctx`, noise,
  `written` masks and the live-bucket pattern still are. Phase 9 worked around it by
  driving the reference live with attention rebound to the port's own
  `flex_attention_masked`, so the kernel could not be the variable. **Being
  re-recorded through the compiled path** — L4 pixel parity depends on it.
- **The decode-ordering comment was wrong, and the test that caught it stands.**
  `enable_async_scheduling=False` does *not* serialize dit→decoder; it only
  disables speculation (`Worker._can_speculate`). What actually stops the DiT
  running twice before a decode is that `NodeManager.pop_ready_nodes` **removes** a
  node from the ready set when it schedules it, and the DiT's persisted controller
  streams are re-injected only at the loop's iteration boundary. Surfaced by a loop
  test asserting `ready_node_names == {vae_decoder}` and failing. Comments and test
  now say the real reason.
- **Latent bug in `mstar/graph/base.py` — not waypoint's, not currently reachable.**
  `GraphStateRegistry.mark_entity_complete` tracks `_num_completed_entities` as a
  *count*, not a set, despite a "no-ops if already done" docstring. Marking one
  entity complete twice inside a 2-node loop body satisfies
  `_num_completed == _num_managed`, fires `complete_iter()`, and `reset_for_iter`
  then silently discards the latent queued in the other node's `ready_signals`.
  Single-node loop bodies (wan22, cosmos3) cannot hit it; waypoint is held off it
  only by the pop semantics above. Any future multi-node loop body is one re-ingest
  away. **Reported, not fixed — shared engine code, outside this port's scope.**
- **The `369 keys` in `weight_loader.py`'s key map is stale** — the shipped
  checkpoint has **393**. Prose only; nothing computes from it, and the loader's
  completeness contract passes on the real file.
- **Stream C's in-flight work fails 6 tests in `test_waypoint_shell.py`**
  (`KeyError: 'vae_encoder'`, `NameError: WorkerGraphIO`) — the Phase 10 VAE nodes
  do not exist yet. Not caused by the phase 8 dependency changes: the only tracked
  phase 8 edit is +17 lines of `[project.optional-dependencies]`, and the other
  217 tests in the ring/flex/waypoint set stay green.

## Historical decisions

These record the choices made on 2026-09-10 after phase 9. The first choice was
superseded by WP-001 on 2026-09-11; it remains here to explain the implementation
and measurements that followed it.

- **Superseded: the port ships the mathematically exact tables by default.** The
  reference serves bf16-quantized RoPE
  and conditioner tables because of the lossy `NoCastModule._apply` round-trip
  above. Rather than be bug-compatible, the port stays exact and gains a
  reference-compat flag alongside the existing `full_global_ring`, which is already
  there "to restore the reference's allocation for an A/B parity run". Flag on ⇒
  bit-exact over 41 frames, which validates every other line of the port; flag off ⇒
  exact-table serving differed from the reference by one characterised difference.
  Rejected: bug-compatibility (ships a ~1.8 rad phase defect the reference's authors
  appear not to have intended) and dropping bit-exactness as the gate (leaves no
  zero for a future regression to fail against).
- **`patch_cached_noise_conditioning` sits behind the same flag.** It is measurably
  **not** a no-op — the planning assumption fell the wrong way. The cause is the
  batch shape, not the LUT: the reference builds the table by evaluating the fp32
  MLP on all 5 sigmas at once (M=5 ⇒ TF32 tensor-core GEMM under
  `float32_matmul_precision('high')`), while serving one sigma at a time is an M=1
  GEMV that stays exact fp32. Measured batch5-vs-batch1: **3.125e-02 @ `high`, 0.0 @
  `highest`**, 3.125e-02 @ `medium`. Per-sigma the patch is off the live path by
  1.5625e-02, except sigma 0.75 at 3.125e-02. Only the `CachedDenoiseStepEmb` half
  diverges — **`CachedCondHead` measures 0.0 at all five sigmas and needs no compat
  treatment**. Serving keeps the exact per-sigma GEMV; parity runs reproduce the
  reference. One flag, so tables and LUT can never disagree about which model is
  being served.

Three independent measurements converge on the M=5 TF32 story — phase 8's
calibration probe (`high` vs `highest` = 0.03125 on the conditioner), phase 7's pin
test (all precisions identical at the served M=1 shape), and phase 9's direct
batch5-vs-batch1 comparison. Phase 8 read its 0.03125 as only proving the probe was
live; it was also the signal, unrecognised at the time.

## Assumptions made

Carried in from planning, to be confirmed or killed by the work:

- ~~**`patch_cached_noise_conditioning` is a numerical no-op.**~~ **Killed by phase
  9.** The reasoning was that the LUT is built by running the same fp32 island on
  bf16 sigmas and rounding to bf16, which is what the port's live path returns. That
  missed the batch shape entirely. See *Decisions taken* above — a good example of
  why the plan said to test this rather than assert it.
- **The patched reference is the reference.** `world_engine.py:84` applies
  `apply_inference_patches` unconditionally, so parity targets the patched model.
- **Same-torch parity.** The reference pins `torch==2.11.0`; this env has 2.9.1.
  Both sides run at 2.9.1 so the comparison means something; this deviates from the
  reference's pin deliberately.
- **The two matmul-precision settings differ, but not observably** — mstar sets
  `'high'`, the reference `'medium'`. Settled by phase 8: the oracle records under
  `'high'` (serving's value, set *after* importing world_engine, which sets
  `'medium'` at import), and a calibration probe run at record time measures
  `'high'` and `'medium'` as bit-identical on this build — 0.0 on both a 4096²
  fp32 GEMM and on the NoiseConditioner LUT. `'highest'` differs from both (0.104
  and 0.03125), which is what shows the probe is live. So the flag cannot explain
  a phase 9 mismatch on this box; it can on another. Phase 7 reached the same
  answer by a second, independent route: at the served shape (B=N=1) the
  `NoiseConditioner` matmuls are GEMVs (M=1), where all three settings are
  bit-identical. Its pin test asserts both the setting and the reason it does not
  bite, with an N=8 precondition that fails if the flag ever stops being live.
- **Historical implementation: `reference_compat` + `compile_dit` needed one eager frame first.** The sigma LUT
  is built on first forward and cached per device, because it needs loaded weights —
  the same contract `compile_regions`' docstring already states for the RoPE and
  token-grid tables. An operational constraint on the parity configuration only;
  the exact-table default at that time had no LUT. The active plan instead requires
  post-load table materialization before compile/warmup.
- **Both LUTs are built under the same `float32_matmul_precision`.** At `'highest'`
  the batch-5 GEMM stops rounding and the compat conditioner would converge on the
  exact one; a test asserts the exact path still differs, so that surfaces red
  rather than silently. The fixture ordering that guarantees it is commented.
- **The port's global-ring compaction touches only slots the reference never
  addresses** — the reference's live region is `[0, port ring_len)` plus scratch
  `[L, capacity)`, with `[8192, 65536)` dead. Not taken on faith: phase 9's
  `test_the_port_compacts_only_slots_the_reference_never_addresses` proves it
  against the oracle's `written` masks, which survive the oracle's eager-attention
  defect.
- **VAE nodes use `disable_autocast = True`, not the fp32-island mixin.** TAEHV is
  uniformly bf16, so the recorded island set would be empty, and
  `EngineManager.build` skips the blanket cast entirely under `disable_autocast` —
  stronger than restoring dtypes after a cast that already rounded. The only such
  mixin is wan22's, and this port deliberately does not import across models.
- **The emitted payload is raw uint8 RGB** `[temporal_compression, H, W, 3]` in C
  order, no container: a per-step mp4 would be an unplayable fragment.
- **A 1-frame seed clip is repeated to fill `temporal_compression`**, matching
  `gen_sample.py`'s `seed_frame_x4`. Anything but 1 or exactly
  `temporal_compression` is refused at the API boundary. Aspect ratio is checked,
  resolution is not — the AE resizes 16:9 input onto its own grid.
- **The noise must be supplied, not reproduced.** The reference draws bf16 on
  device unseeded; the port draws fp32 from a seeded CPU generator. The oracle
  saves the exact noise tensor per frame so phase 9 feeds the port the same draw —
  without that there is nothing bit-exact to compare.
