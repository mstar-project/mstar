# Qwen3-TTS Tier A — validation results

**Environment:** DGX node, NVIDIA B200, M\* branch `keisuke/mstar-sim`, colocated and Talker/Codec-disagg layouts.  
**Workload:** text → audio, 64 output frames, concurrency 4, 12 timed requests + 2 warmup (14 in step/profile logs).  
**SSS:** not used for this model (no Qwen3-TTS arch in vibesim).

---

## Pipeline (M\*-Sim only)

| Step | Command surface | GPU |
|------|-----------------|-----|
| 1. Capture | `mstar serve qwen3_tts` + benchmark | Yes |
| 2. Harvest | `mstar harvest` → stepdb | No |
| 3. Calibrate | `mstar calibrate` → `timing.json` | No |
| 4. Validate | `mstar validate` gates V1–V3 | No |
| 5. Simulate | `mstar simulate` (+ placement YAML) | No |
| 6. Ranking | V4 when two layouts are measured | Optional GPU for 2nd capture |

Configs: colocated `configs/qwen3tts.yaml`; disagg Talker @ rank 0, Codec @ rank 1 (`configs/qwen3tts_disagg.yaml` on this branch).

---

## Validation gates (colocated v2 capture)

| Gate | Outcome | Notes |
|------|---------|--------|
| **V1** semantics | FAIL | Decode batch count ~30% low vs logs; codec 1 sim step vs 11 measured — aggregation gap in DES, not harvest |
| **V2** step costs | **PASS** | Medians match capture for populated stepdb buckets |
| **V3** E2E | FAIL | Sim TTFT/E2E ~23–26% faster than measured on B200 |

Automate on **V2**. Review **V1** codec and **V3** before absolute latency claims. MR reference for this model class is ~16% worst-case E2E on H200 @ c=4; B200 run landed higher.

---

## Measured throughput (representative)

| Layout | Source | req/s | E2E median (approx.) |
|--------|--------|------:|---------------------:|
| Colocated | HTTP benchmark (v2) | 4.53 | ~850 ms |
| Colocated | Profile recovery (12 req) | 4.82 | — |
| Disagg (GPUs 4+5) | Profile recovery | 5.13 | ~621 ms |

Disagg wins on both throughput and E2E at this operating point.

---

## Simulate (same knobs: 14 req, prompt 32, output 64)

| Layout | sim req/s | E2E p50 | Codec worker GPU util |
|--------|----------:|--------:|----------------------:|
| Colocated | 6.61 | 535 ms | — |
| Disagg | 7.17 | 495 ms | **0%** (artifact) |

Sim ordering matches measurement (**V4 concordant**). Disagg sim **req/s delta is directional** — many `codec_chunk` / prefill rows are **missing** in stepdb lookup (zero-cost in timeline), so Codec-rank utilization in sim is not trustworthy yet.

---

## Stepdb / coverage

| Capture | Step records | stepdb rows |
|---------|-------------:|------------:|
| Tier A initial | 360 | 9 |
| Tier A v2 re-capture | 397 | 9 (unchanged) |
| Disagg measured | 419 | 7 |

Re-capture alone did not widen buckets. Missing costs track **prompt/KV shape in sim** (e.g. validate at 32 tokens vs capture prefill KV ~256) and sparse harvest grid.

---

## Integration harness (dual-backend POC)

| Manifest backend | Run | Acceptance |
|------------------|-----|------------|
| `mstar` | Full report from v2 artifacts | PASS (V2; WARN V1/V3 + review strings) |
| `sss` | Unified smoke preset dry-run | PASS (preset validate) |

Do **not** compare `throughput_req_s` across backends. See `shared-experiment-layer.md` for the vibesim launcher hook.

---

## Code on this branch

| Change | File |
|--------|------|
| `simulate --json` serialization | `mstar/sim/simulate_cli.py` |
| Disagg placement example | `configs/qwen3tts_disagg.yaml` |

Base simulator: **`keisuke/mstar-sim`**.
