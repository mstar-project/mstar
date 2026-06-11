# Ming-flash-omni-2.0 task-accuracy spot checks — 4×H100

Both runs against the same `vllm-omni 0.19.0` server + hybrid snapshot
(inclusionAI thinker + Jonathan1909 metadata/talker) used for the T2T
scaling sweep. Sampling is small — these are directional spot checks,
not publishable numbers. Dated 2026-06-06.

## Headline

| Suite | Items | Accuracy | Parse rate | Wall (s) | req/s |
|-------|-------|----------|------------|----------|-------|
| MMLU (0-shot, ~5/subject) | 285 | **78.9%** | 99.3% | 12.6 | 22.7 |
| VideoMME (chunk1 subset, stratified) | 51 | **56.9%** | 100.0% | 576.1 | 0.09 |

## MMLU breakdown

Sample: 285 items (cais/mmlu test, ~5 per subject across all 57 subjects). 0-shot.
Prompt: `<question>\n\nA. ...\nB. ...\nC. ...\nD. ...\n\nAnswer with just the letter (A, B, C, or D):`

### Per-subject (sorted by accuracy, worst first)

| Subject | Correct/Total | Accuracy |
|---------|--------------|----------|
| econometrics | 1/5 | 20% |
| philosophy | 2/5 | 40% |
| global_facts | 2/5 | 40% |
| virology | 2/5 | 40% |
| international_law | 3/5 | 60% |
| high_school_mathematics | 3/5 | 60% |
| electrical_engineering | 3/5 | 60% |
| conceptual_physics | 3/5 | 60% |
| business_ethics | 3/5 | 60% |
| high_school_chemistry | 3/5 | 60% |
| ... | ... | ... |
| professional_accounting | 5/5 | 100% |
| high_school_psychology | 5/5 | 100% |
| human_sexuality | 5/5 | 100% |
| high_school_computer_science | 5/5 | 100% |
| miscellaneous | 5/5 | 100% |
| high_school_government_and_politics | 5/5 | 100% |
| high_school_us_history | 5/5 | 100% |
| logical_fallacies | 5/5 | 100% |
| prehistory | 5/5 | 100% |
| high_school_european_history | 5/5 | 100% |

## VideoMME breakdown

Sample: 51 items from chunk1 (videos_chunked_01.zip, 30 videos), stratified evenly across short/medium/long durations.
Prompt: `<question>\n\nA. <opt>\nB. <opt>\nC. <opt>\nD. <opt>\n\nAnswer with just the letter (A, B, C, or D):`
Video sent as base64-inlined `data:video/mp4` content part on `/v1/chat/completions`.

### By duration

| Duration | Correct/Total | Accuracy |
|----------|--------------|----------|
| short | 13/17 | 76.5% |
| medium | 5/17 | 29.4% |
| long | 11/17 | 64.7% |

### By task type

| Task type | Correct/Total | Accuracy |
|-----------|--------------|----------|
| Temporal Reasoning | 0/3 | 0% |
| Counting Problem | 1/6 | 17% |
| OCR Problems | 1/4 | 25% |
| Attribute Perception | 1/4 | 25% |
| Action Recognition | 3/5 | 60% |
| Object Reasoning | 4/6 | 67% |
| Temporal Perception | 2/3 | 67% |
| Object Recognition | 6/8 | 75% |
| Information Synopsis | 5/6 | 83% |
| Spatial Reasoning | 1/1 | 100% |
| Action Reasoning | 2/2 | 100% |
| Spatial Perception | 3/3 | 100% |

## Caveats

- **Small N** — MMLU 5/subject and VideoMME ~17/duration are not enough
  for headline-quality numbers, especially the per-bucket breakdowns
  (e.g. VideoMME medium=29% is suspicious vs short=77% / long=65% and
  could be sample variance).
- **VideoMME videos limited to chunk1** — only 1 of the 20 dataset
  zip chunks was extracted (4.9 GB on `/dev/shm`). The full VideoMME is
  ~30 GB and would need extra disk to land in this container's overlay.
- **0-shot** for both — no in-context examples. Published Ming numbers
  may use chain-of-thought / few-shot for higher scores.
- **Greedy decoding** (`temperature=0`) on the thinker; matches the
  benchmark wiring used everywhere else in this branch.

## How to reproduce

Server: see [`benchmark/vllm_omni_instructions.md`](../../benchmark/vllm_omni_instructions.md) for the launch recipe.
Eval scripts were scratch (not committed) — both ~80 LOC, sending
`/v1/chat/completions` requests in a loop with the standard OpenAI
shape. JSON output ships per-item details next to this SUMMARY.