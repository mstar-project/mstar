# Report format

Write `results/report.md` on the box as you go (one section per suite as it finishes), so a run that dies halfway still leaves a readable partial report. After the copy-back it sits next to `media/`, `bench/` and `logs/` in the local results directory, and media links in it are relative to that directory.

The reader is the person who asked for the run. They know the codebase and will read the diff themselves; the report tells them what the PR does on real hardware. Lead with the verdict, keep the evidence one click away.

```markdown
# PR #<n> live test: <PR title>

<one line: verdict — e.g. "Claims hold; no regressions outside noise", "Claim 2 does not hold", "Blocked: <why>">

PR head `<sha>` vs main `<sha>` (merge-base `<sha>`) · <provider> <N>x <GPU> in <region>, $<rate>/hr, <wall time>, ~$<cost> · rust extension active on both arms

## Claims

| Claim (from the PR description) | Check | Result |
| --- | --- | --- |
| <claim, quoted or paraphrased> | <what was run> | ✅ holds / ❌ does not hold / ⚠️ partial — <number or one-line evidence> |

## Regressions

| Suite | Config | Metric | main | PR | Δ | Noise floor | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5 0.8B c=64 | qwen3_5_0.8b.yaml | text tok/s | 5,120 ± 60 | 5,090 ± 70 | −0.6% | ±1.5% (main vs main) | noise |

One row per suite × concurrency × metric that matters (TTFT p50, ITL, tok/s or audio-sec/s, RTF p50; `failed` must be 0). Mean ± spread across restarts. Say "regression" only when the delta exceeds the noise floor measured on this box in this run.

## Edge cases

- <case>: <what was sent> → <what happened>, with log excerpt or file link

## Media

| Suite | Prompt | main | PR |
| --- | --- | --- | --- |
| qwen3omni t2s | <short prompt> | [wav](media/qwen3omni_t2s/p0.main.wav) | [wav](media/qwen3omni_t2s/p0.pr.wav) |

## Not tested

- <what the plan skipped and why: hardware unavailable, no client support, out of budget, blocked by a failure>

## Failures and logs

- <anything that crashed, with the server log tail and the path under logs/>
```

## Rules

- Every number carries its harness, config and concurrency (AGENTS.md invariant 10). A mechanism you inferred but did not check is labeled a hypothesis.
- Report what was run, including what failed or was skipped. A suite that could not run is a "Not tested" line, never silently dropped.
- Length-sensitive metrics (req/s, audio-sec/s, RTF tails) only compare when output volume matches; check `Output bytes` first (benchmarking skill, trap 3).
- No debugging narrative. State what happened and the evidence.

## Posting to the PR

Only when explicitly asked. Then post the report body (without local paths; media links don't resolve on GitHub, so list them as "available on request") under this header:

```markdown
> 🤖 **Automated live test, AI-generated.** This report was produced by an automated workflow (Claude Code) that provisioned <N>x <GPU>, ran the PR head and main, and summarized the results. It has not been checked line by line by a human. Numbers come from a single allocation and carry the run-to-run noise described in `.claude/skills/benchmarking/SKILL.md`; treat small deltas as provisional, and verify anything that would block the merge.
```

Post with `gh pr comment <n> --body-file <file>`, and show the user the exact text first.
