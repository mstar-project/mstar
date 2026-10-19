# Report format

Write `results/report.md` on the box as you go (one section per suite as it finishes), so a run that dies halfway still leaves a readable partial report. After the copy-back it sits next to `media/`, `bench/` and `logs/` in the local results directory, and media links in it are relative to that directory.

The reader is the person who asked for the run, and possibly the PR's author and reviewers. They know the codebase and will read the diff themselves; the report tells them what the PR does on real hardware. Lead with the verdict, keep the evidence one click away.

**The report is about the PR, not about the run.** Most of what goes wrong during a live test is the test setup: provisioning, the image, model staging, disk, ports, the benchmark client, a flag the harness got wrong, a script in this skill. Those problems and their fixes are not findings, even when they took most of the run, and they never go in `report.md`. Before writing anything down as a finding, ask: *would this also happen on the base branch with the same setup?* If yes, or if you only saw it before fixing something in the harness, it is not about the PR.

- A setup problem that was fixed: leave it out of the report entirely.
- A setup problem that kept a check from running: one "Not tested" line with a short neutral reason ("not run: server did not start on either arm in this environment"), no story.
- A bug that reproduces on the base branch too (a pre-existing issue the run happened to hit): not a finding about this PR. Note it in `harness-notes.md`, not in the report, unless it stops a PR claim from holding, in which case say so on that claim's row.

**Headline numbers give the PR the benefit of the doubt.** When an environment problem slows the PR's code on this box and you find a working configuration (a kernel backend that fails here and falls back, a driver or NCCL feature the box lacks), the claims table, the verdict and the performance table use the working configuration's numbers. The degraded number appears once, in a performance bullet that says what failed and which setting worked. If the fallback would also hit production users (it's on the base branch, or the PR adds it), say so in that bullet; that's what makes it worth a reviewer's attention. Apply the same working configuration to the base arm and the competitors whenever it applies to them.

Write all of those to `results/harness-notes.md` instead: what went wrong, the fix or workaround, and what to change in this skill or its scripts so the next run doesn't hit it. That file stays local, is never posted, and is how the workflow gets better.

Keep it short enough to read in a minute: aim for under ~3,500 characters, about one screen. Whatever didn't make the cut stays in the local results directory (per-run tables, worker phases, protocol details, versions, logs), and the reader can ask for it.

```markdown
## Live test: <PR title>

**<verdict in one or two sentences, e.g. "Works; no regressions outside noise." / "The port works, but claim 2 doesn't hold.">**

<N>x <GPU>, <TP/config>, <tensor transport> · PR head `<sha>` vs <base> `<sha>` · <only if any: a dependency below the repo's floor, e.g. "Mooncake 0.3.8.post1, below the 0.3.13.post1 floor">

### Claims

| Claim | Result |
| --- | --- |
| <claim, paraphrased short> | ✅ / ❌ / ⚠️ / ➖ <the number or a few words of evidence> |

### Performance

<one table, only when the PR claims a number or a regression suite ran: one row per arm/system, columns per metric, concurrencies inline as "a / b / c">

- **<finding in bold>.** <one or two sentences: mechanism with a code path, labelled a hypothesis if unverified>
- <at most three or four bullets>

<one line of caveats that change how a number reads, only if any>

### Other

- **Regressions:** <one line, or a small table only when something moved>
- **Edge cases:** <one line each, only ones the PR's code is responsible for>
- **Failures caused by the PR:** <only failures on the PR arm that the base arm doesn't have under the same setup; omit otherwise>
- **Not tested:** <comma-separated, a few words of reason each>
- **Media:** <links when there is media; omit otherwise>
```

## Result marks

- ✅ held, ❌ failed on the setup it was measured on, ⚠️ held only partly (some cells, some restarts).
- ➖ not exercised here, with the reason: the base bug didn't fire on this box (so neither arm can show the fix), or the claim's number came from another GPU, harness or metric and this run can't stand in for it ("not comparable: B200 claim, host-bound H200"). Never ⚠️ for these; ⚠️ says the PR fell short.
- ➖ inconclusive, too, when an output changes by no more than base's own run-to-run variation and there is no reference to say which side is right. Better: add a reference arm when one is cheap (fp32, `'highest'` matmul precision, or the reference implementation).
- Before marking a description claim ❌, check which test, harness and inputs the claim was measured with. When your run used a different one (the PR's stricter new test run on base, say), name both in the row, so the ❌ explains itself.
- A case the description hands to another open PR is "owned by #N" with a link, not a ⚠️ here. Test it on that PR, or on both merged, if it matters to the verdict.

## Rules

- No cost, price, provider, region, driver/library versions or wall time. The hardware line is enough. (Competitor engine versions are fine in the performance table, and a dependency below the repo's floor goes in the header line.)
- A rate claim (stalls, hangs, failures per run) states the counts on each arm and whether they separate: a one-sided Fisher exact test at p < 0.05 is the rule. When they don't, say "not separable at N runs per arm" and, if cheap, how many runs would be needed; don't call it fixed or broken.
- A section with nothing in it is left out, not filled with "none".
- No methodology paragraphs: the protocol follows the benchmarking skill unless a deviation changes how a number reads, and then it's one clause in the caveats line.
- Every number carries its harness, config and concurrency (AGENTS.md invariant 10), via the table it sits in. A mechanism you inferred but did not check is labeled a hypothesis.
- Put the exact client command for every number in the report (a collapsed block is fine), so the author can rerun it. Set the client's model-dependent defaults explicitly (`tts_speech_bench --sample-rate` is 24 kHz by default; a 48 kHz model then reads 2x audio-s/s and half the RTF), and say whether a concurrency is closed-loop or one burst (`tts_speech_bench` sends each repeat as a single burst of c).
- Next to a claimed number, name the claim's GPU, harness and metric when they differ from yours. A mismatch is ➖ not comparable, not a failure.
- Every compared number has at least two repeats per arm, three by default (server restarts for perf; runs or samples for fault-injection, quality and output checks). Two are enough only when they agree very closely. A cell with one run per arm is "1 run, not a comparison", never a delta or a verdict.
- "Within noise" names its rule in one clause, e.g. "inside the spread across restarts of either arm".
- An output diff between arms needs a same-code control (two processes of one arm). If the same prompts differ in every PR process, suspect a changed input; if a different prompt differs in each process, it is a near-tie fork, so say which pattern you saw.
- Report what was run, including what failed or was skipped. A suite that could not run is a "Not tested" line, never silently dropped.
- Length-sensitive metrics (req/s, audio-sec/s, RTF tails) only compare when output volume matches; check `Output bytes` first (benchmarking skill, trap 3).
- No debugging narrative. State what happened and the evidence.
- Nothing about the harness, the box, or how the run was made to work (see above). The header line already says what hardware it ran on; that is all the reader needs about the environment.

## Posting to the PR

Only when explicitly asked. Then post the report body (without local paths; media links don't resolve on GitHub, so say "media available on request") under this header:

```markdown
> 🤖 **Automated live test, AI-generated** (Claude Code). Not checked line by line by a human. Numbers come from one allocation, so treat small deltas as provisional.
```

Post with `gh pr comment <n> --body-file <file>`. Show the user the exact text first, unless the request said to post without asking. Before posting, reread the body once for anything about the test setup that slipped in, and cut it.
