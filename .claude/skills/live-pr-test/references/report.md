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

<N>x <GPU>, <TP/config> · PR head `<sha>` vs <base> `<sha>`

### Claims

| Claim | Result |
| --- | --- |
| <claim, paraphrased short> | ✅ / ❌ / ⚠️ <the number or a few words of evidence> |

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

## Rules

- No cost, price, provider, region, driver/library versions or wall time. The hardware line is enough. (Competitor engine versions are fine in the performance table.)
- A section with nothing in it is left out, not filled with "none".
- No methodology paragraphs: the protocol follows the benchmarking skill unless a deviation changes how a number reads, and then it's one clause in the caveats line.
- Every number carries its harness, config and concurrency (AGENTS.md invariant 10), via the table it sits in. A mechanism you inferred but did not check is labeled a hypothesis.
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
