# Decisions

Gaurav's calls on PRs and corrections to his drafts, newest first, as of Oct 27 2026. They are newer than `../SKILL.md` and win over it where the two differ. The calls are decisions on record for hard rule R2 and the Overlap pass; the standards are how he finds and states a problem.

## Calls

- 2026-10-27 #800, #804: #800 merges first and #804 is evaluated after it. Why: #804's DROID deployment depends on #800's camera feed and prefix scales.
- 2026-10-26 #404, #413: no hand-written count, cap, range, limit or key list where the engine state, the model's declarations, its checkpoint config or the sampler resource could give it. Each such value is a finding naming the source it could come from.
- 2026-10-25 #754: closed in favor of #419. A follower-report wait was a second mechanism beside rank 0 deciding and sending (`ScheduleTPNode` already carries `prefix_hits` and `page_trails`), so the fix moved into #419 as rank 0 sending each row's span. Why: derive from what the system already decides rather than have each node declare its shape-setting inputs.
- 2026-10-25 #754: fix a rare failure on the failure path (the failed row runs on zeros and the request fails through the conductor) rather than gating every request on it. Why: the gate made every TP request wait for each follower's report before its first step, to cover a read that almost never fails.
- 2026-10-25 #568: the code was right, and cold and warm boot time against main was the one thing left to settle. Why: the only plausible regression was boot time, and the author owns that evidence.
- 2026-10-24 #476: the LiveKit example parked; `stream_format` streaming on `/v1/audio/speech` and the disconnect release asked for as their own PR. Why: `examples/livekit_kokoro.py` already points LiveKit's TTS plugin at M*, and M* refuses tool calls.
- 2026-10-23 #379: the sliding-window attention split into its own PR (now #855). The Rust runtime's new-token fix stays in #379 with a follower test.

## Standards

- 2026-10-27: a test that guards nothing or has a wrong oracle is a finding on that test.
- 2026-10-27: no ask to split a PR unless the split changes what lands.
- 2026-10-26: report the structural and non-blocking points a review finds, each marked as a follow-up or not a merge condition, beside the merge conditions.
- 2026-10-25: answer a question from the code and configs before asking the author.
- 2026-10-25: ask for a GPU run only when nothing else can show the point. CPU tests that already cover a case count.
- 2026-10-25: don't hand the author's own "known gaps" back to them. Raise one only with a run or a mechanism that shows it matters.
- 2026-10-25: a PR that changes a default path is re-measured at the head under review. A description table from an earlier head doesn't settle it on its own.
- 2026-10-24: test at realistic sizes. A stress config no deployment runs (a 48-page KV pool) is not evidence for or against a PR.
- 2026-10-24: name a bug as a bug in the first words, then the mechanism, then the fix.
- 2026-10-24: say each fact once, and name the cause in one plain sentence.
- 2026-10-24: ratios are 1x or more, with an arrow for the direction (1.13x ↑, 9.9x ↓, 1.0x when level).
- 2026-10-24: name PRs by number and commits as "this head" or "main" rather than a hash where possible.
- 2026-10-23: recheck every carried fact against the code before it's reported.
