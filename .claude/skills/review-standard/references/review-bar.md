# The review bar

Gaurav's reviews of #461, #465 and #557 (Oct 19 to 24 2026) are the bar for every review. This file outranks `../SKILL.md` on rigor, and `decisions.md` still wins on a specific call.

As main's primitives matter more, how a change sits in the architecture is the first thing a review judges. Performance and testing are separate axes, each judged on its own. A review that is right on one axis and silent on another isn't finished.

This pass runs no code. Where his bar says "run" or "measure", the pass traces the same point through the code, derives it, or takes it from the PR's own numbers or the live test's in `pr-conversation.json`, and a run that only a GPU can do belongs to the live test.

## Rigor

1. **The change in its whole context.** Every changed file in full in `pr-head/`, not the diff. The callers and callees of every changed function (search both trees for them). The other files of the module that share its state. Every PR in its stack, read as one design before any one PR is reviewed, as far as the inputs carry them. The open PRs that touch the same functions, as far as the description, the threads, `decisions.md` and the map name them.

2. **Architecture.** Does the change use or extend the primitive that owns its concern, or build beside it? Run the mstar-way checks, read the map entry of every mechanism it touches (Extend via, Breaks if), and check the rulings. Say what the next model, or the next change to this module, has to know or touch because of it. Look for a second way to do something main already does, tuned constants that decide behavior, hand-kept lists, options nothing uses, silent fallbacks, graph structure given up to dodge a runtime cost, and state with no owner on some exit path. A finding names the existing mechanism and how the change would sit on it:

   > Every arena producer already holds its namespace lock, taken in `SharedMemoryCommunicationManager.__init__`. If segments were named under the namespace, one lock and one sweep would cover both, and `create_locked`, its retry and the rustc 1.89 floor would go. (#552)

   > Holding new requests behind a pending reload is admission control. It belongs with #307, not in a readiness check. (#461)

3. **Performance.** What the change costs per step and per request, against what it has to cost: derive the work it must do and the bytes it must move, then compare with what was measured. A change to a default path is measured at the head against its base on stock configs, host time beside device time, and on a workload it should leave alone; in this pass that measurement is the live test's, and the finding names the run. An unsettled regression is reported with its numbers or settled by analysis, never left out.

4. **Testing and claims.** Trace the new tests against main's code and count the ones that would fail for the bug apart from the ones that would fail only for a missing name. Check the oracle. Give the test-to-code ratio when it's out of proportion. Check every claim in the description against the code and against main:

   > The description says the request that raised "kept its concurrency slot forever". On main, the request timeout (600 s by default) aborts it and frees the slot. (#465)

   > I ran the new tests on main: 15 of 17 fail, 13 on the #464 bug and 2 only on the missing `_send_for_teardown`. (#465)

5. **Every finding carries its proof**, in the same sentence or the next: a derivation, the chain of lines that does it, a test the PR already has (read, not run), or a number from the PR or the live test. His reviews used runs too:

   > In my run on current main, with Orpheus's LLM and SNAC on two workers, I closed 600 streams after 3 chunks and resubmitted each under the same id. The SNAC worker ended with 83 `INPUT_SIGNALS` parked for good. (#557)

   This pass states the chain of lines instead, and names that run as the way to settle it.

6. **What gets reported.** A finding about the PR's code or its defaults is reported, with its fix. It is never softened to a nit or dropped because it seemed strict. A point that changes neither what lands nor what the next change does is dropped, and so is one a test already guards. On Oct 26 #766 went out with both of its findings dropped, a 1 ms retry poll and a test-only wrap; both were real.

   A change to a default path with no measurement at its head is itself a finding (hard rule R7), whose fix names the run the live test should make.


What the Oct 26 duplex drafts missed:

- #770's notes held two findings about the shipped default: the cap admits 32 live calls while 16 keep real time, and the late-packet A/B at 32 calls was unsettled. With their starts spread over 2 s, 32 calls fell behind on both builds, so the analysis settled the A/B and pointed at the cap.
- #770's notes held the reason the codec moved into the talker's step: a node of its own cost 10 ms per tick. That is the worker's cost per iteration, and the next streaming model meets the same choice, so the review says it.
- #595's critique flagged `BATCHED_CHUNK_S` under R4 (its comment: "One second is a judgement, not a measured crossover.") and the session overrode it.
- #770's draft asked to share the iSTFT helper because a change to one copy would make the stream differ, but `test_decode_stream_is_the_full_decode` already fails if it does. The reason that holds is that every change to the inverse STFT has to be made twice.
- None of the six read the stack as one design, and five had no measurement at the head for a default-path change.

## Reference reviews

Posted text, verbatim. Read them as examples of good catches: each names what breaks, the existing mechanism the change should sit on, and the smallest structural fix. The verdict words in them (Request changes) are his posting layer, which this pass doesn't produce.

**#461, Request changes (Oct 19 2026)**

> The reload rules in the last three commits switch on and off with two one-second wall-clock windows, so they need to be replaced (see the comment on `_APPENDING_S`). Your reply to the other review says the stall reproduced only with the CPU loaded, because fewer decode steps fit in the 1 s window. Whether requests finish shouldn't depend on that.
>
> The victim order is a real fix for #460, but on its own it stalled 6 of 7 runs against 0 of 28 on main in your first reproduction. It should land on top of a reload rule that doesn't depend on step time.
>
> Holding new requests behind a pending reload is admission control. It belongs with #307, not in a readiness check.

Its inline comment on `_APPENDING_S` ran the author's own replays with the fake clock moved 0.01 s to 3 s per pass, gave the pass/fail table (both fail from 1.1 s), and ended:

> I'd count growth in the node's steps since the last scan instead of in seconds.

**#465, Request changes (Oct 19 2026)**

> The per-message isolation in `run`, with `_fail_after_error` and `completion_sent`, fixes #464, and I'd keep it as it is.
>
> The teardown resend queue should go (see the inline comment on `_send_for_teardown`). With `_send_for_teardown` as a catch-and-log, a `DRAIN_REQUEST` that fails to send still has its drain record. The record's TTL then finishes the teardown and sends the client its error. This also removes the hand-picked backoff and the 120 s deadline, which now decide whether a `REMOVE_REQUEST` is ever delivered.
>
> The description says the request that raised "kept its concurrency slot forever". On main, the request timeout (600 s by default) aborts it and frees the slot.
>
> On tests: they add 519 lines for 202 lines of code. About half of them model a dead socket as `communicator.send` raising `OSError`, which neither transport does for a dead peer, so those can go with the queue.
>
> I ran the new tests on main: 15 of 17 fail, 13 on the #464 bug and 2 only on the missing `_send_for_teardown`.

Its inline comment on `conductor.py`:

> A send to a dead peer doesn't raise on either transport: in my runs it hung in `send` once libzmq's 1000-message high-water mark filled, and the only raise I could get was the Rust send's EINTR on SIGINT, at shutdown. So I'd keep this as a catch-and-log and drop the resend queue and its backoff.

**#557, Request changes (Oct 19 2026)**

> The leak this PR targets is real. In my run on current main, with Orpheus's LLM and SNAC on two workers, I closed 600 streams after 3 chunks and resubmitted each under the same id. The SNAC worker ended with 83 `INPUT_SIGNALS` parked for good.
>
> Since #519 is now in this branch's base, each submission gets its own key, so the same key never reaches a worker twice. That makes the id-reuse half of this PR unnecessary (see the inline comment on the `discard`).
>
> I'd like the PR cut down to two things: dropping late messages for a removed request, and handling the conductor's REMOVE in `_remove_request` (see the other inline comment). The description should then be rewritten against current main.

Its inline comment on `worker.py`:

> This adds a second path for a REMOVE whose id has no handle, next to `_remove_request`'s no-handle branch, and it skips that branch's `scheduler.clear_wire_rid`. I'd send the conductor's REMOVE through `_remove_request` instead, with `_mark_removed` in the no-handle branch before the follower guard.

## Further examples from his method

- **The ask names the layer and the interface.** The strongest architectural asks name the layer a mechanism belongs in and the interface it should have there (DSA as a resource over a paged KV cache on #387, an `AllReduce` base class on #386, `DeviceCapability` and `Resource.can_capture()` on #262). When the abstraction already exists on main, the ask names the file, the interface and where to register: "see mstar/distributed/all_reduce.py ... implement the `AllReduce` interface and add the backend to `AllReduce.new`" (#379, Oct 24), and the author did exactly that the same night.
- **A back-of-envelope first.** On #379, Command A+ reads about 50 GB of bf16 weights per decode token (about 25B active parameters), so 8 H100s bottom out near 1.9 ms a token, and the measured 5.8 ms was a third of that. On #376, collectives were 38 ms of a 285 ms SP2 step, which ruled out custom collective kernels as the next lever before anyone wrote one. On #419 a requested run was for a polling cost of about 0.1 ms on a step of several milliseconds, so the question was whether a slowdown was expected at all.
- **A trace claim checked against the trace.** The TP4 prefix-caching numbers on #379 couldn't be cache hits, since `kv/manager.py enable_prefix_cache` turned caching off above one rank, and #405 was what added it at TP > 1.
- **A rebase-only finding that the rebase won't show.** On #379, #262 had renamed the graph-capture hook, so after a rebase nothing would call the PR's `get_cuda_graph_configs` and decode would run eager. That was a finding on the hook; the conflicts, the import fix and the tests CI would fail were not.
- **A pre-existing bug the PR's result depends on is the PR's to fix.** On #800 a fresh install loaded no SigLIP weights, so the PR's LeRobot match held only on an older transformers, and `_stage` broadcasting a smaller input was how the one-camera bug stayed silent. Both were asked for in the PR.
- **A reply that deflects is wrong until checked.** On #808 "making the loop length follow `max_tokens` would be a change to the graph runtimes" was wrong: main already treats a loop's `max_iters` as a ceiling that each request's own stop ends early (Flux2, OmniVoice). Never call a change too big without checking how big it is.
- **A suggestion is weighed for the problem it points at.** On #836 a reviewer suggested Qwen3-Omni's `process_prompt` turn the prefix cache off for audio output. He was right that the conductor imported worker code for it, but taking the suggestion whole would have made correctness depend on every future model remembering to opt out, so the rule stayed in the conductor, derived from the graph.
