# Qwen3-TTS: the codec beside the Talker

The Talker's decode frame was close to the competitors'; the gap came from decode waiting behind codec windows on one worker loop and one stream. This note records what recovered it and the form that shipped. All numbers: 2x B300 (sm_103), `-O2`, SHM arena, tts1's protocol (200 sentences, closed loop, warmup 5 waves), one session at a time, stacked on the kernels branch.

## Result

Paired median against base over 5 rounds, range in brackets. 3000 requests per arm, 0 errors, 0 runaways.

| arm | c=1 | c=8 | c=32 |
|---|---|---|---|
| side step (this branch) | -1.3% [-11, +7] | +15.0% [-2, +18] | +4.7% [-4, +15] |
| split config with `mps: true` | -3.7% [-9, +2] | +11.9% [-5, +24] | +14.9% [-7, +24] |

The decode stall across a codec step at c=1 (nsys, p50) goes from 7.7 ms on base to 2.7 ms here and 1.5 ms on split.

The two don't compete. The side step is what a colocated deployment gets, with no daemon and no second process. Split with `mps: true` is a one-line config change on a feature that already exists (`mstar/conductor/mps.py`) and stays the better choice at c=32. On split the lane does not exist, since the Codec is alone on its worker.

## Why a second stream alone does nothing

The loop is host-bound (`event_sync` p50 0.05-0.4 ms on this base), so a step costs a pass of the main loop whatever stream it runs on. Putting the codec's kernels on a side stream with the chain unchanged left the step mix and `iter_total` where they were (decode 65% and 57% of iterations at c=8 and c=32, before and after). The gain comes from taking the codec out of the chain, which raises the decode share to 90% and 78% and takes wall time per decode step from 7.4 to 7.0 ms at c=8. Split moves the same host work behind a second GIL, which is why it wins at c=32, where the main thread is the limit: 6.8 ms of main-thread CPU per decode step against 7.8 here and 8.3 on base.

## The form

- **Declaration.** `NodeSubmodule.side_stream_walks`, on `CodecSubmodule` for `codec_chunk`. Nothing else in the model changes.
- **Which keys run there.** `EngineManager.side_stream_keys`: declared walks, on a device with streams, beside another node. A parallel node stays on the chain, because when a step lands is per rank and a lockstep instance has to agree (invariant 11). `_refuse_unsuitable_side_walks` fails startup for a walk that isn't one node's alone, or a node with resources: a side step is neither speculated nor planned ahead, so there is nothing to admit and nothing for the chain to wait on.
- **Streams and ordering.** `SideStepLane` owns the stream and the one step in flight. Each step fences behind the main stream, because the main thread clones and stacks its inputs there. Postprocess waits for the step to land, because it frees those inputs first, and they are main-stream allocations a main-stream step could reuse.
- **Capture.** `Engine._capture_stream_for` gives the node a capture stream of its own. Graphs captured on one stream must not overlap. The chain's never do, but a side step overlaps all of them.
- **Scheduler.** `MicroScheduler.side_keys` keeps the walk out of the untargeted scan, the fairness peek and the backlog; the lane asks for it by target. No graph runtime change, so Python and Rust stay in step (invariant 5): both serve `get_ready_nodes` as before and the filter sits above them.
- **Postprocess order.** A side step lands between two chain steps, so it skips `clear_pending_loop_stops`. Those stops belong to the chain's next postprocess; clearing them routes the speculative step past EOS, and the audio gains a frame.
- **Rollback.** A side step fails alone: a raise fails its rows, not the chain's, and a chain error leaves it to finish. It has no speculation to clear and no plan to reset.

## Two things that went wrong, and what caught them

1. **The pending stop.** A seeded c=1 compare showed 3-6 of 200 WAVs one frame longer, every common sample identical. Caught by the byte compare.
2. **Prefill beside a codec window.** With both graphs captured on torch's one default stream, about 1 request in 1700 at c=8 and 1 in 12000 at c=32 read its sentence correctly and then never reached EOS (top minus EOS logit stayed above 4), running up to the 8192-frame cap. Seeded c=1 stayed 200/200 byte-identical, since at c=1 a prefill never overlaps a codec step, and WER stayed in band, since one babbling file in 200 is a rounding error. It showed up as a single request holding a c=8 cell open for 30 s, halving its audio-s/s.

| soak at c=8 | requests | over 60 s of audio | over 3x its sentence's median |
|---|---|---|---|
| base | 5000 | 0 | 1 |
| lane, kernels on the main stream | 5000 | 0 | 1 |
| side stream, chain unchanged | 5000 | 0 | 0 |
| lane + side stream | 5000 | 3 | 11 |
| same, plan cache off | 8000 | 6 | 14 |
| same, pre-plan off | 8000 | 4 | 12 |
| same, prefill waits for the side step | 10000 | 0 | 3 |
| same, codec on its own capture stream | 10000 | 0 | 4 |
| this branch | 10000 | 0 | 0 |

What the two graphs share through the capture stream is a hypothesis: a cuBLAS workspace would fit, since prefill and codec are both captured with `compile=False` while decode is compiled, but a harness of overlapped GEMMs at decode sizes never reproduced it (0 of 15000, same-stream and own-stream alike). The fix is proven by the soak; the mechanism is not.

## Invariants

1 (the model declares, the engine and worker execute). 4 (only `CodecSubmodule` declares; every other node takes three cheap checks per pass). 5 (no runtime or wire change). 7 (worked through against the stage order: the chain's stages are untouched and the lane sits at the top of a pass and after N's postprocess). 8 (the config decides placement; the declaration does nothing on split). 11 (parallel nodes stay on the chain). 14 (a step's rids are in `_in_flight_rids` until it finishes, so a teardown waits).

## Tests

`test/modular/test_side_steps.py`: stream choice, the fence's order, one step in flight, when a step may postprocess, input lifetime, the pending-stop case through the real `_postprocess_batch`, failure isolation both ways, the scheduler filter including the backlog, capture streams, startup refusal. Any change here also needs a c=8 soak with a control arm: `seeded c=1` and WER cannot see the failure above.

## Next

- `mps: true` in `configs/qwen3tts_1p7b_split.yaml`. First fix a bug in it: on any GPU but 0 the workers exit with `No CUDA GPUs are available`, reproduced twice with `--gpus 1`. MPS clients see the daemon's GPU as device 0, and a worker keeps `CUDA_VISIBLE_DEVICES=1`. Every number above ran on GPU 0, so nothing else here is affected.
- Prefill mixed into decode steps: a combined `talker_mixed` walk with a packed capture ran on this model as a scratch, cutting prefill's own steps to zero at c=8 and c=32, with TTFA p50 43 to 34 ms at c=8 and 64 to 60 ms at c=32 (p95 132 to 108), and +13% and +11% audio-s/s against the same 3 rounds' base. Worth its own PR.
- Fish S2-Pro, Qwen3-Omni Code2Wav and Orpheus SNAC have the same layout. Each needs its own soak before declaring.
- The GPU thread still runs the side step's host work, so N+1's launch waits for it: only 22% of a codec span overlaps decode graphs, against 65% on split. A thread of its own did not help (+7% at c=8, 0% at c=32 over 7 rounds, against +6% and 0%), because the work holds the GIL.
