# Model shape map

This is a non-exhaustive catalog of patterns worth inspecting, not a taxonomy
that every model must fit. Derive the target design before consulting the
examples. Do not force a model into the least-wrong row.

## Derive the target first

Answer these questions in the model port report:

1. Which compute stages can be independent graph nodes?
2. Which Walks and loops traverse those nodes?
3. Which tensors cross node, Walk, iteration, or partition boundaries?
4. Which persistent state requires engine admission, stable storage, capacity,
   dependency ordering, or cleanup?
5. Which checked-out graph and resource primitives express each requirement?

Classify every persistent value from its behavior before selecting reference
code. Engine lifecycle invariants are constraints on the design, not optional
features inherited only when a similar model happens to use them.

## Common patterns

| target pattern | examples to inspect | state/resource pattern | eager starting point |
|---|---|---|---|
| Autoregressive backbone followed by a codec or output stage | `orpheus` | Backbone uses KV, attention, position, and sampler resources; output streams to a separate stage. | Match token and codec outputs one request at a time. |
| Encoder-decoder with fixed source context | `whisper` | Encoder is cacheless; decoder has separate self-KV and cross-context resources plus position and sampler. | Verify encoder features, context write, then greedy decode. |
| Audio tower feeding an autoregressive language model | `higgs_audio` | Ragged/cacheless tower attention followed by autoregressive resources. | Verify audio preprocessing and tower output before language decode. |
| Two autoregressive stages connected as a live stream | `qwen3_omni`, `qwen3_tts` | Each stage declares its own resource keys; async partitions and stream buffers connect them. | Validate each stage directly, then one-token chunk routing. |
| Conditional multimodal Walks with CFG branches | `bagel` | Walk-specific branches share declared resources; step declarations encode forks and combined labels. | Start with uncaptured single-request Walks for each modality. |
| Diffusion or flow loop with stateless compute | `cosmos3`, `wan22` | Loop state normally travels on graph edges or request state; declare attention resources only where actual attention planning needs them. | Share initial latents with the reference and compare every denoise step. |
| Vision-language-action policy | `pi05` | Vision encoder plus language KV resources and an action flow loop; no sampler when actions come from flow matching. | Verify vision/language prefix, then every action denoise step. |
| World-model rollout with growing autoregressive context | `vjepa2` | One-shot predictor may need no resource; action-conditioned rollout declares KV and attention for persistent history. | Compare each rollout iteration and loop-back signal. |
| Fixed-horizon interactive world with overwrite-in-place history | Waypoint draft PR #251 | A custom engine-owned ring resource retains several resident worlds; resident worlds and step batching are independent. | Validate prime and rollout with capture disabled, including slot reuse and cleanup. |

Borrow only the graph or resource patterns whose invariants match the target.
Port mathematical details from the upstream implementation and checkpoint
config; do not inherit hardcoded dimensions, sampling defaults, state ownership,
or optimization settings from a reference model.

## When no pattern matches

An unmatched topology is a normal porting case. It is not permission to bypass
the engine shape:

1. Build the graph directly from the target's stages, Walks, loops, and tensor
   dependencies using the checked-out graph primitives.
2. Fit persistent behavior to existing resource kinds wherever their invariants
   match, even if no existing model combines them in the same way.
3. Compose references per stage and per resource; do not copy one model package
   merely because its modality is similar.
4. For genuinely new pooled state, add a resource kind through the existing
   spec, request-config, step, and `Resource` extension interfaces.
5. Keep the path eager and reduce batching or optimization scope when needed to
   establish correctness without weakening lifecycle ownership.

Never use a model-owned pool, allocator, capacity limit, stable shared buffer,
dependency scheduler, or cleanup lifecycle as a fallback or temporary shortcut.
Push the model decomposition toward the existing engine shape by splitting
nodes, Walks, routed tensors, request state, and resources more precisely.

`evidence-blocked` is valid only after a concrete required behavior cannot be
represented by existing resource kinds or by a new resource using the existing
extension interfaces. Show the attempted mapping and the exact missing
interface. Lack of a close reference, implementation effort, or loss of a
preferred optimization is not a blocker.

## Multi-reference ports

Use more than one reference when topology and state behavior differ by stage.
For example, an audio encoder plus two streaming autoregressive decoders can use
Higgs Audio for the tower, Whisper for fixed cross-context if present, and
Qwen3-Omni for partition routing. State the source and matching invariant of
each borrowed pattern in the design table.
