# Model shape map

A non-exhaustive catalogue of topologies already in the tree, for finding precedent once you have derived your own decomposition. It is not a taxonomy every model must fit — do not force a model into the least-wrong row.

Adapted from the map drafted in PR [#253](https://github.com/mstar-project/mstar/pull/253).

## Derive your design first

Answer these before looking at the table, and write the answers down:

1. Which compute stages can be independent graph nodes, and which are better fused for a larger CUDA graph?
2. Which walks and loops traverse those nodes?
3. Which tensors cross node, walk, iteration or partition boundaries?
4. Which persistent state needs engine admission, stable storage, capacity, dependency ordering or cleanup?
5. Which existing graph and resource primitives express each requirement?

Classify every persistent value from its *behaviour* before picking reference code. Engine lifecycle ownership is a constraint on the design, not a feature you inherit only if a similar model happens to use it.

## Patterns

| Target pattern | Inspect | State / resources | Eager starting point |
|---|---|---|---|
| AR backbone followed by a codec or output stage | `orpheus` | Backbone uses KV, attention, position, sampler; output streams to a separate stage | Match token and codec outputs one request at a time |
| Encoder-decoder with fixed source context | `whisper` | Cacheless encoder; decoder has separate self-KV plus cross-context resources, position, sampler | Verify encoder features, then the context write, then greedy decode |
| Audio tower feeding an AR language model | `higgs_audio` | Ragged/cacheless tower attention, then AR resources | Verify audio preprocessing and tower output before language decode |
| Two AR stages connected as a live stream | `qwen3_omni`, `qwen3_tts` | Each stage declares its own resource keys; async partitions and stream buffers connect them. qwen3-omni also fuses talker + code predictor into one node for capture size | Validate each stage directly, then one-token chunk routing |
| Conditional multimodal walks with CFG branches | `bagel` | Walk-specific branches share declared resources; step declarations encode forks and combined labels | Start with uncaptured single-request walks per modality |
| Diffusion or flow loop over stateless compute | `cosmos3`, `wan22` | Loop state normally rides graph edges or request state; declare attention resources only where planning actually needs them. These know their iteration count at ingestion, so they exit early in `prepare_inputs` for the async worker's extra step | Share initial latents with the reference and compare every denoise step |
| Vision-language-action policy | `pi05` | Vision encoder plus language KV resources and an action flow loop; no sampler when actions come from flow matching | Verify the vision/language prefix, then every action denoise step |
| World-model rollout with growing AR context | `vjepa2` | One-shot predictor may need no resource at all; action-conditioned rollout declares KV and attention for persistent history | Compare each rollout iteration and the loop-back signal |
| Fixed-horizon interactive world, overwrite-in-place history | `waypoint` | A custom engine-owned ring resource retaining several resident worlds; resident worlds and step batching are independent | Validate prime and rollout with capture disabled, including slot reuse and cleanup |

Borrow only the graph and resource patterns whose invariants match. Port the mathematics from the upstream implementation and the checkpoint config; do not inherit hardcoded dimensions, sampling defaults, state ownership or optimization settings from whichever model you copied the file layout from.

## When nothing matches

An unmatched topology is a normal porting case, not permission to bypass the engine:

1. Build the graph from your model's own stages, walks, loops and tensor dependencies using the graph primitives.
2. Fit persistent behaviour to existing resource kinds wherever their invariants match, even if no existing model combines them that way.
3. Compose references per stage and per resource; don't copy a model package because its modality is similar.
4. For genuinely new pooled state, add a resource kind through the existing spec, request-config, step and `Resource` interfaces.
5. Reduce batching and optimization scope to establish correctness — never lifecycle ownership.

Splitting nodes, walks, routed tensors, request state and resources more precisely is almost always the way out. A model-owned pool, allocator, capacity limit, stable shared buffer, dependency scheduler or cleanup lifecycle is not an acceptable temporary shortcut.
