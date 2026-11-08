# Performance checklist

What the performance pass (SKILL.md §4.3) looks for. Leads, not rules: AGENTS.md's "Performance
ideas from past work" and patterns C4, C6 and C15 are further leads, and AGENTS.md invariants 6, 7
and 10 are the citable rules.

## Where time goes

On each hot path the PR adds or changes (the per-step forward, `prepare_inputs`, the sampler, the
scheduler tick, the data worker):

- host syncs (`.item()`, `.tolist()`, `.cpu()`, a blocking copy, a graph replay waiting on the host)
  outside `check_stop`;
- work that falls out of a captured graph or `torch.compile`: eager fallback, a recompile on a new
  shape, a graph break;
- extra memory traffic: copies, `contiguous()`, fp32 upcasts, re-reading weights;
- per-step work that could be done once at load or capture time;
- launch-bound loops of tiny kernels;
- Python on the critical path per token or step, per-request objects built every step (invariant 6);
- memory that shrinks the KV pool or the batch size;
- missed overlap: copies and compute on one stream, the conductor waiting on the engine;
- an algorithm that grows faster than it needs to (quadratic buffers, per-request re-planning).

## Standard techniques first

A missing one on a default path is a finding (SKILL.md §4.3 gives the priority). Name the shared
component that already does it (the engine's KV pool, `mstar/model/components/`, the ragged
attention resources) when the PR rolls its own or skips it.

- **Autoregressive models:** a KV cache, paged through the engine's pool rather than a private
  tensor per request; prefix caching where prompts repeat; continuous batching (not batch-of-one,
  not waiting for a full batch); captured decode; the sampler on the GPU; incremental
  detokenization.
- **Diffusion and flow models:** conditioning (text-encoder output, the prompt's cross-attention
  K/V, timestep and RoPE tables) computed once per request, not every step; CFG's two branches in
  one batched forward; requests at the same step batched together; the VAE decoded once at the end,
  tiled only where memory needs it.
- **Everything:** fused attention (FlashInfer, FlashAttention) on the hot path; shared fused norms,
  RoPE and activations; bf16 where the reference is bf16; weights loaded once and shared; encoder
  outputs cached when several steps read them; pinned, non-blocking H2D copies; a fast tokenizer;
  request-independent work hoisted to load or capture time.
- **Big models:** TP, or a stated reason not to, when weights don't fit comfortably on one GPU.

## The shape, not just the lines

These often outweigh line-level findings. The expected speedup is usually larger and less certain
and the difficulty higher; say so in the finding's fields and name the design you'd compare against.

- **Node split and placement:** what crosses a GPU or process boundary and how big it is (#375 fused
  the text encoder with the connectors because its hidden states were about 385 MB per prompt,
  against about 12 MB after the connectors). Does placement leave a GPU idle while another is the
  bottleneck?
- **Interface granularity:** per-request or batched calls, and APIs returning host values (Python
  lists, `.item()`) that force a sync at every call.
- **Where state lives and who owns it:** host or device, copies forced by an ownership boundary,
  caches that never evict or are rebuilt every step.
- **Shape policy:** padding and buckets against the number of captures and compiles; shapes that
  change every step.
- **Fit with the fast paths:** capture, `torch.compile`, TP, continuous batching, prefix sharing, the
  SHM arena. Does it quietly opt out of one?
- **Scheduling and overlap:** can stages of different requests overlap? Can the CPU prepare step N+1
  while step N runs?
- **Static reservations:** workspace or pool sizes fixed up front that cut KV or batch capacity
  (#375's ~6 GiB of FlashInfer workspace).
- **Kernel and precision choices** baked into the design: SDPA vs FlashInfer, fp32 where bf16
  matches the reference, a fused path that can't be used.
- **Outside the GPU step:** serialization between processes, tokenization and media decode/encode on
  the critical path, Python in the conductor loop, startup (weight loading, compile and capture).
- **Will it scale:** the next likely config (TP2 → TP4, longer video, bigger batch, a second model
  reusing it). Linear, quadratic, or a cliff?
