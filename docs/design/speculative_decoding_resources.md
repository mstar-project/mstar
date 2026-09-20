# Speculative decoding, from the resource angle

Status: design sketch. Nothing here is implemented. Written against Qwen3.5,
whose checkpoints already ship an MTP head, but the resource contract is meant
to be model-agnostic.

## 1. What the checkpoint gives us

Qwen3.5 ships a complete multi-token-prediction module that nothing currently
loads — `transformers` drops it too (`modeling_qwen3_5.py`:
`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`). For the 4B, 15 tensors:

```
mtp.pre_fc_norm_hidden.weight      (2560,)
mtp.pre_fc_norm_embedding.weight   (2560,)
mtp.fc.weight                      (2560, 5120)     <- 2 * hidden in
mtp.layers.0.{self_attn,mlp,norms} one decoder layer
mtp.norm.weight                    (2560,)
```

`fc` taking `2 * hidden` is the tell: it consumes a concatenation, which is the
DeepSeek-V3 MTP shape. Per position *t*, given the target's last hidden `h_t`
and the sampled `x_{t+1}`:

```
h' = fc(concat(pre_fc_norm_hidden(h_t), pre_fc_norm_embedding(embed(x_{t+1}))))
h' = mtp.layers.0(h')
logits = lm_head(mtp.norm(h'))          -> distribution over x_{t+2}
```

Chaining `h'` back in gives proposals beyond *t+2*.

Two properties that drive everything below:

- **`mtp.layers.0` is full-attention, not gated-delta-net.** It has
  `q/k/v/o_proj` and `q_norm`/`k_norm`, no `conv1d`/`A_log`/`dt_bias`. Its
  geometry matches the main stack exactly (`q_proj` is `(8192, 2560)` = 16
  heads x 256 x 2, so it carries the same output gate; `k_proj` is 4 KV heads).
  So the draft head needs **one extra layer of KV and no recurrent state**, and
  it shards under TP exactly like `Qwen3_5Attention`.
- **It shares the embedding and the head.** `mtp_use_dedicated_embeddings` is
  `false` and there is no `lm_head` under `mtp.*`.

## 2. The invariant everything rests on

Every resource plans host-side today. `GDNPrefillWrapper.plan(spans: list[int])`
builds `cu_seqlens` on the host and copies H2D; `_plan_conv` derives the Triton
grid from host ints; `PositionStep(advance=...)` takes host ints. The acceptance
count is produced on device.

So the rule is:

> **Shapes are host-side, fixed, and sized for the maximum. Values are
> device-side and carry the actual.**

A step always proposes *k*, so the token count per request is constant and
preplannable. What varies — how far each cache advanced, which positions —
lives in device-resident index tensors. Over-provisioned work must no-op rather
than corrupt; the GDN path already does exactly this (`batch_ptr` filled with
`PAD_SLOT_ID`, kernel returns early at `kernels.py:118`), and
`chunk_gated_delta_rule` already takes `cu_seqlens` as a device tensor.

The moment one resource wants an exact host-side length at plan time, the
pipeline serialises.

## 3. The two-phase resource contract

Add to the resource interface, alongside `plan` / `commit`:

- **`reserve(step, n_draft)`** — plan-time, host-side, deterministic. Claims
  whatever a *k+1*-token speculative step could need. Part of `plan`.
- **`commit_draft(accepted)`** — after the acceptance count is known. Promotes
  what was accepted and releases the rest.

`reserve` means genuinely different things per resource, and the interface
should admit that rather than forcing one shape.

### KV cache

- **reserve**: pages for *k+1* tokens. Over-allocate so no allocation decision
  depends on acceptance.
- **commit**: advance the cursor by `accepted`. Rejected positions are simply
  never read again — the cache is positionally addressed and append-only, so
  rollback is not advancing.
- The MTP layer's KV rides along as one more layer of the same pool (same
  geometry), rather than a second pool.

### Recurrent state — delta-net

This is the one that shapes the design.

The chunked path already separates read from write: `GDNPrefillWrapper.run`
gathers `initial = index_select(state, 0, slots)`, calls
`chunk_gated_delta_rule(initial_state=initial, output_final_state=True)` which
returns a **fresh** `final`, and only then scatters with
`state.index_copy_(0, slots, final)`. The destructive path is the *decode*
kernel (`gated_delta_rule_decode_pretranspose(initial_state=state,
initial_state_indices=slots)`), which updates the pool in place — but
speculative verify is a multi-token step (`is_decode = all(s == 1 for s in
spans)` is false for *k+1* > 1), so verify takes the chunked kernel. **The
in-place path is the one speculation replaces.**

That does *not* make reserve a no-op, because of *when* the scatter happens.
Each of the 24 GDN layers scatters during its own forward, long before the LM
head produces the logits that decide acceptance. Deferring means holding 24
layers' `final` live — for the 4B that is the same 48 MiB/row, moved from the
pool into activation memory. So:

- **reserve**: **one** draft slot per request. Not *k+1* — the chunked kernel
  returns one `final` per *segment*, the state after all *k+1* tokens, not a
  state per prefix length. *k+1* slots buys nothing you can fill without *k+1*
  chained calls.
- **commit**:
  - all accepted -> promote the draft, free the old committed slot.
  - partially accepted -> recompute over the *j* accepted tokens from the
    committed slot, which survived precisely because the draft went elsewhere.
    Then promote.
  - none accepted -> free the draft.

Promotion is a **free-list swap, not a copy**. The pool allocates by index
(`SlotState(index=self._free.pop())`) and addresses through a device
`slot_indices` tensor, so promoting is repointing the request's label at the
draft index and returning the old one. This is deliberately *not*
`_apply_fork`, which is a real copy — its docstring says why: "the state is
mutated in place, so two labels cannot share it."

### Recurrent state — conv window

Comes along free. A slot spans both blocks (`for tensor in
self._blocks.values(): tensor[:, slot.index]`), so a draft slot isolates the
conv window too. No `cache_indices` scratch redirection needed.

Worth noting the size asymmetry for anyone costing this: per slot on the 4B,
the delta-net state is 48 MiB and the conv window 1.12 MiB.

### Position

`PositionStep(advance=...)` already accepts an explicit per-request advance —
Qwen3-Omni uses it because a vision span's 3D MRoPE covers fewer positions than
it has tokens. Speculation is the same shape of problem: advance by `accepted`,
not by the span. The mechanism exists.

### Sampler

Verification is rejection sampling against the draft distribution, so the
sampler needs both the target and draft probabilities for the proposed tokens.
Greedy collapses to an equality test. This is the one resource that needs new
*math* rather than new bookkeeping.

## 4. Pool sizing

Two slots per concurrent request instead of one: ~98 MiB per request on the 4B,
~72 MiB on the 27B at TP=4.

`max_slots` is currently floored by capture, not concurrency — the runner holds
a dummy row per `(config, slot)` for the whole pass, so the floor is
`2x32 + 2x8 = 80`, hence `max_slots: 96`. Doubling rows would take that to 160.

That floor should be removed independently: `DummyRowPool.reset(dummy_rids)`
after each capture passes `free=False`, and only `release_all()` frees. The
residency buys nothing — `test_dummy_row_pages.py` establishes that a replay
pads with **zero-length** rows, so none of that storage is read again. Freeing
between captures makes the floor the largest single bucket (~32) instead of the
sum, and decouples it from *how many* capture configs exist.

One implementation note: `reset(free=True)` releases the slot but leaves the rid
in `DummyRowPool._held`, and `ensure()` only ingests names beyond `len(held)` —
so freeing between captures must also drop those rids from `_held`, or the next
`ensure` hands back names whose slots were freed and the plan addresses
`NO_SLOT`. Harmless today because the only `free=True` caller is `release_all()`,
after the last `ensure`.

## 5. Where the D2H lands

The acceptance count is device-side and the host needs it eventually. This is
structural, not incidental.

It is not, however, on the critical path. Shapes are max-sized (§2), so
*planning* step N+1 does not need it. What needs it is host-side bookkeeping —
scheduler decisions, freeing requests, token accounting — all of which tolerate
a step or two of lag. The engine already eats a per-step D2H of exactly this
kind: `check_stop` calls `.item()` on the sampled token, deliberately moved off
the GPU thread (`postprocess`'s comment says so). The acceptance count can ride
the same asynchronous path.

What speculation *does* serialise is a single request's chain: step N+1's input
tokens depend on N's acceptance, so one request cannot have two speculative
verifies in flight. Across requests the pipeline still overlaps, so the async
postprocess worker keeps earning its keep — it just stops helping *within* one
request's chain. Preplanning for N+1 stays valid because the shape is fixed.

## 6. Graph-walk placement

Two candidate shapes:

**(A) One node.** `LLM` runs verify and draft in one forward, with two labels —
`main` for the target stream, `draft` for the MTP head.

**(B) Two nodes.** `Loop(Sequential([mtp, LLM]))`, the draft head its own node.

**Recommendation: (A) for MTP.** Four reasons, in rough order of weight:

1. **`h_t` never leaves the node.** MTP consumes the target's last hidden state,
   pre-`lm_head`. Across a node boundary that becomes a graph edge carrying
   `[k+1, hidden]` every step — materialised, registered with the tensor
   manager, routed, unpersisted. Inside one node it is a local variable.
2. **Shared weights.** MTP shares `embed_tokens` and `lm_head`. Two nodes are
   two submodules, so either the embedding and head are duplicated (1.2 GiB on
   the 4B) or two submodules alias one `nn.Module`, which node placement does
   not guarantee is co-located.
3. **Labels already express this.** `SubmoduleStep` segments carry a label and
   the resources are cursored per label — BAGEL uses it for CFG. "Two streams
   in one node" is the mechanism's existing purpose.
4. **Capture.** `get_cuda_graph_configs` is per-submodule, so two nodes mean two
   capture sets and a higher slot floor (§4). One node captures draft and verify
   together.

Ordering within the step also argues for it: the natural loop body is
verify(proposals) -> accept -> draft-from-accepted-`h`, and both halves want
`h` from the same forward.

**(B) is the right shape for a *general* draft model** — EAGLE with its own
weights, or a genuinely separate small model — where there is no weight sharing
and the drafter could even sit on another rank. If we expect to support that,
the node boundary is the right abstraction and MTP becomes the degenerate
co-located case. Worth deciding which of those we are building before writing
the interface, because it is the one choice here that is expensive to reverse.

The draft chain itself (*k* autoregressive steps over one layer) should be an
unrolled loop **inside** the node's forward, not a graph `Loop`. It is
fixed-length and single-layer; a nested graph loop would buy nothing and cost a
capture bucket per iteration.

## 7. Open questions

- **Measure acceptance rate first.** The rollback cost on the recurrent layers
  is paid every speculative step whether or not tokens are accepted, and 24 of
  the 4B's 32 layers (48 of the 27B's 64) are GDN. The MTP weights are already
  in the checkpoint, so this is a throwaway script against HF, no engine work.
  It decides whether any of the above is worth building.
- **Trees.** *b* branches means *b* draft slots — the recurrence cannot share
  state across branches the way tree attention shares a KV prefix. Linear pool
  cost on top of per-branch recompute probably keeps a 3:1 hybrid on chains.
- **Does the MTP layer's KV need rolling back separately?** It grows by the
  proposal count, not the accepted count, and its cursor may want to differ from
  the main stream's. Likely another label rather than another pool.
- **`mtp_num_hidden_layers` > 1** on some future size would make the draft head
  deep enough that its own capture config starts to make sense, weakening (A).
