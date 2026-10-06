# Context parallelism (CP) for autoregressive nodes

Divide the token axis of one request across `cp` ranks of one instance. Then the KV cache per rank falls by `1/cp` past the head-shard limit. The prefill compute of a long prompt falls by `1/cp`. Each decode step reads `1/cp` of the resident KV per rank. The ranks of one CP group hold the same attention heads and different tokens.

The reference model is the Qwen3-Omni-30B-A3B thinker. It is a mixture-of-experts text decoder with 48 layers, 32 query heads, 4 KV heads and head dimension 128. It runs at TP2 today. Its 4 KV heads cap the head-shard of the KV cache at 4 ranks. CP is the path to a larger context per instance past that cap.

CP is opt-in per `node_groups` entry with `cp_size`. The default `cp_size: 1` keeps the current behavior byte for byte.

- **Status:** draft for review. No code is written. Pinned to M* commit `33baea08`.
- **Area:** `engine/resources/{kv,attn,position,sampler}`, `distributed`, `model/base`, `model/qwen3_omni`. No worker, conductor or Rust runtime change.
- **Related:** [chunked prefill plan](https://github.com/mstar-project/mstar/blob/chunked-prefill-2/docs/design/chunked_prefill_plan.md), issue #210 (prefix KV reuse), PR #198 (windowed generation), vLLM DCP and PCP, arXiv 2411.01783 (ring attention for 1M-token prefill).

Notation:

- `cp` = `cp_size`, `r` = CP rank, `P` = `page_size`, `I_tok` = `cp_interleave_tokens`.
- `L` = global token count of a stream, `T` = new tokens in this step.
- `Hq_l` and `Hkv_l` = head counts of this rank after the `tp*sp` split, `D` = head dimension.

Sections:

- Design: [0 Decision log](#0-decision-log) · [1 Rules and invariants](#1-design-rules-and-invariants) · [2 Mesh](#2-configuration-and-mesh) · [3 KV ownership](#3-kv-ownership) · [4 Attention](#4-attention) · [5 Model API](#5-model-facing-api) · [6 Followers and graphs](#6-followers-and-cuda-graphs)
- Review: [7 Interactions](#7-interactions-with-other-work) · [8 Checks](#8-correctness-checks) · [9 Caveats](#9-caveats) · [10 Plan](#10-implementation-plan-and-follow-ups) · [11 Acceptance](#11-benchmark-acceptance) · [12 Non-goals](#12-non-goals)

---

## 0. Decision log

| # | Decision | Where |
|---|---|---|
| 1 | `cp_size` is a third mesh axis next to `tp_size` and `sp_size`. `tp*sp` divides the heads and `cp` divides the tokens. "CP carved out of TP" is phase 2. | [§2](#2-configuration-and-mesh) |
| 2 | KV ownership is interleaved by token through one pure class, `CPLayout`, with the knob `cp_interleave_tokens` (default 1). Every rank reserves the per-rank maximum page count, so the allocator state stays symmetric. | [§3](#3-kv-ownership) |
| 3 | One merge primitive for all row types: all-gather `(O, LSE)` and merge by log-sum-exp. Decode is local paged attention plus this merge. | [§4.1](#41-lse-merge), [§4.2](#42-decode) |
| 4 | Prefill all-gathers the new K/V and divides the Q rows into zigzag chunks. A prefill over resident context selects pass-Q or pass-KV by a cost rule. Both ship in v1 behind one interface. Ring is a later implementation of the same interface. | [§4.3](#43-prefill-of-a-fresh-prompt), [§4.4](#44-prefill-over-resident-context), [§4.6](#46-pass-q-and-pass-kv-implementation) |
| 5 | A CP-capable model slices its inputs, declares the global span, and samples after rank 0 broadcasts the last hidden rows. Every rank samples from the same seed. | [§5](#5-model-facing-api) |
| 6 | The CP degree is fixed per instance and equal on the prefill and decode sides. No conductor or Rust change beyond the instance size `tp*sp*cp`. Five features are asserted off in v1. | [§7](#7-interactions-with-other-work), [§10](#10-implementation-plan-and-follow-ups) |

---

## 1. Design rules and invariants

Every change in sections 2 to 6 follows from one of these five rules.

1. **The instance is `cp*sp*tp` ranks in lockstep, in row-major order `[cp][sp][tp]`.** Rank 0 leads. All other ranks follow the current `ScheduleTPNode` FIFO. No layer above the attention resource learns a new concept.
2. **`tp*sp` divides the heads. `cp` divides the tokens.** Every `KVConfig.shard()` call and every head-degree call site takes `head_shard_size = tp*sp`, never the joint world size.
3. **`stored_len` stays global. `CPLayout` derives every per-rank number from it.** Token `g` lives on rank `(g // I_tok) % cp`. Every rank reserves the per-rank maximum page count.
4. **Decode runs local paged attention, then all-gathers `(O, LSE)` and merges.** All ranks hold the same heads, so no Q exchange and no reduce-scatter is necessary. The all-gather is captured in CUDA graphs, as the SP all-gather is.
5. **Prefill runs zigzag Q chunks against the all-gathered new K/V and writes the owned pages.** Rank 0 always owns the last chunk of each request. Prompts that are too short for `2*cp` page-sized chunks use replicated prefill.

Every PR keeps these invariants. The checks in [§8](#8-correctness-checks) test them.

- **I1. Partition.** For every stream, the owned-token sets of the `cp` ranks partition `[0, L)`, and `sum_r local_len_r(L) == L`.
- **I2. Determinism.** The chunk layout, the page ownership and every collective shape are pure functions of `(T, stored_len, cp, P, I_tok)`. Every rank knows all five values, so no rank can take a different branch.
- **I3. Symmetry.** The free-page count, the admission verdict and the CUDA-graph capture result are identical on every rank of the instance.
- **I4. Full output per rank.** Every attention call ends with the full merged `O` for the `Hq_l` heads of the rank. The row-parallel `o_proj` and its TP all-reduce do not change.
- **I5. Zero cost when off.** `cp_size: 1` produces plan tensors identical to `main`.

---

## 2. Configuration and mesh

### 2.1 Configuration

```yaml
node_groups:
  - node_names: [Thinker]
    ranks: [0, 1, 2, 3]
    tp_size: 2
    cp_size: 2        # TP groups {0,1},{2,3}; CP groups {0,2},{1,3}
```

`WorkerGraph` ([base.py](../../mstar/model/base.py)) gets `cp_size`, `_cp_ranks` and `_cp_comm_size`, in the same form as `_sp_ranks`. `tp_size` is still rewritten to the instance size. Thus `ShardingGroup`, the conductor and the Rust shard code do not change. `get_sharding_config` validates `cp_enabled_nodes` in the same way as `sp_enabled_nodes`. A CP node declares no `shard_dim` entries. Its signals are replicated, and each rank cuts its own slice locally.

### 2.2 Communication groups

```python
@dataclass
class JointGroups:
    tp_group: CommGroup
    sp_group: CommGroup
    cp_group: CommGroup                 # trivial when cp_size == 1

    @property
    def rank(self):                     # row-major [cp][sp][tp]; 0 iff instance leader
        return (self.cp_group.rank * self.sp_group.world_size
                + self.sp_group.rank) * self.tp_group.world_size + self.tp_group.rank

    @property
    def world_size(self): ...           # cp * sp * tp

    @property
    def head_shard_size(self):          # tp * sp: the degree attention HEADS are split by
        return self.tp_group.world_size * self.sp_group.world_size

    def broadcast(self, tensor, src=0): # tp, then sp, then cp
```

`GlobalParallelConfig` ([communication.py](../../mstar/distributed/communication.py)) builds the CP groups from `_cp_ranks`. It adds them to the sorted group schedule, so every rank creates the groups in the same order.

### 2.3 Call sites that change

| Site | Today | With CP |
|---|---|---|
| `KVConfig.shard()`, two call sites ([manager.py](../../mstar/engine/resources/kv/manager.py), [attn/base.py](../../mstar/engine/resources/attn/base.py)) | joint world size | `head_shard_size` |
| piecewise CUDA-graph configs ([engine.py](../../mstar/engine/engine.py)) | joint world size as head degree | `head_shard_size` |
| `agree_across_ranks`, `post_warmup_validate`, `_verify_tp_async_sched_agrees` | loop over `[tp_group, sp_group]` | loop over `[tp_group, sp_group, cp_group]` |
| `BaseSampler._broadcast_tokens` | TP group | joint group |

The `head_shard_size` line is the one line that must not be wrong. If `cp` is part of the head degree, `KVConfig.shard()` divides the heads by `cp` and drops heads with no error.

---

## 3. KV ownership

### 3.1 The CPLayout class

Today every rank holds every token, so the page of token `g` is `g // P` on every rank. Under CP a rank holds a subset of the tokens. The map from a global token to `(owner rank, local page, offset)` is then new state. The allocator, the plan builder, the position manager, the fork copy and the tests all need it. One pure class holds the arithmetic, so the sites cannot drift. It has no communication and no engine state, so its tests run on a CPU.

```python
# mstar/distributed/cp/layout.py
@dataclass(frozen=True)
class CPLayout:
    cp_size: int; cp_rank: int; page_size: int; interleave_tokens: int = 1   # I_tok; 1 = vLLM default

    def local_len(self, L: int) -> int          # tokens of this rank in [0, L); I_tok divides P or is a multiple of P
    def local_pages(self, L: int) -> int        # pages this rank uses for L tokens
    def reserve_pages(self, L: int) -> int      # ceil(max_r local_len(L, r) / P): the per-rank MAX
    def owner_of_token(self, g: int) -> int     # (g // I_tok) % cp
    def owned_mask(self, g: Tensor) -> Tensor
    def local_index(self, g: Tensor) -> Tensor  # (g // (I_tok*cp)) * I_tok + g % I_tok ; page = local_index // P
```

With `I_tok = 1`, token `g` lives on rank `g % cp`. With `I_tok = P`, logical page `p` lives on rank `p % cp`. A physical page then holds tokens from a super-page of `cp * I_tok` global tokens. The knob trades the write-scatter cost against empty shards for short prompts and against the prefix-cache hit granularity ([§4.5](#45-cost-model-and-levers)).

### 3.2 KV manager changes

All changes are in [manager.py](../../mstar/engine/resources/kv/manager.py).

| Function | Change |
|---|---|
| `_alloc` | Reserve `layout.reserve_pages(L)` per stream on every rank. Use the first `local_pages(L)`. At most one page per stream per rank is idle. |
| `_sequence_views` | The view length becomes `local_len(stored_len + span)`. `SequenceView` gets `global_length`, `global_to_compute` and `q_rows`. `to_compute` becomes the local Q-row count. |
| `_compute_plan_state` | Compute the global index `g` per token as today. Owned tokens map through `local_index(g)` to `(page, offset)`. Non-owned tokens map to `SINK_PAGE`. |
| `_decode_plan_state` | The owner of the new token is `owner_of_token(stored_len)`. Other ranks write to `SINK_PAGE`. |
| `_assert_symmetric_free_pages` | Add the CP group. The assertion stays strict. |
| `KVConfig` ([config.py](../../mstar/engine/resources/kv/config.py)) | Carries `cp_size`, `cp_rank`, `cp_interleave_tokens`. `max_num_pages` stays a per-rank knob. |

### 3.3 Position ids

`PositionManager` emits one contiguous range per view today. Under CP prefill the local rows are two chunks. Thus the manager turns `q_rows` into explicit positions through the current `pos_ids` override. The counters advance by the global span and stay symmetric. Positions are counters independent of the KV length. The thinker advances them by more than the span, so `stored_len` and the position counter must never derive from each other.

### 3.4 Symmetric reservation

Exact per-rank page counts differ by up to one page per stream per rank. The free-page totals then differ. A request can then fail allocation on one rank and pass on another. The allocation-failure handler and TP-async speculation both assume that every rank reaches the same verdict. A different verdict on one rank stops the next collective.

The alternative is exact counts plus an `all_reduce(min)` on admission. That adds a collective to the scheduler path. The reservation of the maximum costs one page per stream per rank and keeps every current symmetry assertion strict (I3).

### 3.5 Forks, offload and prefix cache

- Forks copy local pages to local pages. Both streams start at logical token 0, so ownership matches. Only the copy count changes, to the reserved page count.
- CPU offload is per rank. Victim selection uses a wall-clock LRU and is already uncoordinated across TP ranks. Under CP the reclaimable page counts also differ per rank. CP nodes require `cpu_offload_pages == 0` in v1.
- The prefix cache keys off the joint world size, which already includes `cp`. It stays off at world size > 1, as today.

---

## 4. Attention

### 4.1 LSE merge

Both FlashInfer wrappers return the log-sum-exp (LSE) on request. The merge is a pure function with no communication inside.

```python
# mstar/distributed/cp/merge.py
def merge_partial_attention(o: Tensor[N, B, H, D], lse: Tensor[N, B, H]) -> Tensor[B, H, D]:
    # lse <- -inf where NaN or +inf (empty shard / padding row)
    # m = max_n lse ; w_n = exp(lse_n - m) ; o = sum_n w_n * o_n / sum_n w_n

def cp_all_gather_merge(cp_group, o_local, lse_local) -> Tensor[B, H, D]:
    # all_gather both along a new leading dim into a static buffer, then merge_partial_attention
```

The LSE stays in fp32 from the kernel to the merge, in the natural-log base. The manager checks the log base of the kernel one time at startup against a small reference. Every CP rank ends with the full merged `O` for its `Hq_l` heads (I4).

### 4.2 Decode

`FlashInferManager.run` gets one branch when `cp_group.world_size > 1`.

1. Run local paged attention with `return_lse=True`.
2. A row with zero local KV gets one `SINK_PAGE` in the plan, so the kernel never sees an empty row. Such rows are streams with fewer than `cp` tokens and CUDA-graph padding rows. The plan sets their `lse` to `-inf`.
3. `cp_all_gather_merge` into a static buffer per bucket, because a captured graph bakes the address.

The all-gather moves `cp * B * Hq_l * (D + 1) * 2` bytes per layer per step. That is a few hundred KB. The CUDA graph replays the all-gather in the same way as the SP all-gather.

### 4.3 Prefill of a fresh prompt

This path serves `stored_len == 0`. Cut `[0, T)` into `2*cp` chunks. Rank `r` takes chunks `r` and `2cp-1-r`. Each chunk is one FlashInfer ragged row with `kv_len = chunk_end`, so the end-aligned causal mask is exact.

```python
# mstar/distributed/cp/prefill.py
def zigzag_chunks(T: int, cp: int) -> list[list[tuple[int, int]]]   # per rank, two (start, end)
def gather_new_kv(cp_group, k_loc, v_loc, local_rows, T) -> (k_full, v_full)
    # padded all_gather (gather_sequence), then one precomputed index restores global order
def ragged_rows(chunks) -> (qo_indptr, kv_indptr)
```

Per layer, inside `AttentionCallable.__call__`:

1. Gather the new K/V. Q, K and V travel in one `gather_sequence` call.
2. Write the full K/V. The plan from [§3.2](#32-kv-manager-changes) sends non-owned rows to `SINK_PAGE`.
3. Run the local Q chunks through a new ragged prefill wrapper over the full new K/V. The ragged path reads all new keys, so no cross-rank merge is necessary.
4. If `T < 2 * cp * P`, the step uses replicated prefill. Every rank computes all `T` rows and writes only its owned pages. PR 2 ships with this as the only prefill mode.

Zigzag balances the causal FLOPs to within one chunk. It puts the last chunk on rank 0, which [§5.3](#53-sample) uses.

### 4.4 Prefill over resident context

This path serves `stored_len > 0`. A prefill that extends a stream with resident tokens must also attend to the resident pages. Examples are the thinker's second text prefill after the modality walks, a multi-turn chat, and every chunk after the first one under chunked prefill. Each rank holds only a subset of the resident pages. A single causal kernel call over a partial page list infers the query position from the local key count. That position is wrong, and the kernel gives an incorrect result with no error.

The pass-Q method has three parts. vLLM DCP uses the same method. [§4.6](#46-pass-q-and-pass-kv-implementation) gives the pass-KV alternative and the rule that selects between them.

1. **Part A.** All-gather the new Q rows across CP in the same collective as K and V. Run non-causal paged attention of all `T` new Q rows against the local resident pages, with LSE. Every rank holds a different slice of the resident keys, so every rank must see every new query. A rank that runs only its local Q chunk here gives each query `1/cp` of the resident context.
2. **Part B.** Run causal ragged attention of the local Q chunk against the all-gathered new K/V, with LSE. This is the same call as in [§4.3](#43-prefill-of-a-fresh-prompt).
3. **Part C.** All-gather the Part A partials `(O_A, LSE_A)` across CP. Keep the rows of this rank's chunk. Merge them with the Part B partial.

Every rank runs Part A, also with no resident pages, and reports `lse = -inf` in that case. Thus the collective is unconditional per stream (I2). Decode is this path with one query and no Part B. vLLM gathers Q along heads, because its DCP ranks hold different Q heads. M* gathers Q along tokens, because CP ranks hold the same heads and different chunks.

### 4.5 Cost model and levers

Constants: thinker at TP2×CP2, `Hq_l = 16`, `Hkv_l = 2`, `D = 128`, bf16, 48 layers. One H100 node: about 15 µs per collective, about 350 GB/s per rank for a gather, about 2.5 TB/s HBM. Per layer:

| Term | Formula | Note |
|---|---|---|
| New K/V gather (pass-KV for new tokens) | `(cp-1)/cp * T * Hkv_l * D * 4` bytes | 64K tokens: 32 MB, about 105 µs. No overlap with compute. |
| Resident context, pass-Q | `2 * (cp-1)/cp * T * Hq_l * D * 2` bytes | Q out, partial `O` and LSE back. Grows with `T` only. |
| Resident context, pass-KV | `(cp-1)/cp * L * Hkv_l * D * 4` bytes | Grows with `L`. Needs a gather from the paged cache. |
| Attention FLOPs per rank | `2 * T^2 * Hq_l * D / cp` | 64K tokens: about 15 ms. The only term that CP divides. |
| MoE expert weight read | about 604 MB per rank at TP2 | About 0.25 ms per layer from about 128 local tokens. CP does not reduce it. |
| Decode all-gather | `cp * B * Hq_l * (D+1) * 2` bytes | Latency-bound, about 15 µs per layer. |

Pass-Q is cheaper than pass-KV when `T / L < Hkv_l / Hq_l`. For the thinker that ratio is `1/8`. A second turn or a modality-walk prefill adds a small `T` to a large `L`, so pass-Q wins. An MLA model has one latent KV head, so pass-KV always wins there.

| Prompt (thinker TP2×CP2) | Gather per layer | Layer time per rank | Comm share |
|---|---|---|---|
| 512 tokens | 0.26 MB, about 16 µs | about 0.25 ms (weight-read floor) | about 6 % |
| 4K tokens | 2 MB, about 20 µs | about 0.3 ms | about 7 % |
| 64K tokens | 32 MB, about 105 µs | about 18 ms | about 0.6 % |

Decode at `B = 8` with a 32K context: CP2 saves about 2.5 ms of KV reads per step and pays about 0.7 ms of all-gather latency. The 12 ms of expert reads stay. The net win is about 12 % of the step. At `B = 1` CP decode loses. The break-even is between 75K and 0.3M resident tokens per instance.

The model motivates these levers. Each lever has a v1 choice and a trigger to revisit it in [§10](#10-implementation-plan-and-follow-ups).

| Lever | Cost term | v1 choice | Trigger to revisit |
|---|---|---|---|
| Pass-Q or pass-KV for resident context | the `T / L` rule | both, selected per step by the rule ([§4.6](#46-pass-q-and-pass-kv-implementation)) | PR 5 measures the crossover |
| Ring or all-gather for the new K/V | overlap of the gather with compute | all-gather, no overlap (0.6 % at 64K) | gather above about 20 % of a prefill step, or a 1M-token target |
| Replicated prefill threshold | empty shards and gather overhead at small `T` | `T < 2 * cp * P` | PR 5 measurement |
| Interleave `I_tok` | write-scatter cost, empty shards, prefix-cache granularity | 1 (vLLM default) | PR 5 sweep over 1, `P`, `4P` |
| CP degree per instance | short-prompt loss of about 6 % from the floor | fixed per instance | short prompts dominate a `cp > 1` instance. Dense models first. |
| Sink-page writes | `(cp-1)/cp` of the scatter bandwidth (0.1 % at 64K) | accept | a profile shows the scatter above 5 % of a layer |

These are pen-and-paper values. PR 5 measures them.

### 4.6 Pass-Q and pass-KV implementation

Both methods for resident context are implementations of one interface behind `AttentionCallable`. The interface takes the local Q rows, the gathered new K/V, the view and the cache. It returns the full `O` for the local rows (I4).

```python
# mstar/distributed/cp/attention.py
class CPAttention(Protocol):
    def prefill(self, q_local, k_new, v_new, view, kv) -> Tensor      # full O for the local Q rows

class PassQ(CPAttention):   ...   # §4.4: Parts A, B, C
class PassKV(CPAttention):  ...   # below: chunked resident gather, then Part B

def select_method(T_total: int, L_total: int, Hq_l: int, Hkv_l: int) -> type[CPAttention]:
    # pass-Q iff 2 * T_total * Hq_l < L_total * Hkv_l ; batch totals, so every rank selects the same method (I2)
```

**Pass-KV for resident context.** The resident K/V travels instead of Q. The method has four steps per layer.

1. Divide the global resident range `[0, stored_len)` into fixed chunks of `C` tokens. The chunk count is a function of `stored_len`, so it is identical on every rank (I2). The last chunk is padded to `C`.
2. Per chunk, each rank gathers its owned rows from the paged cache into a contiguous workspace. This needs a gather kernel from the paged cache (M* has only `read_tokens` today). The ranks then all-gather the workspace across CP, and one precomputed index restores the global order.
3. Run non-causal ragged attention of the local Q chunk rows against the full chunk K/V, with LSE. Merge online with the running `(O, LSE)` of the stream.
4. Run Part B of [§4.4](#44-prefill-over-resident-context) over the gathered new K/V and merge. No Part C is necessary, because no Q left the rank.

The workspace holds one chunk of full K/V per rank, `C * Hkv_l * D * 4` bytes, and its size does not depend on the rank. `C` is a knob with a default of `4096` tokens. A larger `C` means fewer collectives and more transient memory.

**Selection.** The policy runs one time per prefill step from the batch totals of `T` and `L`. Thus one method runs per step, and every rank selects the same method with no exchange. Pass-Q moves `2 * T * Hq_l * D * 2` bytes per rank and pass-KV moves `L * Hkv_l * D * 4` bytes per rank, both times `(cp-1)/cp`. For the thinker, pass-Q wins when the step adds less than one eighth of the resident length.

A decode step, a modality-walk prefill and a short second turn select pass-Q. A long second turn after a short first turn selects pass-KV. An MLA model, with one latent KV head, selects pass-KV at every ratio.

**Shared parts.** Both methods use the same ragged wrapper, the same merge and the same gathered new K/V. Both report `lse = -inf` for a stream with no resident tokens, so the collectives stay unconditional per stream. Ring pass-KV is a later implementation of `CPAttention` that replaces the all-gather in step 2 with point-to-point exchange and overlaps it with step 3.

---

## 5. Model-facing API

A CP-capable `ARNodeSubmodule` does three things in prefill. Decode needs none of them. The thinker ([qwen3_omni/submodules.py](../../mstar/model/qwen3_omni/submodules.py)) is the reference, and it joins `cp_enabled_nodes`.

### 5.1 Slice the inputs

`prepare_inputs` builds the full `ARNodeInputs` as today. For a CP prefill that is not in replicated mode, it then keeps `split_inputs(start, end)` for the two chunks of this rank, concatenated. Positions, deepstack tensors and the MRoPE advance stay global, because `split_inputs` cuts them with the rows. The thinker overrides `split_inputs` to cut its `tensor_inputs`, which the base class refuses to do.

### 5.2 Declare the global span

`declare_step` keeps `Segment.span` as the full `input_seq_len`. `SubmoduleStep` gets `cp_rows`, so the attention resource plans the local Q rows.

### 5.3 Sample

Under zigzag, the last token of each request is the last row of the second chunk on rank 0. Thus `select_last_hidden` is correct on rank 0 only. Rank 0 broadcasts the `[bs, hidden]` last rows over the CP group. That is one latency-bound collective of tens of KB per prefill step. Every rank then runs the real `lm_head` and the sampler from the same seed, exactly as TP ranks do today. The current joint-group `_broadcast_tokens` stays as the agreement check.

Decode needs no hidden broadcast, because every CP rank holds the full hidden state. The redundant `lm_head` on the followers adds nothing to the critical path, because the step time is the time of the leader.

### 5.4 Thinker states

`thinker_states` are per-rank shards under CP prefill. The thinker all-gathers them and restores the sequence order only when `audio_output` is set.

---

## 6. Followers and CUDA graphs

1. Every branch that guards a collective derives its condition from the step declaration or the KV plan, never from a local tensor shape. "This rank has resident pages" differs per rank. "This stream has resident tokens" does not (I2).
2. The decode buckets capture the all-gather of [§4.2](#42-decode). The capture agreement already ANDs the result across the joint group, which [§2](#2-configuration-and-mesh) extends to the CP axis (I3).
3. Padding rows read garbage from `SINK_PAGE`. Their LSE is finite garbage that affects only their own rows. The NaN and infinity guard in the merge is mandatory.
4. CP prefill runs eager in v1. The prefill buckets stop at 2048 tokens, so long prompts run eager today. A graph saves 5 to 10 µs per kernel launch, which is below 0.1 % of a 64K prefill.

The followers get no new wire data. `ScheduleTPNode` carries the node, the walk, the request ids and the speculation flags. A follower computes its chunk layout and its page ownership from those values and its own rank.

---

## 7. Interactions with other work

Three features compose with no new work.

1. **Chunked prefill.** A chunk is a prefill with `stored_len > 0`, so it uses [§4.4](#44-prefill-over-resident-context). Inside a chunk, `split_inputs` composes: the chunk first, then zigzag. The chunk progress is per rank and identical by determinism.
2. **Speculation** (TP async follow). The flags travel on `ScheduleTPNode` unchanged. A CP decode step adds one lockstep collective per layer, as the TP all-reduce does.
3. **Prefix caching** (#210). Under CP a cached prefix is a set of per-rank page lists under one hash, with a hit granularity of `cp * I_tok` tokens. The per-rank refcount of #210 does not change.

Five features are asserted off in v1. Each has a follow-up in [§10](#10-implementation-plan-and-follow-ups).

1. **Windowed generation** (#198). A release of the oldest pages shifts every later logical page and its owner. The fix is a per-stream logical base offset.
2. **PD disaggregation.** Equal `cp` on both sides keeps the per-rank pull unchanged. But the conductor merges publish info from the first instance rank only, and CP page lists differ by rank. The fix is to publish the list of every rank. A different `cp` on the two sides needs a re-interleave.
3. **Other attention backends.** CP nodes require FlashInfer. Dense, xpu and cross attention assert `cp == 1`. Cross-attention labels stay replicated.
4. **Multi-token decode** (`span > 1` on a paged step). No model does it today. The new tokens can belong to different owners and need [§4.4](#44-prefill-over-resident-context).
5. **Ulysses SP with CP on an AR node.** The two compose in principle. SP on an AR paged node is unused today, so `sp*cp > 1` on an AR node is rejected.

---

## 8. Correctness checks

Checks 1 to 4 run in CI on one GPU. Check 5 needs 2 to 4 GPUs.

1. I1: for every stream and every `I_tok`, the owned-token sets partition `[0, L)`.
2. The concatenation of the ranks' `read_tokens` in layout order reproduces the global K/V that was written. The test runs random K/V through `cp` managers on one GPU.
3. `merge_partial_attention` over any division of a sequence equals full attention (bf16 `atol 1e-2`, fp32 `1e-5`). The test includes an empty shard and `N = 1`.
4. I5: `cp_size: 1` produces `KVPlanState` tensors identical to `main`.
5. Thinker text under `cp2`, `cp4` and `tp2*cp2` equals the TP1 baseline under the protocol in `test/distributed/test_qwen3omni_thinker_tp2_vs_tp1.py` on 2K, 16K and 32K prompts. A test pins that zigzag puts the last chunk on rank 0.

---

## 9. Caveats

These are the places where the method can be wrong, not the places where code can have a bug. Group A causes a stop or an incorrect output with no error.

**A. Head and token split.** The head degree is `tp*sp`, not the joint world size. A site that uses the joint size as the head degree drops heads with no error, and two such sites exist today ([§2.3](#23-call-sites-that-change)). Every agreement check and the token broadcast must run over all three axes, or they skip ranks. Exact per-rank page counts make admission and OOM differ across ranks and stop the next collective. Symmetric reservation prevents it at one page per stream per rank ([§3.4](#34-symmetric-reservation)).

**B. Ownership and the sink page.** Prefill has two different "local" token sets: the Q rows (zigzag) and the stored KV (token-interleaved). They are independent only because the new K/V is all-gathered, so the write plan filters by ownership, never by Q rows.

Non-owned rows go to a sink page. That trades `(cp-1)/cp` of the scatter bandwidth and one garbage row for a plan with no per-rank branches. The same sink page serves streams with fewer than `cp` tokens and padding rows, so the kernel call and the collective stay unconditional. Thus the merge must guard against NaN and infinity.

**C. Resident context and positions.** Any attention with more than one new query over a partial page list needs the three-part method of [§4.4](#44-prefill-over-resident-context). A single kernel call gives an incorrect result with no error. Every multimodal thinker request and every chunk after the first one under chunked prefill take this path. Positions are counters independent of the KV length, so `stored_len` and the position counter must never derive from each other ([§3.3](#33-position-ids)).

**D. Model.** The sampler relies on zigzag to put the last chunk on rank 0 ([§5.3](#53-sample)). `thinker_states` are shards that the thinker gathers only for audio output. MoE routing per rank sees only the local tokens, so the expert load differs per rank and the result does not. The expert weight read is a per-layer floor that CP does not reduce, which caps the win for short prompts and for decode ([§4.5](#45-cost-model-and-levers)).

**E. Scope.** Only FlashInfer supports CP in v1. Five features are asserted off: windowed retention, PD with a different `cp` or first-rank-only publish, CPU offload, multi-token decode, and SP with CP on an AR node ([§7](#7-interactions-with-other-work)).

---

## 10. Implementation plan and follow-ups

One PR per row. Each PR keeps the invariants of [§1](#1-design-rules-and-invariants) and the checks of [§8](#8-correctness-checks) that exist at that point.

| PR | Scope | Done when |
|---|---|---|
| 1. Mesh, groups, LSE | `cp_size` axis, `head_shard_size` at every head-degree site, three-axis agreement checks, `return_lse`, `CPLayout` and the merge with unit tests | check 4 passes and the current TP2 and TP2×SP2 tests do not change |
| 2. KV ownership and decode CP | `CPLayout` in the allocator and the plans, symmetric reservation, sink page, decode merge path, replicated prefill as the only prefill mode, assertions for the features that are off | checks 1 to 3 pass and CUDA-graph decode replays with the all-gather inside |
| 3. Prefill CP with pass-Q | zigzag chunks, K/V gather, ragged wrapper, thinker slice, span and sample, `CPAttention` with `PassQ` | check 5 passes on 16K and 32K text prompts |
| 4. Pass-KV and the policy | paged-cache gather kernel, fixed-chunk workspace, chunk loop with online merge, `PassKV`, `select_method` | check 5 passes with each method forced, and the two methods agree |
| 5. Measurement | time to first token, time per output token, capacity in 32K sessions, `I_tok` sweep, the pass-Q/pass-KV crossover | the table gives the prompt length at which CP2 and CP4 beat TP2, and the defaults for `I_tok` and `C` |

**Follow-ups, in priority order, each with its trigger from [§4.5](#45-cost-model-and-levers):**

1. *Ring pass-KV and pass-Q* behind the same interface. Trigger: a measured all-gather above about 20 % of a prefill step, or a 1M-token target that exceeds the transient memory. Prerequisites: point-to-point send and receive (M* has `all_gather` and `all_to_all` only), a second CUDA stream with events, and double-buffered K/V.
2. *Fused merge and compaction before the K/V write.* Trigger: a profile shows the merge or the sink scatter above 5 % of a layer. With `I_tok = 1` the owned rows of the gathered buffer are a strided view, so the compaction is free.
3. *Length-based CP degree* (two node groups plus a conductor predicate). Trigger: short requests dominate a `cp > 1` instance. Dense models are the first candidates, because the thinker's floor keeps the short-prompt loss near 6 %.
4. *CP carved from TP*, *PD with different `cp`*, *prefix cache under CP* (with #210), *windowed generation under CP* (with #198).
5. *CP prefill CUDA-graph capture* (lowest priority). Trigger: a profile shows launch latency above 5 % of a CP prefill step, after items 1 to 4.
6. *Cheaper `lm_head` on the critical path* (low priority, a TP item). The thinker's `lm_head` reads about 310 MB per rank per step at TP2, about 0.12 ms. Trigger: a profile shows `lm_head` above 5 % of a decode step.

---

## 11. Benchmark acceptance

Two predictions from the cost model are the acceptance checks for PR 5.

- Thinker geometry: 48 layers, `Hq = 32`, `Hkv = 4`, `D = 128`. Hardware: one 8×H100 node.
- Prediction 1: CP2 prefill is about 1.9× faster than the TP1-equivalent compute above about 24K tokens.
- Prediction 2: decode CP is slower than TP at `B = 1`. The break-even is between 75K and 0.3M resident tokens per instance.
- The per-layer all-gather time comes from the current NVTX ranges.

---

## 12. Non-goals

1. Ring attention in v1. No point-to-point primitive exists, and the all-gather is safe under CUDA graphs. It is a follow-up behind the `CPAttention` interface.
2. CP on non-autoregressive nodes (DiT, encoders). Ulysses SP covers them.
3. Sliding-window, Mamba, linear-attention or attention-sink layers under CP. No in-tree AR model has them, and vLLM asserts them off too.
4. Changes to conductor routing or to the Rust runtime. The instance abstraction absorbs the new axis.
5. A reduction of weight memory. The weights are replicated across CP ranks by design, as in the paper.
