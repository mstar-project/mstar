# Context parallelism (CP) for autoregressive nodes

Divide the token axis of one request across `cp` ranks of one instance. The ranks of one CP group hold the same attention heads and different tokens. The KV cache per rank falls by `1/cp` past the head-shard limit. The attention compute of a long prefill falls by `1/cp`. Each decode step reads `1/cp` of the resident KV per rank.

CP is opt-in per `node_groups` entry with `cp_size`. The default `cp_size: 1` keeps the current behavior byte for byte.

- **Status:** draft for review. No code is written. Pinned to M* commit `33baea08`.
- **Area:** `engine/resources/{kv,attn,position,sampler}`, `distributed`, `model/base`, `model/qwen3_omni`. No worker, conductor or Rust runtime change.
- **Related:** #210 and #279, #280 (prefix KV reuse), #293 (windowed generation), [chunked prefill plan](https://github.com/mstar-project/mstar/blob/chunked-prefill-2/docs/design/chunked_prefill_plan.md), vLLM DCP and PCP, arXiv 2411.01783 (ring attention for 1M-token prefill).

Notation:

- `cp` = `cp_size`, `r` = CP rank, `P` = `page_size`, `I_tok` = `cp_interleave_tokens`.
- `L` = resident token count of a stream, `T` = new tokens of a stream in this step.
- `Hq_l` and `Hkv_l` = head counts of this rank after the `tp*sp` split, `D` = head dimension.

Sections:

- Design: [0 Decision log](#0-decision-log) · [1 Rules and invariants](#1-design-rules-and-invariants) · [2 Mesh](#2-configuration-and-mesh) · [3 KV ownership](#3-kv-ownership) · [4 Attention](#4-attention) · [5 Model API](#5-model-facing-api) · [6 Followers and graphs](#6-followers-and-cuda-graphs)
- Review: [7 Interactions](#7-interactions-with-other-work) · [8 Checks](#8-correctness-checks) · [9 Open risks](#9-open-risks) · [10 Plan](#10-implementation-plan-and-follow-ups) · [11 Non-goals](#11-non-goals)

## Workload and expected gains

The reference model is the Qwen3-Omni-30B-A3B thinker, a mixture-of-experts text decoder. It has 48 layers, hidden size 2048, 32 query heads, 4 KV heads and head dimension 128. It has 128 experts with 8 active per token. It runs at TP2 today with `max_seq_len: 32768` in every config. Its 4 KV heads replicate past TP4, so TP4 is the last TP degree that divides the KV cache and the attention compute.

At the current shape and cap, CP is close to TP at equal GPU count. CP2×TP2 and TP4 divide the attention compute and the KV cache by the same factor. TP4 also halves the expert weight read per layer, which CP leaves intact.

CP moves fewer bytes per token. One K/V gather costs about 0.5 KB per token per layer. TP activation all-reduces cost about 8 to 12 KB per token per layer. By pen and paper, CP2×TP2 beats TP4 on prefill above about 12K prompt tokens. It loses on decode at every batch size that fits in 32K sessions.

The case for CP is the context past 32K. Past TP4, TP adds no KV capacity and no attention division. Attention compute grows with `T^2` and decode KV reads grow with the resident tokens, while the expert floor stays constant. At 128K tokens, attention is above 95 % of a thinker prefill layer, and a request holds about 13 GB of KV. Only CP divides that further.

The work therefore has two parts: raise the thinker cap to 64K and then 128K with RoPE scaling, and a dense long-context text model as the second target. We propose Qwen3-32B (dense, 8 KV heads, 128K with YaRN). It is not in the tree, so this is a proposal. [§4.6](#46-cost-model-and-levers) gives the numbers.

---

## 0. Decision log

| # | Decision | Where |
|---|---|---|
| 1 | `cp_size` is a third mesh axis next to `tp_size` and `sp_size`. `tp*sp` divides the heads and `cp` divides the tokens. "CP carved out of TP" is a follow-up. | [§2](#2-configuration-and-mesh) |
| 2 | KV ownership is interleaved by token through one pure class, `CPLayout`, with the knob `cp_interleave_tokens` (default 1). Every rank reserves the per-rank maximum page count, so the allocator state stays symmetric. | [§3](#3-kv-ownership) |
| 3 | One merge primitive for all row types: all-gather `(O, LSE)` and merge by log-sum-exp. Decode is local paged attention plus this merge. | [§4.1](#41-lse-merge), [§4.2](#42-decode) |
| 4 | Text prefill divides the Q rows into zigzag chunks over one gather of the new K/V. Vision walks and short prompts run replicated. A row over resident context selects pass-Q or single-shot pass-KV by a per-row cost rule. The exchange of K/V is inside the `CPAttention` contract, so a ring replaces the all-gather later with no caller change. | [§4.3](#43-prefill-of-a-fresh-prompt) to [§4.7](#47-pass-q-and-pass-kv-implementation) |
| 5 | A CP-capable model slices its text prefill inputs and declares the global span. Rank 0 owns the last row of every request, and the current rank-0 token broadcast overwrites the followers. No new collective at sampling. | [§5](#5-model-facing-api) |
| 6 | The CP degree is fixed per instance. No conductor or Rust change beyond the instance size `tp*sp*cp`. PD, prefix cache, windowed generation, CPU offload and SP-with-CP are asserted off on CP nodes in v1. | [§7](#7-interactions-with-other-work) |

---

## 1. Design rules and invariants

Five rules give every change in sections 2 to 6.

1. **The instance is `cp*sp*tp` ranks in lockstep, in row-major order `[cp][sp][tp]`.** Rank 0 leads. All other ranks follow the current `ScheduleTPNode` FIFO. No layer above the attention resource learns a new concept.
2. **`tp*sp` divides the heads. `cp` divides the tokens.** Every `KVConfig.shard()` call and every head-degree call site takes `head_shard_size = tp*sp`, never the joint world size.
3. **`stored_len` stays global. `CPLayout` derives every per-rank number from it.** Token `g` lives on rank `(g // I_tok) % cp`.
4. **Decode runs local paged attention, then all-gathers `(O, LSE)` and merges.** All ranks hold the same heads, so no Q exchange and no reduce-scatter is necessary. The all-gather is captured in CUDA graphs, as the SP all-gather is.
5. **Prefill runs zigzag Q chunks against the gathered new K/V and writes the owned pages.** Rank 0 always owns the last chunk of each request. Rows that are too short for `2*cp` page-sized chunks, and vision walks, run replicated.

Every PR keeps these invariants. The checks in [§8](#8-correctness-checks) test them.

- **I1. Partition.** For every stream, the owned-token sets of the `cp` ranks partition `[0, L)`, and `sum_r local_len_r(L) == L`.
- **I2. Determinism.** The row layout, the page ownership, the method per row and every collective shape are pure functions of two things. These are the per-row `(T, L)` and the instance constants `(cp, P, I_tok)`. Every rank knows these values, so no rank can take a different branch.
- **I3. Symmetry.** The free-page count, the admission verdict and the CUDA-graph capture result are identical on every rank of the instance.
- **I4. Full output per rank.** Every attention call ends with the full merged `O` for the `Hq_l` heads of the rank. The row-parallel `o_proj` and its TP all-reduce do not change.
- **I5. Zero cost when off.** `cp_size: 1` produces plan tensors identical to `main`.

---

## 2. Configuration and mesh

```yaml
node_groups:
  - node_names: [Thinker]
    ranks: [0, 1, 2, 3]
    tp_size: 2
    cp_size: 2        # TP groups {0,1},{2,3}; CP groups {0,2},{1,3}
```

`WorkerGraph` ([base.py](../../mstar/model/base.py)) gets `cp_size` and the CP rank lists, in the same form as the SP ones. `tp_size` is still rewritten to the instance size, so `ShardingGroup`, the conductor and the Rust shard code do not change. A CP node declares no `shard_dim` entries. Its signals are replicated, and each rank cuts its own slice locally.

`JointGroups` gets `cp_group`, a row-major `rank`, and `head_shard_size = tp * sp`. `GlobalParallelConfig` ([communication.py](../../mstar/distributed/communication.py)) builds the CP groups in the sorted group schedule, so every rank creates them in the same order.

| Site | Today | With CP |
|---|---|---|
| `KVConfig.shard()`: five call sites in `kv/manager.py`, `kv/ring/manager.py`, `attn/base.py`, `attn/ragged/base.py`, `attn/cross.py` | joint world size | `head_shard_size` |
| piecewise CUDA-graph configs and the two capture passes ([engine.py](../../mstar/engine/engine.py)) | joint world size as head degree | `head_shard_size` |
| `agree_across_ranks`, `post_warmup_validate`, `_verify_tp_async_sched_agrees` | loop over `[tp_group, sp_group]` | loop over `[tp_group, sp_group, cp_group]` |
| `BaseSampler._broadcast_tokens` | TP group | joint group |

The `head_shard_size` sites are the lines that must not be wrong. If `cp` is part of the head degree, `KVConfig.shard()` divides the heads by `cp` and drops heads with no error.

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

With `I_tok = 1`, token `g` lives on rank `g % cp`. With `I_tok = P`, logical page `p` lives on rank `p % cp`. The knob trades the write-scatter cost against empty shards for short prompts and against the prefix-cache hit granularity ([§4.6](#46-cost-model-and-levers)).

### 3.2 KV manager changes

All changes are in [manager.py](../../mstar/engine/resources/kv/manager.py).

| Function | Change |
|---|---|
| `_alloc` | Reserve `layout.reserve_pages(L)` per stream on every rank. Use the first `local_pages(L)`. At most `ceil(I_tok / P)` pages per stream per rank are idle, one page for `I_tok <= P`. |
| `_sequence_views` | The view length becomes `local_len(stored_len + span)`. `SequenceView` gets `global_length`, `global_to_compute` and `q_rows`. `to_compute` becomes the local Q-row count. |
| `_compute_plan_state` | Compute the global index `g` per token as today. Owned tokens map through `local_index(g)` to `(page, offset)`. Non-owned tokens map to `SINK_PAGE`. |
| `_decode_plan_state` | The owner of the new token is `owner_of_token(stored_len)`. Other ranks write to `SINK_PAGE`. |
| free-page symmetry check after warmup | Add the CP group. The check stays strict. |
| `KVConfig` ([config.py](../../mstar/engine/resources/kv/config.py)) | Carries `cp_size`, `cp_rank`, `cp_interleave_tokens`. `max_num_pages` stays a per-rank knob. |

### 3.3 Position ids

`PositionManager` emits one contiguous range per view today. Under zigzag the local rows are two chunks, so the manager turns `q_rows` into explicit positions through the current `pos_ids` override. The counters advance by the global span and stay symmetric. Positions are counters independent of the KV length. The thinker computes its 3D MRoPE positions itself, and the position advance of a vision walk differs from its token count. Thus `stored_len` and the position counter must never derive from each other.

### 3.4 Symmetric reservation

Exact per-rank page counts differ by up to one page per stream per rank. The free-page totals then differ. A request can then fail allocation on one rank and pass on another. The allocation-failure handler and TP-async speculation both assume that every rank reaches the same verdict. A different verdict on one rank stops the next collective.

The alternative is exact counts plus an `all_reduce(min)` on admission. That adds a collective to the scheduler path. The reservation of the maximum costs one page per stream per rank and keeps every current symmetry check strict (I3).

### 3.5 Forks, offload and prefix cache

- Forks copy local pages to local pages. Both streams start at logical token 0, so ownership matches. Only the copy count changes, to the reserved page count.
- CPU offload is per rank. Victim selection uses a wall-clock LRU and is already uncoordinated across TP ranks. Under CP the reclaimable page counts also differ per rank. CP nodes require `cpu_offload_pages == 0` in v1.
- The prefix cache (#279, #280) is in the tree and refuses a world size above 1. It stays off on CP nodes. Under CP a cached prefix becomes a set of per-rank page lists under one hash with a hit granularity of `cp * I_tok` tokens. That is a follow-up.

---

## 4. Attention

### 4.1 LSE merge

Both FlashInfer wrappers return the log-sum-exp (LSE) on request. The merge is a pure function with no communication inside.

```python
# mstar/distributed/cp/merge.py
def merge_partial_attention(o: Tensor[N, B, H, D], lse: Tensor[N, B, H]) -> Tensor[B, H, D]:
    # lse <- -inf where NaN or +inf ; o <- 0 where not finite   (empty shard, sink-page row)
    # m = max_n lse ; if m == -inf: return zeros          (only padding rows reach this)
    # w_n = exp(lse_n - m) ; o = sum_n w_n * o_n / sum_n w_n

def cp_all_gather_merge(cp_group, o_local, lse_local) -> Tensor[B, H, D]:
    # all_gather both along a new leading dim into a static buffer, then merge_partial_attention
```

The guard covers both tensors. A sink-page row can return NaN in `O`, and zero times NaN is NaN, so `O` is cleaned before the weights apply. A row with every partial empty has `m == -inf` and returns zeros. A real stream never reaches that case, because at least one rank holds a finite partial for every row. The LSE stays in fp32 in the natural-log base. Every CP rank ends with the full merged `O` for its `Hq_l` heads (I4).

### 4.2 Decode

`FlashInferManager.run` gets one branch when `cp_group.world_size > 1`.

1. Run local paged attention with `return_lse=True`.
2. A row with zero local KV gets one `SINK_PAGE` in the plan, so the kernel never sees an empty row. Such rows are streams with fewer than `cp` tokens and CUDA-graph padding rows. The plan sets their `lse` to `-inf`.
3. `cp_all_gather_merge` into a static buffer per bucket, because a captured graph bakes the address.

The all-gather moves `cp * B * Hq_l * (D + 1) * 2` bytes per layer per step. That is a few hundred KB. The CUDA graph replays the all-gather in the same way as the SP all-gather.

### 4.3 Prefill of a fresh prompt

This path serves text rows with `L == 0` and `T >= 2 * cp * P`. Cut `[0, T)` into `2*cp` chunks. Rank `r` takes chunks `r` and `2cp-1-r`. Each chunk is one FlashInfer ragged row with `kv_len = chunk_end`, so the end-aligned causal mask is exact.

```python
# mstar/distributed/cp/prefill.py
def zigzag_chunks(T: int, cp: int) -> list[list[tuple[int, int]]]   # per rank, two (start, end)
def exchange_new_kv(cp_group, k_loc, v_loc, local_rows, T) -> (k_full, v_full)
    # v1: padded all_gather (gather_sequence) plus one precomputed index that restores global order
def ragged_rows(chunks) -> (qo_indptr, kv_indptr)
```

Per layer, inside `AttentionCallable.__call__`, three steps run. Exchange the new K/V in one `gather_sequence` call. Write the full K/V through the plan of [§3.2](#32-kv-manager-changes). Run the local Q chunks through a new ragged prefill wrapper over the full new K/V. The ragged path reads all new keys, so no cross-rank merge is necessary. Zigzag balances the causal FLOPs to within one chunk and puts the last chunk on rank 0, which [§5.3](#53-sample) uses.

**Replicated rows.** A row with `T < 2 * cp * P`, and every vision walk, runs replicated: every rank computes all `T` rows and writes only its owned pages. No gather runs. Vision walks keep their captured graphs, their one-request batch and their deepstack inputs exactly as today. CP saves KV memory on them and no compute. PR 2 ships with replicated rows as the only prefill mode.

### 4.4 Prefill over resident context

This path serves rows with `L > 0`. A prefill that extends a stream with resident tokens must also attend to the resident pages. Examples are every thinker walk after the system prompt, a multi-turn chat, and every chunk after the first one under chunked prefill. Each rank holds only a subset of the resident pages. A single causal kernel call over a partial page list infers the query position from the local key count. That position is wrong, and the kernel gives an incorrect result with no error.

Two methods are correct. Both end with the same merge. [§4.7](#47-pass-q-and-pass-kv-implementation) gives the rule that selects between them per row.

**Pass-Q**, the vLLM DCP method, in three parts:

1. **Part A.** Gather the new Q rows across CP in the same collective as K and V. Run non-causal paged attention of all `T` new Q rows against the local resident pages, with LSE. Every rank holds a different slice of the resident keys, so every rank must see every new query.
2. **Part B.** Run causal ragged attention of the local Q chunk against the gathered new K/V, with LSE. This is the call of [§4.3](#43-prefill-of-a-fresh-prompt).
3. **Part C.** All-gather the Part A partials `(O_A, LSE_A)` across CP. Keep the rows of this rank's chunk. Merge them with the Part B partial.

**Single-shot pass-KV**, in two parts. Each rank reads its owned resident rows of the stream into a contiguous workspace, and the ranks all-gather the workspace. Each rank then runs causal ragged attention of its local Q chunk against the resident keys followed by the new keys, with LSE. No Q leaves the rank and no Part C is necessary. The workspace holds `L * Hkv_l * D * 4` bytes, so this method serves `L` up to a knob `cp_pass_kv_max_tokens` (default 8192).

For a replicated row, every rank already holds all `T` queries and all new K/V. Part A runs on all rows, Part B runs locally on all rows, and Part C merges. That is the decode path with `T` above one. A rank with no resident pages for a stream reports `lse = -inf`, so every collective stays unconditional per row (I2).

### 4.5 Path matrix

| Row | Q layout | Method | Collectives per layer |
|---|---|---|---|
| Text, `L == 0`, `T >= 2cpP` | zigzag | fresh prompt ([§4.3](#43-prefill-of-a-fresh-prompt)) | new K/V gather |
| Text, `L == 0`, `T < 2cpP`; vision walk, `L == 0` | replicated | local causal, write owned pages | none |
| Text, `L > 0`, `T >= 2cpP`, `T * Hq_l < L * Hkv_l` or `L` above the pass-KV cap | zigzag | pass-Q | new K/V and Q gather, `(O_A, LSE_A)` gather |
| Text, `L > 0`, `T >= 2cpP`, otherwise | zigzag | single-shot pass-KV | new K/V gather, resident K/V gather |
| Text, `L > 0`, `T < 2cpP`; vision walk, `L > 0`; multi-token decode | replicated | Part A on all rows, local Part B, Part C | `(O_A, LSE_A)` gather |
| Decode, `T == 1` | replicated | Part A, Part C | `(O, LSE)` gather, captured |

Rows of one method in one step are grouped into one kernel call and one set of collectives. The collective shapes are sums over the rows of that method, so they are identical on every rank (I2). Multi-token decode is therefore a supported path. No in-tree model uses it, so no v1 check covers it.

### 4.6 Cost model and levers

Constants: thinker at TP2×CP2, `Hq_l = 16`, `Hkv_l = 2`, `D = 128`, bf16, 48 layers. One H100 node: about 15 µs per collective, about 350 GB/s per rank for a gather, about 2.5 TB/s HBM. Per layer:

| Term | Formula per rank | Note |
|---|---|---|
| New K/V gather | `(cp-1)/cp * T * Hkv_l * D * 4` | about 0.5 KB per token. No overlap with compute. |
| Pass-Q extra | `(cp-1)/cp * T * Hq_l * D * 2 * 2` | Q out, partial `O` and LSE back. About 4 KB per new token. |
| Pass-KV extra | `(cp-1)/cp * L * Hkv_l * D * 4` | About 0.5 KB per resident token. |
| TP activation all-reduces | `2 * (tp-1)/tp * T * hidden * 2`, twice | About 8 KB per token at TP2, 12 KB at TP4. |
| Attention FLOPs | `2 * T * (L + T) * Hq_l * D / cp`, causal halves the `T * T` part | 32K fresh: about 3.7 ms. 128K: about 60 ms. The only term CP divides. |
| MoE expert weight read | about 604 MB at TP2, 302 MB at TP4 | About 0.25 ms from about 128 local tokens. TP divides it, CP does not. |
| Decode all-gather | `cp * B * Hq_l * (D+1) * 2` | Latency-bound, about 15 µs. |

Pass-Q is cheaper than pass-KV when `T / L < Hkv_l / Hq_l`, which is `1/8` for the thinker. The thinker prefill schedule follows the prompt order: the system prompt, then one walk per attachment, then the user text. A modality walk is therefore a large `T` over a small `L` and selects pass-KV. The user text after the attachments is a small `T` over a large `L` and selects pass-Q. A dense model with `Hkv / Hq = 1/4` moves the threshold, and an MLA model with one latent KV head selects pass-KV at every ratio.

| Fresh text prompt (thinker TP2×CP2) | Gather per layer | Layer time per rank | Comm share |
|---|---|---|---|
| 512 tokens | 0.26 MB, about 16 µs | about 0.25 ms (weight-read floor) | about 6 % |
| 4K tokens | 2 MB, about 20 µs | about 0.3 ms | about 7 % |
| 32K tokens | 16 MB, about 60 µs | about 4 ms | about 1.5 % |
| 128K tokens | 64 MB, about 200 µs | about 60 ms | about 0.3 % |

A row over resident context adds the pass-Q or pass-KV bytes to the gather column. For a 16K user text over a 16K context under pass-Q, that is about 32 MB more, about 2 % of the layer.

Decode at `B = 8` with a 32K context: CP2 saves about 2.5 ms of KV reads per step and pays about 0.7 ms of all-gather latency. The 12 ms of expert reads stay. The net win against TP2 is about 12 % of the step. TP4 halves the expert reads instead and wins decode at this shape. CP decode beats TP only when KV reads dominate, above about 75K to 0.3M resident tokens per instance.

The model motivates these levers. Each lever has a v1 choice and a trigger to revisit it in [§10](#10-implementation-plan-and-follow-ups).

| Lever | Cost term | v1 choice | Trigger to revisit |
|---|---|---|---|
| Pass-Q or pass-KV per row | the `T / L` rule | both, single-shot pass-KV up to the cap | PR 5 measures the crossover and the cap |
| Ring or all-gather | overlap of the gather with compute | all-gather, no overlap | gather above about 20 % of a prefill step |
| Replicated prefill threshold | empty shards and gather overhead at small `T` | `T < 2 * cp * P`, and all vision walks | PR 5 measurement, zigzag vision follow-up |
| Interleave `I_tok` | write-scatter cost, empty shards, prefix-cache granularity | 1 (vLLM default) | PR 5 sweep over 1, `P`, `4P` |
| CP degree per instance | short-prompt loss of about 6 % from the floor | fixed per instance | short prompts dominate a `cp > 1` instance |
| Sink-page writes | `(cp-1)/cp` of the scatter bandwidth | accept | a profile shows the scatter above 5 % of a layer |

These are pen-and-paper values. PR 5 measures them at equal GPU count.

### 4.7 Pass-Q and pass-KV implementation

Both methods are implementations of one interface behind `AttentionCallable`. The interface takes the local Q rows, the local new K/V, the CP group, the view and the cache. It returns the full `O` for the local rows (I4). The exchange of the new K/V is inside the contract. Thus the all-gather and a ring are two implementations of the same contract, for the new tokens and for the resident context alike.

```python
# mstar/distributed/cp/attention.py
class CPAttention(Protocol):
    def prefill(self, q_local, k_local, v_local, cp_group, view, kv) -> Tensor   # full O for the local Q rows

class AllGatherPassQ(CPAttention):   ...   # all-gather new K/V and Q, then §4.4 Parts A, B, C
class AllGatherPassKV(CPAttention):  ...   # all-gather new K/V and the resident K/V, then one causal ragged call
# RingPassKV, RingPassQ: follow-ups behind the same contract

def select_method(T: int, L: int, Hq_l: int, Hkv_l: int, pass_kv_max: int) -> type[CPAttention]:
    # per row: pass-KV iff T * Hq_l >= L * Hkv_l and L <= pass_kv_max, else pass-Q
```

The rule runs per row from values every rank knows (I2). The rule counts bytes, and at the crossover the two methods cost the same by definition. Communication grows with the sum of `T` and `L`, while compute grows with their product. Thus a wrong choice costs the most where both are small and the step is cheap in any case. For the thinker the choice is a few percent either way.

Both methods use the same ragged wrapper, the same merge and the same exchange of the new K/V. Both report `lse = -inf` for a stream with no resident tokens.

**Single-shot against ring.** The all-gather implementations move the whole K/V in one collective and then compute. A ring cuts the K/V into `cp` blocks and passes each block to the next rank point to point. It computes on one block while the next block arrives. Ring wins in two cases. The K/V is too large for one workspace, or the exchange is a large share of the step and the overlap pays. That is the 1M-token regime of the paper.

Under a 128K cap the exchange is below 2 % of the step, so the all-gather is enough. The loop for the large-and-large case waits for the ring. The ring needs a point-to-point primitive, a second CUDA stream and double-buffered K/V, none of which exist today.

---

## 5. Model-facing API

A CP-capable `ARNodeSubmodule` does two things in text prefill. Decode and replicated rows need none of them. The thinker ([qwen3_omni/submodules.py](../../mstar/model/qwen3_omni/submodules.py)) is the reference, and it joins `cp_enabled_nodes`.

### 5.1 Slice the inputs

`prepare_inputs` builds the full `ARNodeInputs` as today. For a zigzag text row, it then keeps `split_inputs(start, end)` for the two chunks of this rank, concatenated. Positions and the MRoPE advance stay global, because `split_inputs` cuts them with the rows. Vision walks are replicated, so their deepstack inputs and their `.item()` syncs stay as they are.

### 5.2 Declare the global span

`declare_step` keeps `Segment.span` as the full `input_seq_len`. `SubmoduleStep` gets `cp_rows`, so the attention resource plans the local Q rows.

### 5.3 Sample

Under zigzag, the last token of each request is the last row of the second chunk on rank 0. Thus `select_last_hidden` is correct on rank 0 only. The followers run `lm_head` and the sampler on their own last rows, which are not the true rows, with the same shapes. The current `_broadcast_tokens` is an in-place overwrite from rank 0, which [§2](#2-configuration-and-mesh) moves to the joint group. The follower tokens are therefore correct after the overwrite.

The penalty state and the RNG offsets advance from the broadcast tokens, as they do on TP followers today. No new collective is necessary. Decode and replicated rows hold the true last rows on every rank.

### 5.4 Thinker states

`thinker_states` are per-rank shards under a zigzag row. The thinker all-gathers them and restores the sequence order only when `audio_output` is set.

---

## 6. Followers and CUDA graphs

1. Every branch that guards a collective derives its condition from the step declaration or the KV plan, never from a local tensor shape. "This rank has resident pages" differs per rank. "This stream has resident tokens" does not (I2).
2. The decode buckets capture the all-gather of [§4.2](#42-decode). The capture agreement already ANDs the result across the joint group, which [§2](#2-configuration-and-mesh) extends to the CP axis (I3).
3. Padding rows read garbage from `SINK_PAGE`. Their partials affect only their own rows, and the merge guard of [§4.1](#41-lse-merge) cleans them.
4. Zigzag text prefill runs eager in v1. The text prefill buckets stop at 2048 tokens, so long text prompts run eager today. Vision prefill is captured up to 16K tokens, and it keeps its graphs because vision walks run replicated.

The followers get no new wire data. `ScheduleTPNode` carries the node, the walk, the request ids and the speculation flags. A follower computes its row layout and its page ownership from those values and its own rank.

---

## 7. Interactions with other work

Two features compose with no new work.

1. **Chunked prefill.** A chunk after the first one is a row with `L > 0`, so it uses [§4.4](#44-prefill-over-resident-context). A mixed step holds decode rows next to chunk rows. A rank's row set is every decode row, plus the zigzag rows of every chunk row. The token budget of the chunked plan stays in global tokens, so every rank cuts the same chunks (I2). The per-rank compute is then `sum(decode rows) + sum(T_i / cp)`, so a CP node can raise its budget by up to `cp`. That scale is a knob, default 1 in v1.
2. **Speculation** (TP async follow). The flags travel on `ScheduleTPNode` unchanged. A CP decode step adds one lockstep collective per layer, as the TP all-reduce does.

Five features are asserted off on CP nodes in v1. Each has a follow-up in [§10](#10-implementation-plan-and-follow-ups).

1. **Windowed generation** (#293, in the tree). A release of the oldest pages shifts every later logical page and its owner. The fix is a per-stream logical base offset.
2. **PD disaggregation.** The world-size match and the publish path assume one page list per instance. CP page lists differ by rank. The fix is to publish the list of every rank, with equal `cp` on both sides as the first supported case.
3. **Prefix cache** (#279, #280). Off at world size above 1 today. Under CP it needs per-rank page lists under one hash.
4. **Other attention backends.** CP nodes require FlashInfer. Dense, xpu and cross attention assert `cp == 1`. Cross-attention labels stay replicated.
5. **Ulysses SP together with CP on one AR node.** The two compose in principle. SP on an AR paged node is unused today, so `sp > 1` and `cp > 1` on the same AR node is rejected.

---

## 8. Correctness checks

Checks 1 to 4 run in CI on one GPU. Check 5 needs 4 GPUs.

1. I1: for every stream and every `I_tok`, the owned-token sets partition `[0, L)`.
2. The concatenation of the ranks' `read_tokens` in layout order reproduces the global K/V that was written. The test runs random K/V through `cp` managers on one GPU.
3. `merge_partial_attention` over any division of a sequence equals full attention (bf16 `atol 1e-2`, fp32 `1e-5`). The test includes an empty shard, a NaN partial, an all-empty row and `N = 1`.
4. I5: `cp_size: 1` produces `KVPlanState` tensors identical to `main`.
5. A new equivalence protocol, because the current TP2-vs-TP1 script is a one-prompt smoke. Greedy decode (`temperature 0`, `top_k 1`) of 256 tokens on fixed prompts: text at 2K, 16K and 32K, one image prompt, one two-turn prompt, and one prompt under chunked prefill. The token sequence under `cp2`, `cp4` and `tp2*cp2` must equal the TP1 sequence (strict), with the first 64 tokens as the loose level. A test pins that zigzag puts the last chunk on rank 0. A test forces each resident-context method and compares the two.

---

## 9. Open risks

These are the places where the plan can be wrong, not the places where code can have a bug.

- **The context cap.** The gains past 32K depend on a RoPE-scaled thinker that is not validated. If the quality at 64K is not acceptable, the thinker stays a correctness bed and the gains move to the dense target.
- **The equal-GPU comparison.** The 12K crossover against TP4 comes from pen-and-paper constants for all-reduce and gather bandwidth. A measured NVLink all-reduce can move it by a factor of two in either direction.
- **The `T / L` rule.** The rule counts bytes only. Kernel efficiency differs between one causal ragged call over resident plus new keys and a non-causal paged call over resident pages. PR 5 measures both.
- **Vision replicated.** A 16K vision walk on `cp` ranks computes `cp` times the attention of one rank. CP gives no prefill speedup on vision in v1, only KV memory.
- **The chunked budget unit.** The budget stays in global tokens for symmetry. A mixed step then under-fills the CP ranks by up to `cp`. The scale knob is a guess until measured.
- **The MoE floor.** The 604 MB per layer assumes almost all experts active from about 128 local tokens. A skewed router changes the floor and the short-prompt share.

---

## 10. Implementation plan and follow-ups

One PR per row. Each PR keeps the invariants of [§1](#1-design-rules-and-invariants) and the checks of [§8](#8-correctness-checks) that exist at that point.

| PR | Scope | Done when |
|---|---|---|
| 0. Context cap | `max_seq_len` 64K for the thinker with RoPE scaling, a quality check at 64K, page pool sized for it | the TP2 quality check passes at 64K |
| 1. Mesh, groups, LSE | `cp_size` axis, `head_shard_size` at every head-degree site, three-axis agreement checks, `return_lse`, `CPLayout` and the merge with unit tests | check 4 passes and the current TP2 and TP2×SP2 tests do not change |
| 2. KV ownership and decode CP | `CPLayout` in the allocator and the plans, symmetric reservation, sink page, decode merge path, replicated rows as the only prefill mode, assertions for the features that are off | checks 1 to 3 pass and CUDA-graph decode replays with the all-gather inside |
| 3. Zigzag text prefill | zigzag chunks, K/V gather, ragged wrapper, thinker slice and span, `CPAttention` with `PassQ` and single-shot `PassKV`, the per-row rule | check 5 passes |
| 4. Chunked prefill under CP | row sets of a mixed step, the budget scale knob, the chunked prompt in check 5 | check 5 passes with chunked prefill on |
| 5. Measurement at equal GPU count | time to first token and time per output token for CP2×TP2 against TP4 at 16K, 32K and 64K, capacity in 32K and 64K sessions, the `I_tok` sweep, the pass-KV cap and crossover | the table gives the prompt length at which CP2×TP2 beats TP4, and the defaults for `I_tok` and the pass-KV cap |

**Follow-ups, in priority order, each with its trigger from [§4.6](#46-cost-model-and-levers):**

1. *Dense long-context target.* Bring up Qwen3-32B or the chosen model at 128K under CP2 and CP4. Trigger: PR 5 confirms the crossover on the thinker.
2. *Chunked pass-KV* for `L` above the cap, with a gather kernel from the paged cache and an online merge. Trigger: a measured row with `L` above the cap and `T / L` above the rule.
3. *Ring pass-KV and pass-Q* behind `CPAttention`, for the new tokens and the resident context. Trigger: a measured exchange above about 20 % of a prefill step, or a target past 64K that exceeds the workspace.
4. *Zigzag vision walks*, with graph capture per bucket and the deepstack slice. Trigger: vision prefill dominates a CP instance.
5. *Fused merge and compaction before the K/V write.* Trigger: a profile shows the merge or the sink scatter above 5 % of a layer.
6. *Length-based CP degree* (two node groups plus a conductor predicate). Trigger: short requests dominate a `cp > 1` instance.
7. *CP carved from TP*, *PD under CP*, *prefix cache under CP*, *windowed generation under CP*.
8. *Low priority:* CUDA-graph capture of zigzag text prefill, below 0.1 % of a 64K step. A cheaper `lm_head` on the decode critical path, a TP item.

---

## 11. Non-goals

1. CP on non-autoregressive nodes (DiT, encoders). Ulysses SP covers them.
2. Sliding-window, Mamba, linear-attention or attention-sink layers under CP. No in-tree AR model has them, and vLLM asserts them off too.
3. A reduction of weight memory. The weights are replicated across CP ranks by design, as in the paper.
