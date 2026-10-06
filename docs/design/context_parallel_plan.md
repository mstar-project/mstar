# Context parallelism (CP) for autoregressive nodes

Divide the token axis of one request across `cp` ranks of one instance. Then the KV cache per rank falls by `1/cp` past the head-shard limit. The prefill compute of a long prompt falls by `1/cp`. Each decode step reads `1/cp` of the resident KV per rank. The ranks of one CP group hold the same attention heads and different tokens.

Opt-in per `node_groups` entry with `cp_size`. The default `cp_size: 1` keeps the current behavior byte for byte. The Qwen3-Omni thinker is the reference model.

- **Status:** draft for review. No code is written. Pinned to M* commit `33baea08`. Lives on branch `cp-rfc`.
- **Area:** `engine/resources/{kv,attn,position,sampler}`, `distributed`, `model/base`, `model/qwen3_omni`. No worker, conductor or Rust runtime change.
- **Related:** chunked prefill plan ([chunked_prefill_plan.md](chunked_prefill_plan.md), branch `chunked-prefill-2`), issue #210 (prefix KV reuse), PR #198 (windowed generation), vLLM DCP and PCP (`vllm/v1/attention/ops/{dcp,pcp}.py`), arXiv 2411.01783 (ring attention for 1M-token prefill).
- **Background notes:** `~/context-parallelism/notes/01..06` (vLLM survey, cost model, M* extension points, model survey, decision log, phase plan) and `08`, `09` (caveat surveys, 45 and 78 items).

Notation:

- `cp` = `cp_size`, `r` = CP rank, `P` = `page_size`, `I_tok` = `cp_interleave_tokens`.
- `L` = global token count of a stream, `T` = new tokens in this step.
- `Hq_l` and `Hkv_l` = head counts of this rank after the `tp*sp` split, `D` = head dimension.

---

## 0. Decisions

The design log `notes/05_design_cp_mstar.md` records decisions D1 to D9 from 2026-10-01. This document applies them. Three later decisions are marked with their date.

| Item | Decision | Section |
|---|---|---|
| D1 | `cp_size` is a third mesh axis next to `tp_size` and `sp_size`. "CP carved out of TP" is phase 2. | §2 |
| D2 | KV ownership is interleaved by token with the knob `cp_interleave_tokens`. | §3.1 |
| D3 | Decode merges by one all-gather of `(O, LSE)` and a local LSE merge. | §4.2 |
| D4 | Prefill all-gathers the new K/V and divides the Q rows into zigzag chunks. Ring pass-KV and pass-Q are later implementations of the same interface. | §4.3, §10 |
| D5 | The CP degree is fixed per instance. Length-based routing is a follow-up. | §10 |
| D6 | One all-gather per batch through `gather_sequence` with per-request sizes. | §4.3 |
| D7 | Rank 0 broadcasts the last hidden rows over the CP group. Every rank runs `lm_head` and samples from the same seed. The token broadcast stays as the agreement check. | §5.3 |
| D8 | No Rust runtime or conductor change beyond the instance size `tp*sp*cp`. | §2 |
| D9 | Equal `cp` on the prefill and decode instances in v1. | §7 |
| 2026-10-01 | Every rank reserves the per-rank maximum page count, so allocator state stays byte-symmetric. | §3.4 |
| 2026-10-06 | Prefill over resident context (pass-Q) ships in v1, in PR 3. | §4.4 |
| 2026-10-06 | `cp_interleave_tokens` defaults to 1, which is vLLM's default. PR 4 measures 1, `P` and `4P`. | §3.1 |

One item is open: where this document lands. The options are this file on branch `cp-rfc`, a GitHub issue in the style of #210, or both.

---

## 1. Design rules

Every change in sections 2 to 6 follows from one of these five rules.

1. **The instance is `cp*sp*tp` ranks in lockstep, in row-major order `[cp][sp][tp]`.** Rank 0 leads. All other ranks follow the current `ScheduleTPNode` FIFO. No layer above the attention resource learns a new concept.
2. **`tp*sp` divides the heads. `cp` divides the tokens.** Every `KVConfig.shard()` call and every head-degree call site takes `head_shard_size = tp*sp`, never the joint world size.
3. **`stored_len` stays global. `CPLayout` derives every per-rank number from it.** Token `g` lives on rank `(g // I_tok) % cp`. Every rank reserves the per-rank maximum page count.
4. **Decode runs local paged attention, then all-gathers `(O, LSE)` and merges.** All ranks hold the same heads, so no Q exchange and no reduce-scatter is necessary. The all-gather is captured in CUDA graphs, as the SP all-gather is.
5. **Prefill runs zigzag Q chunks against the all-gathered new K/V and writes the owned pages.** Rank 0 always owns the last chunk of each request. Prompts that are too short for `2*cp` page-sized chunks use replicated prefill.

---

## 2. Configuration and mesh

### 2.1 `node_groups`

```yaml
node_groups:
  - node_names: [Thinker]
    ranks: [0, 1, 2, 3]
    tp_size: 2
    cp_size: 2        # TP groups {0,1},{2,3}; CP groups {0,2},{1,3}
```

`WorkerGraph` ([base.py](../../mstar/model/base.py), lines 69 to 130) gets `cp_size`, `_cp_ranks` and `_cp_comm_size`, in the same form as `_sp_ranks`. `tp_size` is still rewritten to the instance size. Thus `ShardingGroup`, the conductor and `rust/src/graph/shard.rs` do not change.

`get_sharding_config` (`base.py`, lines 347 to 397) validates `cp_enabled_nodes` in the same way as `sp_enabled_nodes`. A CP node declares no `shard_dim` entries. Its signals are replicated, and each rank cuts its own slice locally.

### 2.2 `JointGroups`

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

`GlobalParallelConfig` ([communication.py](../../mstar/distributed/communication.py), lines 371 to 437) builds `world_cp_groups` from `wg._cp_ranks`. It adds them to the sorted `new_group` schedule, so every rank creates the groups in the same order. Line 379 tests `cp_size > 1` together with `tp` and `sp`.

### 2.3 Call sites that change

| Site | Today | With CP |
|---|---|---|
| `KVConfig.shard()` at [manager.py](../../mstar/engine/resources/kv/manager.py) line 310 and [attn/base.py](../../mstar/engine/resources/attn/base.py) line 60 | `joint_comm_group.world_size` | `head_shard_size` |
| `get_piecewise_cuda_graph_configs` at [engine.py](../../mstar/engine/engine.py) line 512 | joint world size as head degree | `head_shard_size` |
| `agree_across_ranks` ([cuda_graph_runner.py](../../mstar/engine/cuda_graph_runner.py) lines 47 to 49), `post_warmup_validate`, `_verify_tp_async_sched_agrees` ([worker.py](../../mstar/worker/worker.py) lines 1651 to 1680) | loop over `[tp_group, sp_group]` | loop over `[tp_group, sp_group, cp_group]` |
| `BaseSampler._broadcast_tokens` ([sampler/utils.py](../../mstar/engine/resources/sampler/utils.py) lines 268 to 279) | TP group | joint group |

The `head_shard_size` line is the one line that must not be wrong. If `cp` is part of the head degree, `KVConfig.shard()` divides the heads by `cp` and drops heads with no error. The same number reaches the piecewise CUDA-graph configs. The caveat survey (`notes/08`, items 3 and 27) found both sites.

---

## 3. KV ownership

### 3.1 `CPLayout`

One pure class holds all the ownership arithmetic. It has no communication and no engine state, so its tests run on a CPU.

```python
# mstar/distributed/cp/layout.py
@dataclass(frozen=True)
class CPLayout:
    cp_size: int; cp_rank: int; page_size: int; interleave_tokens: int = 1   # I_tok; 1 = vLLM default, P = one page per turn

    def local_len(self, L: int) -> int
        # I_tok must divide P or be a multiple of P
        # base = (L // I_tok // cp) * I_tok ; rem = L - base * cp
        # return base + clamp(rem - rank * I_tok, 0, I_tok)   (vLLM utils.py:1108-1153)
    def local_pages(self, L: int) -> int                     # pages this rank uses for L tokens
    def reserve_pages(self, L: int) -> int                   # ceil(max_r local_len(L, r) / P): the per-rank MAX
    def owner_of_token(self, g: int) -> int                  # (g // I_tok) % cp
    def owned_mask(self, g: torch.Tensor) -> torch.Tensor
    def local_index(self, g: torch.Tensor) -> torch.Tensor   # (g // (I_tok*cp)) * I_tok + g % I_tok ; page = local_index // P
```

The knob `cp_interleave_tokens` in `resources.kv` sets `I_tok`. The default is 1, which is vLLM's `cp_kv_cache_interleave_size` default. With `I_tok = 1`, token `g` lives on rank `g % cp`. With `I_tok = P`, logical page `p` lives on rank `p % cp`. A physical page then holds tokens from a super-page of `cp * I_tok` global tokens.

PR 4 measures `I_tok` in 1, `P` and `4P` on three values: the write-scatter time, the number of empty shards for short prompts, and the prefix-cache hit granularity. The measured best value becomes the default.

### 3.2 `KVManager` changes

All six changes are in [manager.py](../../mstar/engine/resources/kv/manager.py).

| Function | Lines | Change |
|---|---|---|
| `_alloc` | 1754 to 1792 | Reserve `layout.reserve_pages(L)` per stream on every rank. Use the first `local_pages(L)`. At most one page per stream per rank is idle. |
| `_sequence_views` | 881 to 900 | `length = local_len(stored_len + span)` and `page_idxs = page_indices[:local_pages]`. `SequenceView` ([plan.py](../../mstar/engine/resources/kv/plan.py) lines 65 to 78) gets `global_length`, `global_to_compute` and `q_rows`. `to_compute` becomes the local Q-row count. |
| `_compute_plan_state` | 901 to 954 | Compute the global index `g` per token as today. Owned tokens map through `local_index(g)` to `(page, offset)`. Non-owned tokens get `token_to_page = SINK_PAGE` and `token_to_cache = 0`. |
| `_decode_plan_state` | 955 to 979 | The owner of the new token is `owner_of_token(stored_len)`. Other ranks write to `SINK_PAGE`. One H2D copy, as today. |
| `_assert_symmetric_free_pages` | 1609 to 1633 | Add the CP group. The assertion stays strict. `assert_pages_conserved` (lines 1545 to 1606, debug only) indexes with `local_pages(stored_len)`. |
| `KVConfig` | [config.py](../../mstar/engine/resources/kv/config.py) lines 41 to 110 | Carries `cp_size`, `cp_rank`, `cp_interleave_tokens`. `max_num_pages` stays a per-rank knob. |

### 3.3 Position ids

`PositionManager._build_pos_ids` ([position/manager.py](../../mstar/engine/resources/position/manager.py) lines 333 to 343) emits one contiguous range per view. Under CP prefill the local rows are two chunks. Thus the manager turns `q_rows` into explicit positions through the current `PositionStep.pos_ids` override. The counters advance by the global `Segment.span` (lines 282 to 318) and stay symmetric. The thinker uses `custom_pos_ids`, which `split_inputs` already cuts.

### 3.4 Why every rank reserves the maximum

Exact per-rank page counts differ by up to one page per stream per rank. The free-page totals then differ. A request can then fail allocation on one rank and pass on another. `_handle_allocation_failure` ([worker.py](../../mstar/worker/worker.py) lines 1581 to 1639) assumes that every rank raises on the same batch. TP-async speculation also voids heads from each rank's own admission verdict. A different verdict on one rank stops the next collective.

The alternative is exact counts plus an `all_reduce(min)` on admission. That adds a collective to the scheduler path. The reservation of the maximum costs one page per stream per rank and keeps every current symmetry assertion strict.

### 3.5 Forks, offload, prefix cache

- Forks (`_reserve_fork`, `_apply_fork`, lines 1681 to 1752) copy local pages to local pages. Both streams start at logical token 0, so ownership matches. Only the copy count changes, to the reserved page count.
- Offload (lines 1268 to 1471, [cpu_page_pool.py](../../mstar/engine/resources/kv/cpu_page_pool.py)) is per rank. Victim selection uses a wall-clock LRU and is already uncoordinated across TP ranks (`worker.py` lines 1594 to 1606). Under CP the per-rank `reclaimable()` values also differ. CP nodes require `cpu_offload_pages == 0` in v1.
- `enable_prefix_cache` (lines 407 to 427) keys off `_world_size`, which is the joint size and already includes `cp`.

---

## 4. Attention

### 4.1 LSE and the merge

Both FlashInfer wrappers return the log-sum-exp (LSE) on request. The merge is a pure function with no communication inside.

```python
# mstar/distributed/cp/merge.py
def merge_partial_attention(o: Tensor[N, B, H, D], lse: Tensor[N, B, H]) -> Tensor[B, H, D]:
    # lse <- -inf where NaN or +inf (empty shard / padding row)
    # m = max_n lse ; w_n = exp(lse_n - m) ; o = sum_n w_n * o_n / sum_n w_n

def cp_all_gather_merge(cp_group, o_local, lse_local) -> Tensor[B, H, D]:
    # all_gather both along a new leading dim into a static buffer, then merge_partial_attention
```

- `run(q, kv, return_lse=False)` on both wrappers ([wrappers.py](../../mstar/engine/resources/attn/wrappers.py) lines 146 to 202 and 281 to 331). FlashInfer 0.6.18 supports `return_lse=True` on both paged wrappers with the natural-log base.
- The LSE stays in fp32 from the kernel to the merge. vLLM commit f05603fa28 crashed when it packed the LSE into bf16.
- The manager checks the log base of the kernel one time at startup against a small reference. vLLM commit 5b4cb69523 corrected a base mismatch that corrupted the softmax denominator with no error.
- Every CP rank ends with the full merged `O` for its `Hq_l` heads. That is the input that the row-parallel `o_proj` expects. The TP all-reduce in `o_proj` does not change.

### 4.2 Decode

`FlashInferManager.run` ([flashinfer.py](../../mstar/engine/resources/attn/flashinfer.py) lines 206 to 239) gets one branch when `cp_group.world_size > 1`.

1. `o, lse = wrapper.run(q, kv, return_lse=True)`.
2. A row with zero local KV gets one `SINK_PAGE` in the plan, so the kernel never sees an empty row. Such rows are streams with fewer than `cp` tokens and CUDA-graph padding rows. The plan sets their `lse` to `-inf` from its `local_length == 0` mask. This is vLLM's `mask_dcp_empty_shards_` (`ops/dcp.py` lines 72 to 96).
3. `o = cp_all_gather_merge(cp_group, o, lse)`. The all-gather target is a static buffer per bucket, because a captured graph bakes the address. vLLM commit 9fd737badc made the same correction.

The all-gather moves `cp * B * Hq_l * (D + 1) * 2` bytes per layer per step. That is a few hundred KB. The CUDA graph replays the all-gather in the same way as the SP all-gather ([sequence_parallel.py](../../mstar/model/components/distributed/sequence_parallel.py) lines 109 to 121).

### 4.3 Prefill, `stored_len == 0`

Cut `[0, T)` into `2*cp` chunks. Rank `r` takes chunks `r` and `2cp-1-r`. Each chunk is one FlashInfer ragged row.

```python
# mstar/distributed/cp/prefill.py
def zigzag_chunks(T: int, cp: int) -> list[list[tuple[int, int]]]   # per rank, two (start, end)
def gather_new_kv(cp_group, k_loc, v_loc, local_rows, T) -> (k_full, v_full)
    # padded all_gather (gather_sequence), then one precomputed index restores global order
def ragged_rows(chunks) -> (qo_indptr, kv_indptr)
    # each chunk is a row with kv_len = chunk_end, so the end-aligned causal mask is exact
```

Per layer, inside `AttentionCallable.__call__` ([convenience.py](../../mstar/engine/resources/convenience.py) lines 52 to 70):

1. `k_full, v_full = gather_new_kv(...)`. Q, K and V travel in one `gather_sequence` call.
2. `kv.write_kv(k_full, v_full)`. The plan from §3.2 sends non-owned rows to `SINK_PAGE`.
3. `attn.run(q_local, ...)` through a new `FlashInferRaggedPrefillWrapper` around `BatchPrefillWithRaggedKVCacheWrapper`, over `k_full, v_full` with the chunk rows. The ragged path reads the full new K/V, so no cross-rank merge is necessary.
4. If `T % (2*cp) != 0`, the chunks differ by one row. `gather_sequence` pads, and the index drops the pad.
5. If `T < 2 * cp * P`, the step uses replicated prefill. Every rank computes all `T` rows and writes only its owned pages. PR 2 ships with this as the only prefill mode.

Zigzag balances the causal FLOPs to within one chunk. It puts the last chunk on rank 0, which §5.3 uses.

### 4.4 Prefill over resident context, `stored_len > 0`

This path ships in v1, in PR 3. The `stored_len == 0` assertion exists only in PR 2, where replicated prefill is the only prefill mode.

A prefill that extends a stream with resident tokens must also attend to the resident pages. Examples are the thinker's second `prefill_text` after the modality walks ([qwen3_omni_model.py](../../mstar/model/qwen3_omni/qwen3_omni_model.py) lines 836 to 875), a multi-turn chat, and every chunk after the first one under chunked prefill. Each rank holds only a subset of the resident pages. The FlashInfer end-aligned causal mask computes the query position from `kv_len - qo_len`. Over a partial page list that position is wrong, and the kernel gives an incorrect result with no error.

The correct method has three parts. vLLM uses the same method in `flash_attn.py` lines 1547 to 1700.

1. **Part A.** All-gather the new Q rows across CP in the same collective as K and V. Run non-causal paged attention of all `T` new Q rows against the local resident pages, with LSE. Every rank holds a different slice of the resident keys, so every rank must see every new query. A rank that runs only its local Q chunk here gives each query `1/cp` of the resident context, which is wrong.
2. **Part B.** Run causal ragged attention of the local Q chunk against the all-gathered new K/V, with LSE. This is the same call as in §4.3.
3. **Part C.** All-gather the Part A partials `(O_A, LSE_A)` across CP. Keep the rows of this rank's chunk. Merge them with the Part B partial through `merge_partial_attention`.

Every rank runs Part A, also with no resident pages, and reports `lse = -inf` in that case. Thus the collective is unconditional per stream. Decode is this path with one query and no Part B. vLLM gathers Q along heads, because its DCP ranks hold different Q heads. M* gathers Q along tokens, because CP ranks hold the same heads and different chunks.

### 4.5 Cost per layer

| Step | Communication | Redundant compute |
|---|---|---|
| Prefill | one all-gather of `(cp-1)/cp * T * Hkv_l * D * 4` bytes, no overlap (thinker TP2×CP2 at 64K: 32 MB; Llama-70B TP8 at 1M: 512 MB) | embeddings and MRoPE tables for all `T` tokens before the slice; `(cp-1)/cp` of the K/V scatter bandwidth lands in `SINK_PAGE` |
| Decode | one all-gather of `cp * B * Hq_l * (D+1) * 2` bytes plus one elementwise merge | MLP, `lm_head` and sampler run `cp` times |

The thinker's MoE layer has a per-layer floor that CP does not reduce. From about 128 local tokens, almost all 128 experts get a token. Each rank at TP2 then reads about 604 MB of expert weights per layer (302M parameters in bf16). That is about 0.25 ms per layer at 2.5 TB/s, on every CP rank, in prefill and in decode. The floor hides the all-gather latency for short prompts and limits the decode win.

| Prompt (thinker TP2×CP2) | Received per layer | Comm per layer | Layer time per rank | Comm share |
|---|---|---|---|---|
| 512 tokens | 0.26 MB | ~16 µs (latency) | ~0.25 ms (weight-read floor) | ~6 % |
| 4K tokens | 2 MB | ~20 µs | ~0.3 ms (floor plus FLOPs) | ~7 % |
| 64K tokens | 32 MB | ~105 µs | ~18 ms (attention FLOPs) | ~0.6 % |

Decode at `B = 8` with a 32K context: CP2 saves about 2.5 ms of KV reads per step and pays about 0.7 ms of all-gather latency. The 12 ms of expert reads stay. The net win is about 12 % of the step. These are pen-and-paper values. PR 4 measures them.

---

## 5. Model-facing API

A CP-capable `ARNodeSubmodule` does three things in prefill. Decode needs none of them. The thinker ([qwen3_omni/submodules.py](../../mstar/model/qwen3_omni/submodules.py)) is the reference, and it joins `cp_enabled_nodes`.

### 5.1 Slice the inputs per rank

`prepare_inputs` (lines 376 to 540) builds the full `ARNodeInputs` as today. For a CP prefill that is not in replicated mode, it then keeps `split_inputs(start, end)` for the two chunks of this rank, concatenated. Positions, deepstack tensors and `mrope_pos_advance` stay global, because `split_inputs` cuts them with the rows. The thinker overrides `split_inputs` to cut its `tensor_inputs`, which the base class refuses to do ([submodule_base.py](../../mstar/model/submodule_base.py) lines 679 to 686).

### 5.2 Declare the global span

`declare_step` (lines 540 to 585) keeps `Segment.span` as the full `input_seq_len`. `SubmoduleStep` gets `cp_rows`, so the attention resource plans the local Q rows.

### 5.3 Sample

Under zigzag, the last token of each request is the last row of the second chunk on rank 0. Thus `select_last_hidden` is correct on rank 0 only. Rank 0 broadcasts the `[bs, hidden]` last rows over the CP group. That is one latency-bound collective of tens of KB per prefill step. Every rank then runs the real `lm_head` and `sampler.sample` from the same seed, exactly as TP ranks do today. The current joint-group `_broadcast_tokens` stays as the agreement check.

Decode needs no hidden broadcast, because every CP rank holds the full hidden state. The redundant `lm_head` on the followers adds nothing to the critical path, because the step time is the time of the leader.

### 5.4 `thinker_states`

`thinker_states` (`layer_0_embed`, `layer_n_hidden`) are per-rank shards under CP prefill. The thinker all-gathers them and restores the sequence order only when `audio_output` is set.

---

## 6. Followers and CUDA graphs

1. Every branch that guards a collective derives its condition from the step declaration or the KV plan, never from a local tensor shape. "This rank has resident pages" differs per rank. "This stream has resident tokens" does not.
2. `zigzag_chunks`, `CPLayout` and the replicated-prefill threshold are pure functions of `(T, stored_len, cp, P, I_tok)`. Every rank knows all five values.
3. The decode buckets capture the all-gather of §4.2. `_buckets_captured_everywhere` ([cuda_graph_runner.py](../../mstar/engine/cuda_graph_runner.py) lines 487 to 512) already ANDs the capture result across the joint group, which §2 extends to the CP axis.
4. Padding rows read garbage from `SINK_PAGE`. Their LSE is finite garbage that affects only their own rows. The NaN and infinity guard in the merge is mandatory, so that a `NaN` cannot pass through `max`.
5. CP prefill runs eager in v1. `PREFILL_TOKEN_BUCKETS` stop at 2048 tokens, so long prompts run eager today. A graph saves 5 to 10 µs per kernel launch, which is below 0.1 % of a 64K prefill.

The followers get no new wire data. `ScheduleTPNode` ([ipc_format.py](../../mstar/utils/ipc_format.py) lines 105 to 111) carries the node, the walk, the request ids and the speculation flags. A follower computes its chunk layout and its page ownership from those values and its own rank.

---

## 7. Interactions with other work

Three features compose with no new work:

1. **Chunked prefill** ([chunked_prefill_plan.md](chunked_prefill_plan.md)). A chunk is a prefill with `stored_len > 0`, so it uses §4.4. Inside a chunk, `split_inputs` composes: the chunk first, then zigzag. `ChunkProgress` is per rank and identical by determinism.
2. **Speculation** (TP async follow). The flags travel on `ScheduleTPNode` unchanged. A CP decode step adds one lockstep collective per layer, as the TP all-reduce does.
3. **Prefix caching** (#210). Off at world size > 1 today. Under CP a cached prefix is a set of per-rank page lists under one hash, with `effective_page = cp * I_tok` tokens (vLLM `kv_cache_utils.py` lines 718 to 800). The per-rank refcount of #210 does not change.

Five features are asserted off in v1. Each has a named follow-up in §10.

1. **Windowed generation** (#198). `_release_oldest_locked` (`manager.py` lines 1208 to 1220) deletes the front of `page_indices`. That shifts every later logical page and its owner. The fix is a per-stream logical base offset. In v1, `retention is None` on CP nodes.
2. **PD disaggregation.** Equal `cp` on both sides keeps the per-rank pull ([transfer.py](../../mstar/engine/resources/kv/transfer.py) lines 323 to 363) unchanged. But the conductor merges `resource_publish_info` from the first instance rank only ([conductor.py](../../mstar/conductor/conductor.py) lines 1283 to 1296), and CP page lists differ by rank. The fix is to publish the list of every rank. A different `cp` on the two sides needs a re-interleave. vLLM requires that one degree divides the other (`base_worker.py` lines 2485 to 2492).
3. **Other attention backends.** `attn/dense.py` line 178 asserts `view.start == 0`. `attn/xpu.py` pads with `SINK_PAGE`. `attn/cross.py` holds replicated encoder KV. CP nodes require FlashInfer. Cross-attention labels stay replicated.
4. **Multi-token decode** (`span > 1` on a paged step). No model does it today. The new tokens can belong to different owners and need §4.4.
5. **Ulysses SP with CP on an AR node.** The two compose in principle. SP on an AR paged node is unused today (Cosmos3 only), so `sp*cp > 1` on an AR node is rejected.

---

## 8. Correctness checks

Checks 1 to 4 run in CI on one GPU. Check 5 needs 2 to 4 GPUs.

1. For every stream, `sum_r local_len_r(L) == L`, and the owned-token sets partition `[0, L)`.
2. The concatenation of the ranks' `read_tokens` in layout order reproduces the global K/V that was written. The test runs random K/V through `cp` managers on one GPU.
3. `merge_partial_attention` over any division of a sequence equals full attention (bf16 `atol 1e-2`, fp32 `1e-5`). The test includes an empty shard and `N = 1`.
4. `cp_size: 1` produces `KVPlanState` tensors identical to `main` (golden check). Thus PR 1 and PR 2 do not change behavior when CP is off.
5. Thinker text under `cp2`, `cp4` and `tp2*cp2` equals the TP1 baseline under the protocol in `test/distributed/test_qwen3omni_thinker_tp2_vs_tp1.py` (strict, loose, smoke) on 2K, 16K and 32K prompts.

---

## 9. Caveats

Twenty-seven items in five groups, in the order of what occurs if an item is missed. Group A causes a stop or an incorrect output with no error.

### A. Stops or incorrect output with no error (PR 1 and PR 2)

1. `KVConfig.shard()` and `engine.py` line 512 must divide the heads by `tp*sp`, not by the joint world size. Otherwise heads are dropped with no error. (§2.3)
2. Exact per-rank page counts make admission and OOM differ across ranks and stop the next collective. Symmetric reservation prevents it. (§3.4)
3. Any step with `to_compute > 1` over paged CP KV needs §4.4. Without it the FlashInfer end-aligned causal mask is incorrect with no error. PR 2 asserts `stored_len == 0`.
4. Every `[tp, sp]` loop and `_broadcast_tokens` must include the CP axis. Otherwise the capture agreement and the token broadcast skip ranks. (§2.3)
5. The LSE must be fp32 and natural-log. The all-gather target must be a static buffer under capture. (§4.1, §4.2)

### B. Plans and allocator

6. Two different "local" token sets exist in prefill: the Q rows (zigzag) and the stored KV (token-interleaved). They are independent only because K/V is all-gathered. The write plan filters by ownership, never by Q rows. (§3.2, §4.3)
7. `SINK_PAGE` writes cause benign write races, wasted bandwidth, and garbage or NaN that the merge must guard against. (§6)
8. Streams with fewer than `cp` tokens, and padding rows, have zero local KV on some ranks. Plan one `SINK_PAGE` and mask the LSE. (§4.2)
9. `_build_pos_ids` emits a contiguous range per view. CP prefill rows are two chunks. (§3.3)
10. `assert_pages_conserved` indexes local page lists by global counts (debug only). (§3.2)

### C. Prefill

11. Short prompts leave empty shards: `-inf` mask plus replicated prefill. (§4.3)
12. The sampler relies on zigzag to put the last chunk on rank 0. `zigzag_chunks` must guarantee it, and a test must pin it. (§5.3)
13. The thinker's multimodal schedule is `[prefill_text] + modality walks + [prefill_text]` on one label (`qwen3_omni_model.py` lines 836 to 875). Every multimodal request hits caveat 3 and runs through §4.4. The first PR 4 benchmark is text-only, so the numbers stay comparable with the cost model.
14. The thinker carries `tensor_inputs` (deepstack) that the base `split_inputs` refuses to cut (`submodule_base.py` lines 679 to 686). It needs the override. (§5.1)
15. `thinker_states` are shards under CP prefill. All-gather only when `audio_output`. (§5.4)
16. `mrope_pos_advance` (`submodules.py` lines 555 to 560) advances the positions by more than the span. Positions are counters independent of KV lengths. Thus `stored_len` and the position counter must never derive from each other.

### D. Graphs, kernels, names

17. CP prefill is eager. CP decode must capture the all-gather. (§6)
18. NCCL `new_group` order: the CP groups must be in the sorted union on every rank. (§2.2)
19. `WorkerGraph.tp_size` and `ShardingGroup.tp_size` mean "instance size". Document it. Do not rename in this change.
20. MoE routing per rank sees only the local tokens (`components/moe.py`, TP-sharded experts with all-reduce, no EP). The expert load differs per rank. The result does not.
21. Only the FlashInfer backend. Dense, xpu and cross attention assert `cp == 1`. (§7)

### E. Features asserted off in v1

22. Retention and windowed releases shift the logical page indices. (§7)
23. PD requires equal `cp`, and the conductor merges publish info from the first instance rank only. (§7)
24. Offload victim selection is per rank and uncoordinated: `cpu_offload_pages == 0`. (§3.5)
25. Multi-token decode can cross owners. (§7)
26. SP with CP on an AR node. (§7)
27. The prefix cache stays off at world size > 1, as today. (§3.5)

Sources: the design pass and the two surveys. `notes/08` has 45 items and `notes/09` has 78 items. Nothing in the surveys contradicts the design. The surveys found items 2, 9, 10, 24 and the `engine.py` line 512 site.

---

## 10. Delivery plan and follow-ups

The estimates assume one engineer with 2 to 4 GPUs on one node. The total is about 3 weeks to a measured CP2 and CP4 thinker.

**PR 1. Mesh, groups, LSE (2 days, no behavior change).**
- `cp_size` in YAML and `WorkerGraph`. `cp_group` on `JointGroups`, `WorkerParallelGroups`, `GlobalParallelConfig`.
- `head_shard_size` at the two `shard()` call sites and `engine.py` line 512. Every `[tp, sp]` loop gets `cp`. `_broadcast_tokens` over the joint group.
- `return_lse` on both wrappers. `mstar/distributed/cp/{layout,merge}.py` with unit tests.
- Done when: no code path reads `cp_size > 1`, correctness check 4 passes, and the current TP2 and TP2×SP2 tests do not change.

**PR 2. KV ownership and decode CP (5 days).**
- `CPLayout` in `_alloc` (symmetric reservation), `_sequence_views`, `_compute_plan_state`, `_decode_plan_state`. `SINK_PAGE` row for empty shards.
- Assertions for `cpu_offload_pages`, retention, PD, dense and xpu.
- `FlashInferManager.run` merge path. Replicated prefill as the only prefill mode. `configs/qwen3omni_thinker_cp2.yaml`.
- Done when: correctness checks 1 to 3 pass, the TP1-vs-CP2 protocol passes on short prompts, and CUDA-graph decode replays with the all-gather inside.

**PR 3. Prefill CP (7 days, includes §4.4).**
- `zigzag_chunks`, `gather_new_kv`, `ragged_rows`, `FlashInferRaggedPrefillWrapper`.
- Thinker `prepare_inputs` slice, `declare_step.cp_rows`, explicit `pos_ids` for generic models, last-hidden-row broadcast before the sampler, `thinker_states` all-gather.
- §4.4 pass-Q path: Q in the K/V gather. Part A over all `T` gathered Q rows against the local resident pages. Part B over the gathered new K/V. `(O_A, LSE_A)` all-gather and merge. Unconditional per stream.
- Done when: correctness check 5 passes on 16K and 32K text prompts for `cp2`, `cp4` and `tp2*cp2`.

**PR 4. Measurement (2 days).**
- `benchmark/cp_ttft.py`: time to first token at 4K to 64K for TP1, TP2, TP4, CP2, CP4 and TP2×CP2. Time per output token at `B` in {1, 8, 32}. Capacity in 32K sessions. `I_tok` in {1, `P`, `4P`}.
- Results next to the predictions of the cost model (`notes/model_cp.py` with the thinker's shape).
- Done when: the table gives the prompt length at which CP2 and CP4 beat TP2 on this hardware, and the default `I_tok`.

**Follow-ups, each with its trigger:**

1. *Pass-KV for resident context and the pass-Q/pass-KV heuristic* (`T / L` against `Hkv_l / Hq_l`). Trigger: the first MLA model on a CP node, or a measured second-turn prefill with `T / L` above `Hkv_l / Hq_l`.
2. *Ring pass-KV and pass-Q* behind the same `AttentionCallable` branch. Trigger: a measured all-gather above about 20 % of a prefill step at the target length, or a 1M-token target that exceeds the transient memory.
3. *Fused Triton merge, compaction before the K/V write.* Trigger: NVTX shows the merge or the sink scatter above 5 % of a layer. With `I_tok = 1` the owned rows of the gathered buffer are a strided view, so the compaction is free.
4. *Elastic or length-based CP degree* (two node groups plus a conductor predicate, notes `05` F1). Trigger: short requests dominate a `cp > 1` instance. For the thinker the MoE weight-read floor keeps the short-prompt loss near 6 %, so dense models are the first candidates.
5. *CP carved from TP* (vLLM DCP placement behind a `heads_identical=False` flag), *PD with different `cp`*, *prefix cache under CP* (with #210), *windowed generation under CP* (with #198).
6. *CP prefill CUDA-graph capture* (lowest priority). The recipe is the decode recipe: bucket `T`, fix the zigzag chunk layout per bucket, and give each bucket static gather buffers. Trigger: a profile shows launch latency above 5 % of a CP prefill step, after follow-ups 1 to 5.
7. *Cheaper `lm_head` on the critical path* (low priority, a TP item). The thinker's `lm_head` reads about 310 MB per rank per step at TP2, about 0.12 ms. Options: a vocabulary shard across TP with one all-gather of the top candidates, or a fused top-k over the sharded logits. Trigger: a profile shows `lm_head` above 5 % of a decode step.

**Prerequisites for pass-KV (follow-ups 1 and 2).** Two different mechanisms carry the name.

- *Chunked all-gather pass-KV for resident context* (vLLM MLA style, `mla_attention.py` lines 3027 to 3110). It uses the current all-gather only. It needs four parts:
  - (a) A gather kernel from the paged cache into a contiguous workspace. vLLM has `cp_gather_cache`. M* has only `read_tokens` in [cache.py](../../mstar/engine/resources/kv/cache.py).
  - (b) A workspace with a size from the global `stored_len` in fixed chunks, so that every rank gathers the same bytes.
  - (c) A chunk loop that runs ragged attention per chunk and merges by LSE online.
  - (d) The policy object that selects pass-Q or pass-KV from `(T, L, Hq_l, Hkv_l)`.
- *Ring pass-KV for the new tokens* (the paper). Comm overlaps compute. It needs all of the above plus three more parts. Point-to-point send and receive in `distributed/communication.py` (M* has `all_gather` and `all_to_all` only). A second CUDA stream with events. Double-buffered K/V. Prefill is eager, so graph capture of point-to-point is not required.

The `CPAttention` interface from decision D4 lands in PR 3 with the pass-Q path as its first implementation. Both pass-KV variants are later implementations of the same interface.

---

## 11. Benchmark acceptance

Two predictions from the cost model are the acceptance checks for PR 4.

- Thinker geometry: 48 layers, `Hq = 32`, `Hkv = 4`, `d = 128`, from the HF `config.json` of Qwen3-Omni-30B-A3B. The dataclass defaults in [qwen3_omni/config.py](../../mstar/model/qwen3_omni/config.py) lines 41 to 71 are placeholders that the loader overrides.
- Hardware: one 8×H100 node, Slurm partitions `team1` or `guest`.
- Prediction 1: CP2 prefill is about 1.9× faster than the TP1-equivalent compute above about 24K tokens.
- Prediction 2: decode CP is slower than TP at `B = 1`. The break-even is between 75K and 0.3M resident tokens per instance, as the two cost models give different constants.
- The per-layer all-gather time comes from the current NVTX ranges.

---

## 12. Non-goals

1. Ring attention (pass-KV, pass-Q) in v1. No point-to-point primitive exists, and the all-gather is safe under CUDA graphs. It is a follow-up behind the same seam.
2. CP on non-autoregressive nodes (DiT, encoders). Ulysses SP covers them.
3. Sliding-window, Mamba, linear-attention or attention-sink layers under CP. No in-tree AR model has them, and vLLM asserts them off too (`kv_cache_interface.py` lines 866 to 870, `flashinfer.py` lines 1409 to 1414).
4. Changes to conductor routing or to the Rust runtime. The instance abstraction absorbs the new axis.
5. A reduction of weight memory. The weights are replicated across CP ranks by design, as in the paper.
