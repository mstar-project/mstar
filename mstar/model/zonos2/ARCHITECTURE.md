# ZONOS2 Model Architecture: Computation Graph Decomposition

This document maps the ZONOS2 model's hierarchical computation graph to the currently implemented components in `mstar.model.components`, and identifies gaps.

---

## Part 1: Hierarchical Computation Graph Decomposition

### Level 0: Top-Level Request Flow

```
Sequential {
  1. Input Tokenization (text → bytes)
  2. Prefill Phase
  3. Decode Loop (Dynamic)
}
```

---

### Level 1: Prefill Phase

```
Sequential {
  1. MultiEmbedding
  2. emb_norm
  3. TransformerBlockLoop (Static, N iterations)
  4. out_norm
  5. MultiOutputHead
  6. Softcap
  7. Extract last-token logits
}
```

---

### Level 2: MultiEmbedding (Parallel across codebooks)

```
Parallel {
  ├─ Embed column 0 (codebook 0)
  ├─ Embed column 1 (codebook 1)
  ├─ ...
  ├─ Embed column 8 (codebook 8)
  └─ Embed column 9 (text)
} → Sequential { Sum all embeddings }
```

**Pattern**: Parallel independent operations → Sequential reduction (sum)

---

### Level 2: TransformerBlockLoop (Static loop, N iterations)

```
StaticLoop (N iterations) {
  Sequential {
    1. attention_norm
    2. Attention
    3. ffn_norm
    4. FeedForward OR MoEFeedForward
  }
}
```

**Pattern**: Unrolled sequential composition of identical blocks

---

### Level 3: Attention (Parallel across heads)

```
Sequential {
  1. Linear projections (wq, wkv) → Q, K, V
  2. RoPE positional encoding
  
  3. Parallel {
       For each head (H heads):
         Sequential {
           ├─ QK normalization
           ├─ Temperature scaling (per-head learnable param)
           ├─ FlashAttention (softmax, matmul with V)
           └─ Headwise gating: sigmoid(gater(x)) × attn_output
         }
     }
  
  4. Concatenate heads
  5. wo projection (output)
  6. AllReduce (if tensor parallel)
}
```

**Key**: Q/K/V projections are linear (single fused GEMM), then per-head computation parallelizes.

---

### Level 3: FeedForward (Simple sequential)

```
Sequential {
  1. w_in: hidden → intermediate
  2. GELU activation
  3. w_out: intermediate → hidden
}
```

---

### Level 3: MoEFeedForward (Parallel expert computation)

```
Sequential {
  1. Router: hidden → expert_logits (sequential)
  
  2. Expert Dropout Augmentation (optional, sequential blend)
  
  3. Parallel {
       For each expert (E experts):
         Sequential {
           ├─ gate_up_proj (fused w1 || w3)
           ├─ SiLU activation
           ├─ down_proj (w2)
           └─ Gating: sigmoid(gate) × output
         }
     }
  
  4. Combine / AllGather (weighted sum across experts)
  5. AllReduce (if tensor parallel)
}
```

**Key**: Sparse activation — only top-K experts compute per token. Dispatch/combine involve data movement.

---

### Level 2: MultiOutputHead (Parallel across codebooks)

```
Sequential {
  1. Linear projection: hidden → (9 × 1026)
  2. Reshape to (*, 9, 1026)
}
```

Or decomposed:
```
Parallel {
  ├─ Linear CB0 projections
  ├─ Linear CB1 projections
  ├─ ...
  └─ Linear CB8 projections
} → Sequential { Stack outputs → (*, 9, 1026) }
```

---

### Level 1: Decode Loop (Dynamic, data-dependent termination)

```
DynamicLoop (until EOS condition) {
  Sequential {
    1. Frame tokenization (previous codebook outputs)
    2. MultiEmbedding (same structure as prefill)
    3. TransformerBlockLoop (same structure, with KV cache)
    4. out_norm
    5. MultiOutputHead
    6. Softcap
    7. Sampling
    8. EOS Check (conditional branch → terminate or continue)
  }
}
```

**Key difference from prefill**: Autoregressive — output of frame N becomes input to frame N+1.

---

### Level 2: Sampling (Parallel per-codebook)

```
Sequential {
  1. Repetition penalty (per-codebook filtering)
  2. Temperature scaling (per-sequence, broadcast)
  3. Top-k filtering
  4. Top-p / Min-p filtering
  5. Softmax per codebook
  
  6. Parallel {
       For each codebook (9 codebooks):
         Multinomial sample (independent per-codebook RNG)
     }
}
```

---

## Part 2: Parallelism Hierarchy Summary

| Level | Type | Examples | Notes |
|-------|------|----------|-------|
| **1** | Static Loop | N transformer blocks | Critical path; sequential across blocks |
| **2** | Sequential | Norm → Attention → Norm → FFN | Within each block |
| **3** | Parallel | 10 embeddings, H heads, E experts, 9 codebooks | Can execute concurrently |
| **4** | Sequential | QK norm → attention → gating (within head) | Sub-operations within parallel units |
| **5** | Dynamic Loop | Autoregressive decode until EOS | Inherently sequential; each frame depends on prior |

**Bottleneck**: Sequential transformer layers (critical path for latency).

**Parallelism sources**:
- **Codebooks** (9-way): multi-output prediction, independent sampling
- **Heads** (H-way): standard attention parallelism
- **Experts** (E-way, MoE only): sparse activation across routed experts

---

## Part 3: Implementation Mapping

### ✅ Implemented Components

#### Core Layers (Single-Block)

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| `Attention` | `attention.py` | ✅ Complete | QK-norm, GQA, RoPE, qkv consolidation, per-head gating |
| `ParallelAttention` | `distributed/attention.py` | ✅ Complete | TP-aware: QKV sharded, o_proj all-reduces |
| `RMSNorm` | `norm.py` | ✅ Complete | Llama-style (weight) and Gemma-style (1+weight) |
| `AdaRMSNorm` | `norm.py` | ✅ Complete | Adaptive conditioning via dense(cond) → scale/shift/gate |
| `DecoderLayer` | `decoder_layer.py` | ✅ Complete | Standard pre-norm: norm→attn→residual, norm→ffn→residual |
| `GatedDecoderLayer` | `decoder_layer.py` | ✅ Complete | Pre-norm with gated residuals for AdaRMS |

#### Feedforward & MLP

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| `MLP` | `mlp.py` | ✅ Complete | Simple 2-layer: in→act→out |
| `GatedMLP` | `mlp.py` | ✅ Complete | SwiGLU: separate gate/up, consolidation post-load |
| `FusedGatedMLP` | `mlp.py` | ✅ Complete | SwiGLU: fused gate+up from construction |
| `ParallelGatedMLP` | `distributed/mlp.py` | ✅ Complete | TP-aware gated MLP |

#### Linear Projections

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| `FusedColumnLinear` | `linear.py` | ✅ Complete | Concatenates output shards; supports weight_loader for checkpoint slicing |
| `ColumnParallelLinear` | `distributed/linear.py` | ✅ Complete | Output sharded; gathers on return if requested |
| `RowParallelLinear` | `distributed/linear.py` | ✅ Complete | Row-parallel with AllReduce on output |
| `QKVParallelLinear` | `distributed/linear.py` | ✅ Complete | Fused QKV projection with TP sharding |
| `MergedColumnParallelLinear` | `distributed/linear.py` | ✅ Complete | Fused gate/up with TP sharding |

#### MoE Components

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| `TopKRouter` | `moe.py` | ✅ Complete | Softmax top-k routing; returns logits, weights, indices |
| `SparseMoeBlock` | `moe.py` | ✅ Complete | Top-K dispatch to experts (no shared expert) |
| `SparseMoeBlockWithSharedExpert` | `moe.py` | ✅ Complete | Top-K + sigmoid-gated shared expert |
| `ParallelSparseMoeBlock` | `moe.py` | ✅ Complete | TP-aware sparse MoE (no shared) |
| `ParallelSparseMoeBlockWithSharedExpert` | `moe.py` | ✅ Complete | TP-aware sparse MoE + shared expert |
| `dispatch_experts_fused` | `moe.py` | ✅ Complete | Per-expert SwiGLU dispatch (fallback for Triton) |

#### Embedding

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| `VocabParallelEmbedding` | `distributed/embedding.py` | ✅ Complete | Row-parallel embedding; all-reduces to replicate |

---

### ❌ Missing Components

#### 1. MultiEmbedding & MultiOutputHead

**What it is**: 
- Parallel embedding lookup across 10 columns (9 audio codebooks + text token), summed into single hidden state
- Inverse for output: parallel projection to 9 codebook logits

**Current workaround**: 
- Models likely stack VocabParallelEmbedding calls or use manual embedding + sum

**Rationale for component**:
- Explicit dataflow: "Parallel {10 embeddings} → Sum"
- Checkpoint loading: weight_loader pattern for per-column embedding tables
- TP integration: seamless distribution of 10 embedding shards across ranks

**Sketch**:
```python
class MultiEmbedding(nn.Module):
    """N independent embeddings summed into single output."""
    def __init__(self, column_configs: dict[str|int, dict]):
        # column_configs: {"cb0": {"vocab": 1024, "dim": 4096}, ...}
        for col_id, cfg in column_configs.items():
            self.register_module(f"embed_{col_id}", 
                VocabParallelEmbedding(...))
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (seq_len, n_columns)
        # output: (seq_len, hidden_size) — sum of all embeddings

class MultiOutputHead(nn.Module):
    """Output projection to N independent output vocabularies."""
    def __init__(self, column_configs: dict[str|int, dict]):
        # similar structure, but for projections
        # output: (*, n_columns, vocab_size)
```

---

#### 2. Expert Dropout Augmentation (EDA) in MoE

**What it is**: 
- ZONOS2's MoE layers (after the first) blend the previous layer's router hidden states with the current layer's:
  - `router_hidden = router_states_prev * scale + router_hidden_current`
  - This threads router context across consecutive MoE layers

**Current state**: 
- `TopKRouter` is stateless; each layer routes independently
- No mechanism to propagate router state across layers

**Rationale for component**:
- Improves expert specialization by maintaining context across layers
- Requires explicit state threading between MoE blocks

**Sketch**:
```python
class MoEBlockWithEDA(nn.Module):
    """Sparse MoE with Expert Dropout Augmentation.
    
    Blends previous layer's router hidden state into current routing:
        router_hidden = scale * prev_router_hidden + current_router_hidden
    """
    def __init__(self, ..., eda_scale: float = 1.0):
        self.router = TopKRouter(...)
        self.eda_scale = eda_scale
    
    def forward(self, hidden_states, prev_router_hidden=None):
        # If prev_router_hidden is provided, blend it in
        # Return (output, current_router_hidden) for chaining
```

---

#### 3. Per-Head Gating in Attention (explicit component)

**What it is**:
- After attention output, apply `sigmoid(gater(x))` per head or per element
- Multiplies gated value with attention result

**Current state**:
- Gating exists implicitly in attention workflow
- No explicit `HeadwiseGating` or `ElementwiseGating` component

**Rationale for component**:
- ZONOS2 uses headwise gating; modeling it explicitly makes variants clear
- Reusable for other attention schemes

**Sketch**:
```python
class HeadwiseGating(nn.Module):
    """Apply sigmoid gating per attention head."""
    def __init__(self, num_heads: int, head_dim: int):
        self.gater = nn.Linear(head_dim, 1, bias=False)
    
    def forward(self, attn_output: torch.Tensor) -> torch.Tensor:
        # attn_output: (num_tokens, num_heads, head_dim)
        # output: sigmoid-gated attn_output
```

---

#### 4. Per-Codebook Gating in Output (optional)

**What it is**:
- After multi-output projection, optionally apply per-codebook scaling
- Codebook-specific learned gates

**Current state**: Not present in current models (observation-based on ZONOS2 architecture)

**Rationale for component**:
- Allows different codebooks to have different prediction strengths
- Modular if future models use it

---

#### 5. Tokenization & Prompt Assembly (PromptBuilder)

**What it is**:
- UTF-8 byte tokenization (via NeMo text normalization)
- Frame construction: pad audio codebooks + text token
- Shear pattern application: codebook j shifted by j frames
- Pre-computed silence frames appended

**Current state**: 
- Likely in `tts/prompt.py` or similar higher-level module
- Not in `mstar.model.components`

**Rationale for component**:
- Separable concern: input formatting is distinct from model
- Could be in `mstar.model.components.tokenization` or separate namespace

---

#### 6. Sampling Pipeline (TTSSampler)

**What it is**:
- Applies filters in sequence: repetition penalty, temperature, top-k, top-p, min-p
- Multinomial sampling per codebook (independent)
- Supports per-request seeding

**Current state**:
- Likely in `tts/sampler.py` at higher level
- Not in `mstar.model.components`

**Rationale for component**:
- Distinct from model: post-processing of logits
- Could be in `mstar.model.components.sampling` or separate namespace

---

#### 7. TransformerStack & AutoregressiveDecoder (Orchestration)

**What it is**:
- `TransformerStack`: composes N `DecoderLayer`s with residual streams
- `AutoregressiveDecoder`: manages decode loop, EOS checking, frame assembly

**Current state**:
- Implicit: caller loops over DecoderLayers manually
- No explicit composition

**Rationale for component**:
- Encapsulates the static loop pattern
- Clearer intent: "this is N transformer blocks"
- Could enable optimizations like fused backward

**Sketch**:
```python
class TransformerStack(nn.Module):
    """Sequence of N pre-norm decoder layers."""
    def __init__(self, num_layers: int, layer_factory: Callable):
        self.layers = nn.ModuleList([layer_factory() for _ in range(num_layers)])
    
    def forward(self, hidden_states, cache_handle):
        for layer in self.layers:
            hidden_states = layer(hidden_states, cache_handle)
        return hidden_states

class AutoregressiveDecoder(nn.Module):
    """Orchestrates prefill + decode loop."""
    def __init__(self, model, sampler):
        ...
    
    def generate(self, prompt, max_tokens, **sampling_params):
        # Prefill
        # Decode loop with EOS check
```

---

## Part 4: Coverage Summary

| Category | Completeness | Gap |
|----------|--------------|-----|
| **Attention & Norms** | 95% | Per-head gating could be explicit |
| **FFN/MLP** | 90% | Missing per-codebook gating (optional) |
| **MoE** | 85% | Missing EDA blending across layers |
| **Tensor Parallelism** | 100% | Complete coverage |
| **Multi-codebook ops** | 30% | Missing `MultiEmbedding` / `MultiOutputHead` |
| **Control flow** | 10% | Missing explicit `TransformerStack` / `AutoregressiveDecoder` |
| **Sampling & tokenization** | 0% | Likely in higher-level modules; not in components |

---

## Part 5: Recommended Next Steps

### High Priority
1. **Add `MultiEmbedding` & `MultiOutputHead`** — ZONOS2-specific, needed for multi-codebook models
2. **Add EDA support to MoE** — Improves ZONOS2 expert specialization

### Medium Priority
3. **Explicit `TransformerStack`** — Clearer intent, enables stack-level optimizations
4. **HeadwiseGating component** — Currently implicit; explicit variant useful for debugging/variants

### Lower Priority
5. **Sampling pipeline** — Likely in `tts/` namespace; could be unified in `components.sampling`
6. **PromptBuilder** — Currently in `tts/prompt.py`; could be mirrored in components for clarity

---

## Part 6: File Organization

Current structure:
```
mstar/model/components/
├── __init__.py
├── attention.py          # Attention, ParallelAttention
├── decoder_layer.py      # DecoderLayer, GatedDecoderLayer
├── linear.py             # FusedColumnLinear
├── mlp.py                # MLP, GatedMLP, FusedGatedMLP
├── moe.py                # TopKRouter, SparseMoeBlock, MoE variants
├── norm.py               # RMSNorm, AdaRMSNorm
└── distributed/
    ├── __init__.py
    ├── attention.py      # ParallelAttention
    ├── embedding.py      # VocabParallelEmbedding
    ├── linear.py         # ColumnParallelLinear, RowParallelLinear, etc.
    └── mlp.py            # ParallelGatedMLP
```

Suggested additions:
```
mstar/model/components/
├── multi_embedding.py    # MultiEmbedding, MultiOutputHead (new)
├── moe_eda.py            # MoEBlockWithEDA (new, or extend moe.py)
├── gating.py             # HeadwiseGating, ElementwiseGating (new)
├── stack.py              # TransformerStack (new, or extend decoder_layer.py)
└── distributed/
    ├── multi_embedding.py    # ParallelMultiEmbedding (new)
```

---

## References

- ZONOS2 Architecture: `docs/tts_architecture.md`
- Model Config: `models/zonos2.py`
- Current Components: `mstar/model/components/`
