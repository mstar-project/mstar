# Complete System Design: Disaggregated Multimodal Inference System

**Scope**: A modular, extensible, high-performance serving system for multimodal models including unified models (BAGEL, Show-o2, Janus Pro), omni models (Qwen2.5-Omni, Qwen3-Omni), VLMs (Qwen3-VL), SpeechLMs (VoxServe models), and world models.

---

## Table of Contents

1. [Vision and Design Principles](#1-vision-and-design-principles)
2. [Supported Model Families](#2-supported-model-families)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [API Server](#4-api-server)
5. [Cluster Manager / Config](#5-cluster-manager--config)
6. [Conductor](#6-conductor)
7. [Execution Strategy](#7-execution-strategy)
8. [Workers](#8-workers)
9. [Computation Graph Model](#9-computation-graph-model)
10. [Inter-Worker Communication](#10-inter-worker-communication)
11. [KV Cache Management](#11-kv-cache-management)
12. [Concrete Request Flows](#12-concrete-request-flows)
13. [Technology Decisions](#13-technology-decisions)
14. [Resolved Design Tensions](#14-resolved-design-tensions)
15. [Open Questions](#16-open-questions)
- [Appendix A: Pipeline Parallelism Note](#appendix-a-pipeline-parallelism-note)
- [Appendix B: Scheduling Detail](#appendix-b-scheduling-detail-note-3)
- [Appendix C: Node Completion Signaling](#appendix-c-node-completion-signaling-note-11)
- [Appendix D: Design Notes from Atindra & Naomi](#appendix-d-design-notes-from-atindra--naomi)
- [Appendix E: Qwen 2.5 Omni vs. Qwen 3 Omni Comparison](#appendix-e-qwen25-omni-vs-qwen3-omni-comprehensive-architectural-comparison)

---

## 1. Vision and Design Principles

### 1.1 Vision

Extend the VoxServe SpeechLM serving system into a general-purpose multimodal inference system that supports arbitrary combinations of input and output modalities (text, image, audio, video) across diverse model architectures. The system must handle:

- **Autoregressive text generation** (standard LLM decode)
- **Rectified flow / ODE-based image generation** (BAGEL, Show-o2)
- **Discrete VQ-based image generation** (Janus Pro)
- **Streaming speech generation** (Qwen-Omni Talker, VoxServe SpeechLMs)
- **Vision-language understanding** (Qwen3-VL, understanding paths of unified models)
- **World model inference** (DreamerV3, DIAMOND)
- **Any future modality combination** without architectural changes

### 1.2 Design Principles

These principles are derived from 13 specific problems identified in the scrapped old design (see [Section 14](#14-resolved-design-tensions) for details).

| # | Principle | Rationale |
|---|-----------|-----------|
| P1 | **Separate logical from physical** | Computation nodes are logical; workers are physical GPU processes. The mapping is deployment config, not architecture. |
| P2 | **Workers declare capabilities, not pool membership** | Workers report what they can do (`encode_image`, `llm_prefill`, `flow_step`, etc.). The conductor routes by capability + affinity. No "pool ownership" confusion. |
| P3 | **Execution plan belongs on the model** | The model knows its own architecture. The model provides `get_graph_walk_graphs()` (and related orchestration helpers) so execution is defined by the model, not a separate strategy object. |
| P4 | **Co-location expresses tight coupling** | For Show-o2/JanusFlow where the flow head is interleaved with the LLM, the execution plan declares them as a single node. The conductor forces co-location. |
| P5 | **One process per GPU** | Never multiple worker processes on the same GPU without explicit coordination. Co-location = one process with multiple models loaded. |
| P6 | **Conductor is the inter-worker coordinator** | The conductor orchestrates lifecycle, selects workers, coordinates handoffs. It never touches GPU tensors and never manages batching. |
| P7 | **Workers handle internal batching** | Each worker has its own micro-scheduler for continuous batching. The conductor dispatches work; the worker decides when to execute. |
| P8 | **Start fresh, borrow primitives** | Build a clean architecture for multimodal disaggregation from day one. Copy proven primitives from VoxServe (FlashInfer, paged KV cache, CUDA graphs, ZMQ) and vLLM (scheduler interface, cache management, executor abstraction). |
| P9 | **Node as the fundamental unit** | Every computation is a node with: `(inputs, outputs, graph_edges)`. Nodes compose into graphs via `Sequential`, `Parallel`, and `Loop` primitives. Nodes are scheduled with graph-walk context and node-level readiness. |
| P10 | **Graph-driven scheduling** | The computation graph (not hardcoded graph walk enums) drives all scheduling. Ready/waiting queues operate on graph nodes, enabling arbitrary DAGs. |

---

## 2. Supported Model Families

### 2.1 Architecture Taxonomy

| Model | Type | Generation Method | Flow/LLM Coupling | Disaggregatable? |
|-------|------|-------------------|-------------------|-----------------|
| **BAGEL** | Unified (understand + generate) | Rectified Flow (configurable, default 24 steps) | Integrated -- each flow step runs a full LLM forward pass, but KV cache is frozen (`update_past_key_values=False`) and reused across steps | Partially -- KV cache transferred once (frozen), but LLM forward pass computation still needed per step. Less cross-GPU traffic than Show-o2 since KV doesn't change. |
| **Show-o2** | Unified (understand + generate) | ODE Flow Matching (50 steps) | Interleaved -- each flow step is a FULL LLM forward pass | No -- LLM + diffusion head must be co-located |
| **Janus Pro** | Unified (understand + generate) | Autoregressive VQ (576 tokens) | N/A -- pure AR | Yes -- just standard AR decode |
| **Qwen2.5-Omni** | Omni (text+image+audio in, text+speech out) | AR text + AR speech codec + Flow DiT | Thinker streams last-layer hidden states + input embeddings (element-wise sum, 3584-dim/token) and text token IDs to Talker before `<EOS>`. Talker recomputes TMRoPE internally from token IDs. | Yes -- Thinker and Talker are separate models on separate GPUs |
| **Qwen3-Omni** | Omni (text+image+audio+video in, text+speech out) | AR text + AR speech codec (16-layer RVQ, MTP for residual codebooks) + ConvNet Code2Wav | Thinker streams **both** layer-0 embeddings and layer-24 hidden states (`accept_hidden_layer=24`) for **all** tokens, plus token IDs. The Talker selectively routes: `text_projection(layer_0_embed)` for text tokens, `hidden_projection(layer_24_hidden)` for multimodal tokens (audio/image/video). Assistant-turn and decode tokens always use `text_projection`. | Yes -- Thinker (30B-A3B MoE) and Talker (3B-A0.3B MoE) are separate |
| **Qwen3-VL** | VLM (text+image+video in, text out) | Autoregressive text only | N/A -- but DeepStack injects ViT (SigLIP-2) features at the first three LLM layers | Standard LLM serving with vision encoder + multi-level feature injection |
| **VoxServe SpeechLMs** | SpeechLM (text in, audio out) | AR text→codec + detokenizer | LLM + detokenizer pipeline | Yes -- LLM and detokenizer can be disaggregated |
| **World Models** | RL world model | RSSM / Diffusion U-Net | N/A | Yes -- encoder, dynamics, actor are separable |

### 2.2 Key Architectural Patterns

**Pattern 1: Frozen-KV Flow (BAGEL)**
```
Text + Image context → LLM prefill → LLM decode (AR text) → [DONE_WITH_FWD]
                                          ↓ (KV cache frozen after text generation)
                         For each of ~24 flow steps:
                           noised latents → vae2llm projection → LLM forward (read-only KV) → llm2vae → velocity
                           update latents via Euler step
                         Final latents → VAE decode → Image
```
Each flow step runs a **full LLM forward pass** through the shared transformer backbone (verified in `bagel.py:_forward_flow()` which calls `self.language_model.forward_inference()`). However, the KV cache is **frozen** (`update_past_key_values=False`, `is_causal=False`) -- the text/image conditioning KV is read but never written during flow steps. Only the query-side latent embeddings change per step. The velocity prediction is extracted via a linear layer (`llm2vae`), not a separate diffusion head. 3x KV cache for CFG (conditional + text-CFG + image-CFG). This is cheaper than Show-o2 because KV cache can be transferred once and reused, but it is NOT a "single conditioning transfer" model -- the LLM computation is required at every step.

**Pattern 2: Interleaved Flow + LLM (Show-o2)**
```
Text tokens + noise latents → LLM forward pass → velocity prediction at image positions
                                    ↓
                              Euler step on latents
                                    ↓
                            Updated latents → LLM forward pass again (50× total)
                                    ↓
                              Final latents → VAE decode → Image
```
Each of the 50 flow steps requires a FULL LLM forward pass. No KV cache reuse across steps (latents change). CFG doubles all compute.

**Pattern 3: Thinker-Talker Streaming (Qwen2.5-Omni / Qwen3-Omni)**
```
Inputs → Encoders (ViT, Audio) → Thinker (AR text) ──STREAM──→ Talker (AR speech codec)
                                       ↓                              ↓
                                  DONE_WITH_FWD              Flow DiT (2.5) / Code2Wav (3)
                                       ↓                              ↓
                                 Text output                    Audio stream
```
Thinker streams hidden states + text token IDs to Talker before finishing its own generation. stream flag enables this inter-worker producer-consumer pattern. What exactly gets streamed differs: Qwen2.5-Omni sends last-layer hidden states + input embeddings (element-wise sum); Qwen3-Omni sends both layer-0 embeddings and layer-24 hidden states for all tokens (the Talker-side selectively routes them through `text_projection` or `hidden_projection` depending on token type). See [Appendix E](#appendix-e-qwen25-omni-vs-qwen3-omni-comprehensive-architectural-comparison) for the full comparison.

**Pattern 4: Standard SpeechLM (VoxServe Models)**
```
Text → Preprocess → LLM prefill → LLM decode (AR codec tokens) → Detokenizer → Audio chunks
```
VoxServe's existing pattern. Detokenizer runs at intervals (every 10-50 tokens) to generate audio chunks for streaming.

---

## 3. System Architecture Overview

```
                                      ┌───────────────────────┐
                                      │  Cluster Manager /    │
                                      │  Config (small)       │
                                      │  • cluster_config.yaml│
                                      │  • GPU allocation     │
                                      │  • autoscaling        │
                                      └──────────┬────────────┘
                                                 │ deployment-time only
                                                 ▼
┌──────────────┐                     ┌──────────────────────────────────────────┐
│  API Server  │ ──── ZMQ ────────→  │                 Conductor                │
│              │                     │                                          │
│  • FastAPI   │                     │  ┌─────────────┐  ┌──────────────────┐   │
│  • HTTP      │                     │  │   Worker    │  │ Request Lifecycle│   │
│  • Streaming │                     │  │   Registry  │  │  • req mgmt      │   │
│  • Preprocess│                     │  │   (static + │  │    (steps a-f)   │   │
│    Worker    │                     │  │    dynamic) │  │  • fwd pass mgmt │   │
│  • Tokenize  │                     │  └─────────────┘  │    (steps g-i)   │   │
└──────────────┘                     │                   └──────────────────┘   │
       ▲                             │  ┌──────────────────────────────────┐    │
       │                             │  │ WorkerGraph assignment           │    │
       │ emit_to_client             │  │ • {req → {worker_graph_id: worker_id}} │    │
       │                             │  │ • Graph-walk transitions (model ABC) │    │
       │                             │  └──────────────────────────────────┘    │
       │                             └────────────────┬───────────────────┬─────┘
       │                                              │                   │
       │                                    ┌─────────┘           ┌───────┘
       │                                    │                     │
       │                             ┌──────▼──────────┐   ┌──────▼──────────┐
       │                             │  Worker 0       │   │  Worker 1       │
       │                             │  (GPU 0)        │   │  (GPU 1)        │
       └──────────────────────────── │  • LLM (AR)     │◄─►│  • ViT (E/D)   │
                                     │  • Engines      │IPC│  • VAE (E/D)   │
                                     │  • MicroSched   │ZMQ│  • Engines     │
                                     │  • WorkerGraphsMgr │ + │  • MicroSched  │
                                     │  • Ready/Wait Q │RDM│  • WorkerGraphsMgr│
                                     └─────────────────┘ A └────────────────┘

                     ┌─────────────────────────────────────┐
                     │        Model (ABC)                  │
                     │  (per model, defines all graph walks)    │
                     │                                     │
                     │  • get_graph_walk_graphs() → {graph_walk: G}  │
                     │  • get_node_engine_types()          │
                     │  • get_initial_forward_pass_args(meta) │
                     │  • get_forward_pass_args(meta)         │
                     │  • process_prompt(text) → tensors    │
                     │  • get_submodule(node) → nn.Module  │
                     └─────────────────────────────────────┘
```

### 3.1 Component Responsibilities

| Component | Responsibility | What It Does NOT Do |
|-----------|---------------|---------------------|
| **API Server** | HTTP endpoints, streaming responses, ZMQ communication with Conductor, **tokenization** via `model.process_prompt()` (in the PreprocessWorker thread), media loading (images, audio, video) | GPU computation, batching |
| **Cluster Manager** | Deployment-time GPU allocation, autoscaling policy, config loading | Runtime request routing (deployment-time only) |
| **Conductor** | Macro-level request lifecycle: worker selection, worker graph assignment, routing inputs at the start of each forward pass, determining the next computation graph walk (via `model.get_forward_pass_args`), EOS/completion detection, passing per-request metadata (e.g., cache_labels for CFG). Does NOT manage ready/waiting queues or batch scheduling — those are on workers. | GPU computation, batching, tensor operations, intra-forward-pass scheduling |
| **Model** (replaces "Execution Strategy") | Defines computation graphs for each graph walk via `get_graph_walk_graphs()`, engine types via `get_node_engine_types()`, forward pass orchestration (`get_initial_forward_pass_args`, `get_forward_pass_args`), tokenization (`process_prompt`), and `NodeSubmodule` wrappers (preprocess + forward for each node). Lives on the `Model` ABC. | Scheduling, worker selection, communication. |
| **Workers** | GPU computation via engines, internal batch scheduling (MicroScheduler), KV cache management (via CacheHandle in AREngine), streaming output directly to API Server (EMIT_TO_CLIENT), inter-worker communication (Mooncake RDMA + ZMQ), worker graph queue management and completion detection (WorkerGraphsManager with ready/waiting queues), per-request graph walk tracking (applied locally from conductor signals), buffering persist signals and new tokens before reporting to conductor. | Macro-level graph walk transitions (EOS detection, deciding next graph walk), cross-worker scheduling, worker selection. |

---

## 4. API Server

### 4.1 Design

Borrowed from VoxServe with minimal modifications. The API server is a FastAPI application that:

1. Receives HTTP requests (text, image, audio, video inputs)
2. Serializes them and sends to the Conductor via ZMQ PUSH socket
3. Receives results from Workers via ZMQ PULL socket (streaming audio/image/text chunks)
4. Streams responses back to clients via async generators

### 4.2 Key Interfaces

```
Client ──HTTP POST──→ API Server ──ZMQ PUSH──→ Conductor (request socket)
Workers ──ZMQ PUSH──→ API Server (result socket) ──HTTP stream──→ Client
```

**Message format (generalized from VoxServe)**:
- Request: `json_metadata | binary_payload` (metadata includes request_id, modalities, model_kwargs)
- Response: `request_id | CHUNK_TYPE | data` where CHUNK_TYPE is `AUDIO`, `IMAGE`, `TEXT`, `VIDEO`, or `COMPLETION`

### 4.3 Data Worker and Tokenization

The API server includes a `PreprocessWorker` (running in a separate thread) that handles:
1. **Tokenization** via `model.process_prompt(prompt, input_modalities, output_modalities, **model_kwargs)` — each model defines its own tokenization logic, system prompt insertion (e.g., BAGEL's think mode), and output key naming.
2. **Media loading** — images (torchvision), audio (torchaudio), video (torchcodec) are loaded and tensor-encoded.
3. **Tensor registration** — all input tensors are stored in the `MooncakeCommunicationManager` and registered for RDMA send.
4. **Conductor notification** — a `NewRequestConductor` message is sent with `initial_signals` (tensor edge info), `input_modalities`, `output_modalities`, `input_metadata`, and `model_kwargs`.

If no model is provided (lightweight mode), the data worker falls back to UTF-8 byte encoding.

### 4.4 Adaptations from VoxServe

- Generalize message format from audio-only to arbitrary modality chunks
- Support multiple output streams per request (e.g., text + audio simultaneously for Omni models)
- ~Add input streaming support for incremental inputs (text chunks, video frames)~ (already implemented in VoxServe)
- Result socket receives from Workers directly (not through Conductor) for streaming data

### 4.5 What to Reuse

| From VoxServe | Reuse Strategy |
|---------------|----------------|
| FastAPI app structure, endpoints | Adapt directly |
| ZMQ socket setup (PUSH/PULL IPC) | Copy directly |
| Streaming response pattern (async generator + Queue) | Generalize to multi-modality |
| Request timeout enforcement | Copy directly |
| Multi-process spawning of Conductor subprocess | Adapt directly |

NOTE: We might want to rethink ZMQ for inter-node communication. ZMQ does have support for TCP, but TCP is probably not the most performant communication solution.

---

## 5. Cluster Manager / Config

### 5.1 Design

A small, deployment-time-only component that reads `cluster_config.yaml` and initializes the system. It does NOT participate in the runtime request path.

### 5.2 Config YAML Structure

The config uses `node_groups` — each group lists the graph node names it handles and which GPU ranks can execute it. The conductor uses this to break graph walk graphs into worker graphs and assign them to workers.

```yaml
# BAGEL example: LLM on GPU 0, encoders/decoders on GPU 1
node_groups:
  - node_names: ["LLM"]
    ranks: [0]
  - node_names: ["vit_encoder", "vae_encoder", "vae_decoder"]
    ranks: [1]
```

```yaml
# BAGEL colocated (single GPU): everything on rank 0
node_groups:
  - node_names: ["LLM", "vit_encoder", "vae_encoder", "vae_decoder"]
    ranks: [0]
```

```yaml
# Show-o2 example: LLM + flow co-located (required), VAE separate
# Optional 'graph_walks' field restricts a group to specific computation graph walks
node_groups:
  - node_names: ["LLM", "diffusion_head", "euler_step", "vit_encoder", "text_emb"]
    ranks: [0]
  - node_names: ["vae_decoder"]
    ranks: [1]
```

Node names in the YAML must match the `name` fields in the model's `get_graph_walk_graphs()` output.
When `ranks` has multiple entries, the worker graph is replicated for data parallelism and the conductor randomly assigns requests to one rank per worker graph.
The optional `graph_walks` field restricts a node group to specific computation graph walks (e.g., `graph_walks: ["image_gen"]`); if omitted, the group is active for all graph walks.

### 5.3 Responsibilities

1. Parse config YAML
2. Initialize Worker processes on specified GPUs
3. Populate the Conductor's Worker Registry with static worker properties
4. Handle autoscaling decisions (future work)

### 5.4 Architectural Note

The Cluster Manager sits above the Conductor and is NOT in the runtime request path:
```
Cluster Manager → (creates) → Conductor + Workers
API Server → (sends requests to) → Conductor  (no Cluster Manager involvement)
```
This was explicitly corrected from an earlier design that incorrectly placed the Cluster Manager between the API Server and Conductor.

---

## 6. Conductor

The Conductor is the central runtime coordinator.
As we have opted for a more decentralized, IPC-heavy scheduling procedure, the coordinator:

1. Assigns workers to worker graphs (contiguous computation graph sections) upon receiving a new request, and sends the workers information about what worker graph(s) they are handling for a request, and what workers are handling other worker graphs (for output routing),
2. During a forward pass, receives information from workers about worker graph completion, and also information about tensors that will persist across forward passes (e.g., a newly-generated token will need to be added to the inputs of the next forward pass, and image embeddings may also persist for the next forward pass),
3. At the end of each full model forward pass, calls `model.get_forward_pass_args(...)` to advance the computation graph walk (e.g., prefill → decode → image_gen) and detect request completion (EOS, image generation done) via `request_done`,
4. At the beginning of each full model forward pass, sends inputs and state information (e.g., which graph walk we are in) to the appropriate graph nodes on the appropriate workers.

The intra-forward-pass communication, scheduling, and batching is being handled on the worker level for now, and workers send their outputs directly to other workers (and to the API server when appropriate).

### 6.1 Inputs

- **Requests**: From API Server via ZMQ
- **Config**: From Cluster Manager at startup
- **Worker updates**: Dynamic status from workers via ZMQ

### 6.2 Internal Subcomponents

#### 6.2.1 Worker Registry

Indexes all workers by capability and tracks their state.

**Static properties** (set at deployment, don't change):
- `rank / gpu`: Physical GPU assignment (and if sharded across multiple GPUs -- not yet fully designed)
- `components: list[(model, submodel, capability)]`: What the worker can do
  - Example: `(show-o2, LLM, llm_prefill)`, `(show-o2, LLM, llm_decode)`, `(qwen3-omni, talker, speech_decode)`

**Dynamic properties** (workers push updates):
- Queue depth: Number of pending requests
- KV cache pages: Remaining tokens in last page
- Per-request compute: Estimated FLOPs (e.g., `Loop([LLM, ODE], 50)` vs single LLM pass)
- Per-request SLOs: Target latencies
- Per-request flags: pending / processing / completed

#### 6.2.2 Scheduling

At **initialization** time, the conductor assigns worker graphs to workers.
In the data-parallel case, a worker graph may be assigned to multiple workers, and requests will be routed to one of the data-parallel ranks per worker graph.
If there are multiple graph walks (e.g., prefill, decode, image generation), components of those will have separate worker graphs where appropriate.
Worker graphs for all graph walks will be assigned to workers at the beginning of a request, though only the worker graphs for one graph walk will be run by workers during each forward pass.

Specifically, upon instatiation, the conductor calls `model.get_worker_graphs(model_config_file)` and creates a mapping of worker graph id to worker graph.
A **WorkerGraph** is a contiguous section of a model computation graph that will be assigned to a worker. `model.get_worker_graphs` populates the worker graphs with a list of what workers can execute each worker graph (i.e., have the right sub-models loaded) and what computation graph walk the worker graph is active for.

Then, it communicates to each worker what worker graphs they will be processing, as well as what graph nodes are in other worker graphs (required for output routing).

For each request, the scheduler contains multiple roles that operate at different timescales:

**Request Management** -- Runs once per new request:

| Step | Action | Inputs |
|------|--------|--------|
| a | Get request from queue, extract metadata | `model_name`, initial input/output modalities |
| b | Determine worker plan for request (e.g., random routing for DP, though we should use a more sophisticated system in the future) to get a worker_graph to worker mapping for this request | server config (worker->gpu mappings) |
| c | Initialize a `RequestData` object for this request to track the progress of the current request, as well as signals that persist between forward passes |  Worker plan from (b), initial input/output modalities |
| d | Communicate to the relevant workers the worker graph IDs for this request, as well as the worker graph to worker mapping (so that the workers can route their outputs properly to other workers) | outputs from (b) and (c) | 

**Intra-forward Pass Management** -- Runs every conductor full model forward pass (i.e., when all worker graphs have been completed).

| Step | Action | Details |
|------|--------|---------|
| e | Check for "persistent signals" from workers | The workers may push information about tensors that will be used in future forward passes, which will be used as inputs in future forward passes |
| f | Check for "worker graph done" signals from workers | Update what worker graphs have been completed for the current request forward pass. If all worker graphs for the current computation graph walk have been completed, we have completed a forward pass. |

**At the end of each forward pass and beginning of the next**:

| Step | Action | Details |
|------|--------|---------|
| g | Update request lifecycle | Call `model.get_forward_pass_args(metadata, persist_signals, new_tokens)` to advance graph walk transitions (e.g., prefill → decode → image_gen) and detect completion (EOS, image done). If `request_done` is set, end request (send REMOVE_REQUEST to workers). |
| h | Read `fwd_args.inputs` | Get the `GraphEdge` inputs for the current forward pass (e.g., if we are doing image generation, this will include noisy latents). Each `GraphEdge` includes tensor name, what graph node it is routed to, and the current tensor location (IP address, memory address, size)  |
| i | Dispatch inputs to workers | Send the input `GraphEdge`s for the current forward pass to the appropriate workers. Also include metadata of which graph walk we are in (e.g. prefill, decode, image_gen) and per-request metadata (e.g., cache_labels, snapshot_after), because these are required for routing and cache orchestration |

**Note**: sending the current computation graph walk may be replaced by a more detailed "request state", which will additionally include information about, e.g., what input token indices are text vs. image vs. video.


**Key data structures**:

- `{worker_graph_id: worker_graph}`:
```python
@dataclass
class WorkerGraph:
    section: GraphSection
    graph_walks: set[str] # e.g., prefill, decode, image_gen
    consumes_stream: bool # for thinker-talker
    ranks: list[int] # one-to-one mapping of worker to rank
    worker_graph_id: str
```
- `{req_id: request_data}`:
```python
@dataclass
class RequestData:
    current_forward_metadata: CurrentForwardMetadata
    fwd_inputs: list[GraphEdge] # inputs that the conductor has sent to the current fwd pass
    persist_signals: dict[str, list[TensorPointerInfo]] # signals passed back to conductor
    worker_graph_to_worker: dict[str, str] # worker graph id to worker id
    new_tokens: dict[str, list[int]] # name -> tokens, for this fwd pass (e.g., {"new_token": [42, 17]})

    # for tracking progress
    all_worker_graph_ids: set[str] # across all graph walks
    current_worker_graph_ids: set[str] # for the current fwd pass computation graph walk
    completed_worker_graph_ids: set[str] # for the current full model fwd pass
```
where `CurrentForwardMetadata` is:
```python
@dataclass
class CurrentForwardMetadata:
    input_modalities: list[str]
    output_modalities: list[str]
    graph_walk: str
    is_prefill: bool
    request_done: bool = False  # set by model.get_forward_pass_args() on EOS/completion
    kwargs: dict  # model-specific metadata (e.g., prefill_schedule, cfg scales)
```
See the [computation graph model section](#9-computation-graph-model) for information about `GraphEdge` and `TensorPointerMetadata`.


### 6.4 WorkerGraph Persistence (Note 7)

The conductor does NOT recompute the execution plan every forward pass. Instead:

1. On server initialization graph walk setup: determine node->worker and worker->gpu mappings from yaml file. This results in a static list of worker graphs that are being handled by each worker.
2. On a new request arriving: compute the execution plan based on input/output modalities, stored in the request state. Only recompute IF input/output modalities change (e.g., `<BOI>` triggers adding flow worker graph). The conductor only sends the inputs to the proper workers for this forward pass, as well as what graph walk of computation we are in (it does **not** need to send new worker graph information to workers); the execution of the proper worker graphs for each forward pass can be handled via tensor routing between workers.

The worker needs to know only (1) what the computation worker graphs are to compute, (2) what's in the incoming request queue, and (3) where to send the output, which is decided by the conductor as metadata for each request. This is assuming worker graph-to-worker mapping is static (i.e., there never exists LLM-only worker and LLM+flow worker at the same time)

This enables:
- Reduced recomputation overhead
- "Talker stays alive for duration of request" paradigm
- Preference for, e.g., request0's decode always running on GPU1 (avoids KV cache transfer)

---


## 7. Execution Strategy

When defining a model, the user must define a computation graph for each graph walk, as well as: logic for determining the computation graph walk at each full model forward pass, the "full model" inputs at each new forward pass, and code for actually executing each graph node.

**Note**: This is currently in the `Model` class, but it might make sense to split execution-planning concerns into a dedicated strategy object later (for example, to support policy hot-swaps per request).

### 7.1 Computation Graph and WorkerGraphs
For each graph walk, the user must define a **computation** graph, which specifies the discrete computation nodes, their execution order, and how their outputs are routed.

To make graph definition more intuitive than, e.g., a generic DAG, nodes can be organized in `Sequential`, `Parallel`, and `Loop` configurations.

The user must define `self.get_graph_walk_graphs()`, which returns a mapping of computation graph walk name (e.g., `prefill`, `decode`, `image_gen`) to computation graph.

Once the graphs are defined, the `Model` class automatically parses it, along with the cluster config, to produce a list of **WorkerGraphs**, or groups of nodes that will be assigned to a worker together.
This is produced by `model.get_worker_graphs(config_path)`, which calls `self.get_graph_walk_graphs()` and performs the logic to break the graphs from each graph walk into worker graphs.
See the [computation graph model section](#9-computation-graph-model) for information.

### 7.2 Model ABC Methods

In addition to the computation graph, each model implements the following abstract methods from the `Model` base class (`model/base.py`):

**Tokenization (called by API server data worker):**

- `process_prompt(prompt, input_modalities, output_modalities, **kwargs) → NameToTensorList` — Tokenizes the user prompt and produces initial text tensors (e.g., tokenized input, system prompt). Called by the API server's `PreprocessWorker` to convert raw text to model-specific tensor format. Output keys (e.g., `"text_inputs"`, `"system_prompt"`) are referenced by `get_forward_pass_args` via `persist_signals`.

**Forward pass orchestration (called by conductor):**

- `get_initial_forward_pass_args(input_modalities, output_modalities, input_signals, model_kwargs) → ForwardPassArgs` — Determines the starting graph walk and constructs model-specific metadata (e.g., BAGEL's prefill schedule with multi-cache annotations for CFG).

- `get_forward_pass_args(metadata, persist_signals, new_tokens) → ForwardPassArgs` — Returns the external inputs and request state for the next forward pass. Uses `persist_signals` (tensors that persisted from previous forward passes), the current graph walk and `new_tokens` to construct `GraphEdge`s with `next_node`, `name`, and `tensor_info` fields.

- `get_forward_pass_args(metadata, persist_signals, new_tokens) → ForwardPassArgs` — Called after each full model forward pass to advance graph walk transitions (e.g., prefill_text → prefill_vit → decode → image_gen). Graph walk transitions are schedule-driven for models like BAGEL; `new_tokens` is checked for EOS.

**Node engine types:**

- `get_node_engine_types() → dict[str, EngineType]` — Returns the engine type (`EngineType.AR`, `EngineType.ENC_DEC`, `EngineType.FLOW`, `EngineType.AUDIO_CODEC`) for each node name. Used by the conductor to build engine configs for workers.

**Submodule access (called by engine manager on workers):**

- `get_submodule(node_name) → nn.Module | None` — Returns the `NodeSubmodule` wrapper for a node, or `None` for dummy mode. Workers call this to get the actual PyTorch module to execute. Submodule creation is lazy (created on first access).

**Node execution:**

- The worker scheduler prepares `NodeBatch` payloads and dispatches them directly to the assigned engine. There is no global `step()` dispatcher; execution behavior is implemented by the model’s node submodule.

### 7.3 Full Graph Example (BAGEL)

BAGEL has **5 separate graph walk graphs** rather than one monolithic graph, because: (1) the output mode is known upfront from the API request (no BOI token detection), and (2) the LLM is a "fat node" that absorbs text_emb, lm_head, and flow_proj to avoid unnecessary IPC. Graph walk transitions are schedule-driven via `get_forward_pass_args()`.

The 4 nodes are: `vit_encoder` (enc_dec), `vae_encoder` (enc_dec), `LLM` (ar), `vae_decoder` (enc_dec).

```python
def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
    # -- prefill_text: just the LLM node (text embedding is internal) --
    prefill_text = GraphNode(
        name="LLM",
        input_ids=["text_inputs"],
            outputs=[],   # No output — conductor notified via WORKER_GRAPHS_DONE
    )

    # -- prefill_vit: ViT encoder -> LLM (bidirectional attention) --
    prefill_vit = Sequential([
        GraphNode(
            name="vit_encoder",
            input_ids=["image_inputs"],
            outputs=[GraphEdge(next_node="LLM", name="vit_emb")],
        ),
        GraphNode(name="LLM", input_ids=["vit_emb"], outputs=[]),
    ])

    # -- prefill_vae: VAE encoder -> LLM (bidirectional attention) --
    prefill_vae = Sequential([
        GraphNode(
            name="vae_encoder",
            input_ids=["image_inputs"],
            outputs=[GraphEdge(next_node="LLM", name="vae_emb")],
        ),
        GraphNode(name="LLM", input_ids=["vae_emb"], outputs=[]),
    ])

    # -- decode: single LLM node (embed + transformer + lm_head) --
    decode = GraphNode(
        name="LLM",
        input_ids=["text_inputs"],
        outputs=[
            GraphEdge(
                next_node=EMIT_TO_CLIENT, name="new_token",
                output_modality="text", is_new_token=True,
                persist=True,
            ),
        ],
    )

    # -- image_gen: denoising loop (LLM does 3-pass CFG + Euler) -> VAE decode --
    image_gen = Sequential([
        Loop(
            section=GraphNode(
                name="LLM",
                input_ids=["latents"],
                outputs=[GraphEdge(next_node="LLM", name="latents")],
            ),
            n_iters=self.num_timesteps - 1,  # N-1 Euler steps for N timesteps
            outputs=[GraphEdge(next_node="vae_decoder", name="latents")],
        ),
        GraphNode(
            name="vae_decoder",
            input_ids=["latents"],
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT, name="image_output",
                    output_modality="image", persist=True,
                ),
            ],
        ),
    ])

    return dict(
        prefill_text=prefill_text,
        prefill_vit=prefill_vit,
        prefill_vae=prefill_vae,
        decode=decode,
        image_gen=image_gen,
    )
```

**Prefill schedule and multi-cache orchestration (CFG)**:

For image generation, BAGEL uses 3 KV caches (main, cfg_text, cfg_img) for classifier-free guidance. The prefill schedule annotates each step with `cache_labels` (which caches to update) and optional `snapshot_after` (to deepcopy a cache):

| Prefill step | `cache_labels` | `snapshot_after` |
|---|---|---|
| system text | `["main", "cfg_img"]` | — |
| image (VAE) | `["main"]` | `("main", "cfg_text")` after last image |
| user text | `["main", "cfg_img"]` | — |

After prefill: `main` = full, `cfg_img` = text-only, `cfg_text` = system+image only.

During `image_gen`, all 3 caches are frozen (read-only, `write_cache=False`). Each denoising step runs the LLM 3x against different caches, combines velocities via the CFG formula, and performs an Euler step.

The `LLMSubmodule` dispatches based on the `graph_walk` argument (passed through `NodeBatch.graph_walk`), handling prefill_text/vit/vae/decode/image_gen with graph walk-specific logic. Cache labels and snapshot metadata flow from `metadata.kwargs["prefill_schedule"]` through the conductor → `InputSignals.per_request_metadata` → worker → `NodeBatch.per_request_metadata` → engine → `submodule.forward(**metadata)`.

**Note**: BAGEL uses the same LLM backbone for both text generation and flow steps. During flow steps, the LLM reads frozen KV cache (`write_cache=False`, `is_causal=False`) and processes noised latents. Velocity is extracted via `llm2vae` (a linear layer), not a separate diffusion head.

### 7.4 Image Generation Graph (Show-o2 -- Interleaved)
**AR Text Generation Graph Walk**:
```python
Sequential([
    Parallel([
        GraphNode(
            name="vit_encoder",
            input_ids=["image"],
            outputs=[
                GraphEdge(
                    next_node="LLM",
                    name="img_emb"
                )
            ]
        ),
        GraphNode(
            name="text_emb",
            input_ids=["text"],
            outputs=[
                GraphEdge(
                    next_node="LLM",
                    name="text_emb"
                )
            ]
        ),
    ]),
    GraphNode(
        name="LLM_text_gen",  # AR text generation graph walk
        input_ids=["text_emb", "img_emb"],
        outputs=[
            GraphEdge(next_node=EMIT_TO_CLIENT, name="text_tokens"),
        ]
    )
])
```

**Image generation (flow matching)**:
```python
Sequential([
    Parallel([
        GraphNode(
            name="vit_encoder",
            input_ids=["image"],
            outputs=[
                GraphEdge(
                    next_node="LLM",
                    name="img_emb"
                )
            ]
        ),
        GraphNode(
            name="text_emb",
            input_ids=["text"],
            outputs=[
                GraphEdge(
                    next_node="LLM",
                    name="text_emb"
                )
            ]
        ),
    ]),
    Loop(
        section=Sequential([
            GraphNode(
                name="LLM",  # Full LLM forward pass at each flow step
                input_ids=["text_emb", "img_emb", "latents"],
                outputs=[
                    GraphEdge(
                        next_node="diffusion_head",
                        name="hidden_states"
                    )
                ]
            ),
            GraphNode(
                name="diffusion_head",
                input_ids=["hidden_states"],
                outputs=[
                    GraphEdge(
                        next_node="euler_step",
                        name="velocity"
                    )
                ]
            ),
            GraphNode(
                name="euler_step",
                input_ids=["velocity", "latents"],
                outputs=[
                    GraphEdge(
                        next_node="LLM",
                        name="latents"
                    )  # loop-back
                ]
            ),
        ]),
        n_iters=50,
        outputs=[
            GraphEdge(
                next_node="vae_decoder",
                name="latents"
            )
        ]
    ),
    GraphNode(
        name="vae_decoder",
        input_ids=["latents"],
        outputs=[
            GraphEdge(
                next_node=EMIT_TO_CLIENT,
                name="image"
            )
        ]
    )
])
```

Note: For Show-o2, the `LLM`, `diffusion_head`, and `euler_step` nodes MUST be co-located on the same worker because they execute 50 times in tight sequence. The config YAML's `try_to_colocate` enforces this.

### 7.5 Full Graph Example (Qwen3-Omni -- Thinker-Talker)

```python
full_graph = Sequential([
    Parallel([
        GraphNode(
            name="text_emb",
            input_ids=["text"],
            outputs=[
                GraphEdge(
                    next_node="thinker",
                    name="text_emb"
                )
            ]
        ),
        GraphNode(
            name="vit_encoder",
            input_ids=["image"],
            outputs=[
                GraphEdge(
                    next_node="thinker",
                    name="img_emb"
                )
            ]
        ),
        GraphNode(
            name="audio_encoder",
            input_ids=["audio"],
            outputs=[
                GraphEdge(
                    next_node="thinker",
                    name="audio_emb"
                )
            ]
        ),
    ]),
    Parallel([
        # Thinker branch (AR text generation)
        GraphNode(
            name="thinker",
            input_ids=["text_emb", "img_emb", "audio_emb"],
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="text_tokens"
                ),
                GraphEdge(
                    next_node="talker",
                    name="hidden_states",
                    persist=False
                ),
                # Note: in a producer-consumer stream, only the consumer needs to
                # know that this is a streaming operation. The producer can just send
                # outputs via IPC as normal.
            ]
        ),
        # Talker branch (AR speech codec generation)
        # Note: The talker is an AR model that runs autoregressively (many forward passes),
        # but this is managed at the worker level (the Talker loop in Section 10.2),
        # not as a Loop in the graph. The graph represents it as a single node
                # because its lifecycle is managed via stream buffering, not graph readiness.
        Sequential([
            GraphNode(
                name="talker",
                consumes_stream=True,
                input_ids=["hidden_states"],  # from thinker via stream
                outputs=[
                    GraphEdge(
                        next_node="mtp_module",
                        name="codec_tokens"
                    )
                ]
            ),
            GraphNode(
                name="mtp_module",
                input_ids=["codec_tokens"],
                outputs=[
                    GraphEdge(
                        next_node="code2wav",
                        name="full_codebook"
                    )
                ]
            ),
            GraphNode(
                name="code2wav",
                input_ids=["full_codebook"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="audio_chunk"
                    )
                ]
            ),
        ])
    ]),
])
```


### 7.6 Note: GraphEdges in Full Graph Construction

The graph edge info associated with each tensor triggers the onset of the `next_node` as well as inter-worker or worker-to-conductor communication (tensor sharing).

```python
@dataclass
class TensorPointerInfo:
    dims: list[int]
    dtype: str
    nbytes: int
    address: int
    uuid: str  # for tensor cleanup and list[tensor] indexing
    source_session_id: str # e.g., f"{HOSTNAME}:{client_engine.get_rpc_port()}"
    source_entity: str # which {worker, api_server} the tensor is on

@dataclass
class GraphEdge:
    next_node: str
    name: str
    tensor_info: list[TensorPointerInfo] = field(default_factory=list)
    # Flags
    persist: bool = field(default=False)
    is_new_token: bool = field(default=False)
    output_modality: str = field(default="")  # text | image | video | audio (for EMIT_TO_CLIENT)
```

---

## 8. Workers and Engines

### 8.1 Design

Workers are long-lived GPU processes. Each worker:
1. Loads model components at startup based on its config
2. Registers capabilities with the Conductor's Worker Registry
3. Receives dispatched nodes from the Conductor via ZMQ
4. Executes computation, manages internal batching
5. Sends completion notifications back to the Conductor
6. Streams output directly to the API Server (for EMIT_TO_CLIENT)
7. Communicates with other workers for inter-worker data transfer (node outputs and prefill→decode KV cache)

### 8.2 Engines

Each worker has one or more **engines** that wrap their model components for efficient execution. An `EngineManager` (on each worker) maps node names to engine instances, loads submodules via `model.get_submodule(node_name)`, and dispatches execution.

Engine types are defined by the `EngineType` enum:
```python
class EngineType(Enum):
    AR = "ar"           # Autoregressive (LLM with KV cache)
    FLOW = "flow"       # Flow matching / diffusion
    ENC_DEC = "enc_dec" # Stateless encoder/decoder (ViT, VAE)
    AUDIO_CODEC = "audio_codec"
```

All engines inherit `BaseEngine` and implement:
- `load_model(submodules, model_config, device)`: receives `dict[str, nn.Module]` keyed by node name (from `model.get_submodule()`) and performs engine-specific initialization.
- `execute_batch(batch: NodeBatch) → NodeOutput`: runs computation for a batch of requests.
- `add_request(request_id)` / `remove_request(request_id)`: per-request lifecycle (e.g., KV cache allocation/deallocation).
- `warmup()`: optional CUDA graph capture.

```python
@dataclass
class NodeBatch:
    """Input to an engine's execute_batch()."""
    node_name: str
    graph_walk: str
    request_ids: list[str]
    per_request_input_tensors: dict[str, NameToTensorList]  # {rid: {name: list[tensor]}}
    metadata: dict = field(default_factory=dict)
    per_request_metadata: dict[str, dict] = field(default_factory=dict)  # {rid: {key: value}}
```

The `per_request_metadata` field carries model-specific metadata from the conductor (e.g., `cache_labels`, `snapshot_after` for BAGEL's multi-cache CFG). It flows: conductor → `InputSignals.per_request_metadata` → worker → `NodeBatch.per_request_metadata` → engine → `submodule.forward(**metadata)`.

#### 8.2.1 NodeSubmodule (preprocess/forward pattern)

Model nodes are wrapped in `NodeSubmodule` subclasses (`model/base.py`), which separate preprocessing from computation:

```python
class NodeSubmodule(torch.nn.Module):
    def preprocess(self, **inputs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convert variable-length list[Tensor] inputs to fixed tensors.
        NOT compiled — handles Python-level variability (cu_seqlens, .item()).
        Default: assert each input has exactly 1 tensor and unwrap it."""

    def forward(self, **kwargs) -> NameToTensorList:
        """Pure tensor → NameToTensorList computation. Compilable + CUDA-graphable."""
```

Engines call `submodule.preprocess(**inputs)` then `submodule(**preprocessed, **metadata)`. The preprocess step extracts Python scalars (`.item()`) and computes cu_seqlens so that `forward()` operates only on fixed tensors (CUDA graph compatible).

#### 8.2.2 Autoregressive Engine (AREngine)

For transformer-based nodes (LLM). Manages paged KV cache with FlashInfer attention.

**Key feature: CacheHandle** — Per-request cache handle created by the engine and passed to `submodule.forward()`. The engine provides cache infrastructure; the submodule decides cache semantics:

```python
class CacheHandle:
    def run_attention(self, q, k, v, layer_idx, is_causal=True, write_cache=True) -> Tensor
    def set_active_label(self, label: str)    # Switch active KV cache (e.g., "main" → "cfg_text")
    def snapshot(self, from_label, to_label)  # Deepcopy KV cache between labels
    def save_seq_position(self) -> int        # Save position for flow matching rewind
    def restore_seq_position(self, pos: int)  # Rewind to saved position
```

This separation enables BAGEL's 3-cache CFG without the engine knowing about CFG semantics. The AR engine's `execute_batch` creates a `CacheHandle` per request and passes it alongside `graph_walk` and metadata to the submodule.

KV cache state is keyed by `(request_id, cache_label)` to support multiple caches per request. `add_request()` accepts optional `cache_labels` to initialize multiple caches (default: `["main"]`).

**Features**:
- Paged KV cache with block-based `PageAllocator`
- FlashInfer prefill and decode wrappers (with naive fallback when FlashInfer is unavailable)
- Pause/resume for interleaved loops (LLM ↔ flow)

#### 8.2.3 Encoder/Decoder Engine (EncoderDecoderEngine)

For stateless nodes (ViT encoder, VAE encoder/decoder). No KV cache — straightforward `preprocess` → `forward` per request.

Passes `per_request_metadata` to `submodule.forward()` alongside preprocessed inputs.

**Characteristics**:
- Stateless per request (no `add_request` / `remove_request` overhead)
- CUDA-graph-friendly (fixed shapes per request)


#### 8.3 Scheduler

Each worker handles its own batch scheduling via a `MicroScheduler` class; at every worker loop, the worker calls `scheduler.get_next_batch(worker_graphs_manager)` to get the next node name and batch of inputs that should be run.
The conductor and other workers dispatch work items; the worker accumulates and batches them.

**For AR decode nodes**:
- Continuous batching: requests join/leave batch dynamically
- Batch selection based on: prefill priority, KV cache availability, SLO urgency
- Chunked prefill: process prompt in chunks to avoid blocking decode requests (handled at worker level, not conductor)

**For flow/diffusion nodes**:
- Batch multiple requests' flow steps together
- CFG parallelism: handled within the submodule (e.g., BAGEL's LLMSubmodule runs 3 forward passes per step internally)

**For encoder nodes**:
- Simple batching: accumulate images/audio, process in batch
- Stateless: no KV cache management needed

### 8.4 Worker Internal Architecture

The worker (`worker/worker.py`) integrates four components:
1. **WorkerGraphsManager** — Tracks per-request graph state: which worker graphs are assigned, ready/waiting queues, graph walk tracking, output routing.
2. **EngineManager** — Maps node names to engine instances, handles `load_model()`, `add_request()`, `remove_request()`.
3. **MicroScheduler** — Selects the next batch to run based on ready nodes across all requests.
4. **MooncakeCommunicationManager** — Handles RDMA tensor transfers (start_read, get_ready, register_for_send).

### 8.5 Worker State Management

State is mainly managed within workers (or between workers, via IPC), with some information passed from the conductor to workers at the beginning of each forward pass.

- **Within-worker state** (worker manages): KV cache (via CacheHandle), per-request graph queues, engine state.
- **Between-worker synchronization** (handled by IPC between workers): tensor transfers, node output routing.
- **Conductor-to-worker synchronization**: At the beginning of each forward pass, the conductor sends `InputSignals` with: inputs (`list[GraphEdge]`), graph_walk, and `per_request_metadata` (e.g., `cache_labels`, `snapshot_after` for CFG). After the forward pass starts, subsequent signals pass between workers directly via IPC.

### 8.6 Streaming Output

Workers stream output directly to the API Server (not through the Conductor):

```
Worker ──ZMQ PUSH──→ API Server result socket
```

This avoids the conductor becoming a bottleneck for high-bandwidth data (audio waveforms, image tensors). The conductor only receives lightweight completion notifications.
Tensors are transferred to the API server via the same methods is inter-worker communication, as described in **Worker Communication** below.

### 8.7 Worker Communication
Worker communication includes two layers: (1) message-passing of JSON messages via ZMQ, and (2) tensor communication via Mooncake Transfer Engine.
For tensor communication, we are using a read/pull-based paradigm: when tensors need to be transferred, the sender passes a ZMQ message with the information needed for the receiver to initiate a read.
Then, the receiver reads the tensor via (2) and sends an ACK back to the sender.
See the [Inter-Process Communication](#10-inter-process-communication) section for more information.

---

## 9. Computation Graph Model

### 9.1 Class Hierarchy

```
GraphSection (ABC)
├── GraphNode          # Node: a single computation unit
├── Sequential          # Nodes execute in order
├── Parallel            # Nodes execute concurrently
└── Loop                # Node/worker graph repeated N times
```

A `GraphSection` is an abstract class that represents any arbitrary (contiguous) part of a graph: a single node (`GraphNode`), or different compositions of nodes.
This abstraction is necessary for `Sequential`, `Parallel`, and `Loop` blocks to hold/organize arbitrary graphs (e.g., a `Loop` can hold a `Sequential`, `Parallel` or even another `Loop`).

<!-- **Note on evolution from whiteboard to code**: The original whiteboard (IMG_0736) proposed `GraphSection` with concrete fields `inputs: list[str]` and `name: str`. The actual implementation evolved to use **abstract methods only** -- no concrete fields on the base class. -->

**Abstract methods on GraphSection** (all subclasses must implement):
```python
DestToInputs = dict[str, list[GraphEdge]]

class GraphSection(ABC):
    def get_node_names(self) -> list[str]: ...             # Names of all nodes in this section
    def get_inputs(self) -> list[GraphEdge]: ...         # External/loop-back inputs
    def get_outputs(self) -> list[GraphEdge]: ...        # External/loop-back outputs
    def ingest_inputs(self, node_to_inputs: DestToInputs): ...  # Mark inputs as received; MUTATES node_to_inputs
    def split_off_ready(self) -> tuple[list[GraphNode], GraphSection | None]: ...  # Split ready nodes from waiting
```

As a reminder, the `GraphEdge` is the following dataclass:
```python
@dataclass
class GraphEdge:
    next_node: str
    name: str
    tensor_info: list[TensorPointerInfo] = field(default_factory=list)
    # Flags
    persist: bool = field(default=False)
    is_new_token: bool = field(default=False)
```

**Critical behavior**: `ingest_inputs()` **mutates** its argument `node_to_inputs`, removing entries that were consumed. After the call, only un-consumed (external) entries remain. This mutation-based protocol is how external outputs (i.e., those that must be routed to other workers) are computed.

### 9.2 GraphNode

The fundamental computation unit (leaf node).

```python
@dataclass
class GraphNode(GraphSection):
    name: str
    input_ids: set[str]              # Named inputs this node needs (coerced from list in __post_init__)
    outputs: list[GraphEdge]
    consumes_stream: bool = False     # For stream consumer nodes (e.g., Talker)

    # Populated as predecessors complete — maps input name to the GraphEdge
    # carrying the tensor info (address, uuid, etc.)
    ready_inputs: dict[str, GraphEdge] = field(default_factory=dict)

    def is_ready(self) -> bool:
        return self.input_ids.issubset(set(self.ready_inputs.keys()))
```

**Key methods**:
- `ingest_inputs(node_to_inputs)`: If this node's name is in `node_to_inputs`, ingest any input IDs that this node needs and hasn't already received. Remove consumed entries from `node_to_inputs` (mutation). Returns the list of ingested `GraphEdge`s.
- `split_off_ready()`: Returns `([self], None)` if ready, or `([], self)` if still waiting.
- `get_inputs()`: Returns `[GraphEdge(next_node=self.name, name=id) for id in self.input_ids]`.
- `get_outputs()`: Returns `self.outputs`.

Visual:
```
             ┌──────────┐
  inp_A ──→  │  Node    │ ──→ out_X → [dest1, dest2]
  inp_B ──→  │          │ ──→ out_Y → [dest3]
             └──────────┘
```

### 9.3 Sequential

Nodes execute in order.

```python
@dataclass
class Sequential(GraphSection):
    sections: list[GraphSection]
```

**Key behaviors**:
- `split_off_ready()`: Only checks the **first** section. If the first section is ready, it is removed and the rest form a new Sequential. This enforces sequential ordering -- later sections cannot become ready until earlier ones complete.
- `_get_inputs_outputs()`: Filters internal signals. Iterates sections, tracking accumulated outputs. When a section's input matches a prior section's output, it is recognized as internal and excluded from the external inputs list. This ensures only truly external inputs/outputs are reported.

Visual:
```
  A ──→ B ──→ C
```

### 9.4 Parallel

Nodes execute concurrently. Fork/join pattern.

```python
@dataclass
class Parallel(GraphSection):
    sections: list[GraphSection]
```

**Key behaviors**:
- `split_off_ready()`: Checks **ALL** sections (contrast with Sequential which only checks first). Each branch independently splits off ready nodes. This allows all branches to make progress concurrently.

Visual:
```
       ┌──→ A ──────────┐
  ──→──┤                 ├──→ D
       └──→ B ──→ C ──→─┘
```

### 9.5 Loop

    A node or worker graph repeated N times. Handles loop-back signals (outputs that feed back as inputs to the next iteration) and external inputs.

```python
@dataclass
class Loop(GraphSection):
    section: GraphSection           # Template ("clean" copy, never mutated)
    n_iters: int
    outputs: list[GraphEdge]   # Final iteration outputs (replace loop-backs)
    curr_iter: int = field(default=0)
    external_inputs: list[GraphEdge] = field(default=None)     # Inputs from OUTSIDE the loop (not loop-back)
    loop_back_signals: list[GraphEdge] = field(default=None)
    curr_iter_section: GraphSection = field(default=None)    # Mutable copy for current iteration
    nxt_iter_section: GraphSection = field(default=None)     # Pre-populated copy for next iteration
```

**`external_inputs`** (critical field, computed in `__post_init__`): Identifies which inputs come from outside the loop vs. from loop-back signals. This distinction is essential for nested loops -- external inputs are re-injected each iteration, while loop-back inputs only come from the previous iteration's outputs.

**`loop_back_signals`** (critical field, computed in `__post_init__`): Identifies which outputs loop back.
These outputs must be removed in the final loop iteration.

**Key internal methods**:
- `_get_loop_back_signals()`: Identifies signals that appear in both the section's inputs AND outputs -- these are the loop-back edges.
- `_replace_outputs_for_final_iter(section)`: On the last iteration, removes loop-back edges from node outputs and replaces them with `self.outputs` (the final output destinations). Applied recursively through nested sections.
- `_advance_one_iter()`: Promotes `nxt_iter_section` to `curr_iter_section`, creates a fresh `deepcopy` for the new `nxt_iter_section`, increments `curr_iter`, and re-ingests `external_inputs`.
- `split_off_ready()`: If `curr_iter_section` is None (iteration consumed), calls `_advance_one_iter()`. Then splits off ready nodes from the current iteration. On last iteration, returns the waiting section directly (no Loop wrapper). Otherwise, returns the Loop itself as the waiting section.
- `ingest_inputs(node_to_inputs)`: First ingests into `curr_iter_section`. Then separates external inputs from loop-back inputs, and only passes loop-back inputs to `nxt_iter_section` (to prevent external inputs from prematurely populating the next iteration). This logic is required for nested loops.

Visual:
```
  A ──→ B ──→ (loop-back to A, 50×)
              ──→ (final output after 50 iterations)
```

**Loop dispatch granularity**: A `Loop(50)` can be dispatched as a single unit (worker runs all 50 iterations, sends one completion) or broken into `Loop(10) × 5` (enabling round-robin across workers, reducing head-of-line blocking, enabling checkpointing between chunks).

<!-- ### 9.6 Signal Routing (needs/produces/ptr)

Three equivalent representations of graph edges:

```python
SignalToDests = dict[str, list[str]]              # {signal: [dest_node_names]}
SignalToDestsAndFlags = dict[str, list[GraphEdge]]  # {signal: [GraphEdge(dest, flags)]}
DestToInputs = dict[str, list[str]]              # {dest_node: [required_signals]}
```

These can be converted between each other:
- `remove_flags()`: SignalToDestsAndFlags → SignalToDests (strips GraphEdge, keeps dest names)
- `get_node_to_inputs_mapping()`: SignalToDests → DestToInputs (inverts the mapping).
- `get_signal_to_dest_mapping()`: DestToInputs → SignalToDests (inverts back)

Helper: `update_list_dicts(signals, new_signals)` merges two `dict[str, list]` by extending existing keys and adding new ones. Used by `Parallel` when combining inputs/outputs across branches.

**Mapping direction note**: The whiteboard (IMG_0728) uses `destination → [output_names]` notation (e.g., `DONE_W_FWD: [tokens]`). The code uses the **inverse**: `output_name → [destinations]` (e.g., `"tokens": [GraphEdge("DONE_WITH_FWD")]`). Both representations are equivalent; the code's direction was chosen because it maps naturally to a node's `outputs` dict where each named output lists where it goes. -->

### 9.6 GraphEdge

```python
@dataclass
class GraphEdge:
    next_node: str
    name: str # name of the signal that is being passed (e.g., text_emb, latents, etc.)
    tensor_info: list[TensorPointerInfo] = field(default_factory=list)
    # Flags
    persist: bool = field(default=False)
    is_new_token: bool = field(default=False)
    output_modality: str = field(default="")  # text | image | video | audio (only for EMIT_TO_CLIENT)
    _persist_for_loop: bool = field(default=False)  # internal: don't cleanup between loop iters
```

`tensor_info` is a **list** of `TensorPointerInfo` objects holding data required for inter-worker tensor communication (see [Inter-Process Communication Section](#10-inter-process-communication)). A list allows a single graph edge to carry multiple tensors (e.g., multiple images for one input).

The `persist` flag is used for signals that **persist between forward passes** and need to be sent back to the conductor to be passed as inputs into the next forward pass.

The `is_new_token` flag is needed so that the conductor knows where to check for `<EOS>`, `<BOI>`, etc.

#### 9.6.1 Special/"Flag" `next_node` values
**(1) EMIT_TO_CLIENT (formerly STREAM)**

Signals that output should be streamed to the client via the API Server.

**Flow**: Worker → API Server (direct ZMQ, bypasses Conductor)

**Examples**:
- Text tokens streamed as they're generated
- Audio chunks streamed at detokenizer intervals
- Image sent after flow loop completes

**(2) STREAM**

A flag for **producer-consumer streaming between workers**. Distinct from EMIT_TO_CLIENT (which is client-facing).

**Purpose**: Enable inter-worker streaming without routing through the Conductor.

**Primary use case**: Qwen-Omni Thinker→Talker streaming. Both Qwen2.5-Omni and Qwen3-Omni use a Thinker-Talker architecture where the Thinker streams data to the Talker BEFORE the Thinker finishes generating text. What exactly gets streamed differs between the two models (see comparison in [Appendix E](#appendix-e-qwen25-omni-vs-qwen3-omni-comprehensive-architectural-comparison)).

**Flow**:
```
Thinker Worker ──STREAM (hidden states + token IDs)──→ Talker Worker
Thinker Worker ──WORKER_GRAPHS_DONE──→ Conductor
Conductor starts next forward pass
...
Conductor ──(at <EOS>)──→ Talker Worker: STOP signal (ZMQ msg: req_id, STOP)
```

**Why to have a STREAM flag**: This tells the producer (e.g., thinker) worker that an output will be consumed in a streaming fashion, i.e., the consumer (e.g., talker) will read the outputs while the next thinker forward pass starts.
This will allow the thinker to ensure that the tensors it produces remain in a queue until they have been fully read by the talker (whereas, for non-stream outputs, the producer worker can automatically overwrite the tensors for each new forward pass).

**Potential Talker Worker pseudocode** (from IMG_0725):

```python
# Talker worker state
buffer = {req_id: {"hidden_states": [...], "token_ids": [...]}}  # consumed from stream
status = {req_id: "WAITING" | "TALKING"}

# Talker loop (runs continuously):
while True:
    # 1. If received STOP from Conductor, remove req_id from buffer/status
    for stop_msg in recv_stop_signals():
        del buffer[stop_msg.req_id]
        del status[stop_msg.req_id]

    # 2. Consume from stream data flow (add new req_ids, buffer data)
    for stream_msg in recv_stream_data():
        if stream_msg.req_id not in buffer:
            buffer[stream_msg.req_id] = {"hidden_states": [], "token_ids": []}
            status[stream_msg.req_id] = "WAITING"
        buffer[stream_msg.req_id]["hidden_states"].extend(stream_msg.hidden_states)
        buffer[stream_msg.req_id]["token_ids"].extend(stream_msg.token_ids)

    # 3. For all req_ids: check if can talk (enough data in buffer)
    for req_id in buffer:
        if status[req_id] == "WAITING" and enough_in_buffer(req_id):
            status[req_id] = "TALKING"

    # 4. Run talker forward for current batch (all TALKING requests)
    batch = [req_id for req_id in status if status[req_id] == "TALKING"]
    talker_outputs = run_talker_batch(batch)

    # 5. Emit audio output to API Server (EMIT_TO_CLIENT)
    for req_id, audio_chunk in talker_outputs:
        stream_to_api_server(req_id, audio_chunk)
```

### 9.7 RequestQueues

The conductor maintains one `RequestQueues` per in-flight request:

```python
@dataclass
class RequestQueues:
    ready: list[GraphNode]   # Nodes with all inputs available
    waiting: GraphSection     # Remaining graph structure

    def process_new_inputs(
        self,
        new_inputs: list[GraphEdge]
    ) -> ProcessedInputs:
        """
        Processes all outputs that feed into the waiting graph section, and
        return a dictionary of external output edges (ones that are feeding
        to different worker graphs)
        """
        if self.waiting is None:
            return ProcessedInputs(
                routed_to_this_worker_graph=set(),
                for_other_worker_graphs=new_inputs,
            )

        new_inputs: DestToGraphEdges = get_node_to_inputs_mapping(new_inputs)
        ingested = self.waiting.ingest_inputs(new_inputs)
        external_outputs = sum( new_inputs.values(), start=[])
        
        self._update_ready_waiting()
        return ProcessedInputs(
            for_other_worker_graphs=external_outputs,
            routed_to_this_worker_graph=ingested
        )
```

The key insight: `ingest_inputs` consumes entries from its argument. After the call, only un-consumed entries remain in the dict -- these are external outputs (like EMIT_TO_CLIENT, DONE_WITH_FWD) that don't correspond to any waiting node.

`test/test_request_queues.py` contains a stress test of this system using a Show-o2-style graph with nested loops and parallel branches.

---

## 10. Inter-Process Communication

### 10.1 Message Types

Here, we detail the messages that can be sent to the conductor, worker, and API server via IPC.

**Communication to conductor** (`ConductorMessageType`):

| Flow | Direction | Purpose | Message Content |
|------|-----------|---------|-----------------|
**New Request** | API server → Conductor | Notify conductor of new work | `req_id`, `initial_signals: dict[str, list[TensorPointerInfo]]`, `initial_input_modalities`, `initial_output_modalities`, `input_metadata`, `model_kwargs` |
**Worker Graphs Done** | Worker → Conductor | Notify worker graph completion | `req_id`, `worker_graph_ids`, `persist_signals: dict[str, list[TensorPointerInfo]]`, `new_tokens: dict[str, list[int]]` |

**Communication to workers** (`WorkerMessageType`):

| Flow | Direction | Purpose | Message Content |
|------|-----------|---------|-----------------|
**New Request** | Conductor → Worker | Notify worker will be handling worker graphs this request | `req_id`, `worker_graph_ids`, `worker_graph_to_worker: dict[str, str]`, `initial_graph_walk`, `initial_inputs: list[GraphEdge]`, `per_request_metadata: dict` |
**Remove Request** | Conductor → Worker | Remove request upon `<EOS>` or rescheduling | `req_id` |
**Input Signals** | Conductor → Worker or Worker → Worker | Send inputs to graph nodes | `req_id`, `graph_walk`, `inputs: list[GraphEdge]`, `per_request_metadata: dict` |
**Tensor Received** | Worker → Worker or Worker → API server | ACK that RDMA tensor read has finished | `req_id`, `successful_tensors: list[NameAndUuid]`, `failed_tensor_ids: list[NameAndUuid]` |

The `per_request_metadata` field on `NewRequest` and `InputSignals` carries model-specific metadata from the conductor to workers (e.g., `cache_labels`, `snapshot_after` for BAGEL's multi-cache CFG orchestration).

**Communication to API server**:

| Flow | Direction | Purpose | Message Content |
|------|-----------|---------|-----------------|
**Result Chunk** | Worker → API server | Stream output | `req_id`, `modality`, `graph_edge: GraphEdge`, `metadata` dict |
**Request Complete** | Conductor → API server | Request has finished processing | `req_id`


**Design evolution note**: Note 14 of the design discussions stated: "Previously, the conductor handled what graph nodes are ready to run and which are waiting for inputs. Now that everything passes between workers via inter-worker communication, this no longer fits in the conductor. The ready/waiting logic moves to the worker level."
The conductor sends inputs and some state information (e.g., "we are currently in decode graph walk doing AR text generation") to the workers at the beginning of each forward pass, and the workers handle the rest via direct communication of signals and state (KV cache) to the appropriate destination worker. 
<!-- **Current resolution**: The final design retains ready/waiting queue tracking at the **Worker** level (via `RequestQueues`, Sections 6.2.3 and 9.8) for macro-level graph progression, while Note 14's insight is incorporated as a two-level split:

- The **Conductor** handles macro-level scheduling: which worker runs which node, graph-level readiness (are all inputs for a node available?), request lifecycle tracking, and worker assignment.
- The **Workers** handle micro-level readiness: buffering partial inputs (e.g., the Talker buffering stream data until enough has arrived to start talking), managing continuous batching (when to actually execute a dispatch), and inter-worker data reception.

This means the Conductor tracks "is this graph node logically ready?" while the Worker tracks "do I have enough data in my buffers to actually run this computation?" -->

### 10.2 Current Communication Implementation: Mooncake Transfer Engine + ZMQ

From the Mooncake research, the inter-worker communication layer should:

1. **Use the Mooncake Transfer Engine** with topology-aware, multi-path data transfer
2. **Support multiple protocols**: RDMA (for production), TCP (fallback), NVLink (intra-node), shared memory (same-machine)
3. **Stream layer-by-layer / chunk-by-chunk**: Do not wait for full computation to complete before beginning transfer
4. **Use pull-based transfer**: The receiving worker initiates data pull to handle burstiness naturally.

Mooncake includes a `mooncake-transfer-engine` package that we can use out-of-the box, along with ZMQ for passing of metadata.

Specifically, there are two layers of inter-worker communication:
- **(1) ZMQ**: For control messages, metadata, and notifications (e.g., "you can now read the latents from IP X at memory address Y", "I have completed this worker graph", "stop talking")
- **(2) Mooncake Transfer Engine**: For the actual heavy lifting of tensor data transfer (e.g., KV cache blocks and tensors transferred between worker graphs/workers and from workers to the API server).

**(1) Message-Passing Communication**:
The `BaseCommunicator` abstract class handles message-level worker communication, with interfaces for `send` and `get_all_new_messages`, and is implemented by `ZMQCommunicator`.
Each entity (worker, conductor, api server) has one `ZMQCommunicator` instance, and calls the `send` and `get_all_new_messages` methods when appropriate.

**(2) Tensor Communicator**:
The `TensorCommunicationManager` abstract class (a) holds all tensors used by a worker, and (b) handles tensor-level worker-worker (e.g., when `next_node` is a part of a worker graph that's in a different worker) and worker-api_server (streaming out) communication.
It is implemented by `MooncakeCommunicationManager`.

To read in tensors, the receiver first calls `communication_manager.start_read_tensors(...)`, which allocates an appropriately-sized buffer and calls
```python
self.engine.transfer_read_on_cuda(
    info.source_session_id,
    dst.data_ptr(),
    info.address,
    info.nbytes,
    stream.cuda_stream,
)
```
from Mooncake transfer engine to start a read.
The `start_read_tensors` returns immediately, but keeps track of a cuda Event for each `transfer_read_on_cuda`, which it can later query for completion.

To see what currently-being-read tensors have finished transferring, the receiver calls `communication_manager.get_ready_tensors()`, which queries all stored cuda Events and return the names of all ready tensors.
This also sends ACK messages back to the senders of those tensors, so that they can clean up any temporary buffers or state associated with those tensors.


**Tensor transfer high-level example**:
`Worker_0` has latents that are needed on the `Worker_1` (via its worker graph). `Worker_0` sends `address` and `n_bytes` information (via ZMQ). `Worker_1` allocates the required memory for latents, uses `transfer_read_on_cuda` to pull the tensors from `Worker_0`.
While the tensors are being transferred, `Worker_1` performs computation if applicable.
When the tensors are fully read in, `Worker_1` and sends a feedback message to `Worker_0` for confirmation.

**Actual worker main loop** (from `worker/worker.py`):
```python
def run(self) -> None:
    while True:
        # 1. Process ZMQ messages (new requests, input signals, removals)
        self._process_messages()

        # 2. Check for completed RDMA transfers, feed ready edges to worker graph queues
        self._check_ready_tensors()

        # 3. Pick next batch via MicroScheduler (selects node + requests)
        batch = self.scheduler.get_next_batch(self.worker_graphs_manager)
        if batch is None:
            continue

        # 4. Gather input tensors for the batch
        node_batch = self._build_node_batch(batch)

        # 5. Execute via engine (AR, enc_dec, etc.)
        engine = self.engine_manager.get_engine(batch.node_name)
        output = engine.execute_batch(node_batch)

        # 5b. Free consumed input tensors
        self._cleanup_consumed_inputs(batch)

        # 6. Route outputs through WorkerGraphsManager (determines destinations)
        routing_per_request = {
            rid: self.worker_graphs_manager.process_node_outputs(rid, node.outputs)
            for rid, node in batch.node_objects.items()
        }

        # 7. Store output tensors, register RDMA if needed
        self._store_outputs(batch, output, routing_per_request)

        # 8. Send outputs to other workers / conductor / API server
        for request_id in batch.node_objects:
            self._send_outputs(request_id, routing_per_request[request_id])
```

### 10.3 KV Cache Transfer

From Mooncake's architecture:
- KV cache blocks are **paged and hashed** using chain hashes (each block hash includes all preceding hashes)
- This enables **prefix-level deduplication** across the entire cluster
- Transfers are **overlapped with computation**: as each LLM layer finishes, its KV cache is asynchronously streamed

<!-- **For our system**: Take the IPC logic from Mooncake. Start with ZMQ for initial implementation; migrate to RDMA if profiling shows it's the bottleneck. -->

### 10.4 vLLM KV Connector Interface

From vLLM's codebase, the `KVConnector` abstraction provides a clean boundary:

```python
class KVConnector:
    def pre_forward(self, scheduler_output) -> None:
        """Load required KV cache before computation."""
    def post_forward(self, scheduler_output, wait_for_save=True) -> KVConnectorOutput:
        """Save and transfer KV cache after computation."""
```

This pattern generalizes to any inter-worker data transfer:
```python
class NodeConnector:
    def load_inputs(self, node_id, input_spec) -> None:
        """Asynchronously load inputs for a node."""
    def save_outputs(self, node_id, output_spec) -> None:
"""Asynchronously stream outputs to downstream nodes."""
    def register_buffers(self, buffer_spec) -> None:
        """Register memory regions for zero-copy transfers."""
```

### 10.5 Talker Worker Buffer Management

This can be handled directly by the `TensorCommunicationManager` implementation on the Thinker and Talker workers. The tensor manager on the Thinker maintains an output buffer of hidden states that it streams to the Talker, and the tensor manager on the Talker maintains an input buffer.

The Thinker keeps adding outputs to the buffer, removing them upon receipt of read ACK from the Talker.
The Talker reads inputs into its buffer, reading from the buffer once enough data is present.

The Talker worker also maintains a status flag for the request (e.g., `"status": {req_id: "WAITING" | "TALKING"}`).

**Talker loop**:
1. If received STOP from Conductor, remove req_id
2. Consume from stream (add new req_ids, buffer tensors)
3. For all req_ids: check if enough data to talk (status = TALKING or sufficient buffer)
4. Run Talker forward for current batch
5. Emit audio output to API Server

---

## 11. KV Cache Management

### 11.1 Architecture

KV cache management spans three levels:

| Level | Component | Responsibility |
|-------|-----------|----------------|
| **Global** | Conductor (Worker Registry) | Track which worker holds which request's KV cache; decide when to transfer |
| **Local** | Worker (KV Cache Manager) | Paged allocation, prefix caching, block management |
| **Transfer** | Transfer Engine (Mooncake) | Physical data movement between workers |

### 11.2 Paged KV Cache

Borrowed from vLLM:
- KV cache is divided into fixed-size **blocks** (e.g., 16 tokens per block)
- Each request has a **physical block table** mapping logical positions to physical blocks
- **Prefix caching**: Blocks with matching token sequences are shared via hash-based deduplication
- **Dynamic allocation**: Blocks are allocated on demand as sequences grow

Borrowed from VoxServe:
- Page allocation/deallocation logic
- KV page tracking per request (`kv_pages`, `kv_token_len`, `kv_last_page_len`)
- Integration with FlashInfer's paged attention wrappers

### 11.3 Cross-Worker KV Transfer

When a request moves between workers (e.g., prefill on Worker 0, decode on Worker 1):

1. Conductor decides to transfer (based on load balancing or co-location requirements)
2. Conductor notifies source worker: "stream KV cache for req_id to Worker 1"
3. Source worker uses Transfer Engine to send KV cache layer-by-layer
4. Destination worker receives blocks, maps them into its local KV cache manager
5. Destination worker notifies Conductor: "ready to serve req_id"

### 11.4 Tiered Storage (Future)

Following Mooncake's pattern, support tiered KV cache storage:
- **GPU VRAM**: Hot cache (active requests)
- **CPU DRAM**: Warm cache (recently accessed, prefix cache)
- **SSD**: Cold cache (evicted but recoverable)

### 11.5 Chunked Prefill

From Note 13: "Chunked prefill is NOT on the conductor level. Send everything to the worker and have the worker handle chunking."

This supports:
- Running prefill on an all-text chunk while a future image input is still being encoded
- Workers managing their own prefill scheduling without conductor involvement
- Inter-worker communication delivering encoded images to the worker as they become available

---

## 12. Concrete Request Flows

### 12.1 Text + Image Understanding (e.g., BAGEL, Qwen3-VL)

```
1. User sends: "Describe this image" + image.jpg
2. API Server data worker:
   a. Calls model.process_prompt("Describe this image") → tokenized text tensor
   b. Loads image.jpg → image tensor
   c. Registers tensors for RDMA, sends NewRequestConductor to Conductor
3. Conductor:
   a. Calls model.get_initial_forward_pass_args(["text", "image"], ["text"], initial_signals, model_kwargs)
      → Builds prefill schedule (e.g., for BAGEL: prefill_text, prefill_vit)
   b. Assigns worker graphs to workers (e.g., LLM → Worker 0, vit_encoder → Worker 1)
   c. Sends NewRequest to each worker with worker graph assignments
4. Prefill graph walk (schedule-driven):
   Step 0: prefill_text — Conductor sends text_inputs to Worker 0 (LLM)
   Step 1: prefill_vit — Conductor sends image_inputs to Worker 1 (vit_encoder)
     → Worker 1 runs ViT, sends vit_emb to Worker 0 via IPC
     → Worker 0 runs LLM forward (bidirectional for image tokens)
   Each step ends with WORKER_GRAPHS_DONE → Conductor
5. Conductor transitions to decode graph walk
6. Decode loop: For every forward pass, conductor sends previous token to Worker 0
   - Worker 0 runs LLM decode, emits new_token → API Server (EMIT_TO_CLIENT)
   - Worker 0 sends WORKER_GRAPHS_DONE with new_token to Conductor
7. At <EOS>, conductor sends REMOVE_REQUEST to all workers
8. Conductor finishes request
```

**Special: DeepStack multi-level feature injection** for Qwen3-VL requires ViT features from 3 intermediate layers to be injected at LLM layers 1, 2, 3. This is handled internally by the worker since ViT and LLM are co-located.

### 12.2 Image Generation -- BAGEL Pattern (Schedule-Driven, Frozen-KV Flow)

Output mode is known **upfront** from the API request's `output_modalities` (no BOI token detection). Prefill is schedule-driven: a sequential list of `(graph_walk_name, step_kwargs)` entries that walk through interleaved text and image inputs with multi-cache annotations for CFG.

```
1. User sends: "Generate a sunset over mountains" + reference_image.jpg
2. API Server data worker:
   a. Calls model.process_prompt("Generate...") → tokenized text tensor
   b. Loads reference_image.jpg → image tensor
   c. Registers tensors for RDMA send
   d. Sends NewRequestConductor to Conductor with initial_signals, modalities

3. Conductor:
   a. Calls model.get_initial_forward_pass_args(["text", "image"], ["image"], initial_signals, model_kwargs)
      → Builds prefill schedule with CFG annotations:
        [("prefill_text", {cache_labels: ["main", "cfg_img"]}),
         ("prefill_vae",  {cache_labels: ["main"], snapshot_after: ("main", "cfg_text")})]
   b. Assigns worker graphs to workers:
      LLM → Worker 0, vit_encoder + vae_encoder + vae_decoder → Worker 1
   c. Sends NewRequest to workers with worker graph assignments

4. Prefill graph walk (multiple forward passes, schedule-driven):
   Step 0: prefill_text
     - Conductor sends text_inputs to Worker 0 (LLM node)
     - Worker 0: LLMSubmodule._forward_prefill_text()
       → embed_tokens → LLM forward (causal) writing to "main" and "cfg_img" caches
     - Worker 0 sends WORKER_GRAPHS_DONE → Conductor
   Step 1: prefill_vae
     - Conductor sends image_inputs to Worker 1 (vae_encoder node)
     - Worker 1: VAE encode → sends vae_emb to Worker 0 (via IPC)
     - Worker 0: LLMSubmodule._forward_prefill_vae()
       → BOI + vae_emb + EOI → LLM forward (bidirectional) writing to "main" only
       → snapshot("main", "cfg_text") — creates text-only CFG cache
     - Worker 0 sends WORKER_GRAPHS_DONE → Conductor

5. Conductor: all prefill steps done → transitions to image_gen graph walk
   (no BOI token, direct transition from schedule)

6. Image generation (49 flow matching steps on Worker 0):
   For each step:
     a. LLMSubmodule._forward_image_gen() runs 3 LLM forward passes:
        - set_active_label("main") → LLM forward (read-only KV, write_cache=False)
        - set_active_label("cfg_text") → LLM forward (read-only KV)
        - set_active_label("cfg_img") → LLM forward (read-only KV)
     b. llm2vae projection on all 3 outputs
     c. CFG velocity combination:
        v_final = v_cfg_img + img_scale * (v_cfg_text + text_scale * (v_main - v_cfg_text) - v_cfg_img)
     d. Euler step: x_{t+1} = x_t + v_final * dt
   Loop completion → latents ready

7. Transfer latents from Worker 0 to Worker 1 via RDMA
8. Worker 1 runs VAE decode → emits image → API Server (EMIT_TO_CLIENT)
9. Conductor marks request done (image_gen graph walk complete = one image per request)
```

**Key differences from previous design**: No BOI token detection — output mode known upfront. 5 separate graph walk graphs instead of one monolithic graph. Multi-cache CFG (3 caches: main, cfg_img, cfg_text) orchestrated via CacheHandle with `set_active_label()` and `snapshot()`, not by tripling the batch size. Schedule annotations (`cache_labels`, `snapshot_after`) flow from model → conductor → worker → engine → submodule.

**With think_mode**: After prefill, first enters `decode` graph walk to generate reasoning text. When EOS is detected, transitions to `image_gen` graph walk (thinking done, now generate).

### 12.3 Image Generation -- Show-o2 Pattern (Interleaved Flow)

```
1. User sends: "Generate a sunset over mountains"
2. API Server → Conductor: {text: "Generate...", model: "show-o2"}
3. Conductor:
   a. Instantiates a new `RequestData` object
   b. Assign: ALL nodes → Worker 0 (forced co-location: LLM + diffusion_head + euler_step)
   c. vae_decoder → Worker 1
   d. Ready: text_emb (has text input)
4. Dispatch text_emb → Worker 0
... Worker 0 runs decode (sending appropriate WORKER_GRAPHS_DONE signals to coductor), produces <BOI>, which conductor sees and starts image generation graph walk ...
5. Worker 0 starts Loop(50):
   For each of 50 iterations:
     a. LLM full forward pass (text_emb + img_emb + current latents)
     b. Diffusion head processes LLM hidden states at image positions
     c. Euler step updates latents
   End loop
   - Loop completion → Conductor: latents ready
6. Conductor: vae_decoder becomes ready
7. Dispatch vae_decoder → Worker 1
8. Worker 1: VAE decode → emit image → API Server
... Conductor waits for <EOS> ...
9. Conductor finishes request
```

**Key**: 50 LLM forward passes (no KV cache reuse across flow steps). CFG doubles all compute. This is why Show-o2 is much more expensive than BAGEL for image generation.

### 12.4 Speech-to-Speech -- Qwen3-Omni (Thinker-Talker)

```
1. User sends: audio recording + image
2. API Server → Conductor: {audio: audio.wav, image: img.jpg, model: "qwen3-omni"}
3. Conductor:
   a. Instantiates a new `RequestData` object
   b. Assign: vit + audio_encoder + thinker → Worker 0 (30B MoE)
              talker + mtp + code2wav → Worker 1 (3B MoE)
   c. Ready: vit_encoder (has image), audio_encoder (has audio)
4. Dispatch encoder inputs → Worker 0 (parallel). Send WORKER_GRAPHS_DONE signal.
Thinker becomes ready on Worker 0 (internal Worker scheduling).
5. Worker 0 runs thinker decode loop:
   - Emits text tokens → API Server (EMIT_TO_CLIENT)
   - Streams layer-0 embeddings + layer-24 hidden states (accept_hidden_layer) + token IDs → Worker 1 (stream)
6. Once WORKER_GRAPHS_DONE for the thinker is sent to the conductor, the conductor immediately starts a new forward pass. 
7. Worker 1 (talker loop, running concurrently):
   - Buffers incoming hidden states + token IDs from stream
   - When sufficient data: runs talker AR decode
   - Talker → MTP module (residual codebooks) → Code2Wav → audio chunks
   - Emits audio → API Server (EMIT_TO_CLIENT)
   - On receiving STOP from Conductor: finishes current buffer, winds down
8. Conductor finishes request when both thinker and talker complete
```

**Key features**:
- Thinker and Talker run CONCURRENTLY on different GPUs
- Stream enables hidden state communication before thinker finishes
- Layer-0 embeddings + layer-24 hidden states + text token IDs travel via stream (see [Appendix E](#appendix-e-qwen25-omni-vs-qwen3-omni-comprehensive-architectural-comparison) for Qwen2.5-Omni vs. Qwen3-Omni differences)
- Talker maintains per-request buffer with WAITING/TALKING status

### 12.5 SpeechLM -- VoxServe Pattern (Orpheus, CosyVoice, etc.)

```
1. User sends: "Hello world" (text-to-speech)
2. API Server → Conductor: {text: "Hello world", model: "orpheus"}
3. Conductor:
   a. Instantiates a new `RequestData` object
   b. Assign: preprocess + LLM + detokenizer → Worker 0
4. Worker 0 runs VoxServe-style loop (with inputs passing in from the conductor at every forward pass, and WORKER_GRAPHS_DONE messages passing from the worker to the condutor):
   a. Preprocess: tokenize text, allocate KV cache, init decoder cache
   b. LLM prefill
   c. LLM decode loop (continuous batching):
      - At model-specific intervals (e.g., 28 tokens for Orpheus, 10 for CSM, 25 for GLM): run detokenizer
      - Detokenizer produces audio chunk → emit to API Server (EMIT_TO_CLIENT)
5. Conductor finishes request upon seeing <EOS>
```

**Key**: This is the existing VoxServe pattern, preserved as-is within the new architecture. The model-level node execution path internally handles the prefill→decode→detokenize pipeline.

---

## 13. Technology Decisions

### 13.1 What to Borrow from VoxServe

| Component | How |
|-----------|-----|
| API Server (`APIServer` class in launch.py) | Adapt -- generalize message format, remove VoxServe-specific endpoints |
| FlashInfer attention wrappers (`flashinfer_utils.py`) | Copy directly (`FlashInferPrefillWrapper` + `FlashInferDecodeWrapper`) |
| Paged KV cache allocation (in `CudaGraphWorker`) | Copy directly (tensor allocation pattern) |
| CUDA graph capture pattern (`_initialize_decode_cuda_graphs`) | Copy directly (capture/replay logic) |
| ZMQ IPC socket setup (launch.py + scheduler/base.py) | Copy directly (context init, socket binding, HWM config) |
| Continuous batching pattern (scheduler/base.py `_step` + `_select_*`) | Adapt for multi-node |
| `DecoderCache` generic state container (tokenizer/base.py) | Copy directly (slicing, copy_from, cat, device movement) |
| Streaming audio/image output (`async_stream_chunks`) | Generalize to multi-modality |

### 13.2 What to Borrow from vLLM

| Component | How |
|-----------|-----|
| Scheduler interface (`SchedulerInterface`, `SchedulerOutput`) | Adapt -- computation-agnostic design works for arbitrary nodes |
| Block-based KV cache management (`KVCacheManager`) | Adapt -- replace "KV cache" with "intermediate activation cache" |
| Executor abstraction (`collective_rpc`, pluggable backends) | Reference -- good pattern for distributed execution |
| Request state management | Reference -- extend for computation DAG tracking |
| Prefix caching (hash-based block dedup) | Borrow directly for KV cache sharing |

### 13.3 What to Borrow from vLLM-Omni

| Component | How |
|-----------|-----|
| Node configuration YAML format | Adapt directly |
| `OmniRequest` with `prompt_embeds` field | Reference -- extend for arbitrary inter-node tensors |
| `OmniConnectors` IPC (shared memory, serialization) | Borrow directly for inter-worker data transfer |
| `PromptEmbedsPayload` tensor serialization | Borrow directly |
| Diffusion engine/scheduler/worker separation | Reference pattern for DiT Worker |

### 13.4 What to Borrow from Mooncake

| Component | How |
|-----------|-----|
| KV cache IPC logic (Note 12) | Borrow -- transfer engine for inter-worker communication |
| Layer-by-layer stream during computation | Borrow pattern for stream implementation |
| Content-addressable block hashing for dedup | Borrow for prefix caching across workers |
| Topology-aware multi-path transfer | Future -- start with ZMQ/TCP, add RDMA later |
| Prediction-based early rejection | Borrow pattern for SLO-aware scheduling |

### 13.5 What to Write Custom

| Component | Rationale |
|-----------|-----------|
| **Conductor** (all) | Unique to our architecture: request lifecycle management, worker graph assignment, graph walk transitions |
| **Model ABC** (all) | Model-specific computation graphs, graph walk definitions, NodeSubmodule wrappers |
| **Worker internals** (WorkerGraphsManager, MicroScheduler) | Graph-aware ready/waiting queues, batch scheduling, worker graph completion detection |
| **Worker Registry** | Capability-based routing unique to our design |
| **Computation Graph Model** (GraphSection hierarchy) | Working implementation exists |
| **stream flag handling** | Thinker-Talker streaming pattern specific to our system |
| **WorkerGraph persistence** logic | Re-plan only when modalities change |

### 13.6 Insert Later (Not Initial Implementation)

- **FlashInfer** optimized attention kernels (beyond what VoxServe provides)
- **CUDA graphs** for all worker types (start with DiT Worker, extend to vLLM Engine Worker)
- **RDMA** inter-worker communication (start with ZMQ/TCP)
- **RadixAttention** prefix sharing (start with simple hash-based prefix matching)
- **Autoscaling** (start with static configuration)

---

## 14. Resolved Design Tensions

### 14.1 Tension 1: Per-Step Dispatch vs. Loop-as-Unit

**Apparent problem**: The conductor loop (step h) shows per-step dispatch (worker completes one step, reports back, conductor dispatches next). But Loop(50) dispatches 50 steps as a unit.

**Resolution**: These are two ends of a spectrum controlled by Loop dispatch granularity.
- `Loop(50)`: Worker runs all 50 steps, sends one completion. Best for tightly-coupled operations (Show-o2 interleaved flow+LLM).
- `Loop(1)`: One completion per step. Best when the conductor needs control between steps.
- `Loop(10) × 5`: Middle ground. Enables round-robin across workers, reduces HoL blocking, enables checkpointing between chunks.

The conductor loop step h just says "check for messages" -- it doesn't prescribe frequency.

### 14.2 Tension 2: Who Constructs needs/produces/ptr?

**Resolution**: The **Execution Strategy** (on the model) defines the **template** (all possible routes). The **Conductor** instantiates it per-request by enabling/disabling routes and assigning workers. Steps a-f are the instantiation process.

### 14.3 Tension 3: Request Management vs. Node Management Timing

**Resolution**: Request management (steps a-f) runs ONE TIME per new request or per requeue after forward pass completion. Node management (steps g-i) runs EVERY conductor loop iteration. They operate at different timescales. No conflict.

### 14.4 Tension 4: Two Diagrams of Step 2 (Image Generation)

**Resolution**: They are TWO DIFFERENT model architectures:
- **Diagram 1 (BAGEL pattern)**: LLM decode runs once, then 24 flow steps each re-entering the LLM backbone with frozen KV
- **Diagram 2 (Show-o2 pattern)**: LLM_dec inside the loop, 50 × (LLM + diffusion_head + euler_step) with NO KV reuse across steps

Not an expanded view of the same thing.

### 14.5 Old Design Problem: Pool-Worker Ownership

**Resolution**: Pools are eliminated entirely. Workers declare capabilities. The Worker Registry indexes capabilities. The Conductor routes by capability + affinity. No pool "owns" a worker.

### 14.6 Old Design Problem: execution_plan() on GenerationStrategy

**Resolution**: The execution plan (computation graph) belongs on the **model**, not a separate strategy. The model provides `get_graph_walk_graphs()` and graph-walk orchestration helpers; execution is performed by engine-backed nodes on workers.

### 14.7 Old Design Problem: Multiple Workers on Same GPU

**Resolution**: One worker process per GPU (Principle P5). Co-location = one UnifiedWorker process with multiple model components loaded, not separate processes competing for VRAM.

### 14.8 Old Design Problem: Interleaved vs. Independent Flow Heads

**Resolution**: The computation graph and node design express this directly.
- **BAGEL**: Flow matching is absorbed into the LLM node as a "fat node" (`LLMSubmodule`). The LLM does 3-pass CFG + llm2vae + Euler step internally via `_forward_image_gen()`. The `image_gen` graph walk graph is `Loop(LLM) → vae_decoder`. CacheHandle enables multi-cache CFG without the engine knowing about CFG. Only the final latents → VAE can be disaggregated.
- **Show-o2**: LLM + diffusion_head + euler_step are all inside a `Loop` in `Sequential`. Config YAML node_groups forces them onto the same worker.

| Architecture | Flow Separate from LLM? | Cross-GPU Transfers if Separated | Config |
|---|---|---|---|
| BAGEL | No (flow uses LLM backbone, frozen KV) | ~24 query tensors + ~24 hidden states if separated | Co-locate LLM + flow; only VAE separate |
| Show-o2 | No (interleaved, LLM invoked per step) | 2 × 50 = 100 if separated | MUST co-locate |
| JanusFlow | No (interleaved) | 2 × 30 if separated | MUST co-locate |
| Janus Pro | N/A (pure AR) | 0 | Standard AR |
| Qwen2.5-Omni | Yes (separate Thinker/Talker models) | Dense hidden states (~3584-dim/token) + token IDs, streamed via stream | Separate workers OK |
| Qwen3-Omni | Yes (separate Thinker/Talker models) | Both layer-0 embeddings + layer-24 hidden states (~2048-dim/token each, both sent for all tokens) + token IDs, streamed via stream. Talker-side routes text_projection, multimodal_hidden_projection. | Separate workers OK |

---

## 15. Open Questions

### 15.1 Designed but Not Yet Implemented

1. **Worker selection policy**: How the conductor chooses among capable workers (load balancing, KV affinity, SLO-awareness). Currently random selection for data-parallel ranks.

2. **Batching at the conductor level**: Whether the conductor should batch-select which requests' nodes to dispatch together, or leave all batching to workers.

3. ~~**Model hierarchy / interface**: Resolved — `Model` ABC in `model/base.py` with `get_graph_walk_graphs()`, `get_node_engine_types()`, `get_submodule()`, `process_prompt()`, etc. `BagelModel` and `DummyModel` are concrete implementations.~~

4. **Streaming across disaggregated nodes**: How streaming chunks (audio, partial images) work when nodes are on different GPUs. Currently: workers stream directly to API Server.

5. **SLO prediction / early rejection**: The Mooncake-style prediction-based rejection is referenced but not specified for our system.

6. **Error handling / fault tolerance**: What happens when a worker crashes mid-generation. Recovery, request reassignment, state checkpointing.

7. **Worker health monitoring**: Heartbeat mechanism, failure detection.

8. **Auto-scaling policies**: When/how to add/remove workers based on load.

9. **Partial output routing**: The current routing maps `{enabled_node: [outputs feeding to the node]}`, routing whole named outputs. But some nodes produce a single tensor where different *slices* need to go to different destinations (e.g., LLM hidden states at image positions → flow_head, hidden states at text positions → text decoder). Either nodes must produce separately-named outputs for each slice, or the routing must support index-based sub-selection.

10. **Loop breakup policy**: When the conductor breaks `Loop(50)` into `Loop(10) × 5` for HoL blocking avoidance, round-robin scheduling, or checkpointing -- is this static (from config YAML) or dynamic (based on queue depth, worker load)? The motivations are listed but the policy is undefined.

11. **Batched cross-request FlashInfer**: Current AR engine executes per-request (one CacheHandle per request). Batching FlashInfer calls across requests (single `plan()` + `run()` for multiple requests' attention) is a future optimization.

### 15.2 Explicitly Deferred

1. **Tensor parallelism / pipeline parallelism**: LLM can be decomposed as LLM_part1, LLM_part2, etc. (Note 1). Same as having different nodes in series. Deferred to post-v1.

2. **Full RadixAttention tree**: Start with simple prefix matching; full tree-based sharing later.

3. ~~**RDMA inter-worker communication**: Start with ZMQ/TCP; add Mooncake Transfer Engine when profiling shows transfer is the bottleneck.~~

4. **CUDA MPS / MIG for GPU sharing**: Currently one process per GPU. Multi-tenant GPU sharing deferred.

### 15.3 Requires Further Research

1. **Scheduling estimated runtimes** (Note 2): Need per-node runtime estimates, dynamically updated based on hardware load and previous task durations.

~~2. **Nested loop scheduling**: When a `Loop(10)` chunk completes, should the conductor assign the next chunk to the same or different worker? Depends on state locality vs. load balancing.~~

3. **World model integration**: World models (DreamerV3, DIAMOND, V-JEPA 2, Cosmos) have fundamentally different characteristics from SpeechLMs/VLMs:
   - **State structure**: RSSM state (deterministic GRU hidden `h_t` + stochastic categorical `z_t`, e.g., DreamerV3 uses 32×32 categorical) vs. KV cache. This is a tiny rolling state (~few KB) rather than a growing sequence cache (~MB-GB).
   - **Loop structure**: Environment step loop (observe → encode → RSSM → predict → act) at real-time control frequencies (e.g., 20Hz for robotics), not token-by-token generation.
   - **Session model**: Episode-based (start, steps, terminal) not session-based. Requests are long-running environment interactions, not one-shot generations.
   - **Actor-Critic on imagined trajectories**: World models interleave inference with imagination rollouts (predict future states without env interaction), a compute pattern unlike anything in our current graph model.
   - **Candidate approach**: A specialized `WorldModelWorker` wrapping RSSM state update + prediction heads, with the Conductor managing the env-step loop. Alternatively, express the RSSM as a `Loop(GraphNode(...))` where the node handles the encode→dynamics→predict cycle.

---

## Appendix A: Pipeline Parallelism Note

From Note 1: "LLM can be decomposed as LLM_part1, LLM_part2, etc. Same as having different components in series."

This means pipeline parallelism is naturally expressed in our computation graph:
```python
Sequential([
    GraphNode(name="llm_layers_0_15", ...),
    GraphNode(name="llm_layers_16_31", ...),
])
```
Each sub-node can be assigned to a different worker. The conductor handles the handoff. This is a future extension, not initial implementation.

## Appendix B: Scheduling Detail (Note 3)

Each node needs to know:
1. What inputs from previous nodes it needs
2. What future nodes its outputs enable

This must work across LLM forward passes. When any worker completes:
- Conductor sees from (2) what future nodes these outputs contribute to
- For each future node, conductor checks from (1) whether it's ready
- Must also encode token indices minutia for interleaved image and text token ordering

**Token indices**: All data carries token indices throughout the pipeline. Every piece of data knows its position in the original sequence, critical for multimodal attention masking.

## Appendix C: Node Completion Signaling (Note 11)

When a node within a worker graph completes, the worker sends a "node completed" message to the conductor. This doesn't necessarily trigger any action but is important for:
- SLO-aware scheduling (estimating remaining time)
- Progress monitoring
- Debugging

The message can include a flag saying "this also means worker graph completed" to distinguish intra-worker graph progress from worker graph completion.

## Appendix D: Design Notes from Atindra & Naomi

These notes were captured during whiteboard sessions and design discussions. They are referenced throughout this document as "Note N".

---

**Note 1:** pipeline parallelism note: the LLM can be decomposed as LLM_part1, LLM_part2, etc... This should be the same as having different components in series

**Note 2:** on scheduling: need to store estimated runtimes of nodes, and update dynamically based on current hardware load / how long previous tasks took / etc.

**Note 3:** on token routing / scheduling: each node needs to know (1) what inputs from previous nodes it needs, and (2) what future nodes its outputs enable. This may also need to work across LLM forward passes. When any worker completes a task, it will send the outputs back to the conductor. The conductor will see from (2) what future nodes these outputs contribute to. For each of these future nodes, it will know from (1) whether the node is ready to go or whether it needs to wait for more work to finish (e.g., the text encoding has finished, but image encoding is still working, and the LLM decode node needs both). We will have to also encode, somewhere in here, the minutia of token indices and such so that we know what order to place, e.g., interleaved image and text tokens.

**Note 4:** completion signaling (6) and batching across requests (1): can take these mechanisms from voxserve

**Note 5:** Streaming: Is either just another flag like EMIT_TO_CLIENT (aka/formerly STREAM in the diagrams above) or another process (TBD). It signifies producer-consumer streaming of tensors (between nodes like the Thinker node and the Talker node). Streaming makes sense because:
- We need inter-worker communication (we would not need it if we relied on the conductor to pass the data around to the talker because the DONE_WITH_FWD flag already wakes the conductor up and the conductor already has the computation graph).
- Qwen2.5-Omni which sends hidden states from thinker to talker (so before DONE_WITH_FWD flag is on). The Qwen3-Omni architecture works only with DONE_WITH_FWD flag without streaming if we are willing to take the conductor communication hit.
  - **[Correction]**: The original note said "layer 18." Verified findings: Qwen2.5-Omni sends **last-layer** hidden states + input embeddings (element-wise sum, 3584-dim/token), NOT from a specific intermediate layer. Qwen3-Omni sends from layer 24 (`accept_hidden_layer=24` in the released model config).

**Note 6:** For every forward pass, we are making a computation graph and dispatching it. Have worker graphs all ready per request. The scheduling part of conductor launches them when they are done (for example flow part of worker graph is launched at \<BOI\> and killed at \<EOI\>). We keep the talker worker graph alive throughout.

**Note 7:** idea on persistence of worker graphs: For every request, the conductor holds the current computation graph / execution plan. When the next forward pass starts, it retrieves a new execution plan from the model IF the input or output modalities change. If we have a new execution plan, the conductor formulates the new plan as a series of additions / subtractions from the current computation graph.
- The addition/subtraction mechanism can be trivial: the model can produce the "full possible computation" graph, and the get\_execution\_plan function can return whether each node is active or not.
- `showo2_model.execution_plan(input_modalities, output_modalities, metadata)` -- per-request, per-forward pass
- This solves a few issues:
  - reduces recomputation of execution plans (don't need to recompute for each fwd pass, just when the input or output modalities change)
  - how to specify that the "talker is on for the duration of the request" paradigm
  - how to make the preference for, e.g., decode for request0 always running on GPU1 to prevent KV cache transfer, a first-class concept

*(Note 8 does not exist -- numbering skips from 7 to 9)*

**Note 9:** modification of needs/produces/edge design: every worker graph has a list of inputs that it needs. We merge the produces/edge concepts: every worker graph can have multiple outputs that feed into different worker graphs, some of which will be ready in the middle of the worker graph execution. This is handled by multiple edges to next worker graphs, each of which is associated with the corresponding produced tensors.

**Note 10:** Whenever we change worker graphs, the conductor should send the corresponding workers information to the extent of "when you get tensors X and Y, run worker graph Z"

**Note 11:** When a node within a worker graph completes, send a "node completed" message back to the conductor. This doesn't really trigger anything but is important for SLO-aware scheduling. We can also have a flag in the message saying "this also means the worker graph has completed"

**Note 12:** KV cache management: take the IPC logic from mooncake

**Note 13:** chunked prefill: Not on the conductor level. Just send everything to the worker and have the worker handle chunking. This supports, e.g., running prefill on an all-text chunk while a future image input is being encoded because everything is now through inter-worker communication.

**Note 14:** note on scheduling: Previously, we had the conductor handling what graph nodes are ready to run and which are still waiting for their inputs. Now that we have everything passing between workers via inter-worker communication, this no longer fits into the conductor. We still need this logic (which graph nodes are ready, which are still waiting for inputs), but it'll be at the worker level. I'm working on code/pseudocode for that right now.

**Note 15:** NXT_STEP has been renamed to DONE_WITH_FWD. They are one and the same.

**Note 16:** For now, continuous batching, cuda graphs, chunked prefill etc. happen on the worker and we have some mooncake style inter-worker communication.

**Note 17:** Should we fashion our workers after voxserve in any way (though they are highly specific for SpeechLMs) or just use vLLM Engine Worker for standard AR nodes: Thinker, Talker, standard LLM decode and FlashInfer Custom Worker (interleaved nodes: Show-o2 LLM+flow loops, shared-backbone routing) and DiT Worker (diffusion nodes)?

----

## Appendix E: Qwen2.5-Omni vs. Qwen3-Omni: comprehensive architectural comparison

These two models share the Thinker-Talker pattern but differ substantially in architecture, what flows via stream, and how speech is generated. Verified from:
- [HuggingFace Transformers Qwen2.5-Omni implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py)
- Qwen2.5-Omni technical report (arXiv:2503.20215)
- Qwen3-Omni technical report (arXiv:2509.17765)
- [Qwen3-Omni-30B-A3B-Instruct config.json](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/main/config.json)
- [HuggingFace Transformers Qwen3-Omni implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_omni_moe)
- vllm-omni Qwen2.5-Omni: [stage_input_processors/qwen2_5_omni.py](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/stage_input_processors/qwen2_5_omni.py), [models/qwen2_5_omni/qwen2_5_omni_talker.py](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/models/qwen2_5_omni/qwen2_5_omni_talker.py)
- vllm-omni Qwen3-Omni: [models/qwen3_omni/qwen3_omni.py](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py), [stage_input_processors/qwen3_omni.py](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/stage_input_processors/qwen3_omni.py), [models/qwen3_omni/qwen3_omni_moe_code_predictor_mtp.py](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_code_predictor_mtp.py), [models/qwen3_omni/qwen3_omni_code2wav.py](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_code2wav.py)

**Architecture:**

| | **Qwen2.5-Omni-7B** | **Qwen3-Omni-30B-A3B** |
|---|---|---|
| **Thinker** | Dense Transformer, 28 layers, h=3584, ~7B params | MoE Transformer, 48 layers, h=2048, 128 experts (8 active), ~30B total / ~3B active |
| **Talker** | Dense dual-track Transformer, 24 layers, h=896, ~1B params | MoE Transformer, 20 layers, h=1024, 128 experts (6 active), ~3B total / ~0.3B active |
| **MTP module** | None | Dense Transformer, 5 layers, ~80M params — predicts residual codebooks 1–15 after Talker predicts codebook 0 |
| **Audio encoder** | Whisper-large-v3 (from Qwen2-Audio) | AuT (Audio Transformer, 650M params, trained from scratch on 20M hours) |
| **Vision encoder** | Qwen2.5-VL ViT (~675M) | SigLIP2-So400M (~540M, from Qwen3-VL) |
| **Audio codec** | Single codebook, 8193 tokens, 25 Hz (40ms/frame) | 16-layer RVQ, 2048 codebook, 12.5 Hz (80ms/frame) |
| **Speech decoder** | Flow-Matching DiT (22 layers, 10 ODE steps) + BigVGAN, ~449M. Block-wise: must wait for context window. | Causal ConvNet (Code2Wav), ~200M, single forward pass. Frame-by-frame: immediate waveform after each token. |
| **TMRoPE** | First 16 angles = temporal, 40ms/ID, 2s chunk interleaving | Interleaved 24/20/20 angles (temporal/height/width), 80ms/ID, direct absolute-time alignment (no chunking) |

**Thinker→Talker data flow (what travels via stream):**

| | **Qwen2.5-Omni** | **Qwen3-Omni** |
|---|---|---|
| **Which hidden states?** | **Last-layer** output + input embeddings (element-wise sum): `final_hidden[i] + input_embed[i]`, 3584-dim per token. Multimodal positions (audio/image/video) are zeroed out in the input embeddings before summing. | **Both** layer-0 embeddings **and** layer-24 hidden states are transferred for **all** tokens (full-sequence tensors). `accept_hidden_layer=24` in the released model config (note: the HuggingFace Transformers default is 18, but the actual model overrides to 24). The **selective routing** happens on the **Talker side**: text tokens → `text_projection(layer_0_embed)`, multimodal tokens (audio/image/video) → `hidden_projection(layer_24_hidden)`. Assistant-turn and streaming decode tokens always use `text_projection(layer_0_embed)`. |
| **Text token IDs** | Original prompt token IDs + generated token IDs. The Talker uses these to **recompute TMRoPE position IDs internally** via its own `get_rope_index()`. TMRoPE IDs are NOT directly transferred. | Full sequence token IDs (prompt + generated). Used for ChatML segment parsing and positional encoding. |
| **Text/multimodal decoupling** | Talker consumes Thinker's hidden states + text token embeddings **together** (coupled — same representation for all tokens). | Decoupled: text tokens use shallow layer-0 embeddings (via `text_projection`), multimodal tokens use deep layer-24 hidden states (via `hidden_projection`). The Talker builds a `multimodal_mask` from token IDs to route each token. This decoupling enables separate system prompts for Thinker and Talker, and allows RAG/safety filters to intervene on text before it reaches the Talker. |
| **Projection** | Single `nn.Linear(3584→896)` (`thinker_to_talker_proj`). The 3584 is the shared `embedding_size` (both codec embeddings and thinker hidden states live in this space); the 896 is the talker's internal transformer `hidden_size`. | Two separate MLPs: `text_projection` and `hidden_projection`, each `Linear(2048→2048)→SiLU→Linear(2048→1024)`. |
| **Talker input formula** | `talker_input = thinker_to_talker_proj(codec_embed(codec_token) + thinker_hidden)` — element-wise add then project. | User-turn text: `text_projection(layer_0_embed)`. User-turn multimodal: `hidden_projection(layer_24_hidden)`. Assistant-turn: always `text_projection(layer_0_embed)`. Streaming decode: always `text_projection(layer_0_embed)`. Codec embeddings added separately. |
| **Speech decoder** | AR codec → Flow-Matching DiT (10 ODE steps) + BigVGAN → waveform | AR codec (codebook 0) → MTP module (codebooks 1–15) → Code2Wav ConvNet → waveform |
| **Data volume per token** | Dense: ~3584 × 2 bytes (fp16) ≈ **7 KB/token** | Dense: ~2048 × 2 bytes (fp16) × 2 streams (layer-0 + layer-24) ≈ **8 KB/token** total |

**Key insight**: The stream channel carries **dense hidden-state vectors**, not lightweight position IDs. For Qwen3-Omni, both layer-0 and layer-24 tensors are sent (so ~2 × 2048 × 2 bytes ≈ **8 KB/token** total). For a 200-token response, that is ~1.4 MB of hidden states for Qwen2.5-Omni, ~1.6 MB for Qwen3-Omni (two streaming channels). This has bandwidth implications for inter-GPU transfer.

**Qwen3-Omni streaming consideration**: Because Qwen3-Omni extracts hidden states from an intermediate layer (layer 24 out of 48) rather than the final layer, the Thinker's text generation (which uses all 48 layers) is architecturally decoupled from the Talker's input. This means streaming could theoretically be replaced by DONE_WITH_FWD + Conductor forwarding. However, the whiteboard notes present this as a trade-off rather than a settled decision. Streaming avoids the conductor communication hit but adds complexity. For Qwen2.5-Omni, the last-layer hidden states are tightly coupled to the Thinker's text generation, making streaming more clearly necessary.
