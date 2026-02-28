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
10. [Communication Protocols and Flags](#10-communication-protocols-and-flags)
11. [Inter-Worker Communication](#11-inter-worker-communication)
12. [KV Cache Management](#12-kv-cache-management)
13. [Concrete Request Flows](#13-concrete-request-flows)
14. [Technology Decisions](#14-technology-decisions)
15. [Resolved Design Tensions](#15-resolved-design-tensions)
16. [Open Questions](#16-open-questions)
- [Appendix A: Pipeline Parallelism Note](#appendix-a-pipeline-parallelism-note)
- [Appendix B: Scheduling Detail](#appendix-b-scheduling-detail-note-3)
- [Appendix C: Stage Completion Signaling](#appendix-c-stage-completion-signaling-note-11)
- [Appendix D: Design Notes from Atindra & Naomi](#appendix-d-design-notes-from-atindra--naomi)

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

These principles are derived from 13 specific problems identified in the scrapped old design (see Section 15 for details).

| # | Principle | Rationale |
|---|-----------|-----------|
| P1 | **Separate logical from physical** | Computation stages are logical; workers are physical GPU processes. The mapping is deployment config, not architecture. |
| P2 | **Workers declare capabilities, not pool membership** | Workers report what they can do (`encode_image`, `llm_prefill`, `flow_step`, etc.). The conductor routes by capability + affinity. No "pool ownership" confusion. |
| P3 | **Execution plan belongs on the model** | The model knows its own architecture. The model provides `get_execution_strategy()` which returns both the computation graph and the `run_stage()` function. |
| P4 | **Co-location expresses tight coupling** | For Show-o2/JanusFlow where the flow head is interleaved with the LLM, the execution plan declares them as a single stage. The conductor forces co-location. |
| P5 | **One process per GPU** | Never multiple worker processes on the same GPU without explicit coordination. Co-location = one process with multiple models loaded. |
| P6 | **Conductor is the inter-worker coordinator** | The conductor orchestrates lifecycle, selects workers, coordinates handoffs. It never touches GPU tensors and never manages batching. |
| P7 | **Workers handle internal batching** | Each worker has its own micro-scheduler for continuous batching. The conductor dispatches work; the worker decides when to execute. |
| P8 | **Start fresh, borrow primitives** | Build a clean architecture for multimodal disaggregation from day one. Copy proven primitives from VoxServe (FlashInfer, paged KV cache, CUDA graphs, ZMQ) and vLLM (scheduler interface, cache management, executor abstraction). |
| P9 | **Stage as the fundamental unit** | Every computation is a stage with: `(inputs, outputs, pointers)`. Stages compose into graphs via `Sequential`, `Parallel`, and `Loop` primitives. Stages are run via a step function with `(inputs, state, metadata)`. |
| P10 | **Graph-driven scheduling** | The computation graph (not hardcoded phase enums) drives all scheduling. Ready/waiting queues operate on graph stages, enabling arbitrary DAGs. |

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
Inputs → Encoders (ViT, Audio) → Thinker (AR text) ──RELAY──→ Talker (AR speech codec)
                                       ↓                              ↓
                                  DONE_WITH_FWD              Flow DiT (2.5) / Code2Wav (3)
                                       ↓                              ↓
                                 Text output                    Audio stream
```
Thinker streams hidden states + text token IDs to Talker before finishing its own generation. RELAY flag enables this inter-worker producer-consumer pattern. What exactly gets streamed differs: Qwen2.5-Omni sends last-layer hidden states + input embeddings (element-wise sum); Qwen3-Omni sends both layer-0 embeddings and layer-24 hidden states for all tokens (the Talker-side selectively routes them through `text_projection` or `hidden_projection` depending on token type). See Section 10.2 for the full comparison.

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
│  (from       │                     │                                          │
│   VoxServe)  │                     │  ┌─────────────┐  ┌──────────────────┐   │
│              │                     │  │   Worker    │  │   Scheduler      │   │
│  • FastAPI   │                     │  │   Registry  │  │   • req mgmt     │   │
│  • HTTP      │                     │  │   (static + │  │     (steps a-f)  │   │
│  • Streaming │                     │  │    dynamic) │  │   • stage mgmt   │   │
│              │                     │  └─────────────┘  │     (steps g-i)  │   │
└──────────────┘                     │                   └──────────────────┘   │
       ▲                             │  ┌──────────────────────────────────┐    │
       │                             │  │ Dispatching of stages/subgraphs  │    │
       │ stream out                  │  │ • ready/waiting queues           │    │
       │                             │  │ • {req → {stage: worker_id}}     │    │
       │                             │  └──────────────────────────────────┘    │
       │                             └────────────────┬───────────────────┬─────┘
       │                                              │                   │
       │                                    ┌─────────┘           ┌───────┘
       │                                    │                     │
       │                             ┌──────▼──────┐       ┌──────▼──────┐
       │                             │  Worker 0   │       │  Worker 1   │
       │                             │  (GPU 0)    │  IPC  │  (GPU 1)    │
       └──────────────────────────── │  • LLM      │◄─────►│  • Flow     │
                                     │  • Encoder  │       │  • VAE      │
                                     │  • AR Head  │       │  • Talker   │
                                     └─────────────┘       └─────────────┘

                     ┌─────────────────────────────────────┐
                     │        Execution Strategy           │
                     │  (per model, part of model obj)     │
                     │                                     │
                     │  • Full Graph (all possible stages) │
                     │  • f(inp_modalities, out_modalities │
                     │      flags/meta) → Active Graph     │
                     │  • run_stage(stage_id, inputs,      │
                     │      state, flags) → {out: tensor}  │
                     └─────────────────────────────────────┘
```

### 3.1 Component Responsibilities

| Component | Responsibility | What It Does NOT Do |
|-----------|---------------|---------------------|
| **API Server** | HTTP endpoints, streaming responses, ZMQ communication with Conductor | Touch GPU tensors, manage batching |
| **Cluster Manager** | Deployment-time GPU allocation, autoscaling policy, config loading | Runtime request routing (deployment-time only) |
| **Conductor** | Request lifecycle, worker selection, subgraph dispatching (i.e., dispatches contiguous graph sections to each worker), routing of inputs to the graph (e.g., input text, image, embeddings from the prev forward pass), determining computation path (prefill vs. decode vs. image gen, e.g.) | GPU computation, batching, tensor operations |
| **Execution Strategy** | Defines a list of computation graphs for every "execution phase" of the model, where an execution phase is, e.g., prefill, autoregressive decode, image generation (for models that have different inference paths for image generation), map modalities to active phase, provide `run_stage()` or `step()` function. This is currently _not_ a separate class but lives on the `Model` class. | Scheduling, worker selection, communication. |
| **Workers** | GPU computation, internal batching, continuous batching, KV cache management, streaming output, inter-worker communication. Handles input/output queues for its own subgraphs, and sends outputs to other workers when computation finishes. | Request lifecycle (e.g., checking for BOI or EOS tokens), cross-worker scheduling / batching. |

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

### 4.3 Adaptations from VoxServe

- Generalize message format from audio-only to arbitrary modality chunks
- Support multiple output streams per request (e.g., text + audio simultaneously for Omni models)
- ~Add input streaming support for incremental inputs (text chunks, video frames)~ (already implemented in VoxServe)
- Result socket receives from Workers directly (not through Conductor) for streaming data

### 4.4 What to Reuse

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

```yaml
cluster:
  workers:
    - rank: 0
      gpu: "cuda:0"
      components:
        - model: "show-o2"
          submodel: "llm"
          capabilities: ["llm_prefill", "llm_decode", "flow_step"]
        - model: "show-o2"
          submodel: "vit"
          capabilities: ["encode_image"]
    - rank: 1
      gpu: "cuda:1"
      components:
        - model: "show-o2"
          submodel: "vae"
          capabilities: ["vae_encode", "vae_decode"]

  try_to_colocate:
    - ["llm", "flow_head"]  # for Show-o2 interleaved pattern

  autoscaling:
    enabled: false
    min_workers: 2
    max_workers: 8

  gpu_preferences:
    llm: "H100"
    vae: "A100"
```

Capability names are defined by the system in per-model basis. In other words, users can't use custom modules (i.e., something finer than what the system defines) in a computation graph.

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

1. Assigns workers to subgraphs (contiguous computation graph sections) upon receiving a new request, and sends the workers information about what subgraph(s) they are handing for a request, and what workers are handling other subgraphs (for output routing),
2. During a forward pass, receives information from workers about subgraph completion, and also information about tensors that will persist across forward passes (e.g., a newly-generated token will need to be added to the inputs of the next forward pass, and image embeddings may also persist for the next forward pass),
3. At the end of each full model forward pass, updates the current input/output modalities and computation "phase" (e.g., checks for BOI or EOS tokens),
4. At the beginning of each full model forward pass, sends inputs and state information (e.g., which computation phase we are in) to the appropriate graph stages on the appropriate workers.

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

At **initialization** time, the conductor assigns subgraphs to workers.
In the data-parallel case, a subgraph may be assigned to multiple workers, and requests will be routed to one of the data-parallel ranks per subgraph.
If there are multiple "phases" of computation (e.g., prefill, decode, image generation), components of those will have separate subgraphs where appropriate.
Subgraphs for all phases will be assigned to workers at the beginning of a request, though only the subgraphs for one phase will be run by workers during each forward pass.

Specifically, upon instatiation, the conductor calls `model.get_subgraphs(model_config_file)` and creates a mapping of subgraph id to subgraph.
A **subgraph** is a contiguous section of a model computation graph that will be assigned to a worker. `model.get_subgraphs` populates the subgraphs with a list of what workers can execute that subgraph (i.e., have the right sub-models loaded) and what computation phase the subgraph is active for.

Then, it communicates to each worker what subgraphs they will be processing, as well as what graph stages are in other subgraphs (required for output routing).

For each request, the scheduler contains multiple roles that operate at different timescales:

**Request Management** -- Runs once per new request:

| Step | Action | Inputs |
|------|--------|--------|
| a | Get request from queue, extract metadata | `model_name`, initial input/output modalities |
| b | Determine worker plan for request (e.g., random routing for DP, though we should use a more sophisticated system in the future) to get a subgraph to worker mapping for this request | server config (worker->gpu mappings) |
| c | Initialize a `RequestData` object for this request to track the progress of the current request, as well as signals that persist between forward passes |  Worker plan from (b), initial input/output modalities |
| d | Communicate to the relevant workers the subgraph IDs for this request, as well as the subgraph to worker mapping (so that the workers can route their outputs properly to other workers) | outputs from (b) and (c) | 

**Intra-forward Pass Management** -- Runs every conductor full model forward pass (i.e., when all subgraphs have been completed).

| Step | Action | Details |
|------|--------|---------|
| e | Check for "persistent signals" from workers | The workers may push information about tensors that will be used in future forward passes, which will be used as inputs in future forward passes |
| f | Check for "subgraph done" signals from workers | Update what subgraphs have been completed for the current request forward pass. If all subgraphs for the current computation phase have been completed, we have completed a forward pass. |

**At the end of each forward pass and beginning of the next**:

| Step | Action | Details |
|------|--------|---------|
| g | Update request lifecycle | Update lifecycle state (saw `<BOI>`, saw `<EOS>`, etc.), wrangle inputs for next forward pass. If `<EOS>`, end request (send signals to workers and API server). Also set which subgraphs are active for the next model forward pass (used to track when the fwd pass is done) |
| h | `model.get_forward_pass_inputs(...)` | Get the inputs for the current forward pass (e.g., if we are doing image generation, this will include noisy latents). This will be in the form of `GraphPointer` objects, which include tensor name, what graph stage it is routed to, and the current tensor location (IP address, memory address, size)  |
| i | Dispatch inputs to workers | Send the input `GraphPointer`s for the current forward pass to thj appropriate workers. Also include metadata of which phase we are in (e.g. prefill, decode, image_gen), because that is required for routing |

**Note**: sending the current computation phase may be replaced by a more detailed "request state", which will additionally include information about, e.g., what input token indices are text vs. image vs. video.


**Key data structures**:

- `{subgraph_id: subgraph}`:
```python
@dataclass
class Subgraph:
    section: GraphSection
    phases: set[str] # e.g., prefill, decode, image_gen 
    consumes_stream: bool # for thinker-talker
    ranks: list[int] # one-to-one mapping of worker to rank
    subgraph_id: str
```
- `{req_id: request_data}`:
```python
@dataclass
class RequestData:
    current_forward_metadata: CurrentForwardMetadata
    fwd_inputs: list[GraphPointer] # inputs that the conductor has sent to the current fwd pass
    persist_signals: dict[str, TensorPointerInfo] # signals passed back to conductor
    subgraph_to_worker: dict[str, str] # subgraph id to worker id
    new_tokens: list[int] # for this fwd pass, used to check for BOI, EOS, etc.

    # for tracking progress
    all_subgraph_ids: set[str] # across all phases
    current_subgraph_ids: set[str] # for the current fwd pass computation phase
    completed_subgraph_ids: set[str] # for the current full model fwd pass
```
where `CurrentForwardMetadata` is:
```python
@dataclass
class CurrentForwardMetadata:
    input_modalities: list[str]
    output_modalities: list[str]
    phase: str
    is_prefill: bool
```
See the **computation graph model** section for information about `GraphPointer` and `TensorPointerMetadata`.


### 6.4 Subgraph Persistence (Note 7)

The conductor does NOT recompute the execution plan every forward pass. Instead:

1. On server initialization phase: determine stage->worker and worker->gpu mappings from yaml file. This results in a static list of subgraphs that are being handled by each worker.
2. On a new request arriving: compute the execution plan based on input/output modalities, stored in the request state. Only recompute IF input/output modalities change (e.g., `<BOI>` triggers adding flow subgraph). The conductor only sends the inputs to the proper workers for this forward pass, as well as what phase of computation we are in (it does **not** need to send new subgraph information to workers); the execution of the proper subgraphs for each forward pass can be handled via tensor routing between workers.

The worker needs to know only (1) what the computation subgraphs are to compute, (2) what's in the incoming request queue, and (3) where to send the output, which is decided by the conductor as metadata for each request. This is assuming subgraph-to-worker mapping is static (i.e., there never exists LLM-only worker and LLM+flow worker at the same time)

This enables:
- Reduced recomputation overhead
- "Talker stays alive for duration of request" paradigm
- Preference for, e.g., request0's decode always running on GPU1 (avoids KV cache transfer)

---


## 7. Execution Strategy

When defining a model, the user must define a computation graph for each phase of computation, as well as: logic for determining the computation phase at each full model forward pass, the "full model" inputs at each new forward pass, and code for actually executing each graph stage.

**Note**: This is currently in the `Model` class, but it might make more sense to pull this logic out into an `ExecutionStrategy` class, which can be retrieved via `model.get_execution_strategy`.

### 7.1 Computation Graph and Subgraphs
For each phase of computation, the user must define a **computation** graph, which specifies the discrete computation stages, their execution order, and how their outputs are routed.

To make graph definition more intuitive than, e.g., a generic DAG, stages can be organized in `Sequential`, `Parallel`, and `Loop` configurations.

The user must define `self.get_phase_graphs()`, which returns a mapping of computation phase name (e.g., `prefill`, `decode`, `image_gen`) to computation graph.

Once the graphs are defined, the `Model` class automatically parses it, along with the cluster config, to produce a list of **Subgraphs**, or groups of stages that will be assigned to a worker together.
This is produced by `model.get_subgraphs(config_path)`, which calls `self.get_phase_graphs()` and performs the logic to break the graphs from each phase into subgraphs.
See the **computation graph model** section for information.

### 7.2 Input Wrangling and Stage Running

In addition to the computation graph, the user must define the following functions for forward pass execution:

<!-- get_initial_forward_metadata, get_forward_pass_inputs, update_for_next_forward, step in model/base.py-->

<!-- @Irmak TODO EDIT FROM HERE -->

`get_initial_forward_metadata()` returns `CurrentForwardMetadata` object given input-output modalities (such as `image`, `text`, `audio`) with `prefill` phase and `is_prefill` flag set to `True`.

`get_forward_pass_inputs()` returns a list of `GraphPointer`s with `next_stage` and `name` fields. It checks for persisting signals (outputs of the previous fwd pass) to be merged with new inputs. Called by the conductor. These are the external inputs that go from the conductor to the workers at the beginning of the current forward pass; all other signals are handled internally via IPC.

`update_for_next_forward()` is called by the conductor after the full fwd pass. The metadata is updated (primarily checking for input-output modality changes to update the `phase`).

**Note: these functions are subject to change, they are our best guess for what needs to be defined.**

<!-- 
### 7.2 Three Outputs

1. **Full Graph**: The complete computation graph using GraphSection primitives (GraphStage, Sequential, Parallel, Loop). Represents ALL possible stages for this model regardless of input/output modalities.

2. **Active Graph Function**: `f(input_modalities, output_modalities, flags/meta) → active graph`. Selects which stages are active for a given request. Example:
   - Text-only input, text output → Only: text_tok → LLM prefill → LLM decode → STREAM
   - Text input, image output → Full pipeline including flow head and VAE

3. **`run_stage(stage_id, inputs, state, flags) → {output_name: tensor}`**: Called by workers. Takes named input tensors, returns named output tensors. Flags include metadata like flow step number, prefill vs. decode, etc. Every time the active graph changes, a new `stage_id` is issued. -->

### 7.3 Full Graph Example (BAGEL)

**AR Text Generation Phase**:
```python
full_graph = Sequential([
    Parallel([
        GraphStage(
            name="vit_encoder",
            input_ids=["image"],
            outputs=[
                GraphPointer(next_stage="LLM", name="img_emb")
            ]
        ),
        GraphStage(
            name="text_emb",
            input_ids=["text"],
            outputs=[
                GraphPointer(next_stage="LLM", name="text_emb")
            ]
        ),
    ]),
    GraphStage(
        name="LLM_text_gen",  # AR text generation phase
        input_ids=["text_emb", "img_emb"],
        outputs=[
            GraphPointer(next_stage="STREAM_OUT", name="text_tokens"),
        ]
    )
])
```

**Image generation phase**: triggered when conductor detects `<BOI>`. Each flow step runs the full LLM backbone with frozen KV cache.
```python
full_graph = Sequential([
    Parallel([
        GraphStage(
            name="vit_encoder",
            input_ids=["image"],
            outputs=[
                GraphPointer(next_stage="LLM", name="img_emb")
            ]
        ),
        GraphStage(
            name="text_emb",
            input_ids=["text"],
            outputs=[
                GraphPointer(next_stage="LLM", name="text_emb")
            ]
        ),
    ]),
    # Image generation phase -- triggered when conductor detects <BOI>
    # Each flow step runs the full LLM backbone with frozen KV cache
    Loop(
        section=GraphStage(
            name="LLM_flow_step",  # Full LLM forward pass per step (frozen KV)
            input_ids=["text_emb", "img_emb", "latents"],
            outputs=[
                GraphPointer(
                    next_stage="LLM_flow_step",
                    name="latents"
                )  # loop-back
            ]
        ),
        n_iters=24,
        outputs=[
            GraphPointer(
                next_stage="vae_decoder",
                name="latents"
            )
        ]
    ),
    GraphStage(
        name="vae_decoder",
        input_ids=["latents"],
        outputs=[
            GraphPointer(
                next_stage="STREAM_OUT",
                name="image",
                back_to_conductor=True
            )
        ]
    )
])
```


Note: BAGEL uses the same LLM backbone for both text generation and flow steps. During flow steps, the LLM reads frozen KV cache (`update_past_key_values=False`) and processes noised latents projected via `vae2llm`. Velocity is extracted via `llm2vae` (a linear layer), not a separate diffusion head. CFG requires 3x forward passes per step (conditional + text-CFG + image-CFG), handled at the worker level by tripling the batch size. The `text_emb` and `img_emb` inputs to the flow loop are external inputs that persist across iterations (handled by `Loop.external_inputs`).

### 7.4 Image Generation Graph (Show-o2 -- Interleaved)
**AR Text Generation Phase**:
```python
Sequential([
    Parallel([
        GraphStage(
            name="vit_encoder",
            input_ids=["image"],
            outputs=[
                GraphPointer(
                    next_stage="LLM",
                    name="img_emb"
                )
            ]
        ),
        GraphStage(
            name="text_emb",
            input_ids=["text"],
            outputs=[
                GraphPointer(
                    next_stage="LLM",
                    name="text_emb"
                )
            ]
        ),
    ]),
    GraphStage(
        name="LLM_text_gen",  # AR text generation phase
        input_ids=["text_emb", "img_emb"],
        outputs=[
            GraphPointer(next_stage="STREAM_OUT", name="text_tokens"),
        ]
    )
])
```

**Image generation (flow matching)**:
```python
Sequential([
    Parallel([
        GraphStage(
            name="vit_encoder",
            input_ids=["image"],
            outputs=[
                GraphPointer(
                    next_stage="LLM",
                    name="img_emb"
                )
            ]
        ),
        GraphStage(
            name="text_emb",
            input_ids=["text"],
            outputs=[
                GraphPointer(
                    next_stage="LLM",
                    name="text_emb"
                )
            ]
        ),
    ]),
    Loop(
        section=Sequential([
            GraphStage(
                name="LLM",  # Full LLM forward pass at each flow step
                input_ids=["text_emb", "img_emb", "latents"],
                outputs=[
                    GraphPointer(
                        next_stage="diffusion_head",
                        name="hidden_states"
                    )
                ]
            ),
            GraphStage(
                name="diffusion_head",
                input_ids=["hidden_states"],
                outputs=[
                    GraphPointer(
                        next_stage="euler_step",
                        name="velocity"
                    )
                ]
            ),
            GraphStage(
                name="euler_step",
                input_ids=["velocity", "latents"],
                outputs=[
                    GraphPointer(
                        next_stage="LLM",
                        name="latents"
                    )  # loop-back
                ]
            ),
        ]),
        n_iters=50,
        outputs=[
            GraphPointer(
                next_stage="vae_decoder",
                name="latents"
            )
        ]
    ),
    GraphStage(
        name="vae_decoder",
        input_ids=["latents"],
        outputs=[
            GraphPointer(
                next_stage="STREAM_OUT",
                name="image"
            )
        ]
    )
])
```

Note: For Show-o2, the `LLM`, `diffusion_head`, and `euler_step` stages MUST be co-located on the same worker because they execute 50 times in tight sequence. The config YAML's `try_to_colocate` enforces this.

### 7.5 Full Graph Example (Qwen3-Omni -- Thinker-Talker)

```python
full_graph = Sequential([
    Parallel([
        GraphStage(
            name="text_emb",
            input_ids=["text"],
            outputs=[
                GraphPointer(
                    next_stage="thinker",
                    name="text_emb"
                )
            ]
        ),
        GraphStage(
            name="vit_encoder",
            input_ids=["image"],
            outputs=[
                GraphPointer(
                    next_stage="thinker",
                    name="img_emb"
                )
            ]
        ),
        GraphStage(
            name="audio_encoder",
            input_ids=["audio"],
            outputs=[
                GraphPointer(
                    next_stage="thinker",
                    name="audio_emb"
                )
            ]
        ),
    ]),
    Parallel([
        # Thinker branch (AR text generation)
        GraphStage(
            name="thinker",
            input_ids=["text_emb", "img_emb", "audio_emb"],
            outputs=[
                GraphPointer(
                    next_stage="STREAM_OUT",
                    name="text_tokens"
                ),
                GraphPointer(
                    next_stage="talker",
                    name="hidden_states",
                    back_to_conductor=False
                ),
                # Note: in a producer-consumer stream, only the consumer needs to
                # know that this is a streaming operation. The producer can just send
                # outputs via IPC as normal.
            ]
        ),
        # Talker branch (AR speech codec generation)
        # Note: The talker is an AR model that runs autoregressively (many forward passes),
        # but this is managed at the worker level (the Talker loop in Section 10.2),
        # not as a Loop in the graph. The graph represents it as a single stage
        # because its lifecycle is managed via RELAY buffering, not graph readiness.
        Sequential([
            GraphStage(
                name="talker",
                consumes_stream=True,
                input_ids=["hidden_states"],  # from thinker via RELAY
                outputs=[
                    GraphPointer(
                        next_stage="mtp_module",
                        name="codec_tokens"
                    )
                ]
            ),
            GraphStage(
                name="mtp_module",
                input_ids=["codec_tokens"],
                outputs=[
                    GraphPointer(
                        next_stage="code2wav",
                        name="full_codebook"
                    )
                ]
            ),
            GraphStage(
                name="code2wav",
                input_ids=["full_codebook"],
                outputs=[
                    GraphPointer(
                        next_stage="STREAM_OUT",
                        name="audio_chunk"
                    )
                ]
            ),
        ])
    ]),
])
```


### 7.6 Note: GraphPointers in Full Graph Construction

The graph pointer info associated with each tensor triggers the onset of the `next_stage` as well as inter-worker or worker-to-conductor communication (tensor sharing).

```python
@dataclass
class TensorPointerInfo:
    dims: list[int]
    dtype: str
    nbytes: int
    address: int
    source_session_id: str # e.g., f"{HOSTNAME}:{client_engine.get_rpc_port()}"
    source_entity: str # which {worker, api_server} the tensor is on

@dataclass
class GraphPointer:
    next_stage: str
    name: str
    tensor_info: TensorPointerInfo | None = field(default=None)
    # Flags
    back_to_conductor: bool = field(default=False)
    is_new_token: bool = field(default=False)
```

---

## 8. Workers

### 8.1 Design

Workers are long-lived GPU processes. Each worker:
1. Loads model components at startup based on its config
2. Registers capabilities with the Conductor's Worker Registry
3. Receives dispatched stages from the Conductor via ZMQ
4. Executes computation, manages internal batching
5. Sends completion notifications back to the Conductor
6. Streams output directly to the API Server (for STREAM_OUT)
7. Communicates with other workers for inter-worker data transfer (RELAY, KV cache)

### 8.2 Worker Types

Three worker types handle different computation patterns:

#### 8.2.1 vLLM Engine Worker

For transformer-based stages (LLM prefill, LLM decode, encoder forward passes).

**Borrows from vLLM**:
- Continuous batching with dynamic scheduling
- Paged KV cache with block-based management
- Prefix caching (hash-based block deduplication)
- Speculative decoding support
- Executor abstraction (`collective_rpc` for distributed execution)
- Chunked prefill (handled at worker level per Note 13, not conductor level)

**Key insight from vLLM**: The `SchedulerOutput` data structure is computation-agnostic. It describes *what* to schedule (request IDs, token counts, encoder inputs) without knowing *how* computation works. This makes it reusable for arbitrary transformer stages.

**Key insight from VoxServe**: The `CudaGraphWorker` pattern captures CUDA graphs for different batch size and sequence length buckets, providing predictable execution times and reduced kernel launch overhead.

#### 8.2.2 FlashInfer Custom Worker

For attention-heavy stages that need custom kernels but aren't standard transformer decode (e.g., understanding encoders with bidirectional attention, depth transformers).

**Borrows from VoxServe**:
- FlashInfer prefill/decode wrappers
- Paged KV cache allocation logic
- CUDA graph capture pattern

**Uses FlashInfer directly** for:
- Custom attention patterns (causal, bidirectional, sliding window)
- Fused RoPE / RMSNorm kernels
- Variable-length sequence batching

#### 8.2.3 DiT Worker

For diffusion/flow-based stages (rectified flow, ODE solvers, VAE encode/decode).

**Characteristics**:
- No KV cache (stateless per step)
- Fixed-shape computation per step (batch × channels × height × width)
- CUDA-graph-friendly (fixed shapes)
- CFG parallelism (2x or 3x batch for classifier-free guidance)

**Implementation**: Custom worker that handles the flow step loop internally when dispatched a `Loop(flow_step, n_iters=N)` subgraph (e.g., N=24 for BAGEL, N=50 for Show-o2).

### 8.3 Worker Capabilities

Workers declare capabilities as tuples: `(model_name, submodel_name, capability_type)`.

Examples:
```python
worker_0_capabilities = [
    ("show-o2", "LLM", "llm_prefill"),
    ("show-o2", "LLM", "llm_decode"),
    ("show-o2", "diffusion_head", "flow_step"),  # co-located with LLM
    ("show-o2", "vit", "encode_image"),
]
worker_1_capabilities = [
    ("show-o2", "vae", "vae_encode"),
    ("show-o2", "vae", "vae_decode"),
]
```

### 8.4 Internal Batching

Workers handle their own batching. The conductor dispatches work items; the worker accumulates and batches them.

**For AR decode stages** (vLLM Engine Worker):
- Continuous batching: requests join/leave batch dynamically
- Batch selection based on: prefill priority, KV cache availability, SLO urgency
- Chunked prefill: process prompt in chunks to avoid blocking decode requests

**For flow/diffusion stages** (DiT Worker):
- Batch multiple requests' flow steps together
- CFG parallelism: double/triple batch for guidance

**For encoder stages**:
- Simple batching: accumulate images/audio, process in batch
- Stateless: no need for KV cache management

### 8.5 Worker State Management

From the original whiteboard: "Update in worker, synchronization in conductor."

- **Within-worker state** (worker manages): KV cache, noise schedule, step counter, latents, decoder cache
- **Between-worker synchronization** (conductor manages): Which worker has which request's state, when to transfer KV cache, request lifecycle tracking

### 8.6 Streaming Output

Workers stream output directly to the API Server (not through the Conductor):

```
Worker ──ZMQ PUSH──→ API Server result socket
```

This avoids the conductor becoming a bottleneck for high-bandwidth data (audio waveforms, image tensors). The conductor only receives lightweight completion notifications.

### 8.7 Worker Communication
The new `BaseCommunicator` class handles tensor-level worker-worker (`next_stage` is a part of a subgraph that's in a different worker), worker-conductor (tensor has `back_to_conductor` flag set) or worker-api_server (streaming out) communication.

High-level example:
`Worker_0` has latents that are needed on the `Worker_1` subgraph. `Worker_0` sends `address` and `n_bytes` information (via ZMQ). `Worker_1` allocates the required memory for latents, uses `transfer_read_on_cuda` to pull the tensors from `Worker_0` and sends a feedback message for confirmation.

```python
self.engine.transfer_read_on_cuda(
    info.source_session_id,
    dst.data_ptr(),
    info.address,
    info.nbytes,
    stream.cuda_stream,
)
```

---

## 9. Computation Graph Model

### 9.1 Class Hierarchy

```
GraphSection (ABC)
├── GraphStage          # Leaf: a single computation unit
├── Sequential          # Stages execute in order
├── Parallel            # Stages execute concurrently
└── Loop                # Stage/subgraph repeated N times
```

**Note on evolution from whiteboard to code**: The original whiteboard (IMG_0736) proposed `GraphSection` with concrete fields `inputs: list[str]` and `name: str`. The actual implementation evolved to use **abstract methods only** -- no concrete fields on the base class.

**Abstract methods on GraphSection** (all subclasses must implement):
```python
class GraphSection(ABC):
    def get_stage_names(self) -> list[str]: ...       # Names of all stages in this section
    def get_inputs(self) -> SignalToDests: ...         # External/loop-back inputs
    def get_outputs(self) -> SignalToDests: ...        # External/loop-back outputs
    def ingest_inputs(self, stage_to_inputs: DestToInputs): ...  # Mark inputs as received; MUTATES stage_to_inputs
    def split_off_ready(self) -> tuple[list[GraphStage], GraphSection | None]: ...  # Split ready stages from waiting
```

**Critical behavior**: `ingest_inputs()` **mutates** its argument `stage_to_inputs`, removing entries that were consumed. After the call, only un-consumed (external) entries remain. This mutation-based protocol is how external outputs are computed.

### 9.2 GraphStage

The fundamental computation unit (leaf node).

```python
@dataclass
class GraphStage(GraphSection):
    name: str
    input_ids: set[str]           # Named inputs this stage needs (coerced from list in __post_init__)
    outputs: SignalToDestsAndFlags  # {output_name: [GraphPointer(dest, flags)]}
    ready_input_ids: set[str] = field(default_factory=set)  # Populated as predecessors complete

    def is_ready(self) -> bool:
        return self.input_ids.issubset(self.ready_input_ids)
```

**Key methods**:
- `ingest_inputs(stage_to_inputs)`: If this stage's name is in `stage_to_inputs`, ingest any input IDs that this stage needs and hasn't already received. Remove consumed entries from `stage_to_inputs` (mutation).
- `split_off_ready()`: Returns `([self], None)` if ready, or `([], self)` if still waiting.
- `get_inputs()`: Returns `{inp: [self.name] for inp in self.input_ids}` -- maps each input to this stage name.
- `get_outputs()`: Returns `remove_flags(self.outputs)` -- strips GraphPointers to plain names.

Visual:
```
             ┌──────────┐
  inp_A ──→  │  Stage   │ ──→ out_X → [dest1, dest2]
  inp_B ──→  │          │ ──→ out_Y → [dest3]
             └──────────┘
```

### 9.3 Sequential

Stages execute in order.

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

Stages execute concurrently. Fork/join pattern.

```python
@dataclass
class Parallel(GraphSection):
    sections: list[GraphSection]
```

**Key behaviors**:
- `split_off_ready()`: Checks **ALL** sections (contrast with Sequential which only checks first). Each branch independently splits off ready stages. This allows all branches to make progress concurrently.

Visual:
```
       ┌──→ A ──────────┐
  ──→──┤                 ├──→ D
       └──→ B ──→ C ──→─┘
```

### 9.5 Loop

A stage or subgraph repeated N times. Handles loop-back signals (outputs that feed back as inputs to the next iteration) and external inputs.

```python
@dataclass
class Loop(GraphSection):
    section: GraphSection           # Template ("clean" copy, never mutated)
    n_iters: int
    outputs: SignalToDestsAndFlags   # Final iteration outputs (replace loop-backs)
    curr_iter: int = field(default=0)
    external_inputs: SignalToDests = field(default=None)     # Inputs from OUTSIDE the loop (not loop-back)
    curr_iter_section: GraphSection = field(default=None)    # Mutable copy for current iteration
    nxt_iter_section: GraphSection = field(default=None)     # Pre-populated copy for next iteration
```

**`external_inputs`** (critical field, computed in `__post_init__`): Identifies which inputs come from outside the loop vs. from loop-back signals. This distinction is essential for nested loops -- external inputs are re-injected each iteration, while loop-back inputs only come from the previous iteration's outputs.

**Key internal methods**:
- `_get_loop_back_signals()`: Identifies signals that appear in both the section's inputs AND outputs -- these are the loop-back edges.
- `_replace_outputs_for_final_iter(section)`: On the last iteration, removes loop-back pointers from stage outputs and replaces them with `self.outputs` (the final output destinations). Applied recursively through nested sections.
- `_advance_one_iter()`: Promotes `nxt_iter_section` to `curr_iter_section`, creates a fresh `deepcopy` for the new `nxt_iter_section`, increments `curr_iter`, and re-ingests `external_inputs`.
- `split_off_ready()`: If `curr_iter_section` is None (iteration consumed), calls `_advance_one_iter()`. Then splits off ready stages from the current iteration. On last iteration, returns the waiting section directly (no Loop wrapper). Otherwise, returns the Loop itself as the waiting section.
- `ingest_inputs(stage_to_inputs)`: First ingests into `curr_iter_section`. Then separates external inputs from loop-back inputs, and only passes loop-back inputs to `nxt_iter_section` (to prevent external inputs from prematurely populating the next iteration). This logic is required for nested loops.

Visual:
```
  A ──→ B ──→ (loop-back to A, 50×)
              ──→ (final output after 50 iterations)
```

**Loop dispatch granularity**: A `Loop(50)` can be dispatched as a single unit (worker runs all 50 iterations, sends one completion) or broken into `Loop(10) × 5` (enabling round-robin across workers, reducing head-of-line blocking, enabling checkpointing between chunks).

### 9.6 Signal Routing (needs/produces/ptr)

Three equivalent representations of graph edges:

```python
SignalToDests = dict[str, list[str]]              # {signal: [dest_stage_names]}
SignalToDestsAndFlags = dict[str, list[GraphPointer]]  # {signal: [GraphPointer(dest, flags)]}
DestToInputs = dict[str, list[str]]              # {dest_stage: [required_signals]}
```

These can be converted between each other:
- `remove_flags()`: SignalToDestsAndFlags → SignalToDests (strips GraphPointer, keeps dest names)
- `get_stage_to_inputs_mapping()`: SignalToDests → DestToInputs (inverts the mapping)
- `get_signal_to_dest_mapping()`: DestToInputs → SignalToDests (inverts back)

Helper: `update_list_dicts(signals, new_signals)` merges two `dict[str, list]` by extending existing keys and adding new ones. Used by `Parallel` when combining inputs/outputs across branches.

**Mapping direction note**: The whiteboard (IMG_0728) uses `destination → [output_names]` notation (e.g., `DONE_W_FWD: [tokens]`). The code uses the **inverse**: `output_name → [destinations]` (e.g., `"tokens": [GraphPointer("DONE_WITH_FWD")]`). Both representations are equivalent; the code's direction was chosen because it maps naturally to a stage's `outputs` dict where each named output lists where it goes.

### 9.7 GraphPointer

```python
@dataclass
class GraphPointer:
    next_stage: str                        # Name of destination stage (or flag like "STREAM_OUT")
    back_to_conductor: bool = field(default=False)  # Default: direct worker-to-worker
```

The `back_to_conductor` flag determines whether data flows:
- **Worker-to-worker directly** (back_to_conductor=False, the default): For RELAY, inter-worker tensor transfer
- **Worker-to-conductor-to-worker** (back_to_conductor=True): For DONE_WITH_FWD, lifecycle events

### 9.8 RequestQueues

The conductor maintains one `RequestQueues` per in-flight request:

```python
@dataclass
class RequestQueues:
    ready: list[GraphStage]   # Stages with all inputs available
    waiting: GraphSection     # Remaining graph structure

    def process_new_inputs(self, new_inputs: SignalToDestsAndFlags) -> SignalToDests:
        """
        1. Convert new_inputs from SignalToDestsAndFlags to DestToInputs format
        2. Call self.waiting.ingest_inputs(converted) -- this MUTATES the dict,
           removing consumed entries. What remains = external (un-consumed) outputs.
        3. Call self._update_ready_waiting() to split off newly-ready stages
        4. Return external outputs as SignalToDests
        """
        if self.waiting is None:
            return remove_flags(new_inputs)

        new_inputs = get_stage_to_inputs_mapping(remove_flags(new_inputs))
        self.waiting.ingest_inputs(new_inputs)  # mutates new_inputs
        external_outputs = new_inputs           # what's left = external
        self._update_ready_waiting()            # split_off_ready internally
        return get_signal_to_dest_mapping(external_outputs)
```

The key insight: `ingest_inputs` consumes entries from its argument. After the call, only un-consumed entries remain in the dict -- these are external outputs (like STREAM_OUT, DONE_WITH_FWD) that don't correspond to any waiting stage.

A working implementation `computation_graph_scratch_work.py` of this computation graph model exists with a stress test using a Show-o2-style graph with nested loops and parallel branches.

---

## 10. Communication Protocols and Flags

### 10.1 ZMQ Communication (3 Flows)

From whiteboard IMG_0708:

| Flow | Direction | Purpose | Message Content |
|------|-----------|---------|-----------------|
| **Enqueue** | Conductor → Worker | Dispatch stage work | `(req_id, stage_id, input_tensors, metadata)` |
| **Completion** | Worker → Conductor | Notify stage completion | `(req_id, stage_name, output_names, flags)` |
| **Rescheduling** | Conductor → Worker | Remove/reassign request | `(req_id, REMOVE)` |

### 10.2 Three Flags

#### STREAM_OUT (formerly STREAM)

Signals that output should be streamed to the client via the API Server.

**Flow**: Worker → API Server (direct ZMQ, bypasses Conductor)

**Examples**:
- Text tokens streamed as they're generated
- Audio chunks streamed at detokenizer intervals
- Image sent after flow loop completes

#### DONE_WITH_FWD (also NXT_STEP)

Signals that the current forward pass is complete. This triggers the Conductor to:

1. Update lifecycle state (saw `<BOI>`, saw `<EOS>`, etc.)
2. Wrangle inputs for next forward pass (pass along encoded images, add new text tokens)
3. If `<EOS>` not seen, requeue request (triggers steps a-f again)
4. At `<EOS>`, send stop signal to Talker if RELAY is active

#### RELAY

A flag for **producer-consumer streaming between workers**. Distinct from STREAM_OUT (which is client-facing).

**Purpose**: Enable inter-worker data streaming without routing through the Conductor.

**Primary use case**: Qwen-Omni Thinker→Talker streaming. Both Qwen2.5-Omni and Qwen3-Omni use a Thinker-Talker architecture where the Thinker streams data to the Talker BEFORE the Thinker finishes generating text. What exactly gets streamed differs between the two models (see comparison below).

**Flow**:
```
Thinker Worker ──RELAY (hidden states + token IDs)──→ Talker Worker
Thinker Worker ──DONE_WITH_FWD──→ Conductor
Conductor ──(at <EOS>)──→ Talker Worker: STOP signal (ZMQ msg: req_id, STOP)
```

**Conductor RELAY state**:
- Pushes hidden states and token IDs to talker worker: ZMQ msg `(req_id, hidden_states, token_ids, metadata)`
- Maintains `{req_id: talker_id}` mapping to know which talker serves which request

**Talker Worker internals** (from IMG_0725):

```python
# Talker worker state
buffer = {req_id: {"hidden_states": [...], "token_ids": [...]}}  # consumed from RELAY stream
status = {req_id: "WAITING" | "TALKING"}

# Talker loop (runs continuously):
while True:
    # 1. If received STOP from Conductor, remove req_id from buffer/status
    for stop_msg in recv_stop_signals():
        del buffer[stop_msg.req_id]
        del status[stop_msg.req_id]

    # 2. Consume from RELAY streams (add new req_ids, buffer data)
    for relay_msg in recv_relay_data():
        if relay_msg.req_id not in buffer:
            buffer[relay_msg.req_id] = {"hidden_states": [], "token_ids": []}
            status[relay_msg.req_id] = "WAITING"
        buffer[relay_msg.req_id]["hidden_states"].extend(relay_msg.hidden_states)
        buffer[relay_msg.req_id]["token_ids"].extend(relay_msg.token_ids)

    # 3. For all req_ids: check if can talk (enough data in buffer)
    for req_id in buffer:
        if status[req_id] == "WAITING" and enough_in_buffer(req_id):
            status[req_id] = "TALKING"

    # 4. Run talker forward for current batch (all TALKING requests)
    batch = [req_id for req_id in status if status[req_id] == "TALKING"]
    talker_outputs = run_talker_batch(batch)

    # 5. Stream audio output to API Server (STREAM_OUT)
    for req_id, audio_chunk in talker_outputs:
        stream_to_api_server(req_id, audio_chunk)
```

**Why RELAY exists**: The Thinker streams hidden states to the Talker BEFORE `DONE_WITH_FWD` fires — the Thinker is still generating text while the Talker has already begun speech synthesis. Going through the Conductor would add latency. Direct worker-to-worker streaming is essential.

**Qwen2.5-Omni vs. Qwen3-Omni: comprehensive architectural comparison**

These two models share the Thinker-Talker pattern but differ substantially in architecture, what flows via RELAY, and how speech is generated. Verified from:
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

**Thinker→Talker data flow (what travels via RELAY):**

| | **Qwen2.5-Omni** | **Qwen3-Omni** |
|---|---|---|
| **Which hidden states?** | **Last-layer** output + input embeddings (element-wise sum): `final_hidden[i] + input_embed[i]`, 3584-dim per token. Multimodal positions (audio/image/video) are zeroed out in the input embeddings before summing. | **Both** layer-0 embeddings **and** layer-24 hidden states are transferred for **all** tokens (full-sequence tensors). `accept_hidden_layer=24` in the released model config (note: the HuggingFace Transformers default is 18, but the actual model overrides to 24). The **selective routing** happens on the **Talker side**: text tokens → `text_projection(layer_0_embed)`, multimodal tokens (audio/image/video) → `hidden_projection(layer_24_hidden)`. Assistant-turn and streaming decode tokens always use `text_projection(layer_0_embed)`. |
| **Text token IDs** | Original prompt token IDs + generated token IDs. The Talker uses these to **recompute TMRoPE position IDs internally** via its own `get_rope_index()`. TMRoPE IDs are NOT directly transferred. | Full sequence token IDs (prompt + generated). Used for ChatML segment parsing and positional encoding. |
| **Text/multimodal decoupling** | Talker consumes Thinker's hidden states + text token embeddings **together** (coupled — same representation for all tokens). | Decoupled: text tokens use shallow layer-0 embeddings (via `text_projection`), multimodal tokens use deep layer-24 hidden states (via `hidden_projection`). The Talker builds a `multimodal_mask` from token IDs to route each token. This decoupling enables separate system prompts for Thinker and Talker, and allows RAG/safety filters to intervene on text before it reaches the Talker. |
| **Projection** | Single `nn.Linear(3584→896)` (`thinker_to_talker_proj`). The 3584 is the shared `embedding_size` (both codec embeddings and thinker hidden states live in this space); the 896 is the talker's internal transformer `hidden_size`. | Two separate MLPs: `text_projection` and `hidden_projection`, each `Linear(2048→2048)→SiLU→Linear(2048→1024)`. |
| **Talker input formula** | `talker_input = thinker_to_talker_proj(codec_embed(codec_token) + thinker_hidden)` — element-wise add then project. | User-turn text: `text_projection(layer_0_embed)`. User-turn multimodal: `hidden_projection(layer_24_hidden)`. Assistant-turn: always `text_projection(layer_0_embed)`. Streaming decode: always `text_projection(layer_0_embed)`. Codec embeddings added separately. |
| **Speech decoder** | AR codec → Flow-Matching DiT (10 ODE steps) + BigVGAN → waveform | AR codec (codebook 0) → MTP module (codebooks 1–15) → Code2Wav ConvNet → waveform |
| **Data volume per token** | Dense: ~3584 × 2 bytes (fp16) ≈ **7 KB/token** | Dense: ~2048 × 2 bytes (fp16) × 2 streams (layer-0 + layer-24) ≈ **8 KB/token** total |

**Key insight**: The RELAY stream carries **dense hidden-state vectors**, not lightweight position IDs. For Qwen3-Omni, both layer-0 and layer-24 tensors are sent (so ~2 × 2048 × 2 bytes ≈ **8 KB/token** total). For a 200-token response, that is ~1.4 MB of hidden states for Qwen2.5-Omni, ~1.6 MB for Qwen3-Omni (two streams). This has bandwidth implications for inter-GPU transfer.

**Qwen3-Omni RELAY consideration**: Because Qwen3-Omni extracts hidden states from an intermediate layer (layer 24 out of 48) rather than the final layer, the Thinker's text generation (which uses all 48 layers) is architecturally decoupled from the Talker's input. This means RELAY could theoretically be replaced by DONE_WITH_FWD + Conductor forwarding. However, the whiteboard notes present this as a trade-off rather than a settled decision. RELAY avoids the conductor communication hit but adds complexity. For Qwen2.5-Omni, the last-layer hidden states are tightly coupled to the Thinker's text generation, making RELAY more clearly necessary.

### 10.3 ptr Mechanism

From whiteboard IMG_0728, the `ptr` (pointer) concept maps outputs to destinations:

```python
ptr = {
    "DONE_WITH_FWD": ["tokens"],           # flag
    "RELAY": ["thinker_hidden_states"],     # flag (inter-worker stream)
    "flow_head": ["latents"],              # subgraph destination
    "STREAM_OUT": ["text_tokens"],          # client stream
}
```

In the final dataclass design, this is encoded directly in `GraphStage.outputs`:
```python
outputs = {
    "tokens": [GraphPointer("DONE_WITH_FWD", back_to_conductor=True)],
    "thinker_hidden_states": [GraphPointer("talker", back_to_conductor=False)],  # RELAY
    "latents": [GraphPointer("flow_head")],
    "text_tokens": [GraphPointer("STREAM_OUT")],
}
```

---

## 11. Inter-Worker Communication

### 11.1 Design Philosophy

**Design evolution note**: Note 14 of the design discussions stated: "Previously, the conductor handled what graph stages are ready to run and which are waiting for inputs. Now that everything passes between workers via inter-worker communication, this no longer fits in the conductor. The ready/waiting logic moves to the worker level."

**Current resolution**: The final design retains ready/waiting queue tracking at the **Conductor** level (via `RequestQueues`, Sections 6.2.3 and 9.8) for macro-level graph progression, while Note 14's insight is incorporated as a two-level split:

- The **Conductor** handles macro-level scheduling: which worker runs which stage, graph-level readiness (are all inputs for a stage available?), request lifecycle tracking, and worker assignment.
- The **Workers** handle micro-level readiness: buffering partial inputs (e.g., the Talker buffering RELAY data until enough has arrived to start talking), managing continuous batching (when to actually execute a dispatch), and inter-worker data reception.

This means the Conductor tracks "is this graph stage logically ready?" while the Worker tracks "do I have enough data in my buffers to actually run this computation?"

### 11.2 Mooncake-Inspired Transfer Engine

From the Mooncake research, the inter-worker communication layer should:

1. **Use a Mooncake-style Transfer Engine** with topology-aware, multi-path data transfer
2. **Support multiple protocols**: RDMA (for production), TCP (fallback), NVLink (intra-node), shared memory (same-machine)
3. **Stream layer-by-layer / chunk-by-chunk**: Do not wait for full computation to complete before beginning transfer
4. **Use pull-based transfer**: The receiving worker initiates data pull to handle burstiness naturally

### 11.3 KV Cache Transfer

From Mooncake's architecture:
- KV cache blocks are **paged and hashed** using chain hashes (each block hash includes all preceding hashes)
- This enables **prefix-level deduplication** across the entire cluster
- Transfers are **overlapped with computation**: as each LLM layer finishes, its KV cache is asynchronously streamed

**For our system**: Take the IPC logic from Mooncake (Note 12). Start with ZMQ for initial implementation; migrate to RDMA if profiling shows it's the bottleneck.

### 11.4 vLLM KV Connector Interface

From vLLM's codebase, the `KVConnector` abstraction provides a clean boundary:

```python
class KVConnector:
    def pre_forward(self, scheduler_output) -> None:
        """Load required KV cache before computation."""
    def post_forward(self, scheduler_output, wait_for_save=True) -> KVConnectorOutput:
        """Save/stream KV cache after computation."""
```

This pattern generalizes to any inter-worker data transfer:
```python
class StageConnector:
    def load_inputs(self, stage_id, input_spec) -> None:
        """Asynchronously load inputs for a stage."""
    def save_outputs(self, stage_id, output_spec) -> None:
        """Asynchronously stream outputs to downstream stages."""
    def register_buffers(self, buffer_spec) -> None:
        """Register memory regions for zero-copy transfers."""
```

### 11.5 Talker Worker Buffer Management

For the Thinker-Talker pattern (Qwen-Omni), the Talker worker maintains:

```python
talker_state = {
    "buffer": {req_id: {"hidden_states": [...], "token_ids": [...]}},  # consumed from RELAY stream
    "status": {req_id: "WAITING" | "TALKING"},
}
```

**Talker loop**:
1. If received STOP from Conductor, remove req_id
2. Consume from RELAY streams (add new req_ids, buffer tensors)
3. For all req_ids: check if enough data to talk (status = TALKING or sufficient buffer)
4. Run Talker forward for current batch
5. Stream audio output to API Server

---

## 12. KV Cache Management

### 12.1 Architecture

KV cache management spans three levels:

| Level | Component | Responsibility |
|-------|-----------|----------------|
| **Global** | Conductor (Worker Registry) | Track which worker holds which request's KV cache; decide when to transfer |
| **Local** | Worker (KV Cache Manager) | Paged allocation, prefix caching, block management |
| **Transfer** | Transfer Engine (Mooncake) | Physical data movement between workers |

### 12.2 Paged KV Cache

Borrowed from vLLM:
- KV cache is divided into fixed-size **blocks** (e.g., 16 tokens per block)
- Each request has a **physical block table** mapping logical positions to physical blocks
- **Prefix caching**: Blocks with matching token sequences are shared via hash-based deduplication
- **Dynamic allocation**: Blocks are allocated on demand as sequences grow

Borrowed from VoxServe:
- Page allocation/deallocation logic
- KV page tracking per request (`kv_pages`, `kv_token_len`, `kv_last_page_len`)
- Integration with FlashInfer's paged attention wrappers

### 12.3 Cross-Worker KV Transfer

When a request moves between workers (e.g., prefill on Worker 0, decode on Worker 1):

1. Conductor decides to transfer (based on load balancing or co-location requirements)
2. Conductor notifies source worker: "stream KV cache for req_id to Worker 1"
3. Source worker uses Transfer Engine to send KV cache layer-by-layer
4. Destination worker receives blocks, maps them into its local KV cache manager
5. Destination worker notifies Conductor: "ready to serve req_id"

### 12.4 Tiered Storage (Future)

Following Mooncake's pattern, support tiered KV cache storage:
- **GPU VRAM**: Hot cache (active requests)
- **CPU DRAM**: Warm cache (recently accessed, prefix cache)
- **SSD**: Cold cache (evicted but recoverable)

### 12.5 Chunked Prefill

From Note 13: "Chunked prefill is NOT on the conductor level. Send everything to the worker and have the worker handle chunking."

This supports:
- Running prefill on an all-text chunk while a future image input is still being encoded
- Workers managing their own prefill scheduling without conductor involvement
- Inter-worker communication delivering encoded images to the worker as they become available

---

## 13. Concrete Request Flows

### 13.1 Text-Only VLM (e.g., Qwen3-VL)

```
1. User sends: "Describe this image" + image.jpg
2. API Server → Conductor: {text: "Describe...", image: image.jpg, model: "qwen3-vl"}
3. Conductor:
   a. strategy = qwen3_vl.get_execution_strategy()
   b. active_graph = strategy.get_active_graph(inp=[TEXT, IMAGE], out=[TEXT])
   c. Assign: vit_encoder → Worker 0, LLM → Worker 0 (co-located)
   d. Ready: vit_encoder (has image), text_emb (has text)
4. Dispatch vit_encoder and text_emb to Worker 0 (parallel)
5. Worker 0 completes both → hidden states ready → LLM becomes ready
6. Dispatch LLM to Worker 0
7. Worker 0 runs LLM decode loop (continuous batching, AR token generation)
   - Streams text tokens → API Server (STREAM_OUT)
   - At <EOS>: sends DONE_WITH_FWD → Conductor
8. Conductor finishes request
```

**Special: DeepStack multi-level feature injection** for Qwen3-VL requires ViT features from 3 intermediate layers to be injected at LLM layers 1, 2, 3. This is handled internally by the worker since ViT and LLM are co-located.

### 13.2 Image Generation -- BAGEL Pattern (Frozen-KV Flow)

```
1. User sends: "Generate a sunset over mountains"
2. API Server → Conductor: {text: "Generate...", model: "bagel"}
3. Conductor:
   a. active_graph = bagel.get_active_graph(inp=[TEXT], out=[IMAGE])
   b. Assign: text_emb + LLM + flow_loop → Worker 0, vae → Worker 1
      (LLM must be co-located with flow loop since each flow step runs the LLM backbone)
   c. Ready: text_emb (has text input)
4. Dispatch text_emb → Worker 0
5. Worker 0: text_emb completes → LLM prefill + AR text decode
   - Streams text tokens → API Server (STREAM_OUT)
   - At <BOI>: DONE_WITH_FWD → Conductor
6. Conductor: flow loop phase begins. KV cache is now frozen on Worker 0.
7. Worker 0: runs 24 flow steps internally:
   For each step:
     a. Project noised latents via vae2llm into LLM embedding space
     b. Run full LLM forward pass (reading frozen KV cache, update_past_key_values=False)
     c. Extract velocity via llm2vae linear layer
     d. Euler step to update latents
   CFG: 3x forward passes per step (conditional + text-CFG + image-CFG)
   - Completion → Conductor: flow loop done, latents ready
8. Transfer latents from Worker 0 to Worker 1
9. vae_decoder becomes ready → Worker 1 runs VAE decode
10. Worker 1 streams image → API Server (STREAM_OUT)
11. Conductor finishes request
```

**Key**: The LLM is co-located with the flow loop because each of the 24 flow steps requires a full LLM forward pass. However, the KV cache is frozen after text generation, so it is reused (read-only) across all flow steps. The 3x CFG multiplier means each flow step actually runs 3 LLM forward passes (72 total). Only the final latents are transferred to Worker 1 for VAE decoding.

### 13.3 Image Generation -- Show-o2 Pattern (Interleaved Flow)

```
1. User sends: "Generate a sunset over mountains"
2. API Server → Conductor: {text: "Generate...", model: "show-o2"}
3. Conductor:
   a. active_graph = showo2.get_active_graph(inp=[TEXT], out=[IMAGE])
   b. Assign: ALL stages → Worker 0 (forced co-location: LLM + diffusion_head + euler_step)
   c. vae_decoder → Worker 1
   d. Ready: text_emb (has text input)
4. Dispatch text_emb → Worker 0
5. Worker 0 starts Loop(50):
   For each of 50 iterations:
     a. LLM full forward pass (text_emb + img_emb + current latents)
     b. Diffusion head processes LLM hidden states at image positions
     c. Euler step updates latents
   End loop
   - Loop completion → Conductor: latents ready
6. Conductor: vae_decoder becomes ready
7. Dispatch vae_decoder → Worker 1
8. Worker 1: VAE decode → stream image → API Server
9. Conductor finishes request
```

**Key**: 50 LLM forward passes (no KV cache reuse across flow steps). CFG doubles all compute. This is why Show-o2 is much more expensive than BAGEL for image generation.

### 13.4 Speech-to-Speech -- Qwen3-Omni (Thinker-Talker)

```
1. User sends: audio recording + image
2. API Server → Conductor: {audio: audio.wav, image: img.jpg, model: "qwen3-omni"}
3. Conductor:
   a. active_graph includes: encoders, thinker (AR text), talker (AR speech)
   b. Assign: vit + audio_encoder + thinker → Worker 0 (30B MoE)
              talker + mtp + code2wav → Worker 1 (3B MoE)
   c. Ready: vit_encoder (has image), audio_encoder (has audio)
4. Dispatch encoders → Worker 0 (parallel)
5. Worker 0 completes encoders → thinker becomes ready
6. Dispatch thinker → Worker 0
7. Worker 0 runs thinker decode loop:
   - Streams text tokens → API Server (STREAM_OUT)
   - Streams layer-0 embeddings + layer-24 hidden states (accept_hidden_layer) + token IDs → Worker 1 (RELAY)
   - At <EOS>: sends DONE_WITH_FWD → Conductor
8. Worker 1 (talker loop, running concurrently):
   - Buffers incoming hidden states + token IDs from RELAY
   - When sufficient data: runs talker AR decode
   - Talker → MTP module (residual codebooks) → Code2Wav → audio chunks
   - Streams audio → API Server (STREAM_OUT)
   - On receiving STOP from Conductor: finishes current buffer, winds down
9. Conductor finishes request when both thinker and talker complete
```

**Key features**:
- Thinker and Talker run CONCURRENTLY on different GPUs
- RELAY enables hidden state streaming before thinker finishes
- Layer-0 embeddings + layer-24 hidden states + text token IDs travel via RELAY (see Section 10.2 for Qwen2.5-Omni vs. Qwen3-Omni differences)
- Talker maintains per-request buffer with WAITING/TALKING status

### 13.5 SpeechLM -- VoxServe Pattern (Orpheus, CosyVoice, etc.)

```
1. User sends: "Hello world" (text-to-speech)
2. API Server → Conductor: {text: "Hello world", model: "orpheus"}
3. Conductor:
   a. active_graph = orpheus.get_active_graph(inp=[TEXT], out=[AUDIO])
   b. Assign: preprocess + LLM + detokenizer → Worker 0
4. Worker 0 runs VoxServe-style loop:
   a. Preprocess: tokenize text, allocate KV cache, init decoder cache
   b. LLM prefill
   c. LLM decode loop (continuous batching):
      - At model-specific intervals (e.g., 28 tokens for Orpheus, 10 for CSM, 25 for GLM): run detokenizer
      - Detokenizer produces audio chunk → stream to API Server (STREAM_OUT)
      - At EOS: DONE_WITH_FWD → Conductor
5. Conductor finishes request
```

**Key**: This is the existing VoxServe pattern, preserved as-is within the new architecture. The model's `run_stage()` internally handles the prefill→decode→detokenize pipeline.

---

## 14. Technology Decisions

### 14.1 What to Borrow from VoxServe

| Component | How |
|-----------|-----|
| API Server (`APIServer` class in launch.py) | Adapt -- generalize message format, remove VoxServe-specific endpoints |
| FlashInfer attention wrappers (`flashinfer_utils.py`) | Copy directly (`FlashInferPrefillWrapper` + `FlashInferDecodeWrapper`) |
| Paged KV cache allocation (in `CudaGraphWorker`) | Copy directly (tensor allocation pattern) |
| CUDA graph capture pattern (`_initialize_decode_cuda_graphs`) | Copy directly (capture/replay logic) |
| ZMQ IPC socket setup (launch.py + scheduler/base.py) | Copy directly (context init, socket binding, HWM config) |
| Continuous batching pattern (scheduler/base.py `_step` + `_select_*`) | Adapt for multi-stage |
| `DecoderCache` generic state container (tokenizer/base.py) | Copy directly (slicing, copy_from, cat, device movement) |
| Streaming audio/image output (`async_stream_chunks`) | Generalize to multi-modality |

### 14.2 What to Borrow from vLLM

| Component | How |
|-----------|-----|
| Scheduler interface (`SchedulerInterface`, `SchedulerOutput`) | Adapt -- computation-agnostic design works for arbitrary stages |
| Block-based KV cache management (`KVCacheManager`) | Adapt -- replace "KV cache" with "intermediate activation cache" |
| Executor abstraction (`collective_rpc`, pluggable backends) | Reference -- good pattern for distributed execution |
| Request state management | Reference -- extend for computation DAG tracking |
| Prefix caching (hash-based block dedup) | Borrow directly for KV cache sharing |

### 14.3 What to Borrow from vLLM-Omni

| Component | How |
|-----------|-----|
| Stage configuration YAML format | Adapt directly |
| `OmniRequest` with `prompt_embeds` field | Reference -- extend for arbitrary inter-stage tensors |
| `OmniConnectors` IPC (shared memory, serialization) | Borrow directly for inter-worker data transfer |
| `PromptEmbedsPayload` tensor serialization | Borrow directly |
| Diffusion engine/scheduler/worker separation | Reference pattern for DiT Worker |

### 14.4 What to Borrow from Mooncake

| Component | How |
|-----------|-----|
| KV cache IPC logic (Note 12) | Borrow -- transfer engine for inter-worker communication |
| Layer-by-layer streaming during computation | Borrow pattern for RELAY implementation |
| Content-addressable block hashing for dedup | Borrow for prefix caching across workers |
| Topology-aware multi-path transfer | Future -- start with ZMQ/TCP, add RDMA later |
| Prediction-based early rejection | Borrow pattern for SLO-aware scheduling |

### 14.5 What to Write Custom

| Component | Rationale |
|-----------|-----------|
| **Conductor** (all) | Unique to our architecture: graph-based scheduling, ready/waiting queues, subgraph management |
| **Execution Strategy** (all) | Model-specific computation graphs, `run_stage()` per model family |
| **Worker Registry** | Capability-based routing unique to our design |
| **Computation Graph Model** (GraphSection hierarchy) | Working implementation exists in `computation_graph_scratch_work.py` |
| **RequestQueues** with graph-aware ready/waiting | Working implementation exists |
| **RELAY flag handling** | Thinker-Talker streaming pattern specific to our system |
| **Subgraph persistence** logic | Re-plan only when modalities change |

### 14.6 Insert Later (Not Initial Implementation)

- **FlashInfer** optimized attention kernels (beyond what VoxServe provides)
- **CUDA graphs** for all worker types (start with DiT Worker, extend to vLLM Engine Worker)
- **RDMA** inter-worker communication (start with ZMQ/TCP)
- **RadixAttention** prefix sharing (start with simple hash-based prefix matching)
- **Autoscaling** (start with static configuration)

---

## 15. Resolved Design Tensions

### 15.1 Tension 1: Per-Step Dispatch vs. Loop-as-Unit

**Apparent problem**: The conductor loop (step h) shows per-step dispatch (worker completes one step, reports back, conductor dispatches next). But Loop(50) dispatches 50 steps as a unit.

**Resolution**: These are two ends of a spectrum controlled by Loop dispatch granularity.
- `Loop(50)`: Worker runs all 50 steps, sends one completion. Best for tightly-coupled operations (Show-o2 interleaved flow+LLM).
- `Loop(1)`: One completion per step. Best when the conductor needs control between steps.
- `Loop(10) × 5`: Middle ground. Enables round-robin across workers, reduces HoL blocking, enables checkpointing between chunks.

The conductor loop step h just says "check for messages" -- it doesn't prescribe frequency.

### 15.2 Tension 2: Who Constructs needs/produces/ptr?

**Resolution**: The **Execution Strategy** (on the model) defines the **template** (all possible routes). The **Conductor** instantiates it per-request by enabling/disabling routes and assigning workers. Steps a-f are the instantiation process.

### 15.3 Tension 3: Request Management vs. Stage Management Timing

**Resolution**: Request management (steps a-f) runs ONE TIME per new request or per requeue after forward pass completion. Stage management (steps g-i) runs EVERY conductor loop iteration. They operate at different timescales. No conflict.

### 15.4 Tension 4: Two Diagrams of Step 2 (Image Generation)

**Resolution**: They are TWO DIFFERENT model architectures:
- **Diagram 1 (BAGEL pattern)**: LLM decode runs once, then 24 flow steps each re-entering the LLM backbone with frozen KV
- **Diagram 2 (Show-o2 pattern)**: LLM_dec inside the loop, 50 × (LLM + diffusion_head + euler_step) with NO KV reuse across steps

Not an expanded view of the same thing.

### 15.5 Old Design Problem: Pool-Worker Ownership

**Resolution**: Pools are eliminated entirely. Workers declare capabilities. The Worker Registry indexes capabilities. The Conductor routes by capability + affinity. No pool "owns" a worker.

### 15.6 Old Design Problem: execution_plan() on GenerationStrategy

**Resolution**: The execution plan (computation graph) belongs on the **model**, not the strategy. The model knows its own architecture. The model provides `get_execution_strategy()` which returns the full graph, the active graph function, and `run_stage()`. There is no separate `GenerationStrategy` class with `initialize()/step()`.

### 15.7 Old Design Problem: Multiple Workers on Same GPU

**Resolution**: One worker process per GPU (Principle P5). Co-location = one UnifiedWorker process with multiple model components loaded, not separate processes competing for VRAM.

### 15.8 Old Design Problem: Interleaved vs. Independent Flow Heads

**Resolution**: The computation graph expresses this directly.
- **BAGEL**: Each flow step runs the full LLM backbone with frozen KV cache. The LLM and flow loop MUST be co-located (flow is not a separate model -- it reuses the same LLM). Only the final latents → VAE can be disaggregated.
- **Show-o2**: LLM + diffusion_head + euler_step are all inside a `Loop` in `Sequential`. Config YAML `try_to_colocate` forces them onto the same worker.

| Architecture | Flow Separate from LLM? | Cross-GPU Transfers if Separated | Config |
|---|---|---|---|
| BAGEL | No (flow uses LLM backbone, frozen KV) | ~24 query tensors + ~24 hidden states if separated | Co-locate LLM + flow; only VAE separate |
| Show-o2 | No (interleaved, LLM invoked per step) | 2 × 50 = 100 if separated | MUST co-locate |
| JanusFlow | No (interleaved) | 2 × 30 if separated | MUST co-locate |
| Janus Pro | N/A (pure AR) | 0 | Standard AR |
| Qwen2.5-Omni | Yes (separate Thinker/Talker models) | Dense hidden states (~3584-dim/token) + token IDs, streamed via RELAY | Separate workers OK |
| Qwen3-Omni | Yes (separate Thinker/Talker models) | Both layer-0 embeddings + layer-24 hidden states (~2048-dim/token each, both sent for all tokens) + token IDs, streamed via RELAY. Talker-side routes text→`text_projection`, multimodal→`hidden_projection`. | Separate workers OK |

---

## 16. Open Questions

### 16.1 Designed but Not Yet Implemented

1. **Worker selection policy**: How the conductor chooses among capable workers (load balancing, KV affinity, SLO-awareness). Currently unspecified beyond "use Worker Registry dynamic properties."

2. **Batching at the conductor level**: Whether the conductor should batch-select which requests' stages to dispatch together, or leave all batching to workers.

3. **Model hierarchy / interface**: The concrete Python class hierarchy for models. The Execution Strategy is defined conceptually but the exact class interface (`BaseMultimodalModel.get_execution_strategy()`) needs implementation.

4. **Streaming across disaggregated stages**: How streaming chunks (audio, partial images) work when stages are on different GPUs. Currently: workers stream directly to API Server.

5. **SLO prediction / early rejection**: The Mooncake-style prediction-based rejection is referenced but not specified for our system.

6. **Error handling / fault tolerance**: What happens when a worker crashes mid-generation. Recovery, request reassignment, state checkpointing.

7. **Worker health monitoring**: Heartbeat mechanism, failure detection.

8. **Auto-scaling policies**: When/how to add/remove workers based on load.

9. **Partial output routing**: The current routing maps `{enabled_stage: [outputs feeding to the stage]}`, routing whole named outputs. But some stages produce a single tensor where different *slices* need to go to different destinations (e.g., LLM hidden states at image positions → flow_head, hidden states at text positions → text decoder). Either stages must produce separately-named outputs for each slice, or the routing must support index-based sub-selection.

10. **Loop breakup policy**: When the conductor breaks `Loop(50)` into `Loop(10) × 5` for HoL blocking avoidance, round-robin scheduling, or checkpointing -- is this static (from config YAML) or dynamic (based on queue depth, worker load)? The motivations are listed but the policy is undefined.

### 16.2 Explicitly Deferred

1. **Tensor parallelism / pipeline parallelism**: LLM can be decomposed as LLM_part1, LLM_part2, etc. (Note 1). Same as having different stages in series. Deferred to post-v1.

2. **Full RadixAttention tree**: Start with simple prefix matching; full tree-based sharing later.

3. **RDMA inter-worker communication**: Start with ZMQ/TCP; add Mooncake Transfer Engine when profiling shows transfer is the bottleneck.

4. **CUDA MPS / MIG for GPU sharing**: Currently one process per GPU. Multi-tenant GPU sharing deferred.

### 16.3 Requires Further Research

1. **Scheduling estimated runtimes** (Note 2): Need per-stage runtime estimates, dynamically updated based on hardware load and previous task durations.

2. **Nested loop scheduling**: When a `Loop(10)` chunk completes, should the conductor assign the next chunk to the same or different worker? Depends on state locality vs. load balancing.

3. **World model integration**: World models (DreamerV3, DIAMOND, V-JEPA 2, Cosmos) have fundamentally different characteristics from SpeechLMs/VLMs:
   - **State structure**: RSSM state (deterministic GRU hidden `h_t` + stochastic categorical `z_t`, e.g., DreamerV3 uses 32×32 categorical) vs. KV cache. This is a tiny rolling state (~few KB) rather than a growing sequence cache (~MB-GB).
   - **Loop structure**: Environment step loop (observe → encode → RSSM → predict → act) at real-time control frequencies (e.g., 20Hz for robotics), not token-by-token generation.
   - **Session model**: Episode-based (start, steps, terminal) not session-based. Requests are long-running environment interactions, not one-shot generations.
   - **Actor-Critic on imagined trajectories**: World models interleave inference with imagination rollouts (predict future states without env interaction), a compute pattern unlike anything in our current graph model.
   - **Candidate approach**: A specialized `WorldModelWorker` wrapping RSSM state update + prediction heads, with the Conductor managing the env-step loop. Alternatively, express the RSSM as a `Loop(GraphStage(...))` where the stage's `run_stage()` handles the encode→dynamics→predict cycle.

---

## Appendix A: Pipeline Parallelism Note

From Note 1: "LLM can be decomposed as LLM_part1, LLM_part2, etc. Same as having different components in series."

This means pipeline parallelism is naturally expressed in our computation graph:
```python
Sequential([
    GraphStage(name="llm_layers_0_15", ...),
    GraphStage(name="llm_layers_16_31", ...),
])
```
Each sub-stage can be assigned to a different worker. The conductor handles the handoff. This is a future extension, not initial implementation.

## Appendix B: Scheduling Detail (Note 3)

Each stage needs to know:
1. What inputs from previous stages it needs
2. What future stages its outputs enable

This must work across LLM forward passes. When any worker completes:
- Conductor sees from (2) what future stages these outputs contribute to
- For each future stage, conductor checks from (1) whether it's ready
- Must also encode token indices minutia for interleaved image and text token ordering

**Token indices**: All data carries token indices throughout the pipeline. Every piece of data knows its position in the original sequence, critical for multimodal attention masking.

## Appendix C: Stage Completion Signaling (Note 11)

When a stage within a subgraph completes, the worker sends a "stage completed" message to the conductor. This doesn't necessarily trigger any action but is important for:
- SLO-aware scheduling (estimating remaining time)
- Progress monitoring
- Debugging

The message can include a flag saying "this also means subgraph completed" to distinguish intra-subgraph progress from subgraph completion.

## Appendix D: Design Notes from Atindra & Naomi

These notes were captured during whiteboard sessions and design discussions. They are referenced throughout this document as "Note N".

---

**Note 1:** pipeline parallelism note: the LLM can be decomposed as LLM_part1, LLM_part2, etc... This should be the same as having different components in series

**Note 2:** on scheduling: need to store estimated runtimes of stages, and update dynamically based on current hardware load / how long previous tasks took / etc.

**Note 3:** on token routing / scheduling: each stage needs to know (1) what inputs from previous stages it needs, and (2) what future stages its outputs enable. This may also need to work across LLM forward passes. When any worker completes a task, it will send the outputs back to the conductor. The conductor will see from (2) what future stages these outputs contribute to. For each of these future stages, it will know from (1) whether the stage is ready to go or whether it needs to wait for more work to finish (e.g., the text encoding has finished, but image encoding is still working, and the LLM decode stage needs both). We will have to also encode, somewhere in here, the minutia of token indices and such so that we know what order to place, e.g., interleaved image and text tokens.

**Note 4:** completion signaling (6) and batching across requests (1): can take these mechanisms from voxserve

**Note 5:** RELAY: Is either just another flag like STREAM_OUT (aka/formerly STREAM in the diagrams above) or another process (TBD). It signifies producer-consumer streaming of tokens (between stages like the Thinker stage and the Talker Stage). RELAY makes sense because:
- We need inter-worker communication (we would not need it if we relied on the conductor to pass the data around to the talker because the DONE_WITH_FWD flag already wakes the conductor up and the conductor already has the computation graph).
- Qwen2.5-Omni which sends hidden states from thinker to talker (so before DONE_WITH_FWD flag is on). The Qwen3-Omni architecture works only with DONE_WITH_FWD flag without RELAY if we are willing to take the conductor communication hit.
  - **[Correction]**: The original note said "layer 18." Verified findings: Qwen2.5-Omni sends **last-layer** hidden states + input embeddings (element-wise sum, 3584-dim/token), NOT from a specific intermediate layer. Qwen3-Omni sends from layer 24 (`accept_hidden_layer=24` in the released model config). See Section 10.2 for the full Qwen2.5-Omni vs. Qwen3-Omni comparison.

**Note 6:** For every forward pass, we are making a computation graph and dispatching it. Have subgraphs all ready per request. The scheduling part of conductor launches them when they are done (for example flow part of subgraph is launched at \<BOI\> and killed at \<EOI\>). We keep the talker subgraph alive throughout.

**Note 7:** idea on persistence of subgraphs: For every request, the conductor holds the current computation graph / execution plan. When the next forward pass starts, it retrieves a new execution plan from the model IF the input or output modalities change. If we have a new execution plan, the conductor formulates the new plan as a series of additions / subtractions from the current computation graph.
- The addition/subtraction mechanism can be trivial: the model can produce the "full possible computation" graph, and the get\_execution\_plan function can return whether each stage is active or not.
- `showo2_model.execution_plan(input_modalities, output_modalities, metadata)` -- per-request, per-forward pass
- This solves a few issues:
  - reduces recomputation of execution plans (don't need to recompute for each fwd pass, just when the input or output modalities change)
  - how to specify that the "talker is on for the duration of the request" paradigm
  - how to make the preference for, e.g., decode for request0 always running on GPU1 to prevent KV cache transfer, a first-class concept

*(Note 8 does not exist -- numbering skips from 7 to 9)*

**Note 9:** modification of needs/produces/pointer design: every subgraph has a list of inputs that it needs. We merge the produces/pointer concepts: every subgraph can have multiple outputs that feed into different subgraphs, some of which will be ready in the middle of the subgraph execution. This is handled by multiple pointers to next subgraphs, each of which is associated with the corresponding produced tensors.

**Note 10:** Whenever we change subgraphs, the conductor should send the corresponding workers information to the extent of "when you get tensors X and Y, run subgraph Z"

**Note 11:** When a stage within a subgraph completes, send a "stage completed" message back to the conductor. This doesn't really trigger anything but is important for SLO-aware scheduling. We can also have a flag in the message saying "this also means the subgraph has completed"

**Note 12:** KV cache management: take the IPC logic from mooncake

**Note 13:** chunked prefill: Not on the conductor level. Just send everything to the worker and have the worker handle chunking. This supports, e.g., running prefill on an all-text chunk while a future image input is being encoded because everything is now through inter-worker communication.

**Note 14:** note on scheduling: Previously, we had the conductor handling what graph stages are ready to run and which are still waiting for their inputs. Now that we have everything passing between workers via inter-worker communication, this no longer fits into the conductor. We still need this logic (which graph stages are ready, which are still waiting for inputs), but it'll be at the worker level. I'm working on code/pseudocode for that right now.

**Note 15:** NXT_STEP has been renamed to DONE_WITH_FWD. They are one and the same.

**Note 16:** For now, continuous batching, cuda graphs, chunked prefill etc. happen on the worker and we have some mooncake style inter-worker communication.

**Note 17:** Should we fashion our workers after voxserve in any way (though they are highly specific for SpeechLMs) or just use vLLM Engine Worker for standard AR stages: Thinker, Talker, standard LLM decode and FlashInfer Custom Worker (interleaved stages: Show-o2 LLM+flow loops, shared-backbone routing) and DiT Worker (diffusion stages)?
