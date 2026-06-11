# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`mstar` is a disaggregated multimodal inference engine. It serves multimodal models (vision, audio, text, actions) over HTTP via a graph-based execution system where logical computation nodes are decoupled from physical GPU workers.

## Common Commands

```bash
# Install (editable)
pip install -e ".[dev]"

# Lint
ruff check .

# Format
ruff format .

# Run all modular tests
pytest test/modular/

# Run a single test
pytest test/modular/test_phase1.py::TestPageAllocator::test_allocate_and_free

# Run integration tests (requires GPU + model weights)
pytest test/integration/

# Start the server
mstar-serve --config configs/<model>.yaml --host 0.0.0.0 --port 8000
```

## Architecture

**Request flow:** HTTP request → API Server (FastAPI) → Conductor → Workers (one per GPU) → streaming results back to API Server.

### Key components

- **API Server** (`mstar/api_server/`): FastAPI endpoint handling tokenization, media loading, and result collection. Entry point is `entrypoint.py`. Accepts `POST /generate`.
- **Conductor** (`mstar/conductor/`): Central coordinator that manages request lifecycle, selects workers, routes inputs, and detects completion.
- **Workers** (`mstar/worker/`): One process per GPU. Each has an engine manager, micro-scheduler (continuous batching), and KV cache manager. Workers route tensors directly to downstream workers.
- **Engines** (`mstar/engine/`): Execution backends — `AREngine` (autoregressive), `FlowEngine` (diffusion/ODE), `EncoderDecoderEngine` (vision/audio encoding), `AudioCodecEngine`.
- **Models** (`mstar/model/`): Each model inherits from `Model` (base.py) and defines its computation graph, forward pass orchestration, tokenization, and engine types. Registered via `registry.py`.
- **Graph** (`mstar/graph/`): Computation graph primitives — `GraphNode`, `Sequential`, `Parallel`, `Loop`, `GraphEdge`. Models declare graph walks that the conductor executes.
- **Communication** (`mstar/communication/`): ZMQ-based IPC/TCP for inter-process messaging. Tensor transport via RDMA or TCP.
- **Streaming** (`mstar/streaming/`): Handles streaming output with configurable chunking policies.

### Core design principles

- **Models define execution plans**: Each model provides its own graph walks (prefill, decode, image_gen, etc.) via `get_graph_walk_graphs()`.
- **Disaggregated**: Computation nodes (logical) map to workers (physical) via YAML config `node_groups`.
- **Graph-driven scheduling**: The conductor walks the computation graph to coordinate multi-engine pipelines.

### Supported models

BAGEL, Show-o2, Janus Pro, Qwen2.5-Omni, Qwen3-Omni, Pi0.5, Orpheus. Dummy models exist for testing without GPU.

## Config Format

YAML files in `configs/` map model nodes to GPU ranks:
```yaml
model: "pi05"
max_seq_len: 2048
node_groups:
  - node_names: ["vit_encoder"]
    ranks: [0]
  - node_names: ["LLM"]
    ranks: [0]
```

## Code Style

- Ruff for linting and formatting. Line length: 120. Target: Python 3.12.
- CI runs `ruff check --output-format=github .` on PRs to main.
