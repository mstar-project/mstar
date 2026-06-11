<p align="center">
  <img src="assets/mstar-logo.svg" alt="M*" width="300">
</p>

<h3 align="center">A universal serving system for composite, any-to-any multimodal models</h3>

<p align="center">
  <em>Models are dataflow graphs &nbsp;·&nbsp; requests are <strong>Walks</strong> &nbsp;·&nbsp; one runtime serves them all</em>
</p>

<p align="center">
  <a href="#quickstart"><b>Quickstart</b></a> &nbsp;·&nbsp;
  <a href="#supported-models"><b>Models</b></a> &nbsp;·&nbsp;
  <a href="#how-it-works"><b>How it works</b></a> &nbsp;·&nbsp;
  <a href="https://mstar-project.github.io/mstar/"><b>Docs</b></a> &nbsp;·&nbsp;
  <a href="https://mstar-project.github.io/"><b>Blog</b></a> &nbsp;·&nbsp;
  <a href="https://mstar-project.github.io/mstar.pdf"><b>Paper</b></a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-7b61ff.svg" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/python-3.12-22d3ee.svg" alt="Python 3.12">
  <img src="https://img.shields.io/badge/modalities-text_image_audio_video_action-6a8cff.svg" alt="Modalities">
  <a href="https://mstar-project.github.io/mstar/"><img src="https://img.shields.io/badge/docs-online-3b82f6.svg" alt="Docs"></a>
  <a href="https://mstar-project.github.io/mstar.pdf"><img src="https://img.shields.io/badge/paper-PDF-b31b1b.svg" alt="Paper (PDF)"></a>
</p>

---

## What is M*?

**M\*** (pronounced *"M-star"*) is a serving system for the new generation of **composite multimodal models** — models built from structurally distinct components (vision encoders, transformer backbones, diffusion and flow heads, audio codecs, action generators, world-model predictors) whose execution path changes with the input and the task.

LLM serving stacks assume inference is a single autoregressive loop. Composite models broke that assumption. M\*'s core idea is the **Walk Graph**: a model is a dataflow graph of its components, and every request is a *Walk* over that graph. A single runtime serves unified multimodal models, omni models, speech LMs, vision-language-action policies, and world models — at or above the performance of engines specialized for each.

**Fast** — per-component fast paths, matched to each component's bottleneck:
- Paged attention (FlashInfer) and continuous batching for autoregressive backbones
- CUDA-graph capture for encoders and decode
- Classifier-free-guidance parallelism for diffusion / flow
- Sliding-window chunk streaming for audio codecs
- Component-level disaggregation with pluggable tensor transport (shared memory, TCP, RDMA)

**Flexible** — the abstraction mirrors the model:
- One small Python file per model declares its component graph and its Walks
- A YAML file maps components to GPUs at per-component, per-walk granularity — arbitrary disaggregation, no code changes
- Text, image, audio, video, and robot actions, in and out
- A **Python SDK**, an **OpenAI-compatible API**, and a native streaming endpoint

> **Roadmap.** M\* is evolving toward *many-model, agentic* multimodal serving — routing requests across many models and tools within one graph-scheduled runtime.

## Quickstart

```bash
pip install -e .            # install M*
mstar serve bagel          # one command — launch a server (default: http://localhost:8000)
```

Other models: `mstar serve qwen3_omni` · `mstar serve orpheus` · `mstar serve pi05` · `mstar serve vjepa2`

**Python SDK** — works for every model (text, image, audio, video):

```python
from mstar import MStarClient
client = MStarClient("http://localhost:8000")

client.chat("What is the capital of France?").text          # text
client.generate_image("a cat in a hat")                     # → PNG bytes   (BAGEL)
client.tts("Hello there", voice="tara").to_wav("out.wav")   # → speech      (Orpheus)

for event in client.chat("Tell me a story", stream=True):   # streaming
    print(getattr(event, "text", ""), end="", flush=True)
```

**OpenAI-compatible API** — drop-in for `bagel`, `qwen3_omni`, and `orpheus`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

client.chat.completions.create(model="bagel", messages=[{"role": "user", "content": "hi"}])
client.audio.speech.create(model="orpheus", input="hi", voice="tara")   # text-to-speech
client.images.generate(model="bagel", prompt="a cat")                   # image generation
```

Runnable scripts and `curl` examples live in [`examples/`](examples/). Power users can launch any
deployment with an explicit config: `mstar-serve --config configs/<model>.yaml`.

## Supported models

| Model | Family | Input → Output | Endpoints |
|-------|--------|----------------|-----------|
| [BAGEL](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) | Unified multimodal | text, image → text, image | `/v1/chat/completions`, `/v1/images/generations` |
| [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | Omni | text, image, audio, video → text, speech | `/v1/chat/completions` |
| [Orpheus](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) | Speech LM | text → speech | `/v1/audio/speech` |
| [Pi0.5](https://huggingface.co/lerobot/pi05_base) | Vision-language-action | text, image, state → robot actions | `/generate` |
| [V-JEPA 2 / 2-AC](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) | World model | video (+ actions) → latents, rollouts | `/generate` |

Every model is reachable through the SDK and the native `/generate` endpoint; the OpenAI-compatible
routes cover the chat, speech, and image models.

## How it works

```
HTTP / SDK  →  API Server  →  Conductor  →  Workers (one per GPU)  →  streaming results
                                  │              │
                          walks the graph,   own subgraphs; route tensors
                          schedules walks    directly to one another
```

A model declares a **computation graph** of components and a set of named **Walks** (e.g.
`prefill`, `decode`, `image_gen`). The **Conductor** turns each request into a walk over that graph
and schedules it; **Workers** each own a subgraph on their GPU and stream tensors directly to one
another. Logical graph structure is decoupled from physical placement, so the same model runs
single-GPU or fully disaggregated by changing only the YAML `node_groups`. Four composable
primitives — `Sequential`, `Parallel`, `Loop`, and a cross-partition
`StreamingGraphEdge` — express every model family above. See the [paper](https://mstar-project.github.io/mstar.pdf) for the full design.

## Performance

Across every model we benchmark, M\* matches or beats the system specialized for that family — unified
models (BAGEL), omni and speech models (Qwen3-Omni, Orpheus), and world models (V-JEPA 2) — by executing
only the components each request needs and giving each its own fast path: paged attention and continuous
batching for autoregressive backbones, classifier-free-guidance parallelism for diffusion, chunk streaming
for audio codecs, and persistent-cache loops for world-model rollouts.

Benchmark numbers shift as systems evolve — ours and everyone else's — so rather than freeze figures here
that go stale, we keep the current results and full methodology in the
[blog post](https://mstar-project.github.io/) and the [paper](https://mstar-project.github.io/mstar.pdf).

## Contributing

Issues and pull requests are welcome. Found a bug, or want a model or feature supported?
**[Open an issue](https://github.com/mstar-project/mstar/issues).** To add a model yourself,
follow the [Adding a New Model](https://mstar-project.github.io/mstar/adding_models.html) guide.
PRs to `main` go through review and CI (`ruff`).

## Citation

If you use M\* in your research, please cite:

```bibtex
@article{mstar2026,
  title  = {M*: A Modular, Extensible, Serving System for Multimodal Models},
  author = {Jha, Atindra and Sagan, Naomi and Kamahori, Keisuke and Sivgin, Irmak and
            Sanda, Rohan and Gao, Steven and Horowitz, Mark and Zettlemoyer, Luke and
            Hsu, Olivia and Leskovec, Jure and Kasikci, Baris and Wang, Stephanie},
  year   = {2026},
  note   = {Preprint}
}
```

From Stanford University & the University of Washington. Correspondence: `atindra@cs.stanford.edu`.

## Acknowledgments

M\* builds on ideas and proven primitives from the open-source community — paged attention and
continuous batching ([vLLM](https://github.com/vllm-project/vllm)),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer) kernels, streaming speech serving
(VoxServe), and RDMA tensor transport ([Mooncake](https://github.com/kvcache-ai/Mooncake)).

## License

[Apache License 2.0](LICENSE).
