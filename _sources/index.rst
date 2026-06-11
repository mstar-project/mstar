M* Documentation
================

``mstar`` is the package behind **M\*** ("M-star"), a **disaggregated, any-to-any
multimodal inference engine**. It serves models built from structurally distinct
components — vision encoders, transformer backbones, diffusion / flow heads, audio codecs,
action generators, world-model predictors — whose execution path changes with the input
and the task.

The core abstraction is the *graph walk*: each model declares its computation as a
dataflow graph of components, and every request is a walk over that graph. A request flows
``HTTP / SDK → API server → conductor → workers → streamed results``; the conductor schedules
the model's graph walks to coordinate multi-engine pipelines across one or more GPUs. Logical
graph structure is decoupled from physical placement, so the *same* model runs single-GPU
or fully disaggregated by changing only a YAML config.

One runtime serves unified multimodal models, omni models, speech LMs,
vision-language-action policies, and world models — through a **Python SDK**, an
**OpenAI-compatible API**, and a native streaming endpoint.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   serving
   clients

.. toctree::
   :maxdepth: 2
   :caption: Reference

   architecture
   models
   api

.. toctree::
   :maxdepth: 2
   :caption: Contributing

   adding_models
