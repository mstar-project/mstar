M\* documentation
=================

M\* (“M-star”) serves models that generate or process text, images, speech, video, and actions. Start with a supported model and deployment config, then send requests through the Python SDK, an OpenAI-compatible endpoint, or the native API. Available interfaces depend on the model and task.

- `Install M\* <installation.html>`__ and `send your first request <quickstart.html>`__.
- `Choose a model <models.html>`__ and check its dependencies, inputs, and deployment examples.
- `Configure a deployment <serving.html>`__ or `connect a client <clients.html>`__.
- `Follow a request through the system <architecture.html>`__ or `add a model <adding_models.html>`__.

A model declares a graph of components, such as an encoder, a transformer, or an audio decoder. Each request follows graph walks through those components. The deployment YAML assigns nodes to GPU ranks, so supported placement changes do not require rewriting the model. The conductor coordinates the walks; workers execute ready nodes and return streamed results.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   serving
   metrics
   clients
   evaluation

.. toctree::
   :maxdepth: 2
   :caption: Reference

   architecture
   models
   asr
   api
   environment_variables

.. toctree::
   :maxdepth: 2
   :caption: Contributing

   adding_models
