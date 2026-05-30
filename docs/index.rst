mminf Documentation
===================

``mminf`` is a **disaggregated multimodal inference engine**. It serves multimodal
models (vision, audio, text, actions) over HTTP via a graph-based execution system
where logical computation nodes are decoupled from physical GPU workers.

A request flows ``HTTP → API server → conductor → workers → streamed results``. Each
model declares its own computation graph (a *graph walk*), and the conductor walks
that graph to coordinate multi-engine pipelines across one or more GPUs.

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
