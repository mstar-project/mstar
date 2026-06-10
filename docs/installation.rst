Installation
============

Requirements
------------

- **Python 3.12+**.
- **Linux with an NVIDIA GPU** and a recent **CUDA** toolkit for the GPU model
  families. A CPU-only machine can exercise the graph/worker plumbing in *dummy mode*
  (submodules return ``None``) for development and the modular tests, but not real model
  inference.
- Enough GPU memory for the model you intend to serve — several families (e.g.
  BAGEL-7B, Qwen3-Omni-30B) are multi-GPU-class models.

Install from source
-------------------

``mminf`` is installed from source in editable mode:

.. code-block:: bash

   git clone https://github.com/merceod/multimodal_inference.git
   cd multimodal_inference
   pip install -e .

This pulls in the core runtime (PyTorch, FastAPI/Uvicorn, ZMQ, …) and installs the two
console scripts, ``mminf`` and ``mminf-serve``.

.. important::

   ``mminf`` pins **PyTorch 2.9** (``torch==2.9.1`` / ``torchvision==0.24.1`` /
   ``torchaudio==2.9.1``). The Qwen3-Omni extra depends on ``sgl-kernel``, which is built
   against torch 2.9 — newer torch will not work. By default pip installs the PyTorch wheel
   for the *newest* CUDA toolkit (currently CUDA 13), which may not match your machine. If
   your CUDA version differs, install the matching torch build **before** ``pip install -e .``
   — see `Matching your CUDA toolkit`_ below.

Optional dependencies
---------------------

Model families and some output formats need extra packages, exposed as pip *extras*:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Extra
     - Installs / use for
   * - ``.[bagel]``
     - BAGEL runtime: ``transformers``, ``flashinfer-python``, ``safetensors``,
       ``einops``, ``Pillow``, ``torchvision`` / ``torchaudio`` / ``torchcodec``,
       ``huggingface-hub``, ``regex``, and ``mooncake-transfer-engine`` (RDMA transport).
   * - ``.[qwen3_omni]``
     - Qwen3-Omni runtime: the BAGEL set plus ``flash-attn``, ``qwen-omni-utils``,
       ``sgl-kernel``, and ``datasets``.
   * - ``.[orpheus]``
     - Orpheus TTS runtime: ``transformers``, ``flashinfer-python``, ``safetensors``,
       ``einops``, ``huggingface-hub``, ``mooncake-transfer-engine``.
   * - ``.[pi05]``
     - Pi0.5 runtime: ``transformers``, ``flashinfer-python``, ``safetensors``,
       ``triton``, ``huggingface-hub``, ``mooncake-transfer-engine``.
   * - ``.[vjepa2]`` / ``.[vjepa2_ac]``
     - V-JEPA 2 runtime: ``safetensors``, ``torchcodec``, ``huggingface-hub``,
       ``mooncake-transfer-engine`` (``vjepa2_ac`` also adds ``flashinfer-python``).
   * - ``.[audio]``
     - ``soundfile`` — only needed to return **non-WAV** audio containers (mp3/flac/…)
       from the OpenAI/SDK audio surfaces. WAV/PCM output works without it.
   * - ``.[dev]``
     - ``ruff`` + ``pytest`` for linting and the test suite.
   * - ``.[all]``
     - The union of every model extra above — installs the full runtime for all model
       families in one shot (including ``flash-attn`` and ``sgl-kernel``). Convenient for a
       machine that serves multiple models; heavier and slower to install than a single
       family's extra.

Combine extras as needed:

.. code-block:: bash

   pip install -e ".[bagel,audio,dev]"

.. note::

   ``torch``, ``torchvision``, and ``torchaudio`` are already in the base install; each
   model extra adds that family's remaining runtime — FlashInfer for the autoregressive
   backbones, Transformers, safetensors, any codec/media libraries, and the Mooncake RDMA
   transport for disaggregated deployments.

GPU libraries
-------------

The GPU model families depend on:

- **FlashInfer** (``flashinfer-python``) — paged attention and continuous batching for the
  autoregressive backbones (every model with a ``KV_CACHE`` node runs attention through it).
- **flash-attn** — used by Qwen3-Omni.
- **mooncake-transfer-engine** — RDMA tensor transport for multi-GPU, disaggregated
  deployments. Single-node deployments can use shared-memory (``SHM``) or ``TCP`` transport
  instead (see :doc:`serving`).

These are installed by the extras above. Make sure your installed ``torch`` matches your
CUDA version *before* installing them.

Matching your CUDA toolkit
--------------------------

PyPI's default ``torch`` wheel targets the newest CUDA release. To get the build for *your*
CUDA toolkit, install the pinned torch trio from PyTorch's CUDA-specific index **first**, then
install ``mminf`` (pip then sees the requirement as already satisfied and won't pull a
different build):

.. code-block:: bash

   # CUDA 12.8
   pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
       --index-url https://download.pytorch.org/whl/cu128
   pip install -e ".[bagel]"      # add the extras you need

Swap ``cu128`` for your toolkit (e.g. ``cu126``, ``cu130``). Check your driver's CUDA version
with ``nvidia-smi`` and pick the closest build at
https://pytorch.org/get-started/locally/. Keep all three packages on the same ``2.9`` /
``0.24`` line — ``torchvision`` and ``torchaudio`` are versioned in lockstep with ``torch``.

Verify the install
------------------

.. code-block:: bash

   python -c "import mminf; print('mminf import OK')"
   mminf --help
   mminf-serve --help

Next: :doc:`quickstart`.
