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

``mstar`` is installed from source in editable mode. We recommend `uv
<https://docs.astral.sh/uv/>`_ to create the Python 3.12 environment:

.. code-block:: bash

   git clone https://github.com/mstar-project/mstar.git
   cd mstar

   # Create and activate a Python 3.12 virtualenv (--seed adds pip to it)
   uv venv --python 3.12 --seed
   source .venv/bin/activate

   uv pip install --torch-backend=auto -e .

This pulls in the core runtime (PyTorch, FastAPI/Uvicorn, ZMQ, …) and installs the two
console scripts, ``mstar`` and ``mstar-serve``.

.. important::

   **Always pass** ``--torch-backend=auto``. ``mstar`` floors PyTorch at 2.9
   (``torch>=2.9.1`` / ``torchvision>=0.24.1`` / ``torchaudio>=2.9.1``). The flag tells 
   ``uv`` to detect your driver's CUDA version and fetch the **matching** torch build
   — cu128 on a CUDA 12.x box, cu130 on a CUDA 13.x box. This matters because source-
   compiled extensions (``flash-attn``) and the JIT-built MoE kernel for Qwen3-Omni
   compile against your *system* CUDA toolkit, whose major version must match torch's.
   Without the flag ``uv`` installs PyPI's default (cu128) build. You can set it once
   with ``export UV_TORCH_BACKEND=auto`` instead of repeating the flag. See `Matching your
   CUDA toolkit`_ for details and the manual fallback.

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
     - Qwen3-Omni runtime: the BAGEL set plus ``qwen-omni-utils``, ``datasets``, and
       ``ninja`` (speeds up the JIT build of the vendored MoE align kernel).
       **Also needs** ``flash-attn``, which is installed separately —
       see `flash-attn (Qwen3-Omni)`_.
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
       families in one shot. Convenient for a machine that serves multiple models; heavier
       and slower to install than a single family's extra. (Still excludes ``flash-attn`` —
       see `flash-attn (Qwen3-Omni)`_.)

Combine extras as needed (keep ``--torch-backend=auto`` on every install):

.. code-block:: bash

   uv pip install --torch-backend=auto -e ".[bagel,audio,dev]"

.. tip::

   If you're just getting started or have the disk/time to spare, ``.[all]`` is the
   recommended install — it pulls every model family's runtime so any model works out of
   the box, with no need to track which extra goes with which model:

   .. code-block:: bash

      uv pip install --torch-backend=auto -e ".[all,dev]"

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
- **flash-attn** — used by Qwen3-Omni. **Not installed by any extra**; install it separately
  (see `flash-attn (Qwen3-Omni)`_).
- **mooncake-transfer-engine** — RDMA tensor transport for multi-GPU, disaggregated
  deployments. Single-node deployments can use shared-memory (``SHM``) or ``TCP`` transport
  instead (see :doc:`serving`).

Apart from ``flash-attn``, these are installed by the extras above. Your installed ``torch``
must match your system CUDA toolkit — ``--torch-backend=auto`` handles that for you (next
section).

flash-attn (Qwen3-Omni)
-----------------------

``flash-attn`` is only needed for **Qwen3-Omni**, and it is **not** pulled in by
``.[qwen3_omni]`` or ``.[all]`` — you install it as a separate step. The reason: flash-attn
publishes no wheels on PyPI, so ``pip``/``uv`` fall back to compiling it from source, which is
slow. When a prebuilt GitHub wheel matches your stack, use it to skip the build; otherwise
build from source (see `No matching wheel — build from source`_ below).

The wheels live on flash-attn's `GitHub releases
<https://github.com/Dao-AILab/flash-attention/releases>`_, named by CUDA major, torch
**minor** version, Python tag, and C++ ABI. ``mstar`` no longer pins torch (floor
``>=2.9.1``), so the wheel's ``torchX.Y`` tag must match **whatever torch you actually
installed**, and its ``cu1x`` must match that torch's CUDA build — not your system toolkit.
Check both first:

.. code-block:: bash

   python -c "import torch; print(torch.__version__, torch.version.cuda)"
   # e.g. 2.12.1+cu130 13.0  ->  torch2.12 / cu13
   #      2.9.1+cu128  12.8  ->  torch2.9  / cu12

Pick the release that has a wheel for your torch minor (each flash-attn release lists which
``torchX.Y`` tags it ships), then install it by **direct URL** (don't use ``--find-links`` —
uv sorts the ``+cu13…`` local version above ``+cu12…`` and will grab cu13 even on a CUDA 12
box). The examples below are for **torch 2.9**; substitute the ``torch2.9`` segment with your
own tag. Note the prebuilt wheels **top out at** ``torch2.10`` (``cu13``, in release
``v2.8.3``) — there is **no** ``torch2.11+`` wheel, so on newer torch (e.g. 2.12) you must
build from source instead:

.. code-block:: bash

   # torch built for CUDA 12.x (cu12)
   uv pip install \
     "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

   # torch built for CUDA 13.x (cu13)
   uv pip install \
     "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

Because it's a binary wheel nothing compiles, so your *system* CUDA version is irrelevant —
only the torch build matters. Verify with:

.. code-block:: bash

   python -c "import flash_attn; print(flash_attn.__version__)"

(An ``undefined symbol`` error on import means the wheel's ``cu1x`` / ``torchX.Y`` tag doesn't
match your installed torch — recheck ``python -c "import torch; print(torch.__version__,
torch.version.cuda)"`` and pick the matching wheel.)

No matching wheel — build from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If no prebuilt wheel matches your torch minor (the wheels stop at ``torch2.10``, so this is
the case on **torch 2.11+**), compile from source. A source build compiles against **whatever
torch you have installed**, so the torch-minor mismatch goes away. flash-attn 2.8.3 builds
against a CUDA-13 toolkit, but **only if you restrict the target architectures** (see below) —
the default arch list pulls in Blackwell targets that don't compile under nvcc 13.x. It is a
long compile (tens of minutes).

Three things to get right:

* ``FLASH_ATTN_CUDA_ARCHS`` — set it to **only your GPU's compute capability** (H100 →
  ``"90"``, A100 → ``"80"``). By default flash-attn builds ``80;90;100;120``, and on a CUDA 13
  toolkit that includes the Blackwell targets (``sm_100``/``sm_120``). flash-attn 2.8.3's
  kernels **fail to compile for Blackwell under nvcc 13.x** — the build dies on
  ``flash_bwd_hdim256_*`` with a bare ``[code=255]`` (the only hint upstream is a ``double4``
  *deprecation* warning, which is not itself the error). Restricting the arch list avoids the
  broken targets and cuts build time several-fold.
* ``--no-build-isolation`` — build against your *installed* torch. Without it, uv builds in an
  isolated env that pulls a fresh (possibly different-minor) torch, defeating the point.
* ``psutil`` must be importable **before** the build — flash-attn's ``setup.py`` imports it to
  size the parallel compile, and it is not declared as a build dependency.

.. code-block:: bash

   uv pip install psutil
   FLASH_ATTN_CUDA_ARCHS="90" uv pip install flash-attn==2.8.3.post1 --no-build-isolation
   python -c "import flash_attn; print(flash_attn.__version__)"

Matching your CUDA toolkit
--------------------------

PyPI's default ``torch`` wheel targets one specific CUDA release (cu128), which may not match
your machine. ``flash-attn`` compiles from source, and the vendored MoE align kernel
JIT-compiles on first use, both against your *system* CUDA — so a mismatch with torch's
CUDA breaks the build. The simplest fix is to let ``uv`` choose the right build
automatically:

.. code-block:: bash

   uv pip install --torch-backend=auto -e ".[all]"

``--torch-backend=auto`` detects your driver (via ``nvidia-smi``) and selects the matching
PyTorch index — cu128 on CUDA 12.x, cu130 on CUDA 13.x — for the runtime *and* for the
isolated environment that builds ``flash-attn``. The same command therefore
works unchanged across machines. (Needs a recent ``uv`` — run ``uv pip install --help`` and
look for ``--torch-backend`` if unsure; ``export UV_TORCH_BACKEND=auto`` is equivalent.)

**Manual fallback.** If you can't use the flag, install the pinned torch trio from the
matching CUDA index **first**, then install ``mstar`` (the resolver then treats the
requirement as already satisfied):

.. code-block:: bash

   # pick the index for your CUDA toolkit: cu128 (CUDA 12.8), cu130 (CUDA 13.x), …
   uv pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
       --index-url https://download.pytorch.org/whl/cu128
   uv pip install -e ".[all]"

Check your driver's CUDA version with ``nvidia-smi`` and pick the closest build at
https://pytorch.org/get-started/locally/. Keep all three packages on the same ``2.9`` /
``0.24`` line — ``torchvision`` and ``torchaudio`` are versioned in lockstep with ``torch``.

Verify the install
------------------

.. code-block:: bash

   python -c "import mstar; print('mstar import OK')"
   mstar --help
   mstar-serve --help

Troubleshooting
---------------

``CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symptom — a convolution (or another cuDNN op) aborts with::

   RuntimeError: ... cudnn_status: CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH

This affects **CUDA 12** installs (a ``+cu12x`` torch) running on a host that *also* has a
**newer system cuDNN (≥ 9.21)** on the default loader path (e.g. ``/usr/lib64``). PyTorch's
cu12 wheels pin ``nvidia-cudnn-cu12==9.20.0.48``, and that slim wheel omits the
``libcudnn_engines_tensor_ir`` engine. When cuDNN needs that engine it loads it from the
system copy instead — and a 9.20 dispatcher paired with a ≥ 9.21 engine is the mismatch.
Boxes with no system cuDNN, or a matching one, are unaffected, as are CUDA 13 installs
(they use ``nvidia-cudnn-cu13``).

Fix — complete the environment's own cuDNN so it never falls back to the system copy:

.. code-block:: bash

   uv pip install --no-deps -U "nvidia-cudnn-cu12>=9.21"

``--no-deps`` leaves torch in place; pip may print a harmless warning that torch pins
``9.20.0.48``. cuDNN is ABI-compatible within the v9 series, so the newer patch runs fine.

Next: :doc:`quickstart`.
