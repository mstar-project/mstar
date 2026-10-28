Installation
============

Requirements
------------

- **Python 3.12+**.
- **Linux with an NVIDIA GPU** and a recent **CUDA** toolkit for the GPU model
  families, or **Intel GPUs with XPU PyTorch** for BAGEL
  (see `Using Docker image on Intel GPU`_).
  A CPU-only machine can exercise the graph/worker plumbing in *dummy mode*
  (submodules return ``None``) for development and the modular tests, but not real model
  inference.
- Enough GPU memory for the model you intend to serve — several families (e.g.
  BAGEL-7B, Qwen3-Omni-30B) are multi-GPU-class models.

Install from PyPI
-----------------

Released versions are on PyPI as ``mstar-ai`` (``mstar-project`` and ``mstar-serve`` are
alias packages that install the same thing). The import package and the console scripts
are still ``mstar``. We recommend `uv <https://docs.astral.sh/uv/>`_ to create the
Python 3.12 environment:

.. code-block:: bash

   # Create and activate a Python 3.12 virtualenv (--seed adds pip to it)
   uv venv --python 3.12 --seed
   source .venv/bin/activate

   uv pip install --torch-backend=auto "mstar-ai[all]"
   mstar serve bagel

The extras are the same as for a source install (see `Optional dependencies`_), so
``mstar-ai[bagel]`` or ``mstar-ai[qwen3_omni,audio]`` work the same way. The default
deployment configs ship inside the package, so ``mstar serve <model>`` works without a
checkout. Keep ``--torch-backend=auto`` on every install, see the note below.

Install from source
-------------------

To work on the code, install from source in editable mode:

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

   **For CUDA installs, always pass** ``--torch-backend=auto``. ``mstar`` floors
   PyTorch at 2.10 (``torch>=2.10.0`` / ``torchvision>=0.25.0`` / ``torchaudio>=2.10.0``).
   The flag tells
   ``uv`` to detect your driver's CUDA version and fetch the **matching** torch build
   — cu128 on a CUDA 12.x box, cu130 on a CUDA 13.x box. This matters because source-
   compiled extensions (``flash-attn``) and the JIT-built MoE kernel for Qwen3-Omni
   compile against your *system* CUDA toolkit, whose major version must match torch's.
   Without the flag ``uv`` installs PyPI's default (cu128) build. You can set it once
   with ``export UV_TORCH_BACKEND=auto`` instead of repeating the flag. See `Matching your
   CUDA toolkit`_ for details and the manual fallback.

Using Docker image on Intel GPU
-------------------------------

Build the image using ``docker/Dockerfile.xpu``:

.. code-block:: bash

   git clone https://github.com/mstar-project/mstar.git
   cd mstar
   docker build -t mstar:xpu - < docker/Dockerfile.xpu


The host must have a working Intel GPU driver and expose ``/dev/dri``. Run an
interactive shell and check GPU access:

.. code-block:: bash

   docker run --rm -it --device=/dev/dri --ipc=host \
     -v /dev/dri/by-path:/dev/dri/by-path:ro \
     -p 8000:8000 \
     -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
     mstar:xpu
   python -c "import torch; print(torch.xpu.is_available(), torch.xpu.device_count())"

Inside the container, serve BAGEL using the TP2 deployment with three Intel GPUs:
one for the encoders and decoder, and two for the LLM.

.. code-block:: bash

   cd /mstar-workspace/mstar
   mstar serve bagel --config configs/bagel_xpu_tp2.yaml

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
   * - ``.[bagel_xpu]``
     - BAGEL runtime for Intel XPU. Pins ``vllm-xpu-kernels==0.1.15.4`` for
       batched token sampling with per-request RNG state on the device.
       Install the matching XPU build of PyTorch.
   * - ``.[command_a_plus]``
     - Command A+ runtime: ``flashinfer-python>=0.6.18`` (its one-kernel all-reduce),
       ``transformers>=5.17`` (the version its Hugging Face parity checks use),
       ``huggingface-hub``, ``safetensors``, ``mooncake-transfer-engine``, and ``ninja``
       (speeds up the JIT build of the vendored MoE align kernel). Set ``CUDA_HOME`` to a
       CUDA toolkit matching PyTorch's so that kernel can build. ``.[all]`` allows older
       FlashInfer and Transformers for the other models, so add ``.[command_a_plus]``
       when installing both.
   * - ``.[qwen3_omni]``
     - Qwen3-Omni runtime: the BAGEL set plus ``qwen-omni-utils``, ``datasets``, and
       ``ninja`` (speeds up the JIT build of the vendored MoE align kernel). It needs
       ``transformers>=5.18``, whose ``transformers.vision_utils`` the vision encoder uses.
       **Also needs** ``flash-attn``, which is installed separately —
       see `flash-attn (Qwen3-Omni, LingBot-World)`_.
   * - ``.[orpheus]``
     - Orpheus TTS runtime: ``transformers``, ``flashinfer-python``, ``safetensors``,
       ``einops``, ``huggingface-hub``, ``mooncake-transfer-engine``.
   * - ``.[fish_s2_pro]``
     - Fish Audio S2-Pro runtime: ``transformers``, ``flashinfer-python``, ``safetensors``,
       ``soundfile`` (reference-clip decoding), ``huggingface-hub``, ``mooncake-transfer-engine``.
   * - ``.[omnivoice]``
     - OmniVoice TTS. The extra is empty on purpose: the runtime is the
       ``omnivoice`` package, installed separately from git —
       see `omnivoice (OmniVoice)`_.
   * - ``.[pi05]``
     - Pi0.5 runtime: ``transformers``, ``flashinfer-python``, ``safetensors``,
       ``triton``, ``huggingface-hub``, ``mooncake-transfer-engine``.
   * - ``.[vjepa2]`` / ``.[vjepa2_ac]``
     - V-JEPA 2 runtime: ``safetensors``, ``torchcodec``, ``huggingface-hub``,
       ``mooncake-transfer-engine`` (``vjepa2_ac`` also adds ``flashinfer-python``).
   * - ``.[glm]``
     - GLM runtime: ``transformers``, ``flashinfer-python``, ``safetensors``,
       ``triton``, ``huggingface-hub``, ``mooncake-transfer-engine``, and ``ninja``
       (speeds up the JIT build of the vendored MoE align kernel).
   * - ``.[waypoint]``
     - Index-hosted Waypoint dependencies: ``huggingface-hub``, ``safetensors``,
       and ``tensordict`` for reference validation. The pinned TAEHV implementation
       must be installed separately as shown below; keeping its direct URL out of
       package metadata allows ``m-star`` to be published on PyPI. **Also needs**
       ``flash-attn-4``, which is installed separately —
       see `flash-attn-4 (Waypoint, FA3 sm90 kernel)`_.
   * - ``.[asr]``
     - Speech recognition models: ``transformers``, ``flashinfer-python``, ``safetensors``,
       ``einops``, ``soundfile`` (WAV / FLAC / OGG / MP3 uploads), ``torchcodec``
       (FFmpeg-backed decoding of the other formats), ``huggingface-hub``,
       ``mooncake-transfer-engine``, and ``datasets`` / ``jiwer`` for
       ``benchmark/asr_eval.py``.
   * - ``.[audio]``
     - ``soundfile`` — only needed to return **non-WAV** audio containers (mp3/flac/…)
       from the OpenAI/SDK audio surfaces. WAV/PCM output works without it.
   * - ``.[audioseal]``
     - ``audioseal`` (``>=0.2,<0.3``, MIT code and weights) for the opt-in AudioSeal
       watermark on generated speech (Qwen3-TTS ``audio_watermark``, see the
       Qwen3-TTS notes in :doc:`models`). Not part of ``.[all]``; a deployment that
       enables the watermark without it fails at startup with this install hint.
       The ``python -m mstar.model.audio_watermark detect`` CLI also needs ``.[audio]``.
   * - ``.[metrics]``
     - ``prometheus-client`` for ``--enable-metrics`` (Prometheus metrics at
       ``/metrics``, see :ref:`serving-metrics`). A serving feature rather than a
       model's runtime, so not part of ``.[all]``.
   * - ``.[dev]``
     - ``ruff`` + ``pytest`` for linting and the test suite.
   * - ``.[all]``
     - The union of the index-hosted dependencies from every model extra above.
       Convenient for a machine that serves multiple models; heavier and slower to
       install than a single family's extra. It excludes the separately installed
       TAEHV, ``flash-attn`` and ``qwen-tts`` packages; see below,
       `flash-attn (Qwen3-Omni, LingBot-World)`_ and `qwen-tts (Qwen3-TTS)`_.

Combine extras as needed (keep ``--torch-backend=auto`` on every install):

.. code-block:: bash

   uv pip install --torch-backend=auto -e ".[bagel,audio,dev]"

Waypoint's dependencies and pinned TAEHV source are two installs. PyPI and other
standards-conformant indices reject distributions whose metadata declares a direct-URL
dependency, so the ``waypoint`` extra intentionally does not name TAEHV. Check the
installer version before installing its source archive: an old pip may report success
while producing an empty ``UNKNOWN`` wheel.

.. code-block:: bash

   uv --version                 # must be 0.4.0 or newer
   uv pip install --torch-backend=auto -e ".[waypoint]"
   uv pip install --no-deps \
     "taehv @ https://github.com/madebyollin/taehv/archive/7dc60ec6601af2e668e31bc70acc4cb3665e4c22.zip"

Or, in an existing Python 3.12 environment:

.. code-block:: bash

   python -m pip install --upgrade "pip>=24.3"
   python -m pip install -e ".[waypoint]"
   python -m pip install --no-deps \
     "taehv @ https://github.com/madebyollin/taehv/archive/7dc60ec6601af2e668e31bc70acc4cb3665e4c22.zip"

.. tip::

   If you're just getting started or have the disk/time to spare, ``.[all]`` installs all
   index-hosted model dependencies in one shot. Waypoint still needs the pinned TAEHV
   command above, Qwen3-Omni still needs ``flash-attn``, and Qwen3-TTS still needs
   ``qwen-tts`` (see `qwen-tts (Qwen3-TTS)`_):

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
- **flash-attn** — used by Qwen3-Omni and by LingBot-World's ``reference_numerics`` mode. **Not installed by any extra**; install it separately
  (see `flash-attn (Qwen3-Omni, LingBot-World)`_).
- **flash-attn-4** — Waypoint's flex-attention ``FLASH`` backend. **Not installed by any
  extra**; install it separately (see `flash-attn-4 (Waypoint, FA3 sm90 kernel)`_).
- **mooncake-transfer-engine** (0.3.13.post1 or newer) — RDMA tensor transport for
  multi-GPU, disaggregated deployments. Single-node deployments can use shared-memory
  (``SHM``) or ``TCP`` transport instead (see :doc:`serving`, which also covers the network
  it needs). The extras install Mooncake's CUDA 12 wheel; next to a CUDA 13 torch
  (``+cu130``) swap in its CUDA 13 build, as the Docker images do (see
  :ref:`mooncake-cuda13`).

Apart from ``flash-attn`` and ``flash-attn-4``, these are installed by the extras above. Your
installed ``torch`` must match your system CUDA toolkit — ``--torch-backend=auto`` handles
that for you (next section).

omnivoice (OmniVoice)
---------------------

``omnivoice`` is only needed for **OmniVoice**, and it is **not** pulled in by
``.[omnivoice]`` or ``.[all]`` — you install it as a separate step:

.. code-block:: bash

   uv pip install "git+https://github.com/k2-fsa/OmniVoice.git"

It has to come from git rather than PyPI: the released package has no
``omnivoice.models.omnivoice_flashinfer``, and that is the packed-attention
forward mstar's graph calls, so a PyPI install fails at model load with
``ImportError: cannot import name 'omnivoice_flashinfer'``. A direct git URL
cannot live in ``pyproject.toml`` either, because PyPI rejects any
``requires_dist`` entry carrying one, which is why this is an install step
and not a dependency. Verify with:

.. code-block:: bash

   python -c "from omnivoice.models import omnivoice_flashinfer; print('ok')"

qwen-tts (Qwen3-TTS)
--------------------

**Qwen3-TTS** loads its 12 Hz speech tokenizer from the ``qwen-tts`` package, which
``.[qwen3_tts]`` and ``.[all]`` do **not** install. Add it as a separate step, without
its dependencies:

.. code-block:: bash

   uv pip install --no-deps "qwen-tts==0.1.1"

``qwen-tts`` pins ``transformers==4.57.3`` and ``accelerate==1.12.0``. Installed with
them, it can't share an environment with ``diffusers>=0.40`` (Wan 2.2, Cosmos3,
FLUX.2 [klein]), which needs ``huggingface-hub>=1.23``, or with OmniVoice
(``transformers>=5.3``) or Qwen3-Omni (``transformers>=5.18``). Installing it after them
downgrades their dependencies.
mstar reads only the speech-tokenizer source files and runs them under transformers
4.57.3 or 5.x, so the package's own requirements are not needed. ``pip check`` still
lists them (``transformers==4.57.3``, ``accelerate==1.12.0``, ``gradio``, ``sox`` and
others) as unmet for ``qwen-tts``; that is expected.

flash-attn (Qwen3-Omni, LingBot-World)
--------------------------------------

``flash-attn`` is only needed for **Qwen3-Omni** and **LingBot-World** with ``reference_numerics: true``, and it is **not** pulled in by
``.[qwen3_omni]``, ``.[lingbot_world]`` or ``.[all]`` — you install it as a separate step. The reason: flash-attn
publishes no wheels on PyPI, so ``pip``/``uv`` fall back to compiling it from source, which is
slow. When a prebuilt GitHub wheel matches your stack, use it to skip the build; otherwise
build from source (see `No matching wheel — build from source`_ below).

The wheels live on flash-attn's `GitHub releases
<https://github.com/Dao-AILab/flash-attention/releases>`_, named by CUDA major, torch
**minor** version, Python tag, and C++ ABI. ``mstar`` no longer pins torch (floor
``>=2.10.0``), so the wheel's ``torchX.Y`` tag must match **whatever torch you actually
installed**, and its ``cu1x`` must match that torch's CUDA build — not your system toolkit.
Check both first:

.. code-block:: bash

   python -c "import torch; print(torch.__version__, torch.version.cuda)"
   # e.g. 2.12.1+cu130 13.0  ->  torch2.12 / cu13
   #      2.10.0+cu130 13.0  ->  torch2.10 / cu13

Pick the release that has a wheel for your torch minor (each flash-attn release lists which
``torchX.Y`` tags it ships), then install it by **direct URL** (don't use ``--find-links`` —
uv sorts the ``+cu13…`` local version above ``+cu12…`` and will grab cu13 even on a CUDA 12
box). The example below is for **torch 2.10**, the floor. The prebuilt wheels **top out at**
``torch2.10``, and only for ``cu13`` (release ``v2.8.3``): on a cu12 torch, or on torch 2.11+,
you must build from source instead:

.. code-block:: bash

   # torch 2.10 built for CUDA 13.x (cu13)
   uv pip install \
     "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

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

flash-attn-4 (Waypoint, FA3 sm90 kernel)
----------------------------------------

Waypoint's DiT attention runs torch flex-attention with the ``FLASH`` backend by
default. That backend needs the **flash-attn-4** package, which provides
``flash_attn.cute`` — the CuTe DSL rewrite of flash-attn; on H100 it runs the
FA3-style sm90 kernel with TMA and warpgroup specialisation. It is **not on
PyPI** as of 2026-09-17, and it is **not** pulled in by any extra.

FLASH needs **torch 2.11 or newer**: older torch does not pass flash-attn-4 the
block-sparse block size, and capture fails with ``Block sparse tensors ...
require explicit sparse_block_size[0]``. Note this rules out the flash-attn
prebuilt wheels above, which stop at ``torch2.10``.

Install it from the upstream repo's ``flash_attn/cute`` subdirectory, pinned to
the revision this was tested against (with ``nvidia-cutlass-dsl`` 4.7.1 and
torch 2.12.1):

.. code-block:: bash

   git clone https://github.com/Dao-AILab/flash-attention
   git -C flash-attention checkout 1bda8f9290cd48d030f1516f0e680cd464ef3554
   uv pip install --torch-backend=auto ./flash-attention/flash_attn/cute

This is pure Python plus ``nvidia-cutlass-dsl`` — there is no CUDA extension to
build. Its kernels are JIT-compiled on first use, which adds roughly a minute to
the first server startup.

If ``flash-attn-4`` isn't importable or torch is older than 2.11, the server
logs a warning and falls back to the previous Triton flex kernel. It is correct
but slower — about 1.8x per attention call at 720p. Set
``MSTAR_FLEX_BACKEND=TRITON`` to choose it explicitly, or
``MSTAR_FLEX_BACKEND=FLASH`` to fail at startup instead of falling back.

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

.. note::

   **Mooncake on a CUDA 13 torch.** The extras install Mooncake's CUDA 12 wheel, which can't
   load next to a cu130 torch, so ``TCP`` and ``RDMA`` transport fail at startup; ``SHM`` is
   not affected. Swap in its CUDA 13 build as described in :ref:`mooncake-cuda13`.

**Manual fallback.** If you can't use the flag, install the pinned torch trio from the
matching CUDA index **first**, then install ``mstar`` (the resolver then treats the
requirement as already satisfied):

.. code-block:: bash

   # pick the index for your CUDA toolkit: cu128 (CUDA 12.8), cu130 (CUDA 13.x), …
   uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
       --index-url https://download.pytorch.org/whl/cu128
   uv pip install -e ".[all]" "torchcodec==0.10.*"

Check your driver's CUDA version with ``nvidia-smi`` and pick the closest build at
https://pytorch.org/get-started/locally/. Keep all three packages on the same ``2.10`` /
``0.25`` line — ``torchvision`` and ``torchaudio`` are versioned in lockstep with ``torch``.
``torchcodec`` doesn't declare its ``torch`` version, so the resolver otherwise picks the
newest release (0.17 at the time of writing), which fails to load next to torch 2.10; 0.10
is the release built against it.
The ``[asr]`` extra pins that pair itself (``torch`` 2.10 with ``torchcodec`` 0.10, as in the
Docker images), so installing it puts ``torch`` on 2.10.

Verify the install
------------------

.. code-block:: bash

   python -c "import mstar; print('mstar import OK')"
   mstar --help
   mstar-serve --help

Optional: Rust support
----------------------

Parts of the runtime can run on Rust components (vendored in ``rust/``),
each optional and selectable per process — without the extension everything
runs on the pure-Python paths as before. Migrated so far (see
:doc:`environment_variables` for the flags):

* **ZeroMQ control mesh** — ``MSTAR_RUST_ZMQ``: same endpoints, same wire
  format as pyzmq.
* **SHM tensor arena** — ``MSTAR_SHM_ARENA``: replaces the per-tensor file
  transport; requires the extension, interoperates on the same descriptor
  wire (and depends on the transport above only in the sense that both ship
  in the same extension).
* **Message encoding** — ``MSTAR_WIRE_CODEC``: typed msgpack rather than
  pickle, which is what lets one end of an edge be Rust. On by default,
  extension or not. The legacy ``pickle`` codec is no longer supported.
  Loosely typed metadata uses data-only extensions; upgrade all deployment
  processes together, since legacy pickle frames are rejected.
* **Graph runtime** — ``MSTAR_RUST_GRAPH``: graph state, scheduling,
  routing and speculation, and the frames a worker sends. Unlike the others
  this is not free-standing — it holds a share of the Rust transport and of
  the Rust tensor bookkeeper, so it runs only where ``MSTAR_RUST_ZMQ``
  resolved to Rust and the msgpack codec is in use. ``AUTO`` takes it
  wherever that holds, so on a machine with the extension built, installing
  it is what moves a worker onto the Rust runtime.

Build the extension into your environment with `maturin
<https://www.maturin.rs>`_ (needs a Rust toolchain; ``rustup`` works):

.. code-block:: bash

   uv pip install maturin
   maturin develop --release -m rust/Cargo.toml

Build with ``--release`` — an unoptimized debug build (maturin's default)
costs real latency on the hot receive path. Verify with:

.. code-block:: bash

   python -c "import mstar_rust; print('mstar_rust OK')"
   pytest test/rust

``test/rust`` is skipped wholesale when the extension is not importable, so
a green run there without the build above proves nothing.

The Rust-side tests need one extra flag. The crate's default features turn on
``pyo3/extension-module``, which tells pyo3 *not* to link libpython — right
for the cdylib, which resolves CPython symbols from the interpreter that
imports it, wrong for ``cargo test``, which is an executable and has to link
it itself. Without this the link fails on ``undefined symbol:
PyEval_RestoreThread``:

.. code-block:: bash

   export LD_LIBRARY_PATH="$(python -c 'import sysconfig
   print(sysconfig.get_config_var("LIBDIR"))')"
   cargo test --release --no-default-features --manifest-path rust/Cargo.toml

Optional: the Rust HTTP frontend
--------------------------------

``mstar-server`` (``rust/server/``) serves the HTTP surface — routes, OpenAI
translation, SSE streaming, uploads — from Rust, with the Python process
keeping preprocessing and the conductor protocol. Build and opt in:

.. code-block:: bash

   cargo build --release --manifest-path rust/server/Cargo.toml
   mstar serve qwen3_omni --rust-frontend
   # or, with an explicit config: mstar-serve --config <model.yaml> --rust-frontend

The binary is resolved via ``--rust-frontend-bin`` / ``MSTAR_SERVER_BIN`` /
``$PATH`` / the in-repo build. Its knobs are listed in
:doc:`environment_variables`.

The OpenAI routes take the model's adapter from the same registry as the Python
frontend (``ADAPTER_REGISTRY`` in ``mstar/api_server/openai/adapters.py``):
``mstar-server`` ports adapter classes, not model names, so every model of a
ported class is served. Chat is served for every model the Python frontend serves
it for. Some speech, image and video adapters, and the transcription and realtime
routes, are not ported yet (``NOT_PORTED`` in
``test/rust/test_rust_openai_adapters.py``); there ``mstar-server`` answers 404,
logs a warning at startup, and ``/generate`` still works.

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

.. _mooncake-cuda13:

``libcudart.so.12: cannot open shared object file``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symptom: with ``--tensor-comm-protocol TCP`` or ``RDMA``, startup fails with::

   RuntimeError: Mooncake TransferEngine is required for RDMA/TCP protocol. Failed to load
   mooncake: ImportError: libcudart.so.12: cannot open shared object file: ...

The extras install ``mooncake-transfer-engine``, Mooncake's default wheel, which is built for
CUDA 12. A CUDA 13 torch (``+cu130`` or newer, which ``--torch-backend=auto`` picks on a
CUDA 13 driver) doesn't bring the CUDA 12 runtime, so the wheel can't load. ``SHM`` is not
affected.

Fix: swap in Mooncake's CUDA 13 build, which installs the same ``mooncake`` package:

.. code-block:: bash

   uv pip uninstall mooncake-transfer-engine
   uv pip install --reinstall --no-deps mooncake-transfer-engine-cuda13

Both wheels install the same files, so installing an extra again puts the CUDA 12 build back.
Repeat the swap afterwards.

Next: :doc:`quickstart`.
