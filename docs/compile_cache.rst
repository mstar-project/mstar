Compile-cache bundles
=====================

A first start compiles every captured forward with Inductor (tuning it at ``-O1`` and
``-O2``), Triton builds the kernels, and on ROCm MIOpen searches its convolution solvers.
All three cache what they produce (:ref:`serving:Restarts and the compile cache`), so a
start on the same host skips that work. A *bundle* carries those caches to another host or
container with the same software stack, so its first start is warm too.

.. code-block:: bash

   # on a host that has served the configs you want to ship warm
   mstar cache pack /shared/mstar-bundles --note "qwen3_5_9b, whisper_large"

   # on the new host, before serving
   mstar cache unpack /shared/mstar-bundles
   # or in one step
   mstar serve qwen3_5_9b --compile-bundle /shared/mstar-bundles

The stack key
-------------

A bundle is only useful on the stack that compiled it, so bundles are kept per key:

.. code-block:: text

   <product>/torch-<torch version>-<hip|cuda>-<version>-triton-<triton version>
   mi325x-vf/torch-2.12.1+rocm7.2-hip-7.2.53211-triton-3.7.1

``mstar cache key`` prints this host's. The product is the GPU's name, lower-cased with
``AMD Instinct`` dropped. The manifest also records the GPU arch (``gfx942``, ``sm_90``).

- The arch, torch, HIP or CUDA, and Triton versions must match. Inductor and Triton key
  their own entries on them, so a bundle from another stack would unpack without error and
  never be read. ``mstar cache unpack`` refuses it (``--force`` unpacks it anyway).
- The product doesn't have to match. Autotuning picks are timings taken on the product
  that compiled them: an MI300X bundle loads on an MI325X (both gfx942) and starts warm, but
  its kernels were tuned on the MI300X. The startup line says so.
- Nothing in the key covers the driver, ``ptxas`` or the M* version. A newer M* that changes
  a captured forward's graph compiles that graph again on first start and keeps the rest.

Layout
------

A bundle root holds one directory per key; each is a bundle:

.. code-block:: text

   <root>/<product>/<torch-...-triton-...>/
       manifest.json        the key, member count and size per tar, and one history entry per pack
       inductor.tar         TORCHINDUCTOR_CACHE_DIR, less a Triton dir inside it
       triton.tar           TRITON_CACHE_DIR
       miopen.tar           MIOPEN_USER_DB_PATH (ROCm)
       miopen-kernels.tar   MIOPEN_CUSTOM_CACHE_DIR, only when it is a different directory
       .append.lock         taken by every pack

Each tar is uncompressed and holds its directory's files as ``./<path>``, so it can be
read and merged with plain ``tar``. ``unpack`` and ``pack`` find these directories the way
a server started from the same environment does: an exported variable, else M*'s default
for Inductor and Triton (``~/.cache/mstar/torchinductor`` and its ``triton``
subdirectory, see :doc:`environment_variables`), else the library's own (MIOpen's
``~/.config/miopen`` and ``~/.cache/miopen``). ``mstar cache dirs`` prints them.

Not in a bundle: FlashInfer's JIT builds (``FLASHINFER_WORKSPACE_BASE``), AITER's and
FlyDSL's on ROCm (``AITER_JIT_DIR``, ``FLYDSL_RUNTIME_CACHE_DIR``), which are keyed by their
own package versions rather than by this key, and the weights.

Building a bundle
-----------------

#. Point the caches at empty directories (or start from a bundle for the same key), for
   example ``export XDG_CACHE_HOME=$(mktemp -d)``.
#. Start each config you want warm and send it a request of every shape that compiles on
   first use, such as each image size a diffusion model will serve. Most of the work
   happens before ``/health`` answers: every capture bucket is compiled at startup.
#. Stop the servers, then ``mstar cache pack ROOT``. Packing must not run while a server
   is writing to the caches.

``pack`` adds to the bundle for this host's key under ``ROOT``, creating it if needed. It
appends the Inductor and Triton files the bundle doesn't have and those whose size or
modification time differ from its last copy; their entries never change once written.
MIOpen keeps each database in one file (the find and perf databases as ``key=value``
text, compiled kernels in SQLite), so a changed one is merged with the bundle's copy,
this host's record winning where both have one: caches warmed separately, each from
empty, add up rather than replace each other. Each tar is copied, appended to and renamed
into place, so a concurrent ``unpack`` never reads a partial tar, and the lock serializes
packs, so several hosts with the same key can add to one bundle in turn.

Unpacking
---------

``mstar cache unpack SRC`` takes a bundle or a root of bundles, picks the one for this
host's key, and extracts each tar into its cache directory over what is there; where both
have a file, the bundle's copy wins, and with repeated copies in a tar the last one does.
It exits with status 3 when ``SRC`` has no bundle for this stack, and ``mstar serve
--compile-bundle`` then starts cold with a warning. ``unpack`` leaves
``mstar-bundle.json`` in the Inductor directory, naming the bundle and its key.

At startup each worker logs its stack key and what it found:

.. code-block:: text

   Worker audio_encoder: compile cache: stack mi325x-vf/torch-...-triton-3.7.1; bundle /shared/mstar-bundles/mi325x-vf/torch-... unpacked 2026-10-29T21:04:11+00:00 matches
   Worker decoder: compile cache: stack mi325x-vf/...; no bundle unpacked into /root/.cache/mstar/torchinductor

A bundle built for another arch, torch, HIP, CUDA or Triton logs a warning instead, since
its entries won't be used.

``mstar cache show ROOT`` lists the bundles under a root with their sizes and last packs.
