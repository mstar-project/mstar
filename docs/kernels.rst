Adding a Kernel
===============

Kernels are engine-level: a model never calls one by name. A resource or a
layer component calls an ``mstar::<name>`` op, and the op runs whichever
implementation serves the process's platform and GPU architecture. This page
covers declaring such an op, building C++/HIP sources for it, tuning it per
device, and measuring it with the kernel harness.

Where things live
-----------------

``mstar/kernels/registry.py``
   ``KernelOp``: one declaration per op (schema, fake, reference,
   implementations per platform and architecture, shapes, parity bar).
   ``python -m mstar.kernels [--probe]`` lists every op.
``mstar/kernels/numerics.py``
   The parity bar (``Tolerance``) and the fp64 reference evaluation.
``mstar/kernels/hip/build.py``
   ``JitExtension``: C++/CUDA/HIP sources built into a torch extension on
   first use, per architecture.
``mstar/kernels/configs/<arch>/[<product>/]<kernel>.json``
   Tuned launch parameters per device, keyed by the op's shape classes.
``mstar/platforms/``
   What the hardware is (``current_platform``, ``Platform.device_facts()``)
   and ``PlatformDispatch``. Nothing else branches on the vendor.
``benchmark/kernels/``
   The harness: specs, candidates, the oracle, captured microbenchmarks, the
   roofline, the results store, and the served-step trace tools.

Declaring an op
---------------

.. code-block:: python

   from mstar.kernels import KernelOp, ShapePolicy, Tolerance

   def _reference(x, weight, eps):        # plain torch: CPU tensors, fallback, oracle
       ...

   def _fake(x, weight, eps):             # shapes and dtypes only
       return torch.empty_like(x)

   def _flashinfer():                     # a factory: imports, returns the kernel
       import flashinfer
       return lambda x, weight, eps: flashinfer.norm.rmsnorm(x, weight, eps=eps)

   MY_NORM = KernelOp(
       "my_norm", "(Tensor x, Tensor weight, float eps) -> Tensor",
       fake=_fake, reference=_reference,
       impls={"cuda": _flashinfer, "rocm": _rocm_norm, "rocm:gfx950": _rocm_norm_gfx950},
       shapes=ShapePolicy(dims=("rows", "hidden"), key=..., inputs=..., warm=(...,)),
       tolerance=Tolerance.rounded_once(),
   )

- The op is defined once through ``torch.library``, so a traced or compiled
  caller sees one opaque node, and its fake serves ``torch.compile``.
- Implementation keys are ``cuda``, ``rocm``, ``xpu``, ``default`` and
  ``<platform>:<arch>``; an architecture's key wins over its platform's. A
  factory runs once per process, on the op's first accelerator call or on
  ``warm``, so declaring an op loads no kernel library.
- Without ``fallback_to_reference``, a platform whose factory fails (or that
  has none) raises when the op is called there, naming the op, the platform
  and the cause; ``op.unavailable_reason()`` says the same at build time,
  which is where a resource should refuse a configuration it cannot serve.
- Nothing is built inside a captured graph. Before it captures anything the
  engine resolves every op (``registry.warm_all()``: imports and JIT builds,
  no kernel runs) and logs one line per op naming what runs
  (``op.selection()``; an implementation may carry a ``kernel_selection``
  string saying what it chose). The shapes a model uses then compile in the
  eager forwards that precede each capture, and ``op.warm(shapes)`` resolves
  and runs an op on chosen shapes explicitly.
- Add the declaring module to ``DECLARING_MODULES`` in the registry; a CPU
  test fails on a module that declares an op without being listed.

Numerics
--------

An implementation is checked against the reference evaluated in fp64 on the
CPU (``mstar.kernels.numerics``):

``rounded_once``
   fp32 math and one round-to-nearest-even per output: every element within
   one ulp of the fp64 result rounded to the output dtype, and the relative
   norm distance from that rounded result under ``rel_norm`` (a truncating
   store fails it).
``accumulated``
   a long fp32 reduction rounded once (GEMMs): elementwise within ``rtol``
   plus ``atol_scale`` times the largest reference magnitude.
``exact``
   bit for bit, for kernels that claim it.

``test/modular/test_kernel_ops_parity.py`` runs every registered op on a GPU
against its reference on its warm shapes and dtypes, and checks a captured
replay against eager; a new op is covered when it is declared.

Building C++ and HIP sources
----------------------------

.. code-block:: python

   from mstar.kernels.hip.build import JitExtension

   EXT = JitExtension("_mstar_my_op", [CSRC / "my_op.cu"], headers=[CSRC / "my_op.cuh"],
                      extra_cuda_cflags=["-O3"])
   EXT.unavailable_reason()   # builds once per process; None when it loaded

CUDA sources are hipified on ROCm. Code objects target the visible device's
architecture, or the ``archs`` named (several make one fat binary, e.g.
``archs=("gfx942", "gfx950")``). The extension is named by a hash of its
sources, flags, architectures and torch build, so checkouts share builds and
a change rebuilds. ``TORCH_EXTENSIONS_DIR`` chooses where builds go.

Tuned configs
-------------

``op.config(shape)`` reads ``configs/<arch>/<product>/<name>.json`` if the
product has one, else ``configs/<arch>/<name>.json``, and returns the entry
for the shape itself (the optional ``shapes`` table, keyed by
``configs.shape_key``: dimensions in name order, ``hidden4096-rows1``), else
for the shape's class (``ShapePolicy.classes``), else the file's ``default``.
Products of one architecture share code objects, not timings. Configs are
written by the harness, and only after a model-level A/B shows the change
helps: ``ResultsStore.promote`` writes one family's winning knobs per class,
and ``ResultsStore.promote_best`` picks an implementation per shape across
families (a kernel of this repo only where it beats the shipped one by a
margin), optionally keeping each class's default so that shapes nobody
searched run what they ran before.

An entry may name the implementation as well as its knobs, for an op that
chooses per shape. On AMD GPUs:

- ``mstar::fused_add_rms_norm`` runs the Triton kernel
  (``mstar/kernels/triton/norm.py``) for entries with ``"impl": "triton"`` and
  FlashInfer's ROCm fork otherwise, including on every product without a
  config.
- Decode buckets of up to 16 rows send each linear where the
  ``decode_gemv`` entry for its ``(rows, n, k)`` says: ``"aiter"`` (AITER's
  skinny GEMM), ``"triton"`` or ``"hip"`` (``mstar::decode_gemv``,
  ``mstar/kernels/{triton,hip}/gemv``) or ``"hipblaslt"`` (left to
  inductor), and a gated MLP's gate/up projection and ``silu(gate) * up``
  become one ``mstar::decode_gemv_silu_mul`` where that op's entry names a
  kernel. Without a config, 1-2 row linears go to AITER and the rest to
  hipBLASLt. The graph passes are in ``mstar/platforms/rocm_gemm.py``; their
  cache key covers the configs, so a changed config recompiles.

Measuring with the harness
--------------------------

.. code-block:: bash

   python -m benchmark.kernels.search --spec rms_norm --list
   python -m benchmark.kernels.search --spec rms_norm --budget 8 --store <dir> --table out.md

A ``KernelSpec`` (``benchmark/kernels/specs/``) names the shapes, inputs,
reference, tolerance, bytes and FLOPs per shape, and the candidates on a
device: the shipped op, torch expressions, and Triton or HIP kernels over the
knob spaces in ``benchmark/kernels/candidates.py``. For each shape the search
drops candidates that miss the bar, times the rest together in captured
graphs with operands rotated past the memory-side cache, and writes a record
per shape to the results store under ``<arch>/<product>/<torch-hip-triton>/``
with each candidate's time, percent of the roofline floor
(``Platform.device_facts()``), oracle verdict and register/LDS statistics.
Stored winners seed the next search. ``benchmark/kernels/search.sbatch`` runs
one search as a self-contained batch job. Times from a shared or virtualized
GPU rank candidates; they are not performance claims.

Shapes come from served steps: ``benchmark/kernels/trace/proftrigger`` opens a
profiler window in a running worker, ``trace/ktrace.py`` turns the trace into
a per-step kernel table, and ``trace/harvest.py`` collects shapes from a model
config, the server log and traces into ``benchmark/kernels/shapes/``.

Checklist
---------

1. Declare the op (or add an implementation key to an existing one) and list
   its module in ``DECLARING_MODULES``.
2. Have the resource that runs it refuse, at build time, what
   ``unavailable_reason()`` says it cannot serve.
3. Run the CPU tests (``test/modular/test_kernel_registry.py``) and, on each
   GPU platform touched, ``test/modular/test_kernel_ops_parity.py``.
4. Search it with the harness on the target devices; promote a config only
   after a model-level A/B on an exclusive node, stating the chip.
