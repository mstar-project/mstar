"""Kernel harness: measure kernel candidates against their op's reference and
the roofline floor, per device, and keep the results.

One search (``python -m benchmark.kernels.search --spec <name>``) takes a
``KernelSpec`` (``spec``; the built-in ones are in ``specs/``) and, for each
shape and dtype, builds its candidates (the shipped registry op, torch,
Triton and HIP variants over the knob spaces in ``candidates``), drops those
that miss the op's tolerance against the fp64 reference, times the rest in
captured graphs with operands rotated past the device's caches (``bench``),
and writes one JSON record per shape to the results store (``store``), keyed
by architecture, product and the torch/HIP/Triton stack, with roofline
floors (``roofline``) and register/LDS statistics (``isa``). ``trace/`` holds
the served-step tools: a profiler window in a running worker
(``proftrigger``), the per-step kernel table (``ktrace``, ``kseq``,
``kwindow``) and the shape harvest that feeds a spec (``harvest``).

Numbers from a shared or virtualized GPU rank candidates; they are not
claims. A winner reaches ``mstar/kernels/configs`` only after a model-level
A/B. See docs/kernels.rst.
"""
