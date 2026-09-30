"""CTA-pipelining (arXiv:2607.07862) for a two-layer MLP across two GPUs.

Two concurrently resident GEMM kernels, one per GPU, coupled through
peer-memory tile writes and per-row-block release/acquire counters instead of
tensor parallelism's all-reduce. Two kernel back ends:

* Triton (``kernels.py`` / ``mlp.py``, exported here), portable, simple.
* CUTLASS CuTe DSL Hopper warp-specialised persistent WGMMA GEMM
  (``cute_gemm.py`` / ``cute_mlp.py``); import those modules directly, they
  need the ``nvidia-cutlass-dsl`` package and an sm_90 GPU.

See README.md in this directory.
"""

from mstar.utils.cta_pipelining.kernels import (
    ACTIVATIONS,
    cta_pipe_consumer_kernel,
    cta_pipe_producer_kernel,
    launch_consumer,
    launch_producer,
)
from mstar.utils.cta_pipelining.mlp import (
    CTAPipelinedMLP,
    ensure_peer_access,
    mlp_reference,
    verify_peer_kernel_access,
)

__all__ = [
    "ACTIVATIONS",
    "CTAPipelinedMLP",
    "cta_pipe_consumer_kernel",
    "cta_pipe_producer_kernel",
    "ensure_peer_access",
    "launch_consumer",
    "launch_producer",
    "mlp_reference",
    "verify_peer_kernel_access",
]
