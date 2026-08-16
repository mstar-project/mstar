"""Bitwise parity of the LingBot fast UniPC path vs the wan22 reference.

The precomputed-constant solver (``components/unipc_fast.py``) must produce
``torch.equal`` outputs against ``mstar.model.wan22.components.unipc`` at every
step — it is a restructuring, not a reimplementation. Runs the full loop on
random DiT-output stand-ins at several (steps, shape) configs.

GPU-only (the serving path runs the solver on device); skipped without CUDA.
"""

import pytest
import torch

from mstar.model.lingbot.components.unipc_fast import verify_against_reference

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@requires_cuda
@pytest.mark.parametrize(
    ("num_steps", "shape"),
    [
        (30, (16, 9, 24, 40)),  # 33 frames @ 320x192
        (20, (16, 3, 24, 40)),  # 9 frames @ 320x192
        (40, (16, 13, 60, 104)),  # 49 frames @ 832x480
    ],
)
def test_unipc_fast_bitwise_parity(num_steps, shape):
    verify_against_reference(num_steps, 3.0, shape)
