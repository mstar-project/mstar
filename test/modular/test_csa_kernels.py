"""The CSA kernels compile for Hopper as their launchers launch them, on any host.

Triton type-checks a kernel, and sizes its shared memory, when a GPU first launches it. Here the
real launcher runs over a recorder and Triton's compiler builds every launch it made as that GPU
would. What the kernels compute is held to the torch path on a GPU
(``test/deepseek_v41/gpu_csa.py --mode kernels --only scores,attend``).
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

pytest.importorskip("triton.backends.nvidia")

from mstar.engine.resources.csa import kernels  # noqa: E402
from mstar.engine.resources.csa.config import CSAConfig, CSASpec, Fp4Rows  # noqa: E402
from mstar.testing_utils.triton_compile import compile_launches, record_launches  # noqa: E402

KEYS = Fp4Rows(block=32)


# index heads a rank holds x key width: the checkpoint on one rank and on eight (four heads, in
# the dot's least tile of 16), the reduced model on one rank and on two
@pytest.mark.parametrize("heads, dim", [(32, 128), (4, 128), (16, 32), (8, 64)])
def test_decode_scores_compiles_for_hopper_within_its_shared_memory(monkeypatch, heads, dim):
    launches = record_launches(monkeypatch, kernels, "_decode_scores_kernel")
    rows, width = 3, 4096
    q = torch.zeros(rows, heads, dim, dtype=torch.bfloat16)
    w = torch.zeros(rows, heads, dtype=torch.bfloat16)
    # a key plane as its cache declares it: the row's FP4 bytes in bf16 elements
    plane = torch.zeros(4, 16, -(-KEYS.nbytes(dim) // 2), dtype=torch.bfloat16)
    index = torch.zeros(rows, dtype=torch.long)
    out = kernels.decode_scores(q, w, plane, torch.zeros(9, dtype=torch.long), index, index, index, width)
    assert out.shape == (rows, width) and out.dtype == torch.float32
    arguments = launches.arguments()
    # a launch reads the plane as bytes, a whole key row apart, and pads the heads to the dot's least
    assert arguments["plane"].dtype == torch.uint8 and arguments["ROW"] == plane.shape[-1] * 2 >= KEYS.nbytes(dim)
    assert (arguments["NH"], arguments["NHP"]) == (heads, max(16, heads))
    assert compile_launches(launches) == 1


# heads a rank holds x latent width x entries a row: the checkpoint on one, two, four and eight
# ranks (eight heads in the dot's least tile of 16), the reduced model
@pytest.mark.parametrize("heads, dim, entries", [
    (64, 512, 640), (32, 512, 640), (16, 512, 640), (8, 512, 640), (4, 32, 64)])
def test_attend_compiles_for_hopper_within_its_shared_memory(monkeypatch, heads, dim, entries):
    launches = record_launches(monkeypatch, kernels, "_attend_kernel")
    rows = 3
    out = kernels.attend(
        torch.zeros(rows, heads, dim, dtype=torch.bfloat16), torch.zeros(rows, entries, dim, dtype=torch.bfloat16),
        torch.zeros(rows, entries, dtype=torch.bool), torch.zeros(heads), dim ** -0.5)
    assert out.shape == (rows, heads, dim) and out.dtype == torch.bfloat16
    arguments = launches.arguments()
    # the sums start at zero, a row of them per padded head; the validity mask goes in as bytes
    assert arguments["acc"].shape == (rows, max(16, heads), dim) and not arguments["acc"].any()
    assert arguments["valid"].dtype == torch.uint8 and arguments["BD"] == min(128, dim)
    assert compile_launches(launches) == 1


def test_a_deployment_turns_each_kernel_on_or_off():
    spec = CSASpec(resource_key="csa", nodes={"n"}, config=CSAConfig(
        caches={1: "kv"}, topk=8, sm_scale=1.0, index_head_dim=32, latent_dim=32))
    # the score kernel is the torch path's arithmetic and runs unasked; the softmax kernel is not
    assert (spec.config.score_kernel, spec.config.attend_kernel) == (True, False)
    spec.apply_yaml_overrides(score_kernel=False, attend_kernel=True)
    assert (spec.config.score_kernel, spec.config.attend_kernel) == (False, True)
    with pytest.raises(ValueError, match="attend_kernel must be true or false"):
        spec.apply_yaml_overrides(attend_kernel="yes")
