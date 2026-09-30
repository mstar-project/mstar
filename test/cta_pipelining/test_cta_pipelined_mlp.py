"""Tests for the CTA-pipelined two-layer MLP (mstar.utils.cta_pipelining).

Two layers of checking:

* ``test_interpreter_protocol`` runs the producer and consumer kernels on CPU
  under ``TRITON_INTERPRET=1`` (in a subprocess, because the interpreter flag
  is read when triton is imported). It validates the tiling, activation, bias,
  row-major tile mapping and the counter protocol (every row block reaches
  ``n_ready``), with no GPU. The system fence is compiled out there.
* ``test_two_gpu_*`` need two peer-capable CUDA devices and compare the real
  concurrent two-GPU pipeline against cuBLAS.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch


def _two_peer_gpus() -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return False
    return torch.cuda.can_device_access_peer(0, 1) and torch.cuda.can_device_access_peer(1, 0)


def run_interpreter_check() -> None:
    """Producer then consumer, sequentially, on CPU tensors under the Triton
    interpreter. Sequential order is required: the consumer's spin-wait only
    terminates because the producer already ran."""
    assert os.environ.get("TRITON_INTERPRET") == "1"
    import triton

    from mstar.utils.cta_pipelining import launch_consumer, launch_producer, mlp_reference
    from mstar.utils.cta_pipelining.kernels import ACT_GELU_TANH

    torch.manual_seed(0)
    M, K, N1, N2 = 80, 64, 96, 48  # M, N1 deliberately not tile multiples
    bm, bn, bk = 32, 32, 32
    x = torch.randn(M, K)
    w1, b1 = torch.randn(N1, K) * K**-0.5, torch.randn(N1) * 0.1
    w2, b2 = torch.randn(N2, N1) * N1**-0.5, torch.randn(N2) * 0.1
    h = torch.zeros(M, N1)
    counters = torch.zeros(triton.cdiv(M, bm), dtype=torch.int32)
    y = torch.zeros(M, N2)

    launch_producer(x, w1, b1, h, counters, act=ACT_GELU_TANH, fence=False, block_m=bm, block_n=bn, block_k=bk,
                    num_warps=4, num_stages=1)
    n_ready = triton.cdiv(N1, bn)
    assert counters.tolist() == [n_ready] * counters.numel(), counters.tolist()
    launch_consumer(h, w2, b2, y, counters, M=M, n_ready=n_ready, fence=False, block_m=bm, block_n=bn, block_k=bk,
                    num_warps=4, num_stages=1)

    h_ref = torch.nn.functional.gelu(torch.nn.functional.linear(x, w1, b1), approximate="tanh")
    torch.testing.assert_close(h, h_ref, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(y, mlp_reference(x, w1, b1, w2, b2, "gelu_tanh"), atol=2e-3, rtol=2e-3)
    print("interpreter check OK")


def test_interpreter_protocol():
    pytest.importorskip("triton")
    env = {**os.environ, "TRITON_INTERPRET": "1"}
    proc = subprocess.run(
        [sys.executable, __file__, "--interpret"], env=env, capture_output=True, text=True, timeout=600, check=False
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "interpreter check OK" in proc.stdout


@pytest.mark.skipif(not _two_peer_gpus(), reason="needs two peer-capable CUDA devices")
@pytest.mark.parametrize("shape", [(1000, 256, 512, 256), (4096, 3072, 14336, 3072)])
@pytest.mark.parametrize("output_on", ["producer", "consumer"])
def test_two_gpu_matches_reference(shape, output_on):
    from mstar.utils.cta_pipelining import CTAPipelinedMLP, mlp_reference, verify_peer_kernel_access

    d0, d1 = torch.device("cuda", 0), torch.device("cuda", 1)
    verify_peer_kernel_access(d0, d1)
    verify_peer_kernel_access(d1, d0)
    M, K, N1, N2 = shape
    torch.manual_seed(0)
    x = torch.randn(M, K, device=d0, dtype=torch.bfloat16)
    w1 = (torch.randn(N1, K, device=d0) * K**-0.5).to(torch.bfloat16)
    b1 = (torch.randn(N1, device=d0) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(N2, N1, device=d0) * N1**-0.5).to(torch.bfloat16)
    b2 = (torch.randn(N2, device=d0) * 0.02).to(torch.bfloat16)
    ref = mlp_reference(x, w1, b1, w2, b2, "gelu_tanh").float()

    mlp = CTAPipelinedMLP(
        w1, b1, w2, b2, producer_device=d0, consumer_device=d1,
        output_device=d0 if output_on == "producer" else d1,
    )
    # Two forwards exercise the counter reset / buffer reuse path.
    for _ in range(2):
        y = mlp(x)
        assert y.device == (d0 if output_on == "producer" else d1)
        torch.cuda.synchronize(d0)
        torch.cuda.synchronize(d1)
        torch.testing.assert_close(y.float().to(d0), ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not _two_peer_gpus(), reason="needs two peer-capable CUDA devices")
def test_two_gpu_growing_tokens():
    """Buffer growth between calls (small M, then larger M) keeps results right."""
    from mstar.utils.cta_pipelining import CTAPipelinedMLP, mlp_reference

    d0, d1 = torch.device("cuda", 0), torch.device("cuda", 1)
    K, N1, N2 = 256, 1024, 256
    torch.manual_seed(1)
    w1 = (torch.randn(N1, K, device=d0) * K**-0.5).to(torch.bfloat16)
    w2 = (torch.randn(N2, N1, device=d0) * N1**-0.5).to(torch.bfloat16)
    mlp = CTAPipelinedMLP(w1, None, w2, None, producer_device=d0, consumer_device=d1, activation="silu")
    for M in (300, 2048):
        x = torch.randn(M, K, device=d0, dtype=torch.bfloat16)
        y = mlp(x)
        torch.cuda.synchronize(d0)
        torch.cuda.synchronize(d1)
        ref = mlp_reference(x, w1, None, w2, None, "silu").float()
        torch.testing.assert_close(y.float(), ref, atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    if "--interpret" in sys.argv:
        run_interpreter_check()
