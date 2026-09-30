"""Tests for the CUTLASS (CuTe DSL) CTA-pipelined MLP (``cute_gemm.py`` /
``cute_mlp.py``). All need an sm_90 GPU; the two-GPU tests need two
peer-capable devices."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass")


def _sm90(idx: int = 0) -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(idx)[0] == 9


def _two_peer_sm90() -> bool:
    if not (torch.cuda.is_available() and torch.cuda.device_count() >= 2 and _sm90(0) and _sm90(1)):
        return False
    return torch.cuda.can_device_access_peer(0, 1) and torch.cuda.can_device_access_peer(1, 0)


def _weights(K, N1, N2, bias, device, seed=0):
    torch.manual_seed(seed)
    w1 = (torch.randn(N1, K, device=device) * K**-0.5).to(torch.bfloat16)
    w2 = (torch.randn(N2, N1, device=device) * N1**-0.5).to(torch.bfloat16)
    b1 = (torch.randn(N1, device=device) * 0.02).to(torch.bfloat16) if bias else None
    b2 = (torch.randn(N2, device=device) * 0.02).to(torch.bfloat16) if bias else None
    return w1, b1, w2, b2


@pytest.mark.skipif(not _sm90(), reason="needs an sm_90 GPU")
@pytest.mark.parametrize("role", ["plain", "consumer"])
def test_gemm_cluster_1x2_same_output(role):
    """Cluster (1, 2) with A multicast: bit-identical to (1, 1), M not a multiple of the tile."""
    from mstar.utils.cta_pipelining.cute_gemm import ROLE_CONSUMER, ROLE_PLAIN, CuteGemmOp

    d0 = torch.device("cuda", 0)
    M, K, N = 1000, 512, 1024
    w, b, _, _ = _weights(K, N, N, True, d0, seed=6)
    x = torch.randn(M, K, device=d0, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream(d0)
    r = ROLE_CONSUMER if role == "consumer" else ROLE_PLAIN
    counters = torch.ones(8, dtype=torch.int32, device=d0) if role == "consumer" else None
    outs = []
    for cl in ((1, 1), (1, 2)):
        out = torch.empty(M, N, device=d0, dtype=torch.bfloat16)
        CuteGemmOp(r, "gelu_tanh", True, cluster_shape_mn=cl).launch(
            x, w, b, out, counters, n_ready=1, device=d0, stream=stream
        )
        outs.append(out)
    torch.cuda.synchronize(d0)
    assert torch.equal(outs[0], outs[1])


@pytest.mark.skipif(not _sm90(), reason="needs an sm_90 GPU")
@pytest.mark.parametrize("activation,bias", [("gelu_tanh", True), ("silu", False), ("none", True)])
@pytest.mark.parametrize("M", [1000, 4096])
def test_plain_gemm_matches_cublas(activation, bias, M):
    from mstar.utils.cta_pipelining import mlp_reference
    from mstar.utils.cta_pipelining.cute_mlp import CutePlainMLP

    d0 = torch.device("cuda", 0)
    K, N1, N2 = 512, 1024, 512
    w1, b1, w2, b2 = _weights(K, N1, N2, bias, d0)
    x = torch.randn(M, K, device=d0, dtype=torch.bfloat16)
    ref = mlp_reference(x, w1, b1, w2, b2, activation).float()
    y = CutePlainMLP(w1, b1, w2, b2, activation=activation)(x)
    torch.cuda.synchronize(d0)
    torch.testing.assert_close(y.float(), ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not _sm90(), reason="needs an sm_90 GPU")
@pytest.mark.parametrize("raster_group", [3, 32])
def test_plain_gemm_raster_group_same_output(raster_group):
    """Grouped tile order: bit-identical to row-block-major. 8 row blocks give groups of
    3, 3, 2 (short last group) or one short group of 8."""
    from mstar.utils.cta_pipelining.cute_gemm import ROLE_PLAIN, CuteGemmOp

    d0 = torch.device("cuda", 0)
    M, K, N = 1000, 512, 1024
    w, b, _, _ = _weights(K, N, N, True, d0, seed=4)
    x = torch.randn(M, K, device=d0, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream(d0)
    outs = []
    for g in (1, raster_group):
        out = torch.empty(M, N, device=d0, dtype=torch.bfloat16)
        CuteGemmOp(ROLE_PLAIN, "gelu_tanh", True, raster_group=g).launch(x, w, b, out, None, device=d0, stream=stream)
        outs.append(out)
    torch.cuda.synchronize(d0)
    assert torch.equal(outs[0], outs[1])


@pytest.mark.skipif(not _two_peer_sm90(), reason="needs two peer-capable sm_90 GPUs")
@pytest.mark.parametrize("shape", [(1000, 256, 512, 256), (4096, 3072, 14336, 3072)])
@pytest.mark.parametrize("output_on", ["producer", "consumer"])
def test_two_gpu_matches_reference(shape, output_on):
    from mstar.utils.cta_pipelining import mlp_reference
    from mstar.utils.cta_pipelining.cute_mlp import CuteCTAPipelinedMLP

    d0, d1 = torch.device("cuda", 0), torch.device("cuda", 1)
    M, K, N1, N2 = shape
    w1, b1, w2, b2 = _weights(K, N1, N2, True, d0)
    x = torch.randn(M, K, device=d0, dtype=torch.bfloat16)
    ref = mlp_reference(x, w1, b1, w2, b2, "gelu_tanh").float()
    mlp = CuteCTAPipelinedMLP(
        w1, b1, w2, b2, producer_device=d0, consumer_device=d1,
        output_device=d0 if output_on == "producer" else d1,
    )
    for _ in range(2):  # second call exercises counter reset / buffer reuse
        y = mlp(x)
        torch.cuda.synchronize(d0)
        torch.cuda.synchronize(d1)
        torch.testing.assert_close(y.float().to(d0), ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not _two_peer_sm90(), reason="needs two peer-capable sm_90 GPUs")
def test_two_gpu_growing_tokens():
    from mstar.utils.cta_pipelining import mlp_reference
    from mstar.utils.cta_pipelining.cute_mlp import CuteCTAPipelinedMLP

    d0, d1 = torch.device("cuda", 0), torch.device("cuda", 1)
    w1, _, w2, _ = _weights(256, 1024, 256, False, d0, seed=1)
    mlp = CuteCTAPipelinedMLP(w1, None, w2, None, producer_device=d0, consumer_device=d1, activation="silu")
    assert mlp.N1p == mlp.N1 and mlp.local is None  # shape not in DEFAULT_PRODUCER_SHARE: unsplit
    for M in (300, 2048):
        x = torch.randn(M, 256, device=d0, dtype=torch.bfloat16)
        y = mlp(x)
        torch.cuda.synchronize(d0)
        torch.cuda.synchronize(d1)
        ref = mlp_reference(x, w1, None, w2, None, "silu").float()
        torch.testing.assert_close(y.float(), ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not _two_peer_sm90(), reason="needs two peer-capable sm_90 GPUs")
@pytest.mark.parametrize("share,tiles", [(1.0, 11), (0.75, 8), (0.5, 6), (0.62, 7)])
def test_two_gpu_producer_share(share, tiles):
    """Column-split GEMM 1: the producer writes H[:, :N1p] (a column slice, row stride N1),
    the consumer GPU computes H[:, N1p:] itself; x copied in the forward or given on GPU 1."""
    from mstar.utils.cta_pipelining import mlp_reference
    from mstar.utils.cta_pipelining.cute_mlp import CuteCTAPipelinedMLP

    d0, d1 = torch.device("cuda", 0), torch.device("cuda", 1)
    K, N1, N2 = 768, 2816, 512  # 11 n-tiles of 256 in GEMM 1
    w1, b1, w2, b2 = _weights(K, N1, N2, True, d0, seed=3)
    mlp = CuteCTAPipelinedMLP(w1, b1, w2, b2, producer_device=d0, consumer_device=d1, producer_share=share)
    assert mlp.N1p == tiles * 256 and mlp.n_ready == tiles
    assert mlp.consumer.cluster_size == 2  # N2 = 2 n-tiles: fc2 cluster (1, 2)
    for M in (1000, 4096):
        x = torch.randn(M, K, device=d0, dtype=torch.bfloat16)
        ref = mlp_reference(x, w1, b1, w2, b2, "gelu_tanh").float()
        for x_consumer in (None, x.to(d1)):
            y = mlp(x) if x_consumer is None else mlp(x, x_consumer)
            torch.cuda.synchronize(d0)
            torch.cuda.synchronize(d1)
            torch.testing.assert_close(y.float(), ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not _two_peer_sm90(), reason="needs two peer-capable sm_90 GPUs")
@pytest.mark.parametrize("share", [1.0, 0.75])
def test_two_gpu_epoch_counters_varying_tokens(share):
    """Epoch counters: growing then shrinking M inside one reservation (row blocks
    past M must still advance an epoch), the int32 guard reset, then growth past
    the reservation (reallocation restarts the epoch); also with a column-split GEMM 1."""
    from mstar.utils.cta_pipelining import mlp_reference
    from mstar.utils.cta_pipelining.cute_mlp import CuteCTAPipelinedMLP

    d0, d1 = torch.device("cuda", 0), torch.device("cuda", 1)
    w1, b1, w2, b2 = _weights(256, 1024, 256, True, d0, seed=2)
    mlp = CuteCTAPipelinedMLP(
        w1, b1, w2, b2, producer_device=d0, consumer_device=d1, activation="gelu_tanh", max_tokens=2048,
        producer_share=share,
    )
    for i, M in enumerate((1000, 2048, 300, 1500, 3000)):
        if i == 3:
            mlp._epoch = (1 << 30) // mlp.n_ready  # force the overflow guard (sync + zero)
        x = torch.randn(M, 256, device=d0, dtype=torch.bfloat16)
        y = mlp(x)
        torch.cuda.synchronize(d0)
        torch.cuda.synchronize(d1)
        ref = mlp_reference(x, w1, b1, w2, b2, "gelu_tanh").float()
        torch.testing.assert_close(y.float(), ref, atol=5e-2, rtol=5e-2)



@pytest.mark.skipif(not _two_peer_sm90(), reason="needs two peer-capable sm_90 GPUs")
@pytest.mark.parametrize(
    "shape,x_replicated,output_on",
    [((3072, 14336, 3072), False, "producer"), ((3072, 14336, 3072), True, "consumer"),
     ((256, 1024, 256), False, "producer")],
)
def test_two_gpu_cuda_graph(shape, x_replicated, output_on):
    """use_graph=True: one graph per M, three calls per M with different x (the static output is
    checked before the next call overwrites it), back to the first M's graph, then eager again.
    Wan shape: the graph-mode default share for max_tokens=4096 (40/56) applies; (256, 1024, 256): unsplit."""
    from mstar.utils.cta_pipelining import mlp_reference
    from mstar.utils.cta_pipelining.cute_mlp import CuteCTAPipelinedMLP

    d0, d1 = torch.device("cuda", 0), torch.device("cuda", 1)
    K, N1, N2 = shape
    w1, b1, w2, b2 = _weights(K, N1, N2, True, d0, seed=5)
    mlp = CuteCTAPipelinedMLP(
        w1, b1, w2, b2, producer_device=d0, consumer_device=d1, max_tokens=4096, use_graph=True,
        output_device=d0 if output_on == "producer" else d1,
    )
    assert mlp.N1p == (40 * 256 if N1 == 14336 else N1)
    for M in (1000, 4096, 1000):
        for _ in range(3):
            x = torch.randn(M, K, device=d0, dtype=torch.bfloat16)
            y = mlp(x, x.to(d1)) if x_replicated else mlp(x)
            ref = mlp_reference(x, w1, b1, w2, b2, "gelu_tanh").float()
            torch.testing.assert_close(y.float().to(d0), ref, atol=5e-2, rtol=5e-2)
    assert len(mlp._graphs) == 2
    mlp.use_graph = False  # eager after replays: the epoch counters restart
    y = mlp(x)
    torch.testing.assert_close(y.float().to(d0), ref, atol=5e-2, rtol=5e-2)

