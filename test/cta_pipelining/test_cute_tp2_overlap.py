"""Tests for the TP2 MLP with the fc2 reduce-scatter / all-reduce fused into the fc2 epilogue
(``cute_tp2.py``, ``ROLE_REDUCE`` in ``cute_gemm.py``). All need two peer-capable sm_90 GPUs."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass")


def _two_peer_sm90() -> bool:
    if not (torch.cuda.is_available() and torch.cuda.device_count() >= 2):
        return False
    if any(torch.cuda.get_device_capability(i)[0] != 9 for i in (0, 1)):
        return False
    return torch.cuda.can_device_access_peer(0, 1) and torch.cuda.can_device_access_peer(1, 0)


pytestmark = pytest.mark.skipif(not _two_peer_sm90(), reason="needs two peer-capable sm_90 GPUs")

D0, D1 = torch.device("cuda", 0), torch.device("cuda", 1)


def _weights(K, N1, N2, bias, seed=0):
    torch.manual_seed(seed)
    w1 = (torch.randn(N1, K, device=D0) * K**-0.5).to(torch.bfloat16)
    w2 = (torch.randn(N2, N1, device=D0) * N1**-0.5).to(torch.bfloat16)
    b1 = (torch.randn(N1, device=D0) * 0.02).to(torch.bfloat16) if bias else None
    b2 = (torch.randn(N2, device=D0) * 0.02).to(torch.bfloat16) if bias else None
    return w1, b1, w2, b2


def _sync():
    torch.cuda.synchronize(D0)
    torch.cuda.synchronize(D1)


def _check(mlp, ys, ref, M):
    """Sharded: GPU 0 holds rows [0, r0), GPU 1 the rest (``split``). Replicated: the full output on
    both GPUs, bit-identical (both reduce-add the same two bf16 partials)."""
    if mlp.output == "sharded":
        (_, n0), _ = mlp.split(M)
        r0 = min(M, n0 * mlp.tile_m)
        assert (ys[0].shape[0], ys[1].shape[0]) == (r0, M - r0)
        y = torch.cat([ys[0], ys[1].to(D0)])
    else:
        assert ys[0].shape[0] == ys[1].shape[0] == M
        assert torch.equal(ys[0], ys[1].to(D0))
        y = ys[0]
    torch.testing.assert_close(y.float(), ref, atol=5e-2, rtol=5e-2)


def _counters_at_epoch(mlp):
    """Every counter (own rows: the peer's signals; the rest: ``_bump``) at epoch * n_ready."""
    return all(bool((c == mlp._epoch * mlp.n_ready).all()) for c in mlp._counters)


@pytest.mark.parametrize("output", ["sharded", "replicated"])
@pytest.mark.parametrize(
    "shape",
    [
        (100, 256, 512, 512),  # one row block: GPU 1 owns none (fc2 on GPU 1 is range A only)
        (1000, 256, 512, 512),  # ragged M, odd number of row blocks (8 -> 4 + 4; 1000 % 128 != 0)
        (4096, 3072, 14336, 3072),  # Wan2.2 FFN, fc2 cluster (1, 2)
    ],
)
def test_tp2_overlap_matches_reference(shape, output):
    from mstar.utils.cta_pipelining import mlp_reference
    from mstar.utils.cta_pipelining.cute_tp2 import CuteTP2OverlapMLP

    M, K, N1, N2 = shape
    w1, b1, w2, b2 = _weights(K, N1, N2, True)
    x = torch.randn(M, K, device=D0, dtype=torch.bfloat16)
    ref = mlp_reference(x, w1, b1, w2, b2, "gelu_tanh").float()
    mlp = CuteTP2OverlapMLP(w1, b1, w2, b2, devices=(D0, D1), output=output)
    for _ in range(2):  # second call: epoch 2, reused H / counters
        ys = mlp(x, x.to(D1))
        _sync()
        _check(mlp, ys, ref, M)
        assert _counters_at_epoch(mlp)


@pytest.mark.parametrize("output", ["sharded", "replicated"])
def test_tp2_overlap_varying_tokens(output):
    """One object, varying M (the row-block split, and so which counters the peer signals and which
    ``_bump`` advances, changes between forwards), growth past the reservation (reallocation restarts
    the epoch), the int32 guard reset; no bias, silu."""
    from mstar.utils.cta_pipelining import mlp_reference
    from mstar.utils.cta_pipelining.cute_tp2 import CuteTP2OverlapMLP

    w1, _, w2, _ = _weights(256, 1024, 512, False, seed=1)
    mlp = CuteTP2OverlapMLP(w1, None, w2, None, devices=(D0, D1), activation="silu", max_tokens=1100,
                            output=output)
    for i, M in enumerate((1100, 300, 1000, 129, 1100, 2500)):
        if i == 4:
            mlp._epoch = (1 << 30) // mlp.n_ready  # force the overflow guard (sync + zero)
        x = torch.randn(M, 256, device=D0, dtype=torch.bfloat16)
        ys = mlp(x, x.to(D1))
        _sync()
        _check(mlp, ys, mlp_reference(x, w1, None, w2, None, "silu").float(), M)
        assert _counters_at_epoch(mlp)


@pytest.mark.parametrize("output", ["sharded", "replicated"])
def test_tp2_overlap_back_to_back(output):
    """Forwards queued without host syncs, alternating M: ``_bump`` must only write counters no
    signal targets (a read-modify-write of the whole array loses in-flight peer ``red.add`` updates
    and a later forward spins forever), and each GPU's fc2 must not signal the peer before the
    peer's previous forward is done. Checks the last forward and the counter invariant."""
    from mstar.utils.cta_pipelining import mlp_reference
    from mstar.utils.cta_pipelining.cute_tp2 import CuteTP2OverlapMLP

    w1, b1, w2, b2 = _weights(256, 512, 512, True, seed=2)
    mlp = CuteTP2OverlapMLP(w1, b1, w2, b2, devices=(D0, D1), max_tokens=1100, output=output)
    xs = {M: torch.randn(M, 256, device=D0, dtype=torch.bfloat16) for M in (1000, 1100)}
    x1s = {M: x.to(D1) for M, x in xs.items()}
    mlp(xs[1000], x1s[1000])  # compile + module load outside the queued run
    _sync()
    for i in range(24):
        M = (1000, 1100)[i % 2]
        ys = mlp(xs[M], x1s[M])
    _sync()
    _check(mlp, ys, mlp_reference(xs[M], w1, b1, w2, b2, "gelu_tanh").float(), M)
    assert _counters_at_epoch(mlp)


def test_tp2_overlap_shares_compiled_ops():
    """One compiled op per GEMM, launched on both devices (see ``CuteTP2OverlapMLP.__init__``: a
    per-device fc2 op deadlocked on the DSL's module load behind the peer's spinning fc2)."""
    from mstar.utils.cta_pipelining.cute_gemm import ROLE_PLAIN, ROLE_REDUCE
    from mstar.utils.cta_pipelining.cute_tp2 import CuteTP2OverlapMLP

    w1, b1, w2, b2 = _weights(256, 512, 512, True, seed=3)
    mlp = CuteTP2OverlapMLP(w1, b1, w2, b2, devices=(D0, D1))
    assert mlp.fc1.role == ROLE_PLAIN and mlp.fc2.role == ROLE_REDUCE
    assert mlp.fc2.cluster_size == 2  # N2 = 2 n-tiles: cluster (1, 2)
    x = torch.randn(300, 256, device=D0, dtype=torch.bfloat16)
    mlp(x, x.to(D1))
    _sync()
    assert mlp.fc2._compiled is not None and mlp.fc2._compiled_key == (torch.bfloat16,)
