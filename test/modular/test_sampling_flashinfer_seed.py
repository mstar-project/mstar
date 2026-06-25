"""Pin the FlashInfer seed/offset contract the Talker's graph sampler relies on.

The Talker samples stochastically (temperature ~0.9, top_k ~50) under CUDA-graph
capture. ``CudaGraphableSampler.sample_with_config`` passes per-request philox
``seed``/``offset`` as captured int tensors and advances ``offset_buf += 1``
in-graph so each replay steps the RNG (see the comments in
``mstar/utils/sampling.py`` about a frozen offset never reaching EOS).

This requires a FlashInfer build whose ``*_sampling_from_probs`` binding accepts
tensor seed/offset under graph capture. FlashInfer 0.6.3 rejected tensors at the
C/TVM-FFI layer ("Mismatched type on argument #7"); the binding was reworked in
0.6.4 (verified: 0.6.3 rejects; 0.6.4 / 0.6.5 / 0.6.7.post3 accept and advance
the RNG under capture), which ``pyproject.toml`` now pins as the floor. These
tests fail fast on a too-old build and pin the two facts the design depends on:

  1. a captured tensor ``offset`` ADVANCES the RNG per replay (stochastic
     sampling actually varies step to step);
  2. passing ``offset=None`` under capture is NOT a valid shortcut -- it makes
     FlashInfer read its default CUDA generator's ``current_seed()``, which is
     illegal during capture. (Pins why the captured path must pass a real
     tensor, not None.)

Skips when CUDA / FlashInfer are unavailable.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FlashInfer sampler requires CUDA"
)


def _flashinfer_or_skip():
    try:
        import flashinfer  # noqa: F401

        return flashinfer
    except Exception:
        pytest.skip("flashinfer not installed; sampler path unavailable")


def _stochastic_probs(batch: int, vocab: int, device: torch.device):
    """A non-degenerate distribution so RNG-stepped samples can actually vary."""
    torch.manual_seed(0)
    return torch.softmax(torch.randn(batch, vocab, device=device) / 0.9, dim=-1)


def test_tensor_seed_offset_accepted_under_capture():
    """Pinned FlashInfer must accept captured tensor seed/offset and advancing
    offset_buf in-graph must step the RNG -- the exact pattern
    CudaGraphableSampler.sample_with_config relies on.

    Fails loudly (not skips) on a too-old build that rejects tensor seed, since
    that build silently breaks stochastic Talker sampling.
    """
    flashinfer = _flashinfer_or_skip()
    dev = torch.device("cuda")
    B, V = 1, 2048
    probs = _stochastic_probs(B, V, dev)
    top_k = torch.tensor([50], device=dev, dtype=torch.int32)
    top_p = torch.tensor([1.0], device=dev)
    seed_buf = torch.zeros(B, device=dev, dtype=torch.long)
    offset_buf = torch.zeros(B, device=dev, dtype=torch.long)

    # Eager warmup so all lazy kernel/workspace init happens outside capture.
    for _ in range(2):
        flashinfer.sampling.top_k_top_p_sampling_from_probs(
            probs, top_k, top_p, deterministic=True, seed=seed_buf, offset=offset_buf
        )
    torch.cuda.synchronize()

    # Capture: sample, then advance the offset buffer in-graph (mirrors the
    # mstar CudaGraphableSampler design). A private pool avoids cross-test
    # graph-memory aliasing in the shared pytest process.
    pool = torch.cuda.graph_pool_handle()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g, pool=pool):
            out = flashinfer.sampling.top_k_top_p_sampling_from_probs(
                probs, top_k, top_p, deterministic=True,
                seed=seed_buf, offset=offset_buf,
            )
            offset_buf += 1
    except Exception as exc:  # too-old FlashInfer rejects the tensor binding
        pytest.fail(
            "Pinned FlashInfer must accept tensor seed/offset under CUDA-graph "
            f"capture, but capture raised: {exc!r}. Check the flashinfer-python "
            "pin in pyproject.toml (>=0.6.4)."
        )

    tokens = []
    for _ in range(10):
        g.replay()
        torch.cuda.synchronize()
        tokens.append(int(out.item()))

    assert int(offset_buf.item()) == 10, "in-graph offset_buf += 1 did not accumulate"
    # The whole point: a stepped offset yields a varying stochastic stream.
    # A frozen offset would return the same token every replay (the bug a
    # scalar/None shortcut would silently reintroduce).
    assert len(set(tokens)) > 1, (
        "captured tensor offset did not advance the RNG -- stochastic sampling "
        f"is frozen (got identical tokens: {tokens[:3]}...)"
    )


def test_none_offset_is_illegal_under_capture():
    """Pins why the captured path must pass a real tensor, not None: with
    offset=None, FlashInfer falls back to its default CUDA generator, whose
    current_seed() read is illegal during graph capture. Guards against anyone
    "simplifying" the sampler to seed=None/offset=None."""
    flashinfer = _flashinfer_or_skip()
    dev = torch.device("cuda")
    B, V = 1, 2048
    probs = _stochastic_probs(B, V, dev)
    top_k = torch.tensor([50], device=dev, dtype=torch.int32)
    top_p = torch.tensor([1.0], device=dev)

    for _ in range(2):
        flashinfer.sampling.top_k_top_p_sampling_from_probs(
            probs, top_k, top_p, deterministic=True, seed=None, offset=None
        )
    torch.cuda.synchronize()

    pool = torch.cuda.graph_pool_handle()
    g = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="(?i)current_seed|graph capture"):
        with torch.cuda.graph(g, pool=pool):
            flashinfer.sampling.top_k_top_p_sampling_from_probs(
                probs, top_k, top_p, deterministic=True, seed=None, offset=None
            )
