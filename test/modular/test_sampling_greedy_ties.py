"""Greedy rows on the CUDA-graph sampler path must return the argmax for every
seed, including when the top-1 is an exact tie.

Encoding greedy as ``(temperature=1.0, top_k=1)`` is not greedy: FlashInfer's
top-k is rejection sampling and accepts any token with no strictly-greater
probability, so tied maxima are broken by the per-request philox seed and
identical greedy requests diverge at the first tie (routine in bf16 logits).
``SamplerBuffers`` now keeps ``temperature == 0`` in the device row and the
prep kernel emits a one-hot at argmax on-device (``include_greedy=True``).

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
    except Exception:
        pytest.skip("flashinfer not installed; sampler path unavailable")


def _tied_logits(device: torch.device, vocab: int = 64) -> tuple[torch.Tensor, int]:
    """Two exact maxima; ``torch.argmax`` picks the first, index 3."""
    logits = torch.zeros(1, vocab, device=device)
    logits[0, 3] = 5.0
    logits[0, 41] = 5.0
    assert int(torch.argmax(logits[0])) == 3
    return logits, 3


def _greedy_cfg(seed: int):
    from mstar.engine.resources.sampler.utils import SamplingConfig

    cfg = SamplingConfig(temperature=0.0)
    cfg.set_seed(seed)
    return cfg


def test_greedy_tie_is_argmax_for_every_seed_eager():
    _flashinfer_or_skip()
    from mstar.engine.resources.sampler.utils import SamplerBuffers

    dev = torch.device("cuda")
    rid = "r1"
    SLOT = 0
    logits, argmax = _tied_logits(dev)

    bufs = SamplerBuffers.allocate(max_batch_size=1, device=dev)
    bufs.register_request(rid, _greedy_cfg(0))
    sampler = bufs.sampler_for(1, SLOT)

    seen = set()
    for seed in range(64):
        bufs.update_request_config(rid, _greedy_cfg(seed))
        bufs.gather_static([rid], 1, SLOT)
        bufs.gather_dynamic([rid], 1, SLOT)
        for _ in range(4):  # the in-graph offset advances per sample
            seen.add(int(sampler.sample([rid], logits).item()))
    assert seen == {argmax}, f"greedy broke a top-1 tie by seed: got {sorted(seen)}"


def test_greedy_tie_is_argmax_for_every_seed_under_capture():
    _flashinfer_or_skip()
    from mstar.engine.resources.sampler.utils import SamplerBuffers

    dev = torch.device("cuda")
    rid = "r1"
    SLOT = 0
    logits, argmax = _tied_logits(dev)

    bufs = SamplerBuffers.allocate(max_batch_size=1, device=dev)
    bufs.register_request(rid, _greedy_cfg(0))
    sampler = bufs.sampler_for(1, SLOT)
    bufs.gather_static([rid], 1, SLOT)
    bufs.gather_dynamic([rid], 1, SLOT)
    for _ in range(2):  # warm up triton autotune + flashinfer outside capture
        sampler.sample([rid], logits)
    torch.cuda.synchronize()

    pool = torch.cuda.graph_pool_handle()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=pool):
        out = sampler.sample([rid], logits)

    seen = set()
    for seed in range(64):
        bufs.update_request_config(rid, _greedy_cfg(seed))
        bufs.gather_static([rid], 1, SLOT)
        for _ in range(4):
            g.replay()
            torch.cuda.synchronize()
            seen.add(int(out.item()))
    assert seen == {argmax}, f"greedy broke a top-1 tie by seed: got {sorted(seen)}"
