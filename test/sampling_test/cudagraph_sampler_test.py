#!/usr/bin/env python3
"""Direct test of the CUDA-graph sampler machinery — no engine, no model.

Manually drives ``SamplerBuffers`` the way the sampler resource does:
  register per-request configs → gather (per-request params/seed/offset into the
  static per-step buffers, by slot) → replay a captured CUDA graph that samples
  out of those buffers → scatter the advanced offsets and repeat, with a real
  (bs>1) batch.

The reference is eager ``sample_tokens`` fed the *same* gathered buffer values,
so the check is purely "does the captured graph sample identically to eager
given identical (params, seed, offset)". The batch layout is FIXED across the
loop, so flashinfer's position-dependent RNG (see flashinfer_batch_test.py) is
held constant and can't confound the comparison — this isolates the gather +
graph-replay + per-request offset-advance logic that the sampling branch owns.

    python test/sampling_test/cudagraph_sampler_test.py

Requires a GPU + flashinfer.
"""

import torch

from mstar.engine.resources.sampler.utils import (
    SamplerBuffers,
    SamplingConfig,
    sample_tokens,
)

DEV = torch.device("cuda:0")
V = 4096
BS, ITERS = 4, 8
SLOT = 0


def _eager(logits, sampler):
    """sample_tokens fed the exact gathered buffer values (no penalty here)."""
    return sample_tokens(
        logits=logits,
        temperature=sampler.temperature_buf,
        top_k=sampler.top_k_buf,
        top_p=sampler.top_p_buf,
        seed=sampler.seed_buf,
        rand_offset=sampler.offset_buf,
    ).to(torch.int64)


def main() -> int:
    if not torch.cuda.is_available():
        print("needs CUDA")
        return 2

    bufs = SamplerBuffers.allocate(max_batch_size=BS, device=DEV)

    # Distinct per-request configs + seeds (so a slot/gather mixup would show).
    rids = [f"r{i}" for i in range(BS)]
    for i, rid in enumerate(rids):
        cfg = SamplingConfig(temperature=0.7 + 0.1 * i, top_k=40 + i, top_p=1.0)
        cfg.set_seed(1000 + i)          # distinct per-request seed
        bufs.register_request(rid, cfg)

    gen = torch.Generator(device=DEV).manual_seed(0)
    logits = torch.randn(BS, V, device=DEV, dtype=torch.bfloat16, generator=gen)

    # The sampler holds zero-copy views of the slot's buffers, so one instance
    # stays valid across every gather / capture / replay below.
    sampler = bufs.sampler_for(BS, SLOT)

    # ---- warm up (autotune the fused/flashinfer kernels) then capture -------
    for _ in range(3):
        bufs.gather_static(rids, BS, SLOT)
        bufs.gather_dynamic(rids, BS, SLOT)
        sampler.sample(rids, logits)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graphed = sampler.sample(rids, logits)

    # ---- loop: gather (refreshes offset) → replay → compare to eager -------
    mismatches = 0
    for it in range(ITERS):
        bufs.gather_static(rids, BS, SLOT)
        bufs.gather_dynamic(rids, BS, SLOT)  # updates the static buffers
        # eager first: `sample` advances offset_buf in place, and the graph's
        # copy of that increment must not be double-counted in the reference
        expected = _eager(logits, sampler)
        graph.replay()
        bufs.scatter_offset(SLOT)  # persist the in-graph-advanced offsets
        torch.cuda.synchronize()
        if not torch.equal(graphed, expected):
            mismatches += 1
            print(f"iter {it}: mismatch graphed={graphed.tolist()} "
                  f"eager={expected.tolist()}")

    # Sanity: distinct per-request seeds/configs should spread the batch's
    # tokens (else the gather is collapsing everyone onto one slot's config).
    print(f"row spread (distinct tokens across {BS} requests): "
          f"{len(set(graphed.tolist()))}")

    if mismatches:
        print(f"\nFAIL: {mismatches} graphed-vs-eager mismatches")
        return 1
    print(f"\nPASS: graphed sampler == eager over {ITERS} iters at bs={BS} "
          f"(gather + graph replay + per-request offset all correct)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
