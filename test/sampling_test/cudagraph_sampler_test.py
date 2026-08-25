#!/usr/bin/env python3
"""Direct test of the CUDA-graph sampler machinery — no engine, no model.

Manually drives ``MultiSamplerBuffers`` the way the engine does:
  register per-request configs → gather (per-request params/seed/offset into the
  static per-step buffers, by slot) → replay a captured CUDA graph that samples
  main + aux out of those buffers → advance and repeat in a loop, with a real
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
    MultiSamplerBuffers,
    MultiSamplingConfig,
    SamplingConfig,
    sample_tokens,
)

DEV = torch.device("cuda:0")
V, AUX_V = 4096, 2048
BS, ITERS, NUM_AUX = 4, 8, 15
AUX = "aux"


def _eager(logits, cg, offset_delta):
    """sample_tokens fed the exact gathered buffer values (no penalty here)."""
    off = cg.offset_buf + offset_delta
    return sample_tokens(
        logits=logits,
        temperature=cg.temperature_buf,
        top_k=cg.top_k_buf,
        top_p=cg.top_p_buf,
        seed=cg.seed_buf,
        rand_offset=off,
    ).to(torch.int64)


def main() -> int:
    if not torch.cuda.is_available():
        print("needs CUDA")
        return 2

    # Distinct per-request configs + seeds (so a slot/gather mixup would show).
    main_cfgs, aux_cfgs = {}, {}
    default = MultiSamplingConfig(
        main=SamplingConfig(temperature=0.9, top_k=50, top_p=1.0),
        aux={AUX: SamplingConfig(temperature=1.0, top_k=50, top_p=0.8)},
    )
    bufs = MultiSamplerBuffers.allocate(max_batch_size=BS, device=DEV, config=default)

    rids = [f"r{i}" for i in range(BS)]
    for i, rid in enumerate(rids):
        cfg = MultiSamplingConfig(
            main=SamplingConfig(temperature=0.7 + 0.1 * i, top_k=40 + i, top_p=1.0),
            aux={AUX: SamplingConfig(temperature=0.9, top_k=32 + i, top_p=0.8)},
        )
        cfg.set_seed(1000 + i)          # distinct per-request seed
        bufs.register_request(rid, cfg)

    gen = torch.Generator(device=DEV).manual_seed(0)
    main_logits = torch.randn(BS, V, device=DEV, dtype=torch.bfloat16, generator=gen)
    aux_logits = torch.randn(BS, AUX_V, device=DEV, dtype=torch.bfloat16, generator=gen)

    # ---- warm up (autotune the fused/flashinfer kernels) then capture -------
    for _ in range(3):
        s = bufs.gather_for_request_ids(rids, BS)
        s.sample(rids, main_logits)
        for _ in range(NUM_AUX):
            s.sample_aux(AUX, rids, aux_logits)
    torch.cuda.synchronize()

    cap = bufs.gather_for_request_ids(rids, BS)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_main = cap.sample(rids, main_logits)
        g_aux = torch.stack(
            [cap.sample_aux(AUX, rids, aux_logits) for _ in range(NUM_AUX)], dim=1
        )

    # ---- loop: gather (advances offset) → replay → compare to eager ---------
    mismatches = 0
    for it in range(ITERS):
        cg = bufs.gather_for_request_ids(rids, BS)  # updates the static buffers
        e_main = _eager(main_logits, cg.main, offset_delta=0)
        e_aux = torch.stack(
            [_eager(aux_logits, cg.aux[AUX], offset_delta=g) for g in range(NUM_AUX)],
            dim=1,
        )
        graph.replay()
        bufs.scatter_offsets()  # persist the in-graph-advanced offsets to master
        torch.cuda.synchronize()
        if not torch.equal(g_main, e_main):
            mismatches += 1
            print(f"iter {it}: MAIN mismatch graphed={g_main.tolist()} "
                  f"eager={e_main.tolist()}")
        if not torch.equal(g_aux, e_aux):
            mismatches += 1
            print(f"iter {it}: AUX mismatch")

    # Sanity: distinct per-request seeds/configs should spread the batch's
    # tokens (else the gather is collapsing everyone onto one slot's config).
    print(f"row spread (distinct main tokens across {BS} requests): "
          f"{len(set(g_main.tolist()))}")

    if mismatches:
        print(f"\nFAIL: {mismatches} graphed-vs-eager mismatches")
        return 1
    print(f"\nPASS: graphed sampler == eager over {ITERS} iters at bs={BS} "
          f"(gather + graph replay + per-request offset all correct)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
