#!/usr/bin/env python3
"""Is CUDA-graph sampling batch-invariant — and if not, whose fault?

No engine, no CUDA graphs, no loops — just the kernels. We fix one "reference"
row (identical probs, seed, offset), drop it into batches of different size and
at different positions (padding the rest with random distractor rows), and check
its sampled token never changes.

Two paths are checked:
  - ``sample_cuda_graphable_gpu`` — mstar's path (fused softmax → flashinfer).
  - raw ``flashinfer...top_k_top_p_sampling_from_probs`` on **fixed pre-computed
    probs**, skipping the fused kernel entirely.

If the raw-flashinfer path is also batch-variant, the position-dependence is
flashinfer's own RNG (it folds the batch row index into philox) — nothing mstar
does can beat it, and batched decoding cannot be bit-reproducible.

    python test/sampling_test/flashinfer_batch_test.py

Requires a GPU + flashinfer.
"""

import torch

from mstar.engine.resources.sampler.utils import sample_cuda_graphable_gpu

DEV = "cuda"
V = 2048
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
TOP_K, TOP_P, TEMP = 50, 1.0, 1.0


def _mstar_sample(logits_rows, seeds, offsets):
    """mstar path: fused_temperature_softmax → flashinfer, on raw logits."""
    bs = logits_rows.shape[0]
    return sample_cuda_graphable_gpu(
        logits_rows,
        temperature=torch.full((bs,), TEMP, device=DEV),
        top_k=torch.full((bs,), TOP_K, dtype=torch.int32, device=DEV),
        top_p=torch.full((bs,), TOP_P, device=DEV),
        seed=torch.as_tensor(seeds, dtype=torch.long, device=DEV),
        offset=torch.as_tensor(offsets, dtype=torch.long, device=DEV),
    )


def _raw_flashinfer_sample(probs_rows, seeds, offsets):
    """Raw flashinfer on already-softmaxed probs — no fused kernel involved."""
    import flashinfer
    bs = probs_rows.shape[0]
    out = flashinfer.sampling.top_k_top_p_sampling_from_probs(
        probs_rows,
        torch.full((bs,), TOP_K, dtype=torch.int32, device=DEV),
        torch.full((bs,), TOP_P, device=DEV),
        deterministic=True,
        seed=torch.as_tensor(seeds, dtype=torch.long, device=DEV),
        offset=torch.as_tensor(offsets, dtype=torch.long, device=DEV),
    )
    out = out[0] if isinstance(out, tuple) else out
    return out.to(torch.int64)


def _logits_row(gen):
    return torch.randn(V, device=DEV, dtype=torch.bfloat16, generator=gen)


def _probs_row(gen):
    return torch.softmax(
        torch.randn(V, device=DEV, dtype=torch.float32, generator=gen), dim=-1
    )


def _batch_invariance(name, sample_fn, make_row) -> int:
    """Reference row at every (bs, position) must sample identically to bs=1."""
    gen = torch.Generator(device=DEV).manual_seed(0)
    ref = make_row(gen)
    ref_seed, ref_offset = 12345, 7
    base = sample_fn(ref.unsqueeze(0), [ref_seed], [ref_offset])[0].item()

    bad = []
    for bs in BATCH_SIZES:
        for pos in range(bs):
            rows = torch.stack([make_row(gen) for _ in range(bs)])
            rows[pos] = ref  # identical row for the reference position
            seeds = [1000 * i + 1 for i in range(bs)]
            offsets = [i + 100 for i in range(bs)]
            seeds[pos], offsets[pos] = ref_seed, ref_offset  # identical rng
            tok = sample_fn(rows, seeds, offsets)[pos].item()
            if tok != base:
                bad.append((bs, pos, tok))

    if bad:
        print(f"[{name}] BATCH-VARIANT: ref token={base}, differs at "
              f"(bs,pos,tok): {bad[:12]}{' …' if len(bad) > 12 else ''}")
        return 1
    print(f"[{name}] batch-invariant: ref token={base} at every (bs,pos) ✓")
    return 0


def test_offset_determinism() -> int:
    """Same row+seed+offset twice must match; distinct offsets should explore."""
    gen = torch.Generator(device=DEV).manual_seed(1)
    ref = _logits_row(gen)
    toks = {}
    for off in range(32):
        a = _mstar_sample(ref.unsqueeze(0), [999], [off])[0].item()
        b = _mstar_sample(ref.unsqueeze(0), [999], [off])[0].item()
        if a != b:
            print(f"NON-DETERMINISTIC: offset={off} gave {a} then {b}")
            return 1
        toks[off] = a
    print(f"offset-deterministic ✓ ({len(set(toks.values()))}/32 distinct)")
    return 0


def main() -> int:
    if not torch.cuda.is_available():
        print("needs CUDA")
        return 2
    rc = 0
    rc |= _batch_invariance("mstar sample_cuda_graphable_gpu", _mstar_sample, _logits_row)
    rc |= _batch_invariance("raw flashinfer from_probs", _raw_flashinfer_sample, _probs_row)
    rc |= test_offset_determinism()
    print("\nPASS" if rc == 0 else "\nFAIL")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
