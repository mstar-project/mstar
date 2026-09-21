"""GPU: the inverse of Marlin's repack (``marlin/layout.py``) recovers every weight bit for bit: an expert's
MXFP4 codes and scales, repacked with the backend's own helpers, dequantized through the position tables,
equal ``dequant_mxfp4`` of the original tensors. Both expert shapes of a latent MoE at TP8-like widths."""
import pytest
import torch

from mstar.model.kimi_k3.reference.mxfp4 import dequant_mxfp4

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = torch.device("cuda")


@pytest.mark.parametrize("n,k", [(512, 512), (512, 256), (768, 3584), (3584, 384)])
def test_marlin_layout_tables_invert_the_repack(n, k):
    from mstar.utils.fused_moe.marlin import _load_ops, prepare_scales, repack_experts
    from mstar.utils.fused_moe.marlin.layout import dequant_marlin_reference

    ops = _load_ops()
    torch.manual_seed(n + k)
    e = 2
    packed = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, device=DEV)
    scale = torch.randint(100, 140, (e, n, k // 32), dtype=torch.uint8, device=DEV)
    w_marlin = repack_experts(ops, packed, n, k)
    s_marlin = prepare_scales(scale, n, k)
    for i in range(e):
        want = dequant_mxfp4(packed[i], scale[i])
        got = dequant_marlin_reference(w_marlin[i], s_marlin[i], n, k)
        assert got.shape == want.shape and torch.equal(got, want), (got.float() - want.float()).abs().max()


@pytest.mark.parametrize("n,k", [(512, 512), (768, 3584), (3584, 384)])
def test_dequant_kernel_matches_the_reference_bit_for_bit(n, k):
    from mstar.utils.fused_moe.marlin import _load_ops, prepare_scales, repack_experts
    from mstar.utils.fused_moe.marlin.dequant import dequant_marlin_experts

    ops = _load_ops()
    torch.manual_seed(7 * n + k)
    e = 3
    packed = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, device=DEV)
    scale = torch.randint(100, 140, (e, n, k // 32), dtype=torch.uint8, device=DEV)
    w_marlin = repack_experts(ops, packed, n, k)
    s_marlin = prepare_scales(scale, n, k)
    got = dequant_marlin_experts(w_marlin, s_marlin, n, k)
    want = torch.stack([dequant_mxfp4(packed[i], scale[i]) for i in range(e)])
    assert torch.equal(got, want), (got.float() - want.float()).abs().max()


def test_dequant_kernel_speed_at_the_layer_shapes():
    """A layer's experts at the TP8 per-rank shapes (224 experts): the two dequantizations should take
    about a millisecond together (1.85 GB written); printed, asserted loosely."""
    import time

    from mstar.utils.fused_moe.marlin import _load_ops, prepare_scales, repack_experts
    from mstar.utils.fused_moe.marlin.dequant import dequant_marlin_experts

    ops = _load_ops()
    e, latent, inter = 224, 3584, 384
    tensors = []
    for n, k in ((2 * inter, latent), (latent, inter)):
        packed = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, device=DEV)
        scale = torch.randint(118, 128, (e, n, k // 32), dtype=torch.uint8, device=DEV)
        tensors.append((repack_experts(ops, packed, n, k), prepare_scales(scale, n, k), n, k))
    outs = [torch.empty(e, n, k, dtype=torch.bfloat16, device=DEV) for _, _, n, k in tensors]
    def run():
        for (w, s, n, k), o in zip(tensors, outs, strict=True):
            dequant_marlin_experts(w, s, n, k, out=o)
    run()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        run()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 10 * 1e3
    print(f"\ndequant of a layer's experts: {ms:.2f} ms for {sum(o.numel() * 2 for o in outs) / 1e9:.2f} GB")
    assert ms < 5.0
