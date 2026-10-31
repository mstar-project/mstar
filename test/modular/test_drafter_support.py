"""Engine pieces a drafter in the forward needs beyond the taps: the attention plan that hides a
block's trailing rows (``hide_tail_views``, ``AttentionStep.hide_last``) and the CPU RoPE
kernel the CPU drafter tests rotate with. A drafter through them (DSpark) is test_dspark.py's."""

from __future__ import annotations

import pytest
import torch

from mstar.engine.resources.kv.plan import SequenceView, build_paged_indptrs, hide_tail_views
from mstar.engine.resources.position.rope import rope_apply_qk_inplace


def test_hiding_a_rows_tail_shortens_its_keys_only():
    page = 4
    views = [
        SequenceView("a", "main", [5, 6, 7], length=10, to_compute=3),
        # its last kept row is the first of its third page: hiding it drops that page
        SequenceView("b", "main", [1, 2, 3], length=9, to_compute=3),
    ]
    out = build_paged_indptrs(hide_tail_views(views, 1, page), page)
    assert out.qo_indptr.tolist() == [0, 3, 6]
    assert out.kv_lens.tolist() == [9, 8]
    assert out.paged_kv_indices.tolist() == [5, 6, 7, 1, 2]
    with pytest.raises(ValueError, match="hide_last"):
        hide_tail_views([SequenceView("c", "main", [1], length=3, to_compute=1)], 2, page)


@pytest.mark.parametrize("interleave", [False, True])
def test_the_cpu_rope_kernel_rotates_as_the_gpu_kernel_does(interleave):
    gen = torch.Generator().manual_seed(0)
    q, k = torch.randn(6, 4, 128, generator=gen), torch.randn(6, 2, 128, generator=gen)
    pos = torch.tensor([0, 1, 7, 100, 3, 257])
    args = (None, None, interleave, 1.0, 1e6)
    qc, kc = q.clone(), k.clone()
    rope_apply_qk_inplace(qc, kc, pos, *args)
    # rotating position p by its angles, as complex numbers
    half = 64
    inv = 1e6 ** (-torch.arange(0, 128, 2, dtype=torch.float64) / 128)
    turn = torch.polar(torch.ones(6, half, dtype=torch.float64), pos[:, None].double() * inv)[:, None]
    for x, got in ((q, qc), (k, kc)):
        x = x.double()
        pairs = x.view(*x.shape[:2], half, 2) if interleave else torch.stack((x[..., :half], x[..., half:]), -1)
        rot = torch.view_as_real(torch.view_as_complex(pairs.contiguous()) * turn)
        want = rot.flatten(-2) if interleave else torch.cat((rot[..., 0], rot[..., 1]), -1)
        torch.testing.assert_close(got.double(), want, atol=2e-4, rtol=0)
    if torch.cuda.is_available():
        # FlashInfer's kernel takes 16-bit q and k
        qb, kb = q.bfloat16(), k.bfloat16()
        qg, kg = qb.cuda(), kb.cuda()
        rope_apply_qk_inplace(qb, kb, pos, *args)
        rope_apply_qk_inplace(qg, kg, pos.cuda(), *args)
        torch.testing.assert_close(qg.cpu(), qb, atol=3e-2, rtol=0)
        torch.testing.assert_close(kg.cpu(), kb, atol=3e-2, rtol=0)
