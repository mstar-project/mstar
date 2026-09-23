"""Per-request tensors that slice one storage are cloned/copied once per storage and handed
back as views in the original order; unrelated or non-contiguous tensors keep their own op."""
import torch

from mstar.utils.coalesce import apply_coalesced, clone_coalesced, storage_spans


def test_slices_of_one_tensor_form_one_span():
    base = torch.arange(64, dtype=torch.long)
    parts = [base[i:i + 1] for i in range(64)]
    spans = storage_spans(parts)
    assert len(spans) == 1 and spans[0].view.numel() == 64
    clones = clone_coalesced(parts)
    assert all(torch.equal(c, p) for c, p in zip(clones, parts, strict=True))
    assert all(c.shape == (1,) for c in clones)
    # the clones no longer alias the base
    base.add_(100)
    assert clones[3].item() == 3 and parts[3].item() == 103
    # one result buffer behind all of them
    assert len({c.untyped_storage().data_ptr() for c in clones}) == 1


def test_mixed_groups_and_orders():
    a = torch.arange(10, dtype=torch.int32)
    b = torch.arange(6, dtype=torch.float32)
    lone = torch.tensor([7, 8, 9])
    strided = torch.arange(12).view(3, 4)[:, 1]  # not contiguous
    tensors = [a[4:6], b[1:3], a[0:2], lone, strided, b[5:6]]
    spans = storage_spans(tensors)
    assert len(spans) == 4  # a-group, b-group, lone, strided
    out = apply_coalesced(tensors, lambda v: v.clone())
    for o, t in zip(out, tensors, strict=True):
        assert torch.equal(o, t) and o.shape == t.shape
    # a span covers only what its members touch: a[0:6] -> 6 elements, b[1:6] -> 5
    sizes = sorted(s.view.numel() for s in spans)
    assert sizes == [3, 3, 5, 6]


def test_span_op_result_is_sliced_per_member():
    base = torch.arange(8, dtype=torch.long)
    parts = [base[2:4], base[6:8], base[0:1]]
    calls = []

    def op(v):
        calls.append(v.numel())
        return v * 10

    out = apply_coalesced(parts, op)
    assert calls == [8] and [o.tolist() for o in out] == [[20, 30], [60, 70], [0]]


def test_tiled_members_are_split_once_and_gapped_ones_are_sliced():
    # rows of one batch result, in order: the fast path (one split call), 1-D and 2-D shapes
    rows = torch.arange(64 * 4, dtype=torch.long).view(64, 4)
    parts = [rows[i:i + 1] for i in range(64)]
    clones = clone_coalesced(parts)
    assert all(c.shape == (1, 4) and torch.equal(c, p) for c, p in zip(clones, parts, strict=True))
    flat = [rows.view(-1)[i:i + 1] for i in range(256)]
    clones = clone_coalesced(flat)
    assert all(c.shape == (1,) and torch.equal(c, p) for c, p in zip(clones, flat, strict=True))
    # out of order or with gaps: the span is still one op, members are sliced individually
    gapped = [rows[5:6], rows[2:3], rows[9:11]]
    spans = storage_spans(gapped)
    assert len(spans) == 1 and spans[0].view.numel() == 9 * 4
    clones = clone_coalesced(gapped)
    assert all(c.shape == p.shape and torch.equal(c, p) for c, p in zip(clones, gapped, strict=True))
    assert len({c.untyped_storage().data_ptr() for c in clones}) == 1
