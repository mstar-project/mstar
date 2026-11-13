"""The offload/reload fingerprint's digest (`tensor_digest`, `TensorDigest`) folds a tensor's words
a chunk at a time, and the KV pool's (`KVManager._pages_digest`) a layer's pages at a time: the same
digest as the one-shot sums over the whole tensor, with bounded scratch."""

import pytest
import torch

from mstar.engine.resources.base import DIGEST_CHUNK, TensorDigest, tensor_digest
from mstar.engine.resources.kv.manager import KVManager


def _one_shot(t: torch.Tensor) -> torch.Tensor:
    """The digest over the whole tensor at once: its words summed plain and weighted by
    ``(i + 1) % 65521 + 1``, in wrapping int64."""
    flat = t.detach().reshape(-1).contiguous()
    words = flat.view(torch.int32) if flat.numel() * flat.element_size() % 4 == 0 else flat.view(torch.uint8)
    words = words.to(torch.int64)
    weights = torch.arange(1, words.numel() + 1, dtype=torch.int64) % 65521 + 1
    return torch.stack([words.sum(), (words * weights).sum()])


def _tensors():
    g = torch.Generator().manual_seed(0)
    return [
        torch.randn(3, 70001, generator=g).to(torch.bfloat16),
        torch.randn(5, 123, generator=g),
        # past 65521 words, so the weights wrap; and words large enough that the sums do
        torch.randint(-2**31, 2**31 - 1, (200_003,), generator=g, dtype=torch.int32),
        torch.randint(-2**62, 2**62, (4097,), generator=g, dtype=torch.int64),
        # bytes that aren't whole words: digested as bytes
        torch.randint(0, 255, (1001,), generator=g, dtype=torch.uint8),
    ]


@pytest.mark.parametrize("chunk", [1, 7, 65521, 65522, 100_000, DIGEST_CHUNK])
def test_the_digest_is_the_one_shot_digest_whatever_the_chunk(chunk):
    for t in _tensors():
        assert torch.equal(tensor_digest(t, chunk), _one_shot(t)), (t.dtype, tuple(t.shape), chunk)


def test_pieces_added_in_order_digest_as_their_concatenation():
    t = _tensors()[1]  # rows of whole 32-bit words
    digest = TensorDigest(t.device, chunk=100)
    for piece in t:
        digest.add(piece)
    assert torch.equal(digest.value(), _one_shot(t))
    with pytest.raises(ValueError, match="whole 32-bit words"):
        TensorDigest(t.device).add(torch.zeros(3, dtype=torch.uint8))


def test_a_pools_pages_digest_a_layer_at_a_time_as_they_did_all_at_once():
    """An NHD pool ``[layers, pages, 2, page_size, heads, head_dim]``, some of its pages, out of order."""
    g = torch.Generator().manual_seed(1)
    storage = torch.randn(6, 9, 2, 4, 2, 8, generator=g).to(torch.bfloat16)
    pages = torch.tensor([7, 2, 5, 3])
    assert torch.equal(KVManager._pages_digest(storage, pages), _one_shot(storage.index_select(1, pages)))
    # one byte a page, a layer's pages not whole words: the whole copy, as before
    odd, some = torch.randint(0, 255, (3, 9, 1), generator=g, dtype=torch.uint8), torch.tensor([1, 4, 8])
    assert torch.equal(KVManager._pages_digest(odd, some), _one_shot(odd.index_select(1, some)))


def test_a_fingerprint_holds_one_layers_pages_and_one_chunks_scratch(monkeypatch):
    """The largest copy is one layer's pages (not every layer's), and the largest int64 scratch
    is one chunk (not the whole tensor's words)."""
    layers, num_pages = 6, 4
    storage = torch.randn(layers, 9, 2, 4, 2, 8).to(torch.bfloat16)
    pages = torch.tensor([7, 2, 5, 3])
    seen = {"index_select": 0, "arange": 0}
    index_select, arange = torch.Tensor.index_select, torch.arange

    def counted_index_select(self, *args, **kwargs):
        out = index_select(self, *args, **kwargs)
        seen["index_select"] = max(seen["index_select"], out.numel())
        return out

    def counted_arange(*args, **kwargs):
        out = arange(*args, **kwargs)
        seen["arange"] = max(seen["arange"], out.numel())
        return out

    monkeypatch.setattr(torch.Tensor, "index_select", counted_index_select)
    monkeypatch.setattr(torch, "arange", counted_arange)
    monkeypatch.setattr(TensorDigest.__init__, "__defaults__", (64,))
    KVManager._pages_digest(storage, pages)

    one_layer = num_pages * storage[0, 0].numel()
    assert seen["index_select"] == one_layer, seen
    assert seen["arange"] == 64, seen
