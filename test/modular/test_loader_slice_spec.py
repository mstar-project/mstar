"""The loader iterators' ``slice_spec`` path: sliced safetensors reads.

The generic loader has every rank read the full checkpoint and keep its
slice (8x the bytes at TP8). ``slice_spec`` lets a TP-aware caller ask the
iterator for ``get_slice(key)[..., start:stop]`` on one dim instead of the
full tensor. These tests pin the iterator half of the fast read path
(the model-side read plan lives with the model that builds it).
"""
import json

import torch
from safetensors.torch import save_file

from mstar.model.loader.iterators import (
    TensorSlice,
    iter_safetensors_file,
    iter_safetensors_shards,
)


def _write_sharded(tmp_path, tensors):
    save_file(tensors, str(tmp_path / "model-00001-of-00001.safetensors"))
    index = {"weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))


def test_iterator_slice_spec_reads_exact_shards(tmp_path):
    torch.manual_seed(0)
    a = torch.randn(8, 6)
    b = torch.randn(6, 8)
    c = torch.randn(4)
    _write_sharded(tmp_path, {"a": a, "b": b, "c": c})

    specs = {"a": TensorSlice(0, 2, 5), "b": TensorSlice(1, 4, 8)}
    out = dict(iter_safetensors_shards(tmp_path, keys={"a", "b", "c"},
                                       slice_spec=specs.get))
    assert torch.equal(out["a"], a[2:5])
    assert torch.equal(out["b"], b[:, 4:8])
    assert torch.equal(out["c"], c)  # no spec -> full read


def test_single_file_slice_spec_and_default_full_read(tmp_path):
    torch.manual_seed(1)
    w = torch.randn(4, 10)
    path = tmp_path / "model.safetensors"
    save_file({"w": w}, str(path))

    sliced = dict(iter_safetensors_file(path, slice_spec={"w": TensorSlice(1, 3, 7)}.get))
    assert torch.equal(sliced["w"], w[:, 3:7])
    # slice_spec omitted -> the pre-existing full-tensor path, unchanged.
    full = dict(iter_safetensors_file(path))
    assert torch.equal(full["w"], w)


def test_slice_spec_slices_any_dim(tmp_path):
    torch.manual_seed(2)
    v = torch.randn(6)
    t = torch.randn(2, 3, 4)
    _write_sharded(tmp_path, {"v": v, "t": t})

    specs = {"v": TensorSlice(0, 1, 4), "t": TensorSlice(2, 1, 3)}
    out = dict(iter_safetensors_shards(tmp_path, slice_spec=specs.get))
    assert torch.equal(out["v"], v[1:4])
    assert torch.equal(out["t"], t[:, :, 1:3])
