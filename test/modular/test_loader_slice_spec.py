"""The loader iterators' ``slice_spec`` path: sliced safetensors reads."""
import json

import pytest
import torch
from safetensors.torch import save_file

from mstar.model.loader.iterators import (
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

    specs = {"a": (0, 2, 5), "b": (1, 4, 8)}
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

    sliced = dict(iter_safetensors_file(path, slice_spec={"w": (1, 3, 7)}.get))
    assert torch.equal(sliced["w"], w[:, 3:7])
    # slice_spec omitted -> the pre-existing full-tensor path, unchanged.
    full = dict(iter_safetensors_file(path))
    assert torch.equal(full["w"], w)


def test_slice_spec_rejects_unsupported_dim(tmp_path):
    save_file({"w": torch.zeros(2, 3, 4)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="dim=2"):
        list(iter_safetensors_shards(tmp_path, slice_spec=lambda _k: (2, 0, 1)))
