"""The loader iterators read tensors with ``pread`` instead of safetensors'
mmap. These tests pin that every yielded tensor — full reads, contiguous
and strided slices, every dtype, several shards — is bit-identical to what
``safe_open`` returns, in the same order.
"""
import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from mstar.model.loader import iterators
from mstar.model.loader.iterators import (
    TensorSlice,
    iter_safetensors_file,
    iter_safetensors_shards,
)


def _reference(paths, keys=None, prefix=None, slice_spec=None):
    """The safe_open reader the iterators used before."""
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if (prefix is not None and not key.startswith(prefix)) or \
                        (keys is not None and key not in keys):
                    continue
                spec = slice_spec(key) if slice_spec is not None else None
                if spec is None:
                    yield key, f.get_tensor(key)
                    continue
                dim, start, stop = spec
                sl = f.get_slice(key)
                index = [slice(None)] * len(sl.get_shape())
                index[dim] = slice(start, stop)
                yield key, sl[tuple(index)]


def _assert_same(got, want):
    assert [k for k, _ in got] == [k for k, _ in want]
    for (key, a), (_, b) in zip(got, want, strict=True):
        assert a.dtype == b.dtype, key
        assert a.shape == b.shape, key
        assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                           b.contiguous().reshape(-1).view(torch.uint8)), key


def _fp8(*shape):
    return (torch.randn(*shape) * 4).to(torch.float8_e4m3fn)


def _checkpoint(tmp_path):
    """Two shards shaped like an fp8 block-scaled MoE checkpoint."""
    torch.manual_seed(0)
    shards = [{}, {}]
    for layer in range(2):
        p = f"model.layers.{layer}.mlp"
        shard = shards[layer]
        for e in range(3):
            for proj, shape in (("gate_proj", (64, 32)), ("up_proj", (64, 32)),
                                ("down_proj", (32, 64))):
                shard[f"{p}.experts.{e}.{proj}.weight"] = _fp8(*shape)
                shard[f"{p}.experts.{e}.{proj}.weight_scale_inv"] = torch.rand(
                    shape[0] // 16, shape[1] // 16)
        shard[f"{p}.shared_experts.down_proj.weight"] = _fp8(32, 48)
        shard[f"{p}.shared_experts.down_proj.weight_scale_inv"] = torch.rand(2, 3)
        shard[f"model.layers.{layer}.input_layernorm.weight"] = torch.randn(32).bfloat16()
        shard[f"model.layers.{layer}.conv.weight"] = torch.randn(24, 1, 4)
    shards[0]["model.embed_tokens.weight"] = torch.randn(50, 32).bfloat16()
    shards[0]["model.flags"] = torch.tensor([True, False, True])
    shards[0]["model.ids"] = torch.arange(7, dtype=torch.int64)
    shards[0]["model.scalar"] = torch.tensor(3.5, dtype=torch.float64)
    shards[0]["model.empty"] = torch.empty(0, 4, dtype=torch.float16)
    shards[1]["lm_head.weight"] = torch.randn(50, 32).bfloat16()
    shards[1]["model.e5m2"] = torch.randn(8, 8).to(torch.float8_e5m2)

    weight_map = {}
    paths = []
    for i, shard in enumerate(shards):
        name = f"model-{i + 1:05d}-of-00002.safetensors"
        save_file(shard, str(tmp_path / name))
        weight_map.update({k: name for k in shard})
        paths.append(tmp_path / name)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}))
    return paths, weight_map


def _tp_spec(rank, size):
    """Row slices for gate/up, column slices for down, per TP rank."""
    def spec(key):
        if ".experts." not in key:
            return None
        scale = key.endswith("_scale_inv")
        if ".down_proj." in key:
            cols = (4 if scale else 64) // size
            return TensorSlice(1, rank * cols, (rank + 1) * cols)
        rows = (4 if scale else 64) // size
        return TensorSlice(0, rank * rows, (rank + 1) * rows)
    return spec


@pytest.mark.parametrize("threads, ahead", [(8, 512 << 20), (3, 1)])
def test_shards_match_safe_open_bit_for_bit(tmp_path, monkeypatch, threads, ahead):
    monkeypatch.setattr(iterators, "_READ_THREADS", threads)
    monkeypatch.setattr(iterators, "_READ_AHEAD_BYTES", ahead)
    paths, weight_map = _checkpoint(tmp_path)

    _assert_same(list(iter_safetensors_shards(tmp_path)), list(_reference(paths)))
    for rank in range(4):
        spec = _tp_spec(rank, 4)
        _assert_same(
            list(iter_safetensors_shards(tmp_path, slice_spec=spec)),
            list(_reference(paths, slice_spec=spec)),
        )
    keys = {k for k in weight_map if "experts.1." in k or k.startswith("lm_head")}
    _assert_same(
        list(iter_safetensors_shards(tmp_path, keys=keys, slice_spec=_tp_spec(1, 2))),
        list(_reference(paths, keys=keys, slice_spec=_tp_spec(1, 2))),
    )
    _assert_same(
        list(iter_safetensors_shards(tmp_path, prefix="model.layers.1.")),
        list(_reference(paths, prefix="model.layers.1.")),
    )


def test_every_slice_dim_matches_safe_open(tmp_path):
    torch.manual_seed(1)
    t = torch.randn(3, 5, 6).bfloat16()
    path = tmp_path / "model.safetensors"
    save_file({"t": t, "row": torch.randn(1, 8, 4)}, str(path))
    # Plain (dim, start, stop) tuples are specs too: read plans build them.
    for spec in (TensorSlice(0, 1, 3), TensorSlice(1, 2, 5), TensorSlice(2, 0, 4),
                 TensorSlice(-1, 1, 6), TensorSlice(1, 3, 3), (1, 0, 2)):
        _assert_same(
            list(iter_safetensors_file(path, slice_spec={"t": spec, "row": spec}.get)),
            list(_reference([path], slice_spec={"t": spec, "row": spec}.get)),
        )


def test_unknown_dtype_falls_back_to_safe_open(tmp_path, monkeypatch):
    path = tmp_path / "model.safetensors"
    w = torch.randn(4, 6).bfloat16()
    save_file({"w": w, "v": torch.randn(3)}, str(path))
    dtypes = dict(iterators._DTYPES)
    del dtypes["BF16"]
    monkeypatch.setattr(iterators, "_DTYPES", dtypes)
    spec = {"w": TensorSlice(1, 2, 5)}.get
    _assert_same(list(iter_safetensors_file(path, slice_spec=spec)),
                 list(_reference([path], slice_spec=spec)))


@pytest.mark.parametrize("ahead_tensors", [1, 256])
def test_every_opened_file_is_closed(tmp_path, monkeypatch, ahead_tensors):
    monkeypatch.setattr(iterators, "_READ_AHEAD_TENSORS", ahead_tensors)
    opened, closed = [], []
    real_open, real_close = iterators.os.open, iterators.os.close
    monkeypatch.setattr(iterators.os, "open", lambda *a: opened.append(real_open(*a)) or opened[-1])
    monkeypatch.setattr(iterators.os, "close", lambda fd: (closed.append(fd), real_close(fd)))
    _checkpoint(tmp_path)

    it = iter_safetensors_shards(tmp_path)
    next(it)
    it.close()  # a consumer that stops early
    assert opened and sorted(opened) == sorted(closed)

    opened.clear()
    closed.clear()
    assert len(list(iter_safetensors_shards(tmp_path))) > 0
    assert len(opened) == 2 and sorted(opened) == sorted(closed)


def test_a_slice_dim_past_the_tensor_raises(tmp_path):
    """As safe_open's get_slice: a column spec that also matched a 1-d bias wrapped onto
    dim 0 and read a plausible-shaped wrong shard."""
    path = tmp_path / "model.safetensors"
    save_file({"bias": torch.randn(8)}, str(path))
    with pytest.raises(IndexError, match="out of range"):
        list(iter_safetensors_file(path, slice_spec={"bias": TensorSlice(1, 0, 4)}.get))


def test_reads_are_split_below_2_gib(tmp_path, monkeypatch):
    # macOS rejects one iovec of 2 GiB or more with EINVAL
    sizes = []
    real = iterators.os.preadv
    monkeypatch.setattr(iterators, "_PREAD_MAX", 16)
    monkeypatch.setattr(iterators.os, "preadv",
                        lambda fd, bufs, off: sizes.append(len(bufs[0])) or real(fd, bufs, off))
    path = tmp_path / "model.safetensors"
    t = torch.arange(40, dtype=torch.int32)
    save_file({"t": t}, str(path))
    (_, got), = list(iter_safetensors_file(path))
    assert torch.equal(got, t) and max(sizes) <= 16
