"""CUDA IPC KV transfer: a read is one copy per page and token range over its
layers, not one per (layer, page), and lands exactly the pages and tokens it
names, read from another process's cache."""

import os
import subprocess
import sys

import pytest
import torch

from mstar.engine.resources.kv import page_copier
from mstar.engine.resources.kv.cache import KVCache
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.transfer import KVReadInfo, _contiguous_runs, _layers_by_page


def _config(**kwargs):
    defaults = dict(num_layers=3, num_kv_heads=2, head_dim=4, max_seq_len=32, max_num_pages=4, page_size=8)
    return PagedKVConfig(**{**defaults, **kwargs})


def test_reads_group_by_page_and_token_range():
    reads = [
        KVReadInfo(layer_idx=layer, local_page_idx=lp, remote_page_idx=rp, token_start=0, token_end=end)
        for lp, rp, end in ((2, 0, 8), (3, 1, 5))
        for layer in (2, 0, 1)
    ]

    assert _layers_by_page(reads) == {(2, 0, 0, 8): [2, 0, 1], (3, 1, 0, 5): [2, 0, 1]}


@pytest.mark.parametrize("layers, runs", [
    ([2, 0, 1], [(0, 3)]),
    ([0, 2, 3], [(0, 1), (2, 4)]),
    ([1, 1], [(1, 2)]),
])
def test_layers_become_contiguous_runs(layers, runs):
    assert _contiguous_runs(layers) == runs


def test_layer_rows_is_one_view_a_page_or_one_each_for_k_and_v():
    cache = KVCache(_config(), torch.device("cpu"))
    cache.tensor.copy_(torch.arange(cache.tensor.numel()).view_as(cache.tensor))

    [whole] = cache.layer_rows(0, 3, 1, 0, 8)
    k, v = cache.layer_rows(1, 3, 2, 3, 6)

    assert torch.equal(whole, cache.tensor[:, 1])
    assert torch.equal(k, cache.tensor[1:3, 2, 0, 3:6])
    assert torch.equal(v, cache.tensor[1:3, 2, 1, 3:6])
    # each layer's row is contiguous, so a row is one memcpy
    assert all(view[0].is_contiguous() for view in (whole, k, v))


def _copiers():
    if not torch.cuda.is_available():
        return [pytest.param(None, marks=pytest.mark.skip(reason="needs a GPU"))]
    out = []
    for cls in (page_copier.CudaBindingsCopier, page_copier.CudartCopier, page_copier.TorchCopier):
        try:
            out.append(pytest.param(cls(), id=cls.name))
        except (ImportError, OSError, AttributeError, RuntimeError) as e:
            out.append(pytest.param(None, id=cls.name, marks=pytest.mark.skip(reason=str(e))))
    return out


@pytest.mark.parametrize("copier", _copiers())
def test_copy_rows_copies_exactly_the_view(copier):
    src = KVCache(_config(), torch.device("cuda"))
    dst = KVCache(_config(), torch.device("cuda"))
    src.tensor.normal_()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    for d, s in zip(dst.layer_rows(0, 3, 3, 0, 8), src.layer_rows(0, 3, 1, 0, 8), strict=True):
        copier.copy_rows(d, s, stream)
    for d, s in zip(dst.layer_rows(1, 3, 0, 2, 7), src.layer_rows(1, 3, 2, 2, 7), strict=True):
        copier.copy_rows(d, s, stream)
    stream.synchronize()

    expected = torch.zeros_like(dst.tensor)
    expected[:, 3] = src.tensor[:, 1]
    expected[1:3, 0, :, 2:7] = src.tensor[1:3, 2, :, 2:7]
    assert torch.equal(dst.tensor, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA IPC needs a GPU")
def test_a_read_from_another_process_lands_the_named_pages_and_tokens():
    from mstar.engine.resources.kv.transfer import CudaIpcKVTransferEngine

    cache = KVCache(_config(), torch.device("cuda:0"))
    cache.tensor.normal_()
    engine = CudaIpcKVTransferEngine(cache)
    torch.cuda.synchronize()
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"kv_ipc_{os.getpid()}.pt")
    torch.save({"info": engine.get_kv_transfer_info(), "src": cache.tensor.cpu()}, path)
    reader_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    try:
        pythonpath = os.pathsep.join([".", os.environ.get("PYTHONPATH", "")])
        reader = subprocess.run(
            [sys.executable, "-c", _READER, path, reader_device], capture_output=True,
            check=False, timeout=120, env={**os.environ, "PYTHONPATH": pythonpath},
        )
        assert reader.returncode == 0, reader.stderr.decode()[-2000:]
        assert reader.stdout.decode().split()[-1] == "ok"
    finally:
        engine.shutdown()
        os.unlink(path)


# Pages 0 and 1 whole into 2 and 3, then tokens [0, 5) of page 2 into page 0,
# every layer: the shape of a prefill -> decode handoff. Page 1 stays zero.
_READER = """
import sys, torch
from mstar.engine.resources.kv.cache import KVCache
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.transfer import CudaIpcKVTransferEngine, KVReadInfo

blob = torch.load(sys.argv[1], weights_only=False)
device = torch.device(sys.argv[2])
torch.cuda.set_device(device)
config = PagedKVConfig(num_layers=3, num_kv_heads=2, head_dim=4, max_seq_len=32, max_num_pages=4, page_size=8)
cache = KVCache(config, device)
engine = CudaIpcKVTransferEngine(cache)
reads = [
    KVReadInfo(layer_idx=layer, local_page_idx=local, remote_page_idx=remote, token_start=0, token_end=end)
    for local, remote, end in ((2, 0, 8), (3, 1, 8), (0, 2, 5))
    for layer in range(3)
]
engine.read_batched_async(blob["info"], reads).result()
# the Future is the readiness gate: read on another stream, no device sync first
with torch.cuda.stream(torch.cuda.Stream(device)):
    got = cache.tensor.cpu()
src = blob["src"]
expected = torch.zeros_like(src)
expected[:, 2] = src[:, 0]
expected[:, 3] = src[:, 1]
expected[:, 0, :, :5] = src[:, 2, :, :5]
assert torch.equal(got, expected)
engine.shutdown()
print("ok")
"""
