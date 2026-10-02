"""A KV retrieve future covers the device copies, not just their submission."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.kv.cache import KVCache, KVLayout
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.manager import CacheStream, KVManager
from mstar.engine.resources.kv.transfer import (
    CudaIpcKVTransferEngine,
    CudaIpcKVTransferInfo,
    KVReadInfo,
)


def test_cuda_ipc_retrieve_waits_for_all_device_copies(monkeypatch):
    """Hold device completion after copy_ returns, as a CUDA copy does.

    A consumer can use a different stream (including a speculative plan), so
    it must not observe a completed retrieve while these writes are pending.
    No CUDA allocation is needed to force this ordering.
    """
    submitted = Event()
    complete = Event()
    copies = []
    visible = []
    device = torch.device("cuda:2")

    class Completion:
        def synchronize(self):
            if copies:
                assert complete.wait(timeout=5)
                visible.extend(copies)

    stream = SimpleNamespace(record_event=Completion)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: stream)
    remote = object()
    monkeypatch.setattr(
        "mstar.engine.resources.kv.transfer.rebuild_cuda_tensor",
        lambda *args: remote,
    )

    def chunk_view(*, layer_idx, page_idx, token_start, token_end, tensor=None):
        source = (layer_idx, page_idx, token_start, token_end)
        if tensor is remote:
            return source

        def copy_(value):
            copies.append((source, value))
            if len(copies) == 2:
                submitted.set()

        return SimpleNamespace(copy_=copy_)

    engine = CudaIpcKVTransferEngine.__new__(CudaIpcKVTransferEngine)
    engine._device = device
    engine._kv_cache = SimpleNamespace(
        layout=KVLayout.NHD, chunk_view=chunk_view,
    )
    engine._executor = ThreadPoolExecutor(max_workers=1)
    engine._pending = []
    descriptor = CudaIpcKVTransferInfo(
        cuda_share=(0, b"handle", 64, 0, b"counter", 0, b"event", False),
        size=(1,), stride=(1,), offset=0, dtype="torch.float32",
        requires_grad=False, layout=KVLayout.NHD,
    )
    reads = [
        KVReadInfo(layer_idx=i, local_page_idx=3, remote_page_idx=7,
                   token_start=1, token_end=4)
        for i in range(2)
    ]
    try:
        future = engine.read_batched_async(descriptor, reads)
        manager = KVManager.__new__(KVManager)
        manager._streams = {
            0: {"main": CacheStream(read_pending=True, read_future=future)},
        }
        assert submitted.wait(timeout=5)
        with pytest.raises(TimeoutError):
            future.result(timeout=0.05)
        assert manager._check_ready(0, "main") == (False, None)
        assert visible == []
        complete.set()
        future.result(timeout=5)
        assert manager._check_ready(0, "main") == (True, None)
        assert visible == [
            ((i, 3, 1, 4), (i, 7, 1, 4)) for i in range(2)
        ]
    finally:
        complete.set()
        engine.shutdown()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="CUDA IPC device visibility requires two CUDA devices",
)
def test_cuda_ipc_retrieve_is_visible_on_an_independent_consumer_stream():
    """A plan stream must see the copied prefix as soon as the future resolves."""
    cfg = PagedKVConfig(
        num_layers=2, num_kv_heads=1, head_dim=4,
        max_num_pages=8, page_size=8, max_seq_len=64,
    )
    source = KVCache(cfg, torch.device("cuda:0"), torch.float32)
    destination = KVCache(cfg, torch.device("cuda:1"), torch.float32)
    source.tensor.fill_(42)
    torch.cuda.synchronize(source.device)
    torch.cuda.synchronize(destination.device)
    producer = CudaIpcKVTransferEngine(source)
    reader = CudaIpcKVTransferEngine(destination, max_workers=1)
    copy_stream = torch.cuda.Stream(device=destination.device)
    consumer_stream = torch.cuda.Stream(device=destination.device)
    do_read = reader._do_read

    def delayed_read(*args):
        with torch.cuda.stream(copy_stream):
            # Widen the enqueue/completion window without holding the GIL.
            torch.cuda._sleep(500_000_000)
            return do_read(*args)

    reader._do_read = delayed_read
    try:
        with torch.cuda.device(destination.device):
            future = reader.read_batched_async(
                producer.get_kv_transfer_info(),
                [
                    KVReadInfo(
                        layer_idx=i, local_page_idx=3, remote_page_idx=1,
                        token_start=0, token_end=4,
                    )
                    for i in range(2)
                ],
            )
            future.result(timeout=10)
            with torch.cuda.stream(consumer_stream):
                observed = destination.tensor[:, 3, :, :4].clone()
            consumer_stream.synchronize()
            assert torch.all(observed == 42).item()
    finally:
        copy_stream.synchronize()
        reader.shutdown()
        producer.shutdown()
