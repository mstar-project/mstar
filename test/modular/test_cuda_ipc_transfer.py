import time
import zmq
import torch
import torch.multiprocessing as mp
from torch.multiprocessing.reductions import rebuild_cuda_tensor
from random import randint

PORT = 5555

NUM_REQUESTS = 5


def sender():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.bind(f"tcp://127.0.0.1:{PORT}")

    device = "cuda:0"

    num_layers = 28
    max_num_pages = 512
    page_size = 128
    num_kv_heads = 32
    head_dim = 128

    kv_cache = torch.zeros(
            num_layers,
            max_num_pages,
            2,
            page_size,
            num_kv_heads,
            head_dim,
            device=device,
            dtype=torch.float32,
        ).contiguous()


    for i in range(NUM_REQUESTS):
        # Pick a slice to set to 1
        target_page = i
        for l in range(num_layers):
            kv_cache[l, target_page] = i + l / num_layers

        storage = kv_cache.untyped_storage()
        cuda_share = storage._share_cuda_()

        meta = {
            "size": kv_cache.size(),
            "stride": kv_cache.stride(),
            "offset": kv_cache.storage_offset(),
            "dtype": str(kv_cache.dtype),
            "requires_grad": kv_cache.requires_grad,
            "num_layers": num_layers,
            "target_page": target_page,
        }

        print(f"[Sender] Sending metadata for iter {i}...")
    
        sock.send_pyobj((cuda_share, meta))

    print("[Sender] Done. Sleeping to keep storage alive...")
    time.sleep(10)  # keep process alive so IPC memory is valid


def receiver():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.connect(f"tcp://127.0.0.1:{PORT}")

    for i in range(NUM_REQUESTS):
        start_total = time.perf_counter()

        cuda_share, meta = sock.recv_pyobj()
        recv_time = time.perf_counter()

        (
            storage_device,
            storage_handle,
            storage_size_bytes,
            storage_offset_bytes,
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
        ) = cuda_share

        # dtype reconstruction
        dtype = getattr(torch, meta["dtype"].split(".")[-1])

        tensor = rebuild_cuda_tensor(
            torch.Tensor,
            meta["size"],
            meta["stride"],
            meta["offset"],
            torch.UntypedStorage,
            dtype,
            storage_device,
            storage_handle,
            storage_size_bytes,
            storage_offset_bytes,
            meta["requires_grad"],
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
        )

        rebuild_time = time.perf_counter()

        # Extract the slice
        total_to_cuda_time = 0
        correct = True
        for l in range(meta["num_layers"]):
            tic = time.perf_counter()
            target = tensor[
               l, meta["target_page"],
            ]

            # Move to cuda:1
            target = target.contiguous().to("cuda:1")
            toc = time.perf_counter()
            total_to_cuda_time += toc - tic
            # Check correctness
            correct &= torch.all(target == i + l / meta["num_layers"]).item()

        print(f"[Receiver] Correct: {correct}")

        print(f"Timing breakdown (iter {i}):")
        print(f"  ZMQ recv: {(recv_time - start_total)*1000:.2f} ms")
        print(f"  Rebuild: {(rebuild_time - recv_time)*1000:.2f} ms")
        print(f"  Transfer to cuda:1: {(total_to_cuda_time)*1000:.2f} ms")
        print(f"  Total time: {(rebuild_time - start_total + total_to_cuda_time)*1000:.2f} ms\n")


if __name__ == "__main__":
    mp.set_start_method("spawn")

    p1 = mp.Process(target=sender)
    p2 = mp.Process(target=receiver)

    p2.start()
    time.sleep(1)  # ensure receiver is ready
    p1.start()

    p1.join()
    p2.join()