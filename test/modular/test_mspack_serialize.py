import os
import time
import uuid
import statistics
from typing import Any
import torch
from multiprocessing import shared_memory as _shm


# ========================
# Your SHM helpers (unchanged)
# ========================
def shm_write_bytes(payload: bytes, name: str | None = None) -> dict[str, Any]:
    try:
        shm = _shm.SharedMemory(create=True, size=len(payload), name=name)
    except FileExistsError:
        if name:
            try:
                existing = _shm.SharedMemory(name=name)
                existing.unlink()
            except Exception:
                pass
            shm = _shm.SharedMemory(create=True, size=len(payload), name=name)
        else:
            raise

    mv = memoryview(shm.buf)
    mv[: len(payload)] = payload
    del mv

    meta = {"name": shm.name, "size": len(payload)}
    shm.close()
    return meta


def shm_read_bytes(meta: dict[str, Any]) -> bytes:
    shm = _shm.SharedMemory(name=meta["name"])
    mv = memoryview(shm.buf)
    data = bytes(mv[: meta["size"]])
    del mv

    shm.close()
    shm.unlink()
    return data


# ========================
# File helpers
# ========================
BASE_DIR = "/dev/shm/mminf_test"
os.makedirs(BASE_DIR, exist_ok=True)


def file_write_bytes(payload: bytes, path: str):
    with open(path, "wb") as f:
        f.write(payload)


def file_read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


# ========================
# Data generation
# ========================
def make_payload():
    tensor = torch.randn(16, 4096)
    return (
        tensor.detach()
        .contiguous()
        .cpu()
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )


# ========================
# Benchmark
# ========================
def benchmark(n_iters=50):
    shm_times = []
    file_times = []

    payload = make_payload()

    for i in range(n_iters):
        # ----------------
        # SHM benchmark
        # ----------------
        name = f"mminf_{uuid.uuid4().hex}"

        t0 = time.perf_counter()
        meta = shm_write_bytes(payload, name=name)
        out = shm_read_bytes(meta)
        t1 = time.perf_counter()

        assert out == payload
        shm_times.append(t1 - t0)

        # ----------------
        # File benchmark
        # ----------------
        path = os.path.join(BASE_DIR, f"{uuid.uuid4().hex}.bin")

        t0 = time.perf_counter()
        file_write_bytes(payload, path)
        out = file_read_bytes(path)
        t1 = time.perf_counter()

        assert out == payload
        file_times.append(t1 - t0)

        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    # ========================
    # Results
    # ========================
    def summarize(name, times):
        print(f"\n{name}:")
        print(f"  mean: {statistics.mean(times)*1e3:.3f} ms")
        print(f"  p50 : {statistics.median(times)*1e3:.3f} ms")
        print(f"  p95 : {statistics.quantiles(times, n=20)[18]*1e3:.3f} ms")
        print(f"  min : {min(times)*1e3:.3f} ms")
        print(f"  max : {max(times)*1e3:.3f} ms")

    summarize("SharedMemory", shm_times)
    summarize("File (/dev/shm)", file_times)


if __name__ == "__main__":
    benchmark(n_iters=10)