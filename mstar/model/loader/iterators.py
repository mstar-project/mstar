"""Streaming iterators over safetensors checkpoints.

Yields ``(key, tensor)`` one at a time so the full state_dict never has
to fit in memory.

``slice_spec`` lets TP-aware callers read only their shard of a tensor:
``slice_spec(key)`` returns a ``TensorSlice`` or ``None`` for a full read.
A slice that is contiguous on disk (dim 0, in practice) reads just those
bytes — for checkpoints dominated by expert tensors this cuts per-rank IO
by the TP factor (GLM-5.2 at TP8: ~704 GB -> ~120 GB per rank). A slice
on a later dim is strided on disk, so the whole tensor is read and sliced
in memory.

Tensors are read with ``pread`` into private buffers by a small thread
pool, a bounded window ahead of the consumer, instead of being faulted in
through safetensors' mmap: on network and ZFS filesystems a page fault
moves a few pages at a time, while large reads stream and several in
flight hide the filesystem's latency.
"""
from __future__ import annotations

import json
import math
import os
import struct
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from safetensors import safe_open


class TensorSlice(NamedTuple):
    """The half-open range of ``dim`` a caller wants: ``[start, stop)``."""

    dim: int
    start: int
    stop: int


SliceSpec = Callable[[str], TensorSlice | None]

# Reads run on this many threads and stay at most this many bytes ahead of
# the consumer: enough to keep a network filesystem busy, small enough that a
# slow consumer never buffers much of a shard.
_READ_THREADS = 8
_READ_AHEAD_BYTES = 512 << 20
_READ_AHEAD_TENSORS = 256

_DTYPES: dict[str, torch.dtype] = {
    name: getattr(torch, attr)
    for name, attr in (
        ("BOOL", "bool"), ("U8", "uint8"), ("I8", "int8"),
        ("I16", "int16"), ("U16", "uint16"), ("I32", "int32"),
        ("U32", "uint32"), ("I64", "int64"), ("U64", "uint64"),
        ("F16", "float16"), ("BF16", "bfloat16"), ("F32", "float32"),
        ("F64", "float64"), ("C64", "complex64"),
        ("F8_E4M3", "float8_e4m3fn"), ("F8_E5M2", "float8_e5m2"),
        ("F8_E8M0", "float8_e8m0fnu"),
    )
    if hasattr(torch, attr)
}


class _Read(NamedTuple):
    """One tensor's bytes on disk and how to turn them into the tensor."""

    offset: int
    nbytes: int
    dtype: torch.dtype
    shape: tuple[int, ...]
    narrow: TensorSlice | None  # applied in memory, after the read


# per call: macOS rejects an iovec of 2 GiB or more (EINVAL); Linux caps a call below that
_PREAD_MAX = 1 << 30


def _pread_into(fd: int, buf: memoryview, offset: int) -> None:
    done = 0
    while done < len(buf):
        n = os.preadv(fd, [buf[done:done + _PREAD_MAX]], offset + done)
        if n <= 0:
            raise EOFError(f"safetensors file truncated at byte {offset + done}")
        done += n


def _read_tensor(fd: int, rd: _Read) -> torch.Tensor:
    # numpy for the byte work: a torch copy here would open an OpenMP team
    # per read thread.
    buf = torch.empty(rd.nbytes, dtype=torch.uint8)
    _pread_into(fd, memoryview(buf.numpy()), rd.offset)
    shape = rd.shape
    if rd.narrow is not None:
        dim, start, stop = rd.narrow
        outer = math.prod(shape[:dim])
        inner = math.prod(shape[dim + 1:]) * rd.dtype.itemsize
        shape = shape[:dim] + (stop - start,) + shape[dim + 1:]
        sliced = torch.empty(outer * (stop - start) * inner, dtype=torch.uint8)
        np.copyto(
            sliced.numpy().reshape(outer, stop - start, inner),
            buf.numpy().reshape(outer, rd.shape[dim], inner)[:, start:stop],
        )
        buf = sliced
    return buf.view(rd.dtype).view(shape)


def _read_with_safetensors(path: Path, key: str, spec: TensorSlice | None) -> torch.Tensor:
    """For dtypes the byte reader has no torch view for (packed fp4, ...)."""
    with safe_open(str(path), framework="pt", device="cpu") as f:
        if spec is None:
            return f.get_tensor(key)
        dim, start, stop = spec
        sl = f.get_slice(key)
        index = [slice(None)] * len(sl.get_shape())
        index[dim] = slice(start, stop)
        return sl[tuple(index)].contiguous()


def _plan_read(meta: dict, base: int, spec: TensorSlice | None) -> _Read | None:
    dtype = _DTYPES.get(meta["dtype"])
    if dtype is None:
        return None
    shape = tuple(meta["shape"])
    begin, end = meta["data_offsets"]
    itemsize = dtype.itemsize
    if end - begin != math.prod(shape) * itemsize:
        raise ValueError(f"safetensors entry {meta} has a size that does not match its shape")
    if spec is None:
        return _Read(base + begin, end - begin, dtype, shape, None)
    dim, start, stop = spec
    if not -len(shape) <= dim < len(shape):
        # as safe_open's get_slice: a wrapped dim would read another axis's shard
        raise IndexError(f"slice dim {dim} out of range for a {len(shape)}-d tensor {meta}")
    dim %= len(shape)
    start, stop, _ = slice(start, stop).indices(shape[dim])
    stop = max(start, stop)
    if math.prod(shape[:dim]) == 1:
        # Nothing outer to stride over: the slice is one contiguous run.
        inner = math.prod(shape[dim + 1:]) * itemsize
        out = shape[:dim] + (stop - start,) + shape[dim + 1:]
        return _Read(base + begin + start * inner, (stop - start) * inner, dtype, out, None)
    return _Read(base + begin, end - begin, dtype, shape, TensorSlice(dim, start, stop))


def _pread_exact(fd: int, n: int, offset: int) -> bytes:
    buf = bytearray(n)
    _pread_into(fd, memoryview(buf), offset)
    return bytes(buf)


def _iter_files(
    paths: list[Path],
    device: torch.device | str,
    prefix: str | None,
    keys: set[str] | None,
    slice_spec: SliceSpec | None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Read the selected tensors of ``paths`` in order, prefetching ahead."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    open_fds: set[int] = set()

    def jobs() -> Iterator[tuple[str, int, Callable[[], torch.Tensor], int | None]]:
        for path in paths:
            fd = os.open(path, os.O_RDONLY)
            open_fds.add(fd)
            (header_len,) = struct.unpack("<Q", _pread_exact(fd, 8, 0))
            header = json.loads(_pread_exact(fd, header_len, 8))
            header.pop("__metadata__", None)
            # safe_open.keys() order, which callers have always seen.
            selected = [
                key for key in sorted(header)
                if (prefix is None or key.startswith(prefix))
                and (keys is None or key in keys)
            ]
            for i, key in enumerate(selected):
                spec = slice_spec(key) if slice_spec is not None else None
                rd = _plan_read(header[key], 8 + header_len, spec)
                last_fd = fd if i == len(selected) - 1 else None
                if rd is None:
                    # a dtype torch reads here but this path does not: its bytes still count
                    # against the read-ahead
                    begin, end = header[key]["data_offsets"]
                    yield key, end - begin, partial(_read_with_safetensors, path, key, spec), last_fd
                else:
                    yield key, rd.nbytes, partial(_read_tensor, fd, rd), last_fd
            if not selected:
                open_fds.discard(fd)
                os.close(fd)

    pending: deque[tuple[str, int, Future, int | None]] = deque()
    in_flight = 0
    todo = jobs()
    pool = ThreadPoolExecutor(_READ_THREADS, thread_name_prefix="safetensors-read")
    try:
        exhausted = False
        while True:
            while not exhausted and (not pending or (
                    in_flight < _READ_AHEAD_BYTES and len(pending) < _READ_AHEAD_TENSORS)):
                job = next(todo, None)
                if job is None:
                    exhausted = True
                    break
                key, nbytes, read, last_fd = job
                pending.append((key, nbytes, pool.submit(read), last_fd))
                in_flight += nbytes
            if not pending:
                return
            key, nbytes, future, last_fd = pending.popleft()
            tensor = future.result()
            in_flight -= nbytes
            if last_fd is not None:
                open_fds.discard(last_fd)
                os.close(last_fd)
            if device.type != "cpu":
                tensor = tensor.to(device)
            yield key, tensor
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        todo.close()
        for fd in open_fds:
            os.close(fd)


def iter_safetensors_file(
    path: str | Path,
    device: torch.device | str = "cpu",
    prefix: str | None = None,
    keys: set[str] | None=None,
    slice_spec: SliceSpec | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(key, tensor)`` from a single safetensors file."""
    yield from _iter_files([Path(path)], device, prefix, keys, slice_spec)


def iter_safetensors_shards(
    repo_dir: str | Path, device: torch.device | str = "cpu",
    prefix: str | None = None,
    keys: set[str] | None=None,
    slice_spec: SliceSpec | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(key, tensor)`` from a sharded HF safetensors checkpoint.

    Looks for ``model.safetensors.index.json`` in ``repo_dir``; if absent,
    falls back to a single ``model.safetensors`` file.
    """
    repo_dir = Path(repo_dir)
    index_path = repo_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)

        if prefix is not None or keys is not None:
            relevant_keys = [
                key for key in index["weight_map"]
                if (prefix is None or key.startswith(prefix)) and \
                   (keys is None or key in keys)
            ]
            shard_files = sorted(set([
                index["weight_map"][key] for key in relevant_keys
            ]))
        else:
            shard_files = sorted(set(index["weight_map"].values()))

        yield from _iter_files(
            [repo_dir / shard_file for shard_file in shard_files],
            device, prefix, keys, slice_spec,
        )
        return
    single = repo_dir / "model.safetensors"
    if single.exists():
        yield from iter_safetensors_file(
            single, device=device,
            prefix=prefix, keys=keys, slice_spec=slice_spec,
        )
        return
    raise FileNotFoundError(
        f"No safetensors checkpoint found in {repo_dir} "
        f"(looked for model.safetensors.index.json and model.safetensors)"
    )
