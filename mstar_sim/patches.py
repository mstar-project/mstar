"""The few places where mstar, and torch itself, are adjusted for the emulated GPU, all
from outside.

- ``cap_captures``: capture buckets above ``MSTAR_SIM_MAX_CAPTURE_TOKENS`` /
  ``MSTAR_SIM_MAX_CAPTURE_BATCH`` are dropped, as if the model had declared a
  smaller set, for a faster start-up (each bucket is captured once). ``0``
  keeps every bucket, and a step larger than any bucket runs eagerly as in
  production.
- ``kv_scatter``: the paged-KV write is priced by the slots it writes (a
  Triton launch is priced by its arguments' size, here the whole cache layer).
- ``conv_kernels``: mstar's Triton conv kernels are priced by their real
  work and table shape (a launch's traffic would count the whole state pool).
- ``header_only_weights``: weights are read from safetensors headers only.
- ``worker_flag``: a model built on the CPU in a worker allocates nothing.
- ``rust_poll``: mstar's poll for other processes is a wait the clock knows
  (:mod:`mstar_sim.waits`).
- ``no_uvloop``: uvloop's loop waits where the clock can't see, so asyncio's stands in.
- ``swap_or_assign``, ``relax_blockwise_scaled_mm_meta``: torch's own (every process, mstar
  or not; ``mstar_sim.device.install`` applies them).

Each patch checks every mstar name it relies on at install, whether or not its
option is on (:func:`_require`). One whose target is missing is skipped with a
warning naming it (``SKIPPED``), so the emulator still runs on an older tree
(a bisect, a corpus of old bugs). On the current tree none may be skipped:
``mstar_sim/tests/test_device.py`` checks that in CI, so the mstar change that
moves a target fails there.
"""

from __future__ import annotations

import bisect
import functools
import importlib
import math
import os
import sys
import threading

import torch

from mstar_sim.checkpoint import SMALL_WEIGHT


def _replace(owner, name: str, value) -> None:
    """``owner.name = value``, for an attribute mstar still has."""
    _require(owner, name)
    setattr(owner, name, value)


def _require(owner, name: str) -> None:
    if not hasattr(owner, name):
        raise AttributeError(f"{getattr(owner, '__name__', owner)}.{name} is gone: mstar changed under "
                             "mstar_sim; update mstar_sim/patches.py")


def _cap(values, limit):
    if not limit or not values:
        return values
    kept = [v for v in values if v <= limit]
    return kept or [min(values)]


def cap_captures() -> None:
    # targets first, whether or not the caps are set: a rename fails every run
    cfg = importlib.import_module("mstar.engine.accelerator_graph_config")
    runner = importlib.import_module("mstar.engine.accelerator_graph_runner")
    _require(cfg.AcceleratorGraphConfig, "__init__")
    _require(cfg.PackedAcceleratorGraphConfig, "__init__")
    _require(cfg.PiecewiseAcceleratorGraphConfig, "__post_init__")
    tokens = int(os.environ.get("MSTAR_SIM_MAX_CAPTURE_TOKENS", "0"))
    batch = int(os.environ.get("MSTAR_SIM_MAX_CAPTURE_BATCH", "0"))
    if not tokens and not batch:
        return
    for cls in (runner.AcceleratorGraphRunner, *runner.AcceleratorGraphRunner.__subclasses__()):
        if "CAPTURE_BATCH_SIZES" in vars(cls):
            cls.CAPTURE_BATCH_SIZES = _cap(cls.CAPTURE_BATCH_SIZES, batch)
    for name in dir(runner):
        obj = getattr(runner, name)
        if isinstance(obj, type) and "CAPTURE_BATCH_SIZES" in vars(obj):
            obj.CAPTURE_BATCH_SIZES = _cap(obj.CAPTURE_BATCH_SIZES, batch)

    base_init = cfg.AcceleratorGraphConfig.__init__

    @functools.wraps(base_init)
    def init(self, *args, **kwargs):
        base_init(self, *args, **kwargs)
        self.capture_batch_sizes = _cap(self.capture_batch_sizes, batch)
        if hasattr(self, "capture_token_lengths"):
            self.capture_token_lengths = _cap(self.capture_token_lengths, tokens)

    cfg.AcceleratorGraphConfig.__init__ = init
    packed_init = cfg.PackedAcceleratorGraphConfig.__init__

    @functools.wraps(packed_init)
    def packed(self, *args, **kwargs):
        packed_init(self, *args, **kwargs)
        self.capture_token_lengths = _cap(self.capture_token_lengths, tokens)

    cfg.PackedAcceleratorGraphConfig.__init__ = packed
    post = cfg.PiecewiseAcceleratorGraphConfig.__post_init__

    @functools.wraps(post)
    def post_init(self):
        self.capture_batch_sizes = _cap(self.capture_batch_sizes, batch)
        if hasattr(self, "total_tokens"):
            self.total_tokens = _cap(self.total_tokens, tokens)
        post(self)

    cfg.PiecewiseAcceleratorGraphConfig.__post_init__ = post_init
    for sub in cfg.PiecewiseAcceleratorGraphConfig.__subclasses__():
        if "__post_init__" in vars(sub):
            sub_post = sub.__post_init__

            def sub_init(self, _sub_post=sub_post):
                if hasattr(self, "total_tokens"):
                    self.total_tokens = _cap(self.total_tokens, tokens)
                self.capture_batch_sizes = _cap(self.capture_batch_sizes, batch)
                _sub_post(self)

            sub.__post_init__ = sub_init


def kv_scatter() -> None:
    from mstar_sim import clock

    cache = importlib.import_module("mstar.engine.resources.kv.cache")

    # (layer, k, v, page_idx, cache_idx); older trees put the cache and a layer index first:
    # either way k is the fourth from last
    @clock.kernel_fn("triton._kv_scatter_nhd_kernel", writes_args=True,
                     cost=lambda *a: (0.0, 4.0 * a[-4].numel() * a[-4].element_size()))
    def scatter_nhd(*args):
        return None  # in place

    _replace(cache, "_kv_scatter_nhd_eager", scatter_nhd)


def conv_kernels() -> None:
    from mstar_sim import kernels

    mod = importlib.import_module("mstar.utils.causal_conv1d")
    _replace(mod, "causal_conv1d_fn", kernels.causal_conv1d_fn)
    _replace(mod, "causal_conv1d_update", kernels.causal_conv1d_update)


def no_graphs() -> None:
    """``MSTAR_SIM_NO_GRAPHS``: ``1`` captures nothing, so every step runs the eager
    forward (module hooks fire, ops can be inspected one by one); ``A,B`` runs only
    those graph nodes eagerly (one whose capture needs a real Triton build, say)."""
    runner = importlib.import_module("mstar.engine.accelerator_graph_runner")
    engine = importlib.import_module("mstar.engine.engine")
    _require(engine.Engine, "_build_piecewise_runners")
    for attr in ("_capture_configs", "_submodule_name"):  # set in the runner's __init__
        if attr not in runner.AcceleratorGraphRunner.__init__.__code__.co_names:
            raise AttributeError(f"AcceleratorGraphRunner.{attr} is gone: mstar changed under mstar_sim; "
                                 "update mstar_sim/patches.py")
    value = os.environ.get("MSTAR_SIM_NO_GRAPHS", "")
    if not value:
        return
    nodes = None if value == "1" else set(value.split(","))

    def eager(node) -> bool:
        return nodes is None or node in nodes

    init = runner.AcceleratorGraphRunner.__init__

    @functools.wraps(init)
    def no_capture(self, *args, **kwargs):
        init(self, *args, **kwargs)
        if eager(self._submodule_name):
            self._capture_configs = []

    runner.AcceleratorGraphRunner.__init__ = no_capture
    build = engine.Engine._build_piecewise_runners

    def piecewise(self, node_name, *args, **kwargs):
        return {} if eager(node_name) else build(self, node_name, *args, **kwargs)

    _replace(engine.Engine, "_build_piecewise_runners", piecewise)


# ------------------------------------------------------------------- weights

_DTYPES = {
    "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
    "U8": torch.uint8, "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


def _parse(fd: int) -> tuple[int, dict]:
    """A safetensors file's data offset and header; refuses a copy that read its small tensors as zeros."""
    import json
    import struct

    (n,) = struct.unpack("<Q", os.pread(fd, 8, 0))
    header = json.loads(os.pread(fd, n, 8))
    if (header.pop("__metadata__", None) or {}).get("mstar_sim") == "shapes_only":  # only VERSION 2 wrote it
        raise ValueError(f"{os.readlink(f'/proc/self/fd/{fd}')} is a mstar_sim copy whose small tensors read as "
                         "zeros: rebuild it with python -m mstar_sim.checkpoint <repo> --hf-home <dir>")
    return 8 + n, header


def _header(path) -> dict:
    fd = os.open(path, os.O_RDONLY)
    try:
        return _parse(fd)[1]
    finally:
        os.close(fd)


@functools.lru_cache(maxsize=256)
def _small_spans(fd: int, file: tuple[int, int, int]) -> list[tuple[int, int]]:
    """The byte ranges of the file's small tensors (``file``: its device, inode and mtime)."""
    base, header = _parse(fd)
    return sorted((base + v["data_offsets"][0], base + v["data_offsets"][1]) for v in header.values() if _small(v))


def _small(info: dict) -> bool:
    n = 1
    for d in info["shape"]:
        n *= d
    return n < SMALL_WEIGHT


def _meta(info: dict) -> torch.Tensor:
    return torch.empty(info["shape"], dtype=_DTYPES.get(info["dtype"], torch.uint8), device="meta")


def _on(tensor: torch.Tensor, device) -> torch.Tensor:
    """Where safetensors would put it: a cuda device takes it (and its memory) there."""
    if device is None or str(device) == "cpu":
        return tensor
    return tensor.to(device if not isinstance(device, int) else f"cuda:{device}")


def _read(path, name):
    with _real_safe_open(path, framework="pt") as f:
        return f.get_tensor(name)


_real_safe_open = None


class _MetaSlice:
    def __init__(self, info):
        self.info = info

    def get_shape(self):
        return list(self.info["shape"])

    def get_dtype(self):
        return self.info["dtype"]

    def __getitem__(self, index):
        return _meta(self.info)[index]


class _MetaSafeOpen:
    """``safetensors.safe_open`` over a file's header only: tensors are meta."""

    def __init__(self, path, framework="pt", device="cpu", **_):  # transformers 5.22 passes backend=
        self._path = path
        self._header = _header(path)
        self._device = device

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def keys(self):
        return sorted(self._header)

    def offset_keys(self):
        return sorted(self._header, key=lambda k: self._header[k]["data_offsets"][0])

    def metadata(self):
        return {}

    def get_tensor(self, name):
        info = self._header[name]
        return _on(_read(self._path, name) if _small(info) else _meta(info), self._device)

    def get_slice(self, name):
        return _MetaSlice(self._header[name])


def header_only_weights() -> None:
    """Weights are read from safetensors headers only: every tensor above
    ``SMALL_WEIGHT`` elements is meta, so no large weight is read and a
    checkpoint of sparse files will do (``mstar_sim.checkpoint`` keeps the small ones' bytes)."""
    import safetensors
    import safetensors.torch

    global _real_safe_open
    _real_safe_open = _real_safe_open or safetensors.safe_open
    # the library first: weights stay header-only even where mstar's own reader moved
    safetensors.safe_open = _MetaSafeOpen
    safetensors.torch.safe_open = _MetaSafeOpen
    safetensors.torch.load_file = lambda path, device="cpu": {
        k: _on(_read(path, k) if _small(v) else _meta(v), device) for k, v in _header(path).items()
    }
    iterators = importlib.import_module("mstar.model.loader.iterators")
    _replace(iterators, "safe_open", _MetaSafeOpen)

    if not hasattr(iterators, "_read_tensor"):
        return  # an older loader reads through safe_open, patched above
    real_read = iterators._read_tensor

    def read_tensor(fd, rd):
        """Real bytes for a read of a small tensor (a shard of a large one's too would be a hole)."""
        st = os.fstat(fd)
        spans = _small_spans(fd, (st.st_dev, st.st_ino, st.st_mtime_ns))
        i = bisect.bisect_right(spans, (rd.offset, math.inf)) - 1
        if rd.nbytes == 0 or (i >= 0 and rd.offset < spans[i][1]):
            return real_read(fd, rd)
        shape = rd.shape
        if rd.narrow is not None:
            dim, start, stop = rd.narrow
            shape = shape[:dim] + (stop - start,) + shape[dim + 1:]
        return torch.empty(shape, dtype=rd.dtype, device="meta")

    _replace(iterators, "_read_tensor", read_tensor)


def worker_flag() -> None:
    """Mark worker processes, where large float CPU allocations go value-less."""
    conductor = importlib.import_module("mstar.conductor.conductor")
    global _worker_target_impl
    _worker_target_impl = conductor._worker_process_target
    _replace(conductor, "_worker_process_target", worker_process_target)


_worker_target_impl = None


def worker_process_target(*args, **kwargs):
    from mstar_sim import device

    device.WORKER = True
    conductor = importlib.import_module("mstar.conductor.conductor")
    impl = _worker_target_impl or conductor._worker_process_target
    return impl(*args, **kwargs)


def rust_poll() -> None:
    """The Rust communicator on the deployment's clock (:mod:`mstar_sim.waits`). Its poll is a
    wait on other processes: its thread counts as blocked, a write to its eventfd
    (``EventWakeup``) wakes it, the mover's eventfd can end it, and its timeout is a host
    deadline that holds until the clock reaches it. A send reads the clock, so a thread past
    the bound waits before it sends."""
    communicator = importlib.import_module("mstar.communication.rust_communicator").RustZMQCommunicator
    for name in ("__init__", "_poll_once", "send", "try_send_bytes"):
        _require(communicator, name)
    from mstar_sim import clock, waits

    if clock.FAST:
        return
    init, poll_once = communicator.__init__, communicator._poll_once
    send, try_send = communicator.send, communicator.try_send_bytes

    def initialized(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self._inner.register_wakeup_fd(waits._wakefd[0])

    def polled(self, timeout_ms):
        if timeout_ms <= 0:
            return poll_once(self, timeout_ms)

        event = getattr(self, "event", None)

        def poll(seconds):
            kind = poll_once(self, max(1, math.ceil(seconds * 1000)))
            # a wake with no event is the mover's; one with an event may be real, and is now drained
            return None if kind == "timeout" or (kind == "wake" and event is None) else kind

        kind = waits.polling(poll, event.fd if event is not None else None, timeout_ms / 1000)
        if kind == "msg":
            waits.delivered()
        return kind or "timeout"

    def sending(self, entity_id, msg):
        waits.read()
        send(self, entity_id, msg)

    def sending_bytes(self, entity_id, data):
        waits.read()
        return try_send(self, entity_id, data)

    communicator.__init__ = initialized
    communicator._poll_once = polled
    communicator.send = sending
    communicator.try_send_bytes = sending_bytes


def no_uvloop() -> None:
    """libuv's own epoll wait is invisible to the registry, and its clock to the patches: under
    the emulator ``uvloop`` hands back asyncio's loop, which they cover."""
    from mstar_sim import clock

    try:
        uvloop = importlib.import_module("uvloop")
    except ImportError:
        return
    if clock.FAST:
        return
    import asyncio

    uvloop.install = lambda: None
    uvloop.new_event_loop = asyncio.new_event_loop
    uvloop.run = lambda main, **kwargs: asyncio.run(main, debug=kwargs.get("debug"))


def swap_or_assign() -> None:
    """Moving a module swaps its parameters in place (so tied weights stay tied), but a
    swap refuses a tensor something else holds (weight_norm's hooks): that module alone
    is then moved or loaded by assignment, as torch does without the swap, on its thread
    alone (the swap setting reads per thread)."""
    from torch import nn

    future, assigning = torch.__future__, threading.local()
    swaps = future.get_swap_module_params_on_conversion
    future.get_swap_module_params_on_conversion = lambda: swaps() and not getattr(assigning, "on", False)

    def by_assignment(call, *args):
        assigning.on = True
        try:
            return call(*args)
        finally:
            assigning.on = False

    apply, load = nn.Module._apply, nn.Module._load_from_state_dict

    def apply_one(self, fn, recurse=True):
        try:
            return apply(self, fn, recurse)
        except RuntimeError as e:  # its own parameters (a child retries its own)
            if not str(e).startswith("_apply(): Couldn't swap"):
                raise
            return by_assignment(apply, self, fn, False)

    def load_one(self, state_dict, prefix, metadata, strict, missing, unexpected, errors):
        marks = len(missing), len(unexpected), len(errors)
        load(self, state_dict, prefix, metadata, strict, missing, unexpected, errors)
        if any(e.startswith("While swapping") for e in errors[marks[2]:]):
            for found, mark in zip((missing, unexpected, errors), marks, strict=True):
                del found[mark:]
            by_assignment(load, self, state_dict, prefix, metadata, strict, missing, unexpected, errors)

    nn.Module._apply = apply_one
    nn.Module._load_from_state_dict = load_one


def relax_blockwise_scaled_mm_meta() -> None:
    """torch 2.10's meta check for block-scaled fp8 GEMMs (``_scaled_mm_v2``, 1x128 x 128x128
    blocks) floors ``N / 128``, so a weight whose last 128-row block is partial (GLM-5.2's
    fused q_a + kv_a) fails it, though the CUDA kernel takes it with ceil-div scales. Such a
    weight on such a recipe is checked as if its last block were whole: every other check
    still applies."""
    import torch._meta_registrations as mr

    check = getattr(mr, "_check_scaled_mm_sizes_v2", None)
    if check is None or getattr(check, "_mstar_sim", False):
        return

    blocks = torch.nn.functional.ScalingType.BlockWise128x128.value

    def relaxed(self, mat2, scale_a, recipe_a, scale_b, recipe_b, *args, out_dtype=None, **kwargs):
        args = (scale_a, recipe_a, scale_b, recipe_b, *args)
        try:
            return check(self, mat2, *args, out_dtype=out_dtype, **kwargs)
        except RuntimeError as refused:
            n = mat2.shape[1]
            if n % 128 == 0 or blocks not in recipe_b:
                raise
            whole = torch.empty_strided((mat2.shape[0], -(-n // 128) * 128), mat2.stride(), dtype=mat2.dtype,
                                        device=mat2.device)
            try:
                check(self, whole, *args, out_dtype=out_dtype, **kwargs)
            except RuntimeError:
                raise refused from None
            return torch.empty(self.shape[0], n, dtype=out_dtype or self.dtype, device=self.device)

    relaxed._mstar_sim = True
    mr._check_scaled_mm_sizes_v2 = relaxed


def install() -> None:
    try:
        importlib.import_module("mstar")
    except ModuleNotFoundError as e:
        if e.name != "mstar":
            raise  # mstar is there but broken: not a run to make without the patches
        return  # not an mstar process
    for patch in PATCHES:
        try:
            patch()
        except (ImportError, AttributeError) as e:
            SKIPPED.append((patch.__name__, str(e)))
            print(f"mstar_sim: WARNING: patch {patch.__name__} skipped: {e}", file=sys.stderr, flush=True)


SKIPPED: list[tuple[str, str]] = []


PATCHES = (cap_captures, no_graphs, kv_scatter, conv_kernels, header_only_weights, worker_flag, rust_poll,
           no_uvloop)
