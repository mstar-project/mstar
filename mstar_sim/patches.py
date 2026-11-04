"""The few places where mstar is adjusted for the emulated GPU, all from outside.

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

Each patch checks every mstar name it relies on at install, whether or not its
option is on (:func:`_require`). One whose target is missing is skipped with a
warning naming it (``SKIPPED``), so the emulator still runs on an older tree
(a bisect, a corpus of old bugs). On the current tree none may be skipped:
``mstar_sim/tests/test_device.py`` checks that in CI, so the mstar change that
moves a target fails there.
"""

from __future__ import annotations

import functools
import importlib
import os
import sys

import torch


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

    @clock.kernel_fn("triton._kv_scatter_nhd_kernel", static=True,
                     cost=lambda layer, k, v, *a: (0.0, 4.0 * k.numel() * k.element_size()))
    def scatter_nhd(layer, k, v, page_idx, cache_idx):
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


def _header(path) -> dict:
    import json
    import struct

    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


# tensors this small are read for real (a LoRA alpha, a norm, a test's fixture):
# cheap, and host code reads their values; only weights this large go value-less
SMALL_WEIGHT = 1 << 16


def _small(info: dict) -> bool:
    n = 1
    for d in info["shape"]:
        n *= d
    return n < SMALL_WEIGHT


def _meta(info: dict) -> torch.Tensor:
    return torch.empty(info["shape"], dtype=_DTYPES.get(info["dtype"], torch.uint8), device="meta")


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

    def __init__(self, path, framework="pt", device="cpu"):
        self._path = path
        self._header = _header(path)

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
        return _read(self._path, name) if _small(info) else _meta(info)

    def get_slice(self, name):
        return _MetaSlice(self._header[name])


def header_only_weights() -> None:
    """Weights are read from safetensors headers only: every tensor above
    ``SMALL_WEIGHT`` elements is meta, so no large weight is read and a
    checkpoint of sparse files will do (its small tensors read as zeros)."""
    import safetensors
    import safetensors.torch

    global _real_safe_open
    _real_safe_open = _real_safe_open or safetensors.safe_open
    # the library first: weights stay header-only even where mstar's own reader moved
    safetensors.safe_open = _MetaSafeOpen
    safetensors.torch.safe_open = _MetaSafeOpen
    safetensors.torch.load_file = lambda path, device="cpu": {
        k: _read(path, k) if _small(v) else _meta(v) for k, v in _header(path).items()
    }
    iterators = importlib.import_module("mstar.model.loader.iterators")
    _replace(iterators, "safe_open", _MetaSafeOpen)

    real_read = iterators._read_tensor

    def read_tensor(fd, rd):
        shape = rd.shape
        if rd.nbytes < SMALL_WEIGHT * rd.dtype.itemsize:
            return real_read(fd, rd)
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


PATCHES = (cap_captures, no_graphs, kv_scatter, conv_kernels, header_only_weights, worker_flag)
