"""``torch.distributed`` across emulated GPUs.

Every emulated GPU is its own process, as on a real node. Process groups that
would be NCCL are gloo (the control traffic mstar sends over CPU groups is
real). A collective over simulated-GPU tensors moves no data (the tensors
have none) and is charged its NCCL time (:func:`mstar_sim.cost.collective_seconds`,
measured tables where they exist) on the current stream, on every rank alike,
so symmetric ranks stay level in simulated time, as SGLang's simulator and
Maya assume.
"""

from __future__ import annotations

import functools

import torch
import torch.distributed as dist

from mstar_sim import clock, cost, device


class _Done:
    """The ``Work`` of a collective that finished at launch (host side)."""

    def wait(self, timeout=None):
        return True

    def is_completed(self):
        return True

    def is_success(self):
        return True

    def get_future(self):
        fut = torch.futures.Future()
        fut.set_result(None)
        return fut


def _sim_tensors(args, kwargs) -> list[torch.Tensor]:
    flat = device.tree_flatten((args, kwargs))[0]
    return [t for t in flat if isinstance(t, device.SimTensor)]


def _group_size(kwargs, args, position: int | None) -> int:
    group = kwargs.get("group")
    if group is None and position is not None and len(args) > position:
        group = args[position]
    return dist.get_world_size(group) if dist.is_initialized() else 1


def _collective(name: str, op: str, group_pos: int | None):
    real = getattr(dist, name)

    @functools.wraps(real)
    def call(*args, **kwargs):
        tensors = _sim_tensors(args, kwargs)
        if not tensors:
            return real(*args, **kwargs)
        n = _group_size(kwargs, args, group_pos)
        nbytes = max(t.numel() * t.element_size() for t in tensors)
        if not clock.FAST:
            clock.charge(cost.collective_seconds(op, n, nbytes, graph=clock.capturing() is not None))
        return _Done() if kwargs.get("async_op") else None

    return call


# name -> (cost family, positional index of `group`)
_COLLECTIVES = {
    "all_reduce": ("all_reduce", 2),
    "all_gather_into_tensor": ("all_gather", 2),
    "all_gather": ("all_gather", 2),
    "reduce_scatter_tensor": ("reduce_scatter", 3),
    "reduce_scatter": ("reduce_scatter", 3),
    "all_to_all_single": ("alltoall", 4),
    "broadcast": ("all_gather", 2),
    "reduce": ("all_reduce", 3),
    "send": ("all_gather", 2),
    "recv": ("all_gather", 2),
    "isend": ("all_gather", 2),
    "irecv": ("all_gather", 2),
}


def _gloo(backend):
    if backend is None:
        return None
    text = str(backend)
    if "nccl" in text:
        return "gloo"
    return backend


def install() -> None:
    init = dist.init_process_group

    @functools.wraps(init)
    def init_process_group(*args, **kwargs):
        if args:
            args = (_gloo(args[0]), *args[1:])
        if "backend" in kwargs:
            kwargs["backend"] = _gloo(kwargs["backend"])
        kwargs.pop("device_id", None)  # no device to bind a gloo group to
        return init(*args, **kwargs)

    dist.init_process_group = init_process_group
    new_group = dist.new_group

    @functools.wraps(new_group)
    def new_group_gloo(*args, **kwargs):
        if "backend" in kwargs:
            kwargs["backend"] = _gloo(kwargs["backend"])
        return new_group(*args, **kwargs)

    dist.new_group = new_group_gloo
    default_backend = dist.get_default_backend_for_device
    dist.get_default_backend_for_device = (
        lambda d: "gloo" if str(getattr(d, "type", d)).startswith("cuda") else default_backend(d)
    )
    for name, (op, group_pos) in _COLLECTIVES.items():
        if hasattr(dist, name):
            setattr(dist, name, _collective(name, op, group_pos))
