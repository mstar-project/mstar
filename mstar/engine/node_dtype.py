"""Which dtype each node runs in, resolved once at load.

A node's dtype is its param dtype and its autocast dtype both: the engine
manager casts the node to it, and every forward, warm-up and capture runs
under autocast in it. ``None`` is a node that keeps the dtype it was built in
and runs without autocast (``NodeSubmodule.disable_autocast``).
"""
import contextvars
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

import torch

if TYPE_CHECKING:
    from mstar.model.base import Model
    from mstar.model.submodule_base import NodeSubmodule

logger = logging.getLogger(__name__)


def resolve_model_dtype(model: "Model", model_config: dict) -> torch.dtype | None:
    """The model-wide dtype: YAML ``autocast_dtype`` over ``get_autocast_dtype``.

    A model returning ``None`` pinned its numerics, so YAML may only agree.
    """
    model_dtype = model.get_autocast_dtype()
    if "autocast_dtype" not in model_config:
        return model_dtype
    yaml_dtype = model_config["autocast_dtype"]
    if model_dtype is None and yaml_dtype is not None:
        raise ValueError(
            f"{type(model).__name__} runs without autocast (get_autocast_dtype "
            f"is None), so YAML autocast_dtype={yaml_dtype!r} cannot turn it on; "
            "drop the key or set it to null"
        )
    return yaml_dtype


def resolve_node_dtype(
    submodule: "NodeSubmodule", model_dtype: torch.dtype | None,
) -> torch.dtype | None:
    """One node's dtype: ``None`` if it opted out, else its own override, else the model's."""
    if submodule.disable_autocast:
        return None
    return submodule.get_autocast_dtype() or model_dtype


_current_node: contextvars.ContextVar[tuple[str | None, torch.dtype | None]] = (
    contextvars.ContextVar("mstar_current_node", default=(None, None))
)
_warned: set[tuple[str | None, str]] = set()


@contextmanager
def node_scope(node_name: str | None, dtype: torch.dtype | None = None) -> Iterator[None]:
    """Name the node whose forward runs inside, and its resolved dtype, for
    ``warn_implicit_downcast``."""
    token = _current_node.set((node_name, dtype))
    try:
        yield
    finally:
        _current_node.reset(token)


def warn_implicit_downcast(op: str, src: torch.dtype, dst: torch.dtype) -> None:
    """Warn once per node and op that ``op`` ran a node's ``src`` input in ``dst``
    rather than the node's dtype. Silent while dynamo traces, which would
    graph-break on it; an eager or custom-op caller still reports.

    A custom op inside a compiled forward sees no autocast, so a node that
    autocasts in ``dst`` lands here too; that is the dtype it asked for."""
    if torch.compiler.is_compiling():
        return
    node, node_dtype = _current_node.get()
    if node_dtype == dst or (node, op) in _warned:
        return
    _warned.add((node, op))
    reason = "the node has autocast off" if node_dtype is None else f"the node runs in {node_dtype}"
    logger.warning(
        "%s runs node %s's %s input in %s: %s, and the kernel takes no %s",
        op, node or "<unknown>", src, dst, reason, src,
    )
