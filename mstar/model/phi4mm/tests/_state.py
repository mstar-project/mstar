"""The vision node's state as the engine hands it over: the vision array's handle over a stand-in for its ragged
resource, and the node's step opened on that handle around a forward."""

from __future__ import annotations

from contextlib import contextmanager

from mstar.engine.resources import StepContext
from mstar.engine.state.adapters import StateSetup, _resolve, build_handle
from mstar.model.keys import VISION_ATTN
from mstar.model.phi4mm.config import VISION_TOKENS
from mstar.model.phi4mm.vision import vision_array
from mstar.model.submodule_base import engine_step


def bind_vision(node, attn) -> None:
    """Bind ``node``'s vision array to ``attn``, which runs its read."""
    array = vision_array(node.vision_config)
    node.bind_node_state({VISION_TOKENS: build_handle(array, {VISION_ATTN: attn}, _resolve([array], StateSetup()))})


@contextmanager
def vision_step(node, rids, inputs):
    """The node's step over ``inputs``, its state expanded and its handle open until the block ends."""
    step = engine_step(node, graph_walk="prefill_vision", request_ids=rids, inputs=inputs)
    step.set_ctx(StepContext(request_ids=rids, graph_walk="prefill_vision", slot=0, capture=False))
    handle = node.state_handles[VISION_TOKENS]
    handle.begin_step(step)
    try:
        yield step
    finally:
        handle.end_step(step)
