"""A verifying node's mixed prompt-plus-verify capture."""
from __future__ import annotations

from mstar.engine.accelerator_graph_config import PackedAcceleratorGraphConfig


class MixedVerifyGraphConfig(PackedAcceleratorGraphConfig):
    """A packed capture of a verifying node's mixed steps, whose lean resources get their twins as a batched
    capture's do (`AcceleratorGraphRunner._get_slot_specs`): its verify rows run ``k + 1`` tokens each through the
    sampler, so an all-greedy step's argmax (`greedy_verify`) saves what it saves on a verify step. The node's
    other packed captures (its prompt steps, a row's one sampled token each) keep main's batched-only twins."""

    verify_lean_twins = True
