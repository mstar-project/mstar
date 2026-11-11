"""The ``model_kwargs.draft`` block: whether a node drafts and verifies tokens
(speculative decoding), with which drafter, how many draft tokens, and in which
mode.

Kept free of torch and the resources, so the conductor parses it before any
worker exists (`mstar.validation.validate_deployment`).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class DraftMode(Enum):
    # asynchronous when every resource on the node can be corrected on the
    # device and async scheduling is on, else synchronous, with the reason logged
    AUTO = "auto"
    # the worker plans a verify step only after the step before it was
    # post-processed, so every host length is exact when a step is planned
    SYNCHRONOUS = "synchronous"
    # step N+1 is planned from upper bounds while N runs, and corrected on the device
    ASYNCHRONOUS = "asynchronous"


class DraftMethod:
    """A kind of drafter: whether it loads a checkpoint of its own."""

    __slots__ = ("name", "needs_model")

    def __init__(self, name: str, needs_model: bool):
        self.name = name
        self.needs_model = needs_model


# The drafters the engine knows. A model may serve others (a test's scripted
# drafter, say) by passing its own table to `DraftConfig.parse`.
DSPARK = DraftMethod("dspark", needs_model=True)
MTP = DraftMethod("mtp", needs_model=False)
DRAFT_METHODS: Mapping[str, DraftMethod] = {m.name: m for m in (DSPARK, MTP)}

_KEYS = ("method", "model", "num_draft_tokens", "mode")


@dataclass(frozen=True)
class DraftConfig:
    """One node's draft setup, fixed for the deployment. ``num_draft_tokens``
    (``k``) sizes the verify capture, the KV slack, the random-number stride
    and the take-back amounts, so every rank derives the same from it."""

    method: str
    num_draft_tokens: int
    model: str | None = None
    mode: DraftMode = DraftMode.AUTO

    @property
    def block(self) -> int:
        """Tokens a verify step runs a request: the last emitted token and the drafts."""
        return self.num_draft_tokens + 1

    @property
    def kv_slack(self) -> int:
        """How far a request's reserved KV length runs past its verified one at
        most: two committed verify steps whose results are not yet applied."""
        return 2 * self.block

    @classmethod
    def parse(cls, block: Any, methods: Mapping[str, DraftMethod] = DRAFT_METHODS) -> DraftConfig:
        """``model_kwargs.draft`` as written in the serving config. Raises
        ``ValueError`` naming every problem with it."""
        if not isinstance(block, Mapping):
            raise ValueError(f"model_kwargs.draft must be a mapping, got {type(block).__name__}")
        problems = [f"unknown key {key!r} (known: {', '.join(_KEYS)})" for key in block if key not in _KEYS]
        method = block.get("method")
        kind = methods.get(method) if isinstance(method, str) else None
        if kind is None:
            problems.append(f"method must be one of {sorted(methods)}, got {method!r}")
        model = block.get("model")
        if model is not None and not (isinstance(model, str) and model):
            problems.append(f"model must be a checkpoint name or path, got {model!r}")
        if kind is not None:
            if kind.needs_model and model is None:
                problems.append(f"method {method!r} drafts from its own checkpoint: set model")
            if not kind.needs_model and model is not None:
                problems.append(f"method {method!r} drafts from the target's checkpoint: drop model")
        k = block.get("num_draft_tokens")
        if not (isinstance(k, int) and not isinstance(k, bool) and k >= 1):
            problems.append(f"num_draft_tokens must be a positive integer, got {k!r}")
        mode = block.get("mode", DraftMode.AUTO.value)
        try:
            mode = DraftMode(mode)
        except ValueError:
            problems.append(f"mode must be one of {[m.value for m in DraftMode]}, got {mode!r}")
        if problems:
            raise ValueError("model_kwargs.draft: " + "; ".join(problems))
        return cls(method=method, num_draft_tokens=k, model=model, mode=mode)

    @classmethod
    def from_model_kwargs(
        cls, model_kwargs: Mapping[str, Any] | None, methods: Mapping[str, DraftMethod] = DRAFT_METHODS,
    ) -> DraftConfig | None:
        """The parsed block, or None when the config has none: without it
        speculative decoding is off, whatever the checkpoint ships."""
        block = (model_kwargs or {}).get("draft")
        return None if block is None else cls.parse(block, methods)
