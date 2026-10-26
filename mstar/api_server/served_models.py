"""The names an OpenAI request's ``model`` field may carry, as vLLM reads it.

The base model's name runs the base model; with LoRA on (a ``resources: lora:`` block in
the serving config), an adapter's name runs that adapter. Adapters come from the config
and from runtime loads (``/v1/load_lora_adapter``), so the set changes while serving.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from typing import Any

from mstar.engine.resources.lora.config import (
    LORA_ADAPTER_KWARG,
    lora_enabled,
    served_lora_adapters,
)
from mstar.model.registry import model_init_kwargs


class ServedModels:
    """Read by request handlers; changed by runtime loads and unloads. ``adapters`` is a
    tuple replaced whole under a lock, so a reader always sees one consistent set."""

    def __init__(self, base: Iterable[str], adapters: Iterable[str] = (), lora: bool = False):
        # names that select the base model: the config's model name and its checkpoint repo
        self.base = frozenset(base)
        self._adapters: tuple[str, ...] = tuple(adapters)
        self.lora = lora or bool(self._adapters)
        self._lock = threading.Lock()

    @classmethod
    def of(cls, model_name: str, model_config: Mapping[str, Any] | None) -> "ServedModels":
        repo = model_init_kwargs(model_name).get("model_path_hf")
        config = model_config or {}
        return cls(
            base=(n for n in (model_name, repo) if n),
            adapters=served_lora_adapters(config),
            lora=lora_enabled(config),
        )

    @property
    def adapters(self) -> tuple[str, ...]:
        return self._adapters

    def add(self, name: str) -> None:
        with self._lock:
            if name not in self._adapters:
                self._adapters = (*self._adapters, name)

    def remove(self, name: str) -> bool:
        """Take ``name`` out of selection; False if it was not selectable."""
        with self._lock:
            if name not in self._adapters:
                return False
            self._adapters = tuple(a for a in self._adapters if a != name)
            return True

    def adapter_for(self, requested: str | None) -> str | None:
        """The adapter a request's ``model`` names; None runs the base model.

        A deployment without LoRA does not read the field, as before LoRA. With LoRA on,
        any other name that is not a selectable adapter is a ``ValueError`` (a 400),
        including one being unloaded: an unload takes the name out of selection here
        before any worker hears of it.
        """
        if not self.lora or not requested or requested in self.base:
            return None
        self.check(requested)
        return requested

    def check(self, adapter: object) -> None:
        """Refuse (``ValueError``, a 400) an adapter name that is not selectable now:
        how a ``lora_adapter`` kwarg (``/generate``, ``extra_body``) is checked."""
        if not isinstance(adapter, str) or adapter not in self._adapters:
            raise ValueError(
                f"unknown LoRA adapter {adapter!r}; this deployment serves "
                f"{sorted(self._adapters)}"
            )

    def select(self, requested: str | None, model_kwargs: dict | None) -> dict | None:
        """``model_kwargs`` naming the adapter a request runs under: the one its OpenAI
        ``model`` field names, or its ``lora_adapter`` kwarg (``/generate``,
        ``extra_body``), each checked. Both, naming different adapters, is a
        ``ValueError``."""
        adapter = self.adapter_for(requested)
        explicit = (model_kwargs or {}).get(LORA_ADAPTER_KWARG)
        if explicit is None:
            return model_kwargs if adapter is None else {**(model_kwargs or {}), LORA_ADAPTER_KWARG: adapter}
        self.check(explicit)
        if adapter is not None and explicit != adapter:
            raise ValueError(
                f"model {requested!r} and lora_adapter {explicit!r} name different adapters"
            )
        return model_kwargs
