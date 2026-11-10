"""The names an OpenAI request's ``model`` field may carry, as vLLM reads it.

The base model's name runs the base model; with LoRA on (a ``resources: lora:`` block in
the serving config), an adapter's name runs that adapter. Adapters come from the config
and from runtime loads (``/v1/load_lora_adapter``), so the set changes while serving.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

from mstar.engine.resources.lora.config import LORA_ADAPTER_KWARG, LORA_DIGEST_KWARG, lora_enabled
from mstar.engine.resources.lora.identity import adapter_prefix_root
from mstar.model.registry import model_init_kwargs


class Selected(NamedTuple):
    """What ``ServedModels.select`` settles for a request."""

    model_kwargs: dict | None
    # the root its prefix-cache key chains hang from: its adapter's, or empty for the
    # base model, whose pages adapters' requests must not be served
    prefix_root: bytes = b""


class ServedModels:
    """Read by request handlers; changed by runtime loads and unloads. ``digests`` maps
    each selectable adapter to its weights' digest, which a request on it carries
    (``LORA_DIGEST_KWARG``: a worker serving other weights under the name refuses it) and
    keys its prompt's pages under (``adapter_prefix_root``), apart from the base model's
    and from other weights'. The mapping is replaced whole under a lock, so a reader
    always sees one consistent set."""

    def __init__(self, base: Iterable[str], digests: Mapping[str, bytes] | None = None, lora: bool = False):
        # names that select the base model: the config's model name and its checkpoint repo
        self.base = frozenset(base)
        self._digests: Mapping[str, bytes] = dict(digests or {})
        self.lora = lora or bool(self._digests)
        self._lock = threading.Lock()

    @classmethod
    def of(
        cls, model_name: str, model_config: Mapping[str, Any] | None,
        digests: Mapping[str, bytes] | None = None,
    ) -> "ServedModels":
        """``digests`` are the serving config's adapters' (``resolve_lora_adapters``)."""
        repo = model_init_kwargs(model_name).get("model_path_hf")
        return cls(
            base=(n for n in (model_name, repo) if n), digests=digests,
            lora=lora_enabled(model_config or {}),
        )

    @property
    def adapters(self) -> tuple[str, ...]:
        return tuple(self._digests)

    def add(self, name: str, digest: bytes) -> None:
        """Make ``name`` selectable, serving the weights ``digest`` names."""
        with self._lock:
            self._digests = {**self._digests, name: digest}

    def remove(self, name: str) -> bytes | None:
        """Take ``name`` out of selection; its digest, or None if it was not selectable."""
        with self._lock:
            digest = self._digests.get(name)
            if digest is not None:
                self._digests = {n: d for n, d in self._digests.items() if n != name}
            return digest

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

    def check(self, adapter: object) -> bytes:
        """``adapter``'s digest; ``ValueError`` (a 400) for a name that is not selectable
        now: how a ``lora_adapter`` kwarg (``/generate``, ``extra_body``) is checked."""
        digest = self._digests.get(adapter) if isinstance(adapter, str) else None
        if digest is None:
            raise ValueError(
                f"unknown LoRA adapter {adapter!r}; this deployment serves "
                f"{sorted(self._digests)}"
            )
        return digest

    def select(self, requested: str | None, model_kwargs: dict | None) -> Selected:
        """``model_kwargs`` naming the adapter a request runs under and its digest, and
        that adapter's prefix-cache root: the one its OpenAI ``model`` field names, or its
        ``lora_adapter`` kwarg (``/generate``, ``extra_body``), each checked. Both, naming
        different adapters, is a ``ValueError``."""
        adapter = self.adapter_for(requested)
        explicit = (model_kwargs or {}).get(LORA_ADAPTER_KWARG)
        if explicit is not None:
            digest = self.check(explicit)
            if adapter is not None and explicit != adapter:
                raise ValueError(
                    f"model {requested!r} and lora_adapter {explicit!r} name different adapters"
                )
            adapter = explicit
        elif adapter is None:
            return Selected(model_kwargs)
        else:
            digest = self.check(adapter)
        kwargs = {**(model_kwargs or {}), LORA_ADAPTER_KWARG: adapter, LORA_DIGEST_KWARG: digest}
        return Selected(kwargs, adapter_prefix_root(digest))
