"""Which weights an adapter name stands for, as the prefix cache keys them.

A request on an adapter fills KV pages that differ from the base model's for the same
tokens, so its prompt's key chain hangs from a root that names the adapter's weights
(``AdapterPrefixRoots``). The preprocess worker, which keys prompts, reads each adapter
once when it is registered; nothing about an adapter travels per step.
"""

from __future__ import annotations

import hashlib
import logging
import threading

import torch

from mstar.engine.resources.kv.keys import fingerprint
from mstar.engine.resources.lora.checkpoint import read_lora_file

logger = logging.getLogger(__name__)

# folded into every adapter's root, so no adapter digest can coincide with another use of the chain root
_NAMESPACE = "lora-adapter"


def _field(hasher, payload: bytes) -> None:
    hasher.update(len(payload).to_bytes(8, "little"))
    hasher.update(payload)


def adapter_digest(adapter) -> bytes:
    """SHA-256 over everything an adapter's delta depends on, rank-independent.

    ``adapter`` is a ``LoraAdapter`` as ``read_lora_file`` returns it: every module's
    name, alpha (the sidecar's included), and A and B with their dtype and shape, so
    that equal bytes of a different shape or dtype digest apart.
    """
    hasher = hashlib.sha256()
    for module in sorted(adapter.layers):
        layer = adapter.layers[module]
        _field(hasher, module.encode())
        _field(hasher, repr(float(layer.alpha)).encode())
        for t in (layer.a, layer.b):
            t = t.detach().contiguous()
            _field(hasher, f"{t.dtype}{tuple(t.shape)}".encode())
            _field(hasher, t.view(torch.uint8).numpy().tobytes())
    return hasher.digest()


def read_adapter_digest(spec, path: str) -> bytes:
    """The digest of the adapter at ``path``, read as the pool reads it (``spec.convert_keys``)."""
    return adapter_digest(read_lora_file(path, convert_keys=spec.convert_keys))


def adapter_prefix_root(digest: bytes) -> bytes:
    """The root an adapter's key chains hang from.

    The digest alone, not the name: KV depends on the weights, so two names for the
    same weights share pages, and one name reloaded with other weights does not.
    """
    return fingerprint(_NAMESPACE, digest)


class AdapterPrefixRoots:
    """Adapter name -> the root its requests' key chains hang from.

    A name that is not registered has no root, and a request on it must not be keyed
    at all: keyed under the base model's empty root, its pages would serve base
    requests. ``register`` and ``unregister`` are the hooks for loading and unloading
    an adapter at runtime; reads happen on the preprocess thread, writes on whichever
    thread registers.
    """

    def __init__(self, roots: dict[str, bytes] | None = None):
        self._lock = threading.Lock()
        self._roots: dict[str, bytes] = dict(roots or {})

    @classmethod
    def read(cls, spec, digests: dict[str, bytes] | None = None) -> "AdapterPrefixRoots":
        """Register every adapter ``spec`` (a ``LoraSpec``) serves under ``digests``, or,
        for one not in it, under the digest of its file, read once."""
        roots = cls()
        for name, path in spec.adapters.items():
            digest = (digests or {}).get(name)
            roots.register(name, digest if digest is not None else read_adapter_digest(spec, path))
        return roots

    def register(self, name: str, digest: bytes) -> None:
        with self._lock:
            self._roots[name] = adapter_prefix_root(digest)
        logger.info("LoRA %s: prefix-cache root %s", name, digest.hex()[:16])

    def unregister(self, name: str) -> None:
        with self._lock:
            self._roots.pop(name, None)

    def root(self, name: str) -> bytes | None:
        return self._roots.get(name)
