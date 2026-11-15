"""Which weights an adapter name stands for, as the prefix cache keys them.

A request on an adapter fills KV pages that differ from the base model's for the same
tokens, so its prompt's key chain hangs from a root that names the adapter's weights
(``adapter_prefix_root``). The API server takes it from the adapter's digest when it
selects the adapter for a request (``ServedModels``); nothing about an adapter travels
per step.
"""

from __future__ import annotations

import hashlib

import torch

from mstar.engine.resources.kv.keys import fingerprint

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


def adapter_prefix_root(digest: bytes) -> bytes:
    """The root an adapter's key chains hang from.

    The digest alone, not the name: KV depends on the weights, so two names for the
    same weights share pages, and one name reloaded with other weights does not.
    """
    return fingerprint(_NAMESPACE, digest)
