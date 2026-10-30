"""Verify locally regenerated NVIDIA PersonaPlex reference fixtures on load."""

import hashlib
import json

import torch

from mstar.model.personaplex.config import REFERENCE_SHA


def _digest_value(value):
    if isinstance(value, torch.Tensor):
        contiguous = value.detach().cpu().contiguous()
        raw = contiguous.view(torch.uint8).numpy().tobytes()
        return [str(value.dtype), list(value.shape), hashlib.sha256(raw).hexdigest()]
    if isinstance(value, dict):
        return {key: _digest_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_digest_value(item) for item in value]
    return value


def verify_reference_fixture(data: dict) -> dict:
    """Check the oracle's source pin and canonical content hash, excluding argv."""
    if data.get("source") != "NVIDIA/personaplex" or data.get("source_sha") != REFERENCE_SHA:
        raise ValueError("PersonaPlex reference fixture has a different source or source SHA")
    content = {key: value for key, value in data.items() if key not in ("argv", "content_sha256")}
    digest = _digest_value(content)
    canonical = json.dumps(digest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != data.get("content_sha256"):
        raise ValueError("PersonaPlex reference fixture content SHA256 mismatch")
    return data


def load_binary_fixture(path):
    return verify_reference_fixture(torch.load(path, weights_only=True))
