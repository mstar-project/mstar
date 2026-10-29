"""Reading LoRA checkpoints: shared by the static merge and the per-request adapter pool.

Accepted checkpoint layouts (``.safetensors``):

* diffusers / PEFT: ``[transformer.]<module>.lora_A.weight`` + ``.lora_B.weight`` (also the
  ``base_model.model.`` prefix), optionally ``<module>.alpha``; without an alpha the scaling
  comes from an ``adapter_config.json`` / ``config.json`` next to the weights when there is one
  (``lora_alpha`` / ``r``, per-module ``alpha_pattern`` / ``rank_pattern``, and ``use_rslora``
  for ``lora_alpha / sqrt(r)``, as PEFT applies them), else it is ``1`` (``lora_alpha = r``).
* Kohya-style naming ``lora_down`` / ``lora_up`` for A / B.
* model-native layouts (e.g. the BFL ``double_blocks.N.img_attn.qkv`` keys of FLUX.2) through a
  model-supplied ``convert_keys`` hook that rewrites them to the reference module paths.

``<module>`` is a reference-library module path; the model's checkpoint remap (the same one the
weight loader uses) and its stacked-shard rules turn it into the native parameter and, for
fused projections, the shard it lands on (a ``LoraSpec``'s ``module_map``, and the static
merge's ``resolve_lora_module``).
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

import torch

logger = logging.getLogger(__name__)

_PREFIXES = ("transformer.", "base_model.model.", "diffusion_model.")


@dataclass(frozen=True)
class LoraFile:
    """One adapter checkpoint: a safetensors path and its strength."""

    path: str
    scale: float = 1.0

    @classmethod
    def parse(cls, raw) -> "LoraFile":
        if isinstance(raw, str):
            return cls(path=raw)
        return cls(path=str(raw["path"]), scale=float(raw.get("scale", 1.0)))


class LoraLayer(NamedTuple):
    """One module's adapter: ``W += scale * alpha / r * b @ a``."""

    a: torch.Tensor  # [r, in]
    b: torch.Tensor  # [out, r]
    alpha: float


@dataclass
class LoraAdapter:
    """``module path -> LoraLayer`` in reference-library naming."""

    layers: dict[str, LoraLayer] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.layers)


def load_lora_file(path: str, device="cpu") -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(path, device=str(device))


KeyConverter = Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]

_SIDECAR_CONFIGS = ("adapter_config.json", "config.json")


def _pattern_value(patterns: dict, module: str):
    """PEFT's ``get_pattern_key``: the value of the first pattern that names ``module`` or
    a dotted suffix of it (``q_proj``, ``layers.0.self_attn.q_proj``); None if none does."""
    for pattern, value in patterns.items():
        if re.match(rf"(.*\.)?({pattern})$", module):
            return value
    return None


@dataclass(frozen=True)
class SidecarScaling:
    """``(module, rank) -> alpha`` from a PEFT adapter config: ``lora_alpha`` (None:
    ``alpha = r``), unless an ``alpha_pattern`` names the module; ``rslora`` folds PEFT's
    ``alpha / sqrt(r)`` into the alpha. A module a ``rank_pattern`` names must be that rank."""

    lora_alpha: float | None
    rslora: bool
    alpha_pattern: dict
    rank_pattern: dict

    def __call__(self, module: str, rank: int) -> float:
        want = _pattern_value(self.rank_pattern, module)
        if want is not None and int(want) != rank:
            # what the adapter holds, so a client may be told
            raise ValueError(
                f"LoRA module {module!r} is rank {rank}, but its config's rank_pattern says {want}"
            )
        value = _pattern_value(self.alpha_pattern, module)
        if value is None:
            value = rank if self.lora_alpha is None else self.lora_alpha
        return float(value) * (math.sqrt(rank) if self.rslora else 1.0)


def sidecar_default_alpha(path: str) -> Callable[[str, int], float] | None:
    """``(module, rank) -> alpha`` reproducing PEFT's scaling for an adapter whose safetensors has no
    alpha keys.

    PEFT reads ``lora_alpha`` (and ``use_rslora``) from the adapter's config, so ``load_lora_weights`` applies
    ``lora_alpha / r`` — or ``lora_alpha / sqrt(r)`` for rank-stabilized adapters — where the bare weights
    would suggest ``1``; a module an ``alpha_pattern`` names takes that alpha instead, and one a
    ``rank_pattern`` names has that rank (which its weights must have). Looks for ``adapter_config.json`` then
    ``config.json`` next to ``path`` and accepts the training-config spellings too (``alpha``,
    ``lora_rank`` / ``rank``). None when there is nothing to read.
    """
    import json
    from pathlib import Path

    for name in _SIDECAR_CONFIGS:
        candidate = Path(path).parent / name
        if not candidate.is_file():
            continue
        try:
            config = json.loads(candidate.read_text())
        except ValueError:
            continue
        if not isinstance(config, dict):
            continue
        lora_alpha = config.get("lora_alpha", config.get("alpha"))
        alpha_pattern = config.get("alpha_pattern") or {}
        rank_pattern = config.get("rank_pattern") or {}
        if lora_alpha is None and not alpha_pattern:
            continue
        rslora = bool(config.get("use_rslora", False))
        logger.info(
            "LoRA %s: scaling from %s (lora_alpha %s, rslora %s, %d alpha / %d rank patterns)",
            path, name, lora_alpha, rslora, len(alpha_pattern), len(rank_pattern),
        )
        return SidecarScaling(lora_alpha, rslora, alpha_pattern, rank_pattern)
    return None


def normalize_lora_key(key: str) -> str:
    """A raw checkpoint key in reference-library naming: known prefixes stripped, Kohya's
    ``lora_down`` / ``lora_up`` renamed to ``lora_A`` / ``lora_B``."""
    for prefix in _PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    key = key.replace(".lora_down.", ".lora_A.").replace(".lora_up.", ".lora_B.")
    # PEFT's LoRA on an embedding: A [r, vocab] and B [hidden, r], the LoRA of a linear
    # over the token's one-hot, which an embedding target reads as such
    if key.endswith(".lora_embedding_A") or key.endswith(".lora_embedding_B"):
        key = key[:-len("_embedding_A")] + "_" + key[-1] + ".weight"
    return key


_AB_KEY = re.compile(r"^(.*)\.lora_([AB])\.weight$")
_ALPHA_KEY = re.compile(r"^(.*)\.alpha$")


def parse_lora_key(key: str) -> tuple[str, str]:
    """``(module, "A" | "B" | "alpha")`` of a normalized key; raises on anything else."""
    m = _AB_KEY.match(key)
    if m:
        return m.group(1), m.group(2)
    m = _ALPHA_KEY.match(key)
    if m:
        return m.group(1), "alpha"
    raise ValueError(_unsupported(key))


# what a PEFT checkpoint holds besides plain LoRA, and why it cannot be served per request
_UNSUPPORTED = (
    ("lora_magnitude_vector", "a DoRA adapter (use_dora=True), whose magnitude rescales "
     "the merged weight per output row"),
    ("dora_scale", "a DoRA adapter in Kohya's naming, whose magnitude rescales the merged "
     "weight per output row"),
    ("modules_to_save", "a full copy of a module (PEFT's modules_to_save), which replaces "
     "the base weights rather than adding to them"),
    ("original_module", "a full copy of a module (PEFT's modules_to_save), which replaces "
     "the base weights rather than adding to them"),
    ("trainable_tokens", "trained token embeddings (PEFT's trainable_token_indices), which "
     "replace rows of the embedding"),
    ("lora_A.bias", "a bias on the LoRA factors (lora_bias=True)"),
    ("lora_B.bias", "a bias on the LoRA factors (lora_bias=True)"),
)


def _unsupported(key: str) -> str:
    for marker, what in _UNSUPPORTED:
        if marker in key:
            return f"LoRA key {key!r} is {what}; only plain LoRA (lora_A / lora_B) is served"
    return (
        f"unrecognized LoRA key {key!r}: not a lora_A / lora_B / alpha tensor (a full "
        "weight, a classification head, or another PEFT method such as OFT?)"
    )


def normalize_lora_state_dict(
    state_dict: dict[str, torch.Tensor], convert_keys: KeyConverter | None = None,
    default_alpha: Callable[[str, int], float] | None = None,
) -> LoraAdapter:
    """Group raw LoRA tensors into per-module ``(A, B, alpha)`` triples in reference-library naming."""
    sd = {normalize_lora_key(key): value for key, value in state_dict.items()}
    if convert_keys is not None:
        sd = convert_keys(sd)
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    alphas: dict[str, float] = {}
    for key, value in sd.items():
        module, kind = parse_lora_key(key)
        if kind == "alpha":
            alphas[module] = float(value.item())
        else:
            pairs.setdefault(module, {})[kind] = value
    adapter = LoraAdapter()
    for module, ab in pairs.items():
        if set(ab) != {"A", "B"}:
            raise ValueError(f"LoRA module {module!r} is missing lora_{'B' if 'A' in ab else 'A'}")
        a, b = ab["A"], ab["B"]
        rank = a.shape[0]
        if b.shape[1] != rank:
            raise ValueError(f"LoRA module {module!r}: A is rank {rank} but B has {b.shape[1]} columns")
        fallback = default_alpha(module, rank) if default_alpha is not None else float(rank)
        adapter.layers[module] = LoraLayer(a=a, b=b, alpha=alphas.get(module, fallback))
    return adapter


def read_lora_file(path: str, convert_keys: KeyConverter | None = None) -> LoraAdapter:
    """``normalize_lora_state_dict`` over a safetensors file, with its sidecar's scaling."""
    return normalize_lora_state_dict(
        load_lora_file(path), convert_keys=convert_keys, default_alpha=sidecar_default_alpha(path),
    )

