"""Voice registry: the bundled Kokoro voice packs and blends of them.

A voice pack is a ``[510, 1, 256]`` tensor with one style vector per phoneme
count; a request's style is the row for its chunk's phoneme-string length. A
blend is a weighted sum of packs, spelled ``af_bella+af_sky`` (equal weights),
``af_bella(2)+af_sky(1)`` or ``af_bella-am_adam(0.5)`` (Kokoro-FastAPI's syntax;
weights are normalized by their absolute sum), or ``af_bella,af_sky`` (the
``kokoro`` package's comma mean).
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

_TERM = re.compile(r"^\s*([A-Za-z0-9_.\-]+?)\s*(?:\(\s*([0-9]*\.?[0-9]+)\s*\))?\s*$")


class VoiceRegistry:
    def __init__(self, voices_dir: str | Path, pack_rows: int, style_dim: int):
        self._dir = Path(voices_dir)
        self._pack_rows = pack_rows
        self._style_dim = style_dim
        self._packs: dict[str, torch.Tensor] = {}
        self._blends: dict[str, torch.Tensor] = {}
        if not self._dir.is_dir():
            raise FileNotFoundError(f"Kokoro voices directory not found: {self._dir}")
        self._names = sorted(p.stem for p in self._dir.glob("*.pt"))
        if not self._names:
            raise FileNotFoundError(f"No voice packs (*.pt) in {self._dir}")

    @property
    def names(self) -> list[str]:
        """The bundled voices, sorted."""
        return list(self._names)

    def __contains__(self, name: str) -> bool:
        return name in self._packs or (self._dir / f"{name}.pt").is_file()

    def pack(self, name: str) -> torch.Tensor:
        """``[pack_rows, 2 * style_dim]`` style table of one bundled voice."""
        pack = self._packs.get(name)
        if pack is None:
            path = self._dir / f"{name}.pt"
            if not path.is_file():
                raise ValueError(f"Unknown Kokoro voice {name!r}; available: {', '.join(self._names)}")
            pack = torch.load(path, map_location="cpu", weights_only=True).float().reshape(self._pack_rows, -1)
            if pack.shape[-1] != 2 * self._style_dim:
                raise ValueError(
                    f"Voice pack {name!r} has style width {pack.shape[-1]}, expected {2 * self._style_dim}"
                )
            self._packs[name] = pack
        return pack

    @staticmethod
    def parse_blend(spec: str) -> list[tuple[str, float]]:
        """``"af_bella(2)+af_sky-am_adam(0.5)"`` -> ``[(af_bella, 2), (af_sky, 1), (am_adam, -0.5)]``."""
        if not spec or not spec.strip():
            raise ValueError("Voice must not be empty")
        terms: list[tuple[str, float]] = []
        sign = 1.0
        for part in re.split(r"([+,\-])", spec.replace(" ", "")):
            if part in ("+", ","):
                sign = 1.0
            elif part == "-":
                sign = -1.0
            elif part:
                match = _TERM.match(part)
                if match is None:
                    raise ValueError(f"Cannot parse voice term {part!r} in {spec!r}")
                name, weight = match.group(1), match.group(2)
                terms.append((name, sign * (float(weight) if weight else 1.0)))
        if not terms:
            raise ValueError(f"Cannot parse voice {spec!r}")
        return terms

    def resolve(self, spec: str) -> torch.Tensor:
        """Style table for a voice or blend spec, cached by spec string."""
        table = self._blends.get(spec)
        if table is None:
            terms = self.parse_blend(spec)
            total = sum(abs(w) for _, w in terms)
            if total == 0:
                raise ValueError(f"Voice blend {spec!r} has zero total weight")
            table = sum((w / total) * self.pack(name) for name, w in terms)
            self._blends[spec] = table
        return table

    def language_of(self, spec: str) -> str:
        """Kokoro voice names start with their language code (``af_heart`` -> ``a``)."""
        return self.parse_blend(spec)[0][0][0].lower()

    def style(self, spec: str, num_phonemes: int) -> torch.Tensor:
        """``[2 * style_dim]`` style vector for a chunk of ``num_phonemes`` phoneme characters."""
        row = min(max(num_phonemes, 1), self._pack_rows) - 1
        return self.resolve(spec)[row]
