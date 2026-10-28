"""The DiT scaffold's VAE decode compiles in the mode its compile level names
(``decode_utils.VAE_COMPILE_MODES``): max-autotune at -O1 and -O2, none at -O0."""

from types import SimpleNamespace

import pytest
import torch

from mstar.engine import accelerator_graph_runner
from mstar.model.components.diffusion import decode_utils


@pytest.fixture
def at_level(monkeypatch):
    def use(level: int) -> None:
        monkeypatch.setenv("MSTAR_COMPILE_LEVEL", str(level))
        accelerator_graph_runner.compile_level.cache_clear()

    yield use
    accelerator_graph_runner.compile_level.cache_clear()


@pytest.mark.parametrize(
    ("level", "mode"),
    [(0, "default"), (1, "max-autotune-no-cudagraphs"), (2, "max-autotune-no-cudagraphs")],
)
def test_the_vae_decode_compiles_in_its_levels_mode(monkeypatch, at_level, level, mode):
    seen = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: seen.append(kwargs) or fn)
    at_level(level)
    decode_utils.compile_vae_decode(SimpleNamespace(decode=lambda latents: latents))
    assert seen == [{"fullgraph": False, "dynamic": False, "mode": mode}]
