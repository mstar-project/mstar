"""Kimi's compile regions (``KimiK3LLMSubmodule.compile_regions``: the AttnRes reads with their norms,
each MLP block, the output read) resolve on the submodule and trace under Dynamo with dynamic shapes:
on CPU, compiled with the ``aot_eager`` backend as the engine compiles them (``Engine._compile_regions``
on an accelerator: ``torch.compile(target, dynamic=True)`` set over the attribute), a prefill and
decode steps give the eager engine's tokens and logits. The tiny release has MXFP4 experts, so the MoE
region traces the experts' custom op."""
import pytest
import torch

from mstar.model.kimi_k3.tests.test_mxfp4 import PROMPT, _engine, _greedy
from mstar.model.kimi_k3.tests.test_truncate import make_tiny_release


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    from mstar.model.kimi_k3.tests import engine_helpers as H

    H.install_cpu_stubs(monkeypatch)
    torch.set_num_threads(2)


def _compile(sub, backend: str) -> list[str]:
    """What ``Engine._compile_regions`` does to each region, with ``backend``."""
    done = []
    for path in sub.compile_regions():
        owner_path, _, name = path.rpartition(".")
        owner = sub
        for part in filter(None, owner_path.split(".")):
            owner = getattr(owner, part)
        target = getattr(owner, name)
        if isinstance(target, torch.nn.Module):
            owner, name, target = target, "forward", target.forward
        setattr(owner, name, torch.compile(target, dynamic=True, backend=backend))
        done.append(path)
    return done


def test_the_regions_cover_every_layer_and_can_be_turned_off(tmp_path):
    make_tiny_release(tmp_path)
    _, sub, _ = _engine(tmp_path)
    sub.compile_reads = True
    regions = sub.compile_regions()
    layers = sub.language_model.model.layers
    assert len(regions) == 3 * len(layers) + 1
    assert "language_model.model.layers.0.mlp" in regions  # the dense layer
    assert "language_model.model.layers.1.block_sparse_moe" in regions
    sub.compile_reads = False
    assert len(sub.compile_regions()) == len(layers)  # the MLP blocks only
    sub.compile_glue = False
    assert sub.compile_regions() == {}


def test_compiled_regions_decode_as_the_eager_engine(tmp_path):
    import torch._dynamo

    make_tiny_release(tmp_path)
    engine, sub, _ = _engine(tmp_path)
    eager_tokens, eager_logits = _greedy(engine)
    torch._dynamo.reset()
    engine, sub, _ = _engine(tmp_path)
    sub.compile_reads = True  # every region, the reads' too
    with torch._dynamo.config.patch(recompile_limit=16, error_on_recompile=False):
        assert len(_compile(sub, "aot_eager")) == len(sub.compile_regions())
        tokens, logits = _greedy(engine)
    assert torch.equal(tokens, eager_tokens)
    torch.testing.assert_close(logits, eager_logits, atol=1e-5, rtol=0)
    assert tokens.numel() == len(PROMPT) // len(PROMPT) + 3  # the prefill's token and three decode steps
