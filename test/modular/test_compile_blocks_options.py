"""``compile_blocks`` compiles every graph of a block under its Inductor options, also
when an eager call in the middle (an attention, a cache write) splits the block into
graphs. CPU Inductor; no GPU."""

from __future__ import annotations

import pytest
import torch
import torch._inductor.config as inductor_config
from torch import nn
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files

from mstar.engine.accelerator_graph_runner import deterministic_inductor_options
from mstar.model.components.diffusion.compile_utils import _inductor_options, compile_blocks


class _SeenConfig(CustomGraphPass):
    """Records, per compiled graph, the Inductor flags in effect while it compiles."""

    def __init__(self):
        self.graphs = []

    def __call__(self, graph):
        self.graphs.append((inductor_config.deterministic, inductor_config.emulate_precision_casts))

    def uuid(self):
        return get_hash_for_files((__file__,))


@torch.compiler.disable
def _eager_middle(x):
    return x * 2


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(64)

    def forward(self, x):
        x = self.norm(x) + 1
        x = _eager_middle(x)
        return self.norm(x) * 3


@pytest.mark.skipif(not deterministic_inductor_options(), reason="this torch has no Inductor deterministic mode")
def test_a_block_split_by_an_eager_call_compiles_deterministic(torch_flag):
    seen = _SeenConfig()
    torch_flag(inductor_config, "post_grad_custom_post_pass", seen)
    torch_flag(inductor_config, "force_disable_caches", True)
    torch._dynamo.reset()
    block = _Block()
    compile_blocks([block])
    with torch.no_grad():
        block(torch.randn(5, 64))
    # one graph before the eager call and one after it, both deterministic and rounding as eager
    assert seen.graphs == [(True, True), (True, True)]


def test_autotune_keeps_the_level_mode_and_drops_deterministic():
    options = _inductor_options({"mode": "max-autotune-no-cudagraphs", "dynamic": False}, deterministic=False)
    assert options["max_autotune"] and options["emulate_precision_casts"]
    assert not options.get("deterministic", False)
