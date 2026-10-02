"""GPU check for MSTAR_KIMI_SHARED_STREAM: the shared expert on a side stream
gives the single-stream result eagerly, compiled, and replayed from a CUDA
graph captured under ``fail_on_recompile`` (how the decode step runs)."""
from dataclasses import replace

import pytest
import torch

from mstar.model.kimi_k2_7.components.moe import KimiSparseMoeBlock, _SideStream
from mstar.model.kimi_k2_7.config import KimiK2Config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _block() -> KimiSparseMoeBlock:
    cfg = replace(KimiK2Config.reduced(), hidden_size=1024, moe_intermediate_size=2048, n_routed_experts=8)
    torch.manual_seed(0)
    block = KimiSparseMoeBlock(cfg).cuda().to(torch.bfloat16)
    with torch.no_grad():
        for p in block.parameters():
            p.normal_(0, 0.02)
    block.gate.weight.data = block.gate.weight.data.float()
    if getattr(block.gate, "e_score_correction_bias", None) is not None:
        block.gate.e_score_correction_bias.data.zero_()
    return block


@torch.no_grad()
def test_shared_stream_matches_single_stream_eager_compiled_and_captured():
    block = _block()
    x = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16)

    block._shared_stream = None
    ref = block(x).clone()

    block._shared_stream = _SideStream()
    eager = block(x)
    torch.testing.assert_close(eager, ref, rtol=0, atol=0)

    compiled = torch.compile(block, mode="max-autotune-no-cudagraphs", fullgraph=False, dynamic=False)
    torch.testing.assert_close(compiled(x), ref, rtol=1e-2, atol=1e-2)

    static_x = x.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            compiled(static_x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.compiler.set_stance("fail_on_recompile"), torch.cuda.graph(graph):
        static_out = compiled(static_x)
    for seed in range(3):
        torch.manual_seed(seed)
        new_x = torch.randn_like(static_x)
        static_x.copy_(new_x)
        graph.replay()
        block._shared_stream = None
        want = block(new_x)
        block._shared_stream = _SideStream()
        torch.testing.assert_close(static_out, want, rtol=1e-2, atol=1e-2)
