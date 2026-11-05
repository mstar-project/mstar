"""Parity of each kernel the ragged ``auto`` backend can pick for uniform segments, through the
resource's own wrapper (eager, and replayed from a CUDA graph), against fp32 attention at the image
DiTs' sizes; and what the shipped table picks on this GPU."""

from __future__ import annotations

import pytest
import torch

from mstar.engine.resources.attn.ragged.select import RAGGED_ATTENTION, SelectedRaggedPrefillWrapper, table_rows
from mstar.engine.resources.attn.workspace import Workspace
from mstar.kernels import Choice, registry
from mstar.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform.has_cuda_capability(9, 0),
    reason="cuDNN attention and FlashInfer's ragged prefill need a CUDA GPU of capability 9.0+",
)
pytest.importorskip("flashinfer")

DEVICE = torch.device("cuda:0")
D = 128
# (heads, segment tokens, segments): klein 4B and 9B at 1024^2 ([512 text | 4096 image]), Z-Image's
# main layers at a 64-token caption, its image refiner and its caption refiner
SIZES = {
    "klein4b": (24, 4608, 2), "klein9b": (32, 4608, 1), "zimage": (30, 4160, 2),
    "zimage_image": (30, 4096, 1), "zimage_caption": (30, 64, 8),
}
CHOICES = (Choice("flashinfer"), Choice("cudnn"))


@pytest.fixture(scope="module")
def workspace():
    return Workspace(torch.empty(256 << 20, dtype=torch.uint8, device=DEVICE), "test")


def _inputs(heads, seq, segments):
    g = torch.Generator(device=DEVICE).manual_seed(seq + heads)
    return [torch.randn(segments * seq, heads, D, device=DEVICE, dtype=torch.bfloat16, generator=g)
            for _ in range(3)]


def _reference(q, k, v, seq, segment=0, causal=False):
    """fp32 attention (no TF32) of one segment, and its log-sum-exp: [seq, H, D], [seq, H]."""
    rows = slice(segment * seq, (segment + 1) * seq)
    qf, kf, vf = (t[rows].double() for t in (q, k, v))
    scores = torch.einsum("shd,thd->hst", qf, kf) * D ** -0.5
    if causal:
        scores = scores.masked_fill(torch.ones(seq, seq, device=DEVICE, dtype=torch.bool).triu(1), float("-inf"))
    out = torch.einsum("hst,thd->shd", torch.softmax(scores, -1), vf)
    return out.float(), torch.logsumexp(scores, -1).T.float()


def _wrapper(workspace, choice, heads, seq, segments, graph):
    kw = dict(accelerator_graph=True, max_num_segments=segments, max_total_tokens=segments * seq) if graph else {}
    return SelectedRaggedPrefillWrapper(lambda: workspace, heads, heads, D, rows=((None, choice),), device=DEVICE,
                                        q_data_type=torch.bfloat16, **kw)


def _replayed(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    graph.replay()
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "captured"])
@pytest.mark.parametrize("choice", CHOICES, ids=lambda c: c.impl)
@pytest.mark.parametrize("size", SIZES)
def test_each_kernel_matches_fp32_at_the_dit_sizes(workspace, size, choice, graph):
    heads, seq, segments = SIZES[size]
    q, k, v = _inputs(heads, seq, segments)
    w = _wrapper(workspace, choice, heads, seq, segments, graph)
    w.plan(torch.arange(0, (segments + 1) * seq, seq, dtype=torch.int32))
    assert w.impl.split(":")[0] == choice.impl
    out = _replayed(lambda: w.run(q, k, v)) if graph else w.run(q, k, v)
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    tolerance = RAGGED_ATTENTION.impl(choice.impl).tolerance
    for segment in {0, segments - 1}:
        ref, _ = _reference(q, k, v, seq, segment)
        assert tolerance.error(out[segment * seq:(segment + 1) * seq], ref) <= 1.0, segment


@pytest.mark.parametrize("causal", [False, True], ids=["bidirectional", "causal"])
@pytest.mark.parametrize("choice", CHOICES, ids=lambda c: c.impl)
def test_each_kernel_returns_the_natural_log_sum_exp(workspace, choice, causal):
    heads, seq, segments = 30, 640, 2
    q, k, v = _inputs(heads, seq, segments)
    w = _wrapper(workspace, choice, heads, seq, segments, graph=False)
    w.plan(torch.arange(0, (segments + 1) * seq, seq, dtype=torch.int32), causal=causal)
    out, lse = w.run(q, k, v, return_lse=True)
    assert lse.shape == (segments * seq, heads) and lse.dtype == torch.float32
    for segment in range(segments):
        ref, ref_lse = _reference(q, k, v, seq, segment, causal)
        rows = slice(segment * seq, (segment + 1) * seq)
        assert RAGGED_ATTENTION.impl(choice.impl).tolerance.error(out[rows], ref) <= 1.0
        torch.testing.assert_close(lse[rows], ref_lse, atol=2e-2, rtol=1e-3)


def test_the_shipped_table_picks_cudnn_for_the_dits_on_sm103():
    if current_platform.cuda_capability(DEVICE) != (10, 3):
        pytest.skip("the shipped ragged_attention table covers sm_103")
    registry.reset_lookups()
    for heads, seq, _ in SIZES.values():
        rows = table_rows(heads, heads, D, torch.bfloat16, DEVICE)
        assert rows is not None and RAGGED_ATTENTION.row_for(rows, seq).impl == "cudnn", (heads, seq)
