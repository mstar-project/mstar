"""``RaggedPrefillWrapper``: cacheless varlen self-attention, eager and under
CUDA-graph capture.

The graph-mode cases are the load-bearing ones: they pin that a wrapper captured
once at a bucket ceiling can be re-planned and replayed under shorter, raggeder
layouts — including ones padded out with zero-length segments — which is what
lets the engine bucket encoder graphs instead of capturing one per exact layout.
"""

import pytest
import torch
import torch.nn.functional as F

from mstar.utils.flashinfer_utils import (
    SUPPORTED_HEAD_DIMS,
    RaggedPrefillWrapper,
    padded_head_dim,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ragged attention requires CUDA"
)

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
NUM_HEADS = 16
# 72 is what Qwen3-Omni's AuT and the SigLIP2-style ViTs use, and it is NOT a
# FlashInfer-supported head dim — so every case here also exercises the pad.
HEAD_DIM = 72
MAX_SEGMENTS = 4
MAX_TOKENS = 512
# bf16 attention against an fp32 reference; observed max err is ~8e-3.
TOL = 3e-2


def _workspace() -> torch.Tensor:
    return torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)


def _cu_seqlens(seg_lens: list[int]) -> torch.Tensor:
    cu = [0]
    for seg_len in seg_lens:
        cu.append(cu[-1] + seg_len)
    return torch.tensor(cu, dtype=torch.int32)


def _qkv(total_tokens: int, seed: int = 0):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return tuple(
        torch.randn(
            total_tokens, NUM_HEADS, HEAD_DIM,
            dtype=DTYPE, device=DEVICE, generator=gen,
        )
        for _ in range(3)
    )


def _reference(q, k, v, cu_seqlens, causal=False):
    """Per-segment SDPA at the true head dim, accumulated in fp32."""
    scale = HEAD_DIM ** -0.5
    out = torch.zeros_like(q)
    cu = cu_seqlens.tolist()
    for start, end in zip(cu[:-1], cu[1:], strict=True):
        if end <= start:
            continue
        qs, ks, vs = (
            t[start:end].transpose(0, 1).unsqueeze(0).float() for t in (q, k, v)
        )
        o = F.scaled_dot_product_attention(qs, ks, vs, scale=scale, is_causal=causal)
        out[start:end] = o.squeeze(0).transpose(0, 1).to(out.dtype)
    return out


def _assert_close(got, want):
    err = (got.float() - want.float()).abs().max().item()
    assert err < TOL, f"max_abs_err={err} exceeds {TOL}"


# --------------------------------------------------------------------------- #
# head_dim padding
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("head_dim", "expected"),
    [(64, 64), (72, 128), (128, 128), (129, 256), (256, 256)],
)
def test_padded_head_dim_rounds_up(head_dim, expected):
    assert padded_head_dim(head_dim) == expected


def test_padded_head_dim_rejects_oversized():
    with pytest.raises(ValueError, match="exceeds the largest supported"):
        padded_head_dim(SUPPORTED_HEAD_DIMS[-1] + 1)


def test_sm_scale_defaults_to_true_head_dim():
    """Not the padded one — FlashInfer's own default would use the padded dim."""
    wrapper = RaggedPrefillWrapper(
        workspace_buffer=_workspace(),
        num_qo_heads=NUM_HEADS, num_kv_heads=NUM_HEADS, head_dim=HEAD_DIM,
        device=DEVICE, q_data_type=DTYPE,
    )
    assert wrapper.padded_head_dim == 128
    assert wrapper.sm_scale == pytest.approx(HEAD_DIM ** -0.5)


# --------------------------------------------------------------------------- #
# eager
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "seg_lens",
    [[128, 96, 40], [256], [7, 1, 300], [64] * 8],
    ids=["mixed", "single", "very_ragged", "uniform_8"],
)
def test_eager_matches_reference(seg_lens):
    wrapper = RaggedPrefillWrapper(
        workspace_buffer=_workspace(),
        num_qo_heads=NUM_HEADS, num_kv_heads=NUM_HEADS, head_dim=HEAD_DIM,
        device=DEVICE, q_data_type=DTYPE,
    )
    cu = _cu_seqlens(seg_lens)
    q, k, v = _qkv(sum(seg_lens))
    wrapper.plan(cu)
    _assert_close(wrapper.run(q, k, v), _reference(q, k, v, cu))
    assert wrapper.num_segments == len(seg_lens)


def test_eager_causal_matches_reference():
    wrapper = RaggedPrefillWrapper(
        workspace_buffer=_workspace(),
        num_qo_heads=NUM_HEADS, num_kv_heads=NUM_HEADS, head_dim=HEAD_DIM,
        device=DEVICE, causal=True, q_data_type=DTYPE,
    )
    seg_lens = [128, 96, 40]
    cu = _cu_seqlens(seg_lens)
    q, k, v = _qkv(sum(seg_lens))
    wrapper.plan(cu)
    _assert_close(wrapper.run(q, k, v), _reference(q, k, v, cu, causal=True))


def test_eager_accepts_device_cu_seqlens():
    """Works, just costs a sync — the encoders build cu_seqlens on device today."""
    wrapper = RaggedPrefillWrapper(
        workspace_buffer=_workspace(),
        num_qo_heads=NUM_HEADS, num_kv_heads=NUM_HEADS, head_dim=HEAD_DIM,
        device=DEVICE, q_data_type=DTYPE,
    )
    seg_lens = [128, 96, 40]
    cu = _cu_seqlens(seg_lens)
    q, k, v = _qkv(sum(seg_lens))
    wrapper.plan(cu.to(DEVICE))
    _assert_close(wrapper.run(q, k, v), _reference(q, k, v, cu))


# --------------------------------------------------------------------------- #
# CUDA graph mode
# --------------------------------------------------------------------------- #


def _graph_wrapper() -> RaggedPrefillWrapper:
    return RaggedPrefillWrapper(
        workspace_buffer=_workspace(),
        num_qo_heads=NUM_HEADS, num_kv_heads=NUM_HEADS, head_dim=HEAD_DIM,
        max_num_segments=MAX_SEGMENTS, max_total_tokens=MAX_TOKENS,
        device=DEVICE, use_cuda_graph=True, q_data_type=DTYPE,
    )


def _capture(wrapper):
    """Capture ``run`` over full-bucket static buffers, as an engine runner would."""
    static = tuple(
        torch.zeros(MAX_TOKENS, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        for _ in range(3)
    )
    wrapper.plan(_cu_seqlens([MAX_TOKENS // MAX_SEGMENTS] * MAX_SEGMENTS))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            wrapper.run(*static)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = wrapper.run(*static)
    torch.cuda.synchronize()
    return graph, static, static_out


def _replay(wrapper, graph, static, static_out, seg_lens, seed=0):
    total = sum(seg_lens)
    q, k, v = _qkv(total, seed=seed)
    cu = _cu_seqlens(seg_lens)
    wrapper.plan(cu)
    for buf, real in zip(static, (q, k, v), strict=True):
        buf.zero_()
        buf[:total].copy_(real)
    graph.replay()
    torch.cuda.synchronize()
    return static_out[:total], _reference(q, k, v, cu)


@pytest.mark.parametrize(
    "seg_lens",
    [
        [128, 128, 128, 128],   # the captured layout
        [100, 60, 30, 20],      # shorter, every segment used
        [100, 60, 30, 0],       # one zero-length pad segment
        [190, 20, 0, 0],        # two
        [7, 0, 0, 0],           # one real segment, rest padding
    ],
    ids=["as_captured", "shorter", "one_pad_seg", "two_pad_segs", "mostly_pad"],
)
def test_graph_replay_matches_reference(seg_lens):
    wrapper = _graph_wrapper()
    graph, static, static_out = _capture(wrapper)
    got, want = _replay(wrapper, graph, static, static_out, seg_lens)
    _assert_close(got, want)


def test_graph_replay_is_stable_across_changing_layouts():
    """The bucketing case: one captured graph, many different layouts in sequence."""
    wrapper = _graph_wrapper()
    graph, static, static_out = _capture(wrapper)
    layouts = [[128, 128, 128, 128], [100, 60, 30, 0], [7, 0, 0, 0], [64, 64, 64, 64]]
    for i, seg_lens in enumerate(layouts * 2):
        got, want = _replay(wrapper, graph, static, static_out, seg_lens, seed=i)
        _assert_close(got, want)


def test_graph_mode_fewer_segments_are_padded_not_rejected():
    wrapper = _graph_wrapper()
    wrapper.plan(_cu_seqlens([10, 20]))
    assert wrapper.num_segments == 2
    # Padded out to the fixed size the captured kernel expects.
    assert wrapper._cu_host.numel() == MAX_SEGMENTS + 1
    assert wrapper._cu_host.tolist() == [0, 10, 30, 30, 30]


def test_graph_mode_rejects_too_many_segments():
    wrapper = _graph_wrapper()
    with pytest.raises(ValueError, match="segments exceeds"):
        wrapper.plan(_cu_seqlens([10] * (MAX_SEGMENTS + 1)))


def test_graph_mode_rejects_too_many_tokens():
    wrapper = _graph_wrapper()
    with pytest.raises(ValueError, match="tokens exceeds"):
        wrapper.plan(_cu_seqlens([MAX_TOKENS, 1]))


def test_graph_mode_priming_survives_a_small_first_plan():
    """FlashInfer latches max rows on the first plan; the ctor primes at the
    ceiling so planning small first doesn't cap the bucket below its own size."""
    wrapper = _graph_wrapper()
    wrapper.plan(_cu_seqlens([8, 8, 8, 8]))
    wrapper.plan(_cu_seqlens([MAX_TOKENS // MAX_SEGMENTS] * MAX_SEGMENTS))
    assert wrapper.num_segments == MAX_SEGMENTS


def test_graph_mode_requires_bucket_bounds():
    with pytest.raises(AssertionError, match="max_num_segments required"):
        RaggedPrefillWrapper(
            workspace_buffer=_workspace(),
            num_qo_heads=NUM_HEADS, num_kv_heads=NUM_HEADS, head_dim=HEAD_DIM,
            device=DEVICE, use_cuda_graph=True,
        )
