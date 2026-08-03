"""Ragged (cacheless) attention: the FlashInfer wrapper, and both stateless
CUDA-graph runners that drive it.

The load-bearing property throughout: a graph captured once per bucket can be
re-planned before each replay against a different varlen layout, with the
segment count padded out by zero-length segments. Everything runs at
head_dim=72, which FlashInfer has no kernel for, so the pad-to-128 path is
always exercised.
"""

import pytest
import torch
import torch.nn.functional as F

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.attention_state import AttentionMode, RaggedAttentionConfig
from mstar.engine.cuda_graph_config import (
    BasicBatchedCudaGraphConfig,
    PiecewisePackedConfig,
)
from mstar.engine.cuda_graph_runner import (
    PiecewiseCudaGraphRunner,
    StatelessCudaGraphRunner,
)
from mstar.model.submodule_base import ARNodeInputs, NodeSubmodule
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
HEADS, HEAD_DIM = 4, 72
WIDTH = HEADS * HEAD_DIM
MAX_SEGMENTS, MAX_TOKENS = 4, 512
TOL = 3e-2                      # bf16 vs an fp32 reference; observed ~8e-3


def cu(seg_lens: list[int]) -> torch.Tensor:
    out = [0]
    for n in seg_lens:
        out.append(out[-1] + n)
    return torch.tensor(out, dtype=torch.int32)


def qkv(total: int, seed: int = 0):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return tuple(
        torch.randn(total, HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE, generator=gen)
        for _ in range(3)
    )


def ref(q, k, v, cu_seqlens, causal=False):
    """Per-segment SDPA at the true head dim, accumulated in fp32."""
    out = torch.zeros_like(q)
    offsets = cu_seqlens.tolist()
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        if end <= start:
            continue
        qs, ks, vs = (t[start:end].transpose(0, 1).unsqueeze(0).float() for t in (q, k, v))
        o = F.scaled_dot_product_attention(qs, ks, vs, scale=HEAD_DIM ** -0.5, is_causal=causal)
        out[start:end] = o.squeeze(0).transpose(0, 1).to(out.dtype)
    return out


def close(got, want):
    err = (got.float() - want.float()).abs().max().item()
    assert err < TOL, f"max_abs_err={err} exceeds {TOL}"


def workspace():
    return torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)


def wrapper(**overrides) -> RaggedPrefillWrapper:
    kwargs = dict(
        workspace_buffer=workspace(), num_qo_heads=HEADS, num_kv_heads=HEADS,
        head_dim=HEAD_DIM, device=DEVICE, q_data_type=DTYPE,
    )
    kwargs.update(overrides)
    return RaggedPrefillWrapper(**kwargs)


def graph_wrapper() -> RaggedPrefillWrapper:
    return wrapper(
        max_num_segments=MAX_SEGMENTS, max_total_tokens=MAX_TOKENS, use_cuda_graph=True
    )


# --- head-dim padding ------------------------------------------------------

@pytest.mark.parametrize(
    ("head_dim", "expected"), [(64, 64), (72, 128), (128, 128), (129, 256), (256, 256)]
)
def test_padded_head_dim_rounds_up(head_dim, expected):
    assert padded_head_dim(head_dim) == expected


def test_padded_head_dim_rejects_oversized():
    with pytest.raises(ValueError, match="exceeds the largest supported"):
        padded_head_dim(SUPPORTED_HEAD_DIMS[-1] + 1)


def test_sm_scale_uses_true_head_dim_not_padded():
    """FlashInfer's own default would derive it from the padded dim."""
    w = wrapper()
    assert w.padded_head_dim == 128
    assert w.sm_scale == pytest.approx(HEAD_DIM ** -0.5)


# --- eager -----------------------------------------------------------------

@pytest.mark.parametrize("seg_lens", [[128, 96, 40], [7, 1, 300]], ids=["mixed", "ragged"])
def test_eager_matches_reference(seg_lens):
    w, layout = wrapper(), cu(seg_lens)
    q, k, v = qkv(sum(seg_lens))
    w.plan(layout)
    close(w.run(q, k, v), ref(q, k, v, layout))
    assert w.num_segments == len(seg_lens)


def test_eager_causal_matches_reference():
    w, layout = wrapper(causal=True), cu([128, 96, 40])
    q, k, v = qkv(264)
    w.plan(layout)
    close(w.run(q, k, v), ref(q, k, v, layout, causal=True))


def test_eager_accepts_device_cu_seqlens():
    """Works, just costs a sync — encoders build cu_seqlens on device today."""
    w, layout = wrapper(), cu([128, 96, 40])
    q, k, v = qkv(264)
    w.plan(layout.to(DEVICE))
    close(w.run(q, k, v), ref(q, k, v, layout))


# --- wrapper under CUDA graph ----------------------------------------------

def capture_wrapper(w):
    static = tuple(
        torch.zeros(MAX_TOKENS, HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) for _ in range(3)
    )
    w.plan(cu([MAX_TOKENS // MAX_SEGMENTS] * MAX_SEGMENTS))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            w.run(*static)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = w.run(*static)
    torch.cuda.synchronize()
    return graph, static, out


def replay(w, graph, static, out, seg_lens, seed=0):
    total = sum(seg_lens)
    q, k, v = qkv(total, seed=seed)
    layout = cu(seg_lens)
    w.plan(layout)
    for buf, real in zip(static, (q, k, v), strict=True):
        buf.zero_()
        buf[:total].copy_(real)
    graph.replay()
    torch.cuda.synchronize()
    return out[:total], ref(q, k, v, layout)


@pytest.mark.parametrize(
    "seg_lens",
    [[128] * 4, [100, 60, 30, 20], [100, 60, 30, 0], [190, 20, 0, 0], [7, 0, 0, 0]],
    ids=["as_captured", "shorter", "one_pad", "two_pad", "mostly_pad"],
)
def test_graph_replay_matches_reference(seg_lens):
    w = graph_wrapper()
    close(*replay(w, *capture_wrapper(w), seg_lens))


def test_graph_replay_stable_across_changing_layouts():
    """Re-planning one captured wrapper repeatedly must not leak state."""
    w = graph_wrapper()
    graph, static, out = capture_wrapper(w)
    layouts = [[128] * 4, [100, 60, 30, 0], [7, 0, 0, 0], [64] * 4]
    for i, seg_lens in enumerate(layouts * 2):
        close(*replay(w, graph, static, out, seg_lens, seed=i))


def test_graph_mode_pads_fewer_segments():
    w = graph_wrapper()
    w.plan(cu([10, 20]))
    assert w.num_segments == 2
    assert w._cu_host.tolist() == [0, 10, 30, 30, 30]


def test_graph_mode_rejects_too_many_segments():
    with pytest.raises(ValueError, match="segments exceeds"):
        graph_wrapper().plan(cu([10] * (MAX_SEGMENTS + 1)))


def test_graph_mode_rejects_too_many_tokens():
    with pytest.raises(ValueError, match="tokens exceeds"):
        graph_wrapper().plan(cu([MAX_TOKENS, 1]))


def test_graph_mode_priming_survives_a_small_first_plan():
    """FlashInfer latches max rows on the first plan; the ctor primes at the
    ceiling so planning small first can't cap the bucket below its own size."""
    w = graph_wrapper()
    w.plan(cu([8] * 4))
    w.plan(cu([MAX_TOKENS // MAX_SEGMENTS] * MAX_SEGMENTS))
    assert w.num_segments == MAX_SEGMENTS


def test_graph_mode_requires_bucket_bounds():
    with pytest.raises(AssertionError, match="max_num_segments required"):
        wrapper(use_cuda_graph=True)


# --- PiecewiseCudaGraphRunner ----------------------------------------------

PW_TOKENS, PW_BS = 256, 2


def ragged_config(**overrides) -> RaggedAttentionConfig:
    kwargs = dict(
        num_qo_heads=HEADS, num_kv_heads=HEADS, head_dim=HEAD_DIM,
        max_segments_per_request=2, max_tokens_per_request=128,
    )
    kwargs.update(overrides)
    return RaggedAttentionConfig(**kwargs)


def attn_block(static_inputs, static_cm=None, static_attn=None, **kwargs):
    x = static_inputs["x"].reshape(-1, HEADS, HEAD_DIM)
    return {"x": static_attn.run(x, x, x).reshape(-1, WIDTH)}


def piecewise_runner(config, capture_fn=attn_block):
    runner = PiecewiseCudaGraphRunner(
        config=PiecewisePackedConfig(
            capture_fn=capture_fn,
            make_static_inputs=lambda shape: {
                "x": torch.zeros(shape.total_tokens, WIDTH, dtype=DTYPE, device=DEVICE)
            },
            total_tokens=[PW_TOKENS],
            capture_batch_sizes=[PW_BS],
        ),
        device=DEVICE, autocast_dtype=DTYPE, ragged_config=config,
    )
    runner.warmup_and_capture()
    assert runner.graphs, "capture produced no graphs"
    return runner


def test_piecewise_bucket_sizes_its_own_state():
    data = piecewise_runner(ragged_config()).graphs[(PW_BS, PW_TOKENS)]
    # Segment ceiling = bs * max_segments_per_request; token ceiling = the bucket.
    assert data.ragged_state.max_num_segments == PW_BS * 2
    assert data.ragged_state.max_total_tokens == PW_TOKENS


# Piecewise replay numerics live in test_bagel_vit_ragged_attention.py, which
# drives this runner end-to-end through a real ViT. Here we only pin the
# contract edges the integration test can't reach.


def test_piecewise_run_requires_cu_seqlens_when_ragged():
    """Replaying a stale plan would be a silent correctness bug."""
    runner = piecewise_runner(ragged_config())
    with pytest.raises(ValueError, match="pass cu_seqlens"):
        runner.run(
            static_inputs={"x": torch.zeros(128, WIDTH, dtype=DTYPE, device=DEVICE)},
            seq_lens=[64, 64],
        )


def test_piecewise_passes_no_static_attn_without_ragged_config():
    """Capture-fns predating ragged attention keep their exact signature."""
    seen = {}

    def capture_fn(static_inputs, static_cm=None, **kwargs):
        seen["static_attn"] = "static_attn" in kwargs
        return {"x": static_inputs["x"] * 2}

    runner = piecewise_runner(None, capture_fn=capture_fn)
    assert seen["static_attn"] is False
    assert runner.graphs[(PW_BS, PW_TOKENS)].ragged_state is None


@pytest.mark.parametrize(
    "config",
    [ragged_config(enabled_for=AttentionMode.EAGER), ragged_config(max_segments_per_request=None)],
    ids=["eager_only", "no_segment_bound"],
)
def test_piecewise_skips_ragged_state(config):
    runner = piecewise_runner(
        config, capture_fn=lambda static_inputs, static_cm=None, **kw: {"x": static_inputs["x"] * 2}
    )
    assert runner.graphs[(PW_BS, PW_TOKENS)].ragged_state is None


# --- StatelessCudaGraphRunner ----------------------------------------------

SL_SEQ, SL_BS = 64, 2


class _EncoderSubmodule(NodeSubmodule):
    """Encoder-shaped submodule: one varlen segment per request, two backends."""

    disable_torch_compile = True

    def __init__(self, config: RaggedAttentionConfig | None):
        super().__init__()
        self._config = config

    def get_ragged_attention_config(self, tp_world_size: int = 1):
        return self._config

    def get_cuda_graph_configs(self, device, tp_world_size: int = 1):
        return [BasicBatchedCudaGraphConfig(
            capture_graph_walk="encode",
            single_request_inputs=ARNodeInputs(
                input_seq_len=SL_SEQ,
                tensor_inputs={"x": torch.zeros(SL_SEQ, WIDTH, dtype=DTYPE, device=device)},
            ),
            capture_batch_sizes=[SL_BS],
            compile=False,
        )]

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        return ARNodeInputs(input_seq_len=SL_SEQ, tensor_inputs={"x": inputs["x"][0]})

    def preprocess(self, graph_walk, engine_inputs, inputs):
        # Runs OUTSIDE the captured region, at capture and replay — so this is
        # where the layout gets planned.
        attn = engine_inputs.ragged_attention_state
        if attn is not None:
            attn.plan(cu([SL_SEQ] * len(inputs)))
        return {"x": torch.stack([i.tensor_inputs["x"] for i in inputs])}

    def forward_batched(self, graph_walk, engine_inputs, x, **kwargs):
        attn = engine_inputs.ragged_attention_state
        flat = x.reshape(-1, HEADS, HEAD_DIM)
        if attn is not None:
            flat = attn.run(flat, flat, flat)
        else:
            # Second backend, so EAGER-mode capture is a real capture, not a crash.
            s = flat.reshape(x.shape[0], SL_SEQ, HEADS, HEAD_DIM).transpose(1, 2)
            flat = F.scaled_dot_product_attention(
                s, s, s, scale=HEAD_DIM ** -0.5
            ).transpose(1, 2).reshape(-1, HEADS, HEAD_DIM)
        out = flat.reshape(x.shape[0], SL_SEQ, WIDTH)
        return {rid: {"y": [out[i]]} for i, rid in enumerate(engine_inputs.request_ids)}

    def can_batch(self, batch, model_inputs):
        return True


def stateless_runner(config):
    submodule = _EncoderSubmodule(config)
    runner = StatelessCudaGraphRunner(
        submodule_name="encoder", submodule=submodule, device=DEVICE, autocast_dtype=DTYPE,
    )
    runner.warmup_and_capture()
    assert runner.graphs, "capture produced no graphs"
    return runner, submodule


def fwd_info(rid: str) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=rid, graph_walk="encode", requires_cfg=False,
        fwd_index=0, random_seed=0, max_tokens=1, sampling_config={},
    )


def test_stateless_bucket_sizes_its_own_state():
    runner, _ = stateless_runner(ragged_config(max_segments_per_request=1))
    state = runner._ragged.get(runner._ragged_key("encode", SL_BS))
    assert state is not None and state.max_num_segments == SL_BS


@pytest.mark.parametrize(
    "config",
    [None, ragged_config(max_segments_per_request=1, enabled_for=AttentionMode.EAGER)],
    ids=["no_config", "eager_only"],
)
def test_stateless_skips_ragged_state(config):
    runner, _ = stateless_runner(config)
    assert runner._ragged.get(runner._ragged_key("encode", SL_BS)) is None


@pytest.mark.parametrize("rids", [["r0", "r1"], ["r0"]], ids=["full_batch", "padded_batch"])
def test_stateless_replay_matches_reference(rids):
    runner, submodule = stateless_runner(ragged_config(max_segments_per_request=1))
    xs = [torch.randn(SL_SEQ, WIDTH, dtype=DTYPE, device=DEVICE) for _ in rids]
    outputs = runner.run(
        graph_walk="encode",
        request_ids=rids,
        inputs=[ARNodeInputs(input_seq_len=SL_SEQ, tensor_inputs={"x": x}) for x in xs],
        per_request_info={rid: fwd_info(rid) for rid in rids},
        submodule=submodule,
    )
    flat = torch.cat(xs).reshape(-1, HEADS, HEAD_DIM)
    want = ref(flat, flat, flat, cu([SL_SEQ] * len(rids))).reshape(-1, WIDTH)
    for i, rid in enumerate(rids):
        close(outputs[rid]["y"][0], want[i * SL_SEQ:(i + 1) * SL_SEQ])
