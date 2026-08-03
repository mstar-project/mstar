"""Ragged attention under CUDA-graph capture, through both stateless runners.

These are the end-to-end cases the design turns on: a graph captured once per
bucket, re-planned before each replay against a *different* varlen layout, with
the segment count padded out. Both runners are exercised against a per-segment
SDPA reference at head_dim=72, so the head-dim padding is in the loop too.
"""

import pytest
import torch
import torch.nn.functional as F

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.attention_state import RaggedAttentionConfig
from mstar.engine.cuda_graph_config import (
    BasicBatchedCudaGraphConfig,
    PiecewisePackedConfig,
)
from mstar.engine.cuda_graph_runner import (
    PiecewiseCudaGraphRunner,
    StatelessCudaGraphRunner,
)
from mstar.model.submodule_base import ARNodeInputs, NodeSubmodule

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="graph capture requires CUDA"
)

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
NUM_HEADS = 4
HEAD_DIM = 72                 # not FlashInfer-supported; pads to 128
WIDTH = NUM_HEADS * HEAD_DIM
TOL = 3e-2


def _ragged_config(**overrides) -> RaggedAttentionConfig:
    kwargs = {
        "num_qo_heads": NUM_HEADS,
        "num_kv_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "max_segments_per_request": 2,
        "max_tokens_per_request": 128,
    }
    kwargs.update(overrides)
    return RaggedAttentionConfig(**kwargs)


def _cu_seqlens(seg_lens: list[int]) -> torch.Tensor:
    cu = [0]
    for seg_len in seg_lens:
        cu.append(cu[-1] + seg_len)
    return torch.tensor(cu, dtype=torch.int32)


def _reference(x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Self-attention with q=k=v=x, per segment, accumulated in fp32."""
    qkv = x.reshape(-1, NUM_HEADS, HEAD_DIM)
    out = torch.zeros_like(qkv)
    cu = cu_seqlens.tolist()
    for start, end in zip(cu[:-1], cu[1:], strict=True):
        if end <= start:
            continue
        s = qkv[start:end].transpose(0, 1).unsqueeze(0).float()
        o = F.scaled_dot_product_attention(s, s, s, scale=HEAD_DIM ** -0.5)
        out[start:end] = o.squeeze(0).transpose(0, 1).to(out.dtype)
    return out.reshape(-1, WIDTH)


def _assert_close(got, want):
    err = (got.float() - want.float()).abs().max().item()
    assert err < TOL, f"max_abs_err={err} exceeds {TOL}"


# --------------------------------------------------------------------------- #
# PiecewiseCudaGraphRunner
# --------------------------------------------------------------------------- #

PW_TOKENS = 256
PW_BS = 2


def _attn_block(static_inputs, static_cm=None, static_attn=None, **kwargs):
    """Captured region: one varlen self-attention over the packed buffer."""
    x = static_inputs["x"]
    qkv = x.reshape(-1, NUM_HEADS, HEAD_DIM)
    out = static_attn.run(qkv, qkv, qkv)
    return {"x": out.reshape(-1, WIDTH)}


def _piecewise_runner(ragged_config, capture_fn=_attn_block):
    config = PiecewisePackedConfig(
        capture_fn=capture_fn,
        make_static_inputs=lambda shape: {
            "x": torch.zeros(shape.total_tokens, WIDTH, dtype=DTYPE, device=DEVICE)
        },
        total_tokens=[PW_TOKENS],
        capture_batch_sizes=[PW_BS],
    )
    runner = PiecewiseCudaGraphRunner(
        config=config,
        device=DEVICE,
        autocast_dtype=DTYPE,
        ragged_config=ragged_config,
    )
    runner.warmup_and_capture()
    return runner


def test_piecewise_captures_with_ragged_state():
    runner = _piecewise_runner(_ragged_config())
    assert runner.graphs, "capture produced no graphs"
    data = runner.graphs[(PW_BS, PW_TOKENS)]
    assert data.ragged_state is not None
    # segment ceiling = bs * max_segments_per_request; token ceiling = the bucket
    assert data.ragged_state.max_num_segments == PW_BS * 2
    assert data.ragged_state.max_total_tokens == PW_TOKENS


@pytest.mark.parametrize(
    "seg_lens",
    [
        [64, 64, 64, 64],   # the captured layout
        [100, 60, 40, 20],  # shorter, all segments used
        [100, 60, 40, 0],   # one zero-length pad segment
        [120, 40, 0, 0],    # two
        [16, 0, 0, 0],      # one real segment
    ],
    ids=["as_captured", "shorter", "one_pad", "two_pad", "mostly_pad"],
)
def test_piecewise_replay_matches_reference(seg_lens):
    runner = _piecewise_runner(_ragged_config())
    total = sum(seg_lens)
    cu = _cu_seqlens(seg_lens)
    x = torch.randn(total, WIDTH, dtype=DTYPE, device=DEVICE)

    out = runner.run(
        static_inputs={"x": x},
        seq_lens=[total // PW_BS, total - total // PW_BS],
        cu_seqlens=cu,
    )
    _assert_close(out["x"], _reference(x, cu))


def test_piecewise_replay_stable_across_changing_layouts():
    runner = _piecewise_runner(_ragged_config())
    layouts = [[64, 64, 64, 64], [100, 60, 40, 0], [16, 0, 0, 0]]
    for seg_lens in layouts * 2:
        total = sum(seg_lens)
        cu = _cu_seqlens(seg_lens)
        x = torch.randn(total, WIDTH, dtype=DTYPE, device=DEVICE)
        out = runner.run(
            static_inputs={"x": x},
            seq_lens=[total // PW_BS, total - total // PW_BS],
            cu_seqlens=cu,
        )
        _assert_close(out["x"], _reference(x, cu))


def test_piecewise_run_requires_cu_seqlens_when_ragged():
    """Replaying a stale plan would be a silent correctness bug."""
    runner = _piecewise_runner(_ragged_config())
    with pytest.raises(ValueError, match="pass cu_seqlens"):
        runner.run(static_inputs={"x": torch.zeros(128, WIDTH, dtype=DTYPE, device=DEVICE)},
                   seq_lens=[64, 64])


def test_piecewise_without_ragged_config_passes_no_static_attn():
    """Capture-fns that predate ragged attention keep their exact signature."""
    seen = {}

    def capture_fn(static_inputs, static_cm=None, **kwargs):
        seen["static_attn"] = "static_attn" in kwargs
        return {"x": static_inputs["x"] * 2}

    runner = _piecewise_runner(None, capture_fn=capture_fn)
    assert runner.graphs
    assert seen["static_attn"] is False
    assert runner.graphs[(PW_BS, PW_TOKENS)].ragged_state is None


def test_piecewise_without_segment_bound_skips_ragged_state():
    runner = _piecewise_runner(
        _ragged_config(max_segments_per_request=None),
        capture_fn=lambda static_inputs, static_cm=None, **kw: {"x": static_inputs["x"] * 2},
    )
    assert runner.graphs[(PW_BS, PW_TOKENS)].ragged_state is None


# --------------------------------------------------------------------------- #
# StatelessCudaGraphRunner
# --------------------------------------------------------------------------- #

SL_SEQ = 64          # tokens per request; one segment per request
SL_BS = 2


class _EncoderSubmodule(NodeSubmodule):
    """Minimal encoder-shaped submodule: one varlen segment per request."""

    disable_torch_compile = True

    def __init__(self, ragged_config: RaggedAttentionConfig | None):
        super().__init__()
        self._ragged_config = ragged_config

    def get_ragged_attention_config(self, tp_world_size: int = 1):
        return self._ragged_config

    def get_cuda_graph_configs(self, device, tp_world_size: int = 1):
        return [
            BasicBatchedCudaGraphConfig(
                capture_graph_walk="encode",
                single_request_inputs=ARNodeInputs(
                    input_seq_len=SL_SEQ,
                    tensor_inputs={
                        "x": torch.zeros(SL_SEQ, WIDTH, dtype=DTYPE, device=device)
                    },
                ),
                capture_batch_sizes=[SL_BS],
                compile=False,
            )
        ]

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        return ARNodeInputs(input_seq_len=SL_SEQ, tensor_inputs={"x": inputs["x"][0]})

    def preprocess(self, graph_walk, engine_inputs, inputs):
        # Runs OUTSIDE the captured region, at capture and at replay — so this
        # is where the layout gets planned.
        attn = engine_inputs.ragged_attention_state
        if attn is not None:
            attn.plan(_cu_seqlens([SL_SEQ] * len(inputs)))
        return {"x": torch.stack([i.tensor_inputs["x"] for i in inputs])}

    def forward_batched(self, graph_walk, engine_inputs, x, **kwargs):
        attn = engine_inputs.ragged_attention_state
        qkv = x.reshape(-1, NUM_HEADS, HEAD_DIM)
        out = attn.run(qkv, qkv, qkv).reshape(x.shape[0], SL_SEQ, WIDTH)
        return {
            rid: {"y": [out[i]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    def can_batch(self, batch, model_inputs):
        return True


def _stateless_runner(ragged_config):
    submodule = _EncoderSubmodule(ragged_config)
    runner = StatelessCudaGraphRunner(
        submodule_name="encoder",
        submodule=submodule,
        device=DEVICE,
        autocast_dtype=DTYPE,
    )
    runner.warmup_and_capture()
    return runner, submodule


def _fwd_info(rid: str) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=rid, graph_walk="encode", requires_cfg=False,
        fwd_index=0, random_seed=0, max_tokens=1, sampling_config={},
    )


def test_stateless_captures_with_ragged_state():
    runner, _ = _stateless_runner(_ragged_config(max_segments_per_request=1))
    assert runner.graphs, "capture produced no graphs"
    state = runner._ragged.get(runner._ragged_key("encode", SL_BS))
    assert state is not None
    assert state.max_num_segments == SL_BS


def test_stateless_gated_on_submodule_declaring_config():
    runner, _ = _stateless_runner(None)
    assert runner.ragged_config is None
    assert runner._ragged.get(runner._ragged_key("encode", SL_BS)) is None


def test_stateless_replay_matches_reference():
    runner, submodule = _stateless_runner(_ragged_config(max_segments_per_request=1))
    rids = ["r0", "r1"]
    xs = [torch.randn(SL_SEQ, WIDTH, dtype=DTYPE, device=DEVICE) for _ in rids]
    inputs = [
        ARNodeInputs(input_seq_len=SL_SEQ, tensor_inputs={"x": x}) for x in xs
    ]

    outputs = runner.run(
        graph_walk="encode",
        request_ids=rids,
        inputs=inputs,
        per_request_info={rid: _fwd_info(rid) for rid in rids},
        submodule=submodule,
    )

    want = _reference(torch.cat(xs), _cu_seqlens([SL_SEQ] * 2))
    for i, rid in enumerate(rids):
        _assert_close(outputs[rid]["y"][0], want[i * SL_SEQ:(i + 1) * SL_SEQ])


def test_stateless_replay_pads_partial_batch():
    """One real request replaying a bs=2 graph: the unused segment is zero-length."""
    runner, submodule = _stateless_runner(_ragged_config(max_segments_per_request=1))
    x = torch.randn(SL_SEQ, WIDTH, dtype=DTYPE, device=DEVICE)

    outputs = runner.run(
        graph_walk="encode",
        request_ids=["r0"],
        inputs=[ARNodeInputs(input_seq_len=SL_SEQ, tensor_inputs={"x": x})],
        per_request_info={"r0": _fwd_info("r0")},
        submodule=submodule,
    )

    _assert_close(outputs["r0"]["y"][0], _reference(x, _cu_seqlens([SL_SEQ])))
