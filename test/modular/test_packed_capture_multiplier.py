"""A packed capture whose plan carries every input token under several
combined labels (batched classifier-free guidance: cond + uncond in one plan)
declares ``total_tokens_multiplier``. Buckets are then keyed by plan tokens,
the stand-in rows get input tokens, and bucket selection scales the batch's
input tokens by the candidate's multiplier, so the resources size their
per-bucket buffers for the whole plan (the position resource used to assert
"plan label 'cfg' carries 2048 tokens but its captured bucket holds 1024")."""

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.cuda_graph_config import PackedCudaGraphConfig
from mstar.engine.cuda_graph_runner import CudaGraphRunner
from mstar.engine.resources import BucketKey
from mstar.model.submodule_base import ARNodeInputs


def _packed(multiplier: int) -> PackedCudaGraphConfig:
    return PackedCudaGraphConfig(
        capture_graph_walk="prefill",
        capture_token_lengths=[256, 1024],
        make_node_input=lambda n: ARNodeInputs(input_ids=torch.zeros(n, dtype=torch.long), input_seq_len=n),
        total_tokens_multiplier=multiplier,
    )


def test_buckets_are_keyed_by_plan_tokens_and_rows_get_input_tokens():
    cfg = _packed(2)
    assert cfg.get_total_tokens(bs=4) == [512, 2048]
    rows = cfg.get_node_inputs(bs=4, num_tokens=2048)  # the 1024-input-token bucket
    assert [r.input_seq_len for r in rows] == [256, 256, 256, 256]
    plain = _packed(1)
    assert plain.get_total_tokens(bs=4) == [256, 1024]
    assert [r.input_seq_len for r in plain.get_node_inputs(bs=1, num_tokens=1024)] == [1024]
    with pytest.raises(ValueError):
        _packed(0)


def _runner_with(buckets: dict[BucketKey, PackedCudaGraphConfig]) -> CudaGraphRunner:
    runner = object.__new__(CudaGraphRunner)
    runner._buckets = {
        key: SimpleNamespace(config=cfg, slots=[object()], config_idx=0) for key, cfg in buckets.items()
    }
    return runner


def test_selection_scales_the_batch_tokens_by_the_bucket_multiplier():
    guided, plain = _packed(2), _packed(1)
    runner = _runner_with({
        BucketKey("prefill", bs=1, num_tokens=512, cg_key_info=True): guided,
        BucketKey("prefill", bs=1, num_tokens=2048, cg_key_info=True): guided,
        BucketKey("prefill", bs=1, num_tokens=256, cg_key_info=False): plain,
        BucketKey("prefill", bs=1, num_tokens=1024, cg_key_info=False): plain,
    })
    # 300 input tokens need 600 plan tokens under guidance: the 512 bucket is too small
    assert runner.select_bucket("prefill", bs=1, num_tokens=300, cg_key_info=True).num_tokens == 2048
    assert runner.select_bucket("prefill", bs=1, num_tokens=200, cg_key_info=True).num_tokens == 512
    assert runner.select_bucket("prefill", bs=1, num_tokens=1025, cg_key_info=True) is None
    # without guidance the same request counts once
    assert runner.select_bucket("prefill", bs=1, num_tokens=300, cg_key_info=False).num_tokens == 1024
    assert runner.select_bucket("prefill", bs=1, num_tokens=200, cg_key_info=False).num_tokens == 256
