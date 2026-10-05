"""CPU checks for sampler hints, the batched XPU ABI and host synchronization."""

import sys
from contextlib import nullcontext
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from mstar.engine.resources.sampler import utils


@pytest.fixture
def kernel(monkeypatch):
    package = ModuleType("vllm_xpu_kernels")
    package.__path__ = []
    extension = ModuleType("vllm_xpu_kernels._xpu_C")
    package._xpu_C = extension
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, extension.__name__, extension)

    def fill_tokens(output, *args):
        output.copy_(torch.arange(output.numel(), dtype=output.dtype))

    sampler = Mock(side_effect=fill_tokens)
    monkeypatch.setattr(torch.ops._xpu_C, "topk_topp_sampler", sampler, raising=False)
    return sampler


def _sample(logits, **overrides):
    batch = logits.shape[0]
    options = dict(
        temperature=torch.ones(batch), top_k=torch.zeros(batch, dtype=torch.int32),
        top_p=torch.ones(batch), repetition_penalty=1.0, seen_token_mask=None,
        run_greedy=True, top_k_zero_count=None,
        seed=torch.arange(batch, dtype=torch.int64) + 100,
        rand_offset=torch.arange(batch, dtype=torch.int64) * logits.shape[1],
    )
    options.update(overrides)
    return utils._sample_xpu(logits, **options)


@pytest.mark.parametrize("zero_count", [None, 2])
def test_explicit_rng_uses_one_batched_call_without_host_reads(kernel, monkeypatch, zero_count):
    logits = torch.tensor([[1., 3., 2.], [4., 2., 3.], [1., 2., 4.]])
    with monkeypatch.context() as guard:
        def forbidden(*args, **kwargs):
            pytest.fail("explicit XPU RNG or parameters were read back to the CPU")

        for name in ("cpu", "item", "tolist"):
            guard.setattr(torch.Tensor, name, forbidden)
        tokens = _sample(
            logits, top_k=torch.tensor([0, 2, 0], dtype=torch.int32),
            top_p=torch.tensor([1., 0.8, 0.9]), run_greedy=False,
            top_k_zero_count=zero_count,
        )
    assert tokens.tolist() == [0, 1, 2]
    kernel.assert_called_once()
    output, _, scores, top_k, top_p, mode, rng, scale = kernel.call_args.args
    assert output.shape == (3,)
    assert output.dtype == top_k.dtype == rng.dtype == torch.int64
    assert scores.dtype == torch.float32
    assert rng.is_contiguous() and rng.shape == (3, 2)
    torch.testing.assert_close(rng, torch.tensor([[100, 0], [101, 3], [102, 6]]))
    torch.testing.assert_close(top_k, torch.tensor([3, 2, 3]))
    torch.testing.assert_close(top_p, torch.tensor([1., 0.8, 0.9]))
    assert mode == "raw_logits" and scale == 1.0


def test_all_top_k_disabled_uses_unfiltered_kernel_parameter(kernel):
    _sample(torch.ones(2, 8), top_k_zero_count=2)
    assert kernel.call_args.args[3] is None


def test_no_disabled_top_k_rows_preserve_kernel_parameters(kernel):
    _sample(torch.ones(2, 8), top_k=torch.tensor([2, 4]), top_k_zero_count=0)
    torch.testing.assert_close(kernel.call_args.args[3], torch.tensor([2, 4]))


@pytest.mark.parametrize("zero_count", [0, 1, 2, None])
def test_cuda_fast_path_requires_every_top_k_row_disabled(monkeypatch, zero_count):
    flashinfer = ModuleType("flashinfer")
    tokens = torch.zeros(2, dtype=torch.int64)
    flashinfer.sampling = Mock()
    flashinfer.sampling.top_p_sampling_from_probs.return_value = tokens
    flashinfer.sampling.top_k_top_p_sampling_from_probs.return_value = tokens
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(utils, "fused_temperature_softmax", Mock(return_value=torch.ones(2, 8)))
    top_k = {0: [2, 4], 1: [0, 4], 2: [0, 0], None: [0, 4]}[zero_count]
    actual = utils._sample_cuda(
        torch.ones(2, 8), torch.ones(2), torch.tensor(top_k), torch.ones(2),
        1.0, None, False, zero_count, None, None,
    )
    torch.testing.assert_close(actual, tokens)
    if zero_count == 2:
        flashinfer.sampling.top_p_sampling_from_probs.assert_called_once()
        flashinfer.sampling.top_k_top_p_sampling_from_probs.assert_not_called()
    else:
        flashinfer.sampling.top_p_sampling_from_probs.assert_not_called()
        flashinfer.sampling.top_k_top_p_sampling_from_probs.assert_called_once()


def test_mixed_greedy_rows_keep_argmax_tie_behavior(kernel):
    tokens = _sample(
        torch.tensor([[5., 5., 1.], [3., 2., 1.], [1., 4., 4.]]),
        temperature=torch.tensor([0., 0.7, 0.]),
    )
    assert tokens.tolist() == [0, 1, 1]


def test_filters_keep_penalty_temperature_and_min_p_order(kernel):
    logits = torch.tensor([[2., -2., 0.], [1., 3., -1.]])
    _sample(
        logits, temperature=torch.tensor([2., 0.5]),
        repetition_penalty=torch.tensor([2., 1.5]),
        seen_token_mask=torch.tensor([[True, True, False], [False, True, True]]),
        min_p=torch.tensor([0., 1.]),
    )
    # Penalized -> temperature-scaled logits; min_p=1 retains only the max.
    torch.testing.assert_close(
        kernel.call_args.args[2], torch.tensor([[0.5, -2., 0.], [-float("inf"), 4., -float("inf")]]),
    )


@pytest.mark.parametrize("missing", ["seed", "rand_offset", "both"])
def test_missing_rng_fields_reserve_only_required_generator_state(kernel, monkeypatch, missing):
    monkeypatch.setattr(torch.xpu, "current_device", lambda: 0)
    monkeypatch.setattr(torch.xpu, "default_generators", (object(),))
    reserve = Mock(return_value=(777, 8))
    monkeypatch.setattr(utils, "_xpu_generator_seed_offset", reserve)
    options = {}
    if missing in ("seed", "both"):
        options["seed"] = None
    if missing in ("rand_offset", "both"):
        options["rand_offset"] = None
    _sample(torch.ones(3, 8), **options)
    reserve.assert_called_once_with(
        torch.xpu.default_generators[0], 24 if missing in ("rand_offset", "both") else 0,
    )
    rng = kernel.call_args.args[6]
    expected_seed = [777] * 3 if missing in ("seed", "both") else [100, 101, 102]
    expected_offset = [8, 16, 24] if missing in ("rand_offset", "both") else [0, 8, 16]
    torch.testing.assert_close(rng[:, 0], torch.tensor(expected_seed))
    torch.testing.assert_close(rng[:, 1], torch.tensor(expected_offset))
