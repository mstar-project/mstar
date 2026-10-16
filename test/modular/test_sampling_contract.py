"""The sampling contract, on the engine sampler.

Each knob has one meaning on both paths: a ``top_k=0`` row is unfiltered even
beside a ``top_k>0`` row, ``top_p`` keeps the nucleus, tiny temperatures are
greedy, and None never replaces a value. Draws are FlashInfer's, so a seed is
not batch-invariant."""

from __future__ import annotations

import math

import pytest
import torch

from mstar.engine.resources import SamplingReqConfig
from mstar.engine.resources.sampler.config import SamplerSpec, derive_seed
from mstar.engine.resources.sampler.utils import (
    Sampler,
    canonical_sampling_values,
    sample_tokens,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="sampler kernels need CUDA")
DEV = "cuda"


def _t(values, dtype=torch.float32):
    return torch.tensor(values, device=DEV, dtype=dtype)


@cuda
def test_top_k_zero_rows_are_unfiltered_beside_top_k_rows():
    """Was: an eager batch mixing top_k 0 and >0 drew token 0 for the 0 rows."""
    V = 64
    logits = torch.randn(4, V, device=DEV)
    logits[:, 37] += 40.0  # certain at T=1, so any other token is a bug
    out = sample_tokens(
        logits, _t([0.0, 1.0, 1.0, 1.0]), _t([0, 0, 5, 5], torch.int32), _t([1.0] * 4),
        seed=torch.arange(4, device=DEV), rand_offset=torch.zeros(4, dtype=torch.long, device=DEV),
        top_k_zero_count=2,
    )
    assert out.tolist() == [37, 37, 37, 37]


@cuda
def test_top_p_draws_stay_in_the_nucleus_and_follow_the_distribution():
    V, n = 4096, 20000
    row = torch.randn(1, V, device=DEV) * 2
    logits = row.expand(n, V).contiguous()
    seeds = torch.arange(n, device=DEV)
    zeros = torch.zeros(n, dtype=torch.long, device=DEV)
    p = row[0].softmax(-1)
    toks = sample_tokens(logits, _t([1.0] * n), _t([0] * n, torch.int32), _t([1.0] * n),
                         seed=seeds, rand_offset=zeros)
    emp = torch.bincount(toks, minlength=V).float() / n
    top = p.topk(3).indices
    assert torch.allclose(emp[top], p[top], atol=0.006)
    nucleus = sample_tokens(logits, _t([1.0] * n), _t([0] * n, torch.int32), _t([0.3] * n),
                            seed=seeds, rand_offset=zeros)
    sp, si = p.sort(descending=True)
    cut = int((sp.cumsum(0) < 0.3).sum()) + 1
    assert set(nucleus.tolist()) <= set(si[:cut].tolist())



def test_tiny_temperatures_are_greedy_and_greedy_turns_filters_off():
    assert canonical_sampling_values(1e-50, 50, 0.5, 0.1) == (0.0, 0, 1.0, 0.0)
    assert canonical_sampling_values(0.0, 50, 0.5, 0.1) == (0.0, 0, 1.0, 0.0)
    assert canonical_sampling_values(0.7, 50, 0.5, 0.1) == (0.7, 50, 0.5, 0.1)
    assert canonical_sampling_values(0.7, 2**40, 1.0, 0.0) == (0.7, 0, 1.0, 0.0)


def test_none_keeps_the_current_value():
    sampler = Sampler(device=torch.device("cpu"))
    sampler.add_request("r")
    sampler.set_config("r", temperature=0.9, top_k=40, top_p=0.8)
    sampler.set_config("r", temperature=None, top_k=None, top_p=None, min_p=None,
                       repetition_penalty=None)
    cfg = sampler._sampling_config["r"]
    assert (cfg.temperature, cfg.top_k, cfg.top_p, cfg.min_p, cfg.repetition_penalty) == (
        0.9, 40, 0.8, 0.0, 1,
    )



def test_validate_refuses_what_the_node_cannot_do():
    plain = SamplerSpec(resource_key="s", nodes={"n"}, vocab_size=10, enable_repetion_penalty=False)
    SamplingReqConfig(repetition_penalty=1, min_p=0.0).validate(plain)
    with pytest.raises(ValueError, match="repetition_penalty"):
        SamplingReqConfig(repetition_penalty=1.2).validate(plain)
    with pytest.raises(ValueError, match="min_p"):
        SamplingReqConfig(min_p=0.1).validate(plain)
    capable = SamplerSpec(resource_key="s", nodes={"n"}, vocab_size=10, enable_min_p=True)
    SamplingReqConfig(repetition_penalty=1.2, min_p=0.1).validate(capable)
    with pytest.raises(ValueError, match="penalize_prompt"):
        SamplingReqConfig(penalize_prompt="no").validate()


def test_each_sampler_of_a_request_gets_its_own_stable_seed():
    assert derive_seed(7, "talker") == derive_seed(7, "talker")
    assert derive_seed(7, "talker") != derive_seed(7, "thinker")
    assert 0 <= derive_seed(-(2**63), "x") < 2**63
    cfg = SamplingReqConfig()
    cfg.apply_conductor_config(seed=7, resource_key="talker")
    assert cfg.seed == derive_seed(7, "talker")



# ── SamplingReqConfig.validate ranges ───────────────────────────────────────

def test_defaults_and_unset_fields_pass():
    SamplingReqConfig().validate()
    # None means "the sampler's default"; the wire codec round-trips it
    SamplingReqConfig(temperature=None, top_k=None, top_p=None, min_p=None,
                      repetition_penalty=None, ignore_eos=None).validate()


@pytest.mark.parametrize("kwargs", [
    dict(temperature=0), dict(temperature=1), dict(temperature=2.5), dict(temperature=1e6),
    dict(top_p=1), dict(top_p=1e-6), dict(top_k=0), dict(top_k=50),
    dict(min_p=0), dict(min_p=1), dict(repetition_penalty=1e-3), dict(repetition_penalty=1e9),
    dict(ignore_eos=True),
])
def test_in_range_values_pass(kwargs):
    SamplingReqConfig(**kwargs).validate()


@pytest.mark.parametrize("kwargs, name", [
    (dict(temperature="hot"), "temperature"),
    (dict(temperature="0.7"), "temperature"),
    (dict(temperature=-1), "temperature"),
    (dict(temperature=math.nan), "temperature"),
    (dict(temperature=math.inf), "temperature"),
    (dict(temperature=True), "temperature"),
    (dict(top_p=0), "top_p"),
    (dict(top_p=-1), "top_p"),
    (dict(top_p=2), "top_p"),
    (dict(top_p=math.nan), "top_p"),
    (dict(min_p=-0.1), "min_p"),
    (dict(min_p=1.5), "min_p"),
    (dict(repetition_penalty=0), "repetition_penalty"),
    (dict(repetition_penalty=-2), "repetition_penalty"),
    (dict(repetition_penalty=math.inf), "repetition_penalty"),
    (dict(top_k=-1), "top_k"),
    (dict(top_k=1.5), "top_k"),
    (dict(top_k="5"), "top_k"),
    (dict(top_k=True), "top_k"),
    (dict(ignore_eos="yes"), "ignore_eos"),
])
def test_bad_values_raise_value_error(kwargs, name):
    with pytest.raises(ValueError, match=name):
        SamplingReqConfig(**kwargs).validate()
