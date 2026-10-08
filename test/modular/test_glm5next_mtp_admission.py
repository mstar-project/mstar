"""MTP is greedy only: a sampling request is refused as a 400 by process_prompt,
not as a 500 by the worker after it reached every rank."""
from types import SimpleNamespace

import pytest

from mstar.model.glm5_next.config import SAMPLER
from mstar.model.glm5_next.glm5_next_model import Glm5NextModel
from mstar.model.glm5_next.submodules import Glm5NextLLMSubmodule


def _mtp_model() -> Glm5NextModel:
    return Glm5NextModel("x", tokenizer_mode="byte", mtp_num_draft_tokens=3)


@pytest.mark.parametrize("kwargs", [
    {"temperature": 0.7}, {"repetition_penalty": 1.1}, {"repetition_penalty": 0.0},
])
def test_process_prompt_refuses_sampling_under_mtp(kwargs):
    with pytest.raises(ValueError, match="greedy only"):
        _mtp_model().process_prompt("hi", ["text"], ["text"], **kwargs)


def test_process_prompt_takes_greedy_under_mtp():
    m = _mtp_model()
    assert m.process_prompt("hi", ["text"], ["text"])["text_inputs"][0].tolist() == [104, 105]
    m.process_prompt("hi", ["text"], ["text"], temperature=0.0, repetition_penalty=1.0)


def test_the_worker_refuses_an_explicit_zero_penalty():
    # `penalty or 1.0` read 0 as unset, and the sampler divided by it
    sub = object.__new__(Glm5NextLLMSubmodule)
    info = SimpleNamespace(request_id="r", resource_configs={
        SAMPLER: SimpleNamespace(temperature=0.0, repetition_penalty=0.0)})
    with pytest.raises(RuntimeError, match="greedy only"):
        sub._check_mtp_sampling(info)


@pytest.mark.parametrize("kernel", ["Triton", "fused", "auto "])
def test_an_unknown_moe_kernel_is_refused(kernel):
    # it resolved to the uncapturable reference loop: every step eager, /health fine
    from mstar.model.glm5_next.components.moe import Glm5NextSparseMoeBlock
    from mstar.model.glm5_next.config import Glm5NextModelConfig

    cfg = Glm5NextModelConfig.reduced_fp8()
    cfg.moe_quant_kernel = kernel
    with pytest.raises(ValueError, match="moe_quant_kernel"):
        Glm5NextSparseMoeBlock(cfg)

