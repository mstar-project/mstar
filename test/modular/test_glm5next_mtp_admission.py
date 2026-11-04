"""MTP is greedy only: process_prompt refuses a sampling request as a 400, before it
reaches any rank."""
import pytest

from mstar.model.glm5_next.glm5_next_model import Glm5NextModel


def _mtp_model() -> Glm5NextModel:
    return Glm5NextModel("x", tokenizer_mode="byte", mtp_num_draft_tokens=3)


@pytest.mark.parametrize("kwargs", [
    {"temperature": 0.7}, {"repetition_penalty": 1.1}, {"repetition_penalty": 0.9},
])
def test_process_prompt_refuses_sampling_under_mtp(kwargs):
    with pytest.raises(ValueError, match="greedy only"):
        _mtp_model().process_prompt("hi", ["text"], ["text"], **kwargs)


def test_process_prompt_takes_greedy_under_mtp():
    m = _mtp_model()
    assert m.process_prompt("hi", ["text"], ["text"]).new_input_tensors["text_inputs"][0].tolist() == [104, 105]
    m.process_prompt("hi", ["text"], ["text"], temperature=0.0, repetition_penalty=1.0)


@pytest.mark.parametrize("kernel", ["Triton", "fused", "auto "])
def test_an_unknown_moe_kernel_is_refused(kernel):
    # it would resolve to the uncapturable reference loop and serve every step eager
    from mstar.model.glm5_next.components.moe import Glm5NextSparseMoeBlock
    from mstar.model.glm5_next.config import Glm5NextModelConfig

    cfg = Glm5NextModelConfig.reduced_fp8()
    cfg.moe_quant_kernel = kernel
    with pytest.raises(ValueError, match="moe_quant_kernel"):
        Glm5NextSparseMoeBlock(cfg)


@pytest.mark.parametrize("k", [-1, -3])
def test_a_negative_draft_count_is_refused(k):
    # -1 would size the cache at zero rows and fail building the graph; -3 would serve unnoticed
    with pytest.raises(ValueError, match="mtp_num_draft_tokens"):
        Glm5NextModel("x", tokenizer_mode="byte", mtp_num_draft_tokens=k)
