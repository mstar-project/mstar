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


@pytest.mark.parametrize("name", ["glm5_next_tp8.yaml", "glm5_next_tp8_mtp.yaml"])
def test_shipped_configs_admit_one_request_per_kda_slot(name):
    # past the slots a prefill waits in the worker, and its wait broke the decode
    # speculation chain every other step
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / name).read_text())
    assert cfg["max_concurrent_requests"] == cfg["resources"]["kda_state"]["max_slots"] - 1
