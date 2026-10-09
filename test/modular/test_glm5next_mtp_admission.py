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


@pytest.mark.parametrize("name", ["glm5_next_tp8.yaml", "glm5_next_tp8_mtp.yaml",
                                  "glm5_next_tp8_longctx.yaml"])
def test_shipped_configs_admit_one_request_per_kda_slot(name):
    # past the slots a prefill waits in the worker, and its wait broke the decode
    # speculation chain every other step
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / name).read_text())
    assert cfg["max_concurrent_requests"] == cfg["resources"]["kda_state"]["max_slots"] - 1


def test_bucket_overrides_reach_the_config():
    m = Glm5NextModel("x", tokenizer_mode="byte", prefill_token_buckets=[128, 256, 512],
                      prefill_capture_batch_sizes=[1])
    assert m.config.prefill_token_buckets == [128, 256, 512]
    assert m.config.prefill_capture_batch_sizes == [1]


def test_eager_prefill_is_not_held_to_the_captured_rows():
    from mstar.model.glm5_next.config import Glm5NextModelConfig

    sub = object.__new__(Glm5NextLLMSubmodule)
    sub.config = Glm5NextModelConfig.reduced()
    sub.config.prefill_graphs = True
    sub.config.moe_quant_kernel = "auto"
    if sub._captured_prefill_rows() is None:
        pytest.skip("no fused KDA on this host")
    sub._prefill_captured = False  # get_cuda_graph_configs declared none
    assert sub._captured_prefill_rows() is None
