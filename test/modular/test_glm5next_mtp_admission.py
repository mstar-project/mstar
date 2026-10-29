"""MTP verifies its drafts through the sampler (the engine's draft-verify): a temperature is
served, by rejection sampling; a repetition penalty, which the verify can't apply, is refused
by the sampler spec at the conductor, as a 400, before the request reaches any rank."""
import pytest

from mstar.model.glm5_next.config import SAMPLER
from mstar.model.glm5_next.glm5_next_model import Glm5NextModel
from mstar.model.glm5_next.submodules import Glm5NextLLMSubmodule


def _mtp_model() -> Glm5NextModel:
    return Glm5NextModel("x", tokenizer_mode="byte", mtp_num_draft_tokens=3)


def _validate(m: Glm5NextModel, kwargs: dict) -> None:
    """What the conductor does with the request's sampling settings."""
    spec = next(s for s in m.get_node_resources() if s.resource_key == SAMPLER)
    m.get_request_resource_configs({}, kwargs)[SAMPLER].validate(spec)


@pytest.mark.parametrize("kwargs", [{"repetition_penalty": 1.1}, {"repetition_penalty": 0.9},
                                    {"temperature": 0.7, "repetition_penalty": 1.1}])
def test_a_penalty_is_refused_under_mtp(kwargs):
    with pytest.raises(ValueError, match="can't apply repetition_penalty"):
        _validate(_mtp_model(), kwargs)
    # without MTP the sampler applies it
    _validate(Glm5NextModel("x", tokenizer_mode="byte"), kwargs)


@pytest.mark.parametrize("kwargs", [{}, {"temperature": 0.0}, {"temperature": 0.7, "top_k": 20}])
def test_sampling_is_served_under_mtp(kwargs):
    m = _mtp_model()
    ids = m.process_prompt("hi", ["text"], ["text"], **kwargs).new_input_tensors["text_inputs"][0]
    assert ids.tolist() == [104, 105]
    _validate(m, kwargs)
    # the sampler verifies, at a temperature by rejection sampling
    spec = next(s for s in m.get_node_resources() if s.resource_key == SAMPLER)
    assert spec.draft_verify and spec.enable_sampled_verify


def test_an_explicit_zero_penalty_is_refused():
    # `penalty or 1.0` read 0 as unset, and the sampler divided by it
    with pytest.raises(ValueError, match="repetition_penalty"):
        _validate(_mtp_model(), {"repetition_penalty": 0.0})


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
    # the reduced config's 16 is below the fused KDA prefill's head dims (kda_triton.supported)
    sub.config.linear_head_dim = 32
    sub.config.prefill_graphs = True
    sub.config.moe_quant_kernel = "auto"
    if sub._captured_prefill_rows() is None:
        pytest.skip("no Triton for the fused KDA kernels")
    sub._prefill_captured = False  # get_accelerator_graph_configs declared none
    assert sub._captured_prefill_rows() is None
