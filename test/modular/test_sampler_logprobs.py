"""Per-token log-probs: the values against ``log_softmax``, the request
field's validation against the node's ``SamplerSpec``, and the sampler's
rows of them (eager and graph sampler) that reach the host."""

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.engine import Engine
from mstar.engine.resources.sampler import utils
from mstar.engine.resources.sampler.config import (
    SAMPLED_LOGPROB,
    SamplerSpec,
    SamplerStep,
    SamplingReqConfig,
)
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import SlotLease, StepContext
from mstar.model.submodule_base import BatchedModelOutput

V = 50


def _logits():
    torch.manual_seed(0)
    return torch.randn(3, V, dtype=torch.bfloat16) * 4


@pytest.mark.parametrize("config, error", [
    (dict(return_logprobs=1), "return_logprobs must be true or false"),
    (dict(return_logprobs=True, logprobs_mode="processed"), "logprobs_mode must be 'raw'"),
])
def test_logprobs_fields_are_validated(config, error):
    with pytest.raises(ValueError, match=error):
        SamplingReqConfig(**config).validate()


def test_a_deployment_turns_logprobs_on():
    spec = SamplerSpec(resource_key="sampler", nodes={"lm"}, vocab_size=V)
    SamplingReqConfig().validate(spec)
    with pytest.raises(ValueError, match="does not report logprobs"):
        SamplingReqConfig(return_logprobs=True).validate(spec)
    with pytest.raises(ValueError, match="enable_logprobs must be true or false"):
        spec.apply_yaml_overrides(enable_logprobs="yes")
    with pytest.raises(TypeError):
        spec.apply_yaml_overrides(enable_logprob=True)
    spec.apply_yaml_overrides(enable_logprobs=True)
    SamplingReqConfig(return_logprobs=True).validate(spec)


# -- the sampler's rows ------------------------------------------------------
#
# The draw itself needs a GPU kernel; argmax stands in for it, which leaves
# the scoring and its row mask, the part under test, as they run.


@pytest.fixture(autouse=True)
def _argmax_draw(monkeypatch):
    monkeypatch.setattr(utils, "sample_tokens", lambda logits, **kw: logits.argmax(-1))
    monkeypatch.setattr(utils, "sample_accelerator_graphable_gpu", lambda logits, *a, **kw: logits.argmax(-1))


def _resource(graph: bool = False, enable: bool = True) -> SamplerResource:
    res = SamplerResource(
        vocab_size=None, enable_repetion_penalty=False, device=torch.device("cpu"),
        enable_logprobs=enable,
    )
    if graph:
        res.build_accelerator_graph_buffers([SimpleNamespace(slot=0)], max_bs=4, max_seq_len=1)
    return res


def _step(res, rids, logits, kept_rids=None, padded_bs=None):
    """Plan, sample and commit one step; the rows ``step_host_rows`` hands over."""
    lease = None if padded_bs is None else SlotLease(slot=0, bucket=SimpleNamespace(bs=padded_bs))
    ctx = StepContext(request_ids=tuple(rids), graph_walk="w", slot=0, capture=False, slot_lease=lease)
    step = SamplerStep(apply_penalty=False, kept_rids=kept_rids)
    res.plan(step, ctx)
    res.sample(list(rids), logits)
    res.commit(step, ctx)
    return res.step_host_rows()


def _expected(logits, temperature):
    scaled = logits.float() / torch.tensor(temperature)[:, None]
    return scaled.log_softmax(-1).gather(1, logits.argmax(-1, keepdim=True)).squeeze(1)


def test_the_eager_sampler_scores_the_rows_that_asked():
    res = _resource()
    res.ingest_request(0, SamplingReqConfig(temperature=0.7, return_logprobs=True))
    res.ingest_request(1, SamplingReqConfig(temperature=0.7))
    res.ingest_request(2, SamplingReqConfig(temperature=0.0, return_logprobs=True))
    logits = _logits()

    rows = _step(res, [0, 1, 2], logits)[SAMPLED_LOGPROB]

    expected = _expected(logits, [0.7, 0.7, 1.0])  # greedy at T = 1
    torch.testing.assert_close(rows[[0, 2]], expected[[0, 2]])
    assert rows[1].isnan()


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_no_rows_reach_the_host_unless_a_request_asked(graph):
    res = _resource(graph=graph)
    padded_bs = 4 if graph else None
    logits = _logits()[:1].expand(4 if graph else 1, -1)
    res.ingest_request(0, SamplingReqConfig(temperature=1.0))
    assert _step(res, [0], logits, padded_bs=padded_bs) == {}
    res.ingest_request(1, SamplingReqConfig(temperature=1.0, return_logprobs=True))
    rows = _step(res, [0], logits, padded_bs=padded_bs)
    if graph:
        # a resident request asked, none of the step's rows: all NaN
        assert rows[SAMPLED_LOGPROB].isnan().all()
    else:
        assert rows == {}
    res.remove_request(1)
    assert _step(res, [0], logits, padded_bs=padded_bs) == {}


def test_a_node_without_the_capability_refuses_at_ingest():
    res = _resource(enable=False)
    with pytest.raises(ValueError, match="enable_logprobs=False"):
        res.ingest_request(0, SamplingReqConfig(return_logprobs=True))
    assert 0 not in res._sampler._sampling_config


def test_the_graph_sampler_writes_a_row_per_padded_slot_row():
    res = _resource(graph=True)
    res.ingest_request(0, SamplingReqConfig(temperature=0.5, return_logprobs=True))
    res.ingest_request(1, SamplingReqConfig(temperature=0.5))
    logits = torch.cat([_logits(), _logits()[:1]])  # the bucket's 4 rows

    rows = _step(res, [0, 1], logits, padded_bs=4)[SAMPLED_LOGPROB]

    assert rows.data_ptr() == res._cg_buffers.logprob[0].data_ptr(), "the slot's static row"
    torch.testing.assert_close(rows[0], _expected(logits, [0.5] * 4)[0])
    assert rows[1:].isnan().all(), "a row that did not ask, and padding rows, carry none"


def test_a_recycled_slot_does_not_inherit_the_previous_requests_flag():
    res = _resource(graph=True)
    res.ingest_request(0, SamplingReqConfig(temperature=1.0, return_logprobs=True))
    res.ingest_request(9, SamplingReqConfig(temperature=1.0, return_logprobs=True))
    logits = torch.cat([_logits(), _logits()[:1]])
    assert not _step(res, [0], logits, padded_bs=4)[SAMPLED_LOGPROB][0].isnan()
    slot = res._cg_buffers._rid_to_slot[0]
    res.remove_request(0)

    res.ingest_request(1, SamplingReqConfig(temperature=1.0))
    assert res._cg_buffers._rid_to_slot[1] == slot
    assert _step(res, [1], logits, padded_bs=4)[SAMPLED_LOGPROB][0].isnan()


# -- the engine hook ---------------------------------------------------------


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def step_host_rows(self):
        return self.rows


def test_resource_rows_join_the_stop_buffers_without_touching_the_models_dict():
    tokens, logprobs = torch.tensor([5, 6]), torch.tensor([-0.1, float("nan")])
    stop = {"new_token": tokens}
    out = BatchedModelOutput(check_stop_buffers=stop)
    mgmt = SimpleNamespace(resources={"kv": _Rows({}), "sampler": _Rows({SAMPLED_LOGPROB: logprobs})})

    Engine._add_resource_host_rows(mgmt, out)

    assert out.check_stop_buffers == {"new_token": tokens, SAMPLED_LOGPROB: logprobs}
    assert stop == {"new_token": tokens}


def test_resource_rows_with_nowhere_to_go_raise():
    rows = _Rows({SAMPLED_LOGPROB: torch.tensor([-0.1])})
    with pytest.raises(RuntimeError, match="no row-addressed stop buffers"):
        Engine._add_resource_host_rows(SimpleNamespace(resources={"sampler": rows}), BatchedModelOutput())
    # two resources under one key would keep only the last one's
    out = BatchedModelOutput(check_stop_buffers={"new_token": torch.tensor([1])})
    with pytest.raises(RuntimeError, match="already has"):
        Engine._add_resource_host_rows(SimpleNamespace(resources={"talker": rows, "code_pred": rows}), out)


def test_a_second_sample_in_one_step_raises_while_logprobs_are_on():
    res = _resource()
    res.ingest_request(0, SamplingReqConfig(return_logprobs=True))
    _step(res, [0], _logits()[:1])
    with pytest.raises(RuntimeError, match="one sample"):
        res.sample([0], _logits()[:1])
    _step(res, [0], _logits()[:1])  # the next plan starts a new step
    # without them, a node may sample as often as it likes
    plain = _resource(enable=False)
    plain.ingest_request(0, SamplingReqConfig())
    plain.sample([0], _logits()[:1])
    plain.sample([0], _logits()[:1])
