"""CPU tests for the DiT scaffold's ``DenoiseLoopSubmodule`` with a toy model.

Structural behaviour only — no weights, no GPU: seeded noise at iteration 0, the step
index from the engine's loop counter, the async-overshoot veto, equal-shape batching,
stacked preprocess and per-row outputs, the stop boundary, the CUDA-graph bucket key
and the ragged-attention step declaration.
"""

from __future__ import annotations

import sys

import pytest
import torch

sys.path.insert(0, ".")

from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.model.components.diffusion.denoise_loop import LATENTS, DenoiseLoopSubmodule  # noqa: E402
from mstar.model.components.diffusion.flow_match import FlowMatchConfig, FlowMatchSchedule, euler_step  # noqa: E402
from mstar.model.submodule_base import ModelInputsFromEngine  # noqa: E402

LOOP = "denoise_loop"
WALK = "image_gen"


class ToyDenoise(DenoiseLoopSubmodule):
    """Velocity = -latents scaled by a learned-free constant; shape key = token count."""

    def __init__(self, **kwargs):
        super().__init__(loop_name=LOOP, **kwargs)
        self.scale = torch.nn.Parameter(torch.tensor(0.5))  # gives the module a device
        self.calls: list[tuple] = []

    def shape_key_for(self, fwd_info):
        return (int(fwd_info.step_metadata["tokens"]), 3)

    def schedule_for(self, fwd_info, shape_key):
        steps = int(fwd_info.step_metadata["num_inference_steps"])
        return FlowMatchSchedule.build(FlowMatchConfig(), steps, shape_key[0])

    def seed_latents(self, fwd_info, shape_key, generator):
        return torch.randn(shape_key[0], 4, generator=generator)

    def request_inputs(self, fwd_info, inputs, shape_key):
        return {"cond": inputs["cond"][0]}

    def num_tokens(self, shape_key):
        return shape_key[0] + shape_key[1]

    def capture_request_inputs(self, shape_key, device):
        return {LATENTS: torch.zeros(shape_key[0], 4, device=device), "cond": torch.zeros(3, device=device)}

    def denoise(self, engine_inputs, shape_key, latents, timestep, sigma, sigma_next, **cond):
        self.calls.append((tuple(latents.shape), tuple(timestep.shape), tuple(sigma.shape), tuple(cond["cond"].shape)))
        velocity = -latents * self.scale
        return euler_step(latents, velocity, sigma, sigma_next)


def _info(rid: str, k: int, steps: int = 4, tokens: int = 8, seed: int = 0) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=rid, graph_walk=WALK, fwd_index=k, random_seed=seed, max_tokens=0,
        step_metadata={"tokens": tokens, "num_inference_steps": steps},
        dynamic_loop_iter_counts={LOOP: k},
    )


def _inputs(latents=None):
    d = {"cond": [torch.zeros(3)]}
    if latents is not None:
        d[LATENTS] = [latents]
    return d


def test_iteration_zero_seeds_from_request_seed_and_is_repeatable():
    sub = ToyDenoise()
    a = sub.prepare_inputs(WALK, _info("r0", 0, seed=7), _inputs())
    b = sub.prepare_inputs(WALK, _info("r1", 0, seed=7), _inputs())
    c = sub.prepare_inputs(WALK, _info("r2", 0, seed=8), _inputs())
    assert torch.equal(a.tensor_inputs[LATENTS], b.tensor_inputs[LATENTS])
    assert not torch.equal(a.tensor_inputs[LATENTS], c.tensor_inputs[LATENTS])
    assert a.input_seq_len == 8 + 3 and a.resource_step_info == (8, 3)
    state = sub.request_state("r0")
    assert state["num_steps"] == 4 and state["schedule"].num_steps == 4
    # step scalars come off the device-resident schedule
    assert torch.equal(a.tensor_inputs["sigma"], state["sigmas"][0:1])
    assert torch.equal(a.tensor_inputs["timestep"], state["timesteps"][0:1])


def test_later_iterations_take_the_loop_back_latents_and_step_k():
    sub = ToyDenoise()
    sub.prepare_inputs(WALK, _info("r0", 0), _inputs())
    x = torch.ones(8, 4)
    out = sub.prepare_inputs(WALK, _info("r0", 2), _inputs(x))
    assert out.tensor_inputs[LATENTS] is x
    sched = sub.request_state("r0")["schedule"]
    assert torch.equal(out.tensor_inputs["sigma"], sched.sigmas[2:3])
    assert torch.equal(out.tensor_inputs["sigma_next"], sched.sigmas[3:4])


def test_overshoot_iteration_is_vetoed():
    sub = ToyDenoise()
    sub.prepare_inputs(WALK, _info("r0", 0, steps=2), _inputs())
    assert sub.prepare_inputs(WALK, _info("r0", 2, steps=2), _inputs(torch.ones(8, 4))) is None


@pytest.mark.parametrize("steps,k,expect", [(1, 0, True), (4, 2, False), (4, 3, True), (4, 4, True)])
def test_check_stop_boundary(steps, k, expect):
    sub = ToyDenoise()
    sub.prepare_inputs(WALK, _info("r0", 0, steps=steps), _inputs())
    assert (LOOP in sub.check_stop("r0", _info("r0", k, steps=steps), {})) is expect
    assert sub.check_stop("unknown", _info("unknown", k), {}) == set()


def test_can_batch_only_equal_shapes():
    sub = ToyDenoise()
    a = sub.prepare_inputs(WALK, _info("a", 0, tokens=8), _inputs())
    b = sub.prepare_inputs(WALK, _info("b", 1, tokens=8, steps=6), _inputs(torch.ones(8, 4)))
    c = sub.prepare_inputs(WALK, _info("c", 0, tokens=12), _inputs())
    assert sub.can_batch(None, [a, b])          # same shape, different step / step count
    assert not sub.can_batch(None, [a, c])      # different shape
    assert not sub.can_batch(None, [a])         # single request runs the plain forward
    assert sub.max_batch_size(WALK) == 8


def test_preprocess_stacks_and_forward_batched_splits_rows():
    sub = ToyDenoise()
    a = sub.prepare_inputs(WALK, _info("a", 0, tokens=8), _inputs())
    b = sub.prepare_inputs(WALK, _info("b", 1, tokens=8, steps=6), _inputs(torch.ones(8, 4)))
    engine_inputs = ModelInputsFromEngine(request_ids=["a", "b"], per_request_info={})
    kwargs = sub.preprocess(WALK, engine_inputs, [a, b])
    assert kwargs[LATENTS].shape == (2, 8, 4) and kwargs["cond"].shape == (2, 3)
    assert kwargs["sigma"].shape == (2, 1, 1) and kwargs["timestep"].shape == (2,)
    assert kwargs["shape_key"] == (8, 3)
    out = sub.forward_batched(WALK, engine_inputs, **kwargs)
    assert set(out) == {"a", "b"}
    assert out["a"][LATENTS][0].shape == (8, 4)
    # row b took its own sigma pair (step 1 of a 6-step schedule)
    sched_b = sub.request_state("b")["schedule"]
    expected_b = euler_step(torch.ones(8, 4), -torch.ones(8, 4) * 0.5, sched_b.sigmas[1], sched_b.sigmas[2])
    assert torch.equal(out["b"][LATENTS][0], expected_b)
    assert sub.calls[-1] == ((2, 8, 4), (2,), (2, 1, 1), (2, 3))
    # the single-request path goes through forward and returns the row directly
    single = sub.forward(WALK, ModelInputsFromEngine(request_ids=["a"], per_request_info={}),
                         **sub.preprocess(WALK, engine_inputs, [a]))
    assert single[LATENTS][0].shape == (8, 4)


def test_cg_key_info_and_declare_step_agree():
    sub = ToyDenoise(attn_resource_key="dit_attn")
    infos = {"a": _info("a", 0), "b": _info("b", 3)}
    assert sub.cg_key_info(WALK, infos) == (8, 3)
    infos["c"] = _info("c", 0, tokens=16)
    assert sub.cg_key_info(WALK, infos) is None
    # padding rows (no step metadata) do not vote
    pad = CurrentForwardPassInfo(request_id="pad", graph_walk=WALK, fwd_index=0, random_seed=0, max_tokens=1)
    assert sub.cg_key_info(WALK, {"a": _info("a", 0), "pad": pad}) == (8, 3)
    a = sub.prepare_inputs(WALK, _info("a", 0), _inputs())
    b = sub.prepare_inputs(WALK, _info("b", 0), _inputs())
    step = sub.declare_step(WALK, ["a", "b"], [a, b])
    assert step.cg_key_info == (8, 3)
    assert [(s.request_id, s.label, s.span) for s in step.segments] == [("a", "main", 11), ("b", "main", 11)]
    assert set(step.steps) == {"dit_attn"} and step.steps["dit_attn"].causal is False
    assert ToyDenoise().declare_step(WALK, ["a"], [a]) is None  # no resource declared -> nothing to plan


def test_cuda_graph_configs_one_bucket_per_shape():
    sub = ToyDenoise(capture_shapes=[(WALK, (8, 3)), (WALK, (16, 3))], capture_batch_sizes=(4, 1))
    configs = sub.get_cuda_graph_configs(torch.device("cpu"))
    assert [c.additional_key_info for c in configs] == [(8, 3), (16, 3)]
    cfg = configs[0]
    assert cfg.capture_graph_walk == WALK and cfg.capture_batch_sizes == [1, 4]
    assert cfg.single_request_inputs.input_seq_len == 11
    assert cfg.single_request_inputs.resource_step_info == (8, 3)
    assert set(cfg.single_request_inputs.tensor_inputs) == {LATENTS, "cond", "sigma", "sigma_next", "timestep"}
    assert cfg.get_total_tokens(4) == [44]
    assert cfg.caps_eager_batch_size is False and cfg.compile is False
    assert ToyDenoise().get_cuda_graph_configs(torch.device("cpu")) == []
