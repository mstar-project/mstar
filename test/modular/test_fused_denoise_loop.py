"""CPU tests for the DiT scaffold's fused loop (``fused_steps``): a fixed schedule run in one engine step.

The contract: one fused step gives the per-step loop's sample bit for bit, ``denoise`` sees each round's step index
and scalars, the step's inputs carry every step's scalars, a capture placeholder holds the whole schedule, and a
schedule the request sets (not the bucket) is refused.
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
STEPS = 4


class FixedToy(DenoiseLoopSubmodule):
    """A toy DiT on a schedule its bucket fixes (as FastWan's DMD and H3-Image's grid are)."""

    def __init__(self, **kwargs):
        super().__init__(loop_name=LOOP, **kwargs)
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.seen: list[tuple[int, float]] = []

    def bucket_key_for(self, fwd_info):
        return int(fwd_info.step_metadata["tokens"])

    def schedule_settings(self, fwd_info):
        return None

    def schedule_for(self, walk, settings, bucket_key):
        return FlowMatchSchedule.build(FlowMatchConfig(), STEPS, bucket_key)

    def seed_latents(self, bucket_key, generator):
        return torch.randn(bucket_key, 4, generator=generator)

    def request_inputs(self, fwd_info, inputs, bucket_key):
        return {"cond": inputs["cond"][0]}

    def num_tokens(self, bucket_key):
        return bucket_key

    def capture_request_inputs(self, bucket_key, device):
        return {LATENTS: torch.zeros(bucket_key, 4, device=device), "cond": torch.zeros(3, device=device)}

    def denoise(self, engine_inputs, bucket_key, latents, timestep, sigma, sigma_next, **cond):
        info = engine_inputs.per_request_info[engine_inputs.request_ids[0]]
        self.seen.append((self.step_index(info), float(sigma.flatten()[0])))
        velocity = -latents * self.scale + cond["cond"].sum()
        return euler_step(latents, velocity, sigma, sigma_next)


def _info(k: int, seed: int = 3) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id="r", graph_walk=WALK, fwd_index=k, random_seed=seed, max_tokens=0, rid_handle=0,
        step_metadata={"tokens": 8}, dynamic_loop_iter_counts={LOOP: k},
    )


def _step(sub, k, latents=None):
    inputs = {"cond": [torch.full((3,), 0.1)]}
    if latents is not None:
        inputs[LATENTS] = [latents]
    info = _info(k)
    node = sub.prepare_inputs(WALK, info, inputs)
    engine_inputs = ModelInputsFromEngine(request_ids=["r"], per_request_info={"r": info})
    return node, sub.forward(WALK, engine_inputs, **sub.preprocess(WALK, engine_inputs, [node]))[LATENTS][0]


def test_a_fused_step_gives_the_per_step_loops_sample_bit_for_bit():
    per_step, fused = FixedToy(), FixedToy(fused_steps=True)
    sample = None
    for k in range(STEPS):
        _, sample = _step(per_step, k, sample)
    node, fused_sample = _step(fused, 0)
    assert torch.equal(fused_sample, sample)
    # every round ran with its own step index and scalars, as the per-step loop's steps did
    assert fused.seen == per_step.seen
    assert [k for k, _ in fused.seen] == list(range(STEPS))
    # the one step's inputs carry every step's scalars
    schedule = fused.schedule_of(_info(0), 8)
    assert torch.equal(node.tensor_inputs["sigma"], schedule.sigmas[:STEPS])
    assert torch.equal(node.tensor_inputs["sigma_next"], schedule.sigmas[1:STEPS + 1])
    # and the round is not left behind for the next request's step
    assert fused._fused_round is None


def test_a_fused_loop_runs_one_iteration_of_a_schedule_its_bucket_fixes():
    fused = FixedToy(fused_steps=True)
    with pytest.raises(RuntimeError, match="fused loop runs one iteration"):
        _step(fused, 1, torch.zeros(8, 4))

    class RequestSteps(FixedToy):
        def schedule_settings(self, fwd_info):
            return STEPS

        def schedule_for(self, walk, settings, bucket_key):
            return FlowMatchSchedule.build(FlowMatchConfig(), settings, bucket_key)

    with pytest.raises(RuntimeError, match="fused loop runs one iteration"):
        _step(RequestSteps(fused_steps=True), 0)


def test_a_fused_capture_holds_every_step_of_the_buckets_schedule():
    fused = FixedToy(fused_steps=True, capture_buckets=[(WALK, 8)], capture_batch_sizes=(1,))
    (config,) = fused.get_accelerator_graph_configs(torch.device("cpu"))
    scalars = config.single_request_inputs.tensor_inputs
    assert all(scalars[name].shape == (STEPS,) for name in fused.STEP_SCALARS)


def test_fused_steps_take_no_step_cache():
    with pytest.raises(ValueError, match="fused_steps takes no step schedule or step cache"):
        FixedToy(fused_steps=True, step_cache=object())
