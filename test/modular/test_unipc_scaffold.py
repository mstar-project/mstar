"""The scaffold's UniPC against diffusers' scheduler, and a batch of rows against
each row alone. CPU only; Wan's end-to-end oracle is ``test_wan22_reference_equivalence``."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers import UniPCMultistepScheduler

from mstar.model.components.diffusion.denoise_loop import UniPCDenoiseLoopSubmodule
from mstar.model.components.diffusion.flow_match import FlowMatchSchedule
from mstar.model.components.diffusion.unipc import (
    LAST_SAMPLE,
    MODEL_OUTPUTS,
    SOLVER_STATE,
    UniPCState,
    initial_state,
    make_unipc_tables,
    unipc_step,
)
from mstar.model.cosmos3.submodules import _UniPCSampler, native_flow_sigmas

BASE = {
    "num_train_timesteps": 1000, "solver_order": 2, "prediction_type": "flow_prediction", "predict_x0": True,
    "solver_type": "bh2", "lower_order_final": True, "use_flow_sigmas": True, "final_sigmas_type": "zero",
    "timestep_spacing": "linspace",
}
# the three schedules these checkpoints serve
SCHEDULES = {
    "wan22": ({**BASE, "flow_shift": 5.0, "use_karras_sigmas": False}, None),
    "cosmos3_native_flow": ({**BASE, "flow_shift": 12.0, "use_karras_sigmas": False}, native_flow_sigmas),
    "cosmos3_karras": (
        {**BASE, "flow_shift": 1.0, "use_karras_sigmas": True, "sigma_min": 0.147, "sigma_max": 200.0}, None,
    ),
}


def _reference(config, sigmas, steps) -> UniPCMultistepScheduler:
    scheduler = UniPCMultistepScheduler.from_config(config)
    if sigmas is None:
        scheduler.set_timesteps(steps, device="cpu")
    else:
        scheduler.set_timesteps(steps, device="cpu", sigmas=sigmas(steps, 1000))
    return scheduler


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("shape", [(1, 4, 2, 3, 3), (1, 17)], ids=["video", "packed"])
@pytest.mark.parametrize("steps", [1, 2, 3, 8, 20])
def test_a_schedule_is_bitwise_the_reference(schedule, dtype, shape, steps):
    """Every step of a schedule, from the same inputs, in lockstep with diffusers."""
    config, sigmas = SCHEDULES[schedule]
    reference = _reference(config, sigmas, steps)
    sampler = _UniPCSampler(_reference(config, sigmas, steps), torch.device("cpu"))
    gen = torch.Generator().manual_seed(steps)
    ours = theirs = torch.randn(shape, generator=gen, dtype=torch.float32).to(dtype)
    state = initial_state(ours)
    for k in range(steps):
        output = torch.randn(shape, generator=gen, dtype=torch.float32).to(dtype)
        theirs = reference.step(output, reference.timesteps[k], theirs, return_dict=False)[0]
        ours, state = sampler.step(output, ours, k, state)
        assert torch.equal(ours, theirs), f"step {k}"
        assert ours.dtype == dtype


def test_wan_tables_match_the_reference_config():
    config, _ = SCHEDULES["wan22"]
    reference = _reference(config, None, 20)
    sigmas, timesteps = make_unipc_tables(20, config["flow_shift"])
    assert torch.equal(reference.sigmas, sigmas) and torch.equal(reference.timesteps, timesteps)


def test_the_first_step_reads_no_state():
    """Zeros are never read at iteration 0, so any state gives the same step."""
    sigmas, _ = make_unipc_tables(8, 5.0)
    sample, output = torch.ones(1, 4), torch.full((1, 4), 0.25)
    zeros = UniPCState(**dict(zip(("model_outputs", "last_sample"), initial_state(sample).values(), strict=True)))
    junk = UniPCState(model_outputs=torch.full((2, 1, 4), 7.0), last_sample=torch.full((1, 4), 7.0))
    assert torch.equal(
        unipc_step(output, sample, zeros, sigmas, sigmas, 0, 8).sample,
        unipc_step(output, sample, junk, sigmas, sigmas, 0, 8).sample,
    )


class _Loop(UniPCDenoiseLoopSubmodule):
    """The smallest loop on the UniPC base: rows of ``[1, 4]`` and a linear model."""

    def __init__(self):
        super().__init__(loop_name="loop", max_batch_size=4)
        self.forward_rows = []

    def get_device(self):
        return torch.device("cpu")

    def bucket_key_for(self, fwd_info):
        return "key"

    def schedule_for(self, fwd_info, bucket_key):
        sigmas, timesteps = make_unipc_tables(fwd_info.step_metadata["steps"], 5.0)
        return FlowMatchSchedule(sigmas=sigmas, timesteps=timesteps, mu=None)

    def seed_latents(self, fwd_info, bucket_key, generator):
        return torch.randn(1, 4, generator=generator)

    def request_inputs(self, fwd_info, inputs, bucket_key):
        return {}

    def num_tokens(self, bucket_key):
        return 4

    def model_output(self, engine_inputs, bucket_key, latents, timestep):
        self.forward_rows.append(latents.shape[0])
        return latents * 0.5 + timestep.view(-1, 1, 1) * 1e-3


def _info(rid, steps, k):
    return SimpleNamespace(
        request_id=rid, rid_handle=rid, random_seed=hash(rid) % 1000, step_metadata={"steps": steps},
        dynamic_loop_iter_counts={"loop": k},
    )


def _step(sub, rows):
    """One engine pass over ``{rid: (info, carried)}``."""
    prepared = [sub.prepare_inputs("walk", info, carried) for info, carried in rows.values()]
    engine_inputs = SimpleNamespace(
        request_ids=list(rows), per_request_info={rid: row[0] for rid, row in rows.items()},
    )
    return sub.forward_batched("walk", engine_inputs, **sub.preprocess("walk", engine_inputs, prepared))


def _solo(rid, steps):
    sub, carried, seen = _Loop(), {}, []
    for k in range(steps):
        carried = _step(sub, {rid: (_info(rid, steps, k), carried)})[rid]
        seen.append(carried)
    return seen


def test_rows_at_different_steps_share_a_forward():
    """Three schedules overlapping in one forward step exactly as each does alone:
    a batch at different ramp positions, and rows finishing at different iterations."""
    want = {rid: _solo(rid, steps) for rid, steps in (("a", 6), ("b", 4), ("c", 2))}
    sub = _Loop()
    start = {"a": 0, "b": 1, "c": 2}  # the iteration each row joins the batch at
    steps = {"a": 6, "b": 4, "c": 2}
    carried = {rid: {} for rid in steps}
    for tick in range(6):
        rows = {
            rid: (_info(rid, steps[rid], tick - start[rid]), carried[rid])
            for rid in steps if 0 <= tick - start[rid] < steps[rid]
        }
        out = _step(sub, rows)
        for rid, row in out.items():
            carried[rid] = row
            k = tick - start[rid]
            for name in (sub.LATENTS, *SOLVER_STATE):
                assert torch.equal(row[name][0], want[rid][k][name][0]), (rid, k, name)
    assert max(sub.forward_rows) == 3


def test_a_loop_seeds_and_carries_its_solver_state():
    sub = _Loop()
    info = _info("r", 3, 0)
    prepared = sub.prepare_inputs("walk", info, {})
    assert set(prepared.tensor_inputs) == {"latents", MODEL_OUTPUTS, LAST_SAMPLE, "timestep"}
    assert prepared.tensor_inputs[MODEL_OUTPUTS].shape == (2, 1, 4)
    assert not np.any(prepared.tensor_inputs[LAST_SAMPLE].numpy())
    # past iteration 0 a missing edge is a routing bug, not a fresh loop
    with pytest.raises(RuntimeError, match="missing at iteration 1"):
        sub.prepare_inputs("walk", _info("r", 3, 1), {})
