"""Cosmos3's distilled sampler takes its SDE noise as a tensor and its step index from the loop.

diffusers' `FlowMatchEulerDiscreteScheduler.step` draws the stochastic step's noise from a generator and counts
its own steps, so each request held a scheduler of its own. Drawn outside the step, with the loop's index, the
update is the same to the bit, one sampler serves every request of a schedule, and the draw can move to wherever
the request's generator is kept.
"""

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from mstar.model.cosmos3.config import Cosmos3Config
from mstar.model.cosmos3.submodules import Cosmos3DiTSubmodule

SIGMAS = (1.0, 0.9375, 0.8333333333333334, 0.625)


def _sampler():
    config = Cosmos3Config(compile_denoise=False, is_distilled=True, distilled_sigmas=SIGMAS)
    config.scheduler.stochastic_sampling = True
    dit = Cosmos3DiTSubmodule(transformer=None, config=config, scheduler=None)
    return dit._new_scheduler(len(SIGMAS), torch.device("cpu"))


def _velocity(k: int, shape) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(100 + k))


def test_the_distilled_step_is_diffusers_step_to_the_bit():
    sampler = _sampler()
    reference = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0, stochastic_sampling=True)
    reference.set_timesteps(sigmas=list(SIGMAS), device="cpu")
    want = torch.randn(1, 16, 3, 4, 4, generator=torch.Generator().manual_seed(0))
    got = want.clone()
    ours, theirs = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    for k in range(len(SIGMAS)):
        v = _velocity(k, want.shape)
        want = reference.step(v, reference.timesteps[k], want, generator=theirs, return_dict=False)[0]
        got, solver = sampler.step(v, got, k, {}, noise=sampler.noise(got, ours))
        assert solver == {}
        assert torch.equal(got, want), f"step {k} left diffusers' update"


def test_one_distilled_sampler_serves_interleaved_requests():
    sampler = _sampler()
    alone = {}
    for seed in (1, 2):
        x, gen = torch.ones(1, 16, 2, 4, 4) * seed, torch.Generator().manual_seed(seed)
        for k in range(len(SIGMAS)):
            x, _ = sampler.step(_velocity(k, x.shape), x, k, {}, noise=sampler.noise(x, gen))
        alone[seed] = x
    xs = {seed: torch.ones(1, 16, 2, 4, 4) * seed for seed in (1, 2)}
    gens = {seed: torch.Generator().manual_seed(seed) for seed in (1, 2)}
    for k in range(len(SIGMAS)):
        for seed in (1, 2):
            x = xs[seed]
            xs[seed], _ = sampler.step(_velocity(k, x.shape), x, k, {}, noise=sampler.noise(x, gens[seed]))
    for seed in (1, 2):
        assert torch.equal(xs[seed], alone[seed]), "a step count held in the sampler would cross two requests"
