"""CPU tests for the DiT scaffold's ``DenoiseLoopSubmodule`` with a toy model.

Structural behaviour only — no weights, no GPU: seeded noise at iteration 0, the step
index from the engine's loop counter, the async-overshoot veto, equal-shape batching,
stacked preprocess and per-row outputs, the stop boundary, the CUDA-graph bucket key
and the ragged-attention step declaration.
"""

from __future__ import annotations

import dataclasses
import sys

import pytest
import torch

sys.path.insert(0, ".")

from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.engine.resources.convenience import RaggedAttentionCallable  # noqa: E402
from mstar.model.components.diffusion.denoise_loop import (  # noqa: E402
    LATENTS,
    DenoiseLoopSubmodule,
    StepScalar,
)
from mstar.model.components.diffusion.flow_match import (  # noqa: E402
    FlowMatchConfig,
    FlowMatchSchedule,
    euler_step,
)
from mstar.model.submodule_base import ModelInputsFromEngine  # noqa: E402

LOOP = "denoise_loop"
WALK = "image_gen"


class ToyDenoise(DenoiseLoopSubmodule):
    """Velocity = -latents scaled by a learned-free constant; shape key = token count."""

    def __init__(self, **kwargs):
        super().__init__(loop_name=LOOP, **kwargs)
        self.scale = torch.nn.Parameter(torch.tensor(0.5))  # gives the module a device
        self.calls: list[tuple] = []

    def bucket_key_for(self, fwd_info):
        return (int(fwd_info.step_metadata["tokens"]), 3)

    def schedule_for(self, fwd_info, bucket_key):
        steps = int(fwd_info.step_metadata["num_inference_steps"])
        return FlowMatchSchedule.build(FlowMatchConfig(), steps, bucket_key[0])

    def seed_latents(self, fwd_info, bucket_key, generator):
        return torch.randn(bucket_key[0], 4, generator=generator)

    def request_inputs(self, fwd_info, inputs, bucket_key):
        return {"cond": inputs["cond"][0]}

    def num_tokens(self, bucket_key):
        return bucket_key[0] + bucket_key[1]

    def capture_request_inputs(self, bucket_key, device):
        return {LATENTS: torch.zeros(bucket_key[0], 4, device=device), "cond": torch.zeros(3, device=device)}

    def denoise(self, engine_inputs, bucket_key, latents, timestep, sigma, sigma_next, **cond):
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


def test_missing_loop_back_latents_past_iteration_zero_raises():
    sub = ToyDenoise()
    sub.prepare_inputs(WALK, _info("r0", 0), _inputs())
    with pytest.raises(RuntimeError, match=r"\['latents'\] missing at iteration 2 of request r0"):
        sub.prepare_inputs(WALK, _info("r0", 2), _inputs())


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
    assert kwargs["bucket_key"] == (8, 3)
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
    sub = ToyDenoise(capture_buckets=[(WALK, (8, 3)), (WALK, (16, 3))], capture_batch_sizes=(4, 1))
    configs = sub.get_cuda_graph_configs(torch.device("cpu"))
    assert [c.additional_key_info for c in configs] == [(8, 3), (16, 3)]
    cfg = configs[0]
    assert cfg.capture_graph_walk == WALK and cfg.capture_batch_sizes == [1, 4]
    assert cfg.single_request_inputs.input_seq_len == 11
    assert cfg.single_request_inputs.resource_step_info == (8, 3)
    assert set(cfg.single_request_inputs.tensor_inputs) == {LATENTS, "cond", "sigma", "sigma_next", "timestep"}
    assert cfg.get_total_tokens(4) == [44]
    assert cfg.caps_eager_batch_size is False and cfg.compile is False
    assert cfg.input_seq_dims == {key: 0 for key in cfg.single_request_inputs.tensor_inputs}
    assert ToyDenoise().get_cuda_graph_configs(torch.device("cpu")) == []


class TwoSpanDenoise(ToyDenoise):
    """A model whose refiner layers attend over the image tokens alone."""

    def attention_segments(self, bucket_key):
        return (("image", bucket_key[0]), ("main", self.num_tokens(bucket_key)))


def test_declare_step_lists_every_attention_span_per_row():
    sub = TwoSpanDenoise(attn_resource_key="dit_attn")
    a = sub.prepare_inputs(WALK, _info("a", 0), _inputs())
    b = sub.prepare_inputs(WALK, _info("b", 1), _inputs(torch.ones(8, 4)))
    step = sub.declare_step(WALK, ["a", "b"], [a, b])
    assert [(s.request_id, s.label, s.span) for s in step.segments] == [
        ("a", "image", 8), ("a", "main", 11), ("b", "image", 8), ("b", "main", 11),
    ]
    # a capture bucket's rows carry the shape key, so their spans match the real rows'
    cfg = TwoSpanDenoise(attn_resource_key="dit_attn", capture_buckets=[(WALK, (8, 3))]).get_cuda_graph_configs(
        torch.device("cpu"),
    )[0]
    pad = cfg.single_request_inputs
    assert [(s.label, s.span) for s in sub.declare_step(WALK, ["pad"], [pad]).segments] == [("image", 8), ("main", 11)]


class UniPCDenoise(ToyDenoise):
    """A multi-step scheduler: the solver's history rides along as extra loop-back
    edges, the way wan22's UniPC carries ``unipc_model_outputs`` / ``unipc_last_sample``."""

    SOLVER_STATE = ("history",)

    def seed_loop_back(self, fwd_info, bucket_key, generator):
        return {
            **super().seed_loop_back(fwd_info, bucket_key, generator),
            "history": torch.zeros(bucket_key[0], 4),
        }

    def capture_request_inputs(self, bucket_key, device):
        return {
            **super().capture_request_inputs(bucket_key, device),
            "history": torch.zeros(bucket_key[0], 4, device=device),
        }

    def denoise(self, engine_inputs, bucket_key, latents, timestep, sigma, sigma_next, history, **cond):
        velocity = -(latents + history) * self.scale
        return {LATENTS: euler_step(latents, velocity, sigma, sigma_next), "history": velocity}


def test_every_loop_back_edge_is_seeded_then_carried():
    sub = UniPCDenoise()
    assert sub.loop_back_names == (LATENTS, "history")

    first = sub.prepare_inputs(WALK, _info("r0", 0), _inputs())
    assert set(first.tensor_inputs) >= {LATENTS, "history"}
    assert torch.equal(first.tensor_inputs["history"], torch.zeros(8, 4))

    # a later iteration takes both back from the Loop rather than re-seeding
    carried = {"cond": [torch.zeros(3)], LATENTS: [torch.ones(8, 4)], "history": [torch.full((8, 4), 2.0)]}
    later = sub.prepare_inputs(WALK, _info("r0", 1), carried)
    assert later.tensor_inputs[LATENTS] is carried[LATENTS][0]
    assert later.tensor_inputs["history"] is carried["history"][0]

    # solver state lost on the way back is an error, not a fresh seed
    with pytest.raises(RuntimeError, match=r"\['history'\] missing at iteration 1"):
        sub.prepare_inputs(WALK, _info("r0", 1), {"cond": [torch.zeros(3)], LATENTS: [torch.ones(8, 4)]})


def test_a_mapping_from_denoise_is_split_across_rows_and_edges():
    sub = UniPCDenoise()
    a = sub.prepare_inputs(WALK, _info("a", 0, tokens=8), _inputs())
    b = sub.prepare_inputs(WALK, _info("b", 0, tokens=8), _inputs())
    engine_inputs = ModelInputsFromEngine(request_ids=["a", "b"], per_request_info={})
    out = sub.forward_batched(WALK, engine_inputs, **sub.preprocess(WALK, engine_inputs, [a, b]))

    assert set(out) == {"a", "b"}
    for rid in ("a", "b"):
        assert set(out[rid]) == {LATENTS, "history"}
        assert out[rid][LATENTS][0].shape == (8, 4) and out[rid]["history"][0].shape == (8, 4)
    # the single-request path splits the same way
    single = sub.forward(WALK, ModelInputsFromEngine(request_ids=["a"], per_request_info={}),
                         **sub.preprocess(WALK, engine_inputs, [a]))
    assert set(single) == {LATENTS, "history"}


def test_capture_buckets_stage_every_loop_back_edge():
    sub = UniPCDenoise(capture_buckets=[(WALK, (8, 3))])
    tensor_inputs = sub.get_cuda_graph_configs(torch.device("cpu"))[0].single_request_inputs.tensor_inputs
    assert set(tensor_inputs) == {LATENTS, "history", "cond", "sigma", "sigma_next", "timestep"}


@dataclasses.dataclass(frozen=True)
class GuidedSchedule(FlowMatchSchedule):
    """A schedule carrying a per-step guidance scale next to the sigmas."""

    guidances: torch.Tensor = dataclasses.field(default_factory=lambda: torch.zeros(0))


class GuidedDenoise(ToyDenoise):
    """A scheduler with a per-step scalar of its own, declared rather than
    threaded through prepare_inputs / preprocess / _run by hand."""

    STEP_SCALARS = {
        **ToyDenoise.STEP_SCALARS,
        "guidance": StepScalar("guidances", broadcast=False, capture_fill=3.5),
    }

    def schedule_for(self, fwd_info, bucket_key):
        base = super().schedule_for(fwd_info, bucket_key)
        # the only thing a new scalar needs: a source on the schedule. The base
        # stages it on the device and slices it per step.
        return GuidedSchedule(
            sigmas=base.sigmas, timesteps=base.timesteps, mu=base.mu,
            guidances=torch.arange(float(base.num_steps)),
        )

    def denoise(self, engine_inputs, bucket_key, latents, timestep, sigma, sigma_next, guidance, **cond):
        self.calls.append(tuple(guidance.shape))
        return euler_step(latents, -latents * self.scale * guidance.view(-1, 1, 1), sigma, sigma_next)


def test_a_declared_step_scalar_is_sliced_reshaped_and_popped():
    sub = GuidedDenoise()
    a = sub.prepare_inputs(WALK, _info("a", 0), _inputs())
    b = sub.prepare_inputs(WALK, _info("b", 2), _inputs(torch.ones(8, 4)))
    # sliced at each row's own step index, and offset=1 still reads k + 1
    assert torch.equal(a.tensor_inputs["guidance"], torch.tensor([0.0]))
    assert torch.equal(b.tensor_inputs["guidance"], torch.tensor([2.0]))
    sched_b = sub.request_state("b")["schedule"]
    assert torch.equal(b.tensor_inputs["sigma_next"], sched_b.sigmas[3:4])

    engine_inputs = ModelInputsFromEngine(request_ids=["a", "b"], per_request_info={})
    kwargs = sub.preprocess(WALK, engine_inputs, [a, b])
    assert kwargs["guidance"].shape == (2,)  # broadcast=False stays [B]
    out = sub.forward_batched(WALK, engine_inputs, **kwargs)
    assert sub.calls[-1] == (2,)  # reached denoise by name, not left in **cond
    assert out["a"][LATENTS][0].shape == (8, 4)

    # and the capture bucket stages it at the declared fill
    cfg = GuidedDenoise(capture_buckets=[(WALK, (8, 3))]).get_cuda_graph_configs(torch.device("cpu"))[0]
    assert torch.equal(cfg.single_request_inputs.tensor_inputs["guidance"], torch.tensor([3.5]))
    # the base staged it off the schedule, so nothing had to touch the state
    assert torch.equal(sub.request_state("a")["guidances"], torch.arange(4.0))


def test_a_step_scalar_the_schedule_does_not_carry_says_so():
    """Otherwise it surfaces as an AttributeError (or, before the state was
    populated from the table, a KeyError) from inside the slicing dict."""
    class Undeclared(ToyDenoise):
        STEP_SCALARS = {**ToyDenoise.STEP_SCALARS, "cfg": StepScalar("cfg_scales")}

    with pytest.raises(AttributeError, match="STEP_SCALARS\\['cfg'\\] reads 'cfg_scales'"):
        Undeclared().prepare_inputs(WALK, _info("a", 0), _inputs())


def test_a_step_scalar_too_short_for_its_offset_says_so():
    """``timesteps`` has one entry per step, so reading k + 1 off it runs out on
    the last step -- an empty slice that would only surface as a shape error
    once preprocess tried to stack it."""
    class OffsetPastTheEnd(ToyDenoise):
        STEP_SCALARS = {
            **ToyDenoise.STEP_SCALARS,
            "next_timestep": StepScalar("timesteps", offset=1),
        }

    with pytest.raises(ValueError, match="needing 5 entries for 4 steps, but timesteps has 4"):
        OffsetPastTheEnd().prepare_inputs(WALK, _info("a", 0), _inputs())


def test_ragged_for_binds_the_label():
    class Resource:
        def run(self, q, k, v, label=None):
            return label

    sub = TwoSpanDenoise(attn_resource_key="dit_attn")
    resource = Resource()
    sub.bind_node_resources({"dit_attn": resource})
    image = sub.ragged_for("image")
    assert isinstance(image, RaggedAttentionCallable)
    assert image.attn is resource and image.label == "image"
    assert image(None, None, None) == "image"
    assert sub._ragged()(None, None, None) == "main"
    # one callable object per label, so a compiled region's identity guards hold across steps
    assert sub.ragged_for("image") is image and sub._ragged() is sub._ragged()
    # a rebinding starts over: a callable holding the old resource would attend
    # through a plan nothing writes to any more
    new_resource = Resource()
    sub.bind_node_resources({"dit_attn": new_resource})
    assert sub.ragged_for("image") is not image
    assert sub.ragged_for("image").attn is new_resource
    assert sub.ragged_for("main")(None, None, None) == "main"
    assert ToyDenoise().ragged_for("image") is None  # no resource declared -> SDPA
    unbound = TwoSpanDenoise(attn_resource_key="dit_attn")
    assert unbound.ragged_for("image") is None  # declared but not bound yet -> SDPA
