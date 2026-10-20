"""GR00T N1.7's ``denoise_step`` node: one flow-matching step of the action head.

The loop itself is the scaffold's (``DenoiseLoopSubmodule``), and the scaffold has its own
tests. What is here is the hooks -- what GR00T's step *is* -- and the places where GR00T's
convention is not the scaffold's.

The schedule comes first, because it is the one place where GR00T reads a flow-matching
schedule backwards from every image model on the scaffold. ``FlowMatchSchedule.build`` gives
the image convention: sigmas descending from 1 and timesteps ``[1000, 750, 500, 250]``.
GR00T's time runs the other way -- ``t_cont = k / 4``, conditioned on
``int(t_cont * 1000)``, with ``actions += dt * velocity`` -- so our sigmas *ascend* from 0
and ``sigma_next - sigma`` is the reference's ``dt = 1/4``. The timesteps below are derived
from the reference's own expression (``gr00t_n1d7.py:398-399``) rather than from the
implementation, and ``build``'s descending schedule is pinned beside them so that nobody
"simplifies" one into the other: both are four steps of a flow schedule, both load the
checkpoint, and both run.

Then the two counts, which are one off from each other on purpose. The DT sees **41** rows
(the state row plus 40 action rows) and the latents the loop carries are **[40, 132]**. The
step declaration is sized by the first and the trajectory by the second, so a test that lets
them be confused is the point of that section.

Then the loop state's dtype, where this file's first answer was **wrong**. It used to pin an
fp32 trajectory carried through a bf16 forward, on the claim that the reference integrates in
fp32. The reference does not. It is served bf16 (``gr00t_policy.py:111`` casts the whole model,
the checkpoint's ``model_dtype`` is ``bfloat16`` and every ``action_head.*`` tensor in it is
BF16), it draws ``actions`` at ``vl_embeds.dtype`` (``gr00t_n1d7.py:347-351``) and it updates
with a Python-float ``dt`` (``gr00t_n1d7.py:435``), so its carried trajectory is bf16 at every
step. fp32 is the better *integrator*, and that is exactly why it had to go: over four steps it
moved the trajectory 0.0243 away from the reference while the R7 block-phase off-by-one moves it
0.0391, so a tolerance wide enough to admit our own extra precision is within 1.6x of admitting
a real wiring bug. It cost the end-to-end parity bar, which is the only test that catches that
class of defect. The section now pins the carried dtype as the *weights'*, and the step as the
reference's bf16 arithmetic to the bit.

After those: the bucket key, read off ``step_metadata`` because the span lengths are needed
before their tensors arrive; the conditioning tensors, which carry **no** batch dim because
the base stacks them; the attention callables, memoised per binding because a compiled
region guards on their identity; and the step declaration, which names both DT resources and
is the first thing to read the three span labels and ``DT_XATTN_PAIRS``.

Every value test runs a **shallow** DT. The real stack is 32 blocks of 1536 x 32 x 48 --
1.12B parameters, 4.2 GiB per fp32 copy -- so the only place the real widths appear is the
last section, on ``meta``, where nothing is allocated.
"""

import dataclasses
import functools
import glob
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.attn.ragged.config import RaggedCrossAttentionStep, cross_label
from mstar.engine.resources.step import Segment
from mstar.model.components.diffusion.denoise_loop import LATENTS, DenoiseLoopSubmodule
from mstar.model.components.diffusion.flow_match import FlowMatchConfig, FlowMatchSchedule
from mstar.model.gr00t_n17.components.dit import ActionPositionEmbedding, DiT
from mstar.model.gr00t_n17.components.embodiment import (
    build_action_decoder,
    build_action_encoder,
)
from mstar.model.gr00t_n17.config import (
    DT_ATTN,
    DT_XATTN,
    DT_XATTN_PAIRS,
    SA_SPAN,
    VL_IMAGE_SPAN,
    VL_TEXT_SPAN,
    Gr00tN17Config,
    Gr00tN17StackConfig,
)
from mstar.model.gr00t_n17.submodules import (
    Gr00tN17DenoiseStepSubmodule,
    Gr00tN17Shape,
)
from mstar.model.submodule_base import ModelInputsFromEngine

_HUB = Path.home() / ".cache/huggingface/hub"
_GR00T = _HUB / "models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495"
_BACKBONE = next(iter(glob.glob(str(_HUB / "models--nvidia--Cosmos-Reason2-2B/snapshots/*"))), None)

_LOOP = "action_gen"
_WALK = "action_gen"

# The checkpoint's numbers, spelled out rather than read off the config: these are what the
# tests below exist to catch changing. The last section checks them against the checkpoint.
_HORIZON = 40          # `action_horizon`: the action rows, and the latents' rows
_ACTION_DIM = 132      # `max_action_dim`: the latents' width
_ROWS = 1 + _HORIZON   # what the DT sees: the state row, then the action rows
_STEPS = 4             # `num_inference_timesteps`
_BUCKETS = 1000        # `num_timestep_buckets`

# The reference's schedule, written out from `gr00t_n1d7.py:398-399`:
#   t_cont = t / float(num_inference_timesteps); t_discretized = int(t_cont * num_timestep_buckets)
# Derived here rather than imported, so the test's expectation is the reference's expression
# and not the implementation's `floor` of a sigma.
_REF_TIMESTEPS = tuple(float(int(t / _STEPS * _BUCKETS)) for t in range(_STEPS))
# `dt = 1.0 / self.num_inference_timesteps` (`gr00t_n1d7.py:355`), which is what
# `actions + dt * pred_velocity` steps by -- and what `sigma_next - sigma` has to be.
_REF_DT = 1.0 / _STEPS

# A tiny DT. `input_embedding_dim` is the stack's width and not a free number: the action
# encoder's output and the state row are both that wide and the DT runs on them unprojected,
# so `input_embedding_dim == dit.inner_dim` -- 1536 in the real checkpoint, 8 here.
_HEADS, _HEAD_DIM = 2, 4
_WIDTH = _HEADS * _HEAD_DIM
_BRIDGE_DIM = 7        # the bridge's rows, a different width from the stack's
_OUT_DIM = 6           # `dit.output_dim`: what the action decoder reads
_DECODER_HIDDEN = 5    # `hidden_size`
_CATEGORIES = 3        # `max_num_embodiments`
_EMBODIMENT = 2        # the id every test runs under: the last category, so an off-by-one bites
_DEPTH = 2             # one cross block and one self block, the shortest stack that alternates

# The two spans. Different lengths on purpose -- the real ones are 13 or 21 text rows against
# a constant 448 image rows -- so neither span can stand in for the other.
_TXT, _IMG = 5, 9
_SHAPE = Gr00tN17Shape(text_rows=_TXT, image_rows=_IMG)

_BASE_CONFIG = Gr00tN17Config(
    text=None, vision=None, bridge=None,
    dit=Gr00tN17StackConfig(
        num_layers=_DEPTH, num_attention_heads=_HEADS,
        attention_head_dim=_HEAD_DIM, output_dim=_OUT_DIM,
    ),
    deepstack_visual_indexes=(), backbone_embedding_dim=_BRIDGE_DIM,
    input_embedding_dim=_WIDTH, hidden_size=_DECODER_HIDDEN,
    max_num_embodiments=_CATEGORIES, max_seq_len=64,
    action_horizon=_HORIZON, max_action_dim=_ACTION_DIM,
    num_inference_timesteps=_STEPS, num_timestep_buckets=_BUCKETS,
)


@pytest.fixture(autouse=True)
def _pinned_threads():
    """fp32 CPU forwards compared with ``torch.equal``; reduction order follows the thread
    count, as it does in the other gr00t tests."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


# ----------------------------------------------------------------------------
# The node, built shallow
# ----------------------------------------------------------------------------


def _config(depth: int = _DEPTH) -> Gr00tN17Config:
    """The tiny config at ``depth`` blocks, by ``replace`` so one description of the model
    serves every depth and nothing drifts between them."""
    return dataclasses.replace(
        _BASE_CONFIG, dit=dataclasses.replace(_BASE_CONFIG.dit, num_layers=depth),
    )


def _randomize(module: nn.Module, seed: int, scale: float = 0.1) -> None:
    """Filled, which is not optional: ``ActionPositionEmbedding`` allocates its table with
    ``torch.empty``, so an unfilled node reads whatever the allocator last left there.

    Scaled down: the fill runs through the residual blocks and through the swish in the
    action encoder, and at 0.3 a few blocks of random weights already saturate both.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(scale * torch.randn(parameter.shape, generator=generator))


def _node(
    depth: int = _DEPTH, seed: int = 0, dtype: torch.dtype = torch.float32,
) -> Gr00tN17DenoiseStepSubmodule:
    """Ours, filled, shallow. ``dtype`` casts the whole node -- weights, forward and, because
    the carried trajectory follows the weights, the loop state too. That is what the dtype
    section runs at bf16 and at fp32 to tell the two apart."""
    config = _config(depth)
    node = Gr00tN17DenoiseStepSubmodule(
        config,
        build_action_encoder(config),
        ActionPositionEmbedding(config),
        DiT(config),
        build_action_decoder(config),
        loop_name=_LOOP,
    )
    _randomize(node, seed)
    return node.to(dtype).eval()


def _info(
    rid: str = "r0", k: int = 0, text: int = _TXT, image: int = _IMG, seed: int = 0,
) -> CurrentForwardPassInfo:
    """One request's forward-pass info, with the span counts in ``step_metadata``.

    Supplied directly: nothing fills these yet -- the ``preprocess`` that counts the prompt's
    rows is the next checkpoint -- so the key's source is this dict and not a producer node.
    The handle, not the id, is what the base keys its per-request state by.
    """
    return CurrentForwardPassInfo(
        request_id=rid, graph_walk=_WALK, fwd_index=k, random_seed=seed, max_tokens=0,
        rid_handle=abs(hash(rid)) % 1000,
        step_metadata={"vl_text_rows": text, "vl_image_rows": image},
        dynamic_loop_iter_counts={_LOOP: k},
    )


def _walk_inputs(
    config: Gr00tN17Config, text: int = _TXT, image: int = _IMG, seed: int = 5,
    dtype: torch.dtype = torch.float32, embodiment: int = _EMBODIMENT,
) -> dict:
    """The edges the ``prefill`` walk routes into this node, in the shapes those nodes emit:
    the bridge's two spans are ``[rows, backbone_embedding_dim]`` and the state encoder's
    token is ``[1, 1, input_embedding_dim]``, both already without a batch dim and both in
    the model's dtype. ``embodiment_id`` arrives as the request sent it, ``[1]`` and long.
    """
    generator = torch.Generator().manual_seed(seed)

    def rows(*shape):
        return torch.randn(shape, generator=generator).to(dtype)

    return {
        "vl_text": [rows(text, config.backbone_embedding_dim)],
        "vl_image": [rows(image, config.backbone_embedding_dim)],
        "state_features": [rows(1, 1, config.input_embedding_dim)],
        "embodiment_id": [torch.tensor([embodiment])],
    }


def _stacked_cond(
    config: Gr00tN17Config, batch: int = 1, text: int = _TXT, image: int = _IMG,
    seed: int = 6, dtype: torch.dtype = torch.float32,
) -> dict:
    """``denoise``'s conditioning for a batch of ``batch`` rows, stacked as the base stacks
    it: a leading batch dim on every tensor and ``[B]`` for the embodiment ids."""
    generator = torch.Generator().manual_seed(seed)

    def rows(*shape):
        return torch.randn(shape, generator=generator).to(dtype)

    return {
        "vl_text": rows(batch, text, config.backbone_embedding_dim),
        "vl_image": rows(batch, image, config.backbone_embedding_dim),
        "state": rows(batch, 1, config.input_embedding_dim),
        "embodiment_id": torch.full((batch,), _EMBODIMENT, dtype=torch.long),
    }


def _latents(
    batch: int = 1, seed: int = 7, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``[B, 40, 132]``: the trajectory the loop carries, in the dtype ``seed_latents`` draws
    it in -- the **weights'**, as the reference draws ``actions`` at ``vl_embeds.dtype``.

    So ``dtype`` here is the dtype of the node the caller built, and not a free choice: fp32
    for the default node, bf16 for the bf16 one. Drawn in fp32 and cast either way, so the
    bf16 trajectory is the fp32 one rounded and the two chains start at the same place.
    """
    return torch.randn(
        batch, _HORIZON, _ACTION_DIM, generator=torch.Generator().manual_seed(seed),
    ).to(dtype)


def _scalars(step: int, batch: int = 1) -> dict:
    """Step ``step``'s scalars as the base slices and views them: ``sigma`` / ``sigma_next``
    at ``[B, 1, 1]`` to broadcast over ``[B, L, C]``, ``timestep`` at a plain ``[B]``."""
    schedule = FlowMatchSchedule(
        sigmas=torch.arange(_STEPS + 1, dtype=torch.float32) / _STEPS,
        timesteps=torch.tensor(_REF_TIMESTEPS),
        mu=None,
    )
    return {
        "sigma": schedule.sigmas[step].expand(batch).view(-1, 1, 1),
        "sigma_next": schedule.sigmas[step + 1].expand(batch).view(-1, 1, 1),
        "timestep": schedule.timesteps[step].expand(batch),
    }


# ----------------------------------------------------------------------------
# The ragged resources' stand-in
# ----------------------------------------------------------------------------


class _SpanAttention:
    """Stands in for one of the two resources the engine plans from the step's segments.

    Built from that resource's own segment list, so the spans it slices are the spans the
    node declared -- a declaration that disagreed with the forward would land here as a
    wrong slice rather than as a bar to be argued about. A label of the form
    ``q<-kv`` (``cross_label``) is a cross-attention pair: the ``q`` spans attend the ``kv``
    spans, request for request. Never causal, as both DT attentions are.

    The per-segment call keeps the batch axis, so at one request it is exactly the call
    ``sdpa_attention`` makes -- which is what lets the ragged path be compared to the SDPA
    fallback with ``torch.equal`` rather than against a bar.
    """

    def __init__(self, segments):
        self.spans: dict[str, list[int]] = {}
        for segment in segments:
            self.spans.setdefault(segment.label, []).append(segment.span)

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, label: str) -> torch.Tensor:
        q_label, _, kv_label = label.partition("<-")
        kv_label = kv_label or q_label
        # the plan's layout is the declaration's, so a caller handing this label tensors of
        # another span's length is a disagreement between the two. Checked rather than
        # sliced: `k[0:9]` of a five-row tensor quietly returns five rows and the wrong
        # attention then reads as the right one.
        for name, tensor in ((q_label, q), (kv_label, k), (kv_label, v)):
            if name not in self.spans:
                raise ValueError(f"no span {name!r} in this step; it declared {sorted(self.spans)}")
            if tensor.shape[0] != sum(self.spans[name]):
                raise ValueError(
                    f"{label}: {tensor.shape[0]} row(s) for span {name!r}, declared "
                    f"{sum(self.spans[name])}"
                )
        out, q_at, kv_at = [], 0, 0
        for q_span, kv_span in zip(self.spans[q_label], self.spans[kv_label], strict=True):
            query = q[q_at:q_at + q_span].transpose(0, 1).unsqueeze(0)
            key = k[kv_at:kv_at + kv_span].transpose(0, 1).unsqueeze(0)
            value = v[kv_at:kv_at + kv_span].transpose(0, 1).unsqueeze(0)
            out.append(
                F.scaled_dot_product_attention(query, key, value).squeeze(0).transpose(0, 1)
            )
            q_at += q_span
            kv_at += kv_span
        return torch.cat(out)


def _bind(node: Gr00tN17DenoiseStepSubmodule, step, keys=(DT_ATTN, DT_XATTN)) -> None:
    """Bind the way the engine does: the resources it planned from this step's declaration,
    under the keys the node declared them with."""
    node.bind_node_resources(
        {key: _SpanAttention(step.steps[key].segments) for key in keys}
    )


def _declare(node: Gr00tN17DenoiseStepSubmodule, rids=("r0",), infos=None):
    """``prepare_inputs`` for every request, then the step they declare together."""
    infos = infos or [_info(rid) for rid in rids]
    prepared = [
        node.prepare_inputs(
            _WALK, info,
            _walk_inputs(
                node.config, info.step_metadata["vl_text_rows"],
                info.step_metadata["vl_image_rows"], dtype=next(node.parameters()).dtype,
            ),
        )
        for info in infos
    ]
    return prepared, node.declare_step(_WALK, list(rids), prepared)


@torch.no_grad()
def _by_hand(node: Gr00tN17DenoiseStepSubmodule, latents: torch.Tensor, cond: dict,
             timestep: torch.Tensor) -> torch.Tensor:
    """The reference's iteration up to the velocity (``gr00t_n1d7.py:401-432``), on this
    node's own children: encode the actions at the discretised time, add the position table
    to the action rows, concatenate the state row in front, run the DT, decode, and keep the
    last ``action_horizon`` rows. The Euler update is deliberately *not* here -- it is what
    the dtype section compares against.
    """
    dtype = next(node.model.parameters()).dtype
    cat_ids = cond["embodiment_id"].reshape(-1)
    actions = node.action_encoder(latents.to(dtype), timestep, cat_ids)
    rows = torch.cat([cond["state"], node.position_embedding(actions)], dim=1)
    out = node.model(rows, cond["vl_text"], cond["vl_image"], timestep, node.attends())
    return node.action_decoder(out, cat_ids)[:, -_HORIZON:]


# ----------------------------------------------------------------------------
# The schedule: GR00T's time, read the other way round
# ----------------------------------------------------------------------------


def test_the_schedule_is_four_ascending_sigmas_and_the_references_buckets():
    """``sigmas`` is ``[0, .25, .5, .75, 1]`` and ``timesteps`` is ``[0, 250, 500, 750]``,
    both float32, with one more sigma than there are steps.

    The timesteps are the reference's own ``int(t / num_inference_timesteps *
    num_timestep_buckets)`` computed in this file, so the expectation is the reference's
    expression and not a second reading of the implementation. The terminal sigma is not
    decoration: ``STEP_SCALARS`` reads ``sigmas[k + 1]`` at the last step, so a schedule with
    only ``num_steps`` sigmas is rejected by the base (the test below).

    If this fails the model is conditioned on the wrong time -- every bucket is an index into
    a trained sinusoid -- and the trajectory is integrated over the wrong interval.
    """
    schedule = _node().schedule_for(None, _SHAPE)

    assert schedule.num_steps == _STEPS
    assert schedule.sigmas.shape == (_STEPS + 1,)
    assert schedule.timesteps.shape == (_STEPS,)
    assert schedule.sigmas.dtype is torch.float32
    assert schedule.timesteps.dtype is torch.float32
    assert torch.equal(schedule.sigmas, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert torch.equal(schedule.timesteps, torch.tensor(_REF_TIMESTEPS))
    assert schedule.mu is None  # no resolution-dependent shift: GR00T has no image grid


def test_sigma_ascends_so_the_step_increment_is_the_references_dt():
    """``sigma_next - sigma`` is ``+1/4`` at every step, which is the reference's
    ``dt = 1.0 / num_inference_timesteps``, and it is what ``euler_step`` multiplies the
    velocity by.

    Said as a difference rather than as a list of sigmas because the difference is the only
    thing the arithmetic uses: a descending schedule of the same spacing would give ``-1/4``
    and integrate the trajectory backwards, producing finite, plausible actions.
    """
    schedule = _node().schedule_for(None, _SHAPE)
    steps = schedule.sigmas[1:] - schedule.sigmas[:-1]

    assert torch.equal(steps, torch.full((_STEPS,), _REF_DT))
    assert bool((steps > 0).all())
    # the first sigma is 0: GR00T's "sigma" is its time, not a noise level, so the loop
    # starts at t = 0 where an image model starts at full noise
    assert float(schedule.sigmas[0]) == 0.0
    assert float(schedule.sigmas[-1]) == 1.0


def test_the_image_conventions_schedule_is_a_different_one():
    """``FlowMatchSchedule.build`` -- what every other denoise loop on the scaffold uses --
    gives sigmas descending ``[1, .75, .5, .25, 0]`` and timesteps ``[1000, 750, 500, 250]``
    at the same four steps.

    Pinned here so that "why is this schedule hand-written when the scaffold builds one?"
    has an answer in the test file: the two are different schedules, and swapping ours for
    ``build``'s would condition the DT on the reversed bucket ladder and step the trajectory
    the wrong way. Both are four-step flow schedules of float32 tensors, so nothing but these
    values tells them apart.
    """
    built = FlowMatchSchedule.build(FlowMatchConfig(), _STEPS, image_seq_len=_IMG)
    ours = _node().schedule_for(None, _SHAPE)

    assert torch.equal(built.sigmas, torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]))
    assert torch.equal(built.timesteps, torch.tensor([1000.0, 750.0, 500.0, 250.0]))
    assert not torch.equal(built.sigmas, ours.sigmas)
    assert not torch.equal(built.timesteps, ours.timesteps)
    # the two are each other's reverse in sigma, which is why a "simplification" to `build`
    # would still look like a schedule and still run
    assert torch.equal(built.sigmas.flip(0), ours.sigmas)


def test_the_steps_scalars_come_off_our_schedule_at_every_step():
    """The base slices ``sigmas[k]``, ``sigmas[k + 1]`` and ``timesteps[k]`` out of whatever
    ``schedule_for`` returned, and checks the lengths it needs -- so a schedule one sigma
    short raises here rather than running three steps and reading garbage on the fourth.

    Driven through ``prepare_inputs`` at all four iterations, which is also the only place
    the terminal sigma is read.
    """
    node = _node()
    inputs = _walk_inputs(node.config)
    for k in range(_STEPS):
        prepared = node.prepare_inputs(_WALK, _info("r0", k), {**inputs, LATENTS: [_latents(1)[0]]})
        assert float(prepared.tensor_inputs["sigma"]) == k * _REF_DT
        assert float(prepared.tensor_inputs["sigma_next"]) == (k + 1) * _REF_DT
        assert float(prepared.tensor_inputs["timestep"]) == _REF_TIMESTEPS[k]
    # the request runs exactly `num_inference_timesteps` steps; the base vetoes the next
    assert node.request_state(_info("r0").rid_handle)["num_steps"] == _STEPS
    assert node.prepare_inputs(
        _WALK, _info("r0", _STEPS), {**inputs, LATENTS: [_latents(1)[0]]},
    ) is None


# ----------------------------------------------------------------------------
# The two counts: 41 rows into the DT, 40 rows of latents
# ----------------------------------------------------------------------------


def test_the_dt_sees_41_rows_while_the_latents_carry_40():
    """``num_tokens`` is ``1 + action_horizon`` -- the state row plus the action rows -- and
    the latents are ``[action_horizon, max_action_dim]``. The two are one apart, and both are
    plausible token counts for this node.

    The 41 sizes the attention segment, the capture bucket and ``input_seq_len``; the 40
    sizes the trajectory. Using the latents' 40 as the token count would declare a span one
    row short of the rows the DT attends, which the ragged plan would accept and silently cut
    the last action row's attention out of; using 41 for the latents would add a row to the
    trajectory that the decoder's ``[-action_horizon:]`` slice then throws away.
    """
    node = _node()

    assert node.num_tokens(_SHAPE) == _ROWS == 41
    assert node.num_tokens(_SHAPE) == 1 + node.config.action_horizon
    seeded = node.seed_latents(None, _SHAPE, torch.Generator().manual_seed(0))
    assert seeded.shape == (_HORIZON, _ACTION_DIM) == (40, 132)
    # the weights', not fp32: this node is fp32, so the two coincide here, and an
    # assertion on fp32 would pass against a `seed_latents` hardcoded to it
    assert seeded.dtype is next(node.parameters()).dtype
    # the counts are one apart, and neither is the other
    assert node.num_tokens(_SHAPE) == seeded.shape[0] + 1
    # the token count does not move with the spans; the spans are declared separately
    assert node.num_tokens(Gr00tN17Shape(_TXT * 3, _IMG * 7)) == _ROWS


def test_the_41_rows_reach_the_dt_and_the_40_come_back():
    """The declaration's token count is the rows the forward actually builds. Pinned by
    watching the DT's own input: ``prepare_inputs`` says 41, the step's ``sa`` segment says
    41, the tensor handed to the DT has 41 rows, and what ``denoise`` returns has 40.

    Nothing in the node's shapes would catch a confusion here -- both counts flow through
    without a shape error -- so the only place it is visible is between the declaration and
    the forward, which is what this test straddles.
    """
    node = _node()
    prepared, step = _declare(node)
    _bind(node, step)
    seen = []
    node.model.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape))

    cond = _stacked_cond(node.config)
    out = node.denoise(None, _SHAPE, latents=_latents(1), **_scalars(0), **cond)

    assert prepared[0].input_seq_len == _ROWS
    assert [segment.span for segment in step.steps[DT_ATTN].segments] == [_ROWS]
    assert seen == [torch.Size((1, _ROWS, _WIDTH))]
    assert out.shape == (1, _HORIZON, _ACTION_DIM)


# ----------------------------------------------------------------------------
# The loop state's dtype: the weights', because the reference's is
# ----------------------------------------------------------------------------


def _ref_euler(actions: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    """The reference's Euler update, written out rather than imported: ``actions + dt *
    pred_velocity * vel_strength`` (``gr00t_n1d7.py:435``) in that order, with ``dt`` the
    Python float ``1 / num_inference_timesteps`` (``:355``) and ``vel_strength`` the
    ``torch.ones_like(actions)`` the reference uses when no RTC action is supplied (``:356``).

    Nothing is upcast here, because nothing is upcast there: at a bf16 model every operand and
    every rounding is bf16. That is the arithmetic our step has to reproduce -- not an fp32
    ideal for it to beat.
    """
    return actions + _REF_DT * velocity * torch.ones_like(actions)


def test_the_carried_dtype_is_the_weights_and_does_not_change_mid_loop():
    """The carried trajectory is in the **weights'** dtype at every iteration: a bf16 model
    carries bf16, an fp32 model carries fp32, and neither changes between iteration 0 and
    iteration 3.

    This section used to pin fp32 against a bf16 model, on the claim that the reference
    integrates in fp32. **That claim was wrong** -- the reference is served bf16, draws
    ``actions`` at ``vl_embeds.dtype`` and updates with a Python-float ``dt``, so its
    trajectory is bf16 throughout (this module's docstring has the citations). fp32 is the
    better integrator and that is the trap: it cost the end-to-end parity bar, because a
    tolerance wide enough to admit our own extra precision (0.0243 over four steps) is within
    1.6x of admitting the R7 block-phase off-by-one (0.0391). Do not "improve" it back.

    Stability across the loop is a second property, and the one with a cost attached:
    ``euler_step`` returns in the *velocity's* dtype while ``seed_latents`` draws at the
    *weights'*, and those two agree only because both are the weights' dtype. If they ever
    disagreed, iteration 0 would take a wide ``latents`` and every later iteration a narrow
    one -- one bucket needing two captured graphs, which the Loop will not do, so the capture
    misses and the whole loop quietly runs eager.
    """
    for dtype in (torch.bfloat16, torch.float32):
        node = _node(dtype=dtype)
        cond = _stacked_cond(node.config, dtype=dtype)
        start = _latents(1, dtype=dtype)

        carried, seen = start, []
        for k in range(_STEPS):
            carried = node.denoise(None, _SHAPE, latents=carried, **_scalars(k), **cond)
            seen.append(carried.dtype)

        # the premise: the forward really is in `dtype`, so what `euler_step` returns in the
        # velocity's dtype is testable at all
        assert _by_hand(node, carried, cond, _scalars(0)["timestep"]).dtype is dtype
        assert seen == [dtype] * _STEPS, seen
        assert carried.shape == (1, _HORIZON, _ACTION_DIM)
        # the same dtype the loop's own seed arrives in, which is what makes it stable: the
        # carried dtype is the seed's dtype and not a second decision
        assert node.seed_latents(None, _SHAPE, torch.Generator().manual_seed(0)).dtype is dtype
        assert not torch.equal(carried, start)  # the four steps moved the trajectory


def test_four_bf16_iterations_are_the_references_arithmetic_to_the_bit():
    """The whole schedule, output fed back as the Loop feeds it back, equal to the reference's
    own chain at **every** iteration -- ``torch.equal``, not a tolerance.

    ``euler_step`` sums in fp32 and rounds back to the velocity's dtype, which reads like an
    approximation of the reference's pure-bf16 update and is not one. For two bf16 operands and
    a ``dt`` that is an exact power of two, ``(a.float() + dt * v.float()).to(bfloat16)`` is
    bit-identical to bf16 ``a + dt * v``: scaling by a power of two only moves an exponent, so
    ``dt * v`` is exact in bf16, and rounding the exact sum once through fp32 is the same as
    rounding it once, because fp32's 24 mantissa bits are more than twice bf16's 8. Matching
    the reference therefore costs nothing -- which is the other half of why the fp32 carry had
    to go: it bought accuracy nobody asked for with the one bar that catches a wiring bug.

    The power of two is load-bearing. The last block is the same identity at ``dt = 0.3``,
    where it fails outright: a schedule of 3 or 5 inference steps would land there and this
    step would stop being the reference's, so the equivalence is a fact about GR00T's four
    steps and not about fp32 in general.

    If this fails the trajectory is no longer the reference's bit for bit and the parity bar
    goes from exact to a tolerance -- the fp32 carry's exact failure mode, which the last
    assertion keeps in view by stepping that chain alongside and showing it differs.
    """
    node = _node(dtype=torch.bfloat16)
    cond = _stacked_cond(node.config, dtype=torch.bfloat16)
    start = _latents(1, dtype=torch.bfloat16)

    ours = reference = start
    fp32_carry = start.float()  # what this file used to pin: the better integrator
    for k in range(_STEPS):
        scalars = _scalars(k)
        # the hand-written chain steps by the schedule's own dt, so it is not a second
        # schedule that can drift from the node's
        assert float(scalars["sigma_next"] - scalars["sigma"]) == _REF_DT

        ours = node.denoise(None, _SHAPE, latents=ours, **scalars, **cond)
        velocity = _by_hand(node, reference, cond, scalars["timestep"])
        assert velocity.dtype is torch.bfloat16  # the premise: the forward *is* bf16
        reference = _ref_euler(reference, velocity)
        assert reference.dtype is torch.bfloat16
        assert torch.equal(ours, reference), f"iteration {k}"

        # the fp32 carry, stepped from the same place: `_by_hand` casts it down for the
        # forward exactly as the old node did, and only the carried state stays wide
        fp32_carry = fp32_carry + _REF_DT * _by_hand(
            node, fp32_carry, cond, scalars["timestep"],
        ).float()

    assert not torch.equal(ours, start)
    # the fp32 carry is a *different* trajectory from the reference's, however much better an
    # integrator it is. This is the comparison that priced the reversal.
    assert not torch.equal(ours.float(), fp32_carry)

    # the identity the step leans on, on the very tensors it just ran on
    assert torch.equal(
        (start.float() + _REF_DT * velocity.float()).to(torch.bfloat16),
        start + _REF_DT * velocity,
    )
    # and its precondition, both ways round: exact at 1/4, broken at 0.3
    assert _REF_DT == 2.0**-2
    a = torch.randn(4096, generator=torch.Generator().manual_seed(3)).bfloat16()
    v = torch.randn(4096, generator=torch.Generator().manual_seed(4)).bfloat16()
    for dt, exact in ((_REF_DT, True), (0.3, False)):
        assert torch.equal((a.float() + dt * v.float()).to(torch.bfloat16), a + dt * v) is exact, dt


# ----------------------------------------------------------------------------
# The bucket key, read off the request's step metadata
# ----------------------------------------------------------------------------


def test_the_bucket_key_is_the_two_span_counts_off_step_metadata():
    """The span lengths are needed before the bridge's tensors arrive -- the key picks the
    capture bucket and decides who batches with whom -- so they ride on the request in
    ``step_metadata`` rather than being measured off a tensor.

    Read straight out of the dict, and missing is an error rather than a default: a zero or
    an absent entry would key a bucket whose spans are not the request's, and the step's
    segments would then be planned at the wrong lengths.
    """
    node = _node()

    assert node.bucket_key_for(_info(text=13, image=448)) == Gr00tN17Shape(13, 448)
    assert node.bucket_key_for(_info(text=21, image=448)) == Gr00tN17Shape(21, 448)
    # the key reaches the prepared inputs, which is what `declare_step` and `can_batch` read
    prepared, _ = _declare(node)
    assert prepared[0].resource_step_info == _SHAPE

    bare = CurrentForwardPassInfo(
        request_id="r0", graph_walk=_WALK, fwd_index=0, random_seed=0, max_tokens=0,
    )
    with pytest.raises(KeyError):
        node.bucket_key_for(bare)
    # a padding row of a captured bucket carries no metadata, and the base drops it from the
    # vote rather than asking for its key
    assert node.cg_key_info(_WALK, {"r0": _info("r0"), "pad": bare}) == _SHAPE


def test_two_requests_batch_only_when_both_spans_match():
    """``Gr00tN17Shape`` is a frozen dataclass, so ``==`` and ``__hash__`` are generated --
    and both are load-bearing. ``can_batch`` compares keys in a set and the graph buckets are
    keyed by it, so a key that compared by identity would batch nothing and capture nothing,
    while one that ignored a field would stack two requests whose conditioning is not the
    same rectangle.

    The two fields are checked separately: the real image span is constant at 448 and only
    the text span varies, so a key that dropped ``image_rows`` would pass every recorded
    request and fail the first time a camera set changed.
    """
    node = _node()
    # distinct request ids per case: the base caches a request's key for the walk, so the
    # same id with different metadata would be answered from that cache
    same = [_info("r0"), _info("r1")]
    other_text = [_info("r0"), _info("r2", text=_TXT + 1)]
    other_image = [_info("r0"), _info("r3", image=_IMG + 1)]

    assert Gr00tN17Shape(13, 448) == Gr00tN17Shape(13, 448)
    assert hash(Gr00tN17Shape(13, 448)) == hash(Gr00tN17Shape(13, 448))
    assert len({Gr00tN17Shape(13, 448), Gr00tN17Shape(13, 448)}) == 1
    assert Gr00tN17Shape(13, 448) != Gr00tN17Shape(21, 448)
    assert Gr00tN17Shape(13, 448) != Gr00tN17Shape(13, 449)
    assert len({Gr00tN17Shape(13, 448), Gr00tN17Shape(21, 448), Gr00tN17Shape(13, 449)}) == 3
    with pytest.raises(dataclasses.FrozenInstanceError):
        Gr00tN17Shape(13, 448).text_rows = 21

    assert node.can_batch(None, _declare(node, ("r0", "r1"), same)[0]) is True
    assert node.can_batch(None, _declare(node, ("r0", "r2"), other_text)[0]) is False
    assert node.can_batch(None, _declare(node, ("r0", "r3"), other_image)[0]) is False
    # a single row never batches, whatever its key
    assert node.can_batch(None, _declare(node, ("r0",))[0]) is False
    # and the graph bucket follows the same key: a mixed batch matches no capture
    assert node.cg_key_info(_WALK, {info.request_id: info for info in same}) == _SHAPE
    assert node.cg_key_info(_WALK, {info.request_id: info for info in other_text}) is None
    assert node.cg_key_info(_WALK, {info.request_id: info for info in other_image}) is None


# ----------------------------------------------------------------------------
# The conditioning: no batch dim, and the capture's twin of it
# ----------------------------------------------------------------------------

# rank of each conditioning entry as one request carries it, and what it has to be once the
# base has stacked the batch. A leading 1 anywhere here would arrive at `denoise` as a
# spurious axis -- `[B, 1, T, width]` cross-attention rows, a `[B, 1]` embodiment id.
_REQUEST_RANKS = {"vl_text": 2, "vl_image": 2, "state": 2, "embodiment_id": 0}


def test_the_request_inputs_carry_no_batch_dim_and_stack_into_the_forwards_shapes():
    """``request_inputs`` returns one request's conditioning, each tensor **without** a batch
    dim, because ``preprocess`` stacks the batch itself. The state token is the one to watch:
    the state encoder emits ``[1, 1, input_embedding_dim]`` and this hook reshapes it to
    ``[1, D]``, so what the DT gets is ``[B, 1, D]`` -- one state row per request. Left at
    rank 3 it would stack to ``[B, 1, 1, D]`` and the concatenation with the action rows
    would fail, or worse, broadcast.

    Checked as a rank on each side rather than as a shape, so the test says the one thing it
    is about: the batch axis is the base's to add and this hook's not to.
    """
    node = _node()
    inputs = _walk_inputs(node.config)
    single = node.request_inputs(_info(), inputs, _SHAPE)

    assert set(single) == set(_REQUEST_RANKS)
    assert {name: tensor.dim() for name, tensor in single.items()} == _REQUEST_RANKS
    assert single["vl_text"].shape == (_TXT, node.config.backbone_embedding_dim)
    assert single["vl_image"].shape == (_IMG, node.config.backbone_embedding_dim)
    assert single["state"].shape == (1, node.config.input_embedding_dim)
    assert int(single["embodiment_id"]) == _EMBODIMENT
    # the spans are not pulled from the key, so a key disagreeing with the tensors is caught
    # by the step's segments rather than silently reshaping them
    assert torch.equal(single["vl_text"], inputs["vl_text"][0])
    assert torch.equal(single["vl_image"], inputs["vl_image"][0])

    prepared, _ = _declare(node, ("r0", "r1"))
    stacked = node.preprocess(_WALK, None, prepared)
    for name, rank in _REQUEST_RANKS.items():
        assert stacked[name].dim() == rank + 1, name
        assert stacked[name].shape[0] == 2, name
    assert stacked[LATENTS].shape == (2, _HORIZON, _ACTION_DIM)
    assert stacked["state"].shape == (2, 1, node.config.input_embedding_dim)
    assert stacked["embodiment_id"].shape == (2,)


def test_driving_the_node_is_one_denoise_call_per_row_and_each_row_is_its_own():
    """``prepare_inputs -> declare_step -> preprocess -> forward_batched``, as the engine runs
    it: the two rows that come back are the two rows of one stacked ``denoise``, each from its
    own conditioning.

    The two requests are given different conditioning **and** different embodiment ids, and
    the rows are required to differ. A batched forward that served row 0's embodiment to
    everyone -- the per-embodiment weights are gathered by ``W[cat_ids]``, so one row's id
    broadcast over the batch changes no shape -- would return two plausible trajectories for
    the wrong robot, and this is the only test that would see it.
    """
    node = _node()
    rids = ["r0", "r1"]
    infos = [_info(rid) for rid in rids]
    prepared = [
        node.prepare_inputs(
            _WALK, info, _walk_inputs(node.config, seed=seed, embodiment=embodiment),
        )
        for info, seed, embodiment in zip(infos, (5, 11), (0, _EMBODIMENT), strict=True)
    ]
    step = node.declare_step(_WALK, rids, prepared)
    _bind(node, step)
    stacked = node.preprocess(_WALK, None, prepared)
    engine = ModelInputsFromEngine(request_ids=rids, per_request_info={})

    rows = node.forward_batched(_WALK, engine, **stacked)

    assert list(rows) == rids
    assert stacked["embodiment_id"].tolist() == [0, _EMBODIMENT]
    batched = node.denoise(
        None, _SHAPE,
        **{name: stacked[name] for name in (LATENTS, "sigma", "sigma_next", "timestep")},
        **{name: stacked[name] for name in _REQUEST_RANKS},
    )
    for index, rid in enumerate(rids):
        assert list(rows[rid]) == [LATENTS]
        assert rows[rid][LATENTS][0].shape == (_HORIZON, _ACTION_DIM)
        assert torch.equal(rows[rid][LATENTS][0], batched[index])
    assert not torch.equal(batched[0], batched[1])

    # each row answers to its own embodiment id and to no one else's: changing row 0's moves
    # row 0 and leaves row 1 bit-identical. Same batch size and same shapes on both sides, so
    # this is `torch.equal` with no reduction order in it.
    moved = node.denoise(
        None, _SHAPE,
        **{name: stacked[name] for name in (LATENTS, "sigma", "sigma_next", "timestep")},
        **{name: stacked[name] for name in _REQUEST_RANKS if name != "embodiment_id"},
        embodiment_id=torch.tensor([_EMBODIMENT, _EMBODIMENT]),
    )
    assert not torch.equal(moved[0], batched[0])
    assert torch.equal(moved[1], batched[1])


def test_the_captures_placeholders_are_the_shapes_and_dtypes_a_real_row_brings():
    """A graph is captured on ``capture_request_inputs`` and replayed on what
    ``request_inputs`` and the seeded noise produce, so the two have to agree on every name,
    shape and dtype. They disagree in two different ways if they are written twice: a shape
    mismatch misses the capture and runs eager (slow, silent), and a dtype mismatch replays
    the graph with a buffer the kernels will reinterpret.

    Every dtype here is the model's, the **latents' included**: the loop carries the weights'
    dtype because the reference does. This test used to pin an fp32 latents placeholder against
    a bf16 model -- the same wrong claim as the dtype section above, and the worse of the two
    failures to get wrong, since a shape mismatch only misses the capture while a dtype
    mismatch replays the graph over a buffer the kernels reinterpret.

    Run at both dtypes, because a placeholder with a *hardcoded* dtype is wrong on exactly one
    of the two passes: fp32 (what this node used to do) fails the bf16 pass, bf16 fails the
    fp32 pass, and a single-dtype test would be blind to one of them.
    """
    cpu = torch.device("cpu")
    for dtype in (torch.bfloat16, torch.float32):
        node = _node(dtype=dtype)
        capture = node.capture_request_inputs(_SHAPE, cpu)
        inputs = _walk_inputs(node.config, dtype=dtype)
        real = {
            LATENTS: node.seed_latents(None, _SHAPE, torch.Generator().manual_seed(0)),
            **node.request_inputs(_info(), inputs, _SHAPE),
        }

        assert set(capture) == set(real) == {LATENTS, *_REQUEST_RANKS}
        for name, tensor in real.items():
            assert capture[name].shape == tensor.shape, (dtype, name)
            assert capture[name].dtype is tensor.dtype, (dtype, name)
            assert capture[name].device == cpu, (dtype, name)
        # spelled out as well as compared: the loop above passes if both sides are wrong
        # together, which is exactly how the fp32 claim survived
        assert capture[LATENTS].dtype is dtype
        assert capture["vl_text"].dtype is dtype
        assert capture["embodiment_id"].dtype is torch.long  # an index, not a width
    # and the placeholder spans are the key's, so every bucket captures its own lengths
    wide = node.capture_request_inputs(Gr00tN17Shape(21, 448), cpu)
    assert wide["vl_text"].shape == (21, node.config.backbone_embedding_dim)
    assert wide["vl_image"].shape == (448, node.config.backbone_embedding_dim)


# ----------------------------------------------------------------------------
# The attention callables: one per span, one set per binding
# ----------------------------------------------------------------------------


def test_the_attention_callables_are_memoised_per_binding_and_dropped_on_rebinding():
    """A compiled transformer region guards on the **identity** of the callables it is handed,
    so a fresh one per step retraces that frame every step until Dynamo gives up. One set for
    the life of a binding, and a new set after the next one -- the old callables hold the old
    resources, and handing them out after a rebind would run the retired resource's plan.

    Identity is checked per span as well as on the dict: a dict rebuilt with the same
    callables inside it would be harmless, and a cached dict holding freshly built callables
    would not be.
    """
    node = _node()
    _, step = _declare(node)
    _bind(node, step)

    first = node.attends()
    assert node.attends() is first
    assert all(node.attends()[span] is first[span] for span in first)

    _bind(node, step)
    second = node.attends()
    assert second is not first
    assert all(second[span] is not first[span] for span in first)
    assert set(second) == set(first)


def test_each_span_reads_its_own_resource_under_the_label_the_step_declared():
    """Three spans over two resources: ``sa`` is self-attention on ``DT_ATTN``, and the two
    bridge spans are cross-attention pairs on ``DT_XATTN`` under ``cross_label(sa, span)``.

    The label is the whole coupling. A ragged plan is keyed by it, so a callable asking for a
    label the step did not declare finds no plan -- and the pairs ``declare_step`` publishes
    are where those labels come from, which is why they are read off ``DT_XATTN_PAIRS`` here
    rather than written out a second time.
    """
    node = _node()
    _, step = _declare(node)
    _bind(node, step)
    attends = node.attends()
    resources = node.node_resources

    assert set(attends) == {SA_SPAN, VL_TEXT_SPAN, VL_IMAGE_SPAN}
    ragged = {span: fn.keywords["ragged"] for span, fn in attends.items()}
    assert all(isinstance(fn, functools.partial) for fn in ragged.values())
    assert ragged[SA_SPAN].func.__self__ is resources[DT_ATTN]
    assert ragged[SA_SPAN].keywords["label"] == SA_SPAN
    for q_label, kv_label in DT_XATTN_PAIRS:
        assert ragged[kv_label].func.__self__ is resources[DT_XATTN]
        assert ragged[kv_label].keywords["label"] == cross_label(q_label, kv_label)
    # every label a callable asks for is a label the step's resource declared
    for key, span_labels in ((DT_ATTN, [SA_SPAN]), (DT_XATTN, [SA_SPAN, VL_TEXT_SPAN, VL_IMAGE_SPAN])):
        assert {segment.label for segment in step.steps[key].segments} == set(span_labels)


def test_nothing_bound_leaves_attends_empty_and_the_dt_reads_every_span_with_sdpa():
    """Nothing bound is the **parity backend**: the map is empty, the DT answers every span
    with SDPA, and that is how the CPU tests and an unconfigured deployment run at all.

    An unbound span is *left out* rather than mapped to ``None`` -- a ``None`` would reach
    ``joint_attention`` as the ragged callable and raise mid-forward.
    """
    node = _node()
    _, step = _declare(node)
    _bind(node, step, keys=())

    assert node.attends() == {}
    out = node.denoise(
        None, _SHAPE, latents=_latents(1), **_scalars(0), **_stacked_cond(node.config),
    )
    assert out.shape == (1, _HORIZON, _ACTION_DIM)


@pytest.mark.parametrize(
    "bound,missing",
    [((DT_ATTN,), DT_XATTN), ((DT_XATTN,), DT_ATTN)],
    ids=["self-attention only", "cross-attention only"],
)
def test_binding_one_of_the_two_resources_is_refused(bound, missing):
    """All or nothing. Partial binding used to be silent, and that is incoherent: the node
    declares a step for **both** resources, so the engine plans a kernel the forward would
    not use -- and a captured graph would record SDPA where the plan says ragged. The
    arithmetic stays right, which is what makes it dangerous.

    Nothing bound stays legal (the test above); only the half-configured case raises. Found
    by C08's blind QA, which noted the sibling ``BridgeAttention`` raises on a missing
    resource while this node did not.
    """
    node = _node()
    _, step = _declare(node)
    _bind(node, step, keys=bound)

    with pytest.raises(RuntimeError, match=missing):
        node.attends()


def test_the_two_resources_planned_from_the_step_are_the_sdpa_fallback_at_one_request():
    """The ragged path and the SDPA fallback are the same computation: at one request each
    declared span is one segment, and the stand-in's per-segment call is exactly the call
    ``sdpa_attention`` makes -- so this is ``torch.equal`` and not a bar.

    That makes it a test of the wiring rather than of a kernel: it is the segments the node
    declared being sliced by the labels the node's callables ask for. A span declared at the
    wrong length, a cross pair with its two labels the wrong way round, or a callable bound
    to the other resource all land here as a different tensor.
    """
    node = _node()
    latents, scalars, cond = _latents(1), _scalars(2), _stacked_cond(node.config)

    sdpa = node.denoise(None, _SHAPE, latents=latents, **scalars, **cond)
    _, step = _declare(node)
    _bind(node, step)
    ragged = node.denoise(None, _SHAPE, latents=latents, **scalars, **cond)

    assert set(node.attends()) == {SA_SPAN, VL_TEXT_SPAN, VL_IMAGE_SPAN}
    assert torch.equal(ragged, sdpa)


# ----------------------------------------------------------------------------
# The step declaration: both DT resources, and the spans each one plans
# ----------------------------------------------------------------------------


def test_the_step_names_both_dt_resources_with_the_spans_each_one_plans():
    """``DT_ATTN`` is planned over the ``sa`` segments alone; ``DT_XATTN`` is planned over
    ``sa + text + image`` with ``DT_XATTN_PAIRS``, because a cross plan needs both ends of
    every pair in its layout. One resource is one *kind* of attention, so the DT's two cannot
    share one even at the same 32 x 48 head geometry.

    Order is pinned, not just membership: the segments are grouped by label -- every
    request's ``sa``, then every request's text, then every request's image -- which is how
    the pairs line a request's query span up with its own key span. Interleaving them per
    request would pair request 0's queries with request 0's text and request 1's queries with
    request 0's image.

    Nothing here is declared at the top level: ``SubmoduleStep.segments`` is left None so
    neither resource silently inherits the other's layout.
    """
    node = _node()
    rids = ["r0", "r1"]
    _, step = _declare(node, rids, [_info("r0"), _info("r1")])

    assert set(step.steps) == {DT_ATTN, DT_XATTN}
    assert step.segments is None
    assert step.cg_key_info == _SHAPE

    self_attn = step.steps[DT_ATTN]
    assert isinstance(self_attn, AttentionStep)
    assert self_attn.causal is False
    assert self_attn.segments == tuple(Segment(rid, SA_SPAN, _ROWS) for rid in rids)

    cross = step.steps[DT_XATTN]
    assert isinstance(cross, RaggedCrossAttentionStep)
    assert cross.pairs == DT_XATTN_PAIRS == ((SA_SPAN, VL_TEXT_SPAN), (SA_SPAN, VL_IMAGE_SPAN))
    assert cross.segments == (
        *(Segment(rid, SA_SPAN, _ROWS) for rid in rids),
        *(Segment(rid, VL_TEXT_SPAN, _TXT) for rid in rids),
        *(Segment(rid, VL_IMAGE_SPAN, _IMG) for rid in rids),
    )
    # both ends of every declared pair are in the layout, which is what a cross plan needs
    labels = {segment.label for segment in cross.segments}
    assert all({q, kv} <= labels for q, kv in cross.pairs)


def test_the_declared_spans_are_each_requests_own_lengths():
    """Spans come from each row's own key, so a batch the scheduler built out of two keys --
    or a captured bucket's padding rows -- is declared at the lengths its rows actually
    carry rather than at the first row's. The graph key then comes back None, which is the
    base's way of saying this batch matches no capture.
    """
    node = _node()
    infos = [_info("r0"), _info("r1", text=_TXT + 4, image=_IMG + 2)]
    _, step = _declare(node, ("r0", "r1"), infos)

    assert step.cg_key_info is None
    assert step.steps[DT_XATTN].segments == (
        Segment("r0", SA_SPAN, _ROWS), Segment("r1", SA_SPAN, _ROWS),
        Segment("r0", VL_TEXT_SPAN, _TXT), Segment("r1", VL_TEXT_SPAN, _TXT + 4),
        Segment("r0", VL_IMAGE_SPAN, _IMG), Segment("r1", VL_IMAGE_SPAN, _IMG + 2),
    )


def test_attention_segments_lists_the_same_three_spans_the_step_declares():
    """``attention_segments`` is the scaffold's hook for "every span one request's layers
    attend over", and this node's ``declare_step`` builds its segments directly instead of
    going through it. The two are therefore two descriptions of one layout, and this test is
    what keeps them from drifting: a span added to one and not the other is a plan with no
    entry for a label the forward asks for.
    """
    node = _node()
    _, step = _declare(node)

    assert node.attention_segments(_SHAPE) == (
        (SA_SPAN, _ROWS), (VL_TEXT_SPAN, _TXT), (VL_IMAGE_SPAN, _IMG),
    )
    assert [(segment.label, segment.span) for segment in step.steps[DT_XATTN].segments] == list(
        node.attention_segments(_SHAPE)
    )


def test_the_node_declares_no_single_base_resource():
    """The base's own ``declare_step`` plans one resource under ``attn_resource_key``; this
    node passes ``None`` and declares two itself, as ``ltx2_5`` does for its four. Left set,
    the base would plan a third resource over one ``"main"`` span of 41 rows that no layer
    reads -- and ``ragged_for`` would hand the DT callables bound to it.
    """
    node = _node()

    assert node.attn_resource_key is None
    assert node.loop_back_names == (LATENTS,)  # one Euler step, no solver state to carry
    assert DenoiseLoopSubmodule.SOLVER_STATE == ()
    assert node.ragged_for(SA_SPAN) is None
    assert node.max_batch_size(_WALK) == 8


# ----------------------------------------------------------------------------
# The checkpoint's own numbers, on `meta`
# ----------------------------------------------------------------------------


@pytest.mark.skipif(not (_GR00T.exists() and _BACKBONE), reason="checkpoint not cached")
def test_the_checkpoints_numbers_are_the_ones_these_tests_run():
    """Every count above is spelled out, so this is where they are tied to the checkpoint:
    40 action rows of 132, four steps over 1000 buckets, and 41 rows into the DT.

    The node is built on ``meta`` -- 32 blocks of 1536 x 32 x 48 is 1.12B parameters and
    4.2 GiB per fp32 copy -- so nothing is allocated. ``input_embedding_dim`` is checked
    against the stack's width because the state row and the action rows go into the DT
    unprojected: if the checkpoint ever separated them the concatenation would be the first
    thing to break.
    """
    config = Gr00tN17Config.from_hf(_GR00T, _BACKBONE)
    with torch.device("meta"):
        node = Gr00tN17DenoiseStepSubmodule(
            config,
            build_action_encoder(config),
            ActionPositionEmbedding(config),
            DiT(config),
            build_action_decoder(config),
            loop_name=_LOOP,
        )

    assert (config.action_horizon, config.max_action_dim) == (_HORIZON, _ACTION_DIM)
    assert (config.num_inference_timesteps, config.num_timestep_buckets) == (_STEPS, _BUCKETS)
    assert config.input_embedding_dim == config.dit.inner_dim
    key = Gr00tN17Shape(text_rows=13, image_rows=448)  # a recorded prefill: four camera views
    assert node.num_tokens(key) == _ROWS
    assert node.seed_latents(None, key, torch.Generator().manual_seed(0)).shape == (40, 132)
    schedule = node.schedule_for(None, key)
    assert torch.equal(schedule.timesteps, torch.tensor(_REF_TIMESTEPS))
    placeholders = node.capture_request_inputs(key, torch.device("meta"))
    assert placeholders[LATENTS].shape == (40, 132)
    assert placeholders["vl_text"].shape == (13, config.backbone_embedding_dim)
    assert placeholders["state"].shape == (1, config.input_embedding_dim)
