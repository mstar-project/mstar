"""GR00T N1.7's ``denoise_step`` node: one flow-matching step of the action head on the
scaffold's ``DenoiseLoopSubmodule``. Covers the hooks: the schedule (ascending, unlike the image
models), the 41-row/40-row counts, the carried dtype, the bucket key, batching, the
conditioning, the attention callables and the step declaration. Value tests run a shallow DT.
"""

import dataclasses

import pytest
import torch
import torch.nn.functional as F

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.attn.ragged.config import RaggedCrossAttentionStep, cross_label
from mstar.engine.resources.convenience import RaggedAttentionCallable
from mstar.engine.resources.step import Segment
from mstar.model.components.diffusion.denoise_loop import LATENTS
from mstar.model.components.diffusion.flow_match import FlowMatchSchedule
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
from mstar.worker.batch_builder import off_group_rids
from mstar.worker.micro_scheduler import ScheduledBatch

from .helpers import BACKBONE, GR00T, needs_checkpoint, randomize

_LOOP = "action_gen"
_WALK = "action_gen"

# The checkpoint's numbers, spelled out rather than read off the config; the last test ties
# them to the checkpoint.
_HORIZON = 40          # `action_horizon`: the action rows, and the latents' rows
_ACTION_DIM = 132      # `max_action_dim`: the latents' width
_ROWS = 1 + _HORIZON   # what the DT sees: the state row, then the action rows
_STEPS = 4             # `num_inference_timesteps`
_BUCKETS = 1000        # `num_timestep_buckets`

# The reference's schedule, written out from `gr00t_n1d7.py:398-399`:
#   t_cont = t / float(num_inference_timesteps); t_discretized = int(t_cont * num_timestep_buckets)
# so the expectation is the reference's expression, not the implementation's `floor` of a sigma.
_REF_TIMESTEPS = tuple(float(int(t / _STEPS * _BUCKETS)) for t in range(_STEPS))
# `dt = 1.0 / self.num_inference_timesteps` (`gr00t_n1d7.py:355`): what `sigma_next - sigma` has to be.
_REF_DT = 1.0 / _STEPS

# A tiny DT. The state row and the action rows enter the DT unprojected, so
# `input_embedding_dim == dit.inner_dim` (1536 in the checkpoint, 8 here).
_HEADS, _HEAD_DIM = 2, 4
_WIDTH = _HEADS * _HEAD_DIM
_BRIDGE_DIM = 7        # the bridge's rows, a different width from the stack's
_OUT_DIM = 6           # `dit.output_dim`: what the action decoder reads
_DECODER_HIDDEN = 5    # `hidden_size`
_CATEGORIES = 3        # `max_num_embodiments`
_EMBODIMENT = 2        # the id every test runs under: the last category, so an off-by-one bites
_DEPTH = 2             # one cross block and one self block, the shortest stack that alternates

# Different span lengths on purpose (real: 13 or 21 text rows, 448 image rows), so neither
# span can stand in for the other.
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


# ----------------------------------------------------------------------------
# The node, built shallow
# ----------------------------------------------------------------------------


def _config(depth: int = _DEPTH) -> Gr00tN17Config:
    """The tiny config at ``depth`` blocks."""
    return dataclasses.replace(
        _BASE_CONFIG, dit=dataclasses.replace(_BASE_CONFIG.dit, num_layers=depth),
    )


def _node(
    depth: int = _DEPTH, seed: int = 0, dtype: torch.dtype = torch.float32,
) -> Gr00tN17DenoiseStepSubmodule:
    """Ours, filled, shallow. ``dtype`` casts the whole node, and so the carried loop state."""
    config = _config(depth)
    node = Gr00tN17DenoiseStepSubmodule(
        config,
        build_action_encoder(config),
        ActionPositionEmbedding(config),
        DiT(config),
        build_action_decoder(config),
        loop_name=_LOOP,
    )
    # filled because `ActionPositionEmbedding` allocates with `torch.empty`; 0.1 because at
    # 0.3 a few random blocks already saturate the residuals and the action encoder's swish
    randomize(node, seed, scale=0.1)
    return node.to(dtype).eval()


def _info(
    rid: str = "r0", k: int = 0, text: int = _TXT, image: int = _IMG, seed: int = 0,
) -> CurrentForwardPassInfo:
    """One request's forward-pass info, with the span counts in ``step_metadata``. The base
    keys per-request state by the handle, not the id."""
    return CurrentForwardPassInfo(
        request_id=rid, graph_walk=_WALK, fwd_index=k, random_seed=seed, max_tokens=0,
        # distinct per id and stable across processes (`hash()` is salted per process)
        rid_handle=int(rid.removeprefix("r")),
        step_metadata={"vl_text_rows": text, "vl_image_rows": image},
        dynamic_loop_iter_counts={_LOOP: k},
    )


def _walk_inputs(
    config: Gr00tN17Config, text: int = _TXT, image: int = _IMG, seed: int = 5,
    dtype: torch.dtype = torch.float32, embodiment: int = _EMBODIMENT,
) -> dict:
    """The edges the ``prefill`` walk routes here, in the shapes those nodes emit: spans
    ``[rows, backbone_embedding_dim]``, state token ``[1, 1, input_embedding_dim]``, and
    ``embodiment_id`` as the request sent it, ``[1]`` long."""
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
    """``denoise``'s conditioning for ``batch`` rows, stacked as the base stacks it."""
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
    """``[B, 40, 132]`` in the node's dtype, as ``seed_latents`` draws it. Drawn in fp32 and
    cast, so the bf16 and fp32 chains start at the same place."""
    return torch.randn(
        batch, _HORIZON, _ACTION_DIM, generator=torch.Generator().manual_seed(seed),
    ).to(dtype)


def _scalars(step: int, batch: int = 1) -> dict:
    """Step ``step``'s scalars as the base slices them: sigmas ``[B, 1, 1]``, timestep ``[B]``."""
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
    """Stands in for one DT resource, built from its own declared segments, so a
    declaration that disagrees with the forward shows up as a wrong slice. ``q<-kv``
    (``cross_label``) pairs the ``q`` spans with the ``kv`` spans, request for request.
    At one request the per-segment call is exactly ``sdpa_attention``'s, so the ragged path
    compares to the SDPA fallback with ``torch.equal``.
    """

    def __init__(self, segments):
        self.spans: dict[str, list[int]] = {}
        for segment in segments:
            self.spans.setdefault(segment.label, []).append(segment.span)

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, label: str) -> torch.Tensor:
        q_label, _, kv_label = label.partition("<-")
        kv_label = kv_label or q_label
        # checked rather than sliced: `k[0:9]` of a five-row tensor quietly returns five rows
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
    """Bind as the engine does: resources planned from this step, under the declared keys."""
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
    """The reference's iteration up to the velocity (``gr00t_n1d7.py:401-432``) on this
    node's children: encode, add positions, state row in front, DT, decode, last
    ``action_horizon`` rows. No Euler update: that is what the dtype tests compare."""
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
    """GR00T's time runs the other way from the image models' ``FlowMatchSchedule.build``:
    sigmas ascend from 0, so ``sigma_next - sigma`` is the reference's ``dt = +1/4``, and the
    timesteps are the reference's buckets. Reversed, the DT is conditioned on the wrong time
    and the trajectory integrates backwards, still finite and plausible.
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
    steps = schedule.sigmas[1:] - schedule.sigmas[:-1]
    assert torch.equal(steps, torch.full((_STEPS,), _REF_DT))


def test_the_steps_scalars_come_off_our_schedule_at_every_step():
    """The base slices ``sigmas[k]``, ``sigmas[k + 1]`` and ``timesteps[k]`` out of our
    schedule at all four iterations (the only place the terminal sigma is read), and vetoes
    a fifth step, which async speculation would otherwise run.
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


def test_the_41_rows_reach_the_dt_and_the_40_come_back():
    """The DT sees 41 rows (state row + 40 action rows); the latents carry 40. The 41 sizes
    the attention segment, capture bucket and ``input_seq_len``; the 40 sizes the trajectory.
    Swapped, both flow through without a shape error, so this pins the declaration, the DT's
    actual input and the output against each other.
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
    assert node.num_tokens(_SHAPE) == _ROWS == 1 + node.config.action_horizon
    assert node.seed_latents(None, _SHAPE, torch.Generator().manual_seed(0)).shape == (_HORIZON, _ACTION_DIM)
    # the token count does not move with the spans; the spans are declared separately
    assert node.num_tokens(Gr00tN17Shape(_TXT * 3, _IMG * 7)) == _ROWS


# ----------------------------------------------------------------------------
# The loop state's dtype: the weights', because the reference's is
# ----------------------------------------------------------------------------


def _ref_euler(actions: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    """The reference's Euler update, ``actions + dt * pred_velocity * vel_strength``
    (``gr00t_n1d7.py:435``), with ``dt`` a Python float (``:355``) and ``vel_strength`` the
    ``ones_like`` used without RTC (``:356``). Nothing upcast: at a bf16 model it is all bf16.
    """
    return actions + _REF_DT * velocity * torch.ones_like(actions)


def test_the_carried_dtype_is_the_weights_and_does_not_change_mid_loop():
    """The carried trajectory is in the weights' dtype at every iteration, because the
    reference's is: it is served bf16 (``gr00t_policy.py:111``), draws ``actions`` at
    ``vl_embeds.dtype`` (``gr00t_n1d7.py:347-351``) and steps with a Python-float ``dt``
    (``:435``). Do not "improve" it to an fp32 carry: over four steps that moves the
    trajectory 0.0243 from the reference, while an off-by-one in the DT's block alternation moves it
    0.0391, so a parity tolerance admitting the first is within 1.6x of admitting the second.

    It must also not change mid-loop: ``euler_step`` returns the velocity's dtype and
    ``seed_latents`` the weights'; if they differed, one bucket would need two graphs and the
    loop would quietly run eager.
    """
    for dtype in (torch.bfloat16, torch.float32):
        node = _node(dtype=dtype)
        cond = _stacked_cond(node.config, dtype=dtype)
        start = _latents(1, dtype=dtype)

        carried, seen = start, []
        for k in range(_STEPS):
            carried = node.denoise(None, _SHAPE, latents=carried, **_scalars(k), **cond)
            seen.append(carried.dtype)

        # the premise: the forward is in `dtype`, so what `euler_step` returns in the
        # velocity's dtype is testable at all
        assert _by_hand(node, carried, cond, _scalars(0)["timestep"]).dtype is dtype
        assert seen == [dtype] * _STEPS, seen
        assert carried.shape == (1, _HORIZON, _ACTION_DIM)
        # the same dtype the loop's own seed arrives in, which is what makes it stable: the
        # carried dtype is the seed's dtype and not a second decision
        assert node.seed_latents(None, _SHAPE, torch.Generator().manual_seed(0)).dtype is dtype
        assert not torch.equal(carried, start)  # the four steps moved the trajectory


def test_four_bf16_iterations_are_the_references_arithmetic_to_the_bit():
    """Four steps fed back as the Loop feeds them, ``torch.equal`` to the reference's bf16
    chain at every iteration. ``euler_step`` sums in fp32 and rounds once, which is
    bit-identical to bf16 ``a + dt * v`` only because ``dt = 1/4`` is a power of two (a ``dt``
    that is not need not round the same). The fp32 carry, stepped alongside, must differ.
    """
    node = _node(dtype=torch.bfloat16)
    cond = _stacked_cond(node.config, dtype=torch.bfloat16)
    start = _latents(1, dtype=torch.bfloat16)

    ours = reference = start
    fp32_carry = start.float()  # the better integrator, and not the reference's
    for k in range(_STEPS):
        scalars = _scalars(k)
        # the hand-written chain steps by the schedule's own dt, so it is not a second
        # schedule that can disagree with the node's
        assert float(scalars["sigma_next"] - scalars["sigma"]) == _REF_DT

        ours = node.denoise(None, _SHAPE, latents=ours, **scalars, **cond)
        velocity = _by_hand(node, reference, cond, scalars["timestep"])
        assert velocity.dtype is torch.bfloat16  # the premise: the forward *is* bf16
        reference = _ref_euler(reference, velocity)
        assert reference.dtype is torch.bfloat16
        assert torch.equal(ours, reference), f"iteration {k}"

        # `_by_hand` casts the fp32 carry down for the forward; only the carried state stays wide
        fp32_carry = fp32_carry + _REF_DT * _by_hand(
            node, fp32_carry, cond, scalars["timestep"],
        ).float()

    assert not torch.equal(ours, start)
    assert not torch.equal(ours.float(), fp32_carry)


# ----------------------------------------------------------------------------
# The bucket key, read off the request's step metadata
# ----------------------------------------------------------------------------


def test_the_bucket_key_is_the_two_span_counts_off_step_metadata():
    """The span lengths are needed before the bridge's tensors arrive, so they ride in
    ``step_metadata``. Missing is an error, not a default: a zero would key a bucket whose
    spans are not the request's and plan the segments at the wrong lengths.
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
    """A key that ignored a field would stack two requests whose conditioning is not the same
    rectangle. Each field is checked separately: the real image span is constant at 448, so a
    key that dropped ``image_rows`` would pass every recorded request.
    """
    node = _node()
    # distinct request ids per case: the base caches a request's key for the walk, so the
    # same id with different metadata would be answered from that cache
    same = [_info("r0"), _info("r1")]
    other_text = [_info("r0"), _info("r2", text=_TXT + 1)]
    other_image = [_info("r0"), _info("r3", image=_IMG + 1)]
    assert node.can_batch(None, _declare(node, ("r0", "r1"), same)[0]) is True
    assert node.can_batch(None, _declare(node, ("r0", "r2"), other_text)[0]) is False
    assert node.can_batch(None, _declare(node, ("r0", "r3"), other_image)[0]) is False
    # a single row never batches, whatever its key
    assert node.can_batch(None, _declare(node, ("r0",))[0]) is False
    # and the graph bucket follows the same key: a mixed batch matches no capture
    assert node.cg_key_info(_WALK, {info.request_id: info for info in same}) == _SHAPE
    assert node.cg_key_info(_WALK, {info.request_id: info for info in other_text}) is None
    assert node.cg_key_info(_WALK, {info.request_id: info for info in other_image}) is None


def _off_group(node: Gr00tN17DenoiseStepSubmodule, infos: dict[int, CurrentForwardPassInfo]):
    """``off_group_rids`` over one batch, grouped as ``Engine.capture_group`` groups a
    captured walk: by ``cg_key_info`` per rid if the node opts in, else not at all."""
    def capture_group(rid: int):
        if not node.split_batches_by_capture_key(_WALK):
            return None
        return node.cg_key_info(_WALK, {rid: infos[rid]})

    batch = ScheduledBatch(_LOOP, _WALK, request_to_worker_graph={rid: 0 for rid in infos})
    return off_group_rids(batch, capture_group), [capture_group(rid) for rid in infos]


def test_the_scheduler_batches_each_prompt_length_apart(monkeypatch):
    """A DT batch mixing prompt lengths matches no capture and runs eager one request at a
    time, so the node opts in and the scheduler's grouping keeps the lengths apart."""
    node = _node()
    assert node.split_batches_by_capture_key(_WALK) is True
    # the real lengths: only the text span differs
    infos = {
        rid: _info(f"r{rid}", text=text, image=448)
        for rid, text in enumerate((13, 21, 13, 21))
    }

    off, groups = _off_group(node, infos)
    assert groups == [Gr00tN17Shape(13, 448), Gr00tN17Shape(21, 448)] * 2
    # the first rid anchors: the other length is held back, the same length stays
    assert off == {1, 3}
    assert _off_group(node, {rid: infos[rid] for rid in (0, 2)})[0] == set()

    # opted out, every rid is in one group and the mixed batch goes ahead whole
    monkeypatch.setattr(node, "split_batches_by_capture_key", lambda graph_walk: False)
    off, groups = _off_group(node, infos)
    assert groups == [None] * 4
    assert off == set()


# ----------------------------------------------------------------------------
# The conditioning: no batch dim, and the capture's twin of it
# ----------------------------------------------------------------------------

# rank of each conditioning entry for one request; a leading 1 would reach `denoise` as a
# spurious axis (`[B, 1, T, width]` rows, a `[B, 1]` embodiment id).
_REQUEST_RANKS = {"vl_text": 2, "vl_image": 2, "state": 2, "embodiment_id": 0}


def test_the_request_inputs_carry_no_batch_dim_and_stack_into_the_forwards_shapes():
    """``request_inputs`` returns each tensor without a batch dim, since ``preprocess``
    stacks the batch. The state token ``[1, 1, D]`` must become ``[1, D]``: left at rank 3 it
    stacks to ``[B, 1, 1, D]`` and the concatenation with the action rows fails or broadcasts.
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
    """The engine's path (``prepare_inputs -> declare_step -> preprocess ->
    forward_batched``) returns the rows of one stacked ``denoise``, each from its own
    conditioning and embodiment. Weights are gathered by ``W[cat_ids]``, so one row's id
    broadcast over the batch changes no shape; this is the only test that would see it.
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

    # changing row 0's embodiment moves row 0 and leaves row 1 bit-identical (same shapes on
    # both sides, so no reduction order differs)
    moved = node.denoise(
        None, _SHAPE,
        **{name: stacked[name] for name in (LATENTS, "sigma", "sigma_next", "timestep")},
        **{name: stacked[name] for name in _REQUEST_RANKS if name != "embodiment_id"},
        embodiment_id=torch.tensor([_EMBODIMENT, _EMBODIMENT]),
    )
    assert not torch.equal(moved[0], batched[0])
    assert torch.equal(moved[1], batched[1])


def test_the_captures_placeholders_are_the_shapes_and_dtypes_a_real_row_brings():
    """A graph captured on ``capture_request_inputs`` replays on what ``request_inputs`` and
    the seeded noise produce. A shape mismatch misses the capture and runs eager; a dtype
    mismatch replays over a buffer the kernels reinterpret. Every dtype is the model's, latents
    included; both dtypes run because a hardcoded placeholder dtype is wrong on exactly one.
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
        # spelled out as well: the loop above passes if both sides are wrong together
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
    """A compiled region guards on the callables' identity, so fresh ones per step retrace
    every step. After a rebind the old callables would run the retired resource's plan.
    Identity is checked per span too: a cached dict of fresh callables would still retrace.
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
    """``sa`` is self-attention on ``DT_ATTN``; the bridge spans are cross pairs on
    ``DT_XATTN`` under ``cross_label(sa, span)``. A ragged plan is keyed by label, so a callable
    asking for an undeclared label finds no plan.
    """
    node = _node()
    _, step = _declare(node)
    _bind(node, step)
    attends = node.attends()
    resources = node.node_resources

    assert set(attends) == {SA_SPAN, VL_TEXT_SPAN, VL_IMAGE_SPAN}
    ragged = {span: fn.keywords["ragged"] for span, fn in attends.items()}
    assert all(isinstance(fn, RaggedAttentionCallable) for fn in ragged.values())
    assert ragged[SA_SPAN].attn is resources[DT_ATTN]
    assert ragged[SA_SPAN].label == SA_SPAN
    for q_label, kv_label in DT_XATTN_PAIRS:
        assert ragged[kv_label].attn is resources[DT_XATTN]
        assert ragged[kv_label].label == cross_label(q_label, kv_label)
    # every label a callable asks for is a label the step's resource declared
    for key, span_labels in ((DT_ATTN, [SA_SPAN]), (DT_XATTN, [SA_SPAN, VL_TEXT_SPAN, VL_IMAGE_SPAN])):
        assert {segment.label for segment in step.steps[key].segments} == set(span_labels)


def test_nothing_bound_leaves_attends_empty_and_the_dt_reads_every_span_with_sdpa():
    """Nothing bound is the parity backend: every span runs on SDPA. Unbound spans are left
    out, not mapped to ``None``, which would reach ``joint_attention`` and raise mid-forward.
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
    """All or nothing: the node declares both resources, so a half binding plans a kernel the
    forward does not use and a captured graph records SDPA where the plan says ragged, with
    the arithmetic still right. Regression: partial binding was once silent.
    """
    node = _node()
    _, step = _declare(node)
    _bind(node, step, keys=bound)

    with pytest.raises(RuntimeError, match=missing):
        node.attends()


def test_the_two_resources_planned_from_the_step_are_the_sdpa_fallback_at_one_request():
    """At one request the ragged path is the SDPA fallback exactly, so this tests the wiring:
    a span declared at the wrong length, a cross pair reversed, or a callable bound to the
    other resource all give a different tensor.
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
    """``DT_ATTN`` plans the ``sa`` segments; ``DT_XATTN`` plans ``sa + text + image`` with
    ``DT_XATTN_PAIRS`` (a cross plan needs both ends of each pair). Order is pinned, grouped by
    label, which is how a request's queries line up with its own keys. Top-level
    ``segments`` stays None so neither resource inherits the other's layout.
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
    """A batch of two keys is declared at each row's own lengths, not the first row's, and
    its graph key is None (matches no capture)."""
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
    """``declare_step`` builds its segments without the scaffold's ``attention_segments``,
    so the two describe one layout twice; a span added to only one leaves a label unplanned.
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
    """The node passes ``attn_resource_key=None`` and declares its two resources itself.
    Left set, the base would plan a third resource that no layer reads, and ``ragged_for``
    would hand the DT callables bound to it.
    """
    node = _node()

    assert node.attn_resource_key is None
    assert node.loop_back_names == (LATENTS,)  # one Euler step, no solver state to carry
    assert node.ragged_for(SA_SPAN) is None
    assert node.max_batch_size(_WALK) == 8


# ----------------------------------------------------------------------------
# The checkpoint's own numbers, on `meta`
# ----------------------------------------------------------------------------


@needs_checkpoint
def test_the_checkpoints_numbers_are_the_ones_these_tests_run():
    """Ties the spelled-out counts to the checkpoint. Built on ``meta``: the real DT is
    1.12B parameters, 4.2 GiB per fp32 copy."""
    config = Gr00tN17Config.from_hf(GR00T, BACKBONE)
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
