"""Contract tests for the Waypoint serving shell: the model, its node
submodule, and the resources it declares.

Not the DiT, the ring numerics, or the weight remap -- those have their own
suites. What's under test is the shell layer between the engine and a working
model: ring geometry, frame_pos rank, the ``_seq_dim`` collision, noise
statelessness, the off-by-one stop, admit-queue sizing, resource binding, and
decode ordering -- each of which fails silently when it is wrong.

CPU-only, checkpoint-free, no engine. Uses the real 720P config since its
numbers are the ones that collide; the DiT is built on ``torch.device("meta")``
and never materialized (a real 720P bf16 build is ~2.6 GB). Two exceptions read
a value back: the noise draw takes an explicit device, and the frame-clock
tests run over ``_HostOnlyDit``. The VAE section runs on the 360P config with a
fake ``taehv`` package.
"""

import dataclasses
import importlib
import logging
import pathlib
import sys

import pytest
import torch
import yaml

sys.path.insert(0, ".")

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import (
    AttentionSpec,
    AttnBackend,
    KVSpec,
    RingKVConfig,
    RingKVStep,
    StepContext,
)
from mstar.engine.resources.runner import topo_sort
from mstar.graph.base import GraphEdge, Loop, Sequential, SpeculativeNodeInfo
from mstar.graph.graph_io import WorkerGraphIO
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs
from mstar.model.waypoint.components.attention import WaypointAttention
from mstar.model.waypoint.components.dit import WaypointDiT
from mstar.model.waypoint.components.taehv import decode_latent, initial_decoder_histories
from mstar.model.waypoint.config import (
    WaypointConfig,
    waypoint_1_5_1b_360p,
    waypoint_1_5_1b_720p,
)
from mstar.model.waypoint.submodules import (
    ATTN_RESOURCE,
    KV_RESOURCE,
    PRIME_WALK,
    ROLLOUT_LOOP_NAME,
    ROLLOUT_WALK,
    WaypointDitSubmodule,
    WaypointVaeEncoderSubmodule,
    _rollout_capture_batch_sizes,
)
from mstar.model.waypoint.waypoint_model import (
    DIT_NODE,
    VAE_ENCODER_NODE,
    WaypointModel,
)


@pytest.fixture(scope="module")
def config():
    return waypoint_1_5_1b_720p()


@pytest.fixture(scope="module")
def model():
    return WaypointModel(skip_weight_loading=True)


@pytest.fixture(scope="module")
def submodule(config):
    """The real submodule over a meta-built DiT.

    Meta, not a stub: ``bind_node_resources`` walking ``self.modules()`` and
    ``self.dit.dtype`` surviving ``cast_serving_dtypes`` are under test here,
    and a stub would assert them against itself. Meta also keeps
    ``prepare_inputs``' shapes/dtypes honest while allocating nothing.

    ``_FakeTaehv`` (defined later; fixtures resolve names at call time) only
    needs its structural facts to build shapes, not real weights.
    """
    with torch.device("meta"):
        dit = WaypointDiT(config)
    dit.cast_serving_dtypes()
    return WaypointDitSubmodule(dit, _FakeTaehv(), config)


class _HostOnlyDit(torch.nn.Module):
    """Stands in for the DiT in tests that need to read a value back.

    ``prepare_inputs`` only touches the DiT for ``.dtype`` and ``get_device``
    for one parameter, so this exercises host-side bookkeeping only. The meta
    build above can't be read back (``.item()`` raises on a meta tensor), and
    materializing 720P just to read a frame counter isn't worth the memory.
    """

    def __init__(self, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1, dtype=dtype))

    @property
    def dtype(self) -> torch.dtype:
        return self.marker.dtype


@pytest.fixture
def host_submodule(config):
    return WaypointDitSubmodule(_HostOnlyDit(), _FakeTaehv(), config)


def _history_outputs(submodule: WaypointDitSubmodule) -> dict:
    """A plausible ``outputs`` dict for the nine decoder histories, for tests
    that call ``postprocess`` directly (bypassing ``forward``) to isolate the
    clock bookkeeping it also does."""
    return {
        f"decoder_history_{idx}": [value]
        for idx, value in enumerate(submodule._zero_histories(submodule.get_device()))
    }


def _fwd_info(
    request_id: str = "r0",
    graph_walk: str = ROLLOUT_WALK,
    random_seed: int = 1234,
    num_steps: int = 8,
    loop_iter: int | None = None,
) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=request_id,
        graph_walk=graph_walk,
        fwd_index=0,
        random_seed=random_seed,
        max_tokens=0,
        resource_configs={},
        step_metadata={"is_prefill": graph_walk == PRIME_WALK, "num_steps": num_steps},
        resource_publish_info={},
        loop_stop_times={},
        dynamic_loop_iter_counts=(
            {} if loop_iter is None else {ROLLOUT_LOOP_NAME: loop_iter}
        ),
    )


def _controller_stream(config, frames: int) -> dict[str, list[torch.Tensor]]:
    """A scripted stream shaped the way ``process_prompt`` emits it."""
    return {
        "mouse": [torch.zeros((1, frames, 2))],
        "button": [torch.zeros((1, frames, config.n_buttons))],
        "scroll": [torch.zeros((1, frames, 1))],
    }


# ---------------------------------------------------------------------------
# Declared resources
# ---------------------------------------------------------------------------


def test_declares_exactly_the_ring_and_the_flex_attention_over_it(model):
    specs = model.get_node_resources()
    assert [type(s) for s in specs] == [KVSpec, AttentionSpec]
    assert [s.resource_key for s in specs] == [KV_RESOURCE, ATTN_RESOURCE]
    assert all(s.nodes == {DIT_NODE} for s in specs)

    assert isinstance(specs[0].config, RingKVConfig)
    # FLEX, not the FLASHINFER default: a paged kernel reassociates the
    # accumulation over KV blocks, breaking bit-exactness against the reference.
    assert specs[1].config.backend is AttnBackend.FLEX
    assert specs[1].config.kv_cache == KV_RESOURCE


def test_ring_geometry_is_copied_from_the_config_layer_for_layer(model, config):
    """Waypoint's 24 layers are not alike, and none of the three per-layer
    numbers is derivable from the others under the compacted global ring.
    Asserted one layer at a time so a summarizing regression names the layer it
    broke."""
    ring = model.get_node_resources()[0].config
    assert ring.num_layers == config.n_layers == 24
    assert ring.num_kv_heads == config.n_kv_heads
    assert ring.num_qo_heads == config.n_heads
    assert ring.head_dim == config.d_head
    assert ring.tokens_per_frame == config.tokens_per_frame
    # Sizing is a deployment question (`apply_yaml_overrides` runs after this
    # hook); what's pinned here is the *default* of one session.
    assert ring.num_sessions == 1

    assert len(ring.layers) == config.n_layers
    for i, layer in enumerate(ring.layers):
        assert layer.ring_frames == config.ring_frames(i), f"ring_frames, layer {i}"
        assert layer.ring_buckets == config.ring_buckets(i), f"ring_buckets, layer {i}"
        assert layer.pinned_dilation == config.pinned_dilation(i), f"dilation, layer {i}"

    # The heterogeneity itself: if these two collapse to one value the loop
    # above would pass against a config that had also been flattened.
    assert {ring.layers[i].pinned_dilation for i in config.global_layers} == {
        config.global_pinned_dilation
    }
    local = set(range(config.n_layers)) - config.global_layers
    assert {ring.layers[i].pinned_dilation for i in local} == {1}


def test_ring_frames_and_buckets_are_read_as_two_separate_questions():
    """They agree on every layer of the compacted 720P ring, which is exactly
    what makes deriving one from the other look safe. Under the reference's
    sizing (``full_global_ring=True``) a global layer is 128 frames indexed
    by 16 buckets, and a derived value serves that layer a shredded history."""
    model = WaypointModel(skip_weight_loading=True)  # fresh: mutated below
    model.config = dataclasses.replace(model.config, full_global_ring=True)
    ring = model.get_node_resources()[0].config

    layer = ring.layers[min(model.config.global_layers)]
    assert layer.ring_frames == model.config.global_window == 128
    assert layer.ring_buckets == 16
    assert layer.ring_frames != layer.ring_buckets


def test_attention_resolves_after_the_cache_it_names(model):
    specs = model.get_node_resources()
    by_key = {s.resource_key: s for s in specs}
    assert by_key[ATTN_RESOURCE].depends_on() == {KV_RESOURCE}
    assert by_key[KV_RESOURCE].depends_on() == set()
    # topo_sort only calls depends_on(), so the specs stand in for the built
    # resources here; the order it returns is the order the engine builds in.
    assert topo_sort(by_key) == (KV_RESOURCE, ATTN_RESOURCE)


# ---------------------------------------------------------------------------
# Graph walks
# ---------------------------------------------------------------------------


def test_prime_encodes_commits_and_initializes_decoder_without_emitting(model):
    """Encoding the seed and dropping the latent would prime the ring but not
    the decoder: the functional decoder spends its first call on
    ``frames_to_trim`` of temporal memory, so the first rollout frame would
    silently pay for it instead."""
    walks = model.get_graph_walk_graphs()
    assert set(walks) == {PRIME_WALK, ROLLOUT_WALK}
    assert model.nodes == [DIT_NODE, VAE_ENCODER_NODE]

    prime = walks[PRIME_WALK]
    assert isinstance(prime, Sequential)
    assert [s.name for s in prime.sections] == [VAE_ENCODER_NODE, DIT_NODE]
    encoder, dit = prime.sections
    assert encoder.input_names == {"image_inputs"}
    assert [(e.name, e.next_node) for e in encoder.outputs] == [("latent", DIT_NODE)]
    # The dit node's contract is unchanged by the node bracketing it.
    assert dit.input_names == {"latent", "mouse", "button", "scroll"}
    # Nothing to route: the decode happens inside this forward, and the
    # reconstructed seed frames it produces are internal-only.
    assert dit.outputs == []

    io = prime.get_inputs_outputs()
    assert io.ext_inputs == {
        ("image_inputs", VAE_ENCODER_NODE),
        ("mouse", DIT_NODE), ("button", DIT_NODE), ("scroll", DIT_NODE),
    }
    assert io.ext_outputs == []


def test_the_rollout_loop_decodes_and_emits_every_iteration(model, config):
    rollout = model.get_graph_walk_graphs()[ROLLOUT_WALK]
    assert isinstance(rollout, Loop)
    assert rollout.name == ROLLOUT_LOOP_NAME  # what check_stop's signal is keyed by
    assert rollout.max_iters == config.max_frames

    dit = rollout.section
    assert dit.name == DIT_NODE
    assert dit.input_names == {"mouse", "button", "scroll", "clock"}
    assert {(e.name, e.next_node) for e in dit.outputs} == {
        ("clock", DIT_NODE), ("video_output", EMIT_TO_CLIENT),
    }
    assert rollout.accumulated_outputs == []
    # The "clock" self loop-back makes the dit a same-node speculation target
    # (see below), so async scheduling can dispatch iteration N+1 before N
    # finishes; overshoot is vetoed host-side in prepare_inputs, not by this.
    assert dit.enable_async_scheduling is True
    # The controller streams stay loop-external; "clock" is the only
    # loop-back, or the conductor would try to re-inject it every walk step.
    assert rollout._external_inputs == {
        ("mouse", DIT_NODE), ("button", DIT_NODE), ("scroll", DIT_NODE),
    }
    assert rollout._loop_back_inputs == {("clock", DIT_NODE)}


def test_the_clock_loop_back_makes_dit_a_same_node_speculation_target(model):
    """``ingest_for_speculation`` must propose the dit as ready for its own
    next iteration once the loop-external streams are ready and the empty
    "clock" loop-back (iteration 0) has been ingested -- the self-edge this
    graph shape depends on for speculation to fire at all.
    """
    rollout = model.get_graph_walk_graphs()[ROLLOUT_WALK]
    wgio = WorkerGraphIO(rollout)
    dit = wgio.get_node(DIT_NODE)
    for name in ("mouse", "button", "scroll"):
        wgio.ingest_input(GraphEdge(next_node=DIT_NODE, name=name, persist=True))
    wgio.ingest_input(GraphEdge(next_node=DIT_NODE, name="clock"))
    assert wgio.ready_node_names == {DIT_NODE}

    assert wgio.ingest_for_speculation(dit.outputs, DIT_NODE) == [
        SpeculativeNodeInfo(node_name=DIT_NODE, is_new_loop_iter=True, loop_name=ROLLOUT_LOOP_NAME)
    ]


def test_the_rollout_loop_closes_after_exactly_num_steps_frames(model):
    """Driven through ``WorkerGraphIO``: exercises the loop's own
    iteration/finish-signal bookkeeping, not node ordering (there's only one
    node, so nothing can decode a frame twice or skip one).

    ``register_loop_finish_signal`` (what ``check_stop`` calls) fires during
    postprocess of the loop's last iteration, so that iteration's frame must
    still be emitted before the loop reports done. The overshoot guard for a
    speculative iteration dispatched before the signal lands is separate --
    the host-side veto in ``prepare_inputs``, checked below.
    """
    num_steps = 3
    rollout = model.get_graph_walk_graphs()[ROLLOUT_WALK]
    wgio = WorkerGraphIO(rollout)
    for name in ("mouse", "button", "scroll"):
        wgio.ingest_input(GraphEdge(next_node=DIT_NODE, name=name, persist=True))
    wgio.ingest_input(GraphEdge(next_node=DIT_NODE, name="clock"))

    emitted = []
    for step in range(num_steps):
        assert wgio.ready_node_names == {DIT_NODE}
        wgio.ready_node_names.discard(DIT_NODE)
        if step == num_steps - 1:
            wgio.register_loop_finish_signal(ROLLOUT_LOOP_NAME)  # what check_stop does
        completion = wgio.mark_node_complete(DIT_NODE)
        # mark_node_complete already stripped filtered_signals (e.g. the final
        # iteration's "clock" loop-back) from output_edges; the rest is ours to route.
        for edge in completion.output_edges:
            if edge.next_node == EMIT_TO_CLIENT:
                emitted.append(edge.name)
                continue
            wgio.ingest_input(edge)
        assert rollout.is_done == (step == num_steps - 1)

    assert emitted == ["video_output"] * num_steps


def test_the_overshoot_veto_fires_only_past_num_steps(submodule):
    """The other half of the guarantee above: a speculative iteration must
    never reach a forward once ``num_steps`` is spent, and never fires
    during prime (no ``rollout_step`` to overshoot).
    """
    num_steps = 3
    rid = "overshoot"
    submodule.request_states.pop(rid, None)
    inputs = _controller_stream(submodule.config, frames=num_steps)

    for expected_step in range(num_steps):
        prepared = submodule.prepare_inputs(
            ROLLOUT_WALK, _fwd_info(request_id=rid, num_steps=num_steps), inputs
        )
        assert prepared is not None, f"step {expected_step} must not be vetoed"
        submodule.postprocess(
            rid, _fwd_info(request_id=rid, num_steps=num_steps),
            {"video_output": [torch.zeros(1)], **_history_outputs(submodule)},
        )

    vetoed = submodule.prepare_inputs(
        ROLLOUT_WALK, _fwd_info(request_id=rid, num_steps=num_steps), inputs
    )
    assert vetoed is None, "an async-overshoot iteration must be vetoed, not run"

    # Prime never overshoots: it has no num_steps to compare rollout_step
    # against, regardless of how far along the rollout counter is.
    prime_inputs = {
        **_controller_stream(submodule.config, frames=1),
        "latent": [torch.zeros((1, 1, *submodule.config.latent_shape))],
    }
    assert submodule.prepare_inputs(
        PRIME_WALK, _fwd_info(request_id=rid, graph_walk=PRIME_WALK, num_steps=num_steps),
        prime_inputs,
    ) is not None

    submodule.cleanup_request(rid)


# ---------------------------------------------------------------------------
# Per-step inputs: rank and the _seq_dim collision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("walk", [PRIME_WALK, ROLLOUT_WALK])
def test_no_prepared_tensor_is_rank_zero(submodule, config, walk):
    """A 0-dim tensor reaches ``_intern_static_buffer`` (reads
    ``stored.shape[0]``) and the worker's output fanout (reads ``dims[0]``)
    -- both IndexError at capture/warmup, far from the line that made it."""
    inputs = _controller_stream(config, frames=4)
    inputs["latent"] = [torch.zeros((1, 1, *config.latent_shape))]
    node_inputs = submodule.prepare_inputs(walk, _fwd_info(graph_walk=walk), inputs)

    assert node_inputs.tensor_inputs, "prepare_inputs emitted no tensors"
    for name, tensor in node_inputs.tensor_inputs.items():
        assert tensor.ndim >= 1, f"{name} is rank-0"

    prepared = submodule.preprocess(walk, None, [node_inputs])
    for name, value in prepared.items():
        assert isinstance(value, torch.Tensor), name
        assert value.ndim >= 1, f"{name} is rank-0 after preprocess"


@pytest.mark.parametrize("walk", [PRIME_WALK, ROLLOUT_WALK])
def test_no_prepared_tensor_carries_tokens_per_frame_in_its_shape(
    submodule, config, walk
):
    """``CudaGraphRunner._seq_dim`` finds the dim equal to ``input_seq_len``
    and hoists it to the front of a shared static buffer; a tensor carrying
    that number for an unrelated reason gets silently transposed under replay.

    At 720P nothing collides, but the margin is thin: a 256-token-per-frame
    variant would put ``button``'s ``n_buttons=256`` in the crosshairs. With
    ``step_batch_size > 1``, 360p at bs=2 does collide with ``n_buttons``;
    that case is handled via the config's ``input_seq_dims`` override instead
    (see ``get_cuda_graph_configs`` and ``test_cuda_graph_capture.py``).
    """
    inputs = _controller_stream(config, frames=4)
    inputs["latent"] = [torch.zeros((1, 1, *config.latent_shape))]
    node_inputs = submodule.prepare_inputs(walk, _fwd_info(graph_walk=walk), inputs)

    assert node_inputs.input_seq_len == config.tokens_per_frame
    for name, tensor in node_inputs.tensor_inputs.items():
        assert config.tokens_per_frame not in tuple(tensor.shape), (
            f"{name} has shape {tuple(tensor.shape)}, which carries "
            f"input_seq_len={config.tokens_per_frame}"
        )


def test_prepared_shapes_and_dtypes_are_the_capture_template_exactly(submodule, config):
    """``_capture_one`` bakes the config's template; replay re-stages only
    what ``preprocess`` returns into it. A mismatched key, shape, or dtype is
    either a stale-address read or a silent eager fallback -- asserted
    against each other rather than a literal for that reason."""
    templates = {
        cfg.capture_graph_walk: cfg.single_request_inputs
        for cfg in submodule.get_cuda_graph_configs(torch.device("meta"))
    }
    inputs = _controller_stream(config, frames=4)
    inputs["latent"] = [torch.zeros((1, 1, *config.latent_shape))]

    for walk, template in templates.items():
        prepared = submodule.prepare_inputs(walk, _fwd_info(graph_walk=walk), inputs)
        assert set(prepared.tensor_inputs) == set(template.tensor_inputs), walk
        assert prepared.input_seq_len == template.input_seq_len, walk
        for name, tensor in prepared.tensor_inputs.items():
            baked = template.tensor_inputs[name]
            assert tensor.shape == baked.shape, f"{walk}/{name}"
            assert tensor.dtype == baked.dtype, f"{walk}/{name}"

    # The one shape that is spelled out, because it is the one the runner
    # indexes into and the one a "scalar frame index" instinct would get wrong.
    frame_pos = templates[ROLLOUT_WALK].tensor_inputs["frame_pos"]
    assert frame_pos.shape == (1,) and frame_pos.dtype == torch.int64


def test_input_seq_dims_covers_every_static_input_on_dim_zero(submodule):
    """Every tensor the DiT's config templates -- including ``button``, whose
    ``n_buttons`` can collide with a bucket's token count -- is a per-request
    row ``preprocess`` concatenates on dim 0. ``input_seq_dims`` must declare
    exactly those keys, all on dim 0, or the runner falls back to its
    size-based guess for whichever key is missing."""
    for cfg in submodule.get_cuda_graph_configs(torch.device("meta")):
        assert cfg.input_seq_dims is not None, cfg.capture_graph_walk
        assert set(cfg.input_seq_dims) == set(cfg.single_request_inputs.tensor_inputs), (
            cfg.capture_graph_walk
        )
        assert set(cfg.input_seq_dims.values()) == {0}, cfg.capture_graph_walk


# ---------------------------------------------------------------------------
# Noise: stateless in (seed, frame_pos)
# ---------------------------------------------------------------------------


def test_noise_is_a_pure_function_of_seed_and_frame_pos(submodule):
    """The draw must not depend on how many frames were drawn before it: a
    generator advanced in place accumulates state ``get_state`` doesn't
    serialize, so a resumed rollout would silently diverge from the one it
    resumed."""
    device, dtype = torch.device("cpu"), torch.float32
    first = submodule._frame_noise(4242, 7, device, dtype)
    # Two intervening draws: if the helper carried a generator, these would
    # move it and the repeat below would come back different.
    submodule._frame_noise(4242, 0, device, dtype)
    submodule._frame_noise(999, 7, device, dtype)
    repeat = submodule._frame_noise(4242, 7, device, dtype)
    assert torch.equal(first, repeat)


def test_noise_differs_across_frames_and_across_seeds(submodule, config):
    device, dtype = torch.device("cpu"), torch.float32
    frames = [submodule._frame_noise(4242, k, device, dtype) for k in range(4)]
    assert frames[0].shape == (1, 1, *config.latent_shape)
    for a in range(len(frames)):
        for b in range(a + 1, len(frames)):
            # Re-seeding from the request seed alone gives every frame the same
            # noise and the video stops evolving.
            assert not torch.equal(frames[a], frames[b]), f"frames {a} and {b} match"

    # Adjacent seeds must not share frame k: `seed + frame_pos` would make
    # seeds 0 and 1 agree on every frame but the first (hence the splitmix
    # finalizer).
    assert not torch.equal(
        submodule._frame_noise(0, 3, device, dtype),
        submodule._frame_noise(1, 3, device, dtype),
    )


def test_prime_is_idle_and_rollout_zero_receives_action_zero(
    host_submodule, config
):
    """Prime advances the ring clock but not the user-action cursor."""
    rid = "clock"
    host_submodule.request_states.pop(rid, None)
    inputs = _controller_stream(config, frames=2)
    # Distinguishable rows, so a slice off by one is visible.
    for row in range(2):
        inputs["scroll"][0][0, row, 0] = float(row + 1)

    prime_inputs = {
        **inputs,
        "latent": [torch.zeros((1, 1, *config.latent_shape), dtype=torch.float32)],
    }
    prime = host_submodule.prepare_inputs(
        PRIME_WALK, _fwd_info(request_id=rid, graph_walk=PRIME_WALK), prime_inputs
    )
    assert int(prime.tensor_inputs["frame_pos"][0]) == 0
    assert float(prime.tensor_inputs["scroll"][0, 0, 0]) == 0.0
    host_submodule.postprocess(
        rid, _fwd_info(request_id=rid, graph_walk=PRIME_WALK),
        _history_outputs(host_submodule),
    )

    seen = []
    for _ in range(2):
        node_inputs = host_submodule.prepare_inputs(
            ROLLOUT_WALK, _fwd_info(request_id=rid), inputs
        )
        seen.append((
            int(node_inputs.tensor_inputs["frame_pos"][0]),
            float(node_inputs.tensor_inputs["scroll"][0, 0, 0]),
        ))
        host_submodule.postprocess(
            rid, _fwd_info(request_id=rid), _history_outputs(host_submodule)
        )

    assert seen == [(1, 1.0), (2, 2.0)]
    with pytest.raises(IndexError, match="action index 2"):
        host_submodule.prepare_inputs(
            ROLLOUT_WALK, _fwd_info(request_id=rid), inputs
        )

    host_submodule.cleanup_request(rid)
    assert rid not in host_submodule.request_states


def test_declare_step_carries_the_same_clock_prepare_inputs_reads(
    host_submodule, config
):
    """One source for the clock, not two: `RingKVManager.admit` checks the
    declared frame against `commit`'s last-recorded one, so a mismatch would
    refuse valid frames and pass desynced ones -- the check inverted.

    Read on the host, off `PerRequestState`, rather than off `NodeInputs`'
    `frame_pos` (a `[1]` device tensor by then, so reading it back would sync
    every step).
    """
    rid = "declare"
    host_submodule.request_states.pop(rid, None)
    inputs = _controller_stream(config, frames=3)

    for expected in range(3):
        node_inputs = host_submodule.prepare_inputs(
            ROLLOUT_WALK, _fwd_info(request_id=rid), inputs
        )
        step = host_submodule.declare_step(
            graph_walk=ROLLOUT_WALK, request_ids=[rid], inputs=[node_inputs],
        )
        kv_step = step.get(KV_RESOURCE)

        assert isinstance(kv_step, RingKVStep), (
            "a KVStep here would admit and commit while silently switching the "
            "clock check off"
        )
        assert kv_step.frames == ((rid, expected),)
        assert dict(kv_step.frames)[rid] == int(
            node_inputs.tensor_inputs["frame_pos"][0]
        )

        host_submodule.postprocess(
            rid, _fwd_info(request_id=rid), _history_outputs(host_submodule)
        )

    host_submodule.cleanup_request(rid)


def test_declare_step_names_a_clock_for_every_request_in_the_batch(host_submodule):
    """No batch shape declines to answer: ``RingKVManager``'s continuity check
    -- the only thing standing between a stalled clock and a world quietly
    rewriting its own history -- must get a clock for every request in the
    batch, not just batches of one.

    The clocks below are genuinely different, so a declaration that broadcast
    one request's frame across the batch fails here rather than passing on a
    coincidence.
    """
    for i, rid in enumerate(("a", "b", "c")):
        host_submodule.request_states.pop(rid, None)
        for _ in range(i * 2):
            host_submodule.postprocess(
                rid, _fwd_info(request_id=rid), _history_outputs(host_submodule)
            )

    step = host_submodule.declare_step(
        graph_walk=ROLLOUT_WALK, request_ids=["a", "b", "c"], inputs=[],
    )

    assert step.get(KV_RESOURCE).frames == (("a", 0), ("b", 2), ("c", 4))
    for rid in ("a", "b", "c"):
        host_submodule.cleanup_request(rid)


# ---------------------------------------------------------------------------
# Stop condition
# ---------------------------------------------------------------------------


def test_check_stop_fires_at_exactly_num_steps(submodule):
    """N frames means firing while iteration N-1 is postprocessed: the loop
    counter still reads N-1 there and the stop ends that iteration. One early
    truncates the video; one late commits an extra frame into the ring."""
    num_steps = 6
    fired = [
        bool(
            submodule.check_stop(
                "r0", _fwd_info(num_steps=num_steps, loop_iter=k), {}
            )
        )
        for k in range(num_steps + 2)
    ]
    assert fired == [False] * (num_steps - 1) + [True, True, True]
    assert submodule.check_stop(
        "r0", _fwd_info(num_steps=num_steps, loop_iter=num_steps - 1), {}
    ) == {ROLLOUT_LOOP_NAME}


def test_check_stop_never_signals_rollout_loop_during_prime(submodule):
    assert submodule.check_stop(
        "r0",
        _fwd_info(
            graph_walk=PRIME_WALK,
            num_steps=1,
            loop_iter=0,
        ),
        {},
    ) == set()


# ---------------------------------------------------------------------------
# Serialization gate
# ---------------------------------------------------------------------------


def _write_config(tmp_path, name: str, **extra) -> str:
    body = {
        "model": "waypoint",
        "max_seq_len": 512,
        "node_groups": [
            {"node_names": [VAE_ENCODER_NODE, DIT_NODE], "ranks": [0]}
        ],
        **extra,
    }
    path = tmp_path / name
    path.write_text(yaml.safe_dump(body))
    return str(path)


def _validate(model, path: str) -> None:
    """What the Conductor does at startup with the deployment YAML."""
    model.validate_config_yaml(yaml.safe_load(pathlib.Path(path).read_text()), path)


def _sessions(n: int) -> dict:
    """The ``resources:`` block a deployment writes to size the ring — the same
    one ``EngineManager.build`` feeds to ``apply_yaml_overrides``, which is why
    the gate reads it here rather than inventing its own key."""
    return {"resources": {KV_RESOURCE: {"num_sessions": n}}}


@pytest.mark.parametrize("limit", [None, 0, -1, True, 1.0, "2"])
def test_validate_config_yaml_refuses_a_deployment_with_no_admit_queue(
    model, tmp_path, limit
):
    """The pool is finite, and this is the primary gate on it.

    The conductor only forms a FIFO admit queue when
    ``max_concurrent_requests`` is set; unset, every request is admitted on
    arrival and anything past the Nth dies terminally at
    ``RingKVManager.admit``, too late to queue.

    ``max_batch_size = 1`` doesn't cover this -- it caps requests per step,
    not how many may exist at once.

    ``True`` is in the parametrize list because ``isinstance(True, int)`` is
    ``True`` in Python, so ``max_concurrent_requests: true`` would otherwise
    silently read as 1.
    """
    extra = {} if limit is None else {"max_concurrent_requests": limit}
    path = _write_config(tmp_path, f"reject_{limit}.yaml", **extra)
    with pytest.raises(ValueError, match="max_concurrent_requests"):
        _validate(model, path)


@pytest.mark.parametrize("worlds", [0, -1, True, 1.0, 1.9, "2"])
def test_validate_config_yaml_refuses_invalid_world_pool_size(
    model, tmp_path, worlds
):
    path = _write_config(
        tmp_path,
        f"invalid_worlds_{worlds}.yaml",
        max_concurrent_requests=1,
        **_sessions(worlds),
    )
    with pytest.raises(ValueError, match=r"resources\.kv\.num_sessions"):
        _validate(model, path)


@pytest.mark.parametrize(("limit", "worlds"), [(2, 1), (8, 4), (2, None)])
def test_validate_config_yaml_refuses_more_arrivals_than_worlds(
    model, tmp_path, limit, worlds
):
    """A queue longer than the pool is a delayed failure: the conductor admits
    ``limit`` requests, the ring hands out ``num_sessions``, and the
    difference dies at ``admit`` with an ``AdmitRuntimeError`` no retry,
    eviction, or reload can clear.

    ``worlds=None`` is the accidental case: raising
    ``max_concurrent_requests`` without touching ``resources``, the shape
    every pre-pool config already has.
    """
    extra = {"max_concurrent_requests": limit}
    if worlds is not None:
        extra |= _sessions(worlds)
    path = _write_config(tmp_path, f"over_{limit}_{worlds}.yaml", **extra)

    with pytest.raises(ValueError, match="exceeds the"):
        _validate(model, path)


@pytest.mark.parametrize(("limit", "worlds"), [(1, None), (1, 1), (4, 4), (8, 8)])
def test_validate_config_yaml_accepts_a_deployment_inside_its_pool(
    model, tmp_path, limit, worlds
):
    """``limit == num_sessions`` is the shape that should be written, at any size.
    The ``(1, None)`` case is the default deployment, which must keep working
    unchanged — the pool is a widening, not a migration."""
    extra = {"max_concurrent_requests": limit}
    if worlds is not None:
        extra |= _sessions(worlds)
    path = _write_config(tmp_path, f"ok_{limit}_{worlds}.yaml", **extra)

    _validate(model, path)
    graphs = model.get_worker_graphs(path)

    assert graphs
    assert {walk for g in graphs for walk in g.graph_walks} == {
        PRIME_WALK,
        ROLLOUT_WALK,
    }


def test_the_shipped_config_serializes_both_nodes_onto_one_rank(model):
    """``configs/waypoint.yaml`` is the deployment and must pass its own gate.

    Both nodes in one group, on rank 0: a node missing from ``node_groups``
    has no rank to run on. There's no decoder node to put a worker boundary
    in front of -- decode is fused into the dit's forward -- so a rank split
    inside the rollout loop isn't expressible any more.
    """
    path = pathlib.Path(__file__).resolve().parents[2] / "configs" / "waypoint.yaml"

    _validate(model, str(path))
    graphs = model.get_worker_graphs(str(path))

    assert {walk for g in graphs for walk in g.graph_walks} == {PRIME_WALK, ROLLOUT_WALK}
    assert {tuple(g.ranks) for g in graphs} == {(0,)}
    by_walk = {walk: g for g in graphs for walk in g.graph_walks}
    assert set(by_walk[PRIME_WALK].section.get_nodes()) == {VAE_ENCODER_NODE, DIT_NODE}
    assert set(by_walk[ROLLOUT_WALK].section.get_nodes()) == {DIT_NODE}


def test_validate_config_yaml_refuses_a_step_wider_than_the_world_pool(tmp_path):
    model = WaypointModel(skip_weight_loading=True, step_batch_size=4)
    path = _write_config(
        tmp_path, "wide_step.yaml", max_concurrent_requests=2, **_sessions(2)
    )
    with pytest.raises(ValueError, match="step_batch_size: 4"):
        _validate(model, path)


def test_validate_config_yaml_warns_about_worlds_no_request_can_reach(
    model, tmp_path, caplog
):
    """Legal, only wasteful -- a warning, not a refusal. Each unreachable
    world is ~816 MiB of ring at 720P allocated and never written, worth a
    log line rather than a failed boot."""
    path = _write_config(
        tmp_path, "underused.yaml", max_concurrent_requests=2, **_sessions(8)
    )

    with caplog.at_level(logging.WARNING):
        _validate(model, path)

    assert any("can never be filled" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Resource binding
# ---------------------------------------------------------------------------


def test_bind_reaches_the_dit_and_all_twenty_four_attention_layers(submodule):
    """One bind has to reach every caller: the 24 attention layers hold their
    own references and call ``upsert``/``attend`` directly. Anything left
    unbound raises mid-forward -- or, during warmup, poisons the capture
    instead of failing a request."""
    kv, attn = object(), object()
    submodule.bind_node_resources({KV_RESOURCE: kv, ATTN_RESOURCE: attn})

    assert submodule.node_resources[KV_RESOURCE] is kv

    layers = [m for m in submodule.modules() if isinstance(m, WaypointAttention)]
    assert len(layers) == 24
    for layer in layers:
        assert layer.kv is kv, f"layer {layer.layer_idx} unbound"
        assert layer.attn is attn, f"layer {layer.layer_idx} unbound"


def test_the_submodule_owns_the_dit_and_is_not_the_dit(submodule):
    """The structural reason the test above can pass at all.

    ``bind_node_resources`` walks ``self.modules()`` but skips ``self``; a
    submodule that *is* the DiT would never be visited, leaving the root
    unbound. Nothing on the root needs a resource today, so this is a guard
    against that changing, not a live bug.
    """
    assert not isinstance(submodule, WaypointDiT)
    assert isinstance(submodule.dit, WaypointDiT) and submodule.dit is not submodule
    # The DiT must be a *child module*, not a plain attribute -- self.modules()
    # is the only thing the walk follows.
    assert any(m is submodule.dit for m in submodule.modules())
    # The submodule itself must not define bind_resources: the walk skips
    # self, so one would never run.
    assert getattr(type(submodule), "bind_resources", None) is None


def test_binding_without_a_declared_resource_fails_at_bind(submodule):
    """Not mid-forward: layers resolve with ``.get`` (correct, since a layer
    may sit on a node owning only some resources), so the node is what still
    knows which keys it declared and can name the missing one.
    """
    for partial in ({ATTN_RESOURCE: object()}, {KV_RESOURCE: object()}, {}):
        with pytest.raises(KeyError):
            submodule.bind_node_resources(partial)


# ---------------------------------------------------------------------------
# Batch size cap
# ---------------------------------------------------------------------------


def test_max_batch_size_is_step_batch_size_for_both_walks(config):
    """Both walks carry up to ``step_batch_size`` rows -- prime rows batch
    too, when several requests are admitted in the same step."""
    batched = dataclasses.replace(config, step_batch_size=4)
    with torch.device("meta"):
        dit = WaypointDiT(batched)
    dit.cast_serving_dtypes()
    submodule = WaypointDitSubmodule(dit, _FakeTaehv(), batched)

    assert submodule.max_batch_size(PRIME_WALK) == 4
    assert submodule.max_batch_size(ROLLOUT_WALK) == 4


def test_dit_can_batch_is_true(submodule):
    """The eager path must batch too: a captured lease replays batched
    regardless, but with ``can_batch`` False a graphs-off (or
    uncaptured-shape) multi-row step falls back to one forward per request."""
    assert submodule.can_batch(batch=None, model_inputs=[]) is True


# ---------------------------------------------------------------------------
# Capture configs
# ---------------------------------------------------------------------------


def test_both_dit_walks_are_optional_captures(submodule, config):
    """Prime and rollout both capture; the declaration order is capture order."""
    configs = submodule.get_cuda_graph_configs(torch.device("meta"))
    # Rollout first: the two share one graph pool and rollout's five forwards
    # are a superset of prime's one, so rollout sizes the pool; the runner's
    # largest-first sort is stable given both specs are (1, tokens_per_frame).
    assert [cfg.capture_graph_walk for cfg in configs] == [ROLLOUT_WALK, PRIME_WALK]
    for cfg in configs:
        assert cfg.compile is False
        # One live world again: the bucket cannot be wider than it.
        assert cfg.capture_batch_sizes == [1]
        # The v1 engine always dispatches batched; a submodule captured on bare
        # `forward` is captured on a method that never runs.
        assert cfg.capture_forward_method == "forward_batched"
        assert cfg.single_request_inputs.input_seq_len == config.tokens_per_frame

    rollout_cfg, prime_cfg = configs
    # Both walks' captured buckets are a real ceiling on the eager batch size:
    # a prime batch bigger than the largest captured bucket is refused, not run eager.
    assert rollout_cfg.caps_eager_batch_size is True
    assert prime_cfg.caps_eager_batch_size is True

    rollout, prime = (cfg.single_request_inputs.tensor_inputs for cfg in configs)
    assert "noise" in rollout and "latent" not in rollout
    assert "latent" in prime and "noise" not in prime
    # The two templates are one shape with the frame tensor renamed. A
    # divergence here is a second static-input family for no reason.
    assert set(rollout) - {"noise"} == set(prime) - {"latent"}
    assert rollout["noise"].shape == prime["latent"].shape
    assert rollout["noise"].dtype == prime["latent"].dtype
    assert submodule.disable_torch_compile is True


@pytest.mark.parametrize(
    "step_batch_size, expected",
    [
        (1, [1]),
        (2, [1, 2]),
        (3, [1, 2, 3]),
        (4, [1, 2, 4]),
        (6, [1, 2, 4, 6]),
        (8, [1, 2, 4, 8]),
        (16, [1, 2, 4, 8, 16]),
    ],
)
def test_rollout_capture_batch_sizes_is_geometric(step_batch_size, expected):
    """Powers of two up to B, then B itself -- so startup grows with log B, not
    B. B that is itself a power of two ends on it once (no duplicate); a B that
    is not appends the odd top bucket the padding path rounds up to."""
    assert _rollout_capture_batch_sizes(step_batch_size) == expected


def test_both_walks_capture_the_same_geometric_buckets(config):
    """With ``step_batch_size > 1`` prime captures the same geometric buckets as
    rollout and caps eagerly at the top of them, so a multi-request prime batch
    replays a captured, padded graph rather than re-tracing eagerly."""
    batched = dataclasses.replace(config, step_batch_size=8)
    with torch.device("meta"):
        dit = WaypointDiT(batched)
    dit.cast_serving_dtypes()
    submodule = WaypointDitSubmodule(dit, _FakeTaehv(), batched)

    configs = submodule.get_cuda_graph_configs(torch.device("meta"))
    assert [cfg.capture_graph_walk for cfg in configs] == [ROLLOUT_WALK, PRIME_WALK]
    for cfg in configs:
        assert cfg.capture_batch_sizes == [1, 2, 4, 8]
        assert cfg.caps_eager_batch_size is True


def test_dit_prime_capture_can_be_declined_on_its_own(config):
    """The A/B control arm: prime off leaves the steady rollout graph alone."""
    no_prime = dataclasses.replace(config, capture_dit_prime=False)
    with torch.device("meta"):
        dit = WaypointDiT(no_prime)
    dit.cast_serving_dtypes()
    submodule = WaypointDitSubmodule(dit, _FakeTaehv(), no_prime)

    configs = submodule.get_cuda_graph_configs(torch.device("meta"))
    assert [cfg.capture_graph_walk for cfg in configs] == [ROLLOUT_WALK]
    assert "noise" in configs[0].single_request_inputs.tensor_inputs


def test_dit_declares_no_capture_when_cuda_graph_is_disabled(config):
    eager_config = dataclasses.replace(config, cuda_graph=False)
    with torch.device("meta"):
        dit = WaypointDiT(eager_config)
    dit.cast_serving_dtypes()
    eager = WaypointDitSubmodule(dit, _FakeTaehv(), eager_config)

    assert eager.get_cuda_graph_configs(torch.device("meta")) == []


# ---------------------------------------------------------------------------
# VAE nodes
# ---------------------------------------------------------------------------


class MemBlock(torch.nn.Module):
    """Small block with the pinned upstream class and shape contract."""

    def __init__(self, channels: int):
        super().__init__()
        # History-shape derivation reads this exact upstream attribute.
        self.conv = torch.nn.ModuleList([
            torch.nn.Conv2d(channels * 2, channels, 1, bias=False)
        ])

    def forward(self, current, past):
        return current + past * 0.25


class TPool(torch.nn.Module):
    def __init__(self, channels: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = torch.nn.Conv2d(channels * stride, channels, 1, bias=False)

    def forward(self, value):
        batch_time, channels, height, width = value.shape
        return self.conv(value.reshape(
            batch_time // self.stride, channels * self.stride, height, width
        ))


class TGrow(torch.nn.Module):
    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride

    def forward(self, value):
        return value.repeat_interleave(self.stride, dim=0)


class _FakeTaehv(torch.nn.Module):
    """Cheap tensor-only TAEHV with the released architecture facts."""

    patch_size = 2
    latent_channels = 32
    image_channels = 3
    t_downscale = 4
    t_upscale = 4
    frames_to_trim = 3
    is_cogvideox = False

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.ModuleList([
            torch.nn.Conv2d(12, 32, 1, stride=8, bias=False),
            TPool(32, self.t_downscale),
            *(MemBlock(32) for _ in range(9)),
        ])
        self.decoder = torch.nn.ModuleList([
            *(MemBlock(32) for _ in range(9)),
            TGrow(self.t_upscale),
        ])
        self.to(torch.bfloat16)

    def preprocess_input_frames(self, frames):
        return torch.nn.functional.pixel_unshuffle(frames, self.patch_size)

    def postprocess_output_frames(self, frames):
        return torch.nn.functional.pixel_shuffle(frames[:, :, :12], self.patch_size).clamp(0, 1)


@pytest.fixture
def taehv_weights():
    return _FakeTaehv()


@pytest.fixture
def ae_config():
    """360P, not the 720P default: the priming path decodes 16 frames per
    session and 720P would allocate tens of MB of host tensors to say the same
    thing. Latent 16x32 -> 256x512 encode grid -> 360x640 out."""
    return waypoint_1_5_1b_360p()


@pytest.fixture
def encoder(taehv_weights, ae_config):
    return WaypointVaeEncoderSubmodule(taehv_weights, ae_config)


class _FakeDit(torch.nn.Module):
    """Stands in for the DiT in the fused-decode tests below.

    What's under test there is TAEHV history bookkeeping inside
    ``WaypointDitSubmodule`` -- isolation, fixed addresses, cleanup -- not the
    denoiser, so this returns a deterministic latent instead of running 24
    attention layers on CPU (real DiT coverage is above and in
    ``test_waypoint_dit.py``).
    """

    def __init__(self, config: WaypointConfig, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.config = config
        self.marker = torch.nn.Parameter(torch.zeros(1, dtype=dtype))

    @property
    def dtype(self) -> torch.dtype:
        return self.marker.dtype

    def generate_frame(self, noise, pos, *, mouse, button, scroll):
        del pos, mouse, button, scroll
        return noise.reshape(1, 1, *self.config.latent_shape).to(self.dtype)

    def append_frame(self, latent, pos, *, mouse, button, scroll):
        del pos, mouse, button, scroll
        return latent.reshape(1, 1, *self.config.latent_shape).to(self.dtype)


@pytest.fixture
def decoder(taehv_weights, ae_config):
    """A ``WaypointDitSubmodule`` over a fake dit: the fused node under test,
    minus the denoiser. Named ``decoder`` because every test below exercises
    the TAEHV history half of this node."""
    return WaypointDitSubmodule(_FakeDit(ae_config), taehv_weights, ae_config)


def _seed_clip(ae_config, value: int = 200) -> torch.Tensor:
    return torch.full((ae_config.temporal_compression, 360, 640, 3), value, dtype=torch.uint8)


def _engine_inputs(
    request_id: str = "r0", graph_walk: str = PRIME_WALK,
) -> ModelInputsFromEngine:
    return ModelInputsFromEngine(
        request_ids=[request_id],
        per_request_info={request_id: _fwd_info(request_id, graph_walk=graph_walk)},
    )


def _decode(
    decoder, *, request_id="r0", graph_walk=ROLLOUT_WALK,
    seed_latent: torch.Tensor | None = None, num_steps: int = 8,
):
    """Drive the fused submodule through one real
    ``prepare_inputs``/``forward``/``postprocess`` cycle -- the same sequence
    the engine runs -- rather than poking ``taehv`` directly, since several
    tests below check that this cycle threads the nine histories correctly.
    """
    info = _fwd_info(request_id, graph_walk=graph_walk, num_steps=num_steps)
    if graph_walk == PRIME_WALK:
        inputs = {"latent": [seed_latent]}
    else:
        inputs = _controller_stream(decoder.config, frames=num_steps)
    prepared = decoder.prepare_inputs(graph_walk, info, inputs)
    assert prepared is not None, "must not be vetoed inside num_steps"
    outputs = decoder.forward(
        graph_walk,
        _engine_inputs(request_id, graph_walk),
        **prepared.tensor_inputs,
    )
    decoder.postprocess(request_id, info, outputs, prepared)
    return outputs


def test_the_encoder_scales_the_clip_once_in_the_ae_dtype(encoder, ae_config):
    """Cast then divide, which is the reference's order. 0-255 is exact in
    bf16, so that divide rounds once; dividing in fp32 and casting after rounds
    twice and lands on a different latent."""
    clip = _seed_clip(ae_config, value=200)
    prepared = encoder.prepare_inputs(PRIME_WALK, _fwd_info(), {"image_inputs": [clip]})

    image = prepared.tensor_inputs["image"]
    assert image.dtype == torch.bfloat16
    assert torch.equal(image, clip.to(torch.bfloat16).div(255))
    assert image.shape == (ae_config.temporal_compression, 360, 640, 3)


def test_the_encoder_emits_the_dit_s_priming_latent(encoder, ae_config):
    image = encoder.prepare_inputs(
        PRIME_WALK, _fwd_info(), {"image_inputs": [_seed_clip(ae_config)]}
    ).tensor_inputs["image"]

    out = encoder.forward(PRIME_WALK, _engine_inputs(), image)

    latent = out["latent"][0]
    # [B, frame, C, h, w] -- the frame axis the dit indexes the ring by, added
    # here rather than left for the dit to guess at.
    assert latent.shape == (1, 1, ae_config.channels, *ae_config.latent_shape[1:])
    assert latent.dtype == torch.bfloat16


def _batch(graph_walk: str) -> ExecutingBatch:
    return ExecutingBatch(
        node_name="Encoder",
        step_context=StepContext(
            request_ids=("r0",), graph_walk=graph_walk, slot=None,
            capture=False, plan_results={},
        ),
        per_request_info={},
    )


def _image_inputs(shape: tuple[int, int, int, int]) -> list[NodeInputs]:
    return [NodeInputs(tensor_inputs={"image": torch.zeros(shape)})]


def test_encoder_only_replays_the_captured_image_shape(encoder, ae_config):
    """The graph's static buffer is sized for exactly one H/W; any other
    16:9 image, or the wrong graph walk, must fall back to eager."""
    captured = (ae_config.temporal_compression, 360, 640, 3)
    bigger = (ae_config.temporal_compression, 720, 1280, 3)
    smaller = (ae_config.temporal_compression, 180, 320, 3)

    assert encoder.can_use_cuda_graphs(_batch(PRIME_WALK), _image_inputs(captured))
    assert not encoder.can_use_cuda_graphs(_batch(PRIME_WALK), _image_inputs(bigger))
    assert not encoder.can_use_cuda_graphs(_batch(PRIME_WALK), _image_inputs(smaller))
    assert not encoder.can_use_cuda_graphs(_batch(ROLLOUT_WALK), _image_inputs(captured))


def test_the_fused_decode_turns_one_frame_into_one_raw_clip(decoder, ae_config):
    out = _decode(decoder, graph_walk=PRIME_WALK, seed_latent=torch.zeros(
        (1, 1, ae_config.channels, *ae_config.latent_shape[1:]), dtype=torch.bfloat16
    ))

    frames = out["video_output"][0]
    # forward() is the pre-split batched call (row B=1 here); the engine's
    # forward_batched drops this leading dim per row before postprocess sees it.
    assert frames.shape == (1, ae_config.temporal_compression, 360, 640, 3)
    assert frames.dtype == torch.uint8
    decoder.cleanup_request("r0")


def test_forward_batched_hands_each_request_its_own_row(decoder, monkeypatch):
    """``forward`` returns ``{key: [batched_tensor]}``; ``forward_batched`` must
    index the tensor's rows, not the one-element list around it, and hand row
    ``i`` to ``request_ids[i]``. ``video_output`` drops the batch dim (what
    ``postprocess`` consumes); every other key keeps a leading 1 (what
    ``prepare_inputs`` builds for the next step)."""
    frames = torch.arange(2 * 4 * 2 * 2 * 3, dtype=torch.uint8).reshape(2, 4, 2, 2, 3)
    clock = torch.tensor([5, 9])
    history = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)
    monkeypatch.setattr(
        decoder, "forward",
        lambda *a, **k: {
            "video_output": [frames], "clock": [clock], "decoder_history_0": [history],
        },
    )
    engine_inputs = ModelInputsFromEngine(
        request_ids=["a", "b"],
        per_request_info={rid: _fwd_info(rid, graph_walk=ROLLOUT_WALK) for rid in "ab"},
    )

    out = decoder.forward_batched(ROLLOUT_WALK, engine_inputs=engine_inputs)

    assert set(out) == {"a", "b"}
    for i, rid in enumerate("ab"):
        assert torch.equal(out[rid]["video_output"][0], frames[i])
        assert torch.equal(out[rid]["clock"][0], clock[i : i + 1])
        assert torch.equal(out[rid]["decoder_history_0"][0], history[i : i + 1])
        assert all(isinstance(v[0], torch.Tensor) for v in out[rid].values())


def test_decode_latent_batches_rows_independently(taehv_weights, ae_config):
    """``decode_latent`` at B=2 equals two independent B=1 calls, row for row.

    MemBlock and TGrow (``_FakeTaehv``'s decoder) are convs/reshapes that
    never mix across the batch dim, so row 1 must not depend on row 0's
    latent -- the property the batched engine path relies on."""
    latent_shape = (1, ae_config.channels, *ae_config.latent_shape[1:])
    generator = torch.Generator().manual_seed(7)
    latents = [
        torch.randn(latent_shape, generator=generator).to(torch.bfloat16)
        for _ in range(2)
    ]
    histories = [initial_decoder_histories(taehv_weights, latent) for latent in latents]

    solo = [
        decode_latent(
            taehv_weights, latent, history, output_size=(360, 640), initialize=True,
        )
        for latent, history in zip(latents, histories, strict=True)
    ]

    batched_latent = torch.cat(latents, dim=0)
    batched_histories = tuple(
        torch.cat([histories[0][idx], histories[1][idx]], dim=0) for idx in range(9)
    )
    batched_frames, batched_state = decode_latent(
        taehv_weights, batched_latent, batched_histories, output_size=(360, 640), initialize=True,
    )

    assert batched_frames.shape == (2, ae_config.temporal_compression, 360, 640, 3)
    for row, (solo_frames, solo_state) in enumerate(solo):
        assert torch.equal(batched_frames[row], solo_frames[0])
        for idx in range(9):
            assert torch.equal(batched_state[idx][row], solo_state[idx][0])


def test_prime_and_rollout_use_the_same_fixed_tensor_histories(decoder, ae_config):
    latent = torch.full(
        (1, 1, ae_config.channels, *ae_config.latent_shape[1:]), 0.125, dtype=torch.bfloat16,
    )
    _decode(decoder, graph_walk=PRIME_WALK, seed_latent=latent, num_steps=1)
    state = decoder.request_state("r0")
    keys = [f"decoder_history_{idx}" for idx in range(9)]
    assert set(state.tensors) == set(keys)
    addresses = [state[key].data_ptr() for key in keys]

    _decode(decoder, graph_walk=ROLLOUT_WALK, num_steps=8)
    assert [state[key].data_ptr() for key in keys] == addresses
    assert all(state[key].dtype == torch.bfloat16 for key in keys)
    decoder.cleanup_request("r0")


def test_fused_decode_histories_are_isolated_interleaved_and_cleaned_up(decoder, ae_config):
    latent = torch.full(
        (1, 1, ae_config.channels, *ae_config.latent_shape[1:]), 0.125, dtype=torch.bfloat16,
    )
    first = _decode(decoder, request_id="a", graph_walk=PRIME_WALK, seed_latent=latent, num_steps=1)
    second = _decode(decoder, request_id="b", graph_walk=PRIME_WALK, seed_latent=latent, num_steps=1)
    assert torch.equal(first["video_output"][0], second["video_output"][0])
    before_b = {
        key: value.clone() for key, value in decoder.request_state("b").tensors.items()
    }
    # A generous num_steps: what's under test is address/isolation stability
    # across interleaved requests, not the overshoot veto (covered above).
    for _ in range(20):
        _decode(decoder, request_id="a", graph_walk=ROLLOUT_WALK, num_steps=100)
        _decode(decoder, request_id="b", graph_walk=ROLLOUT_WALK, num_steps=100)
    assert all(
        not torch.equal(before_b[key], value)
        for key, value in decoder.request_state("b").tensors.items()
    )
    assert all(
        decoder.request_state("a")[key].data_ptr()
        != decoder.request_state("b")[key].data_ptr()
        for key in before_b
    )

    decoder.cleanup_request("a")
    assert "a" not in decoder.request_states
    restarted = _decode(decoder, request_id="a", graph_walk=PRIME_WALK, seed_latent=latent, num_steps=1)
    assert torch.equal(restarted["video_output"][0], first["video_output"][0])
    decoder.cleanup_request("a")
    decoder.cleanup_request("b")


def test_the_encoder_and_the_fused_decode_share_only_weights(
    encoder, decoder, taehv_weights, ae_config
):
    """Encoder prime is stateless; only the fused decode's histories survive a
    request."""
    image = encoder.prepare_inputs(
        PRIME_WALK, _fwd_info(), {"image_inputs": [_seed_clip(ae_config)]}
    ).tensor_inputs["image"]
    latent = encoder.forward(PRIME_WALK, _engine_inputs(), image)["latent"][0]
    _decode(decoder, graph_walk=PRIME_WALK, seed_latent=latent, num_steps=1)
    assert encoder.taehv is decoder.taehv is taehv_weights
    assert encoder.request_states == {}
    assert len(decoder.request_state("r0").tensors) == 9
    decoder.cleanup_request("r0")


def test_ae_graphs_are_compiled_for_capture_but_remain_optional(encoder, decoder):
    encoder_configs = encoder.get_cuda_graph_configs(torch.device("cpu"))
    fused_configs = decoder.get_cuda_graph_configs(torch.device("cpu"))
    assert [cfg.capture_graph_walk for cfg in encoder_configs] == [PRIME_WALK]
    assert {cfg.capture_graph_walk for cfg in fused_configs} == {
        PRIME_WALK, ROLLOUT_WALK,
    }
    assert encoder_configs
    assert all(cfg.compile for cfg in encoder_configs)
    assert encoder.disable_torch_compile is True
    assert encoder.disable_autocast is True

    # The fused dit+decode node compiles its own two reference-shaped regions
    # (WaypointDiT.compile_regions, gated by config.compile_dit); letting the
    # engine also compile this wrapper would fuse across that boundary, so its
    # captures stay uncompiled regardless.
    assert fused_configs
    assert all(cfg.compile is False for cfg in fused_configs)
    assert decoder.disable_torch_compile is True
    assert decoder.disable_autocast is True


def test_ae_nodes_declare_no_capture_when_cuda_graph_is_disabled(
    taehv_weights, ae_config,
):
    eager_config = dataclasses.replace(ae_config, cuda_graph=False)
    encoder = WaypointVaeEncoderSubmodule(taehv_weights, eager_config)
    fused = WaypointDitSubmodule(_FakeDit(eager_config), taehv_weights, eager_config)

    assert encoder.get_cuda_graph_configs(torch.device("cpu")) == []
    assert fused.get_cuda_graph_configs(torch.device("cpu")) == []


def test_the_shell_builds_without_the_taehv_package(monkeypatch):
    """``taehv`` is a separate install with its own checkpoint. Every import of
    it is deferred to the call that needs weights, so the graph, the resources
    and the serialization gate all work on a box that has neither."""
    monkeypatch.setitem(sys.modules, "taehv", None)
    with pytest.raises(ImportError):
        importlib.import_module("taehv")

    unweighted = WaypointModel(skip_weight_loading=True)
    assert set(unweighted.get_graph_walk_graphs()) == {PRIME_WALK, ROLLOUT_WALK}
    assert unweighted.nodes == [DIT_NODE, VAE_ENCODER_NODE]
    assert unweighted.get_node_resources()
    for node in (VAE_ENCODER_NODE, DIT_NODE):
        assert unweighted.get_submodule(node) is None


# ---------------------------------------------------------------------------
# Request shaping
# ---------------------------------------------------------------------------


def test_process_prompt_materializes_the_whole_action_stream(model, config):
    """The rollout is scripted, so the stream is built once at
    request time and sliced per frame. Validated here rather than in
    ``prepare_inputs`` because this runs at the API boundary, where a
    ValueError becomes a 400 instead of killing a rollout mid-flight."""
    actions = [
        {"mouse": (1.0, -2.0), "buttons": [3, 5], "scroll": 0.5},
        {"buttons": [3]},
    ]
    seed = torch.zeros((720, 1280, 3), dtype=torch.uint8)
    out = model.process_prompt(
        None, ["image"], ["video_frame"],
        tensors={"image_inputs": [seed]}, num_steps=2, actions=actions,
    )
    assert set(out) == {"image_inputs", "mouse", "button", "scroll"}
    mouse, button, scroll = out["mouse"][0], out["button"][0], out["scroll"][0]
    assert mouse.shape == (1, 2, 2)
    assert button.shape == (1, 2, config.n_buttons)
    assert scroll.shape == (1, 2, 1)
    assert mouse[0, 0].tolist() == [1.0, -2.0]
    assert button[0, 0].nonzero().flatten().tolist() == [3, 5]
    assert button[0, 1].nonzero().flatten().tolist() == [3]
    with pytest.raises(ValueError, match="out of range"):
        model.process_prompt(
            None, ["image"], ["video_frame"],
            tensors={"image_inputs": [seed]}, num_steps=1,
            actions=[{"buttons": [config.n_buttons]}],
        )
    with pytest.raises(ValueError, match="exactly one action"):
        model.process_prompt(
            None, ["image"], ["video_frame"],
            tensors={"image_inputs": [seed]}, num_steps=1, actions=actions,
        )


@pytest.mark.parametrize("num_steps", [None, 0, -1, True, 1.5, "1"])
def test_process_prompt_rejects_non_positive_integer_steps(model, num_steps):
    seed = torch.zeros((720, 1280, 3), dtype=torch.uint8)
    with pytest.raises(ValueError, match="num_steps > 0"):
        model.process_prompt(
            None,
            ["image"],
            ["video_frame"],
            tensors={"image_inputs": [seed]},
            num_steps=num_steps,
            actions=[],
        )


def test_process_prompt_rejects_steps_past_the_checkpoint_horizon(model, config):
    seed = torch.zeros((720, 1280, 3), dtype=torch.uint8)
    with pytest.raises(ValueError, match="exceeds the checkpoint horizon"):
        model.process_prompt(
            None,
            ["image"],
            ["video_frame"],
            tensors={"image_inputs": [seed]},
            num_steps=config.max_frames + 1,
            actions=[],
        )


@pytest.mark.parametrize(
    ("action", "message"),
    [
        (None, "must be an object"),
        ({"unknown": 1}, "unknown field"),
        ({"mouse": [float("nan"), 0]}, "mouse values must be finite"),
        ({"mouse": [True, 0]}, "mouse values must be numbers"),
        ({"mouse": ["1", 0]}, "mouse values must be numbers"),
        ({"mouse": [1e39, 0]}, "mouse values must be finite"),
        ({"buttons": [1, 1]}, "repeats button id"),
        ({"buttons": [True]}, "button ids must be integers"),
        ({"scroll": float("inf")}, "scroll must be finite"),
        ({"scroll": True}, "scroll must be a number"),
        ({"scroll": "1"}, "scroll must be a number"),
        ({"scroll": 1e39}, "scroll must be finite"),
    ],
)
def test_process_prompt_rejects_invalid_action_values(model, action, message):
    seed = torch.zeros((720, 1280, 3), dtype=torch.uint8)
    with pytest.raises(ValueError, match=message):
        model.process_prompt(
            None,
            ["image"],
            ["video_frame"],
            tensors={"image_inputs": [seed]},
            num_steps=1,
            actions=[action],
        )


def test_the_seed_clip_is_one_latent_frame_of_uint8_rgb(model, config):
    """The streaming encoder emits one latent per ``temporal_compression``
    frames: a short clip would buffer and return nothing, a long one would
    encode twice and leave the second latent unclaimed. Checked at the API
    boundary, so a malformed request is a 400 rather than a rollout that dies
    on a worker."""
    n = config.temporal_compression
    frame = torch.zeros((720, 1280, 3), dtype=torch.uint8)

    def prompt(image):
        return model.process_prompt(
            None,
            ["image"],
            ["video_frame"],
            tensors={"image_inputs": [image]},
            num_steps=2,
            actions=[{}, {}],
        )

    # A still seeds the world by being repeated, which is gen_sample.py's
    # seed_frame_x4.
    clip = prompt(frame)["image_inputs"][0]
    assert clip.shape == (n, 720, 1280, 3) and clip.dtype == torch.uint8
    # A real clip of exactly one latent frame passes through.
    assert prompt(torch.zeros((n, 720, 1280, 3), dtype=torch.uint8))["image_inputs"][
        0
    ].shape == (n, 720, 1280, 3)

    with pytest.raises(ValueError, match="one latent frame"):
        prompt(torch.zeros((n + 1, 720, 1280, 3), dtype=torch.uint8))
    with pytest.raises(ValueError, match="uint8"):
        prompt(torch.zeros((720, 1280, 3), dtype=torch.float32))
    with pytest.raises(ValueError, match="16:9"):
        prompt(torch.zeros((720, 720, 3), dtype=torch.uint8))

    with pytest.raises(ValueError, match="requires one RGB seed"):
        model.process_prompt(
            None, ["tensor"], ["video_frame"], tensors=None,
            num_steps=2, actions=[{}, {}],
        )


def test_the_required_prime_walk_addresses_the_seed_to_the_encoder(model):
    """The seed clip is addressed to the vae_encoder. The
    controller streams stay addressed to the dit on both walks: they are read
    once per frame for the whole rollout, and the encoder never sees them."""
    signals = {name: [object()] for name in ("mouse", "button", "scroll")}

    seeded = model.get_initial_forward_pass_args(
        "p", ["image"], ["video_frame"],
        {**signals, "image_inputs": [object()]}, {"num_steps": 2},
    )

    assert seeded.full_metadata.kwargs["walk_schedule"] == [PRIME_WALK, ROLLOUT_WALK]
    assert seeded.full_metadata.graph_walk == PRIME_WALK
    assert seeded.full_metadata.is_prefill is True
    assert [(e.name, e.next_node) for e in seeded.inputs] == [
        ("image_inputs", VAE_ENCODER_NODE),
        ("mouse", DIT_NODE), ("button", DIT_NODE), ("scroll", DIT_NODE),
    ]

    with pytest.raises(ValueError, match="required seed clip"):
        model.get_initial_forward_pass_args(
            "p", ["tensor"], ["video_frame"], signals, {"num_steps": 2},
        )
    # Nothing is unpersisted: the streams are re-read every frame, and the seed
    # clip goes with the request.
    assert seeded.unpersist_tensors == []


def test_postprocess_emits_the_step_s_frames_as_raw_rgb_bytes(model, config):
    """No container. The emit is per engine step so a client can act on the
    world while it runs, and a per-step mp4 would be a fragment nothing plays."""
    frames = torch.arange(
        config.temporal_compression * 2 * 4 * 3, dtype=torch.uint8
    ).reshape(config.temporal_compression, 2, 4, 3)

    payload = model.postprocess(frames, "video_frame")

    assert payload == frames.numpy().tobytes()
    assert len(payload) == frames.numel()

    with pytest.raises(ValueError, match="uint8"):
        model.postprocess(frames.float(), "video_frame")
    with pytest.raises(ValueError, match="modality"):
        model.postprocess(frames, "image")
