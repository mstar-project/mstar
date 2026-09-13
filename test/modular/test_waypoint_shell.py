"""Contract tests for the Waypoint serving shell: the model, its node
submodule, and the resources it declares.

Nothing here runs the DiT. The 4+1 driver, the ring numerics and the weight
remap have their own suites; what is under test here is the *shell* — the
handful of declarations and host-side hooks that sit between the engine and a
model that already works, every one of which fails silently when it is wrong:

  * a ring geometry summarized instead of copied (a global layer served with a
    local layer's stride still produces smooth video, of the wrong world),
  * a rank-0 ``frame_pos`` (``_intern_static_buffer`` reads
    ``stored.shape[0]``),
  * a per-step tensor whose shape happens to carry ``tokens_per_frame``
    (``_seq_dim`` hoists the matching dim to the front of a shared static
    buffer),
  * noise drawn from an advancing generator instead of ``(seed, frame_pos)``
    (a resumed rollout diverges from the one it resumed),
  * an off-by-one stop that runs one frame past the request and commits it,
  * a deployment whose ``max_concurrent_requests`` is unset or larger than the
    ring's world pool, which is the *only* thing keeping arrivals inside a pool
    that fails terminally when it is overrun,
  * a resource that never reaches the 24 attention layers,
  * a latent that reaches the streaming decoder twice, out of order, or not at
    all -- the prime walk skipping its decode is the version of this the port
    nearly shipped.

CPU-only, checkpoint-free, and no engine. The real 720P config is used
throughout, because the numbers that collide are that config's numbers; the DiT
behind the submodule is built on ``torch.device("meta")`` and never
materialized, since a real 720P bf16 build is ~2.6 GB and none of these
assertions touch a weight. The two places that need a value read back go around
it: the noise draw takes an explicit device (which is why that helper takes
one), and the frame-clock test runs over ``_HostOnlyDit``, since what it
asserts is host bookkeeping the DiT is not part of. The VAE section is the
third: it runs on the 360P config and a fake ``taehv`` package, for the reasons
given there.
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
from mstar.engine.resources import (
    AttentionSpec,
    AttnBackend,
    KVSpec,
    RingKVConfig,
    RingKVStep,
)
from mstar.engine.resources.runner import topo_sort
from mstar.graph.base import GraphEdge, Loop, Sequential
from mstar.graph.graph_io import WorkerGraphIO
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.submodule_base import ModelInputsFromEngine
from mstar.model.waypoint.components.attention import WaypointAttention
from mstar.model.waypoint.components.dit import WaypointDiT
from mstar.model.waypoint.config import waypoint_1_5_1b_360p, waypoint_1_5_1b_720p
from mstar.model.waypoint.submodules import (
    ATTN_RESOURCE,
    KV_RESOURCE,
    PRIME_WALK,
    ROLLOUT_LOOP_NAME,
    ROLLOUT_WALK,
    WaypointDitSubmodule,
    WaypointVaeDecoderSubmodule,
    WaypointVaeEncoderSubmodule,
)
from mstar.model.waypoint.waypoint_model import (
    DIT_NODE,
    VAE_DECODER_NODE,
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
    ``self.dit.dtype`` surviving ``cast_serving_dtypes`` are exactly two of the
    things under test, and a stub would assert them against itself. Meta also
    keeps ``prepare_inputs``' shapes and dtypes honest -- they are computed the
    same way on meta as on cuda -- while allocating nothing.
    """
    with torch.device("meta"):
        dit = WaypointDiT(config)
    dit.cast_serving_dtypes()
    return WaypointDitSubmodule(dit, config)


class _HostOnlyDit(torch.nn.Module):
    """Stands in for the DiT in the tests that need to read a value back.

    ``prepare_inputs`` touches the DiT for exactly one thing -- ``.dtype`` --
    and ``get_device`` for one parameter, so what those tests exercise is the
    submodule's host-side bookkeeping and nothing else. The meta build above
    cannot be read back (``.item()`` raises on a meta tensor) and materializing
    720P to assert on a frame counter is 2.6 GB; a stub is the honest third
    option, and it is confined to the two tests that say so.
    """

    def __init__(self, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1, dtype=dtype))

    @property
    def dtype(self) -> torch.dtype:
        return self.marker.dtype


@pytest.fixture
def host_submodule(config):
    return WaypointDitSubmodule(_HostOnlyDit(), config)


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
    # accumulation over the KV blocks, and bit-exactness against the reference
    # is the only correctness signal this model has.
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
    # One world declared here, because sizing is a deployment question and
    # `apply_yaml_overrides` runs after this hook. What is pinned is the
    # *default*: a node that never says otherwise serves one session.
    assert ring.num_worlds == 1

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
    """The seed frame advances decoder state. Encoding it and dropping the latent would
    prime the world correctly and still corrupt every emitted frame: the
    streaming decoder spends its first call on ``frames_to_trim`` of temporal
    memory, so the first *rollout* frame would pay for it and the whole stream
    would sit one priming short of the world it came from. Silently."""
    walks = model.get_graph_walk_graphs()
    assert set(walks) == {PRIME_WALK, ROLLOUT_WALK}
    assert model.nodes == [DIT_NODE, VAE_DECODER_NODE, VAE_ENCODER_NODE]

    prime = walks[PRIME_WALK]
    assert isinstance(prime, Sequential)
    assert [s.name for s in prime.sections] == [
        VAE_ENCODER_NODE, DIT_NODE, VAE_DECODER_NODE,
    ]
    encoder, dit, decoder = prime.sections
    assert encoder.input_names == {"image_inputs"}
    assert [(e.name, e.next_node) for e in encoder.outputs] == [("latent", DIT_NODE)]
    # The dit node's contract is unchanged by the nodes bracketing it.
    assert dit.input_names == {"latent", "mouse", "button", "scroll"}
    assert [(e.name, e.next_node) for e in dit.outputs] == [("latent", VAE_DECODER_NODE)]
    assert decoder.input_names == {"latent"}
    assert decoder.outputs == []

    # The two `latent` edges are distinct because a section keys on
    # (name, next_node); collapsing them would route the seed latent past the dit.
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

    section = rollout.section
    assert isinstance(section, Sequential)
    dit, decoder = section.sections
    assert (dit.name, decoder.name) == (DIT_NODE, VAE_DECODER_NODE)
    assert [(e.name, e.next_node) for e in dit.outputs] == [("latent", VAE_DECODER_NODE)]
    # Emitted from inside the loop, one frame per iteration -- an interactive
    # world model whose frames only arrive after the rollout ends has no world
    # to interact with.
    assert [e.next_node for e in decoder.outputs] == [EMIT_TO_CLIENT]
    assert rollout.accumulated_outputs == []
    # An overshoot iteration is not a wasted forward: the dit commits a frame
    # into the ring, and a speculative decode is a reorder of a stream that
    # cannot be reordered.
    assert dit.enable_async_scheduling is False
    assert decoder.enable_async_scheduling is False
    # The controller streams stay loop-external; the dit->decoder latent does
    # not become one, or the conductor would re-inject a stale frame.
    assert rollout._external_inputs == {
        ("mouse", DIT_NODE), ("button", DIT_NODE), ("scroll", DIT_NODE),
    }


def test_every_committed_frame_is_decoded_once_including_the_last(model):
    """Driven through ``WorkerGraphIO``, because what is under test is the order
    the worker runs these in, not the order they are declared in.

    The decoder is order-dependent and its memory advances per call, so a latent
    decoded twice, skipped, or taken out of turn shifts every frame after it
    with nothing raised. Two things have to hold. The decode runs between its
    own dit pass and the next one -- which it does because scheduling *pops* a
    node off the ready set, and the dit's controller streams are only
    re-injected at the iteration boundary. And the stop signal, which fires
    during the dit's postprocess, closes the loop only after that iteration's
    decode: ``LoopStateRegistry`` calls ``complete_iter`` once every entity is
    finished, so the finish cannot short-circuit the decoder.
    """
    rollout = model.get_graph_walk_graphs()[ROLLOUT_WALK]
    wgio = WorkerGraphIO(rollout)
    decoder = wgio.get_node(VAE_DECODER_NODE)
    for name in ("mouse", "button", "scroll"):
        wgio.ingest_input(GraphEdge(next_node=DIT_NODE, name=name, persist=True))

    emitted, decoded = [], []

    def run(node_name: str, latent_id: int | None = None) -> None:
        """One scheduling round: pop, execute, route. The pop is what
        ``NodeManager.pop_ready_nodes`` does, and it is the whole reason the dit
        cannot be picked twice in an iteration."""
        assert node_name in wgio.ready_node_names
        wgio.ready_node_names.discard(node_name)
        queued = decoder.ready_signals.ready_inputs.get("latent")
        if queued is not None:
            decoded.append(queued.latent_id)
        for edge in wgio.mark_node_complete(node_name).output_edges:
            if edge.next_node == EMIT_TO_CLIENT:
                emitted.append(edge.name)
                continue
            if edge.next_node == VAE_DECODER_NODE:
                # A fresh edge per iteration: the node's declared outputs are
                # one reused object, so identity is the only way to tell which
                # frame's latent the decoder actually consumed.
                edge = GraphEdge(next_node=edge.next_node, name=edge.name)
                edge.latent_id = latent_id
            wgio.ingest_input(edge)

    for frame in range(3):
        assert wgio.ready_node_names == {DIT_NODE}
        run(DIT_NODE, latent_id=frame)
        assert wgio.ready_node_names == {VAE_DECODER_NODE}
        run(VAE_DECODER_NODE)
        assert rollout.is_done is False
        assert rollout.curr_iter == frame + 1

    wgio.register_loop_finish_signal(ROLLOUT_LOOP_NAME)  # what check_stop does
    run(DIT_NODE, latent_id=3)
    assert rollout.is_done is False, "the loop closed before the frame was decoded"
    assert wgio.ready_node_names == {VAE_DECODER_NODE}
    run(VAE_DECODER_NODE)

    assert rollout.is_done is True
    assert decoded == [0, 1, 2, 3], "a latent was skipped, repeated or reordered"
    assert emitted == ["video_output"] * 4, "one emit per committed frame"


# ---------------------------------------------------------------------------
# Per-step inputs: rank and the _seq_dim collision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("walk", [PRIME_WALK, ROLLOUT_WALK])
def test_no_prepared_tensor_is_rank_zero(submodule, config, walk):
    """A 0-dim tensor reaches ``_intern_static_buffer``, which
    reads ``stored.shape[0]``, and the worker's output fanout reads
    ``dims[0]`` -- both IndexError at capture, i.e. at warmup, far from the
    line that made the tensor."""
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
    """``CudaGraphRunner._seq_dim`` finds the dim equal to
    ``input_seq_len`` and hoists it to the front of a shared static buffer; a
    tensor that carries that number for an unrelated reason gets silently
    transposed under replay.

    ``input_seq_len`` is honestly ``tokens_per_frame`` (512) -- the scheduler is
    told the real token count -- so the burden falls here. At 720P nothing
    collides, but the margin is thin: a 256-token-per-frame variant would put
    ``button``'s ``n_buttons = 256`` straight into the crosshairs.
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
    """``_capture_one`` bakes the config's template and replay re-stages only
    what ``preprocess`` returned into it. A key, shape or dtype that differs
    between the two is either a stale-address read or a silent eager fallback,
    depending on which way it differs -- so they are asserted against each
    other rather than against a literal."""
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


# ---------------------------------------------------------------------------
# Noise: stateless in (seed, frame_pos)
# ---------------------------------------------------------------------------


def test_noise_is_a_pure_function_of_seed_and_frame_pos(submodule):
    """Nothing about the draw may depend on how many
    frames have already been drawn: a generator advanced in place accumulates
    state that ``get_state`` does not serialize, so a resumed rollout would
    diverge from the one it resumed and no assertion anywhere would fire."""
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

    # Adjacent seeds must not share frame k. `seed + frame_pos` would make seeds
    # 0 and 1 agree on every frame but the first, which reads as a broken
    # sampler rather than as a seed collision -- hence the splitmix finalizer.
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
        rid, _fwd_info(request_id=rid, graph_walk=PRIME_WALK), {}
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
        host_submodule.postprocess(rid, _fwd_info(request_id=rid), {})

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
    """One source for the clock, not two. `RingKVManager.admit` checks the
    declared frame against the one its `commit` last recorded, so a step that
    declared a *different* number from the one the forward runs at would refuse
    valid frames and pass desynced ones -- the check inverted.

    Read on the host, off `PerRequestState`: the `frame_pos` in `NodeInputs` is
    a `[1]` device tensor by then and reading it back would be a sync per step.
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

        host_submodule.postprocess(rid, _fwd_info(request_id=rid), {})

    host_submodule.cleanup_request(rid)


def test_declare_step_names_a_clock_for_every_request_in_the_batch(host_submodule):
    """No batch shape declines to answer.

    The singular ``frame_pos: int | None`` this replaces returned ``None``
    whenever the batch was not one request, and ``RingKVManager``'s continuity
    check — the only thing standing between a stalled clock and a world quietly
    rewriting its own history — then did nothing for that step. The reasoning
    was that ``admit`` refuses such a batch anyway, which was true and is still
    true; the problem is that it made the check's coverage depend on a second,
    unrelated refusal staying in place. It does not any more: a batch this
    submodule cannot serve is refused for *being a batch*, with every clock in
    it still named.

    The clocks below are genuinely different, so a declaration that broadcast
    one request's frame across the batch fails here rather than passing on a
    coincidence.
    """
    for i, rid in enumerate(("a", "b", "c")):
        host_submodule.request_states.pop(rid, None)
        for _ in range(i * 2):
            host_submodule.postprocess(rid, _fwd_info(request_id=rid), {})

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
            {"node_names": [VAE_ENCODER_NODE, DIT_NODE, VAE_DECODER_NODE], "ranks": [0]}
        ],
        **extra,
    }
    path = tmp_path / name
    path.write_text(yaml.safe_dump(body))
    return str(path)


def _worlds(n: int) -> dict:
    """The ``resources:`` block a deployment writes to size the ring — the same
    one ``EngineManager.build`` feeds to ``apply_yaml_overrides``, which is why
    the gate reads it here rather than inventing its own key."""
    return {"resources": {KV_RESOURCE: {"num_worlds": n}}}


@pytest.mark.parametrize("limit", [None, 0, -1, True, 1.0, "2"])
def test_get_worker_graphs_refuses_a_deployment_with_no_admit_queue(
    model, tmp_path, limit
):
    """The pool is finite, and this is the primary gate on it.

    The conductor only forms a FIFO admit queue when ``max_concurrent_requests``
    is set: it drains ``waiting_queue`` while ``len(self.requests) <
    max_concurrent_requests``, so an unset value admits every request on arrival
    and everything past the Nth dies terminally at ``RingKVManager.admit`` —
    which sees the batch far too late to queue it.

    ``max_batch_size = 1`` does not cover this and never did. It caps how many
    requests share one *step*; N admitted rollouts alternating steps is now the
    intended shape, but it says nothing about how many may exist at once, which
    is the thing the world pool bounds.

    ``True`` is in the list because ``isinstance(True, int)`` is ``True`` in
    Python: a YAML ``max_concurrent_requests: true`` would otherwise read as the
    number 1 and silently serialize a node sized for eight.
    """
    extra = {} if limit is None else {"max_concurrent_requests": limit}
    path = _write_config(tmp_path, f"reject_{limit}.yaml", **extra)
    with pytest.raises(ValueError, match="max_concurrent_requests"):
        model.get_worker_graphs(path)


@pytest.mark.parametrize("worlds", [0, -1, True, 1.0, 1.9, "2"])
def test_get_worker_graphs_refuses_invalid_world_pool_size(
    model, tmp_path, worlds
):
    path = _write_config(
        tmp_path,
        f"invalid_worlds_{worlds}.yaml",
        max_concurrent_requests=1,
        **_worlds(worlds),
    )
    with pytest.raises(ValueError, match=r"resources\.kv\.num_worlds"):
        model.get_worker_graphs(path)


@pytest.mark.parametrize(("limit", "worlds"), [(2, 1), (8, 4), (2, None)])
def test_get_worker_graphs_refuses_more_arrivals_than_worlds(
    model, tmp_path, limit, worlds
):
    """A queue longer than the pool is not a queue, it is a delayed failure: the
    conductor admits ``limit`` requests, the ring hands out ``num_worlds``, and
    the difference is a set of requests that reach ``admit`` and die there with
    an ``AdmitRuntimeError`` that no retry, eviction or reload can clear.

    The ``worlds=None`` case is the one a deployment writes by accident: raising
    ``max_concurrent_requests`` without touching ``resources`` at all, which is
    the shape every pre-pool config already has.
    """
    extra = {"max_concurrent_requests": limit}
    if worlds is not None:
        extra |= _worlds(worlds)
    path = _write_config(tmp_path, f"over_{limit}_{worlds}.yaml", **extra)

    with pytest.raises(ValueError, match="exceeds the"):
        model.get_worker_graphs(path)


@pytest.mark.parametrize(("limit", "worlds"), [(1, None), (1, 1), (4, 4), (8, 8)])
def test_get_worker_graphs_accepts_a_deployment_inside_its_pool(
    model, tmp_path, limit, worlds
):
    """``limit == num_worlds`` is the shape that should be written, at any size.
    The ``(1, None)`` case is the default deployment, which must keep working
    unchanged — the pool is a widening, not a migration."""
    extra = {"max_concurrent_requests": limit}
    if worlds is not None:
        extra |= _worlds(worlds)
    path = _write_config(tmp_path, f"ok_{limit}_{worlds}.yaml", **extra)

    graphs = model.get_worker_graphs(path)

    assert graphs
    assert {walk for g in graphs for walk in g.graph_walks} == {
        PRIME_WALK,
        ROLLOUT_WALK,
    }


def test_the_shipped_config_serializes_all_three_nodes_onto_one_rank(model):
    """``configs/waypoint.yaml`` is the deployment and has to pass its own gate.

    All three nodes in one group, on rank 0: a node missing from
    ``node_groups`` has no rank to run on and the split fails there, and a
    worker boundary inside the rollout loop would put a process hop between the
    dit and a decoder whose frames must arrive in order.
    """
    path = pathlib.Path(__file__).resolve().parents[2] / "configs" / "waypoint.yaml"

    graphs = model.get_worker_graphs(str(path))

    assert {walk for g in graphs for walk in g.graph_walks} == {PRIME_WALK, ROLLOUT_WALK}
    assert {tuple(g.ranks) for g in graphs} == {(0,)}
    by_walk = {walk: g for g in graphs for walk in g.graph_walks}
    assert set(by_walk[PRIME_WALK].section.get_nodes()) == {
        VAE_ENCODER_NODE, DIT_NODE, VAE_DECODER_NODE,
    }
    assert set(by_walk[ROLLOUT_WALK].section.get_nodes()) == {DIT_NODE, VAE_DECODER_NODE}


def test_get_worker_graphs_warns_about_worlds_no_request_can_reach(
    model, tmp_path, caplog
):
    """Legal, and only wasteful — so a warning and not a refusal. Each
    unreachable world is ~816 MiB of ring at 720P that is allocated, zeroed,
    and never written, which is worth a line in the log rather than a failed
    boot: the deployment still serves correctly."""
    path = _write_config(
        tmp_path, "underused.yaml", max_concurrent_requests=2, **_worlds(8)
    )

    with caplog.at_level(logging.WARNING):
        assert model.get_worker_graphs(path)

    assert any("can never be filled" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Resource binding
# ---------------------------------------------------------------------------


def test_bind_reaches_the_dit_and_all_twenty_four_attention_layers(submodule):
    """One bind on the submodule has to reach every caller: the 24 attention
    layers hold their own references and call ``upsert``/``attend`` directly.
    Anything left unbound raises ``NoneType has no attribute ...`` mid-forward
    -- or, if warmup gets there first, inside a capture, where it poisons the
    graph instead of failing a request."""
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

    ``NodeSubmodule.bind_node_resources`` walks ``self.modules()`` but skips
    ``self`` (``submodule_base.py``: ``if bind is not None and module is not
    self``). A submodule that *is* the DiT -- by subclassing it, or by defining
    ``bind_resources`` on itself -- would therefore never be visited, and
    anything the root came to need would sit unbound.

    Nothing on the DiT root needs a resource today (``commit`` is threaded down
    as an argument), so this is a structural guard rather than a live bug: it
    keeps the walk able to reach the root if that ever changes.
    """
    assert not isinstance(submodule, WaypointDiT)
    assert isinstance(submodule.dit, WaypointDiT) and submodule.dit is not submodule
    # The DiT has to be a *child module*, not a plain attribute -- self.modules()
    # is the only thing the walk follows.
    assert any(m is submodule.dit for m in submodule.modules())
    # And the submodule must not answer bind_resources itself: the walk would
    # skip it, so defining one is a method that never runs.
    assert getattr(type(submodule), "bind_resources", None) is None


def test_binding_without_a_declared_resource_fails_at_bind(submodule):
    """Not mid-forward. The layers resolve with ``.get`` -- correct for a layer,
    which may sit on a node owning only some resources -- so the node is the
    frame that still knows which keys it declared and can name the missing one.
    """
    for partial in ({ATTN_RESOURCE: object()}, {KV_RESOURCE: object()}, {}):
        with pytest.raises(KeyError):
            submodule.bind_node_resources(partial)


# ---------------------------------------------------------------------------
# Capture configs
# ---------------------------------------------------------------------------


def test_only_the_steady_dit_rollout_is_an_optional_capture(submodule):
    """The one-time prime/cache pass is compiled internally but uncaptured."""
    configs = submodule.get_cuda_graph_configs(torch.device("meta"))
    assert len(configs) == 1
    assert configs[0].capture_graph_walk == ROLLOUT_WALK
    for cfg in configs:
        assert cfg.compile is False
        # One live world again: the bucket cannot be wider than it.
        assert cfg.capture_batch_sizes == [1]
        # The v1 engine always dispatches batched; a submodule captured on bare
        # `forward` is captured on a method that never runs.
        assert cfg.capture_forward_method == "forward_batched"

    assert "noise" in configs[0].single_request_inputs.tensor_inputs
    assert "latent" not in configs[0].single_request_inputs.tensor_inputs
    assert submodule.disable_torch_compile is True


def test_dit_declares_no_capture_when_cuda_graph_is_disabled(config):
    eager_config = dataclasses.replace(config, cuda_graph=False)
    with torch.device("meta"):
        dit = WaypointDiT(eager_config)
    dit.cast_serving_dtypes()
    eager = WaypointDitSubmodule(dit, eager_config)

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


@pytest.fixture
def decoder(taehv_weights, ae_config):
    return WaypointVaeDecoderSubmodule(taehv_weights, ae_config)


def _seed_clip(ae_config, value: int = 200) -> torch.Tensor:
    return torch.full((ae_config.temporal_compression, 360, 640, 3), value, dtype=torch.uint8)


def _engine_inputs(
    request_id: str = "r0", graph_walk: str = PRIME_WALK,
) -> ModelInputsFromEngine:
    return ModelInputsFromEngine(
        request_ids=[request_id],
        per_request_info={request_id: _fwd_info(request_id, graph_walk=graph_walk)},
    )


def _decode(decoder, latent, *, request_id="r0", graph_walk=ROLLOUT_WALK):
    info = _fwd_info(request_id, graph_walk=graph_walk)
    prepared = decoder.prepare_inputs(
        graph_walk, info, {"latent": [latent]}
    )
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


def test_the_decoder_turns_one_latent_into_one_raw_clip(decoder, ae_config):
    latent = torch.zeros(
        (1, 1, ae_config.channels, *ae_config.latent_shape[1:]), dtype=torch.bfloat16
    )

    out = _decode(decoder, latent)

    frames = out["video_output"][0]
    assert frames.shape == (ae_config.temporal_compression, 360, 640, 3)
    assert frames.dtype == torch.uint8


def test_decoder_prime_and_steady_state_use_fixed_tensor_histories(decoder, ae_config):
    latent = torch.full(
        (1, 1, ae_config.channels, *ae_config.latent_shape[1:]), dtype=torch.bfloat16
        , fill_value=0.125
    )
    _decode(decoder, latent, graph_walk=PRIME_WALK)
    state = decoder.request_state("r0")
    keys = [f"decoder_history_{idx}" for idx in range(9)]
    assert set(state.tensors) == set(keys)
    addresses = [state[key].data_ptr() for key in keys]

    _decode(decoder, latent * 2, graph_walk=ROLLOUT_WALK)
    assert [state[key].data_ptr() for key in keys] == addresses
    assert all(state[key].dtype == torch.bfloat16 for key in keys)


def test_decoder_histories_are_isolated_interleaved_and_cleaned_up(decoder, ae_config):
    latent = torch.full(
        (1, 1, ae_config.channels, *ae_config.latent_shape[1:]), dtype=torch.bfloat16
        , fill_value=0.125
    )
    first = _decode(decoder, latent, request_id="a", graph_walk=PRIME_WALK)
    second = _decode(decoder, latent, request_id="b", graph_walk=PRIME_WALK)
    assert torch.equal(first["video_output"][0], second["video_output"][0])
    before_b = {
        key: value.clone() for key, value in decoder.request_state("b").tensors.items()
    }
    for _ in range(20):
        _decode(decoder, latent * 2, request_id="a")
        _decode(decoder, latent * 3, request_id="b")
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
    restarted = _decode(decoder, latent, request_id="a", graph_walk=PRIME_WALK)
    assert torch.equal(restarted["video_output"][0], first["video_output"][0])


def test_the_encoder_and_decoder_share_only_weights(
    encoder, decoder, taehv_weights, ae_config
):
    """Encoder prime is stateless; only decoder histories survive a request."""
    image = encoder.prepare_inputs(
        PRIME_WALK, _fwd_info(), {"image_inputs": [_seed_clip(ae_config)]}
    ).tensor_inputs["image"]
    latent = encoder.forward(PRIME_WALK, _engine_inputs(), image)["latent"][0]
    _decode(decoder, latent, graph_walk=PRIME_WALK)
    assert encoder.taehv is decoder.taehv is taehv_weights
    assert encoder.request_states == {}
    assert len(decoder.request_state("r0").tensors) == 9


def test_ae_graphs_are_compiled_for_capture_but_remain_optional(encoder, decoder):
    encoder_configs = encoder.get_cuda_graph_configs(torch.device("cpu"))
    decoder_configs = decoder.get_cuda_graph_configs(torch.device("cpu"))
    assert [cfg.capture_graph_walk for cfg in encoder_configs] == [PRIME_WALK]
    assert {cfg.capture_graph_walk for cfg in decoder_configs} == {
        PRIME_WALK, ROLLOUT_WALK,
    }
    for node, configs in ((encoder, encoder_configs), (decoder, decoder_configs)):
        assert configs
        assert all(cfg.compile for cfg in configs)
        assert node.disable_torch_compile is True
        assert node.disable_autocast is True


def test_ae_nodes_declare_no_capture_when_cuda_graph_is_disabled(
    taehv_weights, ae_config,
):
    eager_config = dataclasses.replace(ae_config, cuda_graph=False)
    encoder = WaypointVaeEncoderSubmodule(taehv_weights, eager_config)
    decoder = WaypointVaeDecoderSubmodule(taehv_weights, eager_config)

    assert encoder.get_cuda_graph_configs(torch.device("cpu")) == []
    assert decoder.get_cuda_graph_configs(torch.device("cpu")) == []


def test_the_shell_builds_without_the_taehv_package(monkeypatch):
    """``taehv`` is a separate install with its own checkpoint. Every import of
    it is deferred to the call that needs weights, so the graph, the resources
    and the serialization gate all work on a box that has neither."""
    monkeypatch.setitem(sys.modules, "taehv", None)
    with pytest.raises(ImportError):
        importlib.import_module("taehv")

    unweighted = WaypointModel(skip_weight_loading=True)
    assert set(unweighted.get_graph_walk_graphs()) == {PRIME_WALK, ROLLOUT_WALK}
    assert unweighted.nodes == [DIT_NODE, VAE_DECODER_NODE, VAE_ENCODER_NODE]
    assert unweighted.get_node_resources()
    for node in (VAE_ENCODER_NODE, VAE_DECODER_NODE):
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
    """``image_inputs`` is what the vae_encoder node consumes, and the streaming
    encoder emits one latent per ``temporal_compression`` frames: a short clip
    would buffer and return nothing, a long one would encode twice and leave the
    second latent unclaimed. Checked here, at the API boundary, so a malformed
    request is a 400 rather than a rollout that dies on a worker."""
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
