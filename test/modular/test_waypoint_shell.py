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
  * a resource that never reaches the 24 attention layers.

CPU-only, checkpoint-free, and no engine. The real 720P config is used
throughout, because the numbers that collide are that config's numbers; the DiT
behind the submodule is built on ``torch.device("meta")`` and never
materialized, since a real 720P bf16 build is ~2.6 GB and none of these
assertions touch a weight. The two places that need a value read back go around
it: the noise draw takes an explicit device (which is why that helper takes
one), and the frame-clock test runs over ``_HostOnlyDit``, since what it
asserts is host bookkeeping the DiT is not part of.
"""

import dataclasses
import logging
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
from mstar.graph.base import GraphNode, Loop
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.waypoint.components.attention import WaypointAttention
from mstar.model.waypoint.components.dit import WaypointDiT
from mstar.model.waypoint.config import waypoint_1_5_1b_720p
from mstar.model.waypoint.submodules import (
    ATTN_RESOURCE,
    KV_RESOURCE,
    PRIME_WALK,
    ROLLOUT_LOOP_NAME,
    ROLLOUT_WALK,
    WaypointDitSubmodule,
)
from mstar.model.waypoint.waypoint_model import DIT_NODE, WaypointModel


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
    num_frames: int = 8,
    loop_iter: int | None = None,
) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=request_id,
        graph_walk=graph_walk,
        fwd_index=0,
        random_seed=random_seed,
        max_tokens=0,
        resource_configs={},
        step_metadata={"is_prefill": graph_walk == PRIME_WALK, "num_frames": num_frames},
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


def test_walks_are_dit_only_and_the_rollout_emits_per_iteration(model, config):
    walks = model.get_graph_walk_graphs()
    assert set(walks) == {PRIME_WALK, ROLLOUT_WALK}
    # The VAE nodes are a later phase. Model.nodes is derived from
    # the walks, so inventing one here would make the engine wait on a node
    # nothing builds.
    assert model.nodes == [DIT_NODE]

    prime = walks[PRIME_WALK]
    assert isinstance(prime, GraphNode) and prime.name == DIT_NODE
    assert "latent" in prime.input_names

    rollout = walks[ROLLOUT_WALK]
    assert isinstance(rollout, Loop)
    assert rollout.name == ROLLOUT_LOOP_NAME  # what check_stop's signal is keyed by
    assert rollout.max_iters == config.max_frames
    section = rollout.section
    assert isinstance(section, GraphNode) and section.name == DIT_NODE
    # Emitted from inside the loop, one frame per iteration -- an interactive
    # world model whose frames only arrive after the rollout ends has no world
    # to interact with.
    assert [e.next_node for e in section.outputs] == [EMIT_TO_CLIENT]
    assert rollout.accumulated_outputs == []
    # An overshoot iteration here is not a wasted forward: it commits a frame
    # into the ring, and there is no undo.
    assert section.enable_async_scheduling is False


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


def test_the_frame_clock_advances_on_the_host_and_drives_the_controller_slice(
    host_submodule, config
):
    """The clock is a host int in PerRequestState, and the device tensor is
    derived from it -- never the other way round. The scripted stream is
    loop-external, so this is the only thing that moves through it."""
    rid = "clock"
    host_submodule.request_states.pop(rid, None)
    inputs = _controller_stream(config, frames=3)
    # Distinguishable rows, so a slice off by one is visible.
    for row in range(3):
        inputs["scroll"][0][0, row, 0] = float(row + 1)

    seen = []
    for _ in range(5):
        node_inputs = host_submodule.prepare_inputs(
            ROLLOUT_WALK, _fwd_info(request_id=rid), inputs
        )
        seen.append(
            (
                int(node_inputs.tensor_inputs["frame_pos"][0]),
                float(node_inputs.tensor_inputs["scroll"][0, 0, 0]),
            )
        )
        host_submodule.postprocess(rid, _fwd_info(request_id=rid), {})

    # The clock advances by one per committed frame; the stream is shorter than
    # the rollout, so its last row holds rather than raising mid-flight.
    assert [pos for pos, _ in seen] == [0, 1, 2, 3, 4]
    assert [scroll for _, scroll in seen] == [1.0, 2.0, 3.0, 3.0, 3.0]

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


def test_check_stop_fires_at_exactly_num_frames(submodule):
    """N frames means firing while iteration N-1 is postprocessed: the loop
    counter still reads N-1 there and the stop ends that iteration. One early
    truncates the video; one late commits an extra frame into the ring."""
    num_frames = 6
    fired = [
        bool(
            submodule.check_stop(
                "r0", _fwd_info(num_frames=num_frames, loop_iter=k), {}
            )
        )
        for k in range(num_frames + 2)
    ]
    assert fired == [False] * (num_frames - 1) + [True, True, True]
    assert submodule.check_stop(
        "r0", _fwd_info(num_frames=num_frames, loop_iter=num_frames - 1), {}
    ) == {ROLLOUT_LOOP_NAME}


# ---------------------------------------------------------------------------
# Serialization gate
# ---------------------------------------------------------------------------


def _write_config(tmp_path, name: str, **extra) -> str:
    body = {
        "model": "waypoint",
        "max_seq_len": 512,
        "node_groups": [{"node_names": [DIT_NODE], "ranks": [0]}],
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


def test_one_capture_config_per_walk_both_uncompiled(submodule):
    """Two configs because the two walks take different input keys, and a walk
    with no bucket would run eager against a ring the other walk's captured
    graph holds baked addresses into.

    ``compile=False`` on both: ``_forward_for`` would otherwise run a
    max-autotune compile of the whole 4+1 driver once per config at warmup, for
    a model whose correctness-critical compile is the ``flex_attention_masked``
    pin inside the attention resource -- which runs regardless. The outer
    compile is an unmeasured throughput bet.
    """
    configs = submodule.get_cuda_graph_configs(torch.device("meta"))
    assert len(configs) == 2
    assert {c.capture_graph_walk for c in configs} == {PRIME_WALK, ROLLOUT_WALK}
    for cfg in configs:
        assert cfg.compile is False
        # One live world again: the bucket cannot be wider than it.
        assert cfg.capture_batch_sizes == [1]
        # The v1 engine always dispatches batched; a submodule captured on bare
        # `forward` is captured on a method that never runs.
        assert cfg.capture_forward_method == "forward_batched"

    by_walk = {c.capture_graph_walk: c for c in configs}
    assert "latent" in by_walk[PRIME_WALK].single_request_inputs.tensor_inputs
    assert "noise" in by_walk[ROLLOUT_WALK].single_request_inputs.tensor_inputs
    assert "noise" not in by_walk[PRIME_WALK].single_request_inputs.tensor_inputs
    assert "latent" not in by_walk[ROLLOUT_WALK].single_request_inputs.tensor_inputs


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
    out = model.process_prompt(
        None, ["tensor"], ["video"], tensors=None, num_frames=4, actions=actions
    )
    assert set(out) == {"mouse", "button", "scroll"}
    mouse, button, scroll = out["mouse"][0], out["button"][0], out["scroll"][0]
    assert mouse.shape == (1, 4, 2)
    assert button.shape == (1, 4, config.n_buttons)
    assert scroll.shape == (1, 4, 1)
    assert mouse[0, 0].tolist() == [1.0, -2.0]
    assert button[0, 0].nonzero().flatten().tolist() == [3, 5]
    assert button[0, 1].nonzero().flatten().tolist() == [3]
    # Unscripted frames are the idle controller, which is what the reference's
    # default CtrlInput() produces.
    assert button[0, 2].sum() == 0 and scroll[0, 2].sum() == 0

    with pytest.raises(ValueError, match="out of range"):
        model.process_prompt(
            None, ["tensor"], ["video"], num_frames=1,
            actions=[{"buttons": [config.n_buttons]}],
        )
    with pytest.raises(ValueError, match="never be read"):
        model.process_prompt(None, ["tensor"], ["video"], num_frames=1, actions=actions)
