"""Contract tests for the Waypoint serving shell: the model, its node
submodule, and the state it declares.

Not the DiT, the ring numerics, or the weight remap -- those have their own
suites. What's under test is the shell layer between the engine and a working
model: ring geometry, frame_pos rank, the ``_seq_dim`` collision, noise
statelessness, the off-by-one stop, admit-queue sizing, state binding, and
decode ordering -- each of which fails silently when it is wrong.

CPU-only, checkpoint-free, no engine. Uses the real 720P config since its
numbers are the ones that collide; the DiT and the ring are built on
``torch.device("meta")`` and never materialized (a real 720P bf16 build is
~2.6 GB). Two exceptions read a value back: the noise draw takes an explicit
device, and the frame-clock tests run over ``_HostOnlyDit``. The VAE section
runs on the 360P config with a fake ``taehv`` package.
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

from mstar.conductor.conductor import _walk_io
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import (
    AttentionSpec,
    AttnBackend,
    KVSpec,
    RingKVConfig,
    StepContext,
)
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.recurrent.config import RecurrentStateSpec
from mstar.engine.resources.runner import topo_sort
from mstar.engine.state.adapters import RingHandle, ValueHandle, node_handles, node_resources
from mstar.graph.base import (
    GraphEdge,
    Loop,
    Sequential,
    SpeculativeNodeInfo,
    TensorPointerInfo,
)
from mstar.graph.graph_io import WorkerGraphIO
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ClientChunk, serving_resources
from mstar.model.submodule_base import ModelInputsFromEngine, engine_step
from mstar.model.waypoint.components.attention import WaypointAttention
from mstar.model.waypoint.components.dit import WaypointDiT
from mstar.model.waypoint.components.taehv import decode_latent, decoder_history_shapes
from mstar.model.waypoint.config import (
    WAYPOINT_VARIANT_360P,
    WaypointConfig,
    waypoint_1_5_1b_360p,
    waypoint_1_5_1b_720p,
)
from mstar.model.waypoint.submodules import (
    ACTIONS_STREAM,
    DECODER,
    KV,
    PRIME_WALK,
    ROLLOUT_LOOP_NAME,
    ROLLOUT_WALK,
    WaypointDitSubmodule,
    WaypointVaeEncoderSubmodule,
)
from mstar.model.waypoint.waypoint_model import (
    DIT_NODE,
    VAE_ENCODER_NODE,
    WaypointModel,
)
from mstar.validation import DeploymentError, validate_deployment
from mstar.validation.client_streams import check_client_streams

# the flex attention over the ring, which takes the key of the ring's first read
FLEX = "local_attn"


@pytest.fixture(scope="module")
def config():
    return waypoint_1_5_1b_720p()


@pytest.fixture(scope="module")
def model():
    return WaypointModel(skip_weight_loading=True)


@pytest.fixture(scope="module")
def submodule(config, model):
    """The real submodule over a meta-built DiT, bound to the handles the model
    declares.

    Meta, not a stub: ``bind_node_state`` walking ``self.modules()`` and
    ``self.dit.dtype`` surviving ``cast_serving_dtypes`` are under test here,
    and a stub would assert them against itself. Meta also keeps
    ``prepare_inputs``' shapes/dtypes honest while allocating nothing.

    ``_FakeTaehv`` (defined later; fixtures resolve names at call time) only
    needs its structural facts to build shapes, not real weights.
    """
    with torch.device("meta"):
        dit = WaypointDiT(config)
    dit.cast_serving_dtypes()
    return _bound(WaypointDitSubmodule(dit, _FakeTaehv(), config), model)


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
def host_submodule(config, model):
    return _bound(WaypointDitSubmodule(_HostOnlyDit(), _FakeTaehv(), config), model, sessions=3)


def _bound(submodule: WaypointDitSubmodule, model: WaypointModel, sessions: int = 1) -> WaypointDitSubmodule:
    """``submodule`` bound to the dit node's handles, over the ring and the
    decoder pool ``model`` declares, sized for ``sessions`` worlds. The ring is
    built on meta, since no test here runs the attention that reads it, and the
    pool beside the submodule's weights."""
    specs = {spec.resource_key: spec for spec in serving_resources(model, _sessions(sessions))}
    resources = {
        KV: build_resource(specs[KV], EngineResourceInfo(device=torch.device("meta"))),
        DECODER: build_resource(specs[DECODER], EngineResourceInfo(device=submodule.get_device())),
        FLEX: None,
    }
    submodule.bind_node_state(node_handles(model, DIT_NODE, resources))
    return submodule


def _resources(submodule: WaypointDitSubmodule) -> dict:
    """The ring and the decoder pool under ``submodule``'s handles, by key."""
    return {KV: submodule.kv._ring, DECODER: submodule.decoder._pool}


def _step(submodule: WaypointDitSubmodule, walk: str, request_ids: list[str]):
    """The step ``submodule`` declares for ``request_ids``, admitted and planned
    on its ring and decoder pool as the engine runs it."""
    step = engine_step(submodule, graph_walk=walk, request_ids=request_ids, inputs=[])
    step.set_ctx(StepContext(request_ids=request_ids, graph_walk=walk, slot=0, capture=False))
    resources = _resources(submodule)
    for rid in request_ids:
        if rid not in resources[KV]._known_rids:
            for resource in resources.values():
                resource.ingest_request(rid)
    for key, resource in resources.items():
        assert resource.admit(step.steps[key], step.ctx).ok, f"{key} should hold every row"
        resource.plan(step.steps[key], step.ctx)
    return step


def _commit(submodule: WaypointDitSubmodule, step) -> None:
    for key, resource in _resources(submodule).items():
        resource.commit(step.steps[key], step.ctx)


def _fwd_info(
    request_id: str = "r0",
    graph_walk: str = ROLLOUT_WALK,
    random_seed: int = 1234,
) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=request_id,
        # prepare_inputs asks the ring for the handle's frames, and a step's
        # rows carry the same value, so one token serves as both.
        rid_handle=request_id,
        graph_walk=graph_walk,
        fwd_index=0,
        random_seed=random_seed,
        max_tokens=0,
        resource_configs={},
        resource_publish_info={},
    )


def _action(config, scroll: float = 0.0) -> dict[str, list[torch.Tensor]]:
    """One frame's ``actions`` chunk, a row shaped the way ``process_prompt`` emits it."""
    row = torch.zeros(2 + config.n_buttons + 1)
    row[-1] = scroll
    return {ACTIONS_STREAM: [row]}


# ---------------------------------------------------------------------------
# Declared state
# ---------------------------------------------------------------------------


def test_declares_exactly_the_ring_the_flex_attention_over_it_and_the_decoder_s_histories(model):
    specs = node_resources(model)
    assert [type(s) for s in specs] == [KVSpec, AttentionSpec, RecurrentStateSpec]
    assert [s.resource_key for s in specs] == [KV, FLEX, DECODER]
    assert all(s.nodes == {DIT_NODE} for s in specs)

    assert isinstance(specs[0].config, RingKVConfig)
    # FLEX, not the FLASHINFER default: a paged kernel reassociates the
    # accumulation over KV blocks, breaking bit-exactness against the reference.
    assert specs[1].config.backend is AttnBackend.FLEX
    assert specs[1].config.kv_cache == KV


def test_ring_geometry_follows_the_config_layer_for_layer(model, config):
    """Waypoint's 24 layers are not alike: the global ones keep every 8th
    frame of the last 128 in 16 slots, the local ones the last 16 frames.
    Asserted one layer at a time so a summarizing regression names the layer it
    broke."""
    ring = node_resources(model)[0].config
    assert ring.num_layers == config.n_layers == 24
    assert ring.num_kv_heads == config.n_kv_heads
    assert ring.num_qo_heads == config.n_heads
    assert ring.head_dim == config.d_head
    assert ring.tokens_per_frame == config.tokens_per_frame
    # Sizing is a deployment question (`apply_yaml_overrides` runs after the
    # declaration); what's pinned here is the *default* of one session.
    assert ring.num_sessions == 1
    assert ring.max_frames == 1 + config.max_frames, "the ring would refuse the seed frame or the last rollout frame"

    assert len(ring.layers) == config.n_layers
    for i, layer in enumerate(ring.layers):
        is_global = config.is_global_layer(i)
        stride = config.global_pinned_dilation if is_global else 1
        slots = (config.global_window if is_global else config.local_window) // stride
        assert layer.ring_frames == slots, f"ring_frames, layer {i}"
        assert layer.ring_buckets == slots, f"ring_buckets, layer {i}"
        assert layer.pinned_dilation == stride, f"dilation, layer {i}"

    # The heterogeneity itself: if these two collapse to one value the loop
    # above would pass against a config that had also been flattened.
    assert {ring.layers[i].pinned_dilation for i in config.global_layers} == {
        config.global_pinned_dilation
    }
    local = set(range(config.n_layers)) - config.global_layers
    assert {ring.layers[i].pinned_dilation for i in local} == {1}


def test_attention_resolves_after_the_cache_it_names(model):
    by_key = {s.resource_key: s for s in node_resources(model)}
    assert by_key[FLEX].depends_on() == {KV}
    assert by_key[KV].depends_on() == set()
    # topo_sort only calls depends_on(), so the specs stand in for the built
    # resources here; the order it returns is the order the engine builds in.
    order = topo_sort(by_key)
    assert order.index(KV) < order.index(FLEX), "the flex attention would build before the ring it reads"


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
    # Prime runs on its own idle action; the client's actions are the rollout's.
    assert dit.input_names == {"latent"}
    # Nothing to route: the decode happens inside this forward, and the
    # reconstructed seed frames it produces are internal-only.
    assert dit.outputs == []

    io = prime.get_inputs_outputs()
    assert io.ext_inputs == {("image_inputs", VAE_ENCODER_NODE)}
    assert io.ext_outputs == []


def test_the_rollout_loop_decodes_and_emits_every_iteration(model, config):
    rollout = model.get_graph_walk_graphs()[ROLLOUT_WALK]
    assert isinstance(rollout, Loop)
    assert rollout.name == ROLLOUT_LOOP_NAME
    assert rollout.max_iters == config.max_frames

    dit = rollout.section
    assert dit.name == DIT_NODE
    assert dit.input_names == {ACTIONS_STREAM, "clock"} and dit.consumes_stream
    assert {(e.name, e.next_node) for e in dit.outputs} == {
        ("clock", DIT_NODE), ("video_output", EMIT_TO_CLIENT),
    }
    assert rollout.accumulated_outputs == []
    # The "clock" self loop-back makes the dit a same-node speculation target
    # (see below), so async scheduling can dispatch iteration N+1 before N
    # finishes; a step past the stream's last row finds none.
    assert dit.enable_async_scheduling is True
    # The actions stream is loop-external; "clock" is the only loop-back.
    assert rollout._external_inputs == {(ACTIONS_STREAM, DIT_NODE)}
    assert rollout._loop_back_inputs == {("clock", DIT_NODE)}
    # one client stream, one row per frame, capped at the horizon
    (stream,) = model.get_partition_topology().client_connections()
    assert stream.edge_name == ACTIONS_STREAM and stream.limits.max_items == config.max_frames
    assert check_client_streams(model, {}) == []


def test_the_clock_loop_back_makes_dit_a_same_node_speculation_target(model):
    """``ingest_for_speculation`` must propose the dit as ready for its own
    next iteration once its actions chunk is in and the empty "clock"
    loop-back (iteration 0) has been ingested -- the self-edge this graph
    shape depends on for speculation to fire at all.
    """
    rollout = model.get_graph_walk_graphs()[ROLLOUT_WALK]
    wgio = WorkerGraphIO(rollout)
    dit = wgio.get_node(DIT_NODE)
    wgio.ingest_input(GraphEdge(next_node=DIT_NODE, name=ACTIONS_STREAM))
    wgio.ingest_input(GraphEdge(next_node=DIT_NODE, name="clock"))
    assert wgio.ready_node_names == {DIT_NODE}

    assert wgio.ingest_for_speculation(dit.outputs, DIT_NODE) == [
        SpeculativeNodeInfo(node_name=DIT_NODE, is_new_loop_iter=True, loop_name=ROLLOUT_LOOP_NAME)
    ]


def test_the_rollout_loop_closes_on_the_stream_s_last_row(model):
    """Driven through ``WorkerGraphIO``: one actions chunk per iteration, and
    the stream-terminated stop (``register_loop_finish_signal``) on the last
    one, whose frame must still be emitted before the loop reports done."""
    frames = 3
    rollout = model.get_graph_walk_graphs()[ROLLOUT_WALK]
    wgio = WorkerGraphIO(rollout)
    wgio.ingest_input(GraphEdge(next_node=DIT_NODE, name="clock"))

    emitted = []
    for step in range(frames):
        wgio.ingest_input(GraphEdge(next_node=DIT_NODE, name=ACTIONS_STREAM))
        assert wgio.ready_node_names == {DIT_NODE}
        wgio.ready_node_names.discard(DIT_NODE)
        if step == frames - 1:
            wgio.register_loop_finish_signal(ROLLOUT_LOOP_NAME)  # the stream's final chunk
        completion = wgio.mark_node_complete(DIT_NODE)
        # mark_node_complete already stripped filtered_signals (e.g. the final
        # iteration's "clock" loop-back) from output_edges; the rest is ours to route.
        for edge in completion.output_edges:
            if edge.next_node == EMIT_TO_CLIENT:
                emitted.append(edge.name)
                continue
            wgio.ingest_input(edge)
        assert rollout.is_done == (step == frames - 1)

    assert emitted == ["video_output"] * frames


def test_a_rollout_step_on_the_empty_final_chunk_is_declined(submodule):
    """The stream's end can arrive after its last row was taken: the empty
    final chunk is declined, which ends the loop with nothing committed into
    the ring. Prime takes no actions at all."""
    rid = "end"
    assert submodule.prepare_inputs(
        ROLLOUT_WALK, _fwd_info(request_id=rid), {ACTIONS_STREAM: []}, is_final_stream_chunk=True,
    ) is None
    prime = {"latent": [torch.zeros((1, 1, *submodule.config.latent_shape))]}
    assert submodule.prepare_inputs(PRIME_WALK, _fwd_info(request_id=rid, graph_walk=PRIME_WALK), prime) is not None


# ---------------------------------------------------------------------------
# Per-step inputs: rank and the _seq_dim collision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("walk", [PRIME_WALK, ROLLOUT_WALK])
def test_no_prepared_tensor_is_rank_zero(submodule, config, walk):
    """A 0-dim tensor reaches ``_intern_static_buffer`` (reads
    ``stored.shape[0]``) and the worker's output fanout (reads ``dims[0]``)
    -- both IndexError at capture/warmup, far from the line that made it."""
    inputs = {**_action(config), "latent": [torch.zeros((1, 1, *config.latent_shape))]}
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
    inputs = {**_action(config), "latent": [torch.zeros((1, 1, *config.latent_shape))]}
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
        for cfg in submodule.get_accelerator_graph_configs(torch.device("meta"))
    }
    inputs = {**_action(config), "latent": [torch.zeros((1, 1, *config.latent_shape))]}

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
    for cfg in submodule.get_accelerator_graph_configs(torch.device("meta")):
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
    """Prime commits the ring's first frame and takes no action; each rollout
    frame takes its own chunk's row."""
    rid = "clock"

    prime_inputs = {"latent": [torch.zeros((1, 1, *config.latent_shape), dtype=torch.float32)]}
    prime = host_submodule.prepare_inputs(
        PRIME_WALK, _fwd_info(request_id=rid, graph_walk=PRIME_WALK), prime_inputs
    )
    assert int(prime.tensor_inputs["frame_pos"][0]) == 0
    assert float(prime.tensor_inputs["scroll"][0, 0, 0]) == 0.0
    _commit(host_submodule, _step(host_submodule, PRIME_WALK, [rid]))

    seen = []
    for row in range(2):
        # distinguishable rows, so a row off by one is visible
        node_inputs = host_submodule.prepare_inputs(
            ROLLOUT_WALK, _fwd_info(request_id=rid), _action(config, scroll=float(row + 1))
        )
        seen.append((
            int(node_inputs.tensor_inputs["frame_pos"][0]),
            float(node_inputs.tensor_inputs["scroll"][0, 0, 0]),
        ))
        _commit(host_submodule, _step(host_submodule, ROLLOUT_WALK, [rid]))

    assert seen == [(1, 1.0), (2, 2.0)]


def test_declare_step_counts_a_frame_for_every_request_in_the_batch(host_submodule, config):
    """No batch shape skips a row: a request whose frame goes uncounted writes
    its next frame over this one and quietly rewrites its own history.

    The counts below are genuinely different, so a frame read off one request
    and broadcast across the batch fails here rather than passing on a
    coincidence.
    """
    for i, rid in enumerate(("a", "b", "c")):
        for _ in range(i * 2):
            _commit(host_submodule, _step(host_submodule, ROLLOUT_WALK, [rid]))

    _commit(host_submodule, _step(host_submodule, ROLLOUT_WALK, ["a", "b", "c"]))

    frames = [
        int(host_submodule.prepare_inputs(ROLLOUT_WALK, _fwd_info(request_id=rid), _action(config))
            .tensor_inputs["frame_pos"][0])
        for rid in ("a", "b", "c")
    ]
    assert frames == [1, 3, 5], "each request should read its own count, one frame past the batch step"


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
    """The ``resources:`` block a deployment writes for ``n`` worlds: ``n``
    ring sessions, and a decoder slot for each beside the pool's sink slot. The
    same block ``EngineManager.build`` feeds to ``apply_yaml_overrides``."""
    return {"resources": {KV: {"num_sessions": n}, DECODER: {"max_slots": n + 1}}}


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
def test_the_deployment_check_refuses_an_invalid_world_pool_size(
    model, tmp_path, worlds
):
    """The model no longer reads ``num_sessions``: the ring's own check on the
    ``resources:`` block refuses it at startup, with the deployment's other
    problems."""
    path = _write_config(
        tmp_path,
        f"invalid_worlds_{worlds}.yaml",
        max_concurrent_requests=1,
        resources={KV: {"num_sessions": worlds}},
    )
    with pytest.raises(DeploymentError, match=r"resources\.kv: num_sessions must be a positive int"):
        validate_deployment(model, yaml.safe_load(pathlib.Path(path).read_text()), path)


@pytest.mark.parametrize(("limit", "worlds"), [(2, 1), (8, 4), (2, None)])
def test_validate_config_yaml_refuses_more_arrivals_than_worlds(
    model, tmp_path, limit, worlds
):
    """A queue longer than the pool would wait forever: the conductor admits
    ``limit`` requests, the ring hands out ``num_sessions``, and the
    difference is refused at ``admit`` until a session is released, which a
    request that never finishes never does. Refused at startup instead.

    ``worlds=None`` is the accidental case: raising
    ``max_concurrent_requests`` without touching ``resources``, the shape
    every pre-pool config already has.
    """
    extra = {"max_concurrent_requests": limit}
    if worlds is not None:
        extra |= _sessions(worlds)
    path = _write_config(tmp_path, f"over_{limit}_{worlds}.yaml", **extra)

    with pytest.raises(ValueError, match=rf"need {limit} sessions of resources\.kv"):
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


@pytest.mark.parametrize("config_name", ["waypoint.yaml", "waypoint_360p.yaml"])
def test_the_shipped_config_serializes_both_nodes_onto_one_rank(model, config_name):
    """``configs/waypoint.yaml`` and ``configs/waypoint_360p.yaml`` are the two
    shipped deployments and each must pass its own gate.

    Both nodes in one group, on rank 0: a node missing from ``node_groups``
    has no rank to run on. There's no decoder node to put a worker boundary
    in front of -- decode is fused into the dit's forward -- so a rank split
    inside the rollout loop isn't expressible any more.
    """
    path = pathlib.Path(__file__).resolve().parents[2] / "configs" / config_name

    _validate(model, str(path))
    graphs = model.get_worker_graphs(str(path))

    assert {walk for g in graphs for walk in g.graph_walks} == {PRIME_WALK, ROLLOUT_WALK}
    assert {tuple(g.ranks) for g in graphs} == {(0,)}
    by_walk = {walk: g for g in graphs for walk in g.graph_walks}
    assert set(by_walk[PRIME_WALK].section.get_nodes()) == {VAE_ENCODER_NODE, DIT_NODE}
    assert set(by_walk[ROLLOUT_WALK].section.get_nodes()) == {DIT_NODE}


def test_validate_config_yaml_refuses_a_step_wider_than_the_cap(tmp_path):
    """A step's rows are requests, so the cap bounds it, even where the ring
    has a session for every row."""
    model = WaypointModel(skip_weight_loading=True, step_batch_size=4)
    path = _write_config(
        tmp_path, "wide_step.yaml", max_concurrent_requests=2, **_sessions(4)
    )
    with pytest.raises(ValueError, match="step_batch_size: 4` exceeds `max_concurrent_requests: 2"):
        _validate(model, path)


def test_validate_config_yaml_warns_about_worlds_no_request_can_reach(
    model, tmp_path, caplog
):
    """Legal, only wasteful -- a warning, not a refusal. Each unreachable
    world is ~816 MiB of ring at 720P allocated and never written, worth a
    log line rather than a failed boot. The ring's pool check logs it."""
    path = _write_config(
        tmp_path, "underused.yaml", max_concurrent_requests=2, **_sessions(8)
    )

    with caplog.at_level(logging.WARNING, logger="mstar.engine.resources.kv.config"):
        _validate(model, path)

    assert any(
        r.name == "mstar.engine.resources.kv.config" and "6 of them are allocated and never filled" in r.getMessage()
        for r in caplog.records
    ), "the ring's six unreachable sessions should be logged"


# ---------------------------------------------------------------------------
# State binding
# ---------------------------------------------------------------------------


def test_bind_reaches_the_submodule_and_all_twenty_four_attention_layers(submodule):
    """One bind has to reach every caller: the submodule reads the ring's
    frames and the decoder's histories, and the 24 attention layers hold their
    own reference to the ring and call ``add``/``read`` directly. Anything
    left unbound raises mid-forward -- or, during warmup, poisons the capture
    instead of failing a request."""
    assert type(submodule.kv) is RingHandle and type(submodule.decoder) is ValueHandle

    layers = [m for m in submodule.modules() if isinstance(m, WaypointAttention)]
    assert len(layers) == 24
    for layer in layers:
        assert layer.state is submodule.kv, f"layer {layer.layer_idx} unbound"


def test_binding_without_a_declared_handle_fails_at_bind(submodule):
    """Not mid-forward: the submodule and its layers take each handle by its
    declared name, so a missing one is named when the node binds."""
    for partial in ({DECODER: object()}, {KV: object()}, {}):
        with pytest.raises(KeyError):
            submodule.bind_node_state(partial)


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
    configs = submodule.get_accelerator_graph_configs(torch.device("meta"))
    # Rollout first: the two share one graph pool and rollout's five forwards
    # are a superset of prime's one, so rollout sizes the pool; the runner's
    # largest-first sort is stable given both specs are (1, tokens_per_frame).
    assert [cfg.capture_graph_walk for cfg in configs] == [ROLLOUT_WALK, PRIME_WALK]
    for cfg in configs:
        assert cfg.compile is False
        # One live world again: the bucket cannot be wider than it.
        assert cfg.capture_batch_sizes == [1]
        # A captured forward returns per-row outputs: the engine maps a replay's
        # entries back by row id.
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


def test_both_walks_capture_the_same_geometric_buckets(config):
    """With ``step_batch_size > 1`` prime captures the same geometric buckets as
    rollout and caps eagerly at the top of them, so a multi-request prime batch
    replays a captured, padded graph rather than re-tracing eagerly."""
    batched = dataclasses.replace(config, step_batch_size=8)
    with torch.device("meta"):
        dit = WaypointDiT(batched)
    dit.cast_serving_dtypes()
    submodule = WaypointDitSubmodule(dit, _FakeTaehv(), batched)

    configs = submodule.get_accelerator_graph_configs(torch.device("meta"))
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

    configs = submodule.get_accelerator_graph_configs(torch.device("meta"))
    assert [cfg.capture_graph_walk for cfg in configs] == [ROLLOUT_WALK]
    assert "noise" in configs[0].single_request_inputs.tensor_inputs


def test_dit_declares_no_capture_when_accelerator_graph_is_disabled(config):
    eager_config = dataclasses.replace(config, accelerator_graph=False)
    with torch.device("meta"):
        dit = WaypointDiT(eager_config)
    dit.cast_serving_dtypes()
    eager = WaypointDitSubmodule(dit, _FakeTaehv(), eager_config)

    assert eager.get_accelerator_graph_configs(torch.device("meta")) == []


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
    """Cheap tensor-only TAEHV with the released architecture facts. Its
    decoder has the pinned one's three stages of three MemBlocks, at 256, 128
    and 64 channels and each at twice the height and width of the one before,
    so its histories are the ones Waypoint's state declares."""

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
            torch.nn.Conv2d(32, 256, 1, bias=False),
            *(MemBlock(256) for _ in range(3)),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(256, 128, 1, bias=False),
            *(MemBlock(128) for _ in range(3)),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 64, 1, bias=False),
            *(MemBlock(64) for _ in range(3)),
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
    minus the denoiser, bound to a decoder pool of two worlds. Named
    ``decoder`` because every test below exercises the TAEHV history half of
    this node."""
    model = WaypointModel(skip_weight_loading=True, variant=WAYPOINT_VARIANT_360P)
    return _bound(WaypointDitSubmodule(_FakeDit(ae_config), taehv_weights, ae_config), model, sessions=2)


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
    seed_latent: torch.Tensor | None = None,
):
    """Drive the fused submodule through one real step -- ``prepare_inputs``,
    the step it declares admitted and planned, ``forward`` between the step's
    calls, then the commit, the same sequence the engine runs -- rather than
    poking ``taehv`` directly, since several tests below check that this cycle
    threads the nine histories correctly.
    """
    info = _fwd_info(request_id, graph_walk=graph_walk)
    if graph_walk == PRIME_WALK:
        inputs = {"latent": [seed_latent]}
    else:
        inputs = _action(decoder.config)
    prepared = decoder.prepare_inputs(graph_walk, info, inputs)
    step = _step(decoder, graph_walk, [request_id])
    step.begin_calls()
    outputs = decoder.forward(
        graph_walk,
        _engine_inputs(request_id, graph_walk),
        **prepared.tensor_inputs,
    )
    step.check_calls()
    _commit(decoder, step)
    step.end_calls()
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
    # capture runs the method its config names, keyed by row
    cfg = encoder.get_accelerator_graph_configs(torch.device("cpu"))[0]
    captured = getattr(encoder, cfg.capture_forward_method)(PRIME_WALK, _engine_inputs(), image=image)
    assert set(captured) == {"r0"}
    assert torch.equal(captured["r0"]["latent"][0], latent)


def _seed_hw_info(
    request_id: str = "r0", *, seed_hw, graph_walk: str = PRIME_WALK,
) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=request_id,
        rid_handle=request_id,
        graph_walk=graph_walk,
        fwd_index=0,
        random_seed=0,
        max_tokens=0,
        step_metadata={"seed_hw": seed_hw},
    )


def test_encoder_captures_one_config_per_seed_size(taehv_weights, ae_config):
    multi = dataclasses.replace(
        ae_config, seed_capture_sizes=((720, 1280), (1080, 1920)),
    )
    encoder = WaypointVaeEncoderSubmodule(taehv_weights, multi)

    configs = encoder.get_accelerator_graph_configs(torch.device("cpu"))

    assert {cfg.additional_key_info for cfg in configs} == {
        (360, 640), (720, 1280), (1080, 1920),
    }
    for cfg in configs:
        h, w = cfg.additional_key_info
        image = cfg.single_request_inputs.tensor_inputs["image"]
        assert image.shape == (ae_config.temporal_compression, h, w, 3)
        assert cfg.capture_graph_walk == PRIME_WALK


def test_cg_key_info_returns_the_captured_size_for_a_matching_seed(
    taehv_weights, ae_config,
):
    multi = dataclasses.replace(ae_config, seed_capture_sizes=((720, 1280),))
    encoder = WaypointVaeEncoderSubmodule(taehv_weights, multi)

    assert encoder.cg_key_info(
        PRIME_WALK, {"r0": _seed_hw_info(seed_hw=(720, 1280))}
    ) == (720, 1280)
    assert encoder.cg_key_info(
        PRIME_WALK, {"r0": _seed_hw_info(seed_hw=(360, 640))}
    ) == (360, 640)


def test_cg_key_info_normalizes_a_list_seed_hw_to_a_tuple(encoder):
    assert encoder.cg_key_info(
        PRIME_WALK, {"r0": _seed_hw_info(seed_hw=[360, 640])}
    ) == (360, 640)


def test_cg_key_info_is_none_for_an_uncaptured_seed_size(encoder):
    assert encoder.cg_key_info(
        PRIME_WALK, {"r0": _seed_hw_info(seed_hw=(540, 960))}
    ) is None


def test_cg_key_info_is_none_off_the_prime_walk(encoder):
    info = {"r0": _seed_hw_info(seed_hw=(360, 640), graph_walk=ROLLOUT_WALK)}
    assert encoder.cg_key_info(ROLLOUT_WALK, info) is None


def test_cg_key_info_is_none_without_a_seed_hw(encoder):
    info = CurrentForwardPassInfo(
        request_id="r0", graph_walk=PRIME_WALK, fwd_index=0, random_seed=0,
        max_tokens=0, step_metadata={},
    )
    assert encoder.cg_key_info(PRIME_WALK, {"r0": info}) is None


def test_cg_key_info_is_none_when_the_batch_disagrees_on_seed_hw(encoder):
    per_request_info = {
        "r0": _seed_hw_info("r0", seed_hw=(360, 640)),
        "r1": _seed_hw_info("r1", seed_hw=(720, 1280)),
    }
    assert encoder.cg_key_info(PRIME_WALK, per_request_info) is None


def test_the_fused_decode_turns_one_frame_into_one_raw_clip(decoder, ae_config):
    out = _decode(decoder, graph_walk=PRIME_WALK, seed_latent=torch.zeros(
        (1, 1, ae_config.channels, *ae_config.latent_shape[1:]), dtype=torch.bfloat16
    ))

    frames = out["video_output"][0]
    # forward() is the pre-split batched call (row B=1 here); the engine's
    # forward_batched drops this leading dim per row before postprocess sees it.
    assert frames.shape == (1, ae_config.temporal_compression, 360, 640, 3)
    assert frames.dtype == torch.uint8


def test_forward_batched_hands_each_request_its_own_row(decoder, monkeypatch):
    """``forward`` returns ``{key: [batched_tensor]}``; ``forward_batched`` must
    index the tensor's rows, not the one-element list around it, and hand row
    ``i`` to ``request_ids[i]``. ``video_output`` drops the batch dim (what
    ``postprocess`` consumes); every other key keeps a leading 1 (what
    ``prepare_inputs`` builds for the next step)."""
    frames = torch.arange(2 * 4 * 2 * 2 * 3, dtype=torch.uint8).reshape(2, 4, 2, 2, 3)
    clock = torch.tensor([5, 9])
    monkeypatch.setattr(
        decoder, "forward", lambda *a, **k: {"video_output": [frames], "clock": [clock]},
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
        assert all(isinstance(v[0], torch.Tensor) for v in out[rid].values())


def test_decode_latent_batches_rows_independently(taehv_weights, ae_config):
    """``decode_latent`` at B=2 equals two independent B=1 calls, row for row.

    ``_FakeTaehv``'s decoder is convs, upsamples and reshapes that never mix
    across the batch dim, so row 1 must not depend on row 0's latent -- the
    property the batched engine path relies on."""
    latent_shape = (1, ae_config.channels, *ae_config.latent_shape[1:])
    generator = torch.Generator().manual_seed(7)
    latents = [
        torch.randn(latent_shape, generator=generator).to(torch.bfloat16)
        for _ in range(2)
    ]
    histories = [
        tuple(latent.new_zeros(1, *shape) for shape in decoder_history_shapes(*latent.shape[-2:]))
        for latent in latents
    ]

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


def test_fused_decode_histories_are_isolated_interleaved_and_cleaned_up(decoder, ae_config, monkeypatch):
    """Each decode reads its request's nine histories as that request's last
    decode wrote them, however requests interleave, and a request primed again
    after its removal starts from zeros."""
    decodes = []
    decode = decoder._decode_latent

    def recording(ae_model, latent, histories, **kwargs):
        frames, written = decode(ae_model, latent, histories, **kwargs)
        decodes.append((histories, written))
        return frames, written

    monkeypatch.setattr(decoder, "_decode_latent", recording)
    shape = (1, 1, ae_config.channels, *ae_config.latent_shape[1:])
    # different seed latents, so a request reading the other's histories shows
    latents = {"a": torch.full(shape, 0.125, dtype=torch.bfloat16), "b": torch.full(shape, 0.5, dtype=torch.bfloat16)}
    first = _decode(decoder, request_id="a", graph_walk=PRIME_WALK, seed_latent=latents["a"])
    _decode(decoder, request_id="b", graph_walk=PRIME_WALK, seed_latent=latents["b"])
    for _ in range(3):
        _decode(decoder, request_id="a", graph_walk=ROLLOUT_WALK)
        _decode(decoder, request_id="b", graph_walk=ROLLOUT_WALK)

    written = {}
    for rid, (read, wrote) in zip("ab" * 4, decodes, strict=True):
        expected = written.get(rid)
        if expected is None:
            assert not any(history.any() for history in read), f"{rid}'s prime should read zeros"
        else:
            assert all(map(torch.equal, read, expected)), f"{rid} should read what its own last decode wrote"
        written[rid] = wrote

    for resource in _resources(decoder).values():
        resource.remove_request("a")
    restarted = _decode(decoder, request_id="a", graph_walk=PRIME_WALK, seed_latent=latents["a"])
    assert torch.equal(restarted["video_output"][0], first["video_output"][0])
    assert not any(history.any() for history in decodes[-1][0]), "a request primed again should start from zeros"


def test_the_encoder_and_the_fused_decode_share_only_weights(
    encoder, decoder, taehv_weights, ae_config
):
    """Encoder prime is stateless; only the fused decode's histories survive a
    request, in its slot of the decoder pool."""
    image = encoder.prepare_inputs(
        PRIME_WALK, _fwd_info(), {"image_inputs": [_seed_clip(ae_config)]}
    ).tensor_inputs["image"]
    latent = encoder.forward(PRIME_WALK, _engine_inputs(), image)["latent"][0]
    _decode(decoder, graph_walk=PRIME_WALK, seed_latent=latent)
    assert encoder.taehv is decoder.taehv is taehv_weights
    assert encoder.request_states == {}
    assert _resources(decoder)[DECODER]._slots["r0"]["main"].has_state, "the prime's histories should be kept"


def test_ae_graphs_are_compiled_for_capture_but_remain_optional(encoder, decoder):
    encoder_configs = encoder.get_accelerator_graph_configs(torch.device("cpu"))
    fused_configs = decoder.get_accelerator_graph_configs(torch.device("cpu"))
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


def test_ae_nodes_declare_no_capture_when_accelerator_graph_is_disabled(
    taehv_weights, ae_config,
):
    eager_config = dataclasses.replace(ae_config, accelerator_graph=False)
    encoder = WaypointVaeEncoderSubmodule(taehv_weights, eager_config)
    fused = WaypointDitSubmodule(_FakeDit(eager_config), taehv_weights, eager_config)

    assert encoder.get_accelerator_graph_configs(torch.device("cpu")) == []
    assert fused.get_accelerator_graph_configs(torch.device("cpu")) == []


def test_seed_capture_sizes_rejects_a_non_16_9_entry(ae_config):
    with pytest.raises(ValueError, match="16:9"):
        dataclasses.replace(ae_config, seed_capture_sizes=((100, 100),))


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
    assert node_resources(unweighted)
    for node in (VAE_ENCODER_NODE, DIT_NODE):
        assert unweighted.get_submodule(node) is None


# ---------------------------------------------------------------------------
# Request shaping
# ---------------------------------------------------------------------------


def test_process_prompt_sends_the_actions_as_the_stream_s_first_chunk(model, config):
    """One row per action, ``[mouse | buttons one-hot | scroll]``, the DiT
    splits by column. Validated here rather than in ``prepare_inputs`` because
    this runs at the API boundary, where a ValueError becomes a 400 instead of
    killing a rollout mid-flight."""
    actions = [
        {"mouse": (1.0, -2.0), "buttons": [3, 5], "scroll": 0.5},
        {"buttons": [3]},
    ]
    seed = torch.zeros((720, 1280, 3), dtype=torch.uint8)
    out = model.process_prompt(
        None, ["image"], ["video_frame"], tensors={"image_inputs": [seed]}, actions=actions,
    )
    assert set(out) == {"image_inputs", ACTIONS_STREAM}
    first, second = out[ACTIONS_STREAM]
    assert first.shape == second.shape == (2 + config.n_buttons + 1,)
    assert first[:2].tolist() == [1.0, -2.0] and float(first[-1]) == 0.5
    assert first[2:-1].nonzero().flatten().tolist() == [3, 5]
    assert second[2:-1].nonzero().flatten().tolist() == [3]
    with pytest.raises(ValueError, match="out of range"):
        model.process_prompt(
            None, ["image"], ["video_frame"],
            tensors={"image_inputs": [seed]}, actions=[{"buttons": [config.n_buttons]}],
        )


@pytest.mark.parametrize("count", ["none", 0, "past the horizon"])
def test_process_prompt_takes_one_to_max_frames_actions(model, config, count):
    actions = {"none": None, 0: [], "past the horizon": [{}] * (config.max_frames + 1)}[count]
    seed = torch.zeros((720, 1280, 3), dtype=torch.uint8)
    with pytest.raises(ValueError, match=f"1 to {config.max_frames} actions"):
        model.process_prompt(
            None, ["image"], ["video_frame"], tensors={"image_inputs": [seed]}, actions=actions,
        )


def test_process_prompt_refuses_the_old_num_steps(model):
    """num_steps went with the actions stream; a client still sending it is told
    the frame count is len(actions), not left with its count ignored."""
    seed = torch.zeros((720, 1280, 3), dtype=torch.uint8)
    with pytest.raises(ValueError, match="no longer takes num_steps"):
        model.process_prompt(
            None, ["image"], ["video_frame"], tensors={"image_inputs": [seed]}, actions=[{}], num_steps=1,
        )


def test_a_live_append_carries_its_actions_in_model_kwargs(model, config):
    def chunk(**fields):
        return ClientChunk(**{
            "text": None, "tensors": {}, "input_metadata": {}, "streams": {},
            "model_kwargs": {}, "request_kwargs": {}, **fields,
        })

    (row,) = model.process_stream_chunk(chunk(model_kwargs={"actions": [{"scroll": 2.0}]}))[ACTIONS_STREAM]
    assert row.shape == (2 + config.n_buttons + 1,) and float(row[-1]) == 2.0
    with pytest.raises(ValueError, match="model_kwargs actions only"):
        model.process_stream_chunk(chunk(text="left"))
    with pytest.raises(ValueError, match="model_kwargs.actions"):
        model.process_stream_chunk(chunk(model_kwargs={"actions": {"scroll": 1}}))


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
            None, ["tensor"], ["video_frame"], tensors=None, actions=[{}, {}],
        )


def _pointer_info(dims: list[int]) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=dims, dtype="uint8", nbytes=0, address=0, stride=[1] * len(dims),
        uuid="u-seed", source_session_id="s", source_entity="test",
    )


@pytest.mark.parametrize("dims", [[720, 1280, 3], [4, 720, 1280, 3]], ids=["image", "clip"])
def test_the_plan_primes_then_rolls_out_with_the_seed_hw_off_the_signal_dims(model, dims):
    plan = model.walk_plan("p", ["image"], ["video_frame"], {"image_inputs": [_pointer_info(dims)]}, {})

    assert [step.walk for step in plan] == [PRIME_WALK, ROLLOUT_WALK] and not any(step.bind for step in plan)
    assert all(step.step_metadata == {"seed_hw": (720, 1280)} for step in plan)
    # the seed clip is addressed to the vae_encoder; the actions arrive on the
    # client stream, not as a walk input
    inputs, _, _ = _walk_io(model)
    assert inputs[PRIME_WALK] == (("image_inputs", VAE_ENCODER_NODE),)
    assert inputs[ROLLOUT_WALK] == (("clock", DIT_NODE),)
    with pytest.raises(ValueError, match="required seed clip"):
        model.walk_plan("p", ["tensor"], ["video_frame"], {}, {})


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


def test_request_kwargs_are_the_keys_process_prompt_reads(model):
    # num_steps is read only to refuse it (test above); actions drives the frames
    assert model.request_kwargs() == {"num_steps", "actions"}
