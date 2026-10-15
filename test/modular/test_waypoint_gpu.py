"""GPU gates for the Waypoint port: fullgraph capture, compiled/eager parity,
CUDA-graph replay, rollout isolation, and the BlockMask rebuild cost.

Checkpoint-free -- weights are random and every claim is a self-consistency
one. The geometry is reduced (4 layers, 128 tokens per frame) but structurally
identical to 720P: one global layer at a dilated stride, controller fusion on
``i % 3 == 0``, GQA live, and local/global rings of *different* capacity so a
layer-indexing mistake can't hide behind a uniform buffer.

There is no eager reference mode: both sides of every comparison run with the
``flex_attention_masked`` compile on, because eager ``flex_attention`` ignores
the no-op ``mask_mod`` and reads unwritten ring slots.

Set ``WAYPOINT_GPU_TESTS=0`` to skip; a full run is a few minutes, most of it
``torch.compile``.
"""

import os
import sys
import time
import zlib

import pytest
import torch

sys.path.insert(0, ".")

from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import (
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
    AttnBackend,
)
from mstar.engine.resources.attn.flex import flex_attention_masked, make_block_mask
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.config import (
    KVSpec,
    RingKVConfig,
    RingKVLayerConfig,
    RingKVStep,
)
from mstar.engine.resources.kv.ring import RingKVManager
from mstar.engine.resources.step import StepContext
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.waypoint.components.dit import WaypointDiT
from mstar.model.waypoint.components.layers import NoiseConditioner
from mstar.model.waypoint.config import WaypointConfig, waypoint_1_5_1b_720p

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(
        os.environ.get("WAYPOINT_GPU_TESTS", "1") == "0",
        reason="WAYPOINT_GPU_TESTS=0",
    ),
]

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16

# bf16 carries 8 mantissa bits, so one ulp at magnitude m is 2**-8 * 2**floor(log2 m).
BF16_EPS = 2.0**-8

# Compiled-vs-eager budget, in ulp of a frame's peak magnitude. Measured worst
# case is 2.35; see the parity test for what it is a budget for.
COMPILE_TOL_ULP = 4

# Frames per rollout in the capture gates. Wraps both rings at least twice:
# the local ring holds 8 frames, and the dilated ring's 4 buckets span
# 4 * 2 = 8 frames.
ROLLOUT_FRAMES = 20


def gpu_config(**overrides) -> WaypointConfig:
    """Reduced geometry with 720P's structure. Layer 3 is the global one
    (period 4, offset -1); its ring holds 4 buckets at stride 2 against the
    local layers' 8 frames, so the two capacities differ."""
    base = {
        "n_layers": 4,
        "n_heads": 2,
        "n_kv_heads": 1,
        "d_model": 64,
        "mlp_ratio": 2,
        "channels": 4,
        "tokens_per_frame": 128,
        "height": 8,
        "width": 16,
        "local_window": 8,
        "global_window": 8,
        "global_pinned_dilation": 2,
        "n_buttons": 8,
    }
    return WaypointConfig(**{**base, **overrides})


def _kv_spec(config: WaypointConfig, num_sessions: int) -> KVSpec:
    return KVSpec(
        resource_key="kv",
        nodes={"dit"},
        config=RingKVConfig(
            num_layers=config.n_layers,
            num_kv_heads=config.n_kv_heads,
            head_dim=config.d_head,
            num_qo_heads=config.n_heads,
            tokens_per_frame=config.tokens_per_frame,
            num_sessions=num_sessions,
            layers=tuple(
                RingKVLayerConfig(
                    ring_frames=config.ring_frames(i),
                    ring_buckets=config.ring_buckets(i),
                    pinned_dilation=config.pinned_dilation(i),
                )
                for i in range(config.n_layers)
            ),
        ),
    )


class _DitNode(NodeSubmodule):
    """Binds through the engine's own walk: one call on the node has to reach
    all of the attention layers."""

    def __init__(self, dit: WaypointDiT):
        super().__init__()
        self.dit = dit

    def prepare_inputs(self, *args, **kwargs):
        raise NotImplementedError("binding stand-in; nothing here runs a step")

    def forward(self, *args, **kwargs):
        raise NotImplementedError("binding stand-in; nothing here runs a step")


def build(config: WaypointConfig, *, seed: int = 0, num_sessions: int = 1):
    """The serving build order -- meta, cast, ``to_empty``, retie -- with random
    weights, wired to a real ring and a real flex backend on the GPU.

    Parameters are filled from a CPU generator so two builds at the same seed
    are bit-identical, letting a test compare two independent instances.
    """
    with torch.device("meta"):
        dit = WaypointDiT(config)
    dit.cast_serving_dtypes()
    dit.to_empty(device=DEVICE)
    dit.retie_cond_proj()

    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for param in dit.parameters():
            param.copy_(torch.randn(param.shape, generator=gen, dtype=torch.float32) * 0.05)
        for block in dit.blocks:
            block.attn.v_lamb.fill_(0.25)
    dit.eval()

    spec = _kv_spec(config, num_sessions)
    info = EngineResourceInfo(device=DEVICE, kv_dtype=DTYPE)
    kv = RingKVManager.build(spec, info)
    attn = AttentionManager.build(
        AttentionSpec(
            resource_key="attn",
            nodes={"dit"},
            config=AttentionConfig(kv_cache="kv", backend=AttnBackend.FLEX),
        ),
        EngineResourceInfo(device=DEVICE, kv_dtype=DTYPE, dependencies={"kv": spec}),
    )
    _DitNode(dit).bind_node_resources({"kv": kv, "attn": attn})
    return dit, kv, attn


def controls(config: WaypointConfig):
    mouse = torch.tensor([[[0.25, -0.5]]], dtype=DTYPE, device=DEVICE)
    button = torch.zeros(1, 1, config.n_buttons, dtype=DTYPE, device=DEVICE)
    button[..., 2] = 1.0
    scroll = torch.tensor([[[1.0]]], dtype=DTYPE, device=DEVICE)
    return mouse, button, scroll


def noise_for(config: WaypointConfig, stream: str, frame: int) -> torch.Tensor:
    """This frame's noise, a pure function of ``(stream, frame)`` -- so an
    interleave order cannot change what a rollout is fed, only where it lands.

    ``crc32`` and not ``hash``: str hashing is salted per process, which would
    make a failure here unreproducible from its own seed."""
    gen = torch.Generator(device="cpu").manual_seed(
        zlib.crc32(f"{stream}:{frame}".encode())
    )
    return torch.randn(
        (1, 1, *config.latent_shape), generator=gen, dtype=torch.float32
    ).to(DEVICE, DTYPE)


# ---- resource lifecycle, driven by hand ------------------------------------


def _ctx(rid: str) -> StepContext:
    return StepContext(request_ids=(rid,), graph_walk="rollout", slot=0, capture=False)


def _step(rid: str, frame: int) -> RingKVStep:
    return RingKVStep(frames=((rid, frame),))


def admit_frame(
    kv: RingKVManager,
    rid: str,
    frame: int,
    attn: AttentionManager | None = None,
) -> None:
    """Admit and stage one frame exactly as the engine's resource runner does.

    KV planning writes the world index read by replay and returns the host ring
    facts the dependent attention plan uses to stage its fixed-address mask.
    ``attn=None`` is reserved for the standalone fallback diagnostic.
    """
    ctx = _ctx(rid)
    outcome = kv.admit(_step(rid, frame), ctx)
    assert outcome.ok, f"{rid} refused at frame {frame}: {outcome.reason}"
    ring_plan = kv.plan(_step(rid, frame), ctx)
    if attn is not None:
        ctx.plan_results["kv"] = ring_plan
        attn.plan(AttentionStep(), ctx)


def _batch_ctx(*rids: str) -> StepContext:
    return StepContext(request_ids=rids, graph_walk="rollout", slot=0, capture=False)


def _batch_step(frames: list[tuple[str, int]]) -> RingKVStep:
    return RingKVStep(frames=tuple(frames))


def admit_batch(
    kv: RingKVManager,
    frames: list[tuple[str, int]],
    attn: AttentionManager,
) -> None:
    """`admit_frame`'s multi-row form: one step naming every ``(rid, frame)``
    pair in ``frames``, in the row order the batched forward's inputs use."""
    ctx = _batch_ctx(*(rid for rid, _ in frames))
    step = _batch_step(frames)
    outcome = kv.admit(step, ctx)
    assert outcome.ok, f"batch {frames} refused: {outcome.reason}"
    ring_plan = kv.plan(step, ctx)
    ctx.plan_results["kv"] = ring_plan
    attn.plan(AttentionStep(), ctx)


def scrub(kv: RingKVManager, *rids: str) -> None:
    """Return every world to the pool and zero every ring. The tests share one
    captured graph, so each one starts from a ring that holds nothing."""
    for rid in rids:
        kv.reset_request(rid, free=True)
        kv.remove_request(rid)
    for layer in kv.layers:
        for world in range(layer.num_sessions):
            layer.reset(world)


def world_snapshot(kv: RingKVManager, session_idx: int):
    return [
        (
            layer.kv[:, :, :, slice(*layer.session_span(session_idx))].clone(),
            layer.written[slice(*layer.session_span(session_idx))].clone(),
        )
        for layer in kv.layers
    ]


def assert_worlds_equal(left, right, what: str) -> None:
    for i, ((kv_a, written_a), (kv_b, written_b)) in enumerate(zip(left, right, strict=True)):
        assert torch.equal(written_a, written_b), f"{what}: layer {i} visibility differs"
        assert torch.equal(kv_a, kv_b), f"{what}: layer {i} ring bytes differ"


# ---------------------------------------------------------------------------
# A.1 -- the fullgraph regions
# ---------------------------------------------------------------------------


def test_compile_regions_holds_under_fullgraph():
    """``fullgraph=True`` on both regions, which is what makes capture possible
    at all -- a break here would leave the driver un-capturable.

    ``torch._dynamo.config.capture_scalar_outputs`` is deliberately NOT set:
    this port compiles clean without it, and setting it here would hide a
    future break behind an unbacked symint.
    """
    config = gpu_config()
    dit, kv, attn = build(config)
    mouse, button, scroll = controls(config)
    kv.ingest_request("r")
    admit_frame(kv, "r", 0, attn)

    assert dit.materialize_runtime_tables(DEVICE) is dit
    assert dit.compile_regions() is dit
    assert dit._regions_compiled
    with torch.no_grad():
        out = dit.generate_frame(
            noise_for(config, "warm", 0),
            torch.tensor([0], dtype=torch.int64, device=DEVICE),
            mouse=mouse, button=button, scroll=scroll,
        )
    torch.cuda.synchronize()

    assert out.shape == (1, 1, *config.latent_shape) and out.dtype == DTYPE
    assert bool(torch.isfinite(out.float()).all())
    assert not torch._dynamo.config.capture_scalar_outputs


# ---------------------------------------------------------------------------
# A.2 -- compiled vs eager
# ---------------------------------------------------------------------------


def test_flex_attention_is_bit_exact_in_and_out_of_a_compiled_region():
    """The floor of the A.2 ladder: the attention kernel itself doesn't change
    under tracing, so any compiled/eager frame difference is attributable to
    the pointwise chains around it, not the kernel the ring is read through.
    """
    gen = torch.Generator(device="cpu").manual_seed(7)
    q = torch.randn(1, 2, 128, 32, generator=gen).to(DEVICE, DTYPE)
    k = torch.randn(1, 1, 640, 32, generator=gen).to(DEVICE, DTYPE)
    v = torch.randn(1, 1, 640, 32, generator=gen).to(DEVICE, DTYPE)
    written = torch.zeros(640, dtype=torch.bool, device=DEVICE)
    written[:384] = True

    def attend(q, k, v, written):
        mask = make_block_mask(q.size(-2), k.size(-2), written)
        return flex_attention_masked(q, k, v, block_mask=mask, enable_gqa=True)

    with torch.no_grad():
        outer_eager = attend(q, k, v, written)
        outer_compiled = torch.compile(attend, fullgraph=True, dynamic=False)(q, k, v, written)
    torch.cuda.synchronize()

    assert torch.equal(outer_eager, outer_compiled)


def test_the_gemms_are_bit_exact_compiled_vs_eager():
    """Second rung: ``nn.Linear`` lowers to the same cuBLAS call either way, so
    the drift is in the fused pointwise chains -- ``rms_norm`` into RoPE,
    adaLN into the residual -- where inductor keeps the intermediate in fp32
    across a fusion while eager round-trips it through bf16.
    """
    gen = torch.Generator(device="cpu").manual_seed(11)
    x = torch.randn(1, 128, 64, generator=gen).to(DEVICE, DTYPE)
    weight = torch.randn(64, 64, generator=gen).to(DEVICE, DTYPE)

    def gemm(x, weight):
        return torch.nn.functional.linear(x, weight)

    def normed(x):
        return torch.nn.functional.rms_norm(x, (x.size(-1),)) * 1.25

    with torch.no_grad():
        assert torch.equal(
            gemm(x, weight),
            torch.compile(gemm, fullgraph=True, dynamic=False)(x, weight),
        )
        norm_eager = normed(x)
        norm_compiled = torch.compile(normed, fullgraph=True, dynamic=False)(x)
        # ...and the fused chain is the one that moves, toward fp32 rather than
        # away from it.
        reference = torch.nn.functional.rms_norm(x.float(), (x.size(-1),)) * 1.25
    torch.cuda.synchronize()

    assert not torch.equal(norm_eager, norm_compiled)
    assert (norm_compiled.float() - reference).abs().max() <= (
        norm_eager.float() - reference
    ).abs().max()


def test_compiled_matches_eager_to_four_bf16_ulp_of_peak():
    """NOT bit-exact: inductor's fp32-carrying pointwise fusions leave a fixed
    handful of bf16 quanta of drift. The bound is measured, not chosen --
    worst 2.35 ulp of peak over 120 frame comparisons across six seeds, so
    ``COMPILE_TOL_ULP`` sits at 4 with ~1.7x headroom.

    Run over the full rollout to check the gap doesn't compound even though
    the ring feeds itself: frame 19's deviation is the same size as frame 0's.

    Ring-slot visibility is checked separately since a mismatch there would
    be a slot bug, not a rounding one.
    """
    config = gpu_config()
    frames = ROLLOUT_FRAMES
    mouse, button, scroll = controls(config)

    def rollout(compiled: bool):
        dit, kv, attn = build(config, seed=0)
        dit.materialize_runtime_tables(DEVICE)
        if compiled:
            dit.compile_regions()
        kv.ingest_request("r")
        latents = []
        with torch.no_grad():
            for frame in range(frames):
                admit_frame(kv, "r", frame, attn)
                latents.append(
                    dit.generate_frame(
                        noise_for(config, "a2", frame),
                        torch.tensor([frame], dtype=torch.int64, device=DEVICE),
                        mouse=mouse, button=button, scroll=scroll,
                    ).clone()
                )
                kv.commit(_step("r", frame), _ctx("r"))
        torch.cuda.synchronize()
        return latents, world_snapshot(kv, kv.session_of("r"))

    eager_latents, eager_ring = rollout(False)
    compiled_latents, compiled_ring = rollout(True)

    for frame, (want, got) in enumerate(zip(eager_latents, compiled_latents, strict=True)):
        peak = want.float().abs().max().item()
        deviation = (want.float() - got.float()).abs().max().item()
        assert deviation <= COMPILE_TOL_ULP * BF16_EPS * peak, (
            f"frame {frame}: compiled and eager diverge by {deviation:.3e} on a "
            f"latent peaking at {peak:.3e} -- {deviation / peak / BF16_EPS:.2f} ulp "
            f"of peak, past the {COMPILE_TOL_ULP} the pointwise fusions account for"
        )

    for i, ((_, eager_written), (_, compiled_written)) in enumerate(
        zip(eager_ring, compiled_ring, strict=True)
    ):
        assert torch.equal(eager_written, compiled_written), (
            f"layer {i}: compiling the regions changed which ring slots are visible"
        )


# ---------------------------------------------------------------------------
# A.3 / A.4 / A.5 -- one captured graph, three claims
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def captured():
    """A compiled DiT and one CUDA graph over ``generate_frame``, plus the
    static buffers a replay reads.

    Two worlds, so the interleave gate can use the same graph the single-world
    gates do -- ``session_idx`` is staged by ``plan`` and read at replay, never
    baked. Capture holds a dummy rid and its warmup frames land in the ring;
    every test scrubs on entry.
    """
    config = gpu_config()
    dit, kv, attn = build(config, seed=0, num_sessions=2)
    dit.materialize_runtime_tables(DEVICE)
    dit.compile_regions()
    mouse, button, scroll = controls(config)

    kv.ingest_request("capture")
    admit_frame(kv, "capture", 0, attn)
    static_noise = torch.zeros(1, 1, *config.latent_shape, dtype=DTYPE, device=DEVICE)
    static_frame = torch.zeros(1, dtype=torch.int64, device=DEVICE)

    static_latent = torch.zeros(1, 1, *config.latent_shape, dtype=DTYPE, device=DEVICE)

    # One pool for both graphs, as ``CudaGraphRunner`` does, and rollout first:
    # its five forwards are a superset of prime's one, so the pool is sized once
    # and prime reuses the blocks rollout freed.
    pool = torch.cuda.graphs.graph_pool_handle()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad():
        for _ in range(3):
            dit.generate_frame(
                static_noise, static_frame, mouse=mouse, button=button, scroll=scroll
            )
        for _ in range(3):
            dit.append_frame(
                static_latent, static_frame, mouse=mouse, button=button, scroll=scroll
            )
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool), torch.no_grad():
        static_out = dit.generate_frame(
            static_noise, static_frame, mouse=mouse, button=button, scroll=scroll
        )
    torch.cuda.synchronize()

    prime_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(prime_graph, pool=pool), torch.no_grad():
        prime_out = dit.append_frame(
            static_latent, static_frame, mouse=mouse, button=button, scroll=scroll
        )
    torch.cuda.synchronize()

    # Warmup and capture wrote real frames into the ring and left the dummy rid
    # holding a world. Asserted here rather than in a test because this is the
    # one place the contamination is unambiguous -- every test below scrubs on
    # entry, so by then it is gone.
    with pytest.raises(RuntimeError, match="capture"):
        kv.post_warmup_validate()

    return {
        "config": config, "dit": dit, "kv": kv, "attn": attn, "graph": graph,
        "noise": static_noise, "frame": static_frame, "out": static_out,
        "prime_graph": prime_graph, "latent": static_latent, "prime_out": prime_out,
        "controls": (mouse, button, scroll),
    }


def replay_frame(captured, rid: str, frame: int, stream: str) -> torch.Tensor:
    """One frame through the captured graph: stage the world, refill the
    inputs, replay, read the output back before the next replay overwrites it."""
    admit_frame(captured["kv"], rid, frame, captured["attn"])
    captured["noise"].copy_(noise_for(captured["config"], stream, frame))
    captured["frame"].fill_(frame)
    captured["graph"].replay()
    torch.cuda.synchronize()
    latent = captured["out"].clone()
    captured["kv"].commit(_step(rid, frame), _ctx(rid))
    return latent


def eager_frame(captured, rid: str, frame: int, stream: str) -> torch.Tensor:
    """The same frame through the same compiled regions, uncaptured: the
    control A.3 needs, since replaying must be compared against the code the
    graph was captured from, not a differently-fused build of it."""
    admit_frame(captured["kv"], rid, frame, captured["attn"])
    mouse, button, scroll = captured["controls"]
    with torch.no_grad():
        latent = captured["dit"].generate_frame(
            noise_for(captured["config"], stream, frame),
            torch.tensor([frame], dtype=torch.int64, device=DEVICE),
            mouse=mouse, button=button, scroll=scroll,
        ).clone()
    torch.cuda.synchronize()
    captured["kv"].commit(_step(rid, frame), _ctx(rid))
    return latent


def replay_prime(captured, rid: str, stream: str) -> torch.Tensor:
    """Prime one world through the captured graph. Prime always sits at frame 0.

    The latent stands in for the VAE encoder's output; ``noise_for`` only has to
    be a deterministic function of ``stream`` here, not real pixels.
    """
    admit_frame(captured["kv"], rid, 0, captured["attn"])
    captured["latent"].copy_(noise_for(captured["config"], stream, 0))
    captured["frame"].fill_(0)
    captured["prime_graph"].replay()
    torch.cuda.synchronize()
    latent = captured["prime_out"].clone()
    captured["kv"].commit(_step(rid, 0), _ctx(rid))
    return latent


def eager_prime(captured, rid: str, stream: str) -> torch.Tensor:
    """The same prime through the same compiled ``_cache_pass``, uncaptured --
    the control replay must be compared against, for the reason ``eager_frame``
    gives."""
    admit_frame(captured["kv"], rid, 0, captured["attn"])
    mouse, button, scroll = captured["controls"]
    with torch.no_grad():
        latent = captured["dit"].append_frame(
            noise_for(captured["config"], stream, 0),
            torch.tensor([0], dtype=torch.int64, device=DEVICE),
            mouse=mouse, button=button, scroll=scroll,
        ).clone()
    torch.cuda.synchronize()
    captured["kv"].commit(_step(rid, 0), _ctx(rid))
    return latent


def test_capture_replays_fixed_address_planned_masks(captured):
    """The manual CUDA gate uses the served preplanned-mask path."""
    attn = captured["attn"]
    assert attn.needs_token_visibility is False
    assert len(attn._planned_masks) == 2
    addresses = {
        key: (
            mask.full_kv_num_blocks.data_ptr(),
            mask.full_kv_indices.data_ptr(),
        )
        for key, mask in attn._planned_masks.items()
    }

    scrub(captured["kv"], "capture")
    captured["kv"].ingest_request("mask-address")
    replay_frame(captured, "mask-address", 0, "mask-address")
    assert {
        key: (
            mask.full_kv_num_blocks.data_ptr(),
            mask.full_kv_indices.data_ptr(),
        )
        for key, mask in attn._planned_masks.items()
    } == addresses
    scrub(captured["kv"], "mask-address")


def test_replay_matches_the_uncaptured_regions_over_two_ring_wraps(captured):
    """20 frames of replay against 20 frames of the same compiled code,
    bit-exact on the emitted latents AND on the ring -- a latent check alone
    would pass on a rollout whose history had quietly gone somewhere else.

    Long enough to wrap both rings twice, so a slot that was only ever
    appended to has to be overwritten and read back.
    """
    config = captured["config"]
    kv = captured["kv"]
    assert ROLLOUT_FRAMES >= 2 * config.local_window
    assert ROLLOUT_FRAMES >= 2 * config.ring_buckets(3) * config.pinned_dilation(3)
    assert config.ring_frames(0) != config.ring_frames(3), "the two rings are the same size"

    scrub(kv, "capture")
    kv.post_warmup_validate()

    kv.ingest_request("uncaptured")
    uncaptured = [
        eager_frame(captured, "uncaptured", f, "a3") for f in range(ROLLOUT_FRAMES)
    ]
    uncaptured_ring = world_snapshot(kv, kv.session_of("uncaptured"))
    scrub(kv, "uncaptured")

    kv.ingest_request("replayed")
    replayed = [
        replay_frame(captured, "replayed", f, "a3") for f in range(ROLLOUT_FRAMES)
    ]
    replayed_ring = world_snapshot(kv, kv.session_of("replayed"))
    scrub(kv, "replayed")

    for frame, (want, got) in enumerate(zip(uncaptured, replayed, strict=True)):
        assert torch.equal(want, got), (
            f"frame {frame}: replay diverged from the code it was captured from by "
            f"{(want.float() - got.float()).abs().max().item():.3e}"
        )
    assert_worlds_equal(uncaptured_ring, replayed_ring, "replay vs uncaptured")
    assert not torch.equal(uncaptured[0], uncaptured[ROLLOUT_FRAMES - 1]), (
        "the rollout is static; a graph that replayed frame 0 forever would pass"
    )


def test_a_second_rollout_starts_from_nothing(captured):
    """Two rollouts in one process on the same fixed ring: the second is fed
    identical noise and must produce identical frames, so it saw neither the
    first rollout's history nor the capture warmup's, both still physically
    in the buffer until something zeroes them.

    ``post_warmup_validate`` is asserted to raise on the dirty ring the first
    rollout leaves behind -- a scrub that silently did nothing would make the
    comparison below vacuous.
    """
    kv = captured["kv"]
    frames = 8

    scrub(kv, "capture")
    kv.post_warmup_validate()

    kv.ingest_request("first")
    first = [replay_frame(captured, "first", f, "a4") for f in range(frames)]
    first_ring = world_snapshot(kv, kv.session_of("first"))
    with pytest.raises(RuntimeError):
        kv.post_warmup_validate()
    scrub(kv, "first")
    kv.post_warmup_validate()

    kv.ingest_request("second")
    second = [replay_frame(captured, "second", f, "a4") for f in range(frames)]
    second_ring = world_snapshot(kv, kv.session_of("second"))

    for frame, (want, got) in enumerate(zip(first, second, strict=True)):
        assert torch.equal(want, got), (
            f"frame {frame}: the second rollout differs from the first by "
            f"{(want.float() - got.float()).abs().max().item():.3e}; it inherited "
            "state from the first rollout or from the capture warmup"
        )
    assert_worlds_equal(first_ring, second_ring, "second rollout vs first")
    scrub(kv, "second")


def test_two_worlds_interleaved_match_the_same_rollouts_run_alone(captured):
    """Two rollouts admitted at once and advanced frame by frame in turn, each
    bit-identical to itself run alone.

    Nothing physical separates the worlds -- they share one buffer per layer,
    folded into the token dimension -- so the whole isolation mechanism is the
    visibility row, and a leak is silent. Both go through the *same* captured
    graph, which is also the claim that ``session_idx`` is read at replay
    rather than baked at capture.
    """
    kv = captured["kv"]
    frames = 10
    assert kv.num_sessions == 2

    scrub(kv, "capture")
    alone = {}
    for stream in ("A", "B"):
        rid = f"alone{stream}"
        kv.ingest_request(rid)
        latents = [replay_frame(captured, rid, f, stream) for f in range(frames)]
        alone[stream] = (latents, world_snapshot(kv, kv.session_of(rid)))
        scrub(kv, rid)

    kv.ingest_request("both_a")
    kv.ingest_request("both_b")
    admit_frame(kv, "both_a", 0, captured["attn"])
    admit_frame(kv, "both_b", 0, captured["attn"])
    assert {kv.session_of("both_a"), kv.session_of("both_b")} == {0, 1}

    interleaved = {"A": [], "B": []}
    for frame in range(frames):
        interleaved["A"].append(replay_frame(captured, "both_a", frame, "A"))
        interleaved["B"].append(replay_frame(captured, "both_b", frame, "B"))

    for stream, rid in (("A", "both_a"), ("B", "both_b")):
        want_latents, want_ring = alone[stream]
        for frame, (want, got) in enumerate(
            zip(want_latents, interleaved[stream], strict=True)
        ):
            assert torch.equal(want, got), (
                f"world {stream} frame {frame}: sharing the node changed the rollout by "
                f"{(want.float() - got.float()).abs().max().item():.3e}"
            )
        assert_worlds_equal(want_ring, world_snapshot(kv, kv.session_of(rid)), f"world {stream}")

    assert not torch.equal(interleaved["A"][0], interleaved["B"][0]), (
        "the two rollouts are identical; an isolation leak would be invisible"
    )
    scrub(kv, "both_a", "both_b")


def test_batched_step_matches_the_same_rollouts_run_one_row_at_a_time():
    """B=2 across two worlds, 8 frames each, matches the identical rollouts
    run one row (B=1) at a time.

    A batched GEMM may pick a different cuBLAS kernel at M=2T than at M=T, so
    the B=1-vs-B=2 comparison is bounded in bf16 ulp rather than exact -- but
    two rows of the SAME batched call fed identical inputs share one kernel
    launch and must be bit-exact.
    """
    config = gpu_config()
    frames = 8
    mouse, button, scroll = controls(config)
    dit, kv, attn = build(config, seed=0, num_sessions=2)
    dit.materialize_runtime_tables(DEVICE)

    # ---- B=1, alternating: each world through its own single-row forward.
    for rid in ("w0", "w1"):
        kv.ingest_request(rid)
    solo_latents = {"w0": [], "w1": []}
    with torch.no_grad():
        for frame in range(frames):
            for stream, rid in (("s0", "w0"), ("s1", "w1")):
                admit_frame(kv, rid, frame, attn)
                out = dit.generate_frame(
                    noise_for(config, stream, frame),
                    torch.tensor([frame], dtype=torch.int64, device=DEVICE),
                    mouse=mouse, button=button, scroll=scroll,
                ).clone()
                solo_latents[rid].append(out)
                kv.commit(_step(rid, frame), _ctx(rid))
    torch.cuda.synchronize()
    solo_ring = {rid: world_snapshot(kv, kv.session_of(rid)) for rid in ("w0", "w1")}
    scrub(kv, "w0", "w1")

    # ---- B=2, batched: both worlds' row for frame f in one forward.
    for rid in ("w0", "w1"):
        kv.ingest_request(rid)
    mouse2 = mouse.expand(2, -1, -1).contiguous()
    button2 = button.expand(2, -1, -1).contiguous()
    scroll2 = scroll.expand(2, -1, -1).contiguous()
    batched_latents = {"w0": [], "w1": []}
    with torch.no_grad():
        for frame in range(frames):
            batch_frames = [("w0", frame), ("w1", frame)]
            admit_batch(kv, batch_frames, attn)
            noise = torch.cat(
                [noise_for(config, "s0", frame), noise_for(config, "s1", frame)], dim=0,
            )
            frame_pos = torch.tensor([frame, frame], dtype=torch.int64, device=DEVICE)
            out = dit.generate_frame(
                noise, frame_pos, mouse=mouse2, button=button2, scroll=scroll2,
            ).clone()
            batched_latents["w0"].append(out[0:1])
            batched_latents["w1"].append(out[1:2])
            kv.commit(_batch_step(batch_frames), _batch_ctx("w0", "w1"))
    torch.cuda.synchronize()
    batch_ring = {rid: world_snapshot(kv, kv.session_of(rid)) for rid in ("w0", "w1")}
    scrub(kv, "w0", "w1")

    for rid in ("w0", "w1"):
        for frame, (want, got) in enumerate(
            zip(solo_latents[rid], batched_latents[rid], strict=True)
        ):
            peak = want.float().abs().max().item()
            deviation = (want.float() - got.float()).abs().max().item()
            assert deviation <= COMPILE_TOL_ULP * BF16_EPS * peak, (
                f"world {rid} frame {frame}: batching diverged from the same row run "
                f"alone by {deviation:.3e} ({deviation / peak / BF16_EPS:.2f} ulp of peak)"
            )
        assert_worlds_equal(solo_ring[rid], batch_ring[rid], f"world {rid}, batched vs alone")

    # Two rows of the SAME batched call, fed identical inputs: must be
    # bit-exact, or a row is reading another row's data or batch position.
    for rid in ("id0", "id1"):
        kv.ingest_request(rid)
    identical_frames = [("id0", 0), ("id1", 0)]
    admit_batch(kv, identical_frames, attn)
    same_noise = noise_for(config, "dup", 0).expand(2, -1, -1, -1, -1).contiguous()
    same_frame_pos = torch.tensor([0, 0], dtype=torch.int64, device=DEVICE)
    with torch.no_grad():
        dup_out = dit.generate_frame(
            same_noise, same_frame_pos, mouse=mouse2, button=button2, scroll=scroll2,
        )
    torch.cuda.synchronize()
    assert torch.equal(dup_out[0], dup_out[1]), (
        "identical inputs on two rows of the same batched call produced different "
        "output; a row is reading something batch-position-dependent"
    )
    kv.commit(_batch_step(identical_frames), _batch_ctx("id0", "id1"))
    scrub(kv, "id0", "id1")


# ---------------------------------------------------------------------------
# A.6 -- the prime graph
# ---------------------------------------------------------------------------


def test_prime_replay_matches_the_uncaptured_prime(captured):
    """Replaying prime writes the ring the compiled cache pass writes.

    The ring *is* prime's whole product -- ``append_frame`` returns its input
    untouched -- so a latent comparison alone would pass on a graph that did
    nothing at all.
    """
    kv = captured["kv"]
    scrub(kv, "capture")

    kv.ingest_request("eager_p")
    eager_prime(captured, "eager_p", "P")
    eager_ring = world_snapshot(kv, kv.session_of("eager_p"))
    scrub(kv, "eager_p")

    kv.ingest_request("graph_p")
    replay_prime(captured, "graph_p", "P")
    graph_ring = world_snapshot(kv, kv.session_of("graph_p"))

    assert_worlds_equal(eager_ring, graph_ring, "primed by replay vs uncaptured")
    assert any(written.any() for _, written in graph_ring), (
        "priming made nothing visible; the comparison above is between two "
        "empty rings and would pass on a graph that never ran"
    )
    scrub(kv, "graph_p")

    # A second, different latent must land somewhere else, or the replay is
    # reproducing capture-time state rather than reading its input buffer.
    kv.ingest_request("other_p")
    replay_prime(captured, "other_p", "Q")
    other_ring = world_snapshot(kv, kv.session_of("other_p"))
    assert not torch.equal(graph_ring[0][0], other_ring[0][0]), (
        "two different seed latents primed the same ring bytes"
    )
    scrub(kv, "other_p")


def test_the_prime_graph_returns_its_own_static_input_buffer(captured):
    """Prime's output aliases its input, and that is the contract downstream:
    ``append_frame`` hands the settled latent straight back, so the consumer
    must copy before the next prime stages over it (the engine does, via
    ``copy_``). Pinned here so adding a ``.clone()`` to ``append_frame`` is a
    decision, not an accident.
    """
    assert captured["prime_out"].data_ptr() == captured["latent"].data_ptr()

    staged = noise_for(captured["config"], "R", 0)
    captured["latent"].copy_(staged)
    assert torch.equal(captured["prime_out"], staged), (
        "the output view did not follow its input buffer, so it is a copy made "
        "at capture time and every replay would return stale bytes"
    )


def test_prime_then_rollout_through_both_graphs_matches_eager(captured):
    """The integration claim, and the shared-pool gate.

    Priming through one graph and then rolling out through the other must be
    bit-identical to the same sequence through the uncaptured compiled regions.
    If the two captures aliased each other in the shared pool, the rollout would
    read latents the prime graph had since overwritten.
    """
    kv = captured["kv"]
    frames = 8
    scrub(kv, "capture")

    kv.ingest_request("eager_pr")
    eager_prime(captured, "eager_pr", "S")
    want = [eager_frame(captured, "eager_pr", f, "S") for f in range(1, frames + 1)]
    want_ring = world_snapshot(kv, kv.session_of("eager_pr"))
    scrub(kv, "eager_pr")

    kv.ingest_request("graph_pr")
    replay_prime(captured, "graph_pr", "S")
    got = [replay_frame(captured, "graph_pr", f, "S") for f in range(1, frames + 1)]
    got_ring = world_snapshot(kv, kv.session_of("graph_pr"))

    for frame, (a, b) in enumerate(zip(want, got, strict=True), start=1):
        assert torch.equal(a, b), (
            f"frame {frame}: priming through the graph changed the rollout by "
            f"{(a.float() - b.float()).abs().max().item():.3e}"
        )
    assert_worlds_equal(want_ring, got_ring, "primed rollout, replay vs uncaptured")
    scrub(kv, "graph_pr")

    # Without this, a prime graph that replayed nothing would pass: the rollout
    # would simply start from an empty world in both arms.
    kv.ingest_request("unprimed")
    unprimed = [replay_frame(captured, "unprimed", f, "S") for f in range(1, frames + 1)]
    assert not torch.equal(got[0], unprimed[0]), (
        "the first rolled-out frame is the same with and without priming; the "
        "prime graph wrote nothing the rollout could read"
    )
    scrub(kv, "unprimed")


# ---------------------------------------------------------------------------
# B -- what the BlockMask rebuild costs
# ---------------------------------------------------------------------------


def test_block_mask_rebuild_cost_at_720p(record_property):
    """A measurement, not a bound: ``FlexAttentionManager.attend`` rebuilds
    the mask on every call (24 layers x 5 passes = 120 times per frame).

    Reported for the eager path, where the rebuild is host work; under the
    compiled regions it is traced into the graph instead, so a captured
    replay pays device time and no host time at all. The block-alignment
    ``torch.equal`` is timed separately since it's a full sync and also what
    blocks capture; ``torch.compiler.is_compiling()`` switches it off.
    """
    config = waypoint_1_5_1b_720p()
    kv_len = config.kv_capacity(0)
    rebuilds = config.n_layers * len(config.scheduler_sigmas)
    assert (rebuilds, kv_len) == (120, 8704)

    written = torch.zeros(kv_len, dtype=torch.bool, device=DEVICE)
    frames = written.view(-1, config.tokens_per_frame)
    frames[:9] = True   # nine committed frames of history
    frames[-1] = True   # the scratch tail is always visible

    def measure(fn, repeats: int) -> float:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / repeats * 1e3

    blocks = written.view(-1, 128)
    per_frame = measure(
        lambda: [make_block_mask(config.tokens_per_frame, kv_len, written)
                 for _ in range(rebuilds)],
        repeats=20,
    )
    # The block-alignment assert, alone. It is a device-to-host sync, so it is
    # both the largest single line in the rebuild and the reason a capture of
    # the eager path is impossible.
    sync = measure(lambda: torch.equal(blocks.any(-1), blocks.all(-1)), repeats=200)
    cached = per_frame / rebuilds * config.n_layers

    record_property("block_mask_ms_per_frame", per_frame)
    print(f"\n[Gate B] {rebuilds} eager BlockMask rebuilds per frame: {per_frame:.2f} ms "
          f"({per_frame / rebuilds * 1e3:.0f} us each), q={config.tokens_per_frame} kv={kv_len}"
          f"\n[Gate B] of which the block-alignment sync: {sync * rebuilds:.2f} ms"
          f"\n[Gate B] a (layer, frame) cache would leave {config.n_layers} rebuilds: "
          f"{cached:.2f} ms, saving {per_frame - cached:.2f} ms/frame")

    # The mask itself is what the timing is about, so assert it is the right one:
    # exactly the written blocks, and nothing partial.
    mask = make_block_mask(config.tokens_per_frame, kv_len, written)
    visible = int(written.view(-1, 128).any(-1).sum())
    assert int(mask.full_kv_num_blocks[0, 0, 0]) == visible == 10 * 4
    assert int(mask.kv_num_blocks.sum()) == 0


# ---------------------------------------------------------------------------
# The fp32 island vs a process-wide matmul precision
# ---------------------------------------------------------------------------


def test_the_fp32_island_is_pinned_against_the_engine_matmul_precision():
    """``mstar/engine/__init__.py`` sets ``float32_matmul_precision``
    process-wide, and ``NoiseConditioner`` is a deliberate fp32 island -- so
    the setting decides what "fp32" means there. Pinned here so a change to
    the engine default fails loudly.

    It doesn't bite *today*: the model serves B == N == 1, so the island's
    matmuls are ``[1, 512] @ [512, 8192]`` -- a GEMV, bit-identical under
    every setting. From N >= 2 the setting reaches it, which a future batched
    step would walk into.
    """
    assert torch.get_float32_matmul_precision() == "high"

    config = waypoint_1_5_1b_720p()
    torch.manual_seed(0)
    island = NoiseConditioner(config.d_model).to(DEVICE, torch.float32).eval()
    previous = torch.get_float32_matmul_precision()
    try:
        outputs = {}
        for served in (1, 8):
            sigma = torch.full((1, served), 0.9, dtype=DTYPE, device=DEVICE)
            for precision in ("highest", "high", "medium"):
                torch.set_float32_matmul_precision(precision)
                with torch.no_grad():
                    outputs[served, precision] = island(sigma).clone()
        torch.cuda.synchronize()
    finally:
        torch.set_float32_matmul_precision(previous)

    for precision in ("high", "medium"):
        assert torch.equal(outputs[1, "highest"], outputs[1, precision]), (
            f"the served shape became sensitive to float32_matmul_precision={precision!r}; "
            "the island is no longer a GEMV and the engine default now changes its numbers"
        )
    assert not torch.equal(outputs[8, "highest"], outputs[8, "medium"]), (
        "precondition: float32_matmul_precision is live on this device at N >= 2"
    )
