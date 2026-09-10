"""Component-level contract tests for the Waypoint-1.5 port: the ring geometry
its config implies, OrthoRoPE and the small layers.

The bar is the reference implementation at ``world_engine/src/``, not "the code
does what the code does". Every failure mode this file guards against is silent:
a wrong ring slot, a re-derived bucket count and a permuted controller concat all
produce plausible video and raise nothing. So the assertions are exact wherever
the reference is exact (bitwise for the compaction A/B, for the RoPE angle
tables, for the ring bytes after a frozen pass) and never widened to accommodate
the implementation.

**The ring and the kernel are engine resources now**, imported below from
``mstar.engine.resources``. Their own contracts -- ownership, the capture
lifecycle, the ``visible`` aliasing hazard, the eager-``flex_attention`` trap --
are pinned next door in ``test_ring_kv_resource.py`` and
``test_flex_attention_resource.py``. What stays here is the half those files
cannot see: that *Waypoint's config* produces the geometry the checkpoint was
trained against, and that the ring arithmetic matches the reference's tables.

CPU-only and checkpoint-free by construction. Numeric work runs on a reduced but
structurally identical config (4 layers / 128 tokens per frame / d_head 32, one
global layer at stride 8); the real 720P config is used only where the assertion
is about geometry rather than activations. ``torch.compile(flex_attention)``
works on CPU in torch 2.9, which is what makes the compaction A/B runnable
without a GPU -- it costs a few seconds of inductor time on first use.
"""

import dataclasses
import math
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    flex_attention,
    noop_mask,
)

from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttnBackend
from mstar.engine.resources.attn.flex import flex_attention_masked, make_block_mask
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.config import (
    KVSpec,
    RingKVConfig,
    RingKVLayerConfig,
    RingKVStep,
)
from mstar.engine.resources.kv.ring import LayerRingCache, RingKVManager
from mstar.engine.resources.step import StepContext
from mstar.model.waypoint.components.layers import (
    MLP,
    AdaLN,
    ControllerInputEmbedding,
    MLPFusion,
    NoiseConditioner,
    ada_gate,
    ada_rmsnorm,
    rms_norm,
)
from mstar.model.waypoint.components.rope import OrthoRoPEAngles, apply_ortho_rope
from mstar.model.waypoint.config import WaypointConfig, waypoint_1_5_1b_720p
from mstar.model.waypoint.ring_geometry import ring_memory_bytes

BLOCK = _DEFAULT_SPARSE_BLOCK_SIZE  # 128
TPF = 128  # tokens per frame in the reduced config == one sparse block


def reduced_config(**overrides) -> WaypointConfig:
    """A 4-layer / 128-token-per-frame Waypoint whose *structure* is the 720P
    model's: one global layer (index 3) at stride 8, one non-global period, GQA
    live at 2 query heads over 1 KV head, controller fusion on ``i % 3 == 0``.

    Only the sizes shrink. ``global_window // global_pinned_dilation == 4``
    addressable slots against a 32-frame reference allocation keeps the 8x
    over-allocation the compaction deviation is about, while letting a test wrap the
    global ring in 32 frames instead of 128.
    """
    base = {
        "n_layers": 4,
        "n_heads": 2,
        "n_kv_heads": 1,
        "d_model": 64,
        "mlp_ratio": 2,
        "channels": 4,
        "tokens_per_frame": TPF,
        "height": 8,
        "width": 16,
        "local_window": 4,
        "global_window": 32,
        "global_pinned_dilation": 8,
        "n_buttons": 8,
    }
    return WaypointConfig(**{**base, **overrides})


def frame_kv(value: float, *, tokens: int = TPF, d_head: int = 8) -> torch.Tensor:
    """``[2, B, H_kv, tokens, D]`` of a single constant, so a ring slot's
    contents identify the frame that wrote it."""
    return torch.full((2, 1, 1, tokens, d_head), float(value))


def ring_slot_values(cache: LayerRingCache) -> list[float]:
    """The K-side value parked in each ring slot (scratch excluded)."""
    return [cache.kv[0, 0, 0, s * cache.tokens_per_frame, 0].item() for s in range(cache.ring_frames)]


def visible_blocks(block_mask) -> set[int]:
    """The KV blocks the mask actually makes visible: ``full_kv_indices``
    truncated to ``full_kv_num_blocks``, which is all the compiled kernel reads.
    """
    n = int(block_mask.full_kv_num_blocks[0, 0, 0])
    return set(block_mask.full_kv_indices[0, 0, 0, :n].tolist())


def ring_kv_spec(config: WaypointConfig, *, num_worlds: int = 1) -> KVSpec:
    """The ``RingKVConfig`` a ``WaypointConfig``'s geometry implies -- which is
    the bridge under test in most of section 3.

    Hand-rolled because the model does not declare its specs yet; when
    ``WaypointModel`` grows a ``get_node_resources`` this becomes a call to it.
    Every field is read off the config rather than restated, so a geometry
    change cannot leave these tests measuring a ring the model no longer asks
    for.
    """
    return KVSpec(
        resource_key="kv",
        nodes={"dit"},
        config=RingKVConfig(
            num_layers=config.n_layers,
            num_kv_heads=config.n_kv_heads,
            head_dim=config.d_head,
            num_qo_heads=config.n_heads,
            tokens_per_frame=config.tokens_per_frame,
            num_worlds=num_worlds,
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


def ring_manager(config: WaypointConfig, *, num_worlds: int = 1) -> RingKVManager:
    """``config``'s rings, allocated. Through ``build(spec, info)`` and not the
    constructor: the spec is what picks ``RingKVManager`` over the paged one."""
    return RingKVManager.build(
        ring_kv_spec(config, num_worlds=num_worlds),
        EngineResourceInfo(device=torch.device("cpu"), kv_dtype=torch.float32),
    )


def _dit_ctx(*rids: str) -> StepContext:
    return StepContext(request_ids=tuple(rids), graph_walk="rollout", slot=0, capture=False)


def waypoint_resources(config: WaypointConfig):
    """The ring and the kernel a Waypoint deployment would get, built through
    the real spec-time factories rather than the constructors: the spec is also
    what cross-checks the flex backend against a ring config."""
    kv_spec = ring_kv_spec(config)
    cpu = torch.device("cpu")
    kv = ring_manager(config)
    attn = AttentionManager.build(
        AttentionSpec(
            resource_key="attn",
            nodes={"dit"},
            config=AttentionConfig(kv_cache="kv", backend=AttnBackend.FLEX),
        ),
        EngineResourceInfo(device=cpu, kv_dtype=torch.float32, dependencies={"kv": kv_spec}),
    )
    return kv, attn


# ---------------------------------------------------------------------------
# 1. The BlockMask's shape
#
# The trap itself -- eager `flex_attention` ignoring a no-op `mask_mod` and
# blending every unwritten ring slot in -- is pinned at its owner in
# `test_flex_attention_resource.py`, along with the compiled-path regression
# guard and `make_block_mask`'s alignment checks. What stays here is the half
# those numeric tests cannot establish about themselves: the mask's structure,
# and the all-visible control that makes their divergence attributable to the
# mask rather than to the two kernels merely computing softmax differently.
# ---------------------------------------------------------------------------


def test_block_mask_is_full_blocks_only_and_carries_a_noop_mask_mod():
    """The whole trap follows from this: visibility is in the index lists and
    nowhere else, so anything that re-derives the mask from ``mask_mod`` sees
    "everything visible".

    Note the exact shape of the fact: ``make_block_mask`` passes
    ``mask_mod=None``, and ``BlockMask.from_kv_blocks`` substitutes
    ``flex_attention.noop_mask``. The hazard is often stated as the BlockMask
    "carrying ``mask_mod=None``"; what it carries is the noop, which is the
    same hazard.
    """
    written = torch.zeros(5 * BLOCK, dtype=torch.bool)
    written[0 * BLOCK : 1 * BLOCK] = True  # one committed frame
    written[4 * BLOCK :] = True  # the permanently visible scratch tail

    bm = make_block_mask(TPF, written.numel(), written)

    assert bm.mask_mod is noop_mask, "a non-noop mask_mod would change the trap's shape"
    assert bm.seq_lengths == (TPF, written.numel())
    # Zero partial blocks: "any token written" and "all tokens written" coincide
    # because writes are whole frames.
    assert int(bm.kv_num_blocks.sum()) == 0
    assert visible_blocks(bm) == {0, 4}
    # Query-uniform: one row, broadcast over the query blocks.
    assert bm.full_kv_num_blocks.shape == (1, 1, TPF // BLOCK)


def test_a_fully_visible_ring_makes_eager_and_compiled_agree():
    """The control for the eager-flex trap, and the reason the divergence next
    door is a diagnosis rather than an observation.

    ``test_eager_flex_attention_does_not_honour_the_block_mask`` shows the two
    kernels disagreeing on a partly-hidden row. On its own that is also what
    two kernels with different softmax numerics would look like. Take the mask
    out -- same q/k/v, every block visible -- and they agree to ~1e-07, which
    leaves the mask as the only thing the disagreement can be attributed to.
    Delete this and the ~1e-01 next door stops meaning "eager ignored the mask".

    The all-visible row has to be built by hand: a live Waypoint ring never
    emits one. The slot the current frame is about to overwrite is always
    hidden (see ``test_mask_hides_the_slot_this_frame_is_about_to_overwrite``),
    so the steady state is capacity minus exactly one block, forever.
    """
    config = reduced_config()
    capacity = config.kv_capacity(0)
    visible = torch.ones(capacity, dtype=torch.bool)
    bm = make_block_mask(TPF, capacity, visible)

    gen = torch.Generator().manual_seed(0xC0FFEE)
    q = torch.randn(1, config.n_heads, TPF, config.d_head, generator=gen)
    k = torch.randn(1, config.n_kv_heads, capacity, config.d_head, generator=gen)
    v = torch.randn(1, config.n_kv_heads, capacity, config.d_head, generator=gen)
    # Every slot holds data here, unlike the trap's fixture: with nothing masked
    # off there is no zero slot for eager to blend in, which is the point.
    kx = k.repeat_interleave(config.n_heads // config.n_kv_heads, dim=1)
    vx = v.repeat_interleave(config.n_heads // config.n_kv_heads, dim=1)
    dense = torch.softmax((q @ kx.transpose(-1, -2)) / config.d_head**0.5, dim=-1) @ vx

    compiled = flex_attention_masked(q, k, v, block_mask=bm, enable_gqa=True)
    eager = flex_attention(q, k, v, block_mask=bm, enable_gqa=True)

    compiled_err = (compiled - dense).abs().max().item()
    eager_err = (eager - dense).abs().max().item()
    print(f"[full_ring] compiled={compiled_err:.3e}  eager={eager_err:.3e}")

    assert compiled_err < 1e-5, f"the compiled path diverged with nothing masked ({compiled_err:.3e})"
    assert eager_err < 1e-5, (
        f"eager disagreed with dense on a fully visible row ({eager_err:.3e}); the eager/compiled "
        "gap is then not purely the mask, and that diagnosis needs re-deriving"
    )


# ---------------------------------------------------------------------------
# 2. Ring rotation and the upsert algorithm
# ---------------------------------------------------------------------------


def make_cache(*, ring_frames: int, ring_buckets: int, dilation: int) -> LayerRingCache:
    """One world, because this section is about the ring *algorithm* — which
    slot a frame lands in, which slot it hides — and that is per world and
    identical at any ``num_worlds``. The folded layout and its isolation are
    pinned where they belong, in ``test_ring_kv_resource.py``; driving them
    again here would only make these tests slower to read."""
    return LayerRingCache(
        num_worlds=1,
        n_kv_heads=1,
        ring_frames=ring_frames,
        ring_buckets=ring_buckets,
        d_head=8,
        tokens_per_frame=TPF,
        pinned_dilation=dilation,
        dtype=torch.float32,
        device="cpu",
    )


def upsert(cache: LayerRingCache, kv, frame_pos, *, commit: bool, world: int = 0):
    """``LayerRingCache.upsert`` with the world index spelled out.

    Not a default on ``upsert`` itself, deliberately. ``world_idx`` is a ``[1]``
    int64 *device* tensor on the forward path and never a Python int — a host
    int is folded into the graph at capture and every replay then serves the
    capture-time world, silently. A default argument is exactly how a caller
    ends up not thinking about which world it writes, so the cache takes it
    positionally and this helper is the only place the zero is written down.
    """
    return cache.upsert(kv, frame_pos, commit, torch.tensor([world], dtype=torch.int64))


@pytest.mark.parametrize(
    ("kind", "dilation", "frames", "expected_slots"),
    [
        # Local layers cycle slots 0..15 with every frame.
        ("local", 1, list(range(20)), [16, 17, 18, 19, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        # Global layers commit on frames 0, 8, 16, ... into slots 0, 1, 2, ...
        ("global", 8, list(range(0, 136)), [128, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]),
    ],
)
def test_ring_slot_rotation(kind, dilation, frames, expected_slots):
    """Global commits land on frames 0, 8, 16, ... in slots
    0, 1, 2, ...; local slots cycle 0..15. The slot holds the frame index that
    last wrote it, so the expected list is the whole history at once."""
    cache = make_cache(ring_frames=16, ring_buckets=16, dilation=dilation)
    for f in frames:
        upsert(cache, frame_kv(f), torch.tensor(f, dtype=torch.int64), commit=True)
    assert ring_slot_values(cache) == [float(v) for v in expected_slots], kind
    assert bool(cache.written[: cache.ring_len].all())


def test_frozen_passes_leave_the_ring_byte_identical():
    """The 4 denoise passes of the 4+1 structure. A cache that mutated on them
    would corrupt the world state permanently, and nothing would raise."""
    cache = make_cache(ring_frames=4, ring_buckets=4, dilation=1)
    for f in range(4):
        upsert(cache, frame_kv(f), torch.tensor(f, dtype=torch.int64), commit=True)

    ring_before = cache.kv[:, :, :, : cache.ring_len].clone()
    written_before = cache.written.clone()
    scratch_before = cache.kv[:, :, :, cache.ring_len :].clone()

    for pass_idx in range(4):  # the four Euler steps, each a different noisy x
        upsert(cache, frame_kv(100 + pass_idx), torch.tensor(4, dtype=torch.int64), commit=False)

    assert torch.equal(cache.kv[:, :, :, : cache.ring_len], ring_before), (
        "a frozen pass wrote the ring; that is amnesia, not a cache miss"
    )
    assert torch.equal(cache.written, written_before)
    # ...but the scratch write is unconditional: it is how the frame being
    # denoised attends to itself between Euler steps.
    assert not torch.equal(cache.kv[:, :, :, cache.ring_len :], scratch_before)
    assert cache.kv[0, 0, 0, cache.ring_len, 0].item() == 103.0


def test_mask_hides_the_slot_this_frame_is_about_to_overwrite():
    """And it applies on frozen passes too, so all five passes of a frame see
    byte-identical KV.

    Asserted through ``make_block_mask`` rather than on the ``visible`` row
    directly: the row is what ``upsert`` returns now, but what the kernel reads
    is the block list built from it, and this is the only place the two are
    checked to agree over a whole 4+1 frame."""
    cache = make_cache(ring_frames=4, ring_buckets=4, dilation=1)
    for f in range(4):
        upsert(cache, frame_kv(f), torch.tensor(f, dtype=torch.int64), commit=True)
    assert set(range(5)) == visible_blocks(
        make_block_mask(TPF, cache.capacity, cache.written)
    ), "precondition: the whole ring plus scratch is written"

    fp = torch.tensor(4, dtype=torch.int64)  # slot 0 is about to be reused
    for commit in (False, False, False, False, True):
        _, _, visible = upsert(cache, frame_kv(4), fp, commit=commit)
        assert visible_blocks(make_block_mask(TPF, cache.capacity, visible)) == {1, 2, 3, 4}, (
            "frame 4 can see the stale frame 0 sitting in the slot it is replacing"
        )


def test_global_layer_commits_nothing_on_non_dilation_frames():
    """``torch.where(write_step, ring_idx, current_idx)``
    redirects the commit onto the scratch slot it just wrote."""
    cache = make_cache(ring_frames=4, ring_buckets=4, dilation=8)
    upsert(cache, frame_kv(0), torch.tensor(0, dtype=torch.int64), commit=True)
    ring_before = cache.kv[:, :, :, : cache.ring_len].clone()
    written_before = cache.written.clone()

    for f in range(1, 8):  # the 7 non-committing frames of every 8
        upsert(cache, frame_kv(f), torch.tensor(f, dtype=torch.int64), commit=True)

    assert torch.equal(cache.kv[:, :, :, : cache.ring_len], ring_before)
    assert torch.equal(cache.written, written_before)
    assert cache.kv[0, 0, 0, cache.ring_len, 0].item() == 7.0  # scratch has the latest

    upsert(cache, frame_kv(8), torch.tensor(8, dtype=torch.int64), commit=True)
    assert ring_slot_values(cache)[:2] == [0.0, 8.0]


def floor_bucket_upsert(cache: LayerRingCache, kv, frame_pos, commit: bool):
    """``LayerRingCache.upsert`` with the round-up dropped: ``bucket = f // d``
    instead of ``(f + d - 1) // d``. Everything else is statement-for-statement
    the same. Used only to A/B the round-up."""
    tokens = cache.tokens_per_frame
    world_idx = torch.tensor([0], dtype=torch.int64)
    world_base = world_idx * cache.capacity
    slot = (frame_pos // cache.pinned_dilation) % cache.ring_buckets
    ring_idx = cache.frame_offsets + slot * tokens + world_base
    current_idx = cache._current_base + world_base

    cache.kv.index_copy_(3, current_idx, kv)

    write_step = frame_pos.remainder(cache.pinned_dilation) == 0
    mask_written = torch.empty_like(cache.written)
    mask_written.copy_(cache.written)
    mask_written &= cache._world_of_slot == world_idx
    mask_written[ring_idx] = mask_written[ring_idx] & ~write_step

    if commit:
        dst = torch.where(write_step, ring_idx, current_idx)
        cache.kv.index_copy_(3, dst, kv)
        cache.written.index_fill_(0, dst, True)
    return mask_written


@pytest.mark.parametrize("dilation", [1, 8])
def test_bucket_round_up_is_faithful_but_currently_unobservable(dilation):
    """Flooring instead of rounding up is said to "rotate the entire history by
    one slot". **That consequence does not hold** for any geometry this
    checkpoint uses, and this test pins the real behaviour rather than the
    claim.

    ``ceil`` and ``floor`` agree on every committing frame
    (``(8j + 7) // 8 == 8j // 8 == j``) and at ``dilation == 1`` they are equal
    outright. They differ only where ``write_step`` is False -- and there
    ``ring_idx`` feeds nothing but ``mask_written[ring_idx] &= ~write_step``,
    which is the identity, and ``torch.where(write_step, ring_idx, current_idx)``
    picks the scratch index. So the ``+ dilation - 1`` is faithfully ported and
    harmless, but it is not load-bearing.
    """
    ceil_cache = make_cache(ring_frames=4, ring_buckets=4, dilation=dilation)
    floor_cache = make_cache(ring_frames=4, ring_buckets=4, dilation=dilation)

    for f in range(24):
        fp = torch.tensor(f, dtype=torch.int64)
        for pass_idx in range(5):
            commit = pass_idx == 4
            kv = frame_kv(f * 10 + pass_idx)
            _, _, ceil_visible = upsert(ceil_cache, kv, fp, commit=commit)
            floor_visible = floor_bucket_upsert(floor_cache, kv, fp, commit)
            # Per token, not per block: the rows are what the mask is built
            # from, so equal rows is the stronger statement of the two.
            assert torch.equal(ceil_visible, floor_visible), f"visibility differs at frame {f}"

    assert torch.equal(ceil_cache.kv, floor_cache.kv)
    assert torch.equal(ceil_cache.written, floor_cache.written)


def test_upsert_rejects_a_wrong_shaped_frame_or_clock():
    cache = make_cache(ring_frames=4, ring_buckets=4, dilation=1)
    fp = torch.tensor(0, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="exactly one frame per upsert"):
        upsert(cache, frame_kv(0, tokens=TPF // 2), fp, commit=True)
    with pytest.raises(RuntimeError, match=r"frame_pos must be a \[\] int64 tensor"):
        upsert(cache, frame_kv(0), torch.tensor([0], dtype=torch.int64), commit=True)
    with pytest.raises(RuntimeError, match=r"frame_pos must be a \[\] int64 tensor"):
        upsert(cache, frame_kv(0), torch.tensor(0, dtype=torch.int32), commit=True)


def test_reset_restores_a_fresh_ring():
    cache = make_cache(ring_frames=4, ring_buckets=4, dilation=1)
    for f in range(4):
        upsert(cache, frame_kv(f + 1), torch.tensor(f, dtype=torch.int64), commit=True)
    cache.reset(0)
    assert not bool(cache.kv.any())
    # The scratch tail stays permanently visible -- masking it removes
    # self-attention.
    assert not bool(cache.written[: cache.ring_len].any())
    assert bool(cache.written[cache.ring_len :].all())


def test_ring_state_is_a_deep_copy_and_is_specific_to_the_compaction_setting():
    """The Waypoint half of the state round trip: a state saved from a compacted
    deployment must not load into a ``full_global_ring`` one.

    The geometries differ only on the global layers (5 frames vs 33 here), so
    every local layer would copy cleanly and only layer 3 would fail -- and if
    the guard were a bare ``copy_`` instead of a shape check, a run where the
    dims happened to broadcast would restore a silently replicated frame. The
    generic clone/copy_/layer-count guarantees are pinned in
    ``test_ring_kv_resource.py``; what is here is that the compaction flag is
    part of a state's identity.
    """
    config = reduced_config()
    kv = ring_manager(config)
    kv.ingest_request("a")
    assert kv.admit(RingKVStep(frames=(("a", 0),)), _dit_ctx("a")).ok
    gen = torch.Generator().manual_seed(3)
    for layer in range(config.n_layers):
        frame = torch.randn(1, 1, TPF, config.d_head, generator=gen)
        kv.upsert(frame, frame, layer, torch.tensor(0, dtype=torch.int64), commit=True)

    state = kv.get_state("a")
    snapshot = [t.clone() for t, _ in state["layers"]]
    for layer in kv.layers:
        layer.reset(kv.world_of("a"))
    assert not any(layer.kv.any() for layer in kv.layers)
    # get_state must clone: the reset above must not have reached the snapshot.
    assert all(torch.equal(a, b) for a, b in zip((t for t, _ in state["layers"]), snapshot, strict=True))

    kv.load_state("a", state)
    assert all(torch.equal(layer.kv, t) for layer, (t, _) in zip(kv.layers, state["layers"], strict=True))

    other = ring_manager(dataclasses.replace(config, full_global_ring=True))
    other.ingest_request("a")
    assert other.admit(RingKVStep(frames=(("a", 0),)), _dit_ctx("a")).ok
    with pytest.raises(ValueError, match="state shape"):
        other.load_state("a", state)


# ---------------------------------------------------------------------------
# 3. The compaction deviation and the num_buckets trap
# ---------------------------------------------------------------------------


def test_720p_ring_geometry_matches_the_contract_table():
    """The "addressable slots" and "ring frames allocated" columns are
    independent, and the compaction deviation turns on keeping them so."""
    config = waypoint_1_5_1b_720p()
    assert sorted(config.global_layers) == [3, 7, 11, 15, 19, 23]

    for i in range(config.n_layers):
        assert config.ring_buckets(i) == 16  # every row of the table
        if config.is_global_layer(i):
            assert (config.ring_frames(i), config.pinned_dilation(i)) == (16, 8)
        else:
            assert (config.ring_frames(i), config.pinned_dilation(i)) == (16, 1)
        assert config.kv_capacity(i) == 17 * 512
        assert config.kv_capacity(i) % BLOCK == 0

    full = dataclasses.replace(config, full_global_ring=True)
    assert full.ring_frames(3) == 128 and full.ring_buckets(3) == 16
    assert full.ring_frames(0) == 16  # local layers are untouched by the flag

    compacted_bytes = sum(ring_memory_bytes(config))
    full_bytes = sum(ring_memory_bytes(full))
    assert compacted_bytes == 816 * 2**20  # 816 MiB
    assert full_bytes - compacted_bytes == 1_409_286_144  # ...saving 1.3125 GiB


def test_ring_buckets_is_an_input_not_a_derivation():
    """The trap. The reference computes
    ``num_buckets = (L // tpf) // dilation``. Against the compacted ring that
    yields 2, not 16 -- a global layer would retain 2 frames instead of 16, with
    no shape error and no exception."""
    config = waypoint_1_5_1b_720p()
    global_layer = 3
    reference_derivation = config.ring_frames(global_layer) // config.pinned_dilation(global_layer)
    assert reference_derivation == 2, "the compacted buffer no longer encodes the bucket count"
    assert config.ring_buckets(global_layer) == 16, (
        "ring_buckets must come from global_window // global_pinned_dilation, "
        "never from ring_frames"
    )
    # And it is genuinely independent of the allocation knob.
    full = dataclasses.replace(config, full_global_ring=True)
    assert full.ring_buckets(global_layer) == config.ring_buckets(global_layer) == 16


def test_compacted_global_ring_addresses_all_sixteen_slots():
    """The behavioural half of the trap: a cache that re-derived its bucket
    count from ``ring_len`` would collide frames 0 and 16 into slot 0 and leave
    14 slots forever empty. Sixteen distinct frames, sixteen distinct slots."""
    cache = make_cache(ring_frames=16, ring_buckets=16, dilation=8)
    assert (cache.ring_len // cache.tokens_per_frame) // cache.pinned_dilation == 2  # the wrong answer
    assert cache.ring_buckets == 16

    for j in range(16):
        f = 8 * j
        upsert(cache, frame_kv(f), torch.tensor(f, dtype=torch.int64), commit=True)

    assert ring_slot_values(cache) == [float(8 * j) for j in range(16)]
    assert bool(cache.written[: cache.ring_len].all()), "a 2-bucket ring would leave 14 slots unwritten"

    with pytest.raises(ValueError, match="ring_buckets"):
        make_cache(ring_frames=4, ring_buckets=8, dilation=8)


def drive_ring(config: WaypointConfig, n_frames: int, seed: int = 7) -> list[torch.Tensor]:
    """Run ``n_frames`` of the real 4+1 pass structure through ``config``'s
    resources and collect every attention output. The K/V/Q streams are drawn
    from a seeded generator so two geometries see byte-identical inputs."""
    kv, attn = waypoint_resources(config)
    gen = torch.Generator().manual_seed(seed)
    outputs = []
    for f in range(n_frames):
        fp = torch.tensor(f, dtype=torch.int64)
        for pass_idx in range(5):
            for layer in range(config.n_layers):
                k = torch.randn(1, 1, TPF, config.d_head, generator=gen)
                v = torch.randn(1, 1, TPF, config.d_head, generator=gen)
                q = torch.randn(1, 2, TPF, config.d_head, generator=gen)
                k_all, v_all, visible = kv.upsert(
                    k, v, layer, fp, commit=pass_idx == 4
                )
                outputs.append(attn.attend(q, k_all, v_all, visible, enable_gqa=True))
    return outputs


def test_compacted_and_full_global_rings_are_bitwise_identical():
    """``from_kv_blocks`` derives the visited list
    from a *stable* descending argsort truncated to the visited count, so
    dropping never-written blocks changes neither which blocks are attended nor
    the order they accumulate in. Bit-equality is therefore the correct bar and
    an ``allclose`` here would be hiding a real difference.

    36 frames wraps the global ring's 4 addressable slots (stride 8) more than
    once and the local rings nine times over.
    """
    compacted = reduced_config()
    full = dataclasses.replace(compacted, full_global_ring=True)
    assert compacted.ring_frames(3) == 4 and full.ring_frames(3) == 32
    assert compacted.kv_capacity(3) == 5 * TPF and full.kv_capacity(3) == 33 * TPF

    a = drive_ring(compacted, 36)
    b = drive_ring(full, 36)
    assert len(a) == len(b) == 36 * 5 * compacted.n_layers

    mismatched = [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if not torch.equal(x, y)]
    assert not mismatched, (
        f"{len(mismatched)}/{len(a)} attention outputs differ between the compacted and the "
        f"reference global ring; first at call {mismatched[0]}"
    )


# ---------------------------------------------------------------------------
# 4. OrthoRoPE
# ---------------------------------------------------------------------------


def reference_angles(config: WaypointConfig, x_pos, y_pos, t_pos):
    """The reference's own angle construction, transcribed from
    ``world_engine/src/model/attn.py::OrthoRoPEAngles``."""
    d_head = config.d_head
    d_xy, d_t = d_head // 8, d_head // 4
    max_freq = min(config.height, config.width) * float(config.rope_nyquist_frac)
    n = (d_xy + 1) // 2
    xy = (torch.linspace(1.0, max_freq / 2, n, dtype=torch.float32) * math.pi).repeat_interleave(2)[:d_xy]
    theta = float(config.rope_theta)
    inv_t = (1.0 / (theta ** (torch.arange(0, d_t, 2, dtype=torch.float32) / d_t))).repeat_interleave(2)

    x = (2.0 * x_pos.float() + 1.0) / config.width - 1.0
    y = (2.0 * y_pos.float() + 1.0) / config.height - 1.0
    t = t_pos.float()
    freqs = torch.cat((x.unsqueeze(-1) * xy, y.unsqueeze(-1) * xy, t.unsqueeze(-1) * inv_t), dim=-1)
    return freqs.cos()[:, None], freqs.sin()[:, None]


def grid_positions(config: WaypointConfig, frame: int):
    idx = torch.arange(config.tokens_per_frame)
    y = idx.div(config.width, rounding_mode="floor")[None]
    x = idx.remainder(config.width)[None]
    t = torch.full((1, config.tokens_per_frame), frame * config.ts_mult, dtype=torch.long)
    return x, y, t


def test_ortho_rope_angles_match_the_reference_construction_bitwise():
    config = waypoint_1_5_1b_720p()
    module = OrthoRoPEAngles(config)
    x_pos, y_pos, t_pos = grid_positions(config, frame=5)

    cos, sin = module(x_pos=x_pos, y_pos=y_pos, t_pos=t_pos)
    ref_cos, ref_sin = reference_angles(config, x_pos, y_pos, t_pos)

    assert cos.shape == sin.shape == (1, 1, config.tokens_per_frame, config.d_head // 2)
    assert torch.equal(cos, ref_cos) and torch.equal(sin, ref_sin)


def test_the_bands_count_rotation_pairs_and_cover_every_head_dim():
    """``d_xy = d_head // 8`` and ``d_t = d_head // 4`` count rotation PAIRS.
    8 + 8 + 16 = 32 pairs = 64 dims -- nothing is unrotated. (An earlier
    reading of this had the top half untouched.)"""
    config = waypoint_1_5_1b_720p()
    d_head = config.d_head
    d_xy, d_t = d_head // 8, d_head // 4
    assert (d_head, d_xy, d_t) == (64, 8, 16)
    assert d_xy + d_xy + d_t == d_head // 2 == 32

    cos, sin = OrthoRoPEAngles(config)(*grid_positions(config, frame=5))
    assert cos.shape[-1] == 32
    # Every band actually rotates at frame 5: no band is a no-op that a reader
    # could mistake for "unrotated dims".
    for name, lo, hi in (("x", 0, d_xy), ("y", d_xy, 2 * d_xy), ("t", 2 * d_xy, 32)):
        assert sin[..., lo:hi].abs().max().item() > 1e-3, f"{name} band has no rotation"


@pytest.mark.parametrize(
    ("axis", "pair_lo", "pair_hi", "dim_lo", "dim_hi"),
    [("x", 0, 8, 0, 16), ("y", 8, 16, 16, 32), ("t", 16, 32, 32, 64)],
)
def test_axis_band_ownership(axis, pair_lo, pair_hi, dim_lo, dim_hi):
    """x owns head dims 0-15, y 16-31, t 32-63. Pair ``p``
    consumes input dims ``2p`` and ``2p+1``, so the pair band and the dim band
    are the same statement twice."""
    config = waypoint_1_5_1b_720p()
    module = OrthoRoPEAngles(config)
    x_pos, y_pos, t_pos = grid_positions(config, frame=3)
    shifted = {
        "x": (x_pos.roll(1, dims=1), y_pos, t_pos),
        "y": (x_pos, y_pos.roll(config.width, dims=1), t_pos),
        "t": (x_pos, y_pos, t_pos + 1),
    }[axis]

    cos_a, sin_a = module(x_pos=x_pos, y_pos=y_pos, t_pos=t_pos)
    cos_b, sin_b = module(x_pos=shifted[0], y_pos=shifted[1], t_pos=shifted[2])

    changed = ((cos_a != cos_b) | (sin_a != sin_b)).flatten(0, 2).any(dim=0)
    assert bool(changed[pair_lo:pair_hi].any()), f"moving {axis} changed no angle in its own band"
    untouched = torch.cat((changed[:pair_lo], changed[pair_hi:]))
    assert not bool(untouched.any()), f"moving {axis} leaked into another axis's band"

    # ...and the same in input-dim terms: zeroing the band's dims makes the
    # rotation independent of that axis.
    gen = torch.Generator().manual_seed(5)
    q = torch.randn(1, 2, config.tokens_per_frame, config.d_head, generator=gen)
    q_zeroed = q.clone()
    q_zeroed[..., dim_lo:dim_hi] = 0.0
    assert not torch.equal(
        apply_ortho_rope(q, (cos_a, sin_a)), apply_ortho_rope(q, (cos_b, sin_b))
    )
    assert torch.equal(
        apply_ortho_rope(q_zeroed, (cos_a, sin_a)), apply_ortho_rope(q_zeroed, (cos_b, sin_b))
    ), f"{axis} does not own head dims {dim_lo}..{dim_hi - 1}"


def test_rotation_is_the_interleaved_pair_form_with_a_concatenated_output():
    """Pairs are read interleaved (``unfold(-1, 2, 2)``) but
    written back with ``cat``, so pair ``p`` lands at output dims ``p`` and
    ``p + 32``. Rewriting this as an in-place interleave is the natural "fix"
    and is wrong; so is reading the pairs split-half."""
    config = waypoint_1_5_1b_720p()
    cos, sin = OrthoRoPEAngles(config)(*grid_positions(config, frame=3))
    half = config.d_head // 2

    gen = torch.Generator().manual_seed(9)
    q = torch.randn(1, 2, config.tokens_per_frame, config.d_head, generator=gen)
    out = apply_ortho_rope(q, (cos, sin))

    even, odd = q[..., 0::2], q[..., 1::2]
    assert torch.equal(out[..., :half], even * cos - odd * sin)
    assert torch.equal(out[..., half:], odd * cos + even * sin)

    # The split-half reading (q[:half], q[half:]) is a different function.
    lo, hi = q[..., :half], q[..., half:]
    split_half = torch.cat((lo * cos - hi * sin, hi * cos + lo * sin), dim=-1)
    assert not torch.equal(out, split_half)
    # ...and so is the in-place interleaved write.
    interleaved = torch.empty_like(out)
    interleaved[..., 0::2] = even * cos - odd * sin
    interleaved[..., 1::2] = odd * cos + even * sin
    assert not torch.equal(out, interleaved)


def test_rope_tables_stay_fp32_whatever_the_serving_dtype_is():
    """``OrthoRoPEAngles``/``OrthoRoPE`` are fp32 islands. The
    port does not use ``NoCastModule``; instead the tables live in a
    ``DeviceTableCache`` outside the module tree, so ``.to(bfloat16)`` cannot
    reach them, and the bodies run in fp32 regardless."""
    config = waypoint_1_5_1b_720p()
    module = OrthoRoPEAngles(config).to(torch.bfloat16)

    xy, inv_t = module._tables.get(torch.device("cpu"))
    assert xy.dtype == inv_t.dtype == torch.float32
    assert list(module.parameters()) == [] and list(module.buffers()) == []

    cos, sin = module(*grid_positions(config, frame=2))
    assert cos.dtype == sin.dtype == torch.float32

    gen = torch.Generator().manual_seed(1)
    q32 = torch.randn(1, 2, config.tokens_per_frame, config.d_head, generator=gen)
    q16 = q32.to(torch.bfloat16)
    out16 = apply_ortho_rope(q16, (cos, sin))
    assert out16.dtype == torch.bfloat16
    # The rotation itself ran in fp32 and rounded once at the end.
    assert torch.equal(out16, apply_ortho_rope(q16.float(), (cos, sin)).to(torch.bfloat16))


def test_rope_rejects_out_of_grid_positions():
    config = waypoint_1_5_1b_720p()
    module = OrthoRoPEAngles(config)
    x_pos, y_pos, t_pos = grid_positions(config, frame=0)
    # torch._assert -- an AssertionError, unlike the torch._check sites elsewhere.
    with pytest.raises(AssertionError, match="pos_ids out of bounds"):
        module(x_pos=x_pos + config.width, y_pos=y_pos, t_pos=t_pos)
    with pytest.raises(ValueError, match="divisible by 8"):
        OrthoRoPEAngles(WaypointConfig(d_model=36, n_heads=1, n_kv_heads=1))


# ---------------------------------------------------------------------------
# 5. The small layers
# ---------------------------------------------------------------------------


def test_rms_norm_is_unweighted_and_scale_invariant():
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(2, 3, 16, generator=gen)
    assert torch.equal(rms_norm(x), F.rms_norm(x, (16,)))
    # No learned gain anywhere in Waypoint: every use site either gets its scale
    # from adaLN or wants a bare normalization.
    assert torch.allclose(rms_norm(x * 7.0), rms_norm(x), atol=1e-6)


def test_ada_rmsnorm_broadcasts_one_modulation_vector_per_frame():
    B, N, T, D = 1, 3, 4, 8
    gen = torch.Generator().manual_seed(4)
    x = torch.randn(B, N * T, D, generator=gen, dtype=torch.float64)
    scale = torch.randn(B, N, D, generator=gen, dtype=torch.float64)
    bias = torch.randn(B, N, D, generator=gen, dtype=torch.float64)

    got = ada_rmsnorm(x, scale, bias)
    want = torch.cat(
        [
            rms_norm(x[:, n * T : (n + 1) * T]) * (1 + scale[:, n : n + 1]) + bias[:, n : n + 1]
            for n in range(N)
        ],
        dim=1,
    )
    assert torch.allclose(got, want, rtol=0, atol=1e-12)
    # A per-frame vector really is constant across that frame's tokens.
    flat = ada_rmsnorm(torch.ones(B, N * T, D, dtype=torch.float64), scale, bias)
    for n in range(N):
        frame = flat[:, n * T : (n + 1) * T]
        assert torch.allclose(frame, frame[:, :1].expand_as(frame), rtol=0, atol=1e-12)


def test_ada_gate_has_no_implicit_one_plus():
    B, N, T, D = 1, 2, 3, 5
    gen = torch.Generator().manual_seed(6)
    x = torch.randn(B, N * T, D, generator=gen)
    assert torch.equal(ada_gate(x, torch.zeros(B, N, D)), torch.zeros_like(x))
    assert torch.equal(ada_gate(x, torch.ones(B, N, D)), x)
    gate = torch.zeros(B, N, D)
    gate[:, 1] = 1.0
    gated = ada_gate(x, gate)
    assert torch.equal(gated[:, :T], torch.zeros(B, T, D)) and torch.equal(gated[:, T:], x[:, T:])


def test_adaln_folds_scale_and_shift_into_one_bias_free_projection():
    D, B, N, T = 8, 1, 2, 3
    torch.manual_seed(0)
    norm = AdaLN(D).to(torch.float64)
    assert norm.fc.bias is None and norm.fc.out_features == 2 * D

    gen = torch.Generator().manual_seed(8)
    x = torch.randn(B, N * T, D, generator=gen, dtype=torch.float64)
    cond = torch.randn(B, N, D, generator=gen, dtype=torch.float64)

    ab = norm.fc(F.silu(cond))
    scale, shift = ab.chunk(2, dim=-1)
    want = torch.cat(
        [
            rms_norm(x[:, n * T : (n + 1) * T]) * (1 + scale[:, n : n + 1]) + shift[:, n : n + 1]
            for n in range(N)
        ],
        dim=1,
    )
    assert torch.allclose(norm(x, cond), want, rtol=0, atol=1e-12)


def test_mlp_is_bias_free_end_to_end():
    """The bias-free-ness is the only reason
    ``F.linear(h, self.mlp.fc2.weight)`` in ``MLPFusion`` is correct. A bias
    would load and then be silently dropped at compute time."""
    mlp = MLP(6, 12, 4)
    assert mlp.fc1.bias is None and mlp.fc2.bias is None
    assert not [n for n, _ in mlp.named_parameters() if n.endswith("bias")]
    gen = torch.Generator().manual_seed(10)
    x = torch.randn(2, 6, generator=gen)
    assert torch.equal(mlp(x), mlp.fc2(F.silu(mlp.fc1(x))))
    assert torch.equal(mlp(x), F.linear(F.silu(F.linear(x, mlp.fc1.weight)), mlp.fc2.weight))


def test_noise_conditioner_fourier_features_and_fp32_island():
    cond = NoiseConditioner(16)
    (freq,) = cond._freq.get(torch.device("cpu"))
    assert torch.equal(freq, torch.logspace(0, -1, steps=256, base=10_000.0, dtype=torch.float32))
    assert freq.dtype == torch.float32
    assert cond.mlp.fc1.in_features == 512 and cond.mlp.fc1.out_features == 64

    sigma = torch.tensor([[1.0, 0.9, 0.75, 0.3, 0.0]])
    phase = (sigma.reshape(-1).float() * 1000)[:, None] * freq[None, :]
    want = cond.mlp(torch.cat((phase.sin(), phase.cos()), dim=-1) * 2**0.5).view(1, 5, 16)
    assert torch.equal(cond(sigma), want)

    # The frequency table is derived state: not a buffer, not in state_dict, and
    # out of reach of a dtype cast.
    assert "freq" not in dict(cond.named_buffers()) and not any("freq" in k for k in cond.state_dict())
    cond.to(torch.bfloat16)
    (freq_after,) = cond._freq.get(torch.device("cpu"))
    assert freq_after.dtype == torch.float32
    with pytest.raises(ValueError, match="even fourier_dim"):
        NoiseConditioner(16, fourier_dim=7)


def test_noise_conditioner_must_stay_fp32_to_serve_fp32_sigma():
    """Why ``FP32_MODULE_PATHS`` exists at all: after the global bf16 cast the
    module's own body still upcasts sigma to fp32, so a bf16 ``mlp`` cannot
    consume it. The failure is loud here, which is the good case -- the point of
    the pin is that the fp32 island is not optional."""
    cond = NoiseConditioner(16).to(torch.bfloat16)
    with pytest.raises(RuntimeError):
        cond(torch.tensor([[1.0]]))
    cond.to(torch.float32)
    assert cond(torch.tensor([[1.0]])).dtype == torch.float32


def test_mlp_fusion_stores_a_packed_fc1_and_splits_it_at_compute_time():
    """``mlp.fc1`` is one ``[D, 2D]`` matrix (one loader key),
    used split so ``cond`` broadcasts over its frame's tokens instead of being
    materialized into a ``[B, N*T, 2D]`` concat. Same arithmetic."""
    config = reduced_config()
    D, N, T = config.d_model, 3, 4
    torch.manual_seed(1)
    fusion = MLPFusion(config).to(torch.float64)

    assert tuple(fusion.mlp.fc1.weight.shape) == (D, 2 * D)
    assert [n for n, _ in fusion.named_parameters()] == ["mlp.fc1.weight", "mlp.fc2.weight"]

    gen = torch.Generator().manual_seed(12)
    x = torch.randn(1, N * T, D, generator=gen, dtype=torch.float64)
    cond = torch.randn(1, N, D, generator=gen, dtype=torch.float64)

    # The nominal form the parameter tree describes: MLP(2D, D, D) on cat([x, cond]).
    cond_per_token = cond.repeat_interleave(T, dim=1)
    want = fusion.mlp(torch.cat((x, cond_per_token), dim=-1))
    assert torch.allclose(fusion(x, cond), want, rtol=0, atol=1e-11)

    # The split is x-half first: swapping the chunks is a different function.
    Wx, Wc = fusion.mlp.fc1.weight.chunk(2, dim=1)
    swapped = F.linear(
        F.silu(F.linear(x.view(1, N, T, D), Wc) + F.linear(cond, Wx).unsqueeze(2)),
        fusion.mlp.fc2.weight,
    ).flatten(1, 2)
    assert not torch.allclose(fusion(x, cond), swapped)


def test_controller_input_embedding_concat_order_is_mouse_button_scroll():
    """The widths sum to 259 under any permutation, so a wrong
    order fails silently. The three fields carry disjoint, self-identifying
    values and the MLP's input is captured directly."""
    config = waypoint_1_5_1b_720p()
    assert config.d_ctrl_in == 259 == 2 + config.n_buttons + 1

    emb = ControllerInputEmbedding(config)
    assert emb.mlp.fc1.in_features == 259
    assert emb.mlp.fc1.out_features == config.d_model * config.mlp_ratio
    assert emb.mlp.fc2.out_features == config.d_model

    seen = []

    class Capture(torch.nn.Module):
        def forward(self, x):
            seen.append(x.clone())
            return x[..., :1]

    emb.mlp = Capture()
    mouse = torch.tensor([[[-1.0, -2.0]]])
    button = torch.arange(256, dtype=torch.float32).view(1, 1, 256) + 100.0
    scroll = torch.tensor([[[7.5]]])
    emb(mouse, button, scroll)

    (packed,) = seen
    assert packed.shape == (1, 1, 259)
    assert torch.equal(packed[..., 0:2], mouse), "field 0:2 is not mouse"
    assert torch.equal(packed[..., 2:258], button), "field 2:258 is not button"
    assert torch.equal(packed[..., 258:259], scroll), "field 258:259 is not scroll"
    # Sensitivity check: a permuted order really would produce a different vector.
    assert not torch.equal(packed, torch.cat((mouse, scroll, button), dim=-1))
