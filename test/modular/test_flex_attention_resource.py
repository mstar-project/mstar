"""The FlexAttention backend as an engine resource: what the factory accepts,
and what the kernel does with a ring's visibility row.

Two different kinds of failure are pinned here, and both are silent.

The first is a *pairing* failure. ``PagedKVConfig`` and ``RingKVConfig`` are
close enough in shape that handing one to the wrong backend gets past
construction: a paged planner walks a page table the ring does not have, a ring
view reaches a kernel that assumes monotonic append. Neither raises on its own,
so ``AttentionManager.build`` cross-checks both directions and these tests
assert it stays that way.

The second is the eager-``flex_attention`` trap. The ``BlockMask`` carries a
no-op ``mask_mod``, so the
eager path rebuilds "everything is visible" and blends every unwritten slot in
as a zero K/V. That is why ``flex.flex_attention_masked`` is a pinned
``torch.compile`` and not a caller's choice, and the test below asserts *both*
halves: that the compiled path matches a masked-dense reference, and that bare
eager does not. If eager ever starts agreeing, the pin has stopped being
load-bearing and the argument for it needs re-deriving rather than the test
being deleted.

A third section is not a failure but a *derisk*, and is labelled as one: the
world pool folds N worlds into the token axis of one ring, so the shape a
batched step would take is B queries against a stride-0 ``expand`` of a single
K/V with the per-world isolation moved into the BlockMask's leading dim. Nothing
ships that path today (the step batch is 1), but if it does not hold bit-exactly
then batching a step is not free and the design owes a different answer, so the
two properties it rests on are asserted rather than assumed.

Checkpoint-free, and CPU except where a test is parametrized over the device:
``torch.compile(flex_attention)`` works on CPU in torch 2.9, at a few seconds of
inductor time on first use.
"""

import sys

import pytest
import torch

sys.path.insert(0, ".")

from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    BlockMask,
    flex_attention,
)

from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
    AttnBackend,
    KVSpec,
    StepContext,
)
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.flex import (
    FlexAttentionManager,
    flex_attention_masked,
    make_block_mask,
)
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.config import (
    PagedKVConfig,
    RingKVConfig,
    RingKVLayerConfig,
)
from mstar.engine.resources.kv.ring.cache import LayerRingCache
from mstar.engine.resources.kv.ring.manager import RingPlan

BLOCK = _DEFAULT_SPARSE_BLOCK_SIZE  # 128
TPF = BLOCK  # one frame per sparse block keeps the geometry readable
KV_LEN = 4 * BLOCK  # 3 ring frames + 1 scratch
N_KV_HEADS = 2
N_QO_HEADS = 4
D_HEAD = 32


def ring_config(n_layers: int = 2) -> RingKVConfig:
    return RingKVConfig(
        num_layers=n_layers,
        num_kv_heads=N_KV_HEADS,
        head_dim=D_HEAD,
        num_qo_heads=N_QO_HEADS,
        tokens_per_frame=TPF,
        layers=tuple(
            RingKVLayerConfig(ring_frames=3, ring_buckets=3, pinned_dilation=1)
            for _ in range(n_layers)
        ),
    )


def paged_config() -> PagedKVConfig:
    return PagedKVConfig(
        num_layers=2,
        num_kv_heads=N_KV_HEADS,
        head_dim=D_HEAD,
        num_qo_heads=N_QO_HEADS,
        max_seq_len=1024,
        max_num_pages=16,
        page_size=BLOCK,
    )


def build_attention(backend: AttnBackend, kv_config) -> AttentionManager:
    """Drive the real spec-time factory, not the manager's constructor: the
    config cross-check lives in ``build`` and is the thing under test."""
    kv_spec = KVSpec(resource_key="kv", nodes={"dit"}, config=kv_config)
    attn_spec = AttentionSpec(
        resource_key="attn",
        nodes={"dit"},
        config=AttentionConfig(kv_cache="kv", backend=backend),
    )
    info = EngineResourceInfo(
        device=torch.device("cpu"),
        kv_dtype=torch.float32,
        dependencies={"kv": kv_spec},
    )
    return AttentionManager.build(attn_spec, info)


def visible_row(committed_blocks: tuple[int, ...]) -> torch.Tensor:
    """A ring visibility row: whole blocks, plus the permanently visible
    scratch frame at the tail (it holds the frame being denoised, so masking it
    would remove self-attention)."""
    visible = torch.zeros(KV_LEN, dtype=torch.bool)
    for b in committed_blocks:
        visible[b * BLOCK : (b + 1) * BLOCK] = True
    visible[KV_LEN - BLOCK :] = True
    return visible


def masked_dense_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, visible: torch.Tensor
) -> torch.Tensor:
    """``softmax(QK^T + mask) V`` written out, with non-visible KV positions
    masked off. Deliberately not another flex call: the reference has to be
    something whose masking cannot share a bug with the thing it checks."""
    q_heads = q.size(1)
    kv_heads = k.size(1)
    k = k.repeat_interleave(q_heads // kv_heads, dim=1)
    v = v.repeat_interleave(q_heads // kv_heads, dim=1)
    scores = (q @ k.transpose(-1, -2)) / (q.size(-1) ** 0.5)
    scores = scores.masked_fill(~visible[None, None, None, :], float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def qkv(visible: torch.Tensor, seed: int = 0xC0FFEE):
    """Query, and a ring whose unwritten slots hold zeros — which is what an
    unwritten ring slot actually holds, and why eager's blending of them is
    wrong rather than merely different."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(1, N_QO_HEADS, TPF, D_HEAD, generator=gen)
    k = torch.zeros(1, N_KV_HEADS, KV_LEN, D_HEAD)
    v = torch.zeros(1, N_KV_HEADS, KV_LEN, D_HEAD)
    n = int(visible.sum())
    k[:, :, visible] = torch.randn(1, N_KV_HEADS, n, D_HEAD, generator=gen)
    v[:, :, visible] = torch.randn(1, N_KV_HEADS, n, D_HEAD, generator=gen)
    return q, k, v


# ---------------------------------------------------------------------------
# 1. The factory, and the config cross-check
# ---------------------------------------------------------------------------


def test_flex_backend_with_a_ring_kv_config_builds_the_flex_manager():
    manager = build_attention(AttnBackend.FLEX, ring_config())
    assert isinstance(manager, FlexAttentionManager)
    assert manager.depends_on() == {"kv"}


def test_flex_backend_rejects_a_paged_kv_config():
    """A paged config behind the flex backend is not a shape error: every field
    the manager reads early is present on both configs."""
    with pytest.raises(TypeError) as excinfo:
        build_attention(AttnBackend.FLEX, paged_config())
    message = str(excinfo.value)
    assert "flex" in message, "the error has to name the backend that was asked for"
    assert "RingKVConfig" in message and "PagedKVConfig" in message


def test_paged_backend_rejects_a_ring_kv_config():
    """The other direction, which is the dangerous one: a ring's storage under
    a backend that assumes monotonic append reads as history rewriting itself."""
    with pytest.raises(TypeError) as excinfo:
        build_attention(AttnBackend.FLASHINFER, ring_config())
    message = str(excinfo.value)
    assert "flashinfer" in message
    assert "PagedKVConfig" in message and "RingKVConfig" in message


def test_requires_kv_write_is_false_for_flex_and_true_for_the_paged_backend():
    """The layer reads this to decide whether to call ``kv.write_kv``. For the
    ring the answer is no — ``upsert`` already wrote the frame and returned the
    view to attend against — and calling one would commit the frame twice."""
    flex_manager = build_attention(AttnBackend.FLEX, ring_config())
    assert flex_manager.requires_kv_write is False

    paged_manager = build_attention(AttnBackend.FLASHINFER, paged_config())
    assert paged_manager.requires_kv_write is True


def test_plan_clears_the_inherited_cursors():
    """``AttentionResource``'s contract: a step that never binds the label or
    layer cursor must not inherit the previous step's."""
    manager = build_attention(AttnBackend.FLEX, ring_config())
    manager.set_default_label("stale")
    manager.set_default_layer_idx(7)
    manager.plan(
        AttentionStep(),
        StepContext(request_ids=("r0",), graph_walk="gen", slot=0, capture=False),
    )
    assert manager.default_label == "main"
    assert manager._default_layer_idx is None


def test_plan_stages_one_reused_mask_per_geometry_and_slot(monkeypatch):
    config = RingKVConfig(
        num_layers=4,
        num_kv_heads=N_KV_HEADS,
        head_dim=D_HEAD,
        num_qo_heads=N_QO_HEADS,
        tokens_per_frame=TPF,
        num_worlds=2,
        layers=(
            RingKVLayerConfig(4, 4, 1),
            RingKVLayerConfig(4, 4, 1),
            RingKVLayerConfig(4, 2, 2),
            RingKVLayerConfig(4, 2, 2),
        ),
    )
    manager = build_attention(AttnBackend.FLEX, config)
    ctx = StepContext(
        request_ids=("r",), graph_walk="rollout", slot=1, capture=False,
        plan_results={"kv": RingPlan("r", 1, 5)},
    )

    manager.plan(AttentionStep(), ctx)

    assert manager.needs_token_visibility is False
    assert len(manager._planned_masks) == 2
    local = manager._mask_for(1, manager._geometry(config.layers[0]))
    assert local is manager._mask_for(1, manager._geometry(config.layers[1]))
    global_mask = manager._mask_for(1, manager._geometry(config.layers[2]))
    assert global_mask is manager._mask_for(1, manager._geometry(config.layers[3]))
    assert local is not global_mask

    # One block per frame. World 1 starts after world 0's five-block span.
    assert local.full_kv_num_blocks.unique().tolist() == [4]
    assert local.full_kv_indices[0, 0, 0, :4].tolist() == [5, 7, 8, 9]
    assert global_mask.full_kv_num_blocks.unique().tolist() == [3]
    assert global_mask.full_kv_indices[0, 0, 0, :3].tolist() == [5, 6, 9]

    addresses = {
        key: (value.full_kv_num_blocks.data_ptr(), value.full_kv_indices.data_ptr())
        for key, value in manager._planned_masks.items()
    }
    table_addresses = {
        key: (value[0].data_ptr(), value[1].data_ptr())
        for key, value in manager._visibility_tables.items()
    }
    monkeypatch.setattr(
        torch,
        "tensor",
        lambda *args, **kwargs: pytest.fail(
            "mask planning must not allocate a new staging tensor"
        ),
    )
    ctx.plan_results["kv"] = RingPlan("r", 1, 6)
    manager.plan(AttentionStep(), ctx)
    assert addresses == {
        key: (value.full_kv_num_blocks.data_ptr(), value.full_kv_indices.data_ptr())
        for key, value in manager._planned_masks.items()
    }
    assert table_addresses == {
        key: (value[0].data_ptr(), value[1].data_ptr())
        for key, value in manager._visibility_tables.items()
    }


def test_planned_masks_match_ring_visibility_across_wraps_and_worlds():
    """The host plan must reproduce the ring's pre-commit visibility exactly.

    This crosses two wraps of both geometries while alternating worlds. It
    compares against ``LayerRingCache.upsert``'s independently maintained
    ``written`` state, so a clock, dilation, overwrite, scratch, or world-offset
    error in the planner cannot satisfy the test by sharing its formula.
    """
    layers = (
        RingKVLayerConfig(ring_frames=4, ring_buckets=4, pinned_dilation=1),
        RingKVLayerConfig(ring_frames=4, ring_buckets=2, pinned_dilation=2),
    )
    config = RingKVConfig(
        num_layers=len(layers),
        num_kv_heads=N_KV_HEADS,
        head_dim=D_HEAD,
        num_qo_heads=N_QO_HEADS,
        tokens_per_frame=TPF,
        num_worlds=2,
        layers=layers,
    )
    manager = build_attention(AttnBackend.FLEX, config)
    caches = [
        LayerRingCache(
            num_worlds=config.num_worlds,
            n_kv_heads=config.num_kv_heads,
            ring_frames=layer.ring_frames,
            ring_buckets=layer.ring_buckets,
            d_head=config.head_dim,
            tokens_per_frame=config.tokens_per_frame,
            pinned_dilation=layer.pinned_dilation,
            dtype=torch.float32,
            device="cpu",
        )
        for layer in layers
    ]
    kv = torch.zeros(2, 1, N_KV_HEADS, TPF, D_HEAD)

    for frame in range(10):
        for world in (0, 1):
            expected = []
            for cache in caches:
                *_, visible = cache.upsert(
                    kv,
                    torch.tensor(frame, dtype=torch.int64),
                    True,
                    torch.tensor([world], dtype=torch.int64),
                )
                expected.append(
                    visible.view(-1, BLOCK).all(-1).nonzero().flatten().tolist()
                )

            ctx = StepContext(
                request_ids=(f"r{world}",),
                graph_walk="rollout",
                slot=0,
                capture=False,
                plan_results={"kv": RingPlan(f"r{world}", world, frame)},
            )
            manager.plan(AttentionStep(), ctx)

            for layer, expected_blocks in zip(layers, expected, strict=True):
                mask = manager._mask_for(0, manager._geometry(layer))
                count = int(mask.full_kv_num_blocks[0, 0, 0])
                assert mask.full_kv_indices[0, 0, 0, :count].tolist() == expected_blocks


def test_capture_plan_requires_preallocated_mask_addresses():
    manager = build_attention(AttnBackend.FLEX, ring_config())
    ctx = StepContext(
        request_ids=("r",), graph_walk="rollout", slot=0, capture=True,
        plan_results={"kv": RingPlan("r", 0, 0)},
    )
    with pytest.raises(RuntimeError, match="not allocated before CUDA graph capture"):
        manager.plan(AttentionStep(), ctx)


# ---------------------------------------------------------------------------
# 2. The mask
# ---------------------------------------------------------------------------


def test_make_block_mask_rejects_unaligned_and_non_multiple_lengths():
    """Both constraints are real rather than defensive: the mask has no partial
    blocks, so a length or a row that does not land on a block boundary would
    silently round the visible region."""
    visible = visible_row((0,))
    with pytest.raises(RuntimeError, match="multiple of block size"):
        make_block_mask(TPF + 1, KV_LEN, visible)
    with pytest.raises(RuntimeError, match="multiple of block size"):
        make_block_mask(TPF, KV_LEN - 1, visible[:-1])

    ragged = torch.zeros(2 * BLOCK, dtype=torch.bool)
    ragged[3] = True  # a partial block, which no whole-frame write can produce
    with pytest.raises(AssertionError, match="block-aligned"):
        make_block_mask(TPF, ragged.numel(), ragged)


# ---------------------------------------------------------------------------
# 3. The kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "committed_blocks"),
    [
        ("one_frame", (0,)),
        ("gappy_ring", (0, 2)),
    ],
)
def test_attend_matches_a_masked_dense_reference(label, committed_blocks):
    visible = visible_row(committed_blocks)
    q, k, v = qkv(visible)
    manager = build_attention(AttnBackend.FLEX, ring_config())

    got = manager.attend(q, k, v, visible, enable_gqa=True)
    want = masked_dense_reference(q, k, v, visible)

    err = (got - want).abs().max().item()
    print(f"[{label}] attend vs masked-dense = {err:.3e}")
    assert got.shape == q.shape
    assert err < 1e-5, f"FlexAttentionManager.attend diverged from masked dense ({err:.3e})"


def test_eager_flex_attention_does_not_honour_the_block_mask():
    """The reason ``flex_attention_masked`` is a ``torch.compile`` pinned
    inside this module rather than a caller's choice.

    Same q/k/v, same ``BlockMask``, two callables: the compiled one iterates
    the block index lists and matches the masked-dense reference; the eager one
    re-derives visibility from the no-op ``mask_mod``, decides everything is
    visible, and blends the unwritten (zero) ring slots in. Nothing raises
    either way, so the guard has to be numeric.

    If both sides ever match, this test is pinning nothing: eager would have
    been fixed upstream, the "unless someone removes the compile" argument
    would have quietly stopped being tested, and it should be re-derived rather
    than deleted.
    """
    visible = visible_row((0,))
    q, k, v = qkv(visible)
    bm = make_block_mask(TPF, KV_LEN, visible)

    dense = masked_dense_reference(q, k, v, visible)
    compiled = flex_attention_masked(q, k, v, block_mask=bm, enable_gqa=True)
    eager = flex_attention(q, k, v, block_mask=bm, enable_gqa=True)

    compiled_err = (compiled - dense).abs().max().item()
    eager_err = (eager - dense).abs().max().item()
    print(f"compiled vs masked-dense={compiled_err:.3e}  eager vs masked-dense={eager_err:.3e}")

    assert compiled_err < 1e-5, (
        f"the compiled path diverged from the masked-dense reference ({compiled_err:.3e})"
    )
    assert eager_err > 1e-2, (
        f"eager flex_attention agreed with the masked reference to {eager_err:.3e}; the "
        "trap may have been fixed upstream, in which case re-derive "
        "the guard rather than deleting it"
    )


def test_attend_takes_the_compiled_path():
    """The regression guard proper: swapping ``flex_attention_masked`` for bare
    ``flex_attention`` inside ``attend`` turns this red, and nothing else
    would."""
    visible = visible_row((0, 2))
    q, k, v = qkv(visible, seed=11)
    bm = make_block_mask(TPF, KV_LEN, visible)
    manager = build_attention(AttnBackend.FLEX, ring_config())

    got = manager.attend(q, k, v, visible, enable_gqa=True)
    want = flex_attention_masked(q, k, v, block_mask=bm, enable_gqa=True)
    trap = flex_attention(q, k, v, block_mask=bm, enable_gqa=True)

    assert torch.equal(got, want), "attend is not going through flex_attention_masked"
    assert (got - trap).abs().max().item() > 1e-2, (
        "attend's output is indistinguishable from the eager path; the mask is not honoured"
    )


def test_flex_attention_masked_is_a_compiled_callable():
    """If someone "simplifies" the pin back to the bare function, this fails
    first and says why."""
    assert flex_attention_masked is not flex_attention
    assert hasattr(flex_attention_masked, "_torchdynamo_orig_callable"), (
        "flex.flex_attention_masked must stay wrapped in torch.compile: with a no-op "
        "mask_mod the eager kernel ignores the ring mask entirely"
    )


def test_masked_dense_reference_is_not_trivially_satisfied():
    """The reference above is only a reference if masking changes the answer;
    an all-visible row would make every assertion in this file pass for the
    wrong reason."""
    visible = visible_row((0,))
    q, k, v = qkv(visible)
    all_visible = torch.ones_like(visible)
    masked = masked_dense_reference(q, k, v, visible)
    unmasked = masked_dense_reference(q, k, v, all_visible)
    assert (masked - unmasked).abs().max().item() > 1e-2


# ---------------------------------------------------------------------------
# 4. The batched step this backend is not asked for yet
# ---------------------------------------------------------------------------
#
# The world pool folds N worlds into the token axis of ONE ring, so the ring is
# `[2, 1, H_kv, N*capacity, D]` and world `w` owns `[w*capacity, (w+1)*capacity)`.
# Today the step batch is 1 and `attend` gets a single row, so isolation is a
# `[kv_len]` visibility row and this section is a derisk, not a contract.
#
# What it derisks is the *only* way batching a step can be made free. When B
# rows of one step attend against that ring, the K/V they share is byte-identical
# -- it is the same buffer -- so the honest shape for it is a stride-0 `expand`
# on dim 0, and the per-row isolation moves out of the tensor and into the
# BlockMask's leading dim: `[B, 1, q_blk, kv_blk]`, one visibility row per world.
# If that does not hold, batching a step means materializing B copies of an
# 816-MiB-per-world ring per layer per pass, which is not a batching strategy.
#
# Two things have to be true and neither is obvious: the compiled kernel must
# accept a batch dim whose stride is 0 (it can trivially not, and the failure
# would be a copy rather than an error), and the per-row answers must be
# *bit-identical* to the single-row calls the node makes today, or the ported
# numerics stop being comparable to the reference the moment concurrency is
# turned on. Both are asserted below, on CPU and on CUDA.

WORLDS = 3
CAPACITY = 2 * BLOCK  # one history frame + the scratch frame, per world
FOLDED_KV = WORLDS * CAPACITY


def world_span(w: int) -> tuple[int, int]:
    return w * CAPACITY, (w + 1) * CAPACITY


def folded_visible_row(w: int, *, history: bool, device) -> torch.Tensor:
    """One world's visibility over the *folded* ring: its own scratch block
    always, its own history block if it has committed one, and nothing outside
    its span ever."""
    row = torch.zeros(FOLDED_KV, dtype=torch.bool, device=device)
    lo, hi = world_span(w)
    row[hi - BLOCK : hi] = True  # scratch: the frame being denoised
    if history:
        row[lo : lo + BLOCK] = True
    return row


def folded_block_mask(rows: list[torch.Tensor], q_len: int) -> "object":
    """``make_block_mask`` with a batch dim: the same query-uniform,
    full-blocks-only construction, stacked over rows.

    Deliberately written here and not in ``flex.py``. The shipped backend takes
    a single ``[kv_len]`` row because the node takes a single row; adding an
    unreachable batched path to the module would be an untested branch on the
    serving path. This is the shape the future change would take, asserted
    against today's kernel.
    """
    b = len(rows)
    q_blocks = q_len // BLOCK
    kv_blocks = FOLDED_KV // BLOCK
    device = rows[0].device

    per_row = torch.stack([r.view(kv_blocks, BLOCK).any(-1) for r in rows])
    assert torch.equal(
        per_row, torch.stack([r.view(kv_blocks, BLOCK).all(-1) for r in rows])
    ), "written must be block-aligned"

    full_bm = per_row[:, None, :].expand(b, q_blocks, kv_blocks)
    full_kv_num_blocks = full_bm.sum(dim=-1, dtype=torch.int32)[:, None].contiguous()
    full_kv_indices = (
        full_bm.argsort(dim=-1, descending=True, stable=True)
        .to(torch.int32)[:, None]
        .contiguous()
    )
    zeros_n = torch.zeros((b, 1, q_blocks), dtype=torch.int32, device=device)
    zeros_i = torch.zeros((b, 1, q_blocks, kv_blocks), dtype=torch.int32, device=device)

    return BlockMask.from_kv_blocks(
        zeros_n,
        zeros_i,
        full_kv_num_blocks,
        full_kv_indices,
        BLOCK_SIZE=BLOCK,
        mask_mod=None,
        seq_lengths=(q_len, FOLDED_KV),
        compute_q_blocks=False,
    )


def folded_ring(device, seed: int = 0x5EED):
    """A folded ring with every world's span filled with distinct noise, and B
    queries. Distinct per span is the point: an output that is invariant to a
    neighbour's bytes has actually been isolated, rather than reading zeros that
    happened to contribute nothing."""
    gen = torch.Generator().manual_seed(seed)
    k = torch.randn(1, N_KV_HEADS, FOLDED_KV, D_HEAD, generator=gen).to(device)
    v = torch.randn(1, N_KV_HEADS, FOLDED_KV, D_HEAD, generator=gen).to(device)
    q = torch.randn(WORLDS, N_QO_HEADS, TPF, D_HEAD, generator=gen).to(device)
    return q, k, v


DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="needs a GPU"
        ),
    ),
]


@pytest.mark.parametrize("device", DEVICES)
def test_a_batched_step_can_share_one_ring_by_stride_0_expand(device):
    """B rows over a stride-0 view of one ring, bit-identical to B single-row
    calls, with zero bytes copied.

    ``torch.equal`` and not a tolerance, on purpose. A batched kernel is allowed
    to tile the KV differently and land within 1e-6 of the serial answer, and
    that would be a real divergence for this port: the whole argument for
    FlexAttention here is that the accumulation order matches the reference's,
    so a batch that reorders it silently costs the only comparison there is.
    If this ever needs a tolerance, batching a step is not free and the finding
    is that, not the tolerance.
    """
    q, k, v = folded_ring(device)
    rows = [
        folded_visible_row(w, history=(w != 1), device=torch.device(device))
        for w in range(WORLDS)
    ]

    k_batched = k.expand(WORLDS, N_KV_HEADS, FOLDED_KV, D_HEAD)
    v_batched = v.expand(WORLDS, N_KV_HEADS, FOLDED_KV, D_HEAD)
    assert k_batched.stride(0) == 0 and v_batched.stride(0) == 0
    assert k_batched.data_ptr() == k.data_ptr(), "the expand copied the ring"
    assert (
        k_batched.untyped_storage().data_ptr() == k.untyped_storage().data_ptr()
    ), "the expand copied the ring"

    batched = flex_attention_masked(
        q, k_batched, v_batched, block_mask=folded_block_mask(rows, TPF), enable_gqa=True
    )
    assert batched.shape == q.shape

    manager = build_attention(AttnBackend.FLEX, ring_config())
    for w in range(WORLDS):
        serial = manager.attend(q[w : w + 1], k, v, rows[w], enable_gqa=True)
        assert torch.equal(batched[w : w + 1], serial), (
            f"world {w} in a batch of {WORLDS} differs from the same world served "
            f"alone by {(batched[w:w+1] - serial).abs().max().item():.3e}; a batched "
            "step would not be the same computation the node performs today"
        )


@pytest.mark.parametrize("device", DEVICES)
def test_a_batched_step_does_not_reach_across_world_spans(device):
    """The isolation half, and the one that can fail silently.

    The previous test would still pass if the mask leaked, because the serial
    reference it compares against uses the same rows and would leak identically.
    So: run the batch, then overwrite every world's span *except* one with fresh
    noise and re-run. That world's output must be bit-identical, because none of
    the bytes that changed are inside its span — and the folded ring is exactly
    the layout where "inside its span" is a claim about a mask rather than about
    a separate allocation.
    """
    q, k, v = folded_ring(device)
    rows = [
        folded_visible_row(w, history=True, device=torch.device(device))
        for w in range(WORLDS)
    ]
    mask = folded_block_mask(rows, TPF)

    def run(k_ring, v_ring):
        return flex_attention_masked(
            q,
            k_ring.expand(WORLDS, N_KV_HEADS, FOLDED_KV, D_HEAD),
            v_ring.expand(WORLDS, N_KV_HEADS, FOLDED_KV, D_HEAD),
            block_mask=mask,
            enable_gqa=True,
        )

    before = run(k, v)

    for kept in range(WORLDS):
        gen = torch.Generator().manual_seed(1000 + kept)
        k2, v2 = k.clone(), v.clone()
        for w in range(WORLDS):
            if w == kept:
                continue
            lo, hi = world_span(w)
            shape = (1, N_KV_HEADS, hi - lo, D_HEAD)
            k2[:, :, lo:hi] = torch.randn(shape, generator=gen).to(device)
            v2[:, :, lo:hi] = torch.randn(shape, generator=gen).to(device)

        after = run(k2, v2)
        assert torch.equal(after[kept], before[kept]), (
            f"world {kept} moved when only other worlds' spans changed: the "
            "BlockMask is not isolating the folded ring"
        )
        others = [w for w in range(WORLDS) if w != kept]
        assert not torch.equal(after[others[0]], before[others[0]]), (
            "no world moved at all, so the rewrite landed nowhere the mask reads "
            "and this test is vacuous"
        )
