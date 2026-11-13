"""Chunk-causal teacher-forcing attention: the mask, the BlockMask the
resource plans for it, and flex_attention over that mask.

The mask is the requirement, so it is checked three ways. Each of its four
rules is asserted directly against token roles this file derives on its own
(always runs). The dense visibility the planned ``BlockMask`` expresses, tile
lists and ``mask_mod`` together, is compared element for element with the
Praxis-1 reference's predicate (runs where ``praxis`` is importable). And the
rule checks are run against masks that are wrong in one way each, which they
must reject, so a pass means something.

CPU only. ``torch.compile(flex_attention)`` compiles once per sequence length
on CPU at about 15 s, so the kernel tests share one length.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from mstar.engine.resources import (
    BucketKey,
    ChunkCausalAttentionStep,
    ChunkCausalLayout,
    RaggedAttentionConfig,
    RaggedAttnBackend,
    RaggedChunkCausalAttentionSpec,
    Segment,
    SlotLease,
    StepContext,
)
from mstar.engine.resources.attn.ragged import chunk_causal as cc
from mstar.engine.resources.attn.ragged.chunk_causal import (
    FlexChunkCausalManager,
    FlexChunkCausalWrapper,
    chunk_causal_blocks,
    chunk_causal_runs,
    dense_visibility,
)
from mstar.engine.resources.base import EngineResourceInfo, build_resource

HQ, HKV, D = 4, 2, 32

# Layouts in the reference's units: (global tokens, tokens per frame, frames per
# chunk, noised frames, clean frames). The encoder's clean stream is the noised
# one's first ``chunks - 1`` chunks.
REF_LAYOUTS = {
    "one_chunk_no_clean": (24, 8, 2, 2, 0),
    "two_chunks_one_clean": (24, 8, 2, 4, 2),
    "four_chunks_three_clean": (32, 8, 2, 8, 6),  # 144 tokens, aligned at block 16 and 48
    "total_not_block_aligned": (32, 5, 2, 8, 6),  # 102 tokens
    "prefix_not_block_aligned": (37, 8, 2, 8, 6),  # 149 tokens
    "short_last_chunk": (13, 4, 3, 7, 4),
}
BLOCK_SIZES = (16, 48, 128)


def layout_of(g, tpf, cf, nf, clf) -> ChunkCausalLayout:
    return ChunkCausalLayout(global_len=g, chunk_len=cf * tpf, noised_len=nf * tpf, clean_len=clf * tpf)


# --- independent token roles and the four rules ------------------------------

def roles(lay: ChunkCausalLayout) -> tuple[np.ndarray, np.ndarray]:
    """``(stream, chunk)`` per token, written out from the layout's definition
    alone: stream 0 global, 1 noised, 2 clean; chunk -1 for globals."""
    stream = np.array([0] * lay.global_len + [1] * lay.noised_len + [2] * lay.clean_len)
    chunk = np.array(
        [-1] * lay.global_len
        + [i // lay.chunk_len for i in range(lay.noised_len)]
        + [i // lay.chunk_len for i in range(lay.clean_len)]
    )
    return stream, chunk


def rule_violations(lay: ChunkCausalLayout, visible: np.ndarray) -> list[str]:
    """Each of the four rules checked against ``visible`` (``[n, n]`` bool),
    as the list of rules it breaks."""
    stream, chunk = roles(lay)
    g, n, c = stream == 0, stream == 1, stream == 2
    qc, kc = chunk[:, None], chunk[None, :]
    broken = []
    # 1. every token reads the global prefix; a global query reads only globals
    if not visible[:, g].all() or visible[np.ix_(g, ~g)].any():
        broken.append("globals")
    # 2. clean is block-causal
    if not np.array_equal(visible[np.ix_(c, c)], (kc <= qc)[np.ix_(c, c)]):
        broken.append("clean block-causal")
    # 3. noised is block-diagonal
    if not np.array_equal(visible[np.ix_(n, n)], (kc == qc)[np.ix_(n, n)]):
        broken.append("noised block-diagonal")
    # 4. noised reads strictly earlier clean chunks; clean reads no noised key
    if not np.array_equal(visible[np.ix_(n, c)], (kc < qc)[np.ix_(n, c)]) or visible[np.ix_(c, n)].any():
        broken.append("cross-stream")
    return broken


def dense_from_block_mask(mask, length: int) -> torch.Tensor:
    """What a ``BlockMask`` lets through, per pair: its ``mask_mod`` inside a
    listed partial tile, everything inside a listed full tile, nothing
    elsewhere. Built from the tile lists, so a missing tile shows up here."""
    q_bs, kv_bs = mask.BLOCK_SIZE
    out = torch.zeros(length, length, dtype=torch.bool)
    qi = torch.arange(length)

    def tiles(num, idx):
        if num is None:
            return
        for qb in range(num.shape[-1]):
            for kb in idx[0, 0, qb, : int(num[0, 0, qb])].tolist():
                yield qb, kb

    for qb, kb in tiles(mask.kv_num_blocks, mask.kv_indices):
        q = qi[qb * q_bs:(qb + 1) * q_bs]
        k = qi[kb * kv_bs:(kb + 1) * kv_bs]
        out[q[:, None], k[None, :]] = mask.mask_mod(0, 0, q[:, None], k[None, :])
    for qb, kb in tiles(mask.full_kv_num_blocks, mask.full_kv_indices):
        out[qb * q_bs:(qb + 1) * q_bs, kb * kv_bs:(kb + 1) * kv_bs] = True
    return out


def planned_mask(layouts, block_size, full_blocks, length=None, **graph):
    w = FlexChunkCausalWrapper(
        num_qo_heads=HQ, num_kv_heads=HKV, head_dim=D, block_size=block_size, full_blocks=full_blocks,
        device=torch.device("cpu"), q_data_type=torch.float32, max_total_tokens=length, **graph,
    )
    w.plan(layouts)
    return w


# --- the four rules, standalone ----------------------------------------------

@pytest.mark.parametrize("name", list(REF_LAYOUTS))
def test_the_predicate_obeys_all_four_rules(name):
    lay = layout_of(*REF_LAYOUTS[name])
    assert rule_violations(lay, dense_visibility([lay]).numpy()) == []


@pytest.mark.parametrize("full_blocks", [False, True])
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("name", list(REF_LAYOUTS))
def test_the_planned_block_mask_obeys_all_four_rules(name, block_size, full_blocks):
    lay = layout_of(*REF_LAYOUTS[name])
    w = planned_mask([lay], block_size, full_blocks)
    dense = dense_from_block_mask(w.block_mask, lay.span)
    assert rule_violations(lay, dense.numpy()) == []
    assert torch.equal(dense, dense_visibility([lay]))


@pytest.mark.parametrize("full_blocks", [False, True])
def test_tile_lists_are_exact(full_blocks):
    """A tile is listed iff some pair in it is visible, and full iff all are
    and it is whole; anything looser visits tiles for nothing, anything
    tighter drops keys."""
    lay = layout_of(*REF_LAYOUTS["prefix_not_block_aligned"])
    bs = 16
    dense = dense_visibility([lay])
    starts, ends, tags = chunk_causal_runs([lay], lay.span)
    any_, partial, full = chunk_causal_blocks(starts, ends, tags, lay.span, bs, full_blocks)
    nb = any_.shape[0]
    for qb in range(nb):
        for kb in range(nb):
            tile = dense[qb * bs:(qb + 1) * bs, kb * bs:(kb + 1) * bs]
            assert any_[qb, kb] == bool(tile.any())
            if full_blocks:
                whole = tile.shape == (bs, bs)
                assert full[qb, kb] == (whole and bool(tile.all()))
                assert partial[qb, kb] == (bool(tile.any()) and not full[qb, kb])
    assert not full_blocks or full.any(), "the layout should have full tiles to test"


def test_packed_requests_see_only_themselves():
    a = layout_of(*REF_LAYOUTS["two_chunks_one_clean"])
    b = layout_of(*REF_LAYOUTS["short_last_chunk"])
    w = planned_mask([a, b], 16, True)
    dense = dense_from_block_mask(w.block_mask, a.span + b.span)
    assert torch.equal(dense[: a.span, : a.span], dense_visibility([a]))
    assert torch.equal(dense[a.span:, a.span:], dense_visibility([b]))
    assert not dense[: a.span, a.span:].any() and not dense[a.span:, : a.span].any()


def test_the_decoder_layout_is_globals_plus_block_diagonal_noise():
    """[globals, noised] with no clean stream, the decoder tower's mask."""
    lay = ChunkCausalLayout(global_len=5, chunk_len=8, noised_len=24)
    dense = dense_visibility([lay]).numpy()
    assert rule_violations(lay, dense) == []
    stream, chunk = roles(lay)
    expected = (stream[None, :] == 0) | ((stream[:, None] == 1) & (chunk[:, None] == chunk[None, :]))
    assert np.array_equal(dense, expected)


def test_graph_mode_padding_is_isolated():
    lay = layout_of(*REF_LAYOUTS["two_chunks_one_clean"])
    length = lay.span + 21
    w = planned_mask([lay], 16, True, length=length, accelerator_graph=True, max_num_segments=2)
    dense = dense_from_block_mask(w.block_mask, length)
    n = lay.span
    assert torch.equal(dense[:n, :n], dense_visibility([lay]))
    assert not dense[:n, n:].any() and not dense[n:, :n].any()
    assert dense[n:, n:].all(), "padding rows must attend something, or they divide by zero"


@pytest.mark.parametrize("fail_on", [1, 2], ids=["before_tags", "after_tags"])
def test_failed_graph_stage_retries_the_same_layout(monkeypatch, fail_on):
    first = KERNEL_LAYOUT
    second = layout_of(24, 8, 2, 8, 6)
    w = planned_mask([first], 128, False, length=first.span,
                     accelerator_graph=True, max_num_segments=1)
    stage = w._stage
    calls = 0

    def fail_stage_once(buffer, array):
        nonlocal calls
        calls += 1
        if calls == fail_on:
            raise RuntimeError("stage failed")
        stage(buffer, array)

    monkeypatch.setattr(w, "_stage", fail_stage_once)
    with pytest.raises(RuntimeError, match="stage failed"):
        w.plan([second])
    monkeypatch.setattr(w, "_stage", stage)
    w.plan([second])

    dense = dense_from_block_mask(w.block_mask, first.span)
    assert torch.equal(dense[:second.span, :second.span], dense_visibility([second]))
    assert not dense[:second.span, second.span:].any()
    assert not dense[second.span:, :second.span].any()


# --- negative controls: wrong masks the checks must reject --------------------

def test_plain_block_causal_is_rejected():
    """Block-causal over the whole sequence by chunk admits a noised query of
    chunk 1 reading noised keys of chunk 0, which this mask must not."""
    lay = layout_of(*REF_LAYOUTS["two_chunks_one_clean"])
    stream, chunk = roles(lay)
    blk = np.concatenate([[0] * lay.global_len, 1 + np.arange(lay.noised_len + lay.clean_len) // lay.chunk_len])
    block_causal = blk[None, :] <= blk[:, None]
    q = lay.global_len + lay.chunk_len  # first noised token of chunk 1
    k = lay.global_len  # first noised token of chunk 0
    assert (stream[q], chunk[q], stream[k], chunk[k]) == (1, 1, 1, 0)
    assert block_causal[q, k] and not dense_visibility([lay])[q, k]
    assert "noised block-diagonal" in rule_violations(lay, block_causal)


def test_a_clean_query_reading_noise_is_rejected():
    lay = layout_of(*REF_LAYOUTS["four_chunks_three_clean"])
    wrong = dense_visibility([lay]).numpy().copy()
    stream, _ = roles(lay)
    wrong[np.ix_(stream == 2, stream == 1)] = True
    assert rule_violations(lay, wrong) == ["cross-stream"]
    # the resource's own mask: no clean query sees a noised key
    assert not dense_visibility([lay]).numpy()[np.ix_(stream == 2, stream == 1)].any()


def test_a_global_query_reading_past_the_prefix_is_rejected():
    lay = layout_of(*REF_LAYOUTS["four_chunks_three_clean"])
    wrong = dense_visibility([lay]).numpy().copy()
    wrong[: lay.global_len, :] = True
    assert rule_violations(lay, wrong) == ["globals"]
    assert not dense_visibility([lay]).numpy()[: lay.global_len, lay.global_len:].any()


def test_noise_reading_its_own_clean_chunk_is_rejected():
    """``k <= q`` across streams (clean of the same chunk) would leak the
    target into its own denoising."""
    lay = layout_of(*REF_LAYOUTS["four_chunks_three_clean"])
    stream, chunk = roles(lay)
    wrong = dense_visibility([lay]).numpy().copy()
    same = (stream[:, None] == 1) & (stream[None, :] == 2) & (chunk[:, None] == chunk[None, :])
    wrong[same] = True
    assert rule_violations(lay, wrong) == ["cross-stream"]


# --- element-for-element against the reference --------------------------------

def reference():
    backbone = pytest.importorskip("praxis.backbone")
    if not hasattr(backbone, "teacher_forcing_mask"):
        # ``praxis`` may resolve to M*'s own model package once it exists
        pytest.skip("praxis.backbone is not the Praxis-1 reference")
    return backbone


def reference_mask(g, tpf, cf, nf, clf, block_size=16, clean_past=True):
    slots = [("global", 1, g), ("noised", nf, tpf)] + ([("clean", clf, tpf)] if clf else [])
    return reference().teacher_forcing_mask(slots, cf, clean_past, block_size, "cpu", False)


def reference_predicate(g, tpf, cf, nf, clf, clean_past=True) -> torch.Tensor:
    n = g + (nf + clf) * tpf
    qi, ki = torch.arange(n)[:, None], torch.arange(n)[None, :]
    return reference_mask(g, tpf, cf, nf, clf, clean_past=clean_past).mask_mod(0, 0, qi, ki).expand(n, n)


@pytest.mark.parametrize("full_blocks", [False, True])
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("name", list(REF_LAYOUTS))
def test_block_mask_equals_the_reference_predicate(name, block_size, full_blocks):
    shape = REF_LAYOUTS[name]
    want = reference_predicate(*shape)
    lay = layout_of(*shape)
    got = dense_from_block_mask(planned_mask([lay], block_size, full_blocks).block_mask, lay.span)
    assert torch.equal(got, want), f"{int((got != want).sum())} pairs differ"
    # and with the reference's own tile lists applied, at its block size
    assert torch.equal(dense_from_block_mask(reference_mask(*shape), lay.span), want)


def test_the_decoder_mask_equals_the_reference():
    g, tpf, cf, nf = 5, 4, 2, 6
    want = reference_predicate(g, tpf, cf, nf, 0, clean_past=False)
    assert torch.equal(dense_visibility([ChunkCausalLayout(g, cf * tpf, nf * tpf)]), want)


def test_the_reference_comparison_can_fail():
    """Block-causal by chunk in place of the resource's mask: the comparison
    above must see the difference."""
    shape = REF_LAYOUTS["two_chunks_one_clean"]
    lay = layout_of(*shape)
    blk = torch.cat([torch.zeros(lay.global_len), 1 + torch.arange(lay.noised_len + lay.clean_len) // lay.chunk_len])
    assert not torch.equal(blk[None, :] <= blk[:, None], reference_predicate(*shape))


# --- flex_attention over the planned mask --------------------------------------

# one length for every compiled call below: CPU inductor compiles per length
KERNEL_LAYOUT = layout_of(37, 8, 2, 8, 6)  # 149 tokens, 4 chunks, prefix and total unaligned


def qkv(n: int, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(n, h, D, generator=gen) for h in (HQ, HKV, HKV))


def sdpa(q, k, v, visible):
    q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=visible, scale=D ** -0.5, enable_gqa=True)
    return out[0].transpose(0, 1)


@pytest.mark.parametrize("full_blocks", [False, True], ids=["partial_tiles", "full_tiles"])
def test_flex_matches_dense_sdpa_under_the_same_mask(full_blocks):
    """Both tile splits: TRITON runs partial tiles only, FLASH full ones too."""
    lay = KERNEL_LAYOUT
    q, k, v = qkv(lay.span)
    w = planned_mask([lay], 128, full_blocks)
    got = w.run(q, k, v)
    want = sdpa(q, k, v, dense_visibility([lay]))
    err = (got - want).abs().max().item()
    print(f"flex (compiled, CPU, fp32, full_blocks={full_blocks}) vs dense SDPA: max abs diff {err:.3e}")
    assert err < 1e-5


def test_eager_flex_agrees_with_compiled():
    """The mask_mod is the exact predicate, so unlike ``attn.flex``'s ring mask
    the eager path is not wrong, only slow."""
    lay = KERNEL_LAYOUT
    q, k, v = qkv(lay.span, seed=5)
    w = planned_mask([lay], 128, cc.full_blocks_for_backend())
    qh, kh, vh = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
    eager = flex_attention(qh, kh, vh, block_mask=w.block_mask, scale=D ** -0.5, enable_gqa=True)
    assert (eager[0].transpose(0, 1) - w.run(q, k, v)).abs().max().item() < 1e-5


def test_flex_matches_dense_sdpa_under_the_reference_mask():
    shape = (37, 8, 2, 8, 6)
    lay = layout_of(*shape)
    assert lay == KERNEL_LAYOUT
    want_mask = reference_predicate(*shape)
    q, k, v = qkv(lay.span, seed=1)
    got = planned_mask([lay], 128, cc.full_blocks_for_backend()).run(q, k, v)
    want = sdpa(q, k, v, want_mask)
    err = (got - want).abs().max().item()
    print(f"flex (compiled, CPU, fp32) vs dense SDPA under the reference mask: max abs diff {err:.3e}")
    assert err < 1e-5
    # and against flex under the reference's own BlockMask: the same kernel
    ref_bm = reference_mask(*shape, block_size=128)
    qh, kh, vh = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
    ref_out = cc.flex_attention_masked(qh, kh, vh, block_mask=ref_bm, scale=D ** -0.5, enable_gqa=True)
    same_kernel = (got - ref_out[0].transpose(0, 1)).abs().max().item()
    print(f"flex under the resource's BlockMask vs under the reference's: max abs diff {same_kernel:.3e}")
    assert same_kernel < 1e-5


def test_attention_output_respects_the_mask():
    """Perturbing a token's value moves exactly the queries that see it."""
    lay = KERNEL_LAYOUT
    q, k, v = qkv(lay.span, seed=2)
    w = planned_mask([lay], 128, cc.full_blocks_for_backend())
    base = w.run(q, k, v)
    stream, chunk = roles(lay)
    # the first clean token of chunk 1
    j = int(np.flatnonzero((stream == 2) & (chunk == 1))[0])
    v2 = v.clone()
    v2[j] += 10.0
    moved = (w.run(q, k, v2) - base).abs().amax(dim=(1, 2)) > 0
    assert torch.equal(moved, dense_visibility([lay])[:, j])


def test_graph_mode_run_zeroes_padding_rows_and_matches_eager():
    lay = layout_of(24, 8, 2, 8, 6)
    length = KERNEL_LAYOUT.span  # same compiled length; the tail is padding
    q, k, v = qkv(length, seed=3)
    q[lay.span:] = float("inf")  # padding rows hold whatever a previous replay left
    w = planned_mask([lay], 128, cc.full_blocks_for_backend(), length=length,
                     accelerator_graph=True, max_num_segments=1)
    out = w.run(q, k, v)
    assert (out[lay.span:] == 0).all()
    want = sdpa(q[:lay.span], k[:lay.span], v[:lay.span], dense_visibility([lay]))
    assert (out[:lay.span] - want).abs().max().item() < 1e-5
    with pytest.raises(ValueError, match="bucket's width"):
        w.run(q[:lay.span], k[:lay.span], v[:lay.span])


# --- the resource --------------------------------------------------------------

def spec(**config) -> RaggedChunkCausalAttentionSpec:
    return RaggedChunkCausalAttentionSpec(
        resource_key="tf_attn", nodes={"encoder"},
        config=RaggedAttentionConfig(num_qo_heads=HQ, num_kv_heads=HKV, head_dim=D, max_segments_per_request=1,
                                     **config),
    )


def manager(**config) -> FlexChunkCausalManager:
    return build_resource(spec(**config), EngineResourceInfo(device=torch.device("cpu"), kv_dtype=torch.float32))


def ctx(n, lease=None) -> StepContext:
    return StepContext(request_ids=list(range(n)), graph_walk="encode", slot=0, capture=False, slot_lease=lease)


def step(layouts, label="main") -> ChunkCausalAttentionStep:
    return ChunkCausalAttentionStep(
        segments=tuple(Segment(request_id=f"r{i}", label=label, span=lay.span) for i, lay in enumerate(layouts)),
        layouts=tuple(layouts),
    )


def test_plan_then_run_through_the_resource():
    m = manager()
    assert isinstance(m, FlexChunkCausalManager) and m.force_double_buffer and m.supports_preplan
    lay = KERNEL_LAYOUT
    m.plan(step([lay]), ctx(1))
    assert torch.equal(dense_from_block_mask(m._wrapper("main").block_mask, lay.span), dense_visibility([lay]))
    q, k, v = qkv(lay.span, seed=4)
    assert (m.run(q, k, v, label="main") - sdpa(q, k, v, dense_visibility([lay]))).abs().max() < 1e-5


def test_a_lease_plans_a_wrapper_sized_by_the_bucket():
    m = manager()
    lease = SlotLease(slot=0, bucket=BucketKey(graph_walk="encode", bs=2, num_tokens=400))
    a, b = layout_of(24, 8, 2, 4, 2), layout_of(13, 4, 3, 7, 4)
    m.plan(step([a, b]), ctx(2, lease=lease))
    w = m._wrapper("main")
    assert w.accelerator_graph and w.max_total_tokens == 400 and w.max_num_segments == 2
    dense = dense_from_block_mask(w.block_mask, 400)
    n = a.span + b.span
    assert torch.equal(dense[:n, :n], dense_from_block_mask(planned_mask([a, b], m.block_size, False).block_mask, n))


def test_the_same_layout_stages_nothing_and_a_dropped_preplan_restages():
    w = planned_mask([KERNEL_LAYOUT], 128, False)
    mask = w.block_mask
    w.plan([KERNEL_LAYOUT])
    assert w.block_mask is mask
    w.forget_plan()
    w.plan([KERNEL_LAYOUT])
    assert w.block_mask is not mask


def test_a_layout_must_span_its_segment():
    bad = ChunkCausalAttentionStep(
        segments=(Segment(request_id="r0", label="main", span=10),), layouts=(ChunkCausalLayout(4, 2, 4),),
    )
    with pytest.raises(ValueError, match="spans 8 tokens, its segment 10"):
        manager().plan(bad, ctx(1))


def test_a_step_without_layouts_is_refused():
    with pytest.raises(ValueError, match="0 layouts for 1 segments"):
        manager().plan(ChunkCausalAttentionStep(segments=(Segment("r0", "main", 8),)), ctx(1))


@pytest.mark.parametrize(
    ("override", "match"),
    [
        (dict(backend=RaggedAttnBackend.FLASHINFER), "backend must be 'auto'"),
        (dict(flashinfer_backend="fa2"), "flashinfer_backend"),
        (dict(compile_op=True), "compile_op"),
        (dict(workspace_mb=64), "workspace_mb"),
    ],
)
def test_fields_flex_does_not_read_are_refused(override, match):
    with pytest.raises(ValueError, match=match):
        manager(**override)


def test_the_block_size_follows_the_backend_and_arch(monkeypatch):
    cpu = torch.device("cpu")
    assert cc.resolve_mask_block_size(None, cpu) == 128
    assert cc.resolve_mask_block_size(64, cpu) == 64
    monkeypatch.setattr(cc, "_FLEX_BACKEND", "FLASH")
    monkeypatch.setattr(cc.current_platform, "cuda_capability", lambda device=None: (10, 0))
    assert cc.resolve_mask_block_size(None, cpu) == 256
    assert cc.resolve_mask_block_size(64, cpu) == 256
    monkeypatch.setattr(cc.current_platform, "cuda_capability", lambda device=None: (9, 0))
    assert cc.resolve_mask_block_size(None, cpu) == 128
    assert cc.full_blocks_for_backend()
