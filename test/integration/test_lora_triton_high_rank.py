"""The Triton LoRA kernels at every launch config ``triton_backend._tiles`` picks: decode and
prefill widths, a rank held in one tile and ranks tiled with their split partials reduced first,
ranks off the tile, in bf16, fp16 and fp32, against the fp32 reference ops over mixed slots with
ragged per-slot rows, a rank-0 slot and base rows; the configs under smaller shared-memory
limits; the finite ``b`` the expand requires past a slot's rank; and tiled ranks captured at
decode and prefill widths. GPU only.

    pytest test/integration/test_lora_triton_high_rank.py -v
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)
pytest.importorskip("triton")

import mstar.engine.resources.lora.triton_backend as tb  # noqa: E402  (registers the CUDA kernels)
from mstar.engine.resources.lora import backend as ref_ops  # noqa: E402
from mstar.engine.resources.lora import ops  # noqa: E402

SLOTS = 6
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
# a live row against the reference, relative to that row's delta: about 2x the worst row over
# this file's cases on an H200 (1.1e-2, 1.6e-3, 7.1e-3; fp32 runs its dots in tf32)
TOLERANCE = {torch.bfloat16: 2.5e-2, torch.float16: 4e-3, torch.float32: 1.5e-2}


def check_rows(got, y0, want, live, dtype, what):
    """Every live row against the reference added to the same ``y`` and rounded the same way,
    relative to the norm of that row's delta."""
    delta = want - y0
    rounded = (y0.to(dtype).float() + delta).to(dtype).float()
    err = (got.float() - rounded).norm(dim=1) / delta.norm(dim=1).clamp_min(1e-6)
    worst = err[live].max().item() if live.any() else 0.0
    assert worst < TOLERANCE[dtype], f"{what}: worst row relative error {worst:.3g}"


def ragged_layout(tokens, g):
    """Tokens over the slots in uneven runs (one slot takes most), as ``(token slots, order, indptr)``,
    the slots int32 as the ops take them."""
    weights = torch.tensor([1.0, 6.0, 2.0, 1.0, 0.5, 3.0], device="cuda")
    token_slots = torch.multinomial(weights, tokens, replacement=True, generator=g)
    order = torch.argsort(token_slots, stable=True).to(torch.int32)
    indptr = torch.zeros(SLOTS + 1, dtype=torch.int32, device="cuda")
    indptr[1:] = torch.cumsum(torch.bincount(token_slots, minlength=SLOTS), 0)
    return token_slots.to(torch.int32), order, indptr


def case(max_rank, in_features, slices, tokens, seed):
    """Weights, slots and rows. ``b`` holds finite values past each slot's rank, where the pool
    holds zeros, so every case also checks that they meet zeros in ``h``."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    # slot 0 is the zero adapter and slot 3 has rank 0; the others are at, under and off the tile
    ranks = torch.tensor([0, max_rank, max(1, max_rank // 4), 0, max(1, max_rank // 2 + 3), max_rank - 1 or 1],
                         dtype=torch.int32, device="cuda")
    a = torch.randn(SLOTS, len(slices) * max_rank, in_features, generator=g, device="cuda") / in_features ** 0.5
    b = torch.randn(SLOTS, sum(slices), max_rank, generator=g, device="cuda") / max_rank ** 0.5
    scalings = torch.rand(SLOTS, generator=g, device="cuda") + 0.5
    token_slots, order, indptr = ragged_layout(tokens, g)
    x = torch.randn(tokens, in_features, generator=g, device="cuda")
    y0 = torch.randn(tokens, sum(slices), generator=g, device="cuda")
    return a, b, scalings, ranks, token_slots, order, indptr, x, y0


def reference(a, b, scalings, ranks, token_slots, x, y0, slices, dtype):
    """The fp32 reference on the inputs as the kernels see them in ``dtype``."""
    a, b, x = (t.to(dtype).float() for t in (a, b, x))
    want = y0.clone()
    h = ref_ops.shrink_reference(x, a, token_slots, ranks, len(slices))
    ref_ops.expand_reference_(want, h, b, token_slots, ranks, scalings, list(slices))
    return want


SHAPES = [  # max_rank, in_features, slices: Phi-4-multimodal's targets, then widths off the tiles
    (256, 3072, (3072, 1024, 1024)), (320, 3072, (8192, 8192)), (320, 8192, (3072,)), (256, 3072, (3072,)),
    (100, 1000, (700, 300)), (16, 3072, (3072, 1024, 1024)), (8, 8192, (3072,)), (64, 1000, (700,)),
]


@pytest.mark.parametrize("dtype", DTYPES.values(), ids=DTYPES.keys())
@pytest.mark.parametrize("tokens", [1, 8, 33, 64, 128, 300, 2048])
@pytest.mark.parametrize("max_rank,in_features,slices", SHAPES)
def test_apply_matches_the_reference(max_rank, in_features, slices, tokens, dtype):
    """``lora_apply_`` (whose shrink writes zeros past each slot's rank for its expand) against
    the fp32 reference, row by row; base rows and the rank-0 slot's rows bitwise untouched."""
    a, b, scalings, ranks, token_slots, order, indptr, x, y0 = case(max_rank, in_features, slices, tokens, tokens)
    want = reference(a, b, scalings, ranks, token_slots, x, y0, slices, dtype)
    got = y0.to(dtype, copy=True)
    ops.lora_apply_(got, x.to(dtype), a.to(dtype), b.to(dtype), order, indptr, token_slots, ranks, scalings,
                    list(slices))
    check_rows(got, y0, want, ranks.long()[token_slots] > 0, dtype, f"apply r{max_rank} T{tokens}")
    untouched = (token_slots == 0) | (token_slots == 3)
    assert torch.equal(got[untouched], y0.to(dtype)[untouched])


@pytest.mark.parametrize("dtype", DTYPES.values(), ids=DTYPES.keys())
@pytest.mark.parametrize("tokens", [5, 300])
@pytest.mark.parametrize("max_rank,in_features,slices", [(320, 3072, (3072, 1024, 1024)), (100, 1000, (700, 300))])
def test_shrink_then_expand_with_anything_past_the_rank(max_rank, in_features, slices, tokens, dtype):
    """The separate ops: ``lora_expand_`` masks ``h`` past each slot's rank itself, since its ``h``
    may come from anywhere (here, inf where the expand must not read)."""
    a, b, scalings, ranks, token_slots, order, indptr, x, y0 = case(max_rank, in_features, slices, tokens, 7)
    want = reference(a, b, scalings, ranks, token_slots, x, y0, slices, dtype)
    h = ops.lora_shrink(x.to(dtype), a.to(dtype), order, indptr, token_slots, ranks, len(slices))
    rank = ranks.long()[token_slots]
    col = torch.arange(h.shape[1], device="cuda") % max_rank
    h = torch.where(col[None, :] < rank[:, None], h, float("inf"))
    got = y0.to(dtype, copy=True)
    ops.lora_expand_(got, h, b.to(dtype), order, indptr, token_slots, ranks, scalings, list(slices))
    check_rows(got, y0, want, rank > 0, dtype, f"shrink + expand r{max_rank} T{tokens}")


@pytest.mark.parametrize("tokens", [8, 300])
def test_shrink_zeros_past_each_slots_rank(tokens):
    """What ``lora_apply_``'s expand relies on: every split's partial holds zeros from a live
    row's slot rank up to the max rank, whatever the buffer held before."""
    max_rank, slices = 320, (3072, 1024, 1024)
    a, _, _, ranks, token_slots, order, indptr, x, _ = case(max_rank, 3072, slices, tokens, 3)
    tiles = tb._tiles(tokens, max_rank, torch.bfloat16, x.device).shrink
    splits, k_per_split = 3, 1024
    h = torch.full((splits, tokens, len(slices) * max_rank), float("nan"), device="cuda")
    tb._shrink_into(h, x.to(torch.bfloat16), a.to(torch.bfloat16), order, indptr, ranks, len(slices), splits,
                    k_per_split, tiles)
    want = ref_ops.shrink_reference(x.to(torch.bfloat16).float(), a.to(torch.bfloat16).float(), token_slots, ranks,
                                    len(slices))
    live = ranks.long()[token_slots] > 0
    err = (h.sum(0)[live] - want[live]).norm(dim=1) / want[live].norm(dim=1)
    assert err.max() < TOLERANCE[torch.bfloat16], f"split partials: worst row relative error {err.max():.3g}"
    col = torch.arange(h.shape[2], device="cuda") % max_rank
    past = col[None, :] >= ranks.long()[token_slots][:, None]
    assert not h[:, live[:, None] & past].any(), "a partial past a slot's rank is not zero"


@pytest.mark.parametrize("op", ["apply", "shrink+expand"])
@pytest.mark.parametrize("tokens", [8, 300])
def test_b_past_a_slots_rank_must_be_finite(op, tokens):
    """The contract ``ops.py`` states: the expand reads ``b[slot, :, rank:]`` (up to the max
    rank, against zeros in ``h``), so finite values there change nothing, bit for bit, and a NaN
    there reaches that slot's rows. The pool's zero padding meets it."""
    max_rank, in_features, slices = 320, 3072, (3072, 1024, 1024)
    a, b, scalings, ranks, token_slots, order, indptr, x, y0 = case(max_rank, in_features, slices, tokens, 5)
    a, x, y0 = a.to(torch.bfloat16), x.to(torch.bfloat16), y0.to(torch.bfloat16)
    past = torch.arange(max_rank, device="cuda") >= ranks[:, None, None]

    def run(b_):
        y = y0.clone()
        if op == "apply":
            ops.lora_apply_(y, x, a, b_.to(torch.bfloat16), order, indptr, token_slots, ranks, scalings, list(slices))
        else:
            h = ops.lora_shrink(x, a, order, indptr, token_slots, ranks, len(slices))
            ops.lora_expand_(y, h, b_.to(torch.bfloat16), order, indptr, token_slots, ranks, scalings, list(slices))
        return y

    zero_padded = run(torch.where(past, 0.0, b))
    assert torch.equal(run(torch.where(past, 1e4, b)), zero_padded)
    rank = ranks.long()[token_slots]
    short = (rank > 0) & (rank < max_rank)
    nan = run(torch.where(past, float("nan"), b))
    assert nan[short].isnan().any(1).all(), "a NaN in b past a slot's rank no longer reaches its rows"
    assert torch.equal(nan[~short], zero_padded[~short])


# per-block shared memory (shared_memory_per_block_optin) of an H200, an A100, sm86 / sm89 and sm75
SMEM_LIMITS = {"sm90": 232448, "sm80": 166912, "sm86": 101376, "sm75": 65536}


def launch_both(tokens, max_rank, dtype, cfg):
    """Every kernel ``lora_apply_`` and the standalone ops can launch at ``cfg``, each result
    checked against the reference: the shrink and expand over one split; the shrink into three
    fp32 split partials, which the expand sums itself when the rank fits one tile and the reduce
    sums first when it doesn't; and the standalone expand. Returns ``(kind, label, compiled
    kernel)`` for each launch."""
    in_features, slices = 3072, (3072, 1024, 1024)
    a, b, scalings, ranks, token_slots, order, indptr, x, y0 = case(max_rank, in_features, slices, tokens, 9)
    want = reference(a, b, scalings, ranks, token_slots, x, y0, slices, dtype)
    live = ranks.long()[token_slots] > 0
    x, a, b = x.to(dtype), a.to(dtype), b.to(dtype)
    tiled = tb._rank_tiles(max_rank)[1] > 1
    launched = []
    for splits, zeros_past_rank in [(1, True), (3, True), (1, False)]:
        name = f"{splits} split{'s' if splits > 1 else ''}" + ("" if zeros_past_rank else ", standalone expand")
        h = torch.empty(splits, tokens, a.shape[1], device="cuda", dtype=dtype if splits == 1 else torch.float32)
        launched.append(("shrink", name, tb._shrink_into(h, x, a, order, indptr, ranks, len(slices), splits,
                                                        in_features // splits, cfg.shrink)))
        if splits > 1 and tiled:
            parts, h = h, torch.empty(1, tokens, a.shape[1], device="cuda", dtype=dtype)
            launched.append(("reduce", name, tb._reduce_into(h, parts)))
        got = y0.to(dtype, copy=True)
        launched.append(("expand", name, tb._expand_from(got, h, b, order, indptr, ranks, scalings, list(slices),
                                                         cfg.expand, zeros_past_rank=zeros_past_rank)))
        check_rows(got, y0, want, live, dtype, f"r{max_rank} T{tokens}, {name}")
    return launched


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("gpu", SMEM_LIMITS)
def test_launch_config_fits_shared_memory(gpu, dtype, monkeypatch):
    """Under each device's shared-memory limit (mocked), the config falls back to fewer stages or
    smaller tiles until its estimate fits, compiles within the limit and still matches the
    reference."""
    limit = SMEM_LIMITS[gpu]
    monkeypatch.setattr(tb, "_smem_limit", lambda index: limit)
    device = torch.device("cuda", torch.cuda.current_device())
    for tokens, max_rank in [(32, 16), (32, 64), (32, 320), (300, 16), (2048, 64), (2048, 320)]:
        cfg = tb._tiles(tokens, max_rank, dtype, device)
        block_r = tb._rank_tiles(max_rank)[0]
        est = tb._shrink_smem(cfg.shrink, block_r, dtype.itemsize), tb._expand_smem(cfg.expand, block_r, dtype.itemsize)
        assert max(est) <= limit, f"{cfg} needs {est} bytes, over {limit}"
        for kind, name, kernel in launch_both(tokens, max_rank, dtype, cfg):
            assert kernel.metadata.shared <= limit, f"{kind} ({name}): {kernel.metadata.shared} > {limit}"
        if gpu == "sm90":  # an H200 runs the preferred configs in either dtype
            prefill = cfg.shrink.m > tb._DECODE_SHRINK[0].m
            assert cfg.shrink == (tb._PREFILL_SHRINK if prefill else tb._DECODE_SHRINK)[0]
            assert cfg.expand == (tb._PREFILL_EXPAND if prefill else tb._DECODE_EXPAND)[0]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("max_rank", [16, 32, 64, 320])
def test_every_candidate_compiles_within_its_estimate(max_rank, dtype):
    """The estimates a config is chosen by bound the compiled kernels' shared memory, for every
    candidate (a Triton upgrade that allocates more fails here, not on a smaller GPU)."""
    block_r = tb._rank_tiles(max_rank)[0]
    for shrinks, expands in ((tb._DECODE_SHRINK, tb._DECODE_EXPAND), (tb._PREFILL_SHRINK, tb._PREFILL_EXPAND)):
        for i in range(max(len(shrinks), len(expands))):
            cfg = tb._Config(shrinks[min(i, len(shrinks) - 1)], expands[min(i, len(expands) - 1)])
            est = {"shrink": tb._shrink_smem(cfg.shrink, block_r, dtype.itemsize),
                   "expand": tb._expand_smem(cfg.expand, block_r, dtype.itemsize),
                   # no config is chosen for the reduce, so it must fit the smallest device
                   "reduce": min(SMEM_LIMITS.values())}
            for kind, name, kernel in launch_both(40, max_rank, dtype, cfg):
                assert kernel.metadata.shared <= est[kind], (
                    f"{kind} ({name}) at {cfg}: {kernel.metadata.shared} > {est[kind]}")


def test_an_fp32_pool_on_a_small_gpu_through_the_op(monkeypatch):
    """``lora_apply_`` in fp32 at prefill width and a tiled rank, as an sm86 / sm89 GPU would pick
    its configs, against the reference."""
    monkeypatch.setattr(tb, "_smem_limit", lambda index: SMEM_LIMITS["sm86"])
    max_rank, in_features, slices, tokens = 320, 3072, (3072, 1024, 1024), 2048
    a, b, scalings, ranks, token_slots, order, indptr, x, y0 = case(max_rank, in_features, slices, tokens, 13)
    want = reference(a, b, scalings, ranks, token_slots, x, y0, slices, torch.float32)
    got = y0.clone()
    ops.lora_apply_(got, x, a, b, order, indptr, token_slots, ranks, scalings, list(slices))
    check_rows(got, y0, want, ranks.long()[token_slots] > 0, torch.float32, "fp32 at sm86's limit")


def capture(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


@pytest.mark.parametrize("op", ["apply", "shrink+expand"])
@pytest.mark.parametrize("tokens", [24, 300])
def test_a_high_rank_inside_a_captured_graph(tokens, op):
    """``lora_apply_``, and the standalone ``lora_shrink`` + ``lora_expand_``, at max rank 320,
    decode and prefill tiles, captured once on the zero adapter and replayed for slot mixes and
    ranks that change between replays: bitwise equal to eager."""
    g = torch.Generator(device="cuda").manual_seed(11)
    max_rank, in_features, slices = 320, 3072, [3072, 1024, 1024]
    bf16 = dict(device="cuda", dtype=torch.bfloat16)
    a = (torch.randn(SLOTS, len(slices) * max_rank, in_features, generator=g, device="cuda") / 55).to(torch.bfloat16)
    b = (torch.randn(SLOTS, sum(slices), max_rank, generator=g, device="cuda") / 18).to(torch.bfloat16)
    scalings = torch.rand(SLOTS, generator=g, device="cuda") + 0.5
    ranks = torch.zeros(SLOTS, dtype=torch.int32, device="cuda")
    token_slots, order, indptr = ragged_layout(tokens, g)
    token_slots.zero_()
    indptr.zero_()
    indptr[1:] = tokens  # every row on the zero adapter
    x, y = torch.zeros(tokens, in_features, **bf16), torch.zeros(tokens, sum(slices), **bf16)

    def run():
        out = y.clone()
        if op == "apply":
            ops.lora_apply_(out, x, a, b, order, indptr, token_slots, ranks, scalings, slices)
        else:
            h = ops.lora_shrink(x, a, order, indptr, token_slots, ranks, len(slices))
            ops.lora_expand_(out, h, b, order, indptr, token_slots, ranks, scalings, slices)
        return out

    graph, out = capture(run)
    for mix_ranks, seed in (([0, 320, 33, 0, 256, 100], 1), ([0, 16, 320, 64, 0, 319], 2), ([0] * SLOTS, 3)):
        gm = torch.Generator(device="cuda").manual_seed(seed)
        ranks.copy_(torch.tensor(mix_ranks, dtype=torch.int32))
        t, o, p = ragged_layout(tokens, gm)
        token_slots.copy_(t)
        order.copy_(o)
        indptr.copy_(p)
        x.copy_(torch.randn(x.shape, generator=gm, device="cuda").to(torch.bfloat16))
        y.copy_(torch.randn(y.shape, generator=gm, device="cuda").to(torch.bfloat16))
        graph.replay()
        eager = run()
        torch.cuda.synchronize()
        assert torch.equal(out, eager), f"replay differs from eager for ranks {mix_ranks}"
        if not any(mix_ranks):
            assert torch.equal(out, y)
