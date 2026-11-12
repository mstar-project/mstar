"""Numerical equivalence between M*'s Praxis-1 components and the reference.

Unlike Pi0.5, whose reference is too heavy to stand up inside a unit test
(``test_pi05_reference_equivalence`` re-implements openpi's math inline), the
Praxis reference package imports under a CPU-only torch and needs no weights.
These tests therefore compare against the real thing rather than against a
second copy of our own reading of it, which is the stronger check: a
misreading of the reference cannot pass on both sides at once.

The reference is a checkout rather than a published package, so the tests
skip unless ``praxis`` is importable. Put the clone on ``PYTHONPATH`` to run
them::

    PYTHONPATH=/path/to/praxis-1-oss pytest test/modular/test_praxis1_reference_equivalence.py

Every comparison below is bit-exact, and each group carries a negative
control that must diverge. A control that cannot fail measures nothing: the
first one written here passed trivially, because at exactly the fit size the
two canvas rules collapse onto the same trim, and it had to be moved to the
native frame to mean anything.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip(
    "praxis",
    reason="the Praxis-1 reference checkout is not on PYTHONPATH",
)

from praxis import bimanual as ref_bi  # noqa: E402
from praxis import robot_io as ref_io  # noqa: E402
from praxis import single_arm as ref_single  # noqa: E402

from mstar.model.praxis1.components import action_rows as port_rows  # noqa: E402
from mstar.model.praxis1.components import canvas as port_canvas  # noqa: E402

TILE_H, TILE_W = 224, 416


def _frames(seed: int, t: int, h: int, w: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(t, h, w, 3), dtype=np.uint8)


def _chw(frames: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# The canvas
# ---------------------------------------------------------------------------


def test_fit_size_matches_reference():
    assert port_canvas.single_arm_fit_size(TILE_W, (720, 1280)) == (
        ref_single.fit_resize_size(TILE_W)
    )


@pytest.mark.parametrize("antialias", [True, False])
@pytest.mark.parametrize(
    "t,h,w",
    [
        (1, 234, 416),  # what the client is asked to send: a trim only
        (3, 234, 416),
        (1, 720, 1280),  # the native frame: a real resize
        (2, 480, 854),  # an off-aspect source, so the crop runs
    ],
)
def test_single_arm_tile_is_bit_exact(t, h, w, antialias):
    frames = _frames(seed=t * 1000 + h, t=t, h=h, w=w)
    want = ref_single._tile_view(frames, TILE_H, TILE_W, 0.03, antialias, None)
    got = port_canvas.tile_view_single_arm(
        _chw(frames), TILE_H, TILE_W, antialias=antialias
    )
    assert got.dtype == want.dtype == torch.uint8
    assert torch.equal(got, want)


@pytest.mark.parametrize("h,w", [(1200, 1920), (256, 416)])
def test_bimanual_row_sizes_match_reference(h, w):
    native = [(h, w)] * 3
    assert port_canvas.bimanual_row_sizes(native) == ref_bi.fit_row_sizes(native)


def test_canvas_control_the_two_rules_differ_on_the_native_frame():
    """The gate must be able to fail.

    At the fit size both rules reduce to the same five-row trim, so a control
    taken there proves nothing. On the native frame one rule crops to the
    tile aspect and the other to the view's, and the answers differ.
    """
    frames = _frames(seed=11, t=1, h=720, w=1280)
    chw = _chw(frames)
    fit = port_canvas.tile_view_single_arm(chw, TILE_H, TILE_W, antialias=True)
    plain = port_canvas.tile_view_single_arm(chw, TILE_H, TILE_W, antialias=False)
    assert not torch.equal(fit, plain)


def test_canvas_lays_views_side_by_side():
    tiles = [torch.full((2, 3, TILE_H, TILE_W), v, dtype=torch.uint8) for v in (7, 9)]
    canvas = port_canvas.build_canvas(tiles)
    assert canvas.shape == (1, 3, 2, TILE_H, 2 * TILE_W)
    assert canvas.dtype == torch.float32
    # Values stay in [0, 255]: the VAE does its own [-1, 1] conversion.
    assert canvas[..., :TILE_W].eq(7).all() and canvas[..., TILE_W:].eq(9).all()


# ---------------------------------------------------------------------------
# The action rows
# ---------------------------------------------------------------------------

RIGS = [
    ("single_arm", 10, ref_single.Q99_DIMS, ref_single.GRIP_DIMS),
    ("bimanual", 20, ref_bi.Q99_DIMS, ref_bi.GRIP_DIMS),
]


def _stats(dim: int, seed: int):
    rng = np.random.default_rng(seed)
    q01 = rng.normal(-1, 0.5, size=dim).astype(np.float32)
    q99 = q01 + np.abs(rng.normal(2, 0.5, size=dim)).astype(np.float32)
    return q01, q99


@pytest.mark.parametrize("rig,dim,q99_dims,grip_dims", RIGS)
def test_row_maps_are_bit_exact(rig, dim, q99_dims, grip_dims):
    q01, q99 = _stats(dim, seed=3)
    rows = np.random.default_rng(5).normal(0, 1.4, size=(24, dim)).astype(np.float32)

    rows_t = torch.from_numpy(rows)
    dims_t = torch.as_tensor(q99_dims, dtype=torch.long)
    grip_t = torch.as_tensor(grip_dims, dtype=torch.long)
    q01_t, q99_t = torch.from_numpy(q01), torch.from_numpy(q99)

    stats = ref_io.NormStats(
        action_q01=q01,
        action_q99=q99,
        proprio_mean=np.zeros(dim, np.float32),
        proprio_std=np.ones(dim, np.float32),
    )

    pairs = [
        (port_rows.clip_normalized(rows_t, dims_t),
         ref_io.clip_normalized(rows, q99_dims)),
        (port_rows.clip_grip(rows_t, grip_t),
         ref_io.clip_grip(rows, grip_dims)),
        (port_rows.q99_denormalize(rows_t, q01_t, q99_t, dims_t),
         ref_io.q99_denormalize(rows, q01, q99, q99_dims)),
        (port_rows.q99_normalize(rows_t, q01_t, q99_t, dims_t),
         ref_io.q99_normalize(rows, q01, q99, q99_dims)),
        (port_rows.denormalize_rows(rows_t, q01_t, q99_t, dims_t),
         ref_io.denormalize_rows(rows, stats, q99_dims)),
    ]
    for got, want in pairs:
        assert torch.equal(got, torch.from_numpy(np.ascontiguousarray(want)))


@pytest.mark.parametrize("rig,dim,q99_dims,grip_dims", RIGS)
def test_proprio_normalize_floors_a_dead_std(rig, dim, q99_dims, grip_dims):
    rng = np.random.default_rng(13)
    rows = rng.normal(0, 1, size=(24, dim)).astype(np.float32)
    mean = rng.normal(0, 1, size=dim).astype(np.float32)
    std = np.abs(rng.normal(1, 0.4, size=dim)).astype(np.float32)
    std[0] = 0.0  # a dim that never moved in training
    got = port_rows.proprio_normalize(
        torch.from_numpy(rows), torch.from_numpy(mean), torch.from_numpy(std)
    )
    want = ref_io.proprio_normalize(rows, mean, std)
    assert torch.equal(got, torch.from_numpy(np.ascontiguousarray(want)))
    assert torch.isfinite(got).all()


@pytest.mark.parametrize("rig,dim,q99_dims,grip_dims", RIGS)
@pytest.mark.parametrize("depth,first_chunk", [(0, True), (1, False)])
def test_token_packing_round_trip(rig, dim, q99_dims, grip_dims, depth, first_chunk):
    n_rows = 21 if depth == 0 else 24
    rows = np.random.default_rng(17).normal(0, 1, size=(n_rows, dim)).astype(np.float32)

    want_packed = ref_io.pack_action_tokens(rows, first_chunk=first_chunk)
    got_packed = port_rows.pack_action_rows(torch.from_numpy(rows), first_chunk)
    assert torch.equal(got_packed, torch.from_numpy(np.ascontiguousarray(want_packed)))

    tokens = torch.from_numpy(np.ascontiguousarray(want_packed))[None]
    want_rows = ref_io.action_rows(tokens, depth, 4, dim)
    got_rows = port_rows.unpack_action_tokens(tokens, depth, dim)
    assert torch.equal(got_rows, torch.from_numpy(np.ascontiguousarray(want_rows)))

    if depth != 0:
        # Only a later chunk round trips exactly: the first chunk's token 0
        # repeats row 0 and the three repeats are dropped on the way back.
        assert torch.equal(got_rows, torch.from_numpy(rows))


def test_rows_control_the_anchor_rule_changes_the_answer():
    """Depth must matter, or the anchor special case is untested."""
    rows = np.random.default_rng(19).normal(0, 1, size=(21, 10)).astype(np.float32)
    tokens = torch.from_numpy(ref_io.pack_action_tokens(rows, first_chunk=True))[None]
    assert port_rows.unpack_action_tokens(tokens, 0, 10).shape == (21, 10)
    assert port_rows.unpack_action_tokens(tokens, 1, 10).shape == (24, 10)
