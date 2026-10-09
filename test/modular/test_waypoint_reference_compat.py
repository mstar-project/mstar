"""``WaypointConfig.reference_compat``: what it changes, and what it must not.

The flag reproduces the reference's lossy derived state so the parity gate can be
bit-exact (``test_waypoint_reference_equivalence.py`` is where that is measured
against the reference itself). Here the two positions are pinned against each other
and against the exact arithmetic, on CPU, with no checkpoint:

  * flag off -- the tables are the mathematically exact fp32 expressions;
  * flag on  -- they are those same tables round-tripped through bf16, which is what
    ``NoCastModule._apply`` leaves behind;
  * the two differ, in every table and through a built DiT.

A flag that silently does nothing in either position is the failure this file exists
to catch.
"""

from dataclasses import replace

import pytest
import torch

from mstar.model.waypoint.components.layers import NoiseConditioner, bf16_roundtrip
from mstar.model.waypoint.components.rope import OrthoRoPEAngles
from mstar.model.waypoint.config import WaypointConfig, waypoint_1_5_1b_720p
from mstar.model.waypoint.weight_loader import build_waypoint_dit

CPU = torch.device("cpu")
SIGMAS = waypoint_1_5_1b_720p().scheduler_sigmas


def _small(**overrides) -> WaypointConfig:
    """A DiT small enough to materialize in a test. d_head stays 16, so OrthoRoPE's
    x/y/t band split is still exercised."""
    return WaypointConfig(
        n_layers=2, n_heads=4, n_kv_heads=2, d_model=64,
        tokens_per_frame=8, height=2, width=4, **overrides,
    )


def _exact_rope_tables(config: WaypointConfig):
    """``world_engine/src/model/attn.py::OrthoRoPEAngles.__init__``, transcribed, so
    the exact path is checked against the reference's expression rather than against
    the port's own code."""
    d_head = config.d_model // config.n_heads
    d_xy, d_t = d_head // 8, d_head // 4
    max_freq = min(config.height, config.width) * float(config.rope_nyquist_frac)
    xy = (
        torch.linspace(1.0, max_freq / 2, (d_xy + 1) // 2, dtype=torch.float32) * torch.pi
    ).repeat_interleave(2)[:d_xy]
    theta = float(config.rope_theta)
    inv_t = (1.0 / (theta ** (torch.arange(0, d_t, 2, dtype=torch.float32) / d_t))).repeat_interleave(2)
    return xy, inv_t


def _exact_fourier_freq(fourier_dim: int = 512, base: float = 10_000.0):
    """``world_engine/src/model/nn.py::NoiseConditioner.__init__``, transcribed."""
    return torch.logspace(0, -1, steps=fourier_dim // 2, base=base, dtype=torch.float32)


# ---------------------------------------------------------------------------
# The three derived tables
# ---------------------------------------------------------------------------


def test_the_experimental_exact_rope_tables_are_the_fp32_expression():
    config = replace(waypoint_1_5_1b_720p(), reference_compat=False)
    xy, inv_t = OrthoRoPEAngles(config)._tables.get(CPU)
    exact_xy, exact_inv_t = _exact_rope_tables(config)
    assert torch.equal(xy, exact_xy)
    assert torch.equal(inv_t, exact_inv_t)
    assert xy.dtype is torch.float32 and inv_t.dtype is torch.float32


def test_the_default_fourier_freq_is_the_exact_fp32_expression():
    (freq,) = NoiseConditioner(64)._freq.get(CPU)
    assert torch.equal(freq, _exact_fourier_freq())
    assert freq.dtype is torch.float32


def test_reference_compat_bf16_roundtrips_all_three_tables():
    """Off vs on must differ in every table, and on must be exactly the round-trip.
    Both halves matter: the first catches a flag that never reaches the builder, the
    second catches one that quantizes differently from ``NoCastModule._apply``."""
    config = replace(waypoint_1_5_1b_720p(), reference_compat=False)
    compat = replace(config, reference_compat=True)
    tables = {
        "rope_angles.xy": (OrthoRoPEAngles(config)._tables.get(CPU)[0],
                           OrthoRoPEAngles(compat)._tables.get(CPU)[0]),
        "rope_angles.inv_t": (OrthoRoPEAngles(config)._tables.get(CPU)[1],
                              OrthoRoPEAngles(compat)._tables.get(CPU)[1]),
        "denoise_step_emb.freq": (
            NoiseConditioner(64)._freq.get(CPU)[0],
            NoiseConditioner(64, reference_compat=True, cached_sigmas=SIGMAS)._freq.get(CPU)[0],
        ),
    }
    for name, (exact, quantized) in tables.items():
        assert not torch.equal(exact, quantized), f"{name}: reference_compat left the table alone"
        assert torch.equal(quantized, bf16_roundtrip(exact)), f"{name}: not a bf16 round-trip"
        assert quantized.dtype is torch.float32, f"{name}: the fp32 island lost its dtype"


def test_the_flag_reaches_a_built_dit():
    """Through ``build_waypoint_dit``'s meta build, ``cast_serving_dtypes`` and
    ``to_empty`` -- the path the derived tables have to survive outside the module
    tree, and the one the compat conditioner's constructor runs under."""
    exact = build_waypoint_dit(
        _small(reference_compat=False), skip_weight_loading=True, device=CPU
    )
    compat = build_waypoint_dit(
        _small(reference_compat=True), skip_weight_loading=True, device=CPU
    )
    for left, right in zip(exact.rope_angles._tables.get(CPU),
                           compat.rope_angles._tables.get(CPU), strict=True):
        assert not torch.equal(left, right)
        assert torch.equal(bf16_roundtrip(left), right)
    (exact_freq,) = exact.denoise_step_emb._freq.get(CPU)
    (compat_freq,) = compat.denoise_step_emb._freq.get(CPU)
    assert not torch.equal(exact_freq, compat_freq)
    assert torch.equal(bf16_roundtrip(exact_freq), compat_freq)
    assert exact.denoise_step_emb.reference_compat is False
    assert compat.denoise_step_emb.reference_compat is True


# ---------------------------------------------------------------------------
# The cached sigma LUT
# ---------------------------------------------------------------------------


def _compat_conditioner(dim: int = 64, sigmas=SIGMAS) -> NoiseConditioner:
    torch.manual_seed(0)
    return NoiseConditioner(dim, reference_compat=True, cached_sigmas=sigmas).eval()


def test_the_cached_lut_serves_the_batch_evaluated_row():
    """Each scheduler sigma reads back its own row of the table the reference builds
    with one batched fp32 call, in bf16. Batch shape is the whole point of the patch,
    so the table is built batched and read one sigma at a time."""
    conditioner = _compat_conditioner()
    table, _, oob = conditioner._reference_lut(CPU)
    assert table.shape == (len(SIGMAS), 64) and table.dtype is torch.bfloat16
    assert int(oob) == len(SIGMAS)

    with torch.no_grad():
        for row, value in enumerate(SIGMAS):
            out = conditioner(torch.tensor([[value]], dtype=torch.bfloat16))
            assert out.shape == (1, 1, 64) and out.dtype is torch.bfloat16
            assert torch.equal(out[0, 0], table[row])
        batched = conditioner(torch.tensor([list(SIGMAS)], dtype=torch.bfloat16))
    assert torch.equal(batched[0], table)


def test_an_off_schedule_sigma_raises_rather_than_reading_a_neighbour():
    """The reference's "no silent wrong": an unknown sigma indexes one past the table.
    Checked on CPU, where the out-of-bounds index is an exception rather than a
    device-side assert that would poison the CUDA context."""
    conditioner = _compat_conditioner()
    with pytest.raises(IndexError):
        conditioner(torch.tensor([[0.5]], dtype=torch.bfloat16))


def test_the_cached_lut_refuses_a_non_bf16_sigma():
    """The key is sigma's bf16 bit pattern, so an fp32 sigma has no valid index."""
    conditioner = _compat_conditioner()
    with pytest.raises(RuntimeError, match="bf16 sigma"):
        conditioner(torch.tensor([[1.0]], dtype=torch.float32))


@pytest.mark.parametrize(
    "sigmas, message",
    [((1.0, 1.0), "collide in bf16"), ((1.0, 1.0 + 2**-9), "collide in bf16"), ((), "sigma schedule")],
)
def test_an_uncacheable_schedule_is_refused_at_construction(sigmas, message):
    """bf16 has 8 mantissa bits, so two schedule entries can round together; the LUT
    would then serve one row for both. Refused where it is cheap to see."""
    with pytest.raises(ValueError, match=message):
        NoiseConditioner(64, reference_compat=True, cached_sigmas=sigmas)


def test_the_default_conditioner_keeps_the_live_per_sigma_path():
    """Flag off: no table, and the fp32 body runs per call in the input's dtype."""
    torch.manual_seed(0)
    conditioner = NoiseConditioner(64).eval()
    assert conditioner.reference_compat is False
    with torch.no_grad():
        out = conditioner(torch.tensor([[1.0, 0.3]], dtype=torch.float32))
    assert out.shape == (1, 2, 64) and out.dtype is torch.float32
    assert not conditioner._lut
