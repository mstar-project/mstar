"""``test/minimax_h3/parity.py``'s verdict on the stages it records: a broken decoder or a
drifting loop exits nonzero, and ``--exact`` holds every compared tensor to the docs' claim."""

from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest

PARITY = Path(__file__).resolve().parents[4] / "test" / "minimax_h3" / "parity.py"
EXACT = {"max_abs": 0.0, "rel_l2": 0.0, "equal_frac": 1.0, "bitexact": True}


@pytest.fixture(scope="module")
def parity():
    if not PARITY.exists():
        pytest.skip("test/minimax_h3 is not in this tree")
    spec = importlib.util.spec_from_file_location("h3_parity", PARITY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides) -> Namespace:
    return Namespace(**{"max_latent_rel": 1e-2, "max_loop_rel": 5e-2, "min_psnr": 40.0, "min_snr": 40.0,
                        "exact": False, **overrides})


def _report(**overrides) -> dict:
    report = {
        "keyframe_latents": dict(EXACT),
        **{f"loop_{kind}_latents_{mode}": dict(EXACT) for kind in ("video", "audio")
           for mode in ("serving", "reference_shapes")},
        "video_decode": {"oracle_latents": {"psnr_db": float("inf"), "equal_frac": 1.0}},
        "audio_decode": {"oracle_latents": {**EXACT, "snr_db": float("inf")}},
        "dit_teacher_forced_serving": [{"k": 0, "video": dict(EXACT), "audio": dict(EXACT)}],
    }
    report.update(overrides)
    return report


def test_a_bit_exact_report_passes(parity):
    assert parity.check_bounds(_report(), _args(exact=True)) == []


def test_the_unbounded_stages_fail_when_off(parity):
    off = {"max_abs": 1.0, "rel_l2": 0.5, "equal_frac": 0.0, "bitexact": False}
    for overrides, failure in (
        ({"video_decode": {"oracle_latents": {"psnr_db": 12.0, "equal_frac": 0.1}}}, "video_decode"),
        ({"audio_decode": {"oracle_latents": {**off, "snr_db": 3.0}}}, "audio_decode"),
        ({"keyframe_latents": off}, "vae_encode"),
        ({"keyframe_latents": {"shape_mismatch": [[1], [2]]}}, "vae_encode"),
        ({"loop_audio_latents_serving": off}, "loop_serving"),
    ):
        assert parity.check_bounds(_report(**overrides), _args()) == [failure]


def test_a_stage_that_did_not_run_is_not_judged(parity):
    assert parity.check_bounds({"video_decode": {}}, _args(exact=True)) == []


def test_exact_names_every_tensor_that_is_not_bit_identical(parity):
    close = {"max_abs": 1e-6, "rel_l2": 1e-5, "equal_frac": 0.9, "bitexact": False}
    report = _report(loop_video_latents_serving=close,
                     dit_teacher_forced_serving=[{"k": 0, "video": dict(EXACT), "audio": close}])
    assert parity.check_bounds(report, _args()) == []
    assert parity.check_bounds(report, _args(exact=True)) == [
        "not_bitexact:loop_video_latents_serving", "not_bitexact:dit_teacher_forced_serving[0].audio"]
