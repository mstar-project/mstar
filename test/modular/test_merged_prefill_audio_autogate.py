"""CPU parity tests for the MERGED_PREFILL_AUDIO occupancy AUTO-GATE.
No GPU / no model weights.

The merge collapses an s2t request's ``prefill_text`` + ``prefill_audio`` into one
``prefill_multimodal_audio`` walk. It wins at low batch but regresses at B32, so it
is gated by live active-request occupancy: merge ONLY when
``occupancy <= MSTAR_MERGED_PREFILL_AUDIO_MAX_BS`` (default 24). This test pins the
gate's PARITY property: whenever the gate declines (occupancy over the ceiling), the
schedule is returned UNCHANGED — byte-identical to the non-merged path — so a single
config is correct at every batch. Both the merged and unmerged walks are captured in
the same boot, so either choice replays a captured graph (validated on GPU: s2t B16
merged 675 tok/s vs B32 unmerged 879).
"""
import pytest

torch = pytest.importorskip("torch")

from mstar.graph.base import TensorPointerInfo
from mstar.model.qwen3_omni import qwen3_omni_model as qm
from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel


class _CondShim:
    _maybe_merge_prefill_schedule = Qwen3OmniModel._maybe_merge_prefill_schedule


def _tpi(uuid: str, n: int = 4) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=[n], dtype="int64", nbytes=n * 8, address=0,
        stride=[1], uuid=uuid, source_session_id="s", source_entity="w",
    )


TEXT = ("prefill_text", {"text_inputs": _tpi("txt")})
AUDIO = ("prefill_audio", {"audio_features": _tpi("af"), "audio_seqlens": _tpi("asl")})
VISION = ("prefill_vision", {"image_features": _tpi("vf"), "image_grid_thw": _tpi("igt")})


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    for f in ("MSTAR_MERGED_PREFILL_AUDIO", "MSTAR_MERGED_PREFILL_AUDIO_MAX_BS"):
        monkeypatch.delenv(f, raising=False)


def _merged(out):
    return len(out) == 1 and out[0][0] == "prefill_multimodal_audio"


# --- default ceiling (24) -----------------------------------------------------
def test_gate_merges_at_or_below_default_ceiling(monkeypatch):
    monkeypatch.setenv("MSTAR_MERGED_PREFILL_AUDIO", "1")
    shim = _CondShim()
    for occ in (1, 8, 16, 24):  # <= 24 -> merge
        out, _, aorder = shim._maybe_merge_prefill_schedule(
            [TEXT, AUDIO], audio_output=False, live_occupancy=occ)
        assert _merged(out) and aorder == "text_first", f"occ={occ} should merge"


def test_gate_declines_above_default_ceiling_byte_identical(monkeypatch):
    monkeypatch.setenv("MSTAR_MERGED_PREFILL_AUDIO", "1")
    shim = _CondShim()
    for occ in (25, 32, 64):  # > 24 -> NO merge, schedule unchanged (parity)
        sched = [TEXT, AUDIO]
        out, vorder, aorder = shim._maybe_merge_prefill_schedule(
            sched, audio_output=False, live_occupancy=occ)
        assert out is sched and vorder is None and aorder is None, \
            f"occ={occ} must be a byte-identical no-op (unchanged schedule)"


def test_boundary_is_inclusive(monkeypatch):
    monkeypatch.setenv("MSTAR_MERGED_PREFILL_AUDIO", "1")
    shim = _CondShim()
    on, _, _ = shim._maybe_merge_prefill_schedule([TEXT, AUDIO], False, live_occupancy=24)
    off, _, _ = shim._maybe_merge_prefill_schedule([TEXT, AUDIO], False, live_occupancy=25)
    assert _merged(on) and not _merged(off)  # <= ceiling merges; ceiling+1 declines


# --- env-configurable ceiling -------------------------------------------------
def test_ceiling_env_override(monkeypatch):
    monkeypatch.setenv("MSTAR_MERGED_PREFILL_AUDIO", "1")
    monkeypatch.setenv("MSTAR_MERGED_PREFILL_AUDIO_MAX_BS", "8")
    assert qm.merged_prefill_audio_max_bs() == 8
    shim = _CondShim()
    merge, _, _ = shim._maybe_merge_prefill_schedule([TEXT, AUDIO], False, live_occupancy=4)
    nomerge, _, _ = shim._maybe_merge_prefill_schedule([TEXT, AUDIO], False, live_occupancy=16)
    assert _merged(merge) and not _merged(nomerge)


def test_ceiling_bad_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MSTAR_MERGED_PREFILL_AUDIO_MAX_BS", "not-an-int")
    assert qm.merged_prefill_audio_max_bs() == 24


# --- backward-compat: occupancy not plumbed -> merge (old always-on behavior) --
def test_none_occupancy_merges(monkeypatch):
    monkeypatch.setenv("MSTAR_MERGED_PREFILL_AUDIO", "1")
    shim = _CondShim()
    out, _, aorder = shim._maybe_merge_prefill_schedule(
        [TEXT, AUDIO], audio_output=False, live_occupancy=None)
    assert _merged(out) and aorder == "text_first"


# --- flag OFF is always a no-op regardless of occupancy -----------------------
def test_flag_off_never_merges(monkeypatch):
    shim = _CondShim()
    for occ in (1, 24, 100, None):
        sched = [TEXT, AUDIO]
        out, vorder, aorder = shim._maybe_merge_prefill_schedule(
            sched, audio_output=False, live_occupancy=occ)
        assert out is sched and vorder is None and aorder is None


# --- gate never applies to speech output (t2s/i2s/s2s) ------------------------
def test_audio_output_never_merges_even_below_ceiling(monkeypatch):
    monkeypatch.setenv("MSTAR_MERGED_PREFILL_AUDIO", "1")
    shim = _CondShim()
    sched = [TEXT, AUDIO]
    out, _, aorder = shim._maybe_merge_prefill_schedule(
        sched, audio_output=True, live_occupancy=1)
    assert out is sched and aorder is None


# --- single-config safety: the audio flag is set GLOBALLY in the one dpenc boot,
# so an i2t request (text+vision, no audio) MUST pass through byte-identical. The
# audio branch needs a prefill_audio walk (absent) and the vision branch needs the
# separate MSTAR_MERGED_PREFILL flag (NOT set in the single boot) -> no-op. This
# pins that i2t correctness does not depend on the audio flag being unset. -------
def test_audio_gate_declines_for_vision_text_i2t_schedule(monkeypatch):
    monkeypatch.delenv("MSTAR_MERGED_PREFILL", raising=False)          # vision merge OFF
    monkeypatch.delenv("MSTAR_CHUNKED_PREFILL_V2_VISION", raising=False)
    monkeypatch.setenv("MSTAR_MERGED_PREFILL_AUDIO", "1")              # audio merge ON (as shipped)
    shim = _CondShim()
    for occ in (1, 8, 24, None):   # even below the audio ceiling, i2t is untouched
        sched = [TEXT, VISION]
        out, vorder, aorder = shim._maybe_merge_prefill_schedule(
            sched, audio_output=False, live_occupancy=occ)
        assert out is sched and vorder is None and aorder is None, \
            f"i2t (text+vision) occ={occ} must be a byte-identical no-op"
