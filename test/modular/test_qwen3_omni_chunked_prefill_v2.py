"""Conductor-driven re-enqueue tests for W5-P1 chunked Thinker prefill (V2).

Exercises the SCHEDULER / CONDUCTOR loop: the Thinker state machine must
re-emit the SAME ``prefill_text`` walk with an advanced ``prefill_chunk_offset``
until a long span is consumed, set ``is_last_prefill`` only on the final chunk,
hold the input tensor alive across chunks (unpersist only on the last), and stay
byte-identical when the flag is OFF or the span fits in one chunk.

All CPU, no GPU, no model weights: a ``_Shim`` borrows just the state-machine
methods off ``Qwen3OmniModel``. P1 chunks text only; vision/audio chunking lands
behind ``MSTAR_CHUNKED_PREFILL_V2_VISION`` later.
"""
import pytest

torch = pytest.importorskip("torch")  # module import pulls torch transitively

from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.graph.base import TensorPointerInfo
from mstar.model.qwen3_omni.qwen3_omni_model import (
    Qwen3OmniModel,
    plan_prefill_chunk,
)


# --- pure planner ------------------------------------------------------------
def test_plan_chunk_no_chunk_when_short():
    # span <= cap => single-shot (None), byte-identical to flag-off.
    assert plan_prefill_chunk(500, 0, 512) is None
    assert plan_prefill_chunk(512, 0, 512) is None
    assert plan_prefill_chunk(0, 0, 512) is None
    assert plan_prefill_chunk(-5, 0, 512) is None


def test_plan_chunk_sequence_cap512():
    # 1100 / cap 512 -> 512, 512, 76; chunks sum to the span.
    assert plan_prefill_chunk(1100, 0, 512) == (512, False)
    assert plan_prefill_chunk(1100, 512, 512) == (512, False)
    assert plan_prefill_chunk(1100, 1024, 512) == (76, True)
    # exact multiple closes on the boundary
    assert plan_prefill_chunk(1024, 512, 512) == (512, True)
    # offset at/past the end is a no-op guard
    assert plan_prefill_chunk(1100, 1100, 512) is None


def test_plan_chunk_bucket_alignment():
    # Non-cap tails fall to the next-lower bucket (256/128) then remainder.
    assert plan_prefill_chunk(1000, 512, 512) == (256, False)
    assert plan_prefill_chunk(1000, 768, 512) == (128, False)
    assert plan_prefill_chunk(1000, 896, 512) == (104, True)


def test_plan_chunk_sums_to_span():
    for span in (513, 600, 1000, 1500, 2048, 4096):
        for cap in (128, 256, 512, 1024):
            off, total = 0, 0
            while True:
                plan = plan_prefill_chunk(span, off, cap)
                if plan is None:
                    total = span  # single-shot covers the whole span
                    break
                cl, done = plan
                total += cl
                off += cl
                if done:
                    break
            assert total == span, (span, cap, total)


# --- conductor state-machine loop (no model weights) -----------------------
class _Shim:
    _text_chunk_bounds = Qwen3OmniModel._text_chunk_bounds
    _chunk_bounds = Qwen3OmniModel._chunk_bounds
    _walk_span_tokens = Qwen3OmniModel._walk_span_tokens
    _log_first_chunk = Qwen3OmniModel._log_first_chunk
    _assert_walk_span = Qwen3OmniModel._assert_walk_span
    _vision_chunk_bounds = Qwen3OmniModel._vision_chunk_bounds
    _get_thinker_forward = Qwen3OmniModel._get_thinker_forward
    _get_thinker_prefill_inputs = Qwen3OmniModel._get_thinker_prefill_inputs


def _tpi(n_tokens: int, uuid: str) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=[n_tokens], dtype="int64", nbytes=n_tokens * 8, address=0,
        stride=[1], uuid=uuid, source_session_id="s", source_entity="w",
    )


def _make_meta(schedule, audio_output=False):
    return CurrentForwardConductorMetadata(
        graph_walk=schedule[0][0],
        is_prefill=True,
        kwargs={
            "prefill_schedule": schedule,
            "prefill_step": 0,
            "audio_output": audio_output,
            "prefill_chunk_offset": 0,
        },
    )


def _drive(shim, meta, persist, max_steps=50):
    """Replay the conductor loop: describe step 0 from the initial metadata,
    then call _get_thinker_forward repeatedly (as _process_done_forward does)
    until the Thinker leaves prefill."""
    steps = []
    schedule = meta.kwargs["prefill_schedule"]
    bounds = shim._chunk_bounds(meta, schedule, 0, persist)
    is_last = (len(schedule) == 1)
    if bounds is not None:
        off, ln, done = bounds
        is_last = is_last and done
        steps.append({"walk": meta.graph_walk, "offset": off, "len": ln,
                      "is_last_prefill": is_last, "unpersist": 0 if not done else 1})
    else:
        steps.append({"walk": meta.graph_walk, "offset": 0, "len": None,
                      "is_last_prefill": is_last, "unpersist": 1})

    for _ in range(max_steps):
        fwd = shim._get_thinker_forward(meta, persist)
        meta = fwd.full_metadata
        meta.kwargs.update(fwd.step_metadata)
        if not meta.is_prefill:
            steps.append({"walk": meta.graph_walk, "decode": True})
            break
        sm = fwd.step_metadata
        steps.append({
            "walk": meta.graph_walk,
            "offset": sm.get("prefill_chunk_offset"),
            "len": sm.get("prefill_chunk_len"),
            "is_last_prefill": sm.get("is_last_prefill"),
            "unpersist": len(fwd.unpersist_tensors),
        })
    return steps


def test_long_text_single_walk_chunks(monkeypatch):
    monkeypatch.setenv("MSTAR_CHUNKED_PREFILL_V2", "1")
    monkeypatch.setenv("MSTAR_PREFILL_CHUNK_TOKENS", "512")
    schedule = [("prefill_text", {"text_inputs": _tpi(1100, "t0")})]
    meta = _make_meta(schedule, audio_output=False)
    persist = {"text_inputs": [schedule[0][1]["text_inputs"]]}
    steps = _drive(_Shim(), meta, persist)

    prefill = [s for s in steps if not s.get("decode")]
    assert [(s["offset"], s["len"]) for s in prefill] == [
        (0, 512), (512, 512), (1024, 76),
    ]
    # is_last_prefill ONLY on the final chunk.
    assert [s["is_last_prefill"] for s in prefill] == [False, False, True]
    # Non-final chunks hold the tensor (unpersist=0); final releases it.
    assert [s["unpersist"] for s in prefill] == [0, 0, 1]
    # Loop terminates into decode.
    assert steps[-1].get("decode") is True


def test_flag_off_no_chunking(monkeypatch):
    monkeypatch.delenv("MSTAR_CHUNKED_PREFILL_V2", raising=False)
    schedule = [("prefill_text", {"text_inputs": _tpi(1100, "t0")})]
    meta = _make_meta(schedule, audio_output=False)
    persist = {"text_inputs": [schedule[0][1]["text_inputs"]]}
    steps = _drive(_Shim(), meta, persist)
    prefill = [s for s in steps if not s.get("decode")]
    # Single full-span step: no chunk metadata, is_last on the one step.
    assert len(prefill) == 1
    assert prefill[0]["len"] is None
    assert prefill[0]["is_last_prefill"] is True
    assert prefill[0]["unpersist"] == 1


def test_short_text_not_chunked(monkeypatch):
    monkeypatch.setenv("MSTAR_CHUNKED_PREFILL_V2", "1")
    monkeypatch.setenv("MSTAR_PREFILL_CHUNK_TOKENS", "512")
    schedule = [("prefill_text", {"text_inputs": _tpi(400, "t0")})]
    meta = _make_meta(schedule, audio_output=False)
    persist = {"text_inputs": [schedule[0][1]["text_inputs"]]}
    steps = _drive(_Shim(), meta, persist)
    prefill = [s for s in steps if not s.get("decode")]
    assert len(prefill) == 1
    assert prefill[0]["len"] is None
    assert prefill[0]["is_last_prefill"] is True


def test_audio_output_disables_chunking(monkeypatch):
    monkeypatch.setenv("MSTAR_CHUNKED_PREFILL_V2", "1")
    monkeypatch.setenv("MSTAR_PREFILL_CHUNK_TOKENS", "512")
    schedule = [("prefill_text", {"text_inputs": _tpi(1100, "t0")})]
    # audio_output True => Talker conditioned => must NOT chunk.
    meta = _make_meta(schedule, audio_output=True)
    persist = {"text_inputs": [schedule[0][1]["text_inputs"]]}
    steps = _drive(_Shim(), meta, persist)
    prefill = [s for s in steps if not s.get("decode")]
    assert len(prefill) == 1
    assert prefill[0]["len"] is None


def test_multi_walk_last_chunk_only_is_last(monkeypatch):
    monkeypatch.setenv("MSTAR_CHUNKED_PREFILL_V2", "1")
    monkeypatch.setenv("MSTAR_PREFILL_CHUNK_TOKENS", "512")
    # Two text walks (e.g. vLLM prefix/suffix layout): first long, second short.
    schedule = [
        ("prefill_text", {"text_inputs": _tpi(600, "t0")}),
        ("prefill_text", {"text_inputs": _tpi(50, "t1")}),
    ]
    meta = _make_meta(schedule, audio_output=False)
    persist = {"text_inputs": [schedule[0][1]["text_inputs"]]}
    steps = _drive(_Shim(), meta, persist)
    prefill = [s for s in steps if not s.get("decode")]
    # walk0: 512, 88 (both not-last); walk1: 50 (last, single-shot).
    assert [s.get("is_last_prefill") for s in prefill] == [False, False, True]


def test_assert_hook_passes_on_valid_span(monkeypatch):
    monkeypatch.setenv("MSTAR_CHUNKED_PREFILL_V2", "1")
    monkeypatch.setenv("MSTAR_PREFILL_CHUNK_TOKENS", "512")
    monkeypatch.setenv("MSTAR_CHUNKED_PREFILL_V2_ASSERT", "1")
    schedule = [("prefill_text", {"text_inputs": _tpi(1500, "t0")})]
    meta = _make_meta(schedule, audio_output=False)
    persist = {"text_inputs": [schedule[0][1]["text_inputs"]]}
    # Should not raise: the summed committed offset matches span - last_chunk.
    steps = _drive(_Shim(), meta, persist)
    assert steps[-1].get("decode") is True


# --- chunked vision (encoder-split) conductor loop -------------------------
def _drive_vision(shim, meta, persist, max_steps=50):
    """Like _drive but the schedule leads with encode_vision (encoder-only, not
    chunked) then a chunked prefill_vision whose span comes from the persisted
    vision_embeds dims. Describes step 0 from the initial metadata."""
    steps = []
    schedule = meta.kwargs["prefill_schedule"]
    walk0 = schedule[0][0]
    bounds = shim._chunk_bounds(meta, schedule, 0, persist)
    is_last = (len(schedule) == 1)
    if bounds is not None:
        off, ln, done = bounds
        is_last = is_last and done
        steps.append({"walk": walk0, "offset": off, "len": ln,
                      "is_last_prefill": is_last})
    else:
        steps.append({"walk": walk0, "offset": 0, "len": None,
                      "is_last_prefill": is_last})
    for _ in range(max_steps):
        fwd = shim._get_thinker_forward(meta, persist)
        meta = fwd.full_metadata
        meta.kwargs.update(fwd.step_metadata)
        if not meta.is_prefill:
            steps.append({"walk": meta.graph_walk, "decode": True})
            break
        sm = fwd.step_metadata
        steps.append({
            "walk": meta.graph_walk,
            "offset": sm.get("prefill_chunk_offset"),
            "len": sm.get("prefill_chunk_len"),
            "is_last_prefill": sm.get("is_last_prefill"),
            "unpersist": len(fwd.unpersist_tensors),
        })
    return steps


def test_vision_chunk_bounds_reads_persist(monkeypatch):
    monkeypatch.setenv("MSTAR_CHUNKED_PREFILL_V2", "1")
    monkeypatch.setenv("MSTAR_CHUNKED_PREFILL_V2_VISION", "1")
    monkeypatch.setenv("MSTAR_PREFILL_CHUNK_TOKENS", "512")
    # 600 vision tokens => span 602 (+2 sentinels) => 512, 90.
    schedule = [("prefill_vision", {})]
    meta = _make_meta(schedule, audio_output=False)
    persist = {"vision_embeds": [_tpi(600, "v0")]}
    assert _Shim()._vision_chunk_bounds(meta, schedule, 0, persist) == (0, 512, False)
    meta.kwargs["prefill_chunk_offset"] = 512
    assert _Shim()._vision_chunk_bounds(meta, schedule, 0, persist) == (512, 90, True)


def test_vision_chunk_bounds_gated_off_without_flag(monkeypatch):
    monkeypatch.setenv("MSTAR_CHUNKED_PREFILL_V2", "1")
    monkeypatch.delenv("MSTAR_CHUNKED_PREFILL_V2_VISION", raising=False)
    schedule = [("prefill_vision", {})]
    meta = _make_meta(schedule, audio_output=False)
    persist = {"vision_embeds": [_tpi(600, "v0")]}
    assert _Shim()._vision_chunk_bounds(meta, schedule, 0, persist) is None


def test_encode_vision_then_chunked_vision(monkeypatch):
    monkeypatch.setenv("MSTAR_CHUNKED_PREFILL_V2", "1")
    monkeypatch.setenv("MSTAR_CHUNKED_PREFILL_V2_VISION", "1")
    monkeypatch.setenv("MSTAR_PREFILL_CHUNK_TOKENS", "512")
    # encode_vision (not chunked) then prefill_vision (600 -> span 602 -> 512,90).
    schedule = [
        ("encode_vision", {"pixel_values": _tpi(9999, "px"),
                           "image_grid_thw": _tpi(3, "grid")}),
        ("prefill_vision", {"image_grid_thw": _tpi(3, "grid")}),
    ]
    meta = _make_meta(schedule, audio_output=False)
    # vision_embeds appears in persist after encode_vision runs.
    persist = {"vision_embeds": [_tpi(600, "v0")], "deepstack": [_tpi(600, "d0")]}
    steps = _drive_vision(_Shim(), meta, persist)
    walks = [s["walk"] for s in steps if not s.get("decode")]
    # encode_vision, then two prefill_vision chunks.
    assert walks == ["encode_vision", "prefill_vision", "prefill_vision"]
    prefill_vision = [s for s in steps
                      if not s.get("decode") and s["walk"] == "prefill_vision"]
    assert [(s["offset"], s["len"]) for s in prefill_vision] == [(0, 512), (512, 90)]
    assert [s["is_last_prefill"] for s in prefill_vision] == [False, True]
    assert steps[-1].get("decode") is True
