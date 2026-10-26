"""``/v1/realtime`` server_vad turn detection over a stubbed engine.

A fake speech probability (frame energy) stands in for Silero, so the
segmentation is exact and the tests need no weights. Every test checks the
invariant that each committed item is exactly the stream between its
``audio_start_ms`` and ``audio_end_ms``: a turn end inside one append must not
drop the start of the next utterance, each sample is committed exactly once,
and each transition fires once on the session timeline.
"""

import base64
import importlib.util
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from test_openai_realtime import _step_samples, client_and_stub  # noqa: E402, F401

from mstar.api_server.openai import realtime_vad  # noqa: E402

FRAME = realtime_vad.FRAME
_REAL_SILERO = realtime_vad.silero_weights() is not None and importlib.util.find_spec("onnxruntime") is not None


class _EnergyVAD:
    """1.0 for a frame with speech-level energy, else 0.0."""

    resets = 0

    def __call__(self, frame):
        assert frame.shape == (FRAME,)
        return 1.0 if np.sqrt(np.mean(frame.astype(np.float64) ** 2)) > 0.05 else 0.0

    def reset(self):
        _EnergyVAD.resets += 1


@pytest.fixture
def vad_client(client_and_stub, monkeypatch):  # noqa: F811
    monkeypatch.setattr(realtime_vad, "silero_factory", lambda: _EnergyVAD)
    return client_and_stub


def _speech(frames, sr=16000):
    n = frames * FRAME * sr // 16000
    return (np.sin(2 * np.pi * 220 * np.arange(n) / sr) * 8000).astype("<i2")


def _silence(frames, sr=16000):
    return np.zeros(frames * FRAME * sr // 16000, dtype="<i2")


def _send(ws, samples, step):
    raw = np.asarray(samples, dtype="<i2").tobytes()
    for i in range(0, len(raw), 2 * step):
        ws.send_text(json.dumps({"type": "input_audio_buffer.append",
                                 "audio": base64.b64encode(raw[i:i + 2 * step]).decode("ascii")}))


def _barrier(ws) -> list[dict]:
    """Every event the server sent for what came before, in order."""
    ws.send_text(json.dumps({"type": "transcription_session.update", "session": {}}))
    events = []
    while (ev := json.loads(ws.receive_text()))["type"] != "transcription_session.updated":
        events.append(ev)
    return events


def _open(ws, **td):
    ws.receive_text()
    ws.send_text(json.dumps({"type": "transcription_session.update", "session": {
        "turn_detection": {"type": "server_vad", **td}, "mstar": {"chunk_seconds": 300}}}))
    updated = json.loads(ws.receive_text())
    assert updated["type"] == "transcription_session.updated", updated
    return updated["session"]


TURN_EVENTS = ("input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped",
               "input_audio_buffer.committed")


def _turns(events) -> list[dict]:
    """(start_ms, end_ms, item_id) per turn, checking each transition fires
    once, in order, on one item, with times that never go back."""
    seq = [e for e in events if e["type"] in TURN_EVENTS]
    turns, last = [], 0
    for i in range(0, len(seq) - len(seq) % 3, 3):
        started, stopped, committed = seq[i:i + 3]
        assert [started["type"], stopped["type"], committed["type"]] == list(TURN_EVENTS), seq
        assert started["item_id"] == stopped["item_id"] == committed["item_id"]
        assert last <= started["audio_start_ms"] < stopped["audio_end_ms"]
        last = stopped["audio_end_ms"]
        turns.append({"start": started["audio_start_ms"], "end": last, "item": committed["item_id"]})
    tail = seq[len(seq) - len(seq) % 3:]
    assert [e["type"] for e in tail] in ([], ["input_audio_buffer.speech_started"]), seq
    return turns


def _assert_items_are_the_stream(turns, finals, stream, sr=16000):
    """Each item is exactly the stream between its start and end times."""
    assert len(finals) == len(turns)
    for turn, heard in zip(turns, finals, strict=True):
        want = stream[turn["start"] * sr // 1000: turn["end"] * sr // 1000]
        if sr == 16000:
            assert heard.shape == want.shape and np.abs(heard - want.astype(np.int32)).max() <= 1
        else:
            assert heard.size == (turn["end"] - turn["start"]) * 16


def test_one_append_with_speech_silence_speech(vad_client):
    """One 32 ms speech frame, sixteen silent frames, one speech frame, all
    in one append: the turn end falls inside the append. The commit takes the first
    turn; the second utterance stays and becomes the next turn."""
    client, stub = vad_client
    heard = _step_samples(stub)
    stream = np.concatenate([_speech(1), _silence(16), _speech(1)])
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        _open(ws)
        _send(ws, stream, step=stream.size)
        events = _barrier(ws)
        assert [e["type"] for e in events if e["type"] in TURN_EVENTS] == list(TURN_EVENTS) + [TURN_EVENTS[0]]
        ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))
        events += _barrier(ws)
    turns = _turns(events)
    assert [(t["start"], t["end"]) for t in turns] == [(0, 544), (544, 576)]
    assert turns[0]["item"] != turns[1]["item"]
    _assert_items_are_the_stream(turns, heard, stream)


def test_end_of_turn_inside_an_append(vad_client):
    """100 ms appends: the sixteenth silent frame completes mid-append and
    the next utterance starts in the same append."""
    client, stub = vad_client
    heard = _step_samples(stub)
    stream = np.concatenate([_speech(10), _silence(16), _speech(10), _silence(16)])
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        _open(ws, prefix_padding_ms=0)
        _send(ws, stream, step=1600)
        events = _barrier(ws)
    turns = _turns(events)
    assert [(t["start"], t["end"]) for t in turns] == [(0, 832), (832, 1664)]
    _assert_items_are_the_stream(turns, heard, stream)


@pytest.mark.parametrize("silence_ms,gap_frames,ends", [
    (499, 16, True), (500, 16, True), (501, 16, True),   # 512 ms of silent frames passes all three
    (500, 15, False),                                     # 480 ms does not
    (512, 16, True), (513, 16, False),                    # exactly the window ends the turn; 1 ms short does not
])
def test_silence_window_boundary(vad_client, silence_ms, gap_frames, ends):
    client, stub = vad_client
    heard = _step_samples(stub)
    stream = np.concatenate([_speech(4), _silence(gap_frames), _speech(4)])
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        _open(ws, silence_duration_ms=silence_ms, prefix_padding_ms=0)
        _send(ws, stream, step=stream.size)
        events = _barrier(ws)
    turns = _turns(events)
    starts = [e for e in events if e["type"] == "input_audio_buffer.speech_started"]
    if ends:
        assert [(t["start"], t["end"]) for t in turns] == [(0, (4 + 16) * 32)]
        assert [e["audio_start_ms"] for e in starts] == [0, (4 + gap_frames) * 32]
    else:
        assert turns == [] and [e["audio_start_ms"] for e in starts] == [0]
    _assert_items_are_the_stream(turns, heard, stream)


def test_zero_silence_window_ends_a_turn_on_its_first_silent_frame(vad_client):
    client, stub = vad_client
    heard = _step_samples(stub)
    stream = np.concatenate([_speech(10), _silence(3)])
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        _open(ws, silence_duration_ms=0, prefix_padding_ms=0)
        _send(ws, stream, step=stream.size)
        events = _barrier(ws)
    turns = _turns(events)
    assert [(t["start"], t["end"]) for t in turns] == [(0, 11 * 32)]
    _assert_items_are_the_stream(turns, heard, stream)


def test_back_to_back_turns_in_10ms_appends(vad_client):
    client, stub = vad_client
    heard = _step_samples(stub)
    stream = np.concatenate([np.concatenate([_speech(6 + 3 * k), _silence(20)]) for k in range(3)])
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        _open(ws)
        _send(ws, stream, step=160)
        events = _barrier(ws)
    turns = _turns(events)
    assert len(turns) == 3
    # the 4 silent frames after each commit are the next turn's prefix padding
    assert [t["start"] for t in turns] == [0, (6 + 16) * 32, (6 + 20 + 9 + 16) * 32]
    _assert_items_are_the_stream(turns, heard, stream)


def test_turn_open_at_close_and_a_client_commit_mid_turn(vad_client):
    client, stub = vad_client
    heard = _step_samples(stub)
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        _open(ws)
        _send(ws, _speech(10), step=1600)
        events = _barrier(ws)
    # the session closed mid-turn: started, never stopped, nothing decoded
    assert [e["type"] for e in events if e["type"] in TURN_EVENTS] == [TURN_EVENTS[0]]
    assert heard == []

    stream = np.concatenate([_speech(10), _speech(1)[:100]])
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        _open(ws)
        _send(ws, stream, step=stream.size)
        ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))
        events = _barrier(ws)
    # a client commit ends the open turn, partial frame included
    turns = _turns(events)
    assert [(t["start"], t["end"]) for t in turns] == [(0, 5220 * 1000 // 16000)]
    assert heard[-1].size == stream.size


def test_turn_reaching_the_item_cap_is_committed_and_speech_goes_on(vad_client):
    """300 s of unbroken speech: the turn ends at the cap (an append past it
    is not refused), and the speech after it starts the next turn."""
    client, stub = vad_client
    heard = _step_samples(stub)
    piece = _speech(625)                       # 20 s
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        _open(ws)
        for _ in range(15):
            _send(ws, piece, step=piece.size)
        _send(ws, _speech(63), step=FRAME * 63)    # 2.016 s more
        events = _barrier(ws)
        assert not [e for e in events if e["type"] == "error"]
        ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))
        events += _barrier(ws)
    turns = _turns(events)
    assert [(t["start"], t["end"]) for t in turns] == [(0, 300_000), (300_000, 302_016)]
    assert [h.size for h in heard] == [300 * 16000, 63 * FRAME]


def test_24khz_input_keeps_the_timeline(vad_client):
    """The same turns at 24 kHz: resampled before segmentation, with the same
    times in ms as at 16 kHz. (The resampler holds the last 1 ms until more
    audio comes, so the stream goes on one frame past the second turn.)"""
    client, stub = vad_client
    heard = _step_samples(stub)
    stream = np.concatenate([_speech(10, 24000), _silence(16, 24000), _speech(10, 24000), _silence(17, 24000)])
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "transcription_session.update", "session": {
            "input_audio_format": {"type": "audio/pcm", "rate": 24000},
            "turn_detection": {"type": "server_vad", "prefix_padding_ms": 0}, "mstar": {"chunk_seconds": 300}}}))
        assert json.loads(ws.receive_text())["type"] == "transcription_session.updated"
        _send(ws, stream, step=240)
        events = _barrier(ws)
    turns = _turns(events)
    assert [(t["start"], t["end"]) for t in turns] == [(0, 832), (832, 1664)]
    _assert_items_are_the_stream(turns, heard, stream, sr=24000)


def test_clear_drops_the_open_turn_and_keeps_the_timeline(vad_client):
    client, _ = vad_client
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        _open(ws, prefix_padding_ms=0)
        _send(ws, np.concatenate([_speech(5), _speech(1)[:200]]), step=FRAME * 5 + 200)
        ws.send_text(json.dumps({"type": "input_audio_buffer.clear"}))
        _send(ws, np.concatenate([_silence(3), _speech(2), _silence(16)]), step=FRAME * 21)
        events = _barrier(ws)
    cut = next(i for i, e in enumerate(events) if e["type"] == "input_audio_buffer.cleared")
    # the cleared turn started and is gone: no stop, no commit
    assert [e["type"] for e in events[:cut] if e["type"] in TURN_EVENTS] == [TURN_EVENTS[0]]
    turns = _turns(events[cut:])
    # cleared audio still counts on the timeline: 5 frames + 200 samples, then 3 silent frames
    assert [t["start"] for t in turns] == [(5 * FRAME + 200) * 1000 // 16000 + 3 * 32]


class _Scripted:
    """Speech probabilities from a list, one per frame (the last one repeats)."""

    def __init__(self, probs):
        self.probs, self.i = list(probs), 0

    def __call__(self, frame):
        p = self.probs[min(self.i, len(self.probs) - 1)]
        self.i += 1
        return p

    def reset(self):
        pass


def _segment(probs, cfg=realtime_vad.ServerVAD(), max_turn=300 * 16000):
    """Feed one frame per probability; return (ops, the stream fed)."""
    stream = np.arange(len(probs) * FRAME, dtype=np.float32)   # every sample distinct
    seg = realtime_vad.TurnSegmenter(_Scripted(probs), cfg, max_turn)
    return seg.feed(stream), stream


def _items(ops) -> list[np.ndarray]:
    items, cur = [], []
    for op in ops:
        if op[0] == "audio":
            cur.append(op[1])
        elif op[0] == "commit":
            items.append(np.concatenate(cur))
            cur = []
    return items


@pytest.mark.parametrize("prefix_ms", [0, 300])
def test_item_cap_is_never_passed_and_speech_across_it_is_kept(prefix_ms):
    """A 3 s cap is not a whole number of 32 ms frames, and the prefix is not
    either: no item passes the cap, and speech going on across a cap split,
    even at a probability between the two thresholds, starts the next turn
    right there, so every sample from the first start on is in one item."""
    probs = [0.0] * 20 + [0.9] * 90 + [0.4] * 20 + [0.9] * 200 + [0.0] * 20
    ops, stream = _segment(probs, realtime_vad.ServerVAD(prefix_padding_ms=prefix_ms), max_turn=3 * 16000)
    items = _items(ops)
    assert len(items) >= 3 and max(i.size for i in items) <= 3 * 16000
    first = 20 * FRAME - prefix_ms * 16
    joined = np.concatenate(items)
    assert np.array_equal(joined, stream[first:first + joined.size])
    starts = [op[1] for op in ops if op[0] == "speech_started"]
    stops = [op[1] for op in ops if op[0] == "speech_stopped"]
    assert starts[1:] == stops[:-1]


def test_borderline_frames_do_not_hold_a_started_silence_open():
    """Silence starts below threshold - 0.15 and only a frame at or above the
    threshold cancels it: frames alternating 0.1 and 0.4 end the turn on the
    frame that completes the window from the first 0.1, a 0.4 one here, and
    a 0.6 frame starts the count over."""
    ops, _ = _segment([0.9] * 10 + [0.1, 0.4] * 30, realtime_vad.ServerVAD(prefix_padding_ms=0))
    assert [op for op in ops if op[0] == "speech_stopped"] == [("speech_stopped", (10 + 16) * 32)]
    probs = [0.9] * 10 + [0.1, 0.4] * 4 + [0.6] + [0.1, 0.4] * 30
    ops, _ = _segment(probs, realtime_vad.ServerVAD(prefix_padding_ms=0))
    assert [op for op in ops if op[0] == "speech_stopped"] == [("speech_stopped", (10 + 8 + 1 + 16) * 32)]


_AT_24KHZ = {"input_audio_format": {"type": "audio/pcm", "rate": 24000},
             "turn_detection": {"type": "server_vad", "prefix_padding_ms": 0}, "mstar": {"chunk_seconds": 300}}


def _open_with(ws, session):
    ws.receive_text()
    ws.send_text(json.dumps({"type": "transcription_session.update", "session": session}))
    updated = json.loads(ws.receive_text())
    assert updated["type"] == "transcription_session.updated", updated


def test_settings_change_keeps_audio_not_scored_yet(vad_client):
    """A partial frame waiting for its VAD decision survives a turn_detection
    change (into the new segmenter, or into the item when it goes to null)."""
    client, stub = vad_client
    heard = _step_samples(stub)
    stream = np.concatenate([_speech(3), _silence(16)])
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        _open(ws, prefix_padding_ms=0)
        _send(ws, stream[:511], step=511)
        ws.send_text(json.dumps({"type": "transcription_session.update", "session": {
            "turn_detection": {"type": "server_vad", "prefix_padding_ms": 0, "threshold": 0.6}}}))
        assert json.loads(ws.receive_text())["session"]["turn_detection"]["threshold"] == 0.6
        _send(ws, stream[511:], step=stream.size)
        events = _barrier(ws)
        turns = _turns(events)
        assert [(t["start"], t["end"]) for t in turns] == [(0, 19 * 32)]
        _send(ws, stream[:300], step=300)
        ws.send_text(json.dumps({"type": "transcription_session.update", "session": {"turn_detection": None}}))
        ws.receive_text()
        ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))
        _barrier(ws)
    _assert_items_are_the_stream(turns, heard[:1], stream)
    assert np.array_equal(heard[1], stream[:300].astype(np.int32))


@pytest.mark.parametrize("sr", [16000, 24000])
def test_settings_change_between_turns_is_accepted_at_both_rates(vad_client, sr):
    """Between turns no audio is in an item, so a turn_detection change goes
    through at 24 kHz as at 16 kHz: what the resampler holds back for its
    lookahead is audio after the turn, and goes on to the new segmenter."""
    client, stub = vad_client
    heard = _step_samples(stub)
    utterance = np.concatenate([_speech(3, sr), _silence(17, sr)])
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        _open_with(ws, {**_AT_24KHZ, "input_audio_format": {"type": "audio/pcm", "rate": sr}})
        _send(ws, utterance, step=utterance.size)
        events = _barrier(ws)
        assert len(_turns(events)) == 1
        ws.send_text(json.dumps({"type": "transcription_session.update", "session": {
            "turn_detection": {"type": "server_vad", "prefix_padding_ms": 0, "threshold": 0.6}}}))
        updated = json.loads(ws.receive_text())
        assert updated["type"] == "transcription_session.updated", updated
        _send(ws, utterance, step=utterance.size)
        events += _barrier(ws)
    turns = _turns(events)
    assert [(t["start"], t["end"]) for t in turns] == [(0, 19 * 32), (20 * 32, 39 * 32)]
    _assert_items_are_the_stream(turns, heard, np.concatenate([utterance, utterance]), sr=sr)


def test_one_long_append_is_one_engine_step(vad_client):
    """64 s of speech in one append decodes in one step, as without turn
    detection, not one step per 2 s chunk over a growing item."""
    client, stub = vad_client
    heard = _step_samples(stub)
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        _open_with(ws, {"turn_detection": {"type": "server_vad"}, "mstar": {"chunk_seconds": 2.0}})
        _send(ws, _speech(2000), step=2000 * FRAME)
        _barrier(ws)
    assert [h.size for h in heard] == [2000 * FRAME // 32000 * 32000]


def test_clear_at_24khz_keeps_the_timeline(vad_client):
    """What the resampler held at a clear still counts on the timeline: 100
    clears of 100 ms each put the next speech at exactly 10 s."""
    client, _ = vad_client
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        _open_with(ws, _AT_24KHZ)
        for _ in range(100):
            _send(ws, np.zeros(2400, dtype="<i2"), step=2400)
            ws.send_text(json.dumps({"type": "input_audio_buffer.clear"}))
        _send(ws, np.concatenate([_speech(10, 24000), _silence(20, 24000)]), step=24000)
        events = _barrier(ws)
    assert [e["audio_start_ms"] for e in events if e["type"] == "input_audio_buffer.speech_started"] == [10_000]


@pytest.mark.parametrize("boundary", ["input_audio_buffer.clear", "input_audio_buffer.commit"])
def test_many_24khz_restarts_do_not_drift_the_timeline(vad_client, boundary):
    """Each commit or clear starts the resampler over and rounds its output
    up by under a sample; the timeline stays on the input clock instead:
    96 restarts after 2,401 input samples each put speech at 9,604 ms."""
    client, _ = vad_client
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        _open_with(ws, _AT_24KHZ)
        for _ in range(96):
            _send(ws, np.zeros(2401, dtype="<i2"), step=2401)
            ws.send_text(json.dumps({"type": boundary}))
        _send(ws, np.concatenate([_speech(10, 24000), _silence(20, 24000)]), step=24000)
        events = _barrier(ws)
    starts = [e["audio_start_ms"] for e in events if e["type"] == "input_audio_buffer.speech_started"]
    assert starts == [96 * 2401 * 1000 // 24000]


def test_a_client_commit_that_completes_a_turn_commits_once(vad_client):
    """At 24 kHz the resampler holds the last 1 ms, so the commit's drained
    tail is what completes the silence window: one item, not a second empty one."""
    client, stub = vad_client
    heard = _step_samples(stub)
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        _open_with(ws, _AT_24KHZ)
        _send(ws, np.concatenate([_speech(1, 24000), _silence(16, 24000)]), step=24000)
        events = _barrier(ws)
        assert not [e for e in events if e["type"] == "input_audio_buffer.committed"]
        ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))
        events = _barrier(ws)
    assert [e["type"] for e in events].count("input_audio_buffer.committed") == 1
    assert [e["type"] for e in events].count("conversation.item.input_audio_transcription.completed") == 1
    assert [h.size for h in heard] == [17 * FRAME]


def test_a_client_commit_between_turns_keeps_earlier_audio_out_of_the_next_turn(vad_client):
    """Audio the VAD had not put in a turn when the client committed (a
    partial frame, the prefix padding) does not leak into the next turn."""
    client, stub = vad_client
    heard = _step_samples(stub)
    stream = np.concatenate([_speech(1)[:511], _speech(3), _silence(16)])
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        _open(ws)
        _send(ws, stream[:511], step=511)
        ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))
        _send(ws, stream[511:], step=stream.size)
        events = _barrier(ws)
    starts = [e["audio_start_ms"] for e in events if e["type"] == "input_audio_buffer.speech_started"]
    assert starts == [511 * 1000 // 16000]
    assert heard[-1].size == stream.size - 511


def test_session_echoes_server_vad_and_validates_it(vad_client):
    client, _ = vad_client
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        session = _open(ws, silence_duration_ms=700)
        assert session["turn_detection"] == {
            "type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300, "silence_duration_ms": 700,
            "create_response": True, "interrupt_response": True}
        for td, needle in (({"type": "server_vad", "threshold": 1.5}, "turn_detection.threshold"),
                           ({"type": "server_vad", "eagerness": "low"}, "turn_detection.'eagerness' is not"),
                           ({"type": "server_vad", "idle_timeout_ms": 5000}, "idle_timeout_ms is not supported"),
                           ({"type": "server_vad", "create_response": "yes"}, "must be a boolean")):
            ws.send_text(json.dumps({"type": "transcription_session.update", "session": {"turn_detection": td}}))
            assert needle in json.loads(ws.receive_text())["error"]["message"]
        ws.send_text(json.dumps({"type": "transcription_session.update", "session": {"turn_detection": None}}))
        assert json.loads(ws.receive_text())["session"]["turn_detection"] is None


def test_server_vad_without_weights_is_refused(client_and_stub, monkeypatch):  # noqa: F811
    client, _ = client_and_stub
    monkeypatch.setattr(realtime_vad, "silero_factory", lambda: None)
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "transcription_session.update",
                                 "session": {"turn_detection": {"type": "server_vad"}}}))
        err = json.loads(ws.receive_text())
    assert err["type"] == "error" and "server_vad is unavailable" in err["error"]["message"]


def test_missing_onnxruntime_disables_server_vad_with_a_warning(monkeypatch, tmp_path, caplog):
    weights = tmp_path / "silero_vad.onnx"
    weights.write_bytes(b"onnx")
    monkeypatch.setenv("MSTAR_SILERO_VAD_PATH", str(weights))
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    realtime_vad.silero_factory.cache_clear()
    try:
        assert realtime_vad.silero_factory() is None
    finally:
        realtime_vad.silero_factory.cache_clear()
    assert "needs onnxruntime" in caplog.text


class _Arg:
    def __init__(self, name, type_):
        self.name, self.type = name, type_


_F, _I = "tensor(float)", "tensor(int64)"
_SILERO_IN = [("input", _F), ("state", _F), ("sr", _I)]


class _FakeSession:
    """An onnxruntime session with given inputs; ``run`` answers like Silero,
    or with the given output shapes, and refuses inputs it does not have."""

    def __init__(self, inputs, out_shapes=((1, 1), (2, 1, 128))):
        self.inputs, self.out_shapes = [_Arg(*a) for a in inputs], out_shapes

    def get_inputs(self):
        return self.inputs

    def run(self, _, feed):
        if set(feed) != {a.name for a in self.inputs}:
            raise ValueError(f"invalid input names {sorted(feed)}")
        return [np.zeros(s, dtype=np.float32) for s in self.out_shapes]


def _fake_onnxruntime(monkeypatch, tmp_path, session):
    weights = tmp_path / "silero_vad.onnx"
    weights.write_bytes(b"onnx")
    monkeypatch.setenv("MSTAR_SILERO_VAD_PATH", str(weights))
    ort = type(sys)("onnxruntime")
    ort.SessionOptions = type("SessionOptions", (), {})
    ort.ExecutionMode = type("ExecutionMode", (), {"ORT_SEQUENTIAL": 0})
    ort.InferenceSession = lambda *a, **k: session
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    realtime_vad.silero_factory.cache_clear()


@pytest.fixture
def _uncached_silero():
    yield
    realtime_vad.silero_factory.cache_clear()


@pytest.mark.parametrize("session,needle", [
    # the silero-vad package's silero_vad_half.onnx and silero_vad_openvino_16k.onnx
    (_FakeSession([("input", _F), ("state", _F)]), "its inputs are"),
    # its silero_vad_16k_sequence.onnx
    (_FakeSession([("input", _F), ("h", _F), ("c", _F)]), "its inputs are"),
    (_FakeSession([("input", _F), ("state", _F), ("sr", _F)]), "its inputs are"),
    (_FakeSession(_SILERO_IN, out_shapes=((1,), (2, 1, 128))), "its outputs have shapes"),
    (_FakeSession(_SILERO_IN, out_shapes=((1, 1),)), "its outputs have shapes"),
])
def test_a_model_with_other_inputs_or_outputs_disables_server_vad(
        monkeypatch, tmp_path, caplog, _uncached_silero, session, needle):
    _fake_onnxruntime(monkeypatch, tmp_path, session)
    assert realtime_vad.silero_factory() is None
    assert "is not the Silero VAD export" in caplog.text and needle in caplog.text


def test_the_silero_inputs_in_any_order_are_accepted(monkeypatch, tmp_path, _uncached_silero):
    """silero_vad_op18_ifless.onnx lists sr before state; inputs go by name."""
    _fake_onnxruntime(monkeypatch, tmp_path, _FakeSession([("input", _F), ("sr", _I), ("state", _F)]))
    make = realtime_vad.silero_factory()
    assert make is not None and make()(np.zeros(FRAME, dtype=np.float32)) == 0.0


def test_server_vad_with_a_wrong_model_is_refused_and_the_session_goes_on(
        client_and_stub, monkeypatch, tmp_path, _uncached_silero):  # noqa: F811
    """A wrong model is refused at the session update with an error event,
    instead of being accepted and closing the socket on the first append."""
    client, stub = client_and_stub
    heard = _step_samples(stub)
    _fake_onnxruntime(monkeypatch, tmp_path, _FakeSession([("input", _F), ("state", _F)]))
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "transcription_session.update",
                                 "session": {"turn_detection": {"type": "server_vad"}}}))
        err = json.loads(ws.receive_text())
        assert err["type"] == "error" and "server_vad is unavailable" in err["error"]["message"]
        _send(ws, _speech(10), step=FRAME * 10)
        ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))
        events = _barrier(ws)
    assert "conversation.item.input_audio_transcription.completed" in [e["type"] for e in events]
    assert [h.size for h in heard] == [10 * FRAME]


def _tiny_onnx(path, inputs):
    """A model with the given (name, element type) inputs and Silero's
    outputs: the mean of ``input`` and ``state`` unchanged."""
    from onnx import TensorProto, helper, save

    types = {"float": TensorProto.FLOAT, "int64": TensorProto.INT64}
    shapes = {"input": [1, None], "state": [2, None, 128], "sr": []}
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["input"], ["output"], axes=[1], keepdims=1),
         helper.make_node("Identity", ["state"], ["stateN"])],
        "tiny",
        [helper.make_tensor_value_info(n, types[t], shapes[n]) for n, t in inputs],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1]),
         helper.make_tensor_value_info("stateN", TensorProto.FLOAT, [2, None, 128])])
    # an IR version every onnxruntime release since 1.10 reads; onnx's default runs ahead of it
    save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8), str(path))


@pytest.mark.skipif(importlib.util.find_spec("onnx") is None or importlib.util.find_spec("onnxruntime") is None,
                    reason="needs onnx and onnxruntime")
def test_a_real_onnx_file_without_sr_is_refused(monkeypatch, tmp_path, _uncached_silero):
    """Through onnxruntime itself: the same graph is accepted with Silero's
    three inputs and refused without ``sr``, as silero_vad_half.onnx is."""
    good, bad = tmp_path / "good.onnx", tmp_path / "bad.onnx"
    _tiny_onnx(good, [("input", "float"), ("state", "float"), ("sr", "int64")])
    _tiny_onnx(bad, [("input", "float"), ("state", "float")])
    monkeypatch.setenv("MSTAR_SILERO_VAD_PATH", str(good))
    realtime_vad.silero_factory.cache_clear()
    make = realtime_vad.silero_factory()
    assert make is not None and make()(np.zeros(FRAME, dtype=np.float32)) == 0.0
    monkeypatch.setenv("MSTAR_SILERO_VAD_PATH", str(bad))
    realtime_vad.silero_factory.cache_clear()
    assert realtime_vad.silero_factory() is None


@pytest.mark.skipif(not _REAL_SILERO, reason="no Silero VAD weights or onnxruntime")
def test_silero_streams_share_one_model_but_not_its_state():
    """Streams interleaved through the one ONNX session score each frame as
    they would alone, and reset starts a stream over."""
    realtime_vad.silero_factory.cache_clear()
    make = realtime_vad.silero_factory()
    with wave.open(str(Path(__file__).parents[1] / "qwen3-omni" / "audio.wav"), "rb") as wf:
        clip = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")[::3][: 40 * FRAME] / 32768.0
    frames = clip.astype(np.float32).reshape(-1, FRAME)
    alone = [p for v in [make()] for p in (v(f) for f in frames)]
    a, b = make(), make()
    together = []
    for f, g in zip(frames, frames[::-1], strict=True):
        together.append(a(f))
        b(g)
    assert together == alone and max(alone) > 0.5
    a.reset()
    assert [a(f) for f in frames] == alone


@pytest.mark.skipif(not _REAL_SILERO, reason="no Silero VAD weights or onnxruntime")
def test_silero_splits_real_speech_into_its_turns(client_and_stub):  # noqa: F811
    """Real Silero on real speech at 24 kHz: the clip twice with 1 s of
    silence between is two turns, each holding its whole clip."""
    realtime_vad.silero_factory.cache_clear()
    client, stub = client_and_stub
    heard = _step_samples(stub)
    with wave.open(str(Path(__file__).parents[1] / "qwen3-omni" / "audio.wav"), "rb") as wf:
        assert wf.getframerate() == 24000 and wf.getnchannels() == 1
        clip = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    gap = np.zeros(24000, dtype="<i2")
    stream = np.concatenate([gap, clip, gap, clip, gap])
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "transcription_session.update", "session": {
            "input_audio_format": {"type": "audio/pcm", "rate": 24000},
            "turn_detection": {"type": "server_vad"}, "mstar": {"chunk_seconds": 300}}}))
        assert json.loads(ws.receive_text())["type"] == "transcription_session.updated"
        _send(ws, stream, step=480)
        events = _barrier(ws)
    turns = _turns(events)
    clip_ms = clip.size * 1000 // 24000
    assert len(turns) == 2
    for turn, clip_start in zip(turns, (1000, 2000 + clip_ms), strict=True):
        assert turn["start"] <= clip_start + 100 and turn["end"] >= clip_start + clip_ms - 100
    _assert_items_are_the_stream(turns, heard, stream, sr=24000)


# ---------------------------------------------------------------- semantic_vad


class _EndOfTurn:
    """Fake end-of-turn model: always returns ``p``, and records each call."""

    def __init__(self, p):
        self.p, self.calls = p, []

    def __call__(self, audio):
        self.calls.append(audio.size)
        return self.p


def _semantic(monkeypatch, p):
    eot = _EndOfTurn(p)
    monkeypatch.setattr(realtime_vad, "smart_turn", lambda: eot)
    return eot


def _run_semantic(client, stream, eagerness="medium"):
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "transcription_session.update", "session": {
            "turn_detection": {"type": "semantic_vad", "eagerness": eagerness}, "mstar": {"chunk_seconds": 300}}}))
        session = json.loads(ws.receive_text())["session"]
        _send(ws, stream, step=1600)
        events = _barrier(ws)
    return session, events


def test_semantic_vad_commits_at_a_pause_the_model_calls_final(vad_client, monkeypatch):
    """P(end of turn) 0.9 at the 300 ms pause ends the turn there, well
    before server_vad's 500 ms window."""
    client, stub = vad_client
    heard = _step_samples(stub)
    eot = _semantic(monkeypatch, 0.9)
    stream = np.concatenate([_speech(10), _silence(40), _speech(5), _silence(20)])
    session, events = _run_semantic(client, stream)
    assert session["turn_detection"] == {"type": "semantic_vad", "eagerness": "medium",
                                         "create_response": True, "interrupt_response": True}
    turns = _turns(events)
    # the pause reaches 300 ms on the 10th silent frame (320 ms); the next turn has 300 ms of prefix
    assert [(t["start"], t["end"]) for t in turns] == [(0, 20 * 32), (50 * 32 - 300, 65 * 32)]
    assert len(eot.calls) == 2 and eot.calls[0] == 20 * FRAME
    _assert_items_are_the_stream(turns, heard, stream)


def test_semantic_vad_keeps_listening_when_the_model_says_not_done(vad_client, monkeypatch):
    """P 0.1 at a pause keeps the turn open: speech after it is the same
    turn, and the silence limit (4 s at medium) ends it."""
    client, stub = vad_client
    heard = _step_samples(stub)
    eot = _semantic(monkeypatch, 0.1)
    stream = np.concatenate([_speech(10), _silence(20), _speech(10), _silence(130)])
    _, events = _run_semantic(client, stream)
    turns = _turns(events)
    assert [(t["start"], t["end"]) for t in turns] == [(0, (10 + 20 + 10 + 125) * 32)]
    assert len(eot.calls) == 2          # once per pause, not once per frame
    _assert_items_are_the_stream(turns, heard, stream)


class _FailingEndOfTurn(_EndOfTurn):
    """Fake end-of-turn model that raises, as onnxruntime or the feature extractor can."""

    def __call__(self, audio):
        self.calls.append(audio.size)
        raise RuntimeError("[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : /srv/models/smart-turn.onnx")


def test_semantic_vad_survives_an_end_of_turn_model_error(vad_client, monkeypatch, caplog):
    """The model raising at a pause is an error event, not a closed socket
    (1011): the session goes on, its turns end at medium's 4 s silence
    limit, the model is not asked again, and every sample is in exactly one
    item. The event has an error id and no server path."""
    client, stub = vad_client
    heard = _step_samples(stub)
    eot = _FailingEndOfTurn(0.9)
    monkeypatch.setattr(realtime_vad, "smart_turn", lambda: eot)
    stream = np.concatenate([_speech(10), _silence(130), _speech(10), _silence(130)])
    with caplog.at_level("ERROR"):
        _, events = _run_semantic(client, stream)
    errors = [e for e in events if e["type"] == "error"]
    assert len(errors) == 1, errors
    error = errors[0]["error"]
    assert error["type"] == "server_error" and error["code"] == "end_of_turn_failed"
    assert "after 4 s of silence" in error["message"] and "/srv" not in error["message"]
    error_id = error["message"].split("error id ")[1].split(")")[0]
    assert error_id in caplog.text and "RUNTIME_EXCEPTION" in caplog.text
    turns = _turns(events)
    # the second turn's prefix is the 5 silent frames left after the first one ended
    assert [(t["start"], t["end"]) for t in turns] == [(0, (10 + 125) * 32), ((10 + 125) * 32, (150 + 125) * 32)]
    assert len(eot.calls) == 1
    _assert_items_are_the_stream(turns, heard, stream)


@pytest.mark.parametrize("eagerness,end_frames", [
    ("high", 7),        # 0.6 >= 0.3: ends at the 200 ms pause (7 frames, 224 ms)
    ("medium", 10),     # 0.6 >= 0.5: ends at the 300 ms pause (10 frames)
    ("auto", 10),
    ("low", 250),       # 0.6 < 0.7: no; the 8 s silence ends it (250 frames)
])
def test_semantic_vad_eagerness(vad_client, monkeypatch, eagerness, end_frames):
    client, stub = vad_client
    heard = _step_samples(stub)
    _semantic(monkeypatch, 0.6)
    stream = np.concatenate([_speech(10), _silence(260)])
    _, events = _run_semantic(client, stream, eagerness)
    turns = _turns(events)
    assert [(t["start"], t["end"]) for t in turns] == [(0, (10 + end_frames) * 32)]
    _assert_items_are_the_stream(turns, heard, stream)


def test_semantic_history_holds_only_its_window_and_is_dropped_at_the_turn_end():
    """The 8 s the model hears are copies, emptied when the turn ends or is
    cleared, so one long append is not kept alive by its frames."""
    seg = realtime_vad.TurnSegmenter(_EnergyVAD(), realtime_vad.SemanticVAD(), 300 * 16000,
                                     end_of_turn=_EndOfTurn(0.1))
    append = np.concatenate([_speech(400), _silence(10)]).astype(np.float32) / 32768
    seg.feed(append)
    assert seg.speaking and not any(np.shares_memory(piece, append) for piece in seg._recent)
    assert sum(piece.size for piece in seg._recent) <= realtime_vad.SMART_TURN_SECONDS * 16000 + FRAME
    seg.feed(np.zeros(130 * FRAME, dtype=np.float32))
    assert not seg.speaking and not seg._recent
    seg.feed(append)
    seg.reset()
    assert not seg._recent


def _semantic_vad_is_refused_and_server_vad_still_works(vad_client, reason, hidden):
    """semantic_vad is refused with an error event naming ``reason`` (and not
    the server path ``hidden``); the session goes on, and server_vad chosen
    next segments as usual."""
    client, stub = vad_client
    heard = _step_samples(stub)
    stream = np.concatenate([_speech(10), _silence(20)])
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "transcription_session.update", "session": {
            "turn_detection": {"type": "semantic_vad", "eagerness": "high"}, "mstar": {"chunk_seconds": 300}}}))
        err = json.loads(ws.receive_text())
        assert err["type"] == "error", err
        message = err["error"]["message"]
        assert "semantic_vad is unavailable" in message and reason in message and hidden not in message
        ws.send_text(json.dumps({"type": "transcription_session.update", "session": {
            "turn_detection": {"type": "server_vad"}, "mstar": {"chunk_seconds": 300}}}))
        updated = json.loads(ws.receive_text())
        assert updated["session"]["turn_detection"]["type"] == "server_vad", updated
        _send(ws, stream, step=1600)
        events = _barrier(ws)
    turns = _turns(events)
    assert [(t["start"], t["end"]) for t in turns] == [(0, (10 + 16) * 32)]
    _assert_items_are_the_stream(turns, heard, stream)


def test_semantic_vad_without_smart_turn_weights_is_refused(vad_client, monkeypatch, tmp_path, _uncached_smart_turn):
    monkeypatch.setenv("MSTAR_SMART_TURN_PATH", str(tmp_path / "missing.onnx"))
    realtime_vad.smart_turn.cache_clear()
    _semantic_vad_is_refused_and_server_vad_still_works(vad_client, "no Smart Turn weights", str(tmp_path))


def test_semantic_vad_settings_are_validated(vad_client, monkeypatch):
    client, _ = vad_client
    _semantic(monkeypatch, 0.9)
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        ws.receive_text()
        for td, needle in (({"type": "semantic_vad", "eagerness": "eager"}, "eagerness must be one of"),
                           ({"type": "semantic_vad", "silence_duration_ms": 500}, "silence_duration_ms' is not")):
            ws.send_text(json.dumps({"type": "transcription_session.update", "session": {"turn_detection": td}}))
            assert needle in json.loads(ws.receive_text())["error"]["message"]


def _smart_turn_available() -> bool:
    import importlib.util

    if importlib.util.find_spec("onnxruntime") is None or importlib.util.find_spec("transformers") is None:
        return False
    return realtime_vad.smart_turn_weights() is not None and realtime_vad.silero_weights() is not None


@pytest.mark.skipif(not _smart_turn_available(), reason="no Smart Turn / Silero weights or onnxruntime")
def test_smart_turn_ends_real_turns_sooner_than_the_silence_window(client_and_stub):  # noqa: F811
    """Real Silero and Smart Turn on real speech at 24 kHz: the clip twice is
    two turns, each ending at the first 300 ms pause after the clip rather
    than after 500 ms of silence."""
    realtime_vad.silero_factory.cache_clear()
    realtime_vad.smart_turn.cache_clear()
    client, stub = client_and_stub
    heard = _step_samples(stub)
    with wave.open(str(Path(__file__).parents[1] / "qwen3-omni" / "audio.wav"), "rb") as wf:
        clip = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    gap = np.zeros(24000, dtype="<i2")
    stream = np.concatenate([gap, clip, gap, clip, gap])
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "transcription_session.update", "session": {
            "input_audio_format": {"type": "audio/pcm", "rate": 24000},
            "turn_detection": {"type": "semantic_vad"}, "mstar": {"chunk_seconds": 300}}}))
        assert json.loads(ws.receive_text())["session"]["turn_detection"]["type"] == "semantic_vad"
        _send(ws, stream, step=480)
        events = _barrier(ws)
    turns = _turns(events)
    clip_ms = clip.size * 1000 // 24000
    assert len(turns) == 2
    for turn, clip_end in zip(turns, (1000 + clip_ms, 2000 + 2 * clip_ms), strict=True):
        assert clip_end - 100 <= turn["end"] < clip_end + 500
    _assert_items_are_the_stream(turns, heard, stream, sr=24000)


def test_missing_smart_turn_is_a_warning_and_no_model(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("MSTAR_SMART_TURN_PATH", str(tmp_path / "missing.onnx"))
    realtime_vad.smart_turn.cache_clear()
    try:
        with caplog.at_level("WARNING", logger=realtime_vad.__name__):
            assert realtime_vad.smart_turn() is None
        assert realtime_vad.smart_turn_problem() == "no Smart Turn weights on this server"
    finally:
        realtime_vad.smart_turn.cache_clear()
    assert "semantic_vad is unavailable" in caplog.text


def test_smart_turn_without_onnxruntime_names_the_package(monkeypatch, tmp_path, _uncached_smart_turn):
    weights = tmp_path / "smart-turn-v3.2-cpu.onnx"
    weights.write_bytes(b"onnx")
    monkeypatch.setenv("MSTAR_SMART_TURN_PATH", str(weights))
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    realtime_vad.smart_turn.cache_clear()
    assert realtime_vad.smart_turn() is None
    assert realtime_vad.smart_turn_problem() == "onnxruntime or transformers is not installed"


def _fake_smart_turn_runtime(monkeypatch, tmp_path, session):
    weights = tmp_path / "smart-turn-v3.2-cpu.onnx"
    weights.write_bytes(b"onnx")
    monkeypatch.setenv("MSTAR_SMART_TURN_PATH", str(weights))
    ort = type(sys)("onnxruntime")
    ort.SessionOptions = type("SessionOptions", (), {})
    ort.InferenceSession = lambda *a, **k: session
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    realtime_vad.smart_turn.cache_clear()


@pytest.fixture
def _uncached_smart_turn():
    yield
    realtime_vad.smart_turn.cache_clear()


_needs_transformers = pytest.mark.skipif(importlib.util.find_spec("transformers") is None,
                                         reason="needs transformers (Whisper's feature extractor)")


@_needs_transformers
@pytest.mark.parametrize("session,needle", [
    (_FakeSession(_SILERO_IN), "its inputs are"),     # MSTAR_SMART_TURN_PATH pointed at Silero
    (_FakeSession([("input_features", _I)], out_shapes=((1, 1),)), "its inputs are"),
    (_FakeSession([("input_features", _F)], out_shapes=((1,),)), "its outputs have shapes"),
    (_FakeSession([("input_features", _F)], out_shapes=()), "its outputs have shapes"),
])
def test_a_smart_turn_file_with_other_inputs_or_outputs_is_no_model(
        monkeypatch, tmp_path, caplog, _uncached_smart_turn, session, needle):
    _fake_smart_turn_runtime(monkeypatch, tmp_path, session)
    with caplog.at_level("WARNING", logger=realtime_vad.__name__):
        assert realtime_vad.smart_turn() is None
    assert "semantic_vad is unavailable" in caplog.text and needle in caplog.text
    assert realtime_vad.smart_turn_problem() == "the Smart Turn file is not the expected model"


@_needs_transformers
def test_the_smart_turn_signature_is_accepted(monkeypatch, tmp_path, _uncached_smart_turn):
    _fake_smart_turn_runtime(monkeypatch, tmp_path, _FakeSession([("input_features", _F)], out_shapes=((1, 1),)))
    model = realtime_vad.smart_turn()
    assert model is not None and model(np.zeros(16000, dtype=np.float32)) == 0.0


@_needs_transformers
def test_semantic_vad_with_a_wrong_smart_turn_file_is_refused(vad_client, monkeypatch, tmp_path, _uncached_smart_turn):
    """Refused at the session update, instead of the first pause raising and
    closing the socket."""
    _fake_smart_turn_runtime(monkeypatch, tmp_path, _FakeSession(_SILERO_IN))
    _semantic_vad_is_refused_and_server_vad_still_works(
        vad_client, "the Smart Turn file is not the expected model", str(tmp_path))


@pytest.mark.skipif(any(importlib.util.find_spec(m) is None for m in ("onnx", "onnxruntime", "transformers")),
                    reason="needs onnx, onnxruntime and transformers")
def test_a_real_onnx_file_with_silero_inputs_is_not_smart_turn(monkeypatch, tmp_path, _uncached_smart_turn):
    path = tmp_path / "silero_shaped.onnx"
    _tiny_onnx(path, [("input", "float"), ("state", "float"), ("sr", "int64")])
    monkeypatch.setenv("MSTAR_SMART_TURN_PATH", str(path))
    realtime_vad.smart_turn.cache_clear()
    assert realtime_vad.smart_turn() is None
