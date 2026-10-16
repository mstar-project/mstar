"""A scripted LiveKit participant that talks to the agent and times it.

It joins a room as a user, waits for the agent, and plays WAV utterances into
its microphone track in real time (silence in between, so the agent's VAD can
close each turn). It listens to the agent's audio and its ``lk.agent.state``
attribute, and reads both sides' transcripts from the ``lk.transcription`` text
streams.

Scenarios (one per conversation; ``mix`` cycles through them):

    turns       N question/answer turns
    bargein     speak over the agent's reply, then expect an answer to the new turn
    cancel      interrupt several times in a row before the agent finishes, then let it answer
    disconnect  leave abruptly while the agent is thinking or speaking

Per turn it records ``turn_latency_s``: from the end of the user's speech to
the agent's first audible audio, which is what a caller hears. It includes the
agent's endpointing delay and the WebRTC path as well as STT, LLM and TTS.
When the user talks over the agent (``bargein``, the last turn of ``cancel``),
the agent must leave ``speaking`` and its earlier audio stop before the user is
done, or the turn counts as a failed interruption; only audio after that stop
counts as the reply.

    python examples/livekit/participant.py --utterances 'utts/*.wav' \\
        --conversations 8 --scenario mix --out results.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import glob
import json
import os
import statistics
import time
import wave

import numpy as np
from livekit import api, rtc

SAMPLE_RATE = 24000
FRAME_SAMPLES = SAMPLE_RATE // 100  # 10 ms
VOICED_RMS = 300.0  # int16 RMS above which an agent frame counts as speech
SCENARIOS = ("turns", "bargein", "cancel", "disconnect")


def load_utterance(path: str) -> np.ndarray:
    """Mono int16 at ``SAMPLE_RATE``, with leading and trailing silence trimmed."""
    with wave.open(path, "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: need 16-bit PCM, got {8 * width}-bit")
    audio = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        n = int(round(len(audio) * SAMPLE_RATE / rate))
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
    frames = audio[: len(audio) // FRAME_SAMPLES * FRAME_SAMPLES].reshape(-1, FRAME_SAMPLES)
    voiced = np.flatnonzero(np.sqrt((frames**2).mean(axis=1)) > VOICED_RMS)
    if voiced.size:
        frames = frames[voiced[0]: voiced[-1] + 1]
    return frames.reshape(-1).astype(np.int16)


class Conversation:
    def __init__(self, url: str, token: str, name: str):
        self.url, self.token, self.name = url, token, name
        self.room = rtc.Room()
        self.source = rtc.AudioSource(SAMPLE_RATE, 1, queue_size_ms=100)
        self._queue: asyncio.Queue[tuple[np.ndarray, asyncio.Future]] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []
        self.agent_joined = asyncio.Event()
        self.agent_audio = asyncio.Event()
        self.voiced: list[float] = []  # wall times of voiced agent frames
        self.states: list[tuple[float, str]] = []
        self.transcripts: list[dict] = []
        self._state_changed = asyncio.Event()

    async def connect(self) -> None:
        room = self.room

        def _state(state: str | None) -> None:
            if state and (not self.states or self.states[-1][1] != state):
                self.states.append((time.time(), state))
                self._state_changed.set()

        def _agent(p: rtc.RemoteParticipant) -> None:
            if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
                self.agent_joined.set()
                _state((p.attributes or {}).get("lk.agent.state"))

        room.on("participant_connected", _agent)

        @room.on("participant_attributes_changed")
        def _attrs(changed: dict, p: rtc.Participant) -> None:
            if "lk.agent.state" in changed:
                _state(changed["lk.agent.state"])

        @room.on("track_subscribed")
        def _track(track: rtc.Track, _pub, p: rtc.RemoteParticipant) -> None:
            if track.kind == rtc.TrackKind.KIND_AUDIO and p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
                self._tasks.append(asyncio.create_task(self._listen(track)))

        def _on_text(reader: rtc.TextStreamReader, identity: str) -> None:
            async def read() -> None:
                text = await reader.read_all()
                attrs = reader.info.attributes or {}
                self.transcripts.append({
                    "t": time.time(), "from": identity, "text": text,
                    "final": attrs.get("lk.transcription_final") == "true",
                })
            self._tasks.append(asyncio.create_task(read()))

        room.register_text_stream_handler("lk.transcription", _on_text)
        await room.connect(self.url, self.token)
        for p in room.remote_participants.values():
            _agent(p)
        track = rtc.LocalAudioTrack.create_audio_track("mic", self.source)
        await room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
        self._tasks.append(asyncio.create_task(self._mic()))

    async def _mic(self) -> None:
        """Real-time microphone: queued utterances, silence otherwise."""
        silence = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        deadline = time.monotonic()
        current: np.ndarray | None = None
        done: asyncio.Future | None = None
        pos = 0
        while True:
            if current is None and not self._queue.empty():
                current, done = self._queue.get_nowait()
                pos = 0
                done.start = time.time()  # type: ignore[attr-defined]
            if current is not None:
                chunk = current[pos: pos + FRAME_SAMPLES]
                pos += FRAME_SAMPLES
                if len(chunk) < FRAME_SAMPLES:
                    chunk = np.pad(chunk, (0, FRAME_SAMPLES - len(chunk)))
            else:
                chunk = silence
            await self.source.capture_frame(rtc.AudioFrame(chunk.tobytes(), SAMPLE_RATE, 1, FRAME_SAMPLES))
            if current is not None and pos >= len(current):
                # capture_frame returns once the frame is queued; the user stops
                # talking when the queue has played out
                done.set_result((done.start, time.time() + self.source.queued_duration))  # type: ignore[attr-defined]
                current = None
            deadline += 0.01
            await asyncio.sleep(max(0.0, deadline - time.monotonic()))

    async def _listen(self, track: rtc.Track) -> None:
        async for ev in rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1):
            samples = np.frombuffer(bytes(ev.frame.data), dtype=np.int16).astype(np.float32)
            if samples.size and np.sqrt((samples**2).mean()) > VOICED_RMS:
                self.voiced.append(time.time())
                self.agent_audio.set()

    async def say(self, audio: np.ndarray) -> asyncio.Future:
        """Queue ``audio``; the future resolves to (start, end) wall times once played."""
        fut = asyncio.get_running_loop().create_future()
        await self._queue.put((audio, fut))
        return fut

    async def first_voiced_after(self, t: float, timeout: float) -> float | None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            i = bisect.bisect_right(self.voiced, t)
            if i < len(self.voiced):
                return self.voiced[i]
            self.agent_audio.clear()
            try:
                await asyncio.wait_for(self.agent_audio.wait(), timeout=max(0.0, end - time.monotonic()))
            except asyncio.TimeoutError:
                return None
        return None

    def stopped_after(self, t: float, gap: float = 0.2) -> float | None:
        """When the agent's audio first paused for ``gap`` seconds after ``t``;
        None while it is still playing (no pause seen yet)."""
        i = bisect.bisect_left(self.voiced, t)
        if i == len(self.voiced) or self.voiced[i] - t > gap:
            return t
        while i + 1 < len(self.voiced) and self.voiced[i + 1] - self.voiced[i] <= gap:
            i += 1
        if i + 1 == len(self.voiced) and time.time() - self.voiced[i] <= gap:
            return None
        return self.voiced[i]

    def left_speaking_by(self, t: float) -> float | None:
        """When the agent's ``lk.agent.state`` last left "speaking" at or before
        ``t`` (0.0 if it never spoke); None while it is still speaking at ``t``."""
        prior = [s for s in self.states if s[0] <= t]
        if prior and prior[-1][1] == "speaking":
            return None
        spoke = [i for i, (_, s) in enumerate(prior) if s == "speaking"]
        return prior[spoke[-1] + 1][0] if spoke else 0.0

    async def wait_state(self, states: tuple[str, ...], after: float, timeout: float) -> float | None:
        end = time.monotonic() + timeout
        while True:
            for t, s in self.states:
                if t >= after and s in states:
                    return t
            remaining = end - time.monotonic()
            if remaining <= 0:
                return None
            self._state_changed.clear()
            try:
                await asyncio.wait_for(self._state_changed.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return None

    async def quiet_for(self, seconds: float, timeout: float) -> float | None:
        """Wait until the agent has been silent for ``seconds``; return when it went quiet."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            last = self.voiced[-1] if self.voiced else 0.0
            if time.time() - last >= seconds:
                return last
            await asyncio.sleep(0.05)
        return None

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.room.disconnect()


def make_token(key: str, secret: str, room: str, identity: str) -> str:
    grants = api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True)
    return api.AccessToken(key, secret).with_identity(identity).with_grants(grants).to_jwt()


async def turn(conv: Conversation, audio: np.ndarray, timeout: float, over_agent: bool = False) -> dict:
    """One user turn and the agent's reply. With ``over_agent`` the user talks
    over the agent: it must leave "speaking" and its earlier audio stop before
    the user is done (else the interruption failed), and only audio after that
    stop is the reply."""
    start, end = await (await conv.say(audio))
    rec = {"user_start": start, "user_end": end}
    after = end
    if over_agent:
        # ``end`` can lie ahead by the source's queue; judge the stop once it has passed
        await asyncio.sleep(max(0.0, end + 0.25 - time.time()))
        # A pause between the agent's phrases is no interruption: LiveKit marks
        # one by leaving "speaking", so the stop is looked for from there, and
        # an agent still speaking when the user is done was not interrupted.
        left = conv.left_speaking_by(end)
        stopped = None if left is None else conv.stopped_after(max(start, left))
        if stopped is not None:
            rec["interrupt_s"] = stopped - start
        if stopped is None or stopped >= end:
            rec.update(interrupt_failed=True, agent_first_audio=None, turn_latency_s=None)
            return rec
        after = max(end, stopped)
    first = await conv.first_voiced_after(after, timeout)
    rec.update(agent_first_audio=first, turn_latency_s=None if first is None else first - end)
    if first is not None:
        thinking = await conv.wait_state(("thinking",), after=end - 0.5, timeout=0.1)
        if thinking is not None and thinking < first:
            rec["after_endpoint_s"] = first - thinking
        rec["agent_quiet"] = await conv.quiet_for(1.0, timeout)
        await conv.wait_state(("listening",), after=first, timeout=5.0)
    return rec


async def run_conversation(args, idx: int, scenario: str, utts: list[np.ndarray], log) -> dict:
    name = f"{args.room_prefix}-{idx}"
    conv = Conversation(args.url, make_token(args.api_key, args.api_secret, name, f"user-{idx}"), name)
    summary = {"room": name, "scenario": scenario, "ok": False, "turns": []}
    t0 = time.time()
    try:
        await conv.connect()
        await asyncio.wait_for(conv.agent_joined.wait(), timeout=args.timeout)
        await conv.wait_state(("listening",), after=0, timeout=args.timeout)
        summary["agent_ready_s"] = time.time() - t0
        pick = lambda k: utts[(idx + k) % len(utts)]  # noqa: E731

        if scenario == "turns":
            for k in range(args.turns):
                summary["turns"].append(await turn(conv, pick(k), args.timeout))
            summary["ok"] = all(t["turn_latency_s"] is not None for t in summary["turns"])

        elif scenario == "bargein":
            _, end = await (await conv.say(pick(0)))
            first = await conv.first_voiced_after(end, args.timeout)
            if first is not None:
                await asyncio.sleep(0.8)
                rec = await turn(conv, pick(1), args.timeout, over_agent=True)
                summary["turns"].append(rec)
                summary["ok"] = rec["turn_latency_s"] is not None

        elif scenario == "cancel":
            for k in range(args.cancels):
                start, end = await (await conv.say(pick(k)))
                await conv.wait_state(("thinking", "speaking"), after=end - 0.5, timeout=args.timeout)
                await asyncio.sleep(0.3 * (k % 3))
            rec = await turn(conv, pick(args.cancels), args.timeout, over_agent=True)
            summary["turns"].append(rec)
            summary["ok"] = rec["turn_latency_s"] is not None

        elif scenario == "disconnect":
            _, end = await (await conv.say(pick(0)))
            state = await conv.wait_state(("thinking", "speaking"), after=end - 0.5, timeout=args.timeout)
            await asyncio.sleep(0.3 + 0.4 * (idx % 3))
            summary["left_at_state"] = conv.states[-1][1] if conv.states else None
            summary["ok"] = state is not None
    except Exception as exc:  # noqa: BLE001
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        summary["transcripts"] = conv.transcripts
        summary["states"] = conv.states
        summary["duration_s"] = time.time() - t0
        await conv.close()
    log.write(json.dumps(summary) + "\n")
    log.flush()
    return summary


def report(results: list[dict]) -> dict:
    lat = [t["turn_latency_s"] for r in results for t in r["turns"] if t.get("turn_latency_s") is not None]
    after = [t["after_endpoint_s"] for r in results for t in r["turns"] if t.get("after_endpoint_s") is not None]
    intr = [t["interrupt_s"] for r in results for t in r["turns"] if t.get("interrupt_s") is not None]
    failed = sum(bool(t.get("interrupt_failed")) for r in results for t in r["turns"])

    def pct(xs: list[float]) -> dict | None:
        if not xs:
            return None
        xs = sorted(xs)
        return {"n": len(xs), "p50": statistics.median(xs), "p90": xs[min(len(xs) - 1, int(0.9 * len(xs)))],
                "max": xs[-1]}

    think = sum("<think>" in t["text"] for r in results for t in r["transcripts"])
    return {
        "conversations": len(results),
        "ok": sum(r["ok"] for r in results),
        "errors": [r["error"] for r in results if "error" in r],
        "by_scenario": {s: sum(r["scenario"] == s for r in results) for s in SCENARIOS},
        "turn_latency_s": pct(lat),
        "after_endpoint_s": pct(after),
        "interrupt_s": pct(intr),
        "interrupts_failed": failed,
        "transcripts_with_think_tags": think,
    }


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default=os.environ.get("LIVEKIT_URL", "ws://localhost:7880"))
    ap.add_argument("--api-key", default=os.environ.get("LIVEKIT_API_KEY", "devkey"))
    ap.add_argument("--api-secret", default=os.environ.get("LIVEKIT_API_SECRET", "secret"))
    ap.add_argument("--utterances", required=True, help="glob of 16-bit WAV files")
    ap.add_argument("--conversations", type=int, default=1)
    ap.add_argument("--scenario", choices=(*SCENARIOS, "mix"), default="turns")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--cancels", type=int, default=3)
    ap.add_argument("--stagger", type=float, default=0.5, help="seconds between conversation starts")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--room-prefix", default=f"mstar-{int(time.time())}")
    ap.add_argument("--out", default="participant_results.jsonl")
    args = ap.parse_args()

    utts = [load_utterance(p) for p in sorted(glob.glob(args.utterances))]
    if not utts:
        raise SystemExit(f"no WAV files match {args.utterances!r}")
    with open(args.out, "w") as log:
        async def start(i: int) -> dict:
            await asyncio.sleep(i * args.stagger)
            scenario = SCENARIOS[i % len(SCENARIOS)] if args.scenario == "mix" else args.scenario
            return await run_conversation(args, i, scenario, utts, log)

        results = await asyncio.gather(*(start(i) for i in range(args.conversations)))
    summary = report(results)
    print(json.dumps(summary, indent=2))
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return 0 if summary["ok"] == summary["conversations"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
