#!/usr/bin/env python3
"""Duplex-style streaming test for the Nemotron-Duplex server.

Sends one user utterance followed by trailing silence — i.e. "keep sending
no-sound frames after the user stops" — and consumes the agent's reply as it
streams back, frame by frame. This exercises the server's full-duplex behavior:
the model should stay silent while the user talks, start replying in the frames
after, speak its turn, and return to silence. We detect that turn boundary and
report it (and stop reading once the agent is done, which cancels the rest of
the request server-side).

    python test/nemotron_duplex/duplex_stream_request.py                 # 30 s reply window
    python test/nemotron_duplex/duplex_stream_request.py --silence 15 --output agent.wav
    python test/nemotron_duplex/duplex_stream_request.py --audio user.wav --silence 20

Note on the transport: the M* engine's /generate takes a fixed audio buffer and
streams the OUTPUT back (there is no incremental-audio-input / WebSocket path in
the engine — that lives in the model's standalone realtime_api.py). So the
"silence until the agent finishes" is a fixed pre-padded window (``--silence``,
the cap the user asked for); the realtime part is the streamed reply we watch.
"""
import argparse
import sys
import time
import wave

from _audio_prep import prepare_user_clip
from _env import get_base_url, load_env

from mstar.client.client import MStarClient
from mstar.client.types import AudioChunk, TextChunk

FRAME_S = 0.08                       # model frame period (~12.5 Hz)
_SILENCE_RMS = 100                   # int16 RMS below this = a silent (non-speech) frame


def _is_special(txt: str) -> bool:
    t = txt.strip()
    return t == "" or t.startswith("<SPECIAL") or t in ("</s>", "<s>", "<pad>", "<unk>")


def _rms_int16(pcm: bytes) -> float:
    import numpy as np

    if not pcm:
        return 0.0
    a = np.frombuffer(pcm, dtype="<i2").astype("float32")
    return float((a ** 2).mean() ** 0.5) if a.size else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=None, help="user speech wav (default: prepared VoiceChat-11B sample)")
    ap.add_argument("--silence", type=float, default=30.0,
                    help="seconds of no-sound frames to send after the user stops (reply window / cap)")
    ap.add_argument("--finish-silence", type=float, default=1.6,
                    help="stop once the agent has been silent this long after speaking")
    ap.add_argument("--output", default="agent_out.wav", help="where to save the agent's speech")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    load_env()
    audio = args.audio or prepare_user_clip(trailing_s=args.silence)
    url = get_base_url()
    print(f"POST {url}/generate (stream)  audio={audio}  reply-window={args.silence:.0f}s")
    print("watching the agent's turn (frame ~= 80 ms):\n")

    c = MStarClient(url, timeout=600)
    gen = c.generate(
        audio=audio, input_modalities=("audio",),
        output_modalities=("text", "audio"), temperature=args.temperature, stream=True,
    )

    pcm = bytearray()
    transcript = []
    sr = 22050
    text_frames = 0                  # agent text tokens seen (frames)
    audio_secs = 0.0                 # seconds of agent audio received
    audible_secs = 0.0               # seconds of that audio that is actually voiced
    first_audible = None
    last_audible = 0.0
    text_started = None              # time of first non-special text token
    text_quiet = 0.0                 # seconds of consecutive EOS/special text
    audio_quiet = 0.0                # seconds of consecutive silent audio
    finished_reason = "reply window ended"
    t0 = time.time()

    try:
        for ev in gen:
            if isinstance(ev, TextChunk):
                text_frames += 1
                if _is_special(ev.text):
                    text_quiet += FRAME_S
                else:
                    text_quiet = 0.0
                    transcript.append(ev.text)
                    if text_started is None:
                        text_started = (text_frames - 1) * FRAME_S
                        print(f"[{text_started:5.1f}s] agent starts replying")
                    sys.stdout.write(ev.text)
                    sys.stdout.flush()
            elif isinstance(ev, AudioChunk):
                pcm += ev.pcm
                sr = ev.sample_rate or sr
                dur = (len(ev.pcm) // 2) / (sr or 1)
                audio_secs += dur
                if _rms_int16(ev.pcm) >= _SILENCE_RMS:
                    audio_quiet = 0.0
                    audible_secs += dur
                    if first_audible is None:
                        first_audible = audio_secs - dur
                    last_audible = audio_secs
                else:
                    audio_quiet += dur
            # Turn is done only after we've actually HEARD the agent (first_audible)
            # and BOTH channels have since gone quiet. The audio pipeline lags the
            # text badly, so gating on text (or on audio before the reply streams)
            # would truncate the spoken reply. If the agent never pauses we run to
            # the --silence cap; if it never voices audio we also run to the cap.
            if (first_audible is not None and text_quiet >= args.finish_silence
                    and audio_quiet >= args.finish_silence):
                finished_reason = "agent finished (text + audio returned to silence)"
                break
    except KeyboardInterrupt:
        finished_reason = "interrupted"
    finally:
        gen.close()   # closing the stream cancels the remaining request server-side

    # --- report ---
    print("\n")
    print(f"=== duplex turn summary ({finished_reason}) ===")
    print(f"  wall time         : {time.time() - t0:.1f}s")
    print(f"  interaction       : ~{text_frames * FRAME_S:.1f}s ({text_frames} input frames)")
    if text_started is None:
        print("  agent reply text  : NONE (all silence) — try more --silence / a different --audio")
    else:
        print(f"  agent reply text  : {''.join(transcript)!r}")
    if first_audible is not None:
        print(f"  audible speech    : {first_audible:.1f}s -> {last_audible:.1f}s span, "
              f"{audible_secs:.1f}s voiced of {audio_secs:.1f}s audio")
    else:
        print(f"  audible speech    : none ({audio_secs:.1f}s of audio, all below threshold)")
    if pcm:
        with wave.open(args.output, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(bytes(pcm))
        print(f"  saved audio       : {args.output} ({len(pcm) // 2 / sr:.1f}s @ {sr} Hz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
