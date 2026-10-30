#!/usr/bin/env python3
"""Make a duplex input clip: a user recording followed by trailing silence for the agent's reply.

PersonaPlex's reply is as long as its input, so the silence is the agent's
turn. Prints where the user's speech ends, which ``benchmark.personaplex.sessions``
takes as ``--speech-end``.

    python test/personaplex/duplex_clip.py --audio question.wav --silence 8 --output user.wav
"""

import argparse
import wave

RATE = 24000


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", required=True, help="24 kHz mono PCM16 WAV of the user speaking")
    p.add_argument("--silence", type=float, default=8.0, help="seconds of silence to append")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    with wave.open(args.audio) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (RATE, 1, 2):
            raise SystemExit(f"{args.audio}: need {RATE} Hz mono PCM16, got {w.getframerate()} Hz, "
                             f"{w.getnchannels()} channel(s), {8 * w.getsampwidth()}-bit")
        speech = w.readframes(w.getnframes())
    with wave.open(args.output, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(speech + b"\x00\x00" * int(args.silence * RATE))
    print(f"speech ends at {len(speech) / 2 / RATE:.3f} s; wrote {args.output}")


if __name__ == "__main__":
    main()
