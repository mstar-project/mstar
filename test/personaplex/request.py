#!/usr/bin/env python3
"""PersonaPlex client: send a prerecorded user WAV, get back the agent track (+ text).

Usage:
    python test/personaplex/request.py --audio user.wav
    python test/personaplex/request.py --audio user.wav --voice NATM1 --output agent.wav
    python test/personaplex/request.py --audio user.wav --audio-only --greedy
    python test/personaplex/request.py --audio user.wav --persona "You are a helpful travel agent."

The response is the agent's side of a full-duplex conversation, time-aligned to
the input: it has the input's length and lags it by one 80 ms frame. Input is
decoded at 24 kHz and channel 0 is used. The server runs each request at batch
size 1: concurrent requests share the GPU but are not batched together.
"""

import argparse
import time

from _env import get_base_url

from mstar.client.client import MStarClient


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", required=True, help="user recording (any format FFmpeg decodes)")
    p.add_argument("--voice", default="NATF2", help="voice preset name (see GET /v1/audio/voices)")
    p.add_argument("--persona", default=None, help="persona text prompt (default: NVIDIA's offline default)")
    p.add_argument("--audio-only", action="store_true", help="request audio without the text stream")
    p.add_argument("--greedy", action="store_true", help="greedy decoding instead of NVIDIA's sampling defaults")
    p.add_argument("--output", default="agent.wav", help="where to write the agent track")
    p.add_argument("--url", default=None, help="server URL (default: HOST/PORT from .env)")
    args = p.parse_args()

    client = MStarClient(args.url or get_base_url(), timeout=900)
    outputs = ("audio",) if args.audio_only else ("audio", "text")
    start = time.perf_counter()
    result = client.generate(
        audio=args.audio, input_modalities=("audio",), output_modalities=outputs,
        voice=args.voice, text_prompt=args.persona, greedy=args.greedy or None,
    )
    elapsed = time.perf_counter() - start
    print(f"saved {result.save_audio(args.output)} in {elapsed:.1f} s")
    if not args.audio_only:
        print(f"agent text: {result.text or ''!r}")


if __name__ == "__main__":
    main()
