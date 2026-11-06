#!/usr/bin/env python3
"""PersonaPlex client: send a prerecorded user WAV, get back the agent track (+ text).

Usage:
    python test/personaplex/request.py --audio user.wav
    python test/personaplex/request.py --audio user.wav --voice NATM1 --output agent.wav
    python test/personaplex/request.py --audio user.wav --audio-only --greedy
    python test/personaplex/request.py --audio user.wav --persona "You are a helpful travel agent."
    python test/personaplex/request.py --audio user.wav --stream   # print chunk arrivals as they come

The response is the agent's side of a full-duplex conversation, time-aligned to
the input: it has the input's length and lags it by one 80 ms frame. Input is
decoded at 24 kHz and channel 0 is used. Concurrent requests can share the
server's LM steps (up to its ``max_batch_size``), which changes rounding, so a
greedy reply can differ slightly from the same request served alone.
"""

import argparse
import time
import wave

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
    p.add_argument("--stream", action="store_true", help="stream the reply and print when each chunk arrives")
    args = p.parse_args()

    client = MStarClient(args.url or get_base_url(), timeout=900)
    outputs = ("audio",) if args.audio_only else ("audio", "text")
    start = time.perf_counter()
    if args.stream:
        stream(client, args, outputs, start)
        return
    result = client.generate(
        audio=args.audio, input_modalities=("audio",), output_modalities=outputs,
        voice=args.voice, text_prompt=args.persona, greedy=args.greedy or None,
    )
    elapsed = time.perf_counter() - start
    print(f"saved {result.save_audio(args.output)} in {elapsed:.1f} s")
    if not args.audio_only:
        print(f"agent text: {result.text or ''!r}")


def stream(client, args, outputs, start) -> None:
    """The reply arrives in chunks while the model works; write them as one track."""
    from mstar.client.types import AudioChunk, TextChunk

    pcm, text = [], []
    for event in client.generate(audio=args.audio, input_modalities=("audio",), output_modalities=outputs,
                                 voice=args.voice, text_prompt=args.persona, greedy=args.greedy or None,
                                 stream=True):
        if isinstance(event, AudioChunk) and event.pcm:
            pcm.append(event.pcm)
            print(f"{time.perf_counter() - start:7.2f} s  audio chunk, {len(event.pcm) // 2 / 24000:.2f} s of sound")
        elif isinstance(event, TextChunk):
            text.append(event.text)
    with wave.open(args.output, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"".join(pcm))
    print(f"saved {args.output} in {time.perf_counter() - start:.1f} s")
    if text:
        print(f"agent text: {''.join(text)!r}")


if __name__ == "__main__":
    main()
