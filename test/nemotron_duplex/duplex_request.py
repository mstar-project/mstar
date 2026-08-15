#!/usr/bin/env python3
"""Nemotron-Duplex client: send user speech, get back agent text + agent speech.

Usage:
    python test/nemotron_duplex/duplex_request.py --audio user.wav
    python test/nemotron_duplex/duplex_request.py --audio user.wav --text-only
    python test/nemotron_duplex/duplex_request.py --audio user.wav --output agent.wav -n 3

The base VoiceChat-11B HF repo ships sample inputs (turn_taking.wav,
interruptions.wav, tool_call.wav) usable for --audio.
"""
import argparse
import sys
import threading
import time

from _env import get_base_url

from mstar.client.client import MStarClient


def one_request(url: str, audio: str, modalities, temperature: float):
    c = MStarClient(url, timeout=600)
    t0 = time.time()
    r = c.generate(
        audio=audio, input_modalities=("audio",),
        output_modalities=modalities, temperature=temperature,
    )
    return r, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="path to a 16 kHz-ish mono wav of user speech")
    ap.add_argument("--output", default="agent_out.wav", help="where to save agent speech")
    ap.add_argument("--text-only", action="store_true", help="request text output only (skip the talker/codec)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("-n", "--concurrency", type=int, default=1, help="fire N identical requests concurrently")
    args = ap.parse_args()

    url = get_base_url()
    modalities = ("text",) if args.text_only else ("text", "audio")
    print(f"POST {url}/generate  audio={args.audio}  out={modalities}  n={args.concurrency}")

    results: dict[int, tuple] = {}

    def worker(i):
        results[i] = one_request(url, args.audio, modalities, args.temperature)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    texts = [results[i][0].text for i in range(args.concurrency)]
    for i in range(args.concurrency):
        r, dt = results[i]
        n_audio = len(r.audio) if getattr(r, "audio", None) is not None else 0
        print(f"  req{i}: {dt:.1f}s  text_len={len(r.text or '')}  audio_samples={n_audio}")

    if args.concurrency > 1:
        print(f"  all identical: {all(t == texts[0] for t in texts)}")

    r0 = results[0][0]
    print("\n=== agent text ===")
    print(r0.text)
    if getattr(r0, "audio", None) is not None and len(r0.audio):
        r0.audio.to_wav(args.output)
        dur = len(r0.audio) / (r0.audio.sample_rate or 22050)
        print(f"\nsaved {args.output}  ({dur:.2f}s @ {r0.audio.sample_rate} Hz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
