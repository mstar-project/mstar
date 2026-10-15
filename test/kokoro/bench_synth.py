"""Time the Kokoro network on one device, eager, per batch size and phoneme
length, to size the CUDA-graph buckets and see the compute ceiling.

    python test/kokoro/bench_synth.py --device cuda --batch-sizes 1 8 32 --phonemes 40 120 300

Prints ms per batch, audio seconds produced per second of compute (throughput)
and the real-time factor (compute s / audio s). Uses random phoneme ids of the
requested lengths and the ``af_heart`` style; needs the checkpoint in the HF cache.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from mstar.model.kokoro.components import KokoroTTS
from mstar.model.kokoro.config import BOUNDARY_TOKEN_ID, KokoroModelConfig
from mstar.model.kokoro.kokoro_model import _resolve_snapshot
from mstar.model.kokoro.voices import VoiceRegistry
from mstar.model.kokoro.weight_loader import load_kokoro_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--phonemes", type=int, nargs="+", default=[40, 120, 300])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--compile", action="store_true", help="torch.compile decode_frames with dynamic shapes")
    parser.add_argument("--compile-text", action="store_true", help="also torch.compile encode_text")
    parser.add_argument("--decoder-dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    args = parser.parse_args()

    local_dir = Path(_resolve_snapshot("hexgrad/Kokoro-82M", None, None))
    config = KokoroModelConfig.from_pretrained(local_dir)
    config.decoder_dtype = args.decoder_dtype
    model = KokoroTTS(config)
    load_kokoro_weights(model, local_dir / config.weights_file)
    model = model.to(args.device).eval()
    if args.compile:
        model.decode_frames = torch.compile(model.decode_frames, dynamic=True)
    if args.compile_text:
        model.encode_text = torch.compile(model.encode_text, dynamic=True)
    voices = VoiceRegistry(local_dir / config.voices_dir, config.style_pack_rows, config.style_dim)
    ids = torch.tensor([v for k, v in config.vocab.items() if k.isalpha()], device=args.device)

    def sync():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    print(f"{'bs':>4} {'T':>4} {'frames':>7} {'ms/batch':>9} {'audio_s/s':>10} {'RTF':>7}")
    for num_phonemes in args.phonemes:
        for bs in args.batch_sizes:
            torch.manual_seed(0)
            body = ids[torch.randint(len(ids), (bs, num_phonemes), device=args.device)]
            boundary = body.new_full((bs, 1), BOUNDARY_TOKEN_ID)
            input_ids = torch.cat([boundary, body, boundary], 1)
            lengths = torch.full((bs,), num_phonemes + 2, device=args.device)
            style = voices.style(args.voice, num_phonemes).to(args.device)[None].expand(bs, -1).contiguous()
            speed = torch.ones(bs, device=args.device)
            times = []
            with torch.no_grad():
                for i in range(args.repeats + 2):
                    sync()
                    t0 = time.perf_counter()
                    audio, frame_lengths, _ = model(input_ids, lengths, style, speed)
                    sync()
                    if i >= 2:
                        times.append(time.perf_counter() - t0)
            ms = statistics.median(times) * 1000
            audio_seconds = frame_lengths.sum().item() * config.samples_per_frame / config.sample_rate
            throughput = audio_seconds / (ms / 1000)
            rtf = (ms / 1000) / audio_seconds
            frames = int(frame_lengths.max())
            print(f"{bs:>4} {num_phonemes + 2:>4} {frames:>7} {ms:>9.1f} {throughput:>10.1f} {rtf:>7.4f}")


if __name__ == "__main__":
    main()
