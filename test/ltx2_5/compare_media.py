#!/usr/bin/env python3
"""Served LTX-2.5 output vs the oracle, at the media level, with yardsticks.

    python test/ltx2_5/compare_media.py --oracle <dir>/t2av --second-oracle <dir>/t2av_efficient \\
        --served <client out dir>

Video PSNR is against the oracle's raw uint8 frames. Three rows put the served number
in context: the H.264 round trip of the oracle's own frames (the codec floor), and a
second reference run under the other SDPA kernel (the reference's own spread).
"""
import argparse
import io
import wave
from pathlib import Path

import numpy as np
import torch


def frames_from_mp4(data: bytes) -> np.ndarray:
    import av

    with av.open(io.BytesIO(data)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def wav_samples(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, w.getnchannels())
    return x.T.astype(np.float32) / 32767.0


def audio_stats(a: np.ndarray, b: np.ndarray) -> str:
    n = min(a.shape[-1], b.shape[-1])
    a, b = a[..., :n], b[..., :n]
    rel = np.linalg.norm(a - b) / np.linalg.norm(b)
    corr = np.corrcoef(a.flatten(), b.flatten())[0, 1]

    def logmel_like(x):
        spec = np.abs(np.fft.rfft(x.reshape(x.shape[0], -1, 1024), axis=-1)) + 1e-5
        return np.log(spec)
    spec_err = np.mean(np.abs(logmel_like(a[:, : n // 1024 * 1024]) - logmel_like(b[:, : n // 1024 * 1024])))
    return f"rel_l2={rel:.3f} corr={corr:.3f} log_spectrum_mae={spec_err:.3f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oracle", required=True)
    p.add_argument("--second-oracle")
    p.add_argument("--served", required=True)
    args = p.parse_args()
    ref = torch.load(Path(args.oracle) / "video.pt").numpy()
    ref_wave = torch.load(Path(args.oracle) / "audio_wave.pt").numpy()

    from mstar.model.ltx2_5.ltx2_5_model import LTX25Model

    codec = frames_from_mp4(LTX25Model(skip_weight_loading=True).postprocess(
        torch.from_numpy(ref).permute(3, 0, 1, 2), "video", {"fps": 24}))
    print(f"codec floor (oracle frames through H.264)   PSNR {psnr(codec, ref):6.2f} dB")
    if args.second_oracle:
        second = torch.load(Path(args.second_oracle) / "video.pt").numpy()
        print(f"second reference run (efficient SDPA)       PSNR {psnr(second, ref):6.2f} dB")
        print(f"   audio  {audio_stats(torch.load(Path(args.second_oracle) / 'audio_wave.pt').numpy(), ref_wave)}")
    served = frames_from_mp4((Path(args.served) / "video.mp4").read_bytes())
    print(f"served (mstar)                              PSNR {psnr(served, ref):6.2f} dB  frames={served.shape}")
    print(f"   audio  {audio_stats(wav_samples(Path(args.served) / 'audio.wav'), ref_wave)}")


if __name__ == "__main__":
    main()
