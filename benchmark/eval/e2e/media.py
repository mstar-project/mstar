"""Inputs for ``compare``: the speech sets from the eval dataset builders, plus request media drawn here
(seeded or synthetic, so every host and every run sends the same bytes and nothing is shipped in the repo).

``python -m benchmark.eval data e2e --out DIR`` writes::

    tts/manifest.jsonl  clone/  asr/  duplex/     speech sets (cases.yaml ``datasets``)
    media/              chat, generation, VLA and world-model inputs (the ``e2e`` suite)
    rp/                 request-path inputs: formats, odd images, a 20-minute WAV, a 4K noise PNG
"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import numpy as np

from benchmark.eval import datasets
from benchmark.eval.datasets import Item

N_TTS, N_ASR, N_DUPLEX = 24, 100, 16
LONG_WAV_S = 20 * 60


def _png(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def scene(width: int = 224, height: int = 224, shift: int = 0):
    """Wall, table, block and gripper (the robotics scene of test/e2e in #410), ``shift`` pixels to the right."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), (200, 200, 205))
    draw = ImageDraw.Draw(image)
    sx, sy = width / 224, height / 224
    draw.rectangle((0, 150 * sy, width, height), fill=(120, 90, 60))
    draw.rectangle(((95 + shift) * sx, 120 * sy, (130 + shift) * sx, 155 * sy), fill=(200, 30, 30))
    draw.rectangle(((100 + shift) * sx, 20 * sy, (125 + shift) * sx, 80 * sy), fill=(60, 60, 60))
    return image


def moving_square_mp4(frames: int = 16, size: int = 256, fps: int = 8) -> bytes:
    import av

    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        for i in range(frames):
            pixels = np.full((size, size, 3), 40, dtype=np.uint8)
            x = 16 + i * (size - 96) // max(frames - 1, 1)
            pixels[96:160, x:x + 64] = (230, 200, 40)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buf.getvalue()


def write_media(out: Path, speech_wav: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "image.png").write_bytes(_png(scene(640, 480)))
    (out / "i2v.png").write_bytes(_png(scene(832, 480)))
    (out / "waypoint_seed.png").write_bytes(_png(scene(640, 360)))
    for name, shift in (("base", 0), ("left", -30), ("right", 30)):
        (out / f"pi05_{name}.png").write_bytes(_png(scene(224, 224, shift)))
    (out / "video.mp4").write_bytes(moving_square_mp4())
    shutil.copyfile(speech_wav, out / "audio.wav")


def _encode_audio(x: np.ndarray, rate: int, fmt: str, codec: str) -> bytes:
    import av

    buf = io.BytesIO()
    with av.open(buf, mode="w", format=fmt) as c:
        s = c.add_stream(codec, rate=rate)
        s.layout = "mono"
        frame = av.AudioFrame.from_ndarray((x * 32767).astype(np.int16)[None, :], format="s16", layout="mono")
        frame.sample_rate = rate
        for p in s.encode(frame):
            c.mux(p)
        for p in s.encode():
            c.mux(p)
    return buf.getvalue()


def write_rp(out: Path, asr_items: list[Item]) -> None:
    """Request-path inputs. Only status codes and identity flags are compared, so content need only be valid."""
    from PIL import Image

    from benchmark.eval.audio import read_wav, write_wav

    out.mkdir(parents=True, exist_ok=True)
    x, rate = read_wav(asr_items[0].audio_path)
    write_wav(out / "speech.wav", x, rate)
    for fmt, container, codec in (("mp3", "mp3", "libmp3lame"), ("flac", "flac", "flac"), ("ogg", "ogg", "libopus"),
                                  ("m4a", "ipod", "aac")):
        (out / f"speech.{fmt}").write_bytes(_encode_audio(x, rate, container, codec))
    clips = [read_wav(it.audio_path)[0] for it in asr_items]
    reps = int(np.ceil(LONG_WAV_S * rate / sum(len(c) for c in clips)))
    write_wav(out / "long.wav", np.concatenate(clips * reps)[: LONG_WAV_S * rate], rate)
    img = scene(640, 480)
    for ext, fmt in (("png", "PNG"), ("jpg", "JPEG"), ("webp", "WEBP"), ("gif", "GIF")):
        img.save(out / f"img.{ext}", format=fmt)
    Image.fromarray((np.asarray(img.convert("L"), dtype=np.uint16) * 257)).save(out / "img16.png")
    rgba = img.convert("RGBA")
    rgba.putalpha(128)
    rgba.save(out / "rgba.png")
    img.convert("CMYK").save(out / "cmyk.jpg", format="JPEG")
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90 degrees on display
    img.save(out / "exif_rot.jpg", format="JPEG", exif=exif)
    frames = [scene(320, 240, s) for s in range(0, 60, 6)]
    frames[0].save(out / "anim.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
    (out / "video.mp4").write_bytes(moving_square_mp4())
    noise = np.random.default_rng(0).integers(0, 256, (2160, 3840, 3), dtype=np.uint8)
    Image.fromarray(noise).save(out / "noise4k.png")


def build(out: str | Path, seedtts_archive: str | None = None) -> Path:
    """Every input ``compare`` sends, under ``out`` (about 150 MB; the 20-minute WAV and 4K PNG are most of it)."""
    out = Path(out).expanduser()
    asr = datasets.load_manifest(datasets.build_librispeech(out / "asr", "test.clean", N_ASR))
    seed = datasets.load_manifest(datasets.build_seedtts(out / "seedtts", "en", N_TTS, seedtts_archive))
    datasets.save_manifest([Item(id=it.id, text=it.text, lang=it.lang) for it in seed], out / "tts" / "manifest.jsonl")
    ref_dir = out / "clone" / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    clone = []
    for it in seed:
        dst = ref_dir / f"{it.id}{Path(it.ref_audio).suffix}"
        shutil.copyfile(it.ref_audio, dst)   # the HF cache path is per host; the copy travels with the directory
        clone.append(Item(id=it.id, text=it.text, lang=it.lang, ref_audio=str(dst), ref_text=it.ref_text))
    datasets.save_manifest(clone, out / "clone" / "manifest.jsonl")
    from benchmark.eval.suites.duplex import compose_items

    datasets.save_manifest(compose_items(asr, out / "duplex" / "audio", N_DUPLEX), out / "duplex" / "manifest.jsonl")
    write_media(out / "media", Path(asr[0].audio_path))
    write_rp(out / "rp", asr[:20])
    (out / "e2e-data.json").write_text(json.dumps({"tts": N_TTS, "asr": N_ASR, "duplex": N_DUPLEX}) + "\n")
    return out
