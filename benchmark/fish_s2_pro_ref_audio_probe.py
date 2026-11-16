"""Isolate Fish S2-Pro reference-clip validation and CPU loading costs.

Run on the host where a headerless FLAC was slow, using the actual WAV and
FLAC clips from the served comparison::

    python -m benchmark.fish_s2_pro_ref_audio_probe \
        --wav reference.wav --flac reference.flac --repeat 3 --output probe.json

Each measurement starts a fresh Python process. Imports and the temporary
copy used to protect the input from validation's delete-on-error behavior are
outside the timed sections. This does not measure model inference or TTFA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MODULE = "benchmark.fish_s2_pro_ref_audio_probe"


def measure_file(path: Path) -> dict:
    import soundfile

    from mstar.api_server.media_io import check_ref_audio
    from mstar.model.fish_s2_pro.config import FishS2ProConfig
    from mstar.model.fish_s2_pro.fish_s2_pro_model import FishS2ProModel

    info = soundfile.info(str(path))
    model = object.__new__(FishS2ProModel)
    model.config = FishS2ProConfig()
    with tempfile.TemporaryDirectory(prefix="mstar-ref-audio-probe-") as directory:
        check_path = Path(directory) / path.name
        shutil.copyfile(path, check_path)
        start = time.perf_counter_ns()
        check_ref_audio(str(check_path), "soundfile")
        validation_ns = time.perf_counter_ns() - start

    start = time.perf_counter_ns()
    loaded = model.load_audio(str(path), "cpu")
    loading_ns = time.perf_counter_ns() - start
    return {
        "file_name": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
        "header_frames": info.frames,
        "header_sample_rate": info.samplerate,
        "decoded_samples": int(loaded.data.numel()),
        "validation_ms": validation_ns / 1e6,
        "load_ms": loading_ns / 1e6,
        "soundfile_version": soundfile.__version__,
        "libsndfile_version": soundfile.__libsndfile_version__,
        "python_version": platform.python_version(),
    }


def run_probe(wav: Path, flac: Path, repeat: int) -> dict:
    if repeat < 1:
        raise ValueError("--repeat must be at least 1")
    for path in (wav, flac):
        if not path.is_file():
            raise FileNotFoundError(path)

    samples: list[dict] = []
    for _ in range(repeat):
        for label, path in (("wav", wav), ("flac", flac)):
            result = subprocess.run(
                [sys.executable, "-m", MODULE, "--one", str(path)],
                check=True, capture_output=True, text=True,
            )
            samples.append({"format": label, **json.loads(result.stdout)})
    return {"repeat": repeat, "samples": samples}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path)
    parser.add_argument("--flac", type=Path)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--one", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.one is not None:
        print(json.dumps(measure_file(args.one), sort_keys=True))
        return
    if args.wav is None or args.flac is None:
        parser.error("--wav and --flac are required")
    report = run_probe(args.wav, args.flac, args.repeat)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
