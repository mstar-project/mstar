"""Build the shared ASR benchmark inputs from the official LibriSpeech release.

Writes, under ``--out`` (the protocol's ``commons/bench/data/asr``):

    test_clean_200/<id>.wav       the first 200 test-clean utterances in utterance-id
                                  order (``1089-134686-0000``, ...), 16 kHz mono PCM16
    test_clean_200/references.tsv <id>\\t<reference text>
    long_form/<id>.wav            one ~10 min file: the consecutive utterances of one
                                  speaker/chapter joined with 0.5 s of silence
    long_form/references.tsv      <id>\\t<reference text>
    manifest.json                 counts, total audio seconds, per-file durations

The source is openslr.org's ``test-clean.tar.gz`` (FLAC + ``*.trans.txt``),
downloaded once into ``--raw-dir``; no ``datasets`` dependency, so the set is
the same wherever it is built. Every engine in the protocol table is
measured on these files by ``benchmark/asr_bench.py``::

    python -m benchmark.asr_data --out $MSTAR_WS_ROOT/commons/bench/data/asr
"""

from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
TEST_CLEAN_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return len(pcm) / sample_rate


def _decode(path: Path) -> np.ndarray:
    """FLAC -> float32 mono at 16 kHz (LibriSpeech is already 16 kHz mono).
    libsndfile via ``soundfile`` needs no FFmpeg on the build host."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import torch
        import torchaudio

        audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, SAMPLE_RATE).numpy()
    return audio


def fetch_test_clean(raw_dir: Path) -> Path:
    """Download and extract ``test-clean.tar.gz`` once; return the split dir."""
    split_dir = raw_dir / "LibriSpeech" / "test-clean"
    if split_dir.is_dir() and any(split_dir.iterdir()):
        return split_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    tarball = raw_dir / "test-clean.tar.gz"
    if not tarball.exists():
        print(f"downloading {TEST_CLEAN_URL} -> {tarball}", flush=True)
        urllib.request.urlretrieve(TEST_CLEAN_URL, tarball)  # noqa: S310 — fixed https URL
    with tarfile.open(tarball) as tar:
        tar.extractall(raw_dir, filter="data")
    return split_dir


def read_transcripts(split_dir: Path) -> dict[str, str]:
    """``utterance id -> reference`` for the whole split, in id order."""
    refs: dict[str, str] = {}
    for trans in sorted(split_dir.rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            if line.strip():
                uid, text = line.split(" ", 1)
                refs[uid] = text.strip()
    return dict(sorted(refs.items()))


def _utterance_path(split_dir: Path, uid: str) -> Path:
    spk, chap, _ = uid.split("-")
    return split_dir / spk / chap / f"{uid}.flac"


def build(out: Path, raw_dir: Path, num_utterances: int, long_form_minutes: float) -> dict:
    split_dir = fetch_test_clean(raw_dir)
    refs = read_transcripts(split_dir)

    short_dir = out / f"test_clean_{num_utterances}"
    short_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"sample_rate": SAMPLE_RATE, "source": TEST_CLEAN_URL, "short": {}, "long_form": {}}
    lines: list[str] = []
    total = 0.0
    for uid in list(refs)[:num_utterances]:
        duration = _write_wav(short_dir / f"{uid}.wav", _decode(_utterance_path(split_dir, uid)))
        manifest["short"][uid] = duration
        total += duration
        lines.append(f"{uid}\t{refs[uid]}")
    (short_dir / "references.tsv").write_text("\n".join(lines) + "\n")
    manifest["short_total_seconds"] = total

    # Long form: one speaker's utterances in id order (chapters back to back),
    # joined with 0.5 s silence, until the target length is reached — the
    # speaker with the most audio, so the file stays one voice. test-clean
    # speakers hold ~8 min each; if none reaches the target the id-ordered
    # split as a whole is used instead.
    long_dir = out / "long_form"
    long_dir.mkdir(parents=True, exist_ok=True)
    by_speaker: dict[str, list[str]] = {}
    for uid in refs:
        by_speaker.setdefault(uid.split("-", 1)[0], []).append(uid)
    target = long_form_minutes * 60.0
    gap = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
    candidates = [(spk, uids) for spk, uids in sorted(by_speaker.items(), key=lambda kv: -len(kv[1]))]
    candidates.append(("test-clean", list(refs)))
    for name, uids in candidates:
        pieces: list[np.ndarray] = []
        texts: list[str] = []
        length = 0.0
        for uid in uids:
            audio = _decode(_utterance_path(split_dir, uid))
            pieces += [audio, gap]
            texts.append(refs[uid])
            length += (len(audio) + len(gap)) / SAMPLE_RATE
            if length >= target:
                break
        if length >= target:
            uid = f"{name}-longform"
            duration = _write_wav(long_dir / f"{uid}.wav", np.concatenate(pieces))
            (long_dir / "references.tsv").write_text(f"{uid}\t{' '.join(texts)}\n")
            manifest["long_form"][uid] = duration
            break
    else:
        raise RuntimeError("test-clean is shorter than the requested long-form length")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="dataset directory (commons/bench/data/asr)")
    parser.add_argument("--raw-dir", default=None, help="where the openslr tarball is kept (default: <out>/raw)")
    parser.add_argument("--num-utterances", type=int, default=200)
    parser.add_argument("--long-form-minutes", type=float, default=10.0)
    args = parser.parse_args()
    out = Path(args.out)
    manifest = build(out, Path(args.raw_dir) if args.raw_dir else out / "raw", args.num_utterances,
                     args.long_form_minutes)
    print(json.dumps({
        "short_files": len(manifest["short"]),
        "short_total_seconds": round(manifest["short_total_seconds"], 1),
        "long_form": {k: round(v, 1) for k, v in manifest["long_form"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
