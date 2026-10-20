"""Eval datasets as JSONL manifests, plus builders that fetch public sets into one.

A manifest line is one item::

    {"id": "1089-134686-0000", "text": "he hoped there would be stew ...",
     "audio_path": "audio/1089-134686-0000.wav", "lang": "en",
     "ref_audio": null, "ref_text": null, "extra": {}}

``text`` is the TTS input or the ASR reference; ``audio_path`` the ASR or
duplex input; ``ref_audio``/``ref_text`` a voice-clone prompt (Seed-TTS).
Relative paths resolve against the manifest's directory, so a built dataset
directory can be copied between hosts as is. A ``.txt`` file is also a
manifest: one TTS sentence per line.

Builders (``python -m benchmark.eval data <name> --out DIR``) download into
the Hugging Face cache (``HF_HOME``) and write 16 kHz mono WAVs plus
``manifest.jsonl``. Dataset terms: LibriSpeech and FLEURS are CC BY 4.0;
Seed-TTS-eval's test set comes from Common Voice (CC0) and DiDiSpeech, for
research use (read from its Hugging Face mirror, or the official Google Drive
archive with gdown).
"""

from __future__ import annotations

import io
import json
import os
import re
import tarfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ASR_RATE = 16000
# Item ids name output files (``<audio_dir>/<id>.wav``), so they are plain names: no paths.
ITEM_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}")


_IPV4_IN_ID = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")


def check_id(item_id: str) -> str:
    """Ids are written to results files verbatim (rescoring matches them to the manifest), so
    they are plain names without paths or address literals."""
    if (not isinstance(item_id, str) or not ITEM_ID.fullmatch(item_id) or ".." in item_id
            or _IPV4_IN_ID.search(item_id)):
        raise ValueError(f"item id {item_id!r} must be a plain name (letters, digits, '.', '_', '-'; no paths "
                         "or IP addresses)")
    return item_id


@dataclass
class Item:
    id: str
    text: str | None = None
    audio_path: str | None = None
    lang: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def load_manifest(path: str | Path, limit: int | None = None) -> list[Item]:
    path = Path(path)
    items: list[Item] = []
    if path.suffix == ".txt":
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
        items = [Item(id=f"{i:04d}", text=ln) for i, ln in enumerate((ln for ln in lines if ln), start=1)]
    else:
        root = path.parent
        known = set(Item.__dataclass_fields__)
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            unknown = set(row) - known
            if unknown or "id" not in row:
                raise ValueError(f"{path}:{n}: needs 'id', unknown keys {sorted(unknown)}")
            for key in ("audio_path", "ref_audio"):
                if row.get(key) and not os.path.isabs(row[key]):
                    row[key] = str(root / row[key])
            items.append(Item(**row))
    ids = [check_id(it.id) for it in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate item ids")
    return items[:limit] if limit else items


def save_manifest(items: list[Item], path: str | Path) -> Path:
    """Write items, storing paths under the manifest's directory relative to it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = path.parent.resolve()
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            row = asdict(it)
            for key in ("audio_path", "ref_audio"):
                p = row.get(key)
                if p and Path(p).resolve().is_relative_to(root):
                    row[key] = str(Path(p).resolve().relative_to(root))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _write_16k(audio_bytes: bytes, dst: Path) -> None:
    import soundfile as sf

    from benchmark.eval.audio import resample, write_wav

    x, rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    write_wav(dst, resample(x.mean(axis=1), int(rate), ASR_RATE), ASR_RATE)


def build_librispeech(out: Path, split: str = "test.clean", limit: int = 200) -> Path:
    """LibriSpeech (``openslr/librispeech_asr``), the first ``limit`` utterances of a split
    (``test.clean``, ``test.other``, ``validation.clean``, ...), read from its parquet shards."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    repo = "openslr/librispeech_asr"
    part, _, sub = split.partition(".")
    sub = sub or "clean"
    if part in ("clean", "other"):  # "clean.test" style
        part, sub = sub, part
    prefix = f"{sub}/{part}/"
    try:
        shards = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                        if f.startswith(prefix) and f.endswith(".parquet"))
    except Exception:  # noqa: BLE001 - offline: the first shard may already be cached
        shards = [f"{prefix}0000.parquet"]
    items: list[Item] = []
    (out / "audio").mkdir(parents=True, exist_ok=True)
    for shard in shards:
        table = pq.read_table(hf_hub_download(repo, shard, repo_type="dataset"), columns=["id", "text", "audio"])
        for row in table.slice(0, limit - len(items)).to_pylist():
            dst = out / "audio" / f"{row['id']}.wav"
            if not dst.exists():
                _write_16k(row["audio"]["bytes"], dst)
            items.append(Item(id=row["id"], text=row["text"], audio_path=str(dst), lang="en"))
        if len(items) >= limit:
            break
    return save_manifest(items, out / "manifest.jsonl")


# FLEURS config prefixes that are not ISO-639-1 (the judge and the server expect "zh", not "cmn").
_FLEURS_LANG = {"cmn": "zh"}


def build_fleurs(out: Path, lang: str = "en_us", split: str = "test", limit: int = 100) -> Path:
    """FLEURS (``google/fleurs``) for one language config, e.g. ``en_us``, ``cmn_hans_cn``.

    The dataset is a TSV plus an audio tarball per language (no parquet), read
    directly. FLEURS records each sentence once per speaker; the first take is kept.
    ``text`` is the raw transcription (cased and punctuated), so it also works as TTS input.
    """
    import csv
    import sys

    from huggingface_hub import hf_hub_download

    csv.field_size_limit(sys.maxsize)
    tsv = hf_hub_download("google/fleurs", f"data/{lang}/{split}.tsv", repo_type="dataset")
    tgz = hf_hub_download("google/fleurs", f"data/{lang}/audio/{split}.tar.gz", repo_type="dataset")
    rows: list[list[str]] = []
    seen: set[str] = set()
    with open(tsv, newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if row[0] in seen:
                continue
            seen.add(row[0])
            rows.append(row)
            if len(rows) >= limit:
                break
    want = {r[1]: r for r in rows}
    found: dict[str, Path] = {}
    (out / "audio").mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz, "r:gz") as tf:
        for member in tf:
            name = os.path.basename(member.name)
            if member.isfile() and name in want and name not in found:
                dst = out / "audio" / f"{lang}_{Path(name).stem}.wav"
                if not dst.exists():
                    _write_16k(tf.extractfile(member).read(), dst)
                found[name] = dst
                if len(found) == len(want):
                    break
    short = lang.split("_", maxsplit=1)[0]
    items = [Item(id=f"{lang}_{r[0]}", text=r[2], audio_path=str(found[r[1]]), lang=_FLEURS_LANG.get(short, short),
                  extra={"normalized_text": r[3]}) for r in rows if r[1] in found]
    return save_manifest(items, out / "manifest.jsonl")


SEEDTTS_GDRIVE_ID = "1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP"  # from github.com/BytedanceSpeech/seed-tts-eval
SEEDTTS_HF_MIRROR = "zhaochenyang20/seed-tts-eval"


def parse_seedtts_meta(meta: Path, locale: str) -> list[Item]:
    """``utt_id|prompt_text|prompt_wav|target_text`` rows of a Seed-TTS ``meta.lst``."""
    items = []
    for line in meta.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("|")
        if len(parts) < 4 or not parts[3].strip():
            continue
        uid, ref_text, ref_wav, text = (p.strip() for p in parts[:4])
        items.append(Item(id=uid, text=text, lang=locale, ref_audio=str(meta.parent / ref_wav), ref_text=ref_text))
    return items


def build_seedtts(out: Path, locale: str = "en", limit: int | None = None, archive: str | None = None) -> Path:
    """Seed-TTS-eval zero-shot set (target text + reference prompt wav and transcript).

    Reads the Hugging Face mirror (``zhaochenyang20/seed-tts-eval``, same layout as the
    official archive) into the HF cache; falls back to the official Google Drive tarball
    (or ``archive``) when the mirror is unreachable.
    """
    meta = None
    if archive is None:
        try:
            from huggingface_hub import snapshot_download

            snap = Path(snapshot_download(SEEDTTS_HF_MIRROR, repo_type="dataset", allow_patterns=[f"{locale}/*"]))
            meta = snap / locale / "meta.lst"
        except Exception:  # noqa: BLE001 - fall back to the official archive
            meta = None
    if meta is None or not meta.exists():
        root = out / "seedtts_testset"
        meta = root / locale / "meta.lst"
        if not meta.exists():
            tar_path = Path(archive) if archive else out / "seedtts_testset.tar"
            if not tar_path.exists():
                import gdown

                out.mkdir(parents=True, exist_ok=True)
                gdown.download(f"https://drive.google.com/uc?id={SEEDTTS_GDRIVE_ID}", str(tar_path), quiet=False)
            with tarfile.open(tar_path) as tar:
                tar.extractall(out, filter="data")
    items = parse_seedtts_meta(meta, locale)
    return save_manifest(items[:limit] if limit else items, out / f"manifest_{locale}.jsonl")
