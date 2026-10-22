"""Full-Duplex-Bench v1.5: the official scoring sources and dataset, fetched at run time.

Nothing here is vendored. Full-Duplex-Bench (github.com/DanielLin94144/Full-Duplex-Bench)
is licensed CC BY-NC 4.0 (non-commercial), so its scoring code and its v1.5 data are
downloaded from the official sources into the Hugging Face cache when a run needs them.
Each is pinned to one revision and checked against its SHA-256, except the v1.5 archives:
Google Drive keeps no revisions, so ``download_dataset`` hashes them and records the hash
with every result as the dataset revision instead of checking it:

* ``get_timing.py``: stop and response intervals from Silero VAD speech spans (its
  ``torch.hub.load`` of Silero is bound to the pinned ``silero-vad`` package, so the
  VAD weights are the package's, not whatever GitHub serves that day);
* ``eval_behavior.py`` and ``instruction/behavior.txt``: the exact behavior-label prompt
  (the f-string and helpers are read from the file's syntax tree, so importing it does
  not create an OpenAI client);
* ``get_transcript/asr.py``: Parakeet TDT 0.6B v2 word timestamps, run unmodified on a
  preloaded local checkpoint;
* the semantic A/F/U judge's frozen prompt, schema and control cases from sgl-project/sglang-omni
  (Apache-2.0), so its quality and coverage numbers are comparable with the reference runner;
* the four v1.5 subsets from the official Google Drive (``gdown``), hashed and recorded.

The pins match the reference runner (``benchmarks/duplex/fdb_v15`` at ``SEMANTIC_REVISION``),
which reports the same metrics, so numbers from either runner use one definition.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import types
import urllib.request
import wave
import zipfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FDB_REPO = "DanielLin94144/Full-Duplex-Bench"
FDB_REVISION = "3e799c45a045256f47d5f1c9cda90157e2d2ec9e"
FDB_LICENSE = "CC BY-NC 4.0"
FDB_FILES = {
    "timing": ("v1_v1.5/evaluation/get_timing.py",
               "4f551da4194ab4d9584db964f4b27223914eecf98cbc312388ecf45eaf6f8a17"),
    "behavior": ("v1_v1.5/evaluation/eval_behavior.py",
                 "0ff8179a437503581d65787da3a43924b45310c31a98d1bcbadce5c8605ca6f2"),
    "instruction": ("v1_v1.5/evaluation/instruction/behavior.txt",
                    "19e5477dac9a9a1e11de126783a0b820b3ecb70db5e91181824fa944e1947977"),
    "asr": ("v1_v1.5/get_transcript/asr.py",
            "aedaee0d50f2bc47947caf6f3899939461e290225a98c0ddc434c360189596cf"),
}
SEMANTIC_REPO = "sgl-project/sglang-omni"
SEMANTIC_REVISION = "ed5adba0e97c8e8e671fcb130c93501382c22d6b"
SEMANTIC_FILES = {
    "prompt": ("benchmarks/duplex/semantic/judge-prompt.txt",
               "330b30b36d7c2a5407c91b7b8666f50a0c9f86e1c513ef8a0df3b2cadbf246e3"),
    "schema": ("benchmarks/duplex/semantic/judge-response.schema.json",
               "5d16454ae41a1e807e72c4b7b31ae9ef0572214c633492ffb9f1f97da2ee2067"),
    "controls": ("benchmarks/duplex/semantic/judge-controls.jsonl",
                 "27999dfed515b5cabb6a38dff2e42190100770e2b65786610ed74996610a8a90"),
    "expected": ("benchmarks/duplex/semantic/control-expected.json",
                 "9020a49ab322a6e51572b650f83035e5c308ffe673151e5c0446e61e1e83f238"),
}
PARAKEET_REPO = "nvidia/parakeet-tdt-0.6b-v2"
PARAKEET_FILE = "parakeet-tdt-0.6b-v2.nemo"
PARAKEET_REVISION = "ae9ad07059c7c739ffaf932226a8fe64ae2620b0"
PARAKEET_SHA256 = "d99e39955c9d3d0350d8fb7c75e40c64a2b2eaeb003883d7c941fd2e8747b28c"

CATEGORIES = ("user_interruption", "user_backchannel", "talking_to_other", "background_speech")
# metadata.json key holding the overlapping event's text, per category.
EVENT_TEXT_KEYS = {
    "user_interruption": "current_turn_text",
    "talking_to_other": "current_turn_text",
    "user_backchannel": "backchannel_text",
    "background_speech": "background_text",
}
DATASET_DRIVE_IDS = {
    "user_interruption": "1wqYcYS4-30W2YMc3TfeMeoiaW9PLf4yT",
    "user_backchannel": "1EGwCd9CdGuh8jeqwPENzbBs5FSEgnYL3",
    "talking_to_other": "1Jh6ER4AUmqGgEZBTV0pcbDaMMQIWA7Kt",
    "background_speech": "1W63k1BlQ0QCgvYCb_8YJNqFfUhBwI97W",
}
RATE = 16000


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cache_root() -> Path:
    hf_home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return Path(hf_home) / "mstar-eval"


def fetch(repo: str, revision: str, rel_path: str, sha256: str) -> Path:
    """One file of a GitHub repo at a pinned revision, cached and hash-checked."""
    dst = cache_root() / "sources" / repo.replace("/", "__") / revision / rel_path
    if not dst.exists() or sha256_file(dst) != sha256:
        url = f"https://raw.githubusercontent.com/{repo}/{revision}/{rel_path}"
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise RuntimeError(f"{repo}@{revision[:12]}:{rel_path} does not match its pinned SHA-256")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(dst)
    return dst


def official_file(key: str) -> Path:
    return fetch(FDB_REPO, FDB_REVISION, *FDB_FILES[key])


def semantic_file(key: str) -> Path:
    return fetch(SEMANTIC_REPO, SEMANTIC_REVISION, *SEMANTIC_FILES[key])


def provenance() -> dict[str, Any]:
    """What the scores are computed with, for the results file."""
    return {
        "fdb": {"repo": FDB_REPO, "revision": FDB_REVISION, "license": FDB_LICENSE,
                "files": {k: v[1] for k, v in FDB_FILES.items()}},
        "semantic_assets": {"repo": SEMANTIC_REPO, "revision": SEMANTIC_REVISION,
                            "files": {k: v[1] for k, v in SEMANTIC_FILES.items()}},
    }


def _load_module(path: Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- timing -----------------------------------------------------------------------------------


@dataclass
class Timing:
    module: types.ModuleType
    record: dict[str, Any]


def load_timing(path: Path | None = None, silero_loader: Callable[[], Any] | None = None) -> Timing:
    """Import the official ``get_timing.py`` with its ``torch.hub.load`` of Silero bound to
    the installed ``silero-vad`` package. Its formulas and constants are used as is."""
    import torch

    path = path or official_file("timing")
    if silero_loader is None:
        def silero_loader():
            from silero_vad import load_silero_vad

            return load_silero_vad(onnx=False)

    def hub_load(repo: str, model: str, *args: Any, **kwargs: Any):
        expected = ("snakers4/silero-vad", "silero_vad", (), {"trust_repo": True, "onnx": False})
        if (repo, model, args, kwargs) != expected:
            raise RuntimeError(f"unexpected torch.hub.load({repo!r}, {model!r}, {args}, {kwargs})")
        return silero_loader(), None

    original = torch.hub.load
    torch.hub.load = hub_load
    try:
        module = _load_module(path, "fdb_v15_get_timing")
    finally:
        torch.hub.load = original
    record = {
        "vad_branch": "VoiceActivityDetector" if hasattr(module, "_VAD") else "get_speech_timestamps",
        "silero_vad": _version("silero-vad"),
        "constants": {k: getattr(module, k) for k in ("SR", "USER_MERGE_GAP", "MODEL_MERGE_GAP")},
        "audio_loader": "official torchaudio.load",
    }
    return Timing(module, record)


def _soundfile_loader(sr_target: int):
    """``torchaudio.load`` without torchcodec: the same float32 [C, T], then the official
    resample and squeeze."""
    import soundfile
    import torch
    import torchaudio

    def load_wav(path):
        x, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        t = torch.from_numpy(x.T.copy())
        if rate != sr_target:
            t = torchaudio.functional.resample(t, rate, sr_target)
        return t.squeeze(0)

    return load_wav


def run_timing(timing: Timing, input_wav: Path, output_wav: Path, folder: Path) -> dict[str, Any]:
    """The official ``process_folder`` on one session: stop and response intervals plus the
    merged speech spans they came from."""
    m = timing.module
    folder.mkdir(parents=True, exist_ok=True)
    for name, src in (("input.wav", input_wav), ("output.wav", output_wav)):
        dst = folder / name
        if dst.resolve() != Path(src).resolve():
            dst.unlink(missing_ok=True)
            os.symlink(Path(src).resolve(), dst)
    if timing.record["audio_loader"].startswith("official"):
        try:
            m.load_wav(folder / "input.wav")
        except (ImportError, RuntimeError, OSError):
            m.load_wav = _soundfile_loader(m.SR)
            timing.record["audio_loader"] = "soundfile.read(float32) + torchaudio resample bridge"
    merged: list[list[tuple[float, float]]] = []
    seg_sec = m.seg_sec

    def capture(wav, gap):
        segs = seg_sec(wav, gap)
        merged.append(segs)
        return segs

    m.seg_sec = capture
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            m.process_folder(folder)
    finally:
        m.seg_sec = seg_sec
    doc = json.loads((folder / m.OUT_FILENAME).read_text())
    return {
        "stop_intervals": doc["latency_stop_list"],
        "response_intervals": doc["latency_resp_list"],
        "user_segments": [[round(s, 3), round(e, 3)] for s, e in merged[0]],
        "model_segments": [[round(s, 3), round(e, 3)] for s, e in merged[1]],
    }


# --- behavior prompt --------------------------------------------------------------------------


@dataclass
class Behavior:
    instruction: str
    template: Callable[[str, str, str, str], str]
    compact: Callable[[Any], str]
    parse_eval: Callable[[Any], dict]
    model: str = "gpt-4o-2024-08-06"
    initial_seed: int = 1

    def user_message(self, transcripts: dict[str, Any]) -> str:
        return self.template(*(self.compact(transcripts[k])
                               for k in ("clean_input", "input", "clean_output", "output")))


def load_behavior(path: Path | None = None, instruction_path: Path | None = None) -> Behavior:
    """The official prompt without importing ``eval_behavior.py`` (which builds an OpenAI
    client at import): its helpers and the ``final_input`` f-string come from the syntax tree."""
    path = path or official_file("behavior")
    instruction_path = instruction_path or official_file("instruction")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    names = ("json_dict_to_compact_text", "extract_json", "parse_eval")
    ns: dict[str, Any] = {"json": json, "Counter": Counter, "Dict": dict, "Any": Any, "Union": Any, "List": list}
    exec(compile(ast.Module(body=[funcs[n] for n in names], type_ignores=[]), str(path), "exec"), ns)  # noqa: S102
    final = [n for n in ast.walk(funcs["eval_behavior_all"]) if isinstance(n, ast.Assign)
             and [getattr(t, "id", None) for t in n.targets] == ["final_input"]]
    if len(final) != 1 or not isinstance(final[0].value, ast.JoinedStr):
        raise RuntimeError("eval_behavior_all's final_input f-string not found")
    fields = ("input_clean_text", "input_noisy_text", "output_clean_text", "output_noisy_text")
    lam = ast.Expression(ast.Lambda(
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=f) for f in fields], kwonlyargs=[], kw_defaults=[],
                           defaults=[]),
        body=final[0].value))
    ast.fix_missing_locations(lam)
    template = eval(compile(lam, str(path), "eval"), {})  # noqa: S307
    return Behavior(instruction_path.read_text(encoding="utf-8"), template, ns["json_dict_to_compact_text"],
                    ns["parse_eval"])


# --- ASR --------------------------------------------------------------------------------------


class ParakeetASR:
    """The official ``get_time_aligned_transcription`` on a preloaded local Parakeet
    checkpoint (SHA-256 checked). Returns the official JSON bytes verbatim, since the judge
    prompt is built from them."""

    def __init__(self, device: str = "cuda"):
        import torch
        from huggingface_hub import hf_hub_download

        nemo_path = hf_hub_download(PARAKEET_REPO, PARAKEET_FILE, revision=PARAKEET_REVISION)
        if sha256_file(nemo_path) != PARAKEET_SHA256:
            raise RuntimeError(f"{PARAKEET_FILE} does not match its pinned SHA-256")
        import nemo.collections.asr as nemo_asr

        model = nemo_asr.models.ASRModel.restore_from(restore_path=nemo_path, map_location=torch.device(device))
        # NeMo's CUDA-graph greedy decoder truncates Parakeet transcripts on some GPUs; the reference runner
        # turns it off too.
        decoding = dict(model.cfg.decoding)
        decoding["greedy"] = {**decoding["greedy"], "use_cuda_graph_decoder": False}
        model.change_decoding_strategy(decoding)
        model.eval()
        self.model, self.device = model, device
        self.official = _load_module(official_file("asr"), "fdb_v15_asr")
        self.official.nemo_asr = types.SimpleNamespace(models=types.SimpleNamespace(
            ASRModel=types.SimpleNamespace(from_pretrained=self._from_pretrained)))
        self.id = f"parakeet:{PARAKEET_REPO}@{PARAKEET_REVISION[:12]}"
        self._cache: dict[str, bytes] = {}

    def _from_pretrained(self, model_name: str):
        if model_name != PARAKEET_REPO:
            raise RuntimeError(f"official ASR asked for {model_name!r}")
        model, device = self.model, self.device
        return types.SimpleNamespace(cuda=lambda: model.cuda() if device.startswith("cuda") else model)

    def transcribe(self, audio: Path, stage: Path) -> dict[str, Any]:
        key = sha256_file(audio)
        if key not in self._cache:
            shutil.rmtree(stage, ignore_errors=True)
            (stage / "item").mkdir(parents=True)
            os.symlink(Path(audio).resolve(), stage / "item" / "audio.wav")
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.official.get_time_aligned_transcription(str(stage), "default", "audio.wav")
                self._cache[key] = (stage / "item" / "audio.json").read_bytes()
            finally:
                shutil.rmtree(stage, ignore_errors=True)
        return json.loads(self._cache[key])


# --- dataset ----------------------------------------------------------------------------------


def download_dataset(zips: Path | None = None) -> tuple[Path, str]:
    """The four v1.5 subsets, extracted under the cache. Returns (root, revision), where the
    revision is a hash of the archives' SHA-256s (the dataset has no version tag)."""
    root = cache_root() / "fdb-v15"
    zips = zips or root / "zips"
    zips.mkdir(parents=True, exist_ok=True)
    for cat, drive_id in DATASET_DRIVE_IDS.items():
        z = zips / f"{cat}.zip"
        if not z.is_file() or z.stat().st_size == 0:
            import gdown

            try:
                gdown.download(id=drive_id, output=str(z), quiet=True)
            except Exception as exc:  # noqa: BLE001 - gdown raises requests, OS and its own errors
                raise SystemExit(f"{z} is missing and its download from Google Drive failed "
                                 f"({type(exc).__name__}); put the official zips there") from None
            if not z.is_file() or z.stat().st_size == 0:
                raise SystemExit(f"{z} is missing and its download from Google Drive failed; "
                                 "put the official zips there")
    sums = "".join(f"{sha256_file(zips / f'{c}.zip')}  ./{c}.zip\n" for c in sorted(DATASET_DRIVE_IDS))
    revision = "sha256:" + hashlib.sha256(sums.encode()).hexdigest()
    # Keyed by the revision, so changed archives are extracted afresh instead of reusing older audio.
    data = root / "data" / revision.removeprefix("sha256:")[:16]
    for cat in CATEGORIES:
        if not (data / cat).is_dir():
            with zipfile.ZipFile(zips / f"{cat}.zip") as zf:
                zf.extractall(data)
    shutil.rmtree(data / "__MACOSX", ignore_errors=True)
    return data, revision


def select(root: Path, per_category: int | None) -> list[tuple[str, str]]:
    """First ``per_category`` samples of each category in numeric id order (None: all), so a
    smaller selection is a prefix of a larger one, as in the reference runner."""
    out = []
    for cat in CATEGORIES:
        ids = sorted((p.name for p in (root / cat).iterdir() if p.is_dir() and p.name.isdigit()), key=int)
        out += [(cat, sid) for sid in (ids if per_category is None else ids[:per_category])]
    return out


def build_manifest(out: Path, per_category: int | None, zips: Path | None = None) -> Path:
    """Copy the selected pairs (16 kHz mono PCM16, as the server and the timing code expect)
    and write ``manifest.jsonl``: one item per pair, ``audio_path`` the overlap input,
    ``extra.clean_audio`` the clean input next to it."""
    from benchmark.eval.audio import read_wav, resample, write_wav
    from benchmark.eval.datasets import Item, save_manifest

    root, revision = download_dataset(zips)
    items = []
    for cat, sid in select(root, per_category):
        src = root / cat / sid
        dst = out / "audio" / cat / sid
        dst.mkdir(parents=True, exist_ok=True)
        marker = dst / ".revision"
        reuse = marker.is_file() and marker.read_text().strip() == revision
        marker.unlink(missing_ok=True)
        for name in ("input.wav", "clean_input.wav"):
            if reuse and (dst / name).exists():
                continue
            with wave.open(str(src / name), "rb") as w:
                native = (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (RATE, 1, 2)
            if native:  # byte for byte, so the timing code sees the official file
                shutil.copyfile(src / name, dst / name)
            else:
                x, rate = read_wav(src / name)
                write_wav(dst / name, resample(x, rate, RATE), RATE)
        marker.write_text(revision + "\n")
        meta = json.loads((src / "metadata.json").read_text(encoding="utf-8"))
        items.append(Item(id=f"{cat}-{sid}", audio_path=str(dst / "input.wav"), lang="en", extra={
            "category": cat, "sample_id": f"{cat}/{sid}", "clean_audio": "clean_input.wav",
            "context_text": meta["context_text"], "event_text": meta[EVENT_TEXT_KEYS[cat]],
            "event_s": meta["timestamps"], "dataset_revision": revision,
        }))
    name = "manifest.jsonl" if per_category is None else f"manifest-{per_category}.jsonl"
    print(f"{len(items)} pairs: " + ", ".join(f"{c} {sum(i.extra['category'] == c for i in items)}"
                                              for c in CATEGORIES), file=sys.stderr)
    return save_manifest(items, out / name)


def _version(dist: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return None
