"""Per-item analysis of a correctness run dir, past the runner's verdict.

    python -m benchmark.correctness.analysis <run_dir> [--reference <ref_dir>] [--no-asr]

The family comes from the run's report.json and the reference from the family, plus whichever
sibling builds exist (math SDPA, fp32 anchor, another seed, a bf16 copy). Per family:

- image: PSNR/SSIM (also at 1/8 scale), color, sharpness and seam statistics, a normalized
  distance against the other seed, contact sheets with a difference heatmap;
- video, audio-video, world: per-frame PSNR/SSIM curves, motion, flicker and color statistics,
  frame strips, a frames sheet, side-by-side mp4s; for audio-video also audio statistics,
  spectrograms, an AV onset check and transcripts;
- tts: audio statistics, duration and log-mel distances, F0, WER, spectrograms;
- actions and latents: errors against the reference and its bf16 copy, jitter, gripper events,
  plots.

Output goes to <run_dir>/analysis/: metrics.json, items.md, a one-page analysis.md and the PNG/mp4
artifacts. CPU only (--device moves Whisper). LPIPS, CLIP score and speaker similarity run when
their package and weights are there, and analysis.md says when they were skipped. Its bars are
proposed thresholds, not verdicts; the verdict stays the runner's.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

from benchmark.correctness import drawing as D
from benchmark.correctness import metrics as M
from benchmark.correctness.sets import DEFAULT_DATA_ROOT

# report.json "family" -> analysis kind
KINDS = {"image": "image", "video": "video", "audio-video": "av", "world-waypoint": "world", "tts": "tts",
         "robotics-pi05": "actions", "robotics-cosmos3-droid": "actions", "robotics-vjepa2-ac": "latents",
         "embedding-vjepa2": "embedding"}
ARMS = {"math": "-sdpa-math", "anchor": "-anchor-fp32"}
CAP_DB = 99.0  # identical frames' inf PSNR, so curves, medians and slopes stay finite
SPREAD_DB = 3.0  # the runner's calibrated floor sits this far under the reference's own spread
HEAT_VMAX = 48.0  # 8-bit levels at the darkest heatmap color
CLIP_ID = "openai/clip-vit-large-patch14"
# Flicker is the largest jump in the second difference of mean luma. Typical values are a fraction of a
# level, so the served jump may exceed the reference's by this factor plus this many levels.
FLICKER_RATIO, FLICKER_LEVELS = 1.25, 1.0
ACTION_PLOT_DIMS = 8  # DROID's 7 joints and the gripper lead every action vector
GRIPPER_DIM = 7


# ---------------------------------------------------------------------------
# run, reference and options
# ---------------------------------------------------------------------------


@dataclass
class RunInfo:
    dir: Path
    report: dict
    family: object
    kind: str
    model_key: str
    items: list
    set_root: Path
    runs: list[Path]
    ref: Path | None
    arms: dict[str, Path]


@dataclass
class Opts:
    workers: int = 1
    asr: bool = True
    optional: bool = True
    sbs: bool = True
    device: str = "cpu"


class Bar(NamedTuple):
    """A headline measure, the reference's own value where there is one, and a proposed threshold."""

    measure: str
    served: object
    spread: object
    rule: str
    ok: bool | None  # None: context only


@dataclass
class Analysis:
    rows: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    bars: list[Bar] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)  # path under the analysis dir -> what it shows
    columns: list[str] = field(default_factory=list)  # items.md columns


def _ok_dir(d: Path) -> bool:
    return (d / "outputs").is_dir() and (d / "meta.json").is_file()


def find_reference(fam, data_root: Path, explicit: Path | None, run: Path) -> Path | None:
    """``<data_root>/references/<id>/<set>`` for the family, or the dir given with --reference."""
    if explicit is not None:
        d = explicit.resolve()
        if not (d / "outputs").is_dir() and (d / fam.set_name / "outputs").is_dir():
            d = d / fam.set_name
        if not (d / "outputs").is_dir():
            raise SystemExit(f"--reference {explicit}: no outputs/ there or under {fam.set_name}/")
        return d
    rid = fam.reference_id()
    names = [rid]
    first = next(iter(sorted((run / "outputs").glob("*.npy"))), None)
    if first is not None:  # array references come per horizon and precision
        names += [f"{rid}-h{np.load(first, mmap_mode='r').shape[0]}-fp32", f"{rid}-fp32"]
    for name in names:
        d = data_root / "references" / name / fam.set_name
        if _ok_dir(d):
            return d
    return None


def find_arms(ref: Path | None) -> dict[str, Path]:
    """Sibling builds of the reference: math SDPA, fp32 anchor, another seed, a bf16 copy."""
    if ref is None:
        return {}
    root, base, set_name = ref.parent.parent, ref.parent.name, ref.name
    out = {arm: root / (base + suffix) / set_name for arm, suffix in ARMS.items()
           if _ok_dir(root / (base + suffix) / set_name)}
    seeds = sorted(p / set_name for p in root.glob(f"{glob.escape(base)}-seed*") if _ok_dir(p / set_name))
    if seeds:
        out["seed"] = seeds[0]
    if base.endswith("-fp32") and _ok_dir(root / (base[:-4] + "bf16") / set_name):
        out["bf16"] = root / (base[:-4] + "bf16") / set_name
    return out


def load_run(run_dir: Path, reference: Path | None, data_root: Path) -> RunInfo:
    import yaml

    from benchmark.correctness.families import family_for_config
    from benchmark.correctness.sets import load_set

    report = json.loads((run_dir / "report.json").read_text())
    kind = KINDS.get(report.get("family"))
    if kind is None:
        raise SystemExit(f"{run_dir}: no analysis for family {report.get('family')!r} yet "
                         f"(supported: {', '.join(sorted(KINDS))})")
    cfg_path = run_dir / "config.yaml"
    cfg = (yaml.safe_load(cfg_path.read_text()) or {}) if cfg_path.is_file() else {}
    model_key = report.get("model") or cfg.get("model")
    fam = family_for_config(report["config"], model_key, cfg, report.get("family_opts") or {})
    sset = load_set(fam.set_name, data_root, verify=False)
    items = fam.select_items(sset.items)
    if report.get("limit"):
        items = items[: report["limit"]]
    runs = [run_dir / name for name in (report.get("runs") or {}) if (run_dir / name / "outputs").is_dir()]
    runs = runs or sorted(p.parent for p in run_dir.glob("r*/outputs"))
    if not runs:
        raise SystemExit(f"{run_dir}: no r*/outputs")
    ref = find_reference(fam, data_root, reference, runs[0])
    return RunInfo(run_dir, report, fam, kind, model_key, items, sset.root, runs, ref, find_arms(ref))


def _served(info: RunInfo, k: int, item_id: str, ext: str) -> str | None:
    if k >= len(info.runs):
        return None
    p = info.runs[k] / "outputs" / f"{item_id}{ext}"
    return str(p) if p.is_file() else None


def _ref(d: Path | None, item_id: str, ext: str) -> str | None:
    if d is None:
        return None
    p = d / "outputs" / f"{item_id}{ext}"
    return str(p) if p.is_file() else None


def _map(fn, jobs: list, workers: int) -> list:
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
        return list(ex.map(fn, jobs))


# ---------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------


def _num(v, nd: int = 3):
    """Rounded float, None for None/NaN, inf kept."""
    if v is None:
        return None
    v = float(v)
    if math.isnan(v):
        return None
    return v if math.isinf(v) else round(v, nd)


def _med(values) -> float | None:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return statistics.median(vals) if vals else None


def _vals(rows: list[dict], key: str) -> list:
    return [r[key] for r in rows if r.get(key) is not None]


def _fmt(v, nd: int = 4) -> str:
    """``nd`` significant digits."""
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return "inf" if math.isinf(v) else f"{v:.{nd}g}"
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt(x, nd) for x in v)
    return str(v)


def _range(values, nd: int = 3) -> str | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return f"{_fmt(_med(vals), nd)} ({_fmt(min(vals), nd)}-{_fmt(max(vals), nd)})"


def luma(img: np.ndarray) -> np.ndarray:
    return np.asarray(img, dtype=np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def lap_var(img: np.ndarray) -> float:
    """Variance of the luma Laplacian, a sharpness measure."""
    y = luma(img)
    lap = -4 * y[1:-1, 1:-1] + y[:-2, 1:-1] + y[2:, 1:-1] + y[1:-1, :-2] + y[1:-1, 2:]
    return float(lap.var())


def hf_energy(img: np.ndarray) -> float:
    """Share of the luma spectrum's power above half the Nyquist frequency (fine detail and noise)."""
    y = luma(img).astype(np.float64)
    p = np.abs(np.fft.rfft2(y - y.mean())) ** 2
    fy = np.fft.fftfreq(y.shape[0])[:, None]
    fx = np.fft.rfftfreq(y.shape[1])[None, :]
    total = p.sum()
    return float(p[np.hypot(fy, fx) > 0.25].sum() / total) if total > 0 else 0.0


def seam_ratio(img: np.ndarray, period: int) -> float:
    """Gradient energy on rows and columns at multiples of ``period`` over the energy elsewhere. A
    tiled VAE or attention block shows up above 1."""
    y = luma(img)
    gx, gy = np.abs(np.diff(y, axis=1)), np.abs(np.diff(y, axis=0))
    on_c = np.arange(gx.shape[1]) % period == period - 1
    on_r = np.arange(gy.shape[0]) % period == period - 1
    if not on_c.any() or not on_r.any():
        return 1.0
    off = gx[:, ~on_c].mean() + gy[~on_r, :].mean()
    return float((gx[:, on_c].mean() + gy[on_r, :].mean()) / max(off, 1e-9))


def flat_blocks(img: np.ndarray, k: int = 16, thr: float = 0.5) -> float:
    y = luma(img)
    h, w = (y.shape[0] // k) * k, (y.shape[1] // k) * k
    if not h or not w:
        return 0.0
    return float((y[:h, :w].reshape(h // k, k, w // k, k).std(axis=(1, 3)) < thr).mean())


def downscale(img: np.ndarray, k: int) -> np.ndarray:
    a = np.asarray(img, dtype=np.float64)
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    return a[:h, :w].reshape(h // k, k, w // k, k, -1).mean(axis=(1, 3))


def image_stats(img: np.ndarray) -> dict:
    x = img.astype(np.float64)
    mx, mn = x.max(axis=-1), x.min(axis=-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0.0)
    return {"rgb_mean": [round(float(v), 2) for v in x.reshape(-1, 3).mean(axis=0)],
            "rgb_std": [round(float(v), 2) for v in x.reshape(-1, 3).std(axis=0)],
            "saturation": _num(sat.mean(), 4), "lapvar": _num(lap_var(img), 1), "hf_energy": _num(hf_energy(img), 5),
            "seam8": _num(seam_ratio(img, 8)), "seam16": _num(seam_ratio(img, 16)),
            "flat16": _num(flat_blocks(img), 4),
            "clipped": _num(((img == 0).all(-1) | (img == 255).all(-1)).mean(), 5)}


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr_from_mse(m: float) -> float:
    return CAP_DB if m == 0 else min(CAP_DB, 10 * math.log10(255.0 ** 2 / m))


def norm_distance(served: float | None, spread: float | None, other_seed: float | None) -> float | None:
    """(D(served, ref) - D(ref, math ref)) / (D(ref, other seed) - D(ref, math ref)). 0 is as close as
    a kernel change, 1 is as far as a different seed."""
    if served is None or spread is None or other_seed is None or other_seed - spread <= 0:
        return None
    return (served - spread) / (other_seed - spread)


def slope(ys) -> float | None:
    y = np.asarray([v for v in ys if v is not None and np.isfinite(v)], dtype=np.float64)
    return float(np.polyfit(np.arange(len(y)), y, 1)[0]) if len(y) > 2 else None


def _stats_ratio(a: dict | None, b: dict | None, key: str) -> float | None:
    if not a or not b or not b.get(key):
        return None
    return _num(a[key] / b[key])


def _rgb_shift(a: dict | None, b: dict | None) -> float | None:
    if not a or not b:
        return None
    return _num(max(abs(x - y) for x, y in zip(a["rgb_mean"], b["rgb_mean"], strict=True)), 2)


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------


def read_audio(path) -> tuple[np.ndarray, int]:
    """[samples, channels] float32 in [-1, 1] and the rate."""
    try:
        import soundfile as sf
    except ImportError:
        from benchmark.correctness.families.av import read_pcm_wav

        return read_pcm_wav(Path(path))
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x, int(sr)


def _lufs(x: np.ndarray, sr: int) -> float | None:
    try:
        import pyloudnorm
    except ImportError:
        return None
    if len(x) < int(0.4 * sr):
        return None
    with np.errstate(divide="ignore"):
        v = pyloudnorm.Meter(sr).integrated_loudness(np.asarray(x, dtype=np.float64))
    return _num(v, 2) if np.isfinite(v) else None


def _spectral(mono: np.ndarray, sr: int, n_fft: int = 1024, hop: int = 256) -> tuple[float | None, float | None]:
    """Median spectral flatness and centroid (Hz) over frames within 40 dB of the loudest."""
    x = np.nan_to_num(np.asarray(mono, dtype=np.float64))
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    idx = np.arange(n_fft)[None, :] + hop * np.arange(1 + (len(x) - n_fft) // hop)[:, None]
    p = np.abs(np.fft.rfft(x[idx] * np.hanning(n_fft), axis=1)) ** 2
    e = p.sum(axis=1)
    if e.max() <= 0:
        return None, None
    voiced = e > e.max() * 1e-4
    flat = np.exp(np.mean(np.log(p + 1e-20), axis=1)) / (np.mean(p, axis=1) + 1e-20)
    cent = p @ np.fft.rfftfreq(n_fft, 1 / sr) / (e + 1e-20)
    return float(np.median(flat[voiced])), float(np.median(cent[voiced]))


def _f0(mono: np.ndarray, sr: int) -> float | None:
    """Median F0 over voiced frames (librosa yin), None without librosa."""
    try:
        import librosa
    except ImportError:
        return None
    x = M.resample(mono, sr, 16000)
    if len(x) < 2048:
        return None
    f0 = librosa.yin(x, fmin=60, fmax=500, sr=16000, frame_length=1024, hop_length=160)
    rms = librosa.feature.rms(y=x, frame_length=1024, hop_length=160)[0]
    n = min(len(f0), len(rms))
    v = f0[:n][rms[:n] > max(rms.max() * 0.1, 1e-4)]
    return float(np.median(v)) if v.size else None


def audio_summary(x: np.ndarray, sr: int, f0: bool = False) -> dict:
    mono = x.mean(axis=1)
    st = M.audio_stats(mono, sr)
    flat, cent = _spectral(mono, sr)
    out = {"sr": sr, "channels": int(x.shape[1]), "duration_s": _num(st["duration_s"]),
           "rms_dbfs": _num(st["rms_dbfs"], 1), "peak": _num(st["peak"]), "silence_frac": _num(st["silence_frac"]),
           "clip_frac": _num(st["clip_frac"], 6), "lead_s": _num(st["lead_silence_s"], 2),
           "trail_s": _num(st["trail_silence_s"], 2), "dc": _num(float(np.abs(x.mean(axis=0)).max()), 5),
           "nonfinite": st["nonfinite"], "lufs": _lufs(x, sr), "flatness": _num(flat, 5), "centroid_hz": _num(cent, 0)}
    if f0:
        out["f0_hz"] = _num(_f0(mono, sr), 1)
    return out


def log_mel_db(mono: np.ndarray, sr: int) -> np.ndarray:
    return 10.0 * M.log_mel(mono, sr)


def snr_db(a: np.ndarray, b: np.ndarray) -> float | None:
    n = min(len(a), len(b))
    err = float(np.sum((a[:n].astype(np.float64) - b[:n]) ** 2))
    sig = float(np.sum(a[:n].astype(np.float64) ** 2))
    if err == 0:
        return math.inf
    return round(10 * math.log10(sig / err), 2) if sig > 0 else None


def dtw_log_mel(a: np.ndarray, sr_a: int, b: np.ndarray, sr_b: int) -> tuple[float | None, float | None]:
    """Log-mel distance and mel-cepstral distortion (dB) along a DTW path after trimming silence, for
    sampled speech that doesn't line up sample by sample. (None, None) without librosa."""
    try:
        import librosa
        from scipy.fft import dct
    except ImportError:
        return None, None
    sr = 16000
    a = librosa.effects.trim(M.resample(a, sr_a, sr), top_db=40)[0]
    b = librosa.effects.trim(M.resample(b, sr_b, sr), top_db=40)[0]

    def mel(x):
        s = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=512, hop_length=160, win_length=400, n_mels=80)
        return 10 * np.log10(s + 1e-12)

    if len(a) < 800 or len(b) < 800:
        return None, None
    A, B = mel(a), mel(b)
    floor = max(A.max(), B.max()) - 80
    A, B = np.maximum(A, floor), np.maximum(B, floor)
    _, wp = librosa.sequence.dtw(X=A, Y=B, metric="euclidean")
    lm = float(np.abs(A[:, wp[:, 0]] - B[:, wp[:, 1]]).mean())
    ca = dct(A / (20 / np.log(10)), axis=0, norm="ortho")[1:14]
    cb = dct(B / (20 / np.log(10)), axis=0, norm="ortho")[1:14]
    mcd = float(10 / np.log(10) * np.sqrt(2) * np.linalg.norm(ca[:, wp[:, 0]] - cb[:, wp[:, 1]], axis=0).mean())
    return lm, mcd


def _frame_env_db(mono: np.ndarray, sr: int, fps: float, n: int) -> list[float | None]:
    """Audio RMS (dBFS) per video frame period."""
    hop = sr / fps
    out = []
    for i in range(n):
        seg = mono[int(i * hop): int((i + 1) * hop)]
        out.append(_num(20 * math.log10(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) + 1e-12), 2)
                   if seg.size else None)
    return out


def _onset(env) -> int | None:
    """First index where an envelope reaches the middle of its range."""
    v = np.asarray([np.nan if e is None else e for e in env], dtype=np.float64)
    if not np.isfinite(v).any() or np.nanmax(v) - np.nanmin(v) < 1e-6:
        return None
    return int(np.nanargmax(v >= np.nanmin(v) + 0.5 * (np.nanmax(v) - np.nanmin(v))))


# ---------------------------------------------------------------------------
# optional models
# ---------------------------------------------------------------------------


def lpips_model():
    """(model, None) or (None, why it isn't available)."""
    try:
        import lpips
    except ImportError:
        return None, "the lpips package is not installed"
    try:
        return lpips.LPIPS(net="alex", verbose=False).eval(), None
    except Exception as e:  # its AlexNet weights come from the network on first use
        return None, f"LPIPS did not load: {str(e)[:160]}"


def lpips_distance(model, a: np.ndarray, b: np.ndarray) -> float:
    import torch

    def t(x):
        return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float() / 127.5 - 1

    with torch.inference_mode():
        return float(model(t(a), t(b)).item())


def clip_model():
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        return None, "transformers is not installed"
    try:
        return (CLIPModel.from_pretrained(CLIP_ID, local_files_only=True).eval(),
                CLIPProcessor.from_pretrained(CLIP_ID, local_files_only=True)), None
    except Exception:
        return None, f"{CLIP_ID} is not in the Hugging Face cache"


def clip_scores(clip, images: list[np.ndarray], texts: list[str]) -> list[float]:
    """CLIPScore per image and its text: 100 x cosine, floored at 0."""
    import torch
    from PIL import Image

    model, proc = clip
    out = []
    for s in range(0, len(images), 8):
        batch = proc(text=texts[s:s + 8], images=[Image.fromarray(x) for x in images[s:s + 8]], return_tensors="pt",
                     padding=True, truncation=True)
        with torch.inference_mode():
            o = model(**batch)
        i = o.image_embeds / o.image_embeds.norm(dim=-1, keepdim=True)
        t = o.text_embeds / o.text_embeds.norm(dim=-1, keepdim=True)
        out += (100 * (i * t).sum(-1)).clamp(min=0).tolist()
    return out


def speaker_model():
    """chatterbox-tts's VoiceEncoder (a Resemblyzer-style LSTM) with ResembleAI/chatterbox's weights. Its
    module is loaded by file so the rest of the package (and its imports) stays out."""
    import importlib.util
    import sys

    from benchmark.correctness.families.tts_models import use_refenv

    use_refenv()
    spec = importlib.util.find_spec("chatterbox")
    locs = list(spec.submodule_search_locations or []) if spec else []
    ve_dir = next((Path(p) / "models" / "voice_encoder" for p in locs
                   if (Path(p) / "models" / "voice_encoder" / "__init__.py").is_file()), None)
    if ve_dir is None:
        return None, "chatterbox-tts (its VoiceEncoder) is not installed"
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        weights = hf_hub_download("ResembleAI/chatterbox", "ve.safetensors", local_files_only=True)
    except Exception:
        return None, "ResembleAI/chatterbox ve.safetensors is not in the Hugging Face cache"
    s = importlib.util.spec_from_file_location("_analysis_voice_encoder", ve_dir / "__init__.py",
                                               submodule_search_locations=[str(ve_dir)])
    mod = importlib.util.module_from_spec(s)
    sys.modules[s.name] = mod
    s.loader.exec_module(mod)
    ve = mod.VoiceEncoder()
    ve.load_state_dict(load_file(weights))
    return ve.eval(), None


def speaker_embeddings(ve, paths: list[str | None]) -> list[np.ndarray | None]:
    wavs, keep = [], []
    for i, p in enumerate(paths):
        if p:
            x, sr = read_audio(p)
            x16 = M.resample(x.mean(axis=1), sr, 16000)
            if len(x16) > 8000:
                wavs.append(x16.astype(np.float32))
                keep.append(i)
    out: list[np.ndarray | None] = [None] * len(paths)
    if wavs:
        for i, v in zip(keep, ve.embeds_from_wavs(wavs, sample_rate=16000), strict=True):
            out[i] = np.asarray(v, dtype=np.float64) / max(np.linalg.norm(v), 1e-12)
    return out


def transcripts(paths: list[str | None], device: str) -> list[str]:
    from benchmark.correctness.families import asr_judge

    texts = asr_judge.transcribe([Path(p) if p else Path("/nonexistent") for p in paths], device)
    return [t if p else None for t, p in zip(texts, paths, strict=True)]


# ---------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------


class ImageJob(NamedTuple):
    id: str
    paths: dict  # r1, r2, ref, math, seed -> png path or None


def _compare_images(row: dict, tag: str, a: np.ndarray | None, b: np.ndarray | None) -> None:
    if a is None or b is None or a.shape != b.shape:
        return
    m = mse(a, b)
    row[f"psnr_{tag}"] = _num(math.inf if m == 0 else 10 * math.log10(255.0 ** 2 / m), 2)
    row[f"rmse_{tag}"] = _num(math.sqrt(m), 3)
    if tag != "r1_r2":
        row[f"ssim_{tag}"] = _num(M.ssim(a, b), 4)
        row[f"psnr8_{tag}"] = _num(M.psnr(downscale(a, 8), downscale(b, 8)), 2)


def _image_item(job: ImageJob) -> dict:
    imgs = {k: M.load_image(p) if p else None for k, p in job.paths.items()}
    row: dict = {"id": job.id}
    for tag, a, b in (("r1_ref", "r1", "ref"), ("r1_r2", "r1", "r2"), ("math_ref", "math", "ref"),
                      ("seed_ref", "seed", "ref")):
        _compare_images(row, tag, imgs.get(a), imgs.get(b))
    if "psnr_r1_r2" in row:
        row["r1_r2_identical"] = math.isinf(row["psnr_r1_r2"])
    for tag in ("r1", "ref", "math"):
        if imgs.get(tag) is not None:
            row[f"{tag}_stats"] = image_stats(imgs[tag])
    if imgs.get("r1") is not None:
        row["size"] = f"{imgs['r1'].shape[0]}x{imgs['r1'].shape[1]}"
    if "psnr_r1_ref" in row and "psnr_math_ref" in row:
        row["spread_gap_db"] = _num(min(row["psnr_r1_ref"], CAP_DB) - min(row["psnr_math_ref"], CAP_DB), 2)
    row["norm_dist_rmse"] = _num(norm_distance(row.get("rmse_r1_ref"), row.get("rmse_math_ref"),
                                               row.get("rmse_seed_ref")))
    rs, fs, ms = row.get("r1_stats"), row.get("ref_stats"), row.get("math_stats")
    row.update(rgb_shift=_rgb_shift(rs, fs), math_rgb_shift=_rgb_shift(ms, fs),
               sharpness_ratio=_stats_ratio(rs, fs, "lapvar"), math_sharpness_ratio=_stats_ratio(ms, fs, "lapvar"),
               hf_ratio=_stats_ratio(rs, fs, "hf_energy"))
    if rs and fs:
        row["seam_shift"] = _num(max(rs["seam8"] - fs["seam8"], rs["seam16"] - fs["seam16"]))
        row["clipped_shift"] = _num(rs["clipped"] - fs["clipped"], 5)
    return row


def analyze_images(info: RunInfo, out: Path, opts: Opts) -> Analysis:
    a = Analysis(columns=["id", "size", "psnr_r1_ref", "ssim_r1_ref", "psnr_math_ref", "spread_gap_db", "psnr8_r1_ref",
                          "psnr8_math_ref", "norm_dist_rmse", "lpips_r1_ref", "lpips_math_ref", "norm_dist_lpips",
                          "rgb_shift", "sharpness_ratio", "hf_ratio", "clip_r1", "clip_ref", "psnr_r1_r2", "prompt"])
    arms = info.arms
    jobs = [ImageJob(it.id, {"r1": _served(info, 0, it.id, ".png"), "r2": _served(info, 1, it.id, ".png"),
                             "ref": _ref(info.ref, it.id, ".png"), "math": _ref(arms.get("math"), it.id, ".png"),
                             "seed": _ref(arms.get("seed"), it.id, ".png")}) for it in info.items]
    rows = _map(_image_item, jobs, opts.workers)
    for it, row in zip(info.items, rows, strict=True):
        row["prompt"] = it.prompt
    _image_optional(a, jobs, rows, [it.prompt for it in info.items], opts)
    a.rows = rows
    _image_sheets(a, info, jobs, rows, out)
    _image_summary(a, rows, bool(arms.get("seed")))
    return a


def _image_optional(a: Analysis, jobs: list[ImageJob], rows: list[dict], prompts: list[str], opts: Opts) -> None:
    if not opts.optional:
        a.skipped["LPIPS, CLIP score"] = "--no-optional"
        return
    model, why = lpips_model()
    if model is None:
        a.skipped["LPIPS"] = why
    else:
        for job, row in zip(jobs, rows, strict=True):
            imgs = {k: M.load_image(p) if p else None for k, p in job.paths.items()}
            for tag, x in (("r1_ref", "r1"), ("math_ref", "math"), ("seed_ref", "seed")):
                if imgs.get(x) is not None and imgs.get("ref") is not None and imgs[x].shape == imgs["ref"].shape:
                    row[f"lpips_{tag}"] = _num(lpips_distance(model, imgs[x], imgs["ref"]), 4)
            row["norm_dist_lpips"] = _num(norm_distance(row.get("lpips_r1_ref"), row.get("lpips_math_ref"),
                                                        row.get("lpips_seed_ref")))
    clip, why = clip_model()
    if clip is None:
        a.skipped["CLIP score"] = why
        return
    for tag in ("r1", "ref", "seed"):
        idx = [i for i, j in enumerate(jobs) if j.paths.get(tag)]
        if idx:
            scores = clip_scores(clip, [M.load_image(jobs[i].paths[tag]) for i in idx], [prompts[i] for i in idx])
            for i, s in zip(idx, scores, strict=True):
                rows[i][f"clip_{tag}"] = _num(s, 2)


def _image_sheets(a: Analysis, info: RunInfo, jobs: list[ImageJob], rows: list[dict], out: Path) -> None:
    differ = any(r.get("r1_r2_identical") is False for r in rows)
    cols = [("r1", "mstar r1")] + ([("r2", "mstar r2")] if differ else []) + \
        [(k, lab) for k, lab in (("ref", "reference"), ("math", "reference, math SDPA"),
                                 ("seed", "reference, another seed")) if any(j.paths.get(k) for j in jobs)]
    with_ref = any(j.paths.get("ref") for j in jobs)
    labels = [lab for _, lab in cols] + ([f"|r1 - reference|, 0-{HEAT_VMAX:g} levels"] if with_ref else [])
    per = 5
    for s in range(0, len(jobs), per):
        grid_rows, row_labels, cell_labels = [], [], []
        for job, row in zip(jobs[s:s + per], rows[s:s + per], strict=True):
            imgs = {k: M.load_image(job.paths[k]) if job.paths.get(k) else None for k, _ in cols}
            cells, cl = [imgs[k] for k, _ in cols], []
            for k, _ in cols:
                cl.append({"r1": f"vs ref {_fmt(row.get('psnr_r1_ref'))} dB, SSIM {_fmt(row.get('ssim_r1_ref'), 3)}",
                           "r2": f"vs r1 {_fmt(row.get('psnr_r1_r2'))} dB",
                           "math": f"vs ref {_fmt(row.get('psnr_math_ref'))} dB",
                           "seed": f"vs ref {_fmt(row.get('psnr_seed_ref'))} dB"}.get(k, ""))
            if with_ref:
                r1, ref = imgs.get("r1"), imgs.get("ref")
                ok = r1 is not None and ref is not None and r1.shape == ref.shape
                cells.append(D.heatmap(D.abs_diff(r1, ref), HEAT_VMAX) if ok else None)
                cl.append("")
            grid_rows.append(cells)
            row_labels.append(f"{job.id}: {row['prompt'][:110]}")
            cell_labels.append(cl)
        first = next((c for c in grid_rows[0] if c is not None), None)
        box = (440, 250) if first is not None and first.shape[1] > first.shape[0] * 1.2 else (330, 330)
        name = f"sheets/sheet_{s // per:02d}.png"
        D.grid(grid_rows, out / name, col_labels=labels, row_labels=row_labels, cell_labels=cell_labels, box=box,
               title=f"{info.report.get('config')} {info.dir.name}: items {s}-{min(s + per, len(jobs)) - 1}")
    if jobs:
        a.artifacts["sheets/"] = f"{per} items per sheet, columns: {'; '.join(labels)}"


def _image_summary(a: Analysis, rows: list[dict], has_seed: bool) -> None:
    n = len(rows)
    psnr, spread = _vals(rows, "psnr_r1_ref"), _vals(rows, "psnr_math_ref")
    a.summary.update(n=n, psnr_r1_ref=M.distribution(psnr), psnr_math_ref=M.distribution(spread),
                     spread_gap_db=M.distribution(_vals(rows, "spread_gap_db")))
    if psnr:
        served = _med(psnr)
        ref_spread = _med(spread)
        a.bars.append(Bar("PSNR served vs reference, median (dB)", served, ref_spread,
                          f"at least the spread minus {SPREAD_DB:g} dB (the runner's calibrated floor)",
                          None if ref_spread is None else served >= ref_spread - SPREAD_DB))
        a.bars.append(Bar("PSNR at 1/8 scale, median (dB)", _med(_vals(rows, "psnr8_r1_ref")),
                          _med(_vals(rows, "psnr8_math_ref")), "context: layout and color only", None))
        under = [r["id"] for r in rows if (r.get("spread_gap_db") or 0) < -SPREAD_DB]
        a.bars.append(Bar(f"items more than {SPREAD_DB:g} dB under their own spread", f"{len(under)}/{n}", None,
                          "context", None))
    _distance_bars(a, rows, bool(psnr), has_seed)
    shifts = _vals(rows, "rgb_shift")
    if shifts:
        a.bars.append(Bar("largest channel-mean shift per image, median (levels)", _med(shifts),
                          _med(_vals(rows, "math_rgb_shift")), "at most 2", _med(shifts) <= 2))
    sharp = _vals(rows, "sharpness_ratio")
    if sharp:
        a.bars.append(Bar("sharpness ratio served/reference, median (range)", _range(sharp),
                          _range(_vals(rows, "math_sharpness_ratio")), "median 0.9-1.1", 0.9 <= _med(sharp) <= 1.1))
    _clip_bars(a, rows)
    same = [r["r1_r2_identical"] for r in rows if "r1_r2_identical" in r]
    if same:
        a.bars.append(Bar("r1 and r2 identical", f"{sum(same)}/{len(same)}", None, "context", None))
    for r in sorted(rows, key=lambda r: r.get("spread_gap_db") if r.get("spread_gap_db") is not None else 0):
        why = []
        if (r.get("spread_gap_db") or 0) < -SPREAD_DB:
            why.append(f"{-r['spread_gap_db']:.1f} dB under its spread ({r['psnr_r1_ref']} vs "
                       f"{r['psnr_math_ref']} dB)")
        if r.get("psnr_r1_ref") is not None and r["psnr_r1_ref"] < 20:
            why.append(f"{r['psnr_r1_ref']} dB from the reference")
        if (r.get("norm_dist_rmse") or 0) > 0.5:
            why.append(f"normalized distance {r['norm_dist_rmse']}")
        if r.get("sharpness_ratio") is not None and not 0.8 <= r["sharpness_ratio"] <= 1.25:
            why.append(f"sharpness ratio {r['sharpness_ratio']}")
        if (r.get("rgb_shift") or 0) > 6:
            why.append(f"channel mean moved {r['rgb_shift']} levels")
        if (r.get("seam_shift") or 0) > 0.05:
            why.append(f"seam ratio up {r['seam_shift']}")
        if (r.get("clipped_shift") or 0) > 0.02:
            why.append(f"{r['clipped_shift']:.1%} more pure black/white pixels")
        if why:
            a.flags.append(f"{r['id']}: " + "; ".join(why))


def _distance_bars(a: Analysis, rows: list[dict], has_ref: bool, has_seed: bool) -> None:
    lp, lp_spread = _vals(rows, "lpips_r1_ref"), _vals(rows, "lpips_math_ref")
    if lp:
        ok = None
        if lp_spread:
            ok = _med(lp) <= 1.5 * _med(lp_spread) and max(lp) <= max(lp_spread)
        a.bars.append(Bar("LPIPS (AlexNet) served vs reference, median / worst", f"{_fmt(_med(lp), 3)} / "
                          f"{_fmt(max(lp), 3)}", f"{_fmt(_med(lp_spread), 3)} / {_fmt(max(lp_spread), 3)}"
                          if lp_spread else None, "median at most 1.5x the spread's, worst at most the spread's", ok))
    for key, what in (("norm_dist_lpips", "LPIPS"), ("norm_dist_rmse", "RMSE")):
        d = _vals(rows, key)
        if d:
            a.bars.append(Bar(f"normalized distance on {what}, median / worst (0 = kernel change, 1 = another seed)",
                              f"{_fmt(_med(d))} / {_fmt(max(d))}", None, "median at most 0.25, every item at most 0.5",
                              _med(d) <= 0.25 and max(d) <= 0.5))
    if has_ref and not has_seed:
        a.notes.append("No second-seed reference, so no normalized distance.")


def _clip_bars(a: Analysis, rows: list[dict]) -> None:
    pairs = [(r["clip_r1"], r["clip_ref"]) for r in rows if r.get("clip_r1") is not None and r.get("clip_ref")
             is not None]
    if not pairs:
        return
    diff = _med(x - y for x, y in pairs)
    seed_min = min(_vals(rows, "clip_seed"), default=None)
    ok = abs(diff) <= 1.0 and (seed_min is None or min(x for x, _ in pairs) >= seed_min)
    a.bars.append(Bar("CLIP score served minus reference, median (100 x cosine)", _num(diff, 2),
                      f"another seed's minimum {_fmt(seed_min)}" if seed_min is not None else None,
                      "within 1, no item under the other seed's minimum", ok))


# ---------------------------------------------------------------------------
# video, audio-video and world rollouts
# ---------------------------------------------------------------------------


class VideoJob(NamedTuple):
    id: str
    kind: str  # video, av or world
    paths: dict  # served r1, r2 (mp4 or npy); reference arms ref, math, anchor, seed (npy); wav paths for av
    model_key: str  # reference frames go through this model's served mp4 encode
    fps: float
    out: str
    sbs: bool
    seed_image: str | None = None


def _curve(a: np.ndarray, b: np.ndarray) -> tuple[list[float], list[float]]:
    n = min(len(a), len(b))
    m = [mse(a[i], b[i]) for i in range(n)]
    return [round(psnr_from_mse(v), 2) for v in m], m


def _curve_summary(row: dict, tag: str, ps: list[float]) -> None:
    row[f"psnr_{tag}_median"] = _num(statistics.median(ps), 2)
    row[f"psnr_{tag}_min"] = _num(min(ps), 2)
    row[f"psnr_{tag}_first"] = _num(ps[0], 2)
    row[f"psnr_{tag}_last"] = _num(ps[-1], 2)
    row[f"slope_{tag}"] = _num(slope(ps), 4)


def temporal(frames: np.ndarray, ssim_pairs: int = 12) -> dict:
    """Mean luma per frame, consecutive-frame mean abs difference (motion), flicker as the second
    difference of mean luma, consecutive-frame SSIM, and sampled sharpness and color."""
    t = len(frames)
    means, motion, prev = [], [], None
    for f in frames:
        y = luma(f)
        means.append(float(y.mean()))
        if prev is not None:
            motion.append(float(np.abs(y - prev).mean()))
        prev = y
    d2 = np.abs(np.diff(np.asarray(means), 2)) if t > 2 else np.zeros(0)
    pairs = sorted({int(round(v)) for v in np.linspace(0, t - 2, min(t - 1, ssim_pairs))}) if t > 1 else []
    sample = sorted({int(round(v)) for v in np.linspace(0, t - 1, min(t, 8))})
    stats = [image_stats(frames[i]) for i in sample]
    mot = np.asarray(motion)
    return {"luma": [round(v, 2) for v in means], "motion": [round(v, 3) for v in motion],
            "motion_median": _num(np.median(mot)) if mot.size else None,
            "motion_max": _num(mot.max()) if mot.size else None,
            "frozen_pairs": int((mot < 0.3).sum()), "flicker_median": _num(np.median(d2), 4) if d2.size else None,
            "flicker_max": _num(d2.max(), 3) if d2.size else None, "luma_drift": _num(means[-1] - means[0], 2),
            "consecutive_ssim_median": _num(statistics.median(M.ssim(frames[i], frames[i + 1]) for i in pairs), 4)
            if pairs else None,
            "lapvar": _num(statistics.median(s["lapvar"] for s in stats), 1),
            "saturation": _num(statistics.median(s["saturation"] for s in stats), 4),
            "hf_energy": _num(statistics.median(s["hf_energy"] for s in stats), 5),
            "rgb_mean": [round(float(v), 2) for v in frames.reshape(-1, 3).mean(axis=0)],
            "flat_frames": int(sum(f.std() < 2 for f in frames))}


def _load_frames(path: str | None, reference: bool, model_key: str, fps: float) -> np.ndarray | None:
    if not path:
        return None
    from benchmark.correctness.families.video import as_served, read_video

    if path.endswith(".npy"):
        frames = np.load(path)
        return as_served(frames, model_key, fps) if reference else frames
    return read_video(Path(path))


def _video_item(job: VideoJob) -> dict:
    out = Path(job.out)
    src = {k: _load_frames(job.paths.get(k), k not in ("r1", "r2"), job.model_key, job.fps)
           for k in ("r1", "r2", "ref", "math", "anchor", "seed")}
    r1, r2, ref = src["r1"], src["r2"], src["ref"]
    row: dict = {"id": job.id}
    for tag, f in src.items():
        if f is not None:
            row[f"{tag}_frames"] = len(f)
            row[f"{tag}_dims"] = f"{f.shape[1]}x{f.shape[2]}"
    if r1 is None:
        return row
    if r2 is not None:
        row["r1_r2_identical"] = bool(np.array_equal(r1, r2))
    curves, mses = {}, {}
    pairs = [("r1_ref", r1, ref), ("math_ref", src["math"], ref), ("anchor_ref", src["anchor"], ref),
             ("seed_ref", src["seed"], ref)]
    if r2 is not None and not row["r1_r2_identical"]:
        pairs += [("r1_r2", r2, r1), ("r2_ref", r2, ref)]
    for tag, x, y in pairs:
        if x is not None and y is not None and x.shape[1:] == y.shape[1:]:
            curves[tag], mses[tag] = _curve(y, x)
            _curve_summary(row, tag, curves[tag])
    row["psnr_curves"] = curves
    if ref is not None and "r1_ref" in curves:
        row["frame_count_match"] = len(r1) == len(ref)
        n = min(len(r1), len(ref))
        idx = sorted({int(round(v)) for v in np.linspace(0, n - 1, min(n, 24))})
        row["ssim_frames"] = idx
        row["ssim_curves"] = {tag: [round(M.ssim(ref[i], x[i]), 4) for i in idx]
                              for tag, x in (("r1_ref", r1), ("math_ref", src["math"]))
                              if x is not None and x.shape[1:] == ref.shape[1:] and len(x) >= n}
        row["ssim_r1_ref_median"] = _num(statistics.median(row["ssim_curves"]["r1_ref"]), 4)
        if "math_ref" in row["ssim_curves"]:
            row["ssim_math_ref_median"] = _num(statistics.median(row["ssim_curves"]["math_ref"]), 4)
        _band(row, curves)
        rm = {tag: math.sqrt(statistics.fmean(v)) for tag, v in mses.items()}
        row["norm_dist_rmse"] = _num(norm_distance(rm.get("r1_ref"), rm.get("math_ref"), rm.get("seed_ref")))
    temps = {tag: temporal(f) for tag, f in (("r1", r1), ("ref", ref), ("math", src["math"])) if f is not None}
    if r2 is not None and not row["r1_r2_identical"]:
        temps["r2"] = temporal(r2)
    row["temporal"] = temps
    _video_ratios(row, temps)
    if job.seed_image:
        from PIL import Image

        seed = np.asarray(Image.open(job.seed_image).convert("RGB").resize((r1.shape[2], r1.shape[1]), Image.BICUBIC))
        row["psnr_seed_image"] = [round(psnr_from_mse(mse(seed, f)), 2) for f in r1]
    if job.kind == "av":
        _av_audio(job, row, temps, out)
    _video_artifacts(job, row, src, out)
    row["_thumbs"] = {tag: [(int(i), D.thumb(f[i], 200)) for i in sorted({int(round(v)) for v in
                                                                            np.linspace(0, len(f) - 1, 8)})]
                      for tag, f in (("ref", ref), ("r1", r1), ("r2", r2 if not row.get("r1_r2_identical") else None))
                      if f is not None}
    return row


def _band(row: dict, curves: dict) -> None:
    """The served curve against the reference's own move (math SDPA, and fp32 anchor for i2v; the
    larger move per frame), frame by frame."""
    arms = [curves[t] for t in ("math_ref", "anchor_ref") if t in curves]
    if not arms:
        return
    served = curves["r1_ref"]
    n = min(len(c) for c in [served, *arms])
    spread = [min(c[i] for c in arms) for i in range(n)]
    below = [i for i in range(n) if served[i] < spread[i] - SPREAD_DB]
    row["spread_median"] = _num(statistics.median(spread), 2)
    row["spread_gap_db"] = _num(statistics.median(served[:n]) - statistics.median(spread), 2)
    row["band_frames_below"] = len(below)
    row["band_first_below"] = below[0] if below else None
    s1, s2 = slope(served[:n]), slope(spread)
    row["slope_gap"] = _num(s1 - s2, 4) if s1 is not None and s2 is not None else None


def _video_ratios(row: dict, temps: dict) -> None:
    t1, tr, tm = temps.get("r1"), temps.get("ref"), temps.get("math")
    for tag, t in (("", t1), ("math_", tm)):
        if not t or not tr:
            continue
        row[f"{tag}motion_ratio"] = _stats_ratio(t, tr, "motion_median")
        row[f"{tag}flicker_max"] = t["flicker_max"]
        if t["flicker_max"] is not None and tr["flicker_max"] is not None:
            row[f"{tag}flicker_excess"] = _num(t["flicker_max"] - FLICKER_RATIO * tr["flicker_max"] - FLICKER_LEVELS, 2)
        row[f"{tag}sharpness_ratio"] = _stats_ratio(t, tr, "lapvar")
        row[f"{tag}rgb_shift"] = _rgb_shift(t, tr)
        n = min(len(t["luma"]), len(tr["luma"]))
        row[f"{tag}luma_offset"] = _num(statistics.fmean(t["luma"][i] - tr["luma"][i] for i in range(n)), 2)
    if t1:
        row["frozen_pairs"] = t1["frozen_pairs"]
        row["flat_frames"] = t1["flat_frames"]


def _av_audio(job: VideoJob, row: dict, temps: dict, out: Path) -> None:
    wav = {k: job.paths.get(f"{k}_wav") for k in ("r1", "r2", "ref", "math")}
    sig = {k: read_audio(p) for k, p in wav.items() if p}
    for k, (x, sr) in sig.items():
        row[f"{k}_audio"] = audio_summary(x, sr)
    r1, ref = sig.get("r1"), sig.get("ref")
    if r1 is not None and ref is not None and r1[1] == ref[1]:
        sr = ref[1]
        row["audio_snr_db"] = snr_db(ref[0].mean(axis=1), r1[0].mean(axis=1))
        row["audio_log_mel_db"] = _num(M.log_mel_distance(ref[0].mean(axis=1), r1[0].mean(axis=1), sr))
        if sig.get("math") is not None and sig["math"][1] == sr:
            row["math_audio_snr_db"] = snr_db(ref[0].mean(axis=1), sig["math"][0].mean(axis=1))
            row["math_audio_log_mel_db"] = _num(M.log_mel_distance(ref[0].mean(axis=1),
                                                                   sig["math"][0].mean(axis=1), sr))
    if r1 is not None and sig.get("r2") is not None:
        row["r1_r2_pcm_identical"] = Path(wav["r1"]).read_bytes() == Path(wav["r2"]).read_bytes()
        row["r1_r2_audio_snr_db"] = snr_db(r1[0].mean(axis=1), sig["r2"][0].mean(axis=1))
    for k in ("r1", "ref"):
        if sig.get(k) is None or k not in temps:
            continue
        x, sr = sig[k]
        env = _frame_env_db(x.mean(axis=1), sr, job.fps, len(temps[k]["luma"]))
        row[f"{k}_audio_env_db"] = env
        mot = temps[k]["motion"]
        m = min(len(mot), len(env) - 1)
        e = np.asarray([np.nan if v is None else v for v in env[1:m + 1]])
        if m > 4 and np.std(mot[:m]) > 0 and np.nanstd(e) > 0 and np.isfinite(e).all():
            row[f"{k}_av_corr"] = _num(np.corrcoef(mot[:m], e)[0, 1])
        on_v, on_a = _onset(mot), _onset(env)
        row[f"{k}_av_lag_frames"] = on_a - (on_v + 1) if on_v is not None and on_a is not None else None
    if row.get("r1_av_lag_frames") is not None and row.get("ref_av_lag_frames") is not None:
        row["av_lag_shift_frames"] = row["r1_av_lag_frames"] - row["ref_av_lag_frames"]
    for k in ("r1", "ref"):
        if row.get(f"{k}_audio"):
            row[f"lufs_{k}"] = row[f"{k}_audio"]["lufs"]
    D.spectrogram([(lab, log_mel_db(sig[k][0].mean(axis=1), sig[k][1]) if k in sig else None)
                   for k, lab in (("ref", "reference"), ("r1", "mstar r1"))],
                  out / "spectrograms" / f"{job.id}.png", f"{job.id}: log-mel, 80 dB range, same scale")


def _psnr_lines(curves: dict) -> list:
    """Curves against the reference. r1 vs r2 sits in another range and gets a chart of its own."""
    return [D.Line(lab, curves[tag], slot, dashed) for tag, lab, slot, dashed in
            (("r1_ref", "mstar vs ref", 0, False), ("math_ref", "math ref vs ref", 2, True),
             ("anchor_ref", "fp32-anchor ref vs ref", 1, True)) if tag in curves]


def _video_artifacts(job: VideoJob, row: dict, src: dict, out: Path) -> None:
    from benchmark.correctness.families.video import server_mp4

    r1, ref = src["r1"], src["ref"]
    r2 = src["r2"] if row.get("r1_r2_identical") is False else None
    n = len(r1)
    idx = sorted({int(round(v)) for v in np.linspace(0, n - 1, 5)})
    curves = row.get("psnr_curves", {})
    strip, labels, cells = [], [], []
    for tag, lab, f in (("ref", "reference", ref), ("r1", "mstar r1", r1), ("r2", "mstar r2", r2),
                        ("math", "reference, math SDPA", src["math"]), ("anchor", "reference, fp32 anchor",
                                                                         src["anchor"])):
        if f is None:
            continue
        strip.append([f[i] if i < len(f) else None for i in idx])
        labels.append(lab)
        c = curves.get(f"{tag}_ref") if tag != "ref" else None
        cells.append([f"frame {i}" + (f", {c[i]:.1f} dB" if c and i < len(c) else "") for i in idx])
    if ref is not None and ref.shape[1:] == r1.shape[1:]:
        strip.append([D.heatmap(D.abs_diff(r1[i], ref[i]), HEAT_VMAX) if i < len(ref) else None for i in idx])
        labels.append(f"|r1 - reference|, 0-{HEAT_VMAX:g} levels")
        cells.append([""] * len(idx))
    box = (260, 150) if r1.shape[2] > r1.shape[1] * 1.2 else (200, 200)
    D.grid(strip, out / "strips" / f"{job.id}.png", row_labels=labels, cell_labels=cells, box=box,
           title=f"{job.id}: frames at 0, 1/4, 1/2, 3/4 and the end")
    charts = []
    lines = _psnr_lines(curves)
    seed_db = row.get("psnr_seed_ref_median")
    if lines:
        charts.append(D.line_chart(lines, f"{job.id} PSNR per frame (dB)" + (", gray: another seed" if seed_db else ""),
                                   hline=seed_db))
    if "r1_r2" in curves:
        charts.append(D.line_chart([D.Line("r2 vs r1", curves["r1_r2"], 3)], f"{job.id} PSNR r2 vs r1 (dB)"))
    ss = row.get("ssim_curves", {})
    if ss:
        charts.append(D.line_chart([D.Line("mstar vs ref", ss["r1_ref"], 0, xs=row["ssim_frames"])]
                                   + ([D.Line("math ref vs ref", ss["math_ref"], 2, True, xs=row["ssim_frames"])]
                                      if "math_ref" in ss else []), f"{job.id} SSIM per frame"))
    temps = row["temporal"]
    for key, title in (("luma", "mean luma per frame"), ("motion", "frame-to-frame mean abs diff")):
        lines = [D.Line(lab, temps[tag][key], slot, tag == "math") for tag, lab, slot in
                 (("ref", "reference", 1), ("r1", "mstar r1", 0), ("math", "math ref", 2)) if tag in temps]
        charts.append(D.line_chart(lines, f"{job.id} {title}"))
    if job.kind == "av" and row.get("r1_audio_env_db"):
        charts.append(D.line_chart([D.Line(lab, row[f"{k}_audio_env_db"], slot) for k, lab, slot in
                                    (("ref", "reference", 1), ("r1", "mstar r1", 0)) if row.get(f"{k}_audio_env_db")],
                                   f"{job.id} audio RMS per video frame (dBFS)"))
    if row.get("psnr_seed_image"):
        charts.append(D.line_chart([D.Line("mstar r1", row["psnr_seed_image"], 0)],
                                   f"{job.id} PSNR of each frame vs the seed image"))
    D.panels(charts, len(charts), out / "curves" / f"{job.id}.png")
    if not job.sbs:
        return
    other = ref if ref is not None and ref.shape[1:] == r1.shape[1:] else r2
    if other is None:
        return
    m = min(len(r1), len(other))
    heat = np.stack([D.heatmap(D.abs_diff(r1[i, ::2, ::2], other[i, ::2, ::2]), HEAT_VMAX) for i in range(m)])
    sbs = np.concatenate([other[:m, ::2, ::2], r1[:m, ::2, ::2], heat], axis=2)
    sbs = sbs[:, : sbs.shape[1] // 2 * 2, : sbs.shape[2] // 2 * 2]
    (out / "sbs").mkdir(parents=True, exist_ok=True)
    (out / "sbs" / f"{job.id}.mp4").write_bytes(server_mp4(sbs, {"crf": "20", "preset": "veryfast"}, job.fps))


def analyze_videos(info: RunInfo, out: Path, opts: Opts) -> Analysis:
    a = Analysis(columns=["id", "r1_frames", "psnr_r1_ref_median", "psnr_r1_ref_min", "psnr_r1_ref_first",
                          "spread_median", "spread_gap_db", "band_frames_below", "slope_gap", "ssim_r1_ref_median",
                          "psnr_seed_ref_median", "norm_dist_rmse", "lpips_r1_ref", "motion_ratio", "flicker_max",
                          "luma_offset", "rgb_shift", "sharpness_ratio", "frozen_pairs", "flat_frames",
                          "r1_r2_identical", "psnr_r1_r2_min", "audio_snr_db", "audio_log_mel_db",
                          "math_audio_log_mel_db", "lufs_r1", "lufs_ref", "r1_av_lag_frames", "ref_av_lag_frames",
                          "transcript_r1", "transcript_ref", "prompt"])
    fam, kind = info.family, info.kind
    fps = float(fam.geometry()["fps"]) if kind == "av" else float(getattr(fam, "fps", 60))
    ext = ".npy" if kind == "world" else ".mp4"
    seed_key = getattr(fam, "size_key", None)
    jobs = []
    for it in info.items:
        paths = {"r1": _served(info, 0, it.id, ext), "r2": _served(info, 1, it.id, ext),
                 "ref": _ref(info.ref, it.id, ".npy")}
        paths.update({arm: _ref(info.arms.get(arm), it.id, ".npy") for arm in ("math", "anchor", "seed")})
        if kind == "av":
            paths.update(r1_wav=_served(info, 0, it.id, ".wav"), r2_wav=_served(info, 1, it.id, ".wav"),
                         ref_wav=_ref(info.ref, it.id, ".wav"), math_wav=_ref(info.arms.get("math"), it.id, ".wav"))
        seed_image = str(info.set_root / it.files[seed_key]) if kind == "world" and seed_key in it.files else None
        jobs.append(VideoJob(it.id, kind, paths, info.model_key, fps, str(out), opts.sbs, seed_image))
    rows = _map(_video_item, jobs, opts.workers)
    thumbs = []
    for it, row in zip(info.items, rows, strict=True):
        row["prompt"] = it.prompt
        thumbs.append(row.pop("_thumbs", {}))
    _video_optional(a, info, jobs, rows, opts)
    if kind == "av":
        _av_transcripts(a, jobs, rows, opts)
    a.rows = rows
    _video_sheet(a, info, rows, thumbs, out)
    _video_summary(a, rows, kind, info.ref is not None, bool(info.arms.get("seed")))
    names = {"strips/": "per item, 5 frames of the reference, mstar r1 (r2 when it differs), the reference arms "
                        "and the |r1 - reference| heatmap",
             "curves/": "per item, PSNR and SSIM per frame against the reference with its arms, mean luma, motion"
                        + (", audio RMS per frame" if kind == "av" else "")
                        + "; summary.png puts every item's PSNR chart side by side",
             "sbs/": "per item, side-by-side mp4 at half size: reference (or r2), mstar r1, |difference|",
             "spectrograms/": "per item, log-mel spectrograms of the reference audio and mstar r1"}
    for name, what in names.items():
        if (out / name).is_dir():
            a.artifacts[name] = what
    return a


def _video_optional(a: Analysis, info: RunInfo, jobs: list[VideoJob], rows: list[dict], opts: Opts) -> None:
    """LPIPS on 8 frames and CLIP on 4 frames per video, when available."""
    if not opts.optional:
        a.skipped["LPIPS, CLIP score"] = "--no-optional"
        return
    if info.ref is None:
        a.skipped["LPIPS"] = "no reference"
        model = None
    else:
        model, why = lpips_model()
        if model is None:
            a.skipped["LPIPS"] = why
    clip, why_clip = (None, "world rollouts have no prompt") if info.kind == "world" else clip_model()
    if clip is None:
        a.skipped["CLIP score"] = why_clip
    if model is None and clip is None:
        return
    for job, row in zip(jobs, rows, strict=True):
        f = {k: _load_frames(job.paths.get(k), k != "r1", job.model_key, job.fps) for k in ("r1", "ref", "math")}
        if f["r1"] is None:
            continue
        if model is not None and f["ref"] is not None and f["ref"].shape[1:] == f["r1"].shape[1:]:
            n = min(len(f["r1"]), len(f["ref"]))
            idx = sorted({int(round(v)) for v in np.linspace(0, n - 1, min(n, 8))})
            for tag in ("r1", "math"):
                if f[tag] is not None and len(f[tag]) >= n:
                    row[f"lpips_{tag}_ref"] = _num(statistics.fmean(lpips_distance(model, f[tag][i], f["ref"][i])
                                                                    for i in idx), 4)
        if clip is not None:
            for tag in ("r1", "ref"):
                if f[tag] is not None:
                    idx = sorted({int(round(v)) for v in np.linspace(0, len(f[tag]) - 1, 4)})
                    row[f"clip_{tag}"] = _num(statistics.fmean(clip_scores(clip, [f[tag][i] for i in idx],
                                                                           [row["prompt"]] * len(idx))), 2)


def _av_transcripts(a: Analysis, jobs: list[VideoJob], rows: list[dict], opts: Opts) -> None:
    if not opts.asr:
        a.skipped["transcripts"] = "--no-asr"
        return
    paths = [j.paths.get("r1_wav") for j in jobs] + [j.paths.get("ref_wav") for j in jobs]
    texts = transcripts(paths, opts.device)
    for k, row in enumerate(rows):
        row["transcript_r1"], row["transcript_ref"] = texts[k], texts[len(rows) + k]


def _video_sheet(a: Analysis, info: RunInfo, rows: list[dict], thumbs: list[dict], out: Path) -> None:
    grid_rows, labels, cells = [], [], []
    for row, th in zip(rows, thumbs, strict=True):
        for tag, lab in (("ref", "reference"), ("r1", "mstar r1"), ("r2", "mstar r2")):
            if tag not in th:
                continue
            grid_rows.append([f for _, f in th[tag]])
            extra = ""
            if tag == "r1" and row.get("psnr_r1_ref_median") is not None:
                extra = f", median frame PSNR vs ref {row['psnr_r1_ref_median']} dB"
            labels.append(f"{row['id']} {lab}: {row['prompt'][:70]}{extra}")
            cells.append([f"frame {i}" for i, _ in th[tag]])
    if grid_rows:
        first = grid_rows[0][0]
        box = (200, 116) if first.shape[1] > first.shape[0] * 1.2 else (160, 160)
        D.grid(grid_rows, out / "sheets" / "frames.png", row_labels=labels, cell_labels=cells, box=box,
               title=f"{info.report.get('config')} {info.dir.name}: 8 evenly spaced frames per video")
        a.artifacts["sheets/frames.png"] = "8 evenly spaced frames per video: reference, mstar r1, r2 when it differs"
    charts = []
    for row in rows:
        lines = _psnr_lines(row.get("psnr_curves", {}))
        if lines:
            charts.append(D.line_chart(lines, f"{row['id']} PSNR per frame (dB)",
                                       hline=row.get("psnr_seed_ref_median")))
    if charts:
        D.panels(charts, min(5, len(charts)), out / "curves" / "summary.png",
                 title=f"{info.report.get('config')} {info.dir.name}: PSNR per frame")


def _video_summary(a: Analysis, rows: list[dict], kind: str, has_ref: bool, has_seed: bool) -> None:
    n = len(rows)
    med = _vals(rows, "psnr_r1_ref_median")
    a.summary.update(n=n, psnr_r1_ref_median=M.distribution(med),
                     spread_median=M.distribution(_vals(rows, "spread_median")))
    if med:
        spread = _med(_vals(rows, "spread_median"))
        a.bars.append(Bar("per-video median frame PSNR vs reference, median over videos (dB)", _med(med), spread,
                          f"at least the spread minus {SPREAD_DB:g} dB (the runner's calibrated floor)",
                          None if spread is None else _med(med) >= spread - SPREAD_DB))
    inside = [r["band_frames_below"] == 0 for r in rows if "band_frames_below" in r]
    if inside:
        a.bars.append(Bar(f"videos inside the reference's band (spread minus {SPREAD_DB:g} dB) at every frame",
                          f"{sum(inside)}/{len(inside)}", None, "at least 80% of the videos",
                          sum(inside) >= 0.8 * len(inside)))
        gaps = _vals(rows, "slope_gap")
        if gaps:
            a.bars.append(Bar("PSNR slope minus the band's, worst video (dB/frame)", min(gaps), None,
                              "at least -0.1", min(gaps) >= -0.1))
    exc = _vals(rows, "flicker_excess")
    if exc:
        a.bars.append(Bar("largest mean-luma jump (second difference) over the reference's allowance, worst video "
                          "(levels)", max(exc), max(_vals(rows, "math_flicker_excess"), default=None),
                          f"at most 0: {FLICKER_RATIO:g}x the reference's largest jump plus {FLICKER_LEVELS:g} level",
                          max(exc) <= 0))
    for key, label, lo, hi in (("motion_ratio", "motion-energy ratio served/reference", 0.8, 1.25),
                               ("sharpness_ratio", "sharpness ratio served/reference", 0.9, 1.1)):
        v = _vals(rows, key)
        if v:
            a.bars.append(Bar(f"{label}, median (range)", _range(v), _range(_vals(rows, f"math_{key}")),
                              f"every video {lo:g}-{hi:g}" if key != "sharpness_ratio" else f"median {lo:g}-{hi:g}",
                              all(lo <= x <= hi for x in v) if key != "sharpness_ratio" else lo <= _med(v) <= hi))
    shifts = _vals(rows, "rgb_shift")
    if shifts:
        a.bars.append(Bar("largest channel-mean shift per video, median (levels)", _med(shifts),
                          _med(_vals(rows, "math_rgb_shift")), "at most 2", _med(shifts) <= 2))
        a.bars.append(Bar("mean luma offset served - reference, median (levels)", _med(_vals(rows, "luma_offset")),
                          _med(_vals(rows, "math_luma_offset")), "context", None))
    _distance_bars(a, rows, has_ref, has_seed)
    _clip_bars(a, rows)
    frozen, flat = sum(_vals(rows, "frozen_pairs")), sum(_vals(rows, "flat_frames"))
    if has_ref:
        ref_frozen = sum(r["temporal"]["ref"]["frozen_pairs"] for r in rows if "ref" in r.get("temporal", {}))
        a.bars.append(Bar("frozen frame pairs (mean abs diff < 0.3) / flat frames, all videos", f"{frozen} / {flat}",
                          ref_frozen, "no more frozen pairs than the reference, no flat frames",
                          frozen <= ref_frozen and flat == 0))
    else:
        a.bars.append(Bar("frozen frame pairs (mean abs diff < 0.3) / flat frames, all videos", f"{frozen} / {flat}",
                          None, "no flat frames", flat == 0))
    same = [r["r1_r2_identical"] for r in rows if "r1_r2_identical" in r]
    if same:
        lows = _vals(rows, "psnr_r1_r2_min")
        a.bars.append(Bar("r1 and r2 identical decoded frames", f"{sum(same)}/{len(same)}"
                          + (f", the others at least {min(lows):.1f} dB" if lows else ""), None, "context", None))
    if kind == "av":
        _av_bars(a, rows)
    if kind == "world":
        _world_bars(a, rows)
    for r in sorted(rows, key=lambda r: r.get("spread_gap_db") if r.get("spread_gap_db") is not None else 0):
        why = []
        if (r.get("spread_gap_db") or 0) < -SPREAD_DB:
            why.append(f"median frame PSNR {-r['spread_gap_db']:.1f} dB under the reference's spread")
        if r.get("band_frames_below"):
            why.append(f"{r['band_frames_below']} frames under the band, first at frame {r['band_first_below']}")
        if r.get("frame_count_match") is False:
            why.append(f"{r.get('r1_frames')} frames against the reference's {r.get('ref_frames')}")
        if r.get("motion_ratio") is not None and not 0.8 <= r["motion_ratio"] <= 1.25:
            why.append(f"motion ratio {r['motion_ratio']}")
        if (r.get("flicker_excess") or 0) > 0:
            why.append(f"largest mean-luma jump {r['flicker_max']} levels, {r['flicker_excess']} over the allowance")
        if (r.get("norm_dist_rmse") or 0) > 0.5:
            why.append(f"normalized distance {r['norm_dist_rmse']}")
        if r.get("flat_frames"):
            why.append(f"{r['flat_frames']} flat frames")
        if (r.get("rgb_shift") or 0) > 6:
            why.append(f"channel mean moved {r['rgb_shift']} levels")
        lufs = (r.get("lufs_r1"), r.get("lufs_ref"))
        if None not in lufs and abs(lufs[0] - lufs[1]) > 2:
            why.append(f"loudness {lufs[0]} vs {lufs[1]} LUFS")
        if r.get("av_lag_shift_frames") is not None and abs(r["av_lag_shift_frames"]) > 1:
            why.append(f"audio onset moved {r['av_lag_shift_frames']} frames against the motion onset")
        if why:
            a.flags.append(f"{r['id']}: " + "; ".join(why))
    a.notes.append(f"PSNR curves cap identical frames at {CAP_DB:g} dB."
                   + (" Reference frames go through the served mp4's encode (SERVER_X264) before any comparison, as "
                      "in the runner." if has_ref else ""))


def _world_bars(a: Analysis, rows: list[dict]) -> None:
    """Rollouts have no reference: how they start from the seed image, move and drift."""
    seed = [(r["psnr_seed_image"][0], r["psnr_seed_image"][-1]) for r in rows if r.get("psnr_seed_image")]
    if seed:
        a.bars.append(Bar("PSNR of the first / last frame vs the seed image, median over rollouts (dB)",
                          f"{_fmt(_med(f for f, _ in seed))} / {_fmt(_med(x for _, x in seed))}", None, "context",
                          None))
    t = [r["temporal"]["r1"] for r in rows if "r1" in r.get("temporal", {})]
    if t:
        a.bars.append(Bar("motion (frame-to-frame mean abs diff), median per rollout", _range(
            [x["motion_median"] for x in t]), None, "context", None))
        a.bars.append(Bar("mean luma drift, first to last frame (levels)", _range([x["luma_drift"] for x in t]), None,
                          "context", None))


def _av_bars(a: Analysis, rows: list[dict]) -> None:
    pairs = [(r["lufs_r1"], r["lufs_ref"]) for r in rows if r.get("lufs_r1") is not None and r.get("lufs_ref")
             is not None]
    if pairs:
        diff = _med(x for x, _ in pairs) - _med(y for _, y in pairs)
        a.bars.append(Bar("integrated loudness, median served minus median reference (LU)", _num(diff, 2), None,
                          "within 2", abs(diff) <= 2))
    else:
        a.skipped.setdefault("loudness (LUFS)", "pyloudnorm is not installed or the clips are under 0.4 s")
    gaps = [r["audio_log_mel_db"] - r["math_audio_log_mel_db"] for r in rows
            if r.get("audio_log_mel_db") is not None and r.get("math_audio_log_mel_db") is not None]
    if gaps:
        a.bars.append(Bar("audio log-mel distance to reference minus the reference's own, median (dB)",
                          _num(_med(gaps)), None, "at most 1 (the runner's av check)", _med(gaps) <= 1))
    clip = [r["r1_audio"]["clip_frac"] for r in rows if r.get("r1_audio")]
    if clip:
        a.bars.append(Bar("clipped samples, worst clip", f"{max(clip):.3%}", None, "under 0.1%", max(clip) < 1e-3))
    # Where most of a clip sits under the -50 dBFS silence threshold, its silent edges say nothing.
    voiced = [r for r in rows if r.get("r1_audio") and r.get("ref_audio") and
              max(r["r1_audio"]["silence_frac"], r["ref_audio"]["silence_frac"]) < 0.5]
    if len(voiced) < len(rows):
        a.notes.append(f"Silent-edge checks use the {len(voiced)} clips under 50% silence on both sides.")
    for edge in ("lead_s", "trail_s"):
        d = [abs(r["r1_audio"][edge] - r["ref_audio"][edge]) for r in voiced]
        if d:
            a.bars.append(Bar(f"{'leading' if edge == 'lead_s' else 'trailing'} silence, largest difference "
                              "from the reference (s)", _num(max(d), 2), None, "within 0.2", max(d) <= 0.2))
    lags = _vals(rows, "av_lag_shift_frames")
    if lags:
        a.bars.append(Bar("audio onset relative to motion onset, served minus reference, worst (frames)",
                          max(lags, key=abs), None, "within 1 frame (a rough sync check)",
                          all(abs(v) <= 1 for v in lags)))


# ---------------------------------------------------------------------------
# speech
# ---------------------------------------------------------------------------


class SpeechJob(NamedTuple):
    id: str
    paths: dict  # r1, r2, ref -> wav path or None
    out: str


def _speech_item(job: SpeechJob) -> dict:
    row: dict = {"id": job.id}
    sig = {}
    for tag in ("r1", "r2", "ref"):
        if job.paths.get(tag):
            x, sr = read_audio(job.paths[tag])
            sig[tag] = (x.mean(axis=1), sr)
            row[f"{tag}_audio"] = audio_summary(x, sr, f0=True)
            for key in ("duration_s", "lufs", "lead_s", "trail_s", "f0_hz"):
                row[f"{key}_{tag}"] = row[f"{tag}_audio"].get(key)
    if "r1" in sig and "ref" in sig:
        (a, sa), (b, sb) = sig["r1"], sig["ref"]
        row["dur_ratio"] = _num(len(a) / sa / max(len(b) / sb, 1e-6))
        row["log_mel_r1_ref"] = _num(M.log_mel_distance(M.resample(a, sa, sb), b, sb), 2)
        lm, mcd = dtw_log_mel(a, sa, b, sb)
        row["dtw_log_mel_r1_ref"], row["mcd_r1_ref"] = _num(lm, 2), _num(mcd, 2)
    if "r1" in sig and "r2" in sig:
        row["r1_r2_identical"] = Path(job.paths["r1"]).read_bytes() == Path(job.paths["r2"]).read_bytes()
        lm, _ = dtw_log_mel(*sig["r1"], *sig["r2"])
        row["dtw_log_mel_r1_r2"] = _num(lm, 2)
    D.spectrogram([(lab, log_mel_db(*sig[k]) if k in sig else None) for k, lab in (("ref", "reference"),
                                                                                    ("r1", "mstar r1"))],
                  Path(job.out) / "spectrograms" / f"{job.id}.png", f"{job.id}: log-mel, 80 dB range, same scale")
    return row


def _runner_transcripts(path: Path, n: int) -> list[str] | None:
    """The runner's cached Whisper transcripts (scoring/judge_r<k>.json, a reference's judge_<model>.json)."""
    from benchmark.correctness.families import asr_judge

    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if data.get("judge") not in (None, asr_judge.JUDGE_MODEL) or len(data.get("texts", [])) != n:
        return None
    return data["texts"]


def analyze_speech(info: RunInfo, out: Path, opts: Opts) -> Analysis:
    from benchmark.correctness.families import asr_judge

    a = Analysis(columns=["id", "wer_r1", "wer_ref", "duration_s_r1", "duration_s_ref", "dur_ratio", "lead_s_r1",
                          "lead_s_ref", "trail_s_r1", "trail_s_ref", "lufs_r1", "lufs_ref", "f0_hz_r1", "f0_hz_ref",
                          "log_mel_r1_ref", "dtw_log_mel_r1_ref", "dtw_log_mel_r1_r2", "mcd_r1_ref", "spk_r1_ref",
                          "spk_r1_ref_centroid", "r1_r2_identical", "heard_r1", "heard_ref", "text"])
    jobs = [SpeechJob(it.id, {"r1": _served(info, 0, it.id, ".wav"), "r2": _served(info, 1, it.id, ".wav"),
                              "ref": _ref(info.ref, it.id, ".wav")}, str(out)) for it in info.items]
    rows = _map(_speech_item, jobs, opts.workers)
    texts = [it.prompt for it in info.items]
    for row, t in zip(rows, texts, strict=True):
        row["text"] = t
    norm = M.make_normalizer("whisper")
    corpus = {}
    sources = [("r1", info.dir / "scoring" / "judge_r1.json"), ("r2", info.dir / "scoring" / "judge_r2.json")]
    if info.ref is not None:
        sources.append(("ref", info.ref / f"judge_{asr_judge.JUDGE_MODEL.split('/')[-1]}.json"))
    for tag, cache in sources:
        if not any(j.paths.get(tag) for j in jobs):
            continue
        hyps = _runner_transcripts(cache, len(jobs))
        if hyps is None and opts.asr:
            hyps = transcripts([j.paths.get(tag) for j in jobs], opts.device)
        if hyps is None:
            a.skipped[f"transcripts of {tag}"] = "--no-asr and no runner transcripts for it"
            continue
        w = M.wer(texts, [h or "" for h in hyps], norm)
        corpus[tag] = w["wer"]
        for row, h, wr in zip(rows, hyps, w["rows"], strict=True):
            row[f"heard_{tag}"], row[f"wer_{tag}"] = h, _num(wr["wer"])
    a.summary["wer"] = {k: _num(100 * v) for k, v in corpus.items()}
    _speaker(a, jobs, rows, opts)
    a.rows = rows
    _speech_summary(a, rows, corpus)
    if (out / "spectrograms").is_dir():
        a.artifacts["spectrograms/"] = "per clip, log-mel spectrograms of the reference and mstar r1"
    return a


def _speaker(a: Analysis, jobs: list[SpeechJob], rows: list[dict], opts: Opts) -> None:
    if not opts.optional:
        a.skipped["speaker similarity"] = "--no-optional"
        return
    ve, why = speaker_model()
    if ve is None:
        a.skipped["speaker similarity"] = why
        return
    e = {tag: speaker_embeddings(ve, [j.paths.get(tag) for j in jobs]) for tag in ("r1", "ref")}
    ref = [v for v in e["ref"] if v is not None]
    served = [v for v in e["r1"] if v is not None]
    if not ref or not served:
        a.skipped["speaker similarity"] = "no reference or served clips long enough"
        return
    cent = np.mean(ref, axis=0)
    cent /= np.linalg.norm(cent)
    for k, row in enumerate(rows):
        x, y = e["r1"][k], e["ref"][k]
        if x is not None and y is not None:
            row["spk_r1_ref"] = _num(x @ y)
        if x is not None:
            row["spk_r1_ref_centroid"] = _num(x @ cent)
        if y is not None and len(ref) > 1:
            others = np.mean([v for j, v in enumerate(e["ref"]) if v is not None and j != k], axis=0)
            row["spk_ref_loo"] = _num(y @ (others / np.linalg.norm(others)))
    sc = np.mean(served, axis=0)
    a.summary["speaker_centroid_cosine"] = _num(sc @ cent / np.linalg.norm(sc), 4)


def _speech_summary(a: Analysis, rows: list[dict], corpus: dict) -> None:
    a.summary["n"] = len(rows)
    if "r1" in corpus:
        ref = corpus.get("ref")
        a.bars.append(Bar("WER r1 / r2 (%)", " / ".join(f"{100 * corpus[k]:.2f}" for k in ("r1", "r2") if k in corpus),
                          _num(100 * ref, 2) if ref is not None else None,
                          "r1 within 1 point of the reference (the runner's bar)",
                          None if ref is None else 100 * (corpus["r1"] - ref) <= 1.0))
    ratios = _vals(rows, "dur_ratio")
    if ratios:
        a.bars.append(Bar("duration ratio served/reference, median (range)", _range(ratios), None, "median 0.85-1.15",
                          0.85 <= _med(ratios) <= 1.15))
    for key, label, tol in (("lufs", "integrated loudness, median (LUFS)", 2.0),
                            ("lead_s", "leading silence, median (s)", 0.2), ("trail_s", "trailing silence, median (s)",
                                                                             0.2)):
        s, r = _med(_vals(rows, f"{key}_r1")), _med(_vals(rows, f"{key}_ref"))
        if s is not None:
            a.bars.append(Bar(label, _num(s, 2), _num(r, 2), f"within {tol:g} of the reference",
                              None if r is None else abs(s - r) <= tol))
        elif key == "lufs":
            a.skipped["loudness (LUFS)"] = "pyloudnorm is not installed or the clips are under 0.4 s"
    clip = [r["r1_audio"]["clip_frac"] for r in rows if r.get("r1_audio")]
    if clip:
        a.bars.append(Bar("clipped samples, worst clip", f"{max(clip):.3%}", None, "under 0.1%", max(clip) < 1e-3))
    f0s, f0r = _med(_vals(rows, "f0_hz_r1")), _med(_vals(rows, "f0_hz_ref"))
    if f0s is not None:
        a.bars.append(Bar("F0, median over clips (Hz)", _num(f0s, 1), _num(f0r, 1), "within 5% of the reference",
                          None if f0r is None else abs(f0s - f0r) <= 0.05 * f0r))
    else:
        a.skipped.setdefault("F0", "librosa is not installed")
    lm = _vals(rows, "dtw_log_mel_r1_ref")
    if lm:
        a.bars.append(Bar("DTW log-mel distance to the reference clip, median (dB)", _num(_med(lm), 2),
                          f"r1 vs r2 {_fmt(_med(_vals(rows, 'dtw_log_mel_r1_r2')))}", "context", None))
    elif rows and "ref_audio" in rows[0]:
        a.skipped.setdefault("DTW log-mel and MCD", "librosa is not installed")
    spk, loo = _vals(rows, "spk_r1_ref_centroid"), _vals(rows, "spk_ref_loo")
    if spk:
        a.bars.append(Bar("speaker similarity to the reference voice's centroid, median", _num(_med(spk)),
                          _num(_med(loo)) if loo else None, "at least 0.9x the reference's own (leave-one-out)",
                          None if not loo else _med(spk) >= 0.9 * _med(loo)))
    same = [r["r1_r2_identical"] for r in rows if "r1_r2_identical" in r]
    if same:
        a.bars.append(Bar("r1 and r2 identical clips", f"{sum(same)}/{len(same)}", None, "context", None))
    for r in sorted(rows, key=lambda r: -(r.get("wer_r1") or 0)):
        why = []
        if (r.get("wer_r1") or 0) > max(0.1, r.get("wer_ref") or 0):
            why.append(f"WER {r['wer_r1']:.2f} (reference {_fmt(r.get('wer_ref'))}), heard {r.get('heard_r1')!r}")
        if r.get("dur_ratio") is not None and not 0.5 <= r["dur_ratio"] <= 2:
            why.append(f"duration ratio {r['dur_ratio']}")
        if r.get("r1_audio") and r["r1_audio"]["clip_frac"] >= 1e-3:
            why.append(f"{r['r1_audio']['clip_frac']:.2%} clipped")
        for edge in ("lead_s", "trail_s"):
            s, ref = r.get(f"{edge}_r1"), r.get(f"{edge}_ref")
            if s is not None and ref is not None and abs(s - ref) > 0.5:
                why.append(f"{'leading' if edge == 'lead_s' else 'trailing'} silence {s} s vs {ref} s")
        if why:
            a.flags.append(f"{r['id']}: " + "; ".join(why))


# ---------------------------------------------------------------------------
# actions, latents and embeddings
# ---------------------------------------------------------------------------


def _load_npy(p: str | None) -> np.ndarray | None:
    return np.load(p).astype(np.float64) if p else None


def _gripper_events(x: np.ndarray) -> list[int]:
    """Steps where the gripper crosses the middle of its range (0.5 for [0, 1] outputs, else 0)."""
    th = 0.5 if x.min() >= -0.05 and x.max() <= 1.05 else 0.0
    return [i for i in range(1, len(x)) if (x[i - 1] - th) * (x[i] - th) < 0]


def action_stats(a: np.ndarray, dims: int) -> dict:
    real = a[:, :dims]
    std = real.std(axis=0)
    step = np.abs(np.diff(real, axis=0)).mean(axis=0) if len(real) > 1 else np.zeros(dims)
    jerk = np.abs(np.diff(real, 3, axis=0)).mean() if len(real) > 3 else None
    out = {"shape": list(a.shape), "finite": bool(np.isfinite(a).all()), "min": _num(real.min()),
           "max": _num(real.max()), "const_dims": int((std < 1e-4).sum()),
           "saturated": _num((np.abs(real) >= 0.999).mean(), 4),
           "jitter": _num((step / np.maximum(std, 1e-6)).mean()), "jerk": _num(jerk, 4)}
    if a.shape[1] > GRIPPER_DIM:
        out["gripper_events"] = _gripper_events(a[:, GRIPPER_DIM])
    return out


def compare_actions(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    big = (np.abs(a) > 0.05) | (np.abs(b) > 0.05)
    return {"maxabs": _num(np.abs(d).max(), 5), "rmse": _num(np.sqrt((d ** 2).mean()), 5),
            "rmse_per_dim": [round(float(v), 5) for v in np.sqrt((d ** 2).mean(axis=0))],
            "corr": _num(np.corrcoef(a.ravel(), b.ravel())[0, 1], 5) if a.std() > 0 and b.std() > 0 else None,
            "sign_agree": _num((np.sign(a[big]) == np.sign(b[big])).mean(), 4) if big.any() else None}


def analyze_actions(info: RunInfo, out: Path, opts: Opts) -> Analysis:
    a = Analysis(columns=["id", "shape", "maxabs_ref", "rmse_ref", "corr_ref", "maxabs_bf16", "ref_spread_maxabs",
                          "gripper_r1", "gripper_ref", "jitter_r1", "jitter_ref", "jerk_r1", "jerk_ref", "r1_r2_maxabs",
                          "task"])
    bf16 = info.arms.get("bf16")
    loaded = []
    for it in info.items:
        x = {"r1": _load_npy(_served(info, 0, it.id, ".npy")), "r2": _load_npy(_served(info, 1, it.id, ".npy")),
             "ref": _load_npy(_ref(info.ref, it.id, ".npy")), "bf16": _load_npy(_ref(bf16, it.id, ".npy"))}
        loaded.append(x)
        row: dict = {"id": it.id, "task": it.prompt}
        if x["r1"] is None:
            a.rows.append(row)
            continue
        dims = min(ACTION_PLOT_DIMS, x["r1"].shape[1])
        s1 = action_stats(x["r1"], dims)
        row.update(shape="x".join(map(str, x["r1"].shape)), r1_stats=s1, jitter_r1=s1["jitter"], jerk_r1=s1["jerk"],
                   gripper_r1=s1.get("gripper_events"), finite=s1["finite"])
        if x["r2"] is not None and x["r2"].shape == x["r1"].shape:
            row["r1_r2_maxabs"] = _num(np.abs(x["r1"] - x["r2"]).max(), 6)
        if x["ref"] is not None and x["ref"].shape == x["r1"].shape:
            sr = action_stats(x["ref"], dims)
            c = compare_actions(x["r1"], x["ref"])
            row.update(ref_stats=sr, jitter_ref=sr["jitter"], jerk_ref=sr["jerk"], gripper_ref=sr.get("gripper_events"),
                       vs_ref=c, maxabs_ref=c["maxabs"], rmse_ref=c["rmse"], corr_ref=c["corr"])
            if x["bf16"] is not None and x["bf16"].shape == x["ref"].shape:
                row["maxabs_bf16"] = _num(np.abs(x["r1"] - x["bf16"]).max(), 5)
                row["ref_spread_maxabs"] = _num(np.abs(x["ref"] - x["bf16"]).max(), 5)
        a.rows.append(row)
    _action_plots(a, info, loaded, out)
    _action_summary(a)
    return a


def _action_plots(a: Analysis, info: RunInfo, loaded: list[dict], out: Path) -> None:
    served = [x["r1"] for x in loaded if x["r1"] is not None]
    if not served:
        return
    dims = min(ACTION_PLOT_DIMS, served[0].shape[1])
    vals = np.concatenate([x[k][:, :dims].ravel() for x in loaded for k in ("r1", "ref") if x[k] is not None])
    lo, hi = float(vals.min()), float(vals.max())
    pad = 0.05 * (hi - lo or 1)
    for it, x in zip(info.items, loaded, strict=True):
        if x["r1"] is None:
            continue
        charts = []
        for k in range(min(dims, x["r1"].shape[1])):
            lines = [D.Line("mstar r1", list(x["r1"][:, k]), 0)]
            if x["ref"] is not None and x["ref"].shape == x["r1"].shape:
                lines.append(D.Line("reference", list(x["ref"][:, k]), 1, True))
            if x["r2"] is not None and x["r2"].shape == x["r1"].shape and not np.array_equal(x["r1"], x["r2"]):
                lines.append(D.Line("r2", list(x["r2"][:, k]), 3))
            charts.append(D.line_chart(lines, f"{it.id} dim {k}" + (" (gripper)" if k == GRIPPER_DIM else ""),
                                       size=(260, 170), ylim=(lo - pad, hi + pad), xlabel="step"))
        D.panels(charts, 4, out / "plots" / f"{it.id}.png", title=f"{it.id}: {it.prompt[:90]}")
    a.artifacts["plots/"] = (f"per episode, the first {dims} action dims over the horizon: mstar r1, the reference "
                             "dashed, r2 when it differs; one y range for every episode")


def _action_summary(a: Analysis) -> None:
    rows = a.rows
    a.summary["n"] = len(rows)
    bad = [r["id"] for r in rows if r.get("finite") is False]
    a.bars.append(Bar("outputs with non-finite values", len(bad), None, "0", not bad))
    m = _vals(rows, "maxabs_ref")
    if m:
        spread = _vals(rows, "ref_spread_maxabs")
        a.bars.append(Bar("max-abs vs reference, worst episode", max(m), max(spread) if spread else None,
                          "at most 0.2 (compare_with_lerobot.py), next to the reference's fp32-vs-bf16 spread",
                          max(m) <= 0.2))
        a.bars.append(Bar("RMSE vs reference, median", _num(_med(_vals(rows, "rmse_ref")), 5), None, "context", None))
        grip = [r["gripper_r1"] == r["gripper_ref"] for r in rows if r.get("gripper_ref") is not None]
        if grip:
            a.bars.append(Bar("episodes with the reference's gripper events", f"{sum(grip)}/{len(grip)}", None, "all",
                              all(grip)))
    j1 = _vals(rows, "jitter_r1")
    if j1:
        a.bars.append(Bar("jitter (mean step change over the dim's spread), median", _num(_med(j1)),
                          _num(_med(_vals(rows, "jitter_ref"))) if _vals(rows, "jitter_ref") else None,
                          "context (recorded DROID actions give 0.1-0.4)", None))
    rep = _vals(rows, "r1_r2_maxabs")
    if rep:
        a.bars.append(Bar("r1 vs r2 max-abs, worst episode", max(rep), None, "context", None))
    if not m:
        a.notes.append("No reference for this policy, so only the served chunks' own statistics.")
    for r in sorted(rows, key=lambda r: -(r.get("maxabs_ref") or 0)):
        why = []
        if (r.get("maxabs_ref") or 0) > 0.2:
            why.append(f"max-abs {r['maxabs_ref']} from the reference")
        if r.get("gripper_ref") is not None and r.get("gripper_r1") != r["gripper_ref"]:
            why.append(f"gripper events at {r.get('gripper_r1')} vs the reference's {r['gripper_ref']}")
        if r.get("r1_stats", {}).get("const_dims"):
            why.append(f"{r['r1_stats']['const_dims']} constant dims")
        if why:
            a.flags.append(f"{r['id']}: " + "; ".join(why))


def _tok_stats(x: np.ndarray, y: np.ndarray) -> dict:
    """Token-level agreement of [tokens, dim] latents."""
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    yn = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    cos = (xn * yn).sum(axis=1)
    nn = np.concatenate([(xn[s:s + 1024] @ yn.T).argmax(axis=1) for s in range(0, len(xn), 1024)])
    return {"cos": _num(M.cosine(x, y), 6), "tok_cos_median": _num(np.median(cos), 5),
            "tok_cos_min": _num(cos.min(), 5), "nn_agree": _num((nn == np.arange(len(x))).mean(), 4)}


def analyze_latents(info: RunInfo, out: Path, opts: Opts) -> Analysis:
    """V-JEPA2-AC rollouts ([steps, tokens, dim]) or V-JEPA2 embeddings ([1, tokens, dim])."""
    a = Analysis(columns=["id", "shape", "cos_ref", "cos_bf16_ref", "dist_ratio", "tok_cos_median", "tok_cos_min",
                          "nn_agree", "steps_cos_ref", "consecutive_maxabs_min", "consecutive_cos", "r1_r2_maxabs"])
    bf16 = info.arms.get("bf16")
    charts = []
    for it in info.items:
        x = _load_npy(_served(info, 0, it.id, ".npy"))
        row: dict = {"id": it.id}
        a.rows.append(row)
        if x is None:
            continue
        x = x.reshape(-1, *x.shape[-2:])
        row.update(shape="x".join(map(str, x.shape)), finite=bool(np.isfinite(x).all()))
        x2 = _load_npy(_served(info, 1, it.id, ".npy"))
        if x2 is not None and x2.size == x.size:
            row["r1_r2_maxabs"] = _num(np.abs(x - x2.reshape(x.shape)).max(), 6)
        if len(x) > 1:
            row["consecutive_maxabs_min"] = _num(min(np.abs(x[k + 1] - x[k]).max() for k in range(len(x) - 1)), 6)
            row["consecutive_cos"] = [_num(M.cosine(x[k], x[k + 1]), 4) for k in range(len(x) - 1)]
        ref, b = _load_npy(_ref(info.ref, it.id, ".npy")), _load_npy(_ref(bf16, it.id, ".npy"))
        if ref is None:
            continue
        ref = ref.reshape(-1, *ref.shape[-2:])
        if ref.shape[1:] != x.shape[1:]:
            continue
        # Only step 0 shares the served context in the AC reference, so step 0 is the one judged.
        row.update(_tok_stats(x[0], ref[0]))
        row["cos_ref"] = row.pop("cos")
        row["steps_cos_ref"] = [_num(M.cosine(x[k], ref[k]), 4) for k in range(min(len(x), len(ref)))]
        if b is not None:
            b = b.reshape(-1, *b.shape[-2:])
            row["cos_bf16_ref"] = _num(M.cosine(b[0], ref[0]), 6)
            row["bf16_steps_cos_ref"] = [_num(M.cosine(b[k], ref[k]), 4) for k in range(min(len(b), len(ref)))]
            row["dist_ratio"] = _num((1 - row["cos_ref"]) / max(1 - row["cos_bf16_ref"], 1e-12))
        if len(x) > 1:
            lines = [D.Line("mstar r1", row["steps_cos_ref"], 0)]
            if "bf16_steps_cos_ref" in row:
                lines.append(D.Line("bf16 reference", row["bf16_steps_cos_ref"], 2, True))
            charts.append(D.line_chart(lines, f"{it.id} cosine to the fp32 reference", size=(300, 200), xlabel="step"))
    if charts:
        D.panels(charts, 5, out / "plots" / "latents.png",
                 title="cosine to the reference per rollout step (only step 0 shares the served context)")
        a.artifacts["plots/latents.png"] = ("per episode, cosine to the fp32 reference per rollout step, with the "
                                            "bf16 copy")
    rows = a.rows
    a.summary["n"] = len(rows)
    bad = [r["id"] for r in rows if r.get("finite") is False]
    a.bars.append(Bar("outputs with non-finite values", len(bad), None, "0", not bad))
    ratio = _vals(rows, "dist_ratio")
    cos = _vals(rows, "cos_ref")
    if ratio:
        a.bars.append(Bar("distance from fp32 (1 - cosine) over the bf16 copy's, median / worst",
                          f"{_fmt(_med(ratio))} / {_fmt(max(ratio))}", None, "at most 2 (the V-JEPA2 family's gate)",
                          max(ratio) <= 2))
    if cos:
        a.bars.append(Bar("cosine to the fp32 reference (step 0), worst", min(cos), min(_vals(rows, "cos_bf16_ref"),
                                                                                       default=None), "context", None))
    rep = [r["id"] for r in rows if r.get("consecutive_maxabs_min") == 0]
    if any("consecutive_maxabs_min" in r for r in rows):
        a.bars.append(Bar("responses with a rollout step equal to the step before", len(rep), None, "0", not rep))
    if not info.ref:
        a.notes.append("No reference, so only finiteness, repeats and r1 vs r2.")
    for r in rows:
        why = []
        if (r.get("dist_ratio") or 0) > 2:
            why.append(f"{r['dist_ratio']}x the bf16 copy's distance from fp32")
        if r["id"] in rep:
            why.append("repeats a rollout step")
        if why:
            a.flags.append(f"{r['id']}: " + "; ".join(why))
    return a


ANALYZERS = {"image": analyze_images, "video": analyze_videos, "av": analyze_videos, "world": analyze_videos,
             "tts": analyze_speech, "actions": analyze_actions, "latents": analyze_latents,
             "embedding": analyze_latents}


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def _cell(v) -> str:
    if isinstance(v, str):
        return v.replace("|", "/").replace("\n", " ")[:90]
    if isinstance(v, list) and len(v) > 8:
        return _fmt(v[:8]) + ", ..."
    return _fmt(v)


def write_items(a: Analysis, path: Path) -> None:
    cols = [c for c in a.columns if any(r.get(c) is not None for r in a.rows)]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(_cell(r.get(c)) for c in cols) + " |" for r in a.rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_default(o):
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_outputs(info: RunInfo, a: Analysis, out: Path, opts: Opts) -> None:
    arms = {k: v.parent.name for k, v in info.arms.items()}
    header = {"run": str(info.dir), "config": info.report.get("config"), "run_id": info.report.get("run_id"),
              "family": info.report.get("family"), "kind": info.kind, "set": info.family.set_name,
              "n_items": len(info.items), "runs": [r.name for r in info.runs],
              "reference": str(info.ref) if info.ref else None, "arms": arms,
              "runner_verdict": info.report.get("verdict"), "served_heads": info.report.get("served_heads"),
              "written": datetime.datetime.now().isoformat(timespec="seconds"),
              "options": {"asr": opts.asr, "optional": opts.optional, "sbs": opts.sbs}}
    payload = {**header, "summary": a.summary, "bars": [b._asdict() for b in a.bars], "flags": a.flags,
               "skipped": a.skipped, "notes": a.notes, "artifacts": a.artifacts, "rows": a.rows}
    (out / "metrics.json").write_text(json.dumps(payload, indent=1, default=_json_default, ensure_ascii=False))
    write_items(a, out / "items.md")
    ref_name = info.ref.parent.name if info.ref else None
    md = [f"# Analysis: {header['config']} `{info.dir.name}`", "",
          f"{info.report.get('family')} family, set `{header['set']}`, {len(info.items)} items, served runs "
          f"{', '.join(header['runs'])}, head {header['served_heads']}. The runner's verdict is "
          f"{header['runner_verdict']} (report.md).",
          f"Reference: `{ref_name}`" + (f", arms {', '.join(f'{k} (`{v}`)' for k, v in arms.items())}" if arms else "")
          if ref_name else "Reference: none found, so only the served outputs' own statistics.", "",
          "The bars are proposed thresholds to calibrate, not verdicts. The reference column is the same measure "
          "on the reference, or, for distances to the reference, on its math-SDPA (or bf16) copy: the spread a "
          "kernel change alone gives.", "",
          "| measure | served | reference | proposed bar | within |", "|---|---|---|---|---|"]
    for b in a.bars:
        md.append(f"| {b.measure} | {_cell(b.served)} | {_cell(b.spread)} | {b.rule} | "
                  f"{'-' if b.ok is None else ('yes' if b.ok else '**no**')} |")
    md += ["", "## Look at these first", ""]
    md += [f"- {f}" for f in a.flags[:15]] or ["- nothing stands out on the measures above"]
    if len(a.flags) > 15:
        md.append(f"- and {len(a.flags) - 15} more in metrics.json")
    if a.notes:
        md += ["", "## Notes", ""] + [f"- {n}" for n in a.notes]
    md += ["", "## Skipped", ""] + ([f"- {k}: {v}" for k, v in a.skipped.items()] or ["- nothing"])
    md += ["", "## Files", "", "- `items.md`: one row per item", "- `metrics.json`: everything, with per-frame curves "
           "and per-item statistics"] + [f"- `{k}`: {v}" for k, v in a.artifacts.items()]
    (out / "analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--reference", type=Path, default=None,
                    help="reference dir (<data-root>/references/<id>[/<set>]); default: the family's own")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out", type=Path, default=None, help="default: <run_dir>/analysis")
    ap.add_argument("--no-asr", action="store_true", help="no Whisper; the runner's cached transcripts are still used")
    ap.add_argument("--no-optional", action="store_true", help="skip LPIPS, CLIP score and speaker similarity")
    ap.add_argument("--no-sbs", action="store_true", help="no side-by-side mp4s")
    ap.add_argument("--device", default="cpu", help="device for Whisper (default cpu)")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    info = load_run(run_dir, args.reference, args.data_root)
    opts = Opts(workers=args.workers, asr=not args.no_asr, optional=not args.no_optional, sbs=not args.no_sbs,
                device=args.device)
    out = (args.out or run_dir / "analysis").resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"{run_dir}: {info.kind}, {len(info.items)} items, reference {info.ref}, arms {sorted(info.arms)}",
          flush=True)
    a = ANALYZERS[info.kind](info, out, opts)
    write_outputs(info, a, out, opts)
    print(f"wrote {out / 'analysis.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
