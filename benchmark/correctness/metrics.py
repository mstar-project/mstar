"""Metric helpers for the correctness runner.

Plain numpy. soundfile, scipy, PIL and transformers are optional and imported lazily.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

# ---------------------------------------------------------------------------
# distributions
# ---------------------------------------------------------------------------


def distribution(values: Sequence[float]) -> dict:
    """n, min, p5, median, mean and max of the finite values. Infinities are counted separately."""
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    finite = arr[np.isfinite(arr)]
    out = {"n": int(arr.size), "n_inf": int(np.isinf(arr).sum()), "n_nan": int(np.isnan(arr).sum())}
    if finite.size:
        out.update(
            min=float(finite.min()),
            p5=float(np.percentile(finite, 5)),
            median=float(np.median(finite)),
            mean=float(finite.mean()),
            max=float(finite.max()),
        )
    return out


# ---------------------------------------------------------------------------
# greedy text: exact match, first divergence, near-tie analysis
# ---------------------------------------------------------------------------


def first_divergence(ref: Sequence, out: Sequence) -> int | None:
    """Index where the sequences first differ, None if equal. A strict prefix diverges at its end."""
    n = min(len(ref), len(out))
    for i in range(n):
        if ref[i] != out[i]:
            return i
    return None if len(ref) == len(out) else n


@dataclass(frozen=True)
class GreedyComparison:
    """One prompt's greedy output vs the reference, in the reference tokenizer's ids."""

    ref_len: int
    out_len: int
    first_divergence: int | None
    exact: bool
    # Details at the divergence. Each side's token (eos_id if it stopped), the reference's logprob gap
    # between the two, and whether the server's token was in the reference top-k.
    ref_token: int | None = None
    out_token: int | None = None
    logprob_gap: float | None = None
    out_in_ref_topk: bool | None = None
    near_tie: bool | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def compare_greedy(
    ref_ids: Sequence[int],
    out_ids: Sequence[int],
    *,
    horizon: int,
    ref_topk: Sequence[dict[int, float]] | None = None,
    eos_id: int | None = None,
    ref_stopped: bool = False,
    out_stopped: bool = True,
    near_tie_nats: float = 1.0,
) -> GreedyComparison:
    """Compare the first ``horizon`` tokens and classify the first divergence.

    ``ref_topk[i]`` maps token id -> logprob for the reference's top-k at step i. A divergence is a
    near tie when the output's token is in that top-k within ``near_tie_nats`` of the reference's
    pick. ``ref_stopped``/``out_stopped`` say whether a side ended on EOS instead of the budget.

    """
    ref = list(ref_ids)[:horizon]
    out = list(out_ids)[:horizon]
    div = first_divergence(ref, out)
    if div is None:
        return GreedyComparison(len(ref), len(out), None, True)
    if div < len(ref) and div < len(out):
        ref_tok, out_tok = ref[div], out[div]
    elif div == len(out) < len(ref):
        # The output ended here. That counts as EOS only if it stopped on its own.
        ref_tok, out_tok = ref[div], (eos_id if out_stopped else None)
    else:  # div == len(ref) < len(out)
        ref_tok, out_tok = (eos_id if ref_stopped else None), out[div]
    gap = in_topk = near = None
    if ref_topk is not None and div < len(ref_topk) and ref_tok is not None and out_tok is not None:
        dist = ref_topk[div]
        in_topk = out_tok in dist
        if in_topk and ref_tok in dist:
            gap = float(dist[ref_tok] - dist[out_tok])
            near = gap <= near_tie_nats
        else:
            near = False
    return GreedyComparison(len(ref), len(out), div, False, ref_tok, out_tok, gap, in_topk, near)


def summarize_greedy(comparisons: Sequence[GreedyComparison], short_horizon: int = 64) -> dict:
    """Roll up per-prompt comparisons."""
    n = len(comparisons)
    if n == 0:
        return {"n": 0}
    exact = sum(c.exact for c in comparisons)
    exact_short = sum(
        c.exact or (c.first_divergence is not None and c.first_divergence >= short_horizon) for c in comparisons
    )
    divs = [c.first_divergence for c in comparisons if c.first_divergence is not None]
    near = sum(bool(c.near_tie) for c in comparisons)
    unexplained = [i for i, c in enumerate(comparisons) if not c.exact and c.near_tie is not True]
    return {
        "n": n,
        "exact": exact,
        "exact_rate": exact / n,
        f"exact_first_{short_horizon}": exact_short,
        f"exact_first_{short_horizon}_rate": exact_short / n,
        "divergence_positions": distribution(divs),
        "near_tie_divergences": near,
        "unexplained": unexplained,
        "max_logprob_gap": max((c.logprob_gap for c in comparisons if c.logprob_gap is not None), default=None),
    }


# ---------------------------------------------------------------------------
# word error rate
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s']|(?<!\w)'|'(?!\w)")


def normalize_basic(text: str) -> str:
    """Lowercase, strip punctuation (keeping apostrophes inside words) and collapse spaces."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split())


def make_normalizer(kind: str = "whisper") -> Callable[[str], str]:
    """Text normalizer. "whisper" uses transformers' EnglishTextNormalizer like the repo's TTS WER
    scripts, and falls back to normalize_basic if transformers is missing."""
    if kind == "whisper":
        try:
            from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
        except ImportError:
            return normalize_basic
        whisper = EnglishTextNormalizer({})
        return lambda s: " ".join(whisper(s).split())
    if kind == "basic":
        return normalize_basic
    raise ValueError(f"unknown normalizer {kind!r}")


def word_edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    """Word-level edit distance."""
    prev = list(range(len(hyp) + 1))
    for i, rw in enumerate(ref, start=1):
        cur = [i]
        for j, hw in enumerate(hyp, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1]


def wer(refs: Sequence[str], hyps: Sequence[str], normalize: Callable[[str], str] = normalize_basic) -> dict:
    """Corpus WER plus per-utterance rows."""
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references vs {len(hyps)} hypotheses")
    rows, errors, words = [], 0, 0
    for r, h in zip(refs, hyps, strict=True):
        rw, hw = normalize(r).split(), normalize(h or "").split()
        e = word_edit_distance(rw, hw)
        errors += e
        words += len(rw)
        rows.append({"errors": e, "words": len(rw), "wer": (e / len(rw)) if rw else float(bool(hw))})
    return {"wer": errors / max(words, 1), "errors": errors, "words": words, "rows": rows}


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------


def read_wav(path) -> tuple[np.ndarray, int]:
    """Read a WAV as mono float32 in [-1, 1]. Returns samples and sample rate."""
    try:
        import soundfile as sf

        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return np.ascontiguousarray(audio.mean(axis=1), dtype=np.float32), int(sr)
    except ImportError:
        import wave

        with wave.open(str(path), "rb") as w:
            sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if width != 2:
            raise ValueError(f"{path}: only 16-bit PCM without soundfile") from None
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        return x.reshape(-1, ch).mean(axis=1), sr


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data[: len(data) - len(data) % 2], dtype="<i2").astype(np.float32) / 32768.0


def write_wav(path, samples: np.ndarray, sample_rate: int) -> None:
    """Write 16-bit PCM. Non-finite samples become 0."""
    import wave

    x = np.nan_to_num(np.asarray(samples, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Resample with scipy, or linear interpolation if scipy isn't installed."""
    if sr_from == sr_to:
        return np.asarray(x, dtype=np.float32)
    n_out = int(round(len(x) * sr_to / sr_from))
    try:
        from scipy.signal import resample_poly
    except ImportError:
        t = np.arange(n_out) * (sr_from / sr_to)
        return np.interp(t, np.arange(len(x)), np.asarray(x, dtype=np.float64)).astype(np.float32)
    g = math.gcd(int(sr_from), int(sr_to))
    return resample_poly(np.asarray(x, dtype=np.float64), sr_to // g, sr_from // g).astype(np.float32)[:n_out]


def _frame_rms_db(x: np.ndarray, sr: int, frame_ms: float) -> np.ndarray:
    n = max(1, int(sr * frame_ms / 1000))
    usable = (len(x) // n) * n
    if usable == 0:
        return np.full(1, -np.inf)
    frames = x[:usable].reshape(-1, n).astype(np.float64)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    with np.errstate(divide="ignore"):
        return 20 * np.log10(rms)


def audio_stats(x: np.ndarray, sr: int, *, silence_db: float = -50.0, frame_ms: float = 20.0) -> dict:
    """Duration, non-finite samples, level, silence and clipping for one clip."""
    x = np.asarray(x, dtype=np.float32)
    nonfinite = int((~np.isfinite(x)).sum())
    clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    db = _frame_rms_db(clean, sr, frame_ms)
    voiced = db > silence_db
    frame_s = frame_ms / 1000.0
    lead = int(np.argmax(voiced)) if voiced.any() else len(db)
    trail = int(np.argmax(voiced[::-1])) if voiced.any() else len(db)
    rms = float(np.sqrt(np.mean(clean.astype(np.float64) ** 2))) if clean.size else 0.0
    return {
        "duration_s": len(x) / sr if sr else 0.0,
        "nonfinite": nonfinite,
        "peak": float(np.abs(clean).max()) if clean.size else 0.0,
        "rms_dbfs": 20 * math.log10(rms) if rms > 0 else -math.inf,
        "silence_frac": float(1.0 - voiced.mean()) if db.size else 1.0,
        "clip_frac": float((np.abs(clean) >= 0.999).mean()) if clean.size else 0.0,
        "lead_silence_s": lead * frame_s,
        "trail_silence_s": trail * frame_s,
    }


def audio_problems(
    stats: dict, *, min_duration_s: float = 0.3, max_silence_frac: float = 0.7, max_clip_frac: float = 0.01
) -> list[str]:
    """List the checks a clip fails."""
    problems = []
    if stats["nonfinite"]:
        problems.append(f"{stats['nonfinite']} non-finite samples")
    if stats["duration_s"] < min_duration_s:
        problems.append(f"duration {stats['duration_s']:.2f}s < {min_duration_s}s")
    if stats["silence_frac"] > max_silence_frac:
        problems.append(f"silence {stats['silence_frac']:.0%} > {max_silence_frac:.0%}")
    if stats["clip_frac"] > max_clip_frac:
        problems.append(f"clipping {stats['clip_frac']:.1%} > {max_clip_frac:.0%}")
    return problems


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    fmax = fmax or sr / 2

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)

    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, bins.size))
    for m in range(n_mels):
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        up = (bins - lo) / max(mid - lo, 1e-9)
        down = (hi - bins) / max(hi - mid, 1e-9)
        fb[m] = np.clip(np.minimum(up, down), 0.0, None)
    return fb


def log_mel(x: np.ndarray, sr: int, *, n_fft: int = 1024, hop: int = 256, n_mels: int = 80) -> np.ndarray:
    """log10 mel energies, [frames, n_mels]."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * np.hanning(n_fft), axis=1)) ** 2
    return np.log10(spec @ _mel_filterbank(sr, n_fft, n_mels).T + 1e-20)


def log_mel_distance(a: np.ndarray, b: np.ndarray, sr: int, dynamic_range_db: float = 80.0) -> float:
    """Mean abs log-mel difference in dB over the common length (0 for identical audio).

    Kokoro's vocoder phase is chaotic, so waveforms don't line up but the spectral envelope does.
    Both sides are floored ``dynamic_range_db`` below the peak so near-silent bands don't dominate.

    """
    n = min(len(a), len(b))
    if n == 0:
        return math.inf
    ma, mb = log_mel(a[:n], sr), log_mel(b[:n], sr)
    floor = max(ma.max(), mb.max()) - dynamic_range_db / 10.0
    return float(10.0 * np.mean(np.abs(np.maximum(ma, floor) - np.maximum(mb, floor))))


# ---------------------------------------------------------------------------
# images and video frames
# ---------------------------------------------------------------------------


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    """PSNR in dB between two same-size arrays (inf when identical)."""
    if a.shape != b.shape:
        raise ValueError(f"size mismatch {a.shape} vs {b.shape}")
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return math.inf if mse == 0 else 20 * math.log10(peak) - 10 * math.log10(mse)


def _to_luma(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float64)
    if img.ndim == 3 and img.shape[-1] >= 3:
        return img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114
    return img if img.ndim == 2 else img[..., 0]


def ssim(a: np.ndarray, b: np.ndarray, peak: float = 255.0, sigma: float = 1.5) -> float:
    """Mean SSIM on luma (Gaussian window, sigma 1.5)."""
    if a.shape != b.shape:
        raise ValueError(f"size mismatch {a.shape} vs {b.shape}")
    x, y = _to_luma(a), _to_luma(b)
    c1, c2 = (0.01 * peak) ** 2, (0.03 * peak) ** 2
    radius = int(3.5 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()

    def blur(z):  # separable Gaussian, reflected edges like scipy's gaussian_filter
        for axis in (0, 1):
            pad = [(0, 0), (0, 0)]
            pad[axis] = (radius, radius)
            zp = np.pad(z, pad, mode="symmetric")
            z = sum(w * np.take(zp, np.arange(i, i + z.shape[axis]), axis=axis) for i, w in enumerate(k))
        return z

    mx, my = blur(x), blur(y)
    vx = blur(x * x) - mx * mx
    vy = blur(y * y) - my * my
    cxy = blur(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return float(s.mean())


def load_image(path) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))


def contact_sheet(rows: Sequence[Sequence[np.ndarray | None]], labels: Sequence[str], path, thumb: int = 256) -> None:
    """Save a labelled grid PNG, one row per sample and one column per source."""
    from PIL import Image, ImageDraw

    ncol = max(len(r) for r in rows)
    pad, head = 4, 18
    sheet = Image.new("RGB", (ncol * (thumb + pad) + pad, len(rows) * (thumb + head + pad) + pad), "white")
    draw = ImageDraw.Draw(sheet)
    for r, row in enumerate(rows):
        y = pad + r * (thumb + head + pad)
        for c, img in enumerate(row):
            x = pad + c * (thumb + pad)
            label = labels[r * ncol + c] if r * ncol + c < len(labels) else ""
            draw.text((x, y), label[:40], fill="black")
            if img is None:
                draw.rectangle([x, y + head, x + thumb, y + head + thumb], outline="red")
                continue
            im = Image.fromarray(np.asarray(img, dtype=np.uint8))
            im.thumbnail((thumb, thumb))
            sheet.paste(im, (x, y + head))
    sheet.save(path)


def frame_psnrs(ref: np.ndarray, out: np.ndarray, peak: float = 255.0) -> list[float]:
    """Per-frame PSNR over the frames both stacks have."""
    n = min(len(ref), len(out))
    return [psnr(ref[i], out[i], peak) for i in range(n)]


# ---------------------------------------------------------------------------
# arrays (actions, latents, embeddings)
# ---------------------------------------------------------------------------


def array_errors(out: np.ndarray, ref: np.ndarray) -> dict:
    """MSE, MAE, max abs and relative L2 of out vs ref (same size)."""
    out = np.asarray(out, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if out.shape != ref.shape:
        raise ValueError(f"size mismatch {out.shape} vs {ref.shape}")
    diff = out - ref
    denom = float(np.linalg.norm(ref))
    return {
        "mse": float(np.mean(diff**2)),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.abs(diff).max()) if diff.size else 0.0,
        "rel_l2": float(np.linalg.norm(diff) / denom) if denom > 0 else (0.0 if not diff.any() else math.inf),
        "nonfinite": int((~np.isfinite(out)).sum()),
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0 if na == nb else 0.0
    return float(a @ b / (na * nb))
