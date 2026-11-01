"""PNG drawing for the correctness analysis: labelled image grids, line charts, difference heatmaps
and spectrograms. PIL only (matplotlib isn't in the serving venv)."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Categorical slots in a fixed order. Each source keeps its slot in every chart: 0 served r1,
# 1 reference, 2 math-SDPA reference (or a bf16 copy), 3 served r2 or the fp32-anchor reference.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SURFACE, INK, INK2, GRID = (252, 252, 251), (11, 11, 11), (82, 81, 78), (225, 224, 220)
# One-hue sequential ramp, near zero first, for heatmaps and spectrograms.
RAMP = ("#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
_FONTS = ("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
          "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf")


def _rgb(c: str) -> tuple[int, int, int]:
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


@functools.lru_cache(maxsize=None)
def font(size: int = 14):
    from PIL import ImageFont

    for p in _FONTS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def heatmap(values: np.ndarray, vmax: float) -> np.ndarray:
    """[H, W] values in [0, vmax] as an RGB uint8 image on the sequential ramp."""
    ramp = np.array([_rgb(c) for c in RAMP], dtype=np.float64)
    t = np.clip(np.asarray(values, dtype=np.float64) / max(vmax, 1e-12), 0, 1) * (len(ramp) - 1)
    lo = np.floor(t).astype(np.int64)
    hi = np.minimum(lo + 1, len(ramp) - 1)
    w = (t - lo)[..., None]
    return (ramp[lo] * (1 - w) + ramp[hi] * w).round().astype(np.uint8)


def abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean absolute difference over the color channels, [H, W] in 8-bit levels."""
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(axis=-1)


def thumb(img: np.ndarray, width: int) -> np.ndarray:
    """Downscale to ``width`` pixels wide (aspect kept)."""
    from PIL import Image

    im = Image.fromarray(np.asarray(img, dtype=np.uint8))
    if im.width > width:
        im = im.resize((width, max(1, round(im.height * width / im.width))), Image.LANCZOS)
    return np.asarray(im)


def grid(rows: Sequence[Sequence[np.ndarray | None]], path: Path, *, col_labels: Sequence[str] = (),
         row_labels: Sequence[str] = (), cell_labels: Sequence[Sequence[str]] = (), box: tuple[int, int] = (320, 320),
         title: str | None = None, fsize: int = 13) -> None:
    """Save a sheet with one row per entry of ``rows``; a None cell is drawn as an empty red box."""
    from PIL import Image, ImageDraw

    f, fb = font(fsize), font(fsize + 1)
    ncol = max(len(r) for r in rows)
    pad, head, rowhead = 6, fsize + 8, fsize + 10
    top = fsize + 14 if title else 0
    colhead = head if col_labels else 0
    height = top + colhead + len(rows) * (rowhead + box[1] + head + pad) + pad
    sheet = Image.new("RGB", (pad + ncol * (box[0] + pad), height), "white")
    d = ImageDraw.Draw(sheet)
    if title:
        d.text((pad, 4), title, fill=INK, font=fb)
    for c, lab in enumerate(col_labels):
        d.text((pad + c * (box[0] + pad), top + 2), lab, fill=INK, font=fb)
    y = top + colhead
    for r, row in enumerate(rows):
        if r < len(row_labels):
            d.text((pad, y + 2), row_labels[r], fill=INK, font=f)
        y += rowhead
        for c, img in enumerate(row):
            x = pad + c * (box[0] + pad)
            if img is None:
                d.rectangle([x, y, x + box[0], y + box[1]], outline=(208, 59, 59))
            else:
                im = Image.fromarray(np.asarray(img, dtype=np.uint8))
                im.thumbnail(box, Image.LANCZOS)
                sheet.paste(im, (x, y))
            labels = cell_labels[r] if r < len(cell_labels) else ()
            if c < len(labels) and labels[c]:
                d.text((x, y + box[1] + 2), labels[c], fill=INK2, font=f)
        y += box[1] + head + pad
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, optimize=True)


class Line(NamedTuple):
    label: str
    ys: Sequence[float | None]
    slot: int
    dashed: bool = False
    xs: Sequence[float] | None = None  # default 0, 1, 2, ...


def line_chart(lines: Sequence[Line], title: str, *, size: tuple[int, int] = (360, 240),
               ylim: tuple[float, float] | None = None, xlabel: str = "frame", hline: float | None = None):
    """One small line chart (a PIL image) with a legend row under the title."""
    from PIL import Image, ImageDraw

    w, h = size
    im = Image.new("RGB", size, SURFACE)
    d = ImageDraw.Draw(im)
    f, fs = font(12), font(11)
    left, right, top, bottom = 46, 10, 40, 22
    pts_all = [(x, y) for ln in lines for x, y in zip(ln.xs if ln.xs is not None else range(len(ln.ys)), ln.ys,
                                                      strict=False) if y is not None and np.isfinite(y)]
    d.text((left, 2), title[:56], fill=INK, font=f)
    if not pts_all:
        d.text((left, top), "no data", fill=INK2, font=f)
        return im
    lo, hi = ylim if ylim is not None else (min(p[1] for p in pts_all), max(p[1] for p in pts_all))
    if ylim is None and hline is not None:
        lo, hi = min(lo, hline), max(hi, hline)
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    x0, x1 = min(p[0] for p in pts_all), max(p[0] for p in pts_all)
    x1 = x1 if x1 > x0 else x0 + 1

    def px(x):
        return left + (w - left - right) * (x - x0) / (x1 - x0)

    def py(v):
        return top + (h - top - bottom) * (1 - (min(max(v, lo), hi) - lo) / (hi - lo))

    for k in range(5):
        v = lo + (hi - lo) * k / 4
        d.line([(left, py(v)), (w - right, py(v))], fill=GRID, width=1)
        d.text((2, py(v) - 6), f"{v:.3g}", fill=INK2, font=fs)
    if hline is not None and lo <= hline <= hi:
        d.line([(left, py(hline)), (w - right, py(hline))], fill=(160, 160, 160), width=1)
    d.text((left, h - 15), f"{x0:g}", fill=INK2, font=fs)
    d.text((w - right - 6 * len(f"{x1:g}"), h - 15), f"{x1:g}", fill=INK2, font=fs)
    d.text(((left + w - right) / 2 - 3 * len(xlabel), h - 15), xlabel, fill=INK2, font=fs)
    lx = left
    for ln in lines:
        col = _rgb(SERIES[ln.slot % len(SERIES)])
        xs = ln.xs if ln.xs is not None else range(len(ln.ys))
        pts = [(px(x), py(y)) for x, y in zip(xs, ln.ys, strict=False) if y is not None and np.isfinite(y)]
        if len(pts) == 1:
            x, y = pts[0]
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=col)
        for a, b in zip(pts, pts[1:], strict=False):
            if ln.dashed:
                for t0 in np.arange(0, 1, 0.34):
                    t1 = min(t0 + 0.17, 1)
                    d.line([(a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0),
                            (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)], fill=col, width=2)
            else:
                d.line([a, b], fill=col, width=2)
        d.line([(lx, 26), (lx + 14, 26)], fill=col, width=2)
        d.text((lx + 17, 20), ln.label, fill=INK, font=fs)
        lx += 26 + int(d.textlength(ln.label, font=fs))
    return im


def panels(imgs: Sequence, ncol: int, path: Path, title: str | None = None) -> None:
    """Tile same-size PIL images into one PNG."""
    from PIL import Image, ImageDraw

    w, h = imgs[0].size
    nrow = (len(imgs) + ncol - 1) // ncol
    top = 24 if title else 0
    sheet = Image.new("RGB", (min(ncol, len(imgs)) * w, nrow * h + top), "white")
    if title:
        ImageDraw.Draw(sheet).text((6, 4), title, fill=INK, font=font(14))
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % ncol) * w, top + (i // ncol) * h))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, optimize=True)


def spectrogram(rows: Sequence[tuple[str, np.ndarray | None]], path: Path, title: str, *, width: int = 900,
                row_h: int = 120, range_db: float = 80.0) -> None:
    """Stacked log-mel spectrograms (``[frames, mels]`` in dB), one shared color scale, low mels at the bottom."""
    from PIL import Image, ImageDraw

    mats = [m for _, m in rows if m is not None and m.size]
    if not mats:
        return
    peak = max(float(m.max()) for m in mats)
    n_frames = max(m.shape[0] for m in mats)
    head = 18
    sheet = Image.new("RGB", (width + 12, 24 + len(rows) * (row_h + head + 6)), "white")
    d = ImageDraw.Draw(sheet)
    d.text((6, 4), title, fill=INK, font=font(13))
    y = 24
    for label, m in rows:
        d.text((6, y), label, fill=INK2, font=font(12))
        y += head
        if m is None or not m.size:
            d.rectangle([6, y, 6 + width, y + row_h], outline=(208, 59, 59))
        else:
            img = heatmap(np.clip(m - (peak - range_db), 0, None)[:, ::-1].T, range_db)
            # Same time scale on every row, so a shorter clip ends early.
            im = Image.fromarray(img).resize((max(1, round(width * m.shape[0] / n_frames)), row_h), Image.BILINEAR)
            sheet.paste(im, (6, y))
        y += row_h + 6
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, optimize=True)
