"""Speed against quality over a run's methods: per-method means, the Pareto frontier, whether the default sits on
it, and the plot.

A method is on the frontier when no other is at least as fast and at least as good, and better in one of the two.
The default passes while nothing is clearly better at both: at least ``min_speed_gain`` faster and
``min_quality_gain`` better at once. The margins keep run-to-run noise from failing it.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple
from xml.sax.saxutils import escape


class Metric(NamedTuple):
    """A quality axis: the record field, how it reads on the plot, whether more is better, and the gain that
    counts as clearly better in the default check."""

    key: str
    title: str
    higher_is_better: bool
    min_gain: float
    digits: int


METRICS = {
    "psnr": Metric("psnr", "PSNR against dense (dB)", True, 0.3, 1),
    "ssim": Metric("ssim", "SSIM against dense", True, 0.005, 3),
    "lpips": Metric("lpips", "LPIPS against dense (lower is better)", False, 0.005, 3),
}
MIN_SPEED_GAIN = 0.02

_MEANS = ("psnr", "ssim", "lpips", "clip", "clip_ref", "flicker", "flicker_ref", "flicker_excess", "motion_psnr",
          "computed_steps", "branch_forwards", "dit_seconds")


@dataclass(frozen=True)
class Point:
    """One method over the run's prompts. Means are over prompts; a None is a value no record carried."""

    name: str
    label: str
    is_default: bool
    prompts: int
    seconds: float
    # the dense run's seconds over this method's, on the prompts both ran
    speedup: float
    dit_speedup: float | None
    psnr: float | None
    # the worst frame of any prompt
    psnr_min: float | None
    ssim: float | None
    lpips: float | None
    lpips_max: float | None
    clip: float | None
    clip_ref: float | None
    flicker: float | None
    flicker_ref: float | None
    flicker_excess: float | None
    motion_psnr: float | None
    computed_steps: float | None
    branch_forwards: float | None
    dit_seconds: float | None
    # the prompt with the lowest PSNR, and that PSNR
    worst_prompt: int | None
    worst_prompt_psnr: float | None
    # every clip bit-identical to its dense clip
    identical: bool

    @property
    def clip_delta(self) -> float | None:
        return None if self.clip is None or self.clip_ref is None else self.clip - self.clip_ref

    def quality(self, metric: Metric) -> float | None:
        """The metric oriented so that more is better."""
        value = getattr(self, metric.key)
        return None if value is None else (value if metric.higher_is_better else -value)

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "clip_delta": self.clip_delta}


def _mean(values: Iterable[Any]) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def aggregate(records: Iterable[Mapping[str, Any]], dense: str = "dense") -> list[Point]:
    """One :class:`Point` per method from a run's records (``kind`` "method"), each against the dense records of
    the prompts it ran. A method without a timed dense counterpart is left out."""
    records = list(records)
    dense_by_prompt = {r["prompt_index"]: r for r in records if r.get("kind") == "dense" and r["method"] == dense}
    by_method: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        if record.get("kind") == "method" and record["prompt_index"] in dense_by_prompt:
            by_method.setdefault(record["method"], []).append(record)
    points = []
    for name, rows in by_method.items():
        refs = [dense_by_prompt[r["prompt_index"]] for r in rows]
        seconds, dense_seconds = _mean(r["seconds"] for r in rows), _mean(r["seconds"] for r in refs)
        means = {key: _mean(r.get(key) for r in rows) for key in _MEANS}
        dense_dit = _mean(r.get("dit_seconds") for r in refs)
        worst = min((r for r in rows if r.get("psnr") is not None), key=lambda r: r["psnr"], default=None)
        points.append(Point(
            name=name, label=rows[0].get("label", name), is_default=bool(rows[0].get("is_default")),
            prompts=len(rows), seconds=seconds, speedup=dense_seconds / seconds,
            dit_speedup=None if means["dit_seconds"] is None or dense_dit is None else dense_dit / means["dit_seconds"],
            psnr_min=min((r["psnr_min"] for r in rows if r.get("psnr_min") is not None), default=None),
            lpips_max=max((r["lpips_max"] for r in rows if r.get("lpips_max") is not None), default=None),
            worst_prompt=None if worst is None else worst["prompt_index"],
            worst_prompt_psnr=None if worst is None else worst["psnr"],
            identical=all(r.get("identical") for r in rows),
            **means,
        ))
    return sorted(points, key=lambda p: (p.speedup, p.name))


def _dominates(a: Point, b: Point, metric: Metric) -> bool:
    qa, qb = a.quality(metric), b.quality(metric)
    return a.speedup >= b.speedup and qa >= qb and (a.speedup > b.speedup or qa > qb)


def frontier(points: Sequence[Point], metric: Metric = METRICS["psnr"]) -> list[Point]:
    """The points no other dominates, slowest first. Two points equal in both stay on it together."""
    scored = [p for p in points if p.quality(metric) is not None]
    return sorted(
        (p for p in scored if not any(_dominates(other, p, metric) for other in scored)),
        key=lambda p: (p.speedup, p.name),
    )


class DefaultVerdict(NamedTuple):
    ok: bool
    default: str | None
    # the methods clearly better than the default at both speed and quality
    better: tuple[str, ...]
    reason: str


def check_default(
    points: Sequence[Point], metric: Metric = METRICS["psnr"], default: str | None = None,
    min_speed_gain: float = MIN_SPEED_GAIN, min_quality_gain: float | None = None,
) -> DefaultVerdict:
    """Whether the default (the point named, else the one flagged) holds its place: it fails when another method
    is at least ``min_speed_gain`` faster (a fraction) and ``min_quality_gain`` better (the metric's own margin by
    default) at once."""
    gain = metric.min_gain if min_quality_gain is None else min_quality_gain
    chosen = [p for p in points if (p.name == default if default is not None else p.is_default)]
    if len(chosen) != 1 or chosen[0].quality(metric) is None:
        which = f"the default {default}" if default is not None else f"a default ({len(chosen)} flagged)"
        return DefaultVerdict(False, default, (), f"{which} is not among the run's scored methods")
    mine = chosen[0]
    better = tuple(
        p.name for p in points
        if p.name != mine.name and p.quality(metric) is not None
        and p.speedup >= mine.speedup * (1 + min_speed_gain) and p.quality(metric) >= mine.quality(metric) + gain
    )
    where = f"{mine.name} at {mine.speedup:.2f}x, {metric.key} {getattr(mine, metric.key):.{metric.digits}f}"
    if better:
        return DefaultVerdict(
            False, mine.name, better,
            f"{where} is beaten at both by {', '.join(better)} (at least {min_speed_gain:.0%} faster and {gain:g} "
            f"better in {metric.key})",
        )
    on = "on" if mine in frontier(points, metric) else "within the margins of"
    return DefaultVerdict(True, mine.name, (), f"{where} is {on} the frontier")


# ---------------------------------------------------------------------------------------------------- the plot
# An emphasis scatter: the frontier in one hue with its line, the default in a second hue and another shape, the
# rest in grey. Colors by role, for a light and a dark surface, from a palette checked for color-vision separation
# and contrast on both; text stays in ink colors, and each point's numbers are in its hover title and the report's
# table. Plain class rules rather than custom properties, so renderers other than browsers draw it too.
_ROLES = (
    # (selector, property, light, dark)
    (".surface", "fill", "#fcfcfb", "#1a1a19"),
    ("text", "fill", "#52514e", "#c3c2b7"),
    (".title", "fill", "#0b0b0b", "#f0efec"),
    (".tick, .axis-title", "fill", "#898781", "#898781"),
    (".grid", "stroke", "#e1e0d9", "#2c2c2a"),
    (".axis", "stroke", "#c3c2b7", "#383835"),
    (".frontier-line", "stroke", "#2a78d6", "#3987e5"),
    (".mark", "stroke", "#fcfcfb", "#1a1a19"),
    (".rest", "fill", "#898781", "#898781"),
    (".on", "fill", "#2a78d6", "#3987e5"),
    (".default", "fill", "#eb6834", "#d95926"),
    (".pt:hover .mark, .pt:focus .mark", "stroke", "#0b0b0b", "#f0efec"),
)
_STYLE = (
    """
  text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; font-size: 11px; }
  .title { font-size: 15px; font-weight: 600; }
  .subtitle { font-size: 12px; }
  .tick { font-variant-numeric: tabular-nums; }
  .grid, .axis { stroke-width: 1; }
  .frontier-line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .mark { stroke-width: 2; }
  .hit { fill: transparent; }
  .pt:hover .mark, .pt:focus .mark { stroke-width: 1.5; }
  .pt:focus { outline: none; }
"""
    + "".join(f"  {selector} {{ {prop}: {light}; }}\n" for selector, prop, light, _ in _ROLES)
    + "  @media (prefers-color-scheme: dark) {\n"
    + "".join(f"    {selector} {{ {prop}: {dark}; }}\n" for selector, prop, _, dark in _ROLES)
    + "  }\n"
)
_WIDTH, _HEIGHT = 880, 540
_LEFT, _RIGHT, _TOP, _BOTTOM = 64, 28, 104, 52
_LABEL_HEIGHT, _CHAR_WIDTH = 13, 6.1


def _ticks(lo: float, hi: float, target: int = 6) -> list[float]:
    """Round tick values covering ``[lo, hi]``."""
    span = max(hi - lo, 1e-9)
    raw = span / target
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    first = math.floor(lo / step) * step
    count = int(math.ceil((hi - first) / step)) + 1
    return [round(first + i * step, 10) for i in range(count)]


def _tooltip(point: Point) -> str:
    parts = [f"{point.name}: {point.speedup:.2f}x ({point.seconds:.2f} s)"]
    for key, text, digits in (("psnr", "PSNR {} dB", 1), ("ssim", "SSIM {}", 3), ("lpips", "LPIPS {}", 3)):
        value = getattr(point, key)
        if value is not None:
            parts.append(text.format(f"{value:.{digits}f}"))
    if point.computed_steps is not None:
        parts.append(f"{point.computed_steps:g} steps computed")
    return ", ".join(parts) + f". {point.label}"


class _Box(NamedTuple):
    x0: float
    y0: float
    x1: float
    y1: float

    def overlaps(self, other: _Box) -> bool:
        return self.x0 < other.x1 and other.x0 < self.x1 and self.y0 < other.y1 and other.y0 < self.y1


def _fit(text: str, width: float) -> str:
    """``text`` cut to ``width`` pixels at the label size, with an ellipsis where it was cut."""
    room = int(width / _CHAR_WIDTH)
    return text if len(text) <= room else text[: max(room - 1, 0)].rstrip() + "\u2026"


def _place_label(text: str, x: float, y: float, taken: list[_Box], plot: _Box) -> tuple[float, float, str] | None:
    """Where a point's label fits: beside it, above or below, then off its corners, the first spot inside the plot
    that covers no mark, frontier line or earlier label. None when none does, and the label is left to the hover
    title and the table."""
    width = len(text) * _CHAR_WIDTH
    spots = [
        (x + 9, y + 4, "start", _Box(x + 9, y - 8, x + 9 + width, y + 6)),
        (x - 9, y + 4, "end", _Box(x - 9 - width, y - 8, x - 9, y + 6)),
        (x, y - 10, "middle", _Box(x - width / 2, y - 10 - _LABEL_HEIGHT, x + width / 2, y - 8)),
        (x, y + 19, "middle", _Box(x - width / 2, y + 8, x + width / 2, y + 8 + _LABEL_HEIGHT)),
    ]
    for dy in (-12, 22, -24, 34):
        spots += [
            (x + 8, y + dy, "start", _Box(x + 8, y + dy - 11, x + 8 + width, y + dy + 3)),
            (x - 8, y + dy, "end", _Box(x - 8 - width, y + dy - 11, x - 8, y + dy + 3)),
        ]
    for tx, ty, anchor, box in spots:
        inside = box.x0 >= plot.x0 and box.x1 <= plot.x1 and box.y0 >= plot.y0 and box.y1 <= plot.y1
        if inside and not any(box.overlaps(other) for other in taken):
            taken.append(box)
            return tx, ty, anchor
    return None


def pareto_svg(
    points: Sequence[Point], metric: Metric = METRICS["psnr"], default: str | None = None,
    title: str = "Step cache: speed against quality", subtitle: str = "",
) -> str:
    """The run's methods as an SVG scatter, speedup across and ``metric`` up: the frontier joined by a line and
    labelled, the default (the point named, else the one flagged) marked, every other method grey. Each point's
    numbers are in its hover title."""
    scored = [p for p in points if getattr(p, metric.key) is not None]
    if not scored:
        raise ValueError(f"no method has a {metric.key} to plot")
    on_frontier = {p.name for p in frontier(scored, metric)}
    is_default = {p.name for p in scored if (p.name == default if default is not None else p.is_default)}
    xs, ys = [p.speedup for p in scored], [getattr(p, metric.key) for p in scored]
    x_ticks = _ticks(min(1.0, *xs), max(xs))
    y_pad = max((max(ys) - min(ys)) * 0.06, 10 ** -metric.digits)
    y_ticks = _ticks(min(ys) - y_pad, max(ys) + y_pad)
    plot = _Box(_LEFT, _TOP, _WIDTH - _RIGHT, _HEIGHT - _BOTTOM)

    def sx(value: float) -> float:
        return plot.x0 + (value - x_ticks[0]) / (x_ticks[-1] - x_ticks[0]) * (plot.x1 - plot.x0)

    def sy(value: float) -> float:
        return plot.y1 - (value - y_ticks[0]) / (y_ticks[-1] - y_ticks[0]) * (plot.y1 - plot.y0)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_WIDTH} {_HEIGHT}" width="{_WIDTH}" '
        f'height="{_HEIGHT}" role="img" aria-label="{escape(title, {chr(34): "&quot;"})}">',
        f"<style>{_STYLE}</style>",
        f'<rect class="surface" width="{_WIDTH}" height="{_HEIGHT}"/>',
        f'<text class="title" x="{_LEFT}" y="28">{escape(title)}</text>',
        f'<text class="subtitle" x="{_LEFT}" y="47">{escape(_fit(subtitle, (plot.x1 - plot.x0) / 1.09))}</text>',
    ]
    # the legend: a key per role, in the same shapes as the marks
    legend_x = _LEFT
    for kind, text in (("on", "on the frontier"), ("default", "the default"), ("rest", "dominated")):
        if kind == "default" and not is_default:
            continue
        out.append(_mark(kind, legend_x + 5, 66))
        out.append(f'<text x="{legend_x + 15}" y="70">{escape(text)}</text>')
        legend_x += 15 + len(text) * _CHAR_WIDTH + 22
    for tick in y_ticks:
        out.append(f'<line class="grid" x1="{plot.x0}" x2="{plot.x1}" y1="{sy(tick):.1f}" y2="{sy(tick):.1f}"/>')
        out.append(
            f'<text class="tick" x="{plot.x0 - 8}" y="{sy(tick) + 4:.1f}" text-anchor="end">'
            f"{tick:.{metric.digits}f}</text>"
        )
    for tick in x_ticks:
        out.append(
            f'<text class="tick" x="{sx(tick):.1f}" y="{plot.y1 + 18}" text-anchor="middle">{tick:g}x</text>'
        )
    out.append(f'<line class="axis" x1="{plot.x0}" x2="{plot.x1}" y1="{plot.y1}" y2="{plot.y1}"/>')
    out.append(
        f'<text class="axis-title" x="{(plot.x0 + plot.x1) / 2:.1f}" y="{_HEIGHT - 12}" text-anchor="middle">'
        "speedup over dense (dense = 1x)</text>"
    )
    out.append(f'<text class="axis-title" x="{_LEFT - 44}" y="{_TOP - 10}">{escape(metric.title)}</text>')

    # what a label may not cover: every mark, and the frontier's line
    taken = [_Box(sx(p.speedup) - 7, sy(getattr(p, metric.key)) - 7, sx(p.speedup) + 7,
                  sy(getattr(p, metric.key)) + 7) for p in scored]
    line = [(sx(p.speedup), sy(getattr(p, metric.key)))
            for p in sorted((p for p in scored if p.name in on_frontier), key=lambda p: p.speedup)]
    if len(line) > 1:
        out.append(f'<polyline class="frontier-line" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in line)}"/>')
        for (x0, y0), (x1, y1) in zip(line, line[1:], strict=False):
            pieces = max(1, int(math.hypot(x1 - x0, y1 - y0) / 5))
            taken += [
                _Box(x0 + (x1 - x0) * i / pieces - 3, y0 + (y1 - y0) * i / pieces - 3,
                     x0 + (x1 - x0) * i / pieces + 3, y0 + (y1 - y0) * i / pieces + 3)
                for i in range(pieces + 1)
            ]
    # grey first, so the frontier and the default sit on top
    order = sorted(scored, key=lambda p: (p.name in is_default, p.name in on_frontier))
    labels = []
    for point in order:
        x, y = sx(point.speedup), sy(getattr(point, metric.key))
        kind = "default" if point.name in is_default else "on" if point.name in on_frontier else "rest"
        out.append(
            f'<g class="pt" tabindex="0"><title>{escape(_tooltip(point))}</title>'
            f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="12"/>{_mark(kind, x, y)}</g>'
        )
    # label the default first, then the frontier, each only where it fits
    for point in sorted(scored, key=lambda p: (p.name not in is_default, p.speedup)):
        if point.name not in is_default and point.name not in on_frontier:
            continue
        text = f"{point.name} (default)" if point.name in is_default else point.name
        spot = _place_label(text, sx(point.speedup), sy(getattr(point, metric.key)), taken, plot)
        if spot is not None:
            labels.append(f'<text x="{spot[0]:.1f}" y="{spot[1]:.1f}" text-anchor="{spot[2]}">{escape(text)}</text>')
    return "\n".join([*out, *labels, "</svg>"]) + "\n"


def _mark(kind: str, x: float, y: float) -> str:
    """A point's mark: a diamond for the default, a dot otherwise, each with a ring in the surface color."""
    if kind == "default":
        return (
            f'<rect class="mark default" x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" '
            f'transform="rotate(45 {x:.1f} {y:.1f})"/>'
        )
    radius = 5 if kind == "on" else 4
    return f'<circle class="mark {kind}" cx="{x:.1f}" cy="{y:.1f}" r="{radius}"/>'
