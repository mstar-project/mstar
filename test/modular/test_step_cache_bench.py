"""The step cache's verification harness (``benchmark/step_cache``): its metrics on synthetic clips, the Pareto
frontier and the default's check, the report, a run's resume, and the regression mode's gates. CPU only, no weights:
clips come from a fake backend, and the optional metrics (LPIPS, CLIP) stay off."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
import threading
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import torch

from benchmark.step_cache import methods as method_grid
from benchmark.step_cache import metrics, pareto, run
from benchmark.step_cache.backends.base import Generated, backend_class
from benchmark.step_cache.backends.server import ServerBackend
from benchmark.step_cache.methods import DENSE, NO_SKIP, Method, plan_cost
from benchmark.step_cache.prompts import CI_PROMPTS, PROMPTS, take
from benchmark.step_cache.report import RunConfig, load_records, write_report
from benchmark.wan22.video_bench_wan22 import VBENCH_PROMPTS
from mstar.model.components.diffusion.step_cache import Interval, step_cache_policy

# --------------------------------------------------------------------- metrics


def _clip(seed: int = 0, frames: int = 8, size: int = 32) -> torch.Tensor:
    """A textured clip that drifts one pixel a frame."""
    gen = torch.Generator().manual_seed(seed)
    base = torch.randint(40, 200, (size, size + frames, 3), generator=gen, dtype=torch.uint8)
    return torch.stack([base[:, t:t + size] for t in range(frames)])


def _smooth_clip(frames: int = 10, size: int = 32) -> torch.Tensor:
    """A soft gradient that drifts slowly: next to no change between frames beyond the drift."""
    x = torch.arange(size, dtype=torch.float32).view(1, 1, size, 1)
    t = torch.arange(frames, dtype=torch.float32).view(frames, 1, 1, 1)
    wave = 128 + 60 * torch.sin(2 * math.pi * (x + t) / 64) * torch.ones(1, size, 1, 3)
    return wave.round().to(torch.uint8)


def test_identical_clips_score_the_caps_and_share_a_digest():
    clip = _clip()
    assert metrics.psnr_per_frame(clip, clip.clone()).tolist() == [metrics.PSNR_CAP] * 8
    assert metrics.ssim_per_frame(clip, clip.clone()).tolist() == pytest.approx([1.0] * 8)
    assert metrics.flicker_excess(clip, clip.clone()) == 0.0
    assert metrics.motion_psnr(clip, clip.clone()) == metrics.PSNR_CAP
    assert metrics.frames_md5(clip) == metrics.frames_md5(clip.numpy())
    other = clip.clone()
    other[3, 5, 7, 1] += 1
    assert metrics.frames_md5(clip) != metrics.frames_md5(other)
    # the digest covers the shape: the same bytes cut into other frames are another clip
    assert metrics.frames_md5(clip) != metrics.frames_md5(clip.reshape(4, 64, 32, 3))


def test_psnr_of_a_known_error_is_the_analytic_value():
    ref = torch.full((3, 16, 16, 3), 100, dtype=torch.uint8)
    psnr = metrics.psnr_per_frame(ref, ref + 10)
    assert psnr.tolist() == pytest.approx([10 * math.log10(255 ** 2 / 100)] * 3)
    # each frame is scored on its own
    test = ref.clone()
    test[1] += 20
    assert metrics.psnr_per_frame(ref, test).tolist() == pytest.approx(
        [metrics.PSNR_CAP, 10 * math.log10(255 ** 2 / 400), metrics.PSNR_CAP]
    )


@pytest.mark.parametrize("window", ["gaussian", "uniform"])
def test_ssim_of_two_flat_frames_is_the_luminance_term(window):
    """Flat frames have no variance, so SSIM is (2ab + C1) / (a^2 + b^2 + C1) whatever the window."""
    a, b, c1 = 100.0, 140.0, (0.01 * 255) ** 2
    ref = torch.full((2, 16, 16, 3), int(a), dtype=torch.uint8)
    test = torch.full((2, 16, 16, 3), int(b), dtype=torch.uint8)
    expected = (2 * a * b + c1) / (a * a + b * b + c1)
    assert metrics.ssim_per_frame(ref, test, window).tolist() == pytest.approx([expected] * 2)


def test_ssim_falls_with_noise_and_refuses_a_frame_smaller_than_its_window():
    clip = _clip()
    gen = torch.Generator().manual_seed(1)
    light = (clip.int() + torch.randint(-4, 5, clip.shape, generator=gen)).clamp(0, 255).to(torch.uint8)
    heavy = (clip.int() + torch.randint(-40, 41, clip.shape, generator=gen)).clamp(0, 255).to(torch.uint8)
    assert 1.0 > metrics.ssim_per_frame(clip, light).mean() > metrics.ssim_per_frame(clip, heavy).mean() > 0.0
    with pytest.raises(ValueError, match="window"):
        metrics.ssim_per_frame(clip[:, :8, :8], clip[:, :8, :8])
    with pytest.raises(ValueError, match="gaussian"):
        metrics.ssim_per_frame(clip, clip, window="box")


@pytest.mark.parametrize("window,kwargs", [
    ("gaussian", {"gaussian_weights": True, "sigma": 1.5, "use_sample_covariance": False}),
    ("uniform", {}),
])
def test_ssim_matches_scikit_image(window, kwargs):
    structural_similarity = pytest.importorskip("skimage.metrics").structural_similarity
    ref = _clip(frames=3, size=48)
    gen = torch.Generator().manual_seed(2)
    test = (ref.int() + torch.randint(-25, 26, ref.shape, generator=gen)).clamp(0, 255).to(torch.uint8)
    ours = metrics.ssim_per_frame(ref, test, window)
    theirs = [
        structural_similarity(a.numpy(), b.numpy(), channel_axis=2, data_range=255, **kwargs)
        for a, b in zip(ref, test, strict=True)
    ]
    assert ours.tolist() == pytest.approx(theirs, abs=1e-3)


def test_an_alternating_brightness_reads_as_flicker_and_a_still_offset_does_not():
    steady = _smooth_clip()
    sign = torch.tensor([6 if t % 2 else -6 for t in range(10)]).view(-1, 1, 1, 1)
    flickering = (steady.int() + sign).clamp(0, 255).to(torch.uint8)
    offset = (steady.int() + 6).clamp(0, 255).to(torch.uint8)
    # the flicker is in the second temporal difference; a constant offset is not
    assert metrics.flicker_excess(steady, flickering) > 1.0
    assert abs(metrics.flicker_excess(steady, offset)) < 0.05
    assert metrics.temporal_flicker(flickering) < metrics.temporal_flicker(steady)
    # the offset moves every frame alike: the per-frame PSNR sees it, the motion PSNR does not
    assert metrics.psnr_per_frame(steady, offset).mean() < 40 < metrics.motion_psnr(steady, offset)
    assert metrics.motion_psnr(steady, flickering) < metrics.motion_psnr(steady, offset)
    # a still clip scores 1, and a clip too short to have a second difference has no excess
    still = steady[:1].repeat(5, 1, 1, 1)
    assert metrics.temporal_flicker(still) == 1.0 and metrics.flicker_excess(steady[:2], flickering[:2]) == 0.0
    assert metrics.flicker_excess(still, flickering[:5]) > 1.0


def test_clips_of_another_shape_or_dtype_are_refused():
    clip = _clip()
    with pytest.raises(ValueError, match="differ in shape"):
        metrics.psnr_per_frame(clip, clip[:4])
    with pytest.raises(ValueError, match="uint8"):
        metrics.psnr_per_frame(clip.float(), clip.float())


def test_a_missing_optional_metric_is_none_with_a_note_and_never_a_crash(monkeypatch):
    monkeypatch.setitem(sys.modules, "lpips", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    scorer = metrics.Scorer.build()
    assert not scorer.lpips.available and not scorer.clip.available
    assert [note.split(":")[0] for note in scorer.notes] == ["LPIPS off", "CLIP score off"]
    clip = _clip()
    quality = scorer.compare(clip, clip.clone(), "a prompt")
    assert quality.lpips is None and quality.clip is None and quality.clip_ref is None
    assert quality.identical and quality.psnr == metrics.PSNR_CAP and quality.psnr_min == metrics.PSNR_CAP
    # asked not to load them, the scorer does not try
    off = metrics.Scorer.build(optional=False)
    assert off.lpips is None and off.clip is None and len(off.notes) == 2


def _mp4(clip: torch.Tensor) -> bytes:
    av = pytest.importorskip("av")
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=8)
        stream.width, stream.height, stream.pix_fmt = clip.shape[2], clip.shape[1], "yuv420p"
        stream.bit_rate = 4_000_000
        for frame in clip:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame.numpy(), format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def test_an_mp4_decodes_to_its_frames():
    clip = _smooth_clip(frames=6)
    decoded = metrics.decode_mp4(_mp4(clip))
    assert decoded.shape == clip.shape and decoded.dtype == torch.uint8
    # lossy, but the same pictures in the same order
    assert metrics.psnr_per_frame(clip, decoded).mean() > 25
    assert metrics.psnr_per_frame(clip, decoded.flip(0)).mean() < metrics.psnr_per_frame(clip, decoded).mean()


# --------------------------------------------------------------------- the frontier and the default


def _point(name: str, speedup: float, psnr: float, *, is_default: bool = False, lpips: float | None = None):
    return pareto.Point(
        name=name, label=f"label of {name}", is_default=is_default, prompts=4, seconds=10 / speedup,
        speedup=speedup, dit_speedup=None, psnr=psnr, psnr_min=psnr - 8, ssim=0.9, lpips=lpips, lpips_max=None,
        clip=None, clip_ref=None, flicker=0.97, flicker_ref=0.97, flicker_excess=0.01, motion_psnr=35.0,
        computed_steps=25, branch_forwards=50.0, dit_seconds=None, worst_prompt=7, worst_prompt_psnr=psnr - 5,
        identical=False,
    )


def test_the_frontier_is_the_points_nothing_beats_at_both():
    points = [
        _point("a", 1.5, 31.0), _point("b", 1.9, 30.0), _point("c", 2.2, 28.8),
        _point("slow-and-worse", 1.4, 30.0), _point("inside", 1.8, 29.0), _point("b-twin", 1.9, 30.0),
        _point("same-speed-worse", 2.2, 28.0),
    ]
    # two points equal in both stay on it together
    assert [p.name for p in pareto.frontier(points)] == ["a", "b", "b-twin", "c"]
    assert [p.name for p in pareto.frontier(points[:1])] == ["a"]
    # LPIPS is better when lower
    by_lpips = [_point("x", 1.5, 30, lpips=0.05), _point("y", 2.0, 30, lpips=0.08), _point("z", 1.8, 30, lpips=0.09)]
    assert [p.name for p in pareto.frontier(by_lpips, pareto.METRICS["lpips"])] == ["x", "y"]
    # a point without the metric is not on its frontier
    assert pareto.frontier([_point("u", 1.5, 30)], pareto.METRICS["lpips"]) == []


def test_a_default_beaten_at_both_fails_and_names_what_beats_it():
    points = [
        _point("default", 1.8, 29.0, is_default=True), _point("better", 1.9, 30.0), _point("also", 2.2, 29.4),
        _point("faster-only", 2.5, 27.0), _point("finer-only", 1.2, 33.0),
    ]
    verdict = pareto.check_default(points)
    assert not verdict.ok and verdict.default == "default" and verdict.better == ("better", "also")
    assert "better, also" in verdict.reason


def test_a_default_on_the_frontier_or_inside_its_margins_passes():
    on = [_point("default", 1.9, 30.0, is_default=True), _point("a", 1.5, 31.0), _point("c", 2.2, 28.8)]
    verdict = pareto.check_default(on)
    assert verdict.ok and verdict.better == () and "is on the frontier" in verdict.reason
    # dominated, but by less than 2% in speed and by less than 0.3 dB: noise, not a better method
    for speedup, psnr in [(1.91, 30.5), (2.2, 30.2), (1.9, 30.0)]:
        near = [_point("default", 1.9, 30.0, is_default=True), _point("near", speedup, psnr)]
        assert pareto.check_default(near).ok
    assert "within the margins" in pareto.check_default(
        [_point("default", 1.9, 30.0, is_default=True), _point("near", 1.91, 30.1)]).reason
    # tighter margins fail the same points; a named default overrides the flag
    assert not pareto.check_default(near[:1] + [_point("near", 1.91, 30.5)], min_speed_gain=0.001).ok
    assert pareto.check_default(on, default="a").default == "a"
    assert not pareto.check_default([_point("a", 1.5, 31.0)]).ok


def test_the_plot_is_one_well_formed_svg_with_every_point_and_the_frontier():
    points = [
        _point("mag<&>", 1.5, 31.0), _point("default", 1.9, 30.0, is_default=True), _point("c", 2.2, 28.8),
        _point("inside", 1.8, 29.0),
    ]
    svg = pareto.pareto_svg(points, title="5B & A14B", subtitle="832x480 <33 frames>")
    root = ET.fromstring(svg)
    ns = {"s": "http://www.w3.org/2000/svg"}
    titles = [t.text for t in root.findall(".//s:g/s:title", ns)]
    # every point carries its numbers in a hover title, names escaped
    assert len(titles) == 4 and any(t.startswith("mag<&>: 1.50x") for t in titles)
    assert all("PSNR" in t and "SSIM" in t for t in titles)
    # the frontier's line joins its three points; the default is the one diamond among the points
    line = root.find(".//s:polyline", ns)
    assert len(line.get("points").split()) == 3
    diamonds = [r for r in root.findall(".//s:g/s:rect", ns) if "default" in r.get("class")]
    assert len(diamonds) == 1
    texts = [t.text for t in root.findall(".//s:text", ns)]
    assert "5B & A14B" in texts and "default (default)" in texts and "the default" in texts
    # every hover target is at least 24 px across
    assert all(float(c.get("r")) >= 12 for c in root.findall(".//s:g/s:circle", ns) if c.get("class") == "hit")
    with pytest.raises(ValueError, match="lpips"):
        pareto.pareto_svg(points, pareto.METRICS["lpips"])


# --------------------------------------------------------------------- a run, with a fake backend


def _strength(method: Method) -> float:
    """How far the fake backend moves a method's clip from dense, from its spec."""
    spec = method.step_cache
    if spec is None or spec == NO_SKIP.step_cache:
        return 0.0
    if spec.get("method") == "plan":
        return 4.0 * (spec["num_steps"] / len(spec["computed"]) - 1)
    return 100 * spec.get("threshold", 0.0) + 4 * (spec.get("every", 1) - 1)


class FakeBackend:
    """Clips from the seed: dense is a fixed clip per seed, and a method adds noise in proportion to how much it
    skips and reports a time that falls with it."""

    name = "fake"
    broken_noskip = False
    # every backend built, so a test can see what a run through ``run.main`` generated
    instances: list[FakeBackend] = []

    def __init__(self):
        self.configured: list[str] = []
        self.generated: list[tuple[str, int]] = []
        self.method = DENSE
        self.closed = False
        type(self).instances.append(self)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> FakeBackend:
        return cls()

    def describe(self) -> dict:
        return {"gpu": "none (fake)", "torch": torch.__version__}

    def configure(self, method: Method) -> None:
        self.configured.append(method.name)
        self.method = method

    def generate(self, prompt: str, seed: int, *, size: str, frames: int, steps: int) -> Generated:
        self.generated.append((self.method.name, seed))
        clip = _clip(seed, frames=frames)
        strength = _strength(self.method)
        if self.broken_noskip and self.method.name == NO_SKIP.name:
            strength = 1.0
        if strength:
            gen = torch.Generator().manual_seed(seed + 1)
            noise = torch.randn(clip.shape, generator=gen) * strength
            clip = (clip.float() + noise).round().clamp(0, 255).to(torch.uint8)
        computed = steps if not strength else max(2, round(steps / (1 + strength / 8)))
        return Generated(
            frames=clip, seconds=10.0 * computed / steps, dit_seconds=9.0 * computed / steps,
            computed_steps=computed, branch_forwards=2.0 * computed, extra={"size": size},
        )

    def close(self) -> None:
        self.closed = True


_CONFIG = RunConfig(model="wan22_ti2v_5b", backend="fake", size="32x32", frames=6, steps=20)
_METHODS = (
    DENSE,
    Method("mag-t0.04-k2", "gentle", {"method": "magcache", "threshold": 0.04, "max_skip_steps": 2}, True),
    Method("int3", "every third", {"method": "interval", "every": 3}),
    NO_SKIP,
)


def _run(tmp_path, backend=None, methods=_METHODS, prompts=None, **kwargs):
    backend = backend or FakeBackend()
    records = run.run_methods(
        backend, methods, prompts or take(3), _CONFIG, tmp_path, metrics.Scorer.build(optional=False),
        log=lambda _: None, **kwargs,
    )
    return backend, records


def test_a_run_scores_every_method_against_dense_of_the_same_prompt(tmp_path):
    backend, records = _run(tmp_path)
    # dense on every prompt, twice on the first two for the floor, then each method
    assert backend.configured == ["dense", "mag-t0.04-k2", "int3", "noskip"]
    assert [r["kind"] for r in records].count("dense") == 3 and [r["kind"] for r in records].count("floor") == 2
    assert len(backend.generated) == 3 + 2 + 3 * 3 == len(records) == len(load_records(tmp_path))
    by = {(r["kind"], r["method"], r["prompt_index"]): r for r in records}
    assert by[("floor", "dense", 0)]["identical"] and by[("method", "noskip", 2)]["identical"]
    assert by[("method", "noskip", 2)]["frames_md5"] == by[("dense", "dense", 2)]["frames_md5"]
    gentle, rough = by[("method", "mag-t0.04-k2", 1)], by[("method", "int3", 1)]
    assert 100 > gentle["psnr"] > rough["psnr"] > 10 and gentle["ssim"] > rough["ssim"]
    assert gentle["seed"] == 1001 and gentle["prompt"] == PROMPTS[1].text and gentle["extra"] == {"size": "32x32"}
    assert gentle["step_cache"] == _METHODS[1].step_cache and gentle["is_default"]


def test_a_resumed_run_generates_only_what_is_missing(tmp_path):
    first, _ = _run(tmp_path, methods=_METHODS[:2], prompts=take(2))
    assert len(first.generated) == 2 + 2 + 2
    with pytest.raises(SystemExit, match="--resume"):
        _run(tmp_path)
    # more methods and one more prompt: only the new clips, and no method that is complete is configured
    second, records = _run(tmp_path, prompts=take(3), resume=True)
    assert second.generated == [
        ("dense", 1002), ("mag-t0.04-k2", 1002), ("int3", 1000), ("int3", 1001), ("int3", 1002),
        ("noskip", 1000), ("noskip", 1001), ("noskip", 1002),
    ]
    third, again = _run(tmp_path, prompts=take(3), resume=True)
    assert third.generated == [] and third.configured == [] and len(again) == len(records)
    # a dense clip lost from disk is generated again, as the methods are scored against it
    (tmp_path / "dense" / "p1.pt").unlink()
    fourth, _ = _run(tmp_path, prompts=take(3), resume=True)
    assert fourth.generated == [("dense", 1001)]
    with pytest.raises(SystemExit, match="another config"):
        run.run_methods(
            FakeBackend(), _METHODS, take(3), _CONFIG._replace(steps=30), tmp_path,
            metrics.Scorer.build(optional=False), resume=True, log=lambda _: None,
        )


def test_the_report_tables_every_method_and_says_where_the_default_sits(tmp_path):
    _, records = _run(tmp_path)
    report = write_report(
        tmp_path, records, env={"gpu": "none (fake)", "torch": "2.x"}, config=_CONFIG, notes=["LPIPS off: no lpips"],
    )
    text = report.path.read_text()
    rows = [line for line in text.splitlines() if line.startswith("| ")]
    # the header, dense, and one row per method, slowest first
    assert [row.split("|")[1].strip() for row in rows] == ["method", "dense", "noskip", "mag-t0.04-k2", "int3"]
    assert "**Hardware:** gpu none (fake); torch 2.x" in text and "32x32 x 6 frames, 20 steps" in text
    assert "**Dense run-to-run floor:** bit-identical on 2 of 2 prompts." in text
    assert "**Note:** LPIPS off: no lpips" in text and "**Default:** PASS" in text
    assert report.verdict.ok and report.verdict.default == "mag-t0.04-k2"
    assert [p.name for p in report.frontier] == ["noskip", "mag-t0.04-k2", "int3"]
    point = {p.name: p for p in report.points}["int3"]
    assert point.speedup == pytest.approx(10.0 / point.seconds) and point.prompts == 3 and point.lpips is None
    assert point.dit_speedup == pytest.approx(point.speedup) and point.worst_prompt in (0, 1, 2)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["frontier"] == ["noskip", "mag-t0.04-k2", "int3"] and summary["default"]["ok"]
    assert summary["config"]["steps"] == 20 and len(summary["points"]) == 3
    assert ET.fromstring((tmp_path / "pareto.svg").read_text()).tag.endswith("svg")
    # another default, named: int3 is the fastest and nothing beats it at both
    assert write_report(tmp_path, records, env={}, config=_CONFIG, default="int3").verdict.default == "int3"


# --------------------------------------------------------------------- the regression mode


def _ci(tmp_path, monkeypatch, name: str, *extra: str, broken: bool = False) -> int:
    monkeypatch.setattr(run, "backend_class", lambda _name: FakeBackend)
    monkeypatch.setattr(FakeBackend, "broken_noskip", broken)
    return run.main([
        "--ci", "--backend", "inproc", "--model", "wan22_ti2v_5b", "--out", str(tmp_path / name), "--size", "32x32",
        "--frames", "6", "--no-optional-metrics", "--baseline", str(tmp_path / "baseline.json"), *extra,
    ])


def test_the_regression_mode_passes_on_its_own_baseline_and_fails_without_one(tmp_path, monkeypatch, capsys):
    assert _ci(tmp_path, monkeypatch, "no-baseline") == 1
    assert "no baseline at" in capsys.readouterr().out
    assert _ci(tmp_path, monkeypatch, "write", "--write-baseline") == 0
    baseline = json.loads((tmp_path / "baseline.json").read_text())
    # the default alone: the no-skip policy has its own gate
    assert list(baseline["methods"]) == ["mag-t0.06-k2"] and baseline["prompts"] == [p.index for p in CI_PROMPTS]
    assert baseline["config"]["steps"] == 20 and baseline["config"]["frames"] == 6
    assert _ci(tmp_path, monkeypatch, "again") == 0
    assert "regression mode: PASS" in capsys.readouterr().out
    records = load_records(tmp_path / "again")
    assert {r["method"] for r in records} == {"dense", "mag-t0.06-k2", "noskip"}
    assert {r["prompt_index"] for r in records} == {p.index for p in CI_PROMPTS}


def test_a_no_skip_policy_that_differs_from_dense_fails_the_regression_mode(tmp_path, monkeypatch, capsys):
    assert _ci(tmp_path, monkeypatch, "write", "--write-baseline") == 0
    assert _ci(tmp_path, monkeypatch, "broken", broken=True) == 1
    out = capsys.readouterr().out
    assert out.count("FAIL noskip differs from dense on prompt") == 3 and "regression mode: FAIL" in out


def test_a_default_under_its_baseline_fails_and_one_within_tolerance_passes(tmp_path, monkeypatch, capsys):
    assert _ci(tmp_path, monkeypatch, "write", "--write-baseline") == 0
    path = tmp_path / "baseline.json"
    baseline = json.loads(path.read_text())
    measured = dict(baseline["methods"]["mag-t0.06-k2"])
    # the baseline asks 0.3 dB and 2% of speedup more than this run has: inside the tolerances
    baseline["methods"]["mag-t0.06-k2"] = {
        "psnr": measured["psnr"] + 0.3, "ssim": measured["ssim"] + 0.005, "speedup": measured["speedup"] * 1.02,
    }
    path.write_text(json.dumps(baseline))
    assert _ci(tmp_path, monkeypatch, "within") == 0
    capsys.readouterr()
    baseline["methods"]["mag-t0.06-k2"]["psnr"] = measured["psnr"] + 2.0
    baseline["methods"]["mag-t0.06-k2"]["speedup"] = measured["speedup"] * 1.5
    path.write_text(json.dumps(baseline))
    assert _ci(tmp_path, monkeypatch, "under") == 1
    out = capsys.readouterr().out
    assert "FAIL mag-t0.06-k2: psnr" in out and "FAIL mag-t0.06-k2: speedup" in out and "ssim" not in out
    # a baseline of another schedule is not compared against
    assert _ci(tmp_path, monkeypatch, "other-steps", "--steps", "10") == 1
    assert "is for another config" in capsys.readouterr().out


def test_a_sweep_takes_its_methods_by_prefix_and_lists_the_grid(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run, "backend_class", lambda _name: FakeBackend)
    assert run.main(["--model", "minimax_h3", "--list-methods"]) == 0
    listed = capsys.readouterr().out.splitlines()
    assert listed[0].startswith("dense:") and "mag-t0.06-k3 (default): MagCache thr 0.06, K 3" in listed
    code = run.main([
        "--backend", "inproc", "--model", "wan22_ti2v_5b", "--out", str(tmp_path / "sweep"), "--size", "32x32",
        "--frames", "6", "--prompts", "2", "--methods", "mag-t0.06,int2+o0", "--no-optional-metrics", "--strict",
    ])
    names = {r["method"] for r in load_records(tmp_path / "sweep")}
    assert names == {"dense", "mag-t0.06-k2", "mag-t0.06-k3", "mag-t0.06-k3+o1", "int2+o0"}
    assert code in (0, 1) and (tmp_path / "sweep" / "report.md").exists()
    with pytest.raises(ValueError, match="unknown backend"):
        backend_class("vllm")


def test_a_methods_file_joins_an_earlier_run_and_its_report(tmp_path, monkeypatch, capsys):
    """Searched plans arrive later as a file: the second invocation reuses the dense clips and the grid's records,
    generates only the file's methods, and reports on everything in the directory."""
    monkeypatch.setattr(run, "backend_class", lambda _name: FakeBackend)
    out = tmp_path / "run"
    common = ["--backend", "inproc", "--model", "wan22_ti2v_5b", "--out", str(out), "--size", "32x32", "--frames",
              "6", "--prompts", "2", "--no-optional-metrics"]
    assert run.main([*common, "--methods", "mag-t0.06-k2,int2+o0"]) == 0
    plans = tmp_path / "plans.json"
    plans.write_text(json.dumps([
        {"name": "search-b20", "label": "searched, 20 steps", "step_cache":
            {"method": "plan", "computed": [0, 1, 2, 5, 8], "num_steps": 10}},
        {"name": "int2+o0", "step_cache": {"method": "interval", "every": 2, "retention_ratio": 0.2, "forecast": 0}},
    ]))
    FakeBackend.instances.clear()
    assert run.main([*common, "--resume", "--methods", "none", "--methods-file", str(plans)]) == 0
    assert FakeBackend.instances[-1].generated == [("search-b20", 1000), ("search-b20", 1001)]
    rows = [line.split("|")[1].strip() for line in (out / "report.md").read_text().splitlines()
            if line.startswith("| ")]
    assert sorted(rows[2:]) == ["int2+o0", "mag-t0.06-k2", "search-b20"]
    # the grid's default is still the one checked, from the first invocation's records
    assert "default: " in capsys.readouterr().out and json.loads((out / "summary.json").read_text())["default"][
        "default"] == "mag-t0.06-k2"
    # a file's own default takes the flag
    plans.write_text(json.dumps([{"name": "search-b20", "is_default": True, "step_cache":
                                  {"method": "plan", "computed": [0, 1, 2, 5, 8], "num_steps": 10}}]))
    assert run.main([*common, "--resume", "--methods", "none", "--methods-file", str(plans)]) == 0
    assert json.loads((out / "summary.json").read_text())["default"]["default"] == "search-b20"
    # a name already in the directory under another setting is refused rather than scored as the old clips
    plans.write_text(json.dumps([{"name": "search-b20", "step_cache": {"method": "interval", "every": 4}}]))
    with pytest.raises(SystemExit, match="another step_cache"):
        run.main([*common, "--resume", "--methods", "none", "--methods-file", str(plans)])


def test_a_methods_file_is_checked_as_it_is_read(tmp_path):
    path = tmp_path / "methods.json"
    spec = {"method": "interval", "every": 2}
    path.write_text(json.dumps([{"name": "a", "step_cache": spec}, {"name": "b", "label": "B", "step_cache": spec,
                                                                    "is_default": True}]))
    extra = method_grid.load_methods(path)
    assert extra == (Method("a", "a", spec), Method("b", "B", spec, True))
    grid = method_grid.grid("wan22_ti2v_5b")
    combined = method_grid.combine(method_grid.select(grid, ["none"]), extra)
    assert [m.name for m in combined] == ["dense", "a", "b"]
    # the file's default takes the flag from the grid's
    assert method_grid.default_of(method_grid.combine(grid, extra)).name == "b"
    assert method_grid.default_of(method_grid.combine(grid, extra[:1])).name == "mag-t0.06-k2"
    # the same name with the same setting is one method; with another setting it is refused
    same = Method("int2+o0", "again", next(m for m in grid if m.name == "int2+o0").step_cache, True)
    assert len(method_grid.combine(grid, [same])) == len(grid)
    assert method_grid.default_of(method_grid.combine(grid, [same])).name == "int2+o0"
    with pytest.raises(ValueError, match="different step_cache"):
        method_grid.combine(grid, [Method("int2+o0", "other", spec)])
    with pytest.raises(ValueError, match="more than one default"):
        method_grid.combine(grid, [Method("x", "x", spec, True), Method("y", "y", spec, True)])
    for bad in ({"name": "a"}, {"name": "a", "step_cache": None}, {"name": "a", "step_cache": spec, "skip": 1}):
        path.write_text(json.dumps([bad]))
        with pytest.raises(ValueError, match="a method is"):
            method_grid.load_methods(path)
    path.write_text(json.dumps({"name": "a", "step_cache": spec}))
    with pytest.raises(ValueError, match="JSON list"):
        method_grid.load_methods(path)


def test_the_toy_model_runs_a_sweep_and_the_regression_mode_in_process(tmp_path, capsys):
    """End to end on the CPU through the real in-process backend: the scaffold's policies on a toy Wan DiT."""
    pytest.importorskip("benchmark.step_cache.backends.inproc")
    out = tmp_path / "sweep"
    code = run.main([
        "--backend", "inproc", "--model", "tiny", "--out", str(out), "--prompts", "2", "--no-optional-metrics",
        "--methods", "mag-t0.06-k2,int2+o1",
    ])
    assert code == 0
    records = {(r["kind"], r["method"], r["prompt_index"]): r for r in load_records(out)}
    assert {name for _, name, _ in records} == {"dense", "mag-t0.06-k2", "int2+o1"}
    # dense is deterministic in one process, and each method's plan is what the policy says
    assert records[("floor", "dense", 0)]["identical"] and records[("dense", "dense", 0)]["extra"]["plan"] is None
    skipping = records[("method", "int2+o1", 1)]
    assert set(skipping["extra"]["plan"]) == {"C", "-"} and skipping["computed_steps"] < 8 and skipping["psnr"] < 100
    # a computed step forwards both guidance branches, a skipping one neither
    assert skipping["computed_steps"] == skipping["extra"]["plan"].count("C")
    assert skipping["branch_forwards"] == 2 * skipping["computed_steps"]
    assert records[("dense", "dense", 1)]["branch_forwards"] == 16.0
    assert (out / "pareto.svg").exists() and "| int2+o1 |" in (out / "report.md").read_text()

    # the regression mode: a no-skip policy through the real scaffold is bit-identical to dense
    baseline = tmp_path / "tiny.json"
    ci = ["--ci", "--backend", "inproc", "--model", "tiny", "--no-optional-metrics", "--baseline", str(baseline)]
    assert run.main([*ci, "--out", str(tmp_path / "ci0"), "--write-baseline"]) == 0
    assert all(r["identical"] for r in load_records(tmp_path / "ci0") if r["method"] == "noskip")
    written = json.loads(baseline.read_text())
    # a toy model's milliseconds are noise: hold the quality floors only
    written["tolerance"]["speedup"] = 1.0
    baseline.write_text(json.dumps(written))
    capsys.readouterr()
    assert run.main([*ci, "--out", str(tmp_path / "ci1")]) == 0
    assert "regression mode: PASS" in capsys.readouterr().out


# --------------------------------------------------------------------- the method grids and the prompts


@pytest.mark.parametrize("model", method_grid.MODELS)
def test_every_method_of_a_grid_is_a_policy_the_scaffold_accepts(model):
    grid = method_grid.grid(model)
    names = [m.name for m in grid]
    assert grid[0] == DENSE and len(set(names)) == len(names) and [m.is_default for m in grid].count(True) == 1
    for method in grid[1:]:
        policy = step_cache_policy(method.step_cache, {"default": (1.0, 0.999, 0.999, 0.999)})
        plan = policy.reuse_plan(30)
        # every method skips some step, never the first or the last
        assert len(plan) == 30 and not plan[0] and not plan[-1] and any(plan), method.name
    # an interval with each forecast order, and MagCache's steps with a forecast
    assert sum("forecast" in m.step_cache for m in grid[1:]) == 12


def test_a_grid_reads_the_named_table_and_is_filtered_by_prefix():
    grid = method_grid.grid("wan22_t2v_a14b", mag_ratios="720p")
    mags = [m for m in grid if m.step_cache and m.step_cache["method"] == "magcache"]
    assert mags and all(m.step_cache["mag_ratios"] == "720p" for m in mags)
    assert method_grid.default_of(grid).name == "mag-t0.12-k4-r0.4"
    assert method_grid.default_of(grid, "int3+o1").name == "int3+o1"
    chosen = method_grid.select(grid, ["int3+o", "mag-t0.3-k6-r0.4+"])
    assert [m.name for m in chosen] == ["dense", "int3+o0", "int3+o1", "int3+o2", "mag-t0.3-k6-r0.4+o1"]
    assert method_grid.select(grid, None) == grid
    with pytest.raises(ValueError, match="matches none"):
        method_grid.select(grid, ["teacache"])
    with pytest.raises(ValueError, match="not one of"):
        method_grid.default_of(grid, "teacache")
    with pytest.raises(ValueError, match="unknown model"):
        method_grid.grid("ltx2_5")
    # the toy model takes TI2V-5B's grid
    assert method_grid.grid("tiny") == method_grid.grid("wan22_ti2v_5b")


def test_a_plans_cost_counts_computed_steps_and_branch_forwards():
    assert plan_cost(Interval(every=1).reuse_plan(10)) == (10, 20.0)
    # 6 steps, one in two computes: 0, 2, 4 and the last; each runs both guidance branches, or one unguided
    every2 = Interval(every=2, retention_ratio=0.0).reuse_plan(6)
    assert plan_cost(every2) == (4, 8.0)
    assert plan_cost(every2, guided=False) == (4, 4.0)


def test_the_prompts_are_fixed_and_start_with_the_video_benchmarks():
    assert len(PROMPTS) == 32 and len({p.text for p in PROMPTS}) == 32
    assert [p.text for p in PROMPTS[:8]] == VBENCH_PROMPTS[:8]
    assert all(p.index == i and p.seed == 1000 + i for i, p in enumerate(PROMPTS))
    assert len({p.dimension for p in PROMPTS}) == 8
    assert len(CI_PROMPTS) == 3 and set(CI_PROMPTS) <= set(PROMPTS) and take(5) == PROMPTS[:5]
    with pytest.raises(ValueError, match="--prompts"):
        take(33)


# --------------------------------------------------------------------- the served backend, without a server


def _server_backend(tmp_path, *argv: str) -> ServerBackend:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="wan22_ti2v_5b")
    parser.add_argument("--out", default=str(tmp_path))
    ServerBackend.add_arguments(parser)
    return ServerBackend.from_args(parser.parse_args(list(argv)))


def test_a_methods_serving_config_is_the_base_with_its_step_cache(tmp_path):
    yaml = pytest.importorskip("yaml")
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({
        "model": "wan22", "model_kwargs": {"compile_dit_sizes": ["832x480x33"], "step_cache": {"method": "none"}},
        "node_groups": [{"node_names": ["dit"], "ranks": [0]}],
    }))
    backend = _server_backend(tmp_path, "--base-config", str(base))
    method = Method("int3", "every third", {"method": "interval", "every": 3, "forecast": 1})
    written = yaml.safe_load(backend.write_config(method).read_text())
    assert written["model_kwargs"] == {"compile_dit_sizes": ["832x480x33"], "step_cache": method.step_cache}
    assert written["node_groups"] == [{"node_names": ["dit"], "ranks": [0]}]
    # dense carries no step cache, whatever the base said
    dense = yaml.safe_load(backend.write_config(DENSE).read_text())
    assert dense["model_kwargs"] == {"compile_dit_sizes": ["832x480x33"]} and dense["model"] == "wan22"
    with pytest.raises(ValueError, match="serves"):
        ServerBackend(model="ltx2_5", out_dir=tmp_path)


class _Videos(BaseHTTPRequestHandler):
    """A stand-in for the server's two routes; records each request's body."""

    bodies: list[dict] = []
    mp4 = b""

    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).bodies.append(body)
        payload = json.dumps({"data": [{"b64_json": base64.b64encode(type(self).mp4).decode()}]}).encode()
        self.send_response(200 if self.path == "/v1/videos/generations" else 404)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def test_a_running_server_is_used_as_it_is_warmed_once_per_shape(tmp_path):
    _Videos.bodies, _Videos.mp4 = [], _mp4(_clip(frames=6))
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Videos)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        backend = _server_backend(
            tmp_path, "--url", url, "--running", "int3", "--request-extra", '{"flow_shift": 12}',
        )
        method = Method("int3", "every third", {"method": "interval", "every": 3})
        # the harness starts nothing: the method asked for has to be the one that server runs
        with pytest.raises(RuntimeError, match="runs 'int3'"):
            backend.configure(DENSE)
        backend.configure(method)
        shape = {"size": "832x480", "frames": 17, "steps": 20}
        first = backend.generate("a prompt", 1003, **shape)
        second = backend.generate("another", 1004, **shape)
        backend.generate("a prompt", 1003, size="832x480", frames=33, steps=20)
        backend.close()
    finally:
        server.shutdown()
        server.server_close()
    assert first.frames.shape == (6, 32, 32, 3) and first.seconds > 0 and second.computed_steps is None
    # one untimed warm-up per shape, then the requests as sent
    assert [(b["prompt"], b["seed"], b["num_frames"]) for b in _Videos.bodies] == [
        (_Videos.bodies[0]["prompt"], 0, 17), ("a prompt", 1003, 17), ("another", 1004, 17),
        (_Videos.bodies[0]["prompt"], 0, 33), ("a prompt", 1003, 33),
    ]
    assert _Videos.bodies[1] == {
        "model": "wan22", "prompt": "a prompt", "negative_prompt": "", "size": "832x480", "seed": 1003,
        "num_inference_steps": 20, "num_frames": 17, "flow_shift": 12,
    }
    assert backend.describe()["url"] == url


def test_a_server_that_does_not_answer_is_an_error_not_a_hang(tmp_path):
    backend = _server_backend(tmp_path, "--url", "http://127.0.0.1:9", "--running", "dense")
    with pytest.raises(RuntimeError, match="no answer"):
        backend.configure(DENSE)
