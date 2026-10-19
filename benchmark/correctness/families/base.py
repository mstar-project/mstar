"""What every model family implements: requests, saving, reference and scoring."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.correctness.client import Request, Response
from benchmark.correctness.report import FamilyScore
from benchmark.correctness.sets import SampleSet


class Family:
    """One model family's correctness check.

    ``opts`` are the --family-opt key=value pairs, as strings.

    """

    name = "base"
    set_name = ""
    default_restarts = 1
    default_concurrency = 1
    # Per-request timeout in seconds. Long generations and first compiles need room.
    request_timeout_s = 1800.0
    # No reference exists for these. The verdict is same-seed agreement between two server instances
    # plus the family's own checks, and the report says so.
    self_consistency_only = False

    def __init__(self, model_key: str, cfg: dict, opts: dict[str, str] | None = None):
        self.model_key = model_key
        self.cfg = cfg
        self.opts = dict(opts or {})

    # --- serving ---------------------------------------------------------------------------
    def select_items(self, items: list) -> list:
        """The items of the set this family runs. Default: all of them."""
        return items

    def requests(self, sset: SampleSet) -> list[Request]:
        raise NotImplementedError

    def save(self, sset: SampleSet, responses: list[Response], out_dir: Path) -> None:
        """Write each item's output to ``out_dir/<id>.<ext>``."""
        raise NotImplementedError

    # --- reference ---------------------------------------------------------------------------
    def reference_id(self) -> str:
        """Folder name under <data_root>/references/ for this implementation and its settings."""
        raise NotImplementedError

    def reference_description(self) -> str:
        return self.reference_id()

    def make_reference(self, sset: SampleSet, ref_dir: Path, device: str) -> None:
        """Write reference outputs for every item to ``ref_dir/outputs`` (eager torch)."""
        raise NotImplementedError(f"{self.name}: no reference implementation available")

    # --- scoring -----------------------------------------------------------------------------
    def score(self, sset: SampleSet, runs: list[Path], ref_dir: Path | None, work: Path,
              device: str) -> FamilyScore:
        raise NotImplementedError

    def protocol(self) -> str:
        if self.self_consistency_only:
            return (f"SELF-CONSISTENCY ONLY. There's no reference for this config ({self.reference_description()}), "
                    "so the verdict is same-seed agreement across server instances plus the family checks")
        return f"reference comparison against {self.reference_description()}"

    # --- helpers -----------------------------------------------------------------------------
    def opt(self, key: str, default=None, cast=str):
        return cast(self.opts[key]) if key in self.opts else default

    @staticmethod
    def ref_outputs(ref_dir: Path | None) -> Path | None:
        if ref_dir is None:
            return None
        d = ref_dir / "outputs"
        return d if d.is_dir() and (ref_dir / "meta.json").is_file() else None


class MathSdpaReference:
    """--family-opt ref_sdpa=math builds a second reference with torch's math SDPA. Scoring the
    default reference then also reports how far the reference moves from itself when only its
    attention kernel changes, which is the spread a served output should be judged against.

    For image-to-video, --family-opt ref_anchor=fp32 builds one whose conditioning frame is encoded in
    fp32. The served encode lands elsewhere inside the bf16 encode error and some videos amplify that,
    but the math-SDPA arm encodes the frame exactly like the reference and two server instances agree
    with each other, so neither shows it."""

    MATH_SUFFIX = "-sdpa-math"
    ANCHOR_SUFFIX = "-anchor-fp32"

    @property
    def ref_sdpa_math(self) -> bool:
        return self.opt("ref_sdpa") == "math"

    @property
    def ref_anchor_fp32(self) -> bool:
        return self.opt("ref_anchor") == "fp32"

    def arm_suffix(self) -> str:
        return (self.MATH_SUFFIX if self.ref_sdpa_math else "") + (self.ANCHOR_SUFFIX if self.ref_anchor_fp32 else "")

    def sdpa_context(self):
        import contextlib

        if not self.ref_sdpa_math:
            return contextlib.nullcontext()
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel(SDPBackend.MATH)

    CALIBRATION_DB = 3.0

    def calibrated_verdict(self, sc, spread: list[float], served_median: float | None,
                           basis: str = "math-SDPA spread") -> None:
        """ROCm verdict next to the raw gate: the served median has to reach the reference's own spread
        (median) minus CALIBRATION_DB. Recorded as a metric, so the raw verdict stays the CUDA gate."""
        import statistics

        if not spread or served_median is None:
            return
        spread_median = statistics.median(spread)
        floor = spread_median - self.CALIBRATION_DB
        verdict = "PASS" if served_median >= floor else "FAIL"
        sc.metrics["rocm_calibrated"] = {"spread_median_db": round(spread_median, 2), "floor_db": round(floor, 2),
                                         "served_median_db": round(served_median, 2), "verdict": verdict,
                                         "basis": basis}
        sc.notes.append(f"Calibrated ROCm verdict {verdict}: served median {served_median:.2f} dB against a floor of "
                        f"{floor:.2f} dB ({self.CALIBRATION_DB:g} dB under the reference's {basis}, median "
                        f"{spread_median:.2f} dB). The raw verdict above is the CUDA-derived gate.")

    def math_reference_dir(self, ref_dir: Path | None) -> Path | None:
        """The math-SDPA sibling of the default reference, when it was built."""
        return self._arm_dir(ref_dir, self.MATH_SUFFIX)

    def anchor_reference_dir(self, ref_dir: Path | None) -> Path | None:
        """The fp32-anchor sibling of the default reference, when it was built."""
        return self._arm_dir(ref_dir, self.ANCHOR_SUFFIX)

    def _arm_dir(self, ref_dir: Path | None, suffix: str) -> Path | None:
        if ref_dir is None or self.arm_suffix():
            return None
        d = ref_dir.parent.parent / (ref_dir.parent.name + suffix) / ref_dir.name
        return d if (d / "meta.json").is_file() else None


def write_meta(ref_dir: Path, **meta) -> None:
    import datetime
    import platform

    info = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "host": platform.node(), **meta}
    try:
        import torch

        info.update(torch=torch.__version__, hip=getattr(torch.version, "hip", None))
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    for mod in ("transformers", "diffusers"):
        try:
            info[mod] = __import__(mod).__version__
        except ImportError:
            pass
    (ref_dir / "meta.json").write_text(json.dumps(info, indent=1, default=str))


def self_consistency_check(sc, values: list[float], name: str, n_runs: int, pass_if, warn_if, rule: str) -> None:
    """Repeatability check for families without a reference. WARN if it couldn't be measured."""
    from benchmark.correctness.report import band

    if n_runs < 2:
        sc.check(name, None, "WARN", "needs --restarts 2 or more, one instance only shows sanity")
    elif not values:
        sc.check(name, None, "ERROR", "no item produced comparable outputs in two instances")
    else:
        worst = min(values) if pass_if(float("inf")) else max(values)
        sc.check(name, worst, band(worst, pass_if, warn_if), rule)


def failed_items(run_dir: Path) -> dict[str, str]:
    """Map of item id to error for the items the server failed."""
    out = {}
    path = run_dir / "responses.jsonl"
    if path.is_file():
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            if not rec.get("ok"):
                out[rec["id"]] = rec.get("error") or "failed"
    return out
