"""Load, validate and select from ``cases.yaml``: which models ``compare`` serves, what it sends per
modality, and how it judges each case."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT = Path(__file__).with_name("cases.yaml")
DRIVERS = ("suite", "client", "rp", "invariants")
SUITES = ("tts", "asr", "duplex")
CLIENT_KINDS = ("chat_text", "chat_image", "chat_audio", "chat_video", "chat_count", "image", "video", "video_i2v",
                "gen_clone", "pi05", "frames")
CHECKS = {
    "text_match": {"min_match"},
    "wer": {"max_increase"},
    "psnr": {"min_db"},
    "action_diff": {"tol"},
    "status_match": set(),
    "duplex": {"tol"},
    "audio_len": {"pct"},
}


class ManifestError(ValueError):
    pass


@dataclass
class Case:
    id: str
    modality: str
    driver: str            # suite | client | rp | invariants
    kind: str              # tts / asr / duplex, a client kind, whisper / omni, chat / speech
    check: dict[str, Any]
    perf: dict[str, Any] = field(default_factory=dict)
    conc: list[int] = field(default_factory=lambda: [1])
    repeats: int = 3
    warmup: int = 2
    n: int | None = None
    data: str | None = None
    args: list[str] = field(default_factory=list)
    optional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Model:
    name: str
    serve: str
    family: str
    tags: list[str]
    cases: list[Case]
    config: str | None = None
    requires: list[str] = field(default_factory=list)
    est_min: float = 10.0              # GPU minutes per arm: only orders the queue (longest first)
    startup_timeout_s: float = 1800.0
    env: dict[str, str] = field(default_factory=dict)
    serve_args: list[str] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    optional: bool = False             # a variant (Mooncake transport): only when named or picked by its own tag

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["cases"] = [c.to_dict() for c in self.cases]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Model:
        return cls(**{**d, "cases": [Case(**c) for c in d["cases"]]})


@dataclass
class Manifest:
    models: dict[str, Model]
    datasets: dict[str, str]
    defaults: dict[str, Any]

    def select(self, spec: str = "all", include_optional: bool = False) -> list[Model]:
        """``all``, comma-separated names, ``tag:<t>`` terms (any match), and ``-name`` / ``-tag:<t>`` exclusions.
        Optional cases (request path, invariants) are dropped unless ``include_optional``. Optional models are
        left out of ``all`` and of tags they share with regular models: name them, or use a tag only optional
        models carry (``tag:mooncake``)."""
        terms = [t.strip() for t in (spec or "all").split(",") if t.strip()]
        inc = [t for t in terms if not t.startswith("-")]
        exc = [t[1:] for t in terms if t.startswith("-")]
        unknown = [t for t in inc + exc if t != "all" and not t.startswith("tag:") and t not in self.models]
        if unknown:
            raise ManifestError(f"unknown model(s): {', '.join(unknown)} (known: {', '.join(self.models)})")
        shared = {t for m in self.models.values() if not m.optional for t in m.tags + [m.family.lower()]}
        out: list[Model] = []
        for m in self.models.values():
            if m.optional:
                if not any(t == m.name or (t.startswith("tag:") and t[4:] not in shared and _match(t, m)) for t in inc):
                    continue
            elif inc and "all" not in inc and not any(_match(t, m) for t in inc):
                continue
            if any(_match(t, m) for t in exc):
                continue
            out.append(m)
        if not include_optional:
            out = [_without_optional(m) for m in out]
        return [m for m in out if m.cases]


def _match(term: str, m: Model) -> bool:
    if term.startswith("tag:"):
        return term[4:] in m.tags or term[4:] == m.family.lower()
    return term == m.name


def _without_optional(m: Model) -> Model:
    m2 = copy.copy(m)
    m2.cases = [c for c in m.cases if not c.optional]
    return m2


def _req(d: dict, key: str, where: str, typ=None):
    if key not in d:
        raise ManifestError(f"{where}: missing '{key}'")
    v = d[key]
    if typ is not None and not isinstance(v, typ):
        raise ManifestError(f"{where}: '{key}' must be {typ.__name__}")
    return v


def _case(raw: dict, where: str, defaults: dict, model_conc: list[int], datasets: dict) -> Case:
    if not isinstance(raw, dict):
        raise ManifestError(f"{where}: a case must be a mapping")
    cid = _req(raw, "id", where, str)
    where = f"{where}.{cid}"
    drivers = [d for d in DRIVERS if d in raw]
    if len(drivers) != 1:
        raise ManifestError(f"{where}: exactly one of {', '.join(DRIVERS)} is required, got {drivers or 'none'}")
    driver = drivers[0]
    kind = raw[driver]
    allowed = {"suite": SUITES, "client": CLIENT_KINDS, "rp": ("whisper", "omni"), "invariants": ("chat", "speech")}
    if kind not in allowed[driver]:
        raise ManifestError(f"{where}: {driver} must be one of {', '.join(allowed[driver])}")
    check = _req(raw, "check", where, dict)
    ck = check.get("kind")
    if ck not in CHECKS:
        raise ManifestError(f"{where}: check.kind must be one of {', '.join(CHECKS)}")
    missing = CHECKS[ck] - set(check)
    if missing:
        raise ManifestError(f"{where}: check {ck} needs {', '.join(sorted(missing))}")
    data = raw.get("data")
    if driver == "suite" and not data:
        raise ManifestError(f"{where}: suite cases need 'data'")
    if data and data not in datasets:
        raise ManifestError(f"{where}: unknown dataset '{data}' (known: {', '.join(datasets)})")
    perf = raw.get("perf") or {}
    if driver in ("suite", "client") and not ("latency" in perf or "throughput" in perf):
        raise ManifestError(f"{where}: perf needs a latency or throughput metric")
    conc = raw.get("conc", model_conc)
    if not conc or not all(isinstance(c, int) and c > 0 for c in conc):
        raise ManifestError(f"{where}: conc must be a list of positive ints")
    return Case(id=cid, modality=str(raw.get("modality", cid)), driver=driver, kind=kind, check=check, perf=perf,
                conc=sorted(conc), repeats=int(raw.get("repeats", defaults.get("repeats", 3))),
                warmup=int(raw.get("warmup", defaults.get("warmup", 2))), n=raw.get("n"), data=data,
                args=[str(a) for a in raw.get("args", [])], optional=bool(raw.get("optional", False)))


def parse(raw: dict) -> Manifest:
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), dict):
        raise ManifestError("the manifest needs a top-level 'models' mapping")
    defaults = raw.get("defaults") or {}
    datasets = raw.get("datasets") or {}
    models: dict[str, Model] = {}
    for name, m in raw["models"].items():
        where = f"models.{name}"
        if not isinstance(m, dict):
            raise ManifestError(f"{where}: must be a mapping")
        conc = m.get("conc", defaults.get("conc", [1, 8]))
        raw_cases = _req(m, "cases", where, list)
        if not raw_cases:
            raise ManifestError(f"{where}: no cases")
        cases = [_case(c, f"{where}.cases[{i}]", defaults, conc, datasets) for i, c in enumerate(raw_cases)]
        ids = [c.id for c in cases]
        if len(set(ids)) != len(ids):
            raise ManifestError(f"{where}: duplicate case ids")
        models[name] = Model(
            name=name, serve=_req(m, "serve", where, str), family=str(m.get("family", "Other")),
            tags=[str(t) for t in m.get("tags", [])], cases=cases, config=m.get("config"),
            requires=[str(r) for r in m.get("requires", [])],
            est_min=float(m.get("est_min", defaults.get("est_min", 10))),
            startup_timeout_s=float(m.get("startup_timeout_s", defaults.get("startup_timeout_s", 1800))),
            env={**{k: str(v) for k, v in (defaults.get("env") or {}).items()},
                 **{k: str(v) for k, v in (m.get("env") or {}).items()}},
            serve_args=[str(a) for a in m.get("serve_args", [])],
            thresholds={**(defaults.get("thresholds") or {}), **(m.get("thresholds") or {})},
            optional=bool(m.get("optional", False)))
    return Manifest(models=models, datasets=datasets, defaults=defaults)


def load(path: str | Path = DEFAULT) -> Manifest:
    return parse(yaml.safe_load(Path(path).read_text()))
