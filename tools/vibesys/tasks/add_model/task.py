"""The ``add-model`` task type: port a new model into mstar under strict
numerical fidelity to an external reference (HF/official) implementation.

The oracle is the reference model in ``reference/reference.py`` (weights fetched
from ``reference/meta.json``); the checker exercises each Walk of mstar's
``get_graph_walk_graphs`` and compares to that oracle per a per-walk tolerance.
"""

from __future__ import annotations

import json
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from tools.vibesys.core import render
from tools.vibesys.core.runner import RunOptions, SynthesisInputs
from tools.vibesys.tasks.base import ImageSpec, TaskType

# Compare kinds understood by the generated checker (see templates/checker.py).
_COMPARE_KINDS = {"exact_tokens", "logit_kl", "cosine", "audio_mse", "ssim"}


@dataclass(frozen=True)
class Walk:
    name: str
    kind: str            # encoder | ar | generative (documentation / grouping)
    compare: str         # one of _COMPARE_KINDS
    tol: float = 0.0

    def validate(self) -> None:
        if self.compare not in _COMPARE_KINDS:
            raise ValueError(f"walk {self.name!r}: unknown compare {self.compare!r} (want {_COMPARE_KINDS})")


@dataclass(frozen=True)
class Spec:
    model: str
    hf_id: str
    revision: str | None
    extra: str                       # pip extra baked into the eval image: .[<extra>]
    reference_model: str             # mstar model to copy from, e.g. "qwen3_omni"
    served_model_name: str
    endpoint: str
    port: int
    modality: str
    headline_metric: str
    headline_walk: str
    result_arg: str
    walks: list[Walk]
    accuracy_mode: str = "smoke"     # smoke (valid output) | strict (numeric vs oracle)
    official_source: dict = field(default_factory=dict)
    accuracy_timeout: int = 900
    benchmark_timeout: int = 900


class AddModelTask(TaskType):
    name = "add-model"
    domain = "llm-serving"

    # ---- spec ---------------------------------------------------------------
    def load_spec(self, path: Path) -> Spec:
        raw = tomllib.loads(path.read_text())
        model = raw["model"]
        headline = raw.get("headline", {})
        walks = [Walk(**w) for w in raw.get("walk", [])]
        for w in walks:
            w.validate()
        if not walks:
            raise ValueError(f"{path}: at least one [[walk]] is required")
        spec = Spec(
            model=model["name"],
            hf_id=model["hf_id"],
            revision=model.get("revision"),
            extra=model.get("extra", model["name"]),
            reference_model=model.get("reference_model", "qwen3_omni"),
            served_model_name=model.get("served_model_name", model["name"]),
            endpoint=model.get("endpoint", "/v1/chat/completions"),
            port=int(model.get("port", 8000)),
            modality=model.get("modality", "text_generation"),
            headline_metric=headline.get("metric", "latency_p50_ms"),
            headline_walk=headline.get("walk", walks[-1].name),
            result_arg=headline.get("result_arg", "--output-json"),
            walks=walks,
            accuracy_mode=model.get("accuracy_mode", "smoke"),
            official_source=raw.get("official_source", {}),
            accuracy_timeout=int(raw.get("accuracy_timeout", 900)),
            benchmark_timeout=int(raw.get("benchmark_timeout", 900)),
        )
        return spec

    # ---- bundle (read-only evaluator) --------------------------------------
    def render_bundle(self, spec: Spec, bundle_dir: Path) -> None:
        t = self.dir / "templates"
        ctx = self._ctx(spec)

        render.render_to(t / "OBJECTIVE.md.tmpl", bundle_dir / "OBJECTIVE.md", ctx)

        # reference/ : oracle + weight manifest
        meta = {"model_id": spec.hf_id}
        if spec.revision:
            meta["revision"] = spec.revision
        if spec.official_source:
            meta["official_source"] = spec.official_source
        render.write(bundle_dir / "reference" / "meta.json", json.dumps(meta, indent=2) + "\n")

        # evaluator/ : its contents are copied to the workspace ROOT by vibesys.
        # Everything is namespaced under a single vibeval/ dir so it can never
        # collide with a top-level path in the mstar seed (e.g. mstar's own
        # benchmark/). checker.py, benchmark.py, and reference.py sit side by side.
        ev = bundle_dir / "evaluator" / "vibeval"
        render.render_to(t / "checker.py.tmpl", ev / "checker.py", ctx)
        render.render_to(t / "benchmark.py.tmpl", ev / "benchmark.py", ctx)
        ref_py = ev / "reference.py"
        render.render_to(t / "reference.py.tmpl", ref_py, ctx)
        override = self.dir / "instances" / f"{spec.model}.reference.py"
        if override.exists():
            shutil.copyfile(override, ref_py)

    # ---- seed (mutable worktree) -------------------------------------------
    def seed_files(self, spec: Spec, seed_dir: Path) -> None:
        t = self.dir / "templates"
        ctx = self._ctx(spec)
        render.render_to(t / "config.yaml.tmpl", seed_dir / "configs" / f"{spec.model}.yaml", ctx)
        run_sh = seed_dir / "run.sh"
        render.render_to(t / "run.sh.tmpl", run_sh, ctx)
        run_sh.chmod(0o755)

    # ---- vibesys wiring -----------------------------------------------------
    def synthesis_inputs(self, spec: Spec, bundle_dir: Path, seed_dir: Path) -> SynthesisInputs:
        return SynthesisInputs(
            domain=self.domain,
            objective_file=bundle_dir / "OBJECTIVE.md",
            accuracy_command=f"python vibeval/checker.py --url http://localhost:{spec.port} --{spec.accuracy_mode}",
            benchmark_command=(
                f"python vibeval/benchmark.py --url http://localhost:{spec.port} {spec.result_arg} result.json"
            ),
            reference_dir=bundle_dir / "reference",
            evaluator_dir=bundle_dir / "evaluator",
            workspace_seed_dir=seed_dir,
            benchmark_metric=spec.headline_metric,
            benchmark_result_arg=spec.result_arg,
            accuracy_timeout=spec.accuracy_timeout,
            benchmark_timeout=spec.benchmark_timeout,
        )

    def run_options(self, spec: Spec, exp_name: str, docker_image: str, **overrides) -> RunOptions:
        # Default skills: the mstar Walk-Graph porting skill, if present in-repo.
        skills = self.dir.parents[2] / "skills" / "add-mstar-model"
        config = self.dir.parents[1] / "agent.toml"  # tools/vibesys/agent.toml
        opts = RunOptions(
            exp_name=exp_name,
            docker_image=docker_image,
            modality=spec.modality,
            extra_skills=skills if skills.is_dir() else None,
            config=config if config.exists() else None,
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(opts, key, value)
        return opts

    def image(self, spec: Spec) -> ImageSpec:
        return ImageSpec(
            dockerfile=self.dir.parents[1] / "Dockerfile",
            build_args={"MSTAR_EXTRA": spec.extra},
        )

    # ---- helpers ------------------------------------------------------------
    def _ctx(self, spec: Spec) -> dict:
        walks_lit = json.dumps(
            [{"name": w.name, "kind": w.kind, "compare": w.compare, "tol": w.tol} for w in spec.walks],
            indent=4,
        )
        node_names = ", ".join(w.name for w in spec.walks)
        return {
            "model": spec.model,
            "hf_id": spec.hf_id,
            "revision": spec.revision or "",
            "reference_model": spec.reference_model,
            "served_model_name": spec.served_model_name,
            "endpoint": spec.endpoint,
            "port": spec.port,
            "modality": spec.modality,
            "headline_metric": spec.headline_metric,
            "headline_walk": spec.headline_walk,
            "result_arg": spec.result_arg,
            "walks_literal": walks_lit,
            "node_names": node_names,
            "walk_table": _walk_table(spec.walks),
        }


def _walk_table(walks: list[Walk]) -> str:
    rows = ["| Walk | Kind | Comparison | Tolerance |", "| --- | --- | --- | --- |"]
    for w in walks:
        tol = "exact" if w.compare == "exact_tokens" else str(w.tol)
        rows.append(f"| `{w.name}` | {w.kind} | {w.compare} | {tol} |")
    return "\n".join(rows)
