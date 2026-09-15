"""The ``add-model`` task type: implement a new model natively in mstar and
verify it against the reference (HF/official) implementation.

The agent must *implement* the model (submodule nodes, walk graph, engine
resources, weight loading), not wrap the upstream pipeline. The task is
described by the client-facing input/output *modalities* (walks are an mstar
implementation detail the agent chooses). The checker and benchmark are
modality-specific
(``templates/modalities/<modality>/``) and compare the client-facing output to
the reference oracle in ``reference.py`` (weights from ``reference/meta.json``).
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

# Output-level comparison metrics understood by the modality checkers.
_COMPARE_KINDS = {"exact_tokens", "logit_kl", "cosine", "audio_mse", "ssim"}


@dataclass(frozen=True)
class Spec:
    model: str
    hf_id: str
    revision: str | None
    reference_model: str | None      # optional mstar structural example
    served_model_name: str
    endpoint: str                    # client-facing OpenAI route
    port: int
    modality: str                    # selects the eval templates + vibesys --modality
    input_modalities: list[str]      # what the client sends (e.g. ["text"])
    output_modalities: list[str]     # what the client gets (e.g. ["video"])
    accuracy_compare: str            # output-level metric vs the reference oracle
    accuracy_tol: float
    accuracy_mode: str               # strict (fidelity vs oracle) | smoke (validity only)
    headline_metric: str
    result_arg: str
    official_source: dict = field(default_factory=dict)
    accuracy_timeout: int = 1200
    benchmark_timeout: int = 1200


class AddModelTask(TaskType):
    name = "add-model"
    domain = "llm-serving"

    # ---- spec ---------------------------------------------------------------
    def load_spec(self, path: Path) -> Spec:
        raw = tomllib.loads(path.read_text())
        model = raw["model"]
        acc = raw.get("accuracy", {})
        headline = raw.get("headline", {})
        compare = acc.get("compare", "ssim")
        if compare not in _COMPARE_KINDS:
            raise ValueError(f"{path}: accuracy.compare={compare!r} not in {_COMPARE_KINDS}")
        return Spec(
            model=model["name"],
            hf_id=model["hf_id"],
            revision=model.get("revision") or None,
            reference_model=model.get("reference_model") or None,
            served_model_name=model.get("served_model_name", model["name"]),
            endpoint=model["endpoint"],
            port=int(model.get("port", 8000)),
            modality=model["modality"],
            input_modalities=list(model.get("input_modalities", ["text"])),
            output_modalities=list(model.get("output_modalities", ["video"])),
            accuracy_compare=compare,
            accuracy_tol=float(acc.get("tol", 0.0)),
            accuracy_mode=model.get("accuracy_mode", "strict"),
            headline_metric=headline.get("metric", "latency_p50_ms"),
            result_arg=headline.get("result_arg", "--output-json"),
            official_source=raw.get("official_source", {}),
            accuracy_timeout=int(raw.get("accuracy_timeout", 1200)),
            benchmark_timeout=int(raw.get("benchmark_timeout", 1200)),
        )

    # ---- bundle (read-only evaluator) --------------------------------------
    def render_bundle(self, spec: Spec, bundle_dir: Path) -> None:
        t = self.dir / "templates"
        ctx = self._ctx(spec)

        render.render_to(t / "OBJECTIVE.md.tmpl", bundle_dir / "OBJECTIVE.md", ctx)

        meta = {"model_id": spec.hf_id}
        if spec.revision:
            meta["revision"] = spec.revision
        if spec.official_source:
            meta["official_source"] = spec.official_source
        render.write(bundle_dir / "reference" / "meta.json", json.dumps(meta, indent=2) + "\n")

        # evaluator/vibeval/ is copied to the workspace ROOT by vibesys (a single
        # dir so it never collides with a top-level path in the mstar seed). The
        # checker/benchmark are MODALITY-specific; model specifics live in reference.py.
        ev = bundle_dir / "evaluator" / "vibeval"
        mod = t / "modalities" / spec.modality
        if not (mod / "checker.py.tmpl").exists():
            raise ValueError(
                f"no eval templates for modality {spec.modality!r}; add "
                f"tools/vibesys/tasks/add_model/templates/modalities/{spec.modality}/"
            )
        render.render_to(mod / "checker.py.tmpl", ev / "checker.py", ctx)
        render.render_to(mod / "benchmark.py.tmpl", ev / "benchmark.py", ctx)
        render.render_to(t / "serve_and_eval.sh.tmpl", ev / "serve_and_eval.sh", ctx)
        ref_py = ev / "reference.py"
        render.render_to(t / "reference.py.tmpl", ref_py, ctx)
        override = self.dir / "instances" / f"{spec.model}.reference.py"
        if override.exists():
            shutil.copyfile(override, ref_py)

    # ---- seed (mutable candidate) ------------------------------------------
    def seed_files(self, spec: Spec, seed_dir: Path) -> None:
        t = self.dir / "templates"
        ctx = self._ctx(spec)
        render.render_to(t / "config.yaml.tmpl", seed_dir / "configs" / f"{spec.model}.yaml", ctx)
        render.render_to(
            t / "PORTING_REPORT.md.tmpl",
            seed_dir / "progress-artifacts" / "evidence" / "model-port-report.md",
            ctx,
        )
        render.render_to(
            t / "PORTING_DECISIONS.jsonl.tmpl",
            seed_dir / "progress-artifacts" / "evidence" / "model-port-decisions.jsonl",
            ctx,
        )
        run_sh = seed_dir / "run.sh"
        render.render_to(t / "run.sh.tmpl", run_sh, ctx)
        run_sh.chmod(0o755)

    # ---- vibesys wiring -----------------------------------------------------
    def synthesis_inputs(self, spec: Spec, bundle_dir: Path, seed_dir: Path) -> SynthesisInputs:
        # serve_and_eval.sh brings the server up on the fixed port + waits for
        # /health, then runs the gate (nothing else starts the server for the
        # framework-owned official commands); uv gives a self-contained Python.
        return SynthesisInputs(
            domain=self.domain,
            objective_file=bundle_dir / "OBJECTIVE.md",
            # The checker runs in the candidate's own env (uv --extra <model>) so
            # the strict path can import + run the reference oracle (torch, the
            # upstream package, video decode); smoke works there too.
            accuracy_command=(
                f"bash vibeval/serve_and_eval.sh "
                f"uv run --extra {spec.model} --with httpx --with numpy python vibeval/checker.py "
                f"--url http://localhost:{spec.port} --{spec.accuracy_mode}"
            ),
            benchmark_command=(
                f"bash vibeval/serve_and_eval.sh "
                f"uv run --no-project --with httpx --with numpy python vibeval/benchmark.py "
                f"--url http://localhost:{spec.port} {spec.result_arg} result.json"
            ),
            reference_dir=bundle_dir / "reference",
            evaluator_dir=bundle_dir / "evaluator",
            workspace_seed_dir=seed_dir,
            # Keep the benchmark command and its result JSON as an observational
            # baseline. The add-model MVP does not register latency as a
            # synthesis objective; a later optimization task can consume the
            # unchanged artifact format.
            benchmark_metric=None,
            benchmark_result_arg=None,
            accuracy_timeout=spec.accuracy_timeout,
            benchmark_timeout=spec.benchmark_timeout,
        )

    def run_options(self, spec: Spec, exp_name: str, docker_image: str, **overrides) -> RunOptions:
        skills = self.dir.parents[3] / ".claude" / "skills" / "add-mstar-model"
        if not skills.is_dir():
            raise FileNotFoundError(
                f"required add-model skill directory is missing: {skills}"
            )
        config = self.dir.parents[1] / "agent.toml"  # tools/vibesys/agent.toml
        opts = RunOptions(
            exp_name=exp_name,
            docker_image=docker_image,
            modality=spec.modality,
            extra_skills=skills,
            config=config if config.exists() else None,
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(opts, key, value)
        return opts

    def image(self, spec: Spec) -> ImageSpec:
        # Minimal CUDA+uv env; mstar and its deps resolve at run time from the
        # candidate's own pyproject via `uv run` (see Dockerfile).
        return ImageSpec(dockerfile=self.dir.parents[1] / "Dockerfile", build_args={})

    # ---- helpers ------------------------------------------------------------
    def _ctx(self, spec: Spec) -> dict:
        if spec.reference_model:
            reference_guidance = (
                f"Inspect `mstar/model/{spec.reference_model}/` as a structural example and "
                "borrow only graph and resource patterns whose invariants match. It is not a "
                "required taxonomy bucket. If it does not fit, derive the model directly from "
                "its stages, Walks, tensor dependencies, persistent state, and the checked-out "
                "engine contracts."
            )
        else:
            reference_guidance = (
                "No reference model is preselected. Derive the target graph and resource "
                "requirements first, consult the model shape map, and record every considered "
                "reference in `model-port-decisions.jsonl`. Select references only where their "
                "graph and resource invariants match."
            )
        return {
            "model": spec.model,
            "hf_id": spec.hf_id,
            "revision": spec.revision or "",
            "reference_guidance": reference_guidance,
            "served_model_name": spec.served_model_name,
            "endpoint": spec.endpoint,
            "port": spec.port,
            "modality": spec.modality,
            "input_modalities": ", ".join(spec.input_modalities),
            "output_modalities": ", ".join(spec.output_modalities),
            "compare": spec.accuracy_compare,
            "tol": spec.accuracy_tol,
            "headline_metric": spec.headline_metric,
            "result_arg": spec.result_arg,
        }
