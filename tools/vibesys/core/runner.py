"""Build and exec the installed ``vibesys`` CLI via its synthesis flags.

We drive VibeSys in *synthesis* mode (``--input-*`` flags) rather than passing a
full on-disk bundle. Synthesis mode is the path built for the pip-installed CLI,
and ``--input-workspace-seed`` copies a *directory* (so the worktree's
uncommitted starter files are carried into the candidate workspace), which a
git-pinned ``[workspace] sources`` bundle cannot do.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SynthesisInputs:
    """The ``--input-*`` pieces a task renders for one run."""

    domain: str
    objective_file: Path
    accuracy_command: str          # shlex-split by vibesys
    benchmark_command: str
    reference_dir: Path | None = None
    evaluator_dir: Path | None = None
    workspace_seed_dir: Path | None = None
    benchmark_metric: str | None = None
    benchmark_result_arg: str | None = None
    accuracy_timeout: int | None = None
    benchmark_timeout: int | None = None


@dataclass
class RunOptions:
    exp_name: str
    backend: str = "cuda"
    docker: bool = True
    docker_image: str | None = None
    local: bool = True
    rounds: int | None = None
    modality: str | None = None
    extra_skills: Path | None = None
    cli_provider: str = "claude"
    config: Path | None = None
    runs_dir: Path | None = None
    extra_env: dict = field(default_factory=dict)
    extra_flags: list[str] = field(default_factory=list)


def build_argv(inputs: SynthesisInputs, opts: RunOptions) -> list[str]:
    argv = [
        "vibesys",
        "--input-domain", inputs.domain,
        "--input-objective-file", str(inputs.objective_file),
        "--input-accuracy-command", inputs.accuracy_command,
        "--input-benchmark-command", inputs.benchmark_command,
        "--exp-name", opts.exp_name,
        "--backend", opts.backend,
        "--agent-backend", "cli",
        "--cli-provider", opts.cli_provider,
    ]
    if opts.config is not None:
        argv += ["--config", str(opts.config)]
    if opts.runs_dir is not None:
        argv += ["--runs-dir", str(opts.runs_dir)]
    if inputs.reference_dir is not None:
        argv += ["--input-reference", str(inputs.reference_dir)]
    if inputs.evaluator_dir is not None:
        argv += ["--input-evaluator-dir", str(inputs.evaluator_dir)]
    if inputs.workspace_seed_dir is not None:
        argv += ["--input-workspace-seed", str(inputs.workspace_seed_dir)]
    if inputs.benchmark_metric and inputs.benchmark_result_arg:
        # The result arg is itself an option string (e.g. "--output-json"), so it
        # must be attached with '=' or argparse treats it as the next flag.
        argv += [
            "--input-benchmark-metric", inputs.benchmark_metric,
            f"--input-benchmark-result-arg={inputs.benchmark_result_arg}",
        ]
    if inputs.accuracy_timeout is not None:
        argv += ["--input-accuracy-timeout", str(inputs.accuracy_timeout)]
    if inputs.benchmark_timeout is not None:
        argv += ["--input-benchmark-timeout", str(inputs.benchmark_timeout)]
    if opts.local:
        argv.append("--local")
    if opts.docker:
        argv.append("--docker")
        if opts.docker_image:
            argv += ["--docker-image", opts.docker_image]
    if opts.rounds is not None:
        argv += ["--max-rounds", str(opts.rounds)]
    if opts.modality:
        argv += ["--modality", opts.modality]
    if opts.extra_skills is not None:
        argv += ["--extra-skills", str(opts.extra_skills)]
    argv += opts.extra_flags
    return argv


def exec_vibesys(argv: list[str], *, dry_run: bool = False, extra_env: dict | None = None) -> int:
    print("+ " + " ".join(_quote(a) for a in argv))
    if dry_run:
        return 0
    if shutil.which("vibesys") is None:
        raise SystemExit(
            "vibesys not found on PATH. Activate the env that has it "
            "(e.g. `conda activate mstar`) or `pip install vibesys`."
        )
    # Strip exported bash functions (BASH_FUNC_module/scl/ml/... from HPC
    # module systems). Non-bash `/bin/sh -c` children the agent spawns cannot
    # parse them and spew "error importing function definition" on every command.
    env = {k: v for k, v in os.environ.items() if not k.startswith("BASH_FUNC_")}
    env.update(extra_env or {})
    return subprocess.call(argv, env=env)


def _quote(arg: str) -> str:
    return f'"{arg}"' if " " in arg else arg
