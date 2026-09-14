"""Focused contract tests for the VibeSys add-model task."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from tools.vibesys.core.runner import build_argv
from tools.vibesys.tasks.add_model.task import AddModelTask

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / ".claude" / "skills" / "add-mstar-model"
REMOVED_APIS = (
    "EngineType",
    "get_node_engine_types",
    "get_kv_cache_config",
    "cache_manager",
)


def _task_and_spec():
    task = AddModelTask()
    return task, task.load_spec(task.instance_path("example"))


def test_skill_frontmatter_and_local_reference_links():
    skill = (SKILL_DIR / "SKILL.md").read_text()
    assert skill.startswith("---\nname: add-mstar-model\ndescription:")

    links = set(re.findall(r"\[[^]]+\]\((references/[^)]+)\)", skill))
    assert links == {
        "references/cuda-graphs.md",
        "references/decision-log.md",
        "references/engine-and-serving.md",
        "references/model-contract.md",
        "references/model-shape-map.md",
        "references/resource-pools.md",
    }
    assert all((SKILL_DIR / link).is_file() for link in links)


def test_run_wires_required_skill_directory():
    task, spec = _task_and_spec()
    opts = task.run_options(spec, "test-exp", "test-image")
    inputs = task.synthesis_inputs(spec, Path("bundle"), Path("seed"))
    argv = build_argv(inputs, opts)

    assert opts.extra_skills == SKILL_DIR
    skill_arg = argv.index("--extra-skills")
    assert Path(argv[skill_arg + 1]) == SKILL_DIR


def test_run_fails_when_skill_directory_is_missing(tmp_path, monkeypatch):
    fake_task_dir = tmp_path / "tools" / "vibesys" / "tasks" / "add_model"
    monkeypatch.setattr(
        AddModelTask,
        "dir",
        property(lambda self: fake_task_dir),
    )

    task = AddModelTask()
    spec = task.load_spec(
        REPO_ROOT / "tools" / "vibesys" / "tasks" / "add_model"
        / "instances" / "example.toml"
    )
    with pytest.raises(FileNotFoundError, match="required add-model skill"):
        task.run_options(spec, "test-exp", "test-image")


def test_rendered_objective_and_port_report_use_resource_contract(tmp_path):
    task, spec = _task_and_spec()
    assert spec.reference_model is None
    bundle = tmp_path / "bundle"
    seed = tmp_path / "seed"
    task.render_bundle(spec, bundle)
    task.seed_files(spec, seed)

    objective = (bundle / "OBJECTIVE.md").read_text()
    report = (
        seed / "progress-artifacts" / "evidence" / "model-port-report.md"
    ).read_text()
    decision_log = (
        seed / "progress-artifacts" / "evidence" / "model-port-decisions.jsonl"
    )
    normalized_objective = " ".join(objective.split())
    normalized_report = " ".join(report.split())

    for current_api in (
        "get_node_resources",
        "get_request_resource_configs",
        "declare_step",
        "engine_inputs.resources",
    ):
        assert current_api in objective
    assert "eager" in objective.lower()
    assert "model-port-report.md" in objective
    assert "model-port-decisions.jsonl" in objective
    assert all(symbol not in objective for symbol in REMOVED_APIS)
    assert "Model Port Report: `mymodel`" in report
    assert "resident capacity" in report
    assert "Direction Status" in report
    assert "Deferred Performance Opportunities" in report
    assert "No reference model is preselected" in normalized_objective
    assert "consult the model shape map" in normalized_objective
    assert "mstar/model//" not in objective
    assert "Never use model-owned allocation" in normalized_objective
    assert "No matching reference" in normalized_objective
    assert "attempted engine mapping" in normalized_report
    assert "Never use model-owned resource lifecycle" in normalized_report
    assert "Strategy Signals" in report

    events = [json.loads(line) for line in decision_log.read_text().splitlines()]
    assert events == [
        {
            "schema_version": 1,
            "seq": 0,
            "phase": "design",
            "event": "log_initialized",
            "subject": "mymodel",
            "options": [],
            "decision": "pending",
            "evidence": [],
            "reason": "Decision log seeded by the add-model task.",
            "skill_signal": "covered",
            "skill_source": "OBJECTIVE.md",
            "skill_note": "",
            "supersedes": None,
        }
    ]

    referenced_bundle = tmp_path / "referenced-bundle"
    task.render_bundle(
        replace(spec, reference_model="vjepa2"),
        referenced_bundle,
    )
    referenced_objective = " ".join(
        (referenced_bundle / "OBJECTIVE.md").read_text().split()
    )
    assert "mstar/model/vjepa2/" in referenced_objective
    assert "not a required taxonomy bucket" in referenced_objective


def test_benchmark_remains_observational_and_emits_compatible_json(tmp_path):
    task, spec = _task_and_spec()
    bundle = tmp_path / "bundle"
    task.render_bundle(spec, bundle)
    inputs = task.synthesis_inputs(spec, bundle, tmp_path / "seed")
    opts = task.run_options(spec, "test-exp", "test-image")
    argv = build_argv(inputs, opts)

    assert inputs.benchmark_command
    assert f"{spec.result_arg} result.json" in inputs.benchmark_command
    assert "--input-benchmark-metric" not in argv
    assert not any(arg.startswith("--input-benchmark-result-arg") for arg in argv)

    evaluator = bundle / "evaluator" / "vibeval"
    (evaluator / "reference.py").write_text(
        "def sample_input():\n"
        "    return {}\n\n"
        "def to_request(value):\n"
        "    return 'POST', '/generate', value\n"
    )
    (evaluator / "httpx.py").write_text(
        "class Response:\n"
        "    def raise_for_status(self):\n"
        "        return None\n\n"
        "class Client:\n"
        "    def __init__(self, timeout):\n"
        "        self.timeout = timeout\n\n"
        "    def request(self, method, url, json):\n"
        "        return Response()\n"
    )

    result_path = tmp_path / "result.json"
    proc = subprocess.run(
        [
            sys.executable,
            "benchmark.py",
            "--url",
            "http://benchmark.invalid",
            "--num-requests",
            "2",
            "--warmup-requests",
            "1",
            spec.result_arg,
            str(result_path),
        ],
        cwd=evaluator,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )

    result = json.loads(result_path.read_text())
    assert "PERF_METRIC:" in proc.stdout
    assert result["num_requests"] == 2
    assert {
        spec.headline_metric,
        "latency_p50_ms",
        "latency_p99_ms",
        "latency_mean_ms",
        "request_throughput",
        "num_requests",
    } <= result.keys()
