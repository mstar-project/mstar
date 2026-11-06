"""The optimization loop's result rows, keep rule, best-so-far line and brief handling.

On a shared host two servers of the same commit differ by a few percent, so a round is kept only when
every candidate round beats every baseline round, not when the means merely differ. Rows are typed
by an agent, so a row with a missing key, a quoted number or a time without its offset is refused
when it is recorded, where it would otherwise break every later read of the file.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, ".")

from benchmark.vibesys import progress, rows
from benchmark.vibesys.cli import main

_CELL = rows.Cell("Qwen3.5-0.8B", "ShareGPT", 1, 64)
_LOOP = Path("benchmark/vibesys/loop.sh").resolve()


def _row(
    role,
    value,
    hypothesis="fused input stage",
    status="probe",
    workload="ShareGPT",
    started="10:00",
    p50=30.0,
    failed=0,
):
    return rows.Row(
        batch="rp2-loop" if status == "probe" else "h200_1106",
        family="Language",
        model="Qwen3.5-0.8B",
        workload=workload,
        gpus=1,
        engine="M* run-base" if role == "baseline" else "M* rp2",
        role=role,
        config="qwen3_5_0.8b.yaml",
        c=64,
        round=1,
        throughput=value,
        unit="tok/s",
        latency="TTFT",
        p50_ms=p50,
        p99_ms=2 * p50,
        requests=640,
        failed=failed,
        status=status,
        started=f"2026-11-06T{started}-05:00",
        error="",
        commit="abc1234",
        hypothesis=hypothesis,
    )


def _round(base, cand, **kw):
    return [_row("baseline", v, **kw) for v in base] + [_row("candidate", v, **kw) for v in cand]


def _at(clock):
    return datetime.fromisoformat(f"2026-11-06T{clock}-05:00")


def _report(tmp_path, **changes):
    results = tmp_path / "results.jsonl"
    obj = json.loads(_row("candidate", 5321.4).to_json()) | changes
    return main(["report", "--results", str(results), json.dumps(obj)]), results


def test_report_refuses_a_row_with_a_missing_key(tmp_path):
    results = tmp_path / "results.jsonl"
    assert main(["report", "--results", str(results), '{"throughput": 1.0}']) == 2
    assert not results.exists(), "a refused row must not reach the results file"


def test_report_refuses_a_quoted_number(tmp_path):
    code, results = _report(tmp_path, c="64")
    assert code == 2 and not results.exists(), "c='64' would never match a cell with c=64"


def test_report_refuses_a_start_time_without_its_offset(tmp_path):
    code, results = _report(tmp_path, started="2026-11-06T10:00")
    assert code == 2 and not results.exists(), "a naive time can't be ordered against the other rows"


def test_report_appends_a_row_that_load_reads_back(tmp_path, capsys):
    code, results = _report(tmp_path)
    assert code == 0
    assert rows.load(results) == [_row("candidate", 5321.4)]
    assert capsys.readouterr().out.strip() == results.read_text().strip(), "report prints the row it wrote"


def test_load_names_the_line_that_is_not_a_row(tmp_path):
    results = tmp_path / "results.jsonl"
    results.write_text(_row("baseline", 1.0).to_json() + "\n--help\n")
    with pytest.raises(ValueError, match=re.escape(f"{results}:2:")):
        rows.load(results)


def test_keep_takes_a_round_whose_every_candidate_round_is_ahead():
    decision = rows.keep(_round([100, 101, 102], [103, 104, 105]), _CELL)
    assert decision.kept, decision.reason


def test_keep_drops_a_round_whose_rounds_overlap():
    decision = rows.keep(_round([100, 101, 106], [103, 104, 105]), _CELL)
    assert not decision.kept, "a higher mean with overlapping rounds is within host noise"


def test_failed_rounds_do_not_count_toward_the_three():
    data = _round([100, 101, 102], [103, 104]) + [_row("candidate", 200, failed=3)]
    decision = rows.keep(data, _CELL)
    assert not decision.kept, "a round with failed requests counted as one of the candidate's three"


def test_a_regression_on_another_cell_drops_the_round():
    data = _round([100, 101, 102], [103, 104, 105]) + _round([50, 51, 52], [40, 41, 42], workload="Random 1024/256")
    decision = rows.keep(data, _CELL)
    assert not decision.kept and "Random 1024/256" in decision.reason, "the regression elsewhere was missed"


def test_overlap_on_another_cell_still_keeps_the_round():
    data = _round([100, 101, 102], [103, 104, 105]) + _round([50, 51, 52], [49, 51, 53], workload="Random 1024/256")
    assert rows.keep(data, _CELL).kept, "noise on another cell is not a regression"


def test_a_lower_p50_counts_as_ahead():
    data = [_row("baseline", 1, p50=30.0) for _ in range(3)] + [_row("candidate", 1, p50=20.0) for _ in range(3)]
    assert rows.keep(data, _CELL, metric="p50_ms").kept, "a lower p50 should count as ahead"


def test_keep_command_decides_only_the_named_round(tmp_path):
    results = tmp_path / "results.jsonl"
    data = _round([100, 101, 102], [103, 104, 105], hypothesis="a") + _round(
        [100, 101, 102], [90, 91, 92], hypothesis="b"
    )
    results.write_text("".join(r.to_json() + "\n" for r in data))
    cell = ["--model", "Qwen3.5-0.8B", "--workload", "ShareGPT", "--c", "64"]
    assert main(["keep", str(results), "--hypothesis", "a", *cell]) == 0
    assert main(["keep", str(results), "--hypothesis", "b", *cell]) == 1, "round b's rows decided round a"


def test_table_has_one_line_per_cell_with_both_arms(capsys, tmp_path):
    results = tmp_path / "results.jsonl"
    data = _round([100, 102], [110, 112], status="counted") + [_row("baseline", 50, status="counted", workload="Image")]
    results.write_text("".join(r.to_json() + "\n" for r in data))
    main(["table", str(results), "--batch", "h200_1106"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3 and "| 1.099 | better |" in lines[2], "a cell measured on one arm has no ratio to show"


def test_kept_steps_scale_to_end_on_the_checkpoint():
    found = [
        progress.Round("a", _at("10:00"), 1.10, True),
        progress.Round("b", _at("11:00"), 1.20, False),
        progress.Round("c", _at("12:00"), 1.05, True),
    ]
    line = progress.levels(found, [progress.Checkpoint(_at("13:00"), 1.30, 100.0)])
    assert math.isclose(line[-1], 1.30), "the line should meet the checkpoint's measured ratio"
    assert line[2] == line[1], "a round not kept should leave the line where it was"


def test_each_checkpoint_ends_its_own_segment():
    found = [progress.Round(name, _at(clock), 1.10, True) for name, clock in (("a", "10:00"), ("b", "12:00"))]
    sweeps = [progress.Checkpoint(_at("11:00"), 1.20, 100.0), progress.Checkpoint(_at("13:00"), 1.50, 100.0)]
    line = progress.levels(found, sweeps)
    assert math.isclose(line[1], 1.20) and math.isclose(line[2], 1.50), "a checkpoint was met in the wrong place"


def test_a_checkpoint_below_the_line_pulls_its_segment_down():
    found = [progress.Round("a", _at("10:00"), 1.10, True), progress.Round("b", _at("12:00"), 1.05, True)]
    sweeps = [progress.Checkpoint(_at("11:00"), 1.10, 100.0), progress.Checkpoint(_at("13:00"), 1.08, 100.0)]
    assert math.isclose(progress.levels(found, sweeps)[-1], 1.08), "the line missed the lower checkpoint"


def test_rounds_and_checkpoints_are_read_by_batch_and_hypothesis():
    data = (
        _round([100, 101, 102], [110, 111, 112], hypothesis="a", started="10:00")
        + _round([110, 111, 112], [108, 109, 110], hypothesis="b", started="11:00")
        + _round([100, 100, 100], [115, 115, 115], hypothesis="checkpoint", status="counted", started="12:00")
    )
    found = progress.rounds(data, _CELL)
    assert [(r.hypothesis, r.kept) for r in found] == [("a", True), ("b", False)]
    sweeps = progress.checkpoints(data, _CELL)
    assert len(sweeps) == 1 and math.isclose(sweeps[0].ratio, 1.15), "the counted rows are one checkpoint"


_STUB = """#!/bin/bash
n=$(( $(cat "$STATE/count" 2>/dev/null || echo 0) + 1 )); echo $n >"$STATE/count"
printf '%s\\0' "$@" >"$STATE/args"
touch "$STATE/STOP"
echo "{\\"session_id\\": \\"s$n\\", \\"is_error\\": ${STUB_ERROR:-false}, \\"num_turns\\": 1, \\"result\\": \\"ok\\"}"
"""


@pytest.mark.skipif(sys.platform != "linux", reason="loop.sh uses GNU coreutils")
class TestLoopBrief:
    """loop.sh against a stub `claude` that records its arguments and stops the loop after one call."""

    def _call(self, tmp_path, brief_text, error=False):
        state, bin_dir, wt = tmp_path / "state", tmp_path / "bin", tmp_path / "wt"
        for d in (state, bin_dir, wt):
            d.mkdir(exist_ok=True)
        (state / "settings.json").write_text("{}")
        (state / "STOP").unlink(missing_ok=True)
        (bin_dir / "claude").write_text(_STUB)
        (bin_dir / "claude").chmod(0o755)
        brief = tmp_path / "brief.md"
        brief.write_text(brief_text)
        env = os.environ | {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "STATE": str(state),
            "STUB_ERROR": "true" if error else "false",
        }
        subprocess.run(
            ["bash", str(_LOOP), str(brief), str(wt), str(state)], env=env, check=True, capture_output=True, timeout=30
        )
        return (state / "args").read_text().split("\0")[:-1], state

    def test_the_first_call_sends_the_brief(self, tmp_path):
        args, state = self._call(tmp_path, "optimize it")
        assert "--resume" not in args and args[-1] == "optimize it"
        assert (state / "brief.sent").read_text() == "optimize it", "the sent brief wasn't recorded"

    def test_an_unchanged_brief_resumes_its_session(self, tmp_path):
        self._call(tmp_path, "optimize it")
        args, _ = self._call(tmp_path, "optimize it")
        assert args[args.index("--resume") + 1] == "s1", "the loop didn't resume the session the brief started"

    def test_an_appended_brief_resumes_with_the_addition_in_the_prompt(self, tmp_path):
        self._call(tmp_path, "optimize it")
        args, _ = self._call(tmp_path, "optimize it\nand profile first")
        assert "--resume" in args and args[-1].endswith("and profile first"), "the addition wasn't sent"

    def test_a_replaced_brief_starts_fresh_and_keeps_the_old_one(self, tmp_path):
        self._call(tmp_path, "optimize it")
        args, state = self._call(tmp_path, "something else")
        assert "--resume" not in args and args[-1] == "something else"
        assert [p.read_text() for p in state.glob("brief.prev-*")] == ["optimize it"], "the old brief was lost"

    def test_a_call_that_starts_a_session_and_fails_still_holds_the_brief(self, tmp_path):
        self._call(tmp_path, "optimize it", error=True)
        args, _ = self._call(tmp_path, "optimize it")
        assert "--resume" in args, "a failed first call made the next one send the brief again"
