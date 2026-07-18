"""The existing graph test suite (test/modular/test_graph.py), re-run with
the Rust walk core engaged — the validation the graph layer's adoption is
gated on. Each scenario (pipeline, fixed-iteration loop, dynamic finish,
nested loops, EOS ready-signal clearing) runs under every ``MSTAR_RUST_WALK``
mode by wrapping the ``WorkerGraphIO`` the tests construct:

* ``shadow`` — Python stays authoritative; the test additionally FAILS on
  any logged divergence, so a silent Rust-side mismatch cannot pass.
* ``1`` — Rust decisions (ready set, doneness, loop indices) drive the
  scenarios; the suite's own assertions validate the behavior.

Skipped unless the ``mstar_rust`` extension is installed."""

import importlib.util
import logging
import sys
from pathlib import Path

import pytest

pytest.importorskip("mstar_rust")

from mstar.graph.graph_io import WorkerGraphIO
from mstar.graph.rust_core import wrap_worker_graph_io

_SPEC = importlib.util.spec_from_file_location(
    "modular_test_graph",
    Path(__file__).resolve().parents[1] / "modular" / "test_graph.py")
_tg = importlib.util.module_from_spec(_SPEC)
sys.modules["modular_test_graph"] = _tg
_SPEC.loader.exec_module(_tg)

SCENARIOS = [
    _tg.test_simple_pipeline,
    _tg.test_diffusion_loop_fixed_iters,
    _tg.test_ar_generation_with_dynamic_finish,
    _tg.test_nested_loops,
    _tg.test_eos_clears_ready_signals,
]
MODES = ["shadow", "1", "pure"]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("mode", MODES)
def test_graph_suite_under_rust_walk(mode, scenario, monkeypatch, caplog):
    monkeypatch.setenv("MSTAR_RUST_WALK", mode)

    def wrapped(graph):
        return wrap_worker_graph_io(WorkerGraphIO(graph), graph, wg_id=0)

    monkeypatch.setattr(_tg, "WorkerGraphIO", wrapped)
    with caplog.at_level(logging.ERROR, logger="mstar.graph.rust_core"):
        scenario()
    diverged = [r.message for r in caplog.records
                if "diverg" in r.message.lower() or "fell back" in r.message.lower()]
    assert not diverged, diverged
