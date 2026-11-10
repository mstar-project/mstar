"""A ported package reaches the engine only through ``mstar.engine.state``.

The contract freezes that one surface, so a ported model that also imports a resource class, or reads a resource off the
engine's inputs, can break with no contract change, and later PRs can't move what it reached for. Nothing at run time
would notice, so this reads each package's source. The base classes are the bridge, so they aren't read. Each exception
says why it stands or which PR deletes it, and an exception nothing uses any more fails too, so the list only shrinks.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, ".")

import pytest

PORTED = (
    "mstar/model/qwen3_5", "mstar/model/command_a_plus", "mstar/model/wan22", "mstar/model/flux2_klein",
    "mstar/model/z_image", "mstar/model/ltx2_5", "mstar/model/qwen3_omni", "mstar/model/pi05",
    "mstar/model/vjepa2", "mstar/model/canary", "mstar/model/components/encoder_decoder.py",
)
FORBIDDEN_ATTRIBUTES = {
    "node_resources", "resources", "request_state", "per_request_states",
    # what a handle holds
    "resource", "positions", "reads", "first_read", "_kv", "_attn", "_pool", "_runners", "_views", "_blocks",
    "_bindings", "_layouts", "_update",
}

_CAPTURE = "a capture's buckets and compile mode, which the engine doesn't pick yet"
ALLOWED = {
    ("mstar.engine.accelerator_graph_config", name): _CAPTURE
    for name in (
        "AcceleratorGraphConfig", "BatchedAcceleratorGraphConfig", "DETERMINISTIC_COMPILE_MODE",
        "PackedAcceleratorGraphConfig", "PiecewiseAcceleratorGraphConfig", "PiecewiseBatchedConfig",
        "PiecewiseCallInputs", "PiecewiseCaptureShape", "geometric_batch_sizes",
    )
} | {
    ("mstar.engine.engine", "ExecutingBatch"): "the annotation of `NodeSubmodule.can_batch`, a base class's hook",
    ("mstar.engine.resources.lora.checkpoint", "LoraFile"): "a model's static LoRA merge at load, which holds no state",
}
# Graph breaks a ported package keeps, by file and function, since no state call sits behind them.
ALLOWED_BREAKS = {
    ("mstar/model/qwen3_omni/components/code2wav.py", "forward"): "the vocoder's transformer, which keeps no state",
}


def _violations(source: str, path: str = "") -> tuple[list[str], set[tuple[str, str]]]:
    """What ``source``, the file at ``path``, does that the lint refuses, and the exceptions it used."""
    found, used = [], set()
    tree = ast.parse(source)
    allowed_breaks = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and (path, node.name) in ALLOWED_BREAKS:
            used.add((path, node.name))
            allowed_breaks.update(decorator.lineno for decorator in node.decorator_list)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("mstar.engine"):
            if node.module == "mstar.engine.state":
                continue
            for alias in node.names:
                key = (node.module, alias.name)
                if key in ALLOWED:
                    used.add(key)
                else:
                    found.append(f"from {node.module} import {alias.name}")
        elif isinstance(node, ast.Import):
            found += [f"import {a.name}" for a in node.names if a.name.startswith("mstar.engine")]
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
            found.append(f".{node.attr} at line {node.lineno}")
        elif (
            isinstance(node, ast.Attribute) and ast.unparse(node).endswith("compiler.disable")
            and node.lineno not in allowed_breaks
        ):
            found.append(f"torch.compiler.disable at line {node.lineno}, where the state calls hold the graph breaks")
    return found, used


def _entry_sources(entry: str) -> list[Path]:
    path = Path(entry)
    return [path] if path.is_file() else [p for p in sorted(path.rglob("*.py")) if "tests" not in p.parts]


def _sources() -> list[Path]:
    return [path for entry in PORTED for path in _entry_sources(entry)]


@pytest.mark.parametrize("path", _sources(), ids=str)
def test_a_ported_package_reaches_the_engine_only_through_the_contract(path):
    found, _ = _violations(path.read_text(), str(path))
    assert not found, f"{path} reaches past the contract: {found}"


@pytest.mark.parametrize("entry", PORTED)
def test_every_ported_entry_yields_a_source(entry):
    assert _entry_sources(entry), f"{entry} yields no source, so the lint would pass it unread"


def test_every_exception_is_still_used():
    used = set()
    for path in _sources():
        used |= _violations(path.read_text(), str(path))[1]
    excepted = ALLOWED.keys() | ALLOWED_BREAKS.keys()
    assert excepted <= used, f"delete these exceptions: {sorted(excepted - used)}"


@pytest.mark.parametrize("source, expected", [
    ("from mstar.engine.resources import KVStep", "from mstar.engine.resources import KVStep"),
    ("from mstar.engine.resources.sampler.resource import SamplerResource", "SamplerResource"),
    ("import mstar.engine.resources", "import mstar.engine.resources"),
    ("def f(e): return e.resources['kv']", ".resources"),
    ("def f(self): return self.node_resources['kv']", ".node_resources"),
    ("def f(self, rid): return self.request_state(rid)", ".request_state"),
    ("def f(e): return e.per_request_states", ".per_request_states"),
    ("def f(self): return self.sampler.resource", ".resource"),
    ("def f(self): return self.kv.positions", ".positions"),
    ("def f(self): return self.kv._kv", "._kv"),
    ("def f(self): return self.kv.first_read", ".first_read"),
    ("def f(self): return self.kv.reads['attn']", ".reads"),
    ("@torch.compiler.disable\ndef f(): pass", "torch.compiler.disable"),
])
def test_the_lint_catches_each_way_past_the_contract(source, expected):
    found, _ = _violations(source)
    assert len(found) == 1 and expected in found[0], found


def test_the_lint_passes_the_contract_itself():
    assert _violations("from mstar.engine.state import Array, sample") == ([], set())


def test_the_lint_lets_a_named_function_keep_its_break_and_no_other():
    source = "@torch.compiler.disable\ndef forward(): pass\n@torch.compiler.disable\ndef other(): pass\n"
    found, used = _violations(source, "mstar/model/qwen3_omni/components/code2wav.py")
    assert used == {("mstar/model/qwen3_omni/components/code2wav.py", "forward")}
    assert len(found) == 1 and "at line 3" in found[0]


def test_the_lint_reads_all_five_reference_packages():
    assert {path.parts[2] for path in _sources()} >= {"wan22", "flux2_klein", "z_image", "ltx2_5", "qwen3_omni"}
