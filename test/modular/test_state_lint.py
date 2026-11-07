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

PORTED = ("mstar/model/qwen3_5", "mstar/model/command_a_plus")
FORBIDDEN_ATTRIBUTES = {
    "node_resources", "resources", "request_state", "per_request_states",
    # what a handle holds
    "resource", "positions", "reads", "first_read", "_kv", "_attn", "_pool", "_linear", "_views", "_blocks",
}

_CAPTURE = "a capture's buckets and compile mode, which the engine doesn't pick yet"
ALLOWED = {
    ("mstar.engine.accelerator_graph_config", name): _CAPTURE
    for name in ("AcceleratorGraphConfig", "BatchedAcceleratorGraphConfig", "DETERMINISTIC_COMPILE_MODE")
}


def _violations(source: str) -> tuple[list[str], set[tuple[str, str]]]:
    """What ``source`` does that the lint refuses, and the exceptions it used."""
    found, used = [], set()
    for node in ast.walk(ast.parse(source)):
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
        elif isinstance(node, ast.Attribute) and ast.unparse(node).endswith("compiler.disable"):
            found.append(f"torch.compiler.disable at line {node.lineno}, where the state calls hold the graph breaks")
    return found, used


def _sources() -> list[Path]:
    return [
        path for package in PORTED for path in sorted(Path(package).rglob("*.py")) if "tests" not in path.parts
    ]


@pytest.mark.parametrize("path", _sources(), ids=str)
def test_a_ported_package_reaches_the_engine_only_through_the_contract(path):
    found, _ = _violations(path.read_text())
    assert not found, f"{path} reaches past the contract: {found}"


def test_every_exception_is_still_used():
    used = set()
    for path in _sources():
        used |= _violations(path.read_text())[1]
    assert ALLOWED.keys() <= used, f"delete these from ALLOWED: {sorted(ALLOWED.keys() - used)}"


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
