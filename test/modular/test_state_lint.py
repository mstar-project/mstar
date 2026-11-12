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
    "mstar/model/z_image", "mstar/model/ltx2_5", "mstar/model/qwen3_omni", "mstar/model/minimax_h3",
    "mstar/model/kokoro", "mstar/model/omnilingual_asr", "mstar/model/omnivoice", "mstar/model/orpheus",
    "mstar/model/voxcpm2",
)
FORBIDDEN_ATTRIBUTES = {
    "node_resources", "resources", "request_state", "request_states", "per_request_states", "sequence_shard",
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
        "PiecewiseCallInputs", "PiecewiseCaptureShape", "PiecewisePackedConfig", "geometric_batch_sizes",
    )
} | {
    ("mstar.engine.engine", "ExecutingBatch"): "the annotation of `NodeSubmodule.can_batch`, a base class's hook",
    ("mstar.engine.resources.lora.checkpoint", "LoraFile"): "a model's static LoRA merge at load, which holds no state",
    ("mstar.engine.resources.lora", "LoraSpec"): "the opt-in LoRA spec, written by hand: the contract declares no LoRA",
    ("mstar.engine.resources", "SeqShardSpec"): (
        "MiniMax-H3's video decoder's sequence shard, written by hand: a contract SeqShard brackets a label an array "
        "opens, and the decoder has no arrays"
    ),
}
# Where an allowed import may stand: the one file whose text says why. Any other ported file importing it fails.
ALLOWED_ONLY_IN = {("mstar.engine.resources", "SeqShardSpec"): "mstar/model/minimax_h3/minimax_h3_model.py"}
# Methods that read the base class's `sequence_shard`, a resource a hand-written `SeqShardSpec` binds, by file and
# `Class.method`.
ALLOWED_SHARD_READS = {
    ("mstar/model/minimax_h3/submodules.py", "H3VideoDecoderSubmodule.declare_step"): (
        "the video decoder's hand-written sequence shard, which a contract SeqShard can't declare (no arrays)"
    ),
    ("mstar/model/minimax_h3/submodules.py", "H3VideoDecoderSubmodule.run_batch"): "the same hand-written shard",
}
# Graph breaks a ported package keeps, by file and function, since no state call sits behind them.
ALLOWED_BREAKS = {
    ("mstar/model/qwen3_omni/components/code2wav.py", "forward"): "the vocoder's transformer, which keeps no state",
    ("mstar/model/wan22/components/dit.py", "_token_runs"): (
        "the timestep runs of a rank's share of the tokens, read off the shard's range outside the DiT's guards"
    ),
}
# what a module defines to be handed the node's built resources, rather than its handles (`bind_state`)
RESOURCE_BINDS = {"bind_resources", "bind_node_resources"}


def _violations(source: str, path: str = "") -> tuple[list[str], set[tuple[str, str]]]:
    """What ``source``, the file at ``path``, does that the lint refuses, and the exceptions it used."""
    found, used = [], set()
    tree = ast.parse(source)
    allowed_breaks, shard_reads = set(), set()
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef):
            for method in cls.body:
                if isinstance(method, ast.FunctionDef) and (path, f"{cls.name}.{method.name}") in ALLOWED_SHARD_READS:
                    used.add((path, f"{cls.name}.{method.name}"))
                    shard_reads.update(range(method.lineno, method.end_lineno + 1))
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
                if key in ALLOWED and ALLOWED_ONLY_IN.get(key, path) == path:
                    used.add(key)
                else:
                    found.append(f"from {node.module} import {alias.name}")
        elif isinstance(node, ast.Import):
            found += [f"import {a.name}" for a in node.names if a.name.startswith("mstar.engine")]
        elif isinstance(node, ast.FunctionDef) and node.name in RESOURCE_BINDS:
            found.append(f"def {node.name} at line {node.lineno}, which takes the built resources")
        elif (
            isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES
            and not (node.attr == "sequence_shard" and node.lineno in shard_reads)
        ):
            found.append(f".{node.attr} at line {node.lineno}")
        elif (
            isinstance(node, ast.Attribute) and ast.unparse(node).endswith("compiler.disable")
            and node.lineno not in allowed_breaks
        ):
            found.append(f"torch.compiler.disable at line {node.lineno}, where the state calls hold the graph breaks")
    return found, used


def _sources() -> list[Path]:
    return [
        path for package in PORTED for path in sorted(Path(package).rglob("*.py")) if "tests" not in path.parts
    ]


@pytest.mark.parametrize("path", _sources(), ids=str)
def test_a_ported_package_reaches_the_engine_only_through_the_contract(path):
    found, _ = _violations(path.read_text(), str(path))
    assert not found, f"{path} reaches past the contract: {found}"


def test_every_exception_is_still_used():
    used = set()
    for path in _sources():
        used |= _violations(path.read_text(), str(path))[1]
    excepted = ALLOWED.keys() | ALLOWED_BREAKS.keys() | ALLOWED_SHARD_READS.keys()
    assert excepted <= used, f"delete these exceptions: {sorted(excepted - used)}"


@pytest.mark.parametrize("source, expected", [
    ("from mstar.engine.resources import KVStep", "from mstar.engine.resources import KVStep"),
    ("from mstar.engine.resources.sampler.resource import SamplerResource", "SamplerResource"),
    ("import mstar.engine.resources", "import mstar.engine.resources"),
    ("def f(e): return e.resources['kv']", ".resources"),
    ("def f(self): return self.node_resources['kv']", ".node_resources"),
    ("def f(self, rid): return self.request_state(rid)", ".request_state"),
    ("def f(self): return self.request_states", ".request_states"),
    ("def f(e): return e.per_request_states", ".per_request_states"),
    ("def f(self): return self.sampler.resource", ".resource"),
    ("def f(self): return self.kv.positions", ".positions"),
    ("def f(self): return self.kv._kv", "._kv"),
    ("def f(self): return self.sequence_shard.split_range('x')", ".sequence_shard"),
    ("def f(self): return self.kv.first_read", ".first_read"),
    ("def f(self): return self.kv.reads['attn']", ".reads"),
    ("@torch.compiler.disable\ndef f(): pass", "torch.compiler.disable"),
    ("class M:\n    def bind_resources(self, resources): pass", "def bind_resources"),
])
def test_the_lint_catches_each_way_past_the_contract(source, expected):
    found, _ = _violations(source)
    assert len(found) == 1 and expected in found[0], found


def test_an_allowed_import_stands_only_in_the_file_that_needs_it():
    source = "from mstar.engine.resources import SeqShardSpec"
    assert _violations(source, "mstar/model/minimax_h3/minimax_h3_model.py")[0] == []
    assert len(_violations(source, "mstar/model/wan22/wan22_model.py")[0]) == 1


def test_the_lint_passes_the_contract_itself():
    assert _violations("from mstar.engine.state import Array, sample") == ([], set())


def test_the_lint_lets_a_named_function_keep_its_break_and_no_other():
    source = "@torch.compiler.disable\ndef forward(): pass\n@torch.compiler.disable\ndef other(): pass\n"
    found, used = _violations(source, "mstar/model/qwen3_omni/components/code2wav.py")
    assert used == {("mstar/model/qwen3_omni/components/code2wav.py", "forward")}
    assert len(found) == 1 and "at line 3" in found[0]


def test_the_lint_reads_the_five_speech_packages():
    assert {path.parts[2] for path in _sources()} >= {"kokoro", "omnilingual_asr", "omnivoice", "orpheus", "voxcpm2"}


def test_the_lint_reads_all_five_reference_packages():
    assert {path.parts[2] for path in _sources()} >= {"wan22", "flux2_klein", "z_image", "ltx2_5", "qwen3_omni"}
