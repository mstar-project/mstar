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
    "mstar/model/canary_qwen", "mstar/model/qwen3_asr", "mstar/model/parakeet", "mstar/model/fish_s2_pro",
    "mstar/model/qwen3_tts",
    "mstar/model/pi05", "mstar/model/vjepa2", "mstar/model/canary", "mstar/model/components/encoder_decoder.py",
    "mstar/model/bagel", "mstar/model/chatterbox", "mstar/model/whisper",
)
FORBIDDEN_ATTRIBUTES = {
    "node_resources", "resources", "request_state", "request_states", "per_request_states", "sequence_shard",
    # what a handle holds
    "resource", "positions", "reads", "first_read", "_kv", "_attn", "_cross", "_pool", "_runners", "_views", "_blocks",
    "_bindings", "_layouts", "_update",
}

_CAPTURE = "a capture's buckets and compile mode, which the engine doesn't pick yet"
ALLOWED = {
    ("mstar.engine.accelerator_graph_config", name): _CAPTURE
    for name in (
        "AcceleratorGraphConfig", "BatchedAcceleratorGraphConfig", "DETERMINISTIC_COMPILE_MODE",
        "PackedAcceleratorGraphConfig", "PiecewiseAcceleratorGraphConfig", "PiecewiseBatchedConfig",
        "PiecewiseCallInputs", "PiecewiseCaptureShape", "PiecewisePackedConfig", "decode_rows_then_prompt",
        "geometric_batch_sizes",
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
# Exceptions for one file, by file, module and name, which a later port of the same kind mustn't inherit.
ALLOWED_IN: dict[tuple[str, str, str], str] = {
    ("mstar/model/bagel/bagel_model.py", "mstar.engine.resources", "KVReqConfig"):
        "per-node and publish labels under CFG parallel and P/D, until P/D labels",
} | {
    ("mstar/model/chatterbox/components/s3gen_graphs.py", module, name):
        "S3Gen's own captures per shape, which hold no state"
    for module, name in (
        ("mstar.engine.accelerator_graph_backend", "CapturedGraph"),
        ("mstar.engine.accelerator_graph_runner", "capture_into_graph"),
    )
}
# Exceptions each scoped to one function and one attribute, by file, the function's qualified name in that file, and
# the attribute: a graph break (`disable`) or a forbidden read the function may keep.
ALLOWED_IN_FUNCTION = {
    ("mstar/model/minimax_h3/submodules.py", "H3VideoDecoderSubmodule.declare_step", "sequence_shard"):
        "the video decoder's hand-written sequence shard, which a contract SeqShard can't declare (no arrays)",
    ("mstar/model/minimax_h3/submodules.py", "H3VideoDecoderSubmodule.run_batch", "sequence_shard"):
        "the same hand-written shard",
    ("mstar/model/qwen3_omni/components/code2wav.py", "Qwen3OmniMoeCode2WavTransformerModel.forward", "disable"):
        "the vocoder's transformer, which keeps no state",
    ("mstar/model/qwen3_tts/components/depth_frame/frame.py", "DepthFrame.run", "disable"):
        "the depth kernel's Gluon launch, which a compiled decode can't trace; no state call sits behind it",
    ("mstar/model/qwen3_tts/submodules.py", "CodecSubmodule.batch_postprocess", "request_state"):
        "the opt-in watermark's per-stream state, until it moves to a watermark node's plain Value (issue #1447)",
    ("mstar/model/wan22/components/dit.py", "_token_runs", "disable"):
        "the timestep runs of a rank's share of the tokens, read off the shard's range outside the DiT's guards",
    ("mstar/model/bagel/components/vit_encoder.py", "BagelViTAttention.attend", "disable"):
        "the ViT's eager varlen attention, flash-attn or a masked SDPA, which keeps no state",
    ("mstar/model/bagel/submodules.py", "ViTEncoderSubmodule._replay_block_loop", "disable"):
        "the ViT region's replay, which runs its captured graph; no state call sits behind it",
}
# what a module defines to be handed the node's built resources, rather than its handles (`bind_state`)
RESOURCE_BINDS = {"bind_resources", "bind_node_resources"}


def _in_functions(node: ast.AST, prefix: str = "", function: str | None = None):
    """Each node under ``node``, with the qualified name of the function it lies in, decorators included, or None."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            name = prefix + child.name
            yield child, name
            yield from _in_functions(child, f"{name}.<locals>.", name)
        elif isinstance(child, ast.ClassDef):
            yield child, function
            yield from _in_functions(child, f"{prefix}{child.name}.", function)
        else:
            yield child, function
            yield from _in_functions(child, prefix, function)


def _violations(source: str, path: str = "") -> tuple[list[str], set[tuple[str, ...]]]:
    """What ``source``, the file at ``path``, does that the lint refuses, and the exceptions it used."""
    found, used = [], set()
    for node, function in _in_functions(ast.parse(source)):
        breaks = isinstance(node, ast.Attribute) and ast.unparse(node).endswith("compiler.disable")
        if (breaks or isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES) and (
            (path, function, node.attr) in ALLOWED_IN_FUNCTION
        ):
            used.add((path, function, node.attr))
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("mstar.engine"):
            if node.module == "mstar.engine.state":
                continue
            for alias in node.names:
                key = (node.module, alias.name)
                if key in ALLOWED and ALLOWED_ONLY_IN.get(key, path) == path:
                    used.add(key)
                elif (path, *key) in ALLOWED_IN:
                    used.add((path, *key))
                else:
                    found.append(f"from {node.module} import {alias.name}")
        elif isinstance(node, ast.Import):
            found += [f"import {a.name}" for a in node.names if a.name.startswith("mstar.engine")]
        elif isinstance(node, ast.FunctionDef) and node.name in RESOURCE_BINDS:
            found.append(f"def {node.name} at line {node.lineno}, which takes the built resources")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
            found.append(f".{node.attr} at line {node.lineno}")
        elif breaks:
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
    excepted = ALLOWED.keys() | ALLOWED_IN.keys() | ALLOWED_IN_FUNCTION.keys()
    assert excepted <= used, f"delete these exceptions: {sorted(excepted - used)}"


def test_an_exception_for_one_file_refuses_the_import_in_another(monkeypatch):
    source = "from mstar.engine.resources import KVReqConfig"
    entry = ("mstar/model/a/a.py", "mstar.engine.resources", "KVReqConfig")
    monkeypatch.setitem(ALLOWED_IN, entry, "a's own")
    assert _violations(source, "mstar/model/a/a.py") == ([], {entry})
    assert _violations(source, "mstar/model/b/b.py")[0] == [source], "the exception would reach every ported package"


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
    source = (
        "class Qwen3OmniMoeCode2WavTransformerModel:\n"
        "    @torch.compiler.disable\n    def forward(self): pass\n"
        "    @torch.compiler.disable\n    def other(self): pass\n"
    )
    found, used = _violations(source, "mstar/model/qwen3_omni/components/code2wav.py")
    assert used == {("mstar/model/qwen3_omni/components/code2wav.py", "Qwen3OmniMoeCode2WavTransformerModel.forward",
                     "disable")}
    assert len(found) == 1 and "at line 4" in found[0]


_TTS = "mstar/model/qwen3_tts/submodules.py"
_WATERMARK = (_TTS, "CodecSubmodule.batch_postprocess", "request_state")


@pytest.mark.parametrize("path, source, found, used", [
    (_TTS, "class CodecSubmodule:\n    def batch_postprocess(self, rid):\n        return self.request_state(rid)\n",
     [], {_WATERMARK}),
    (_TTS, "class CodecSubmodule:\n    def postprocess(self, rid):\n        return self.request_state(rid)\n",
     [".request_state at line 3"], set()),
    (_TTS, "class TalkerSubmodule:\n    def batch_postprocess(self, rid):\n        return self.request_state(rid)\n",
     [".request_state at line 3"], set()),
    (_TTS, "class CodecSubmodule:\n    def batch_postprocess(self, rid):\n        return rid\n", [], set()),
    (_TTS, "class CodecSubmodule:\n    def batch_postprocess(self, rid):\n        self.request_state(rid)\n"
     "        return self.node_resources\n", [".node_resources at line 4"], {_WATERMARK}),
    ("mstar/model/qwen3_omni/components/code2wav.py",
     "class Qwen3OmniMoeCode2WavTransformerModel:\n    def forward(self): pass\n", [], set()),
], ids=["the hook", "another function", "another class's hook", "the hook without it", "the hook reading more",
        "an undecorated forward"])
def test_an_exception_covers_only_its_own_functions_attribute(path, source, found, used):
    """A key skips only its attribute, anywhere in its function, and is used only when it skips one."""
    assert _violations(source, path) == (found, used)


def test_the_lint_reads_the_five_speech_packages():
    assert {path.parts[2] for path in _sources()} >= {"kokoro", "omnilingual_asr", "omnivoice", "orpheus", "voxcpm2"}


def test_the_lint_reads_all_five_reference_packages():
    assert {path.parts[2] for path in _sources()} >= {"wan22", "flux2_klein", "z_image", "ltx2_5", "qwen3_omni"}
