"""A ported package reaches the engine only through ``mstar.engine.state``.

The contract freezes that one surface, so a ported model that also imports a resource class, or reads a resource off the
engine's inputs, can break with no contract change, and later PRs can't move what it reached for. Nothing at run time
would notice, so this reads each package's source. The base classes are the bridge, so they aren't read. Each exception
says why it stands or which PR deletes it, and an exception nothing uses any more fails too, so the list only shrinks.

A package that isn't ported has to be on the frozen list of those that predate this rule, so a new model arrives on
the contract. ``MODEL_REGISTRY`` loads a model only from a listed package, and from one module per unported package,
so a model can't sit where that rule doesn't look. The adapters that bind a declaration to a resource are imported only
by the engine outside the state package, the tests and the files listed with a reason, and ``contract.py`` imports
nothing new from the resources, so the adapters can be replaced without touching a model and importing
``mstar.engine.state`` never loads them.
"""

from __future__ import annotations

import ast
import sys
from collections import Counter
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
)
# The list only shrinks: a package leaves it when its port lands, and nothing is added to it.
UNPORTED = (
    "mstar/model/bagel", "mstar/model/chatterbox", "mstar/model/cosmos3", "mstar/model/cosyvoice3",
    "mstar/model/glm52", "mstar/model/glm5_next", "mstar/model/higgs_audio", "mstar/model/lingbot_world",
    "mstar/model/minicpm_o", "mstar/model/muse", "mstar/model/nemotron_duplex", "mstar/model/phi4mm",
    "mstar/model/ripple", "mstar/model/waypoint", "mstar/model/whisper",
)
# shared code under mstar/model/ that holds no model
SHARED = ("mstar/model/components", "mstar/model/loader")
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
}
# what a module defines to be handed the node's built resources, rather than its handles (`bind_state`)
RESOURCE_BINDS = {"bind_resources", "bind_node_resources"}
_LORA = "the opt-in LoRA spec, which the contract doesn't declare"
# The ported files that still write a spec by hand in `get_node_resources`, each with the one call its override
# returns: a listed file can't return a second spec, which a shared module the lint doesn't read could build.
ALLOWED_NODE_RESOURCES = {
    "mstar/model/qwen3_5/qwen3_5_model.py": ("qwen3_5_lora_spec", _LORA),
    "mstar/model/qwen3_omni/qwen3_omni_model.py": ("self._lora_spec", _LORA),
    "mstar/model/orpheus/orpheus_model.py": ("self._lora_spec", _LORA),
    "mstar/model/qwen3_tts/qwen3_tts_model.py": ("self._lora_spec", _LORA),
    "mstar/model/minimax_h3/minimax_h3_model.py": (
        "SeqShardSpec",
        "the video decoder's sequence shard, which a contract SeqShard can't declare since the decoder has no arrays",
    ),
}
ADAPTERS = "mstar.engine.state.adapters"
_BY_HAND = "the script builds a node's resources and handles by hand, as a test does"
# the files that import the adapters from outside mstar/engine/ and the tests, each with its reason
ADAPTER_IMPORTERS = {
    "mstar/model/base.py": "the model base class resolves a deployment's resource specs from its declarations",
    "mstar/model/submodule_base.py": "`engine_step` expands a node's declared step for the engine",
    "mstar/api_server/data_worker.py": "the data worker converts a prompt's state layout into today's KV layouts",
    "mstar/conductor/conductor.py": "the conductor converts a request's `SampleSettings` into sampler configs",
    "mstar/validation/__init__.py": "`validate_deployment` checks a deployment's state setup before a worker spawns",
    "benchmark/command_a_plus/validate_checkpoint.py": _BY_HAND,
    "benchmark/lora/module_parity.py": _BY_HAND,
    "benchmark/ltx2_5/profile_step.py": _BY_HAND,
}
# the source directories, apart from test/
SOURCE_ROOTS = ("mstar", "mstar_sim", "benchmark", "examples", "scripts")
CONTRACT = "mstar/engine/state/contract.py"
# every name the contract imports from `mstar.engine.resources`
CONTRACT_RESOURCE_IMPORTS = {
    ("mstar.engine.resources.attn.config", "SP_EXCHANGES"),
    ("mstar.engine.resources.kv.config", "Span"),
    ("mstar.engine.resources.step", "Segment"),
}


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


def _returns_one(function: ast.FunctionDef, call: str) -> bool:
    """Whether every return of ``function`` is a list of one call to ``call``."""
    returns = [node.value for node in ast.walk(function) if isinstance(node, ast.Return)]
    return bool(returns) and all(
        isinstance(value, ast.List) and len(value.elts) == 1 and isinstance(value.elts[0], ast.Call)
        and ast.unparse(value.elts[0].func) == call
        for value in returns
    )


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
                else:
                    found.append(f"from {node.module} import {alias.name}")
        elif isinstance(node, ast.Import):
            found += [f"import {a.name}" for a in node.names if a.name.startswith("mstar.engine")]
        elif isinstance(node, ast.FunctionDef) and node.name in RESOURCE_BINDS:
            found.append(f"def {node.name} at line {node.lineno}, which takes the built resources")
        elif isinstance(node, ast.FunctionDef) and node.name == "get_node_resources":
            call = ALLOWED_NODE_RESOURCES.get(path, ("",))[0]
            if call and _returns_one(node, call):
                used.add(path)
            elif call:
                found.append(f"def get_node_resources at line {node.lineno}, which returns more than its one `{call}`")
            else:
                found.append(f"def get_node_resources at line {node.lineno}, where `declare_state` declares the state")
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


def _model_packages() -> list[str]:
    """The directories under mstar/model/ that hold Python source, apart from the shared ones and an editor's dot
    directories (`.ipynb_checkpoints`). A `__pycache__`, or a package another branch left behind, holds none."""
    return sorted(
        str(path) for path in Path("mstar/model").iterdir()
        if path.is_dir() and not path.name.startswith(".") and str(path) not in SHARED and any(path.rglob("*.py"))
    )


def _registered_modules() -> dict[str, str]:
    """The entry PORTED or UNPORTED would name for each module `MODEL_REGISTRY` names. Each string in registry.py
    that starts with ``mstar.`` is such a module, so an entry counts however it is added."""
    tree = ast.parse(Path("mstar/model/registry.py").read_text())
    return {
        node.value: "/".join(node.value.split(".")[:3]) for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("mstar.")
    }


def _imports_adapters(source: str, path: str) -> bool:
    # Most files never name the adapters, and parsing all of them takes seconds.
    if "adapters" not in source:
        return False
    package = Path(path).parent.parts
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            module = node.module
            if node.level:
                # one dot is the file's own package, and each further dot goes up one
                base = ".".join(package[:len(package) + 1 - node.level])
                module = f"{base}.{node.module}" if node.module else base
            names = {alias.name for alias in node.names}
            if module == ADAPTERS or (module == "mstar.engine.state" and "adapters" in names):
                return True
        elif isinstance(node, ast.Import) and ADAPTERS in {alias.name for alias in node.names}:
            return True
    return False


def _adapter_importers() -> set[str]:
    """The files that import the adapters, apart from those in mstar/engine/ and the tests."""
    return {
        str(path) for root in SOURCE_ROOTS for path in Path(root).rglob("*.py")
        if path.parts[:2] != ("mstar", "engine") and "tests" not in path.parts
        and _imports_adapters(path.read_text(), str(path))
    }


def _resource_imports(source: str) -> set[tuple[str, str]]:
    """The module and name of each import ``source`` makes from `mstar.engine.resources`."""
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            found |= {
                (node.module, alias.name) for alias in node.names
                if f"{node.module}.{alias.name}".startswith("mstar.engine.resources")
            }
        elif isinstance(node, ast.Import):
            found |= {(alias.name, "") for alias in node.names if alias.name.startswith("mstar.engine.resources")}
    return found


@pytest.mark.parametrize("path", _sources(), ids=str)
def test_a_ported_package_reaches_the_engine_only_through_the_contract(path):
    found, _ = _violations(path.read_text(), str(path))
    assert not found, (
        f"{path} reaches past the contract: {found}. Reach the engine through `mstar.engine.state`. An import a model "
        "can't avoid goes in ALLOWED with its reason, and state the contract can't declare yet needs its mask or op "
        "and binding in adapters.py first, not a spec returned from `get_node_resources`."
    )


@pytest.mark.parametrize("entry", PORTED)
def test_every_ported_entry_yields_a_source(entry):
    assert _entry_sources(entry), f"{entry} yields no source, so the lint would pass it unread"


def test_every_exception_is_still_used():
    used = _adapter_importers() | _resource_imports(Path(CONTRACT).read_text())
    for path in _sources():
        used |= _violations(path.read_text(), str(path))[1]
    excepted = (
        ALLOWED.keys() | ALLOWED_IN_FUNCTION.keys() | ADAPTER_IMPORTERS.keys() | CONTRACT_RESOURCE_IMPORTS
        | ALLOWED_NODE_RESOURCES.keys()
    )
    assert excepted <= used, f"delete these exceptions: {sorted(excepted - used, key=str)}"


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
    ("class M:\n    def get_node_resources(self): return []", "def get_node_resources"),
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


_OMNI = "mstar/model/qwen3_omni/qwen3_omni_model.py"


@pytest.mark.parametrize("path, body, refused", [
    (_OMNI, "return [self._lora_spec()]", False),
    (_OMNI, "return [self._lora_spec(), *token2wav_resources()]", True),
    (_OMNI, "return [self._lora_spec()] + token2wav_resources()", True),
    (_OMNI, "return token2wav_resources()", True),
    (_OMNI, "if self.speaks:\n            return token2wav_resources()\n        return [self._lora_spec()]", True),
    ("mstar/model/minicpm_o/minicpm_o_model.py", "return token2wav_resources()", True),
], ids=["the listed spec", "a second spec", "a second list", "another spec", "another spec on one path",
        "an unlisted file"])
def test_an_override_returns_only_the_spec_its_file_is_listed_for(path, body, refused):
    """Token2Wav's hand-written specs reached the engine from a ported model this way: a shared module the lint
    doesn't read built them, and the model's ``get_node_resources`` returned them."""
    source = (
        "from mstar.model.components.token2wav.resources import token2wav_resources\n"
        f"class Model:\n    def get_node_resources(self):\n        {body}\n"
    )
    found, used = _violations(source, path)
    assert bool(found) == refused, f"`{body}` in {path} gave {found}"
    assert used == (set() if refused else {path}), "an exception is used only by the override it allows"


def test_the_lint_reads_the_five_speech_packages():
    assert {path.parts[2] for path in _sources()} >= {"kokoro", "omnilingual_asr", "omnivoice", "orpheus", "voxcpm2"}


def test_the_lint_reads_all_five_reference_packages():
    assert {path.parts[2] for path in _sources()} >= {"wan22", "flux2_klein", "z_image", "ltx2_5", "qwen3_omni"}


@pytest.mark.parametrize("package", _model_packages())
def test_a_model_package_is_ported_or_on_the_frozen_list(package):
    assert package in PORTED + UNPORTED, (
        f"{package} is a new model package that is not on the state contract. Declare its state in `declare_state`, "
        "reach the engine only through `mstar.engine.state`, and add it to PORTED. Step 2a of docs/adding_models.rst "
        "shows how. UNPORTED only shrinks, as section 2 of AGENTS.md says."
    )


def test_no_package_is_on_both_lists():
    both = sorted(set(PORTED) & set(UNPORTED))
    assert not both, f"take {both} off UNPORTED, since PORTED lists them"


def test_every_entry_exists_and_every_unported_entry_is_a_package():
    gone = sorted(entry for entry in PORTED + UNPORTED if not Path(entry).exists())
    assert not gone, f"take {gone} off PORTED and UNPORTED, since no such file or package exists"
    not_packages = sorted(set(UNPORTED) - set(_model_packages()))
    assert not not_packages, f"take {not_packages} off UNPORTED, which lists only model packages under mstar/model/"


def test_every_registered_model_lives_in_a_listed_package():
    stray = sorted(module for module, entry in _registered_modules().items() if entry not in PORTED + UNPORTED)
    assert not stray, (
        f"MODEL_REGISTRY loads a model from {stray}, which no package on PORTED or UNPORTED holds. Move it into a "
        "package of its own under mstar/model/."
    )


def test_an_unported_package_registers_no_second_model_module():
    counts = Counter(_registered_modules().values())
    grown = sorted(entry for entry in UNPORTED if counts[entry] > 1)
    assert not grown, f"{grown} register a second model module. Move it into a package of its own under mstar/model/."


def test_only_the_engine_the_tests_and_the_listed_files_import_the_adapters():
    unlisted = sorted(_adapter_importers() - ADAPTER_IMPORTERS.keys())
    assert not unlisted, (
        f"{unlisted} import {ADAPTERS}. Import from `mstar.engine.state` instead, or, if a file needs the adapters, "
        "add it to ADAPTER_IMPORTERS with its reason."
    )


def test_no_other_file_of_the_state_package_imports_the_adapters():
    adapters = Path(*ADAPTERS.split(".")).with_suffix(".py")
    importers = sorted(
        str(path) for path in adapters.parent.rglob("*.py")
        if path != adapters and _imports_adapters(path.read_text(), str(path))
    )
    assert not importers, (
        f"{importers} import the adapters, so importing `mstar.engine.state` would load them and the resources' Triton "
        "kernels. Move the code that needs them into adapters.py."
    )


@pytest.mark.parametrize("source, path, expected", [
    ("from mstar.engine.state.adapters import node_handles", "mstar/model/base.py", True),
    ("from mstar.engine.state import Array, adapters", "mstar/model/base.py", True),
    ("import mstar.engine.state.adapters", "mstar/model/base.py", True),
    ("from .adapters import node_handles", CONTRACT, True),
    ("from . import adapters", CONTRACT, True),
    ("from mstar.engine.state import Array", "mstar/model/base.py", False),
    ("from mstar.api_server.openai import adapters", "mstar/model/base.py", False),
    ("from . import adapters", "mstar/api_server/openai/router.py", False),
])
def test_the_lint_tells_the_state_adapters_from_other_imports(source, path, expected):
    assert _imports_adapters(source, path) is expected, source


def test_the_contract_imports_nothing_new_from_the_resources():
    added = sorted(_resource_imports(Path(CONTRACT).read_text()) - CONTRACT_RESOURCE_IMPORTS)
    assert not added, (
        f"{CONTRACT} now imports {added} from `mstar.engine.resources`. The contract holds declarations only, so what "
        "binds a declaration to a resource belongs in adapters.py."
    )


@pytest.mark.parametrize("source, expected", [
    ("from mstar.engine.resources.kv.config import KVSpec", {("mstar.engine.resources.kv.config", "KVSpec")}),
    ("from mstar.engine import resources", {("mstar.engine", "resources")}),
    ("import mstar.engine.resources.kv", {("mstar.engine.resources.kv", "")}),
    ("from mstar.engine.state.calls import ArrayHandle", set()),
])
def test_the_lint_sees_each_way_to_import_from_the_resources(source, expected):
    assert _resource_imports(source) == expected
