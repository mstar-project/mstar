"""A ported package reaches the engine only through ``mstar.engine.state``.

The contract freezes that one surface, so a ported model that also imports a resource class, or reads a resource off the
engine's inputs, can break with no contract change, and later PRs can't move what it reached for. Nothing at run time
would notice, so this reads each package's source. The base classes are the bridge, so they aren't read. Each exception
says why it stands or which PR deletes it, and an exception nothing uses any more fails too, so the list only shrinks.
A scaffold node takes its handle calls from `DeclaredTokenDecoder` rather than writing them again.

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
    "mstar/model/higgs_audio", "mstar/model/voxcpm2",
    "mstar/model/canary_qwen", "mstar/model/qwen3_asr", "mstar/model/parakeet", "mstar/model/fish_s2_pro",
    "mstar/model/qwen3_tts",
    "mstar/model/pi05", "mstar/model/vjepa2", "mstar/model/canary", "mstar/model/components/encoder_decoder.py",
    "mstar/model/bagel", "mstar/model/chatterbox", "mstar/model/whisper", "mstar/model/phi4mm",
    "mstar/model/nemotron_duplex", "mstar/model/components/fastconformer.py",
    "mstar/model/components/fastconformer_stream.py",
)
# The list only shrinks: a package leaves it when its port lands, and nothing is added to it.
UNPORTED = (
    "mstar/model/cosmos3", "mstar/model/cosyvoice3", "mstar/model/glm52", "mstar/model/glm5_next",
    "mstar/model/lingbot_world", "mstar/model/minicpm_o", "mstar/model/muse", "mstar/model/ripple",
    "mstar/model/waypoint",
)
# shared code under mstar/model/ that holds no model
SHARED = ("mstar/model/components", "mstar/model/loader")
FORBIDDEN_ATTRIBUTES = {
    "node_resources", "resources", "request_state", "request_states", "per_request_states", "sequence_shard",
    # what a handle holds
    "resource", "positions", "reads", "first_read", "_kv", "_attn", "_cross", "_pool", "_runners", "_views", "_blocks",
    "_bindings", "_layouts", "_update",
    # what LoRA's binding sets on a module, and the pool call behind it
    "lora_delta", "lora_params", "layer_delta",
}

_CAPTURE = "a capture's buckets and compile mode, which the engine doesn't pick yet"
ALLOWED = {
    ("mstar.engine.accelerator_graph_config", name): _CAPTURE
    for name in (
        "AcceleratorGraphConfig", "BatchedAcceleratorGraphConfig", "DETERMINISTIC_COMPILE_MODE",
        "PackedAcceleratorGraphConfig", "PiecewiseAcceleratorGraphConfig", "PiecewiseBatchedConfig",
        "PiecewiseCallInputs", "PiecewiseCaptureShape", "PiecewisePackedConfig", "decode_rows_then_prompt",
        "check_inductor_options", "geometric_batch_sizes",
    )
} | {
    ("mstar.engine.engine", "ExecutingBatch"): "the annotation of `NodeSubmodule.can_batch`, a base class's hook",
    ("mstar.engine.resources.lora.checkpoint", "LoraFile"): "a model's static LoRA merge at load, which holds no state",
    ("mstar.engine.resources.kv.config", "DEFAULT_PAGE_SIZE"):
        "the page size a model's default pool sizes count in, until the pool sizes itself",
    ("mstar.engine.resources", "SeqShardSpec"): (
        "MiniMax-H3's video decoder's sequence shard, written by hand: a contract SeqShard brackets a label an array "
        "opens, and the decoder has no arrays"
    ),
}
_LORA = (
    "per-request LoRA, outside the contract for good: adapters are weights the bridge builds from `lora_roots` and "
    "binds, which no request writes, and a request picks one by its kwargs"
)
# Model-layer declarations of what stands outside the contract. A ported file may import these names from the LoRA
# package and nothing else from it, by any form of import: the rest of the package is the bridge's.
LORA_PACKAGE = "mstar.model.components.lora"
OUTSIDE = {("mstar.model.components.lora.modules", name): _LORA for name in ("AdapterRoot", "LoraRoot", "PER_SEGMENT")}
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
# the attribute: a graph break (`disable`), a forbidden read the function may keep, or `def` for a hook it may define.
ALLOWED_IN_FUNCTION = {
    ("mstar/model/minimax_h3/minimax_h3_model.py", "H3Model.get_node_resources", "def"):
        "the video decoder's hand-written sequence shard, as its SeqShardSpec import says",
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
# what a model defines to write resource specs by hand, rather than declare them (`declare_state`, `lora_roots`)
HAND_WRITTEN_SPECS = {"get_node_resources"}
# what builds a LoRA spec, which only the bridge's own files do
LORA_SPEC_BUILDERS = {"declared_lora_spec", "module_lora_spec", "join_lora_specs", "LoraSpec"}
LORA_BRIDGE = {Path("mstar/model/base.py"), Path("mstar/model/components/lora/modules.py")}
ADAPTERS = "mstar.engine.state.adapters"
_BY_HAND = "the script builds a node's resources and handles by hand, as a test does"
# the files that import the adapters from outside mstar/engine/ and the tests, each with its reason
ADAPTER_IMPORTERS = {
    "mstar/model/base.py": "the model base class resolves a deployment's resource specs from its declarations",
    "mstar/model/submodule_base.py": "`engine_step` expands a node's declared step for the engine",
    "mstar/api_server/data_worker.py": "the data worker converts a prompt's state layout into today's KV layouts",
    "mstar/conductor/conductor.py": "the conductor converts a request's `SampleSettings` into sampler configs",
    "mstar/validation/__init__.py": "`validate_deployment` checks a deployment's state setup before a worker spawns",
    "mstar/model/utils.py": "`check_kv_pools_fit` sizes each KV pool from a request's declared layouts at load",
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
    ("mstar.engine.resources.position.config", "LongRoPE"),
    ("mstar.engine.resources.step", "Segment"),
}

_HOOKS = frozenset({
    "bind_state", "node_step", "next_position", "loopback_tokens", "last_token_hidden", "sample_tokens",
})
# The handle hooks a ported scaffold node still writes itself, by file, class and method.
OWN_HOOKS = {
    ("mstar/model/qwen3_5/submodules.py", "LLMSubmodule", "node_step"): "the GDN state's step, beside the KV cache's",
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


def _in_lora_package(name: str) -> bool:
    return name == LORA_PACKAGE or name.startswith(f"{LORA_PACKAGE}.")


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
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if (node.module, alias.name) in OUTSIDE:
                    used.add((node.module, alias.name))
                elif _in_lora_package(f"{node.module}.{alias.name}"):
                    found.append(f"from {node.module} import {alias.name}, which is the bridge's")
        elif isinstance(node, ast.Import):
            found += [
                f"import {a.name}" for a in node.names if a.name.startswith("mstar.engine") or _in_lora_package(a.name)
            ]
        elif isinstance(node, ast.FunctionDef) and node.name in RESOURCE_BINDS:
            found.append(f"def {node.name} at line {node.lineno}, which takes the built resources")
        elif isinstance(node, ast.FunctionDef) and node.name in HAND_WRITTEN_SPECS:
            if (path, function, "def") in ALLOWED_IN_FUNCTION:
                used.add((path, function, "def"))
            else:
                found.append(f"def {node.name} at line {node.lineno}, which writes resource specs by hand")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
            found.append(f".{node.attr} at line {node.lineno}")
        elif breaks:
            found.append(f"torch.compiler.disable at line {node.lineno}, where the state calls hold the graph breaks")
    return found, used


def _bases(sources: list[Path]) -> dict[str, set[str]]:
    """The bases of each class ``sources`` define, by name, joined where two files define one name."""
    bases: dict[str, set[str]] = {}
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef):
                bases.setdefault(node.name, set()).update(ast.unparse(base).rsplit(".", 1)[-1] for base in node.bases)
    return bases


def _descendants(bases: dict[str, set[str]], root: str) -> set[str]:
    """``root`` and the classes in ``bases`` that descend from it."""
    found = {root}
    while grown := {name for name, of in bases.items() if name not in found and of & found}:
        found |= grown
    return found


def _hook_violations(
    source: str, path: str, bases: dict[str, set[str]],
) -> tuple[list[str], set[tuple[str, str, str]]]:
    """The handle hooks a scaffold node in ``source``, the file at ``path``, writes itself, and the exceptions used.
    ``bases`` gives the bases of the classes the ported files define, so a node under a ported scaffold node counts
    too."""
    found, used = [], set()
    declared = _descendants(bases, "DeclaredTokenDecoder")
    sampled = _descendants(bases, "SampledTokenDecoder") | declared
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ClassDef):
            continue
        own = {ast.unparse(base).rsplit(".", 1)[-1] for base in node.bases}
        if not own & sampled:
            continue
        if not own & declared:
            found.append(f"{node.name} subclasses SampledTokenDecoder")
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name in _HOOKS:
                if (path, node.name, item.name) in OWN_HOOKS:
                    used.add((path, node.name, item.name))
                else:
                    found.append(f"{node.name}.{item.name} at line {item.lineno}")
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


def test_a_ported_scaffold_node_writes_no_handle_hook_itself():
    found, used = [], set()
    bases = _bases(_sources())
    for path in _sources():
        in_file, used_in_file = _hook_violations(path.read_text(), str(path), bases)
        found += [f"{path}: {violation}" for violation in in_file]
        used |= used_in_file
    assert not found, f"these repeat DeclaredTokenDecoder's handle calls: {found}"
    assert OWN_HOOKS.keys() <= used, f"delete these exceptions: {sorted(OWN_HOOKS.keys() - used)}"


def test_a_node_under_a_ported_scaffold_node_is_checked_too():
    source = "class CanaryDecoderSubmodule(EncoderDecoderNode):\n    def node_step(self):\n        pass\n"
    bases = {"EncoderDecoderNode": {"DeclaredTokenDecoder"}, "CanaryDecoderSubmodule": {"EncoderDecoderNode"}}
    found, _ = _hook_violations(source, "x.py", bases)
    assert found == ["CanaryDecoderSubmodule.node_step at line 2"], "a hook under EncoderDecoderNode would pass"


@pytest.mark.parametrize("entry", PORTED)
def test_every_ported_entry_yields_a_source(entry):
    assert _entry_sources(entry), f"{entry} yields no source, so the lint would pass it unread"


def test_every_exception_is_still_used():
    used = _adapter_importers() | _resource_imports(Path(CONTRACT).read_text())
    for path in _sources():
        used |= _violations(path.read_text(), str(path))[1]
    excepted = (
        ALLOWED.keys() | ALLOWED_IN.keys() | ALLOWED_IN_FUNCTION.keys() | OUTSIDE.keys() | ADAPTER_IMPORTERS.keys()
        | CONTRACT_RESOURCE_IMPORTS
    )
    assert excepted <= used, f"delete these exceptions: {sorted(excepted - used, key=str)}"


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
    ("class M:\n    def get_node_resources(self): return []", "def get_node_resources"),
    ("from mstar.model.components.lora.modules import module_lora_spec", "module_lora_spec, which is the bridge's"),
    ("from mstar.model.components.lora import modules", "import modules, which is the bridge's"),
    ("from mstar.model.components import lora", "import lora, which is the bridge's"),
    ("import mstar.model.components.lora.modules as m", "import mstar.model.components.lora.modules"),
    ("from mstar.model.components.lora.capable import make_lora_capable", "make_lora_capable, which is the bridge's"),
    ("def f(self): return self.qkv_proj.lora_delta", ".lora_delta"),
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


def test_the_lint_lets_a_ported_file_declare_lora_roots():
    source = "from mstar.model.components.lora.modules import PER_SEGMENT, LoraRoot"
    modules = "mstar.model.components.lora.modules"
    assert _violations(source) == ([], {(modules, "PER_SEGMENT"), (modules, "LoraRoot")})


def test_minimax_h3s_entry_covers_only_its_own_get_node_resources():
    path = "mstar/model/minimax_h3/minimax_h3_model.py"
    source = (
        "class H3Model:\n    def get_node_resources(self): return []\n"
        "class H3Other:\n    def get_node_resources(self): return []\n"
    )
    found, used = _violations(source, path)
    assert used == {(path, "H3Model.get_node_resources", "def")}
    expected = ["def get_node_resources at line 4, which writes resource specs by hand"]
    assert found == expected, "the entry covers another class's get_node_resources"


def test_no_model_builds_a_lora_spec_by_hand():
    """Every model file outside the bridge, ported or not. ``tests`` folders are skipped."""
    found = []
    for path in sorted(Path("mstar/model").rglob("*.py")):
        if "tests" in path.parts or path in LORA_BRIDGE:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and ast.unparse(node.func).split(".")[-1] in LORA_SPEC_BUILDERS:
                found.append(f"{path}:{node.lineno}")
    assert not found, f"these build a LoRA spec by hand, where a model declares `lora_roots`: {found}"


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


def test_the_lint_reads_the_six_speech_packages():
    assert {path.parts[2] for path in _sources()} >= {
        "kokoro", "omnilingual_asr", "omnivoice", "orpheus", "higgs_audio", "voxcpm2",
    }


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
