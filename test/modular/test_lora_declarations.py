"""Each LoRA model's spec is built by the bridge from the roots the model declares (`Model.lora_roots`), where each
model built it by hand before. A spec missing a target refuses, at startup, any adapter that touches it. But a spec that
places an adapter's module at another layer or slice, or ignores it, runs that module wrongly or not at all, and nothing
at run time says so. So each declared spec is held to the hand-built one, recorded in
``fixtures/lora_declared_specs.json`` at the commit before the move.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("safetensors")
import lora_module_helpers as H  # noqa: E402

from mstar.engine.resources.lora import LORA_RESOURCE_KEY  # noqa: E402
from mstar.engine.state.adapters import StateSetup  # noqa: E402
from mstar.model.base import enabled_resources  # noqa: E402
from mstar.model.chatterbox.chatterbox_model import ChatterboxModel  # noqa: E402
from mstar.model.components.lora.modules import AdapterRoot, LoraRoot, declared_lora_spec  # noqa: E402
from mstar.model.higgs_audio.higgs_audio_model import HiggsAudioModel  # noqa: E402
from mstar.model.multimodal import vision_encoder  # noqa: E402
from mstar.model.multimodal_decoder import MediaIntake  # noqa: E402
from mstar.model.orpheus.orpheus_model import OrpheusModel  # noqa: E402
from mstar.model.qwen3_5.config import Qwen3_5VisionConfig  # noqa: E402
from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel  # noqa: E402
from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel  # noqa: E402
from mstar.model.qwen3_tts.qwen3_tts_model import Qwen3TTSModel  # noqa: E402
from mstar.testing_utils.model import make_model  # noqa: E402

HAND_BUILT = json.loads((Path(__file__).parent / "fixtures" / "lora_declared_specs.json").read_text())["models"]
# a serving config with a `resources: lora:` block
LORA_BLOCK = {"resources": {LORA_RESOURCE_KEY: {}}}


def _keys(model, config: dict) -> list[str]:
    return [spec.resource_key for spec in enabled_resources(model, config, StateSetup())]


def _qwen3_5():
    """With its vision tower, so the spec joins two nodes."""
    vision = Qwen3_5VisionConfig(
        depth=2, hidden_size=8, intermediate_size=16, num_heads=2, in_channels=3, patch_size=2, temporal_patch_size=2,
        spatial_merge_size=2, num_position_embeddings=16, out_hidden_size=H.HIDDEN,
    )
    return H.model_object(
        Qwen3_5DenseModel, H.tiny_config(), vision_config=vision,
        media=(MediaIntake("image", vision_encoder(), 0, None),),
        _image_processor=SimpleNamespace(size={"shortest_edge": 65536}, patch_size=vision.patch_size),
    )


def _phi4mm():
    from mstar.model.phi4mm.phi4mm_model import Phi4MMModel
    from mstar.model.phi4mm.tests._paths import snapshot

    if snapshot() is None:
        pytest.skip("Phi-4-multimodal checkpoint not cached")
    return Phi4MMModel(snapshot())


# each LoRA model at the sizes `lora_module_helpers.STACKS` runs. Only Phi-4-multimodal reads a checkpoint.
MODELS = {
    "qwen3_5": _qwen3_5,
    "qwen3_omni": lambda: H.model_object(Qwen3OmniModel, H.STACKS["qwen3_omni_thinker"]().config()),
    "qwen3_tts": lambda: H.model_object(
        Qwen3TTSModel, H.STACKS["qwen3_tts_talker"]().config(), codec_dtype=torch.float32,
    ),
    "orpheus": lambda: H.model_object(OrpheusModel, H.tiny()),
    "chatterbox": lambda: H.model_object(ChatterboxModel, H.STACKS["chatterbox_t3"]().config()),
    "higgs_audio": lambda: H.model_object(HiggsAudioModel, H.STACKS["higgs_audio"]().config()),
    "phi4mm": _phi4mm,
}


def _module_map(part) -> dict:
    return {
        "root_path": part.root_path, "bindings": part.bindings, "unservable": part.unservable,
        "adapter_roots": part._roots, "aliases": part._aliases, "ignore": part._ignore,
        "stacked_params": [dataclasses.asdict(rule) for rule in part._stacked],
        "prefix": part.rules.prefix, "remap": getattr(part.rules.remap, "__qualname__", None),
    }


def _form(model, spec) -> dict:
    """What serving reads of ``spec``, as the fixture's JSON holds it: its targets, each node's module map, where it
    places each target's first module under each layout adapters write it in, the adapters it ships (under the
    model's checkpoint), the pool `LoraSpec.for_node` builds for each node, and whether it lands last among the
    deployment's specs."""
    nodes = sorted(spec.nodes)
    parts = {node: spec.module_map.part(node) for node in nodes}
    first = {}
    for part in parts.values():
        for path, ref in sorted(part.bindings.items(), key=lambda item: item[1].layer, reverse=True):
            first[ref.target] = (part, path)
    modules = sorted({
        *(root.adapter_prefix + path for part, path in first.values() for root in (*part._roots, AdapterRoot("", ""))),
        *(alias for part in parts.values() for alias in part._aliases),
    })
    pools = {node: spec.for_node(node) for node in nodes}
    return json.loads(json.dumps({
        "resource_key": spec.resource_key, "nodes": nodes, "num_layers": spec.num_layers,
        "targets": {name: dataclasses.asdict(target) for name, target in spec.targets.items()},
        "dropped_targets": {name: dataclasses.asdict(target) for name, target in spec.dropped_targets.items()},
        "module_maps": {node: _module_map(part) for node, part in parts.items()},
        "placed": {module: spec.module_map(module) for module in modules},
        "adapters": {name: os.path.relpath(path, model.local_dir) for name, path in spec.adapters.items()},
        "pools": {node: pool and {
            "resource_key": pool.resource_key, "targets": sorted(pool.targets),
            "dropped_targets": sorted(pool.dropped_targets), "ignored": [m for m in modules if pool.ignored_modules(m)],
        } for node, pool in pools.items()},
        "lora_last": _keys(model, LORA_BLOCK)[-1] == LORA_RESOURCE_KEY,
    }, sort_keys=True, default=repr))


@pytest.mark.parametrize("name", MODELS)
def test_each_models_declared_lora_spec_equals_the_one_it_built_by_hand(name):
    model = MODELS[name]()
    spec = declared_lora_spec(model.lora_roots())
    assert spec is not None, f"{name} declares no LoRA roots"
    assert _form(model, spec) == HAND_BUILT[name], f"{name}'s declared spec would serve adapters other than it did"


@pytest.mark.parametrize("name", MODELS)
def test_each_models_lora_roots_name_nodes_it_has(name):
    """A root on a node the model lacks builds a pool that no node binds, so an adapter's modules there load and never
    apply. Nothing checks it at boot, which would build the walk graphs again."""
    model = MODELS[name]()
    declared = {root.node for root in model.lora_roots()}
    assert declared, f"{name} declares no LoRA roots"
    astray = sorted(declared - set(model.nodes))
    assert not astray, f"{name} declares LoRA roots on {astray}, which are not its nodes"


@pytest.mark.parametrize("name", MODELS)
def test_a_lora_block_adds_each_models_lora_spec_last_and_changes_nothing_else(name):
    model = MODELS[name]()
    without = _keys(model, {})
    assert LORA_RESOURCE_KEY not in without, f"{name} would build a LoRA pool without a lora: block"
    assert _keys(model, LORA_BLOCK) == [*without, LORA_RESOURCE_KEY], f"a lora: block would change {name}'s other specs"


def _root(node: str, **kw) -> LoraRoot:
    return LoraRoot(node=node, build=lambda _: nn.Sequential(nn.Linear(4, 4)), root_path="", **kw)


def test_two_roots_on_one_node_are_refused():
    with pytest.raises(ValueError, match=r"name nodes \['N'\] twice"):
        declared_lora_spec((_root("N"), _root("N")))


def test_an_adapter_shipped_as_two_files_under_one_name_is_refused():
    roots = (_root("A", adapters={"x": "/a.safetensors"}), _root("B", adapters={"x": "/b.safetensors"}))
    with pytest.raises(ValueError, match="ship adapter 'x' as two files"):
        declared_lora_spec(roots)


def test_a_built_spec_in_place_of_roots_is_refused():
    built = declared_lora_spec((_root("N"),))
    with pytest.raises(TypeError, match=r"returns LoraRoots, .* not \['LoraSpec'\]"):
        declared_lora_spec((built,))


def test_the_declared_spec_is_built_only_under_a_lora_block():
    model = make_model(lora_roots=lambda: (_root("N"),))
    assert _keys(model, {}) == [], "a deployment without a lora: block would build a pool"
    assert _keys(model, LORA_BLOCK) == [LORA_RESOURCE_KEY]


def test_a_model_that_declares_no_roots_builds_no_lora_spec():
    model = make_model()
    assert declared_lora_spec(model.lora_roots()) is None
    assert _keys(model, LORA_BLOCK) == [], "a model without LoRA would build a pool under a lora: block"
