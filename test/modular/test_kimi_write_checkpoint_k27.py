"""CPU test for the ``k27_code`` variant of ``kimi_write_checkpoint.py``:
checks the writer against the real Kimi-K2.7-Code checkpoint's tensor-name
pattern (from ``k27_ref/model.safetensors.index.json``), the packed/scale/
shape dtypes the load path expects, the config round-trip, and that the real
mstar weight loader consumes the written checkpoint end to end.

Runs at reduced widths (``--hidden``/``--moe-intermediate``/``--vocab``): the
real widths (hidden=7168, intermediate=18432) put a single dense layer at
~800 MB, well over what a CPU test should materialize.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "integration"))

import torch
from kimi_write_checkpoint import K27_REF
from kimi_write_checkpoint import main as write_main
from safetensors import safe_open

from mstar.model.components.quantization import process_weights_after_loading
from mstar.model.kimi_k2_7.components.causal_lm import KimiForCausalLM
from mstar.model.kimi_k2_7.config import KimiK2Config
from mstar.model.loader import load_weights as driver_load_weights

LAYERS = 2
EXPERTS = 4
HIDDEN = 128
MOE_INTERMEDIATE = 64
VOCAB = 32

_EXPERT_IDX_RE = re.compile(r"\.experts\.\d+\.")


def _real_names():
    with open(K27_REF / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    # vision_tower/mm_projector: never requested by a text-only node group
    # (KimiK2Model._create_submodule only builds them for VISION_NODE), so
    # the writer doesn't emit them and they're excluded here too.
    return [k for k in weight_map if "vision_tower" not in k and "mm_projector" not in k]


def _expected_names(num_layers, num_experts):
    """The tensor-name set for ``num_layers``/``num_experts``, derived from
    the real index rather than from the writer, so a naming drift between
    the two is caught."""
    names = _real_names()
    top_level = {n for n in names if not n.startswith("language_model.model.layers.")}
    dense = {n for n in names if n.startswith("language_model.model.layers.0.")}

    moe_source_layer = next(
        i for i in range(1, 61)
        if f"language_model.model.layers.{i}.mlp.gate.weight" in names
    )
    moe_prefix = f"language_model.model.layers.{moe_source_layer}."
    templates = set()
    for name in names:
        if not name.startswith(moe_prefix):
            continue
        rest = name[len(moe_prefix):]
        templates.add(_EXPERT_IDX_RE.sub(".experts.{e}.", rest))

    expected = set(top_level) | dense
    for layer in range(1, num_layers):
        for template in templates:
            if "{e}" in template:
                for expert in range(num_experts):
                    expected.add(
                        f"language_model.model.layers.{layer}." + template.format(e=expert)
                    )
            else:
                expected.add(f"language_model.model.layers.{layer}." + template)
    return expected


def _write_test_checkpoint(out):
    write_main([
        "--out", str(out), "--variant", "k27_code",
        "--layers", str(LAYERS), "--experts", str(EXPERTS),
        "--hidden", str(HIDDEN), "--moe-intermediate", str(MOE_INTERMEDIATE),
        "--vocab", str(VOCAB), "--seed", "0",
    ])


def test_write_k27_code_checkpoint_names_and_dtypes(tmp_path):
    out = tmp_path / "ckpt"
    _write_test_checkpoint(out)

    with open(out / "model.safetensors.index.json") as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    written = set(weight_map)
    expected = _expected_names(LAYERS, EXPERTS)
    assert written == expected, (
        f"missing: {sorted(expected - written)[:10]}; "
        f"extra: {sorted(written - expected)[:10]}"
    )

    def tensor(name):
        with safe_open(str(out / weight_map[name]), framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    gate_packed = tensor("language_model.model.layers.1.mlp.experts.0.gate_proj.weight_packed")
    gate_scale = tensor("language_model.model.layers.1.mlp.experts.0.gate_proj.weight_scale")
    gate_shape = tensor("language_model.model.layers.1.mlp.experts.0.gate_proj.weight_shape")
    assert gate_packed.dtype == torch.int32
    assert gate_scale.dtype == torch.bfloat16
    assert gate_shape.dtype == torch.int64
    assert gate_shape.tolist() == [MOE_INTERMEDIATE, HIDDEN]
    assert gate_packed.shape == (MOE_INTERMEDIATE, HIDDEN // 8)
    assert gate_scale.shape == (MOE_INTERMEDIATE, HIDDEN // 32)

    down_packed = tensor("language_model.model.layers.1.mlp.experts.0.down_proj.weight_packed")
    down_shape = tensor("language_model.model.layers.1.mlp.experts.0.down_proj.weight_shape")
    assert down_shape.tolist() == [HIDDEN, MOE_INTERMEDIATE]
    assert down_packed.shape == (HIDDEN, MOE_INTERMEDIATE // 8)

    embed = tensor("language_model.model.embed_tokens.weight")
    assert embed.dtype == torch.bfloat16
    assert embed.shape == (VOCAB, HIDDEN)
    assert tensor("language_model.lm_head.weight").shape == (VOCAB, HIDDEN)

    bias = tensor("language_model.model.layers.1.mlp.gate.e_score_correction_bias")
    assert bias.dtype == torch.float32
    assert bias.shape == (EXPERTS,)

    dense_mlp = tensor("language_model.model.layers.0.mlp.gate_proj.weight")
    assert dense_mlp.dtype == torch.bfloat16  # layer 0 is dense: no quantization (ignore list)


def test_write_k27_code_checkpoint_config_roundtrip(tmp_path):
    out = tmp_path / "ckpt"
    _write_test_checkpoint(out)

    with open(out / "config.json") as f:
        raw_config = json.load(f)
    assert raw_config["text_config"]["num_hidden_layers"] == LAYERS
    assert raw_config["text_config"]["n_routed_experts"] == EXPERTS
    assert raw_config["text_config"]["quantization_config"] is not None

    cfg = KimiK2Config.from_checkpoint(out, base=KimiK2Config.k27_code())
    assert cfg.num_hidden_layers == LAYERS
    assert cfg.n_routed_experts == EXPERTS
    assert cfg.quantization_config is not None
    assert cfg.quantization_config.num_bits == 4
    assert cfg.quantization_config.group_size == 32


def test_write_k27_code_checkpoint_loads_through_real_loader(tmp_path):
    out = tmp_path / "ckpt"
    _write_test_checkpoint(out)

    cfg = KimiK2Config.from_checkpoint(out, base=KimiK2Config.k27_code())
    # from_checkpoint doesn't override hidden/moe_intermediate/vocab — the
    # writer's shape overrides for this test — so mirror them explicitly to
    # build a model matching what was actually written.
    cfg.hidden_size = HIDDEN
    cfg.moe_intermediate_size = MOE_INTERMEDIATE
    cfg.vocab_size = VOCAB

    with torch.device("meta"):
        model = KimiForCausalLM(cfg)
    model = model.to(torch.bfloat16)
    model.to_empty(device="cpu")
    loaded = driver_load_weights(model, out, device="cpu")
    process_weights_after_loading(model, torch.device("cpu"))

    all_params = set(dict(model.named_parameters()).keys())
    assert loaded == all_params, (
        f"unloaded: {sorted(all_params - loaded)}; spurious: {sorted(loaded - all_params)}"
    )

    experts = model.model.layers[1].mlp.experts
    assert experts.gate_up_proj_packed.dtype == torch.int32
    assert experts.gate_up_proj_scale.dtype == torch.bfloat16
    assert experts.down_proj_packed.dtype == torch.int32
    assert experts.down_proj_scale.dtype == torch.bfloat16
    assert model.model.layers[1].mlp.gate.e_score_correction_bias.dtype == torch.float32
