"""GLM-5.2/5.3 under the quantization plan, against GLM-5.3's real checkpoint layout: the safetensors headers
of zai-org/GLM-5.3 (fetched through the HF API, headers only; layers 0 dense, 3 MoE and 78 MTP, with the routed
experts as one template per projection) and its config.json. Every linear the checkpoint stores in block fp8
loads as stored at the deployment's TP size, kv_b / indexer / shared expert / LM head go where the model
says, and nothing is refused."""

import json
from pathlib import Path

import pytest
import torch

from mstar.distributed.communication import CommGroup
from mstar.model.glm52.components.indexer import is_full_indexer_layer
from mstar.model.glm52.config import Glm52ModelConfig
from mstar.model.glm52.quantization import glm52_quantization_root
from mstar.quantization.checkpoint import CheckpointQuant, TensorInfo
from mstar.quantization.resolve import Action, resolve_plan
from mstar.quantization.spec import parse_quantization

FIXTURE = json.loads((Path(__file__).parent / "glm53_fp8_headers.json").read_text())
ATTN = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj")


def _config(kernels="auto", dsa=False):
    """The checkpoint's geometry, its trunk cut to the first MoE layer and an MTP layer on a FULL indexer slot."""
    cfg = Glm52ModelConfig.from_hf_config(FIXTURE["config"])
    cfg.kernels = kernels
    cfg.mla_absorb = True
    cfg.dsa_long_context = dsa
    cfg.mtp_num_draft_tokens = 1
    cfg.num_hidden_layers = next(n for n in range(4, 40) if is_full_indexer_layer(cfg, n))
    return cfg


def _checkpoint(cfg) -> CheckpointQuant:
    """The fixture's tensors, the MTP layer (78 in the checkpoint) renumbered to follow this trunk."""
    tensors = {}
    rename = {"78": str(cfg.num_hidden_layers)}

    def put(name, entry):
        parts = name.split(".")
        if name.startswith("model.layers."):
            parts[2] = rename.get(parts[2], parts[2])
        tensors[".".join(parts)] = TensorInfo(entry[0], tuple(entry[1]))

    for name, entry in FIXTURE["tensors"].items():
        put(name, entry)
    for layer, template in FIXTURE["expert_templates"].items():
        for e in range(FIXTURE["n_routed_experts"]):
            for suffix, entry in template.items():
                put(f"model.layers.{layer}.mlp.experts.{e}.{suffix}", entry)
    return CheckpointQuant.from_tensors(tensors, FIXTURE["config"]["quantization_config"])


def _plan(cfg, tp=8, quantization=None):
    decl = glm52_quantization_root(cfg, checkpoint_dir=lambda: None)
    with torch.device("meta"):
        root = decl.build(CommGroup(my_global_rank=0, my_group_rank=0, group_members=list(range(tp))))
    return resolve_plan(root, decl, parse_quantization(quantization).for_node("LLM"), _checkpoint(cfg), (9, 0))


def test_the_checkpoint_reads_as_block_fp8():
    ckpt = _checkpoint(_config())
    assert ckpt.method == "fp8" and ckpt.block == (128, 128)
    assert ckpt.bases["model.layers.0.self_attn.kv_a_proj_with_mqa"].format == "fp8_block"
    assert ckpt.bases["model.layers.3.mlp.experts.255.down_proj"].format == "fp8_block"
    assert ckpt.bases["lm_head"].format == "bf16"


@pytest.mark.parametrize("tp", [8, 4])
def test_glm53_loads_its_fp8_linears_as_stored(tp):
    cfg = _config()
    plan = _plan(cfg, tp)
    assert plan.problems == []
    mtp = "mtp.transformer_layer"
    for stack in ("model.layers.0", "model.layers.3", mtp):
        for name in ATTN:
            layer = plan.layers[f"{stack}.self_attn.{name}"]
            assert (layer.action, layer.runtime, layer.block) == (Action.DIRECT, "fp8_block", (128, 128)), layer
        # the absorbed MLA folds kv_b into bf16 w_kc / w_vc
        assert plan.layers[f"{stack}.self_attn.kv_b_proj"].action is Action.DEQUANT
    for path in ("model.layers.0.mlp.gate_up_proj", "model.layers.0.mlp.down_proj",
                 "model.layers.3.mlp.experts", f"{mtp}.mlp.experts"):
        assert plan.layers[path].action is Action.DIRECT, path
    # kernels: auto's decode chain reads the shared expert in bf16
    assert plan.layers["model.layers.3.mlp.shared_expert.gate_up_proj"].action is Action.DEQUANT
    assert plan.layers["lm_head"].action is Action.KEEP
    assert plan.layers["mtp.eh_proj"].action is Action.KEEP


def test_the_reference_kernels_keep_the_shared_expert_fp8_and_dsa_dequantizes_the_indexer():
    plan = _plan(_config(kernels="reference", dsa=True))
    assert plan.problems == []
    assert plan.layers["model.layers.3.mlp.shared_expert.down_proj"].action is Action.DIRECT
    assert plan.layers["model.layers.0.self_attn.indexer.wq_b"].action is Action.DEQUANT
    assert plan.layers["model.layers.0.self_attn.indexer.weights_proj"].action is Action.KEEP


def test_experts_kept_bf16_dequantize_and_a_second_format_is_refused():
    cfg = _config()
    cfg.moe_fp8_resident = False
    assert _plan(cfg).layers["model.layers.3.mlp.experts"].action is Action.DEQUANT
    problems = _plan(_config(), quantization="fp8_channel").problems
    # one line per rule and reason: the stored fp8_block linears, and the experts no fp8_channel kernel serves
    assert len(problems) == 2
    assert "14 layers" in problems[0] and "compounds two roundings" in problems[0]
    assert "serves fp8_channel on no experts layer" in problems[1]


def test_the_model_declares_its_root_and_never_downloads_for_it():
    from mstar.model.glm52.glm52_model import Glm52Model

    model = Glm52Model.__new__(Glm52Model)
    model.config, model.model_path_hf, model.cache_dir = _config(), "zai-org/not-a-cached-repo", None
    (root,) = model.get_quantization_roots()
    assert root.node == "LLM" and root.root_type.__name__ == "Glm52ForCausalLM"
    assert model._local_checkpoint() is None
