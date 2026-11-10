"""Qwen3.5-MoE (``qwen3_5_moe``), text-only, on CPU with no real weights.

- Against both Qwen3.5-397B-A17B checkpoints' own safetensors headers (``qwen3_5_moe_headers.json``: layers 0
  and 3, the experts as one template per layer) the plan serves each linear as stored at TP 8 and 4: nvidia's
  NVFP4-V2 runs its routed experts NVFP4 and its fp8-per-tensor linears fp8 (a fused q|k|v or gate|up, whose parts
  modelopt scaled apart, re-quantized per rank); Qwen's FP8 runs block fp8. Each rank reads only its slice of the
  experts.
- A tiny random model from transformers' ``Qwen3_5MoeForCausalLM`` (the oracle), written in the checkpoints'
  layout, loads through the real loader at TP 1, 2 and 4: the MoE block's output (summed over ranks, each rank
  returning its partial sum) equals transformers' block's.
- The same model in NVFP4-V2's layout (fabricated NVFP4 experts, fp8 per-tensor linears) holds the checkpoint's
  bytes per rank, and its routed output is the dequantized experts' up to NVFP4's activation rounding.

The fixture was made by reading each safetensors file's header of the two snapshots (no tensor data).
"""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mstar.distributed.communication import CommGroup  # noqa: E402
from mstar.quantization.checkpoint import CheckpointQuant, TensorInfo  # noqa: E402
from mstar.quantization.resolve import Action, resolve_plan  # noqa: E402
from mstar.quantization.spec import parse_quantization  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = json.loads((Path(__file__).parent / "qwen3_5_moe_headers.json").read_text())
NVFP4_REPO, FP8_REPO = "nvidia/Qwen3.5-397B-A17B-NVFP4-V2", "Qwen/Qwen3.5-397B-A17B-FP8"
P = "model.language_model.layers"


def _group(rank: int, tp: int) -> CommGroup:
    return CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=list(range(tp)))


def _bare_model(config):
    """The model's declarations need only its config (no checkpoint files)."""
    from mstar.model.qwen3_5.qwen3_5_moe_model import Qwen3_5MoeModel

    model = object.__new__(Qwen3_5MoeModel)
    model.config, model.vision_config, model._submodule_cache = config, None, {}
    return model


def _real_config(repo, tmp_path, layers=None):
    from mstar.model.qwen3_5.config import Qwen3_5Config

    (tmp_path / "config.json").write_text(json.dumps(FIXTURE[repo]["config"]))
    config = Qwen3_5Config.from_hf(tmp_path)
    if layers is not None:
        config.num_hidden_layers, config.layer_types = layers, config.layer_types[:layers]
    return config


def _real_checkpoint(repo, layers) -> CheckpointQuant:
    """The fixture's tensors for ``layers`` stack layers: 0 and 3 as stored, the others copies of layer 0 (the
    linear-attention layers all store alike)."""
    entry = FIXTURE[repo]
    tensors = {}

    def put(layer, suffix, info):
        tensors[f"{P}.{layer}.{suffix}"] = TensorInfo(info[0], tuple(info[1]))

    for name, info in entry["tensors"].items():
        if not name.startswith(P + "."):
            tensors[name] = TensorInfo(info[0], tuple(info[1]))
            continue
        source, suffix = name[len(P) + 1:].split(".", 1)
        for layer in range(layers):
            if str(layer) == source or (source == "0" and layer % 4 != 3):
                put(layer, suffix, info)
    for source, template in entry["expert_templates"].items():
        for layer in range(layers):
            if str(layer) == source or (source == "0" and layer % 4 != 3):
                for e in range(entry["n_routed_experts"]):
                    for suffix, info in template.items():
                        put(layer, f"mlp.experts.{e}.{suffix}", info)
    return CheckpointQuant.from_tensors(tensors, entry["config"]["quantization_config"])


def _real_plan(repo, tmp_path, tp, capability=(10, 3)):
    config = _real_config(repo, tmp_path, layers=4)
    (decl,) = _bare_model(config).get_quantization_roots()
    with torch.device("meta"):
        root = decl.build(_group(0, tp))
    return resolve_plan(root, decl, parse_quantization(None).for_node("LLM"), _real_checkpoint(repo, 4), capability)


def _actions(plan, path):
    layer = plan.layers[path]
    return layer.action, layer.runtime


def test_the_real_config_reads_as_moe(tmp_path):
    config = _real_config(NVFP4_REPO, tmp_path)
    assert config.is_moe and (config.num_experts, config.num_experts_per_tok) == (512, 10)
    assert (config.moe_intermediate_size, config.shared_expert_intermediate_size) == (1024, 1024)
    assert config.intermediate_size == 0 and len(config.full_layer_indices) == 15


@pytest.mark.parametrize("tp", [8, 4])
def test_nvfp4_v2_serves_as_stored(tmp_path, tp):
    plan = _real_plan(NVFP4_REPO, tmp_path, tp)
    assert plan.problems == []
    for layer in range(4):
        experts = plan.layers[f"model.layers.{layer}.mlp.experts"]
        assert (experts.action, experts.runtime, experts.block) == (Action.DIRECT, "nvfp4", (1, 16))
        # modelopt scales the shared expert's gate and up apart: one per-tensor scale per rank needs a requant
        assert _actions(plan, f"model.layers.{layer}.mlp.shared_expert.gate_up_proj") == (Action.REQUANT, "fp8_tensor")
        assert _actions(plan, f"model.layers.{layer}.mlp.shared_expert.down_proj") == (Action.DIRECT, "fp8_tensor")
        assert plan.layers[f"model.layers.{layer}.mlp.shared_expert_gate"].action is Action.KEEP
    fused = plan.layers["model.layers.0.self_attn.in_proj_fused"]
    # q|k|v|z fp8 beside the bf16 a / b gate rows, which stay bf16
    assert (fused.action, fused.runtime, fused.tail) == (Action.REQUANT, "fp8_tensor", 3)
    assert _actions(plan, "model.layers.0.self_attn.out_proj") == (Action.DIRECT, "fp8_tensor")
    assert _actions(plan, "model.layers.3.self_attn.qkv_proj") == (Action.REQUANT, "fp8_tensor")
    assert _actions(plan, "model.layers.3.self_attn.o_proj") == (Action.DIRECT, "fp8_tensor")
    assert plan.layers["lm_head"].action is Action.KEEP


def test_nvfp4_v2_is_refused_where_no_fp4_kernel_runs(tmp_path):
    problems = _real_plan(NVFP4_REPO, tmp_path, 8, capability=(9, 0)).problems
    assert problems and all("format: bf16" in p for p in problems), problems


@pytest.mark.parametrize("tp", [8, 4])
def test_fp8_serves_block_fp8_as_stored(tmp_path, tp):
    plan = _real_plan(FP8_REPO, tmp_path, tp)
    assert plan.problems == []
    for layer in range(4):
        experts = plan.layers[f"model.layers.{layer}.mlp.experts"]
        assert (experts.action, experts.runtime, experts.block) == (Action.DIRECT, "fp8_block", (128, 128))
        for proj in ("gate_up_proj", "down_proj"):
            assert _actions(plan, f"model.layers.{layer}.mlp.shared_expert.{proj}") == (Action.DIRECT, "fp8_block")
    fused = plan.layers["model.layers.0.self_attn.in_proj_fused"]
    assert (fused.action, fused.runtime, fused.tail) == (Action.REQUANT, "fp8_block", 3)
    for path in ("model.layers.0.self_attn.out_proj", "model.layers.3.self_attn.qkv_proj",
                 "model.layers.3.self_attn.o_proj"):
        assert _actions(plan, path) == (Action.DIRECT, "fp8_block"), path


def test_checkpoint_names_land_on_the_stacked_experts():
    from mstar.model.loader.base import _StackedMatcher
    from mstar.model.qwen3_5.weight_loader import qwen3_5_name_remapper, stacked_params

    match = _StackedMatcher(stacked_params(512))

    def place(name):
        native = qwen3_5_name_remapper(name)
        return None if native is None else match(native)

    assert place(f"{P}.3.mlp.experts.511.up_proj.weight_scale_2") == (
        "model.layers.3.mlp.experts.gate_up_proj_scale_2", "up:511")
    assert place(f"{P}.3.mlp.experts.7.down_proj.weight") == ("model.layers.3.mlp.experts.down_proj", "down:7")
    assert place(f"{P}.0.mlp.shared_expert.up_proj.weight") == (
        "model.layers.0.mlp.shared_expert.gate_up_proj.weight", 1)
    assert place(f"{P}.0.mlp.gate.weight") == ("model.layers.0.mlp.gate.weight", None)
    assert place(f"{P}.0.linear_attn.in_proj_z.weight") == ("model.layers.0.self_attn.in_proj_fused.weight", "z")
    # the fp8 KV cache's scales, the MTP layer and the vision tower don't load
    for name in (f"{P}.3.self_attn.k_proj.k_scale", f"{P}.3.self_attn.v_proj.v_scale", "mtp.fc.weight",
                 "model.visual.blocks.0.attn.qkv.weight"):
        assert place(name) is None, name


def _real_keys(repo):
    entry = FIXTURE[repo]
    keys = list(entry["tensors"])
    for layer, template in entry["expert_templates"].items():
        keys += [f"{P}.{layer}.mlp.experts.0.{suffix}" for suffix in template]
    return keys


def test_each_rank_reads_only_its_slice_of_the_experts():
    from mstar.model.qwen3_5.weight_loader import expert_read_slices

    e = f"{P}.3.mlp.experts.0"
    # 128 intermediate rows per rank at TP 8: gate/up rows, down's packed bytes (2 per byte) and scales (1 per 16)
    specs = expert_read_slices(_real_keys(NVFP4_REPO), 1024, tp_rank=3, tp_size=8)
    assert specs[f"{e}.gate_proj.weight"] == specs[f"{e}.up_proj.weight_scale"] == (0, 384, 512)
    assert specs[f"{e}.down_proj.weight"] == (1, 192, 256)
    assert specs[f"{e}.down_proj.weight_scale"] == (1, 24, 32)
    assert not any(k.endswith(("weight_scale_2", "input_scale")) for k in specs)
    # only the routed experts: the shared expert and attention are read whole
    assert all(".mlp.experts." in k for k in specs)
    specs = expert_read_slices(_real_keys(FP8_REPO), 1024, tp_rank=3, tp_size=8)
    assert specs[f"{e}.gate_proj.weight"] == (0, 384, 512) and specs[f"{e}.down_proj.weight"] == (1, 384, 512)
    assert specs[f"{e}.up_proj.weight_scale_inv"] == (0, 3, 4) and specs[f"{e}.down_proj.weight_scale_inv"] == (1, 3, 4)
    # a scale block straddling the slice (TP 16: 64 rows a rank) reads whole
    assert expert_read_slices(_real_keys(FP8_REPO), 1024, tp_rank=3, tp_size=16).get(f"{e}.down_proj.weight") is None


@pytest.mark.parametrize("name", ["qwen3_5_397b_a17b_tp8.yaml", "qwen3_5_397b_a17b_tp4.yaml"])
def test_shipped_configs_validate(name, tmp_path):
    from mstar.validation import validate_deployment

    path = ROOT / "configs" / name
    config = yaml.safe_load(path.read_text())
    assert config["model"] == "qwen3_5_moe"
    deployment = validate_deployment(_bare_model(_real_config(NVFP4_REPO, tmp_path)), config, str(path))
    assert deployment is not None


# --- a tiny model, through the real loader ----------------------------------------------------------------------

TEXT = dict(
    model_type="qwen3_5_moe_text", attn_output_gate=True, head_dim=64, hidden_size=256,
    num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
    num_attention_heads=4, num_key_value_heads=1, linear_conv_kernel_dim=4, linear_key_head_dim=32,
    linear_num_key_heads=4, linear_num_value_heads=8, linear_value_head_dim=32,
    num_experts=8, num_experts_per_tok=2, moe_intermediate_size=128, shared_expert_intermediate_size=128,
    max_position_embeddings=4096, rms_norm_eps=1e-6, tie_word_embeddings=False,
    rope_parameters={"mrope_interleaved": True, "mrope_section": [3, 3, 2], "rope_type": "default",
                     "rope_theta": 10000000, "partial_rotary_factor": 0.25},
)


# the synthetic tokenizer's ids, padded so the vocabulary shards at TP 4
VOCAB = 320


def _hf_model(vocab_size):
    """transformers' Qwen3.5-MoE, the oracle (5.x)."""
    pytest.importorskip("transformers.models.qwen3_5_moe")
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    torch.manual_seed(0)
    hf = Qwen3_5MoeForCausalLM(Qwen3_5MoeTextConfig(**TEXT, vocab_size=vocab_size)).eval()
    with torch.no_grad():
        for param in hf.parameters():
            # not the init's zeros / ones: a swapped or misplaced tensor must show
            param.copy_(torch.randn_like(param) * (0.02 if param.dim() > 1 else 0.1))
    return hf.to(torch.bfloat16)


def _checkpoint_state(hf) -> dict[str, torch.Tensor]:
    """``hf``'s weights under the released checkpoints' names, each routed expert's projections apart."""
    state = {}
    for name, tensor in hf.state_dict().items():
        name = name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name
        if name.endswith("mlp.experts.gate_up_proj"):
            base = name.rpartition(".")[0]
            for e, gate_up in enumerate(tensor):
                gate, up = gate_up.chunk(2)
                state[f"{base}.{e}.gate_proj.weight"], state[f"{base}.{e}.up_proj.weight"] = gate, up
        elif name.endswith("mlp.experts.down_proj"):
            for e, down in enumerate(tensor):
                state[f"{name.rpartition('.')[0]}.{e}.down_proj.weight"] = down
        else:
            state[name] = tensor
    return {k: v.contiguous() for k, v in state.items()}


def _write(root: Path, state: dict[str, torch.Tensor], quantization_config=None) -> Path:
    from safetensors.torch import save_file
    from test_qwen3_5_video import write_synthetic_checkpoint

    write_synthetic_checkpoint(root)
    config = json.loads((root / "config.json").read_text())
    config["architectures"], config["model_type"] = ["Qwen3_5MoeForConditionalGeneration"], "qwen3_5_moe"
    config["text_config"] = {**TEXT, "vocab_size": VOCAB, "eos_token_id": config["text_config"]["eos_token_id"]}
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    (root / "config.json").write_text(json.dumps(config))
    # the tensors a released checkpoint carries beside the text tower, which never load
    extra = {"mtp.fc.weight": torch.zeros(4, 4, dtype=torch.bfloat16),
             "model.visual.patch_embed.proj.weight": torch.zeros(4, 4, dtype=torch.bfloat16)}
    save_file({name: t.clone() for name, t in {**state, **extra}.items()}, str(root / "model.safetensors"))
    return root


@pytest.fixture(scope="module")
def bf16_checkpoint(tmp_path_factory):
    from test_qwen3_5_video import write_synthetic_checkpoint

    tokens = json.loads((write_synthetic_checkpoint(tmp_path_factory.mktemp("tok")) / "config.json").read_text())
    assert tokens["text_config"]["vocab_size"] <= VOCAB
    hf = _hf_model(VOCAB)
    return hf, _write(tmp_path_factory.mktemp("qwen3_5_moe_bf16"), _checkpoint_state(hf))


def _load(path, rank=0, tp=1, dtype=torch.float32, quantized=False):
    """The LLM node's weights root for ``rank`` of ``tp``, loaded through the model's own loader (and, ``quantized``,
    under the quantization plan ``quantization: auto`` resolves from the checkpoint)."""
    from mstar.model.qwen3_5.components.language_model import Qwen3_5ForCausalLM
    from mstar.model.qwen3_5.qwen3_5_moe_model import Qwen3_5MoeModel
    from mstar.model.qwen3_5.weight_loader import load_qwen3_5_weights
    from mstar.quantization.scope import NodeQuantization

    model = Qwen3_5MoeModel(str(path))
    default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        lm = Qwen3_5ForCausalLM(model.config, _group(rank, tp))
    finally:
        torch.set_default_dtype(default)
    if not quantized:
        load_qwen3_5_weights(lm, path)
        return lm, None
    scope = NodeQuantization("LLM", parse_quantization(None).for_node("LLM"), model.get_quantization_roots())
    with scope.activated():
        load_qwen3_5_weights(lm, path)
        scope.finish(lm)
    return lm, next(iter(scope.plans.values()))[1]


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_the_moe_block_matches_transformers(bf16_checkpoint, tp):
    """Each rank's block returns its partial sum (the decoder loop reduces it with the residual add); summed over
    the ranks it is transformers' block, in fp32 on the checkpoint's bf16 weights."""
    hf, path = bf16_checkpoint
    hf = copy.deepcopy(hf).float()
    x = torch.randn(7, TEXT["hidden_size"], generator=torch.Generator().manual_seed(1))
    for layer in (0, 3):
        want = hf.model.layers[layer].mlp(x[None])[0]
        got = sum(_load(path, rank, tp)[0].model.layers[layer].mlp(x) for rank in range(tp))
        assert _rel(got, want) < 1e-5, (layer, tp, _rel(got, want))


def test_the_text_only_model_declares_no_vision(bf16_checkpoint):
    from mstar.model.qwen3_5.config import VISION_ATTN
    from mstar.model.qwen3_5.qwen3_5_moe_model import Qwen3_5MoeModel
    from mstar.quantization.scope import load_submodule

    _, path = bf16_checkpoint
    model = Qwen3_5MoeModel(str(path))
    assert model.vision_config is None and "prefill_vision" not in model.get_graph_walk_graphs()
    assert all(spec.resource_key != VISION_ATTN for spec in model.get_node_resources())
    with pytest.raises(ValueError, match="text-only"):
        model.process_prompt("what is this?", ["text", "image"], ["text"], tensors={"image_inputs": [None]})
    assert len(model.process_prompt("hello", ["text"], ["text"]).new_input_tensors["text_inputs"]) == 1
    submodule = load_submodule(model, "LLM", {}, "cpu")
    walks = {c.capture_graph_walk for c in submodule.get_accelerator_graph_configs(torch.device("cpu"))}
    assert "prefill_vision" not in walks and "decode" in walks


# --- the same model in nvidia/Qwen3.5-397B-A17B-NVFP4-V2's layout ------------------------------------------------

MODELOPT = {"quant_method": "modelopt_mixed", "quant_algo": "MIXED_PRECISION"}
_UNIT = 6.0 * 448.0  # an NVFP4 16-block's largest magnitude in units of its tensor scale
# what modelopt stores fp8 per tensor in the NVFP4-V2 checkpoint (the rest of the non-expert linears stay bf16)
_FP8 = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj", "self_attn.q_proj",
        "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.shared_expert.gate_proj",
        "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj")


def _nvfp4(w, input_scale, tensor_scale):
    """One linear as modelopt stores it: e2m1 bytes, row-major e4m3 block scales, fp32 tensor and input scales."""
    from mstar.model.components.block_scaled import NVFP4
    from mstar.utils.fp8_block import FP8_DTYPE, unswizzle_block_scales

    q = NVFP4.quantize(w, tensor_scale)
    n, k = w.shape
    scale = unswizzle_block_scales(q.scale.view(torch.uint8), n, k // 16).contiguous().view(FP8_DTYPE)
    stored = {"weight": q.data.view(torch.uint8), "weight_scale": scale,
              "weight_scale_2": q.tensor_scale.reshape(()).float(), "input_scale": torch.tensor(input_scale)}
    return stored, NVFP4.dequantize(q).float()


def _fp8_tensor(w):
    from mstar.utils.fp8_block import FP8_DTYPE

    s = w.float().abs().amax() / 448.0
    return {"weight": (w.float() / s).to(FP8_DTYPE), "weight_scale": s.reshape(()), "input_scale": torch.tensor(0.01)}


@pytest.fixture(scope="module")
def nvfp4_checkpoint(bf16_checkpoint, tmp_path_factory):
    """``bf16_checkpoint``'s weights stored as NVFP4-V2 stores them, and the fp32 weights its experts stand for."""
    hf, _ = bf16_checkpoint
    bf16 = _checkpoint_state(hf)
    state, experts = {}, {}
    x = torch.randn(64, TEXT["hidden_size"], generator=torch.Generator().manual_seed(2))
    gate_up_in = float(x.abs().amax()) / _UNIT
    for name, tensor in bf16.items():
        base, _, suffix = name.rpartition(".")
        if ".mlp.experts." in base and suffix == "weight":
            if base.endswith("up_proj"):
                continue
            if base.endswith("gate_proj"):
                # gate and up share their tensor scale and activation scale (modelopt calibrates them together)
                up_base = base.replace("gate_proj", "up_proj")
                gate, up = tensor.float(), bf16[up_base + ".weight"].float()
                shared = torch.maximum(gate.abs().amax(), up.abs().amax()) / _UNIT
                for b, w in ((base, gate), (up_base, up)):
                    stored, experts[b] = _nvfp4(w, gate_up_in, shared)
                    state.update({f"{b}.{k}": v for k, v in stored.items()})
                continue
            stored, experts[base] = _nvfp4(tensor.float(), 0.05, tensor.float().abs().amax() / _UNIT)
            state.update({f"{base}.{k}": v for k, v in stored.items()})
        elif suffix == "weight" and base.endswith(_FP8):
            state.update({f"{base}.{k}": v for k, v in _fp8_tensor(tensor).items()})
            if base.endswith(("self_attn.k_proj", "self_attn.v_proj")):
                state[f"{base}.{base[-6]}_scale"] = torch.tensor(1.0)
        else:
            state[name] = tensor
    return _write(tmp_path_factory.mktemp("qwen3_5_moe_nvfp4"), state, MODELOPT), state, experts


@pytest.mark.parametrize("rank,tp", [(0, 1), (1, 2)])
def test_nvfp4_experts_hold_the_checkpoints_bytes_per_rank(nvfp4_checkpoint, rank, tp):
    from mstar.utils.fp8_block import FP8_DTYPE

    path, stream, _ = nvfp4_checkpoint
    lm, plan = _load(path, rank, tp, dtype=torch.bfloat16, quantized=True)
    assert _actions(plan, "model.layers.0.mlp.experts") == (Action.DIRECT, "nvfp4")
    moe = lm.model.layers[3].mlp
    assert moe.experts_packed.name == "nvfp4"
    inter = TEXT["moe_intermediate_size"] // tp
    rows = slice(rank * inter, (rank + 1) * inter)
    for e in range(TEXT["num_experts"]):
        p = f"{P}.3.mlp.experts.{e}"
        gate_up = torch.cat([stream[f"{p}.gate_proj.weight"][rows], stream[f"{p}.up_proj.weight"][rows]])
        assert torch.equal(moe.experts.gate_up_proj[e], gate_up)
        assert torch.equal(moe.experts.down_proj[e],
                           stream[f"{p}.down_proj.weight"][:, rank * inter // 2:(rank + 1) * inter // 2])
        assert torch.equal(moe.experts.down_proj_scale[e].view(FP8_DTYPE),
                           stream[f"{p}.down_proj.weight_scale"][:, rank * inter // 16:(rank + 1) * inter // 16])
        assert torch.equal(moe.experts.gate_up_proj_scale_2[e].view(torch.float32),
                           stream[f"{p}.gate_proj.weight_scale_2"].repeat(2))
    for layer in (0, 3):
        mlp = lm.model.layers[layer].mlp
        assert mlp.shared_expert.gate_up_proj.quant_format.name == "fp8_tensor"
        assert mlp.shared_expert.down_proj.quant_format.name == "fp8_tensor"
    assert lm.model.layers[3].self_attn.qkv_proj.quant_format.name == "fp8_tensor"
    assert lm.model.layers[0].self_attn.in_proj_fused.quant_format.name == "fp8_tensor"
    assert lm.model.layers[0].self_attn.out_proj.quant_format.name == "fp8_tensor"


def _dequantized_routed(experts, layer, flat, weights, ids):
    """The routed output of the fp32 weights the experts stand for, bf16 activations."""
    want = torch.zeros(flat.shape)
    for t in range(flat.shape[0]):
        for slot in range(ids.shape[1]):
            p = f"{P}.{layer}.mlp.experts.{int(ids[t, slot])}"
            x = flat[t].float()
            gate, up, down = (experts[f"{p}.{n}"] for n in ("gate_proj", "up_proj", "down_proj"))
            want[t] += float(weights[t, slot]) * (down @ (F.silu(gate @ x) * (up @ x)))
    return want


@torch.no_grad()
def test_nvfp4_routed_output_is_the_dequantized_experts_and_sums_over_ranks(bf16_checkpoint, nvfp4_checkpoint):
    path, _, experts = nvfp4_checkpoint
    flat = torch.randn(9, TEXT["hidden_size"], generator=torch.Generator().manual_seed(3)).bfloat16()
    whole = _load(path, dtype=torch.bfloat16, quantized=True)[0].model.layers[3].mlp
    weights, ids, _ = whole.gate(flat)
    got = whole._experts_4bit(flat, weights, ids, skip_invalid=False)
    # NVFP4 rounds the activations to 4 bits per element (the weights here are exact): ~16% on random data
    assert _rel(got, _dequantized_routed(experts, 3, flat, weights, ids)) < 0.2
    # each rank's partial sum over its slice of the intermediate width: the down input's 16-blocks don't straddle
    # the slice, so the ranks round it as one rank does (measured 3e-3: each rank's bf16 output rounding)
    parts = [_load(path, rank, 2, dtype=torch.bfloat16, quantized=True)[0].model.layers[3].mlp for rank in (0, 1)]
    summed = sum(m._experts_4bit(flat, weights, ids, skip_invalid=False).float() for m in parts)
    assert _rel(summed, got) < 2e-2
    # the whole block, fp8 shared expert included, against transformers' bf16 one (a swapped gate / up or a
    # misplaced scale is ~1.0 off; fp8 and NVFP4 on random weights measured 0.2, cosine 0.98)
    out, want = whole(flat).float(), bf16_checkpoint[0].model.layers[3].mlp(flat[None])[0].float()
    assert F.cosine_similarity(out.flatten(), want.flatten(), dim=0) > 0.95
