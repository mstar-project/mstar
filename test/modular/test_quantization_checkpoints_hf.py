"""The 4-bit formats against real checkpoints' headers (``quant_checkpoint_headers.json``: names, dtypes and shapes
read through the HF API at pinned revisions, a few layers each, plus a census of the whole checkpoint): each one is
classified as what it is, the Qwen3.5 int4 checkpoints resolve against M*'s Qwen3.5 at TP 1/2/4 as expected, and the
routed experts of Kimi K3.1 (MXFP4), Kimi K2.6 (int4 g32) and GLM-5.3-Flash / Kimi K3 (modelopt NVFP4) land on the
experts' 4-bit storage, shape for shape, at TP 1/2/4/8."""

import json
import re
from pathlib import Path

import pytest
import torch

from mstar.distributed.communication import CommGroup
from mstar.model.components.moe import ParallelSparseMoeBlock
from mstar.model.loader.base import StackedParamRule, load_weights_into
from mstar.quantization.apply import _is_scale, _unloaded
from mstar.quantization.checkpoint import UNKNOWN, CheckpointQuant, TensorInfo
from mstar.quantization.resolve import Action, loadable, resolve_plan
from mstar.quantization.spec import parse_quantization

HEADERS = json.loads((Path(__file__).parent / "quant_checkpoint_headers.json").read_text())


def ckpt_of(repo: str) -> CheckpointQuant:
    entry = HEADERS[repo]
    tensors = {n: TensorInfo(dt, tuple(shape)) for n, (dt, shape) in entry["tensors"].items()}
    return CheckpointQuant.from_tensors(tensors, entry["quantization_config"])


def bases(ckpt, pattern):
    return {b: s for b, s in ckpt.bases.items() if re.search(pattern, b)}


def test_kimi_k3_1_pro_routed_experts_are_mxfp4_and_the_rest_bf16():
    ckpt = ckpt_of("moonshotai/Kimi-K3.1-Pro")
    experts = bases(ckpt, r"experts\.\d+\.w[123]$")
    assert len(experts) == 2 * 3 * 3  # layers 1 and 3, experts 0/1/895, w1 w2 w3
    assert {(s.format, s.layout, s.block) for s in experts.values()} == {("mxfp4", "compressed-tensors", (1, 32))}
    assert all(loadable(s) for s in experts.values())
    rest = {s.format for b, s in ckpt.bases.items() if b not in experts and s.format != "bf16"}
    assert rest == set(), rest  # attention (KDA, MLA, DSA indexer), shared experts, dense MLP, latent projections
    # the whole checkpoint: 896 experts x 3 projections x 92 MoE layers, packed and scaled
    census = HEADERS["moonshotai/Kimi-K3.1-Pro"]["census"]
    assert census["weight_packed:U8"] == census["weight_scale:U8"] == 896 * 3 * 92


def test_kimi_k2_6_routed_experts_are_int4_group_32_symmetric():
    ckpt = ckpt_of("moonshotai/Kimi-K2.6")
    experts = bases(ckpt, r"experts\.\d+\.(gate|up|down)_proj$")
    assert len(experts) == 3 * 3
    assert {s.params for s in experts.values()} == {(("group", 32), ("scale_dtype", "BF16"), ("symmetric", True))}
    assert all(s.format == "w4a16" and s.layout == "compressed-tensors" and loadable(s) for s in experts.values())
    census = HEADERS["moonshotai/Kimi-K2.6"]["census"]
    assert census["weight_packed:I32"] == census["weight_shape:I32"] == 384 * 3 * 60


def test_glm_5_3_flash_nvfp4_is_modelopt_nvfp4_with_its_mtp_experts_bf16():
    ckpt = ckpt_of("nvidia/GLM-5.3-Flash-NVFP4")
    assert ckpt.method == "modelopt"
    dense = bases(ckpt, r"layers\.0\.mlp\.(gate|up|down)_proj$")
    experts = bases(ckpt, r"layers\.3\.mlp\.experts\.\d+\.(gate|up|down)_proj$")
    mtp = bases(ckpt, r"layers\.45\.mlp\.experts\.\d+\.(gate|up|down)_proj$")
    assert {(s.format, s.layout, s.block) for s in {**dense, **experts}.values()} == {("nvfp4", "modelopt", (1, 16))}
    assert all(loadable(s) for s in experts.values())
    assert {s.format for s in mtp.values()} == {"bf16"}
    attention = bases(ckpt, r"self_attn\.")
    assert {s.format for s in attention.values()} == {"bf16"}


def test_kimi_k3_nvfp4_experts_load_and_its_fp8_attention_is_refused_by_name():
    ckpt = ckpt_of("nvidia/Kimi-K3-NVFP4")
    experts = bases(ckpt, r"experts\.\d+\.w[123]$")
    assert {(s.format, s.layout) for s in experts.values()} == {("nvfp4", "modelopt")}
    # modelopt's FP8_PB_WO: e4m3 with 4-D [N/128, 1, K/128, 1] scales, a layout no loader reads yet
    attention = {s.format for b, s in bases(ckpt, r"self_attn\.(q|k|v|o|g|b)_proj$").items()}
    assert attention == {UNKNOWN}


QWEN = ("QuantTrio/Qwen3.5-27B-AWQ", "Qwen/Qwen3.5-27B-GPTQ-Int4", "cyankiwi/Qwen3.5-27B-AWQ-4bit")
# per checkpoint: (layout, group, symmetric, zero_offset) and what each of the first four layers' linears becomes
EXPECTED = {
    "QuantTrio/Qwen3.5-27B-AWQ": (("awq", 128, False, None), {
        # modules_to_not_convert: layer 0, q/k/v, the delta net's a/b heads (so its fused input is mixed)
        "model.layers.0.mlp.gate_up_proj": "keep", "model.layers.1.mlp.gate_up_proj": "direct",
        "model.layers.1.mlp.down_proj": "direct", "model.layers.1.self_attn.out_proj": "direct",
        "model.layers.1.self_attn.in_proj_fused": "dequant", "model.layers.3.self_attn.qkv_proj": "keep",
        "model.layers.3.self_attn.o_proj": "direct"}),
    "Qwen/Qwen3.5-27B-GPTQ-Int4": (("gptq", 128, True, 1), {
        # the MLPs only (`dynamic` excludes attention)
        "model.layers.0.mlp.gate_up_proj": "direct", "model.layers.0.mlp.down_proj": "direct",
        "model.layers.0.self_attn.in_proj_fused": "keep", "model.layers.3.self_attn.qkv_proj": "keep",
        "model.layers.3.self_attn.o_proj": "keep"}),
    "cyankiwi/Qwen3.5-27B-AWQ-4bit": (("compressed-tensors", 32, True, None), {
        "model.layers.0.mlp.gate_up_proj": "direct", "model.layers.0.self_attn.out_proj": "direct",
        "model.layers.0.self_attn.in_proj_fused": "dequant", "model.layers.3.self_attn.qkv_proj": "direct",
        "model.layers.3.self_attn.o_proj": "direct"}),
}


def qwen_root(repo):
    from mstar.model.qwen3_5.config import Qwen3_5Config
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    text = HEADERS[repo]["text_config"]
    merged = {**text, **text.get("rope_parameters", {})}
    model = Qwen3_5DenseModel.__new__(Qwen3_5DenseModel)
    model.config = Qwen3_5Config(**{k: v for k, v in merged.items() if k in Qwen3_5Config.__dataclass_fields__})
    model.local_dir = None
    return model.get_quantization_roots()[0]


@pytest.mark.parametrize("repo", QWEN)
@pytest.mark.parametrize("tp", [1, 2, 4])
def test_qwen3_5_int4_checkpoints_resolve_against_m_star_qwen3_5(repo, tp):
    decl = qwen_root(repo)
    with torch.device("meta"):
        root = decl.build(CommGroup(my_global_rank=0, my_group_rank=0, group_members=list(range(tp))))
    plan = resolve_plan(root, decl, parse_quantization(None).for_node("LLM"), ckpt_of(repo), (9, 0))
    assert plan.problems == []
    (layout, group, symmetric, zero_offset), actions = EXPECTED[repo]
    for path, action in actions.items():
        assert plan.layers[path].action.value == action, (path, plan.layers[path])
    direct = plan.by_action(Action.DIRECT)
    assert direct and all(l.runtime == "w4a16" and l.source.layout == layout for l in direct)
    assert {(l.source.param("group"), l.source.param("symmetric"), l.source.param("zero_offset")) for l in direct} == {
        (group, symmetric, zero_offset)}
    # served on an H200 as stored, refused nowhere; the mixed delta-net input says why it is bf16
    if any(a == "dequant" for a in actions.values()):
        assert any("different formats" in n for n in plan.notes)


# routed experts: checkpoint names -> the stacked experts storage of a TP block
def _expert_rules(n_experts, names):
    """Per-expert stacked rules over ``names`` (checkpoint projection -> gate/up/down), as a model declares them."""
    rules = []
    for e in range(n_experts):
        for proj, kind in names.items():
            target = ".experts.down_proj" if kind == "down" else ".experts.gate_up_proj"
            rules.append(StackedParamRule(target, f".experts.{proj}.__expert{e}__.weight", f"{kind}:{e}"))
    return rules


def _remap(name):
    m = re.match(r"(.*)\.experts\.(\d+)\.(\w+)\.(weight\w*)$", name)
    return f"{m.group(1)}.experts.{m.group(3)}.__expert{m.group(2)}__.{m.group(4)}" if m else name


EXPERT_CASES = {
    # repo: (layer prefix, projection names, hidden, expert width, experts loaded)
    "moonshotai/Kimi-K3.1-Pro": ("language_model.model.layers.1.block_sparse_moe",
                                 {"w1": "gate", "w3": "up", "w2": "down"}, 3584, 3072, (0, 1)),
    "moonshotai/Kimi-K2.6": ("language_model.model.layers.1.mlp", {"gate_proj": "gate", "up_proj": "up",
                                                                     "down_proj": "down"}, 7168, 2048, (0, 1)),
    "nvidia/GLM-5.3-Flash-NVFP4": ("model.language_model.layers.3.mlp", {"gate_proj": "gate", "up_proj": "up",
                                                                           "down_proj": "down"}, 4096, 2048, (0, 1)),
}


@pytest.mark.parametrize("repo", list(EXPERT_CASES))
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_real_expert_tensors_land_on_the_experts_storage(repo, tp):
    from mstar.quantization.apply import intercept
    from mstar.quantization.resolve import LayerPlan, QuantPlan

    prefix, names, hidden, inter, loaded_experts = EXPERT_CASES[repo]
    ckpt = ckpt_of(repo)
    stored = ckpt.bases[f"{prefix}.experts.0.{next(iter(names))}"]
    n_experts = len(loaded_experts)
    parent = torch.nn.Module()
    parent.moe = block = ParallelSparseMoeBlock(hidden, n_experts, 1, inter, comm_group=CommGroup(
        my_global_rank=tp - 1, my_group_rank=tp - 1, group_members=list(range(tp))))
    block.experts_prepare(stored.format, stored.block, stored)
    from mstar.quantization.apply import _fill_unloaded
    for name, dtype in block.experts_scales().items():
        if _is_scale(name, dtype):
            _fill_unloaded(getattr(block.experts, name), dtype)
    # zeros of each real tensor's dtype and shape, as the checkpoint streams them (experts 0 and 1)
    stream = []
    for name, (dtype, shape) in HEADERS[repo]["tensors"].items():
        m = re.match(re.escape(prefix) + r"\.experts\.(\d+)\.", name)
        if not m or int(m.group(1)) not in loaded_experts:
            continue
        torch_dtype = {"U8": torch.uint8, "I32": torch.int32, "BF16": torch.bfloat16, "F32": torch.float32,
                       "F8_E4M3": torch.float8_e4m3fn}[dtype]
        local = name.replace(f"{prefix}.experts.{m.group(1)}.",
                             f"moe.experts.{loaded_experts.index(int(m.group(1)))}.")
        stream.append((local, torch.ones(shape, dtype=torch.float32).to(torch_dtype)))
    plan = QuantPlan("LLM", {"moe.experts": LayerPlan("moe.experts", "experts", stored.format, stored.format,
                                                      Action.DIRECT, "auto", stored.block, stored)})
    rules = _expert_rules(n_experts, names)
    filled = load_weights_into(parent, intercept(stream, plan, rules, _remap), stacked_params=rules,
                               name_remapper=_remap)
    assert filled == {f"moe.experts.{n}" for n in block._experts_views}, filled
    scales = block.experts_scales().items()
    assert not [n for n, d in scales if _is_scale(n, d) and _unloaded(getattr(block.experts, n), d)]
    shard = inter // tp
    packed = {"mxfp4": 2, "w4a16": 8, "nvfp4": 2}[stored.format]
    data = block.experts.gate_up_proj if stored.format == "nvfp4" else block.experts.gate_up_proj_packed
    assert tuple(data.shape) == (n_experts, 2 * shard, hidden // packed)
    down = block.experts.down_proj if stored.format == "nvfp4" else block.experts.down_proj_packed
    assert tuple(down.shape) == (n_experts, hidden, shard // packed)
