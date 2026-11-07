"""GLM-5.3 as nvidia/GLM-5.3-NVFP4 stores it (modelopt ``MIXED_PRECISION``): routed experts NVFP4, the dense
layers' FFN fp8 per tensor, attention, shared experts and the MTP layer bf16.

Against the checkpoint's own safetensors headers (``glm53_nvfp4_headers.json``: layers 0, 3 and 78, the experts as
one template per layer) the plan loads each linear as stored, and a per-layer rule moves a bf16 linear to a format
at load. On a reduced model the experts hold the checkpoint's bytes on every TP rank, and the block's routed output
is the dequantized experts' up to NVFP4's activation rounding (a swapped gate/up or a misplaced scale is off by
the whole output)."""
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mstar.distributed.communication import CommGroup  # noqa: E402
from mstar.model.components.block_scaled import NVFP4  # noqa: E402
from mstar.model.glm52.components.indexer import is_full_indexer_layer  # noqa: E402
from mstar.model.glm52.config import Glm52ModelConfig  # noqa: E402
from mstar.model.glm52.quantization import glm52_quantization_root  # noqa: E402
from mstar.quantization.checkpoint import CheckpointQuant, TensorInfo  # noqa: E402
from mstar.quantization.resolve import Action, resolve_plan  # noqa: E402
from mstar.quantization.spec import parse_quantization  # noqa: E402
from mstar.utils.fp8_block import FP8_DTYPE, unswizzle_block_scales  # noqa: E402

FIXTURE = json.loads((Path(__file__).parent / "glm53_nvfp4_headers.json").read_text())
ATTN = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")
MODELOPT = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION"}


def _real_config():
    cfg = Glm52ModelConfig.from_hf_config(FIXTURE["config"])
    cfg.mla_absorb = True
    cfg.mtp_num_draft_tokens = 1
    cfg.num_hidden_layers = next(n for n in range(4, 40) if is_full_indexer_layer(cfg, n))
    return cfg


def _real_checkpoint(cfg) -> CheckpointQuant:
    tensors, rename = {}, {"78": str(cfg.num_hidden_layers)}

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


def _real_plan(quantization=None, tp=8):
    cfg = _real_config()
    decl = glm52_quantization_root(cfg, checkpoint_dir=lambda: None)
    with torch.device("meta"):
        root = decl.build(CommGroup(my_global_rank=0, my_group_rank=0, group_members=list(range(tp))))
    return resolve_plan(root, decl, parse_quantization(quantization).for_node("LLM"), _real_checkpoint(cfg),
                        (10, 3))


@pytest.mark.parametrize("tp", [8, 2])
def test_glm53_nvfp4_loads_as_stored(tp):
    plan = _real_plan(tp=tp)
    assert plan.problems == []
    for stack in ("model.layers.0", "model.layers.3", "mtp.transformer_layer"):
        for name in ATTN:
            assert plan.layers[f"{stack}.self_attn.{name}"].action is Action.KEEP, (stack, name)
    experts = plan.layers["model.layers.3.mlp.experts"]
    assert (experts.action, experts.runtime, experts.block) == (Action.DIRECT, "nvfp4", (1, 16))
    # modelopt calibrates gate and up apart: one tensor scale for the fused layer means a requantization
    assert (plan.layers["model.layers.0.mlp.gate_up_proj"].action,
            plan.layers["model.layers.0.mlp.gate_up_proj"].runtime) == (Action.REQUANT, "fp8_tensor")
    assert plan.layers["model.layers.0.mlp.down_proj"].action is Action.DIRECT
    for path in ("model.layers.3.mlp.shared_expert.gate_up_proj", "mtp.transformer_layer.mlp.experts", "lm_head"):
        assert plan.layers[path].action is Action.KEEP, path


def test_a_rule_per_layer_moves_bf16_linears_to_a_format_at_load():
    plan = _real_plan({"layers": [
        {"match": "model.layers.*.self_attn.o_proj", "format": "nvfp4"},
        {"match": "model.layers.*.self_attn.q_b_proj", "format": "fp8_block"},
        {"match": "model.layers.0-2.mlp.*", "format": "bf16"},
    ]})
    assert plan.problems == []
    for layer in (0, 3):
        o = plan.layers[f"model.layers.{layer}.self_attn.o_proj"]
        assert (o.action, o.runtime) == (Action.QUANTIZE, "nvfp4")
        assert plan.layers[f"model.layers.{layer}.self_attn.q_b_proj"].runtime == "fp8_block"
        assert plan.layers[f"model.layers.{layer}.self_attn.kv_a_proj_with_mqa"].action is Action.KEEP
    assert plan.layers["model.layers.0.mlp.gate_up_proj"].action is Action.DEQUANT
    assert plan.layers["model.layers.3.mlp.experts"].runtime == "nvfp4"


def test_each_rank_reads_only_its_slice_of_the_experts():
    from mstar.model.glm52.weight_loader import build_glm52_read_plan

    cfg = _real_config()
    keys = list(FIXTURE["tensors"])
    for layer, template in FIXTURE["expert_templates"].items():
        keys += [f"model.layers.{layer}.mlp.experts.0.{suffix}" for suffix in template]
    _, specs = build_glm52_read_plan(keys, cfg, tp_rank=3, tp_size=8)
    p = "model.layers.3.mlp.experts.0"
    # 256 intermediate rows per rank: gate/up rows, down's packed bytes (2 per byte) and its scales (1 per 16)
    assert specs[f"{p}.gate_proj.weight"] == (0, 768, 1024)
    assert specs[f"{p}.up_proj.weight_scale"] == (0, 768, 1024)
    assert specs[f"{p}.down_proj.weight"] == (1, 384, 512)
    assert specs[f"{p}.down_proj.weight_scale"] == (1, 48, 64)
    assert f"{p}.gate_proj.weight_scale_2" not in specs and f"{p}.down_proj.input_scale" not in specs
    # bf16 experts (the MTP layer's) and the dense fp8 FFN are read whole
    assert not any(k.startswith("model.layers.0.mlp.") for k in specs)


# --- a reduced model in the checkpoint's layout -----------------------------------------------------------------

def _reduced_config():
    cfg = Glm52ModelConfig.reduced()
    cfg.mla_absorb = True
    # 64 columns per rank at TP 2: NVFP4 quantizes K in whole 64-column scale tiles
    cfg.moe_intermediate_size = 128
    return cfg


def _nvfp4(w, input_scale, tensor_scale=None):
    """One linear as modelopt stores it: e2m1 bytes, row-major e4m3 block scales, fp32 tensor and input scales."""
    q = NVFP4.quantize(w, tensor_scale)
    n, k = w.shape
    scale = unswizzle_block_scales(q.scale.view(torch.uint8), n, k // 16).contiguous().view(FP8_DTYPE)
    stored = {"weight": q.data.view(torch.uint8), "weight_scale": scale,
              "weight_scale_2": q.tensor_scale.reshape(()), "input_scale": torch.tensor(input_scale)}
    return stored, NVFP4.dequantize(q)


def _fp8_tensor(w):
    s = w.abs().amax() / 448.0
    w8 = (w / s).to(FP8_DTYPE)
    return {"weight": w8, "weight_scale": s.reshape(()).float(), "input_scale": torch.tensor(0.01)}, w8.float() * s


_UNIT = 6.0 * 448.0  # an NVFP4 16-block's largest magnitude in units of its tensor scale


def _input_scales(refs, layer_prefix, n_experts):
    """Activation scales as modelopt calibrates them (amax / (6 * 448) over a calibration batch): one for every
    expert's gate and up input, one per expert for its down input."""
    x = torch.randn(256, refs[f"{layer_prefix}.experts.0.gate_proj"][2].shape[1],
                    generator=torch.Generator().manual_seed(0))
    down = {}
    for e in range(n_experts):
        p = f"{layer_prefix}.experts.{e}"
        gate, up = (refs[f"{p}.{n}"][2].float() for n in ("gate_proj", "up_proj"))
        down[e] = float((F.silu(x @ gate.t()) * (x @ up.t())).abs().amax()) / _UNIT
    return float(x.abs().amax()) / _UNIT, down


def _fabricate(cfg):
    """A stream in nvidia/GLM-5.3-NVFP4's layout, and the fp32 weights the experts stand for."""
    from test_glm52_moe import _fabricate_checkpoint

    bf16_state, refs = _fabricate_checkpoint(cfg)
    state, experts, calibrated = [], {}, {}
    for name, tensor in bf16_state:
        base, _, suffix = name.rpartition(".")
        if suffix == "weight_scale_inv":
            continue
        if suffix != "weight" or base not in refs:
            state.append((name, tensor))
            continue
        weight = refs[base][2].float()
        if ".mlp.experts." in base:
            layer, _, rest = base.partition(".experts.")
            expert, _, proj = rest.partition(".")
            if layer not in calibrated:
                calibrated[layer] = _input_scales(refs, layer, cfg.n_routed_experts)
            gate_up_in, down_in = calibrated[layer]
            if proj == "up_proj":
                continue
            if proj == "gate_proj":
                # gate and up share their tensor scale, as modelopt calibrates them (one per expert)
                up = refs[base.replace("gate_proj", "up_proj")][2].float()
                shared = torch.maximum(weight.abs().amax(), up.abs().amax()) / _UNIT
                for p, w in (("gate_proj", weight), ("up_proj", up)):
                    stored, ref = _nvfp4(w, input_scale=gate_up_in, tensor_scale=shared)
                    experts[base.replace("gate_proj", p)] = ref
                    state += [(f"{base.replace('gate_proj', p)}.{k}", v) for k, v in stored.items()]
                continue
            stored, ref = _nvfp4(weight, input_scale=down_in[int(expert)])
            experts[base] = ref
            state += [(f"{base}.{k}", v) for k, v in stored.items()]
        elif ".mlp." in base and ".shared_experts." not in base:
            stored, _ = _fp8_tensor(weight)
            state += [(f"{base}.{k}", v) for k, v in stored.items()]
        else:
            state.append((name, weight.bfloat16()))
    return state, experts


def _load(cfg, state, rank=0, tp=1, quantization=None, device="cpu"):
    from mstar.model.glm52._testing import _ST_DTYPES
    from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM
    from mstar.quantization.scope import NodeQuantization

    group = CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=list(range(tp)))
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)  # the served node's dtype: the dense FFN requantizes a bf16 weight
    try:
        with torch.device(device):
            model = Glm52ForCausalLM(cfg, comm_group=group)
    finally:
        torch.set_default_dtype(default)
    infos = {name: TensorInfo(_ST_DTYPES[t.dtype], tuple(t.shape)) for name, t in state}
    ckpt = CheckpointQuant.from_tensors(infos, MODELOPT)
    decl = glm52_quantization_root(cfg, checkpoint_dir=lambda: None)
    scope = NodeQuantization("LLM", parse_quantization(quantization).for_node("LLM"), [decl], checkpoint=ckpt)
    with scope.activated():
        model.load_weights(iter(state))
        scope.finish(model)
    return model


@pytest.mark.parametrize("rank,tp", [(0, 1), (1, 2)])
def test_the_experts_hold_the_checkpoints_bytes_per_rank(rank, tp):
    torch.manual_seed(0)
    cfg = _reduced_config()
    state, _ = _fabricate(cfg)
    stream = dict(state)
    moe = _load(cfg, state, rank, tp).model.layers[1].mlp
    assert moe.experts_packed.name == "nvfp4"
    inter = cfg.moe_intermediate_size // tp
    rows = slice(rank * inter, (rank + 1) * inter)
    for e in range(cfg.n_routed_experts):
        p = f"model.layers.1.mlp.experts.{e}"
        gate_up = torch.cat([stream[f"{p}.gate_proj.weight"][rows], stream[f"{p}.up_proj.weight"][rows]])
        assert torch.equal(moe.experts.gate_up_proj[e], gate_up)
        down = stream[f"{p}.down_proj.weight"][:, rank * inter // 2:(rank + 1) * inter // 2]
        assert torch.equal(moe.experts.down_proj[e], down)
        assert torch.equal(moe.experts.down_proj_scale[e].view(FP8_DTYPE),
                           stream[f"{p}.down_proj.weight_scale"][:, rank * inter // 16:(rank + 1) * inter // 16])
        pair = moe.experts.gate_up_proj_scale_2[e].view(torch.float32)
        assert torch.equal(pair, stream[f"{p}.gate_proj.weight_scale_2"].repeat(2))
    dense = _load(cfg, state, rank, tp).model.layers[0].mlp
    assert dense.gate_up_proj.quant_format.name == "fp8_tensor"
    assert dense.down_proj.quant_format.name == "fp8_tensor"


def _dequantized_routed(experts, flat, weights, ids):
    """The routed output of the fp32 weights the experts stand for, bf16 activations."""
    want = torch.zeros(flat.shape, device=flat.device)
    for t in range(flat.shape[0]):
        for slot in range(ids.shape[1]):
            p = f"model.layers.1.mlp.experts.{int(ids[t, slot])}"
            x = flat[t].float()
            gate, up, down = (experts[f"{p}.{n}"].to(flat.device) for n in ("gate_proj", "up_proj", "down_proj"))
            want[t] += float(weights[t, slot]) * (down @ (F.silu(gate @ x) * (up @ x)))
    return want


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_the_routed_output_is_the_dequantized_experts():
    torch.manual_seed(1)
    cfg = _reduced_config()
    state, experts = _fabricate(cfg)
    moe = _load(cfg, state).model.layers[1].mlp
    flat = torch.randn(9, cfg.hidden_size).bfloat16()
    weights, ids, _ = moe.gate(flat)
    got = moe._experts_4bit(flat, weights, ids, skip_invalid=False)
    # NVFP4 rounds the activations to 4 bits per element (the weights here are exact): ~10% on random data
    assert _rel(got, _dequantized_routed(experts, flat, weights, ids)) < 0.2


def _blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    from mstar.utils.fused_moe import fp4_flashinfer

    return fp4_flashinfer.unavailable_reason(torch.device("cuda")) is None


@pytest.mark.skipif(not _blackwell(), reason="FlashInfer's TRT-LLM FP4 MoE: B200 / B300 with flashinfer")
@pytest.mark.parametrize("tokens", [1, 7, 64, 300])
def test_blackwell_experts_kernel_matches_the_reference(tokens):
    """The checkpoint's experts, converted in place to the TRT-LLM kernel's layout at load, on GLM's own routing.

    Both round an activation block's e4m3 scale to nearest (a stored checkpoint's ``Nvfp4Format.for_stored``), so
    the kernel sits within its accumulation order of the CPU reference; both are ~0.16 from the fp32 weights'
    output on random data (a swapped gate/up or a misplaced scale is ~1.0 off)."""
    torch.manual_seed(2)
    cfg = _reduced_config()
    cfg.hidden_size, cfg.moe_intermediate_size, cfg.n_routed_experts = 512, 256, 8
    state, experts = _fabricate(cfg)
    moe = _load(cfg, state, device="cuda").model.layers[1].mlp
    assert moe._experts_kernel is not None
    reference = _load(cfg, state).model.layers[1].mlp
    flat = torch.randn(tokens, cfg.hidden_size).bfloat16()
    weights, ids, _ = reference.gate(flat)
    fp32 = _dequantized_routed(experts, flat, weights, ids)
    want = reference._experts_4bit(flat, weights, ids, skip_invalid=False)
    got = moe._experts_4bit(flat.cuda(), weights.cuda(), ids.cuda(), skip_invalid=False).cpu()
    assert _rel(got, want) < 0.05, _rel(got, want)
    assert _rel(got, fp32) <= 1.05 * _rel(want, fp32), (_rel(got, fp32), _rel(want, fp32))
    assert F.cosine_similarity(got.float().flatten(), fp32.flatten(), dim=0) > 0.98
