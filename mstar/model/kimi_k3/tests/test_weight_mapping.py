"""Every tensor of the K3.1 Pro / K3 release lands on the port's parameters, at the release's
shapes, at TP1 and on TP8 ranks, through the real loader; every parameter gets one.

``fixtures/k31_tensors.json.gz`` holds the release's non-expert tensors and expert 0's (name,
dtype, shape) from the safetensors headers at the pinned revision; its 896 experts expand to the
index's 497,340 names exactly (checked when the fixture was made, and by the count here). The
model is built on the meta device, so nothing is allocated. The MXFP4 experts load as stored through
the quantization API's storage (a scope given the release's header scan, as the engine's would read
it from the files), or stream as the bf16 tensors their packing stands for (what a dequantized
checkpoint holds)."""
import gzip
import json
import re
from pathlib import Path

import pytest
import torch

from mstar.distributed.communication import CommGroup
from mstar.model.kimi_k3.components.causal_lm import KimiK3ForCausalLM
from mstar.model.kimi_k3.config import KimiK3ModelConfig
from mstar.model.kimi_k3.weight_loader import load_kimi_k3_weights, resolve_index_names

FIXTURE = Path(__file__).parent / "fixtures" / "k31_tensors.json.gz"
_EXPERT0 = re.compile(r"^(.*\.experts)\.0\.(.*)$")
_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "U8": torch.uint8}


def _release() -> dict:
    with gzip.open(FIXTURE) as f:
        return json.loads(f.read())


def _tensors(drop_indexer: bool = False) -> dict[str, tuple[str, list[int]]]:
    """The release's tensors, ``name -> (dtype, shape)``, every expert expanded."""
    rel = _release()
    out = {}
    for name, meta in rel["tensors"].items():
        if drop_indexer and rel["k3_lacks"] in name:
            continue
        m = _EXPERT0.match(name)
        if m:
            for e in range(rel["num_experts"]):
                out[f"{m.group(1)}.{e}.{m.group(2)}"] = tuple(meta)
        else:
            out[name] = tuple(meta)
    return out


def test_fixture_expands_to_the_release_index():
    rel = _release()
    assert len(_tensors()) == rel["total"] == 497_340
    assert len(_tensors(drop_indexer=True)) == rel["k3_total"] == 497_220


def _build(cfg: KimiK3ModelConfig, rank: int = 0, tp: int = 1) -> KimiK3ForCausalLM:
    group = CommGroup.trivial() if tp == 1 else CommGroup(rank, rank, list(range(tp)))
    with torch.device("meta"):
        return KimiK3ForCausalLM(cfg, comm_group=group).to(torch.bfloat16)


def _release_scan(drop_indexer: bool = False):
    """The release's ``CheckpointQuant``, as the quantization API reads it from the headers."""
    from mstar.quantization.checkpoint import CheckpointQuant, TensorInfo

    quant = json.loads((FIXTURE.parent / "k31_config.json").read_text())["text_config"]["quantization_config"]
    tensors = {name: TensorInfo(dtype, tuple(shape)) for name, (dtype, shape) in _tensors(drop_indexer).items()}
    return CheckpointQuant.from_tensors(tensors, quant)


def _scope(cfg: KimiK3ModelConfig, scan):
    """The LLM node's quantization scope with no ``quantization:`` block (``auto``: load as stored)."""
    from mstar.model.kimi_k3.kimi_k3_model import KimiK31Model
    from mstar.quantization.scope import NodeQuantization
    from mstar.quantization.spec import parse_quantization

    model = KimiK31Model("x", config_variant="reduced")
    model.config = cfg
    return NodeQuantization("LLM", parse_quantization(None).for_node("LLM"), model.get_quantization_roots(),
                            device="meta", checkpoint=scan)


@pytest.mark.parametrize("dsa_long_context", [False, True])
def test_every_release_name_maps(dsa_long_context):
    """Every release tensor lands on a parameter of the root as the scope prepares it: the MXFP4
    experts on their packed storage (``gate_up_proj_packed`` / ``_scale``, ``down_proj_*``)."""
    cfg = KimiK3ModelConfig.k31(dsa_long_context=dsa_long_context,
                                max_seq_len=8192 if dsa_long_context else 2048)
    module = _build(cfg)
    plan = _scope(cfg, _release_scan()).plan_for(module)
    experts = [p for p in plan.layers.values() if p.kind == "experts"]
    assert len(experts) == 92 and {(p.stored, p.runtime, p.action.value) for p in experts} == {
        ("mxfp4", "mxfp4", "direct")}
    params = {n for n, _ in module.named_parameters()}
    out = resolve_index_names(list(_tensors()), cfg, params)
    assert len(out["skipped"]) == 168  # vision tower + projector
    assert len(out["dropped"]) == (0 if dsa_long_context else 120)  # the indexer, unbuilt
    assert not out["unmapped"], out["unmapped"][:5]
    targets = {line.split(" -> ")[1] for line in out["loaded"]}
    assert params - targets == set()
    assert "model.layers.1.block_sparse_moe.experts.gate_up_proj_packed" in targets


def _stream(tensors: dict, bf16_experts: bool = True):
    """Meta tensors at the release's shapes; MXFP4 experts as the bf16 weights they pack."""
    for name, (dtype, shape) in tensors.items():
        if bf16_experts and name.endswith(".weight_scale") and ".experts." in name:
            continue
        if bf16_experts and name.endswith(".weight_packed"):
            name, dtype, shape = name[: -len("_packed")], "BF16", [shape[0], 2 * shape[1]]
        yield name, torch.empty(shape, dtype=_DTYPES[dtype], device="meta")


_VARIANTS = {
    "k31": KimiK3ModelConfig.k31,
    "k3": KimiK3ModelConfig.k3,
    "k31_long": lambda: KimiK3ModelConfig.k31(dsa_long_context=True, max_seq_len=8192),
    "k31_ep": lambda: KimiK3ModelConfig.k31(moe_expert_parallel=True),
}


@pytest.mark.parametrize("variant, rank, tp", [
    ("k31", 0, 1), ("k31", 0, 8), ("k31", 7, 8), ("k3", 3, 8), ("k31_long", 5, 8), ("k31_ep", 6, 8),
])
def test_release_shapes_load_strictly(variant, rank, tp):
    """bf16 experts (the dequantized cut): the loaders take the release's shapes (A_log's padding,
    the three convs, the stacked experts, each rank's slices) and fill every parameter."""
    cfg = _VARIANTS[variant]()
    module = _build(cfg, rank, tp)
    loaded = load_kimi_k3_weights(module, _stream(_tensors(drop_indexer=not cfg.dsa)), cfg)
    assert loaded == {n for n, _ in module.named_parameters()}
    layer = module.model.layers[1]
    assert layer.self_attn.forget_gate.A_log.shape == (96 // tp,)
    assert layer.self_attn.forget_gate.dt_bias.dtype == torch.float32
    experts = layer.block_sparse_moe.experts.gate_up_proj
    if variant == "k31_ep":
        assert experts.shape == (896 // tp, 2 * 3072, 3584)
    else:
        assert experts.shape == (896, 2 * 3072 // tp, 3584)


@pytest.mark.parametrize("variant, rank, tp", [("k31", 0, 1), ("k31", 7, 8), ("k3", 2, 8), ("k31_ep", 5, 8)])
def test_release_mxfp4_experts_load_as_stored(variant, rank, tp):
    """The release as it ships: its MXFP4 experts (packed e2m1 u8 and e8m0 scales) load into the
    quantization API's mxfp4 storage, each rank its intermediate slice (TP) or its whole experts
    (EP), and every parameter gets a tensor."""
    cfg = _VARIANTS[variant]()
    module = _build(cfg, rank, tp)
    with _scope(cfg, _release_scan(drop_indexer=not cfg.dsa)).activated():
        loaded = load_kimi_k3_weights(module, _stream(_tensors(drop_indexer=not cfg.dsa), bf16_experts=False), cfg)
    assert loaded == {n for n, _ in module.named_parameters()}
    block = module.model.layers[1].block_sparse_moe
    assert block.experts_packed.name == "mxfp4"
    e, inter = (896 // tp, 3072) if variant == "k31_ep" else (896, 3072 // tp)
    assert block.experts.gate_up_proj_packed.shape == (e, 2 * inter, 3584 // 2)
    assert block.experts.gate_up_proj_scale.shape == (e, 2 * inter, 3584 // 32)
    assert block.experts.down_proj_packed.shape == (e, 3584, inter // 2)
    assert block.experts.down_proj_scale.shape == (e, 3584, inter // 32)


@pytest.mark.parametrize("variant, rank", [("k31", 0), ("k31_ep", 5)])
def test_a_b300_rank_allocates_its_mxfp4_experts_packed(variant, rank):
    """A TP8 / EP8 rank's bytes as the scope prepares the root on meta, which is what ``materialize``'s
    ``to_empty`` then allocates: the experts packed, 168.4 GiB (as bf16 stacks 634 GiB, which ran every
    B300 rank out of memory at load in 04c), 187.1 GiB in all. With tp8's KV and KDA pools (~6.8 GiB;
    tp8_c128 ~13.5) that leaves ~70 GiB of a B300's 267.68 GiB for activations and graphs."""
    cfg = _VARIANTS[variant]()
    module = _build(cfg, rank, 8)
    _scope(cfg, _release_scan()).prepare_meta(module)
    sizes = {n: t.numel() * t.element_size() for n, t in [*module.named_parameters(), *module.named_buffers()]}
    assert not any(t.dtype == torch.bfloat16 for n, t in module.named_parameters() if ".experts." in n)
    weights = 92 * 896 * (2 * 3072 * 3584 + 3584 * 3072) // 8  # every rank an eighth, TP or EP
    assert sum(b for n, b in sizes.items() if ".experts." in n) == weights // 2 + weights // 32
    assert 187 < sum(sizes.values()) / 2**30 < 187.5


def test_mxfp4_experts_outside_the_scope_are_refused():
    cfg = KimiK3ModelConfig.k31()
    with pytest.raises(ValueError, match="quantization scope"):
        load_kimi_k3_weights(_build(cfg), _stream(_tensors(), bf16_experts=False), cfg)


def test_a_log_padding_is_cut_and_checked():
    from mstar.model.kimi_k3.weight_loader import _heads_of_a_log

    padded = torch.cat([torch.arange(1.0, 5.0), torch.zeros(4)])
    assert torch.equal(_heads_of_a_log("x", padded, 4), torch.arange(1.0, 5.0))
    with pytest.raises(ValueError, match="zero padding"):
        _heads_of_a_log("x", torch.ones(8), 4)
