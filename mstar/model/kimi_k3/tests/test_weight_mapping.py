"""Every tensor of the K3.1 Pro / K3 release lands on the port's parameters, at the release's
shapes, at TP1 and on TP8 ranks, through the real loader; every parameter gets one.

``fixtures/k31_tensors.json.gz`` holds the release's non-expert tensors and expert 0's (name,
dtype, shape) from the safetensors headers at the pinned revision; its 896 experts expand to the
index's 497,340 names exactly (checked when the fixture was made, and by the count here). The
model is built on the meta device, so nothing is allocated; the experts stream as the bf16
tensors their MXFP4 packing stands for (what a dequantized checkpoint holds)."""
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


@pytest.mark.parametrize("dsa_long_context", [False, True])
def test_every_release_name_maps(dsa_long_context):
    cfg = KimiK3ModelConfig.k31(dsa_long_context=dsa_long_context,
                                max_seq_len=8192 if dsa_long_context else 2048)
    names = list(_tensors())
    params = {n for n, _ in _build(cfg).named_parameters()}
    out = resolve_index_names(names, cfg, params)
    assert len(out["skipped"]) == 168  # vision tower + projector
    assert len(out["dropped"]) == (0 if dsa_long_context else 120)  # the indexer, unbuilt
    # the MXFP4 scales land on the storage the quantization API's mxfp4 format gives the experts
    unmapped = [line for line in out["unmapped"]
                if not re.search(r"\.weight_scale -> .*\.experts\.(gate_up|down)_proj_scale$", line)]
    assert not unmapped, unmapped[:5]
    assert len(out["unmapped"]) == 92 * 896 * 3
    targets = {line.split(" -> ")[1] for line in out["loaded"]}
    assert params - targets == set()


def _stream(tensors: dict, bf16_experts: bool = True):
    """Meta tensors at the release's shapes; MXFP4 experts as the bf16 weights they pack."""
    for name, (dtype, shape) in tensors.items():
        if bf16_experts and name.endswith(".weight_scale") and ".experts." in name:
            continue
        if bf16_experts and name.endswith(".weight_packed"):
            name, dtype, shape = name[: -len("_packed")], "BF16", [shape[0], 2 * shape[1]]
        yield name, torch.empty(shape, dtype=_DTYPES[dtype], device="meta")


@pytest.mark.parametrize("variant, rank, tp", [
    ("k31", 0, 1), ("k31", 0, 8), ("k31", 7, 8), ("k3", 3, 8), ("k31_long", 5, 8), ("k31_ep", 6, 8),
])
def test_release_shapes_load_strictly(variant, rank, tp):
    """The loaders take the release's shapes (A_log's padding, the three convs, the stacked
    experts, each rank's slices) and fill every parameter; strict loading raises otherwise."""
    cfg = {
        "k31": KimiK3ModelConfig.k31(),
        "k3": KimiK3ModelConfig.k3(),
        "k31_long": KimiK3ModelConfig.k31(dsa_long_context=True, max_seq_len=8192),
        "k31_ep": KimiK3ModelConfig.k31(moe_expert_parallel=True),
    }[variant]
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


def test_mxfp4_experts_need_the_quantization_format():
    cfg = KimiK3ModelConfig.k31()
    with pytest.raises(ValueError, match="mxfp4 experts format"):
        load_kimi_k3_weights(_build(cfg), _stream(_tensors(), bf16_experts=False), cfg)


def test_a_log_padding_is_cut_and_checked():
    from mstar.model.kimi_k3.weight_loader import _heads_of_a_log

    padded = torch.cat([torch.arange(1.0, 5.0), torch.zeros(4)])
    assert torch.equal(_heads_of_a_log("x", padded, 4), torch.arange(1.0, 5.0))
    with pytest.raises(ValueError, match="zero padding"):
        _heads_of_a_log("x", torch.ones(8), 4)
