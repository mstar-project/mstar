"""Kimi K3 / K3.1's model-level declarations on CPU without weights: resources per serving
config, the shipped configs through the deployment check, prompt limits, and the
quantization API's view of the LLM root."""
from pathlib import Path

import pytest
import torch
import yaml

from mstar.engine.resources.dsa.config import DSASpec
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.model.kimi_k3.config import KimiK3ModelConfig
from mstar.model.kimi_k3.kimi_k3_model import (
    QUANTIZABLE_CANDIDATES,
    KimiK3Model,
    KimiK31Model,
    quantization_keep_bf16,
)

ROOT = Path(__file__).resolve().parents[4]
CONFIGS = {
    "kimi_k31_tp8": KimiK31Model, "kimi_k31_ep8": KimiK31Model,
    "kimi_k31_tp8_longctx": KimiK31Model, "kimi_k3_tp8": KimiK3Model, "kimi_k31_tp8_c128": KimiK31Model,
}


def _specs(model) -> dict:
    return {s.resource_key: s for s in model.get_node_resources()}


def test_full_geometry_resources():
    specs = _specs(KimiK31Model("moonshotai/Kimi-K3.1-Pro"))
    kv: PagedKVConfig = specs["kv_cache"].config
    assert (kv.num_layers, kv.head_dim, kv.kv_lora_rank, kv.qk_rope_head_dim) == (24, 576, 512, 64)
    assert kv.max_seq_len == 2048 and kv.num_qo_heads == 96
    assert specs["kda_state"].config.num_layers == 69
    assert "dsa" not in specs
    long = _specs(KimiK31Model("moonshotai/Kimi-K3.1-Pro", dsa_long_context=True, max_seq_len=8192))
    assert isinstance(long["dsa"], DSASpec) and long["dsa"].config.topk == 2048
    assert long["kv_cache"].config.max_seq_len == 8192
    assert _specs(KimiK3Model("moonshotai/Kimi-K3"))["kv_cache"].config.max_seq_len == 32768


@pytest.mark.parametrize("name", list(CONFIGS))
def test_shipped_configs_pass_the_deployment_check(name):
    from mstar.validation import validate_deployment

    path = ROOT / "configs" / f"{name}.yaml"
    cfg = yaml.safe_load(path.read_text())
    model = CONFIGS[name]("moonshotai/Kimi-K3", **cfg.get("model_kwargs", {}))
    validate_deployment(model, cfg, str(path))
    assert cfg["max_seq_len"] == model.config.max_seq_len


def test_bool_knobs_refuse_strings():
    with pytest.raises(ValueError, match="true or false"):
        KimiK31Model("x", config_variant="reduced", dsa_long_context="false")


def test_prompt_limit_and_byte_mode():
    model = KimiK3Model("x", config_variant="reduced", tokenizer_mode="byte")
    out = model.process_prompt("hello", ["text"], ["text"])
    assert out.new_input_tensors["text_inputs"][0].tolist() == list(b"hello")
    assert model.prompt_limit() == model.config.context_limit - 2
    with pytest.raises(ValueError, match="at most"):
        model.process_prompt("x" * model.config.context_limit, ["text"], ["text"])
    assert model.postprocess(torch.tensor([104, 105]), "text") == b"hi"
    assert model.get_loop_caps(max_output_tokens=10) == {"decode_loop": 9}


def test_quantization_roots_cover_every_linear():
    """Every linear of the root is a candidate or kept bf16 and the routed experts are a candidate."""
    from mstar.distributed.communication import CommGroup
    from mstar.quantization.resolve import discover_layers

    model = KimiK31Model("x", config_variant="reduced")
    (root,) = model.get_quantization_roots()
    assert root.node == "LLM" and root.rules.prefix == "language_model."
    with torch.device("meta"):
        module = root.build(CommGroup.trivial())
    layers = discover_layers(module)
    picked = {path for path in layers if root.linears.selected(path)}
    assert "model.layers.1.block_sparse_moe.experts" in picked
    assert "model.layers.3.self_attn.o_proj" in picked  # an MLA layer's
    assert "model.layers.0.self_attn.o_proj" not in picked  # a KDA layer's
    assert "lm_head" not in picked


def test_quantization_patterns_name_real_modules():
    """Every declared pattern matches some module of the tiny root, so none is stale."""
    from fnmatch import fnmatchcase

    from mstar.model.kimi_k3.components.causal_lm import KimiK3ForCausalLM

    cfg = KimiK3ModelConfig.reduced(dsa_long_context=True)  # the indexer built
    with torch.device("meta"):
        names = [n for n, _ in KimiK3ForCausalLM(cfg).named_modules()]
    names += [n + ".experts" for n in names if n.endswith("block_sparse_moe")]

    def matches(name, pattern):
        parts, pats = name.split("."), pattern.split(".")
        return len(parts) == len(pats) and all(fnmatchcase(a, b) for a, b in zip(parts, pats, strict=True))

    for pattern in QUANTIZABLE_CANDIDATES + quantization_keep_bf16(cfg):
        assert any(matches(n, pattern) for n in names), pattern


def test_decode_capture_is_declared_where_it_can_replay():
    """One decode graph per batch size up to the KDA slots; none with dsa_long_context (host-side
    selection) or on MLA's torch fallback (its plan holds the capture's pages)."""
    from types import SimpleNamespace

    from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig
    from mstar.model.kimi_k3.components.causal_lm import KimiK3ForCausalLM
    from mstar.model.kimi_k3.config import ATTN, KDA_STATE
    from mstar.model.kimi_k3.submodules import KimiK3LLMSubmodule

    def configs(cfg, uses_kernel=True, slots=6):
        sub = KimiK3LLMSubmodule(KimiK3ForCausalLM(cfg), cfg)
        sub.node_resources = {ATTN: SimpleNamespace(uses_kernel=uses_kernel),
                              KDA_STATE: SimpleNamespace(config=SimpleNamespace(usable_slots=slots))}
        return sub.get_accelerator_graph_configs(torch.device("meta"))  # a device, not CPU

    (decode,) = configs(KimiK3ModelConfig.reduced())
    assert isinstance(decode, BatchedAcceleratorGraphConfig) and decode.capture_graph_walk == "decode"
    assert decode.capture_batch_sizes == [1, 2, 4] and not decode.compile
    assert configs(KimiK3ModelConfig.reduced(dsa_long_context=True)) == []
    assert configs(KimiK3ModelConfig.reduced(), uses_kernel=False) == []


@pytest.mark.parametrize("name, rows", [("kimi_k31_tp8", 64), ("kimi_k31_ep8", 64), ("kimi_k3_tp8", 64),
                                        ("kimi_k31_tp8_c128", 128), ("kimi_k31_tp8_longctx", 16)])
def test_shipped_configs_decode_as_many_rows_as_they_admit(name, rows):
    """The decode cap (the largest graph that fits the KDA slots) equals the config's admission:
    a lower cap queues the rest of a c=N load behind it (the MoE-gap audit's -38% on Qwen3-Omni).
    The long-context config runs eager, its cap the slots alone."""
    from types import SimpleNamespace

    from mstar.model.kimi_k3.components.causal_lm import KimiK3ForCausalLM
    from mstar.model.kimi_k3.config import ATTN, KDA_STATE
    from mstar.model.kimi_k3.submodules import KimiK3LLMSubmodule

    cfg = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
    kwargs = cfg.get("model_kwargs", {})
    slots = cfg["resources"]["kda_state"]["max_slots"] - 1  # less the sink
    assert slots == kwargs["kda_max_requests"] == cfg["max_concurrent_requests"] == rows
    model = CONFIGS[name]("moonshotai/Kimi-K3", **kwargs)
    sub = KimiK3LLMSubmodule(KimiK3ForCausalLM(KimiK3ModelConfig.reduced()), model.config,
                             decode_capture_batch_sizes=model.decode_capture_batch_sizes)
    pool = SimpleNamespace(config=SimpleNamespace(usable_slots=slots))
    sub.node_resources = {ATTN: SimpleNamespace(uses_kernel=True), KDA_STATE: pool}
    assert sub.max_batch_size("decode") == rows
    graphs = sub.get_accelerator_graph_configs(torch.device("meta"))
    if kwargs.get("dsa_long_context"):
        assert graphs == []  # eager: the slots alone cap it
        return
    assert max(graphs[0].capture_batch_sizes) == rows


def test_decode_ladder_refuses_a_disordered_list():
    from mstar.model.kimi_k3.components.causal_lm import KimiK3ForCausalLM
    from mstar.model.kimi_k3.submodules import KimiK3LLMSubmodule

    cfg = KimiK3ModelConfig.reduced()
    with pytest.raises(ValueError, match="ascend"):
        KimiK3LLMSubmodule(KimiK3ForCausalLM(cfg), cfg, decode_capture_batch_sizes=(1, 8, 4))
