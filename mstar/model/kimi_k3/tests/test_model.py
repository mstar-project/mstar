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
    "kimi_k31_tp8_32k": KimiK31Model, "kimi_k31_tp8_40k": KimiK31Model,
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
    """One decode graph per batch size up to the KDA slots, beside the packed prefill capture (empty
    until the serving config names its buckets); with dsa_long_context two decode graphs, dense and
    selecting (the DSA resource selects on the device), and no prefill; none on MLA's torch fallback
    (its plan holds the capture's pages)."""
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

    decode, prefill = configs(KimiK3ModelConfig.reduced())
    assert prefill.capture_graph_walk == "prefill" and not prefill.get_total_tokens(1)
    assert isinstance(decode, BatchedAcceleratorGraphConfig) and decode.capture_graph_walk == "decode"
    assert decode.capture_batch_sizes == [1, 2, 4] and not decode.compile
    assert decode.additional_key_info is None
    dense, dsa = configs(KimiK3ModelConfig.reduced(dsa_long_context=True))
    assert (dense.additional_key_info, dsa.additional_key_info) == ("dense", "dsa")
    assert all(c.capture_graph_walk == "decode" and c.capture_batch_sizes == [1, 2, 4] for c in (dense, dsa))
    assert configs(KimiK3ModelConfig.reduced(), uses_kernel=False) == []


def test_dsa_decode_replays_the_dense_capture_until_a_row_nears_index_topk():
    """Under dsa_long_context a decode batch's capture key is "dense" while every row is
    DSA_DENSE_MARGIN tokens short of index_topk keys (a row's context lags the step in flight), else
    "dsa"; an unknown row takes "dsa". The declared step follows the capture the engine leased."""
    from mstar.engine.resources.step import BucketKey, SlotLease
    from mstar.model.kimi_k3.components.causal_lm import KimiK3ForCausalLM
    from mstar.model.kimi_k3.config import DSA
    from mstar.model.kimi_k3.submodules import DSA_DENSE_MARGIN, KimiK3LLMSubmodule
    from mstar.model.submodule_base import ARNodeInputs

    cfg = KimiK3ModelConfig.reduced(dsa_long_context=True)
    sub = KimiK3LLMSubmodule(KimiK3ForCausalLM(cfg), cfg)
    edge = cfg.index_topk - DSA_DENSE_MARGIN
    sub.request_state("a").add("context", edge - 1)  # its next token is the edge-th
    sub.request_state("b").add("context", 5)
    assert sub.cg_key_info("decode", {"a": None, "b": None}) == "dense"
    sub.request_state("a").add("context", edge)
    assert sub.cg_key_info("decode", {"a": None, "b": None}) == "dsa"
    assert sub.cg_key_info("decode", {"b": None, "new": None}) == "dsa"
    assert sub.cg_key_info("prefill", {"b": None}) is None
    one = [ARNodeInputs(input_ids=torch.zeros(1, dtype=torch.long), input_seq_len=1)]
    leased = SlotLease(slot=0, bucket=BucketKey(graph_walk="decode", cg_key_info="dsa", bs=1, num_tokens=1))
    step = sub.declare_step("decode", ["b"], one, slot_lease=leased)
    assert step.cg_key_info == "dsa" and not step.steps[DSA].dense
    step = sub.declare_step("decode", ["b"], one)
    assert step.cg_key_info == "dense" and step.steps[DSA].dense
    plain = KimiK3ModelConfig.reduced()
    assert KimiK3LLMSubmodule(KimiK3ForCausalLM(plain), plain).cg_key_info("decode", {"b": None}) is None


@pytest.mark.parametrize("name, rows", [("kimi_k31_tp8", 64), ("kimi_k31_ep8", 64), ("kimi_k3_tp8", 64),
                                        ("kimi_k31_tp8_c128", 128), ("kimi_k31_tp8_longctx", 16),
                                        ("kimi_k31_tp8_32k", 64), ("kimi_k31_tp8_40k", 64)])
def test_shipped_configs_decode_as_many_rows_as_they_admit(name, rows):
    """The decode cap (the largest graph that fits the KDA slots) equals the config's admission:
    a lower cap queues the rest of a c=N load behind it (the MoE-gap audit's -38% on Qwen3-Omni)."""
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
    assert max(graphs[0].capture_batch_sizes) == rows


def test_decode_ladder_refuses_a_disordered_list():
    from mstar.model.kimi_k3.components.causal_lm import KimiK3ForCausalLM
    from mstar.model.kimi_k3.submodules import KimiK3LLMSubmodule

    cfg = KimiK3ModelConfig.reduced()
    with pytest.raises(ValueError, match="ascend"):
        KimiK3LLMSubmodule(KimiK3ForCausalLM(cfg), cfg, decode_capture_batch_sizes=(1, 8, 4))


def test_prefill_is_captured_at_the_named_buckets_within_index_topk_only():
    """The serving config's ``prefill_token_buckets`` (which the engine applies) give the packed
    prefill capture its token counts, at the row counts the KDA slots hold; without them it captures
    nothing, and under dsa_long_context (DSA's selecting prefill) there is none to name."""
    from types import SimpleNamespace

    from mstar.engine.accelerator_graph_config import PackedAcceleratorGraphConfig, apply_token_ladders
    from mstar.model.kimi_k3.components.causal_lm import KimiK3ForCausalLM
    from mstar.model.kimi_k3.config import ATTN, KDA_STATE
    from mstar.model.kimi_k3.submodules import KimiK3LLMSubmodule

    def graphs(cfg, ladder=None, **kwargs):
        sub = KimiK3LLMSubmodule(KimiK3ForCausalLM(cfg), cfg, **kwargs)
        pool = SimpleNamespace(config=SimpleNamespace(usable_slots=4))
        sub.node_resources = {ATTN: SimpleNamespace(uses_kernel=True), KDA_STATE: pool}
        configs = sub.get_accelerator_graph_configs(torch.device("meta"))
        if ladder is not None:
            apply_token_ladders("LLM", configs, {"prefill": ladder}, lambda walk, rows: None,
                                lambda config: config.capture_batch_sizes)
        return [g for g in configs if isinstance(g, PackedAcceleratorGraphConfig)]

    plain = KimiK3ModelConfig.reduced()
    (prefill,) = graphs(plain, [512, 64, 256])
    assert prefill.capture_graph_walk == "prefill" and prefill.capture_token_lengths == {1: [64, 256, 512]}
    assert prefill.capture_batch_sizes == [1, 2, 4], "rows past the KDA slots"
    assert not prefill.caps_eager_batch_size, "a step with more rows than a capture must still run"
    # no row runs past the context the deployment serves (the reduced model's 64 tokens)
    assert prefill.get_total_tokens(1) == [64] and prefill.get_total_tokens(4) == [64, 256]
    assert all(not g.get_total_tokens(bs) for g in graphs(plain) for bs in (1, 2, 4))
    assert not graphs(KimiK3ModelConfig.reduced(dsa_long_context=True), [64])
    with pytest.raises(ValueError, match="ascend"):
        KimiK3LLMSubmodule(KimiK3ForCausalLM(plain), plain, prefill_capture_batch_sizes=(4, 2))


def test_a_prefill_row_off_a_64_key_boundary_runs_eager():
    """Past a mid-way cut the MLA plan peels a row into single-query rows (Kimi's 12 heads a rank),
    which no packed capture planned: such a step must not replay. Rows at 0, on a page, and decode
    rows replay; so does every row where the attention peels none."""
    from types import SimpleNamespace

    from mstar.model.kimi_k3.components.causal_lm import KimiK3ForCausalLM
    from mstar.model.kimi_k3.config import ATTN
    from mstar.model.kimi_k3.submodules import KimiK3LLMSubmodule
    from mstar.model.submodule_base import ARNodeInputs

    cfg = KimiK3ModelConfig.reduced()
    sub = KimiK3LLMSubmodule(KimiK3ForCausalLM(cfg), cfg)
    sub.node_resources = {ATTN: SimpleNamespace(_splits_short_rows=True)}

    def rows(*starts):
        return [ARNodeInputs(input_seq_len=100, chunk_start=s) for s in starts]

    assert sub.can_replay("prefill", rows(0, 128, 896))
    assert not sub.can_replay("prefill", rows(0, 200))
    assert sub.can_replay("decode", [ARNodeInputs(input_seq_len=1, chunk_start=0)])
    sub.node_resources = {ATTN: SimpleNamespace(_splits_short_rows=False)}
    assert sub.can_replay("prefill", rows(200))


def test_a_prompt_is_cut_on_64_token_boundaries():
    """K3.1 declares its prefill chunks' alignment: a chunk cut mid-way ends on a 64-token boundary, so
    the next replays a packed capture (no peeled MLA rows) and keeps its KDA snapshot."""
    from mstar.engine.resources.attn.flashinfer_mla import ROW_ALIGN
    from mstar.engine.resources.linear_attn.kda_triton import CONFIG

    (policy,) = KimiK31Model("moonshotai/Kimi-K3", tokenizer_mode="byte").get_chunking_policies()["LLM"]
    assert policy.chunk_align == ROW_ALIGN == 64
    assert ROW_ALIGN % CONFIG.chunk == 0, "a cut must end one of KDA's chunks"


def test_the_small_m_gemm_table_loads_in_the_process_that_builds_the_submodule(tmp_path, monkeypatch):
    """The conductor pickles the model to each worker without running __init__ (spawn): a table loaded
    there never reached the workers, whose projections all ran cuBLAS (B300 profile 20261108T003606Z).
    The worker's submodule build loads it."""
    import json
    import pickle

    from mstar.utils import small_m_gemm

    path = tmp_path / "table.json"
    path.write_text(json.dumps({"capability": [10, 3], "entries": [
        {"n": 6288, "k": 7168, "out": "bf16", "m": 8, "backend": "fi_tgv"}]}))
    monkeypatch.setattr(small_m_gemm, "_TABLES", {})
    model = KimiK31Model("moonshotai/Kimi-K3", tokenizer_mode="byte", small_m_gemm_table=str(path))
    worker = pickle.loads(pickle.dumps(model))  # what the conductor hands a spawned worker
    small_m_gemm._TABLES.clear()  # a fresh process's dispatcher
    monkeypatch.setattr(worker, "_resolve_checkpoint", lambda: None)  # no weights: the build stops there
    assert worker.get_submodule("LLM") is None
    assert small_m_gemm._TABLES == {(10, 3): {(6288, 7168, "bf16"): [(8, "fi_tgv")]}}


def test_a_small_m_gemm_table_missing_on_the_host_is_a_warning(tmp_path, monkeypatch, caplog):
    from mstar.utils import small_m_gemm

    monkeypatch.setattr(small_m_gemm, "_TABLES", {})
    model = KimiK31Model("moonshotai/Kimi-K3", tokenizer_mode="byte", small_m_gemm_table=str(tmp_path / "nope.json"))
    monkeypatch.setattr(model, "_resolve_checkpoint", lambda: None)
    model.get_submodule("LLM")
    assert not small_m_gemm._TABLES and "not on this host" in caplog.text
