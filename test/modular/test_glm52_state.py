"""GLM-5.2 on the state contract, against ``main``'s hand-written resources: the specs its declarations
build are ``main``'s but for three named differences, ``process_prompt`` lays out the latent array and the
seed, the prompt length rides the walk's step metadata, and a verify step's count edge cuts the verdict as
``main``'s per-request counters did."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    AttnBackend,
    DSAConfig,
    DSASpec,
    KVLayout,
    KVSpec,
    PagedKVConfig,
    PositionSpec,
    SamplerSpec,
)
from mstar.engine.resources.position.config import PositionConfig
from mstar.engine.state.adapters import node_resources
from mstar.model.glm52.config import INDEX_KEYS, KV, MTP_SEED, PROMPT_LEN, SAMPLER, Glm52ModelConfig
from mstar.model.glm52.glm52_model import Glm52Model
from mstar.model.glm52.submodules import Glm52LLMSubmodule
from mstar.model.submodule_base import END_PARTITION, ARNodeInputs

# main's resource keys that the declarations name otherwise: the positions take the latent array's
# position key, and the DSA resource the index key array's name
RENAMED = {"pos": "rope", "dsa": INDEX_KEYS}


def _model(**kwargs) -> Glm52Model:
    model = object.__new__(Glm52Model)
    model.config = Glm52ModelConfig(**kwargs)
    return model


def _main_specs(model: Glm52Model) -> list:
    """``main``'s ``Glm52Model.get_node_resources`` and ``Glm52ModelConfig.position_config``, as they stood
    before the port."""
    cfg, nodes = model.config, {"LLM"}
    num_kv_layers = cfg.num_hidden_layers + (1 if cfg.mtp_num_draft_tokens > 0 else 0)
    if cfg.mla_absorb:
        kv = PagedKVConfig(
            num_layers=num_kv_layers, num_kv_heads=1, head_dim=cfg.cache_latent_dim,
            max_seq_len=model.context_limit(), num_qo_heads=cfg.num_attention_heads, layout=KVLayout.MLA,
            kv_lora_rank=cfg.kv_lora_rank, qk_rope_head_dim=cfg.qk_rope_head_dim,
        )
        attn = AttentionConfig(kv_cache="kv", backend=AttnBackend.FLASHINFER_MLA, sm_scale=cfg.softmax_scale)
    else:
        kv = PagedKVConfig(
            num_layers=num_kv_layers, num_kv_heads=cfg.num_attention_heads, head_dim=cfg.padded_head_dim,
            max_seq_len=model.context_limit(), num_qo_heads=cfg.num_attention_heads,
        )
        attn = AttentionConfig(kv_cache="kv")
    specs = [
        KVSpec(resource_key="kv", nodes=nodes, config=kv),
        AttentionSpec(resource_key="attn", nodes=nodes, config=attn),
        PositionSpec(resource_key="pos", nodes=nodes, config=PositionConfig(
            kv_cache="kv", rotary_dim=cfg.qk_rope_head_dim, interleave=cfg.rope_interleave,
            rope_theta=cfg.rope_theta, rope_table_dtype=torch.float32,
        )),
        SamplerSpec(resource_key="sampler", nodes=nodes, vocab_size=cfg.vocab_size,
                    enable_min_p=bool(model.sampling_defaults.get("min_p"))),
    ]
    if cfg.dsa_long_context:
        specs.append(DSASpec(resource_key="dsa", nodes=nodes, config=DSAConfig(kv_cache="kv", topk=cfg.index_topk)))
    return specs


@pytest.mark.parametrize("kwargs", [
    {}, {"mla_absorb": False}, {"dsa_long_context": True}, {"mtp_num_draft_tokens": 3},
], ids=["absorbed", "naive", "dsa", "mtp"])
def test_the_declarations_build_main_s_specs_but_for_three_named_differences(kwargs):
    model = _model(**kwargs)
    built = {spec.resource_key: spec for spec in node_resources(model)}
    for spec in _main_specs(model):
        got = built.pop(RENAMED.get(spec.resource_key, spec.resource_key))
        assert type(got) is type(spec) and got.nodes == spec.nodes, spec.resource_key
        if isinstance(spec, SamplerSpec):
            assert (got.vocab_size, got.enable_min_p) == (spec.vocab_size, spec.enable_min_p)
        else:
            assert got.config == spec.config, f"{spec.resource_key}: the engine would build another resource"
    # the third: the seed's pool, where main kept fixed buffers on the submodule
    assert set(built) == ({MTP_SEED} if model.config.mtp_num_draft_tokens else set())


@pytest.mark.parametrize("k", [0, 3])
def test_process_prompt_lays_out_the_latent_array_and_the_seed(k):
    model = _model(mtp_num_draft_tokens=k, vocab_size=256)
    model._tokenizer_mode, model._tokenizer = "byte", None
    out = model.process_prompt("Hi", ["text"], ["text"])
    assert set(out.metadata["state_layout"]) == ({KV, MTP_SEED} if k else {KV})
    assert set(out.metadata["state_layout"][KV]) == {"main"}
    assert out.metadata[PROMPT_LEN] == 2
    assert PROMPT_LEN in model.INTERNAL_KWARGS, "a client's prompt_len would reach walk_plan"


def test_the_prompt_length_rides_the_walk_s_step_metadata():
    """Under MTP both steps of the walk carry the prompt length, and the stop check reads it there."""
    model = _model(mtp_num_draft_tokens=3)
    prefill, decode = model.walk_plan("default", ["text"], ["text"], {}, {PROMPT_LEN: 7})
    assert prefill.step_metadata[PROMPT_LEN] == decode.step_metadata[PROMPT_LEN] == 7
    plain, _ = _model().walk_plan("default", ["text"], ["text"], {}, {})
    assert PROMPT_LEN not in plain.step_metadata, "without MTP no step reads it"


def _submodule(k: int, **cfg) -> Glm52LLMSubmodule:
    language_model = torch.nn.Module()
    language_model.lm_head = torch.nn.Identity()
    return Glm52LLMSubmodule(language_model, Glm52ModelConfig(mtp_num_draft_tokens=k, **cfg))


def _info(walk: str, max_tokens: int, prompt_len: int, ignore_eos: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        request_id="r", graph_walk=walk, max_tokens=max_tokens, step_metadata={PROMPT_LEN: prompt_len},
        resource_configs={SAMPLER: SimpleNamespace(ignore_eos=ignore_eos)}, dynamic_loop_iter_counts={},
    )


def _main_cut(target, accepted, budget, emitted, eos, ignore_eos):
    """``main``'s cut of a verify step's verdict, from its per-request counters."""
    out = target[:min(accepted + 1, max(budget - emitted, 1))]
    if not ignore_eos:
        for j, token in enumerate(out):
            if token in eos:
                return out[:j + 1]
    return out


@pytest.mark.parametrize("ignore_eos", [True, False])
@pytest.mark.parametrize("accepted", [0, 2, 3])
@pytest.mark.parametrize("count", [1, 5, 9])
@pytest.mark.parametrize("max_tokens", [6, 10, 64])
def test_the_count_edge_cuts_a_verdict_as_main_did(max_tokens, count, accepted, ignore_eos):
    k, prompt_len = 3, 5
    sub = _submodule(k)
    eos = sub.config.eos_token_ids[0]
    target = [11, eos, 13, 14] if not ignore_eos else [11, 12, 13, 14]
    sub._mailboxes = {None: (torch.tensor([[*target, accepted]]), torch.ones(1, dtype=torch.long))}
    sub._verdicts = None, {"r": 0}
    info = _info("decode", max_tokens, prompt_len, ignore_eos)
    inp = ARNodeInputs(input_ids=torch.tensor([11]), input_seq_len=k + 1, kwargs={"count": count})
    out = sub.unpack_packed_outputs(static_output={}, request_ids=["r"], real_seq_lens=[k + 1], inputs=[inp],
                                    per_request_info={"r": info})["r"]
    budget = min(max_tokens, sub._context_limit() - prompt_len - 2 * (k + 1))
    want = _main_cut(target, accepted, budget, count, sub.config.eos_token_ids, ignore_eos)
    assert out["new_token"][0].tolist() == want
    assert int(out["count"][0]) == count + len(want)
    stops = sub.check_stop("r", info, out)
    assert bool(stops) == (count + len(want) >= budget or (not ignore_eos and want[-1] == eos))


@pytest.mark.parametrize("room", [0, 1, 2])
def test_an_mtp_prefill_at_the_context_limit_stops_after_its_token(room):
    """``main`` clamped the budget in the prefill's preprocess and stopped the prefill row at a budget of 1."""
    k = 3
    sub = _submodule(k)
    prompt_len = sub._context_limit() - 2 * (k + 1) - room
    info = _info("prefill", 64, prompt_len)
    out = {"new_token": [torch.tensor([42])], **sub.unpack_packed_outputs(
        static_output={}, request_ids=["r"], real_seq_lens=[prompt_len], inputs=[], per_request_info={"r": info},
    )["r"]}
    assert int(out["count"][0]) == 1
    assert sub.check_stop("r", info, out) == ({END_PARTITION} if room <= 1 else set())


def test_a_captured_and_padded_prefill_reads_no_prompt_length():
    """A capture's and a padded replay's dummy rows carry no step metadata, and preprocess reads none."""
    sub = _submodule(3)
    sub.get_device = lambda: torch.device("cpu")
    sub.kv = SimpleNamespace(positions=SimpleNamespace(position=lambda rid, label: 0))
    engine_inputs = SimpleNamespace(request_ids=["pad0", "pad1"], per_request_info={
        "pad0": SimpleNamespace(step_metadata={}), "pad1": SimpleNamespace(step_metadata={})})
    inputs = [ARNodeInputs(input_ids=torch.zeros(4, dtype=torch.long), input_seq_len=4) for _ in range(2)]
    assert sub.preprocess("prefill", engine_inputs, inputs)["input_ids"].shape == (8,)


def _reduced(monkeypatch, **fields) -> None:
    """``config_variant: reduced`` building a config with ``fields`` set."""
    cfg = Glm52ModelConfig.reduced()
    for name, value in fields.items():
        setattr(cfg, name, value)
    monkeypatch.setattr(Glm52ModelConfig, "reduced", staticmethod(lambda: cfg))


def test_glm52_refuses_an_indexer_that_rotates_its_own_way(monkeypatch):
    """The index keys rotate through the latent array's Rope, which holds the MLA's pairing."""
    _reduced(monkeypatch, mla_absorb=True, indexer_rope_interleave=not Glm52ModelConfig.reduced().rope_interleave)
    with pytest.raises(ValueError, match="indexer rotates as its MLA does"):
        Glm52Model(model_path_hf="", config_variant="reduced", tokenizer_mode="byte", dsa_long_context=True)


def test_glm52_refuses_dsa_whose_first_layer_reuses_a_pick(monkeypatch):
    """A SHARED layer attends the pick of the FULL layer before it, and a SHARED layer 0 has none."""
    from mstar.model.glm52.components.indexer import is_full_indexer_layer

    _reduced(monkeypatch, mla_absorb=True, index_skip_topk_offset=0, index_topk_freq=2)
    assert not is_full_indexer_layer(Glm52ModelConfig.reduced(), 0)
    with pytest.raises(ValueError, match="layer 0 to be a FULL indexer layer"):
        Glm52Model(model_path_hf="", config_variant="reduced", tokenizer_mode="byte", dsa_long_context=True)
