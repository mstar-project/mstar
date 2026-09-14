import sys

sys.path.insert(0, ".")

from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.engine.resources import AttentionSpec, KVSpec, PositionSpec, SamplerSpec
from mstar.graph.base import Loop
from mstar.model.kimi_k2_7.config import KimiK2Config
from mstar.model.kimi_k2_7.kimi_model import KimiK2Model


def _make_model() -> KimiK2Model:
    model = object.__new__(KimiK2Model)
    model.config = KimiK2Config.reduced()
    model._submodule_cache = {}
    return model


def test_kimi_graph_walks():
    model = _make_model()

    walks = model.get_graph_walk_graphs()
    assert set(walks) == {"prefill", "decode"}
    assert isinstance(walks["decode"], Loop)
    assert walks["decode"].name == "decode_loop"


def test_kimi_declares_one_resource_of_each_kind_on_the_llm_node():
    specs = _make_model().get_node_resources()
    assert [type(spec) for spec in specs] == [
        KVSpec, AttentionSpec, SamplerSpec, PositionSpec
    ]
    assert all(spec.nodes == {"LLM"} for spec in specs)


def test_kimi_kv_cache_config_matches_reduced_mla_dims():
    model = _make_model()
    cfg = model.config

    kv, _attn = model._kv_and_attn_specs()

    assert kv.num_layers == cfg.num_hidden_layers == 2
    assert kv.num_kv_heads == cfg.num_attention_heads == 4
    assert kv.num_qo_heads == cfg.num_attention_heads == 4
    # FlashInfer-SM90 requires padded_head_dim, not raw qk_head_dim.
    assert cfg.qk_head_dim == cfg.qk_nope_head_dim + cfg.qk_rope_head_dim == 24
    assert kv.head_dim == cfg.padded_head_dim == 64
    assert kv.max_seq_len == cfg.max_position_embeddings


def test_kimi_prefill_transitions_to_decode():
    model = _make_model()
    metadata = CurrentForwardConductorMetadata(
        input_modalities=["text"],
        output_modalities=["text"],
        graph_walk="prefill",
        is_prefill=True,
    )

    result = model.get_partition_forward_pass_args(
        partition_name="default",
        partition_metadata=metadata,
        persist_signals={"new_token": []},
    )

    assert result.full_metadata.graph_walk == "decode"
    assert result.full_metadata.is_prefill is False
    assert result.step_metadata["is_prefill"] is False
    assert result.request_done is False


def test_kimi_decode_completion_marks_done():
    model = _make_model()
    metadata = CurrentForwardConductorMetadata(
        input_modalities=["text"],
        output_modalities=["text"],
        graph_walk="decode",
        is_prefill=False,
    )

    result = model.get_partition_forward_pass_args(
        partition_name="default",
        partition_metadata=metadata,
        persist_signals={},
    )

    assert result.request_done is True
    assert result.full_metadata.kwargs["decode_finished"] is True


def test_kimi_get_submodule_is_dummy_mode():
    model = _make_model()
    assert getattr(model, "model_path_hf", None) is None
    assert model.get_submodule("LLM") is None
