"""M0 scaffold tests for Kimi-K2.7 (text backbone).

Dummy mode: the model is built via ``object.__new__`` (no tokenizer, no weights,
no GPU) and only the ``Model`` contract is exercised — the graph, engine types,
KV-cache dims, and the prefill→decode→done state machine. This validates the
serving plumbing in isolation before any MLA/MoE compute exists, exactly as
``docs/adding_models.rst`` prescribes for a new model.
"""
import sys

sys.path.insert(0, ".")

from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.engine.base import EngineType
from mstar.graph.base import Loop
from mstar.model.kimi_k2_7.config import KimiK2Config
from mstar.model.kimi_k2_7.kimi_model import KimiK2Model


def _make_model() -> KimiK2Model:
    model = object.__new__(KimiK2Model)
    model.config = KimiK2Config.reduced()
    model._submodule_cache = {}
    return model


def test_kimi_graph_walks_and_engine_types():
    model = _make_model()

    walks = model.get_graph_walk_graphs()
    assert set(walks) == {"prefill", "decode"}
    assert isinstance(walks["decode"], Loop)
    assert walks["decode"].name == "decode_loop"

    assert model.get_node_engine_types() == {"LLM": EngineType.KV_CACHE}


def test_kimi_kv_cache_config_matches_reduced_mla_dims():
    model = _make_model()
    cfg = model.config

    kv = model.get_kv_cache_config()
    assert len(kv) == 1
    (kv,) = kv

    assert kv.num_layers == cfg.num_hidden_layers == 2
    # Naive/materialized MLA: KV heads == query heads.
    assert kv.num_kv_heads == cfg.num_attention_heads == 4
    assert kv.num_qo_heads == cfg.num_attention_heads == 4
    # M6 FlashInfer-SM90 mitigation: q/k/v are zero-padded from qk_head_dim (24)
    # up to the smallest supported head_dim {64,128,256} >= qk_head_dim, so the
    # paged cache stores head_dim == padded_head_dim == 64 (not the raw 24, which
    # the Hopper prefill kernel static_asserts against).
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
    # M6: get_submodule is the real meta->to_empty->load_weights build, but it
    # returns None in dummy mode when no checkpoint is resolvable. _make_model sets
    # no model_path_hf, so _resolve_checkpoint() -> None -> dummy mode, letting the
    # modular graph tests run without a GPU or weights.
    assert getattr(model, "model_path_hf", None) is None
    assert model.get_submodule("LLM") is None
