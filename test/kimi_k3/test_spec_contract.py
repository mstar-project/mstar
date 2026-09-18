"""Speculative mode of the Kimi K3 model on CPU (``model_kwargs.speculative_tokens``): the acceptance
resource, the pool's prefix blocks, the KDA manager's width, the submodule's k + 1 token rows."""
import torch

from mstar.engine.resources import SPEC_ACCEPTANCE, DeltaNetGeometry, resolve_spec_dependencies
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.runner import topo_sort
from mstar.model.kimi_k3.config import KDA_ATTN, KDA_STATE, MLA_ATTN, MLA_KV, SAMPLER, SPEC
from mstar.model.registry import get_model_class
from mstar.model.submodule_base import ARNodeInputs


def test_speculative_mode_declares_the_acceptance_resource_and_the_prefix_blocks(tiny_dir):
    model = get_model_class("kimi_k3")(model_path_hf=str(tiny_dir), max_output_tokens=16, speculative_tokens=3)
    specs = resolve_spec_dependencies(model.get_node_resources())
    assert SPEC == SPEC_ACCEPTANCE and set(specs) == {MLA_KV, MLA_ATTN, KDA_STATE, KDA_ATTN, SAMPLER, SPEC}
    assert specs[SPEC].num_speculative == 3
    st = specs[KDA_STATE].config
    assert st.blocks["spec_prefix"].shape == (4, 768) and st.blocks["spec_prefix"].dtype is torch.bfloat16
    assert st.blocks["spec_g"].shape == (4, 8, 32) and st.blocks["spec_beta"].shape == (4, 8)
    assert st.blocks["spec_len"].shape == (1,) and st.blocks["spec_len"].dtype is torch.int32
    assert DeltaNetGeometry.speculative_tokens_of(st.blocks) == 3
    st.shard(2)
    assert st.blocks["spec_prefix"].shape == (4, 384) and st.blocks["spec_g"].shape == (4, 4, 32)
    assert st.blocks["spec_beta"].shape == (4, 4) and st.blocks["spec_len"].shape == (1,)
    attn = build_resource(specs[KDA_ATTN], EngineResourceInfo(device=torch.device("cpu"),
                                                               dependencies={KDA_STATE: specs[KDA_STATE]}))
    assert attn.speculative_tokens == 3
    # the cache plans after the verdicts (independent resources are otherwise ordered by name, which
    # would put mla_kv first); the specs answer depends_on like the resources they build
    assert specs[MLA_KV].depends_on() == {SPEC}
    order = topo_sort(specs)
    assert order.index(SPEC) < order.index(MLA_KV) < order.index(MLA_ATTN)
    # without the kwarg nothing changes
    plain = resolve_spec_dependencies(get_model_class("kimi_k3")(model_path_hf=str(tiny_dir)).get_node_resources())
    assert SPEC not in plain and "spec_prefix" not in plain[KDA_STATE].config.blocks


def test_speculative_submodule_rows_carry_k_plus_one_ids(tiny_dir):
    model = get_model_class("kimi_k3")(model_path_hf=str(tiny_dir), speculative_tokens=3)
    sub = model.get_submodule("LLM", device="cpu")
    assert sub.speculative_tokens == 3 and sub.k1 == 4
    # a decode row carries its bonus token (one id) and spans k + 1 tokens, with or without a draft
    inp = sub.prepare_inputs("decode", None, {"text_inputs": [torch.tensor([42])]})
    assert inp.input_ids.tolist() == [42] and inp.input_seq_len == 4
    step = sub.declare_step("decode", ["a"], [inp])
    assert SPEC in step.steps and step.segments[0].span == 4
    sub.cuda_graphs = True
    cfg = sub.get_cuda_graph_configs(torch.device("cpu"))[0]
    assert cfg.single_request_inputs.input_seq_len == 4 and cfg.single_request_inputs.input_ids.numel() == 1
    assert cfg.get_total_tokens(8) == [32]
    # the stub draft repeats the bonus token
    bonus = torch.tensor([[5], [9]])
    assert sub._draft(bonus).tolist() == [[5, 5, 5], [9, 9, 9]]


def test_draft_mode_declares_the_draft_cache_and_rows_carry_one_id(tiny_dir, tmp_path):
    import sys
    sys.path.insert(0, "/scratch/m000137-pm06/atj10/kimi_k3_mstar/bench")
    from make_tiny_dspark import make_tiny_dspark

    from mstar.model.kimi_k3.dspark.model import DSPARK_ATTN, DSPARK_KV

    make_tiny_dspark(str(tiny_dir), str(tmp_path / "draft"), layers=2, aux=(1, 3, 5))
    model = get_model_class("kimi_k3")(model_path_hf=str(tiny_dir), speculative_tokens=3,
                                       speculative_draft=str(tmp_path / "draft"))
    specs = resolve_spec_dependencies(model.get_node_resources())
    assert {DSPARK_KV, DSPARK_ATTN, SPEC} <= set(specs)
    dk = specs[DSPARK_KV].config
    assert dk.num_layers == 2 and dk.latent_dim == 128 + 32 and specs[DSPARK_KV].depends_on() == {SPEC}
    assert specs[DSPARK_ATTN].depends_on() == {DSPARK_KV}
    order = topo_sort(specs)
    assert order.index(SPEC) < order.index(DSPARK_KV) < order.index(DSPARK_ATTN)
    sub = model.get_submodule("LLM", device="cpu")
    assert sub.draft is not None and sub.draft.cfg.target_layer_ids == (1, 3, 5)
    # a decode row carries its bonus token alone and spans k + 1 tokens
    inp = sub.prepare_inputs("decode", None, {"text_inputs": [torch.tensor([42])]})
    assert inp.input_ids.tolist() == [42] and inp.input_seq_len == 4
    step = sub.declare_step("decode", ["a"], [inp])
    assert step.segments[0].span == 4 and DSPARK_KV in step.steps and DSPARK_ATTN in step.steps
    dattn = step.steps[DSPARK_ATTN]
    assert dattn.context_only and not dattn.causal and dattn.segments[0].span == 3
    pre = sub.declare_step("prefill", ["a"], [sub.prepare_inputs("prefill", None, {"text_inputs": [torch.arange(9)]})])
    assert pre.segments[0].span == 9 and DSPARK_KV in pre.steps and DSPARK_ATTN not in pre.steps
    sub.cuda_graphs = True
    cfg = sub.get_cuda_graph_configs(torch.device("cpu"))[0]
    assert cfg.single_request_inputs.input_ids.numel() == 1 and cfg.single_request_inputs.input_seq_len == 4


def test_unpack_cuts_at_a_stop_token_unless_eos_is_ignored(tiny_dir):
    from types import SimpleNamespace

    model = get_model_class("kimi_k3")(model_path_hf=str(tiny_dir), speculative_tokens=3)
    sub = model.get_submodule("LLM", device="cpu")
    stop = next(iter(model.config.stop_token_ids))
    tokens = torch.tensor([[5, 6, stop, 8], [9, 10, 11, 12]])
    verdicts = [SimpleNamespace(accepted=3, tokens=tokens[0].tolist()), SimpleNamespace(accepted=2, tokens=tokens[1].tolist())]
    sub._acceptance = SimpleNamespace(verdicts_for=lambda rids: verdicts)
    static = {"spec_tokens": tokens, "spec_accepted": torch.tensor([3, 2]), "next_inputs": torch.tensor([[8], [11]])}
    info = {"a": SimpleNamespace(resource_configs={SAMPLER: SimpleNamespace(ignore_eos=False)}),
            "b": SimpleNamespace(resource_configs={SAMPLER: SimpleNamespace(ignore_eos=False)})}
    out = sub.unpack_packed_outputs(static, ["a", "b"], [4, 4], [], info)
    assert out["a"]["new_token"][0].tolist() == [5, 6, stop]  # cut after the stop token, the bonus dropped
    assert out["b"]["new_token"][0].tolist() == [9, 10, 11]  # accepted 2 + bonus, no stop token
    assert out["a"]["text_inputs"][0].tolist() == [8]
    info["a"].resource_configs[SAMPLER].ignore_eos = True
    out = sub.unpack_packed_outputs(static, ["a", "b"], [4, 4], [], info)
    assert out["a"]["new_token"][0].tolist() == [5, 6, stop, 8]  # ignore_eos: the whole accepted run
