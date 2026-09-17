"""Speculative mode of the Kimi K3 model on CPU (``model_kwargs.speculative_tokens``): the acceptance
resource, the pool's prefix blocks, the KDA manager's width, the submodule's k + 1 token rows."""
import torch

from mstar.engine.resources import SPEC_ACCEPTANCE, DeltaNetGeometry, resolve_spec_dependencies
from mstar.engine.resources.base import EngineResourceInfo, build_resource
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
    # without the kwarg nothing changes
    plain = resolve_spec_dependencies(get_model_class("kimi_k3")(model_path_hf=str(tiny_dir)).get_node_resources())
    assert SPEC not in plain and "spec_prefix" not in plain[KDA_STATE].config.blocks


def test_speculative_submodule_rows_carry_k_plus_one_ids(tiny_dir):
    model = get_model_class("kimi_k3")(model_path_hf=str(tiny_dir), speculative_tokens=3)
    sub = model.get_submodule("LLM", device="cpu")
    assert sub.speculative_tokens == 3 and sub.k1 == 4
    inp = ARNodeInputs(input_ids=torch.zeros(4, dtype=torch.long), input_seq_len=4)
    step = sub.declare_step("decode", ["a"], [inp])
    assert SPEC in step.steps and step.segments[0].span == 4
    sub.cuda_graphs = True
    cfg = sub.get_cuda_graph_configs(torch.device("cpu"))[0]
    assert cfg.single_request_inputs.input_seq_len == 4 and cfg.single_request_inputs.input_ids.numel() == 4
    assert cfg.get_total_tokens(8) == [32]
    # the stub draft repeats the bonus token
    bonus = torch.tensor([[5], [9]])
    assert sub._draft(bonus).tolist() == [[5, 5, 5], [9, 9, 9]]
