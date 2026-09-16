"""The ``Model`` contract on CPU: registry, resources, walks, prompt processing,
forward-pass sequencing, submodule construction (dummy device)."""
import torch

from mstar.engine.resources import AttnBackend, KVLayout, resolve_spec_dependencies
from mstar.model.kimi_k3.config import KDA_STATE, MLA_ATTN, MLA_KV, SAMPLER
from mstar.model.registry import HF_MODELS, MODEL_REGISTRY, get_model_class


def test_registry_and_resources(tiny_dir):
    assert "kimi_k3" in MODEL_REGISTRY and HF_MODELS["kimi_k3"]["model_path_hf"] == "moonshotai/Kimi-K3"
    cls = get_model_class("kimi_k3")
    model = cls(model_path_hf=str(tiny_dir), max_output_tokens=16)
    specs = resolve_spec_dependencies(model.get_node_resources())
    assert set(specs) == {MLA_KV, MLA_ATTN, KDA_STATE, SAMPLER}
    kv = specs[MLA_KV].config
    assert kv.layout is KVLayout.MLA and kv.num_layers == 2 and kv.latent_dim == 128 + 32
    assert specs[MLA_ATTN].config.backend is AttnBackend.FLASHINFER_MLA
    assert abs(specs[MLA_ATTN].config.sm_scale - (64 + 32) ** -0.5) < 1e-9
    st = specs[KDA_STATE].config
    assert st.num_layers == 6 and st.parts["conv"].shape == (3 * 256, 4) and st.parts["recurrent"].shape == (8, 32, 32)
    st.shard(2)
    assert st.parts["conv"].shape == (384, 4) and st.parts["recurrent"].shape == (4, 32, 32)
    walks = model.get_graph_walk_graphs()
    assert set(walks) == {"prefill", "decode"} and model.nodes == ["LLM"]
    assert model.get_default_sharding_config().tp_enabled_nodes == {"LLM"}


def test_prompt_and_sequencing(tiny_dir):
    model = get_model_class("kimi_k3")(model_path_hf=str(tiny_dir))
    out = model.process_prompt("Hello", ["text"], ["text"])
    ids = out["text_inputs"][0]
    assert ids.dtype == torch.long and ids[0].item() == 163587  # <|open|>message ... chat template applied
    raw = model.process_prompt("Hello", ["text"], ["text"], raw_prompt=True)["text_inputs"][0]
    assert raw.numel() < ids.numel()
    args = model.get_initial_forward_pass_args("default", ["text"], ["text"], {"text_inputs": []})
    assert args.full_metadata.graph_walk == "prefill" and args.full_metadata.is_prefill
    nxt = model.get_partition_forward_pass_args("default", args.full_metadata, {"new_token": []})
    assert nxt.full_metadata.graph_walk == "decode" and not nxt.request_done
    done = model.get_partition_forward_pass_args("default", nxt.full_metadata, {"new_token": []})
    assert done.request_done
    cfgs = model.get_request_resource_configs({}, {"temperature": 0.7, "ignore_eos": True})
    assert cfgs[SAMPLER].temperature == 0.7 and cfgs[SAMPLER].ignore_eos
    assert model.postprocess(torch.tensor([ids[-1]]), "text") == b"<|sep|>"


def test_submodule_builds_on_cpu(tiny_dir):
    model = get_model_class("kimi_k3")(model_path_hf=str(tiny_dir))
    sub = model.get_submodule("LLM", device="cpu", autocast_dtype=torch.float32)
    assert sub is not None and sub.language_model.cfg.num_hidden_layers == 8
    lm = sub.language_model
    assert lm.model.layers[0].self_attn.A_log.dtype == torch.float32
    torch.manual_seed(0)
    ids = torch.randint(0, 1000, (6,))
    with torch.no_grad():
        logits, _ = lm.forward_dense(ids)
    assert logits.shape == (6, 163840) and torch.isfinite(logits).all()
    # the torch reference KDA kernel is host-indexed, so no CUDA-graph capture on CPU
    assert sub.cuda_graphs is False and sub.get_cuda_graph_configs(torch.device("cpu")) == []
    sub.cuda_graphs = True
    cg = sub.get_cuda_graph_configs(torch.device("cpu"))
    # decode only: the varlen KDA prefill kernels size work on the host, so no prefill capture
    assert len(cg) == 1 and cg[0].capture_graph_walk == "decode"


def test_moe_backend_selection_helpers():
    """The Marlin load is retried before falling back, and the backend choice is AND-ed across
    the tensor-parallel group (a trivial group returns the local verdict)."""
    from mstar.distributed.communication import CommGroup
    from mstar.model.kimi_k3.components.language_model import _all_ranks_agree, _retry

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise FileNotFoundError("lock")
        return "loaded"

    assert _retry(flaky, attempts=3, delay_s=0) == "loaded" and calls["n"] == 3
    import pytest as _pytest
    with _pytest.raises(FileNotFoundError):
        _retry(lambda: (_ for _ in ()).throw(FileNotFoundError("x")), attempts=2, delay_s=0)
    assert _all_ranks_agree(True, None) is True and _all_ranks_agree(False, None) is False
    assert _all_ranks_agree(True, CommGroup.trivial()) is True
