import pytest
import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.cache_manager import WorkspaceBufferManager, create_cache_manager
from mstar.engine.kv_store import (
    KVCacheConfig,
    PagedAllocationManager,
    TransferEngineInfo,
)
from mstar.model.kimi_k2_7.components.causal_lm import KimiForCausalLM
from mstar.model.kimi_k2_7.components.moe import KimiSparseMoeBlock
from mstar.model.kimi_k2_7.config import KimiK2Config
from mstar.model.kimi_k2_7.kimi_model import KimiK2Model
from mstar.model.kimi_k2_7.submodules import KimiLLMSubmodule
from mstar.model.submodule_base import ModelInputsFromEngine
from mstar.utils.sampling import Sampler, SamplingConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="serve e2e needs a GPU (real FlashInfer paged cache)",
)

DEVICE = torch.device("cuda")


def _fill_layer(layer, cfg):
    a = layer.self_attn
    for lin in (a.q_a_proj, a.q_b_proj, a.kv_a_proj_with_mqa, a.kv_b_proj, a.o_proj):
        lin.weight.data.normal_(0, 0.03)
    for norm in (a.q_a_layernorm, a.kv_a_layernorm):
        norm.weight.data.normal_(1.0, 0.02)
    layer.input_layernorm.weight.data.normal_(1.0, 0.02)
    layer.post_attention_layernorm.weight.data.normal_(1.0, 0.02)
    mlp = layer.mlp
    if isinstance(mlp, KimiSparseMoeBlock):
        mlp.gate.weight.data.normal_(0, 1)
        mlp.gate.e_score_correction_bias.data = torch.randn(
            cfg.n_routed_experts, device=DEVICE, dtype=torch.float32)
        mlp.experts.gate_up_proj.data.normal_(0, 0.05)
        mlp.experts.down_proj.data.normal_(0, 0.05)
        mlp.shared_expert.gate_up_proj.weight.data.normal_(0, 0.05)
        mlp.shared_expert.down_proj.weight.data.normal_(0, 0.05)
    else:
        mlp.gate_up_proj.weight.data.normal_(0, 0.05)
        mlp.down_proj.weight.data.normal_(0, 0.05)


def _build_reference(cfg):
    model = KimiForCausalLM(cfg).to(device=DEVICE, dtype=torch.bfloat16)
    model.model.embed_tokens.weight.data.normal_(0, 0.05)
    model.model.norm.weight.data.normal_(1.0, 0.02)
    model.lm_head.weight.data.normal_(0, 0.02)
    for layer in model.model.layers:
        _fill_layer(layer, cfg)
    return model.eval()


def _hf_checkpoint(model, cfg):
    inter = cfg.intermediate_size
    moe_inter = cfg.moe_intermediate_size
    shared_inter = cfg.moe_intermediate_size * cfg.n_shared_experts
    m = model.model
    sd = {"model.embed_tokens.weight": m.embed_tokens.weight}
    for i, layer in enumerate(m.layers):
        p = f"model.layers.{i}."
        a = layer.self_attn
        sd[p + "self_attn.q_a_proj.weight"] = a.q_a_proj.weight
        sd[p + "self_attn.q_a_layernorm.weight"] = a.q_a_layernorm.weight
        sd[p + "self_attn.q_b_proj.weight"] = a.q_b_proj.weight
        sd[p + "self_attn.kv_a_proj_with_mqa.weight"] = a.kv_a_proj_with_mqa.weight
        sd[p + "self_attn.kv_a_layernorm.weight"] = a.kv_a_layernorm.weight
        sd[p + "self_attn.kv_b_proj.weight"] = a.kv_b_proj.weight
        sd[p + "self_attn.o_proj.weight"] = a.o_proj.weight
        sd[p + "input_layernorm.weight"] = layer.input_layernorm.weight
        sd[p + "post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight
        mlp = layer.mlp
        if isinstance(mlp, KimiSparseMoeBlock):
            sd[p + "mlp.gate.weight"] = mlp.gate.weight
            sd[p + "mlp.gate.e_score_correction_bias"] = mlp.gate.e_score_correction_bias
            gup, dwn = mlp.experts.gate_up_proj, mlp.experts.down_proj
            for e in range(cfg.n_routed_experts):
                sd[p + f"mlp.experts.{e}.gate_proj.weight"] = gup[e, :moe_inter, :]
                sd[p + f"mlp.experts.{e}.up_proj.weight"] = gup[e, moe_inter:, :]
                sd[p + f"mlp.experts.{e}.down_proj.weight"] = dwn[e]
            sh = mlp.shared_expert
            sd[p + "mlp.shared_experts.gate_proj.weight"] = sh.gate_up_proj.weight[:shared_inter]
            sd[p + "mlp.shared_experts.up_proj.weight"] = sh.gate_up_proj.weight[shared_inter:]
            sd[p + "mlp.shared_experts.down_proj.weight"] = sh.down_proj.weight
        else:
            sd[p + "mlp.gate_proj.weight"] = mlp.gate_up_proj.weight[:inter]
            sd[p + "mlp.up_proj.weight"] = mlp.gate_up_proj.weight[inter:]
            sd[p + "mlp.down_proj.weight"] = mlp.down_proj.weight
    sd["model.norm.weight"] = m.norm.weight
    sd["lm_head.weight"] = model.lm_head.weight
    return {k: v.detach().cpu().clone().contiguous() for k, v in sd.items()}


def _write_checkpoint(tmp_path, seed=0):
    from safetensors.torch import save_file

    torch.manual_seed(seed)
    cfg = KimiK2Config.reduced()
    ref = _build_reference(cfg)
    save_file(_hf_checkpoint(ref, cfg), str(tmp_path / "model.safetensors"))
    return cfg


def _make_real_cache_manager(cfg, dtype, page_size=128, max_num_pages=8):
    num_heads = cfg.num_attention_heads
    head_dim = cfg.padded_head_dim
    kv_cache = torch.zeros(
        cfg.num_hidden_layers, max_num_pages, 2, page_size, num_heads, head_dim,
        dtype=dtype, device=DEVICE,
    ).contiguous()
    kv_cfg = KVCacheConfig(
        num_layers=cfg.num_hidden_layers, num_kv_heads=num_heads, head_dim=head_dim,
        max_seq_len=page_size * max_num_pages, max_num_pages=max_num_pages,
        page_size=page_size, num_qo_heads=num_heads,
    )
    transfer_info = TransferEngineInfo(
        my_entity_id="kimi_serve_e2e", my_session_id="kimi_serve_e2e",
        transfer_engine=LocalTransferEngine("localhost"),
    )
    alloc = PagedAllocationManager(
        config=kv_cfg, kv_cache=kv_cache, transfer_engine_info=transfer_info)
    alloc.add_request("r0", ["main"])
    buffers = WorkspaceBufferManager(64 * 1024 * 1024, device=DEVICE)
    cm = create_cache_manager(
        request_ids=["r0"], active_labels_per_request={"r0": "main"},
        kv_cache=kv_cache, alloc_manager=alloc, buffer_manager=buffers,
        kv_cache_config=kv_cfg, device=DEVICE,
    )
    return cm, alloc


def _greedy_sampler(cfg):
    sampler = Sampler(device=DEVICE)
    sampler.add_request("r0")
    sampler.set_config("r0", vocab_size=cfg.vocab_size, temperature=0.0,
                       top_k=0, top_p=1.0, repetition_penalty=1.0)
    return sampler


def _fwd_info(max_tokens, cfg):
    return CurrentForwardPassInfo(
        request_id="r0", graph_walk="decode", requires_cfg=False, fwd_index=0,
        random_seed=0, max_tokens=max_tokens,
        sampling_config={"LLM": SamplingConfig(vocab_size=cfg.vocab_size,
                                               ignore_eos=cfg.ignore_eos)},
        dynamic_loop_iter_counts={},
    )


def _run_generation(model, submodule, cfg, prompt_ids, max_tokens):
    cm, alloc = _make_real_cache_manager(cfg, torch.bfloat16)
    sampler = _greedy_sampler(cfg)
    engine_inputs = ModelInputsFromEngine(
        request_ids=["r0"], per_request_info={}, cache_manager=cm, sampler=sampler,
    )
    info = _fwd_info(max_tokens, cfg)
    generated: list[int] = []
    stopped = False
    try:
        ar = submodule.prepare_inputs("prefill", None, {"text_inputs": [prompt_ids]})
        packed = submodule.preprocess("prefill", engine_inputs, [ar])
        with torch.no_grad():
            logits = submodule.forward("prefill", engine_inputs, **packed)["logits"][0]
        assert logits.shape == (1, cfg.vocab_size)
        assert torch.isfinite(logits).all()
        next_token = sampler.sample(["r0"], logits).clone()  # (1,)
        generated.append(int(next_token.item()))

        for step in range(max_tokens + 4):  # +slack; check_stop must break first
            ar = submodule.prepare_inputs("decode", None, {"text_inputs": [next_token]})
            packed = submodule.preprocess("decode", engine_inputs, [ar])
            with torch.no_grad():
                out = submodule.forward_batched("decode", engine_inputs, **packed)
            new_token = out["r0"]["new_token"][0]
            outputs = {"new_token": [new_token]}
            submodule.postprocess("r0", info, outputs)  # rebinds text_inputs
            assert outputs["text_inputs"] is outputs["new_token"]
            info.dynamic_loop_iter_counts["decode_loop"] = step
            stop = submodule.check_stop("r0", info, outputs)
            generated.append(int(new_token.item()))
            next_token = new_token
            if stop:
                stopped = True
                break
    finally:
        alloc.cleanup()
        sampler.remove_request("r0")
    return generated, stopped


def test_serve_path_prefill_decode_loop(tmp_path):
    cfg = _write_checkpoint(tmp_path, seed=0)

    model = KimiK2Model(
        model_path_hf="", config_variant="reduced",
        checkpoint_path=str(tmp_path), tokenizer_mode="byte",
    )
    assert model.config.vocab_size == 256

    prompt_tensors = model.process_prompt("hello kimi", ["text"], ["text"])
    prompt_ids = prompt_tensors["text_inputs"][0].to(DEVICE)
    assert prompt_ids.tolist() == list("hello kimi".encode("utf-8"))
    assert prompt_ids.max().item() < cfg.vocab_size

    submodule = model.get_submodule("LLM", device="cuda", autocast_dtype=torch.bfloat16)
    assert isinstance(submodule, KimiLLMSubmodule)
    assert list(submodule.language_model.named_buffers()) == []

    MAX_TOKENS = 6
    generated, stopped = _run_generation(model, submodule, cfg, prompt_ids, MAX_TOKENS)

    assert stopped, "decode loop did not terminate via check_stop"
    # check_stop fires after exactly MAX_TOKENS decode steps.
    assert len(generated) == 1 + MAX_TOKENS, generated
    assert all(0 <= t < cfg.vocab_size for t in generated), generated

    out_bytes = model.postprocess(torch.tensor(generated), "text")
    assert isinstance(out_bytes, bytes)
    assert len(out_bytes) == len(generated)


def test_serve_path_is_deterministic(tmp_path):
    cfg = _write_checkpoint(tmp_path, seed=1)
    model = KimiK2Model(
        model_path_hf="", config_variant="reduced",
        checkpoint_path=str(tmp_path), tokenizer_mode="byte",
    )
    submodule = model.get_submodule("LLM", device="cuda", autocast_dtype=torch.bfloat16)
    prompt_ids = model.process_prompt("serve", ["text"], ["text"])["text_inputs"][0].to(DEVICE)

    runs = [
        _run_generation(model, submodule, cfg, prompt_ids, max_tokens=5)[0]
        for _ in range(2)
    ]
    assert runs[0] == runs[1], runs
    assert len(runs[0]) == 1 + 5
