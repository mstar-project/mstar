"""GLM-5.2 DSA long-context serve e2e: decode PAST index_topk on the real
absorbed paged-latent path (1 GPU, reduced dims, tiny topk=8).

Ports the test_glm52_serve_e2e.py serve harness to the mla_absorb backend
(latent 4D cache, like test_kimi_mla_absorb_serve.py) with
``dsa_long_context=True``. One greedy generation run drives prefill ->
decode through check_stop to prove the sparse path serves and stops
cleanly; then three TEACHER-FORCED replays of that token sequence pin the
two properties that cannot both hold unless the engine half works:

  prefix property — while every context fits topk the flag-on run takes
  the UNTOUCHED dense paged path, so its per-step logits must be BITWISE
  identical both to a dense comparator (topk lifted out of reach) and to
  the flag-off serve;

  engagement — once context exceeds topk the logits must DIFFER from the
  dense comparator: top-8 of >8 positions provably drops keys, so equal
  logits would mean the sparse path never ran.

Teacher forcing (same token fed to every variant) keeps the comparison
per-step: divergence cannot silently propagate through sampled tokens.
The k-store lifecycle is asserted on the real path too: it grows one row
per token per FULL layer and must be EMPTY after ``cleanup_request`` —
the hook ``KVCacheEngine.remove_request`` fires (a leak per request is a
rejection-level bug).
"""
import pytest
import torch

# Module-level skip BEFORE the mstar imports (test_phase1.py precedent):
# the engine import chain pulls triton, which dev machines without a GPU
# don't have — importing first would turn the skip into a collection error.
if not torch.cuda.is_available():
    pytest.skip(
        "DSA long-context serve e2e needs a GPU (real paged latent cache)",
        allow_module_level=True,
    )

from mstar.communication.tensors import LocalTransferEngine
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.cache_manager import WorkspaceBufferManager, create_cache_manager
from mstar.engine.kv_store import (
    KVCacheConfig,
    PagedAllocationManager,
    TransferEngineInfo,
)
from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM
from mstar.model.glm52.components.moe import Glm52SparseMoeBlock
from mstar.model.glm52.config import Glm52ModelConfig
from mstar.model.glm52.glm52_model import Glm52Model
from mstar.model.glm52.submodules import Glm52LLMSubmodule
from mstar.model.submodule_base import ModelInputsFromEngine
from mstar.utils.sampling import MultiSamplingConfig, Sampler, SamplingConfig

DEVICE = torch.device("cuda")
TOPK = 8
PROMPT_LEN = 6
MAX_TOKENS = 14  # final context 6 + 14 = 20, far past topk


def _fill_layer(layer, cfg):
    a = layer.self_attn
    for lin in (a.q_a_proj, a.q_b_proj, a.kv_a_proj_with_mqa, a.kv_b_proj, a.o_proj):
        lin.weight.data.normal_(0, 0.03)
    for norm in (a.q_a_layernorm, a.kv_a_layernorm):
        norm.weight.data.normal_(1.0, 0.02)
    if a.indexer is not None:  # FULL indexer layers (layer 0 at reduced dims)
        for lin in (a.indexer.wq_b, a.indexer.wk, a.indexer.weights_proj):
            lin.weight.data.normal_(0, 0.03)
        a.indexer.k_norm.weight.data.normal_(1.0, 0.02)
        a.indexer.k_norm.bias.data.normal_(0, 0.02)
    layer.input_layernorm.weight.data.normal_(1.0, 0.02)
    layer.post_attention_layernorm.weight.data.normal_(1.0, 0.02)
    mlp = layer.mlp
    if isinstance(mlp, Glm52SparseMoeBlock):
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


def _hf_checkpoint(model, cfg):
    """bf16 passthrough keys (no fp8 pairs — quantization is orthogonal to
    the DSA path and test_glm52_serve_e2e.py already covers it), including
    the FULL-layer indexer weights the sparse path needs."""
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
        if a.indexer is not None:
            sd[p + "self_attn.indexer.wq_b.weight"] = a.indexer.wq_b.weight
            sd[p + "self_attn.indexer.wk.weight"] = a.indexer.wk.weight
            sd[p + "self_attn.indexer.weights_proj.weight"] = a.indexer.weights_proj.weight
            sd[p + "self_attn.indexer.k_norm.weight"] = a.indexer.k_norm.weight
            sd[p + "self_attn.indexer.k_norm.bias"] = a.indexer.k_norm.bias
        sd[p + "input_layernorm.weight"] = layer.input_layernorm.weight
        sd[p + "post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight
        mlp = layer.mlp
        if isinstance(mlp, Glm52SparseMoeBlock):
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
    cfg = Glm52ModelConfig.reduced()
    ref = Glm52ForCausalLM(cfg).to(device=DEVICE, dtype=torch.bfloat16)
    ref.model.embed_tokens.weight.data.normal_(0, 0.05)
    ref.model.norm.weight.data.normal_(1.0, 0.02)
    ref.lm_head.weight.data.normal_(0, 0.02)
    for layer in ref.model.layers:
        _fill_layer(layer, cfg)
    save_file(_hf_checkpoint(ref.eval(), cfg), str(tmp_path / "model.safetensors"))


def _make_latent_cache_manager(cfg, page_size=8, max_num_pages=16):
    latent_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim
    kv_cache = torch.zeros(
        cfg.num_hidden_layers, max_num_pages, page_size, latent_dim,
        dtype=torch.bfloat16, device=DEVICE,
    ).contiguous()  # 4D latent cache (KVCacheEngine's mla_absorb layout)
    kv_cfg = KVCacheConfig(
        num_layers=cfg.num_hidden_layers, num_kv_heads=1, head_dim=latent_dim,
        max_seq_len=page_size * max_num_pages, max_num_pages=max_num_pages,
        page_size=page_size, num_qo_heads=cfg.num_attention_heads,
        attention_backend="mla_absorb",
        softmax_scale=cfg.qk_head_dim ** -0.5,  # no Yarn -> no mscale
        mla_ckv_dim=cfg.kv_lora_rank,
    )
    transfer_info = TransferEngineInfo(
        my_entity_id="glm52_dsa_serve", my_session_id="glm52_dsa_serve",
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
    # ignore_eos: reduced EOS ids live inside the byte vocab; greedy decode
    # on random weights may hit one and break the exact-length assertions.
    return CurrentForwardPassInfo(
        request_id="r0", graph_walk="decode", requires_cfg=False, fwd_index=0,
        random_seed=0, max_tokens=max_tokens,
        sampling_config={"LLM": MultiSamplingConfig(main=SamplingConfig(
            vocab_size=cfg.vocab_size, ignore_eos=True))},
        dynamic_loop_iter_counts={},
    )


def _generate_greedy(submodule, cfg, prompt_ids, max_tokens):
    """The real serve loop (prefill -> decode -> check_stop) on the sparse
    path; returns (generated tokens, stopped-via-check_stop)."""
    cm, alloc = _make_latent_cache_manager(cfg)
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
        assert torch.isfinite(logits).all()
        next_token = sampler.sample(["r0"], logits).clone()
        generated.append(int(next_token.item()))

        for step in range(max_tokens + 4):  # +slack; check_stop must break first
            ar = submodule.prepare_inputs("decode", None, {"text_inputs": [next_token]})
            packed = submodule.preprocess("decode", engine_inputs, [ar])
            with torch.no_grad():
                out = submodule.forward_batched("decode", engine_inputs, **packed)
            new_token = out["r0"]["new_token"][0]
            outputs = {"new_token": [new_token]}
            submodule.postprocess("r0", info, outputs)
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


def _teacher_forced_logits(submodule, cfg, prompt_ids, forced_tokens):
    """Replay a fixed token sequence and collect the logits of every step:
    index 0 = prefill (context PROMPT_LEN), index 1+j = decode of
    ``forced_tokens[j]`` (post-step context PROMPT_LEN + 1 + j)."""
    cm, alloc = _make_latent_cache_manager(cfg)
    engine_inputs = ModelInputsFromEngine(
        request_ids=["r0"], per_request_info={}, cache_manager=cm, sampler=None,
    )
    logits_per_step = []
    try:
        for i, ids in enumerate(
            [prompt_ids, *[torch.tensor([t], device=DEVICE) for t in forced_tokens]]
        ):
            walk = "prefill" if i == 0 else "decode"
            ar = submodule.prepare_inputs(walk, None, {"text_inputs": [ids]})
            packed = submodule.preprocess(walk, engine_inputs, [ar])
            with torch.no_grad():
                logits = submodule.forward(walk, engine_inputs, **packed)["logits"][0]
            logits_per_step.append(logits)
    finally:
        alloc.cleanup()
        submodule.cleanup_request("r0")
    return logits_per_step


def test_dsa_longctx_decode_past_topk(tmp_path):
    _write_checkpoint(tmp_path, seed=0)

    model = Glm52Model(
        model_path_hf="", config_variant="reduced",
        checkpoint_path=str(tmp_path), tokenizer_mode="byte",
    )
    cfg = model.config
    cfg.mla_absorb = True         # sparse path reads the paged latent cache
    cfg.dsa_long_context = True
    cfg.index_topk = TOPK

    submodule = model.get_submodule("LLM", device="cuda", autocast_dtype=torch.bfloat16)
    assert isinstance(submodule, Glm52LLMSubmodule)
    # Long-context serving is eager-only (host-side DSA upkeep would not
    # replay inside a captured graph).
    assert submodule.get_cuda_graph_configs(DEVICE) == []

    prompt_ids = torch.arange(10, 10 + PROMPT_LEN, device=DEVICE)

    # ---- (1) the sparse path serves and stops cleanly ------------------
    generated, stopped = _generate_greedy(submodule, cfg, prompt_ids, MAX_TOKENS)
    assert stopped, "decode loop did not terminate via check_stop"
    assert len(generated) == 1 + MAX_TOKENS, generated
    assert all(0 <= t < cfg.vocab_size for t in generated), generated

    # k-store lifecycle on the real path: one row per token per FULL layer
    # (prefill 6 + one per decode forward), gone after retirement.
    assert submodule._dsa_k_store.tokens("r0", 0) == PROMPT_LEN + MAX_TOKENS
    assert submodule._dsa_k_store.tokens("r0", 1) == 0  # SHARED layer: none
    submodule.cleanup_request("r0")
    assert submodule._dsa_k_store.tracked_requests() == set()

    # ---- teacher-forced replays of the generated sequence --------------
    forced = generated[:-1]  # the last token is never fed back
    sparse = _teacher_forced_logits(submodule, cfg, prompt_ids, forced)
    assert all(torch.isfinite(step).all() for step in sparse)
    # The replay must reproduce the generation (greedy = argmax, and the
    # sparse path is deterministic step for step).
    replay_tokens = [int(step.argmax(-1).item()) for step in sparse]
    assert replay_tokens == generated

    cfg.index_topk = 512  # dense comparator: selection never engages
    dense = _teacher_forced_logits(submodule, cfg, prompt_ids, forced)

    cfg.index_topk = TOPK
    cfg.dsa_long_context = False  # flag-off serve, valid only within topk
    identity_steps = TOPK - PROMPT_LEN  # decode steps whose post ctx <= topk
    flag_off = _teacher_forced_logits(
        submodule, cfg, prompt_ids, forced[:identity_steps])
    cfg.dsa_long_context = True

    # ---- (2) prefix property: bitwise identity while ctx <= topk -------
    # Step i covers context PROMPT_LEN + i; identity for i <= topk - PROMPT_LEN.
    for i in range(identity_steps + 1):
        assert torch.equal(sparse[i], dense[i]), f"step {i} vs dense"
        assert torch.equal(sparse[i], flag_off[i]), f"step {i} vs flag-off"

    # ---- (3) the sparse path actually engages beyond topk --------------
    beyond = [
        not torch.equal(sparse[i], dense[i])
        for i in range(identity_steps + 1, len(sparse))
    ]
    assert any(beyond), (
        "no step beyond topk diverged from the dense comparator — the "
        "sparse path never engaged"
    )
    # By the final step only 8 of 20 positions are attended; equality there
    # would be a numerical accident this test must not tolerate.
    assert beyond[-1], "final step (ctx 20, topk 8) matched dense attention"
