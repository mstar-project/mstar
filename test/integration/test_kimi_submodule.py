"""M6 step 6: submodule-level end-to-end through the REAL paged cache.

This is the M6 correctness gate for serving: it exercises the whole
``KimiK2Model.get_submodule`` -> ``KimiLLMSubmodule`` -> real
``FlashInferCacheManager`` path on the reduced config with synthetic weights.

Two things are validated:

1. **The real build path.** ``get_submodule`` constructs ``KimiForCausalLM`` on
   the meta device, casts to bf16 on meta, ``to_empty(cuda)``, and runs the M5 HF
   loader — the production ``meta -> to_empty -> load_weights`` path (where the M5
   rope-buffer bug would have bitten). We assert the loaded model carries ZERO
   buffers (M6 buffer audit) so no derived tensor survives as garbage.

2. **Serving lifecycle over the paged MLA.** We drive
   ``prepare_inputs -> preprocess -> forward`` through a genuine
   ``FlashInferCacheManager`` (head_dim = padded_head_dim = 64) for a prefill plus
   several decode steps, asserting sane token generation. The prefill logits are
   checked against a mock-cache forward of the SAME loaded model at the
   DeepSeek-correct scale, tying the paged serving path to the validated
   MockCacheHandle goldens.

``mstar-serve`` full-stack e2e (conductor + worker processes + SHM ports + CUDA
graph capture) is NOT run here — that infra isn't stood up in this environment.
This submodule-level test is the required correctness gate; see the M6 notes in
kimi-port-plan for the serve status.

Run:  pytest test/integration/test_kimi_submodule.py -v
"""
import pytest
import torch

from mstar.communication.tensors import LocalTransferEngine
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

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="submodule e2e needs a GPU (real FlashInfer paged cache)",
)

DEVICE = torch.device("cuda")


# --------------------------------------------------------------------------
# Synthetic HF DeepSeek-V3 checkpoint (same serialization as
# test_kimi_weight_loading — un-fuse every fused param back to HF keys).
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Real paged cache + mock cache (DeepSeek-correct scale via padded_head_dim).
# --------------------------------------------------------------------------

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
        my_entity_id="kimi_submodule_test", my_session_id="kimi_session",
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


def _sdpa_causal(q, k, v, scale):
    qt, kt, vt = (t.transpose(0, 1).float() for t in (q, k, v))
    T = q.shape[0]
    causal = torch.triu(torch.full((T, T), float("-inf"), device=q.device), diagonal=1)
    attn = (torch.einsum("hqd,hkd->hqk", qt, kt) * scale + causal).softmax(-1)
    return torch.einsum("hqk,hkd->hqd", attn, vt).transpose(0, 1).to(q.dtype)


class _MockMLACache:
    def __init__(self, head_dim):
        self.scale = head_dim ** -0.5

    def set_layer_idx(self, _i):
        pass

    def advance_seq_lens(self, *_a, **_k):
        pass

    def run_attention(self, q, k, v):
        return _sdpa_causal(q, k, v, self.scale)


def _make_model(cfg, checkpoint_dir) -> KimiK2Model:
    """A KimiK2Model wired to the synthetic checkpoint without a tokenizer.

    object.__new__ skips __init__ (which would pull a tokenizer / the full config),
    so we set only what get_submodule needs — mirroring the modular test builder.
    """
    model = object.__new__(KimiK2Model)
    model.config = cfg
    model.model_path_hf = str(checkpoint_dir)
    model.cache_dir = None
    model._submodule_cache = {}
    return model


# --------------------------------------------------------------------------
# Minimal engine-inputs + lifecycle driver.
# --------------------------------------------------------------------------

def _engine_inputs(cm):
    return ModelInputsFromEngine(
        request_ids=["r0"], per_request_info={}, cache_manager=cm,
    )


def _step(submodule, cm, graph_walk, token_ids):
    """Drive prepare_inputs -> preprocess -> forward for one packed request."""
    engine_inputs = _engine_inputs(cm)
    ar_in = submodule.prepare_inputs(
        graph_walk=graph_walk, fwd_info=None,
        inputs={"text_inputs": [token_ids]},
    )
    packed = submodule.preprocess(graph_walk, engine_inputs, [ar_in])
    with torch.no_grad():
        out = submodule.forward(graph_walk, engine_inputs, **packed)
    return out["logits"][0], packed  # (1, vocab), packed dict


# --------------------------------------------------------------------------
# Test
# --------------------------------------------------------------------------

def test_submodule_prefill_decode_over_real_paged_cache(tmp_path):
    from safetensors.torch import save_file

    torch.manual_seed(0)
    cfg = KimiK2Config.reduced()
    ref = _build_reference(cfg)
    save_file(_hf_checkpoint(ref, cfg), str(tmp_path / "model.safetensors"))

    # --- the real build path: meta -> to(bf16) -> to_empty(cuda) -> load ---
    model = _make_model(cfg, tmp_path)
    submodule = model.get_submodule("LLM", device="cuda", autocast_dtype=torch.bfloat16)
    assert isinstance(submodule, KimiLLMSubmodule)
    # get_submodule caches the built submodule.
    assert model.get_submodule("LLM") is submodule
    # M6 buffer audit: no derived tensor buffer survived the load path as garbage.
    assert list(submodule.language_model.named_buffers()) == []
    p = next(submodule.language_model.parameters())
    assert p.device.type == "cuda" and p.dtype == torch.bfloat16

    # --- prefill over the real paged FlashInfer cache (head_dim = 64) ---
    T = 6
    prompt = torch.randint(0, cfg.vocab_size, (T,), device=DEVICE)
    cm, alloc = _make_real_cache_manager(cfg, torch.bfloat16)
    try:
        prefill_logits, _ = _step(submodule, cm, "prefill", prompt)
        assert prefill_logits.shape == (1, cfg.vocab_size)
        assert torch.isfinite(prefill_logits).all()

        # Reference: the SAME loaded model through the mock cache at the
        # DeepSeek-correct scale (padded_head_dim). Ties the paged serving path to
        # the validated MockCacheHandle goldens. Loose bf16 tolerance (2-layer stack
        # through the real FlashInfer kernel).
        pos = torch.arange(T, device=DEVICE)
        with torch.no_grad():
            ref_hidden = submodule.language_model.model(
                prompt, _MockMLACache(cfg.padded_head_dim), pos)
        ref_logits = submodule.lm_head(ref_hidden[-1:])
        torch.testing.assert_close(prefill_logits, ref_logits, rtol=5e-2, atol=5e-2)

        # --- a few decode steps over the accumulating paged KV cache ---
        next_token = prefill_logits.argmax(-1)  # (1,)
        generated = [int(next_token.item())]
        assert 0 <= generated[-1] < cfg.vocab_size
        for _ in range(4):
            logits, _ = _step(submodule, cm, "decode", next_token)
            assert logits.shape == (1, cfg.vocab_size)
            assert torch.isfinite(logits).all()
            next_token = logits.argmax(-1)
            tok = int(next_token.item())
            assert 0 <= tok < cfg.vocab_size
            generated.append(tok)
    finally:
        alloc.cleanup()

    # Sane generation: right length, all valid ids.
    assert len(generated) == 5
    assert all(0 <= t < cfg.vocab_size for t in generated)


def test_submodule_paged_decode_is_deterministic(tmp_path):
    """Same prompt + fresh cache -> identical first token (paged path is stable and
    the load is reproducible). Cheap guard against nondeterministic KV writes."""
    from safetensors.torch import save_file

    torch.manual_seed(1)
    cfg = KimiK2Config.reduced()
    ref = _build_reference(cfg)
    save_file(_hf_checkpoint(ref, cfg), str(tmp_path / "model.safetensors"))
    model = _make_model(cfg, tmp_path)
    submodule = model.get_submodule("LLM", device="cuda", autocast_dtype=torch.bfloat16)

    T = 5
    prompt = torch.randint(0, cfg.vocab_size, (T,), device=DEVICE)
    tokens = []
    for _ in range(2):
        cm, alloc = _make_real_cache_manager(cfg, torch.bfloat16)
        try:
            logits, _ = _step(submodule, cm, "prefill", prompt)
            tokens.append(int(logits.argmax(-1).item()))
        finally:
            alloc.cleanup()
    assert tokens[0] == tokens[1]
