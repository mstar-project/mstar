"""Absorbed-MLA serve smoke: the FULL model through the real mla_absorb backend.

This is the end-to-end gate for the (now-default) weight-absorbed path. The
absorbed *math* is validated by ``test_kimi_mla_absorb.py`` (CPU) and the wired
attention forward by ``test_kimi_mla_absorb_forward.py``; the SDPA-over-latent
backend by ``test_kimi_mla_absorb_paged.py``. This test closes the loop: a whole
reduced ``KimiForCausalLM`` (all layers) built via the real
``get_submodule`` (meta -> to_empty -> load_weights -> post-load walker that builds
w_kc/w_vc + fused_qkv_a_proj) is driven for a prefill + several decode steps over a
genuine ``MlaAbsorbCacheManager`` + engine-shaped 4D latent cache.

Correctness tie: the SAME synthetic checkpoint is loaded into a NAIVE reference
model (``mla_absorb=False``) and run through the mock cache at the DeepSeek scale;
the absorbed serve's prefill logits must match it (weight absorption is numerically
identical to naive up to fp rounding). This pins the absorbed serve path to the
M4-golden naive reference at full-model scale.

Run:  pytest test/integration/test_kimi_mla_absorb_serve.py -v
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
from mstar.model.kimi_k2_7.components.rope import yarn_get_mscale
from mstar.model.kimi_k2_7.config import KimiK2Config
from mstar.model.kimi_k2_7.kimi_model import KimiK2Model
from mstar.model.kimi_k2_7.submodules import KimiLLMSubmodule
from mstar.model.submodule_base import ModelInputsFromEngine

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="absorbed serve smoke needs a GPU (real paged latent cache)",
)

DEVICE = torch.device("cuda")


# --------------------------------------------------------------------------
# Synthetic checkpoint (same serialization as test_kimi_submodule.py).
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
    """A naive-path (mla_absorb=False) model with random weights = the ground truth."""
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
# Naive mock cache (DeepSeek-correct scale) for the reference logits.
# --------------------------------------------------------------------------

def _sdpa_causal(q, k, v, scale):
    qt, kt, vt = (t.transpose(0, 1).float() for t in (q, k, v))
    t = q.shape[0]
    causal = torch.triu(torch.full((t, t), float("-inf"), device=q.device), diagonal=1)
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


# --------------------------------------------------------------------------
# Real mla_absorb backend (4D latent cache), engine-shaped.
# --------------------------------------------------------------------------

def _absorbed_softmax_scale(cfg):
    r = cfg.rope_scaling
    mscale = yarn_get_mscale(r["factor"], r.get("mscale_all_dim", 0.0))
    return cfg.qk_head_dim ** -0.5 * mscale * mscale


def _make_latent_cache_manager(cfg, dtype, page_size=128, max_num_pages=8):
    latent_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim
    kv_cache = torch.zeros(
        cfg.num_hidden_layers, max_num_pages, page_size, latent_dim,
        dtype=dtype, device=DEVICE,
    ).contiguous()  # 4D latent cache (matches KVCacheEngine's mla_absorb branch)
    kv_cfg = KVCacheConfig(
        num_layers=cfg.num_hidden_layers, num_kv_heads=1, head_dim=latent_dim,
        max_seq_len=page_size * max_num_pages, max_num_pages=max_num_pages,
        page_size=page_size, num_qo_heads=cfg.num_attention_heads,
        attention_backend="mla_absorb", softmax_scale=_absorbed_softmax_scale(cfg),
    )
    transfer_info = TransferEngineInfo(
        my_entity_id="kimi_absorb_serve", my_session_id="kimi_session",
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


def _make_model(cfg, checkpoint_dir) -> KimiK2Model:
    model = object.__new__(KimiK2Model)  # skip __init__ (tokenizer/full config)
    model.config = cfg
    model.model_path_hf = str(checkpoint_dir)
    model.cache_dir = None
    model._submodule_cache = {}
    return model


def _step(submodule, cm, graph_walk, token_ids):
    engine_inputs = ModelInputsFromEngine(
        request_ids=["r0"], per_request_info={}, cache_manager=cm)
    ar_in = submodule.prepare_inputs(
        graph_walk=graph_walk, fwd_info=None, inputs={"text_inputs": [token_ids]})
    packed = submodule.preprocess(graph_walk, engine_inputs, [ar_in])
    with torch.no_grad():
        out = submodule.forward(graph_walk, engine_inputs, **packed)
    return out["logits"][0]


# --------------------------------------------------------------------------
# Test
# --------------------------------------------------------------------------

def test_absorbed_serve_matches_naive_reference(tmp_path):
    from safetensors.torch import save_file

    torch.manual_seed(0)
    # One synthetic checkpoint, built from a naive reference model.
    cfg_naive = KimiK2Config.reduced()            # mla_absorb=False (reduced default)
    assert cfg_naive.mla_absorb is False
    ref = _build_reference(cfg_naive)
    save_file(_hf_checkpoint(ref, cfg_naive), str(tmp_path / "model.safetensors"))

    # Reference logits: the naive model through the mock cache at the DeepSeek scale.
    T = 6
    prompt = torch.randint(0, cfg_naive.vocab_size, (T,), device=DEVICE)
    pos = torch.arange(T, device=DEVICE)
    with torch.no_grad():
        ref_hidden = ref.model(prompt, _MockMLACache(cfg_naive.padded_head_dim), pos)
        ref_logits = ref.lm_head(ref_hidden[-1:])  # (1, vocab)

    # Absorbed serve: load the SAME checkpoint into an mla_absorb model via the real
    # build path; the post-load walker builds w_kc/w_vc + fused_qkv_a_proj.
    cfg_absorb = KimiK2Config.reduced()
    cfg_absorb.mla_absorb = True
    model = _make_model(cfg_absorb, tmp_path)
    submodule = model.get_submodule("LLM", device="cuda", autocast_dtype=torch.bfloat16)
    assert isinstance(submodule, KimiLLMSubmodule)
    # Absorbed models DO carry derived buffers (w_kc/w_vc/fused, persistent=False).
    buf_names = {n for n, _ in submodule.language_model.named_buffers()}
    assert any("w_kc" in n for n in buf_names) and any("fused_qkv_a_proj" in n for n in buf_names)

    cm, alloc = _make_latent_cache_manager(cfg_absorb, torch.bfloat16)
    try:
        prefill_logits = _step(submodule, cm, "prefill", prompt)
        assert prefill_logits.shape == (1, cfg_absorb.vocab_size)
        assert torch.isfinite(prefill_logits).all()
        # Absorbed == naive up to bf16 rounding through a 2-layer stack + real backend.
        torch.testing.assert_close(prefill_logits, ref_logits, rtol=5e-2, atol=5e-2)

        # A few decode steps over the accumulating paged LATENT cache.
        next_token = prefill_logits.argmax(-1)
        generated = [int(next_token.item())]
        for _ in range(4):
            logits = _step(submodule, cm, "decode", next_token)
            assert logits.shape == (1, cfg_absorb.vocab_size)
            assert torch.isfinite(logits).all()
            next_token = logits.argmax(-1)
            tok = int(next_token.item())
            assert 0 <= tok < cfg_absorb.vocab_size
            generated.append(tok)
    finally:
        alloc.cleanup()

    assert len(generated) == 5
