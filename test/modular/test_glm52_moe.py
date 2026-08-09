import torch

from mstar.model.glm52._testing import fake_quantize_fp8_block
from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM
from mstar.model.glm52.components.indexer import is_full_indexer_layer
from mstar.model.glm52.components.moe import Glm52MoEGate, Glm52SparseMoeBlock
from mstar.model.glm52.config import Glm52ModelConfig
from mstar.model.glm52.quantization import dequantize_fp8_block_weight
from mstar.model.glm52.weight_loader import (
    load_glm52_hf_weights,
    restore_fp32_params,
)

BLOCK = (16, 16)


def test_gate_matches_kimi_groupless():
    """GLM's router is KimiMoEGate math with n_group=1 — pin it bitwise."""
    from mstar.model.kimi_k2_7.components.moe import KimiMoEGate

    torch.manual_seed(0)
    kimi = KimiMoEGate(
        hidden_size=64, n_routed_experts=8, num_experts_per_tok=3,
        n_group=1, topk_group=1, routed_scaling_factor=2.5,
        scoring_func="sigmoid", topk_method="noaux_tc", norm_topk_prob=True,
    )
    glm = Glm52MoEGate(
        hidden_size=64, n_routed_experts=8, num_experts_per_tok=3,
        routed_scaling_factor=2.5, norm_topk_prob=True,
    )
    kimi.weight.data.normal_(0, 0.1)
    kimi.e_score_correction_bias.data.normal_(0, 0.5)
    glm.weight.data.copy_(kimi.weight.data)
    glm.e_score_correction_bias.data.copy_(kimi.e_score_correction_bias.data)

    h = torch.randn(17, 64)
    w_kimi, ids_kimi = kimi(h)
    w_glm, ids_glm = glm(h)
    assert torch.equal(ids_kimi, ids_glm)
    assert torch.equal(w_kimi, w_glm)


def _fill_expert_weights(block_fp8, block_bf16, cfg):
    """Give both blocks the same experts: fp8 bytes vs their fp32 dequant."""
    shard = cfg.moe_intermediate_size
    refs = []
    for e in range(cfg.n_routed_experts):
        g = torch.randn(shard, cfg.hidden_size) * 0.1
        u = torch.randn(shard, cfg.hidden_size) * 0.1
        d = torch.randn(cfg.hidden_size, shard) * 0.1
        g8, gs, _ = fake_quantize_fp8_block(g, BLOCK)
        u8, us, _ = fake_quantize_fp8_block(u, BLOCK)
        d8, ds, _ = fake_quantize_fp8_block(d, BLOCK)
        srow = shard // BLOCK[0]

        exp8 = block_fp8.experts
        exp8.gate_up_proj_fp8.data[e, :shard] = g8.view(torch.uint8)
        exp8.gate_up_proj_fp8.data[e, shard:] = u8.view(torch.uint8)
        exp8.gate_up_proj_scale_inv.data[e, :srow] = gs
        exp8.gate_up_proj_scale_inv.data[e, srow:] = us
        exp8.down_proj_fp8.data[e] = d8.view(torch.uint8)
        exp8.down_proj_scale_inv.data[e] = ds

        # fp32 dequant (not bf16) so both paths multiply identical values.
        g32 = dequantize_fp8_block_weight(g8, gs, BLOCK, out_dtype=torch.float32)
        u32 = dequantize_fp8_block_weight(u8, us, BLOCK, out_dtype=torch.float32)
        d32 = dequantize_fp8_block_weight(d8, ds, BLOCK, out_dtype=torch.float32)
        exp16 = block_bf16.experts
        exp16.gate_up_proj.data[e, :shard] = g32
        exp16.gate_up_proj.data[e, shard:] = u32
        exp16.down_proj.data[e] = d32
        refs.append((g8, u8, d8))
    return refs


def test_fp8_resident_dispatch_matches_bf16_dispatch():
    torch.manual_seed(1)
    cfg8 = Glm52ModelConfig.reduced_fp8(block=BLOCK)
    cfg = Glm52ModelConfig.reduced()
    block8 = Glm52SparseMoeBlock(cfg8)
    block = Glm52SparseMoeBlock(cfg)
    assert block8.fp8_experts and not block.fp8_experts

    block8.gate.weight.data.normal_(0, 0.1)
    block8.gate.e_score_correction_bias.data.normal_(0, 0.5)
    block.gate.load_state_dict(block8.gate.state_dict())
    for p, p8 in zip(
        block.shared_expert.parameters(),
        block8.shared_expert.parameters(), strict=True,
    ):
        p8.data.normal_(0, 0.05)
        p.data.copy_(p8.data)

    _fill_expert_weights(block8, block, cfg)

    x = torch.randn(6, cfg.hidden_size) * 0.1
    out8 = block8(x)
    out = block(x)
    # Same loop shape, same fp32 operands -> bitwise equal.
    assert torch.equal(out8, out)


def test_submodule_refuses_post_load_dtype_cast():
    """The engine manager casts the submodule to bf16 AFTER get_submodule;
    that must not re-narrow the fp32 scales/bias restore_fp32_params set."""
    from mstar.model.glm52.submodules import Glm52LLMSubmodule

    torch.manual_seed(3)
    cfg = Glm52ModelConfig.reduced_fp8(block=BLOCK)
    lm = Glm52ForCausalLM(cfg)
    moe = lm.model.layers[1].mlp
    moe.experts.gate_up_proj_scale_inv.data.normal_(0, 0.1)
    moe.gate.e_score_correction_bias.data.normal_(0, 0.5)
    scales_before = moe.experts.gate_up_proj_scale_inv.data.clone()
    bias_before = moe.gate.e_score_correction_bias.data.clone()

    sub = Glm52LLMSubmodule(language_model=lm, config=cfg)
    sub.to(device="cpu", dtype=torch.bfloat16)  # the engine manager's call

    assert moe.experts.gate_up_proj_scale_inv.dtype == torch.float32
    assert moe.gate.e_score_correction_bias.dtype == torch.float32
    assert torch.equal(moe.experts.gate_up_proj_scale_inv.data, scales_before)
    assert torch.equal(moe.gate.e_score_correction_bias.data, bias_before)
    assert moe.experts.gate_up_proj_fp8.dtype == torch.uint8


def test_moe_quant_kernel_resolution():
    """Kimi quant_kernel semantics: auto probes, triton must not downgrade,
    reference (the default until the M1 baseline is banked) keeps the loop."""
    cfg = Glm52ModelConfig.reduced_fp8(block=BLOCK)
    assert cfg.moe_quant_kernel == "reference"

    block = Glm52SparseMoeBlock(cfg)
    block.process_weights_after_loading("cpu")
    assert block._use_fused is False

    cfg_auto = Glm52ModelConfig.reduced_fp8(block=BLOCK)
    cfg_auto.moe_quant_kernel = "auto"
    block_auto = Glm52SparseMoeBlock(cfg_auto)
    block_auto.process_weights_after_loading("cpu")
    assert block_auto._use_fused is False  # no CUDA here -> reference

    cfg_triton = Glm52ModelConfig.reduced_fp8(block=BLOCK)
    cfg_triton.moe_quant_kernel = "triton"
    block_triton = Glm52SparseMoeBlock(cfg_triton)
    try:
        block_triton.process_weights_after_loading("cpu")
        raise AssertionError("explicit triton on CPU must not silently downgrade")
    except RuntimeError:
        pass


def test_cuda_graphs_return_with_fused_dispatch():
    """Graph configs are gated on the RESOLVED dispatch: reference -> none
    (uncapturable host loops), fused -> registered."""
    import torch as _torch

    from mstar.model.glm52.submodules import Glm52LLMSubmodule

    cfg = Glm52ModelConfig.reduced_fp8(block=BLOCK)
    sub = object.__new__(Glm52LLMSubmodule)
    sub.config = cfg

    lm = _torch.nn.Module()
    lm.blk = Glm52SparseMoeBlock(cfg)
    # object.__new__ skipped nn.Module.__init__, so bypass its __setattr__.
    object.__setattr__(sub, "language_model", lm)

    lm.blk._use_fused = False
    assert sub.get_cuda_graph_configs(_torch.device("cpu")) == []
    lm.blk._use_fused = True
    assert len(sub.get_cuda_graph_configs(_torch.device("cpu"))) == 2


def test_autocast_then_restore_keeps_fp8_layout():
    cfg8 = Glm52ModelConfig.reduced_fp8(block=BLOCK)
    block8 = Glm52SparseMoeBlock(cfg8)
    bytes_before = block8.experts.gate_up_proj_fp8.data.clone()

    block8.to(torch.bfloat16)  # the submodule-load autocast
    assert block8.experts.gate_up_proj_fp8.dtype == torch.uint8  # ints immune
    assert block8.experts.gate_up_proj_scale_inv.dtype == torch.bfloat16
    assert block8.gate.e_score_correction_bias.dtype == torch.bfloat16

    restore_fp32_params(block8)
    assert block8.experts.gate_up_proj_scale_inv.dtype == torch.float32
    assert block8.experts.down_proj_scale_inv.dtype == torch.float32
    assert block8.gate.e_score_correction_bias.dtype == torch.float32
    assert torch.equal(block8.experts.gate_up_proj_fp8.data, bytes_before)


def _fp8_entry(state, refs, base, shape):
    w = torch.randn(*shape) * 0.1
    w8, s, deq = fake_quantize_fp8_block(w, BLOCK)
    state.append((f"{base}.weight", w8))
    state.append((f"{base}.weight_scale_inv", s))
    refs[base] = (w8, s, deq)


def _fabricate_checkpoint(cfg, include_mtp=False):
    """HF-style stream for the reduced config + poison keys that must skip.

    ``include_mtp=True`` (M3): emit layer ``num_hidden_layers`` as a full,
    properly-paired MTP layer (decoder inventory via the same loop body —
    the position is FULL by the IndexShare formula — plus the glue keys)
    instead of the poison keys."""
    state: list[tuple[str, torch.Tensor]] = []
    refs: dict[str, tuple] = {}
    hid, q_lora, kv_lora = cfg.hidden_size, cfg.q_lora_rank, cfg.kv_lora_rank
    heads = cfg.num_attention_heads

    state.append(("model.embed_tokens.weight",
                  torch.randn(cfg.vocab_size, hid).bfloat16()))
    state.append(("lm_head.weight", torch.randn(cfg.vocab_size, hid).bfloat16()))
    state.append(("model.norm.weight", torch.randn(hid).bfloat16()))

    for n in range(cfg.num_hidden_layers + (1 if include_mtp else 0)):
        p = f"model.layers.{n}"
        state.append((f"{p}.input_layernorm.weight", torch.randn(hid).bfloat16()))
        state.append((f"{p}.post_attention_layernorm.weight",
                      torch.randn(hid).bfloat16()))
        _fp8_entry(state, refs, f"{p}.self_attn.q_a_proj", (q_lora, hid))
        state.append((f"{p}.self_attn.q_a_layernorm.weight",
                      torch.randn(q_lora).bfloat16()))
        _fp8_entry(state, refs, f"{p}.self_attn.q_b_proj",
                   (heads * cfg.qk_head_dim, q_lora))
        _fp8_entry(state, refs, f"{p}.self_attn.kv_a_proj_with_mqa",
                   (kv_lora + cfg.qk_rope_head_dim, hid))
        state.append((f"{p}.self_attn.kv_a_layernorm.weight",
                      torch.randn(kv_lora).bfloat16()))
        _fp8_entry(state, refs, f"{p}.self_attn.kv_b_proj",
                   (heads * (cfg.qk_nope_head_dim + cfg.v_head_dim), kv_lora))
        _fp8_entry(state, refs, f"{p}.self_attn.o_proj",
                   (hid, heads * cfg.v_head_dim))

        # DSA indexer weights exist only on FULL layers: wq_b/wk are fp8
        # pairs, weights_proj bf16, k_norm a full LayerNorm (weight + bias).
        if is_full_indexer_layer(cfg, n):
            ip = f"{p}.self_attn.indexer"
            _fp8_entry(state, refs, f"{ip}.wq_b",
                       (cfg.index_n_heads * cfg.index_head_dim, q_lora))
            _fp8_entry(state, refs, f"{ip}.wk", (cfg.index_head_dim, hid))
            state.append((f"{ip}.weights_proj.weight",
                          torch.randn(cfg.index_n_heads, hid).bfloat16()))
            state.append((f"{ip}.k_norm.weight",
                          torch.randn(cfg.index_head_dim).bfloat16()))
            state.append((f"{ip}.k_norm.bias",
                          torch.randn(cfg.index_head_dim).bfloat16()))

        if n < cfg.first_k_dense_replace:
            inter = cfg.intermediate_size
            _fp8_entry(state, refs, f"{p}.mlp.gate_proj", (inter, hid))
            _fp8_entry(state, refs, f"{p}.mlp.up_proj", (inter, hid))
            _fp8_entry(state, refs, f"{p}.mlp.down_proj", (hid, inter))
        else:
            moe_inter = cfg.moe_intermediate_size
            state.append((f"{p}.mlp.gate.weight",
                          torch.randn(cfg.n_routed_experts, hid).bfloat16()))
            state.append((f"{p}.mlp.gate.e_score_correction_bias",
                          torch.randn(cfg.n_routed_experts).float()))
            for e in range(cfg.n_routed_experts):
                _fp8_entry(state, refs, f"{p}.mlp.experts.{e}.gate_proj",
                           (moe_inter, hid))
                _fp8_entry(state, refs, f"{p}.mlp.experts.{e}.up_proj",
                           (moe_inter, hid))
                _fp8_entry(state, refs, f"{p}.mlp.experts.{e}.down_proj",
                           (hid, moe_inter))
            _fp8_entry(state, refs, f"{p}.mlp.shared_experts.gate_proj",
                       (moe_inter, hid))
            _fp8_entry(state, refs, f"{p}.mlp.shared_experts.up_proj",
                       (moe_inter, hid))
            _fp8_entry(state, refs, f"{p}.mlp.shared_experts.down_proj",
                       (hid, moe_inter))

    mtp = f"model.layers.{cfg.num_hidden_layers}"
    if include_mtp:
        # The DeepSeek-V3 MTP glue; the decoder inventory for this layer
        # already came out of the loop above.
        state.append((f"{mtp}.enorm.weight", torch.randn(hid).bfloat16()))
        state.append((f"{mtp}.hnorm.weight", torch.randn(hid).bfloat16()))
        _fp8_entry(state, refs, f"{mtp}.eh_proj", (hid, 2 * hid))
        state.append((f"{mtp}.shared_head.norm.weight",
                      torch.randn(hid).bfloat16()))
        return state, refs

    # Poison: MTP layer keys (Phase D) — including the MTP block's own
    # indexer, which must skip by layer index even with load_indexer=True.
    # The fp8 weights deliberately have NO scale sibling — if the skip ever
    # ran after the dequant stream instead of before, the stream would raise
    # "unpaired" and this test would fail.
    state.append((f"{mtp}.enorm.weight", torch.randn(cfg.hidden_size).bfloat16()))
    state.append((f"{mtp}.mlp.experts.0.gate_proj.weight",
                  torch.randn(cfg.moe_intermediate_size, cfg.hidden_size)
                  .to(torch.float8_e4m3fn)))
    state.append((f"{mtp}.self_attn.indexer.wq_b.weight",
                  torch.randn(cfg.index_n_heads * cfg.index_head_dim,
                              cfg.q_lora_rank).to(torch.float8_e4m3fn)))
    return state, refs


def test_load_end_to_end_reduced_fp8():
    torch.manual_seed(2)
    cfg = Glm52ModelConfig.reduced_fp8(block=BLOCK)
    model = Glm52ForCausalLM(cfg)
    state, refs = _fabricate_checkpoint(cfg)

    loaded = load_glm52_hf_weights(
        model, iter(state), cfg.n_routed_experts,
        quant_config=cfg.quantization_config, fp8_experts=True,
        num_hidden_layers=cfg.num_hidden_layers,
    )

    # Completeness both ways: every model param got a tensor, nothing extra.
    assert loaded == set(dict(model.named_parameters()))

    # Dense fp8 modules dequantized bit-exactly (bf16 -> fp32 copy is exact).
    q_a = model.model.layers[0].self_attn.q_a_proj.weight
    _, _, deq = refs["model.layers.0.self_attn.q_a_proj"]
    assert torch.equal(q_a.data, deq.to(q_a.dtype))

    # Dense-layer MLP landed in the fused gate_up param.
    mlp0 = model.model.layers[0].mlp
    inter = cfg.intermediate_size
    _, _, gate_deq = refs["model.layers.0.mlp.gate_proj"]
    _, _, up_deq = refs["model.layers.0.mlp.up_proj"]
    assert torch.equal(mlp0.gate_up_proj.weight.data[:inter], gate_deq.float())
    assert torch.equal(mlp0.gate_up_proj.weight.data[inter:], up_deq.float())

    # Routed experts are fp8-resident: exact bytes and fp32 scales.
    moe = model.model.layers[1].mlp
    shard = cfg.moe_intermediate_size
    srow = shard // BLOCK[0]
    for e in range(cfg.n_routed_experts):
        g8, gs, _ = refs[f"model.layers.1.mlp.experts.{e}.gate_proj"]
        u8, us, _ = refs[f"model.layers.1.mlp.experts.{e}.up_proj"]
        d8, ds, _ = refs[f"model.layers.1.mlp.experts.{e}.down_proj"]
        got = moe.experts.gate_up_proj_fp8.data[e]
        assert torch.equal(got[:shard], g8.view(torch.uint8))
        assert torch.equal(got[shard:], u8.view(torch.uint8))
        assert torch.equal(moe.experts.gate_up_proj_scale_inv.data[e, :srow], gs)
        assert torch.equal(moe.experts.gate_up_proj_scale_inv.data[e, srow:], us)
        assert torch.equal(moe.experts.down_proj_fp8.data[e], d8.view(torch.uint8))
        assert torch.equal(moe.experts.down_proj_scale_inv.data[e], ds)
    assert moe.experts.gate_up_proj_scale_inv.dtype == torch.float32

    # Shared expert dequantized into the fused ParallelGatedMLP param.
    _, _, sg_deq = refs["model.layers.1.mlp.shared_experts.gate_proj"]
    shared = moe.shared_expert.gate_up_proj.weight.data
    assert torch.equal(shared[:shard], sg_deq.float())

    # Router selection bias stays fp32.
    assert moe.gate.e_score_correction_bias.dtype == torch.float32


def test_per_token_group_quant_is_compiler_disabled():
    """The quant wrapper must stay outside torch.compile: Inductor's
    recompile of its Triton kernel crashes (PassManager::run failed,
    08-07 — in-process and subprocess alike) and killed all 296 graph
    captures. The graph break keeps compile+graphs coexisting; this pins
    the wrap so a refactor can't silently drop it."""
    from mstar.utils.fused_moe.kernels import per_token_group_quant_fp8

    assert hasattr(per_token_group_quant_fp8, "_torchdynamo_disable") or hasattr(
        per_token_group_quant_fp8, "_torchdynamo_orig_callable"
    ), "per_token_group_quant_fp8 lost its torch.compiler.disable wrap"
