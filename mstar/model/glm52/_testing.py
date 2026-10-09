"""Test-support helpers for the GLM-5.2 fp8-block path."""
from __future__ import annotations

import torch

from mstar.engine.resources.base import AttentionResource, Resource
from mstar.engine.resources.runner import StepRunner
from mstar.model.glm52.quantization import FP8_DTYPE, dequantize_fp8_block_weight


def fake_quantize_fp8_block(
    weight: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (fp8 weight, fp32 scale_inv, exact bf16 dequantized reference)."""
    out_f, in_f = weight.shape
    bo, bi = block_size
    n_bo, n_bi = -(-out_f // bo), -(-in_f // bi)

    w = weight.to(torch.float32)
    padded = torch.zeros(n_bo * bo, n_bi * bi, dtype=torch.float32)
    padded[:out_f, :in_f] = w
    blocks = padded.view(n_bo, bo, n_bi, bi)
    amax = blocks.abs().amax(dim=(1, 3))  # (n_bo, n_bi)
    scale_inv = amax / 448.0  # e4m3 max normal value
    scale_inv = torch.where(scale_inv == 0, torch.ones_like(scale_inv), scale_inv)

    scale_bc = scale_inv.repeat_interleave(bo, dim=0)[:out_f]
    scale_bc = scale_bc.repeat_interleave(bi, dim=1)[:, :in_f]
    w_fp8 = (w / scale_bc).to(FP8_DTYPE)

    dequant = dequantize_fp8_block_weight(w_fp8, scale_inv, block_size=block_size)
    return w_fp8, scale_inv, dequant


class ReferenceAttentionResource(AttentionResource):
    """CPU stand-in for the naive-path attention resource (the FlashInfer K/V backend):
    dense causal attention over the rows a real NHD ``KVManager`` holds, planned from
    that manager's plan output.
    """

    def __init__(self, kv_cache: str = "kv"):
        self._kv_name = kv_cache
        self._views = None

    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError("test helper; construct directly")

    def depends_on(self):
        return {self._kv_name}

    def plan(self, step, ctx):
        self.reset_default_cursors()
        self._views = ctx.plan_results[self._kv_name]["main"].views

    def qo_indptr_buf(self, label: str = "main"):
        return None

    @torch.compiler.disable
    def run(self, q, label=None, kv_cache_layer=None, k=None, v=None, layer_idx=None):
        del label, k, v, layer_idx
        page_size = kv_cache_layer.shape[2]
        scale = q.shape[-1] ** -0.5
        outs = []
        row = 0
        for view in self._views:
            pages = torch.as_tensor(view.page_idxs, dtype=torch.long)
            # (pages, 2, page_size, H, D) -> (tokens, 2, H, D)
            rows = kv_cache_layer[pages].transpose(0, 1).reshape(
                2, len(view.page_idxs) * page_size, *kv_cache_layer.shape[3:]
            )[:, :view.length]
            keys, vals = rows[0].float(), rows[1].float()
            start = view.length - view.to_compute
            for j in range(view.to_compute):
                qj = q[row + j].float()
                scores = torch.einsum("hd,thd->ht", qj, keys[:start + j + 1]) * scale
                attn = torch.softmax(scores, dim=-1)
                outs.append(torch.einsum("ht,thd->hd", attn, vals[:start + j + 1]).to(q.dtype))
            row += view.to_compute
        return torch.stack(outs)


class GreedySampler(Resource):
    """Argmax per row; enough for the greedy paths the CPU tests drive."""

    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError("test helper; construct directly")

    def sample(self, request_ids, logits, **kwargs):
        del request_ids, kwargs
        return logits.argmax(dim=-1)


class _StubTransferManager:
    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def owns_transfer_info(self, transfer_info, **kwargs):
        del transfer_info, kwargs
        return False

    def remove_request(self, request_id):
        del request_id


def build_cpu_resources(
    config, request_ids: list[str] = (), max_num_pages: int = 64,
    page_size: int = 8,
) -> tuple[dict[str, Resource], StepRunner]:
    """The node's three resources on CPU: a real ``KVManager`` (MLA latent or
    NHD, per ``config.mla_absorb``), the matching attention resource (the
    real MLA one on its torch fallback, or the reference K/V one) and a
    greedy sampler, plus a ``StepRunner`` over them. Requests in
    ``request_ids`` are ingested.
    """
    from mstar.engine.resources.attn.flashinfer_mla import FlashInferMLAManager
    from mstar.engine.resources.kv import manager as manager_mod
    from mstar.engine.resources.kv.config import KVLayout, PagedKVConfig
    from mstar.engine.resources.kv.manager import KVManager
    from mstar.model.glm52.config import (
        ATTN_RESOURCE,
        INDEX_KV_RESOURCE,
        KV_RESOURCE,
        SAMPLER_RESOURCE,
    )

    num_layers = config.num_hidden_layers + (1 if config.mtp_num_draft_tokens > 0 else 0)
    device = torch.device("cpu")
    stub = manager_mod.KVTransferManager
    manager_mod.KVTransferManager = _StubTransferManager
    try:
        if config.mla_absorb:
            kv_cfg = PagedKVConfig(
                num_layers=num_layers, num_kv_heads=1, head_dim=config.cache_latent_dim,
                max_seq_len=config.max_seq_len, max_num_pages=max_num_pages,
                page_size=page_size, num_qo_heads=config.num_attention_heads,
                layout=KVLayout.MLA, kv_lora_rank=config.kv_lora_rank,
                qk_rope_head_dim=config.qk_rope_head_dim,
            )
            kv = KVManager(
                cfg=kv_cfg, name=KV_RESOURCE, joint_comm_group=None,
                transfer_engine_info=None, device=device, dtype=torch.float32,
            )
            class _ReplannedMLA(FlashInferMLAManager):
                # EagerPiecewiseRunner plans before every region call, so a leased
                # fallback plan is never replayed stale
                @property
                def uses_kernel(self) -> bool:
                    return True

            attn = _ReplannedMLA(
                kv_cache=KV_RESOURCE, device=device, dtype=torch.float32,
                kv_config=kv_cfg, sm_scale=config.qk_head_dim ** -0.5,
            )
        else:
            kv_cfg = PagedKVConfig(
                num_layers=num_layers, num_kv_heads=config.num_attention_heads,
                head_dim=config.padded_head_dim, max_seq_len=config.max_seq_len,
                max_num_pages=max_num_pages, page_size=page_size,
                num_qo_heads=config.num_attention_heads,
            )
            kv = KVManager(
                cfg=kv_cfg, name=KV_RESOURCE, joint_comm_group=None,
                transfer_engine_info=None, device=device, dtype=torch.float32,
            )
            attn = ReferenceAttentionResource(KV_RESOURCE)
        index = None
        if config.dsa_long_context:
            from mstar.model.glm52.components.indexer import full_indexer_layers

            d = config.index_head_dim
            index = KVManager(
                cfg=PagedKVConfig(
                    num_layers=len(full_indexer_layers(config)) + (
                        1 if config.mtp_num_draft_tokens > 0 else 0),
                    num_kv_heads=1, head_dim=d,
                    max_seq_len=config.max_seq_len, max_num_pages=max_num_pages,
                    page_size=page_size, layout=KVLayout.MLA, kv_lora_rank=d,
                    qk_rope_head_dim=0),
                name=INDEX_KV_RESOURCE, joint_comm_group=None, transfer_engine_info=None,
                device=device, dtype=torch.float32,
            )
    finally:
        manager_mod.KVTransferManager = stub
    resources = {KV_RESOURCE: kv, ATTN_RESOURCE: attn, SAMPLER_RESOURCE: GreedySampler()}
    if index is not None:
        resources[INDEX_KV_RESOURCE] = index
    runner = StepRunner(resources, node_resources={"LLM": list(resources)})
    for rid in request_ids:
        runner.ingest_request(rid)
    return resources, runner


def build_random_model(cfg, seed: int):
    """A reduced absorbed model with EVERY parameter randomized (MoE expert containers are raw
    ``torch.empty`` at construction — garbage in them NaNs the logits)."""
    from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM
    from mstar.model.glm52.components.moe import Glm52SparseMoeBlock
    from mstar.model.glm52.quantization import process_weights_after_loading

    torch.manual_seed(seed)
    model = Glm52ForCausalLM(cfg)
    model.model.embed_tokens.weight.data.normal_(0, 0.05)
    model.model.norm.weight.data.normal_(1.0, 0.02)
    model.lm_head.weight.data.normal_(0, 0.02)
    for layer in model.model.layers:
        a = layer.self_attn
        for lin in (a.q_a_proj, a.q_b_proj, a.kv_a_proj_with_mqa,
                    a.kv_b_proj, a.o_proj):
            lin.weight.data.normal_(0, 0.03)
        for norm in (a.q_a_layernorm, a.kv_a_layernorm):
            norm.weight.data.normal_(1.0, 0.02)
        if a.indexer is not None:
            a.indexer.wq_b.weight.data.normal_(0, 0.05)
            a.indexer.wk.weight.data.normal_(0, 0.05)
            a.indexer.weights_proj.weight.data.normal_(0, 0.05)
            a.indexer.k_norm.weight.data.normal_(1.0, 0.02)
            a.indexer.k_norm.bias.data.normal_(0, 0.02)
        layer.input_layernorm.weight.data.normal_(1.0, 0.02)
        layer.post_attention_layernorm.weight.data.normal_(1.0, 0.02)
        mlp = layer.mlp
        if isinstance(mlp, Glm52SparseMoeBlock):
            mlp.gate.weight.data.normal_(0, 1)
            mlp.gate.e_score_correction_bias.data = torch.randn(
                cfg.n_routed_experts, dtype=torch.float32)
            mlp.experts.gate_up_proj.data.normal_(0, 0.05)
            mlp.experts.down_proj.data.normal_(0, 0.05)
            mlp.shared_expert.gate_up_proj.weight.data.normal_(0, 0.05)
            mlp.shared_expert.down_proj.weight.data.normal_(0, 0.05)
        else:
            mlp.gate_up_proj.weight.data.normal_(0, 0.05)
            mlp.down_proj.weight.data.normal_(0, 0.05)
    process_weights_after_loading(model, torch.device("cpu"))
    return model.eval()
