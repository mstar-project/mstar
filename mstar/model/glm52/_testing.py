"""Test-support helpers for the GLM-5.2 fp8-block path and its CPU resources."""
from __future__ import annotations

import dataclasses

import torch

from mstar.engine.resources.base import AttentionResource, Resource
from mstar.engine.resources.runner import StepRunner
from mstar.model.submodule_base import BatchedModelOutput
from mstar.utils.fp8_block import FP8_DTYPE, dequantize_fp8_block_weight


def per_request_outputs(raw, request_ids) -> dict[str, dict[str, list[torch.Tensor]]]:
    """A forward's outputs as the engine hands them to ``postprocess``: row ``i`` of each row
    output (a ``[1, ...]`` view) for ``request_ids[i]``, a per-request entry taking precedence."""
    out = BatchedModelOutput.coerce(raw)
    merged = {rid: dict(out.per_rid_outputs.get(rid, {})) for rid in request_ids}
    for name, rows in (out.row_outputs or {}).items():
        for i, rid in enumerate(request_ids):
            merged[rid].setdefault(name, [rows[i:i + 1]])
    return merged


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


# -- DSA's selection in torch: the references the paged path and its kernels are held to ------

def select_topk_causal(
    scores: torch.Tensor, positions: torch.Tensor, topk: int
) -> torch.Tensor:
    """Per-row top-``topk`` key positions of causally masked ``scores``.

    ``scores`` is ``(T, num_keys)`` with every key past a row's position
    already ``-inf``; the result is ``(T, topk)`` int32, -1 padded where a
    row sees fewer than ``topk`` keys. One stable sort for the whole batch:
    no per-token topk launch, no host sync. Ties break toward the earlier
    key position, so a row's selection does not depend on how the batch
    is packed or on which device ran it (torch.topk's tie order is neither).
    """
    num_tokens, num_keys = scores.shape
    k_eff = min(topk, num_keys)
    order = scores.sort(dim=1, descending=True, stable=True).indices[:, :k_eff]
    # the -inf tail sorts last (stable: by ascending index), so rank >= p_t+1
    # is exactly the invisible remainder
    rank = torch.arange(k_eff, device=scores.device)
    invisible = rank.unsqueeze(0) >= positions.unsqueeze(1) + 1
    selection = torch.full(
        (num_tokens, topk), -1, dtype=torch.int32, device=scores.device)
    selection[:, :k_eff] = order.to(torch.int32).masked_fill(invisible, -1)
    return selection


def index_topk(
    q: torch.Tensor, weights: torch.Tensor, keys: torch.Tensor,
    positions: torch.Tensor, topk: int,
) -> torch.Tensor:
    """Top-``topk`` key positions per query row, ``(T, topk)`` int32, -1 padded.

    ``q`` ``(T, H, D)`` and ``weights`` ``(T, H)`` are the indexer's queries and head weights
    (any scales folded in), ``keys`` ``(S, D)`` a request's keys by position and ``positions``
    ``(T,)`` the rows' own: a row sees keys ``0 .. position``, itself included."""
    # score[t, s] = sum_h w[t, h] * relu(q[t, h] . k[s]), in fp32: per-head ReLU BEFORE the
    # weighted sum; the raw weights get no softmax/sigmoid. The engine's bf16 autocast would run
    # both einsums in bf16.
    with torch.autocast(q.device.type, enabled=False):
        dots = torch.einsum("thd,sd->ths", q.float(), keys.float()).relu()
        scores = torch.einsum("th,ths->ts", weights.float(), dots)
    key_pos = torch.arange(keys.shape[0], device=scores.device)
    scores = scores.masked_fill(key_pos.unsqueeze(0) > positions.unsqueeze(1), float("-inf"))
    return select_topk_causal(scores, positions, topk)


def cpu_rope_apply_qk_inplace(
    q, k, pos_ids, cos_sin_cache, rotary_dim, interleave, rope_scale, rope_theta, **llama31,
) -> None:
    """CPU stand-in for ``mstar::rope_apply_qk_inplace``, which has CUDA and XPU kernels only:
    rotates the first ``rotary_dim`` dims of each head of ``q`` and ``k`` in place from the
    position resource's fp32 table, in fp32, as FlashInfer's cos/sin-cache kernel does."""
    del rotary_dim, rope_scale, rope_theta, llama31  # all encoded in the table
    assert cos_sin_cache is not None, "GLM-5.2 rotates from its fp32 table (position_config)"
    half = cos_sin_cache.shape[-1] // 2
    rows = cos_sin_cache[pos_ids.long()]
    cos, sin = rows[:, None, :half], rows[:, None, half:]
    for x in (q, k):
        pe = x[..., : 2 * half].float()
        if interleave:
            x1, x2 = pe[..., 0::2], pe[..., 1::2]
            out = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).flatten(-2)
        else:
            x1, x2 = pe[..., :half], pe[..., half:]
            out = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        x[..., : 2 * half] = out.to(x.dtype)


def install_cpu_rope(monkeypatch) -> None:
    """Route the position resource's RoPE to ``cpu_rope_apply_qk_inplace`` for one test."""
    from mstar.engine.resources.position import rope  # noqa: F401  (registers the op)

    monkeypatch.setattr(torch.ops.mstar, "rope_apply_qk_inplace", cpu_rope_apply_qk_inplace)


def cpu_positions(config, max_seq_len: int | None = None) -> Resource:
    """The model's position resource on CPU. ``rope_dtype=float32``: the tests run fp32, and
    ``apply_qk`` would otherwise round an fp32 q / k to bf16 before rotating."""
    from mstar.engine.resources.position.manager import RopeManager

    return RopeManager(
        config=dataclasses.replace(config.position_config(), rope_dtype=torch.float32),
        device=torch.device("cpu"), max_seq_len=max_seq_len or config.max_seq_len,
    )


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
    """The node's resources on CPU: a real ``KVManager`` (MLA latent or
    NHD, per ``config.mla_absorb``), the matching attention resource (the
    real MLA one on its torch fallback, or the reference K/V one), the
    positions over the cache, a greedy sampler and, under
    ``config.dsa_long_context``, the DSA resource, plus a ``StepRunner`` over
    them. Requests in ``request_ids`` are ingested.
    """
    from mstar.engine.resources.attn.flashinfer_mla import FlashInferMLAManager
    from mstar.engine.resources.dsa.config import INDEX_PLANES, DSAConfig
    from mstar.engine.resources.dsa.manager import DSAManager
    from mstar.engine.resources.kv import manager as manager_mod
    from mstar.engine.resources.kv.config import AuxPlanes, KVLayout, PagedKVConfig
    from mstar.engine.resources.kv.manager import KVManager
    from mstar.model.glm52.config import (
        ATTN_RESOURCE,
        DSA_RESOURCE,
        KV_RESOURCE,
        POS_RESOURCE,
        SAMPLER_RESOURCE,
    )

    num_layers = config.num_hidden_layers + (1 if config.mtp_num_draft_tokens > 0 else 0)
    device = torch.device("cpu")
    stub = manager_mod.KVTransferManager
    manager_mod.KVTransferManager = _StubTransferManager
    try:
        if config.mla_absorb:
            aux = {}
            if config.dsa_long_context:
                from mstar.model.glm52.components.indexer import full_indexer_layers

                # an index plane per FULL layer, and the MTP layer's
                aux[INDEX_PLANES] = AuxPlanes(
                    num_layers=len(full_indexer_layers(config)) + num_layers
                    - config.num_hidden_layers,
                    width=config.index_head_dim)
            kv_cfg = PagedKVConfig(
                num_layers=num_layers, num_kv_heads=1, head_dim=config.cache_latent_dim,
                max_seq_len=config.max_seq_len, max_num_pages=max_num_pages,
                page_size=page_size, num_qo_heads=config.num_attention_heads,
                layout=KVLayout.MLA, kv_lora_rank=config.kv_lora_rank,
                qk_rope_head_dim=config.qk_rope_head_dim, aux_planes=aux,
            )
            kv = KVManager(
                cfg=kv_cfg, name=KV_RESOURCE, joint_comm_group=None,
                transfer_engine_info=None, device=device, dtype=torch.float32,
            )
            class _ReplannedMLA(FlashInferMLAManager):
                # the CPU tests plan before every call, so a leased fallback plan is
                # never replayed stale
                @property
                def uses_kernel(self) -> bool:
                    return True

            attn = _ReplannedMLA(
                kv_cache=KV_RESOURCE, device=device, dtype=torch.float32,
                kv_config=kv_cfg, sm_scale=config.softmax_scale,
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
    finally:
        manager_mod.KVTransferManager = stub
    pos = cpu_positions(config)
    resources = {
        KV_RESOURCE: kv, ATTN_RESOURCE: attn, POS_RESOURCE: pos, SAMPLER_RESOURCE: GreedySampler(),
    }
    if config.dsa_long_context:
        resources[DSA_RESOURCE] = DSAManager(
            DSAConfig(kv_cache=KV_RESOURCE, topk=config.index_topk,
                      sm_scale=config.qk_head_dim ** -0.5, dense_below_topk=True),
            kv_cfg, device,
        )
    runner = StepRunner(resources, node_resources={"LLM": list(resources)})
    for rid in request_ids:
        runner.ingest_request(rid)
    return resources, runner


def build_random_model(cfg, seed: int):
    """A reduced absorbed model with EVERY parameter randomized (MoE expert containers are raw
    ``torch.empty`` at construction — garbage in them NaNs the logits)."""
    from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM
    from mstar.model.glm52.components.moe import Glm52SparseMoeBlock
    from mstar.model.utils import process_weights_after_loading

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
