"""Test-support helpers for the GLM-5.2 fp8-block path and its CPU resources and handles."""
from __future__ import annotations

import dataclasses
import itertools

import torch

from mstar.engine.resources import SubmoduleStep
from mstar.engine.resources.base import AttentionResource, Resource
from mstar.engine.resources.runner import StepRunner
from mstar.engine.state import ArrayStep
from mstar.engine.state.adapters import node_handles, node_resources, state_steps
from mstar.engine.state.state_op import bind_state_ops
from mstar.model.glm52.config import ATTN, INDEX_KEYS, KV, MTP_SEED, SAMPLER
from mstar.model.submodule_base import BatchedModelOutput
from mstar.utils.fp8_block import FP8_DTYPE, dequantize_fp8_block_weight

# the key of the latent array's positions, which the declarations build
POSITIONS = "rope"


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


def cpu_rope_apply_qk_inplace(
    q, k, pos_ids, cos_sin_cache, rotary_dim, interleave, rope_scale, rope_theta, **llama31,
) -> None:
    """CPU stand-in for ``mstar::rope_apply_qk_inplace``, which has CUDA and XPU kernels only:
    rotates the first ``rotary_dim`` dims of each head of ``q`` and ``k`` in place from the
    position resource's fp32 table, in fp32, as FlashInfer's cos/sin-cache kernel does."""
    del rotary_dim, rope_scale, rope_theta, llama31  # all encoded in the table
    assert cos_sin_cache is not None, "GLM-5.2 rotates from its fp32 table (its Rope's table_dtype)"
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


def cpu_model(config):
    """A byte-tokenized `Glm52Model` around ``config``, whose declarations the CPU resources and handles take."""
    from mstar.model.glm52.glm52_model import Glm52Model

    model = Glm52Model(model_path_hf="", tokenizer_mode="byte", config_variant="reduced")
    model.config = config
    return model


def _spec(config, key: str):
    (spec,) = [spec for spec in node_resources(cpu_model(config)) if spec.resource_key == key]
    return spec


def positions_config(config):
    """The latent array's position config, as the declarations build it."""
    return _spec(config, POSITIONS).config


def cpu_positions(config, max_seq_len: int | None = None) -> Resource:
    """The latent array's position resource on CPU. ``rope_dtype=float32``: the tests run fp32, and
    ``apply_qk`` would otherwise round an fp32 q / k to bf16 before rotating."""
    from mstar.engine.resources.position.manager import RopeManager

    return RopeManager(
        config=dataclasses.replace(positions_config(config), rope_dtype=torch.float32),
        device=torch.device("cpu"), max_seq_len=max_seq_len or config.max_seq_len,
    )


def bind_cpu_state(module, config, resources: dict[str, Resource]) -> dict:
    """``module``'s handles over ``resources``, as the engine binds a node's: to a submodule, or to each layer of a
    bare module that takes them."""
    handles = node_handles(cpu_model(config), "LLM", resources)
    bind_state_ops(handles)
    if hasattr(module, "bind_node_state"):
        module.bind_node_state(handles)
    else:
        for each in module.modules():
            if hasattr(each, "bind_state"):
                each.bind_state(handles)
    return handles


def kv_step(handles: dict, segments, walk: str = "prefill") -> SubmoduleStep:
    """A step of ``segments`` that lists the latent array and, where ``handles`` hold them, the index keys, expanded
    as `engine_step` expands a submodule's: what a test driving bare layers opens around their forward."""
    state = {name: ArrayStep() for name in (KV, INDEX_KEYS) if name in handles}
    return state_steps(SubmoduleStep(segments=list(segments), state=state), handles, graph_walk=walk)


def open_step(runner: StepRunner, step, ctx) -> object:
    """Admit and plan ``step``, which `engine_step` expanded, and open its handles, as the engine does before the
    forward. Returns the admit's outcome, and opens nothing on a refusal."""
    step.set_ctx(ctx)
    outcome = runner.admit(step)
    if outcome.ok:
        runner.plan(step)
        step.begin_calls()
    return outcome


def close_step(runner: StepRunner, step) -> None:
    """Check, commit and close ``step`` after its forward, as the engine does."""
    step.check_calls()
    runner.commit(step)
    step.commit_calls()
    step.end_calls()


class ReferenceAttentionResource(AttentionResource):
    """CPU stand-in for the naive-path attention resource (the FlashInfer K/V backend):
    dense causal attention over the rows a real NHD ``KVManager`` holds, planned from
    that manager's plan output.
    """

    requires_kv_write = True

    def __init__(self, kv_cache: str = "kv"):
        self._kv_name = kv_cache
        self._views = None

    def reads_step_kv(self, label: str | None = None) -> bool:
        return False

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

    def select_last_hidden(self, x, label="main"):
        ends = itertools.accumulate(view.to_compute for view in self._views)
        return x[torch.tensor([end - 1 for end in ends], device=x.device)]

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
    """The node's resources on CPU, under the keys the declarations build: a real ``KVManager``
    (MLA latent or NHD, per ``config.mla_absorb``), the matching attention resource (the
    real MLA one on its torch fallback, or the reference K/V one), the positions over the
    cache, a greedy sampler, under ``config.dsa_long_context`` the DSA resource that holds
    the index keys and, under MTP, the seed's pool, plus a ``StepRunner`` over them. Requests
    in ``request_ids`` are ingested.
    """
    from mstar.engine.resources.attn.flashinfer_mla import FlashInferMLAManager
    from mstar.engine.resources.dsa.config import DSAConfig
    from mstar.engine.resources.dsa.manager import DSAManager
    from mstar.engine.resources.kv import manager as manager_mod
    from mstar.engine.resources.kv.config import KVLayout, PagedKVConfig
    from mstar.engine.resources.kv.manager import KVManager
    from mstar.engine.resources.recurrent.pool import RecurrentStatePool

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
                cfg=kv_cfg, name=KV, joint_comm_group=None,
                transfer_engine_info=None, device=device, dtype=torch.float32,
            )
            class _ReplannedMLA(FlashInferMLAManager):
                # the CPU tests plan before every call, so a leased fallback plan is
                # never replayed stale
                @property
                def uses_kernel(self) -> bool:
                    return True

            attn = _ReplannedMLA(
                kv_cache=KV, device=device, dtype=torch.float32,
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
                cfg=kv_cfg, name=KV, joint_comm_group=None,
                transfer_engine_info=None, device=device, dtype=torch.float32,
            )
            attn = ReferenceAttentionResource(KV)
    finally:
        manager_mod.KVTransferManager = stub
    pos = cpu_positions(config)
    resources = {
        KV: kv, ATTN: attn, POSITIONS: pos, SAMPLER: GreedySampler(),
    }
    if config.dsa_long_context:
        resources[INDEX_KEYS] = DSAManager(
            DSAConfig(kv_cache=KV, topk=config.index_topk),
            page_size=page_size, device=device,
        )
    if config.mtp_num_draft_tokens > 0:
        resources[MTP_SEED] = RecurrentStatePool(
            device=device, config=_spec(config, MTP_SEED).config, name=MTP_SEED, kv_dtype=torch.float32,
        )
    runner = StepRunner(resources, node_resources={"LLM": list(resources)})
    for rid in request_ids:
        runner.ingest_request(rid)
    return resources, runner
