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
    def __init__(self, transfer_engine_info, kv_cache):
        del transfer_engine_info, kv_cache

    def get_kv_transfer_info(self):
        return None

    def cleanup(self):
        pass


def build_cpu_resources(
    config, request_ids: list[str] = (), max_num_pages: int = 64,
    page_size: int = 8,
) -> tuple[dict[str, Resource], StepRunner]:
    """The node's three resources on CPU: a real ``KVManager`` (MLA latent or
    NHD, per ``config.mla_absorb``), the matching attention resource (the
    real MLA one on its SDPA fallback, or the reference K/V one) and a
    greedy sampler, plus a ``StepRunner`` over them. Requests in
    ``request_ids`` are ingested.
    """
    from mstar.engine.resources.attn.mla import MlaAttentionManager
    from mstar.engine.resources.kv import manager as manager_mod
    from mstar.engine.resources.kv.config import KVConfig, KVLayout
    from mstar.engine.resources.kv.manager import KVManager
    from mstar.model.glm52.config import ATTN_RESOURCE, KV_RESOURCE, SAMPLER_RESOURCE

    num_layers = config.num_hidden_layers + (1 if config.mtp_num_draft_tokens > 0 else 0)
    device = torch.device("cpu")
    stub = manager_mod.KVTransferManager
    manager_mod.KVTransferManager = _StubTransferManager
    try:
        if config.mla_absorb:
            kv_cfg = KVConfig(
                num_layers=num_layers, num_kv_heads=1, head_dim=config.cache_latent_dim,
                max_seq_len=config.max_seq_len, max_num_pages=max_num_pages,
                page_size=page_size, num_qo_heads=config.num_attention_heads,
                layout=KVLayout.MLA,
            )
            kv = KVManager(
                cfg=kv_cfg, name=KV_RESOURCE, joint_comm_group=None,
                transfer_engine_info=None, device=device, dtype=torch.float32,
            )
            attn = MlaAttentionManager(
                kv_cache=KV_RESOURCE, device=device, dtype=torch.float32,
                kv_config=kv_cfg, softmax_scale=config.qk_head_dim ** -0.5,
                ckv_dim=config.kv_lora_rank,
            )
        else:
            kv_cfg = KVConfig(
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
    resources = {KV_RESOURCE: kv, ATTN_RESOURCE: attn, SAMPLER_RESOURCE: GreedySampler()}
    runner = StepRunner(resources, node_resources={"LLM": list(resources)})
    for rid in request_ids:
        runner.ingest_request(rid)
    return resources, runner


class EagerPiecewiseRunner:
    """A ``PiecewiseCudaGraphRunner`` stand-in that runs the region's Python
    every call instead of replaying a graph: same declare → admit → plan →
    region → commit cycle over the real resources, same static-buffer
    contract (inputs copied into runner-owned buffers padded to the
    bucket), so the captured regions and their step declarations are
    exercised on CPU.
    """

    def __init__(self, label, config, resources, runner, batch_sizes=(1, 2, 4)):
        from mstar.engine.cuda_graph_config import (
            PiecewiseCallInputs,
            PiecewiseConfigType,
        )
        from mstar.engine.cuda_graph_runner import DummyRowPool
        from mstar.model.submodule_base import ModelInputsFromEngine

        self._label = label
        self._config = config
        self._resources = resources
        self._runner = runner
        self._batch_sizes = sorted(batch_sizes)
        self._shapes = {
            (s.bs, s.total_tokens): s for s in config.get_capture_shapes(self._batch_sizes)
        }
        self._packed = config.get_config_type() == PiecewiseConfigType.PACKED
        self._dummy = DummyRowPool(prefix=f"pw_{label}", step_runner=runner, resources=resources)
        self._call_inputs_cls = PiecewiseCallInputs
        self._engine_inputs_cls = ModelInputsFromEngine
        self.calls = 0
        self.staged = 0
        self._static = {}
        # like the real runner's prepare_for_capture: the resources size their
        # static per-slot buffers for the largest bucket, and every replay
        # here carries a lease so they take their captured-graph code paths
        from mstar.engine.resources import CGSlotSpec
        from mstar.engine.resources.step import BucketKey

        self._buckets = {
            key: BucketKey(graph_walk="piecewise", bs=s.bs, num_tokens=s.total_tokens,
                           cg_key_info=label)
            for key, s in self._shapes.items()
        }
        if self._shapes:
            runner.build_cuda_graph_buffers(
                [CGSlotSpec(bucket=b, slot=0, config=config) for b in self._buckets.values()],
                max_bs=max(s.bs for s in self._shapes.values()),
                max_seq_len=max(s.total_tokens for s in self._shapes.values()),
            )

    def _resolve(self, bs, total_tokens):
        padded = next((b for b in self._batch_sizes if b >= bs), None)
        if padded is None:
            return None
        if not self._packed:
            return self._shapes.get((padded, self._config.seq_len * padded))
        cands = sorted(t for (b, t) in self._shapes if b == padded and t >= total_tokens)
        return self._shapes[(padded, cands[0])] if cands else None

    def can_run(self, batch_size, total_tokens=None):
        return self._resolve(batch_size, total_tokens) is not None

    def _buffers_for(self, shape):
        key = (shape.bs, shape.total_tokens)
        if key not in self._static:
            self._static[key] = self._config.make_static_inputs(shape)
        return self._static[key]

    def _prepare(self, static_inputs, request_ids, seq_lens, step_kwargs):
        """Copy the inputs into the bucket's (persistent) buffers, declare and
        plan the step — the half of a replay ``stage`` and ``run`` share."""
        from mstar.engine.resources import SlotLease, StepContext

        real_bs = len(request_ids)
        total = sum(seq_lens) if seq_lens is not None else None
        shape = self._resolve(real_bs, total if self._packed else None)
        assert shape is not None
        dummy_rids = self._dummy.ensure(f"{shape.bs}_{shape.total_tokens}", shape.bs)
        step_ids = [*request_ids, *dummy_rids[real_bs:shape.bs]]
        buffers = self._buffers_for(shape)
        for name, value in static_inputs.items():
            buf = buffers.get(name)
            if buf is None or not isinstance(value, torch.Tensor):
                continue
            n = value.shape[0]
            buf[:n].copy_(value)
            if n < buf.shape[0]:
                buf[n:].zero_()
        step = self._config.declare_step(
            list(step_ids), self._config.replay_seq_lens(shape, seq_lens, real_bs),
            **(step_kwargs or {}),
        )
        bucket = self._buckets[(shape.bs, shape.total_tokens)]
        step.set_ctx(StepContext(
            request_ids=tuple(step_ids), graph_walk="piecewise", slot=0, capture=False,
            slot_lease=SlotLease(slot=0, bucket=bucket),
        ))
        assert self._runner.admit(step).ok
        self._runner.plan(step)
        return shape, step, step_ids, buffers, dummy_rids, real_bs, total

    def stage(self, static_inputs, request_ids=None, seq_lens=None, real_bs=None,
              step_kwargs=None):
        shape, _step, _ids, _bufs, dummy_rids, real_bs, _ = self._prepare(
            static_inputs, request_ids, seq_lens, step_kwargs)
        self._dummy.reset(dummy_rids[real_bs:shape.bs])
        self.staged += 1

    def run(self, static_inputs, request_ids=None, seq_lens=None, real_bs=None,
            step_kwargs=None):
        shape, step, step_ids, buffers, dummy_rids, real_bs, total = self._prepare(
            static_inputs, request_ids, seq_lens, step_kwargs)
        call = self._call_inputs_cls(
            static_inputs=buffers,
            engine_inputs=self._engine_inputs_cls(
                request_ids=list(step_ids), per_request_info={}, resources=dict(self._resources),
            ),
            kwargs=self._config.forward_kwargs,
        )
        try:
            out = self._config.capture_fn(call)
            self._runner.commit(step)
        finally:
            self._dummy.reset(dummy_rids[real_bs:shape.bs])
        self.calls += 1
        if isinstance(out, torch.Tensor):
            out = {"x": out}
        real_len = total if self._packed else real_bs

        class _Out(dict):
            def get_view(self, key, default=None):
                v = self.get(key, default)
                return v if v is None else v[:real_len]

        return _Out({k: (v[:real_len] if isinstance(v, torch.Tensor) else v) for k, v in out.items()})
