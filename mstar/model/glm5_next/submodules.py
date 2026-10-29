"""AR submodule for the GLM-5.3-Flash text backbone on the resource-pool engine."""
from __future__ import annotations

import contextlib
import logging
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig,
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
)
from mstar.engine.accelerator_graph_runner import DEFAULT_CAPTURE_BATCH_SIZES
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import (
    DRAFT_VERIFY,
    AttentionStep,
    DraftPolicy,
    KVStep,
    SamplerStep,
    Segment,
    SlotLease,
    SubmoduleStep,
    with_draft_verify,
)
from mstar.engine.resources.dsa import sparse_mla
from mstar.engine.resources.linear_attn import kda_triton
from mstar.engine.resources.linear_attn.config import LinearAttnStep
from mstar.engine.resources.linear_attn.kda_torch import TorchKDAKernels
from mstar.engine.resources.recurrent.config import RecurrentStep
from mstar.model.glm5_next import fused_decode
from mstar.model.glm5_next.config import (
    ATTN,
    DSA,
    KDA,
    KDA_STATE,
    KV_CACHE,
    LABEL,
    MTP_DRAFT,
    MTP_HIDDEN,
    SAMPLER,
    Glm5NextModelConfig,
)
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    ModelInputsFromEngine,
    NodeInputs,
)
from mstar.utils.h2d import PinnedStager

logger = logging.getLogger(__name__)

# Captured prefill shapes when the config names none (prefill_graphs on). Buckets start
# above the MoE decode path's 64 tokens: the fused runner routes padding rows safely.
# 128-token steps to 1024, then 64 up to one request's max, so a long prompt replays at most
# 63 padding tokens; then 256 to 4096, so a step of several prompts pads at most 255.
PREFILL_TOKEN_BUCKETS = (*range(128, 1024, 128), *range(1024, 2049, 64), *range(2304, 4097, 256))
PREFILL_CAPTURE_BATCH_SIZES = (1, 2, 4)


def _fused_kda(config: Glm5NextModelConfig) -> bool:
    """The KDA layers run the fused kernels, whose prefill is the only capturable one."""
    return (fused_decode._ENABLED and kda_triton._HAS_TRITON
            and kda_triton.supported(config.linear_head_dim))


def _real_row_values(last: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """``values`` with each padding row's entry replaced by that of the row it repeats.

    A padding row repeats the previous row's last index (``_last_rows``), so a scatter at
    ``last`` writes two values to one position; this makes them the same one. Device ops
    only, so a captured replay runs it too.
    """
    n = last.shape[0]
    starts = torch.ones(n, dtype=torch.bool, device=last.device)
    starts[1:] = last[1:] != last[:-1]
    rows = torch.where(starts, torch.arange(n, device=last.device), 0)
    return values.index_select(0, torch.cummax(rows, 0).values)


class Glm5NextLLMSubmodule(ARNodeSubmodule):
    """Embed + 45 hybrid decoder layers + lm_head, one fat TP node, and with MTP the
    layer-45 draft module."""

    # The eager prefill forward hosts the KDA resource (@torch.compiler.disable
    # inside a compiled outer frame) and the fp8 reference MoE loop; the
    # post-capture torch.compile of the eager forwards is untested for it.
    disable_torch_compile: bool = True
    # Weights load in bf16; the KDA conv and delta rule, the router and mHC run
    # in fp32 on purpose, which the engine's bf16 autocast would undo.
    disable_autocast: bool = True

    def __init__(self, language_model: nn.Module, config: Glm5NextModelConfig) -> None:
        super().__init__()
        self.language_model = language_model  # Glm5NextForCausalLM
        self.lm_head = language_model.lm_head
        self.config = config
        # MTP: a decode step verifies the last token and k drafts per request. The model drafts
        # (the layer-45 MTP module) and seeds the next step; the engine verifies, settles the
        # KDA state, rewinds the KV, emits and stops (get_draft_policy).
        self.mtp_k = config.mtp_num_draft_tokens
        self._draft_policy = (
            DraftPolicy(max_drafts=self.mtp_k, stop_ids=frozenset(config.eos_token_ids))
            if self.mtp_k > 0 else None
        )
        # the step's small host indices, one pinned H2D each
        self._long_stager = PinnedStager(torch.long)
        self._int_stager = PinnedStager(torch.int32)
        self._keep_stager = PinnedStager(torch.float32)

    @property
    def _dsa(self):
        """The DSA resource (dsa_long_context), whose ``ctx`` the MLA layers read."""
        return self.node_resources.get(DSA) if self.config.dsa_long_context else None

    @property
    def mtp_block(self) -> int:
        return self.mtp_k + 1

    def get_draft_policy(self, graph_walk: str) -> DraftPolicy | None:
        return self._draft_policy if graph_walk == "decode" else None

    def bind_node_resources(self, resources: dict) -> None:
        if KV_CACHE in resources:
            self.config.check_long_context(resources[KV_CACHE].kv_cache.page_size)
        buckets = self.config.prefill_token_buckets or PREFILL_TOKEN_BUCKETS
        if self.config.dsa_long_context and self.config.prefill_graphs and (
                max(buckets) > sparse_mla.MAX_ROWS):
            # a captured prefill plans its whole bucket at once; the eager plan takes chunks
            raise ValueError(f"dsa_long_context captures prefill buckets up to "
                             f"{sparse_mla.MAX_ROWS} tokens; the largest is {max(buckets)}")
        super().bind_node_resources(resources)
        # The fused kernels where they run; the torch reference elsewhere.
        kda = resources.get(KDA)
        if kda is not None and not _fused_kda(self.config):
            kda.set_kernels(TorchKDAKernels())

    # -- CUDA-graph configs ----------------------------------------------

    def _moe_resolved_fused(self) -> bool:
        """True iff the loaded MoE blocks resolved to the clamped fused fp8
        kernel (``moe_quant_kernel`` "triton"/"auto" on CUDA); False on the
        "reference" default, whose per-hit-expert loop hosts ``.nonzero()``
        and cannot be captured.
        """
        from mstar.model.glm5_next.components.moe import Glm5NextSparseMoeBlock

        lm = getattr(self, "language_model", None)
        if lm is None:
            return False
        for module in lm.modules():
            if isinstance(module, Glm5NextSparseMoeBlock):
                return bool(getattr(module, "_use_fused", False))
        return False

    def _moe_capture_blocked(self, tp_world_size: int) -> bool:
        fp8_reference = (
            self.config.quantization_config is not None
            and self.config.moe_fp8_resident
            and not self._moe_resolved_fused()
        )
        naive_tp = self.config.quantization_config is None and tp_world_size > 1
        return fp8_reference or naive_tp

    def _kda_max_slots(self) -> int | None:
        """Requests the KDA pool can hold at once: its slots but the sink."""
        pool = self.node_resources.get(KDA_STATE) if self.node_resources else None
        return None if pool is None else pool.config.usable_slots

    def max_batch_size(self, graph_walk: str) -> int | None:
        # A step can hold at most one slot per row of real state; the
        # scheduler splits anything larger rather than admit-failing it.
        pool = self.node_resources.get(KDA_STATE) if self.node_resources else None
        if pool is None:
            return None
        # A prefill takes only the slots free now, and waits while none are, so a
        # TP follower refusing one has diverged from rank 0 (the engine raises).
        # With prefill graphs, no more rows than one takes: an eager prefill's
        # last layer compiles the MoE decode kernels for its row count mid-serve.
        if graph_walk == "prefill":
            rows = self._captured_prefill_rows()
            free = pool.num_free_slots
            return free if rows is None else min(free, rows)
        return pool.config.usable_slots

    def _captured_prefill_rows(self) -> int | None:
        """The most rows a captured prefill takes, or None while prefill runs eager."""
        if not self.config.prefill_graphs or not _fused_kda(self.config):
            return None
        # declared no prefill graph (reference MoE, the MLA fallback): an eager prefill
        # held to 4 rows took 16 steps for a burst of 64 prompts
        if getattr(self, "_prefill_captured", True) is False:
            return None
        return max(self.config.prefill_capture_batch_sizes or PREFILL_CAPTURE_BATCH_SIZES)

    def max_step_tokens(self, graph_walk: str) -> int | None:
        cap = self.config.prefill_max_step_tokens
        if graph_walk != "prefill" or cap is None:
            return None
        if cap == "auto":
            return max(self.config.prefill_token_buckets or PREFILL_TOKEN_BUCKETS)
        return cap

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        if self._moe_capture_blocked(tp_world_size):
            logger.warning(
                "glm5_next: reference MoE dispatch active; decode runs eager "
                "(set moe_quant_kernel=auto for the capturable fused kernel)"
            )
            self._prefill_captured = False
            return []
        # The MLA fallback bakes page indices into its plan, so a captured
        # replay would attend the capture's pages (the resource refuses).
        attn = self.node_resources.get(ATTN)
        if device.type == "cuda" and attn is not None and not attn.uses_kernel:
            logger.warning(
                "glm5_next: MLA is on the SDPA fallback, which cannot be "
                "captured; decode runs eager"
            )
            self._prefill_captured = False
            return []
        kv = self.node_resources.get(KV_CACHE)
        if (self.config.dsa_long_context and device.type == "cuda" and kv is not None
                and kv.layer_view().dtype != torch.bfloat16):
            # the DSA kernels take a bf16 cache; the torch paths sync with the host
            logger.warning("glm5_next: dsa_long_context on a %s cache cannot be captured; "
                           "decode runs eager", kv.layer_view().dtype)
            self._prefill_captured = False
            return []
        max_slots = self._kda_max_slots()
        batch_sizes = [
            b for b in DEFAULT_CAPTURE_BATCH_SIZES
            if max_slots is None or b <= max_slots
        ]
        # Graphs capture the eager forward: a Triton crash in Inductor's compile
        # subprocess would fail every capture and fall back to eager silently.
        configs: list[AcceleratorGraphConfig] = [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="decode",
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device),
                    input_seq_len=self.mtp_block,
                ),
                capture_batch_sizes=batch_sizes,
                compile=False,
            ),
        ]
        if self.config.prefill_graphs and not _fused_kda(self.config):
            # The reference KDA prefill loops over spans on the host, so a
            # captured replay would run the capture's dummy spans.
            logger.warning("glm5_next: fused KDA prefill off; prefill runs eager")
        elif self.config.prefill_graphs:
            # One graph per (rows, token bucket); a batch past every bucket
            # runs eager rather than being split.
            configs.append(PackedAcceleratorGraphConfig(
                capture_graph_walk="prefill",
                capture_token_lengths=list(
                    self.config.prefill_token_buckets or PREFILL_TOKEN_BUCKETS),
                make_node_input=lambda n: ARNodeInputs(
                    input_ids=torch.zeros(n, dtype=torch.long, device=device),
                    input_seq_len=n,
                ),
                capture_batch_sizes=[
                    b for b in (self.config.prefill_capture_batch_sizes
                                or PREFILL_CAPTURE_BATCH_SIZES)
                    if max_slots is None or b <= max_slots
                ],
                caps_eager_batch_size=False,
                compile=False,
            ))
        self._prefill_captured = any(c.capture_graph_walk == "prefill" for c in configs)
        return configs

    # -- dtype discipline -------------------------------------------------

    def to(self, *args, **kwargs):
        """Honor device moves; refuse post-load dtype casts."""
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is not None:
            logger.info(
                "Glm5NextLLMSubmodule: ignoring post-load dtype cast to %s "
                "(per-param dtypes are fixed at load; see restore_fp32_params).",
                dtype,
            )
        if device is not None:
            return super().to(device=device, non_blocking=non_blocking)
        return self

    # -- engine seam: prepare -> declare -> preprocess -> forward ---------

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> ARNodeInputs:
        text_inputs = inputs["text_inputs"][0]
        seq_len = text_inputs.shape[0]
        if graph_walk == "decode" and self._draft_policy is not None:
            # One id in, the last emitted token: the step verifies it and k drafts. The context
            # is the engine's (its KV length after each verdict), and the conductor holds it,
            # with the verify rows written ahead of a trim (the layout's lookahead), to kv_rows,
            # which MTP holds to the context limit.
            return ARNodeInputs(input_ids=text_inputs, input_seq_len=self.mtp_block)
        # Without dsa_long_context the serving regime is ctx <= index_topk,
        # where dense MLA computes exactly what GLM-5.3-Flash's DSA would —
        # refuse beyond it rather than serve off-spec logits. Raising here
        # fails only this request (the engine registers the failure and
        # drops the rid from the batch).
        limit = self.config.context_limit
        # the handle, which the batch, check_stop and cleanup_request key by
        state = self.request_state(fwd_info.rid_handle)
        committed = state.get("context", 0)
        if committed + seq_len > limit or committed + seq_len > self.config.kv_rows:
            raise RuntimeError(
                f"request {fwd_info.request_id}: context {committed + seq_len} exceeds "
                f"{'max_seq_len' if self.config.dsa_long_context else 'index_topk'}={limit} "
                f"or {self.config.kv_rows} rows; without dsa_long_context the limit is "
                "index_topk, where dense MLA is exactly GLM-5.3-Flash's DSA."
            )
        if graph_walk == "prefill":
            state.add("prompt_len", committed + seq_len)
        return ARNodeInputs(input_ids=text_inputs, input_seq_len=seq_len)

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        prefill_tokens: dict[str, torch.Tensor] = {}
        if graph_walk == "prefill":
            prefill_tokens = {
                rid: inp.input_ids for rid, inp in zip(request_ids, inputs, strict=True)
            }
        steps = {
            KV_CACHE: KVStep(),
            ATTN: AttentionStep(causal=True),
            KDA_STATE: RecurrentStep(),
            KDA: LinearAttnStep(),
        }
        policy = self.get_draft_policy(graph_walk)
        if policy is None:
            steps[SAMPLER] = SamplerStep(apply_penalty=True, prefill_tracked_tokens=prefill_tokens)
        if self.config.dsa_long_context:
            # every MLA layer attends sparse (the DSA resource): no dense plan over the whole context
            del steps[ATTN]
        step = SubmoduleStep(
            segments=[
                Segment(request_id=rid, label=LABEL, span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps=steps,
        )
        # the verify is the sampler's, through the draft-verify resource, which also marks the
        # KDA step: its layers first replay what the last verify step kept
        return step if policy is None else with_draft_verify(step, policy, DRAFT_VERIFY, SAMPLER)

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        # The per-replay-varying inputs. Everything else the forward reads
        # is a resource plan (slot index, KV write slots, planned kernels).
        out = {"input_ids": torch.cat([inp.input_ids for inp in inputs])}
        if graph_walk == "prefill" and engine_inputs.captured:
            out["last_rows"] = self._last_rows(inputs)
        if self.mtp_k > 0 and graph_walk == "prefill":
            out["mtp_keep"] = self._mtp_keep(engine_inputs, inputs, out["input_ids"].shape[0])
        if self.config.dsa_long_context:
            # the DSA resource stages its rows off the KV plan; the forward opens them
            out.update(self._dsa.step_inputs(
                engine_inputs.step.ctx, engine_inputs.captured, LABEL))
        return out

    def _mtp_keep(
        self, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs], num_tokens: int,
    ) -> torch.Tensor:
        """1 per prefill token, 0 where a request's token sits at position 0: the MTP pass
        zeroes its embedding there, as vLLM does."""
        keep, start = [1.0] * num_tokens, 0
        for rid, inp in zip(engine_inputs.request_ids, inputs, strict=True):
            state = self.request_states.get(rid)
            fresh = state is None or state.get("context", 0) == 0
            if fresh and inp.input_seq_len > 0 and start < num_tokens:
                keep[start] = 0.0
            start += inp.input_seq_len
        # staged in fp32 (no bf16 host view), cast on the device
        return self._keep_stager.to_device(keep, self.lm_head.weight.device).to(self.lm_head.weight.dtype)

    def _last_rows(self, inputs: list[ARNodeInputs]) -> torch.Tensor:
        """Each row's last token, which a captured prefill reads at a fixed
        address: the rows it samples."""
        last, end = [], 0
        for inp in inputs:
            end += inp.input_seq_len
            last.append(max(end - 1, 0))  # a padding row repeats the one before
        return self._int_stager.to_device(last, self.lm_head.weight.device)

    def _forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        last_rows: torch.Tensor | None = None,
        mtp_keep: torch.Tensor | None = None,
        dsa_inputs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if self._dsa is not None:
            self._dsa.ctx = self._dsa.context(dsa_inputs or {})
        try:
            return self._forward_inner(graph_walk, engine_inputs, input_ids, last_rows, mtp_keep)
        finally:
            if self._dsa is not None:
                self._dsa.ctx = None

    def _forward_inner(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        last_rows: torch.Tensor | None,
        mtp_keep: torch.Tensor | None,
    ) -> torch.Tensor:
        attn = engine_inputs.resources[ATTN]
        sampler = engine_inputs.resources[SAMPLER]
        if self.config.dsa_long_context and graph_walk == "prefill" and not engine_inputs.captured:
            return self._prefill_windows(engine_inputs, input_ids, sampler, mtp_keep)
        last = None  # prefill samples each row's last token: the model computes only those
        if last_rows is not None:
            last = last_rows
        elif graph_walk == "prefill":
            positions = torch.arange(input_ids.shape[0], device=input_ids.device)
            last = attn.select_last_hidden(positions, LABEL)
        elif graph_walk != "decode":
            raise ValueError(f"unsupported graph walk: {graph_walk!r}")
        if self.mtp_k > 0 and graph_walk == "prefill":
            # the MTP pass reads the trunk's hidden state at every prompt row
            hidden = self.language_model.model(input_ids)
            last = last.long()
            new_tokens = sampler.sample(
                engine_inputs.request_ids, logits=self.lm_head(hidden.index_select(0, last)))
            self._mtp_prefill(engine_inputs, input_ids, hidden, last, new_tokens, mtp_keep)
            return new_tokens
        hidden = self.language_model.model(input_ids, rows=last)
        logits = self.lm_head(hidden)  # (rows, vocab)
        return sampler.sample(engine_inputs.request_ids, logits=logits)

    def _prefill_windows(
        self, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor, sampler,
        mtp_keep: torch.Tensor | None,
    ) -> torch.Tensor:
        """dsa_long_context's eager prefill: the step's tokens in windows of prefill_window_tokens,
        each through every layer before the next (its KV rows and KDA state land first), so
        activations stay bounded however long the prompt. Samples each request's last row.
        With MTP each window's rows also run the MTP pass (row i pairs the trunk at i with token
        i + 1); a request's last row runs it again once its token is sampled, and seeds."""
        ctx = self._dsa.ctx
        kv, kda = engine_inputs.resources[KV_CACHE], engine_inputs.resources[KDA]
        lm, size, total = self.language_model, self.config.prefill_window_tokens, input_ids.shape[0]
        last = [r0 + n - 1 for r0, n, _ in ctx.spans]
        # one window: the step's own plans (one-token prompts plan as decode, unsplittable)
        split = total > size and len(ctx.spans) != total
        hidden: list[torch.Tensor | None] = [None] * len(last)
        nxt = torch.roll(input_ids, -1) if self.mtp_k > 0 else None
        try:
            for start in range(0, total, size if split else total):
                end = min(start + size, total) if split else total
                ends = [(i, r - start) for i, r in enumerate(last) if start <= r < end]
                with contextlib.ExitStack() as scope:
                    if split:
                        pieces = [(max(r0, start), min(r0 + n, end), req)
                                  for r0, n, req in ctx.spans if r0 < end and r0 + n > start]
                        self._dsa.ctx = ctx.rows(start, end)
                        scope.enter_context(kv.token_window(start, end, LABEL))
                        scope.enter_context(kda.token_window(
                            [(req, b - a) for a, b, req in pieces], LABEL))
                    rows = self._long_stager.to_device([r for _, r in ends] or [end - start - 1], input_ids.device)
                    if nxt is None:
                        out = lm.model(input_ids[start:end], rows=rows)
                    else:
                        full = lm.model(input_ids[start:end])
                        keep = mtp_keep[start:end, None]
                        lm.mtp(lm.model.embed_tokens(nxt[start:end]) * keep, full)
                        out = full.index_select(0, rows)
                for k, (i, _) in enumerate(ends):
                    hidden[i] = out[k]
            last_hidden = torch.stack(hidden)
            new_tokens = sampler.sample(engine_inputs.request_ids, logits=self.lm_head(last_hidden))
            if nxt is not None:
                seeds = []
                for i, r in enumerate(last):  # its MTP row again, now with the sampled token
                    self._dsa.ctx = ctx.rows(r, r + 1)
                    with kv.token_window(r, r + 1, LABEL):
                        embeds = lm.model.embed_tokens(new_tokens[i:i + 1]) * mtp_keep[r:r + 1, None]
                        seeds.append(lm.mtp(embeds, last_hidden[i:i + 1]))
                self._mtp_seed(engine_inputs, torch.cat(seeds), new_tokens)
        finally:
            self._dsa.ctx = ctx
        return new_tokens

    # -- MTP device side ---------------------------------------------------

    def _argmax(self, hidden: torch.Tensor) -> torch.Tensor:
        """The LM head's argmax without gathering the logits: each rank's best over its
        contiguous vocab shard, then the best of those. A tie goes to the lower rank, so the
        lower id, as with a full argmax."""
        head = self.lm_head
        local = F.linear(hidden, head.weight)
        idx = local.argmax(-1)
        if head.tp_size == 1:
            return idx
        val = local.gather(-1, idx[:, None]).float()
        idx = idx[:, None] + head.tp_rank * head.output_size_per_partition
        vals = head.comm_group.all_gather(val, dim=-1)
        ids = head.comm_group.all_gather(idx, dim=-1)
        return ids.gather(-1, vals.argmax(-1, keepdim=True)).squeeze(-1)

    def _draft_tokens(self, hidden: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        """The draft for the token after ``prev``, from the MTP state ``hidden``."""
        del prev  # the tests' oracle drafts read it
        return self._argmax(hidden)

    def _mtp_seed(
        self, engine_inputs: ModelInputsFromEngine, hidden: torch.Tensor, prev: torch.Tensor,
    ) -> None:
        """The next step's draft seed in each row's slot: the MTP state and d1, the draft for
        the token after ``prev`` (the step's next input)."""
        pool = engine_inputs.resources[KDA_STATE]
        pool.scatter_(MTP_HIDDEN, hidden, None)
        pool.scatter_(MTP_DRAFT, self._draft_tokens(hidden, prev)[:, None], None)

    def _mtp_prefill(self, engine_inputs, input_ids, hidden, last, new_tokens, mtp_keep) -> None:
        """The MTP pass over the prompt: row i pairs the trunk's state at token i with the
        embedding of token i + 1 (each request's last row: its sampled token). It fills the
        MTP layer's KV plane and drafts d1 from each request's last row."""
        lm = self.language_model
        # roll, not a shifted copy into empty memory: a captured bucket's last slot is
        # padding, and every slot must hold a valid id for the embedding
        nxt = torch.roll(input_ids, -1)
        nxt.index_copy_(0, last, _real_row_values(last, new_tokens).to(nxt.dtype))
        embeds = lm.model.embed_tokens(nxt) * mtp_keep[:, None]
        self._mtp_seed(engine_inputs, lm.mtp(embeds, hidden).index_select(0, last), new_tokens)

    def _mtp_decode(self, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor) -> torch.Tensor:
        """One verify step. Drafts: d1 from the seed, each later one an MTP pass whose query
        sits at its row of this step's block. Verify: the trunk over [last | drafts], its
        logits to the draft-verify resource, which keeps rank 0's drafts and verdict on every
        rank, settles the KDA state and stages the verdict for the host. Seed: the MTP pass
        over the block with the target's token as each row's next (up to the accepted row it
        equals the drafts), read at the last accepted row. Returns the next input per row."""
        lm, k, block = self.language_model, self.mtp_k, self.mtp_block
        pool = engine_inputs.resources[KDA_STATE]
        dv = engine_inputs.resources[DRAFT_VERIFY]
        n = pool.addressing().num_rows

        h = pool.gather(MTP_HIDDEN, None)
        drafts = [pool.gather(MTP_DRAFT, None)[:, 0]]
        attn = lm.mtp.transformer_layer.self_attn
        latents = h.new_zeros(n, block, attn.kv_lora_rank + attn.mla_cache_kpe)
        index_rows = None
        if self.config.dsa_long_context:  # the MTP layer's own index keys and gates
            d = attn.indexer.head_dim
            index_rows = (h.new_zeros(n, block, d), h.new_zeros(n, block, d))
        for row in range(k - 1):
            h = lm.mtp(lm.model.embed_tokens(drafts[-1]), h, row=row, latents=latents,
                       index_rows=index_rows)
            drafts.append(self._draft_tokens(h, drafts[-1]))

        ids = dv.block_ids(input_ids, dv.agree(torch.stack(drafts, 1)))
        hidden = lm.model(ids.view(-1))
        verdict = dv.verify(list(engine_inputs.request_ids), ids, self.lm_head(hidden))

        each = torch.arange(n, device=ids.device)
        next_input = verdict.next_inputs.view(-1)
        h_block = lm.mtp(lm.model.embed_tokens(verdict.tokens.reshape(-1)), hidden)
        self._mtp_seed(engine_inputs, h_block.view(n, block, -1)[each, verdict.accepted], next_input)
        return next_input

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        last_rows: torch.Tensor | None = None,
        mtp_keep: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        if self.get_draft_policy(graph_walk) is not None:
            return self.forward_batched(graph_walk, engine_inputs, input_ids,
                                        **kwargs)[engine_inputs.request_ids[0]]
        return {"new_token": self._forward(
            graph_walk, engine_inputs, input_ids, last_rows, mtp_keep, kwargs)}

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        last_rows: torch.Tensor | None = None,
        mtp_keep: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList] | BatchedModelOutput:
        if self.get_draft_policy(graph_walk) is not None:
            # the engine cuts the emitted tokens from the verdict; the next input is the token
            # after the accepted drafts
            if self._dsa is not None:
                self._dsa.ctx = self._dsa.context(kwargs)
            try:
                next_input = self._mtp_decode(engine_inputs, input_ids)
            finally:
                if self._dsa is not None:
                    self._dsa.ctx = None
            return {
                rid: {"text_inputs": [next_input[i : i + 1]]}
                for i, rid in enumerate(engine_inputs.request_ids)
            }
        new_tokens = self._forward(
            graph_walk, engine_inputs, input_ids, last_rows, mtp_keep, kwargs)
        # row i is request i: one clone out of the graph and one host copy a step
        return BatchedModelOutput(
            row_outputs={"new_token": new_tokens},
            check_stop_buffers={"new_token": new_tokens},
        )

    # -- slow-postprocess path (worker thread, after execute_batch) -------

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        inputs: ARNodeInputs | None = None,
        **kwargs,
    ):
        # The step ran, so its tokens are the request's context now. A verify step keeps what
        # its verdict accepted, which is the engine's to count.
        if inputs is not None and self.get_draft_policy(request_info.graph_walk) is None:
            state = self.request_state(request_id)
            state.add("context", state.get("context", 0) + inputs.input_seq_len)
        # Seed the next decode step from the emitted token: the walk's
        # text_inputs loop-back edge (get_graph_walk_graphs) carries it. An MTP
        # verify step already set it: the token after the accepted drafts.
        if "new_token" not in outputs:
            return
        outputs.setdefault("text_inputs", outputs["new_token"])
        state = self.request_states.get(request_id)
        if state is not None and state.get("ended_at_prefill"):
            # the reply already ended: this step only closes the decode loop
            del outputs["new_token"]

    def check_stop(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        """Plain decode's stop and the prefill's; a verify step's stop is the engine's
        (``get_draft_policy``) and never reaches here."""
        # The ONLY place .item() on a token value is allowed (slow-postprocess
        # path, off the step-critical GPU thread — base-class contract).
        state = self.request_states.get(request_id)
        if state is not None and state.get("ended_at_prefill"):
            return {"decode_loop"}
        if "new_token" not in outputs:
            return set()
        token = outputs["new_token"][0].item()
        is_eos = token in self.config.eos_token_ids
        ignore_eos = request_info.resource_configs[SAMPLER].ignore_eos
        if request_info.graph_walk == "prefill":
            # no decode_loop on this walk to stop: a stop token ends the reply at the first
            # decode step, and the conductor ends a max_tokens of 1 here
            if not ignore_eos and is_eos:
                self.request_state(request_id).add("ended_at_prefill", True)
            return set()
        # Total generated = 1 prefill-emitted token + (decode iters + 1):
        # max_tokens counts the token the prefill emits, so the decode loop
        # runs one fewer.
        generated = request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 2
        # The next step takes the context to prompt + generated, and it may
        # already be scheduled when this stop lands, so stop while it still
        # fits: past the context limit prepare_inputs refuses it.
        prompt_len = state.get("prompt_len", 0) if state is not None else 0
        full = prompt_len + generated >= self.config.context_limit
        if (not ignore_eos and is_eos) or generated >= request_info.max_tokens or full:
            return {"decode_loop"}
        return set()
