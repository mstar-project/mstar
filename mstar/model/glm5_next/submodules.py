"""AR submodule for the GLM-5.3-Flash text backbone on the resource-pool engine."""
from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    DEFAULT_CAPTURE_BATCH_SIZES,
    AcceleratorGraphConfig,
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
)
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import (
    AttentionStep,
    KVStep,
    SamplerStep,
    Segment,
    SlotLease,
    SubmoduleStep,
)
from mstar.engine.resources.convenience import KDACallable
from mstar.engine.resources.dsa.config import DSAStep
from mstar.engine.resources.linear_attn.config import LinearAttnStep
from mstar.engine.resources.recurrent.config import RecurrentStep
from mstar.model.components.cached_prefix import CACHED, EagerPastCachedPrefix
from mstar.model.glm5_next.components.attention import HOLE_KPE
from mstar.model.glm5_next.config import (
    ATTN,
    DSA,
    KDA,
    KDA_STATE,
    KV_CACHE,
    LABEL,
    MTP_BUDGET,
    MTP_DRAFT,
    MTP_HIDDEN,
    MTP_LEN,
    MTP_POOLS,
    SAMPLER,
    Glm5NextModelConfig,
)
from mstar.model.submodule_base import (
    END_PARTITION,
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    HostRows,
    ModelInputsFromEngine,
    NodeInputs,
)
from mstar.utils.h2d import PinnedStager

logger = logging.getLogger(__name__)

# Captured prefill shapes, which a deployment's serving config `prefill_token_buckets` replaces.
# They start above the MoE decode path's 64 tokens, since the fused runner routes padding rows
# safely; 64 apart from 1024 to one request's 2048, then 256 apart to 4096 for steps of several
# prompts.
PREFILL_TOKEN_BUCKETS = (*range(128, 1024, 128), *range(1024, 2049, 64), *range(2304, 4097, 256))
PREFILL_CAPTURE_BATCH_SIZES = (1, 2, 4)


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


def _shard_best(local: torch.Tensor, first_id: int) -> torch.Tensor:
    """Each row's best logit over one rank's vocab shard and its id, as an int32 pair: the
    logit's fp32 bits, then the id (``first_id`` is the shard's first)."""
    idx = local.argmax(-1, keepdim=True)
    val = local.gather(-1, idx).float().view(torch.int32)
    return torch.cat([val, (idx + first_id).to(torch.int32)], -1)



def _upload(stager: PinnedStager, values: list, device: torch.device) -> torch.Tensor:
    """``values`` on ``device`` in ``stager``'s dtype: one non-blocking copy out of its pinned ring."""
    out = torch.empty(len(values), dtype=stager.dtype, device=device)
    stager.copy_(out, values)
    return out

class Glm5NextLLMSubmodule(EagerPastCachedPrefix, ARNodeSubmodule):
    """Embed + 45 hybrid decoder layers + lm_head, one fat TP node, and with MTP the
    layer-45 draft module."""

    # torch.compile of the eager forwards is untested: they host the KDA resource
    # (compiler-disabled) and the reference MoE loop
    disable_torch_compile: bool = True
    # Weights load in bf16; the KDA conv and delta rule, the router and mHC run
    # in fp32 on purpose, which the engine's bf16 autocast would undo.
    disable_autocast: bool = True

    def __init__(self, language_model: nn.Module, config: Glm5NextModelConfig) -> None:
        super().__init__()
        self.language_model = language_model  # Glm5NextForCausalLM
        self.lm_head = language_model.lm_head
        self.config = config
        # MTP: a decode step verifies the last token and k drafts per request
        self.mtp_k = config.mtp_num_draft_tokens
        # accepted-draft counts since the last log line: a histogram over 0..k
        self._mtp_hist = [0] * (self.mtp_k + 1)
        self._mtp_logged = time.monotonic()
        # the step's small host indices, one pinned H2D each
        self._long_stager = PinnedStager(torch.long)
        self._int_stager = PinnedStager(torch.int32)
        self._keep_stager = PinnedStager(torch.float32)
        # the KDA pool and resource, which settle each verify step's accepted drafts
        self._kda: KDACallable | None = None

    @property
    def _dsa(self):
        """The DSA resource (dsa_long_context), which plans the step's rows for the MLA layers."""
        return self.node_resources.get(DSA) if self.config.dsa_long_context else None

    @property
    def _dsa_ctx(self):
        """The step's DSA plan, or None without the resource."""
        return self._dsa.current(LABEL) if self._dsa is not None else None

    @property
    def mtp_block(self) -> int:
        return self.mtp_k + 1

    def bind_node_resources(self, resources: dict) -> None:
        if KV_CACHE in resources:
            self.config.check_long_context(resources[KV_CACHE].kv_cache.page_size)
        super().bind_node_resources(resources)
        kda = resources.get(KDA)
        self._kda = None if kda is None else KDACallable(resources[KDA_STATE], kda)

    # -- CUDA-graph configs ----------------------------------------------

    def _moe_resolved_fused(self) -> bool:
        """Whether the loaded MoE blocks resolved to the fused fp8 kernel. The reference
        loop hosts ``.nonzero()`` and cannot be captured."""
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

    def _kda_prefill_capturable(self) -> bool:
        kda = self.node_resources.get(KDA) if self.node_resources else None
        return kda is not None and kda.prefill_capturable

    def max_batch_size(self, graph_walk: str) -> int | None:
        # one KDA slot per row: the scheduler splits a larger step rather than fail its admit
        pool = self.node_resources.get(KDA_STATE) if self.node_resources else None
        if pool is None:
            return None
        # A prefill takes only the slots free now, so a TP follower that refuses one has
        # diverged from rank 0. The engine also holds a captured prefill to its largest
        # graph's rows: past them, the eager last layer would compile the MoE decode
        # kernels for its row count mid-serve.
        if graph_walk == "prefill":
            return pool.num_free_slots
        return pool.config.usable_slots

    def batch_cap_varies(self, graph_walk: str) -> bool:
        # the prefill cap follows the free KDA slots
        return graph_walk == "prefill"

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        if self._moe_capture_blocked(tp_world_size):
            logger.warning(
                "glm5_next: reference MoE dispatch active; decode runs eager "
                "(set moe_quant_kernel=auto for the capturable fused kernel)"
            )
            return []
        # The MLA fallback bakes page indices into its plan, so a captured
        # replay would attend the capture's pages (the resource refuses).
        attn = self.node_resources.get(ATTN)
        if device.type == "cuda" and attn is not None and not attn.uses_kernel:
            logger.warning(
                "glm5_next: MLA is on the SDPA fallback, which cannot be "
                "captured; decode runs eager"
            )
            return []
        max_slots = self._kda_max_slots()
        batch_sizes = [
            b for b in DEFAULT_CAPTURE_BATCH_SIZES
            if max_slots is None or b <= max_slots
        ]
        # compile=False: a Triton crash in Inductor's compile subprocess would fail every
        # capture, and serve eager silently
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
        if self.config.prefill_graphs and not self._kda_prefill_capturable():
            # The reference KDA prefill loops over spans on the host, so a
            # captured replay would run the capture's dummy spans.
            logger.warning("glm5_next: fused KDA prefill off; prefill runs eager")
        elif self.config.prefill_graphs:
            # One graph per (rows, token bucket); a batch past every token bucket
            # runs eager rather than being split.
            configs.append(PackedAcceleratorGraphConfig(
                capture_graph_walk="prefill",
                capture_token_lengths=list(PREFILL_TOKEN_BUCKETS),
                make_node_input=lambda n: ARNodeInputs(
                    input_ids=torch.zeros(n, dtype=torch.long, device=device),
                    input_seq_len=n,
                ),
                capture_batch_sizes=[
                    b for b in (self.config.prefill_capture_batch_sizes
                                or PREFILL_CAPTURE_BATCH_SIZES)
                    if max_slots is None or b <= max_slots
                ],
                compile=False,
            ))
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
        if self.mtp_k > 0:
            self._check_mtp_sampling(fwd_info)
            if graph_walk == "decode":
                # one id in, the last emitted token: the step verifies it and k drafts
                seq_len = self.mtp_block
        # Without dsa_long_context dense MLA is GLM-5.3-Flash's DSA only within
        # index_topk: refuse past the limit rather than serve off-spec logits.
        # Raising fails only this request (the engine drops it from the batch).
        limit = self.config.context_limit
        # the handle, which the batch, check_stop and cleanup_request key by
        state = self.request_state(fwd_info.rid_handle)
        committed = state.get("context", 0)
        real = committed
        if self.mtp_k > 0 and graph_walk == "decode":
            # Holes are cache rows, not context. The tokens sent so far are counted at the stop
            # check, which may still be a step behind, so count one a step (the fewest a step
            # sends): exact on the first decode step, and the stop check keeps two blocks free
            # after that. This reads only state every rank updates in step order.
            prompt = state.get("prompt_len", 0)
            real = prompt + max(committed - prompt, 0) // self.mtp_block
        if real + seq_len > limit or committed + seq_len > self.config.kv_rows:
            raise RuntimeError(
                f"request {fwd_info.request_id}: context {real + seq_len} "
                f"({committed + seq_len} cache rows) exceeds "
                f"{'max_seq_len' if self.config.dsa_long_context else 'index_topk'}={limit} "
                f"or {self.config.kv_rows} rows; without dsa_long_context the limit is "
                "index_topk, where dense MLA is exactly GLM-5.3-Flash's DSA."
            )
        if graph_walk == "prefill":
            state.add("prompt_len", committed + seq_len)
            # set by `split_inputs` when this prepare's probe finds a cached prefix
            state.remove(CACHED)
        return ARNodeInputs(input_ids=text_inputs, input_seq_len=seq_len)

    def _check_mtp_sampling(self, fwd_info: CurrentForwardPassInfo) -> None:
        """MTP verifies by argmax: refuse what it would silently ignore."""
        cfg = fwd_info.resource_configs.get(SAMPLER) if fwd_info.resource_configs else None
        if cfg is None:
            return
        # not `or`: an explicit repetition_penalty of 0 must not read as unset
        temperature = 0.0 if cfg.temperature is None else cfg.temperature
        penalty = 1.0 if cfg.repetition_penalty is None else cfg.repetition_penalty
        if temperature > 0.0 or penalty != 1.0:
            raise RuntimeError(
                f"request {fwd_info.request_id}: MTP drafting is greedy only "
                f"(temperature={cfg.temperature}, "
                f"repetition_penalty={cfg.repetition_penalty}); serve it with "
                "mtp_num_draft_tokens: 0"
            )

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
            SAMPLER: SamplerStep(
                apply_penalty=True, prefill_tracked_tokens=prefill_tokens,
            ),
            KDA_STATE: RecurrentStep(),
            KDA: LinearAttnStep(),
        }
        if self.mtp_k > 0 and graph_walk == "decode":
            # the verify and the drafts are argmaxes in the forward; the KDA layers first
            # replay what the last step accepted
            del steps[SAMPLER]
            steps[KDA] = LinearAttnStep(speculative=True)
        if self.config.dsa_long_context:
            # every MLA layer attends sparse over the DSA resource's plan, not a dense one; a
            # verify step's rows move on the device (_trim_on_device)
            del steps[ATTN]
            steps[DSA] = DSAStep(
                rows_may_move=self.config.mtp_trims and graph_walk == "decode")
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label=LABEL, span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps=steps,
            cg_key_info=self._cg_key(graph_walk, request_ids),
        )

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
            out["mtp_budget"] = self._mtp_budget(engine_inputs)
        return out

    def _mtp_keep(
        self, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs], num_tokens: int,
    ) -> torch.Tensor:
        """1 per prefill token, 0 where a request's token sits at position 0: the MTP pass
        zeroes its embedding there, as vLLM does."""
        keep, start = [1.0] * num_tokens, 0
        for rid, inp in zip(engine_inputs.request_ids, inputs, strict=True):
            state = self.request_states.get(rid)
            fresh = state is None or (state.get("context", 0) == 0 and not state.get(CACHED))
            if fresh and inp.input_seq_len > 0 and start < num_tokens:
                keep[start] = 0.0
            start += inp.input_seq_len
        # staged in fp32 (no bf16 host view), cast on the device
        return _upload(self._keep_stager, keep, self.lm_head.weight.device).to(self.lm_head.weight.dtype)

    def _mtp_budget(self, engine_inputs: ModelInputsFromEngine) -> torch.Tensor:
        """``MTP_BUDGET`` as a prefill opens it, a pair per request laid flat: the tokens its
        reply may send after the prefill's, and 1 if a stop token ends it (0 under
        ignore_eos). A padding row gets zeros."""
        flat: list[int] = []
        for rid in engine_inputs.request_ids:
            info = engine_inputs.per_request_info.get(rid)
            sampling = info.resource_configs.get(SAMPLER) if info is not None else None
            if sampling is None:
                flat += (0, 0)
            else:
                flat += (info.max_tokens - 1, int(not sampling.ignore_eos))
        return _upload(self._long_stager, flat, self.lm_head.weight.device)

    def _last_rows(self, inputs: list[ARNodeInputs]) -> torch.Tensor:
        """Each row's last token, which a captured prefill reads at a fixed
        address: the rows it samples."""
        last, end = [], 0
        for inp in inputs:
            end += inp.input_seq_len
            last.append(max(end - 1, 0))  # a padding row repeats the one before
        return _upload(self._int_stager, last, self.lm_head.weight.device)

    def _forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        last_rows: torch.Tensor | None = None,
        mtp_keep: torch.Tensor | None = None,
        mtp_budget: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn = engine_inputs.resources[ATTN]
        sampler = engine_inputs.resources[SAMPLER]
        if mtp_budget is not None:  # an MTP prefill opens the reply's budget
            engine_inputs.resources[KDA_STATE].scatter_(MTP_BUDGET, mtp_budget.view(-1, 2), None)
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
        dsa, ctx = self._dsa, self._dsa.current(LABEL)
        kv, kda = engine_inputs.resources[KV_CACHE], engine_inputs.resources[KDA]
        lm, size, total = self.language_model, self.config.prefill_window_tokens, input_ids.shape[0]
        last = [r0 + n - 1 for r0, n, _ in ctx.spans]
        # one window: the step's own plans (one-token prompts plan as decode, unsplittable)
        split = total > size and len(ctx.spans) != total
        hidden: list[torch.Tensor | None] = [None] * len(last)
        nxt = torch.roll(input_ids, -1) if self.mtp_k > 0 else None
        for start in range(0, total, size if split else total):
            end = min(start + size, total) if split else total
            ends = [(i, r - start) for i, r in enumerate(last) if start <= r < end]
            with contextlib.ExitStack() as scope:
                if split:
                    pieces = [(max(r0, start), min(r0 + n, end), req)
                              for r0, n, req in ctx.spans if r0 < end and r0 + n > start]
                    scope.enter_context(dsa.rows(start, end, LABEL))
                    scope.enter_context(kv.token_window(start, end, LABEL))
                    scope.enter_context(kda.token_window(
                        [(req, b - a) for a, b, req in pieces], LABEL))
                rows = _upload(self._long_stager, [r for _, r in ends] or [end - start - 1], input_ids.device)
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
            # each request's last MTP row again, now with its sampled token: one pass
            rows = _upload(self._long_stager, last, input_ids.device)
            with dsa.pick(last, LABEL), kv.token_rows(rows, LABEL):
                embeds = lm.model.embed_tokens(new_tokens) * mtp_keep.index_select(0, rows)[:, None]
                seeds = lm.mtp(embeds, last_hidden)
                # the seed rows' selection, which the next step's draft passes attend
                pools = dsa.current(LABEL).last_pools if self.config.mtp_shares_selection else None
            self._mtp_seed(engine_inputs, seeds, new_tokens, ctx.pos.index_select(0, rows) + 1, pools)
        return new_tokens

    # -- MTP device side ---------------------------------------------------

    def _broadcast(self, t: torch.Tensor) -> torch.Tensor:
        """Rank 0's copy: every rank must verify the same ids and keep the same tail."""
        return self.lm_head.comm_group.broadcast(t, src=0)

    def _argmax(self, hidden: torch.Tensor) -> torch.Tensor:
        """The LM head's argmax without gathering the logits: each rank's best over its
        contiguous vocab shard, then the best of those, from one gather of every rank's
        (logit, id) pair. A tie goes to the lower rank, so the lower id, as with a full
        argmax."""
        head = self.lm_head
        local = F.linear(hidden, head.weight)
        if head.tp_size == 1:
            return local.argmax(-1)
        best = _shard_best(local, head.tp_rank * head.output_size_per_partition)
        pairs = head.comm_group.all_gather(best, dim=-1).view(best.shape[0], head.tp_size, 2)
        win = pairs[..., 0].contiguous().view(torch.float32).argmax(-1, keepdim=True)
        return pairs[..., 1].gather(-1, win).squeeze(-1).long()

    def _draft_tokens(self, hidden: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        """The draft for the token after ``prev``, from the MTP state ``hidden``."""
        del prev  # the tests' oracle drafts read it
        return self._argmax(hidden)

    def _mtp_seed(
        self, engine_inputs: ModelInputsFromEngine, hidden: torch.Tensor, prev: torch.Tensor,
        ends: torch.Tensor | None = None, pools: torch.Tensor | None = None,
    ) -> None:
        """The next step's draft seed in each row's slot: the MTP state and d1, the draft for
        the token after ``prev`` (the step's next input). A prefill's ``ends`` (device: each
        request's tokens in cache) start its ``MTP_LEN`` under mtp_trims; ``pools`` are the
        seed row's selection (mtp_shares_selection)."""
        pool = engine_inputs.resources[KDA_STATE]
        pool.scatter_(MTP_HIDDEN, hidden, None)
        pool.scatter_(MTP_DRAFT, self._draft_tokens(hidden, prev)[:, None], None)
        if ends is not None and self.config.mtp_trims:
            # MTP_LEN is int64, the DSA positions ``ends`` come from int32
            pool.scatter_(MTP_LEN, ends.long()[:, None], None)
        if pools is not None:
            pool.scatter_(MTP_POOLS, pools, None)

    def _draft_selection(self, ctx, pool, base: torch.Tensor):
        """mtp_shares_selection: the draft passes attend each request's seed selection
        (``MTP_POOLS``, made at ``base - 1``), not their own (``DSAManager.reuse``)."""
        block = self.mtp_block
        pos = (base - 1).to(ctx.pos.dtype).repeat_interleave(block)
        host = (None if ctx.host_pos is None
                else [p - 1 for p in ctx.host_pos[::block] for _ in range(block)])
        return self._dsa.reuse(pos, pool.gather(MTP_POOLS, None).repeat_interleave(block, 0), host, LABEL)

    def _trim_on_device(self, engine_inputs: ModelInputsFromEngine) -> torch.Tensor:
        """mtp_trims: this verify step's rows start at each request's true rows (``MTP_LEN``),
        not where the host's plan put them; that plan, the same on every rank whatever was
        accepted, stays as made, and the DSA resource moves the rows. Spare sparse entries
        attend a row masked as the holes are. Returns each ``MTP_LEN``."""
        # a padding request reads 0 (the sink holds no state): the sink page's first rows,
        # as planned
        base = engine_inputs.resources[KDA_STATE].gather(MTP_LEN, None)[:, 0]
        pos = (base[:, None] + torch.arange(self.mtp_block, device=base.device)).view(-1)
        self._dsa.move_rows(pos, HOLE_KPE, LABEL)
        return base

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
        ctx = self._dsa_ctx
        ends = None if ctx is None else ctx.pos.index_select(0, last) + 1
        seeds = lm.mtp(embeds, hidden).index_select(0, last)
        pools = ctx.last_pools.index_select(0, last) if self.config.mtp_shares_selection else None
        self._mtp_seed(engine_inputs, seeds, new_tokens, ends, pools)

    def _mtp_decode(
        self, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor,
        base: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One verify step: draft k tokens, verify ``[last | drafts]`` with the trunk, then run
        the MTP layer over the block to seed the next step. Returns per row the next input
        token, the target's argmax at each block row, and how many of those it sends
        (``_mtp_sent``)."""
        lm, k, block = self.language_model, self.mtp_k, self.mtp_block
        pool = engine_inputs.resources[KDA_STATE]
        rows = pool.addressing()
        n = rows.num_rows

        # Drafts: d1 was seeded by the last step; each further one is an MTP pass whose
        # query sits at its row of this step's block.
        h = pool.gather(MTP_HIDDEN, None)
        drafts = [pool.gather(MTP_DRAFT, None)[:, 0]]
        attn = lm.mtp.transformer_layer.self_attn
        latents = h.new_zeros(n, block, attn.kv_lora_rank + attn.mla_cache_kpe)
        index_rows = None
        ctx = self._dsa_ctx
        drafting = contextlib.nullcontext()
        if self.config.mtp_shares_selection:
            drafting = self._draft_selection(ctx, pool, base)
        elif self.config.dsa_long_context:  # the MTP layer's own index keys and gates
            d = attn.indexer.head_dim
            index_rows = (h.new_zeros(n, block, d), h.new_zeros(n, block, d))
        # the verify and the seed pass after it run on the step's own plan
        with drafting:
            for row in range(k - 1):
                h = lm.mtp(lm.model.embed_tokens(drafts[-1]), h, row=row, latents=latents,
                           index_rows=index_rows)
                drafts.append(self._draft_tokens(h, drafts[-1]))
        drafts = self._broadcast(torch.stack(drafts, 1))

        # Verify: greedy, the longest prefix of drafts the target agrees with.
        ids = torch.cat([input_ids.view(n, 1), drafts], 1).view(-1)
        hidden = lm.model(ids)
        target = self._argmax(hidden).view(n, block)
        accepted = (drafts == target[:, :k]).long().cumprod(1).sum(1)
        verdict = self._broadcast(torch.cat([target, accepted[:, None]], 1))
        target, accepted = verdict[:, :block], verdict[:, block]
        self._kda.set_prefix_len(accepted)
        if base is not None:  # mtp_trims: the next block starts after the accepted rows
            pool.scatter_(MTP_LEN, (base + accepted + 1)[:, None], None)
        sent = self._mtp_sent(pool, target, accepted)

        # Seed the next step: the MTP pass over the block, with the target's argmax as each
        # row's next token (up to the accepted row it equals the drafts), read at the last
        # accepted row.
        each = torch.arange(n, device=accepted.device)
        next_input = target[each, accepted]
        h_block = lm.mtp(lm.model.embed_tokens(target.reshape(-1)), hidden)
        pools = None
        if self.config.mtp_shares_selection:
            pools = ctx.last_pools.view(n, block, -1)[each, accepted]
        self._mtp_seed(engine_inputs, h_block.view(n, block, -1)[each, accepted], next_input,
                       pools=pools)

        if not self.config.mtp_trims:
            keep = torch.arange(block, device=accepted.device)[None, :] <= accepted[:, None]
            self._mask_holes(engine_inputs, keep)
        return next_input, target, sent

    def _mtp_sent(self, pool, target: torch.Tensor, accepted: torch.Tensor) -> torch.Tensor:
        """How many of its block's tokens each row sends: the accepted drafts and the token
        after them, cut after its first stop token and at the budget left, which this takes
        them from (``MTP_BUDGET``). At least one: a step past the budget is one the stop check
        has ended, whose output the worker drops. It reads rank 0's verdict and state every
        rank holds alike, so every rank cuts the same."""
        budget = pool.gather(MTP_BUDGET, None)
        left, stops = budget.unbind(1)
        sent = torch.minimum(accepted + 1, left.clamp(min=1))
        stop = torch.zeros_like(target, dtype=torch.bool)
        for token in self.config.eos_token_ids:
            stop |= target == token
        stop &= stops[:, None] > 0
        block = target.shape[1]
        ends = torch.arange(1, block + 1, device=target.device)
        sent = torch.minimum(sent, torch.where(stop, ends, block).amin(1))
        left -= sent  # a view of budget
        pool.scatter_(MTP_BUDGET, budget, None)
        return sent

    def _mask_holes(self, engine_inputs: ModelInputsFromEngine, keep: torch.Tensor) -> None:
        """The block's rejected rows stay in the cache, so the next step can be planned
        before this one's verdict exists: a very negative rope slot in every MLA plane takes
        them out of every later softmax (queries carry a rope part of ones)."""
        kv = engine_inputs.resources[KV_CACHE]
        pages, offsets = kv.write_slots(LABEL)
        rope = torch.where(keep.reshape(-1, 1), 0.0, HOLE_KPE)
        width = self.config.kv_lora_rank
        for plane in range(len(self.config.full_attn_layer_indices) + 1):  # the MTP plane last
            cache = kv.layer_view(plane)
            cache[pages, offsets, width:] = rope.to(cache.dtype).expand(-1, cache.shape[-1] - width)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        last_rows: torch.Tensor | None = None,
        mtp_keep: torch.Tensor | None = None,
        mtp_budget: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList | BatchedModelOutput:
        if self.mtp_k > 0 and graph_walk == "decode":
            return self.forward_batched(graph_walk, engine_inputs, input_ids, **kwargs)
        return {"new_token": self._forward(
            graph_walk, engine_inputs, input_ids, last_rows, mtp_keep, mtp_budget)}

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        last_rows: torch.Tensor | None = None,
        mtp_keep: torch.Tensor | None = None,
        mtp_budget: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList] | BatchedModelOutput:
        if self.mtp_k > 0 and graph_walk == "decode":
            # The block's tokens at a fixed width, with the span of each row's that it sends:
            # the worker cuts them once the step has run. The next input is the token after
            # the accepted drafts.
            base = self._trim_on_device(engine_inputs) if self.config.mtp_trims else None
            next_input, target, sent = self._mtp_decode(engine_inputs, input_ids, base)
            return BatchedModelOutput(
                row_outputs={"text_inputs": next_input, "new_token": target},
                # the stop check and a host transport read the cut tokens off this one copy
                check_stop_buffers={"new_token": target},
                output_spans={"new_token": F.pad(sent[:, None], (1, 0))},  # [0, sent) per row
            )
        new_tokens = self._forward(
            graph_walk, engine_inputs, input_ids, last_rows, mtp_keep, mtp_budget)
        # row i is request i: one clone out of the graph and one host copy a step
        return BatchedModelOutput(
            row_outputs={"new_token": new_tokens},
            # "text_inputs": the ids the KV cache keys the reply's pages by (`kv_labels`)
            check_stop_buffers={"new_token": new_tokens, "text_inputs": new_tokens},
        )

    def spanned_outputs(self, graph_walk: str) -> frozenset[str]:
        # a verify step's tokens: fewer than the block's k + 1 unless every draft was accepted
        if self.mtp_k > 0 and graph_walk == "decode":
            return frozenset({"new_token"})
        return frozenset()

    # -- MTP host side ------------------------------------------------------

    def _log_acceptance(self, accepted: int, every_s: float = 10.0) -> None:
        """Rank 0 logs the accepted-draft histogram at most every ``every_s`` seconds, over
        every verify step but a reply's last, whose tokens a stop or the budget may have cut."""
        if self.lm_head.tp_rank != 0:
            return
        self._mtp_hist[accepted] += 1
        now = time.monotonic()
        if now - self._mtp_logged >= every_s:
            self._mtp_logged = now
            steps = sum(self._mtp_hist)
            mean = sum(i * c for i, c in enumerate(self._mtp_hist)) / steps
            logger.info("glm5_next MTP k=%d: %.2f drafts accepted per step (%.2f tokens), "
                        "histogram %s over %d request-steps",
                        self.mtp_k, mean, mean + 1, self._mtp_hist, steps)
            self._mtp_hist = [0] * (self.mtp_k + 1)

    # -- slow-postprocess path (worker thread, after execute_batch) -------

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        inputs: ARNodeInputs | None = None,
        **kwargs,
    ):
        # The step ran, so its tokens are the request's context now, a cached prefix's too.
        if inputs is not None:
            state = self.request_state(request_id)
            state.add("context", state.get("context", 0) + state.get(CACHED, 0) + inputs.input_seq_len)
            state.remove(CACHED)
        # the decode loop's text_inputs edge carries the emitted token; an MTP verify step set
        # it already, to the token after the accepted drafts
        if "new_token" not in outputs:
            return
        outputs.setdefault("text_inputs", outputs["new_token"])

    def check_stop(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "new_token" not in outputs:
            return set()
        if self.mtp_k > 0:
            new_token = outputs["new_token"][0]
            stops = self._mtp_check_stop(request_id, request_info, new_token)
            if request_info.graph_walk == "decode" and not stops:
                # nothing cut this step's tokens short, so all but the last were drafts
                self._log_acceptance(new_token.numel() - 1)
            return stops
        # check_stop gets host tensors, so .item() here does not sync
        return self._stops(request_id, request_info, outputs["new_token"][0].item())

    def check_stop_batched(
        self,
        request_ids: list[str],
        request_infos: dict[str, CurrentForwardPassInfo],
        host_rows: HostRows,
        step_walk: str | None = None,
    ) -> dict[str, set[str]] | None:
        """``check_stop`` for the batch off one ``tolist`` of its tokens."""
        if self.mtp_k > 0:
            return None  # a verify step's row holds several tokens
        pairs = host_rows.first_values("new_token", request_ids)
        if pairs is None:
            return None
        stops: dict[str, set[str]] = {}
        for rid, token in pairs:
            if loops := self._stops(rid, request_infos[rid], token):
                stops[rid] = loops
        return stops

    def _stops(
        self, request_id: str, request_info: CurrentForwardPassInfo, token: int,
    ) -> set[str]:
        """The loops a plain step's emitted ``token`` stops."""
        state = self.request_states.get(request_id)
        is_eos = token in self.config.eos_token_ids
        ignore_eos = request_info.resource_configs[SAMPLER].ignore_eos
        if request_info.graph_walk == "prefill":
            # no decode_loop on this walk to stop: a stop token ends the partition before any
            # decode step runs, and the conductor ends a max_tokens of 1 here
            return {END_PARTITION} if not ignore_eos and is_eos else set()
        # Total generated = 1 prefill-emitted token + (decode iters + 1):
        # max_tokens counts the token the prefill emits, so the decode loop
        # runs one fewer.
        generated = request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 2
        # The next step takes the context to prompt + generated and may already
        # be scheduled when this stop lands, so stop while it still fits, before
        # a step past the context limit (which prepare_inputs refuses) is scheduled. A
        # next step that emits the reply's last token ends the loop at its cap
        # with nothing after it, so it may fill the window.
        prompt_len = state.get("prompt_len", 0) if state is not None else 0
        full = (prompt_len + generated >= self.config.context_limit
                and generated + 1 < request_info.max_tokens)
        if (not ignore_eos and is_eos) or generated >= request_info.max_tokens or full:
            return {"decode_loop"}
        return set()

    def _mtp_check_stop(
        self, request_id: str, request_info: CurrentForwardPassInfo, new_token: torch.Tensor,
    ) -> set[str]:
        """Stop on a stop token among this step's tokens, at max_tokens, or while two more
        blocks fit (the next step may already be scheduled when this stop lands): in the
        context, which counts emitted tokens against context_limit, and in the cache, where
        every verify step keeps its whole block (holes, or rows the device rewrites) against
        kv_rows. ``mtp_checked`` counts the tokens sent up to this step, as ``MTP_BUDGET`` does
        on the device a step ahead."""
        state = self.request_state(request_id)
        before = 0 if request_info.graph_walk == "prefill" else state.get("mtp_checked", 0)
        generated = before + new_token.numel()
        state.add("mtp_checked", generated)
        ignore_eos = request_info.resource_configs[SAMPLER].ignore_eos
        is_eos = not ignore_eos and any(
            t in self.config.eos_token_ids for t in new_token.reshape(-1).tolist())
        if request_info.graph_walk == "prefill":
            # as in _stops: the prefill walk has no decode_loop to stop
            return {END_PARTITION} if is_eos else set()
        steps = request_info.dynamic_loop_iter_counts.get("decode_loop", 0)
        if request_info.graph_walk == "decode":
            steps += 1
        prompt, room = state.get("prompt_len", 0), 2 * self.mtp_block
        full = (prompt + generated + room > self.config.context_limit
                or prompt + steps * self.mtp_block + room > self.config.kv_rows)
        if is_eos or generated >= request_info.max_tokens or full:
            return {"decode_loop"}
        return set()
