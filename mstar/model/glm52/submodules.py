"""AR submodule for the GLM-5.2 text backbone."""
from __future__ import annotations

import itertools
import logging
import os
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig,
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
    distribute_tokens,
)
from mstar.engine.accelerator_graph_runner import DEFAULT_CAPTURE_BATCH_SIZES
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import (
    DRAFT_VERIFY,
    AttentionStep,
    DraftPolicy,
    DSAStep,
    KVStep,
    PositionStep,
    SamplerStep,
    Segment,
    SlotLease,
    SubmoduleStep,
    with_draft_verify,
)
from mstar.model.glm52.config import (
    ATTN_RESOURCE,
    DSA_RESOURCE,
    KV_RESOURCE,
    POS_RESOURCE,
    SAMPLER_RESOURCE,
    Glm52ModelConfig,
)
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    HostRows,
    ModelInputsFromEngine,
    NodeInputs,
)
from mstar.utils.h2d import PinnedStager

logger = logging.getLogger(__name__)

_MAIN = "main"


class _PackedPrefillConfig(PackedAcceleratorGraphConfig):
    """Packed prefill capture without the buckets whose capture rows would pass the context
    limit: ``distribute_tokens`` gives row 0 the remainder, and such a capture fails."""

    def __init__(self, *, max_row_tokens: int, **kwargs):
        super().__init__(**kwargs)
        self.max_row_tokens = max_row_tokens

    def get_total_tokens(self, bs: int) -> list[int]:
        return [n * self.total_tokens_multiplier for n in self.capture_token_lengths
                if distribute_tokens(n, bs)[0] <= self.max_row_tokens]


class Glm52LLMSubmodule(ARNodeSubmodule):
    """Embed + 78 decoder layers + lm_head, one fat TP node."""

    def __init__(self, language_model: nn.Module, config: Glm52ModelConfig):
        super().__init__()
        self.language_model = language_model  # Glm52ForCausalLM
        self.lm_head = language_model.lm_head
        self.config = config
        self._capture_block_warned = False
        # MTP: a decode step drafts k tokens, verifies them with the last emitted token and
        # seeds the next step, all in one captured graph; the eager fallback stays uncompiled.
        # The engine verifies, rewinds, emits and stops (get_draft_policy).
        self._mtp_uncompiled = config.mtp_num_draft_tokens > 0
        self._draft_policy = (
            DraftPolicy(max_drafts=config.mtp_num_draft_tokens, stop_ids=frozenset(config.eos_token_ids))
            if config.mtp_num_draft_tokens > 0 else None
        )
        # the step's per-step indices, in one pinned H2D
        self._index_stager = PinnedStager(torch.long)
        # Per request, in its request_state: "ended_at_prefill", the prefill sampled a stop
        # token: the prefill walk has no loop to stop, so the first decode step stops
        # decode_loop and emits nothing.
        # Which trunk stream the MTP plane pairs drafts against (see
        # _mtp_pair_rows): the post-final-norm one by default.
        self._mtp_pair_postnorm = (
            os.environ.get("MSTAR_GLM52_MTP_PAIR_POSTNORM", "1") == "1"
        )
        if self._draft_policy is not None:
            # the engine logs the acceptance; which pairing produced it is the model's to say
            logger.info(
                "GLM-5.2 MTP: %d drafts a step, paired against the %s trunk stream",
                config.mtp_num_draft_tokens,
                "post-final-norm" if self._mtp_pair_postnorm
                else "pre-final-norm (MSTAR_GLM52_MTP_PAIR_POSTNORM=0)",
            )
        # Each request's seed slot (the MTP state that drafts d1, and d1) in the device seed
        # buffers; slot 0 is the sink that padding and capture rows write.
        self._mtp_slot: dict[str, int] = {}
        self._mtp_free_slots: list[int] = []

    def get_draft_policy(self, graph_walk: str) -> DraftPolicy | None:
        return self._draft_policy if graph_walk == "decode" else None

    def cleanup_request(self, request_id: str):
        slot = self._mtp_slot.pop(request_id, None)
        if slot is not None:
            self._mtp_free_slots.append(slot)
        super().cleanup_request(request_id)

    PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]
    # an MTP row costs k+1 tokens, so the decode capture pads less coarsely
    MTP_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 24, 32, 48, 64]

    @property
    def mtp_k(self) -> int:
        return self.config.mtp_num_draft_tokens

    @property
    def mtp_block(self) -> int:
        return self.mtp_k + 1

    # ── resources ──

    def _kv(self, engine_inputs: ModelInputsFromEngine | None = None):
        res = engine_inputs.resources if engine_inputs is not None else self.node_resources
        return res[KV_RESOURCE]

    def _pos(self, engine_inputs: ModelInputsFromEngine | None = None):
        res = engine_inputs.resources if engine_inputs is not None else self.node_resources
        return res[POS_RESOURCE]

    def _attn(self, engine_inputs: ModelInputsFromEngine | None = None):
        res = engine_inputs.resources if engine_inputs is not None else self.node_resources
        return res[ATTN_RESOURCE]

    def bind_node_resources(self, resources: dict[str, Any]) -> None:
        super().bind_node_resources(resources)
        if self.mtp_k > 0:
            # a request holds at least one page, so the pool bounds the requests in flight
            slots = resources[KV_RESOURCE].config.max_num_pages + 1
            weight = self.lm_head.weight
            self._mtp_seed_hidden = torch.zeros(
                slots, self.config.hidden_size, dtype=weight.dtype, device=weight.device)
            self._mtp_seed_draft = torch.zeros(slots, dtype=torch.long, device=weight.device)
            self._mtp_free_slots = list(range(slots - 1, 0, -1))

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases=None,
        **kwargs,
    ) -> SubmoduleStep | None:
        prefill_tokens = {}
        if graph_walk == "prefill":
            prefill_tokens = {
                rid: inp.input_ids for rid, inp in zip(request_ids, inputs, strict=True)
            }
        steps = {KV_RESOURCE: KVStep(), ATTN_RESOURCE: AttentionStep(causal=True),
                 POS_RESOURCE: PositionStep()}
        policy = self.get_draft_policy(graph_walk)
        if policy is None:
            steps[SAMPLER_RESOURCE] = SamplerStep(
                apply_penalty=True, prefill_tracked_tokens=prefill_tokens)
        if self.config.dsa_long_context:
            steps[DSA_RESOURCE] = DSAStep()
        step = SubmoduleStep(
            segments=[
                Segment(rid, _MAIN, inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps=steps,
        )
        # the verify is the sampler's, through the draft-verify resource
        return step if policy is None else with_draft_verify(step, policy, DRAFT_VERIFY)

    def max_step_tokens(self, graph_walk: str) -> int | None:
        cap = self.config.prefill_max_step_tokens
        if graph_walk != "prefill" or cap is None:
            return None
        if cap == "auto":
            return max(self.config.prefill_batched_token_buckets
                       or self.config.prefill_token_buckets or self.PREFILL_TOKEN_BUCKETS)
        return cap

    # ── capture configs ──

    def _moe_resolved_fused(self) -> bool:
        """True iff the loaded MoE blocks resolved to the fused fp8 kernel."""
        from mstar.model.glm52.components.moe import Glm52SparseMoeBlock

        lm = getattr(self, "language_model", None)
        if lm is None:
            return False
        for module in lm.modules():
            if isinstance(module, Glm52SparseMoeBlock):
                return bool(getattr(module, "_use_fused", False))
        return False

    def _moe_capture_blocked(self, tp_world_size: int) -> bool:
        """The reference MoE dispatch paths (.nonzero() / host loops) are not
        stream-capturable; only the fused fp8 path is.
        """
        fp8_reference = (
            self.config.quantization_config is not None
            and self.config.moe_fp8_resident
            and not self._moe_resolved_fused()
        )
        naive_tp = self.config.quantization_config is None and tp_world_size > 1
        blocked = fp8_reference or naive_tp
        # getattr: some tests build submodules that never ran __init__
        if blocked and not getattr(self, "_capture_block_warned", False):
            self._capture_block_warned = True
            logger.warning(
                "Glm52LLMSubmodule: %s; no CUDA graphs are registered and "
                "decode runs eager, which is far slower. %s",
                "the reference MoE dispatch is active" if fp8_reference
                else "naive TP MoE dispatch is active",
                "moe_quant_kernel 'auto' or 'triton' serves the capturable "
                "fused fp8 kernel." if fp8_reference
                else "Serve the fp8 checkpoint to get the fused kernel.",
            )
        return blocked

    def _compile_flags(self) -> dict[str, Any]:
        # "default" is torch.compile's own mode (cuBLAS GEMMs, no autotuning), the fastest here
        return {"compile": self.config.graph_compile, "compile_mode": "default"}

    @property
    def disable_torch_compile(self) -> bool:
        # graph_compile covers the uncaptured steps too, which the engine compiles
        return getattr(self, "_mtp_uncompiled", False) or not self.config.graph_compile

    def to(self, *args, **kwargs):
        """Honor device moves; refuse post-load dtype casts (per-param dtypes
        are fixed at load — fp32 block scales + router bias, uint8 fp8)."""
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is not None:
            logger.info(
                "Glm52LLMSubmodule: ignoring post-load dtype cast to %s "
                "(per-param dtypes are fixed at load; see restore_fp32_params).",
                dtype,
            )
        if device is not None:
            self.__dict__.pop("_device", None)
            return super().to(device=device, non_blocking=non_blocking)
        return self

    def get_device(self) -> torch.device:
        # cached: the base walks parameters() on every call
        if "_device" not in self.__dict__:
            self._device = super().get_device()
        return self._device

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        if self.config.dsa_long_context:
            # the DSA resource selects per request on the host; a captured
            # decode would skip index upkeep. Eager-only.
            return []
        if self._moe_capture_blocked(tp_world_size):
            return []
        prefill_buckets = self.config.prefill_token_buckets or self.PREFILL_TOKEN_BUCKETS
        prefill_batch_sizes = (
            self.config.prefill_capture_batch_sizes or self.PREFILL_CAPTURE_BATCH_SIZES
        )
        flags = self._compile_flags()
        groups = [(prefill_buckets, prefill_batch_sizes)]
        batched = self.config.prefill_batched_token_buckets
        if batched:
            # one request's buckets and a batch's apart, so each can be fine or coarse
            groups = [(prefill_buckets, [bs for bs in prefill_batch_sizes if bs == 1]),
                      (batched, [bs for bs in prefill_batch_sizes if bs > 1])]
        configs = [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="decode",
                # one id in; under MTP the step verifies it and k drafts
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device),
                    input_seq_len=self.mtp_block,
                ),
                capture_batch_sizes=self.MTP_CAPTURE_BATCH_SIZES if self.mtp_k else None,
                **flags,
            ),
            *(
                _PackedPrefillConfig(
                    max_row_tokens=self._context_limit(),
                    capture_graph_walk="prefill",
                    capture_token_lengths=list(buckets),
                    make_node_input=lambda n: ARNodeInputs(
                        input_ids=torch.zeros(n, dtype=torch.long, device=device),
                        input_seq_len=n,
                    ),
                    capture_batch_sizes=list(sizes),
                    **flags,
                )
                for buckets, sizes in groups if sizes
            ),
        ]
        self._fit_recompile_limit(configs)
        return configs

    def _fit_recompile_limit(self, configs: list[AcceleratorGraphConfig]) -> None:
        """Raise dynamo's recompile limit to what these captures take: a decoder layer's
        frames hold an entry per captured token count and layer kind (dense, MoE, the last
        one on sampled rows), and the shapes past the limit capture uncompiled."""
        tokens = {
            n for config in configs
            for bs in config.capture_batch_sizes or DEFAULT_CAPTURE_BATCH_SIZES
            for n in config.get_total_tokens(bs)
        }
        need = 3 * len(tokens) + 16  # and the engine's compiled fallback
        if torch._dynamo.config.recompile_limit < need:
            logger.info("Glm52LLMSubmodule: dynamo recompile_limit %d -> %d for %d "
                        "captured token counts", torch._dynamo.config.recompile_limit, need,
                        len(tokens))
        self._recompile_limit = need
        self._apply_recompile_limit()

    def _apply_recompile_limit(self) -> None:
        # dynamo's config is per thread, and the engine's step thread starts at its default
        need = getattr(self, "_recompile_limit", 0)
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, need)

    # ── per-step contract ──

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> ARNodeInputs:
        text_inputs = inputs["text_inputs"][0]
        # here, not in preprocess, so the refusal fails this request and not its batch
        # (process_prompt refuses the same prompts first, as a 400)
        if graph_walk == "prefill" and text_inputs.shape[0] > self.config.max_prompt_tokens:
            raise RuntimeError(
                f"request {fwd_info.request_id}: a prompt of {text_inputs.shape[0]} tokens "
                f"exceeds the {self.config.max_prompt_tokens} served"
            )
        if self.config.mtp_num_draft_tokens > 0:
            # the handle, which the batch and cleanup_request key by
            rid = fwd_info.rid_handle
            if graph_walk == "decode":
                # one id in, the last emitted token: the step verifies it and k drafts
                return ARNodeInputs(input_ids=text_inputs, input_seq_len=self.mtp_block)
            if rid not in self._mtp_slot:
                self._mtp_slot[rid] = self._mtp_free_slots.pop()
        return ARNodeInputs(
            input_ids=text_inputs,
            input_seq_len=text_inputs.shape[0],
        )

    def _context_limit(self) -> int:
        """Rows a request may hold: the serving window on the DSA engine
        path, else index_topk, where dense MLA is exactly GLM-5.2's DSA."""
        return self.config.max_seq_len if self.config.dsa_long_context else self.config.index_topk

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        self._apply_recompile_limit()
        pos = self._pos(engine_inputs)
        request_ids = list(engine_inputs.request_ids)
        seq_lens = [inp.input_seq_len for inp in inputs]
        device = self.get_device()
        # Dense MLA equals the reference DSA computation only while every
        # context fits the top-k window; refuse beyond it unless the DSA
        # engine path is on, where the cap is the serving window. The KV pool
        # is sized to this limit, so the conductor holds every request inside
        # it, MTP's verify rows too (the layout's lookahead), and this only
        # fires on a bug. The positions themselves are the position resource's
        # (``_positions``).
        long_context = self.config.dsa_long_context
        limit = self._context_limit()
        topk = self.config.index_topk
        for rid, sl in zip(request_ids, seq_lens, strict=True):
            start = pos.position(rid, _MAIN)  # where the step's first token lands
            if start + sl > limit:
                raise RuntimeError(
                    f"request {rid}: context {start + sl} exceeds {limit}, "
                    + (
                        "the configured max_seq_len serving window."
                        if long_context
                        else "the regime where dense MLA is exactly GLM-5.2's "
                        "DSA computation. Long context needs dsa_long_context=True."
                    )
                )
            if long_context and start + sl > topk and sl > 1:
                raise RuntimeError(
                    f"request {rid}: prefill context {start + sl} exceeds "
                    f"index_topk={topk}; sparse attention beyond topk is "
                    "decode-only."
                )
        # Every per-step index in one pinned H2D (a pageable torch.tensor(..., device=cuda)
        # drains the stream before the step even starts): on prefill each request's last row
        # (for an eager prefill, when no plan buffer carries qo_indptr); under MTP the seed
        # slots (padding and capture rows read and write the sink).
        fields = {}
        if graph_walk == "prefill":
            fields["last_token_indices"] = [end - 1 for end in itertools.accumulate(seq_lens)]
        if self.mtp_k > 0:
            fields["mtp_slots"] = [self._mtp_slot.get(rid, 0) for rid in request_ids]
        out = {}
        if fields:
            staged = self._index_stager.to_device(
                [v for values in fields.values() for v in values], device)
            out = dict(zip(fields, staged.split([len(v) for v in fields.values()]), strict=True))
        out["input_ids"] = (
            inputs[0].input_ids if len(inputs) == 1 else torch.cat([inp.input_ids for inp in inputs]))
        return out

    def _mtp_pair_rows(self, normed: torch.Tensor, prenorm: torch.Tensor) -> torch.Tensor:
        """The trunk stream the MTP plane pairs drafts against: the
        post-final-norm one, which the reference implementation pairs and
        which accepts markedly better, unless
        MSTAR_GLM52_MTP_PAIR_POSTNORM=0."""
        return normed if self._mtp_pair_postnorm else prenorm

    def _positions(self, engine_inputs: ModelInputsFromEngine, num_tokens: int) -> torch.Tensor:
        """The step's positions as the position resource planned them, in the KV plan's
        order, which ``input_ids`` packs in; captured, a slice of the slot's buffer."""
        return self._pos(engine_inputs).pos_ids(_MAIN)[:num_tokens]

    def _hidden(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        rows: torch.Tensor | None = None,
        with_prenorm: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.language_model.model(
            input_ids, position_ids, rows=rows, return_prenorm=with_prenorm)

    @torch.compiler.disable
    def _last_row_index(
        self, engine_inputs: ModelInputsFromEngine, kwargs: dict,
    ) -> torch.Tensor:
        """Each request's last row of a packed prefill: from the plan's
        qo_indptr buffer where the attention resource has one (captured, or
        the MLA resource), else from preprocess's real lengths."""
        qo_indptr = self._attn(engine_inputs).qo_indptr_buf(_MAIN)
        if qo_indptr is not None:
            # the buffer is sized to the capture bucket and tail-filled with
            # the last real offset; slice to the real request count so a
            # padded prefill does not duplicate the final request's row
            n = len(engine_inputs.request_ids)
            return (qo_indptr[1 : n + 1] - 1).long()
        last = kwargs.get("last_token_indices")
        assert last is not None, "eager prefill needs last_token_indices from preprocess"
        return last

    # Outside dynamo: traced, the frame after the qo_indptr read is keyed by the tokens and
    # the rows together, one entry per capture bucket, and past the recompile limit it
    # runs uncompiled.
    @torch.compiler.disable
    def _last_rows(
        self, engine_inputs: ModelInputsFromEngine, hidden: torch.Tensor, kwargs: dict,
    ) -> torch.Tensor:
        return hidden.index_select(0, self._last_row_index(engine_inputs, kwargs))

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        hidden = self._hidden(input_ids, self._positions(engine_inputs, input_ids.shape[0]))
        return {"logits": [self.lm_head(hidden[-1:])]}

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        if self.mtp_k > 0 and graph_walk == "decode":
            # a verify step plans a block of rows per request around its one input token
            position_ids = self._positions(engine_inputs, input_ids.shape[0] * self.mtp_block)
            return self._mtp_decode(engine_inputs, input_ids, position_ids, kwargs["mtp_slots"])
        position_ids = self._positions(engine_inputs, input_ids.shape[0])
        if self.mtp_k > 0 and graph_walk == "prefill":
            return self._mtp_prefill(engine_inputs, input_ids, position_ids, kwargs)
        if graph_walk not in ("prefill", "decode"):
            raise ValueError(f"Batched forward not supported for graph walk: {graph_walk!r}")
        rows = None
        if graph_walk == "prefill" and self.config.prefill_last_layer_rows:
            rows = self._last_row_index(engine_inputs, kwargs)
        hidden = self._hidden(input_ids, position_ids, rows=rows)
        if graph_walk == "prefill" and rows is None:
            hidden = self._last_rows(engine_inputs, hidden, kwargs)
        logits = self.lm_head(hidden)  # (bs, vocab)
        return self._sample(engine_inputs, logits)

    # Keyed by request id, so outside dynamo: traced, the engine's compiled fallback guards
    # on the ids and recompiles this tail for every new request (the sampler runs eager
    # either way).
    @torch.compiler.disable
    def _sample(
        self, engine_inputs: ModelInputsFromEngine, logits: torch.Tensor,
    ) -> BatchedModelOutput:
        new_tokens = engine_inputs.resources[SAMPLER_RESOURCE].sample(
            list(engine_inputs.request_ids), logits=logits)
        # Row i is request i: one clone and one D2H copy per step, not one per request
        return BatchedModelOutput(
            row_outputs={"new_token": new_tokens},
            check_stop_buffers={"new_token": new_tokens},
        )

    # ── MTP: one captured graph per step ──
    #
    # A decode step's block is k+1 rows per request at positions L..L+k: the last emitted token
    # and k drafts. The MTP plane's row p pairs (embed(token p+1), hidden p) at RoPE position
    # p+1, so it shares the trunk's page table and write slots.

    def _mtp_prefill(
        self, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor,
        position_ids: torch.Tensor, kwargs: dict,
    ) -> dict[str, NameToTensorList]:
        """The trunk over the prompt and its sampled token, then the MTP pass over every prompt
        row (each request's last row pairs with its sampled token): it fills the MTP plane and
        seeds the first decode step."""
        lm = self.language_model
        hidden, prenorm = self._hidden(input_ids, position_ids, with_prenorm=True)
        last = self._last_row_index(engine_inputs, kwargs)
        new_tokens = self._sample_tokens(engine_inputs, self.lm_head(hidden.index_select(0, last)))
        # roll, not a shifted copy into empty memory: a captured bucket's tail is padding,
        # and every row must hold a valid id for the embedding
        nxt = torch.roll(input_ids, -1)
        nxt.index_copy_(0, last, new_tokens.to(nxt.dtype))
        h_head, h_raw = lm.mtp(
            lm.model.embed_tokens(nxt), self._mtp_pair_rows(hidden, prenorm), position_ids + 1)
        self._mtp_seed(kwargs["mtp_slots"], h_head.index_select(0, last),
                       h_raw.index_select(0, last), new_tokens)
        return {
            rid: {"new_token": [new_tokens[i : i + 1]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    def _mtp_decode(
        self, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor,
        position_ids: torch.Tensor, mtp_slots: torch.Tensor,
    ) -> dict[str, NameToTensorList]:
        """One verify step. Drafts: d1 from the seed, each later one an MTP pass whose query
        sits at its row of this step's block. Verify: the trunk over [last | drafts], its
        logits to the draft-verify resource, which keeps rank 0's drafts and verdict on every
        rank and stages the verdict for the host. Seed: the MTP pass over the block with the
        target's token as each row's next (up to the accepted row it equals the drafts), read
        at the last accepted row."""
        lm, k, block = self.language_model, self.mtp_k, self.mtp_block
        embed, n = lm.model.embed_tokens, input_ids.shape[0]
        dv = engine_inputs.resources[DRAFT_VERIFY]
        positions = position_ids.view(n, block)
        h, d1 = self._mtp_read_seed(mtp_slots)
        drafts = [d1]
        rows: dict[str, torch.Tensor] = {}
        for row in range(k - 1):
            h_head, h = lm.mtp.forward_block_row(
                embed(drafts[-1]), h, positions[:, row] + 1, row, block, rows)
            drafts.append(self._draft_tokens(h_head, drafts[-1]))
        ids = dv.block_ids(input_ids, dv.agree(torch.stack(drafts, 1)))
        hidden, prenorm = self._hidden(ids.view(-1), position_ids, with_prenorm=True)
        verdict = dv.verify(list(engine_inputs.request_ids), ids, self.lm_head(hidden))
        h_head, h_raw = lm.mtp(
            embed(verdict.tokens.reshape(-1)), self._mtp_pair_rows(hidden, prenorm), position_ids + 1)
        at = torch.arange(n, device=verdict.accepted.device) * block + verdict.accepted
        next_input = verdict.next_inputs.view(-1)
        self._mtp_seed(mtp_slots, h_head.index_select(0, at), h_raw.index_select(0, at), next_input)
        return {
            rid: {"text_inputs": [next_input[i : i + 1]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    # The seed buffers live outside dynamo: they are the submodule's own state, written at
    # fixed addresses that a captured replay keeps.
    @torch.compiler.disable
    def _mtp_read_seed(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._mtp_seed_hidden.index_select(0, slots), self._mtp_seed_draft.index_select(0, slots)

    @torch.compiler.disable
    def _mtp_seed(
        self, slots: torch.Tensor, h_head: torch.Tensor, h_raw: torch.Tensor, prev: torch.Tensor,
    ) -> None:
        """The next step's draft seed in each row's slot: the MTP state and d1, the draft for
        the token after ``prev``. The head reads the shared_head-normed rows; the chain threads
        the raw layer output."""
        self._mtp_seed_hidden.index_copy_(0, slots, h_raw.to(self._mtp_seed_hidden.dtype))
        self._mtp_seed_draft.index_copy_(0, slots, self._draft_tokens(h_head, prev))

    def _draft_tokens(self, h_head: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        """The draft for the token after ``prev``, from the MTP head input ``h_head``."""
        del prev  # the tests' oracle drafts read it
        return self.lm_head(h_head).argmax(-1)

    # Keyed by request id, so outside dynamo (as _sample).
    @torch.compiler.disable
    def _sample_tokens(
        self, engine_inputs: ModelInputsFromEngine, logits: torch.Tensor,
    ) -> torch.Tensor:
        return engine_inputs.resources[SAMPLER_RESOURCE].sample(
            list(engine_inputs.request_ids), logits=logits)

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ):
        if "new_token" not in outputs:
            return
        # an MTP verify step's forward returned the next input (the token after the
        # accepted drafts); new_token is the engine's cut of the verdict
        outputs.setdefault("text_inputs", outputs["new_token"])
        if self.request_state(request_id).get("ended_at_prefill"):
            # the reply already ended: this step only closes the decode loop
            del outputs["new_token"]

    def check_stop(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        tokens = outputs["new_token"][0].reshape(-1).tolist() if "new_token" in outputs else None
        return self._stops(request_id, request_info, tokens)

    def check_stop_batched(
        self,
        request_ids: list[str],
        request_infos: dict[str, CurrentForwardPassInfo],
        host_rows: HostRows,
        step_walk: str | None = None,
    ) -> dict[str, set[str]] | None:
        """``check_stop`` for every request off one ``tolist`` of the step's tokens. A verify
        step's stop is the engine's (``get_draft_policy``) and never reaches here."""
        tokens = host_rows.buffers.get("new_token")
        if not torch.is_tensor(tokens) or tokens.dim() == 0:
            return None
        values = tokens.reshape(tokens.shape[0], -1)[:, 0].tolist()
        row_of = {rid: i for i, rid in enumerate(host_rows.request_ids)}
        stops: dict[str, set[str]] = {}
        for rid in request_ids:
            i = row_of.get(rid)
            if i is None or i >= len(values):
                continue  # no row, no output: the per-request path skips it too
            loops = self._stops(rid, request_infos[rid], [values[i]])
            if loops:
                stops[rid] = loops
        return stops

    def _stops(
        self, request_id: str, request_info: CurrentForwardPassInfo, tokens: list[int] | None,
    ) -> set[str]:
        """The loops the step's new ``tokens`` stop; None when the step emitted none."""
        state = self.request_state(request_id)
        if state.get("ended_at_prefill"):
            return {"decode_loop"}
        if tokens is None:
            return set()
        ignore_eos = request_info.resource_configs[SAMPLER_RESOURCE].ignore_eos
        budget = request_info.max_tokens
        is_eos = tokens[-1] in self.config.eos_token_ids
        # max_tokens counts the token the prefill emits, so the total is
        # 1 + (iters + 1) decode tokens
        generated = request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 2
        if request_info.graph_walk == "prefill":
            # no decode_loop on this walk to stop: a stop token ends the reply at the first
            # decode step, and the conductor ends a max_tokens of 1 here
            if not ignore_eos and is_eos:
                state.add("ended_at_prefill", True)
            return set()
        if (not ignore_eos and is_eos) or generated >= budget:
            return {"decode_loop"}
        return set()
