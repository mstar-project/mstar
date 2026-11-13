"""AR submodule for the GLM-5.2 text backbone."""
from __future__ import annotations

import logging
import time
from collections.abc import Mapping
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
from mstar.engine.engine import ExecutingBatch
from mstar.engine.state import (
    ArrayStep,
    SampleStep,
    Segment,
    SlotLease,
    SubmoduleStep,
    at_last_token,
    position,
    positions,
    sample,
    settle,
)
from mstar.model.glm52.config import INDEX_KEYS, KV, MTP_SEED, PROMPT_LEN, SAMPLER, Glm52ModelConfig
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    HostRows,
    ModelInputsFromEngine,
    NodeInputs,
    TokenStop,
)

logger = logging.getLogger(__name__)

_MAIN = "main"
# How long a read waits for a verify step's verdict before the process exits
_VERDICT_DEADLINE_S = 300.0
# the MTP seed's one layer
_SEED = 0


class _PackedPrefillConfig(PackedAcceleratorGraphConfig):
    """Packed prefill capture without the buckets whose capture rows would pass the context
    limit: ``distribute_tokens`` gives row 0 the remainder, and such a capture fails."""

    def __init__(self, *, max_row_tokens: int, **kwargs):
        super().__init__(**kwargs)
        self.max_row_tokens = max_row_tokens

    def get_total_tokens(self, bs: int) -> list[int]:
        return [n for n in super().get_total_tokens(bs)
                if distribute_tokens(n // self.total_tokens_multiplier, bs)[0] <= self.max_row_tokens]


class Glm52LLMSubmodule(ARNodeSubmodule):
    """Embed + 78 decoder layers + lm_head, one fat TP node."""

    def __init__(self, language_model: nn.Module, config: Glm52ModelConfig):
        super().__init__()
        self.language_model = language_model  # Glm52ForCausalLM
        self.lm_head = language_model.lm_head
        self.config = config
        self._capture_block_warned = False
        self.token_stop = TokenStop(
            loop="decode_loop", buffer="new_token", stop_ids=frozenset(config.eos_token_ids),
            sampler=SAMPLER, prefill_walks=frozenset({"prefill"}),
        )
        # Each captured bucket's mailbox, and the eager steps' under None: the pinned verdict table the host reads
        # and the flag the graph sets once the table is written. `_verdicts` names the mailbox of the step in
        # flight and each real request's row in it. It holds one step, since the engine unpacks a step's outputs
        # before it preprocesses the next step.
        self._mailboxes: dict[Any, tuple[torch.Tensor, torch.Tensor]] = {}
        self._verdicts: tuple[Any, dict] | None = None
        self._one: torch.Tensor | None = None
        # acceptance stats for the periodic log line, before the max_tokens and stop cuts
        self._mtp_stat_emitted = 0
        self._mtp_stat_steps = 0
        self._mtp_stat_logged = 0
        self._mtp_stat_acc_hist = [0] * (config.mtp_num_draft_tokens + 1)

    PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]
    # a decoder layer's frames are traced once per layer kind: dense, MoE, and the last one
    # on the sampled rows
    compiled_entries_per_shape = 3
    # an MTP row costs k+1 tokens, so the decode capture pads less coarsely
    MTP_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 24, 32, 48, 64]

    @property
    def mtp_k(self) -> int:
        return self.config.mtp_num_draft_tokens

    @property
    def mtp_block(self) -> int:
        return self.mtp_k + 1

    # ── state ──

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.kv = handles[KV]
        self.sampler = handles[SAMPLER]
        self.seed = handles.get(MTP_SEED)

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases=None,
        **kwargs,
    ) -> SubmoduleStep | None:
        # The verify and the drafts are argmaxes in the forward: no sampler.
        verify = self.mtp_k > 0 and graph_walk == "decode"
        state = {KV: ArrayStep(verify=verify)}
        if self.config.dsa_long_context:
            state[INDEX_KEYS] = ArrayStep()
        if self.mtp_k > 0:
            state[MTP_SEED] = ArrayStep()
        if not verify:
            prompts = (
                {rid: inp.input_ids for rid, inp in zip(request_ids, inputs, strict=True)}
                if graph_walk == "prefill" else {}
            )
            state[SAMPLER] = SampleStep(apply_penalty=True, prompt_tokens=prompts)
        return SubmoduleStep(
            segments=[
                Segment(rid, _MAIN, inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            state=state,
        )

    def step_is_input_free(self, graph_walk: str) -> bool:
        # every decode row spans one block, as the capture's template row does, and declares nothing else
        return graph_walk == "decode"

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
                "kernels: auto serves the capturable fused fp8 kernel where it "
                "runs." if fp8_reference
                else "Serve the fp8 checkpoint to get the fused kernel.",
            )
        return blocked

    def _compile_flags(self) -> dict[str, Any]:
        # "default" is torch.compile's own mode (cuBLAS GEMMs, no autotuning), the fastest here
        return {"compile": self.config.graph_compile, "compile_mode": "default"}

    @property
    def disable_torch_compile(self) -> bool:
        # graph_compile covers the uncaptured steps too, which the engine compiles. An MTP
        # decode step is one captured graph, and its uncaptured fallback runs eager.
        return self.config.mtp_num_draft_tokens > 0 or not self.config.graph_compile

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
        if self._moe_capture_blocked(tp_world_size):
            return []
        prefill_batch_sizes = (
            self.config.prefill_capture_batch_sizes or self.PREFILL_CAPTURE_BATCH_SIZES
        )
        flags = self._compile_flags()
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
            _PackedPrefillConfig(
                max_row_tokens=self._context_limit(),
                capture_graph_walk="prefill",
                capture_token_lengths=list(self.PREFILL_TOKEN_BUCKETS),
                make_node_input=lambda n: ARNodeInputs(
                    input_ids=torch.zeros(n, dtype=torch.long, device=device),
                    input_seq_len=n,
                ),
                capture_batch_sizes=list(prefill_batch_sizes),
                **flags,
            ),
        ]
        return configs

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
            sampling = fwd_info.resource_configs[SAMPLER]
            # greedy only: the drafts and the verify bypass the sampler, so a temperature
            # would be ignored, and a repetition penalty would move even the prefill's argmax
            if sampling.temperature != 0 or sampling.repetition_penalty != 1:
                raise RuntimeError(
                    f"request {fwd_info.request_id}: MTP speculative decoding is greedy-only "
                    f"but the request asks for temperature={sampling.temperature}, "
                    f"repetition_penalty={sampling.repetition_penalty}. Send "
                    "temperature=0 without a penalty, or serve a k=0 config."
                )
            if graph_walk == "decode":
                # one id in, the last emitted token: the step verifies it and k drafts
                return ARNodeInputs(
                    input_ids=text_inputs, input_seq_len=self.mtp_block, kwargs={"count": int(inputs["count"][0])},
                )
        return ARNodeInputs(
            input_ids=text_inputs,
            input_seq_len=text_inputs.shape[0],
        )

    def _context_limit(self) -> int:
        """Rows a request may hold: the serving window on the DSA engine
        path, else index_topk, where dense MLA is exactly GLM-5.2's DSA."""
        return self.config.max_seq_len if self.config.dsa_long_context else self.config.index_topk

    def _budget(self, info: CurrentForwardPassInfo) -> int:
        """The most tokens a request emits under MTP: its max_tokens, held to what fits before the context limit with
        two verify blocks to spare, since every verify step writes a whole block and the next may already be
        running when the stop lands."""
        room = self._context_limit() - info.step_metadata[PROMPT_LEN] - 2 * self.mtp_block
        return min(info.max_tokens, room)

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        request_ids = list(engine_inputs.request_ids)
        seq_lens = [inp.input_seq_len for inp in inputs]
        # Dense MLA equals the reference DSA computation only while every
        # context fits the top-k window; refuse beyond it unless the DSA
        # engine path is on, where the cap is the serving window. The KV pool
        # is sized to this limit, so the conductor holds every request inside
        # it and this only fires on a bug.
        long_context = self.config.dsa_long_context
        limit = self._context_limit()
        topk = self.config.index_topk
        for rid, sl in zip(request_ids, seq_lens, strict=True):
            start = position(self.kv, rid, _MAIN)  # where the step's first token lands
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
        if self.mtp_k > 0 and graph_walk == "decode":
            self._open_mailbox(engine_inputs)
        return {
            "input_ids": inputs[0].input_ids if len(inputs) == 1 else torch.cat([inp.input_ids for inp in inputs]),
        }

    def _hidden(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.language_model.model(input_ids, position_ids, rows=rows)

    def _last_row_index(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        """Each request's last row of a packed prefill."""
        return at_last_token(self.kv, torch.arange(num_tokens, device=device))

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        hidden = self._hidden(input_ids, positions(self.kv)[: input_ids.shape[0]])
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
            position_ids = positions(self.kv)[: input_ids.shape[0] * self.mtp_block]
            return self._mtp_decode(engine_inputs, input_ids, position_ids)
        # captured, a slice of the slot's buffer
        position_ids = positions(self.kv)[: input_ids.shape[0]]
        if self.mtp_k > 0 and graph_walk == "prefill":
            return self._mtp_prefill(engine_inputs, input_ids, position_ids)
        if graph_walk not in ("prefill", "decode"):
            raise ValueError(f"Batched forward not supported for graph walk: {graph_walk!r}")
        rows = None
        if graph_walk == "prefill" and self.config.tuned_kernels:
            # the last layer's FFN on the rows it samples only: they take the small-batch MoE
            # path, so the first token's numerics move (the reference runs every row)
            rows = self._last_row_index(input_ids.shape[0], input_ids.device)
        hidden = self._hidden(input_ids, position_ids, rows=rows)
        if graph_walk == "prefill" and rows is None:
            hidden = at_last_token(self.kv, hidden)
        logits = self.lm_head(hidden)  # (bs, vocab)
        new_tokens = sample(self.sampler, logits)
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
        self, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor, position_ids: torch.Tensor,
    ) -> dict[str, NameToTensorList]:
        """The trunk over the prompt and its sampled token, then the MTP pass over every prompt
        row (each request's last row pairs with its sampled token): it fills the MTP plane and
        seeds the first decode step."""
        lm = self.language_model
        hidden = self._hidden(input_ids, position_ids)
        last = self._last_row_index(input_ids.shape[0], input_ids.device)
        new_tokens = sample(self.sampler, self.lm_head(hidden.index_select(0, last)))
        # roll, not a shifted copy into empty memory: a captured bucket's tail is padding,
        # and every row must hold a valid id for the embedding
        nxt = torch.roll(input_ids, -1)
        nxt.index_copy_(0, last, new_tokens.to(nxt.dtype))
        # the MTP plane pairs the trunk's post-final-norm state, as the reference does
        h_head, _ = lm.mtp(lm.model.embed_tokens(nxt), hidden, position_ids + 1)
        self._write_seed(h_head.index_select(0, last), new_tokens)
        return {
            rid: {"new_token": [new_tokens[i : i + 1]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    def _mtp_decode(
        self, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor, position_ids: torch.Tensor,
    ) -> dict[str, NameToTensorList]:
        """One verify step. Drafts: d1 from the seed, each later one an MTP pass whose query
        sits at its row of this step's block. Verify: the trunk over [last | drafts], the
        verdict (the target's argmax per row, then the accepted count) settled on the latent array
        and sent to the host's mailbox.
        Seed: the MTP pass over the block with the target's argmax as each row's next token (up
        to the accepted row it equals the drafts), read at the last accepted row."""
        lm, k, block = self.language_model, self.mtp_k, self.mtp_block
        embed, n = lm.model.embed_tokens, input_ids.shape[0]
        block_positions = position_ids.view(n, block)
        seed = self.seed.read(_SEED)
        h, d1 = seed.fields["hidden"], seed.fields["draft"]
        drafts = [d1]
        rows: dict[str, torch.Tensor] = {}
        for row in range(k - 1):
            h_head, _ = lm.mtp.forward_block_row(
                embed(drafts[-1]), h, block_positions[:, row] + 1, row, block, rows)
            drafts.append(self._draft_tokens(h_head, drafts[-1]))
            # the next pass's hnorm reads this pass's post-norm output, as d1's read the
            # trunk's post-final-norm state: the reference chains it so (vLLM #45895)
            h = h_head
        drafts = torch.stack(drafts, 1)
        ids = torch.cat([input_ids.view(n, 1), drafts], 1).view(-1)
        hidden = self._hidden(ids, position_ids)
        target = self.lm_head(hidden).argmax(-1).view(n, block)
        accepted = (drafts == target[:, :k]).long().cumprod(1).sum(1)
        # every rank keeps rank 0's verdict: each trims its KV and seeds its next step by it, so
        # an argmax near-tie read apart on two ranks would leave their streams apart
        verdict = self._broadcast(torch.cat([target, accepted[:, None]], 1))
        target, accepted = verdict[:, :block].contiguous(), verdict[:, block]
        # each row keeps its input token and the drafts it accepted
        settle(self.kv, accepted + 1)
        self._stage_verdict(engine_inputs, verdict)
        h_head, _ = lm.mtp(embed(target.view(-1)), hidden, position_ids + 1)
        at = torch.arange(n, device=accepted.device) * block + accepted
        next_input = target.view(-1).index_select(0, at)
        self._write_seed(h_head.index_select(0, at), next_input)
        return {
            rid: {"text_inputs": [next_input[i : i + 1]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    def _write_seed(self, h_head: torch.Tensor, prev: torch.Tensor) -> None:
        """The next step's draft seed in each row's slot: the MTP state and d1, the draft for
        the token after ``prev``. The head and the next draft pass's hnorm both read the
        shared_head-normed rows."""
        self.seed.write(_SEED, hidden=h_head.to(self.lm_head.weight.dtype), draft=self._draft_tokens(h_head, prev))

    def _draft_tokens(self, h_head: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        """The draft for the token after ``prev``, from the MTP head input ``h_head``."""
        del prev  # the tests' oracle drafts read it
        return self.lm_head(h_head).argmax(-1)

    # outside dynamo: the mailbox is the submodule's own, at addresses a captured replay keeps
    @torch.compiler.disable
    def _stage_verdict(self, engine_inputs: ModelInputsFromEngine, verdict: torch.Tensor) -> None:
        """The host's copy: the verdict first, then the flag that says it is written."""
        verdicts, ready = self._mailboxes[self._mailbox_key(engine_inputs)]
        verdicts[: verdict.shape[0]].copy_(verdict, non_blocking=True)
        ready.copy_(self._one, non_blocking=True)

    # dynamo traces no in-place broadcast
    @torch.compiler.disable
    def _broadcast(self, t: torch.Tensor) -> torch.Tensor:
        """Rank 0's copy: every rank must keep the same tail."""
        return self.lm_head.comm_group.broadcast(t, src=0)

    # ── MTP host side: the verdict mailbox ──

    @staticmethod
    def _mailbox_key(engine_inputs: ModelInputsFromEngine) -> Any:
        """A captured step's capture slot and bucket, None eager: each has its own mailbox, which a
        graph writes at a fixed address."""
        lease = engine_inputs.step.ctx.slot_lease if engine_inputs.step is not None else None
        if not engine_inputs.captured or lease is None or lease.bucket is None:
            return None
        return lease.slot, lease.bucket

    def _open_mailbox(self, engine_inputs: ModelInputsFromEngine) -> None:
        """Clear this verify step's mailbox flag and note where each real row's verdict lands. A mailbox is made, or
        an eager one grows, here, not in the forward: pinned memory can't be allocated under capture."""
        key, rows = self._mailbox_key(engine_inputs), len(engine_inputs.request_ids)
        box = self._mailboxes.get(key)
        if box is None or box[0].shape[0] < rows:
            device = self.lm_head.weight.device
            pin = device.type == "cuda"
            box = self._mailboxes[key] = (
                torch.zeros(rows, self.mtp_block + 1, dtype=torch.long, pin_memory=pin),
                torch.zeros(1, dtype=torch.long, pin_memory=pin),
            )
            if self._one is None:
                self._one = torch.ones(1, dtype=torch.long, device=device)
        box[1].zero_()
        ctx = engine_inputs.step.ctx if engine_inputs.step is not None else None
        # the real rows only: a padded replay's dummy rows run too
        real = set(ctx.request_ids if ctx is not None else engine_inputs.per_request_info)
        self._verdicts = key, {rid: row for row, rid in enumerate(engine_inputs.request_ids) if rid in real}

    def _read_verdicts(self, request_ids: list[str]) -> list[tuple[list[int], int]]:
        """This verify step's ``(tokens, accepted)`` per request, once the graph sets its mailbox's flag. The flag
        stays unset only on a device fault after the launch, which leaves this rank's other ranks in a collective,
        so a missed deadline exits the process rather than failing this rank's batch alone."""
        key, rows = self._verdicts
        verdicts, ready = self._mailboxes[key]
        deadline = time.monotonic() + _VERDICT_DEADLINE_S
        while not int(ready[0]):
            if time.monotonic() > deadline:
                raise SystemExit(f"an MTP verify step never staged its verdict in {_VERDICT_DEADLINE_S:.0f} s")
            time.sleep(0)
        table = verdicts[: max(rows[rid] for rid in request_ids) + 1].tolist()
        block = self.mtp_block
        return [(table[rows[rid]][:block], table[rows[rid]][block]) for rid in request_ids]

    def unpack_packed_outputs(
        self,
        static_output: dict,
        request_ids: list[str],
        real_seq_lens: list[int],
        inputs: list[NodeInputs],
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> dict[str, dict[str, list[torch.Tensor]]]:
        """A verify step's emitted tokens per request: the accepted drafts and the token after
        them, cut at max_tokens and at a stop token (always the last element, which check_stop
        relies on), and the tokens emitted so far, which the prefill starts at one. Waits for the
        device to stage the verdict; the seed pass is still running."""
        if self.mtp_k == 0 or not request_ids:
            return {}
        if per_request_info[request_ids[0]].graph_walk != "decode":
            return {rid: {"count": [torch.ones(1, dtype=torch.long)]} for rid in request_ids}
        eos_ids = self.config.eos_token_ids
        out = {}
        for rid, inp, (target, accepted) in zip(
            request_ids, inputs, self._read_verdicts(request_ids), strict=True,
        ):
            info = per_request_info[rid]
            self._mtp_stat_steps += 1
            self._mtp_stat_emitted += accepted + 1
            self._mtp_stat_acc_hist[accepted] += 1
            count = inp.kwargs["count"]
            emitted = target[: min(accepted + 1, max(self._budget(info) - count, 1))]
            if not info.resource_configs[SAMPLER].ignore_eos:
                for j, token in enumerate(emitted):
                    if token in eos_ids:
                        emitted = emitted[: j + 1]
                        break
            out[rid] = {
                "new_token": [torch.tensor(emitted, dtype=torch.long)],
                "count": [torch.tensor([count + len(emitted)], dtype=torch.long)],
            }
        self._maybe_log_mtp_acceptance()
        return out

    _MTP_STAT_LOG_EVERY = 512  # request-steps between acceptance log lines

    def _maybe_log_mtp_acceptance(self) -> None:
        if self._mtp_stat_steps - self._mtp_stat_logged < self._MTP_STAT_LOG_EVERY:
            return
        self._mtp_stat_logged = self._mtp_stat_steps
        k = self.config.mtp_num_draft_tokens
        mean_emitted = self._mtp_stat_emitted / self._mtp_stat_steps
        logger.info(
            "MTP acceptance: %.2f emitted/step (ceiling %d, plain decode would be "
            "1.00) — draft acceptance rate %.2f over %d request-steps.",
            mean_emitted, k + 1, (mean_emitted - 1.0) / k if k else 0.0,
            self._mtp_stat_steps,
        )
        # conditional per-position profile: p_i = P(draft i accepted |
        # drafts 1..i-1 accepted); a falling profile means the chain degrades
        reached = [sum(self._mtp_stat_acc_hist[i:]) for i in range(k + 1)]
        cond = [
            f"{reached[i] / reached[i - 1]:.2f}" if reached[i - 1] else "-"
            for i in range(1, k + 1)
        ]
        logger.info(
            "MTP acceptance by position: n_acc histogram %s, conditional accept "
            "per position %s",
            self._mtp_stat_acc_hist, " ".join(cond),
        )

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ):
        if "new_token" not in outputs:
            return
        # an MTP verify step's forward returned the next input (the token after the
        # accepted drafts); new_token is the host's cut of the verdict
        outputs.setdefault("text_inputs", outputs["new_token"])

    def check_stop(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if not self.mtp_k:
            return super().check_stop(request_id, request_info, outputs)
        if "new_token" not in outputs:
            return set()
        # in-step truncation leaves a stop id only as the LAST element
        last = int(outputs["new_token"][0].reshape(-1)[-1])
        emitted = int(outputs["count"][0])
        if emitted >= self._budget(request_info) or self._token_stops(request_id, last, request_info):
            return self._stop_names(request_info)
        return set()

    def check_stop_batched(
        self,
        request_ids: list[str],
        request_infos: dict[str, CurrentForwardPassInfo],
        host_rows: HostRows,
    ) -> dict[str, set[str]] | None:
        if self.mtp_k:
            # an MTP verify step emits several tokens a row, which check_stop counts
            return None
        return super().check_stop_batched(request_ids, request_infos, host_rows)
