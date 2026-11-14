"""AR submodule for the GLM-5.3-Flash text backbone on the resource-pool engine."""
from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any, NamedTuple

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
from mstar.engine.state import (
    ArrayStep,
    SampleStep,
    Segment,
    SlotLease,
    SubmoduleStep,
    at_last_token,
    openable,
    sample,
    settle,
)
from mstar.model.glm5_next.config import (
    COUNT,
    KDA_STATE,
    KV_CACHE,
    LABEL,
    MTP_SEED,
    PROMPT_LEN,
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

# How long a read waits for a verify step's verdict before the process exits
_VERDICT_DEADLINE_S = 300.0

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


class _Mailbox(NamedTuple):
    """A verify step's verdict for the host, a row per request of the target's argmax at each block row and then the
    accepted count, and the flag the step sets once the verdict has landed."""

    verdicts: torch.Tensor
    ready: torch.Tensor


class Glm5NextLLMSubmodule(ARNodeSubmodule):
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
        self._int_stager = PinnedStager(torch.int32)
        self._keep_stager = PinnedStager(torch.float32)
        # capture slot (None: eager) -> the mailbox its verify steps write
        self._mailboxes: dict[int | None, _Mailbox] = {}
        # the mailbox of the verify step in flight, which the host reads before the next step is prepared
        self._staged: _Mailbox | None = None
        # requests the KDA Value holds at once, read at bind, before any request holds a slot
        self._kda_slots: int | None = None

    @property
    def mtp_block(self) -> int:
        return self.mtp_k + 1

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.kv, self.kda, self.sampler = handles[KV_CACHE], handles[KDA_STATE], handles[SAMPLER]
        self.seed = handles.get(MTP_SEED)
        self._kda_slots = openable(self.kda)

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

    def max_batch_size(self, graph_walk: str) -> int | None:
        # one KDA slot per row: the scheduler splits a larger step rather than fail its admit
        if self._kda_slots is None:
            return None
        # A prefill takes only the slots free now, so a TP follower that refuses one has
        # diverged from rank 0. The engine also holds a captured prefill to its largest
        # graph's rows: past them, the eager last layer would compile the MoE decode
        # kernels for its row count mid-serve.
        if graph_walk == "prefill":
            return min(openable(self.kda), openable(self.seed)) if self.seed is not None else openable(self.kda)
        return self._kda_slots

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
        max_slots = self._kda_slots
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
        if self.config.prefill_graphs:
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
        if graph_walk == "decode" and seq_len != 1:
            raise RuntimeError(
                f"request {fwd_info.request_id}: a decode row of {seq_len} ids, and a decode step is declared and "
                "captured over rows of one id"
            )
        if self.mtp_k > 0:
            self._check_mtp_sampling(fwd_info)
            if graph_walk == "decode":
                # one id in, the last emitted token: the step verifies it and k drafts
                seq_len = self.mtp_block
        # dsa_long_context is refused at model __init__, so the serving
        # regime is ALWAYS ctx <= index_topk, where dense MLA computes
        # exactly what GLM-5.3-Flash's DSA would — refuse beyond it rather
        # than serve off-spec logits. Raising here fails only this request
        # (the engine registers the failure and drops the rid from the batch).
        context, carried = seq_len, {}
        if graph_walk == "decode":
            # the tokens emitted so far, the last of which this step takes in
            if self.mtp_k > 0:
                count = int(inputs[COUNT][0])
                carried = {COUNT: count}
            else:
                count = fwd_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 1
            # holes are cache rows, not context: the model sees the prompt and the emitted tokens
            context = fwd_info.step_metadata[PROMPT_LEN] + count - 1 + seq_len
        if context > self.config.context_limit:
            raise RuntimeError(
                f"request {fwd_info.request_id}: context {context} exceeds {self.config.context_limit}; "
                "up to index_topk, dense MLA is exactly GLM-5.3-Flash's DSA computation"
            )
        return ARNodeInputs(input_ids=text_inputs, input_seq_len=seq_len, kwargs=carried)

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
        segments = [
            Segment(request_id=rid, label=LABEL, span=inp.input_seq_len)
            for rid, inp in zip(request_ids, inputs, strict=True)
        ]
        if self.mtp_k > 0 and graph_walk == "decode":
            # the verify and the drafts are argmaxes in the forward, so the step samples nothing
            state = {KV_CACHE: ArrayStep(verify=True), KDA_STATE: ArrayStep(verify=True), MTP_SEED: ArrayStep()}
            return SubmoduleStep(segments=segments, state=state)
        prompt_tokens = {}
        if graph_walk == "prefill":
            prompt_tokens = {rid: inp.input_ids for rid, inp in zip(request_ids, inputs, strict=True)}
        state = {
            KV_CACHE: ArrayStep(), KDA_STATE: ArrayStep(),
            SAMPLER: SampleStep(apply_penalty=True, prompt_tokens=prompt_tokens),
        }
        if self.mtp_k > 0:
            state[MTP_SEED] = ArrayStep()
        return SubmoduleStep(segments=segments, state=state)

    def step_is_input_free(self, graph_walk: str) -> bool:
        # every decode row spans one block, as the capture's template row does, and declares nothing else
        return graph_walk == "decode"

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
            out["mtp_keep"] = self._mtp_keep(inputs, out["input_ids"].shape[0])
        if self.mtp_k > 0 and graph_walk == "decode":
            # the step sets the flag once its verdict has landed
            self._staged = self._mailbox(engine_inputs)
            self._staged.ready.zero_()
        return out

    def _mtp_keep(self, inputs: list[ARNodeInputs], num_tokens: int) -> torch.Tensor:
        """1 per prefill token, 0 at each row's first token: GLM-5.3 declares no chunking, so
        that token sits at position 0, where the MTP pass zeroes its embedding, as vLLM does."""
        keep, start = [1.0] * num_tokens, 0
        for inp in inputs:
            if inp.input_seq_len > 0 and start < num_tokens:
                keep[start] = 0.0
            start += inp.input_seq_len
        # staged in fp32 (no bf16 host view), cast on the device
        return _upload(self._keep_stager, keep, self.lm_head.weight.device).to(self.lm_head.weight.dtype)

    def _mailbox_key(self, engine_inputs: ModelInputsFromEngine) -> int | None:
        """A captured step's capture slot, None eager: each has its own mailbox, which a
        graph writes at a fixed address."""
        lease = engine_inputs.step.ctx.slot_lease if engine_inputs.step is not None else None
        if not engine_inputs.captured or lease is None or lease.bucket is None:
            return None
        return lease.slot

    def _mailbox(self, engine_inputs: ModelInputsFromEngine) -> _Mailbox:
        key = self._mailbox_key(engine_inputs)
        if key not in self._mailboxes:
            pin = self.lm_head.weight.device.type == "cuda"
            # a step's rows index it, up to every KDA slot and the sink
            rows = self._kda_slots + 1
            self._mailboxes[key] = _Mailbox(
                torch.zeros(rows, self.mtp_block + 1, dtype=torch.long, pin_memory=pin),
                torch.zeros(1, dtype=torch.long, pin_memory=pin),
            )
        return self._mailboxes[key]

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
    ) -> torch.Tensor:
        last = None  # prefill samples each row's last token: the model computes only those
        if last_rows is not None:
            last = last_rows
        elif graph_walk == "prefill":
            last = at_last_token(self.kv, torch.arange(input_ids.shape[0], device=input_ids.device))
        elif graph_walk != "decode":
            raise ValueError(f"unsupported graph walk: {graph_walk!r}")
        if self.mtp_k > 0 and graph_walk == "prefill":
            # the MTP pass reads the trunk's hidden state at every prompt row
            hidden = self.language_model.model(input_ids)
            last = last.long()
            new_tokens = sample(self.sampler, self.lm_head(hidden.index_select(0, last)))
            self._mtp_prefill(input_ids, hidden, last, new_tokens, mtp_keep)
            return new_tokens
        hidden = self.language_model.model(input_ids, rows=last, decode=graph_walk == "decode")
        return sample(self.sampler, self.lm_head(hidden))

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

    def _mtp_seed(self, hidden: torch.Tensor, prev: torch.Tensor) -> None:
        """The next step's draft seed: the MTP state and d1, the draft for the token after
        ``prev`` (the step's next input)."""
        self.seed.write(0, hidden=hidden, draft=self._draft_tokens(hidden, prev))

    def _mtp_prefill(self, input_ids, hidden, last, new_tokens, mtp_keep) -> None:
        """The MTP pass over the prompt: row i pairs the trunk's state at token i with the
        embedding of token i + 1 (each request's last row: its sampled token). It fills the
        MTP layer's KV plane and drafts d1 from each request's last row."""
        lm = self.language_model
        # roll, not a shifted copy into empty memory: a captured bucket's last slot is
        # padding, and every slot must hold a valid id for the embedding
        nxt = torch.roll(input_ids, -1)
        nxt.index_copy_(0, last, _real_row_values(last, new_tokens).to(nxt.dtype))
        embeds = lm.model.embed_tokens(nxt) * mtp_keep[:, None]
        self._mtp_seed(lm.mtp(embeds, hidden).index_select(0, last), new_tokens)

    def _mtp_decode(self, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor) -> torch.Tensor:
        """One verify step: draft k tokens, verify ``[last | drafts]`` with the trunk, then run
        the MTP layer over the block to seed the next step. The verdict (the target's argmax
        per block row, then the accepted count) goes to the host's mailbox; returns the next
        input token per row."""
        lm, k, block = self.language_model, self.mtp_k, self.mtp_block
        n = input_ids.shape[0]

        # Drafts: d1 was seeded by the last step; each further one is an MTP pass whose
        # query sits at its row of this step's block.
        seed = self.seed.read(0).fields
        h, drafts = seed["hidden"], [seed["draft"]]
        attn = lm.mtp.transformer_layer.self_attn
        latents = h.new_zeros(n, block, attn.kv_lora_rank)
        for row in range(k - 1):
            h = lm.mtp(lm.model.embed_tokens(drafts[-1]), h, row=row, latents=latents)
            drafts.append(self._draft_tokens(h, drafts[-1]))
        drafts = self._broadcast(torch.stack(drafts, 1))

        # Verify: greedy, the longest prefix of drafts the target agrees with.
        ids = torch.cat([input_ids.view(n, 1), drafts], 1).view(-1)
        hidden = lm.model(ids, decode=True)
        target = self._argmax(hidden).view(n, block)
        accepted = (drafts == target[:, :k]).long().cumprod(1).sum(1)
        verdict = self._broadcast(torch.cat([target, accepted[:, None]], 1))
        target, accepted = verdict[:, :block], verdict[:, block]
        settle(self.kda, accepted + 1)

        # The host's copy: the verdict first, then the flag that it has landed.
        box = self._mailbox(engine_inputs)
        box.verdicts[:n].copy_(verdict, non_blocking=True)
        box.ready.copy_(accepted.new_ones(1), non_blocking=True)

        # Seed the next step: the MTP pass over the block, with the target's argmax as each
        # row's next token (up to the accepted row it equals the drafts), read at the last
        # accepted row.
        each = torch.arange(n, device=accepted.device)
        next_input = target[each, accepted]
        h_block = lm.mtp(lm.model.embed_tokens(target.reshape(-1)), hidden)
        self._mtp_seed(h_block.view(n, block, -1)[each, accepted], next_input)
        settle(self.kv, accepted + 1)
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
        if self.mtp_k > 0 and graph_walk == "decode":
            return self.forward_batched(graph_walk, engine_inputs, input_ids)[engine_inputs.request_ids[0]]
        return {"new_token": self._forward(
            graph_walk, engine_inputs, input_ids, last_rows, mtp_keep)}

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
        if self.mtp_k > 0 and graph_walk == "decode":
            # the emitted tokens are cut on the host (unpack_packed_outputs); the next input
            # is the token after the accepted drafts
            next_input = self._mtp_decode(engine_inputs, input_ids)
            return {
                rid: {"text_inputs": [next_input[i : i + 1]]}
                for i, rid in enumerate(engine_inputs.request_ids)
            }
        new_tokens = self._forward(graph_walk, engine_inputs, input_ids, last_rows, mtp_keep)
        # row i is request i: one clone out of the graph and one host copy a step
        return BatchedModelOutput(
            row_outputs={"new_token": new_tokens},
            check_stop_buffers={"new_token": new_tokens},
        )

    # -- MTP host side: the verdict mailbox ---------------------------------

    def _log_acceptance(self, accepted: int, every_s: float = 10.0) -> None:
        """Rank 0 logs the accepted-draft histogram at most every ``every_s`` seconds."""
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

    def _read_verdict(self, rows: int) -> list[tuple[list[int], int]]:
        """The ``(tokens, accepted)`` of the verify step in flight for each of its first ``rows``
        rows, waiting for the device to stage them."""
        box = self._staged
        # A step that failed before its verdict landed never sets the flag. Failing this rank's
        # batch alone would leave the other ranks in its collective, so a missed deadline exits
        # the process: the instance goes down where everyone sees it, rather than hanging.
        deadline = time.monotonic() + _VERDICT_DEADLINE_S
        while not int(box.ready[0]):
            if time.monotonic() > deadline:
                raise SystemExit(f"an MTP verify step never staged its verdict in {_VERDICT_DEADLINE_S:.0f} s")
            time.sleep(0)
        k1 = self.mtp_block
        return [(row[:k1], row[k1]) for row in box.verdicts[:rows].tolist()]

    def unpack_packed_outputs(
        self,
        static_output: dict,
        request_ids: list[str],
        real_seq_lens: list[int],
        inputs: list[NodeInputs],
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> dict[str, NameToTensorList]:
        """A verify step's emitted tokens per request, the accepted drafts and the token after
        them, cut at a stop token and at max_tokens, and the count of tokens emitted after it.
        Waits for the device to stage the verdict (the MTP pass that seeds the next step is
        still running)."""
        if self.mtp_k == 0 or not request_ids:
            return {}
        if per_request_info[request_ids[0]].graph_walk != "decode":
            # the prefill emits each request's first token
            return {rid: {COUNT: [torch.ones(1, dtype=torch.long)]} for rid in request_ids}
        out = {}
        verdicts = self._read_verdict(len(request_ids))
        for rid, inp, (target, accepted) in zip(request_ids, inputs, verdicts, strict=True):
            info, count = per_request_info[rid], inp.kwargs[COUNT]
            emitted = target[:accepted + 1][:max(info.max_tokens - count, 1)]
            if not info.resource_configs[SAMPLER].ignore_eos:
                for i, token in enumerate(emitted):
                    if token in self.config.eos_token_ids:
                        emitted = emitted[:i + 1]
                        break
            self._log_acceptance(accepted)
            out[rid] = {
                "new_token": [torch.tensor(emitted, dtype=torch.long)],
                COUNT: [torch.tensor([count + len(emitted)], dtype=torch.long)],
            }
        return out

    # -- slow-postprocess path (worker thread, after execute_batch) -------

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        inputs: ARNodeInputs | None = None,
        **kwargs,
    ):
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
        # A prefill's stop check gets the step's stop buffers, its sampled tokens alone, and not the count it persists
        # for the decode loop. Its one token stops as a plain step's does.
        if self.mtp_k > 0 and request_info.graph_walk != "prefill":
            return self._mtp_check_stop(request_info, outputs["new_token"][0], int(outputs[COUNT][0]))
        # check_stop gets host tensors, so .item() here does not sync
        return self._stops(request_info, outputs["new_token"][0].item())

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
            if loops := self._stops(request_infos[rid], token):
                stops[rid] = loops
        return stops

    def _stops(self, request_info: CurrentForwardPassInfo, token: int) -> set[str]:
        """The loops a plain step's emitted ``token`` stops."""
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
        full = (request_info.step_metadata[PROMPT_LEN] + generated >= self.config.context_limit
                and generated + 1 < request_info.max_tokens)
        if (not ignore_eos and is_eos) or generated >= request_info.max_tokens or full:
            return {"decode_loop"}
        return set()

    def _mtp_check_stop(
        self, request_info: CurrentForwardPassInfo, new_token: torch.Tensor, count: int,
    ) -> set[str]:
        """Stop on a stop token among this step's tokens, at max_tokens, or while two more
        blocks fit in the context (the next step may already be scheduled when this stop
        lands). ``count`` is the tokens emitted up to this step. The context is the stricter
        bound: every verify step keeps its whole block in the cache, rejected rows as holes,
        and the cache holds ``k + 1`` rows for each token of the context."""
        ignore_eos = request_info.resource_configs[SAMPLER].ignore_eos
        is_eos = not ignore_eos and any(
            t in self.config.eos_token_ids for t in new_token.tolist())
        full = request_info.step_metadata[PROMPT_LEN] + count + 2 * self.mtp_block > self.config.context_limit
        if is_eos or count >= request_info.max_tokens or full:
            return {"decode_loop"}
        return set()
