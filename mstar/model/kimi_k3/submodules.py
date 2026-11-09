"""The ``LLM`` node of Kimi K3 / K3.1: embedding, the hybrid trunk and the LM head, one TP node.

Prefill and decode share the forward; per step it declares the latent KV cache, the MLA
attention over it, the KDA recurrent pool and the KDA resource, the sampler, and under
``dsa_long_context`` the DSA resource. Decode is captured per batch size, DSA included (the routed experts on
CUDA take the fused kernels: bf16, fp8, or MXFP4 on Triton / FlashInfer's FP4 MoE). Prefill is
captured at packed token buckets where the serving config names them (``prefill_token_buckets``,
which the engine applies) and the context stays within index_topk; else eager.
"""
from __future__ import annotations

import logging
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
)
from mstar.engine.accelerator_graph_runner import DEFAULT_CAPTURE_BATCH_SIZES
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import (
    AttentionStep,
    KVStep,
    SamplerStep,
    Segment,
    SlotLease,
    SubmoduleStep,
)
from mstar.engine.resources.attn.flashinfer_mla import ROW_ALIGN
from mstar.engine.resources.dsa.config import DSAStep
from mstar.engine.resources.linear_attn.config import LinearAttnStep
from mstar.engine.resources.recurrent.config import RecurrentStep
from mstar.model.kimi_k3.config import (
    ATTN,
    DSA,
    KDA,
    KDA_STATE,
    KV_CACHE,
    LABEL,
    SAMPLER,
    KimiK3ModelConfig,
)
from mstar.model.submodule_base import (
    END_PARTITION,
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    HostRows,
    InputSeqLenInfo,
    ModelInputsFromEngine,
    NodeInputs,
)

logger = logging.getLogger(__name__)

# Under dsa_long_context a decode batch replays the dense capture while every row is this many tokens
# short of index_topk: a row's context counts its tokens once the step that ran them postprocesses,
# which the async worker may not have done for the step in flight when it picks the next capture.
DSA_DENSE_MARGIN = 16
# A captured prefill's row counts (``prefill_capture_batch_sizes``): a step past the largest runs eager.
DEFAULT_PREFILL_CAPTURE_BATCH_SIZES = (1, 2, 4, 8)


class _PrefillCaptures(PackedAcceleratorGraphConfig):
    """The packed prefill captures, each row count only at token buckets its rows can fill: no row
    holds more than ``row_limit`` tokens (the context the deployment serves), so a 1-row 8192-token
    graph under a 2048-token limit would never replay."""

    def __init__(self, *, row_limit: int, **kwargs):
        super().__init__(**kwargs)
        self.row_limit = row_limit

    def get_total_tokens(self, bs: int) -> list[int]:
        return [n for n in super().get_total_tokens(bs) if n <= bs * self.row_limit]


class KimiK3LLMSubmodule(ARNodeSubmodule):
    # the KDA resource and the AttnRes reads are compiler-disabled or untested under compile
    disable_torch_compile: bool = True
    # bf16 weights; the KDA state, the router, AttnRes and SiTU run fp32 on purpose, which the
    # engine's bf16 autocast would undo
    disable_autocast: bool = True
    # no captured prefill unless the serving config names its buckets (``prefill_token_buckets``)
    PREFILL_TOKEN_BUCKETS: list[int] = []

    def __init__(self, language_model: nn.Module, config: KimiK3ModelConfig,
                 decode_capture_batch_sizes: tuple[int, ...] | None = None, compile_glue: bool = True,
                 compile_reads: bool = False,
                 prefill_capture_batch_sizes: tuple[int, ...] | None = None) -> None:
        super().__init__()
        self.language_model = language_model  # KimiK3ForCausalLM
        self.lm_head = language_model.lm_head
        self.config = config
        self.compile_glue = compile_glue
        self.compile_reads = compile_reads
        # decode graphs' batch sizes, capped by the KDA slots: the decode batch never passes the
        # largest one that fits, so a config with more slots names a longer ladder
        sizes = tuple(decode_capture_batch_sizes or DEFAULT_CAPTURE_BATCH_SIZES)
        if not sizes or list(sizes) != sorted(set(sizes)) or sizes[0] < 1:
            raise ValueError(f"decode_capture_batch_sizes must ascend from 1 up: {sizes}")
        self.decode_capture_batch_sizes = sizes
        rows = tuple(prefill_capture_batch_sizes or DEFAULT_PREFILL_CAPTURE_BATCH_SIZES)
        if not rows or list(rows) != sorted(set(rows)) or rows[0] < 1:
            raise ValueError(f"prefill_capture_batch_sizes must ascend from 1 up: {rows}")
        self.prefill_capture_batch_sizes = rows

    def bind_node_resources(self, resources: dict[str, Any]) -> None:
        if self.config.dsa_long_context:
            kv = resources[KV_CACHE].config
            if kv.tokens_per_page & (kv.tokens_per_page - 1):
                # flashinfer's top-k page-table transform takes power-of-two pages
                raise ValueError(f"{KV_CACHE} page_size={kv.tokens_per_page} must be a power of two "
                                 "under dsa_long_context")
        super().bind_node_resources(resources)

    # -- scheduling ---------------------------------------------------------

    def max_batch_size(self, graph_walk: str) -> int | None:
        """One KDA slot per row: the scheduler splits a larger step rather than fail its admit.
        Not the free slots alone: a chunked prompt's later chunks run in the slot its first took,
        so with every slot held mid-prefill a cap of the free ones would schedule none of them;
        the pool's admit refuses a new row past them, and the scheduler defers it."""
        pool = self.node_resources.get(KDA_STATE) if self.node_resources else None
        return None if pool is None else pool.config.usable_slots

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        """Decode, one graph per batch size (``decode_capture_batch_sizes``) up to the KDA slots;
        under dsa_long_context twice, keyed by ``cg_key_info``: "dense" (dense MLA, the index keys
        written) and "dsa" (every row selects over the serving window and attends sparse). Not on
        MLA's torch fallback (its plan holds the capture's pages).

        Prefill, where the serving config names ``prefill_token_buckets`` (none by default: declared
        empty, for the engine to fill) and dsa_long_context is off:
        one graph per (rows, token bucket), rows up to ``prefill_capture_batch_sizes`` and the KDA
        slots. An eager prefill step issues ~2.2 ms of host work a layer whatever its length (H200
        job 1707: 205 ms over the 93 layers from 64 to 1920 tokens; the B300's 960-token step ran
        216-221 ms), which a replay skips. A step past the largest bucket or row count runs eager."""
        attn = self.node_resources.get(ATTN) if self.node_resources else None
        if device.type != "cpu" and attn is not None and not attn.uses_kernel:
            logger.warning("kimi_k3: MLA is on the torch fallback, which cannot be captured; decode runs eager")
            return []
        pool = self.node_resources.get(KDA_STATE) if self.node_resources else None
        slots = None if pool is None else pool.config.usable_slots
        sizes = [b for b in self.decode_capture_batch_sizes if slots is None or b <= slots]
        keys = ("dense", "dsa") if self.config.dsa_long_context else (None,)
        # compile=False: the forward hosts compiler-disabled resources (KDA, the MLA cache write)
        configs: list[AcceleratorGraphConfig] = [BatchedAcceleratorGraphConfig(
            capture_graph_walk="decode",
            additional_key_info=key,
            single_request_inputs=ARNodeInputs(
                input_ids=torch.zeros(1, dtype=torch.long, device=device), input_seq_len=1),
            capture_batch_sizes=sizes,
            compile=False,
        ) for key in keys]
        if not self.config.dsa_long_context:
            rows = [b for b in self.prefill_capture_batch_sizes if slots is None or b <= slots]

            def ids(n: int) -> ARNodeInputs:
                return ARNodeInputs(input_ids=torch.zeros(n, dtype=torch.long, device=device), input_seq_len=n)

            configs.append(_PrefillCaptures(
                capture_graph_walk="prefill",
                capture_token_lengths=list(self.PREFILL_TOKEN_BUCKETS),
                make_node_input=ids,
                capture_batch_sizes=rows,
                compile=False,
                # a step with more rows than the largest capture runs eager rather than wait
                caps_eager_batch_size=False,
                row_limit=self.config.context_limit,
            ))
        return configs

    def can_replay(self, graph_walk: str, inputs) -> bool:
        """A prefill replays only where every row starts on a ``ROW_ALIGN``-key boundary: elsewhere
        the MLA plan peels the row into single-query rows (``rowwise_causal_indptrs``, for the 12
        heads a rank FlashInfer's query tile doesn't divide), more rows than the capture planned.
        A new prompt starts at 0 and a prefix hit on a page; a chunk the scheduler cut mid-way runs
        eager."""
        if graph_walk != "prefill":
            return True
        attn = self.node_resources.get(ATTN) if self.node_resources else None
        if attn is None or not getattr(attn, "_splits_short_rows", False):
            return True
        return all(inp.chunk_start % ROW_ALIGN == 0 for inp in inputs)

    def cg_key_info(self, graph_walk: str, per_request_info: dict, per_request_input_metadata=None,
                    **kwargs) -> str | None:
        return self._decode_key(graph_walk, per_request_info)

    def _decode_key(self, graph_walk: str, handles) -> str | None:
        """A decode batch's capture under dsa_long_context: "dense" while every row is
        ``DSA_DENSE_MARGIN`` tokens short of index_topk keys, else "dsa"; None otherwise."""
        if not self.config.dsa_long_context or graph_walk != "decode":
            return None
        limit = self.config.index_topk - DSA_DENSE_MARGIN
        for handle in handles:
            state = self.request_states.get(handle)
            if state is None or state.get("context", 0) + 1 > limit:
                return "dsa"
        return "dense"

    def _attn_res_flashinfer(self) -> bool:
        """The AttnRes reads run as FlashInfer's kimi_k3_attn_res on this model's device (``attn_res_read_norm``)."""
        from mstar.model.kimi_k3.components import attn_res

        model = self.language_model.model
        w = model.norm.weight
        return (model.attn_res_flashinfer and w.is_cuda and w.dtype == torch.bfloat16
                and w.shape[0] == attn_res.FI_HIDDEN and attn_res._fi(w.device) is not None)

    def compile_regions(self) -> dict[str, list[tuple]]:
        """The tensor glue between the resources, compiled on its own (the forward is not compiled
        whole: KDA, the MLA cache and attention are resource calls): each layer's two AttnRes
        reads with their norms (``read_attn`` / ``read_mlp``), its MLP block (the latent MoE's
        router, latent projections and norm, the shared experts' SiTU, the experts' custom op),
        and the output read. B300 TP8 decode at c=1 spent 44% of its 40 ms step in ~7,800 small
        elementwise kernels from this glue. Every layer runs the same code objects, so Dynamo
        traces each once and the other layers reuse the graph; dynamic shapes, so one graph per
        region serves every decode bucket and the prefill. Compiled at the capture warm-up."""
        if not self.compile_glue:
            return {}
        # the reads are Triton custom ops (attn_res_kernels) with the residual add fused: around
        # them a region holds only the norm and the block start's copy (compile_reads: false
        # leaves them eager)
        # where FlashInfer's kimi_k3_attn_res serves the reads (attn_res.attn_res_read_norm) each read, its norm and
        # block write are one launch that mutates its inputs, eager: no region around them
        reads = self.compile_reads and not self._attn_res_flashinfer()
        regions: dict[str, list[tuple]] = {"language_model.model.read_out": []} if reads else {}
        for i, layer in enumerate(self.language_model.model.layers):
            base = f"language_model.model.layers.{i}"
            if reads:
                regions[f"{base}.read_attn"] = []
                regions[f"{base}.read_mlp"] = []
            regions[f"{base}.{'block_sparse_moe' if hasattr(layer, 'block_sparse_moe') else 'mlp'}"] = []
        return regions

    # -- dtype discipline ---------------------------------------------------

    def to(self, *args, **kwargs):
        """Honor device moves; refuse post-load dtype casts (the fp32 parameters are fixed at
        load, see ``restore_fp32_params``)."""
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is not None:
            logger.info("KimiK3LLMSubmodule: ignoring post-load dtype cast to %s", dtype)
        if device is not None:
            return super().to(device=device, non_blocking=non_blocking)
        return self

    # -- engine seam ----------------------------------------------------------

    def get_input_sequence_len(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs,
    ) -> InputSeqLenInfo | None:
        """A prompt's token count, which the chunked prefill (``get_chunking_policies``) cuts by; it is
        ``prepare_inputs``' ``input_seq_len`` of the whole prompt. Main's engine manager refuses a node that
        chunks a walk without it, at startup."""
        if graph_walk == "prefill" and inputs.get("text_inputs"):
            return InputSeqLenInfo(int(inputs["text_inputs"][0].shape[0]))
        return None

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs,
    ) -> ARNodeInputs:
        text_inputs = inputs["text_inputs"][0]
        seq_len = text_inputs.shape[0]
        state = self.request_state(fwd_info.rid_handle)
        context = state.get("context", 0)
        limit = self.config.context_limit
        if context + seq_len > limit:
            # fails this request only; the engine drops it from the batch
            raise RuntimeError(
                f"request {fwd_info.request_id}: context {context + seq_len} exceeds the "
                f"{limit}-token limit this deployment serves"
                + ("" if self.config.dsa_long_context or not self.config.dsa else
                   " (index_topk, where dense MLA is exactly K3.1's DSA)"))
        if graph_walk == "prefill":
            state.add("prompt_len", context + seq_len)
        return ARNodeInputs(input_ids=text_inputs, input_seq_len=seq_len)

    def split_inputs(self, graph_walk, fwd_info, inputs, start, end):
        """The prompt's tokens ``[start, end)``, starting at ``start`` (``chunk_start``): past a cached
        prefix the engine cut too, which `postprocess` counts into the context."""
        cut = super().split_inputs(graph_walk, fwd_info, inputs, start, end)
        cut.chunk_start = start
        return cut

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
            prefill_tokens = {rid: inp.input_ids for rid, inp in zip(request_ids, inputs, strict=True)}
        steps = {
            KV_CACHE: KVStep(),
            ATTN: AttentionStep(causal=True),
            SAMPLER: SamplerStep(apply_penalty=True, prefill_tracked_tokens=prefill_tokens),
            KDA_STATE: RecurrentStep(),
            KDA: LinearAttnStep(),
        }
        key = None
        if self.config.dsa_long_context:
            # the capture the engine leased, so the plan matches the graph it replays
            key = (slot_lease.bucket.cg_key_info if slot_lease is not None and slot_lease.bucket is not None
                   else self._decode_key(graph_walk, request_ids))
            steps[DSA] = DSAStep(dense=key == "dense")
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label=LABEL, span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps=steps,
            cg_key_info=key,
        )

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        input_ids = torch.cat([inp.input_ids for inp in inputs])
        if not self.config.moe_mask_padding:
            return {"input_ids": input_ids}
        # the real rows lead a padded bucket; a fill kernel, so a replay restages it with no H2D copy
        real = engine_inputs.real_seq_len
        return {"input_ids": input_ids, "num_valid_tokens": torch.full(
            (1,), input_ids.shape[0] if real is None else real, dtype=torch.int64, device=input_ids.device)}

    def _forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor,
        num_valid_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self._logits(graph_walk, engine_inputs, input_ids, num_valid_tokens)
        return engine_inputs.resources[SAMPLER].sample(engine_inputs.request_ids, logits=logits)

    def _logits(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor,
        num_valid_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(rows, vocab)``: each request's last token's logits in a prefill, every row's in a decode."""
        attn = engine_inputs.resources[ATTN]
        rows = None
        if graph_walk == "prefill":
            # each row samples its last token: the last layer's MLP and the head run only those
            positions = torch.arange(input_ids.shape[0], device=input_ids.device)
            rows = attn.select_last_hidden(positions, LABEL)
        elif graph_walk != "decode":
            raise ValueError(f"unsupported graph walk: {graph_walk!r}")
        return self.language_model(input_ids, rows=rows, num_valid_tokens=num_valid_tokens)

    def forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        return {"new_token": self._forward(graph_walk, engine_inputs, input_ids, kwargs.get("num_valid_tokens"))}

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor,
        **kwargs,
    ) -> BatchedModelOutput:
        new_tokens = self._forward(graph_walk, engine_inputs, input_ids, kwargs.get("num_valid_tokens"))
        # row i is request i: one clone and one host copy a step
        return BatchedModelOutput(
            row_outputs={"new_token": new_tokens}, check_stop_buffers={"new_token": new_tokens})

    # -- postprocess / stop ---------------------------------------------------

    def postprocess(
        self, request_id: str, request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]], inputs: ARNodeInputs | None = None, **kwargs,
    ):
        # the step ran: its tokens are the request's context now, and a prompt's first step starts
        # past the cached prefix it skipped (`split_inputs` sets its chunk_start)
        if inputs is not None:
            state = self.request_state(request_id)
            seq_len = inputs.input_seq_len
            state.add("context", max(state.get("context", 0), inputs.chunk_start) + seq_len)
        if "new_token" in outputs:
            outputs.setdefault("text_inputs", outputs["new_token"])

    def check_stop(
        self, request_id: str, request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "new_token" not in outputs:
            return set()
        # check_stop gets host tensors: .item() does not sync
        return self._stops(request_id, request_info, outputs["new_token"][0].item())

    def check_stop_batched(
        self, request_ids: list[str], request_infos: dict[str, CurrentForwardPassInfo],
        host_rows: HostRows, step_walk: str | None = None,
    ) -> dict[str, set[str]] | None:
        pairs = host_rows.first_values("new_token", request_ids)
        if pairs is None:
            return None
        stops: dict[str, set[str]] = {}
        for rid, token in pairs:
            if loops := self._stops(rid, request_infos[rid], token):
                stops[rid] = loops
        return stops

    def _stops(self, request_id: str, request_info: CurrentForwardPassInfo, token: int) -> set[str]:
        state = self.request_states.get(request_id)
        is_eos = token in self.config.eos_token_ids
        ignore_eos = request_info.resource_configs[SAMPLER].ignore_eos
        if request_info.graph_walk == "prefill":
            # no decode_loop to stop on this walk: a stop token ends the partition here
            return {END_PARTITION} if not ignore_eos and is_eos else set()
        # 1 token from the prefill + (decode iters + 1)
        generated = request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 2
        # stop while the next step (possibly already scheduled) still fits the context limit
        prompt_len = state.get("prompt_len", 0) if state is not None else 0
        full = (prompt_len + generated >= self.config.context_limit
                and generated + 1 < request_info.max_tokens)
        if (not ignore_eos and is_eos) or generated >= request_info.max_tokens or full:
            return {"decode_loop"}
        return set()
