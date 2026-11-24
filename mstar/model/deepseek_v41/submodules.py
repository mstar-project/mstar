"""DeepSeek-V4.1 LLM node submodule wired strictly through ``mstar.engine.state``."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import FINAL_PREFILL, CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig
from mstar.engine.state import (
    ArrayStep,
    SampleStep,
    Segment,
    SlotLease,
    SubmoduleStep,
    host,
    last_sampled,
    sample,
)
from mstar.model.components.deepseek_v41.attention import SharedSelection
from mstar.model.components.deepseek_v41.backbone import DeepSeekV41TextBackbone
from mstar.model.components.deepseek_v41.engram_hash import compute_engram_hashes
from mstar.model.components.deepseek_v41.layer import DeepSeekV41Layer
from mstar.model.components.token_step import TokenStepNode
from mstar.model.deepseek_v41.config import (
    COMPRESS_STATE,
    COMPRESSOR_PARTIAL,
    NGRAM_STATE,
    REQUEST_META,
    SAMPLER,
    WINDOW_STATE,
    DeepSeekV41Config,
)
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    ModelInputsFromEngine,
    TokenStop,
)

logger = logging.getLogger(__name__)


class DeepSeekV41LLMSubmodule(TokenStepNode, ARNodeSubmodule):
    """Causal text LLM node for DeepSeek-V4.1-Flash on ``mstar.engine.state``."""

    PREFILL_WALK = "prefill"
    DECODE_WALK = "decode"
    DECODE_LOOP = "decode_loop"
    SAMPLER = SAMPLER
    MAX_PREFILL_BATCH_SIZE = 16
    # Decode reads routed tokens from postprocess so CPU/stub samplers work seamlessly.
    device_loopback = None

    def __init__(self, model: DeepSeekV41TextBackbone, config: DeepSeekV41Config) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.token_stop = TokenStop(
            loop=self.DECODE_LOOP,
            buffer="new_token",
            stop_ids=frozenset(config.stop_token_ids),
            sampler=self.SAMPLER,
            walk=self.DECODE_WALK,
            prefill_walks=frozenset({self.PREFILL_WALK}),
            last_prefill_flag=FINAL_PREFILL,
            runtime_check=True,
        )
        self.window_state: Any = None
        self.compress_state: Any = None
        self.partial_state: Any = None
        self.ngram_state: Any = None
        self.request_meta: Any = None
        self.sampler_handle: Any = None

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.window_state = handles[WINDOW_STATE]
        self.compress_state = handles.get(COMPRESS_STATE)
        self.partial_state = handles.get(COMPRESSOR_PARTIAL)
        self.ngram_state = handles.get(NGRAM_STATE)
        self.request_meta = handles[REQUEST_META]
        self.sampler_handle = handles[SAMPLER]
        device = self.model.embed.weight.device
        if self.partial_state is not None:
            ratio = max(self.config.compress_ratios[i] for i in self.config.pooled_source_layer_ids)
            self.partial_state.set_start(
                {
                    "kv": torch.zeros((ratio, self.config.head_dim), dtype=torch.float32, device=device),
                    "score": torch.full((ratio, self.config.head_dim), -torch.inf, dtype=torch.float32, device=device),
                }
            )
        if self.ngram_state is not None:
            tail_len = max(1, self.config.engram_max_ngram_size - 1)
            self.ngram_state.set_start(
                {"tail": torch.full((tail_len,), -1, dtype=torch.long, device=device)}
            )

    def get_accelerator_graph_configs(
        self,
        device: torch.device,
        tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        del device, tp_world_size
        logger.warning(
            "deepseek_v41: reference MoE dispatch and host-sync decode ops are active; "
            "decode runs eager without CUDA graph capture"
        )
        return []

    def step_is_input_free(self, graph_walk: str) -> bool:
        del graph_walk
        return False

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        has_loopback: bool = False,
        **kwargs: Any,
    ) -> ARNodeInputs:
        del has_loopback, kwargs
        meta = host(self.request_meta, fwd_info.rid_handle, "main", 0)
        pos = int(meta["pos"])
        raw_ids = inputs.get("text_inputs")
        if raw_ids:
            ids = raw_ids[0].reshape(-1).to(torch.long)
            span = int(ids.numel())
        elif graph_walk == self.DECODE_WALK:
            ids = None
            span = 1
        else:
            raise ValueError("DeepSeek-V4.1 prefill requires text_inputs")
        if pos + span > self.config.max_seq_len:
            raise ValueError(
                f"request {fwd_info.request_id} reaches {pos + span} tokens, past max_seq_len={self.config.max_seq_len}"
            )
        return ARNodeInputs(
            input_seq_len=span,
            input_ids=ids,
            kwargs={"pos": pos, "from_request": True},
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs: Any,
    ) -> SubmoduleStep:
        del slot_lease, piecewise_leases, kwargs
        segments = [
            Segment(request_id=rid, label="main", span=inp.input_seq_len)
            for rid, inp in zip(request_ids, inputs, strict=True)
        ]
        prefill_tokens = (
            {
                rid: inp.input_ids
                for rid, inp in zip(request_ids, inputs, strict=True)
                if inp.input_ids is not None
            }
            if graph_walk != self.DECODE_WALK
            else {}
        )
        host_updates = {
            (rid, "main", 0): {"pos": int(inp.kwargs["pos"]) + int(inp.input_seq_len)}
            for rid, inp in zip(request_ids, inputs, strict=True)
            if inp.kwargs.get("from_request")
        }
        state: dict[str, Any] = {
            WINDOW_STATE: ArrayStep(),
            REQUEST_META: ArrayStep(host=host_updates),
            SAMPLER: SampleStep(apply_penalty=True, prompt_tokens=prefill_tokens),
        }
        if self.compress_state is not None:
            state[COMPRESS_STATE] = ArrayStep()
        if self.partial_state is not None:
            state[COMPRESSOR_PARTIAL] = ArrayStep()
        if self.ngram_state is not None:
            state[NGRAM_STATE] = ArrayStep()
        return SubmoduleStep(segments=segments, state=state)

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor]:
        del graph_walk, engine_inputs
        device = self.model.embed.weight.device
        if any(inp.input_ids is None for inp in inputs):
            sampled = last_sampled(self.sampler_handle).to(device)
            ids_parts = [
                sampled[i : i + 1] if inp.input_ids is None else inp.input_ids.to(device)
                for i, inp in enumerate(inputs)
            ]
        else:
            ids_parts = [inp.input_ids.to(device) for inp in inputs]
        input_ids = torch.cat(ids_parts, dim=0) if len(ids_parts) > 1 else ids_parts[0]

        pos_parts: list[torch.Tensor] = []
        row_req_parts: list[torch.Tensor] = []
        req_first_parts: list[torch.Tensor] = []
        req_end_list: list[int] = []
        last_indices: list[int] = []
        cursor = 0
        for req_idx, inp in enumerate(inputs):
            span = int(inp.input_seq_len)
            start_pos = int(inp.kwargs.get("pos", 0))
            pos_parts.append(torch.arange(start_pos, start_pos + span, dtype=torch.long, device=device))
            row_req_parts.append(torch.full((span,), req_idx, dtype=torch.long, device=device))
            req_first_parts.append(torch.full((span,), cursor, dtype=torch.long, device=device))
            req_end_list.append(start_pos + span)
            last_indices.append(max(cursor + span - 1, 0))
            cursor += span

        return {
            "input_ids": input_ids,
            "pos": torch.cat(pos_parts, dim=0),
            "row_req": torch.cat(row_req_parts, dim=0),
            "req_first": torch.cat(req_first_parts, dim=0),
            "req_end_pos": torch.tensor(req_end_list, dtype=torch.long, device=device),
            "last_token_indices": torch.tensor(last_indices, dtype=torch.long, device=device),
        }

    def _run_step(
        self,
        input_ids: torch.Tensor,
        pos: torch.Tensor,
        row_req: torch.Tensor,
        req_first: torch.Tensor,
        req_end_pos: torch.Tensor,
        last_token_indices: torch.Tensor,
    ) -> torch.Tensor:
        hashes = None
        if self.model.engram_layout is not None and self.ngram_state is not None:
            held_tail = self.ngram_state.read(0).fields["tail"]
            hashes, next_tail = compute_engram_hashes(
                input_ids,
                self.model.engram_layout,
                pos=pos,
                row_req=row_req,
                req_first=req_first,
                prev_tail=held_tail,
            )
            assert next_tail is not None
            self.ngram_state.write(0, tail=next_tail)

        hidden = self.model.embed(input_ids)
        streams = hidden.unsqueeze(1).repeat(1, self.config.hc_mult, 1)  # [T, hc, D]
        pre = torch.zeros(hidden.shape[0], self.config.hc_mult, dtype=torch.float32, device=hidden.device)
        pre[:, 0] = 1.0
        shared = SharedSelection()

        comp_kv_caches: dict[int, torch.Tensor] = {}
        comp_k_caches: dict[int, torch.Tensor] = {}
        if self.compress_state is not None:
            for src_layer in self.config.kv_source_layer_ids:
                fields = self.compress_state.read(src_layer).fields
                comp_kv_caches[src_layer] = fields["compress_kv"]
                comp_k_caches[src_layer] = fields["index_k"]

        for layer_id, layer in enumerate(self.model.layers):
            if layer_id in self.model._engram_index and hashes is not None:
                idx = self.model._engram_index[layer_id]
                # DeepSeekV41Engram expects [B, S, hc, D]; pass [1, T, hc, D] then squeeze.
                streams = self.model.engrams[str(layer_id)](
                    streams.unsqueeze(0), hashes[:, idx, :].unsqueeze(0)
                ).squeeze(0)

            streams_4d = streams.unsqueeze(0)
            pre_3d = pre.unsqueeze(0)
            attn_pre, attn_post, attn_comb = layer.attn_hc.mixes(streams_4d)
            attn_in = layer.attn_norm(DeepSeekV41Layer._collapse(streams_4d, pre_3d)).squeeze(0)

            held_win = self.window_state.read(layer_id).fields["kv"]
            partial_in = None
            if self.partial_state is not None and layer_id in self.config.pooled_source_layer_ids:
                p_fields = self.partial_state.read(layer_id).fields
                partial_in = (p_fields["kv"], p_fields["score"])

            src_layer = layer.attn.kv_source_layer
            ckv = comp_kv_caches[src_layer] if src_layer is not None else None
            ck = comp_k_caches[src_layer] if src_layer is not None else None

            attn_out, next_win, next_ckv, next_ck, next_partial = layer.attn.forward_packed(
                attn_in,
                self.model.rope,
                pos,
                row_req,
                req_first,
                req_end_pos,
                held_win,
                ckv,
                ck,
                partial_in,
                shared,
            )
            self.window_state.write(layer_id, kv=next_win)
            if src_layer is not None and layer.attn.compressor is not None:
                assert next_ckv is not None and next_ck is not None
                comp_kv_caches[src_layer] = next_ckv
                comp_k_caches[src_layer] = next_ck
            if self.partial_state is not None and layer_id in self.config.pooled_source_layer_ids:
                assert next_partial is not None
                self.partial_state.write(layer_id, kv=next_partial[0], score=next_partial[1])

            streams_4d = DeepSeekV41Layer._post(streams_4d, attn_out.unsqueeze(0), attn_post, attn_comb)
            ffn_pre, ffn_post, ffn_comb = layer.ffn_hc.mixes(streams_4d)
            ffn_in = layer.ffn_norm(DeepSeekV41Layer._collapse(streams_4d, attn_pre))
            ffn_out = layer.ffn(ffn_in, None)
            streams = DeepSeekV41Layer._post(streams_4d, ffn_out, ffn_post, ffn_comb).squeeze(0)
            pre = ffn_pre.squeeze(0)

        if self.compress_state is not None:
            for src_layer in self.config.kv_source_layer_ids:
                self.compress_state.write(
                    src_layer,
                    compress_kv=comp_kv_caches[src_layer],
                    index_k=comp_k_caches[src_layer],
                )

        final_hidden = DeepSeekV41Layer._collapse(streams.unsqueeze(0), pre.unsqueeze(0)).squeeze(0)
        normed = self.model.norm(final_hidden)
        last_hidden = normed.index_select(0, last_token_indices)
        logits = self.model.logits(last_hidden)
        return sample(self.sampler_handle, logits)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        pos: torch.Tensor,
        row_req: torch.Tensor,
        req_first: torch.Tensor,
        req_end_pos: torch.Tensor,
        last_token_indices: torch.Tensor,
        **kwargs: Any,
    ) -> NameToTensorList:
        del graph_walk, engine_inputs, kwargs
        tokens = self._run_step(input_ids, pos, row_req, req_first, req_end_pos, last_token_indices)
        return {"new_token": [tokens]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        pos: torch.Tensor,
        row_req: torch.Tensor,
        req_first: torch.Tensor,
        req_end_pos: torch.Tensor,
        last_token_indices: torch.Tensor,
        **kwargs: Any,
    ) -> BatchedModelOutput:
        del graph_walk, engine_inputs, kwargs
        tokens = self._run_step(input_ids, pos, row_req, req_first, req_end_pos, last_token_indices)
        return BatchedModelOutput(
            row_outputs={"new_token": tokens},
            check_stop_buffers={"new_token": tokens, "text_inputs": tokens},
        )
