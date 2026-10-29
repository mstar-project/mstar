# ---------------------------------------------------------------------------
# NodeSubmodule wrappers for Whisper ASR
# ---------------------------------------------------------------------------
#
# Two submodules covering the encoder-decoder pipeline:
#   1. WhisperEncoderSubmodule  (stateless)  — HF WhisperEncoder, one-shot
#   2. WhisperDecoderSubmodule  (KV/attention/cross-attention/positions/sampler)
# ---------------------------------------------------------------------------

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig, BatchedAcceleratorGraphConfig
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.position.manager import PositionManager
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    HostRows,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)
from mstar.model.whisper.components.decoder import WhisperDecoderModel
from mstar.model.whisper.config import (
    ATTN,
    CONTEXT_LABEL,
    CROSS_ATTN,
    CROSS_KV_CACHE,
    KV_CACHE,
    POS,
    SAMPLER,
    WhisperModelConfig,
)
from mstar.utils.knobs import device_loopback_enabled

logger = logging.getLogger(__name__)


# ===================================================================
# 1. WhisperEncoderSubmodule (stateless)
# ===================================================================


class WhisperEncoderSubmodule(NodeSubmodule):
    """Thin wrapper around the HF Whisper audio encoder.

    Consumes the log-mel spectrogram produced by ``process_prompt``
    (a fixed 30 s window: ``(num_mel_bins, 3000)``) and emits
    ``encoder_states`` of shape ``(max_source_positions, d_model)``
    for the decoder's cross-attention. Runs once per request.

    Every clip is padded to that window, so a batch is one fixed
    ``(B, num_mel_bins, 3000)`` mel and each batch size is captured whole:
    uncaptured, the tower's kernel launches take longer than its GPU work.
    """

    CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]

    def __init__(self, audio_encoder: nn.Module, config: WhisperModelConfig):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.config = config

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        conv1, conv2 = self.audio_encoder.conv1, self.audio_encoder.conv2
        frames = self.config.max_source_positions * conv1.stride[0] * conv2.stride[0]
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="prefill",
                single_request_inputs=NodeInputs(
                    tensor_inputs={"audio_features": torch.zeros(
                        self.config.num_mel_bins, frames, device=device,
                    )},
                    # one window per row, so a step's real_seq_len counts its real rows
                    input_seq_len=1,
                ),
                capture_batch_sizes=self.CAPTURE_BATCH_SIZES,
                input_seq_dims={"audio_features": 0},
            ),
        ]

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        return NodeInputs(
            tensor_inputs={"audio_features": inputs["audio_features"][0]},
            input_seq_len=1,
        )

    def can_batch(
        self, batch: ExecutingBatch, model_inputs: list[NodeInputs],
    ) -> bool:
        return True

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor]:
        # Real rows only: a replay's padding rows keep what the static buffer
        # last held, and their outputs are dropped.
        rows = len(inputs) if engine_inputs.real_seq_len is None else engine_inputs.real_seq_len
        feats = torch.stack(
            [inp.tensor_inputs["audio_features"] for inp in inputs[:rows]]
        )
        return {"audio_features": feats.to(self.get_device())}

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        rid = engine_inputs.request_ids[0]
        return self.forward_batched(graph_walk, engine_inputs, audio_features)[rid]

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: torch.Tensor,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        dtype = next(self.audio_encoder.parameters()).dtype
        encoder_states = self.audio_encoder(audio_features.to(dtype)).last_hidden_state
        return {
            rid: {"encoder_states": [encoder_states[i]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }


# ===================================================================
# 2. WhisperDecoderSubmodule
# ===================================================================


class WhisperDecoderSubmodule(ARNodeSubmodule):
    """Autoregressive Whisper decoder.

    Dispatches on graph_walk:
      - prefill: embed the forced decoder prompt
        (``<|startoftranscript|><|lang|><|task|><|notimestamps|>``),
        project ``encoder_states`` to per-layer cross-attention K/V and
        write them into the context stream, fill the self-attention KV
        cache, and sample the first transcript token.
      - decode: embed the previous token, single-step decode.

    Both cache streams belong to the step: ``main`` grows by the token
    count, ``context`` grows by the encoder output at prefill and is
    declared zero-span (read-only) thereafter. Both walks are captured per
    batch size; prefill's capture config names the context it writes per
    row, which the engine sizes the context cache's capture buffers by.
    """

    # Up to the requests the decoder caches hold
    # (`WhisperModel.MAX_CONCURRENT_REQUESTS`), so they all decode in one step.
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    # The encoder's batch sizes: a prefill batch is what one encoder step emitted.
    PREFILL_CAPTURE_BATCH_SIZES = WhisperEncoderSubmodule.CAPTURE_BATCH_SIZES

    # the decode walk reads its input token off the sampler's slot masters
    reads_device_loopback = True

    def __init__(self, decoder: WhisperDecoderModel, config: WhisperModelConfig):
        super().__init__()
        self.decoder = decoder
        self.config = config
        self._suppress_ids: torch.Tensor | None = None
        self._begin_suppress_ids: torch.Tensor | None = None

    def _apply_suppress(self, logits: torch.Tensor, is_first_token: bool) -> torch.Tensor:
        """HF generate parity: mask the always-suppressed token set, plus
        the begin-suppressed set for the first generated token."""
        device = logits.device
        if self._suppress_ids is None:
            self._suppress_ids = torch.tensor(
                self.config.suppress_tokens, dtype=torch.long, device=device,
            )
            self._begin_suppress_ids = torch.tensor(
                self.config.begin_suppress_tokens, dtype=torch.long, device=device,
            )
        if self._suppress_ids.numel():
            logits.index_fill_(-1, self._suppress_ids, float("-inf"))
        if is_first_token and self._begin_suppress_ids.numel():
            logits.index_fill_(-1, self._begin_suppress_ids, float("-inf"))
        return logits

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        prompt_len = self.config.decoder_prompt_len
        context = self.config.max_source_positions
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="decode",
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device),
                    input_seq_len=1,
                ),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
            ),
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="prefill",
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(prompt_len, dtype=torch.long, device=device),
                    input_seq_len=prompt_len,
                    # float32: the encoder's last layer norm leaves it so under autocast
                    tensor_inputs={"encoder_states": torch.zeros(
                        context, self.config.d_model, device=device,
                    )},
                ),
                capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
                input_seq_dims={"input_ids": 0, "encoder_states": 0},
                # every row writes its whole encoder context at prefill
                resource_write_tokens={CROSS_KV_CACHE: context},
            ),
        ]

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        has_loopback: bool = False,
        **kwargs,
    ) -> ARNodeInputs:
        if graph_walk == "decode" and has_loopback and self._device_loopback():
            # one token, read off the sampler's slot master in `preprocess`
            return ARNodeInputs(input_seq_len=1)
        device = self.get_device()
        token_ids = inputs["text_inputs"][0].to(device).reshape(-1)

        tensor_inputs = {}
        if graph_walk == "prefill":
            tensor_inputs["encoder_states"] = inputs["encoder_states"][0].to(device)

        return ARNodeInputs(
            input_seq_len=token_ids.shape[0],
            input_ids=token_ids,
            tensor_inputs=tensor_inputs,
        )

    def uniform_row_inputs(
        self, graph_walk: str, has_loopback: bool = False,
    ) -> ARNodeInputs | None:
        # the loop-back decode row above, built once for the step's rows
        if graph_walk == "decode" and has_loopback and self._device_loopback():
            return ARNodeInputs(input_seq_len=1)
        return None

    def _device_loopback(self) -> bool:
        return (
            device_loopback_enabled()
            and self.node_resources[SAMPLER].has_slot_masters
        )

    def device_loopback_signals(self, graph_walk: str) -> frozenset[str]:
        if graph_walk == "decode" and self._device_loopback():
            return frozenset({"text_inputs"})
        return frozenset()

    def inline_client_signals(self, graph_walk: str) -> dict[str, str]:
        # decode's client edge carries the token the stop check already has on
        # the host; prefill's is persisted for the decode walk
        return {"text_inputs": "new_token"} if graph_walk == "decode" else {}

    def step_is_input_free(self, graph_walk: str) -> bool:
        # a decode row spans one token and no context, as the template does
        return graph_walk == "decode"

    @staticmethod
    def _context_span(inp: ARNodeInputs) -> int:
        """How far this step grows the request's encoder-context stream.

        Non-zero only on the prefill that writes it; every later step reads
        the same pages without extending them.
        """
        encoder_states = inp.tensor_inputs.get("encoder_states")
        return 0 if encoder_states is None else encoder_states.shape[0]

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        context_segments = tuple(
            Segment(
                request_id=rid,
                label=CONTEXT_LABEL,
                span=self._context_span(inp),
            ) for rid, inp in zip(request_ids, inputs, strict=True)
        )
        return SubmoduleStep(
            # the default, for the resources over the self-attention cache
            segments=[
                Segment(
                    request_id=rid,
                    label="main",
                    span=inp.input_seq_len,
                ) for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={
                KV_CACHE: KVStep(),
                ATTN: AttentionStep(causal=True),
                # The context stream: a real span on the prefill that writes
                # it, zero afterwards. `commit` is what turns the prefill's
                # reservation into resident pages the later steps read.
                CROSS_KV_CACHE: KVStep(segments=context_segments),
                CROSS_ATTN: AttentionStep(
                    segments=context_segments, causal=False,
                ),
                SAMPLER: SamplerStep(apply_penalty=False),
                POS: PositionStep(),
            },
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        if graph_walk == "decode" and any(inp.input_ids is None for inp in inputs):
            # Each row's last sampled token, read on the device by slot. Real
            # rows whose last step did not loop back here brought theirs.
            sampler: SamplerResource = engine_inputs.resources[SAMPLER]
            ids = sampler.loopback_tokens(engine_inputs.request_ids)
            real = len(inputs) if engine_inputs.real_seq_len is None else engine_inputs.real_seq_len
            routed = [i for i in range(real) if inputs[i].input_ids is not None]
            if routed:
                ids.index_copy_(
                    0, torch.tensor(routed, device=ids.device),
                    torch.cat([inputs[i].input_ids for i in routed]).to(ids.dtype),
                )
            return {"input_ids": ids}
        preprocessed: dict[str, torch.Tensor | Any] = {
            "input_ids": torch.cat([inp.input_ids for inp in inputs]),
        }
        if graph_walk == "prefill":
            # Concatenated in segment order, which is how the context plan
            # laid the requests' pages out.
            preprocessed["encoder_states"] = torch.cat(
                [inp.tensor_inputs["encoder_states"] for inp in inputs], dim=0,
            )
        return preprocessed

    def _forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        encoder_states: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the decoder and sample one token per request."""
        attn: AttentionManager = engine_inputs.resources[ATTN]
        pos: PositionManager = engine_inputs.resources[POS]
        sampler: SamplerResource = engine_inputs.resources[SAMPLER]

        if encoder_states is not None:
            # prefill only: fills the context pages this step reserved, before
            # the layers read them back
            self.decoder.write_cross_kv(encoder_states)

        input_embeds = self.decoder.embed(input_ids, pos.pos_ids("main"))
        hidden = self.decoder(input_embeds=input_embeds, label="main")

        if graph_walk == "prefill":
            # packed prefill: one hidden per request, at its last token
            hidden = attn.select_last_hidden(hidden)

        logits = self.decoder.lm_head(hidden)
        logits = self._apply_suppress(
            logits, is_first_token=graph_walk == "prefill",
        )
        return sampler.sample(engine_inputs.request_ids, logits=logits)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        return {
            "new_token": [self._forward(
                graph_walk=graph_walk,
                engine_inputs=engine_inputs,
                input_ids=input_ids,
                encoder_states=encoder_states,
            )]
        }

    def can_batch(
        self, batch: ExecutingBatch,
        model_inputs: list[NodeInputs],
    ) -> bool:
        return True

    @staticmethod
    def prefill_rows(context_kv: PagedKVConfig) -> int:
        """Prefill rows the encoder-context cache holds at once, next to its
        sink page: each row reserves a whole context."""
        per_request = -(-context_kv.max_seq_len // context_kv.page_size)
        return (context_kv.max_num_pages - 1) // per_request

    def max_batch_size(self, graph_walk: str) -> int | None:
        # Both walks are bounded by their captured sizes too; this keeps a
        # prefill batch within the contexts the cache holds should a
        # deployment shrink it below the largest prefill bucket.
        if graph_walk != "prefill":
            return None
        return self.prefill_rows(self.node_resources[CROSS_KV_CACHE].config)

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        **kwargs,
    ) -> BatchedModelOutput:
        new_tokens = self._forward(
            graph_walk=graph_walk,
            engine_inputs=engine_inputs,
            input_ids=input_ids,
            encoder_states=encoder_states,
        )
        # Row i is request i: one clone and one host copy per step instead of
        # one per request.
        return BatchedModelOutput(
            row_outputs={"new_token": new_tokens},
            check_stop_buffers={"new_token": new_tokens},
        )

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ):
        # Metadata-only: rebind output name so the decode loop feeds the
        # sampled token back in as the next step's text_inputs.
        if "new_token" not in outputs:
            return
        outputs["text_inputs"] = outputs["new_token"]

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "new_token" not in outputs:
            return set()
        token = outputs["new_token"][0].item()
        return {"decode_loop"} if self._stops(token, request_info) else set()

    def _stops(self, token: int, request_info: CurrentForwardPassInfo) -> bool:
        # the token test first: EOS is rare, the config read behind it is not free
        if token == self.config.eos_token_id and not request_info.resource_configs[SAMPLER].ignore_eos:
            return True
        decoded_tokens = request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 1
        return decoded_tokens >= request_info.max_tokens

    def check_stop_batched(
        self,
        request_ids: list[str],
        request_infos: dict[str, CurrentForwardPassInfo],
        host_rows: HostRows,
        step_walk: str | None = None,
    ) -> dict[str, set[str]] | None:
        """``check_stop`` for every request off one ``tolist``."""
        tokens = host_rows.buffers.get("new_token")
        if not torch.is_tensor(tokens) or tokens.dim() == 0:
            return None
        values = tokens.reshape(tokens.shape[0], -1)[:, 0].tolist()
        rows = host_rows.request_ids
        if tuple(rows[:len(request_ids)]) == tuple(request_ids):
            pairs = zip(request_ids, values, strict=False)
        else:
            row_of = {rid: i for i, rid in enumerate(rows)}
            # no row, no output: the per-request path skips it too
            pairs = (
                (rid, values[i]) for rid in request_ids
                if (i := row_of.get(rid)) is not None and i < len(values)
            )
        return {
            rid: {"decode_loop"} for rid, token in pairs
            if self._stops(token, request_infos[rid])
        }
