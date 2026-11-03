# ---------------------------------------------------------------------------
# NodeSubmodule wrappers for Higgs-Audio STT
# ---------------------------------------------------------------------------
#
# Two submodules:
#   1. HiggsAudioEncoderSubmodule  (stateless) — checkpoint's Whisper-style
#      audio_tower + audio_encoder_proj MLP projector, one-shot at prefill
#   2. HiggsAudioLLMSubmodule      (KV/attention/positions/sampler) — dense Qwen3 LLM
# ---------------------------------------------------------------------------

from __future__ import annotations

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
    PiecewiseAcceleratorGraphConfig,
    PiecewiseBatchedConfig,
    PiecewiseCallInputs,
)
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.higgs_audio.config import (
    ATTN,
    AUDIO_FEATURE_LENS_KEY,
    KV_CACHE,
    ROPE,
    SAMPLER,
    HiggsAudioModelConfig,
)
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    DeviceLoopback,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
    TokenStop,
)
from mstar.utils.h2d import to_device_nonblocking

logger = logging.getLogger(__name__)


# ===================================================================
# 1. HiggsAudioEncoderSubmodule (stateless)
# ===================================================================


class HiggsAudioEncoderSubmodule(NodeSubmodule):
    """Audio tower + projector.

    Consumes per-chunk mel spectrograms from ``process_prompt``
    (``(num_chunks, num_mel_bins, T)`` padded to the longest chunk, plus
    per-chunk mel frame counts) and emits the concatenated LLM-space
    audio embeddings ``(total_audio_tokens, hidden_size)``.

    Mirrors the reference pipeline: batch-encode padded chunks, run the
    projector on the padded batch, then slice each chunk to its valid
    (downsampled) length and concatenate in order. Chunks attend only
    within themselves, so requests whose chunks share a width are encoded
    as one batch.

    The chunk count and, for a clip under one chunk, the mel width change
    per request; compiled, every new shape recompiled on the serving path
    (tens of seconds with the worker blocked). So the tower runs eagerly,
    and full-width chunks replay a CUDA graph by chunk count. A clip under
    one chunk is padded to the full width and replays a second graph whose
    tower masks the padding out (``HiggsAudioTower.forward``'s
    ``valid_frames``), so short clips batch together too.
    """

    disable_torch_compile = True
    # requests per encoder step, and full-width chunks per captured replay
    MAX_BATCH_SIZE = 16
    CAPTURE_CHUNK_COUNTS = [1, 2, 4, 8, 16, 32, 64]
    TOWER = "tower"
    MASKED_TOWER = "tower_masked"

    def __init__(
        self,
        audio_tower: nn.Module,
        projector: nn.Module,
        config: HiggsAudioModelConfig,
        graphs: bool = True,
    ):
        super().__init__()
        self.audio_tower = audio_tower
        self.projector = projector
        self.config = config
        self.graphs = graphs
        # mel frames of a full chunk: the feature extractor's 10 ms hop
        self.chunk_frames = int(config.chunk_size_seconds * 100)

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        features = inputs["audio_features"][0]
        if features.dim() == 2:
            features = features.unsqueeze(0)  # (1, num_mel_bins, T)
        # each chunk's valid mel frames, known on the host since process_prompt
        frames = list(fwd_info.step_metadata[AUDIO_FEATURE_LENS_KEY])
        return NodeInputs(
            tensor_inputs={"audio_features": features},
            kwargs={
                "frames": frames,
                "valid_lengths": [self.config.encoder_output_length(n) for n in frames],
            },
            input_seq_len=features.shape[0],
        )

    def _padded(self, width: int) -> bool:
        """A clip under one chunk: run at the full width, its tail masked,
        captured or not."""
        return width < self.chunk_frames

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        # chunks are padded to their own request's longest, so only requests of
        # one mel width share a batch -- or, padded and masked, short clips
        widths = {inp.tensor_inputs["audio_features"].shape[-1] for inp in model_inputs}
        chunks = sum(inp.input_seq_len for inp in model_inputs)
        same_kind = len(widths) == 1 or all(self._padded(w) for w in widths)
        return same_kind and chunks <= self.CAPTURE_CHUNK_COUNTS[-1]

    def max_batch_size(self, graph_walk: str) -> int:
        return self.MAX_BATCH_SIZE

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        dtype = next(self.audio_tower.parameters()).dtype
        features = [inp.tensor_inputs["audio_features"] for inp in inputs]
        widths = [f.shape[-1] for f in features]
        padded = self._padded(widths[0])
        if padded:
            features = [F.pad(f, (0, self.chunk_frames - f.shape[-1])) for f in features]
        out = {
            "audio_features": torch.cat(features, dim=0).to(device=self.get_device(), dtype=dtype),
            "valid_lengths": [inp.kwargs["valid_lengths"] for inp in inputs],
        }
        if padded:
            # a short clip is one chunk; its mask sum can be one past its mel
            # width, and the slice below counts from the mask sum too
            out["valid_frames"] = to_device_nonblocking(
                [inp.kwargs["frames"][0] for inp in inputs], self.get_device(), dtype=torch.long,
            )
        return out

    def _encode(self, features: torch.Tensor, valid_frames: torch.Tensor | None = None) -> torch.Tensor:
        return self.projector(self.audio_tower(features, valid_frames))

    def _capture_tower(self, inp: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        return {"projected": self._encode(inp.static_inputs["audio_features"])}

    def _capture_masked_tower(self, inp: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        s = inp.static_inputs
        # padding rows replay zeroed: give them one valid frame
        return {"projected": self._encode(s["audio_features"], s["valid_frames"].clamp(min=1))}

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype, tp_world_size: int = 1, **kwargs: Any,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        """Tower + projector over full-width chunks, by chunk count."""
        del autocast_dtype, tp_world_size, kwargs
        if not self.graphs:
            return {}
        dtype = next(self.audio_tower.parameters()).dtype
        mels, frames = self.config.audio_num_mel_bins, self.chunk_frames
        return {
            self.TOWER: PiecewiseBatchedConfig(
                capture_fn=self._capture_tower,
                make_static_inputs=lambda shape: {
                    "audio_features": torch.zeros(shape.bs, mels, frames, dtype=dtype, device=device),
                },
                seq_len=frames,
                capture_batch_sizes=list(self.CAPTURE_CHUNK_COUNTS),
            ),
            # clips under one chunk, one per row, padded to the full width
            self.MASKED_TOWER: PiecewiseBatchedConfig(
                capture_fn=self._capture_masked_tower,
                make_static_inputs=lambda shape: {
                    "audio_features": torch.zeros(shape.bs, mels, frames, dtype=dtype, device=device),
                    "valid_frames": torch.full((shape.bs,), frames, dtype=torch.long, device=device),
                },
                seq_len=frames,
                capture_batch_sizes=[bs for bs in self.CAPTURE_CHUNK_COUNTS if bs <= self.MAX_BATCH_SIZE],
            ),
        }

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: torch.Tensor,
        valid_lengths: list[list[int]],
        valid_frames: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        num_chunks = audio_features.shape[0]
        static_inputs = {"audio_features": audio_features}
        if valid_frames is not None:
            static_inputs["valid_frames"] = valid_frames
        runner = engine_inputs.piecewise_runners.get(self.TOWER if valid_frames is None else self.MASKED_TOWER)
        if (
            runner is not None
            and audio_features.shape[-1] == self.chunk_frames
            and runner.can_run(num_chunks)
        ):
            projected = runner.run(static_inputs=static_inputs, real_bs=num_chunks).get_view("projected")
        else:
            projected = self._encode(audio_features, valid_frames)

        outputs: dict[str, NameToTensorList] = {}
        chunk = 0
        for rid, valid in zip(engine_inputs.request_ids, valid_lengths, strict=True):
            embeds = torch.cat([projected[chunk + i, :n] for i, n in enumerate(valid)], dim=0)
            chunk += len(valid)
            outputs[rid] = {"audio_embeds": [embeds]}
        return outputs

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        **kwargs,
    ) -> NameToTensorList:
        rid = engine_inputs.request_ids[0]
        return self.forward_batched(graph_walk, engine_inputs, **kwargs)[rid]


# ===================================================================
# 2. HiggsAudioLLMSubmodule (KV cache + attention + positions + sampler)
# ===================================================================


class HiggsAudioLLMSubmodule(ARNodeSubmodule):
    """Dense Qwen3 LLM.

    Dispatches on graph_walk:
      - prefill_audio: the ChatML text around the encoder's audio
        embeddings, one span into the KV cache.
      - decode:        embed the previous token, single-step decode.

    Every walk samples a token; ``postprocess`` drops it for a prefill step
    that is not the last one, so the captured graph's output keys stay the
    same whatever the step metadata says. Prefill replays a packed capture
    keyed on ``input_embeds``, while decode carries bare ``input_ids`` and
    embeds inside the graph.
    """

    PREFILL_TOKEN_BUCKETS = [64, 128, 256, 512, 1024, 2048]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]

    def __init__(self, llm: nn.Module, config: HiggsAudioModelConfig):
        super().__init__()
        self.model = llm
        self.config = config
        # the prefill samples the reply's first token
        self.token_stop = TokenStop(
            loop="decode_loop", buffer="new_token", stop_ids=frozenset(config.stop_token_ids),
            sampler=SAMPLER, walk="decode", prefill_walks=frozenset({"prefill_audio"}),
        )
        # decode reads its input ids off the sampler's slot masters
        self.device_loopback = DeviceLoopback(
            walk="decode", signal="text_inputs", sampler=SAMPLER, buffer="new_token",
        )

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        return [
            BatchedAcceleratorGraphConfig(
                # max-autotune until the default compile level (mstar serve -O) is measured on this model
                compile_mode="max-autotune-no-cudagraphs",
                capture_graph_walk="decode",
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device),
                    input_seq_len=1,
                ),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
            ),
            PackedAcceleratorGraphConfig(
                # max-autotune until the default compile level (mstar serve -O) is measured on this model
                compile_mode="max-autotune-no-cudagraphs",
                capture_graph_walk="prefill_audio",
                make_node_input=lambda n: ARNodeInputs(
                    input_seq_len=n,
                    input_embeds=torch.zeros(
                        (n, self.config.hidden_size),
                        device=device, dtype=torch.bfloat16,
                    ),
                ),
                capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
                capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
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
        device = self.get_device()

        if (row := self.uniform_row_inputs(graph_walk, has_loopback)) is not None:
            # the token is read off the sampler's slot master in `preprocess`
            return row

        if graph_walk == "prefill_audio":
            # the audio between the ChatML text around it, prefilled as one span
            audio = inputs["audio_embeds"][0].to(device)
            pre, post = (
                self.model.embed_tokens(inputs[name][0].to(device).reshape(-1))
                if inputs.get(name) else None
                for name in ("text_pre", "text_post")
            )
            dtype = (pre if pre is not None else audio).dtype
            embeds = torch.cat([t.to(dtype) for t in (pre, audio, post) if t is not None], dim=0)
            return ARNodeInputs(
                input_seq_len=embeds.shape[0],
                input_embeds=embeds,
            )

        # decode: embedding happens inside the captured graph; a decode row
        # is one id, so staging it costs 8 bytes instead of a hidden vector
        token_ids = inputs["text_inputs"][0].to(device).reshape(-1)
        return ARNodeInputs(input_seq_len=1, input_ids=token_ids)

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        per_request_info: Mapping[int, CurrentForwardPassInfo] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        kept_rids = None
        if graph_walk != "decode":
            # only the last prefill's token survives `postprocess`
            info = per_request_info or {}
            kept_rids = frozenset(
                rid for rid in request_ids
                if rid in info and info[rid].step_metadata.get("is_last_prefill", False)
            )
        return SubmoduleStep(
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
                SAMPLER: SamplerStep(apply_penalty=False, kept_rids=kept_rids),
                ROPE: PositionStep(),
            },
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        if graph_walk == "decode":
            if all(inp.input_ids is not None for inp in inputs):
                return {"input_ids": torch.cat([inp.input_ids for inp in inputs])}
            # device loop-back: each row's last sampled token, read by slot
            return self.loopback_inputs(engine_inputs, inputs)
        return {
            "input_embeds": torch.cat(
                [inp.input_embeds for inp in inputs], dim=0
            ),
        }

    def _forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None,
        input_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the backbone and sample one token per request."""
        sampler: SamplerResource = engine_inputs.resources[SAMPLER]
        attn: AttentionManager = engine_inputs.resources[ATTN]

        if input_ids is None and input_embeds is None:
            input_ids = sampler.loopback_tokens(engine_inputs.request_ids)
        if input_embeds is None:
            input_embeds = self.model.embed_tokens(input_ids)
        hidden = self.model(input_embeds=input_embeds, label="main")

        if graph_walk != "decode":
            # packed prefill: one hidden per request, at its last token
            hidden = attn.select_last_hidden(hidden)

        logits = self.model.lm_head(hidden)
        return sampler.sample(engine_inputs.request_ids, logits=logits)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        return {
            "new_token": [self._forward(
                graph_walk=graph_walk,
                engine_inputs=engine_inputs,
                input_ids=input_ids,
                input_embeds=input_embeds,
            )]
        }

    def can_batch(
        self, batch: ExecutingBatch,
        model_inputs: list[NodeInputs],
    ) -> bool:
        return True

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> BatchedModelOutput:
        new_tokens = self._forward(
            graph_walk=graph_walk,
            engine_inputs=engine_inputs,
            input_ids=input_ids,
            input_embeds=input_embeds,
        )
        # row i is request i: one clone and one device-to-host copy per step
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
        # Prefill samples on every step so the captured graph's outputs are
        # fixed; only the last prefill's token is a real one.
        if request_info.graph_walk != "decode" and \
                not request_info.step_metadata.get("is_last_prefill", False):
            outputs.pop("new_token", None)
            return
        if "new_token" not in outputs:
            return
        outputs["text_inputs"] = outputs["new_token"]

