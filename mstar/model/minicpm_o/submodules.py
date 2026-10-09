"""MiniCPM-o's node submodules: the Qwen3 LLM and the two encoders.

The LLM's prompt is the whole rendered chat, placeholders included; a
multimodal prefill embeds it and overwrites each placeholder row with the
encoder output that belongs there, at positions ``process_prompt`` read off
the prompt. RoPE is plain 1D over that sequence, so the position resource
advances by the token count with no special casing.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.cuda_graph_config import (
    BatchedCudaGraphConfig,
    CudaGraphConfig,
    PackedCudaGraphConfig,
    PiecewiseBatchedConfig,
    PiecewiseCallInputs,
    PiecewiseCaptureShape,
    PiecewiseCudaGraphConfig,
    PiecewisePackedConfig,
)
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import (
    AttentionStep,
    KVStep,
    PositionStep,
    RaggedCrossAttentionStep,
    SamplerStep,
    Segment,
    SlotLease,
    SubmoduleStep,
)
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.recurrent.config import RecurrentStep
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.components.qwen3_lm import Qwen3DenseLM
from mstar.model.minicpm_o.components.audio import MiniCPMOAudio
from mstar.model.minicpm_o.components.tts import next_history, windowed_frequency_penalty
from mstar.model.minicpm_o.components.vision import MiniCPMOVision, slice_layout
from mstar.model.minicpm_o.config import (
    AUDIO_ATTN,
    LLM_ATTN,
    LLM_KV,
    LLM_POS,
    LLM_SAMPLER,
    PATCHES,
    QUERIES,
    RESAMPLER_ATTN,
    T2W_STATE,
    TTS_ATTN,
    TTS_KV,
    TTS_POS,
    TTS_SAMPLER,
    VISION_ATTN,
    MiniCPMOConfig,
)
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    HostRows,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)

logger = logging.getLogger(__name__)

DECODE = "decode"
DECODE_LOOP = "decode_loop"

# encoder output name -> the prompt positions it fills
_MM_FILLS = {"vision_embeds": "image_positions", "audio_embeds": "audio_positions"}


class LLMSubmodule(ARNodeSubmodule):
    PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024, 2048]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]

    def __init__(self, model: Qwen3DenseLM, config: MiniCPMOConfig):
        super().__init__()
        self.model = model
        self.config = config

    def get_cuda_graph_configs(self, device: torch.device, tp_world_size: int = 1) -> list[CudaGraphConfig]:
        """Decode, and one prefill capture every prefill walk replays: ids are
        embedded in `preprocess`, so the forward always takes embeddings."""
        def dummy(n: int) -> ARNodeInputs:
            return ARNodeInputs(input_ids=torch.zeros(n, dtype=torch.long, device=device), input_seq_len=n)

        return [
            BatchedCudaGraphConfig(
                capture_graph_walk=DECODE,
                single_request_inputs=dummy(1),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
            ),
            PackedCudaGraphConfig(
                capture_graph_walk="prefill_text",
                replay_graph_walks=["prefill_image", "prefill_audio", "prefill_omni"],
                capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
                make_node_input=dummy,
                capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
            ),
        ]

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def _prompt_embeds(self, inputs: NameToTensorList) -> torch.Tensor:
        """The prompt's embeddings with each placeholder row overwritten by the
        encoder output for it, in prompt order."""
        device = self.get_device()
        embeds = self.model.embed(inputs["text_inputs"][0].to(device))
        for name, positions in _MM_FILLS.items():
            if name not in inputs:
                continue
            index = inputs[positions][0].to(device)
            values = inputs[name][0].to(device=device, dtype=embeds.dtype)
            if values.shape[0] != index.shape[0]:
                raise ValueError(
                    f"{name} has {values.shape[0]} rows for {index.shape[0]} placeholder "
                    "tokens; the prompt and the encoder disagree on the attachment's length"
                )
            embeds = embeds.index_copy(0, index, values)
        return embeds

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> ARNodeInputs:
        ids = inputs["text_inputs"][0]
        if graph_walk == DECODE or not any(name in inputs for name in _MM_FILLS):
            return ARNodeInputs(input_seq_len=ids.shape[0], input_ids=ids)
        embeds = self._prompt_embeds(inputs)
        return ARNodeInputs(input_seq_len=embeds.shape[0], input_embeds=embeds)

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, Any]:
        device = self.get_device()
        if all(inp.input_ids is not None for inp in inputs):
            ids = torch.cat([inp.input_ids for inp in inputs]).to(device)
            return {"input_embeds": self.model.embed(ids)}
        # the prefill walks share one capture, so a batch can mix text rows
        # (ids) with multimodal ones (embeddings)
        return {"input_embeds": torch.cat([
            inp.input_embeds if inp.input_embeds is not None else self.model.embed(inp.input_ids.to(device))
            for inp in inputs
        ])}

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        **kwargs,
    ) -> SubmoduleStep:
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={
                LLM_KV: KVStep(),
                LLM_ATTN: AttentionStep(causal=True),
                LLM_POS: PositionStep(),
                # The prompt is not tracked: upstream generates from embeddings,
                # so its penalty sees only generated tokens.
                LLM_SAMPLER: SamplerStep(apply_penalty=True),
            },
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_embeds: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """``new_token``, and on decode ``tts_hidden``: the post-norm hidden of
        each row's input token, which a spoken reply's TTS reads."""
        attn: AttentionManager = engine_inputs.resources[LLM_ATTN]
        sampler: SamplerResource = engine_inputs.resources[LLM_SAMPLER]
        hidden = self.model(input_embeds, label="main")
        out = {}
        if graph_walk == DECODE:
            out["tts_hidden"] = hidden
        else:
            hidden = attn.select_last_hidden(hidden)
        out["new_token"] = sampler.sample(engine_inputs.request_ids, logits=self.model.logits(hidden))
        return out

    def forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_embeds: torch.Tensor, **kwargs,
    ) -> NameToTensorList:
        return {k: [v] for k, v in self._forward(graph_walk, engine_inputs, input_embeds).items()}

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_embeds: torch.Tensor, **kwargs,
    ) -> BatchedModelOutput:
        out = self._forward(graph_walk, engine_inputs, input_embeds)
        return BatchedModelOutput(
            row_outputs=out,
            check_stop_buffers={"new_token": out["new_token"]},
        )

    # ------------------------------------------------------------------
    # Post-step
    # ------------------------------------------------------------------

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        inputs: NodeInputs | None = None,
        **kwargs,
    ):
        # Rebind, not copy: the decode loop routes on `text_inputs`. EOS is
        # tested in `check_stop`, off the GPU thread.
        if "new_token" in outputs:
            outputs["text_inputs"] = outputs["new_token"]
        if "tts_hidden" not in outputs:
            return
        if request_info.step_metadata.get("audio_output", False):
            # the decode loop accumulates (input token, its hidden) pairs: the
            # reply's TTS condition, available whole when the loop ends
            outputs["tts_ids"] = [inputs.input_ids]
        else:
            outputs.pop("tts_hidden")

    def _stops(self, token: int, info: CurrentForwardPassInfo) -> bool:
        hit_eos = (
            not info.resource_configs[LLM_SAMPLER].ignore_eos
            and token in self.config.stop_token_ids
        )
        out_of_budget = info.dynamic_loop_iter_counts.get(DECODE_LOOP, 0) + 1 >= info.max_tokens
        return hit_eos or out_of_budget

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "new_token" not in outputs:
            return set()
        return {DECODE_LOOP} if self._stops(outputs["new_token"][0].item(), request_info) else set()

    def check_stop_batched(
        self,
        request_ids: list[str],
        request_infos: dict[str, CurrentForwardPassInfo],
        host_rows: HostRows,
    ) -> dict[str, set[str]] | None:
        tokens = host_rows.buffers.get("new_token")
        if not torch.is_tensor(tokens) or tokens.dim() == 0:
            return None
        values = tokens.reshape(tokens.shape[0], -1)[:, 0].tolist()
        row_of = {rid: i for i, rid in enumerate(host_rows.request_ids)}
        stops: dict[str, set[str]] = {}
        for rid in request_ids:
            i = row_of.get(rid)
            if i is None or i >= len(values):
                continue
            if self._stops(values[i], request_infos[rid]):
                stops[rid] = {DECODE_LOOP}
        return stops


class VisionEncoderSubmodule(NodeSubmodule):
    """Every slice of every image in a request, packed into one forward: the
    navit SigLIP attends within each slice, then the resampler turns each slice
    into 64 tokens. Requests pack too; their slices stay apart by segment."""

    # Every request brings its own slice count and patch grid, which a compiled
    # forward would recompile for; it runs once a request.
    disable_torch_compile = True

    def __init__(self, model: MiniCPMOVision, config: MiniCPMOConfig):
        super().__init__()
        self.model = model
        self.config = config

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> NodeInputs:
        tgt_sizes = [
            (int(h), int(w)) for t in inputs["image_tgt_sizes"] for h, w in t.reshape(-1, 2).tolist()
        ]
        layout = slice_layout(tgt_sizes, self.config.vision)
        patches = torch.cat(inputs["pixel_values"])
        return NodeInputs(
            input_seq_len=patches.shape[0],
            tensor_inputs={
                "patches": patches,
                "position_ids": layout.position_ids,
                "grid_coords": layout.grid_coords,
            },
            kwargs={"seq_lengths": layout.seq_lengths},
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
        **kwargs,
    ) -> SubmoduleStep:
        patches, queries = [], []
        for rid, inp in zip(request_ids, inputs, strict=True):
            for span in inp.kwargs["seq_lengths"]:
                patches.append(Segment(request_id=rid, label=PATCHES, span=span))
                queries.append(Segment(request_id=rid, label=QUERIES, span=self.config.resampler.num_queries))
        return SubmoduleStep(
            steps={
                VISION_ATTN: AttentionStep(segments=tuple(patches), causal=False),
                RESAMPLER_ATTN: RaggedCrossAttentionStep(
                    segments=tuple(patches + queries), pairs=((QUERIES, PATCHES),),
                ),
            },
        )

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        device = self.get_device()

        def cat(name):
            return torch.cat([inp.tensor_inputs[name] for inp in inputs]).to(device)

        return {
            "patches": cat("patches"),
            "position_ids": cat("position_ids"),
            "grid_coords": cat("grid_coords"),
            "slices_per_request": [len(inp.kwargs["seq_lengths"]) for inp in inputs],
        }

    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs) -> NameToTensorList:
        return self.forward_batched(graph_walk, engine_inputs, **kwargs)[engine_inputs.request_ids[0]]

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        patches: torch.Tensor,
        position_ids: torch.Tensor,
        grid_coords: torch.Tensor,
        slices_per_request: list[int],
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        embeds = self.model(patches, position_ids, grid_coords, sum(slices_per_request))
        q = self.config.resampler.num_queries
        per_request = embeds.split([n * q for n in slices_per_request])
        return {
            rid: {"vision_embeds": [out]}
            for rid, out in zip(engine_inputs.request_ids, per_request, strict=True)
        }


# The Whisper block loop, as a piecewise capture region
AUDIO_BLOCK_LOOP = "audio_block_loop"


class AudioEncoderSubmodule(NodeSubmodule):
    """A request's audio pieces (30 s each at most); each piece is its own
    block-causal segment.

    The conv front end and the pooling run eagerly around a captured block
    loop (24 layers and the projector), one request per replay, bucketed by
    frame count: eager, the encoder is launch-bound (~10 ms whatever the
    clip's length). A request past the largest bucket runs eagerly.
    """

    # 50 frames a second: 5 s up to two full 30 s pieces
    BLOCK_LOOP_FRAME_BUCKETS = [256, 512, 1024, 1536, 3072]

    # Every clip has its own length, which a compiled forward would recompile
    # for; the captured block loop is what makes it fast.
    disable_torch_compile = True

    def __init__(self, model: MiniCPMOAudio, config: MiniCPMOConfig):
        super().__init__()
        self.model = model
        self.config = config

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> NodeInputs:
        mel_lens = [int(n) for t in inputs["audio_feature_lens"] for n in t.reshape(-1).tolist()]
        features = torch.cat(inputs["audio_features"], dim=-1)
        frames = [self.config.audio.conv_frames(n) for n in mel_lens]
        return NodeInputs(
            input_seq_len=sum(frames),
            tensor_inputs={"features": features},
            kwargs={"mel_lens": mel_lens, "frames": frames},
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep | None:
        # leased, the captured block loop plans its own attention per replay
        if (piecewise_leases or {}).get(AUDIO_BLOCK_LOOP):
            return None
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=n)
                for rid, inp in zip(request_ids, inputs, strict=True)
                for n in inp.kwargs["frames"]
            ],
            steps={AUDIO_ATTN: AttentionStep(causal=False)},
        )

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def max_batch_size(self, graph_walk: str) -> int:
        # one request a step, so it can take the captured block loop
        return 1

    def get_piecewise_cuda_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype, tp_world_size: int = 1, **kwargs,
    ) -> dict[str, PiecewiseCudaGraphConfig]:
        d_model = self.config.audio.d_model

        def make_static_inputs(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
            return {"hidden": torch.zeros(shape.total_tokens, d_model, dtype=autocast_dtype, device=device)}

        def declare_step(request_ids: list[str], seq_lens: list[int]) -> SubmoduleStep:
            (rid,) = request_ids
            return SubmoduleStep(
                segments=[Segment(request_id=rid, label="main", span=n) for n in seq_lens],
                steps={AUDIO_ATTN: AttentionStep(causal=False)},
            )

        return {
            AUDIO_BLOCK_LOOP: PiecewisePackedConfig(
                capture_fn=lambda inp: {"hidden": self.model.encode(inp.static_inputs["hidden"])},
                make_static_inputs=make_static_inputs,
                declare_step=declare_step,
                lease_before_step=True,
                total_tokens=self.BLOCK_LOOP_FRAME_BUCKETS,
                capture_batch_sizes=[1],
            )
        }

    @torch.compiler.disable
    def _replay_block_loop(
        self, engine_inputs: ModelInputsFromEngine, hidden: torch.Tensor, frames: list[int],
    ) -> torch.Tensor | None:
        runner = engine_inputs.piecewise_runners.get(AUDIO_BLOCK_LOOP)
        if runner is None or not runner.can_run(len(engine_inputs.request_ids), int(hidden.shape[0])):
            return None
        return runner.run(
            static_inputs={"hidden": hidden},
            request_ids=[engine_inputs.request_ids[0]],
            seq_lens=list(frames),
            real_bs=1,
        ).get_view("hidden")

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        device = self.get_device()
        pieces: list[torch.Tensor] = []
        for inp in inputs:
            pieces += list(inp.tensor_inputs["features"].to(device).split(inp.kwargs["mel_lens"], dim=-1))
        return {
            "pieces": pieces,
            "frames": [n for inp in inputs for n in inp.kwargs["frames"]],
            "tokens_per_request": [
                sum(self.config.audio.pooled_tokens(n) for n in inp.kwargs["mel_lens"]) for inp in inputs
            ],
        }

    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs) -> NameToTensorList:
        return self.forward_batched(graph_walk, engine_inputs, **kwargs)[engine_inputs.request_ids[0]]

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        pieces: list[torch.Tensor],
        frames: list[int],
        tokens_per_request: list[int],
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        hidden = self.model.apm.frontend(pieces)
        encoded = self._replay_block_loop(engine_inputs, hidden, frames)
        if encoded is None:
            encoded = self.model.encode(hidden)
        embeds = self.model.pool(encoded, frames)
        return {
            rid: {"audio_embeds": [out]}
            for rid, out in zip(engine_inputs.request_ids, embeds.split(tokens_per_request), strict=True)
        }


TTS_PREFILL = "tts_prefill"
TTS_DECODE = "tts_decode"
TTS_DECODE_LOOP = "tts_decode_loop"


class TTSSubmodule(ARNodeSubmodule):
    # A reply's text plus two: at most a few hundred rows, once per request
    PREFILL_TOKEN_BUCKETS = [64, 128, 256, 512]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4]
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]

    """The speech-token LM. ``tts_prefill`` reads the whole reply (upstream's
    ``chat`` speaks only once the text is done) and samples the first code;
    ``tts_decode`` feeds each code back until the EOS code.

    Upstream's per-step logits processing, in its order:
      * the window-16 frequency penalty (skipped for the first code), applied
        here as a pure tensor op over the last codes;
      * EOS masked until ``min_new_tokens`` codes exist.
    Both read ``tts_history``, which rides the loop edge instead of living in
    host state: the last ``penalty_window`` codes (-1 for none yet), then the
    number of codes generated so far. Nothing per row crosses from the host.
    then the sampler resource: temperature, top-k/top-p, draw.
    """

    def __init__(self, model, config: MiniCPMOConfig, sampling):
        super().__init__()
        self.model = model
        self.config = config
        self.config_tts = model.config
        self.sampling = sampling

    def get_cuda_graph_configs(self, device: torch.device, tp_world_size: int = 1) -> list[CudaGraphConfig]:
        hidden = self.config_tts.hidden_size

        def extras() -> dict[str, torch.Tensor]:
            return {"history": self._empty_history(device)}

        def prefill_dummy(n: int) -> ARNodeInputs:
            return ARNodeInputs(
                input_embeds=torch.zeros(n, hidden, device=device, dtype=self.model.emb_code.weight.dtype),
                input_seq_len=n, tensor_inputs=extras(),
            )

        return [
            BatchedCudaGraphConfig(
                capture_graph_walk=TTS_DECODE,
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device),
                    input_seq_len=1, tensor_inputs=extras(),
                ),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
            ),
            PackedCudaGraphConfig(
                capture_graph_walk=TTS_PREFILL,
                capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
                make_node_input=prefill_dummy,
                capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
            ),
        ]

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> ARNodeInputs:
        device = self.get_device()
        if graph_walk == TTS_PREFILL:
            ids = torch.cat([t.reshape(-1) for t in inputs["tts_ids"]]).to(device)
            hidden = torch.cat([t.reshape(-1, t.shape[-1]) for t in inputs["tts_hidden"]]).to(device)
            embeds = self.model.condition(ids, hidden)
            history = self._empty_history(device)
        else:
            embeds = None
            history = inputs["tts_history"][0].reshape(-1).to(device)
        return ARNodeInputs(
            input_seq_len=1 if embeds is None else embeds.shape[0],
            input_ids=None if embeds is not None else inputs["tts_code"][0].reshape(1).to(device),
            input_embeds=embeds,
            tensor_inputs={"history": history},
        )

    def _empty_history(self, device: torch.device) -> torch.Tensor:
        """No codes yet: ``penalty_window`` empty slots and a count of 0. Cloned
        from a cached template, so a prefill launches one copy and no H2D."""
        template = getattr(self, "_history_template", None)
        if template is None or template.device != device:
            template = torch.full((self.sampling.penalty_window + 1,), -1, dtype=torch.long)
            template[-1] = 0
            self._history_template = template = template.to(device)
        return template.clone()

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, Any]:
        if inputs[0].input_ids is not None:
            embeds = self.model.emb_code(torch.cat([inp.input_ids for inp in inputs]))
        else:
            embeds = torch.cat([inp.input_embeds for inp in inputs])
        return {
            "input_embeds": embeds,
            "history": torch.stack([inp.tensor_inputs["history"] for inp in inputs]),
        }

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        **kwargs,
    ) -> SubmoduleStep:
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={
                TTS_KV: KVStep(),
                TTS_ATTN: AttentionStep(causal=True),
                TTS_POS: PositionStep(),
                # the window penalty is applied before sampling, here
                TTS_SAMPLER: SamplerStep(apply_penalty=False),
            },
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor,
        history: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        attn: AttentionManager = engine_inputs.resources[TTS_ATTN]
        sampler: SamplerResource = engine_inputs.resources[TTS_SAMPLER]
        hidden = self.model.model(input_embeds, label="main")
        if graph_walk != TTS_DECODE:
            hidden = attn.select_last_hidden(hidden)
        logits = self.model.logits(hidden)
        recent, generated = history[:, :-1], history[:, -1:]
        if graph_walk == TTS_DECODE:
            logits = windowed_frequency_penalty(logits, recent, self.sampling.repetition_penalty)
        eos = self.model.config.eos_code
        suppress_eos = generated[:, 0] < self.sampling.min_new_tokens
        logits[:, eos] = torch.where(suppress_eos, float("-inf"), logits[:, eos])
        code = sampler.sample(engine_inputs.request_ids, logits=logits)
        return {
            "tts_code": code,
            "tts_history": next_history(history, code),
        }

    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs) -> NameToTensorList:
        return {k: [v] for k, v in self._forward(graph_walk, engine_inputs, **kwargs).items()}

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def forward_batched(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs) -> BatchedModelOutput:
        out = self._forward(graph_walk, engine_inputs, **kwargs)
        return BatchedModelOutput(row_outputs=out, check_stop_buffers={"tts_code": out["tts_code"]})

    # ------------------------------------------------------------------
    # Post-step
    # ------------------------------------------------------------------

    def _stops(self, code: int, info: CurrentForwardPassInfo) -> bool:
        generated = info.dynamic_loop_iter_counts.get(TTS_DECODE_LOOP, 0) + 2
        return code == self.model.config.eos_code or generated >= self.sampling.max_new_tokens

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if request_info.graph_walk != TTS_DECODE or "tts_code" not in outputs:
            return set()
        return {TTS_DECODE_LOOP} if self._stops(outputs["tts_code"][0].item(), request_info) else set()

    def check_stop_batched(
        self,
        request_ids: list[str],
        request_infos: dict[str, CurrentForwardPassInfo],
        host_rows: HostRows,
    ) -> dict[str, set[str]] | None:
        codes = host_rows.buffers.get("tts_code")
        if not torch.is_tensor(codes) or codes.dim() == 0:
            return None
        values = codes.reshape(codes.shape[0], -1)[:, 0].tolist()
        row_of = {rid: i for i, rid in enumerate(host_rows.request_ids)}
        stops: dict[str, set[str]] = {}
        for rid in request_ids:
            i = row_of.get(rid)
            info = request_infos[rid]
            if i is None or i >= len(values) or info.graph_walk != TTS_DECODE:
                continue
            if self._stops(values[i], info):
                stops[rid] = {TTS_DECODE_LOOP}
        return stops


class Token2WavSubmodule(NodeSubmodule):
    """Speech codes -> 24 kHz audio, one stream window per step: upstream's
    ``Token2wav.stream``, fed the way its ``streaming_generate`` feeds it (three
    silence codes first, 28-code windows advancing by 25, a last flush).

    The windows are the ``LeftContextChunkPolicy(25, 3)`` stream from the TTS;
    the first window gets the silence prepended here. A request's caches live
    in its slot of the ``T2W_STATE`` pool (``RecurrentStatePool``), which this
    node uses as its backend: it works on the slot's own views, initialised from
    the voice's prepared state on the first window. Their host-side lengths ride
    in the request's state.

    Float32, batched straight on the pool. A voice's full non-last windows enter
    the caches at one of three lengths (the first window, the second, and every
    later one), so rows of one (voice, phase) batch and replay a piecewise
    capture (``t2w/<voice>/<phase>``), bucketed by batch size; the inverse STFT
    and cross-fade stay eager. A last window after the steady phase has one
    of 25 lengths, each its own one-row capture (``t2w/<voice>/last<n>``);
    other last windows (replies under three windows) run eagerly, batched by
    length. The per-request ``Token2Wav.stream`` path is bit-exact against
    the reference at full fp32; this batched path attends the DiT's cache
    in another order and runs its blocks on fused TF32 kernels.
    """

    WINDOW_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]
    # a request ends once, so last windows of one length rarely meet in a pass
    LAST_CAPTURE_BATCH_SIZES = [1]

    disable_torch_compile = True
    disable_autocast = True

    def __init__(self, model, voices: dict, eos_code: int):
        super().__init__()
        self.model = model
        self.voices = voices
        self.eos_code = eos_code

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        is_final_stream_chunk: bool = False,
        **kwargs: Any,
    ) -> NodeInputs | None:
        from mstar.model.minicpm_o.components.token2wav import LEAD_SILENCE, SILENCE_CODE

        codes = inputs["tts_code"][0].reshape(-1).tolist() if inputs.get("tts_code") else []
        # the TTS's stop code is streamed before its stop is seen; it is not speech
        codes = [c for c in codes if c != self.eos_code]
        if not codes:
            # a text-only reply's stream closes with one empty chunk
            return None
        # keyed by the worker handle, as the forward and the engine's cleanup are
        state = self.request_state(fwd_info.rid_handle)
        if not state.get("started", False):
            codes = [SILENCE_CODE] * LEAD_SILENCE + codes
        return NodeInputs(
            tensor_inputs={"codes": torch.tensor([codes], dtype=torch.int32)},
            kwargs={"last": is_final_stream_chunk, "voice": fwd_info.step_metadata.get("voice")},
        )

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[NodeInputs], **kwargs,
    ) -> SubmoduleStep:
        return SubmoduleStep(
            segments=[Segment(request_id=rid, label="main", span=1) for rid in request_ids],
            steps={T2W_STATE: RecurrentStep()},
        )

    @staticmethod
    def window_region(voice: str, phase: int) -> str:
        return f"t2w/{voice}/{phase}"

    @staticmethod
    def last_window_region(voice: str, num_tokens: int) -> str:
        return f"t2w/{voice}/last{num_tokens}"

    def get_piecewise_cuda_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype, tp_world_size: int = 1, **kwargs,
    ) -> dict[str, PiecewiseCudaGraphConfig]:
        from mstar.model.minicpm_o.components.token2wav import (
            LAST_WINDOW_TOKENS,
            PHASE_FIRST,
            PHASE_SECOND,
            PHASE_STEADY,
            WINDOW,
            phase_lengths,
        )

        def static_inputs(num_tokens: int):
            def make(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
                return {"tokens": torch.zeros(shape.bs, num_tokens, dtype=torch.int32, device=device)}
            return make

        def declare_step(request_ids: list[str], seq_lens: list[int]) -> SubmoduleStep:
            return SubmoduleStep(
                segments=[Segment(request_id=rid, label="main", span=1) for rid in request_ids],
                steps={T2W_STATE: RecurrentStep()},
            )

        def region(voice, phase, last):
            lengths = phase_lengths(voice.initial.prompt_frames, phase)

            def capture(call: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
                pool = call.resources[T2W_STATE]
                tokens = call.static_inputs["tokens"]
                return self.model.window_spectrum(
                    {name: pool.block(name, 0) for name in pool.config.blocks},
                    pool.addressing("main").slot_indices[: tokens.shape[0]],
                    voice, lengths, tokens, last,
                )
            return capture

        configs = {}
        for name, voice in self.voices.items():
            for phase in (PHASE_FIRST, PHASE_SECOND, PHASE_STEADY):
                configs[self.window_region(name, phase)] = PiecewiseBatchedConfig(
                    capture_fn=region(voice, phase, last=False),
                    make_static_inputs=static_inputs(WINDOW),
                    declare_step=declare_step,
                    seq_len=WINDOW,
                    capture_batch_sizes=self.WINDOW_CAPTURE_BATCH_SIZES,
                )
            # a reply's last window, entering after at least two full ones
            for n in LAST_WINDOW_TOKENS:
                configs[self.last_window_region(name, n)] = PiecewiseBatchedConfig(
                    capture_fn=region(voice, PHASE_STEADY, last=True),
                    make_static_inputs=static_inputs(n),
                    declare_step=declare_step,
                    seq_len=n,
                    capture_batch_sizes=self.LAST_CAPTURE_BATCH_SIZES,
                )
        return configs

    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs) -> NameToTensorList:
        return self.forward_batched(graph_walk, engine_inputs, **kwargs)[engine_inputs.request_ids[0]]

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        device = self.get_device()
        return {"rows": [
            (inp.tensor_inputs["codes"].to(device), inp.kwargs["last"], inp.kwargs["voice"]) for inp in inputs
        ]}

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, rows: list, **kwargs,
    ) -> dict[str, NameToTensorList]:
        from mstar.model.minicpm_o.components.token2wav import (
            PHASE_STEADY,
            entering_phase,
            state_lengths,
            window_batch_key,
        )

        # Rows of one voice with the same window_batch_key run as one batch: a
        # captured phase when one fits, else the same computation eagerly.
        # Each row advances its own lengths.
        groups: dict[tuple, list[tuple[str, torch.Tensor, dict[str, int]]]] = {}
        for rid, (codes, last, voice_name) in zip(engine_inputs.request_ids, rows, strict=True):
            req = self.request_state(rid)
            lengths = req["lengths"] if "lengths" in req else state_lengths(self.voices[voice_name].initial)
            key = (voice_name, *window_batch_key(lengths, codes.shape[1], last))
            groups.setdefault(key, []).append((rid, codes, lengths))
        pool = engine_inputs.resources[T2W_STATE]
        out = {}
        for key, group in groups.items():
            voice_name, phase, shared, num_tokens, last = key
            shared = dict(shared)
            if phase is not None:
                region, cap = self.window_region(voice_name, phase), self.WINDOW_CAPTURE_BATCH_SIZES[-1]
            elif last and entering_phase(shared) == PHASE_STEADY:
                region, cap = self.last_window_region(voice_name, num_tokens), self.LAST_CAPTURE_BATCH_SIZES[-1]
            else:
                region, cap = None, len(group)
            for i in range(0, len(group), cap):
                out.update(self._windows(engine_inputs, pool, voice_name, region, shared, last, group[i:i + cap]))
        return out

    @torch.compiler.disable
    def _windows(self, engine_inputs, pool, voice_name, region, lengths, last, group) -> dict[str, NameToTensorList]:
        from mstar.model.minicpm_o.components.token2wav import lengths_after

        rids = [rid for rid, _, _ in group]
        tokens = torch.cat([codes for _, codes, _ in group])
        blocks = {name: pool.block(name, 0) for name in pool.config.blocks}
        slots = torch.tensor([pool.slot_index(rid) for rid in rids], dtype=torch.long, device=tokens.device)
        voice = self.voices[voice_name]
        runner = None if region is None else engine_inputs.piecewise_runners.get(region)
        if runner is not None and runner.can_run(len(rids)):
            spectrum = runner.run(static_inputs={"tokens": tokens}, request_ids=rids, real_bs=len(rids))
            wav = self.model.window_finish(
                blocks, slots, spectrum.get_view("magnitude"), spectrum.get_view("phase"), lengths, last,
            )
        else:
            wav = self.model.stream_batched(blocks, slots, voice, lengths, tokens, last)
        out = {}
        for i, (rid, _, row_lengths) in enumerate(group):
            req = self.request_state(rid)
            req.add("lengths", lengths_after(row_lengths, tokens.shape[1], last))
            req.add("started", True)
            out[rid] = {"audio_chunk": [wav[i]]}
        return out
