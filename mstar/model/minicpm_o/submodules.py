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
from typing import TYPE_CHECKING, Any, NamedTuple

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
from mstar.engine.resources.kv.bounded import BoundedKVStep, StreamPosition
from mstar.engine.resources.kv.bounded.manager import BoundedPlan
from mstar.engine.resources.recurrent.config import RecurrentStep
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.components.qwen3_lm import Qwen3DenseLM
from mstar.model.minicpm_o.components.audio import MiniCPMOAudio
from mstar.model.minicpm_o.components.token2wav import (
    CACHE_FAMILIES,
    LAST_WINDOW_TOKENS,
    LEAD_SILENCE,
    SILENCE_CODE,
    WINDOW,
    AttentionCache,
    SlotRows,
    VoiceBank,
    WindowCaches,
    voice_entry,
    window_frames,
    window_tokens,
)
from mstar.model.minicpm_o.components.token2wav_flow import MEL_BINS as MEL_BINS_T2W
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
    T2W_KV,
    T2W_STATE,
    T2W_VOICE,
    T2W_VOICES,
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

if TYPE_CHECKING:
    from mstar.engine.cuda_graph_runner import PiecewiseCudaGraphRunner

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
    """The speech-token LM. ``tts_prefill`` reads the whole reply (upstream's
    ``chat`` speaks only once the text is done) and samples the first code;
    ``tts_decode`` feeds each code back until the EOS code.

    Sampling is the sampler resource's, set up as upstream's TTS sampler
    (``get_request_resource_configs``): temperature, a frequency penalty over
    the last 16 codes, top-p then top-k, and EOS barred until 50 codes exist.
    Upstream samples the first code with temperature and the EOS floor only, so
    the prefill walk declares its sampler step without filters.
    """

    # A reply's text plus two: at most a few hundred rows, once per request
    PREFILL_TOKEN_BUCKETS = [64, 128, 256, 512]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4]
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]

    def __init__(self, model, config: MiniCPMOConfig, sampling):
        super().__init__()
        self.model = model
        self.config = config
        self.config_tts = model.config
        self.sampling = sampling

    def get_cuda_graph_configs(self, device: torch.device, tp_world_size: int = 1) -> list[CudaGraphConfig]:
        hidden = self.config_tts.hidden_size

        def prefill_dummy(n: int) -> ARNodeInputs:
            return ARNodeInputs(
                input_embeds=torch.zeros(n, hidden, device=device, dtype=self.model.emb_code.weight.dtype),
                input_seq_len=n,
            )

        return [
            BatchedCudaGraphConfig(
                capture_graph_walk=TTS_DECODE,
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device), input_seq_len=1,
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
            return ARNodeInputs(input_seq_len=embeds.shape[0], input_embeds=embeds)
        return ARNodeInputs(input_seq_len=1, input_ids=inputs["tts_code"][0].reshape(1).to(device))

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, Any]:
        if inputs[0].input_ids is not None:
            return {"input_embeds": self.model.emb_code(torch.cat([inp.input_ids for inp in inputs]))}
        return {"input_embeds": torch.cat([inp.input_embeds for inp in inputs])}

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
                # the windowed penalty is the sampler's own history, not the
                # presence mask `apply_penalty` drives
                TTS_SAMPLER: SamplerStep(apply_penalty=False, apply_filters=graph_walk != TTS_PREFILL),
            },
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_embeds: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        attn: AttentionManager = engine_inputs.resources[TTS_ATTN]
        sampler: SamplerResource = engine_inputs.resources[TTS_SAMPLER]
        hidden = self.model.model(input_embeds, label="main")
        if graph_walk != TTS_DECODE:
            hidden = attn.select_last_hidden(hidden)
        return {"tts_code": sampler.sample(engine_inputs.request_ids, logits=self.model.logits(hidden))}

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
    the first window gets the silence prepended here. A request's attention caches
    are on the ``T2W_KV`` bounded KV resources, which start from its voice's
    prepared caches; its conv caches and HiFT tails are its slot of ``T2W_STATE``
    (``RecurrentStatePool``). Its stream position (tokens written) rides in the
    request's state.

    Float32. Each window runs in two stages (``forward_batched``), each a piecewise
    capture bucketed by batch size: the flow (``t2w/<voice>/flow28``), for windows
    at any position, and HiFT, separately for a request's first window, whose
    input is shorter. A last window has one of a few lengths, each a one-row
    capture; other shapes run eagerly. The inverse STFT and cross-fade stay eager.
    The per-request ``Token2Wav.stream`` path is bit-exact against the reference at
    full fp32; this one attends the DiT's cache in another order and runs on fused
    TF32 kernels.
    """

    WINDOW_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]
    # a request ends once, so last windows of one length rarely meet in a pass
    LAST_CAPTURE_BATCH_SIZES = [1]

    disable_torch_compile = True
    disable_autocast = True

    def __init__(self, model, voices: dict, eos_code: int, custom_voices: bool = False):
        """``voices`` are the model's fixed ones; with ``custom_voices`` a request may also
        bring its own (a 16 kHz clip, the ``voice_audio`` input of the ``T2W_VOICE`` walk),
        prepared into the ``T2W_VOICES`` pool before its first window."""
        super().__init__()
        self.model = model
        self.eos_code = eos_code
        self.custom_voices = custom_voices
        self.bank = VoiceBank.build(voices)
        self._lead_silence = torch.full((LEAD_SILENCE,), SILENCE_CODE, dtype=torch.int32)

    def _caches(
        self, resources: Mapping, plans: Mapping[str, Any], custom: SlotRows | None = None,
    ) -> WindowCaches:
        """A batch's ``WindowCaches``, on these plans of the cache resources (and its rows
        of the voice pool, ``custom``)."""
        return WindowCaches(**{
            name: AttentionCache(
                resources[T2W_KV[name]], self.bank.sources[name], plans[name],
                None if custom is None else (custom.pool.block(f"src_{name}", 0), custom.rows.slot_indices),
            )
            for name in CACHE_FAMILIES
        })

    def _voice_of(self, rid: str) -> tuple[int, int]:
        """``rid``'s voice, an index into ``bank`` (-1 for its own in the voice pool), and
        its prompt length in tokens."""
        req = self.request_state(rid)
        if "voice_prompt" in req:
            return -1, req["voice_prompt"].tokens.shape[1]
        voice = self.bank.index(req["voice"])
        return voice, self.bank.prompt_tokens[voice]

    def _voice_ids(self, rids: list[str], device: torch.device) -> torch.Tensor:
        return torch.tensor([self._voice_of(r)[0] for r in rids], device=device)

    def _voice_step(self, request_ids: list[str]) -> dict[str, RecurrentStep]:
        """The voice pool's step: an entry for each request with a custom voice."""
        if not self.custom_voices:
            return {}
        return {T2W_VOICES: RecurrentStep(segments=[
            Segment(request_id=rid, label="main",
                    span=int("voice_prompt" in (self.request_states.get(rid) or {})))
            for rid in request_ids
        ])}

    def _cache_steps(
        self, request_ids: list[str], num_tokens: list[int], lasts: list[bool],
    ) -> dict[str, BoundedKVStep]:
        """Each cache resource's step: every row's stream position (in its voice's stream)
        and what its window adds."""
        positions = []
        for rid in request_ids:
            req = self.request_states.get(rid)
            if req is None or not ("voice" in req or "voice_prompt" in req):
                # a capture's dummy row: the resource plans it as padding
                positions.append(None)
            else:
                voice, tokens = self._voice_of(rid)
                positions.append((tokens, req.get("written", 0), voice))
        steps = {}
        for name, family in CACHE_FAMILIES.items():
            steps[T2W_KV[name]] = BoundedKVStep(
                segments=[
                    Segment(request_id=rid, label="main", span=family.span(n, last))
                    for rid, n, last in zip(request_ids, num_tokens, lasts, strict=True)
                ],
                positions={
                    rid: StreamPosition(*family.position(tokens, written), source=voice)
                    for rid, (tokens, written, voice) in (
                        (r, p) for r, p in zip(request_ids, positions, strict=True) if p is not None
                    )
                },
            )
        return steps

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        is_final_stream_chunk: bool = False,
        **kwargs: Any,
    ) -> NodeInputs | None:
        if graph_walk == T2W_VOICE:
            # the voice encoders run on the CPU, as the reference's; the chunk steps
            # declare positions from the prompt's length
            clip = inputs["voice_audio"][0].to("cpu", torch.float32).reshape(-1)
            self.request_state(fwd_info.rid_handle).add("voice_prompt", self.model.voice_encoder(clip))
            return NodeInputs(tensor_inputs={}, kwargs={})
        if not inputs.get("tts_code"):
            return None
        # A host sync, which serializes this node's async pipeline. It is needed either
        # way: a window's length decides its capture and its cache writes. Filtering
        # the stop code on the device would need check_stop to filter outputs (an
        # engine change), and would still leave the length on the device.
        codes = inputs["tts_code"][0].reshape(-1).to("cpu", torch.int32)
        # the TTS's stop code is streamed before its stop is seen; it is not speech
        codes = codes[codes != self.eos_code]
        if codes.numel() == 0:
            # a text-only reply's stream closes with one empty chunk
            return None
        # keyed by the worker handle, as the forward and the engine's cleanup are
        state = self.request_state(fwd_info.rid_handle)
        if not state.get("started", False):
            codes = torch.cat([self._lead_silence, codes])
            if "voice_prompt" not in state:
                state.add("voice", fwd_info.step_metadata.get("voice"))
        return NodeInputs(tensor_inputs={"codes": codes[None]}, kwargs={"last": is_final_stream_chunk})

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[NodeInputs], **kwargs,
    ) -> SubmoduleStep:
        if graph_walk == T2W_VOICE:
            # takes the request's entry of the voice pool, or waits for one
            return SubmoduleStep(
                segments=[Segment(request_id=rid, label="main", span=1) for rid in request_ids],
                steps={T2W_VOICES: RecurrentStep()},
            )
        return SubmoduleStep(
            segments=[Segment(request_id=rid, label="main", span=1) for rid in request_ids],
            steps={
                T2W_STATE: RecurrentStep(),
                **self._voice_step(request_ids),
                **self._cache_steps(
                    request_ids, [inp.tensor_inputs["codes"].shape[1] for inp in inputs],
                    [inp.kwargs["last"] for inp in inputs],
                ),
            },
        )

    @staticmethod
    def flow_region(num_tokens: int, last: bool) -> str:
        return f"t2w/flow{'_last' if last else ''}{num_tokens}"

    @staticmethod
    def hift_region(first: bool, frames: int, last: bool) -> str:
        return f"t2w/hift/{'first' if first else 'next'}{'_last' if last else ''}{frames}"

    def get_piecewise_cuda_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype, tp_world_size: int = 1, **kwargs,
    ) -> dict[str, PiecewiseCudaGraphConfig]:
        def static_tokens(num_tokens: int):
            def make(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
                return {
                    "tokens": torch.zeros(shape.bs, num_tokens, dtype=torch.int32, device=device),
                    # each row's voice, an index into ``bank``
                    "voices": torch.zeros(shape.bs, dtype=torch.long, device=device),
                }
            return make

        def static_mel(frames: int):
            def make(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
                return {"mel": torch.zeros(shape.bs, MEL_BINS_T2W, frames, device=device)}
            return make

        def declare_flow(last: bool):
            def declare_step(request_ids: list[str], seq_lens: list[int]) -> SubmoduleStep:
                return SubmoduleStep(
                    segments=[Segment(request_id=rid, label="main", span=1) for rid in request_ids],
                    steps={
                        T2W_STATE: RecurrentStep(),
                        **self._voice_step(request_ids),
                        **self._cache_steps(request_ids, list(seq_lens), [last] * len(request_ids)),
                    },
                )
            return declare_step

        def declare_hift(request_ids: list[str], seq_lens: list[int]) -> SubmoduleStep:
            return SubmoduleStep(
                segments=[Segment(request_id=rid, label="main", span=1) for rid in request_ids],
                steps={T2W_STATE: RecurrentStep()},
            )

        def flow(last: bool):
            def capture(call: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
                pool = call.resources[T2W_STATE]
                plans = {name: call.resources[key].current for name, key in T2W_KV.items()}
                custom = None
                if self.custom_voices:
                    voices = call.resources[T2W_VOICES]
                    custom = SlotRows(voices, voices.addressing("main"))
                return {"mel": self.model.window_flow(
                    SlotRows(pool, pool.addressing("main")), self.bank, call.static_inputs["voices"],
                    call.static_inputs["tokens"], self._caches(call.resources, plans, custom), last, custom,
                )}
            return capture

        def hift(first: bool, last: bool):
            def capture(call: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
                pool = call.resources[T2W_STATE]
                return self.model.window_vocode(
                    SlotRows(pool, pool.addressing("main")), call.static_inputs["mel"], first, last,
                )
            return capture

        # Full windows at any position share one graph per batch size. A last window has
        # one of HOP lengths and rarely meets another of its length, so each length is
        # a one-row graph: 50 of this node's 65 graphs, most of its ~25 s of capture
        # and ~1 GiB. Padding last windows to one length would batch them and cut that
        # (the flow's time barely depends on length), but needs the padded tokens
        # masked out of attention; HiFT's convs are not causal, so its graphs stay per
        # length.
        #
        # Each row's voice is an input (an index into ``bank``), so one graph serves every
        # voice and rows of different voices share a batch.
        configs = {
            self.flow_region(WINDOW, False): PiecewiseBatchedConfig(
                capture_fn=flow(last=False),
                make_static_inputs=static_tokens(WINDOW),
                declare_step=declare_flow(last=False),
                seq_len=WINDOW,
                capture_batch_sizes=self.WINDOW_CAPTURE_BATCH_SIZES,
            ),
        }
        for n in LAST_WINDOW_TOKENS:
            configs[self.flow_region(n, True)] = PiecewiseBatchedConfig(
                capture_fn=flow(last=True),
                make_static_inputs=static_tokens(n),
                declare_step=declare_flow(last=True),
                seq_len=n,
                capture_batch_sizes=self.LAST_CAPTURE_BATCH_SIZES,
            )
        full = window_frames(WINDOW, False)
        for first in (True, False):
            configs[self.hift_region(first, full, False)] = PiecewiseBatchedConfig(
                capture_fn=hift(first, last=False),
                make_static_inputs=static_mel(full),
                declare_step=declare_hift,
                seq_len=full,
                capture_batch_sizes=self.WINDOW_CAPTURE_BATCH_SIZES,
            )
        for n in LAST_WINDOW_TOKENS:
            frames = window_frames(n, True)
            configs[self.hift_region(False, frames, True)] = PiecewiseBatchedConfig(
                capture_fn=hift(first=False, last=True),
                make_static_inputs=static_mel(frames),
                declare_step=declare_hift,
                seq_len=frames,
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
        if graph_walk == T2W_VOICE:
            return {}
        device = self.get_device()
        return {"rows": [(inp.tensor_inputs["codes"].to(device), inp.kwargs["last"]) for inp in inputs]}

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, rows: list | None = None, **kwargs,
    ) -> dict[str, NameToTensorList]:
        """Each window in two stages: the flow, batched over every row of one window length
        whatever its voice and position, then HiFT, batched over rows that are all or none
        their request's first window."""
        if graph_walk == T2W_VOICE:
            return self._prepare_voices(engine_inputs)
        pool = engine_inputs.resources[T2W_STATE]
        voices = engine_inputs.resources.get(T2W_VOICES)
        step = _NodeStep(
            engine_inputs=engine_inputs,
            state=SlotRows(pool, pool.addressing("main")),
            voices=None if voices is None else SlotRows(voices, voices.addressing("main")),
            plans={name: engine_inputs.resources[key].current for name, key in T2W_KV.items()},
            rows={rid: i for i, rid in enumerate(engine_inputs.request_ids)},
        )
        first, num_tokens, flows = {}, {}, {}
        for rid, (codes, last) in zip(engine_inputs.request_ids, rows, strict=True):
            first[rid] = not self.request_state(rid).get("started", False)
            num_tokens[rid] = codes.shape[1]
            flows.setdefault((codes.shape[1], last), []).append((rid, codes))

        mels, hifts = {}, {}
        for (n, last), group in flows.items():
            region = self.flow_region(n, last)
            for chunk in self._chunks(step, region, group, last):
                rids = [rid for rid, _ in chunk]
                tokens = torch.cat([codes for _, codes in chunk])
                mel = self._run_flow(step, region, rids, tokens, last)
                for i, rid in enumerate(rids):
                    mels[rid] = mel[i]
                    hifts.setdefault((first[rid], mel.shape[-1], last), []).append(rid)

        out = {}
        for (is_first, frames, last), rids in hifts.items():
            region = self.hift_region(is_first, frames, last)
            for chunk in self._chunks(step, region, rids, last):
                wav = self._run_hift(step, region, chunk, torch.stack([mels[rid] for rid in chunk]), is_first, last)
                for i, rid in enumerate(chunk):
                    req = self.request_state(rid)
                    # the attention caches' stream position (``_cache_steps``)
                    req.add("written", req.get("written", 0) + window_tokens(num_tokens[rid], last))
                    req.add("started", True)
                    out[rid] = {"audio_chunk": [wav[i]]}
        return out

    def max_batch_size(self, graph_walk: str):
        # one at a time: a full voice pool then holds back only the request that waits
        return 1 if graph_walk == T2W_VOICE else None

    @torch.compiler.disable
    def _prepare_voices(self, engine_inputs: ModelInputsFromEngine) -> dict[str, NameToTensorList]:
        """Prepare each request's custom voice (the flow over its prompt, as for a bundled
        voice at load) into its entry of the voice pool."""
        pool = engine_inputs.resources[T2W_VOICES]
        rows = SlotRows(pool, pool.addressing("main"))
        layout = dict(pool.config.blocks)
        for i, rid in enumerate(engine_inputs.request_ids):
            voice = self.model.prepare_voice(self.request_state(rid)["voice_prompt"])
            one = SlotRows(pool, rows.rows.select([i]))
            for name, value in voice_entry(voice, layout).items():
                one.set(name, value)
        return {rid: {} for rid in engine_inputs.request_ids}

    def _chunks(self, step: _NodeStep, region: str, group: list, last: bool) -> list[list]:
        """``group`` in batches its captured region can replay (all of it when uncaptured)."""
        runner = step.runner(region)
        if runner is None or not runner.any_graphs:
            return [group]
        cap = (self.LAST_CAPTURE_BATCH_SIZES if last else self.WINDOW_CAPTURE_BATCH_SIZES)[-1]
        return [group[i:i + cap] for i in range(0, len(group), cap)]

    @torch.compiler.disable
    def _run_flow(
        self,
        step: _NodeStep,
        region: str,
        rids: list[str],
        tokens: torch.Tensor,
        last: bool,
    ) -> torch.Tensor:
        """``window_flow`` for these rows: the captured region when it fits, else eagerly."""
        voices = self._voice_ids(rids, tokens.device)
        runner = step.runner(region)
        if runner is not None and runner.can_run(len(rids)):
            out = runner.run(static_inputs={"tokens": tokens, "voices": voices}, request_ids=rids, real_bs=len(rids))
            # copied out: the next replay of this region reuses the buffer
            return out.get_view("mel").clone()
        plans = {name: plan.select(step.select(rids)) for name, plan in step.plans.items()}
        custom = step.voices_of(rids) if step.voices is not None else None
        caches = self._caches(step.engine_inputs.resources, plans, custom)
        return self.model.window_flow(step.state_of(rids), self.bank, voices, tokens, caches, last, custom)

    @torch.compiler.disable
    def _run_hift(
        self,
        step: _NodeStep,
        region: str,
        rids: list[str],
        mel: torch.Tensor,
        first: bool,
        last: bool,
    ) -> torch.Tensor:
        """``window_vocode`` (captured when it fits) and ``window_finish``: the rows' samples."""
        state = step.state_of(rids)
        runner = step.runner(region)
        if runner is not None and runner.can_run(len(rids)):
            out = runner.run(static_inputs={"mel": mel}, request_ids=rids, real_bs=len(rids))
            magnitude, phase = out.get_view("magnitude"), out.get_view("phase")
        else:
            out = self.model.window_vocode(state, mel, first, last)
            magnitude, phase = out["magnitude"], out["phase"]
        return self.model.window_finish(state, magnitude, phase, first, last)


class _NodeStep(NamedTuple):
    """One token2wav forward's view of the node's resources, shared by its window groups."""

    engine_inputs: ModelInputsFromEngine
    # the node step's rows of the state and voice pools and its cache plans, taken before
    # a captured region plans its own
    state: SlotRows
    voices: SlotRows | None
    plans: dict[str, BoundedPlan]
    # request -> its row in the node step
    rows: dict[str, int]

    def runner(self, region: str) -> PiecewiseCudaGraphRunner | None:
        return self.engine_inputs.piecewise_runners.get(region)

    def select(self, rids: list[str]) -> list[int]:
        return [self.rows[rid] for rid in rids]

    def state_of(self, rids: list[str]) -> SlotRows:
        """These requests' rows of the node step's state, for an eager window."""
        return SlotRows(self.state.pool, self.state.rows.select(self.select(rids)))

    def voices_of(self, rids: list[str]) -> SlotRows:
        """These requests' rows of the node step's voice pool."""
        return SlotRows(self.voices.pool, self.voices.rows.select(self.select(rids)))
