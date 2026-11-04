"""MiniCPM-o's node submodules: the Qwen3 LLM, the two encoders, and a spoken
reply's TTS and vocoder (token2wav).

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
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig,
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
    PiecewiseAcceleratorGraphConfig,
    PiecewiseBatchedConfig,
    PiecewiseCallInputs,
    PiecewiseCaptureShape,
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
    window_frames,
)
from mstar.model.minicpm_o.components.token2wav_flow import MEL_BINS as MEL_BINS_T2W
from mstar.model.minicpm_o.components.vision import MiniCPMOVision, slice_layout
from mstar.model.minicpm_o.config import (
    AUDIO_ATTN,
    AUDIO_FEATURE_LENS,
    DECODE_LOOP,
    IMAGE_TGT_SIZES,
    LLM_ATTN,
    LLM_KV,
    LLM_POS,
    LLM_SAMPLER,
    PATCHES,
    PREFILL_WALKS,
    QUERIES,
    RESAMPLER_ATTN,
    T2W_KV,
    T2W_STATE,
    TTS_ATTN,
    TTS_DECODE,
    TTS_DECODE_LOOP,
    TTS_KV,
    TTS_POS,
    TTS_PREFILL,
    TTS_SAMPLER,
    VISION_ATTN,
    MiniCPMOConfig,
)
from mstar.model.submodule_base import (
    EMPTY_INPUT_METADATA,
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    DeviceLoopback,
    InputMetadata,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
    TokenStop,
)
from mstar.utils.h2d import to_device_nonblocking

logger = logging.getLogger(__name__)

DECODE = "decode"

# encoder output name -> the prompt positions it fills
_MM_FILLS = {"vision_embeds": "image_positions", "audio_embeds": "audio_positions"}


class LLMSubmodule(ARNodeSubmodule):
    PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024, 2048]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    # off, as on the other Qwen3DenseLM nodes: compiled, Canary-Qwen's
    # throughput was 7-15% lower (its H100 A/B)
    disable_torch_compile = True

    def __init__(self, model: Qwen3DenseLM, config: MiniCPMOConfig):
        super().__init__()
        self.model = model
        self.config = config
        # A prefill samples the reply's first token, so a stop id there (or a
        # one-token budget) ends the partition; decode's stop ids are checked
        # by the graph runtime, and its budget is the loop's cap.
        self.token_stop = TokenStop(
            loop=DECODE_LOOP, buffer="new_token", stop_ids=frozenset(config.stop_token_ids),
            sampler=LLM_SAMPLER, walk=DECODE, prefill_walks=PREFILL_WALKS, runtime_check=True,
        )
        # decode reads its last token off the sampler's slot masters, and its
        # client token rides inline
        self.device_loopback = DeviceLoopback(
            walk=DECODE, signal="text_inputs", sampler=LLM_SAMPLER, buffer="new_token",
        )

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        """Decode, and one prefill capture every prefill walk replays: ids are
        embedded in `preprocess`, so the forward always takes embeddings."""
        def dummy(n: int) -> ARNodeInputs:
            return ARNodeInputs(input_ids=torch.zeros(n, dtype=torch.long, device=device), input_seq_len=n)

        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk=DECODE,
                single_request_inputs=dummy(1),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
            ),
            PackedAcceleratorGraphConfig(
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
        has_loopback: bool = False,
        **kwargs: Any,
    ) -> ARNodeInputs:
        if (row := self.uniform_row_inputs(graph_walk, has_loopback)) is not None:
            # one token per row, read off the sampler's slot master in
            # `preprocess`; the routed signal is empty here
            return row
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
        if graph_walk == DECODE:
            # A spoken reply's TTS reads each row's input token, so the ids
            # ride to the forward. With device loop-back every row is its
            # request's last sampled token, read by slot.
            if any(inp.input_ids is None for inp in inputs):
                ids = self.loopback_inputs(engine_inputs, inputs)["input_ids"]
            else:
                ids = torch.cat([inp.input_ids for inp in inputs]).to(device)
            return {"input_embeds": self.model.embed(ids), "input_ids": ids}
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
                LLM_SAMPLER: SamplerStep(apply_penalty=True),
            },
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_embeds: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """``new_token``, and on decode the (``tts_ids``, ``tts_hidden``) pair:
        each row's input token and its post-norm hidden, which a spoken
        reply's TTS reads."""
        attn: AttentionManager = engine_inputs.resources[LLM_ATTN]
        sampler: SamplerResource = engine_inputs.resources[LLM_SAMPLER]
        hidden = self.model(input_embeds, label="main")
        out = {}
        if graph_walk == DECODE:
            out["tts_ids"] = input_ids
            out["tts_hidden"] = hidden
        else:
            hidden = attn.select_last_hidden(hidden)
        out["new_token"] = sampler.sample(engine_inputs.request_ids, logits=self.model.logits(hidden))
        return out

    def forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_embeds: torch.Tensor,
        input_ids: torch.Tensor | None = None, **kwargs,
    ) -> NameToTensorList:
        return {k: [v] for k, v in self._forward(graph_walk, engine_inputs, input_embeds, input_ids).items()}

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_embeds: torch.Tensor,
        input_ids: torch.Tensor | None = None, **kwargs,
    ) -> BatchedModelOutput:
        out = self._forward(graph_walk, engine_inputs, input_embeds, input_ids)
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
        **kwargs,
    ):
        # Rebind, not copy: the decode loop routes on `text_inputs`.
        if "new_token" in outputs:
            outputs["text_inputs"] = outputs["new_token"]
        if "tts_hidden" not in outputs:
            return
        if not request_info.step_metadata.get("audio_output", False):
            outputs.pop("tts_ids")
            outputs.pop("tts_hidden")
            return
        # The decode loop accumulates the spoken reply's rows until it ends. A row
        # is a view of the batch's outputs, which would keep every row of every
        # step alive with it.
        for name in ("tts_ids", "tts_hidden"):
            outputs[name] = [t.clone() for t in outputs[name]]


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
        layout = slice_layout([tuple(hw) for hw in fwd_info.step_metadata[IMAGE_TGT_SIZES]], self.config.vision)
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
            return torch.cat([inp.tensor_inputs[name] for inp in inputs])

        return {
            "patches": cat("patches").to(device),
            # built on the host: a pageable copy would wait out the running step
            "position_ids": to_device_nonblocking(cat("position_ids"), device),
            "grid_coords": to_device_nonblocking(cat("grid_coords"), device),
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
        mel_lens = list(fwd_info.step_metadata[AUDIO_FEATURE_LENS])
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

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype, tp_world_size: int = 1, **kwargs,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
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
                can_lease=self._fits_block_loop,
                total_tokens=self.BLOCK_LOOP_FRAME_BUCKETS,
                capture_batch_sizes=[1],
            )
        }

    def _fits_block_loop(self, inputs: list[NodeInputs]) -> bool:
        """Whether the request's pieces fit a replay's attention plan: the
        resource's per-request ceiling, read from it so a YAML override holds."""
        per_request = self.node_resources[AUDIO_ATTN].max_segments_per_request
        return sum(len(inp.kwargs["frames"]) for inp in inputs) <= per_request * len(inputs)

    @torch.compiler.disable
    def _replay_block_loop(
        self, engine_inputs: ModelInputsFromEngine, hidden: torch.Tensor, frames: list[int],
    ) -> torch.Tensor | None:
        """The block loop replayed for the step's one request, or None if not leased."""
        if AUDIO_BLOCK_LOOP not in engine_inputs.piecewise_leases:
            return None
        return engine_inputs.piecewise_runners[AUDIO_BLOCK_LOOP].run(
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
        # A decode row's EOS code ends the loop; the code budget is the loop's
        # cap. The prefill's code is never EOS: the sampler bars it for 50 codes.
        self.token_stop = TokenStop(
            loop=TTS_DECODE_LOOP, buffer="tts_code", stop_ids=frozenset({model.config.eos_code}),
            sampler=TTS_SAMPLER, walk=TTS_DECODE,
        )

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        hidden = self.config_tts.hidden_size

        def prefill_dummy(n: int) -> ARNodeInputs:
            return ARNodeInputs(
                input_embeds=torch.zeros(n, hidden, device=device, dtype=self.model.emb_code.weight.dtype),
                input_seq_len=n,
            )

        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk=TTS_DECODE,
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device), input_seq_len=1,
                ),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
            ),
            PackedAcceleratorGraphConfig(
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


class Token2WavSubmodule(NodeSubmodule):
    """Speech codes -> 24 kHz audio, one stream window per step: upstream's
    ``Token2wav.stream``, fed the way its ``streaming_generate`` feeds it (three
    silence codes first, 28-code windows advancing by 25, a last flush).

    The windows are the ``LeftContextChunkPolicy(25, 3)`` stream from the TTS;
    the first window gets the silence prepended here. A request's attention caches
    are on the ``T2W_KV`` bounded KV resources, which start from its voice's
    prepared caches; its conv caches and HiFT tails are its slot of ``T2W_STATE``
    (``RecurrentStatePool``). Its stream position is the chunk's offset in the
    stream (``prepare_inputs``).

    Float32. Each window runs in two stages (``forward_batched``), each a piecewise
    capture bucketed by batch size: the flow (``t2w/<voice>/flow28``), for windows
    at any position, and HiFT, separately for a request's first window, whose
    input is shorter. A last window has one of a few lengths, each a one-row
    capture; other shapes run eagerly. The inverse STFT and cross-fade stay eager.
    The tests' one-request driver over the same modules is bit-exact against the
    reference at full fp32; this path attends the DiT's cache in another order and
    runs on fused TF32 kernels.
    """

    WINDOW_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]
    # a request ends once, so last windows of one length rarely meet in a pass
    LAST_CAPTURE_BATCH_SIZES = [1]

    disable_torch_compile = True
    disable_autocast = True

    def __init__(self, model, voices: dict, eos_code: int):
        """``voices``: the model's bundled ones, by name."""
        super().__init__()
        self.model = model
        self.eos_code = eos_code
        self.bank = VoiceBank.build(voices)
        self._lead_silence = torch.full((LEAD_SILENCE,), SILENCE_CODE, dtype=torch.int32)

    def _caches(self, resources: Mapping) -> WindowCaches:
        """The step's ``WindowCaches``."""
        return WindowCaches(**{
            name: AttentionCache(resources[T2W_KV[name]], self.bank.sources[name]) for name in CACHE_FAMILIES
        })

    def _voice_of(self, rid: str) -> tuple[int, int]:
        """``rid``'s voice, an index into ``bank``, and its prompt length in tokens."""
        voice = self.bank.index(self.request_state(rid)["voice"])
        return voice, self.bank.prompt_tokens[voice]

    def _voice_ids(self, rids: list[str], device: torch.device) -> torch.Tensor:
        # pinned and queued: a pageable copy would wait out the running step
        return to_device_nonblocking([self._voice_of(r)[0] for r in rids], device, torch.long)

    def _cache_steps(
        self, request_ids: list[str], num_tokens: list[int], lasts: list[bool],
    ) -> dict[str, BoundedKVStep]:
        """Each cache resource's step: every row's stream position (in its voice's stream)
        and what its window adds."""
        positions = []
        for rid in request_ids:
            req = self.request_states.get(rid)
            if req is None or "voice" not in req:
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
        input_metadata: InputMetadata = EMPTY_INPUT_METADATA,
        **kwargs: Any,
    ) -> NodeInputs | None:
        if not inputs.get("tts_code"):
            # a text-only reply's stream closes with one empty chunk
            return None
        # A host sync, which serializes this node's async pipeline. It is needed either
        # way: a window's length decides its capture and its cache writes. Filtering
        # the stop code on the device would need check_stop to filter outputs (an
        # engine change), and would still leave the length on the device.
        codes = inputs["tts_code"][0].reshape(-1).to("cpu", torch.int32)
        # The TTS's stop code is streamed before its stop is seen. It is not speech, and
        # it ends the speech: the window holding it is the last, as upstream's stream
        # ends there.
        last = is_final_stream_chunk or bool((codes == self.eos_code).any())
        codes = codes[codes != self.eos_code]
        chunk = input_metadata.stream_chunks["tts_code"]
        if chunk.context_items == chunk.num_items:
            # No new codes: the stream's final chunk after a window that held the stop
            # code, which only repeats that window's tail. It still runs, voicing
            # nothing (``forward_batched``): a skipped step never completes the node,
            # which then refuses the stream's end.
            return NodeInputs(tensor_inputs={"codes": codes[None, :0]}, kwargs={"last": True})
        # keyed by the worker handle, as the forward and the engine's cleanup are
        state = self.request_state(fwd_info.rid_handle)
        first = chunk.start_offset == 0
        if first:
            codes = torch.cat([self._lead_silence, codes])
            state.add("voice", fwd_info.step_metadata.get("voice"))
        # Window k starts 25k codes into the silence-led stream (its look-back
        # included), which is what its caches hold before it (``_cache_steps``).
        state.add("written", 0 if first else chunk.start_offset + LEAD_SILENCE)
        return NodeInputs(tensor_inputs={"codes": codes[None]}, kwargs={"last": last, "first": first})

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[NodeInputs], **kwargs,
    ) -> SubmoduleStep:
        # This step admits the windows (slots, positions), which has to happen before
        # the forward. Its plan goes unused: each region re-plans the rows it runs.
        # TODO: an admit-only step would save that plan's host work.
        return SubmoduleStep(
            segments=[Segment(request_id=rid, label="main", span=1) for rid in request_ids],
            steps={
                T2W_STATE: RecurrentStep(),
                **self._cache_steps(
                    request_ids, [inp.tensor_inputs["codes"].shape[1] for inp in inputs],
                    [inp.kwargs["last"] for inp in inputs],
                ),
            },
        )

    # Each window's flow and HiFT run through a piecewise region: replayed when a graph
    # fits, else eagerly through the same step, so the resources always hold the plan
    # of exactly the rows being run. Shapes that are never captured (a one-window
    # reply's HiFT) have eager-only regions.

    @staticmethod
    def flow_region(num_tokens: int, last: bool) -> str:
        return f"t2w/flow{'_last' if last else ''}{num_tokens}"

    @staticmethod
    def hift_region(first: bool, frames: int, last: bool) -> str:
        return f"t2w/hift/{'first' if first else 'next'}{'_last' if last else ''}{frames}"

    @staticmethod
    def eager_region(stage: str, first: bool, last: bool) -> str:
        return f"t2w/{stage}_eager/{'first' if first else 'next'}{'_last' if last else ''}"

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype, tp_world_size: int = 1, **kwargs,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
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
            def run(call: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
                pool = call.resources[T2W_STATE]
                return {"mel": self.model.window_flow(
                    SlotRows(pool, pool.addressing("main")), self.bank, call.static_inputs["voices"],
                    call.static_inputs["tokens"], self._caches(call.resources), last,
                )}
            return run

        def hift(first: bool, last: bool):
            def run(call: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
                pool = call.resources[T2W_STATE]
                return self.model.window_vocode(
                    SlotRows(pool, pool.addressing("main")), call.static_inputs["mel"], first, last,
                )
            return run

        def region(run, make_static_inputs, declare_step, seq_len, batch_sizes):
            return PiecewiseBatchedConfig(
                capture_fn=run, make_static_inputs=make_static_inputs, declare_step=declare_step,
                seq_len=seq_len, capture_batch_sizes=batch_sizes, eager_fallback=True,
            )

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
        full = window_frames(WINDOW, False)
        configs = {
            self.flow_region(WINDOW, False): region(
                flow(False), static_tokens(WINDOW), declare_flow(False), WINDOW, self.WINDOW_CAPTURE_BATCH_SIZES),
        }
        for first in (True, False):
            configs[self.hift_region(first, full, False)] = region(
                hift(first, False), static_mel(full), declare_hift, full, self.WINDOW_CAPTURE_BATCH_SIZES)
        for n in LAST_WINDOW_TOKENS:
            frames = window_frames(n, True)
            configs[self.flow_region(n, True)] = region(
                flow(True), static_tokens(n), declare_flow(True), n, self.LAST_CAPTURE_BATCH_SIZES)
            configs[self.hift_region(False, frames, True)] = region(
                hift(False, True), static_mel(frames), declare_hift, frames, self.LAST_CAPTURE_BATCH_SIZES)
        for last in (False, True):
            configs[self.eager_region("flow", False, last)] = region(
                flow(last), static_tokens(WINDOW), declare_flow(last), WINDOW, [])
            for first in (False, True):
                configs[self.eager_region("hift", first, last)] = region(
                    hift(first, last), static_mel(full), declare_hift, full, [])
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
            (inp.tensor_inputs["codes"].to(device), inp.kwargs["last"], inp.kwargs.get("first", False))
            for inp in inputs
        ]}

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, rows: list | None = None, **kwargs,
    ) -> dict[str, NameToTensorList]:
        """Each window in two stages: the flow, batched over every row of one window length
        whatever its voice and position, then HiFT, batched over rows that are all or none
        their request's first window."""
        runners = engine_inputs.piecewise_runners
        out, first, flows = {}, {}, {}
        for rid, (codes, last, is_first) in zip(engine_inputs.request_ids, rows, strict=True):
            if not codes.shape[1]:
                # past its last window (``prepare_inputs``)
                out[rid] = {}
                continue
            first[rid] = is_first
            flows.setdefault((codes.shape[1], last), []).append((rid, codes))

        mels, hifts = {}, {}
        for (n, last), group in flows.items():
            runner = runners.get(self.flow_region(n, last)) or runners[self.eager_region("flow", False, last)]
            for chunk in self._chunks(runner, group, last):
                rids = [rid for rid, _ in chunk]
                tokens = torch.cat([codes for _, codes in chunk])
                mel = self._run_flow(runner, rids, tokens)
                for i, rid in enumerate(rids):
                    mels[rid] = mel[i]
                    hifts.setdefault((first[rid], mel.shape[-1], last), []).append(rid)

        for (is_first, frames, last), rids in hifts.items():
            runner = runners.get(self.hift_region(is_first, frames, last)) or runners[
                self.eager_region("hift", is_first, last)]
            for chunk in self._chunks(runner, rids, last):
                mel = torch.stack([mels[rid] for rid in chunk])
                wav = self._run_hift(engine_inputs, runner, chunk, mel, is_first, last)
                for i, rid in enumerate(chunk):
                    out[rid] = {"audio_chunk": [wav[i]]}
        return out

    def _chunks(self, runner, group: list, last: bool) -> list[list]:
        """``group`` in batches its region can replay (all of it when it runs eagerly)."""
        if not runner.any_graphs:
            return [group]
        cap = (self.LAST_CAPTURE_BATCH_SIZES if last else self.WINDOW_CAPTURE_BATCH_SIZES)[-1]
        return [group[i:i + cap] for i in range(0, len(group), cap)]

    @torch.compiler.disable
    def _run_flow(self, runner, rids: list[str], tokens: torch.Tensor) -> torch.Tensor:
        """``window_flow`` for these rows (``[B, 80, frames]``)."""
        voices = self._voice_ids(rids, tokens.device)
        out = runner.run(
            static_inputs={"tokens": tokens, "voices": voices}, request_ids=rids,
            seq_lens=[tokens.shape[1]] * len(rids), real_bs=len(rids),
        )
        # copied out: the next replay of this region reuses the buffer
        return out.get_view("mel").clone()

    @torch.compiler.disable
    def _run_hift(
        self,
        engine_inputs: ModelInputsFromEngine,
        runner,
        rids: list[str],
        mel: torch.Tensor,
        first: bool,
        last: bool,
    ) -> torch.Tensor:
        """``window_vocode`` and then (eagerly: the inverse STFT syncs) ``window_finish``,
        on the rows the region just planned: the rows' samples."""
        out = runner.run(
            static_inputs={"mel": mel}, request_ids=rids, seq_lens=[mel.shape[-1]] * len(rids), real_bs=len(rids),
        )
        pool = engine_inputs.resources[T2W_STATE]
        state = SlotRows(pool, pool.addressing("main")).head(len(rids))
        return self.model.window_finish(state, out.get_view("magnitude"), out.get_view("phase"), first, last)
