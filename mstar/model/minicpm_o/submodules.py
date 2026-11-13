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
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.components.qwen3_lm import Qwen3DenseLM
from mstar.model.minicpm_o.components.audio import MiniCPMOAudio
from mstar.model.minicpm_o.components.vision import MiniCPMOVision, slice_layout
from mstar.model.minicpm_o.config import (
    AUDIO_ATTN,
    AUDIO_FEATURE_LENS,
    DECODE,
    DECODE_LOOP,
    DECODE_MIXED,
    DECODE_SPOKEN,
    DECODE_WALKS,
    IMAGE_TGT_SIZES,
    LLM_ATTN,
    LLM_KV,
    LLM_POS,
    LLM_SAMPLER,
    PATCHES,
    PREFILL_WALKS,
    QUERIES,
    RESAMPLER_ATTN,
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

# encoder output name -> the prompt positions it fills
_MM_FILLS = {"vision_embeds": "image_positions", "audio_embeds": "audio_positions"}


class LLMSubmodule(ARNodeSubmodule):
    PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024, 2048]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    # Off for the eager path only (a shape no capture holds), as on the other
    # Qwen3DenseLM nodes: compiled, Canary-Qwen's throughput was 7-15% lower
    # (its H100 A/B). The captures below keep `compile=True`: uncompiled, text
    # was 8% slower at c=1 and 3% at c=32 (H200). Inductor picks its kernels
    # per process, so greedy text can differ between boots with an empty
    # compile cache.
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
            sampler=LLM_SAMPLER, walk=DECODE_WALKS, prefill_walks=PREFILL_WALKS, runtime_check=True,
        )
        # a text reply's decode reads its last token off the sampler's slot
        # masters, and its client token rides inline
        self.device_loopback = DeviceLoopback(
            walk=DECODE, signal="text_inputs", sampler=LLM_SAMPLER, buffer="new_token",
        )

    def inline_signals(self, graph_walk: str) -> dict[str, str]:
        # a spoken reply's client token rides inline too; its other outputs go through the host
        if graph_walk == DECODE_SPOKEN:
            return {self.device_loopback.signal: self.device_loopback.buffer}
        return super().inline_signals(graph_walk)

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
                # the capture emits the TTS's pair too, which a text row's walk leaves unrouted
                replay_graph_walks=[DECODE, DECODE_SPOKEN, DECODE_MIXED],
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
        if graph_walk in DECODE_WALKS or not any(name in inputs for name in _MM_FILLS):
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
        if graph_walk in DECODE_WALKS:
            # A spoken reply's TTS reads each row's input token, so the ids
            # ride to the forward. A text row that looped back brings none:
            # its token is read by slot, beside the rows that bring theirs.
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
        reply's TTS reads (a text reply's walk has no edge for them)."""
        attn: AttentionManager = engine_inputs.resources[LLM_ATTN]
        sampler: SamplerResource = engine_inputs.resources[LLM_SAMPLER]
        hidden = self.model(input_embeds, label="main")
        out = {}
        if graph_walk in DECODE_WALKS:
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
        if request_info.graph_walk != DECODE_SPOKEN:  # the row's own walk, in a mixed step too
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
        # A decode row's EOS code ends the loop, checked by the graph runtime; the
        # code budget is the loop's cap. The prefill's code is never EOS: the
        # sampler bars it for 50 codes.
        self.token_stop = TokenStop(
            loop=TTS_DECODE_LOOP, buffer="tts_code", stop_ids=frozenset({model.config.eos_code}),
            sampler=TTS_SAMPLER, walk=TTS_DECODE, runtime_check=True,
        )
        # A decode step reads its last code off the sampler's slot masters, and
        # the code streams to the vocoder inline, so the step runs rows-only.
        self.device_loopback = DeviceLoopback(
            walk=TTS_DECODE, signal="tts_code", sampler=TTS_SAMPLER, buffer="tts_code",
        )

    def inline_signals(self, graph_walk: str) -> dict[str, str]:
        # The prefill's code too, not only the loop-back's: then every item of a
        # reply's stream goes by value, and no window mixes in a stored one.
        return {"tts_code": "tts_code"}

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
                # cuBLAS GEMMs: at decode widths the -O2 template search picks Triton
                # templates that run this step's kernels ~27% slower (GPU time, c=32)
                compile_mode="default",
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
        has_loopback: bool = False,
        **kwargs: Any,
    ) -> ARNodeInputs:
        if (row := self.uniform_row_inputs(graph_walk, has_loopback)) is not None:
            # one code, read off the sampler's slot master in ``preprocess``
            return row
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
        if graph_walk == TTS_DECODE and any(inp.input_ids is None for inp in inputs):
            # a row that looped back on the device brings no code: read by slot,
            # beside the rows that bring theirs (a first decode step)
            return {"input_embeds": self.model.emb_code(self.loopback_inputs(engine_inputs, inputs)["input_ids"])}
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
