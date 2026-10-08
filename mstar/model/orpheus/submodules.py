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
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import (
    AttentionStep,
    KVStep,
    PositionStep,
    SamplerStep,
    Segment,
    SlotLease,
    SubmoduleStep,
    keep_final_chunk_samples,
)
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.orpheus.config import ATTN, KV_CACHE, ROPE, SAMPLER, OrpheusModelConfig
from mstar.model.submodule_base import (
    EMPTY_INPUT_METADATA,
    NO_CHUNKING,
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    ChunkingPolicy,
    InputMetadata,
    InputSeqLenInfo,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)
from mstar.streaming.stream_buffer import StreamChunkInfo

logger = logging.getLogger(__name__)


class OrpheusLLMSubmodule(ARNodeSubmodule):
    """Llama 3.2 3B wrapper for Orpheus TTS.

    Dispatches on graph_walk:
      - prefill: embed text tokens, fill KV cache
      - decode: embed previous token, generate next audio token
      - mixed: prefill and decode rows in one batch
    """

    def __init__(
        self,
        language_model: nn.Module,
        config: OrpheusModelConfig,
    ):
        super().__init__()
        self.language_model = language_model
        self.embed_tokens = language_model.model.embed_tokens
        self.lm_head = language_model.lm_head
        self.config = config

    PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 4, 16, 64]
    MAX_BATCH_TOKENS = 512

    def get_chunking_policy(self, graph_walk: str) -> ChunkingPolicy:
        if graph_walk == "prefill":
            return ChunkingPolicy(chunkable=True, max_batch_tokens=self.MAX_BATCH_TOKENS)
        if graph_walk == "mixed":
            return ChunkingPolicy(max_batch_tokens=self.MAX_BATCH_TOKENS)
        return NO_CHUNKING

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="decode",
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device),
                    input_seq_len=1
                ),
            ),
            PackedAcceleratorGraphConfig(
                capture_graph_walk="prefill",
                # a mixed batch is packed rows too, decode rows one token long
                replay_graph_walks=["prefill", "mixed"],
                capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
                make_node_input=lambda n: ARNodeInputs(
                    input_ids=torch.zeros(
                        (n,), dtype=torch.long, device=device,
                    ), input_seq_len=n
                ),
                capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES
            )
        ]

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> ARNodeInputs:
        return ARNodeInputs(
            input_ids=inputs["text_inputs"][0],
            input_seq_len=inputs["text_inputs"][0].shape[0]
        )

    def get_input_sequence_len(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList, **kwargs,
    ) -> InputSeqLenInfo | None:
        if "text_inputs" not in inputs:
            return None
        return InputSeqLenInfo(inputs["text_inputs"][0].shape[0])

    def speculative_input_seq_len(self, graph_walk, fwd_info, prev_seq_len):
        if graph_walk == "decode":
            return 1
        return super().speculative_input_seq_len(graph_walk, fwd_info, prev_seq_len)

    def declare_inputs_from_spans(self, graph_walk, spans, walks, infos):
        # a row declares its span and walk; its ids only feed the sampler's
        # prompt mask, which a pre-plan leaves alone
        if graph_walk not in ("prefill", "mixed"):
            return None
        return [
            ARNodeInputs(input_seq_len=span, graph_walk=walk)
            for span, walk in zip(spans, walks, strict=True)
        ]

    def declare_step(
        self, graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ):
        prefill_tokens = {
            rid: inp.input_ids for rid, inp in zip(request_ids, inputs, strict=True)
            if (inp.graph_walk or graph_walk) == "prefill"
        }
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
                SAMPLER: keep_final_chunk_samples(SamplerStep(
                    apply_penalty=True,
                    prefill_tracked_tokens=prefill_tokens
                ), request_ids, inputs),
                ROPE: PositionStep()
            }
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        return {
            "text_inputs": torch.cat([inp.input_ids for inp in inputs]),
        }

    def _forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        text_inputs: torch.Tensor,
    ) -> torch.Tensor:
        sampler: SamplerResource = engine_inputs.resources[SAMPLER]
        attn: AttentionManager = engine_inputs.resources[ATTN]
        emb = self.embed_tokens(text_inputs)
        hidden = self.language_model(emb, label="main")

        if graph_walk != "decode":
            hidden = attn.select_last_hidden(hidden)

        logits = self.lm_head(hidden)
        return sampler.sample(
            engine_inputs.request_ids,
            logits=logits
        )


    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        text_inputs: torch.Tensor,
        **kwargs
    ) -> NameToTensorList:
        return {
            "new_token": self._forward(
                graph_walk=graph_walk,
                engine_inputs=engine_inputs,
                text_inputs=text_inputs
            )
        }

    def can_batch(
        self, batch: ExecutingBatch,
        model_inputs: list[NodeInputs]
    ) -> bool:
        return True

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        text_inputs: torch.Tensor,
        **kwargs
    ) -> BatchedModelOutput:
        new_tokens = self._forward(
            graph_walk=graph_walk,
            engine_inputs=engine_inputs,
            text_inputs=text_inputs
        )
        # row i is request i; a per-rid dict here would guard the compiled frame on the ids
        return BatchedModelOutput(row_outputs={"new_token": new_tokens})

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs
    ):
        # Metadata-only: rebind output name for graph routing. EOS check
        # moved to check_stop so the GPU thread doesn't sync on .item() here.
        inputs = kwargs.get("inputs")
        if inputs is not None and not inputs.is_final_chunk:
            outputs.pop("new_token", None)  # only the last chunk's token is real
        if "new_token" not in outputs:
            return
        outputs["text_inputs"] = outputs["new_token"]

    def check_stop(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "new_token" not in outputs:
            return set()
        token = outputs["new_token"][0].item()
        eos_token_id = self.config.stop_token_id
        ignore_eos= request_info.resource_configs[SAMPLER].ignore_eos
        if (not ignore_eos and eos_token_id == token) or \
                (request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 1 >= request_info.max_tokens):
            return {"decode_loop"}
        return set()


class SNACDecoderSubmodule(NodeSubmodule):
    """SNAC 24kHz streaming decoder submodule.

    Receives a window of raw audio tokens from the StreamBuffer (via normal
    graph input routing), decodes all of its frames and emits the audio of
    the frames the window owns: it skips the left context and holds back the
    right context, which the next window emits (see
    ``OrpheusModelConfig.snac_chunk_policy``). The final window emits up to
    the last whole frame.

    Batched: windows of one size decode in one SNAC forward, and every size
    a stream can produce has a capture. The scheduler batches each size
    apart (``split_batches_by_capture_key``), because the decoder is not
    causal and padding would change the frames next to it; a batch that
    mixes sizes anyway decodes each size apart, eagerly.
    """

    # fp32, uncompiled: what ``get_stateless_flavor`` used to buy on the old
    # stateless engine, stated directly now that the v1 engine reads these.
    disable_torch_compile = True
    disable_autocast = True

    CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]

    def __init__(self, snac_model: nn.Module, config: OrpheusModelConfig):
        super().__init__()
        self.snac_model = snac_model
        device = next(self.snac_model.parameters()).device
        self.idx_14 = torch.tensor([1, 4], dtype=torch.long, device=device)
        self.idx_2356 = torch.tensor([2, 3, 5, 6], dtype=torch.long, device=device)
        self.config = config
        self.windows = config.snac_window_frames()

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1
    ) -> list[AcceleratorGraphConfig]:
        """One capture per window size a stream can produce, final flushes included.

        Not compiled, like the eager path (``disable_torch_compile``): the
        captured and eager decodes of a stream then run the same kernels, and
        max-autotune of SNAC's convolutions takes minutes per shape, for each
        of the window sizes times batch sizes.
        """
        del tp_world_size
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="snac_chunk",
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(
                        (frames, self.config.tokens_per_frame), dtype=torch.long, device=device,
                    ),
                    # 1, not the window: batched buckets match on bs
                    input_seq_len=1,
                ),
                additional_key_info=frames,
                capture_batch_sizes=self.CAPTURE_BATCH_SIZES,
                input_seq_dims={"codes_0": 0, "codes_1": 0, "codes_2": 0},
                compile=False,
            )
            for frames in sorted(self.windows, reverse=True)
        ]

    def _stream_chunk(self, input_metadata: InputMetadata | None) -> StreamChunkInfo | None:
        return input_metadata.stream_chunks.get("new_token") if input_metadata else None

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        input_metadata: InputMetadata = EMPTY_INPUT_METADATA,
        is_final_stream_chunk: bool = False,
        **kwargs
    ) -> ARNodeInputs:
        """One window's tokens -> ``(frames, 7)`` codes plus the frames to emit.

        A trailing partial frame (the EOS token the LLM streams last) is not
        audio and is dropped. Without stream chunk info the window is treated
        as a whole utterance.
        """
        tpf = self.config.tokens_per_frame
        right = self.config.snac_right_context_frames
        # an empty final chunk (a no-context stream cut at a chunk boundary
        # without EOS) arrives as no tensor at all
        received = inputs.get("new_token") or [self.idx_14.new_empty(0)]
        tokens = received[0].flatten()
        chunk = self._stream_chunk(input_metadata)
        frames = tokens.numel() // tpf
        final = is_final_stream_chunk or chunk is None or chunk.is_final
        start = max((chunk.context_items if chunk is not None else 0) // tpf - right, 0)
        end = max(frames if final else frames - right, start)
        codes = (tokens[: frames * tpf].view(frames, tpf) - self.config.custom_token_base_id - 10) % 4096
        if frames == 0:
            # a stream shorter than one frame: decode a silent frame, emit nothing
            codes = torch.zeros((1, tpf), dtype=tokens.dtype, device=tokens.device)
        return ARNodeInputs(
            input_ids=codes,
            input_seq_len=1,
            kwargs={"start": start, "end": end},
        )

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[ARNodeInputs]) -> bool:
        return True

    def split_batches_by_capture_key(self, graph_walk: str) -> bool:
        """A batch that mixes window sizes runs eager, so batch each size apart."""
        return graph_walk == "snac_chunk"

    def cg_key_info(
        self,
        graph_walk: str,
        per_request_info: Mapping[str, CurrentForwardPassInfo],
        per_request_input_metadata: Mapping[str, InputMetadata] | None = None,
        **kwargs: Any,
    ) -> Any:
        """The batch's window size in frames when all its windows share one."""
        del graph_walk, kwargs
        per_request_input_metadata = per_request_input_metadata or {}
        frames = set()
        for request_id in per_request_info:
            chunk = self._stream_chunk(per_request_input_metadata.get(request_id))
            if chunk is None:
                return None
            # prepare_inputs decodes a stream shorter than a frame as one frame
            frames.add(max(chunk.num_items // self.config.tokens_per_frame, 1))
        return frames.pop() if len(frames) == 1 else None

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        widths = tuple(item.input_ids.shape[0] for item in inputs)
        width = max(widths)
        stacked = torch.stack([
            torch.nn.functional.pad(item.input_ids, (0, 0, 0, width - item.input_ids.shape[0]))
            for item in inputs
        ])
        codes_0, codes_1, codes_2 = self._extract_snac_codes(stacked)
        return {"codes_0": codes_0, "codes_1": codes_1, "codes_2": codes_2, "widths": widths}

    def _decode(self, codes_0, codes_1, codes_2) -> torch.Tensor:
        audio = self.snac_model.decode([codes_0, codes_1, codes_2])
        return (audio.clamp(-1, 1) * 32767).to(torch.int16).squeeze(1)

    def _decode_rows(self, codes_0, codes_1, codes_2, widths) -> torch.Tensor:
        """``[B, widest * hop]`` int16 audio, each row decoded at its own width."""
        if len(set(widths)) == 1:
            return self._decode(codes_0, codes_1, codes_2)
        hop = self.config.snac_hop_samples
        audio = torch.zeros(
            (codes_0.shape[0], max(widths) * hop), dtype=torch.int16, device=codes_0.device,
        )
        for width in sorted(set(widths)):
            rows = torch.tensor(
                [i for i, w in enumerate(widths) if w == width], device=codes_0.device,
            )
            audio[rows, : width * hop] = self._decode(
                codes_0[rows, :width], codes_1[rows, : 2 * width], codes_2[rows, : 4 * width],
            )
        return audio

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        codes_0: torch.Tensor,
        codes_1: torch.Tensor,
        codes_2: torch.Tensor,
        widths: tuple[int, ...] = (),
        **kwargs,
    ) -> NameToTensorList:
        return {"audio_chunk": [self._decode_rows(codes_0, codes_1, codes_2, widths or (codes_0.shape[1],))[0]]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        codes_0: torch.Tensor,
        codes_1: torch.Tensor,
        codes_2: torch.Tensor,
        widths: tuple[int, ...] = (),
        **kwargs,
    ) -> dict[str, NameToTensorList]: # request_id to tensors
        audio = self._decode_rows(codes_0, codes_1, codes_2, widths or (codes_0.shape[1],) * codes_0.shape[0])
        return {
            rid: {"audio_chunk": [audio[i].detach()]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        inputs: ARNodeInputs | None = None,
        **kwargs
    ):
        """Keep the window's own frames. The geometry rides with the pass's
        inputs: under speculative scheduling the next window of the request is
        prepared before this one is postprocessed."""
        if "audio_chunk" not in outputs:
            return
        if inputs is None:
            raise ValueError("SNAC postprocess needs the pass's inputs for its window geometry")
        hop = self.config.snac_hop_samples
        audio = outputs["audio_chunk"][0].reshape(-1)
        outputs["audio_chunk"][0] = audio[inputs.kwargs["start"] * hop : inputs.kwargs["end"] * hop]

    def _extract_snac_codes(self, mf: torch.Tensor):
        """Split ``(B, frames, 7)`` codes into the three codebook levels.

        Returns:
            (codes_0, codes_1, codes_2) with shapes
            ``(B, frames)``, ``(B, 2 * frames)``, ``(B, 4 * frames)``.
        """
        codes_0 = mf[:, :, 0]
        c1 = torch.index_select(mf, dim=2, index=self.idx_14)
        codes_1 = c1.reshape(mf.shape[0], -1)
        c2 = torch.index_select(mf, dim=2, index=self.idx_2356)
        codes_2 = c2.reshape(mf.shape[0], -1)
        return codes_0, codes_1, codes_2

    def can_use_accelerator_graphs(self, batch, model_inputs):
        widths = {item.input_ids.shape[0] for item in model_inputs}
        return (
            len(widths) == 1 and widths <= set(self.windows)
            and super().can_use_accelerator_graphs(batch, model_inputs)
        )
