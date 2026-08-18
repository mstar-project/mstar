import logging
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.base import NodeBatch
from mstar.engine.kv_store import PositionInfo
from mstar.engine.resources.step import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SubmoduleStep
from mstar.engine.v1.attention_manager import AttentionManager
from mstar.engine.v1.cuda_graph_config import BatchedCudaGraphConfig, CudaGraphConfig, PackedCudaGraphConfig
from mstar.engine.v1.sampler import SamplerResource
from mstar.model.orpheus.config import ATTN, KV_CACHE, ROPE, SAMPLER, OrpheusModelConfig
from mstar.model.submodule_base import ARNodeInputs, ARNodeSubmodule, ModelInputsFromEngine, NodeInputs, NodeSubmodule

logger = logging.getLogger(__name__)


class OrpheusLLMSubmodule(ARNodeSubmodule):
    """Llama 3.2 3B wrapper for Orpheus TTS.

    Dispatches on graph_walk:
      - prefill: embed text tokens, fill KV cache
      - decode: embed previous token, generate next audio token
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

    # PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024]
    # PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]

    # TODO DEBUG
    PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4]

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[CudaGraphConfig]:
        return [
            BatchedCudaGraphConfig(
                capture_graph_walk="decode",
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device),
                    input_seq_len=1
                ),
            ),
            PackedCudaGraphConfig(
                capture_graph_walk="prefill",
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
        pos_info: dict[str, PositionInfo] = {},
        **kwargs,
    ) -> ARNodeInputs:
        return ARNodeInputs(
            input_ids=inputs["text_inputs"][0],
            input_seq_len=inputs["text_inputs"][0].shape[0]
        )

    def declare_step(
        self, graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs]
    ):
        prefill_tokens = {}
        if graph_walk == "prefill":
            prefill_tokens = {
                rid: inp.input_ids for rid, inp in zip(request_ids, inputs)
            }
        return SubmoduleStep(
            segments=[
                Segment(
                    request_id=rid,
                    label="main",
                    span=inp.input_seq_len,
                ) for rid, inp in zip(request_ids, inputs)
            ],
            steps={
                KV_CACHE: KVStep(),
                ATTN: AttentionStep(causal=True),
                SAMPLER: SamplerStep(
                    apply_penalty=True,
                    prefill_tracked_tokens=prefill_tokens
                ),
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

        if graph_walk == "prefill":
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
        self, batch: NodeBatch,
        model_inputs: list[NodeInputs]
    ) -> bool:
        return True

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        text_inputs: torch.Tensor,
        **kwargs
    ) -> dict[str, NameToTensorList]:
        new_tokens = self._forward(
            graph_walk=graph_walk,
            engine_inputs=engine_inputs,
            text_inputs=text_inputs
        )
        return {
            rid: {"new_token": [new_tokens[i : i + 1]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs
    ):
        # Metadata-only: rebind output name for graph routing. EOS check
        # moved to check_stop so the GPU thread doesn't sync on .item() here.
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

    Receives a window of raw audio token tensors from StreamBuffer
    (via normal graph input routing), converts to SNAC codes, and
    decodes the middle region of the audio for low-latency output.

    Supports batched inference: multiple requests are decoded in a
    single SNAC forward pass when all windows have the same frame count.
    """

    def __init__(self, snac_model: nn.Module, config: OrpheusModelConfig):
        super().__init__()
        self.snac_model = snac_model
        device = next(self.snac_model.parameters()).device
        self.idx_14 = torch.tensor([1, 4], dtype=torch.long, device=device)
        self.idx_2356 = torch.tensor([2, 3, 5, 6], dtype=torch.long, device=device)
        self.config = config

        self._orig_seq_len = {}

    def get_stateless_flavor(self) -> str:
        # SNAC runs in fp32 with no autocast and no torch.compile.
        return "audio_codec"

    # _tokens_to_codes pads to multiples of 28 tokens then reshapes to
    # (N_frames, 4, 7); for a single streaming window this is 1 frame.
    @property
    def _num_frames(self) -> int:
        return self.config.snac_window_tokens // (4 * self.config.tokens_per_frame)

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1
    ) -> list[CudaGraphConfig]:
        """Declare the SNAC decode capture.
        """
        # One streaming window is ``snac_window_tokens`` raw tokens
        # (4 frames × 7 codes). Make the dummy tokens a multiple of 28 so
        # ``_tokens_to_codes`` doesn't take the pad branch during capture
        # (the pad branch creates a fresh tensor that would be graph-time
        # churn we don't want in the cached trace).
        tokens_per_window = self.config.snac_window_tokens
        dummy = ARNodeInputs(
            input_ids=torch.zeros((1, 4, 7), dtype=torch.long, device=device),
            input_seq_len=tokens_per_window
        )
        return [
            BatchedCudaGraphConfig(
                capture_graph_walk="snac_chunk",
                single_request_inputs=dummy,
                caps_eager_batch_size=[1, 2, 4, 8, 16]
            ),
        ]

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> ARNodeInputs:
        tokens = inputs["new_token"][0].flatten()
        self._orig_seq_len[fwd_info.request_id] = tokens.shape[0]
        return ARNodeInputs(
            input_ids=self._tokens_to_codes(tokens),
            input_seq_len=tokens.shape[0]
        )

    def can_batch(self, batch: NodeBatch, inputs: list[ARNodeInputs]) -> bool:
        return True

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        stacked = torch.stack([
            input.input_ids for input in inputs
        ], dim=0)
        B, N = stacked.shape[0], stacked.shape[1]
        flat = stacked.reshape(B * N, 4, 7)
        codes_0, codes_1, codes_2 = self._extract_snac_codes(flat)
        return {
            "codes_0": codes_0.reshape(B, N * 4),
            "codes_1": codes_1.reshape(B, N * 8),
            "codes_2": codes_2.reshape(B, N * 16),
        }

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        codes_0: torch.Tensor,
        codes_1: torch.Tensor,
        codes_2: torch.Tensor,
    ) -> NameToTensorList:
        audio_hat = self.snac_model.decode([codes_0, codes_1, codes_2])
        audio_slice = audio_hat[
            :, :, self.config.snac_audio_slice_start:self.config.snac_audio_slice_end,
        ]
        audio_int16 = (audio_slice.clamp(-1, 1) * 32767).to(torch.int16)
        return {"audio_chunk": audio_int16.squeeze(1)}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        codes_0: torch.Tensor,
        codes_1: torch.Tensor,
        codes_2: torch.Tensor,
    )  -> dict[str, NameToTensorList]: # request_id to tensors
        stacked = self.forward(
            graph_walk=graph_walk,
            engine_inputs=engine_inputs,
            codes_0=codes_0,
            codes_1=codes_1,
            codes_2=codes_2
        )

        return {
            rid: {name: [stacked[name][i].detach()] for name in stacked}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    def _tokens_to_codes(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pad raw token IDs to a multiple of 28 and convert to SNAC codes.

        Args:
            tokens: flat 1-D token tensor for one request.

        Returns:
            SNAC codes tensor of shape ``(num_frames, 4, 7)``.
        """
        remainder = tokens.numel() % 28
        if remainder != 0:
            pad_len = 28 - remainder
            pad = tokens[-1].repeat(pad_len)
            tokens = torch.cat([tokens, pad], dim=0)

        tokens = tokens.view(-1, 4, 7)
        return (tokens - self.config.custom_token_base_id - 10) % 4096

    def _extract_snac_codes(self, mf: torch.Tensor):
        """Split (N, 4, 7) codes into the three codebook levels.

        Returns:
            (codes_0, codes_1, codes_2) with shapes
            ``(N, 4)``, ``(N, 8)``, ``(N, 16)``.
        """
        codes_0 = mf[:, :, 0]
        c1 = torch.index_select(mf, dim=2, index=self.idx_14)
        codes_1 = c1.reshape(mf.shape[0], -1)
        c2 = torch.index_select(mf, dim=2, index=self.idx_2356)
        codes_2 = c2.reshape(mf.shape[0], -1)
        return codes_0, codes_1, codes_2

    def can_use_cuda_graphs(self, batch, model_inputs):
        return super().can_use_cuda_graphs(batch, model_inputs) \
            and self.can_batch(batch, model_inputs)
