"""The shared AuT encoder node (Qwen3-ASR, Qwen3-Omni): the packed AuT
encoder over every request's audio, its attention windows declared as
segments of a cacheless ragged attention resource."""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    PiecewiseAcceleratorGraphConfig,
    PiecewiseCallInputs,
    PiecewiseCaptureShape,
    PiecewisePackedConfig,
)
from mstar.engine.resources import (
    AttentionStep,
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    Segment,
    SlotLease,
    SubmoduleStep,
)
from mstar.model.components.audio_features import LogMelSpectrogram
from mstar.model.components.aut_encoder import AuTEncoder, AuTEncoderConfig
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule

if TYPE_CHECKING:
    from mstar.engine.engine import ExecutingBatch

# the node's cacheless windowed attention
AUT_ATTN = "aut_attn"
WINDOW_LABEL = "main"
# The AuT encoder's transformer stack and projector, as a piecewise capture region
AUT_ENCODE = "aut_encode"


def aut_attention_spec(config: AuTEncoderConfig, nodes: set[str], max_frames: int) -> RaggedAttentionSpec:
    """The node's attention resource: every attention window of a clip of up
    to ``max_frames`` mel frames is its own segment."""
    return RaggedAttentionSpec(
        resource_key=AUT_ATTN, nodes=nodes,
        config=RaggedAttentionConfig(
            num_qo_heads=config.encoder_attention_heads,
            num_kv_heads=config.encoder_attention_heads,
            head_dim=config.head_dim,
            max_segments_per_request=-(-max_frames // config.n_window_infer),
        ),
    )


def load_aut_encoder_submodule(
    local_dir: str, config: AuTEncoderConfig, device: str, autocast_dtype: torch.dtype | None,
    log_mel: LogMelSpectrogram | None = None,
) -> AuTEncoderSubmodule:
    """The node over the checkpoint's ``thinker.audio_tower`` weights."""
    from mstar.model.loader.iterators import iter_safetensors_shards

    with torch.device("meta"):
        encoder = AuTEncoder(config, attn_key=AUT_ATTN)
    if autocast_dtype is not None:
        encoder = encoder.to(autocast_dtype)
    encoder.to_empty(device=device)
    prefix = "thinker.audio_tower."
    weights = iter_safetensors_shards(local_dir, device=device, prefix=prefix)
    encoder.load_weights((k.removeprefix(prefix), v) for k, v in weights)
    return AuTEncoderSubmodule(encoder.eval(), log_mel)


class AuTEncoderSubmodule(NodeSubmodule):
    """Packed AuT encoder.

    Consumes one clip of samples per request (Qwen3-ASR's ``process_prompt``),
    turns it into unpadded log-mel features on the GPU (``prepare_inputs``)
    and emits ``audio_embeds`` of shape
    ``(num_audio_tokens, hidden)`` in LLM space. Requests are packed into
    one forward, up to the largest capture's tokens; every 8 s attention
    window of every request is one segment of the node's ragged-attention
    step, so the runner plans the kernel's layout before the forward and
    nothing is recomputed inside.

    Built without a ``log_mel`` it takes each clip's log-mel features as
    given (``audio_features``): Qwen3-Omni's audio encoder is the same AuT,
    fed by the API server's feature extractor.
    """

    MAX_BATCH_SIZE = 8
    # The engine's blanket torch.compile of ``forward_batched`` traces the
    # per-request Python around the tensor work and guards on request ids
    # and batch composition, so batches kept recompiling (1-2 s stalls each,
    # measured on Whisper's decoder). The encoder's kernels are the ragged
    # attention and cuBLAS GEMMs, there is nothing for inductor to fuse.
    disable_torch_compile = True

    def __init__(self, encoder: AuTEncoder, log_mel: LogMelSpectrogram | None = None):
        super().__init__()
        self.encoder = encoder
        # on the materialized encoder's device, and float32 even under a bf16
        # default dtype
        self.log_mel = None if log_mel is None else log_mel.to(
            device=encoder.conv2d1.weight.device, dtype=torch.float32,
        )

    def _mel(self, audio: torch.Tensor) -> torch.Tensor:
        """The spectrogram in float32 whatever dtype the engine cast this
        module to (it casts submodules to the compute dtype after they are
        built; a bf16 STFT would not match the reference features)."""
        if self.log_mel.window.dtype != torch.float32:
            self.log_mel.float()
        return self.log_mel(audio)

    def _param_dtype(self) -> torch.dtype:
        return self.encoder.conv2d1.weight.dtype

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        if self.log_mel is None:
            feats = inputs["audio_features"][0].to(device=self.get_device(), dtype=self._param_dtype())
        else:
            audio = inputs["audio"][0].reshape(-1).to(device=self.get_device(), dtype=torch.float32)
            feats = self._mel(audio).to(self._param_dtype())  # (num_mel_bins, T)
        return NodeInputs(
            tensor_inputs={"audio_features": feats},
            input_seq_len=self.encoder.config.tokens_for_frames(feats.shape[-1]),
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep | None:
        # leased, the captured stack plans its own attention per replay
        if (piecewise_leases or {}).get(AUT_ENCODE):
            return None
        # one segment per attention window, in the packed order the forward
        # lays the requests out in
        segments = [
            Segment(request_id=rid, label=WINDOW_LABEL, span=window)
            for rid, inp in zip(request_ids, inputs, strict=True)
            for window in self.encoder.config.window_lengths(inp.input_seq_len)
        ]
        return SubmoduleStep(segments=segments, steps={AUT_ATTN: AttentionStep(causal=False)})

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        # the clips as they are; the frontend packs their frames
        return {"audio_features": [inp.tensor_inputs["audio_features"] for inp in inputs]}

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype,
        tp_world_size: int = 1, **kwargs,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        """The encoder's stack and projector, by batch size and packed
        token count. At ASR clip lengths (13 tokens a second) its GEMMs are a
        few dozen rows, so eager it is launch-bound: a 7 s clip's stack is
        ~6.3 ms eager and ~1.9 ms replayed. The conv frontend before it runs
        eagerly. A clip past the largest bucket runs eagerly, alone."""
        d_model = self.encoder.config.d_model

        def make_static_inputs(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
            return {"hidden": torch.zeros(shape.total_tokens, d_model, dtype=autocast_dtype, device=device)}

        def declare_step(request_ids: list[str], seq_lens: list[int]) -> SubmoduleStep:
            # Every window of every request is a segment; the kernel needs
            # only the boundaries, so they all name the first row's id.
            # Zero-length rows pad the batch to the bucket.
            return SubmoduleStep(
                segments=[Segment(request_id=request_ids[0], label=WINDOW_LABEL, span=n) for n in seq_lens],
                steps={AUT_ATTN: AttentionStep(causal=False)},
            )

        return {
            AUT_ENCODE: PiecewisePackedConfig(
                capture_fn=self._capture_encode,
                make_static_inputs=make_static_inputs,
                declare_step=declare_step,
                lease_before_step=True,
                total_tokens=self.ENCODE_TOKEN_BUCKETS,
                capture_batch_sizes=self.ENCODE_CAPTURE_BATCH_SIZES,
            )
        }

    # Packed token counts the stack is captured at, and batch sizes (up to
    # MAX_BATCH_SIZE). 4096 tokens is 8 requests of ~39 s.
    ENCODE_TOKEN_BUCKETS = [64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    ENCODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]

    def _capture_encode(self, inp: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        """The captured region: reads the runner's buffer, never rebinds."""
        return {"embeds": self.encoder.encode(inp.static_inputs["hidden"], None)}

    @torch.compiler.disable
    def _replay_encode(
        self, engine_inputs: ModelInputsFromEngine, hidden: torch.Tensor, windows: list[int],
    ) -> torch.Tensor | None:
        """The stack replayed for the whole packed batch, or None if the step
        did not lease it."""
        runner = engine_inputs.piecewise_runners.get(AUT_ENCODE)
        if runner is None:
            return None
        bs = len(engine_inputs.request_ids)
        # an owned copy: the encoder states leave this worker after the step,
        # and the buffer is the next replay's
        return runner.run(
            static_inputs={"hidden": hidden},
            request_ids=list(engine_inputs.request_ids),
            seq_lens=windows,
            real_bs=bs,
        ).get("embeds")

    def _encode(
        self, audio_features: list[torch.Tensor],
        engine_inputs: ModelInputsFromEngine | None = None,
    ) -> list[torch.Tensor]:
        enc = self.encoder
        layout = enc.layout([clip.shape[-1] for clip in audio_features])
        hidden = enc.frontend(audio_features)
        embeds = None
        if engine_inputs is not None:
            embeds = self._replay_encode(engine_inputs, hidden, layout.window_lengths)
        if embeds is None:
            embeds = enc.encode(hidden, layout.cu_seqlens)
        return list(embeds.split(layout.tokens_per_request, dim=0))

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: list[torch.Tensor],
        **kwargs,
    ) -> NameToTensorList:
        return {"audio_embeds": [self._encode(audio_features, engine_inputs)[0]]}

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        # A packed step holds at most the largest capture's tokens. Past it the
        # clips run one by one, each replayed if it fits a bucket, so an eager
        # step's activations are one clip's rather than the whole batch's.
        tokens = sum(inp.input_seq_len for inp in model_inputs)
        return len(model_inputs) <= 1 or tokens <= self.ENCODE_TOKEN_BUCKETS[-1]

    def max_batch_size(self, graph_walk: str) -> int | None:
        del graph_walk
        return self.MAX_BATCH_SIZE

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: list[torch.Tensor],
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        per_request = self._encode(audio_features, engine_inputs)
        return {
            rid: {"audio_embeds": [emb]}
            for rid, emb in zip(engine_inputs.request_ids, per_request, strict=True)
        }
