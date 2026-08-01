"""Engine submodules for Zonos2 TTS.

* :class:`Zonos2LLMSubmodule` — the autoregressive multi-codebook decoder. It
  runs :class:`Zonos2ForCausalLM` and samples a full frame at each step with
  the multi-codebook sampler. It therefore bypasses the single-token sampler of
  the engine: ``forward`` returns ``new_token``, not ``logits``. It also keeps
  the repetition history of each request, and it detects EOS across the delayed
  codebooks in ``check_stop``.

* :class:`Zonos2DACSubmodule` — a stateless audio-codec node. It takes streamed
  frames and emits PCM through the DAC vocoder.

* :class:`Zonos2SpeakerEncoderSubmodule` — a stateless voice-clone node. It
  turns the reference clip of a request into one speaker embedding. The LLM
  writes that embedding over the hidden state of the reserved speaker frame.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.cuda_graph_runner import BasicBatchedCudaGraphConfig
from mstar.engine.kv_cache_engine import BatchedCacheManager
from mstar.model.components.moe import _HAS_FUSED
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)
from mstar.model.zonos2.sampler_buffers import Zonos2SamplerBuffers
from mstar.model.zonos2.tts_sampling import TTSSamplingParams, sample_frame
from mstar.model.zonos2.vocoder import StreamingDacDecoder


class Zonos2LLMSubmodule(ARNodeSubmodule):
    """Autoregressive multi-codebook LLM wrapper.

    Prefill and decode use the same dispatch: embed the frames, run the
    transformer, then sample the per-codebook logits of the last position. The
    submodule returns ``new_token``, the sampled frame ``(1, n_codebooks + 1)``.
    """

    # The default per-step batch capacity of the sampler buffers. The eager path
    # grows it on demand, and ``get_cuda_graph_configs`` pre-sizes it to the
    # largest capture bucket.
    _DEFAULT_MAX_BS = 256

    def __init__(
        self,
        model: nn.Module,
        n_codebooks: int,
        text_vocab: int,
        eoa_id: int,
        params: TTSSamplingParams,
    ):
        super().__init__()
        self.model = model
        self.n_codebooks = n_codebooks
        self.text_vocab = text_vocab
        self.eoa_id = eoa_id
        self.params = params

        # The state of each request. The repetition history and the RNG step
        # live in slot-indexed static buffers, which are graph-safe.
        # ``preprocess`` allocates them when the device is known, or
        # ``get_cuda_graph_configs`` pre-sizes them. ``_eos`` is host-side stop
        # tracking.
        self._sampler_buffers: Zonos2SamplerBuffers | None = None
        self._eos: dict[str, dict] = {}               # EOS countdown tracking
        # The real request ids whose ``buf`` rows of this step are written but
        # not yet synced back to ``master``. The sync is deferred to the
        # ``preprocess`` of the next step, before the register and gather of
        # that step. The write-back therefore stays outside the captured graph.
        # See ``preprocess``.
        self._pending_sync_rids: list[str] | None = None

    # -- CUDA-graph capture --------------------------------------------
    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[BasicBatchedCudaGraphConfig]:
        """Declare the decode capture, which includes the multi-codebook sampler.

        The capture applies only to the fused-MoE path, because only that
        dispatch is graph-safe. The naive path runs eager. Prefill capture is
        future work. It needs a ``FlashInferPackedCudaGraphConfig`` and a static
        ``last_indices`` buffer.

        This method must have no side effect. The eligibility gate
        (``ARNodeSubmodule.can_use_cuda_graphs``) calls it with a dummy CPU
        device only to read the declared walks. ``preprocess`` therefore
        allocates the sampler buffers instead. The runner calls ``preprocess``
        on the real device during the capture warmup, before the graph records
        the buffer addresses. The ``_DEFAULT_MAX_BS`` floor covers every capture
        bucket, so ``ensure_batch_capacity`` never runs inside a capture epoch.
        """
        if not _HAS_FUSED:
            return []
        frame_w = self.n_codebooks + 1
        return [
            BasicBatchedCudaGraphConfig(
                capture_graph_walk="decode",
                requires_cfg=False,
                labels=["main"],
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(
                        1, frame_w, dtype=torch.long, device=device,
                    ),
                    input_seq_len=1,
                ),
            ),
        ]

    # -- input plumbing ------------------------------------------------
    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        seen_token_mask=None,
        pos_info: dict = {},
        **kwargs,
    ) -> ARNodeInputs:
        ids = inputs["text_inputs"][0]  # (num_frames, n_codebooks + 1)
        node_inputs = ARNodeInputs(input_ids=ids, input_seq_len=ids.shape[0])
        # Voice cloning. The embedding comes either from the speaker_encoder
        # node (the prefill_clone walk) or from process_prompt, when the caller
        # supplies a cached one. Absent and empty have the same meaning.
        speaker = inputs.get("speaker_embedding") or []
        if speaker and speaker[0] is not None and speaker[0].numel() > 0:
            node_inputs.tensor_inputs["speaker_embedding"] = speaker[0].reshape(1, -1)
        return node_inputs

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        cache_manager = engine_inputs.cache_manager
        seq_lens = [inp.input_seq_len for inp in inputs]
        cache_manager.set_active_label("main")
        cache_manager.plan_attention(seq_lens=seq_lens, is_causal=True, label="main")
        cache_manager.plan_rope(seq_lens=seq_lens, pos_ids=None, label="main")
        input_ids = torch.cat([inp.input_ids for inp in inputs], dim=0).to(
            device=self.get_device(), dtype=torch.long
        )
        # The last-frame index of each request in the packed sequence. The
        # batched forward uses it to gather the final-position logits of each
        # request. ``get_qo_indptr_buf`` exists only under CUDA-graph capture,
        # so pass the offsets through here instead.
        last_indices = torch.tensor(
            seq_lens, device=self.get_device(), dtype=torch.long
        ).cumsum(0) - 1
        # The host-side sampler lifecycle. It runs on every path (eager, capture
        # warmup, and captured replay), and always outside the graph. It
        # prepares the static buffers that the in-graph sampler in ``forward``
        # and ``forward_batched`` reads. ``padded_bs`` agrees with the logits
        # batch, which the capture can pad.
        self._prepare_sampler_step(engine_inputs, padded_bs=len(inputs))
        speaker_values, speaker_positions = self._collect_speaker_inputs(inputs, seq_lens)
        return {
            "input_ids": input_ids,
            "last_indices": last_indices,
            "speaker_emb_values": speaker_values,
            "speaker_token_positions": speaker_positions,
        }

    def _collect_speaker_inputs(
        self, inputs: list[ARNodeInputs], seq_lens: list[int],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Gather the speaker embeddings and their positions in the packed batch.

        The model injects with ``index_copy`` over the concatenated sequence, so
        a position is absolute, not per request. The speaker slot is frame 0 of
        the prompt of a cloned request, so its absolute index is the exclusive
        prefix sum of ``seq_lens`` for that request.

        The method returns ``(None, None)`` when no request in this batch clones
        a voice. This includes every decode step, where the slot is already in
        the KV cache.
        """
        values: list[torch.Tensor] = []
        positions: list[int] = []
        offset = 0
        for inp, seq_len in zip(inputs, seq_lens, strict=True):
            emb = inp.tensor_inputs.get("speaker_embedding")
            if emb is not None:
                values.append(emb.reshape(1, -1))
                positions.append(offset)
            offset += seq_len

        if not values:
            return None, None

        device = self.get_device()
        return (
            torch.cat(values, dim=0).to(device),
            torch.tensor(positions, device=device, dtype=torch.long),
        )

    def _prepare_sampler_step(
        self, engine_inputs: ModelInputsFromEngine, padded_bs: int,
    ) -> None:
        """Sync, register, and gather for this step. This never runs in a graph.

        The order is important:

        1. Sync the ``buf`` rows of the previous step back to ``master``. This
           uses the slot indices of that step. They are still in
           ``_slot_idx_gpu``, because the gather of this step runs afterwards.
        2. Register the new requests. Each one gets a master slot, and the code
           resets it.
        3. Gather the slot of every request into the per-step ``buf``.

        The sync must come before the register. A request that finishes frees a
        slot, and a new request can reuse it immediately. The reset of the new
        request (step 2) must therefore come after the deferred write-back of
        the request that left (step 1). If not, stale state overwrites the
        reset.
        """
        bufs = self._ensure_buffers(self.get_device(), padded_bs)
        # (1) Flush the writes of the previous step to master.
        if self._pending_sync_rids:
            bufs.sync_after_step(self._pending_sync_rids)
            self._pending_sync_rids = None
        # (2) Recover the real request ids. Under CUDA-graph replay,
        # ``request_ids`` holds dummy capture slots, so prefer
        # ``real_request_ids``. The ``__cg_`` filter also drops the placeholder
        # ids of the capture itself. No real request exists there, so the
        # register and the gather become no-ops onto slot 0.
        rids = engine_inputs.real_request_ids
        if rids is None:
            rids = engine_inputs.request_ids
        real_rids = [r for r in rids if not r.startswith("__cg_")]
        for rid in real_rids:
            bufs.register_request(rid)                        # idempotent
        # (3) Gather the real slots into buf[:len(real_rids)]. Padding rows use
        # slot 0.
        bufs.gather_for_request_ids(real_rids, padded_bs=padded_bs)
        # Keep the real rows for the deferred sync of the next step.
        self._pending_sync_rids = real_rids

    # -- forward + sampling --------------------------------------------
    def can_batch(self, batch, model_inputs) -> bool:
        # ``preprocess`` sets up the varlen packing and the batched FlashInfer
        # plan. The transformer forward vectorises across the packed batch.
        return True

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        speaker_emb_values: torch.Tensor | None = None,
        speaker_token_positions: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        cache_handle: BatchedCacheManager = engine_inputs.cache_manager
        hidden = self.model(
            input_ids, cache_handle, speaker_emb_values, speaker_token_positions,
        )                                                     # (num_frames, hidden)
        logits = self.model.compute_logits(hidden[-1:])       # (1, C, V)
        frame = self._sample_in_graph(logits)                 # (1, C + 1)
        return {"new_token": [frame]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        last_indices: torch.Tensor,
        speaker_emb_values: torch.Tensor | None = None,
        speaker_token_positions: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        cache_handle: BatchedCacheManager = engine_inputs.cache_manager
        hidden = self.model(
            input_ids, cache_handle, speaker_emb_values, speaker_token_positions,
        )                                                     # (total_frames, hidden)
        last_hidden = hidden.index_select(0, last_indices.to(hidden.device))
        logits = self.model.compute_logits(last_hidden)       # (B, C, V)
        frames = self._sample_in_graph(logits)                # (B, C + 1)
        return {
            rid: {"new_token": [frames[i:i + 1]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    def _sample_in_graph(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample inside the graph. This part is fixed-shape and capture-safe.

        The method reads the repetition window and the RNG step of this step
        from the static buffers that ``_prepare_sampler_step`` already gathered.
        It samples a frame, then writes the frame back into the ring. Every
        operation is fixed-shape and in place, so this runs inside the captured
        ``forward_batched`` graph. There is no host sync and no
        ``@torch.compiler.disable``.

        ``(seed, step)`` makes each request reproducible. ``step`` is the frame
        count of the request (``Zonos2SamplerBuffers.offset``). It does not
        depend on the batch position, so a batched run and a sequential run draw
        the same frame.

        The method reads the batch size from ``logits`` (``pb`` is the padded
        batch), so it needs no list of request ids. Host-side code does the
        register, the gather, and the sync.
        """
        bufs = self._sampler_buffers
        pb = logits.shape[0]
        frames = sample_frame(
            logits,
            self.params,
            repetition_token_ids=bufs.repetition_ids(pb),
            text_placeholder=self.text_vocab,
            seed=self.params.seed,
            steps=bufs.steps(pb),
        )                                                     # (pb, C + 1)
        bufs.write_frame(frames, padded_bs=pb)
        return frames

    def _ensure_buffers(self, device, padded_bs: int) -> Zonos2SamplerBuffers:
        """Allocate the per-request sampler buffers, or grow them.

        On first use the method sizes them to ``max(padded_bs,
        _DEFAULT_MAX_BS)``. ``get_cuda_graph_configs`` calls it before the
        capture with the largest capture bucket, so the buffers exist and their
        addresses are fixed before the graph records them.
        ``ensure_batch_capacity`` then grows ``buf`` only on the eager path,
        never inside a capture epoch.
        """
        if self._sampler_buffers is None:
            self._sampler_buffers = Zonos2SamplerBuffers.allocate(
                max_batch_size=max(padded_bs, self._DEFAULT_MAX_BS),
                n_codebooks=self.n_codebooks,
                window=self.params.repetition_window,
                repetition_codebooks=self.params.repetition_codebooks,
                device=device,
            )
        else:
            self._sampler_buffers.ensure_batch_capacity(padded_bs)
        return self._sampler_buffers

    # -- graph routing + stop ------------------------------------------
    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ):
        # Feed the sampled frame back as the next decode input. This is
        # metadata only. It reads no tensor value on the GPU thread.
        if "new_token" in outputs:
            outputs["text_inputs"] = outputs["new_token"]

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "new_token" not in outputs:
            return set()
        frame = outputs["new_token"][0].flatten().tolist()
        audio = frame[: self.n_codebooks]

        st = self._eos.setdefault(
            request_id, {"step": -1, "eos_frame": None, "countdown": 0}
        )
        st["step"] += 1
        # EOS detection agrees with the reference (zonos2 ``tts/sequence.py``).
        # The first frame in which any codebook emits eoa starts a delayed stop
        # countdown. Shift the aligned end frame back by the highest eoa
        # codebook index, because the shear delays that codebook by its index.
        # Then clamp the result at zero.
        if not self.params.ignore_eos and st["eos_frame"] is None:
            eos_cols = [i for i in range(self.n_codebooks) if audio[i] == self.eoa_id]
            if eos_cols:
                st["eos_frame"] = max(0, st["step"] - max(eos_cols))
                st["countdown"] = self.n_codebooks + 1
        if st["eos_frame"] is not None and st["countdown"] > 0:
            st["countdown"] -= 1

        finished = st["eos_frame"] is not None and st["countdown"] <= 0
        max_tokens = getattr(request_info, "max_tokens", None) or self.params.max_tokens
        if request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 1 >= max_tokens:
            finished = True
        return {"decode_loop"} if finished else set()

    def cleanup_request(self, request_id: str):
        if self._sampler_buffers is not None:
            self._sampler_buffers.unregister_request(request_id)
        self._eos.pop(request_id, None)


class Zonos2DACSubmodule(NodeSubmodule):
    """Stateless DAC vocoder node.

    The node takes the streamed frames of a request and emits int16 PCM chunks.
    It decodes incrementally through :class:`StreamingDacDecoder`. On the final
    chunk (``request_id in engine_inputs.final_stream_rids``) it flushes the
    withheld crossfade tail. It drops the last ``n_codebooks - 1``
    shear-alignment frames, because they hold no audio of their own.
    """

    def __init__(self, decoder: StreamingDacDecoder, n_codebooks: int):
        super().__init__()
        self.decoder = decoder
        self.n_codebooks = n_codebooks
        self.frame_width = n_codebooks + 1
        # A marker parameter, so that ``get_device`` and ``.to(device)`` work.
        # The decoder loads the DAC model itself, and it does so lazily.
        self._device_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    def get_stateless_flavor(self) -> str:
        return "audio_codec"  # fp32, no autocast, no torch.compile

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        streamed = inputs.get("new_token", [])
        if not streamed or streamed[0] is None:
            frames = torch.empty(0, self.frame_width, dtype=torch.long)
        else:
            frames = streamed[0].reshape(-1, self.frame_width)
        return NodeInputs(tensor_inputs={"frames": frames})

    def can_batch(self, batch, model_inputs) -> bool:
        # The decoder groups the windows of the same length into one DAC call.
        # The result agrees exactly with per-request decoding, so any
        # co-scheduled set is safe.
        return True

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        # Keep the frames of each request separate. Do not concatenate them.
        # The streaming decoder holds state for each request. The order agrees
        # with ``request_ids``.
        return {"frames_list": [inp.tensor_inputs["frames"] for inp in inputs]}

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        frames_list: list[torch.Tensor],
        **kwargs,
    ) -> NameToTensorList:
        rid = engine_inputs.request_ids[0]
        audio_codes = frames_list[0][:, : self.n_codebooks]
        is_final = rid in engine_inputs.final_stream_rids
        pcm = self.decoder.add_frames(rid, audio_codes, is_final=is_final)
        return {"audio_chunk": [pcm]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        frames_list: list[torch.Tensor],
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        rids = engine_inputs.request_ids
        finals = [rid in engine_inputs.final_stream_rids for rid in rids]
        codes = [f[:, : self.n_codebooks] for f in frames_list]
        out = self.decoder.add_frames_batched(rids, codes, finals)
        return {rid: {"audio_chunk": [out[rid]]} for rid in rids}

    def cleanup_request(self, request_id: str):
        self.decoder.reset(request_id)


class Zonos2SpeakerEncoderSubmodule(NodeSubmodule):
    """Stateless voice-clone node. It maps a reference clip to an embedding.

    The node runs once for each request, and only on the ``prefill_clone``
    walk. Reference clips have arbitrary lengths, so the node encodes the
    requests one at a time instead of padding them into a batch. This is a
    one-time cost for each request, not a cost for each token.
    """

    def __init__(self, encoder: nn.Module, sample_rate: int):
        super().__init__()
        self.encoder = encoder
        self.sample_rate = sample_rate

    def get_stateless_flavor(self) -> str:
        # fp32, no autocast, no torch.compile. The reference runs this encoder
        # in fp32, and the downstream projection was fit against those values.
        return "audio_codec"

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        audio = inputs.get("audio_inputs", [])
        if not audio or audio[0] is None:
            raise ValueError(
                "The speaker_encoder node ran without reference audio; the "
                "prefill_clone walk should only be selected when audio_inputs "
                "is present."
            )
        return NodeInputs(tensor_inputs={"waveform": audio[0]})

    def can_batch(self, batch, model_inputs) -> bool:
        # The node encodes each clip on its own, so co-scheduling never changes
        # a result. It only shares the dispatch.
        return True

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        # The clips have variable lengths. Keep them separate and do not
        # concatenate them.
        return {"waveforms": [inp.tensor_inputs["waveform"] for inp in inputs]}

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        waveforms: list[torch.Tensor],
        **kwargs,
    ) -> NameToTensorList:
        return {"speaker_embedding": [self._embed(waveforms[0])]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        waveforms: list[torch.Tensor],
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        return {
            rid: {"speaker_embedding": [self._embed(waveforms[i])]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    def _embed(self, waveform: torch.Tensor) -> torch.Tensor:
        # ``Zonos2Model.load_audio`` already decoded at the rate of the encoder,
        # so this call normally does no resampling.
        return self.encoder(waveform, self.sample_rate)
