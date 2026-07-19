"""Engine submodules for Zonos2 TTS.

* :class:`Zonos2LLMSubmodule` — the autoregressive multi-codebook decoder.
  It runs :class:`Zonos2ForCausalLM`, samples a full frame per step with the
  custom multi-codebook sampler (so the engine's single-token sampler is
  bypassed — ``forward`` returns ``new_token`` directly, not ``logits``),
  tracks per-request repetition history, and detects EOS across the delayed
  codebooks in ``check_stop``.

* :class:`Zonos2DACSubmodule` — a stateless audio-codec node that consumes
  streamed frames and emits PCM via the DAC vocoder.
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

    Dispatches prefill / decode the same way (embed frames -> transformer ->
    sample the last position's per-codebook logits). Returns ``new_token``:
    the sampled frame ``(1, n_codebooks + 1)``.
    """

    # Default per-step batch capacity for the lazily-allocated sampler buffers
    # (grown on demand in the eager path; Phase 3 pre-sizes to the capture max).
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

        # Per-request state. Repetition history + RNG step live in slot-indexed
        # static buffers (graph-safe); allocated lazily in ``preprocess`` (or
        # pre-sized in ``get_cuda_graph_configs``) once the device is known.
        # ``_eos`` is host-side stop tracking.
        self._sampler_buffers: Zonos2SamplerBuffers | None = None
        self._eos: dict[str, dict] = {}               # EOS countdown tracking
        # Real request ids whose per-step ``buf`` rows are written but not yet
        # synced back to ``master``. Phase 3 defers the sync to the *next*
        # step's ``preprocess`` (before that step's register/gather) so the
        # write-back stays outside the captured graph. See ``preprocess``.
        self._pending_sync_rids: list[str] | None = None

    # -- CUDA-graph capture --------------------------------------------
    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[BasicBatchedCudaGraphConfig]:
        """Declare the decode capture, with the multi-codebook sampler folded
        into the captured ``forward_batched`` (Phase 3).

        Gated on the fused-MoE path: only that dispatch is proven graph-safe
        (Phase 1); the naive path is left to run eager. Prefill capture is a
        follow-on (needs a ``FlashInferPackedCudaGraphConfig`` plus a static
        ``last_indices`` buffer).

        Must stay side-effect-free: the eligibility gate
        (``ARNodeSubmodule.can_use_cuda_graphs``) calls this with a dummy CPU
        device just to read the declared walks. The sampler buffers are instead
        allocated lazily in ``preprocess`` — which the runner invokes on the real
        device during capture warmup, before the graph records their addresses —
        and their ``_DEFAULT_MAX_BS`` floor already covers every capture bucket,
        so ``ensure_batch_capacity`` never fires inside a capture epoch.
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
        return ARNodeInputs(input_ids=ids, input_seq_len=ids.shape[0])

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
        # Per-request last-frame index in the packed sequence — used by the
        # batched forward to gather each request's final-position logits.
        # (``get_qo_indptr_buf`` only exists under CUDA-graph capture, so we
        # thread the offsets through here instead.)
        last_indices = torch.tensor(
            seq_lens, device=self.get_device(), dtype=torch.long
        ).cumsum(0) - 1
        # Host-side sampler lifecycle (runs in every path — eager, capture warmup
        # and captured replay — always outside the graph). Prepares the static
        # buffers that the in-graph sampler in ``forward``/``forward_batched``
        # reads; ``padded_bs`` matches the (possibly capture-padded) logits batch.
        self._prepare_sampler_step(engine_inputs, padded_bs=len(inputs))
        return {"input_ids": input_ids, "last_indices": last_indices}

    def _prepare_sampler_step(
        self, engine_inputs: ModelInputsFromEngine, padded_bs: int,
    ) -> None:
        """Deferred-sync + lazy-register + gather for this step (never captured).

        The ordering is load-bearing:

        1. **sync** the *previous* step's ``buf`` rows back to ``master`` (using
           that step's slot indices, still resident in ``_slot_idx_gpu`` because
           this step's gather runs afterwards),
        2. **register** any new requests (assigns + resets a master slot),
        3. **gather** every request's slot into the per-step ``buf``.

        Sync MUST precede register: when a finishing request frees a slot that a
        new request immediately reuses, the new request's fresh reset (step 2)
        must land *after* the departing request's deferred write-back (step 1),
        or the reset is clobbered by stale state.
        """
        bufs = self._ensure_buffers(self.get_device(), padded_bs)
        # (1) flush the previous step's writes to master.
        if self._pending_sync_rids:
            bufs.sync_after_step(self._pending_sync_rids)
            self._pending_sync_rids = None
        # (2) recover the real request ids. Under CUDA-graph replay
        # ``request_ids`` holds dummy capture slots, so prefer
        # ``real_request_ids``; the ``__cg_`` filter additionally drops the
        # placeholder ids seen during capture itself (no real request exists
        # there — register/gather become no-ops onto slot 0).
        rids = engine_inputs.real_request_ids
        if rids is None:
            rids = engine_inputs.request_ids
        real_rids = [r for r in rids if not r.startswith("__cg_")]
        for rid in real_rids:
            bufs.register_request(rid)                        # idempotent lazy join
        # (3) gather real slots into buf[:len(real_rids)]; padding rows -> slot 0.
        bufs.gather_for_request_ids(real_rids, padded_bs=padded_bs)
        # Retain the real rows for the next step's deferred sync.
        self._pending_sync_rids = real_rids

    # -- forward + sampling --------------------------------------------
    def can_batch(self, batch, model_inputs) -> bool:
        # Varlen packing + batched FlashInfer plan is set up in ``preprocess``;
        # the transformer forward vectorises across the packed batch.
        return True

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        cache_handle: BatchedCacheManager = engine_inputs.cache_manager
        hidden = self.model(input_ids, cache_handle)          # (num_frames, hidden)
        logits = self.model.compute_logits(hidden[-1:])       # (1, C, V)
        frame = self._sample_in_graph(logits)                 # (1, C + 1)
        return {"new_token": [frame]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        last_indices: torch.Tensor,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        cache_handle: BatchedCacheManager = engine_inputs.cache_manager
        hidden = self.model(input_ids, cache_handle)          # (total_frames, hidden)
        last_hidden = hidden.index_select(0, last_indices.to(hidden.device))
        logits = self.model.compute_logits(last_hidden)       # (B, C, V)
        frames = self._sample_in_graph(logits)                # (B, C + 1)
        return {
            rid: {"new_token": [frames[i:i + 1]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    def _sample_in_graph(self, logits: torch.Tensor) -> torch.Tensor:
        """In-graph portion of sampling — fixed-shape, capture-safe.

        Reads the per-step repetition window + RNG step from the static buffers
        that ``preprocess`` (via ``_prepare_sampler_step``) already gathered,
        samples a frame, and writes it back into the ring. All ops are
        fixed-shape / in-place, so this runs *inside* the captured
        ``forward_batched`` graph — no host sync, no ``@torch.compiler.disable``.

        Reproducibility is per-request via ``(seed, step)`` where ``step`` is the
        request's frame count so far (``Zonos2SamplerBuffers.offset``),
        independent of batch position, so batched and sequential draw identically.

        The batch size is read from ``logits`` (``pb`` = padded batch), so this
        needs no request-id list: register/gather/sync are handled host-side.
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
        """Lazily allocate (and grow) the per-request sampler buffers.

        Sized to ``max(padded_bs, _DEFAULT_MAX_BS)`` on first use.
        ``get_cuda_graph_configs`` calls this ahead of capture with the largest
        capture bucket so the buffers exist (and their addresses are fixed)
        before the graph is recorded; ``ensure_batch_capacity`` then only ever
        grows ``buf`` on the eager path, never inside a capture epoch.
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
        # Feed the sampled frame back as the next decode input. Metadata-only
        # (no tensor value reads on the GPU thread).
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
        # EOS detection matches the reference (zonos2 ``tts/sequence.py``): the
        # first frame in which *any* codebook emits eoa starts a delayed stop
        # countdown. The aligned end frame is shifted back by the highest eoa
        # codebook index (that codebook is delayed by its index under the
        # inter-codebook shear) and clamped at zero.
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

    Consumes streamed frames (per request) and emits int16 PCM chunks. Runs
    incrementally via :class:`StreamingDacDecoder`. On the final
    chunk (``request_id in engine_inputs.final_stream_rids``) it flushes the
    withheld crossfade tail; the trailing ``n_codebooks - 1`` shear-alignment
    frames carry no audio of their own and are dropped.
    """

    def __init__(self, decoder: StreamingDacDecoder, n_codebooks: int):
        super().__init__()
        self.decoder = decoder
        self.n_codebooks = n_codebooks
        self.frame_width = n_codebooks + 1
        # Marker parameter so ``get_device`` / ``.to(device)`` work (the DAC
        # model itself is loaded lazily inside the decoder).
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
        # The decoder groups same-length windows into one DAC call (and matches
        # per-request decoding bit-for-bit), so any co-scheduled set is safe.
        return True

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        # Keep each request's frames separate (not concatenated): the streaming
        # decoder is stateful per request. Order matches ``request_ids``.
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
