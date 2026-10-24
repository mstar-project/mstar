# Copyright (c) 2026 OpenBMB.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified for M* in 2026: the reference generate loop
# split into M* graph-node submodules with continuous batching and streamed
# VAE windows.
#
# Original files (OpenBMB/VoxCPM, src/voxcpm/model/voxcpm2.py and
# src/voxcpm/core.py) were released under Apache-2.0, with the full license
# text available at https://github.com/OpenBMB/VoxCPM/blob/main/LICENSE.
#
# This modified file is released under the same license.

# ---------------------------------------------------------------------------
# NodeSubmodule wrappers for VoxCPM2
# ---------------------------------------------------------------------------
#
#   1. RefEncoderSubmodule (no resources; voice cloning only)
#      - AudioVAE encoder over one 16 kHz reference clip -> latent patches,
#        once right-padded (reference prefix) and, for continuation, once
#        left-padded (the audio the model continues), as the reference pads.
#   2. TalkerSubmodule (base LM + residual LM paged KV, positions)
#      - Prefill: text (+ reference / continuation patches) through both LMs,
#        then the first patch from the local DiT.
#      - Decode: one patch per step; the CFM solve (9 DiT passes with CFG) runs
#        inside the captured decode graph.
#      - Streams every patch to the VAE partition; stops on the stop head.
#   3. VAEDecoderSubmodule (no resources)
#      - Decodes each stream window (left context + new patches) to 48 kHz and
#        drops the context's audio. The context covers the decoder's whole
#        causal receptive field, so a stream equals a whole decode.
# ---------------------------------------------------------------------------

from __future__ import annotations

import itertools
import logging
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig, BatchedAcceleratorGraphConfig
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, Segment, SlotLease, SubmoduleStep
from mstar.model.submodule_base import (
    EMPTY_INPUT_METADATA,
    ARNodeInputs,
    ARNodeSubmodule,
    InputMetadata,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)
from mstar.model.voxcpm2.components.audio_vae import AudioVAE
from mstar.model.voxcpm2.components.local import cfm_t_span
from mstar.model.voxcpm2.components.talker import VoxCPM2Generator
from mstar.model.voxcpm2.config import (
    BASE_ATTN,
    BASE_KV,
    BASE_POS,
    MIN_LEN_KEY,
    MODE_REF_CONTINUATION,
    MODE_REFERENCE,
    MODE_ZERO_SHOT,
    PATCH_BUDGET_KEY,
    PROMPT_MODE_KEY,
    REF_AUDIO_END_TOKEN,
    REF_AUDIO_START_TOKEN,
    REF_DIGEST_KEY,
    RES_ATTN,
    RES_KV,
    VoxCPM2Config,
)
from mstar.utils.h2d import PinnedStager

logger = logging.getLogger(__name__)

TALKER_LOOP = "talker_decode_loop"
PREFILL_WALKS = ("talker_prefill", "talker_prefill_clone")


def build_prompt(
    text_ids: torch.Tensor,
    mode: int,
    ref_patches: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The prefill layout of ``_generate_with_prompt_cache`` for one request.

    ``text_ids`` is the tokenized (prompt text +) target text followed by the
    audio-start token. ``ref_patches`` is ``[k, n, patch, feat]``: the
    reference encoding (``k`` = 1) and, for continuation, the prompt encoding
    (``k`` = 2). Returns ``(token_ids, text_mask, audio_patches, audio_pos)``:
    ``audio_pos`` indexes the audio positions, ``audio_patches`` their latents.
    """
    n_text = text_ids.shape[0]
    device = text_ids.device
    if mode == MODE_ZERO_SHOT:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return text_ids, torch.ones(n_text, dtype=torch.bool, device=device), None, empty
    if ref_patches is None:
        raise ValueError("VoxCPM2 voice cloning needs the reference clip's latents")
    ref = ref_patches[0]
    n_ref = ref.shape[0]
    pieces = [
        text_ids.new_full((1,), REF_AUDIO_START_TOKEN),
        text_ids.new_zeros(n_ref),
        text_ids.new_full((1,), REF_AUDIO_END_TOKEN),
        text_ids,
    ]
    audio = [ref]
    audio_pos = [torch.arange(1, 1 + n_ref, device=device)]
    if mode == MODE_REF_CONTINUATION:
        prompt = ref_patches[1]
        start = n_ref + 2 + n_text
        pieces.append(text_ids.new_zeros(prompt.shape[0]))
        audio.append(prompt)
        audio_pos.append(torch.arange(start, start + prompt.shape[0], device=device))
    elif mode != MODE_REFERENCE:
        raise ValueError(f"unknown VoxCPM2 prompt mode {mode}")
    token_ids = torch.cat(pieces)
    audio_pos = torch.cat(audio_pos)
    # index_fill_, not ``text_mask[audio_pos] = False``: that copies the scalar H2D, a sync
    text_mask = torch.ones(token_ids.shape[0], dtype=torch.bool, device=device).index_fill_(0, audio_pos, False)
    return token_ids, text_mask, torch.cat(audio), audio_pos


# ===========================================================================
# 1. RefEncoderSubmodule
# ===========================================================================


class RefEncoderSubmodule(NodeSubmodule):
    """One reference clip -> ``[k, n, patch, feat]`` latent patches.

    Runs once per cloning request, before the clone prefill, in fp32 like the
    reference's AudioVAE. The reference pads the clip to whole patches on the
    right for the isolated reference prefix and on the left for the
    continuation prompt; ``k`` is 1 or 2 accordingly. Encodings are memoised
    by clip content and mode, since a voice is usually reused; the content
    hash is ``reference_digest``, taken in ``process_prompt`` on the host copy.
    """

    disable_torch_compile = True
    disable_autocast = True
    CACHE_SIZE = 64

    def __init__(self, vae: AudioVAE, config: VoxCPM2Config):
        super().__init__()
        self.vae = vae
        self.config = config
        self._cache: OrderedDict[tuple[str, int], torch.Tensor] = OrderedDict()

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs: Any
    ) -> NodeInputs:
        del graph_walk, kwargs
        meta = fwd_info.step_metadata
        clip = inputs["audio_inputs"][0].to(device=self.get_device(), dtype=torch.float32).reshape(-1)
        mode = int(meta[PROMPT_MODE_KEY])
        return NodeInputs(tensor_inputs={"clip": clip}, kwargs={"mode": mode, "key": (meta[REF_DIGEST_KEY], mode)})

    def encode(self, clip: torch.Tensor, mode: int) -> torch.Tensor:
        patch, feat = self.config.patch_size, self.config.feat_dim
        pad = -clip.shape[-1] % self.config.ref_patch_samples
        sides = [(0, pad)] + ([(pad, 0)] if mode == MODE_REF_CONTINUATION else [])
        encodings = []
        for side in sides:
            latents = self.vae.encode(torch.nn.functional.pad(clip, side).unsqueeze(0))
            encodings.append(latents.view(feat, -1, patch).permute(1, 2, 0))
        return torch.stack(encodings)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        clip: torch.Tensor,
        mode: int = MODE_REFERENCE,
        key: tuple | None = None,
        **kwargs: Any,
    ) -> NameToTensorList:
        del graph_walk, engine_inputs, kwargs
        if key in self._cache:
            self._cache.move_to_end(key)
            return {"ref_patches": [self._cache[key]]}
        with torch.autocast(device_type=clip.device.type, enabled=False):
            patches = self.encode(clip, mode)
        self._cache[key] = patches
        while len(self._cache) > self.CACHE_SIZE:
            self._cache.popitem(last=False)
        return {"ref_patches": [patches]}


# ===========================================================================
# 2. TalkerSubmodule
# ===========================================================================


class TalkerSubmodule(ARNodeSubmodule):
    """Both LMs and the patch heads; one latent patch per AR step.

    Runs in the checkpoint's bf16 without autocast, as the reference does:
    autocast would lift the CFM's reductions and the timestep embedding's
    ``exp`` to fp32 and move the whole trajectory off the reference's.

    Per-request CFM noise comes from a CUDA generator seeded with the
    request's seed and drawn one ``[1, feat, patch]`` tensor per step, which is
    the reference's own sequence of ``torch.randn`` calls after
    ``torch.manual_seed(seed)``; it rides into the step as an input, so
    batching and graph replay leave it alone.
    """

    disable_torch_compile = True
    disable_autocast = True
    MAX_BATCH_SIZE = 32
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]

    def __init__(
        self,
        generator: VoxCPM2Generator,
        config: VoxCPM2Config,
        inference_timesteps: int,
        compile_decode: bool = True,
    ):
        super().__init__()
        self.model = generator
        self.config = config
        self.inference_timesteps = inference_timesteps
        self.compile_decode = compile_decode
        self._t_span: torch.Tensor | None = None
        self._index_stager: PinnedStager | None = None

    def _dtype(self) -> torch.dtype:
        return self.model.fusion_concat_proj.weight.dtype

    def t_span(self) -> torch.Tensor:
        if self._t_span is None:
            self._t_span = cfm_t_span(self.inference_timesteps, self.get_device(), self._dtype())
        return self._t_span

    # -- per-request noise -------------------------------------------------

    def _noise(self, state, step: int) -> torch.Tensor:
        """The ``step``-th CFM noise draw of this request, drawn in order and
        kept until a later step is asked for. One launch per row (no host
        sync): each request has its own seeded stream, which one batched draw
        could not reproduce."""
        draws: dict[int, torch.Tensor] = state["noise"]
        drawn = int(state["drawn"])
        while drawn <= step:
            draws[drawn] = torch.randn(
                (1, self.config.feat_dim, self.config.patch_size),
                generator=state["generator"],
                device=self.get_device(),
                dtype=self._dtype(),
            )[0]
            drawn += 1
        state.add("drawn", drawn)
        for old in [k for k in draws if k < step]:
            del draws[old]
        return draws[step]

    # -- prepare / preprocess ----------------------------------------------

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs: Any
    ) -> ARNodeInputs | None:
        del kwargs
        device = self.get_device()
        state = self.request_state(fwd_info.rid_handle)
        if graph_walk in PREFILL_WALKS:
            meta = fwd_info.step_metadata
            mode, max_patches, min_len = (int(meta[k]) for k in (PROMPT_MODE_KEY, PATCH_BUDGET_KEY, MIN_LEN_KEY))
            params = inputs["gen_params"][0].to(device=device, dtype=torch.float32).reshape(-1)
            text_ids = inputs["text_inputs"][0].to(device=device, dtype=torch.long).reshape(-1)
            ref = inputs["ref_patches"][0].to(device=device, dtype=self._dtype()) if "ref_patches" in inputs else None
            token_ids, text_mask, audio, audio_pos = build_prompt(text_ids, mode, ref)
            generator = torch.Generator(device=device)
            generator.manual_seed(int(fwd_info.random_seed))
            # The last position conditions the first patch: zeros after text,
            # the last prompt patch in continuation mode.
            cond = (
                ref[1][-1]
                if mode == MODE_REF_CONTINUATION
                else torch.zeros(self.config.patch_size, self.config.feat_dim, device=device, dtype=self._dtype())
            )
            state.add_all(
                generator=generator,
                noise={},
                drawn=0,
                generated=0,
                max_patches=max_patches,
                min_len=min_len,
                cfg=params[:1].clone(),
            )
            if audio is None:
                audio = torch.zeros(0, self.config.patch_size, self.config.feat_dim, device=device, dtype=self._dtype())
            return ARNodeInputs(
                input_ids=token_ids,
                input_seq_len=token_ids.shape[0],
                tensor_inputs={
                    "text_mask": text_mask,
                    "audio_patches": audio,
                    "audio_pos": audio_pos,
                    "cond_patch": cond,
                    "noise": self._noise(state, 0),
                    "cfg": state["cfg"],
                },
                kwargs={
                    "step": 0,
                    "prompt_patches": ref[1][-self.config.prompt_context_patches :]
                    if mode == MODE_REF_CONTINUATION
                    else None,
                },
            )
        if graph_walk != "talker_decode":
            raise ValueError(f"Unknown VoxCPM2 Talker walk: {graph_walk!r}")
        # Patches committed so far, advanced only by postprocess after a forward
        # ran: a step whose admit was refused prepares the same index again.
        step = int(state["generated"])
        if step >= int(state["max_patches"]):
            logger.debug("VoxCPM2 talker %s: skipping overshoot step %d", fwd_info.request_id, step)
            return None
        prev = inputs["latent_patch"][0].to(device=device, dtype=self._dtype())
        return ARNodeInputs(
            input_seq_len=1,
            tensor_inputs={
                "prev_patch": prev.reshape(self.config.patch_size, self.config.feat_dim),
                "noise": self._noise(state, step),
                "cfg": state["cfg"],
            },
            kwargs={"step": step},
        )

    def _to_device(self, values: list[int]) -> torch.Tensor:
        """Host ints as a device long tensor, in one non-blocking copy from pinned memory."""
        if self._index_stager is None:
            self._index_stager = PinnedStager(torch.long, numel=self.MAX_BATCH_SIZE)
        out = torch.empty(len(values), dtype=torch.long, device=self.get_device())
        self._index_stager.copy_(out, values)
        return out

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs]
    ) -> dict[str, Any]:
        del engine_inputs
        stack = {
            "noise": torch.stack([item.tensor_inputs["noise"] for item in inputs]),
            "cfg": torch.stack([item.tensor_inputs["cfg"] for item in inputs]).reshape(-1),
            "step_index": self._to_device([item.kwargs["step"] for item in inputs]),
        }
        if graph_walk == "talker_decode":
            stack["prev_patch"] = torch.stack([item.tensor_inputs["prev_patch"] for item in inputs])
            return stack
        ends = list(itertools.accumulate(item.input_seq_len for item in inputs))
        starts = [0, *ends[:-1]]
        stack.update(
            token_ids=torch.cat([item.input_ids for item in inputs]),
            text_mask=torch.cat([item.tensor_inputs["text_mask"] for item in inputs]),
            audio_patches=torch.cat([item.tensor_inputs["audio_patches"] for item in inputs]),
            audio_index=torch.cat(
                [item.tensor_inputs["audio_pos"] + start for start, item in zip(starts, inputs, strict=True)]
            ),
            last_token_indices=self._to_device([end - 1 for end in ends]),
            cond_patch=torch.stack([item.tensor_inputs["cond_patch"] for item in inputs]),
            prompt_patches=[item.kwargs.get("prompt_patches") for item in inputs],
        )
        return stack

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        """One causal step over the packed batch, the same span in both LMs."""
        del graph_walk, slot_lease, piecewise_leases, kwargs
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={
                BASE_KV: KVStep(),
                BASE_ATTN: AttentionStep(causal=True),
                BASE_POS: PositionStep(),
                RES_KV: KVStep(),
                RES_ATTN: AttentionStep(causal=True),
            },
        )

    # -- forward -------------------------------------------------------------

    def _run(
        self, graph_walk: str, noise, cfg, step_index, prev_patch=None, **prefill
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del step_index
        if graph_walk == "talker_decode":
            return self.model.decode_step(prev_patch, noise, cfg, self.t_span())
        prefill.pop("prompt_patches", None)
        return self.model.prefill(noise=noise, cfg_value=cfg, t_span=self.t_span(), **prefill)

    def _outputs(
        self, graph_walk: str, patch: torch.Tensor, stop: torch.Tensor, prompt_patches: torch.Tensor | None
    ) -> NameToTensorList:
        stream = [patch]
        if graph_walk == "talker_prefill_clone" and prompt_patches is not None:
            # A continuation's VAE stream leads with the tail of the prompt
            # audio, as the reference decodes it: context, trimmed downstream.
            stream = [*prompt_patches.unbind(0), patch]
        return {"latent_patch": [patch], "latent_patches": stream, "stop_flag": [stop]}

    @staticmethod
    def _stop_flags(stop_logits: torch.Tensor, step_index: torch.Tensor) -> torch.Tensor:
        """``[rows, 2]``: the stop head's verdict and the index of the patch it
        judged, so ``check_stop`` reads the step's own count even when a later
        speculative step has already been postprocessed."""
        return torch.stack([stop_logits.argmax(dim=-1), step_index], dim=-1)

    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs: Any) -> NameToTensorList:
        prompt = kwargs.get("prompt_patches")
        patch, stop_logits = self._run(graph_walk, **kwargs)
        stop = self._stop_flags(stop_logits, kwargs["step_index"])
        return self._outputs(graph_walk, patch[0], stop[0], prompt[0] if prompt else None)

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs: Any
    ) -> dict[str, NameToTensorList]:
        prompt = kwargs.get("prompt_patches") or [None] * len(engine_inputs.request_ids)
        patch, stop_logits = self._run(graph_walk, **kwargs)
        stop = self._stop_flags(stop_logits, kwargs["step_index"])
        return {
            rid: self._outputs(graph_walk, patch[i], stop[i], prompt[i] if i < len(prompt) else None)
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs: Any,
    ) -> None:
        del request_info, kwargs
        if "stop_flag" in outputs:
            state = self.request_state(request_id)
            state.add("generated", int(state.get("generated", 0)) + 1)

    def check_stop(
        self, request_id: str, request_info: CurrentForwardPassInfo, outputs: dict[str, list[torch.Tensor]]
    ) -> set[str]:
        """The reference's loop exit: the stop head fires past ``min_len``
        steps, or the request's patch budget is spent. The stopping step's
        patch is audio and has already been streamed."""
        del request_info
        if "stop_flag" not in outputs:
            return set()
        state = self.request_state(request_id)
        flag, index = (int(v) for v in outputs["stop_flag"][0].reshape(-1).tolist())
        generated, stop = index + 1, flag == 1
        if (stop and generated - 1 > int(state["min_len"])) or generated >= int(state["max_patches"]):
            logger.debug("VoxCPM2 talker %s: stop after %d patches (head=%s)", request_id, generated, stop)
            return {TALKER_LOOP}
        return set()

    # -- batching / graphs ----------------------------------------------------

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return batch.graph_walk in (*PREFILL_WALKS, "talker_decode") and 0 < len(model_inputs) <= self.MAX_BATCH_SIZE

    def max_batch_size(self, graph_walk: str) -> int:
        del graph_walk
        return self.MAX_BATCH_SIZE

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1
    ) -> list[AcceleratorGraphConfig]:
        """Capture the whole decode step (both LMs, the 10-step CFM, the stop head),
        ``torch.compile``'d first unless the deployment set ``compile_talker: false``."""
        del tp_world_size
        dtype = self._dtype()
        cfg = self.config
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="talker_decode",
                single_request_inputs=ARNodeInputs(
                    input_seq_len=1,
                    tensor_inputs={
                        "prev_patch": torch.zeros(cfg.patch_size, cfg.feat_dim, dtype=dtype, device=device),
                        "noise": torch.zeros(cfg.feat_dim, cfg.patch_size, dtype=dtype, device=device),
                        "cfg": torch.full((1,), cfg.cfg_value, dtype=torch.float32, device=device),
                    },
                    kwargs={"step": 1},
                ),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
                compile=self.compile_decode,
            )
        ]

    def can_use_accelerator_graphs(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return (
            batch.graph_walk == "talker_decode"
            and self.can_batch(batch, model_inputs)
            and super().can_use_accelerator_graphs(batch, model_inputs)
        )


# ===========================================================================
# 3. VAEDecoderSubmodule
# ===========================================================================


class VAEDecoderSubmodule(ARNodeSubmodule):
    """Stream windows of latent patches -> 48 kHz PCM, context audio dropped.

    A window is ``context + new`` patches (``ScheduledLeftContextChunkPolicy``).
    The decoder is causal, so padding a window at the end to batch it with
    longer ones never changes its frames, and a context of at least the
    receptive field makes each window's new audio equal a whole decode's.
    """

    disable_torch_compile = True
    disable_autocast = True
    MAX_BATCH_SIZE = 32

    def __init__(self, vae: AudioVAE, config: VoxCPM2Config):
        super().__init__()
        self.vae = vae
        self.config = config
        self._sr_index: torch.Tensor | None = None

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        input_metadata: InputMetadata = EMPTY_INPUT_METADATA,
        **kwargs: Any,
    ) -> ARNodeInputs:
        del graph_walk, kwargs
        chunk = input_metadata.stream_chunks.get("latent_patches")
        data = inputs["latent_patches"][0] if inputs.get("latent_patches") else None
        cfg = self.config
        dtype = next(self.vae.parameters()).dtype
        if data is None:
            patches = torch.zeros(0, cfg.patch_size, cfg.feat_dim, device=self.get_device(), dtype=dtype)
        else:
            patches = data.to(device=self.get_device(), dtype=dtype).reshape(-1, cfg.patch_size, cfg.feat_dim)
        num = patches.shape[0]
        context = min(chunk.context_items if chunk is not None else 0, num)
        logger.debug("VoxCPM2 vae %s: window %d patches, context %d", fwd_info.request_id, num, context)
        return ARNodeInputs(
            input_seq_len=1,
            tensor_inputs={"latents": patches.reshape(-1, cfg.feat_dim).t().contiguous()},
            kwargs={"patches": num, "context": context},
        )

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs]
    ) -> dict[str, torch.Tensor]:
        del graph_walk, engine_inputs
        windows = [item.tensor_inputs["latents"] for item in inputs]
        width = max(window.shape[-1] for window in windows)
        return {"latents": torch.stack([torch.nn.functional.pad(w, (0, width - w.shape[-1])) for w in windows])}

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.shape[-1] == 0:
            return latents.new_zeros(latents.shape[0], 0)
        if self._sr_index is None:
            self._sr_index = self.vae.output_rate_index(latents.device)
        return self.vae.decode(latents, self._sr_index)

    def forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, latents: torch.Tensor, **kwargs: Any
    ) -> NameToTensorList:
        del graph_walk, engine_inputs, kwargs
        return {"audio_chunk": [self._decode(latents)[0]]}

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, latents: torch.Tensor, **kwargs: Any
    ) -> dict[str, NameToTensorList]:
        del graph_walk, kwargs
        audio = self._decode(latents)
        return {rid: {"audio_chunk": [audio[i]]} for i, rid in enumerate(engine_inputs.request_ids)}

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        inputs: ARNodeInputs | None = None,
        **kwargs: Any,
    ) -> None:
        """Keep the window's new patches only: drop the context's audio and any batch padding."""
        del request_id, request_info, kwargs
        if "audio_chunk" not in outputs:
            return
        if inputs is None:
            raise ValueError("VoxCPM2 VAE postprocess needs the pass's inputs for its window geometry")
        per_patch = self.config.patch_samples
        start = int(inputs.kwargs["context"]) * per_patch
        end = int(inputs.kwargs["patches"]) * per_patch
        audio = outputs["audio_chunk"][0].reshape(-1)
        if audio.numel() < end:
            raise ValueError(f"VoxCPM2 VAE produced {audio.numel()} samples for a {end}-sample window")
        outputs["audio_chunk"][0] = audio[start:end]

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        del batch
        return 0 < len(model_inputs) <= self.MAX_BATCH_SIZE

    def max_batch_size(self, graph_walk: str) -> int:
        del graph_walk
        return self.MAX_BATCH_SIZE
