"""mstar engine submodules for Ming-flash-omni-2.0.

Three submodules covering the multimodal-understanding side of the model:

  * ``VisionEncoderSubmodule`` (enc_dec / stateless) — runs Ming's
    Qwen3MoeVisionTransformer + MingVisionProjector, returns
    LLM-space vision embeddings for the Thinker to splice in.

  * ``AudioEncoderSubmodule`` (enc_dec / stateless) — runs
    MingAudioEncoder + MingAudioProjector, returns LLM-space audio
    embeddings (packed across clips).

  * ``BailingMoeV2ThinkerSubmodule`` (AR / KV-cache) — the Ling-2.0
    MoE LLM. Text-only paths are wired today (step 3d–3f); the
    vision/audio prefill paths grow in via this submodule's
    ``prepare_inputs`` dispatch in step 5b.

Reference: mstar's :class:`OrpheusLLMSubmodule`
(`mstar/model/orpheus/submodules.py:20-176`) is the cleanest text-LLM
template; Qwen3-Omni's submodules
(`mstar/model/qwen3_omni/submodules.py`) show the multimodal extensions
and graph-walk dispatch we mirror here.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.kv_store import PositionInfo
from mstar.model.ming_omni_flash.components.model import LingMoeModel
from mstar.model.ming_omni_flash.config import MingFlashOmniModelConfig
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)

logger = logging.getLogger(__name__)


# ===================================================================
# 1. VisionEncoderSubmodule (stateless enc_dec engine)
# ===================================================================


class VisionEncoderSubmodule(NodeSubmodule):
    """Wraps Ming's Qwen3MoeVisionTransformer + MingVisionProjector.

    Runs once per request (stateless), consumes ``(pixel_values,
    image_grid_thw)`` and produces ``vision_embeds`` already projected
    into the Thinker's hidden space (no further linear on the LLM
    side — Ming applies the projector + L2 norm before splicing,
    mirroring ``modeling_bailingmm2.extract_image_feature``).

    ``deepstack`` features are deliberately NOT plumbed in step 5a:
    the released ckpt sets ``use_deepstack=False`` and the deepstack
    list is not consumed by ``modeling_bailingmm2``'s text-out path.
    If/when we enable deepstack splicing, ``build_vision_encoder``
    grows a ``use_deepstack=True`` flag and this submodule's forward
    will return both tensors.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        vision_projector: nn.Module,
        config: MingFlashOmniModelConfig,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.vision_projector = vision_projector
        self.config = config

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        """Pull pixel_values + grid_thw off the conductor's input bundle.

        ``image_grid_thw`` is produced by ``process_prompt`` from the
        HF image processor; for the test path (no processor) a 1-D
        ``[T, H, W]`` tensor also works (we promote it to ``(1, 3)``).
        """
        if "pixel_values" not in inputs or not inputs["pixel_values"]:
            raise ValueError(
                "VisionEncoderSubmodule: missing 'pixel_values' input. "
                "process_prompt must produce this from the image processor."
            )
        pixel_values = inputs["pixel_values"][0]
        grid_thw = inputs.get(
            "image_grid_thw", inputs.get("grid_thw", [None])
        )[0]
        if grid_thw is None:
            raise ValueError(
                "VisionEncoderSubmodule: 'image_grid_thw' is None. "
                "Make sure process_prompt forwarded image_grid_thw from "
                "the HF image processor (BailingMM2Processor)."
            )
        if grid_thw.dim() == 1:
            grid_thw = grid_thw.unsqueeze(0)  # promote to (1, 3)

        return NodeInputs(
            tensor_inputs={
                "pixel_values": pixel_values,
                "grid_thw": grid_thw,
            }
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        """Run encoder → projector → L2-norm.

        Ming applies ``F.normalize(image_embeds, dim=-1)`` after the
        projector (``modeling_bailingmm2.extract_image_feature:101``).
        We mirror that so the Thinker sees the same numeric distribution
        the source model produced during training.
        """
        device = pixel_values.device
        logger.debug(
            "VisionEncoder: pixel_values=%s grid_thw=%s",
            tuple(pixel_values.shape), tuple(grid_thw.shape),
        )
        # The Ming encoder accepts a single torch.Tensor of stacked
        # patches; grid_thw selects which positions / images they belong
        # to. ``use_deepstack=False`` so encoder returns a single tensor.
        with torch.no_grad():
            vision_embeds = self.vision_encoder(
                pixel_values.to(device), grid_thw=grid_thw.to(device),
            )
            if isinstance(vision_embeds, tuple):
                # Defensive: if the encoder was built with
                # ``use_deepstack=True``, drop the deepstack list.
                vision_embeds = vision_embeds[0]
            projected = self.vision_projector(vision_embeds)
            projected = torch.nn.functional.normalize(projected, dim=-1)
        return {"vision_embeds": [projected]}


# ===================================================================
# 2. AudioEncoderSubmodule (stateless enc_dec engine)
# ===================================================================


class AudioEncoderSubmodule(NodeSubmodule):
    """Wraps MingAudioEncoder + MingAudioProjector.

    Consumes a list of variable-length mel spectrograms (one per
    audio clip) and produces packed ``audio_embeds`` ready for the
    Thinker to splice. The packed-sequence forward matches the upstream
    encoder ABI (returns ``(packed, cu_seqlens)``); we drop
    ``cu_seqlens`` after the projector chunks the per-clip lengths
    back via ``MingAudioProjector.compute_output_length`` if needed.

    For step 5a the submodule assumes a single audio clip per request
    (the common case for Q&A / TTS / S2S); multi-clip batched audio
    folds in alongside Thinker batching in a later step.
    """

    def __init__(
        self,
        audio_encoder: nn.Module,
        audio_projector: nn.Module,
        config: MingFlashOmniModelConfig,
    ) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.audio_projector = audio_projector
        self.config = config

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        """Pull mel features + (optional) per-clip lengths.

        ``audio_features`` is the only required input today. It's
        either ``(n_mels, T)`` for a single clip or ``(B, n_mels, T)``
        for already-batched input. ``audio_seqlens`` (the original
        unpadded length per clip) is optional — when present the
        encoder uses it to skip the padded tail.
        """
        if "audio_features" not in inputs or not inputs["audio_features"]:
            raise ValueError(
                "AudioEncoderSubmodule: missing 'audio_features' input. "
                "process_prompt must produce this from the audio processor."
            )
        audio_features = inputs["audio_features"][0]
        audio_seqlens = inputs.get("audio_seqlens", [None])[0]
        return NodeInputs(
            tensor_inputs={
                "audio_features": audio_features,
                "audio_seqlens": audio_seqlens,
            }
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: torch.Tensor,
        audio_seqlens: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """Encoder → projector → L2-norm (if config.audio_encoder.norm_query_embeds).

        Mirrors ``modeling_bailingmm2.extract_audio_feature``:
        L2-normalize along the last dim when ``norm_query_embeds`` is
        set in the audio config (true on the released ckpt).
        """
        device = audio_features.device
        # Accept (n_mels, T) for a single clip or (B, n_mels, T) batched.
        if audio_features.dim() == 2:
            mel_list: list[torch.Tensor] = [audio_features.to(device)]
        elif audio_features.dim() == 3:
            mel_list = [audio_features[i].to(device) for i in range(audio_features.shape[0])]
        else:
            raise ValueError(
                f"AudioEncoderSubmodule: expected audio_features of rank 2 or 3, "
                f"got rank {audio_features.dim()} with shape {tuple(audio_features.shape)}"
            )
        # If audio_seqlens is provided, trim the padded tail of each clip
        # before sending it to the encoder so positional embeddings line up.
        if audio_seqlens is not None:
            mel_list = [
                m[..., : int(audio_seqlens[i].item())]
                for i, m in enumerate(mel_list)
            ]

        logger.debug(
            "AudioEncoder: %d clip(s), per-clip mel T=%s",
            len(mel_list), [int(m.shape[-1]) for m in mel_list],
        )
        with torch.no_grad():
            # Packed encoder returns (total_T', n_state), cu_seqlens int32.
            packed, cu_seqlens = self.audio_encoder(mel_list)
            # Projector expects (B, T, audio_dim) shape — feed one clip
            # at a time when there are multiple, then concat.
            projected_chunks: list[torch.Tensor] = []
            seg_starts = cu_seqlens.tolist()
            for i in range(len(seg_starts) - 1):
                seg = packed[seg_starts[i]:seg_starts[i + 1]].unsqueeze(0)  # (1, T_i, n_state)
                # Projector returns (B, llm_dim, T'_i); transpose to (T'_i, llm_dim).
                projected = self.audio_projector(seg).squeeze(0).transpose(0, 1)
                projected_chunks.append(projected)
            audio_embeds = torch.cat(projected_chunks, dim=0)  # (sum T'_i, llm_dim)

            if self.config.audio_encoder.norm_query_embeds:
                audio_embeds = torch.nn.functional.normalize(audio_embeds, dim=-1)

        return {"audio_embeds": [audio_embeds.to(audio_features.dtype)]}


class BailingMoeV2ThinkerSubmodule(ARNodeSubmodule):
    """Thinker submodule for Ming-flash-omni-2.0.

    Graph walks the dispatch handles:
      * ``prefill`` / ``prefill_text``: embed text token ids, fill KV
        cache, sample first token's logits. (``prefill`` is the legacy
        text-only name kept for backward compat with step 3f; step 5c
        renames the walk to ``prefill_text``.)
      * ``prefill_audio``: splice precomputed audio embeddings between
        ``audio_start`` / ``audio_end`` sentinel embeddings; build
        text-like 3D MRoPE positions for the span; fill KV cache;
        sample first token's logits.
      * ``prefill_vision`` / ``prefill_video``: splice precomputed
        vision embeddings between ``image_start`` / ``image_end``
        (or ``video_start`` / ``video_end``) sentinel embeddings;
        build grid-aware 3D MRoPE positions per
        ``modeling_bailing_moe_v2.get_rope_index:625-647``; fill KV
        cache; sample first token's logits.
      * ``decode`` / ``thinker_decode``: embed the previous token,
        single-step forward, sample next-token logits.

    The submodule does NOT use ``cache_handle.apply_rope`` — Ling-2.0's
    partial 3D ``video_rope`` is applied inline by
    :class:`LingAttention` using the explicit ``position_ids`` argument.
    """

    # Walk names treated as text-only prefill (no embed splicing).
    _TEXT_PREFILL_WALKS = ("prefill", "prefill_text")
    # Walk names treated as autoregressive decode (one-token step).
    _DECODE_WALKS = ("decode", "thinker_decode")

    def __init__(
        self,
        model: LingMoeModel,
        config: MingFlashOmniModelConfig | None = None,
        eos_token_id: int = 156895,
    ) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.eos_token_id = eos_token_id
        # Stash the embed_tokens / lm_head as direct attributes so the
        # engine's CUDA-graph captures don't reach through .model.
        self.embed_tokens = model.embed_tokens
        self.lm_head = model.lm_head

        # Lazily-cached sentinel token embeddings (1, hidden_size each).
        # Recomputed on first use per device; allocated lazily so CPU
        # tests don't materialise the embed table at import time.
        self._image_start_embed: torch.Tensor | None = None
        self._image_end_embed: torch.Tensor | None = None
        self._video_start_embed: torch.Tensor | None = None
        self._video_end_embed: torch.Tensor | None = None
        self._audio_start_embed: torch.Tensor | None = None
        self._audio_end_embed: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Sentinel embedding helpers
    # ------------------------------------------------------------------

    def _sentinel_embed(self, token_id: int, device: torch.device) -> torch.Tensor:
        """Embed a single sentinel token id; small enough to recompute."""
        tok = torch.tensor([int(token_id)], dtype=torch.long, device=device)
        return self.embed_tokens(tok)  # (1, hidden_size)

    def _get_vision_bos_eos(
        self, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config is None:
            raise RuntimeError(
                "BailingMoeV2ThinkerSubmodule.config is None — required for "
                "vision sentinel embeddings. Pass config=... at construction "
                "(step 5b)."
            )
        llm = self.config.thinker_llm
        if self._image_start_embed is None or self._image_start_embed.device != device:
            self._image_start_embed = self._sentinel_embed(llm.image_start_token, device)
            self._image_end_embed = self._sentinel_embed(llm.image_end_token, device)
        return self._image_start_embed, self._image_end_embed

    def _get_video_bos_eos(
        self, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config is None:
            raise RuntimeError("config required for video sentinels.")
        llm = self.config.thinker_llm
        if self._video_start_embed is None or self._video_start_embed.device != device:
            self._video_start_embed = self._sentinel_embed(llm.video_start_token, device)
            self._video_end_embed = self._sentinel_embed(llm.video_end_token, device)
        return self._video_start_embed, self._video_end_embed

    def _get_audio_bos_eos(
        self, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config is None:
            raise RuntimeError("config required for audio sentinels.")
        llm = self.config.thinker_llm
        if self._audio_start_embed is None or self._audio_start_embed.device != device:
            self._audio_start_embed = self._sentinel_embed(llm.audio_start_token, device)
            self._audio_end_embed = self._sentinel_embed(llm.audio_end_token, device)
        return self._audio_start_embed, self._audio_end_embed

    # ------------------------------------------------------------------
    # Image-gen producer-side hidden-state extraction (step 9b)
    # ------------------------------------------------------------------

    @staticmethod
    def extract_image_gen_hidden_states(
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        image_patch_token: int,
    ) -> torch.Tensor:
        """Slice the post-norm hidden states at the ``<imagePatch>`` positions.

        For an image-generation request the prompt carries an
        ``<image><imagePatch>*N</image>`` block (appended by
        :func:`maybe_expand_image_gen_prompt`, step 8b). The DiT condition
        encoder (step 9b) consumes the thinker's hidden states *at those N
        query-token positions* — not the sampled token. This helper pulls them
        out so the Thinker→ImageGen streaming edge can carry
        ``thinker_hidden_states``.

        Args:
            hidden_states: ``(T, H)`` post-norm thinker hidden states (from
                ``LingMoeModel.forward(..., return_hidden_states=True)``).
            token_ids: ``(T,)`` the input token ids for the same forward pass
                (used to locate the patch positions).
            image_patch_token: the ``<imagePatch>`` token id
                (``config.thinker_llm.image_patch_token``, 157157 on the
                released ckpt).

        Returns:
            ``(N, H)`` hidden states at the patch positions, in order.

        Raises:
            ValueError: if shapes disagree or no patch tokens are present.
        """
        if hidden_states.dim() != 2:
            raise ValueError(f"expected (T, H) hidden_states, got {tuple(hidden_states.shape)}")
        if token_ids.dim() != 1:
            token_ids = token_ids.reshape(-1)
        if token_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                f"token_ids length {token_ids.shape[0]} != hidden_states T "
                f"{hidden_states.shape[0]}"
            )
        mask = token_ids.to(hidden_states.device) == int(image_patch_token)
        if not bool(mask.any()):
            raise ValueError(
                f"no <imagePatch> token ({image_patch_token}) found in token_ids; "
                "process_prompt must append the query-token block for image output."
            )
        return hidden_states[mask]

    # ------------------------------------------------------------------
    # ARNodeSubmodule contract
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        pos_info: dict[str, PositionInfo] = {},
        **kwargs,
    ) -> ARNodeInputs:
        """Dispatch on graph_walk to build per-request ARNodeInputs.

        ``**kwargs`` absorbs engine-passed extras (e.g. ``seen_token_mask``
        from the KV-cache engine's sampler) that this submodule doesn't use,
        mirroring the peer models so the engine→submodule contract stays
        forward-compatible.

        Text-only walks return ``input_ids`` (LingMoeModel embeds them
        inline). Multimodal walks return precomputed ``input_embeds``
        + ``custom_pos_ids`` so the position counter stays in sync
        with the sentinel + modality span structure
        ``modeling_bailing_moe_v2.get_rope_index`` would have produced.
        """
        device = self.get_device()
        start_pos = int(
            pos_info.get("main", PositionInfo()).position_id_start
        )

        if graph_walk in self._DECODE_WALKS or graph_walk in self._TEXT_PREFILL_WALKS:
            token_ids = inputs["text_inputs"][0].to(device)
            return ARNodeInputs(
                input_ids=token_ids,
                input_seq_len=token_ids.shape[0],
            )

        if graph_walk == "prefill_audio":
            return self._prepare_prefill_audio(inputs, device, start_pos)

        if graph_walk in ("prefill_vision", "prefill_video"):
            return self._prepare_prefill_vision(
                inputs, device, start_pos, video=(graph_walk == "prefill_video"),
            )

        raise ValueError(
            f"BailingMoeV2ThinkerSubmodule: unknown graph_walk {graph_walk!r}. "
            f"Supported: prefill / prefill_text / prefill_audio / prefill_vision "
            f"/ prefill_video / decode / thinker_decode."
        )

    def _prepare_prefill_audio(
        self,
        inputs: NameToTensorList,
        device: torch.device,
        start_pos: int,
    ) -> ARNodeInputs:
        """Audio prefill: splice ``[bos, audio_embeds, eos]``, text positions."""
        # Local import to keep the components/positions module a leaf in
        # the dependency graph (avoids a circular import at module load).
        from mstar.model.ming_omni_flash.components.positions import (
            get_rope_index_text,
        )
        if "audio_embeds" not in inputs or not inputs["audio_embeds"]:
            raise ValueError(
                "prefill_audio walk: missing 'audio_embeds' input. "
                "Make sure the prefill graph routes the AudioEncoder "
                "output edge into the Thinker."
            )
        audio_embeds = inputs["audio_embeds"][0].to(device)
        bos, eos = self._get_audio_bos_eos(device)
        # Match dtype between sentinel embeds and audio embeds. The
        # encoder's projector returns the LLM's autocast dtype while
        # the embed_tokens table lives in the model's stored dtype —
        # cast sentinels to the audio dtype so the cat is consistent.
        bos = bos.to(audio_embeds.dtype)
        eos = eos.to(audio_embeds.dtype)
        embeds = torch.cat([bos, audio_embeds, eos], dim=0)
        seq_len = embeds.shape[0]
        pos_ids = get_rope_index_text(seq_len, start_pos, device=device)
        return ARNodeInputs(
            input_seq_len=seq_len,
            input_embeds=embeds,
            custom_pos_ids=pos_ids,
        )

    def _prepare_prefill_vision(
        self,
        inputs: NameToTensorList,
        device: torch.device,
        start_pos: int,
        video: bool,
    ) -> ARNodeInputs:
        """Vision prefill: splice ``[bos, vision_embeds, eos]`` + grid positions."""
        from mstar.model.ming_omni_flash.components.positions import (
            get_rope_index_text,
            get_rope_index_vision,
        )
        if "vision_embeds" not in inputs or not inputs["vision_embeds"]:
            raise ValueError(
                "prefill_vision walk: missing 'vision_embeds' input. "
                "Make sure the prefill graph routes the VisionEncoder "
                "output edge into the Thinker."
            )
        vision_embeds = inputs["vision_embeds"][0].to(device)
        grid_thw = inputs.get(
            "image_grid_thw", inputs.get("video_grid_thw", inputs.get("grid_thw", [None])),
        )[0]
        if grid_thw is None:
            raise ValueError(
                "prefill_vision walk: missing 'image_grid_thw' input. "
                "process_prompt must forward this from the image processor."
            )
        grid_thw = grid_thw.to(device)
        if grid_thw.dim() == 1:
            grid = grid_thw
        else:
            # Multi-image / multi-clip support is step 5c (the graph
            # router will sequence one Sequential per image). For 5b
            # we restrict to a single image / clip per request.
            if grid_thw.shape[0] > 1:
                raise NotImplementedError(
                    "prefill_vision: multi-image grid_thw not supported in "
                    "step 5b; one image / clip per request only."
                )
            grid = grid_thw[0]

        # Video walks honor a per-frame timestamp via
        # ``video_second_per_grid``; image walks pass None (one frame).
        seconds_per_grid: float | None = None
        if video:
            spg = inputs.get("video_second_per_grid", [None])[0]
            if spg is not None:
                seconds_per_grid = float(
                    spg.item() if isinstance(spg, torch.Tensor) else spg
                )
            else:
                seconds_per_grid = 1.0  # mirrors the upstream default

        bos, eos = (
            self._get_video_bos_eos(device) if video
            else self._get_vision_bos_eos(device)
        )
        bos = bos.to(vision_embeds.dtype)
        eos = eos.to(vision_embeds.dtype)
        embeds = torch.cat([bos, vision_embeds, eos], dim=0)
        seq_len = embeds.shape[0]

        if self.config is None:
            raise RuntimeError("config required for prefill_vision (spatial_merge_size).")
        spatial_merge = self.config.vision.spatial_merge_size
        bos_pos = get_rope_index_text(1, start_pos, device=device)
        vision_pos = get_rope_index_vision(
            grid_thw=grid,
            start_pos=start_pos + 1,
            spatial_merge_size=spatial_merge,
            device=device,
            second_per_grid_t=seconds_per_grid,
            tokens_per_second=self.config.thinker_llm.tokens_per_second,
        )
        # eos goes one past the largest vision position so the next walk's
        # text positions don't collide with the vision span's T/H/W ranges.
        eos_pos_start = int(vision_pos.max().item()) + 1
        eos_pos = get_rope_index_text(1, eos_pos_start, device=device)
        pos_ids = torch.cat([bos_pos, vision_pos, eos_pos], dim=1)
        if pos_ids.shape != (3, seq_len):
            raise AssertionError(
                f"prefill_vision: position_ids shape {tuple(pos_ids.shape)} "
                f"does not match seq_len={seq_len} (3, T) expectation."
            )
        return ARNodeInputs(
            input_seq_len=seq_len,
            input_embeds=embeds,
            custom_pos_ids=pos_ids,
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        """Plan attention; pack inputs for forward.

        Single-request only in step 3d; batched preprocess folds in
        step 3e+ via ``can_batch`` + ``forward_batched``. The text and
        multimodal paths use mutually exclusive keys downstream so the
        forward can branch on which one is set: ``text_inputs`` for
        the input-ids path, ``input_embeds`` + ``position_ids`` for
        the embeds path.
        """
        if len(inputs) > 1:
            raise NotImplementedError(
                f"BailingMoeV2ThinkerSubmodule: multi-request batching is "
                f"step-3e scope; got {len(inputs)} requests"
            )
        cache_manager = engine_inputs.cache_manager
        seq_lens = [inp.input_seq_len for inp in inputs]

        cache_manager.set_active_label("main")
        cache_manager.plan_attention(
            seq_lens=seq_lens, is_causal=True, label="main",
        )
        # We don't call ``cache_manager.apply_rope`` in attention (we
        # have our own partial 3D rope), but mstar's plan_rope also
        # advances internal position-id state used by ``advance_seq_lens``
        # — keep this call for parity with Orpheus.
        cache_manager.plan_rope(seq_lens=seq_lens, pos_ids=None, label="main")

        inp = inputs[0]
        if inp.input_embeds is not None:
            # Multimodal path: forward gets embeds + explicit positions.
            return {
                "input_embeds": inp.input_embeds,
                "position_ids": inp.custom_pos_ids,
            }
        return {
            "text_inputs": torch.cat([inp.input_ids for inp in inputs]),
        }

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        text_inputs: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        cache_handle = engine_inputs.cache_manager
        request_info = engine_inputs.single_request_info

        # Image-gen prefill carries an <imagePatch> query-token block; when
        # those token ids are present we additionally capture the post-norm
        # hidden states at those positions and stream them to the ImageGen
        # partition. Only meaningful on the text-input path (the block lives
        # in the tokenized prompt); embeds-path multimodal prefills don't
        # carry it.
        want_image_gen = (
            self.config is not None
            and self.config.image_gen is not None
            and text_inputs is not None
            and bool((text_inputs == self.config.thinker_llm.image_patch_token).any())
        )

        if input_embeds is not None:
            if position_ids is None:
                raise ValueError(
                    "BailingMoeV2ThinkerSubmodule.forward: input_embeds "
                    "provided but position_ids is None. prepare_inputs "
                    "must emit custom_pos_ids alongside embeds."
                )
            model_out = self.model(
                cache_handle,
                input_embeds=input_embeds,
                position_ids=position_ids,
            )
        else:
            if text_inputs is None:
                raise ValueError(
                    "BailingMoeV2ThinkerSubmodule.forward: neither "
                    "text_inputs nor input_embeds provided."
                )
            # Text-only path: build 1D positions from the request's
            # position counter (same as step 3f).
            start_pos = 0
            try:
                start_pos = (
                    request_info.position_info.get("main", PositionInfo())
                    .position_id_start
                )
            except AttributeError:
                # ARNodeSubmodule contract may not always provide
                # position_info; fall back to 0.
                pass

            num_tokens = text_inputs.shape[0]
            position_ids_1d = torch.arange(
                start_pos, start_pos + num_tokens,
                dtype=torch.long, device=text_inputs.device,
            )
            model_out = self.model(
                cache_handle,
                input_ids=text_inputs,
                position_ids=position_ids_1d,
                return_hidden_states=want_image_gen,
            )

        if want_image_gen:
            logits, hidden_states = model_out
        else:
            logits = model_out

        # Advance the cache's sequence lengths so the next decode step
        # knows where to read/write. This is the standard post-forward
        # call that mstar's KV cache uses to track positions.
        cache_handle.advance_seq_lens()

        # Sample only the last position's logits (next-token sampling).
        last_logits = logits[-1:, :]
        outputs: NameToTensorList = {"logits": [last_logits]}
        if want_image_gen:
            patch_hidden = self.extract_image_gen_hidden_states(
                hidden_states, text_inputs, self.config.thinker_llm.image_patch_token,
            )
            outputs["thinker_hidden_states"] = [patch_hidden]
        return outputs

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ) -> None:
        """Rebind ``new_token`` → ``text_inputs`` for the decode loop.

        The decode walk's output edge is named ``text_inputs`` so the loop
        feeds the previous sampled token back into the next iteration.
        ``submodule.forward`` returns ``{"logits": [...]}``; the KV-cache
        engine samples that into ``{"new_token": [...]}``; this hook then
        publishes the same tensor under the ``text_inputs`` key so the
        graph router finds an output to attach to the loop edge.

        Mirrors :meth:`OrpheusLLMSubmodule.postprocess`.
        """
        if "new_token" not in outputs:
            return
        outputs["text_inputs"] = outputs["new_token"]

    # ------------------------------------------------------------------
    # Stop conditions
    # ------------------------------------------------------------------

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        """Stop the ``thinker_decode_loop`` when the sampled token is the EOS
        (``<|role_end|>`` for Ming, token id 156895).

        The returned name MUST match the ``Loop(name=...)`` declared in
        ``get_graph_walk_graphs`` (``thinker_decode_loop``). A mismatch makes
        the worker's dynamic-loop registry raise ``KeyError(NodeAndGraphWalk(
        node='decode_loop', ...))`` on the EOS step and crash the rank.
        """
        new_tokens = outputs.get("new_token") or []
        if not new_tokens:
            return set()
        last = new_tokens[-1]
        if isinstance(last, torch.Tensor):
            tok = int(last.flatten()[0].item())
        else:
            tok = int(last)
        if tok == self.eos_token_id:
            return {"thinker_decode_loop"}
        return set()

    def can_batch(self, batch, model_inputs) -> bool:
        # Step 3d is single-request; step 3e adds batching.
        return False


# ===================================================================
# 4. TalkerSubmodule (stateless TTS — text tokens -> waveform)
# ===================================================================


class TalkerSubmodule(NodeSubmodule):
    """Stateless TTS node: talker text token ids -> audio waveform.

    Ming's thinker->talker bridge passes DETOKENIZED TEXT, not streaming
    hidden states (see vllm-omni's pipeline.py: ``thinker2talker`` re-encodes
    the text with the talker's own ``talker/llm`` tokenizer). That makes the
    talker a near-standalone TTS node — much simpler than qwen3_omni's
    streaming-codec handoff. We model it as a single stateless node whose
    forward runs the full AR loop + VAE decode via :class:`TalkerGenerator`.

    The whole per-request generation (LLM prefill + CFM AR decode + AudioVAE
    decode) happens inside one ``forward`` call rather than being unrolled
    into a conductor-driven decode loop, because the CFM step count is
    self-determined by the stop_head (not a token-by-token graph loop).
    This keeps the graph wiring (step 6e-3) trivial: one Talker node,
    one ``EMIT_TO_CLIENT`` audio edge.

    Engine type: STATELESS (no KV cache managed by mstar — the talker LLM
    manages its own ``StaticCache`` internally inside generate_latents).
    """

    def __init__(
        self,
        generator: "Any",  # TalkerGenerator (avoid import cycle at module load)
        config: MingFlashOmniModelConfig,
        max_steps: int = 1000,
        min_new_token: int = 10,
        text_bridge: "Any" = None,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.config = config
        self.max_steps = max_steps
        self.min_new_token = min_new_token
        # Optional Thinker->Talker text bridge: a callable that maps
        # thinker output token ids -> talker_text_inputs token ids
        # (detokenize with the thinker tokenizer, re-encode with the
        # talker/llm tokenizer). When the streaming edge delivers raw
        # thinker tokens, prepare_inputs runs this first. When None
        # (unit-test path / pre-bridged inputs), the inputs are assumed
        # to already be talker-tokenizer ids.
        self.text_bridge = text_bridge
        # Stash embed_tokens so prepare_inputs can map talker text ids ->
        # inputs_embeds without reaching through the generator each time.
        self.embed_tokens = generator.llm.embed_tokens

    def get_stateless_flavor(self) -> str:
        # The talker runs in bf16 with autocast off; mirror the audio_codec
        # flavor (no torch.compile, no autocast) since the CFM ODE loop +
        # AudioVAE ISTFT are numerically sensitive.
        return "audio_codec"

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        """Embed the talker text token ids into the LLM's input space.

        ``talker_text_inputs`` is the token-id tensor produced by the
        Thinker->Talker text bridge (step 6e-3) — already encoded with
        the talker's own ``talker/llm`` tokenizer. We embed it here so
        forward gets ready-to-run ``inputs_embeds``.
        """
        # Two input shapes accepted:
        #   * ``talker_text_inputs`` — already talker-tokenized ids
        #     (unit-test path / pre-bridged).
        #   * ``thinker_tokens`` — raw thinker output ids streamed from
        #     the Thinker partition; run text_bridge to re-tokenize.
        if "talker_text_inputs" in inputs and inputs["talker_text_inputs"]:
            token_ids = inputs["talker_text_inputs"][0]
        elif "thinker_tokens" in inputs and inputs["thinker_tokens"]:
            if self.text_bridge is None:
                raise RuntimeError(
                    "TalkerSubmodule: received 'thinker_tokens' but no "
                    "text_bridge is configured to re-tokenize them."
                )
            token_ids = self.text_bridge(inputs["thinker_tokens"][0])
        else:
            raise ValueError(
                "TalkerSubmodule: missing 'talker_text_inputs' / "
                "'thinker_tokens'. The Thinker->Talker bridge (step 6e-3) "
                "must supply the text ids."
            )
        device = self.embed_tokens.weight.device
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)  # (1, T)
        token_ids = token_ids.to(device)
        inputs_embeds = self.embed_tokens(token_ids)

        # Optional voice-prompt latent (zero-shot cloning); carried
        # through as a tensor input when present.
        prompt_wav_lat = inputs.get("prompt_wav_lat", [None])[0]
        return NodeInputs(
            tensor_inputs={
                "inputs_embeds": inputs_embeds,
                "prompt_wav_lat": prompt_wav_lat,
            }
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs_embeds: torch.Tensor,
        prompt_wav_lat: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """Run the full talker generation: AR latents -> VAE waveform.

        Returns ``{"audio_chunk": [waveform]}`` where waveform is
        ``(1, 1, num_samples)`` at the AudioVAE's sample rate. The
        text-length duration cap is applied to ``max_steps``.
        """
        text_len = inputs_embeds.shape[1]
        max_steps = self.generator.duration_capped_steps(text_len, self.max_steps)
        with torch.no_grad():
            latents = self.generator.generate_latents(
                inputs_embeds,
                prompt_wav_lat=prompt_wav_lat,
                min_new_token=self.min_new_token,
                max_steps=max_steps,
            )
            waveform = self.generator.decode_to_waveform(latents, stream_decode=True)
            waveform = self.generator.trim_trailing_silence(waveform)
        return {"audio_chunk": [waveform]}


# ===================================================================
# 5. ImageGenSubmodule (stateless diffusion — thinker hidden -> image)
# ===================================================================


class ImageGenSubmodule(NodeSubmodule):
    """Stateless image-generation node: thinker hidden states -> RGB image.

    Ming's thinker->imagegen bridge passes the thinker's final hidden states
    sliced at the learnable ``<imagePatch>`` query-token positions (the block
    appended by ``maybe_expand_image_gen_prompt`` in ``process_prompt``, step
    8b). The condition encoder turns those into the DiT's ``cap_feats``, and the
    diffusion pipeline runs the full flow-matching denoise + VAE decode in one
    ``forward`` call — like the Talker, the step count is internal
    (scheduler-determined), not a conductor decode loop. So a single STATELESS
    node with one ``EMIT_TO_CLIENT`` image edge suffices.

    The whole stack (condition encoder + DiT + VAE + optional ByT5) is owned by
    a :class:`MingImagePipeline`; this submodule only marshals inputs and calls
    ``pipeline.generate``.
    """

    def __init__(
        self,
        pipeline: "Any",  # MingImagePipeline (avoid import cycle at module load)
        config: MingFlashOmniModelConfig,
        default_params: "Any" = None,  # MingImageGenSamplingParams
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.config = config
        if default_params is None:
            from mstar.model.ming_omni_flash.components.imagegen_pipeline import (
                MingImageGenSamplingParams,
            )

            ig = config.image_gen
            default_params = MingImageGenSamplingParams(
                height=ig.default_height if ig is not None else 1024,
                width=ig.default_width if ig is not None else 1024,
                num_inference_steps=ig.num_inference_steps if ig is not None else 50,
                guidance_scale=ig.guidance_scale if ig is not None else 2.0,
            )
        self.default_params = default_params

    def get_stateless_flavor(self) -> str:
        # The DiT + VAE denoise loop is numerically sensitive (flow-matching
        # ODE); mirror the talker/audio_codec flavor (no torch.compile, no
        # autocast surprises).
        return "audio_codec"

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        """Pull the thinker hidden states at the query-token positions.

        Accepts either ``thinker_hidden_states`` (already sliced [N, H] or
        [1, N, H] by the thinker->imagegen bridge) or, in the unit-test path,
        a pre-built tensor. An optional ``negative_thinker_hidden_states``
        enables real (non-zero) CFG negatives.
        """
        if "thinker_hidden_states" in inputs and inputs["thinker_hidden_states"]:
            hidden = inputs["thinker_hidden_states"][0]
        else:
            raise ValueError(
                "ImageGenSubmodule: missing 'thinker_hidden_states'. The "
                "Thinker->ImageGen bridge must supply the query-token hidden "
                "states."
            )
        negative = inputs.get("negative_thinker_hidden_states", [None])[0]
        return NodeInputs(
            tensor_inputs={
                "thinker_hidden_states": hidden,
                "negative_thinker_hidden_states": negative,
            }
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        thinker_hidden_states: torch.Tensor,
        negative_thinker_hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """Run condition-encode -> denoise -> VAE decode, emit one image.

        Returns ``{"image": [img]}`` where ``img`` is a ``[B, 3, H, W]`` tensor
        in ``[-1, 1]`` (Z-Image VAE convention); the diffusion output adapter
        converts it to PIL/base64 downstream.
        """
        with torch.no_grad():
            image = self.pipeline.generate(
                thinker_hidden_states,
                self.default_params,
                negative_hidden=negative_thinker_hidden_states,
            )
        return {"image": [image]}
