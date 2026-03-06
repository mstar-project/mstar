"""
BagelModel: Model implementation for BAGEL (ByteDance) unified multimodal model.

BAGEL uses a Qwen2 LLM with MoT (Mixture-of-Transformers) architecture,
SigLIP2 ViT for image understanding, and FLUX VAE for image generation.
The LLM itself serves as the denoiser for rectified flow image generation
(no separate diffusion model).

Architecture (4 stages):
    vit_encoder   (enc_dec) - SigLIP2 ViT + connector + pos embed
    vae_encoder   (enc_dec) - VAE encode + patchify + projection
    LLM           (ar)      - Fat stage: embed + Qwen2 + lm_head + CFG + Euler
    vae_decoder   (enc_dec) - VAE decode to pixels

Phases (5):
    prefill_text  - Text token embedding + LLM prefill (causal)
    prefill_vit   - ViT encoding + LLM prefill (bidirectional for images)
    prefill_vae   - VAE encoding + LLM prefill (bidirectional for images)
    decode        - Autoregressive text generation
    image_gen     - Flow matching loop (3-pass CFG + Euler) + VAE decode

The LLM stage absorbs text_emb, lm_head, and flow_proj because they are
always colocated on the same GPU. Keeping them as separate graph stages
would add unnecessary IPC overhead. CFG requires 3 LLM forward passes +
velocity combination, which is easier as one atomic operation.

Output mode is known upfront from the API request's output_modalities
field (no BOI token detection). Prefill is sequential: text tokens are
processed causally, then each image is processed bidirectionally.
"""

import torch
import torch.nn as nn

from mminf.communication.tensors import NameToTensorList
from mminf.graph.base import (
    GraphPointer,
    GraphSection,
    GraphStage,
    Loop,
    Sequential,
    TensorPointerInfo,
)
from mminf.model.base import STREAM_OUT, CurrentForwardMetadata, Model, StageSubmodule


# ---------------------------------------------------------------------------
# System prompts (used when think_mode=True)
# ---------------------------------------------------------------------------

VLM_THINK_SYSTEM_PROMPT = (
    "You should first think about the reasoning process in the mind "
    "and then provide the user with the answer."
)

GEN_THINK_SYSTEM_PROMPT = (
    "You should first think about the planning process in the mind "
    "and then generate the image."
)


# ---------------------------------------------------------------------------
# StageSubmodule wrappers
# ---------------------------------------------------------------------------


class ViTEncoderSubmodule(StageSubmodule):
    """SigLIP2 ViT + connector + vit_pos_embed: pixel patches -> ViT features.

    Receives preprocessed inputs containing packed pixel values, position IDs,
    cumulative sequence lengths, and max sequence length. Both vit_encoder and
    vae_encoder receive "image_inputs" as their graph input name; routing is
    handled by the graph pointer's next_stage field.
    """

    def __init__(
        self,
        vit_model: nn.Module,
        connector: nn.Module,
        vit_pos_embed: nn.Module,
    ):
        super().__init__()
        self.vit_model = vit_model
        self.connector = connector
        self.vit_pos_embed = vit_pos_embed

    def forward(
        self,
        packed_pixel_values: torch.Tensor,
        packed_position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
    ) -> NameToTensorList:
        features = self.vit_model(
            packed_pixel_values=packed_pixel_values,
            packed_flattened_position_ids=packed_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen.item() if max_seqlen.dim() == 0 else max_seqlen,
        )
        features = self.connector(features)
        pos_emb = self.vit_pos_embed(packed_position_ids)
        features = features + pos_emb
        return {"vit_emb": [features]}


class VAEEncoderSubmodule(StageSubmodule):
    """VAE encode + patchify + vae2llm + time_embedder + latent_pos_embed.

    Encodes an image tensor to VAE latents, patchifies them, and projects
    into the LLM hidden dimension with positional and timestep embeddings.
    """

    def __init__(
        self,
        vae_model: nn.Module,
        vae2llm: nn.Linear,
        time_embedder: nn.Module,
        latent_pos_embed: nn.Module,
        latent_patch_size: int,
        latent_channel: int,
    ):
        super().__init__()
        self.vae_model = vae_model
        self.vae2llm = vae2llm
        self.time_embedder = time_embedder
        self.latent_pos_embed = latent_pos_embed
        self.latent_patch_size = latent_patch_size
        self.latent_channel = latent_channel

    def forward(
        self,
        padded_images: torch.Tensor,
        packed_vae_position_ids: torch.Tensor,
        packed_timesteps: torch.Tensor,
        patchified_h: torch.Tensor,
        patchified_w: torch.Tensor,
    ) -> NameToTensorList:
        latent = self.vae_model.encode(padded_images)

        p = self.latent_patch_size
        h, w = patchified_h.item(), patchified_w.item()
        # Patchify: [batch, C, H, W] -> [num_patches, patch_dim]
        packed_latent = []
        for lat in latent:
            lat = lat[:, :h * p, :w * p].reshape(
                self.latent_channel, h, p, w, p
            )
            lat = torch.einsum("chpwq->hwpqc", lat).reshape(
                -1, p * p * self.latent_channel
            )
            packed_latent.append(lat)
        packed_latent = torch.cat(packed_latent, dim=0)

        # Project to hidden dim with timestep and position embeddings
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        return {"vae_emb": [packed_latent]}


class LLMSubmodule(StageSubmodule):
    """Fat LLM wrapper that dispatches based on phase.

    Absorbs text_emb, lm_head, and flow_proj into a single stage to avoid
    unnecessary IPC overhead. Phase-based dispatch handles:

      - prefill_text: embed_tokens -> LLM forward (causal, mode="und")
      - prefill_vit:  BOI + vit_emb + EOI -> LLM forward (bidirectional)
      - prefill_vae:  BOI + vae_emb + EOI -> LLM forward (bidirectional)
      - decode:       embed_tokens -> LLM forward -> lm_head -> argmax
      - image_gen:    3-pass CFG -> llm2vae -> velocity combine -> Euler step

    BOI/EOI tokens (<|vision_start|>, <|vision_end|>) are structural
    delimiters manually inserted around image embeddings during prefill.
    They are NOT predicted by the model (excluded from CE loss during
    training).

    During image_gen, classifier-free guidance requires 3 LLM forward
    passes with different KV caches (main, cfg_text, cfg_img). The
    velocities are combined via:
        v_final = v_cfg_img + img_scale * (
            v_cfg_text + text_scale * (v_main - v_cfg_text) - v_cfg_img
        )
    followed by an Euler step: x_{t+1} = x_t + v_final * dt.
    """

    def __init__(
        self,
        language_model: nn.Module,
        llm2vae: nn.Linear,
        boi_token_id: int | None = None,
        eoi_token_id: int | None = None,
    ):
        super().__init__()
        self.language_model = language_model
        self.embed_tokens = language_model.model.embed_tokens
        self.lm_head = language_model.lm_head
        self.llm2vae = llm2vae
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id

    def forward(self, phase: str, **kwargs) -> NameToTensorList:
        if phase == "prefill_text":
            return self._forward_prefill_text(**kwargs)
        elif phase == "prefill_vit":
            return self._forward_prefill_vit(**kwargs)
        elif phase == "prefill_vae":
            return self._forward_prefill_vae(**kwargs)
        elif phase == "decode":
            return self._forward_decode(**kwargs)
        elif phase == "image_gen":
            return self._forward_image_gen(**kwargs)
        else:
            raise ValueError(f"Unknown LLM phase: {phase!r}")

    def _forward_prefill_text(self, text_inputs: torch.Tensor, **kwargs) -> NameToTensorList:
        """embed_tokens -> LLM forward (causal, mode='und') -> KV cache update."""
        emb = self.embed_tokens(text_inputs)
        self.language_model(emb, is_causal=True, mode="und", **kwargs)
        return {"prefill_text_done": [torch.tensor([1])]}

    def _forward_prefill_vit(self, vit_emb: torch.Tensor, **kwargs) -> NameToTensorList:
        """Wrap vit_emb with BOI/EOI tokens -> LLM forward (bidirectional)."""
        combined = self._wrap_with_boi_eoi(vit_emb)
        self.language_model(combined, is_causal=False, mode="und", **kwargs)
        return {"prefill_vit_done": [torch.tensor([1])]}

    def _forward_prefill_vae(self, vae_emb: torch.Tensor, **kwargs) -> NameToTensorList:
        """Wrap vae_emb with BOI/EOI tokens -> LLM forward (bidirectional)."""
        combined = self._wrap_with_boi_eoi(vae_emb)
        self.language_model(combined, is_causal=False, mode="und", **kwargs)
        return {"prefill_vae_done": [torch.tensor([1])]}

    def _forward_decode(self, text_inputs: torch.Tensor, **kwargs) -> NameToTensorList:
        """embed_tokens -> LLM forward -> lm_head -> argmax."""
        emb = self.embed_tokens(text_inputs)
        hidden = self.language_model(emb, is_causal=True, mode="und", **kwargs)
        logits = self.lm_head(hidden[:, -1:])
        token = torch.argmax(logits, dim=-1)
        return {"new_token": [token]}

    def _forward_image_gen(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        next_timestep: torch.Tensor,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        **kwargs,
    ) -> NameToTensorList:
        """3-pass CFG -> llm2vae -> velocity combine -> Euler step."""
        # 3 LLM forwards with different KV caches
        v_main = self.language_model(
            latents, is_causal=False, mode="gen",
            cache_label="main", **kwargs,
        )
        v_cfg_text = self.language_model(
            latents, is_causal=False, mode="gen",
            cache_label="cfg_text", **kwargs,
        )
        v_cfg_img = self.language_model(
            latents, is_causal=False, mode="gen",
            cache_label="cfg_img", **kwargs,
        )

        # Project to VAE space
        v_main = self.llm2vae(v_main)
        v_cfg_text = self.llm2vae(v_cfg_text)
        v_cfg_img = self.llm2vae(v_cfg_img)

        # CFG velocity combination
        v_final = v_cfg_img + cfg_img_scale * (
            v_cfg_text + cfg_text_scale * (v_main - v_cfg_text) - v_cfg_img
        )

        # Euler step: x_{t+1} = x_t + v * dt
        dt = next_timestep - timestep
        latents = latents + v_final * dt
        return {"latents": [latents]}

    def _wrap_with_boi_eoi(self, emb: torch.Tensor) -> torch.Tensor:
        """Wrap embeddings with <|vision_start|> and <|vision_end|> tokens."""
        if self.boi_token_id is None or self.eoi_token_id is None:
            return emb
        device = emb.device
        boi_ids = torch.tensor([self.boi_token_id], device=device)
        eoi_ids = torch.tensor([self.eoi_token_id], device=device)
        boi_emb = self.embed_tokens(boi_ids)
        eoi_emb = self.embed_tokens(eoi_ids)
        return torch.cat([boi_emb, emb, eoi_emb], dim=0)


class VAEDecoderSubmodule(StageSubmodule):
    """VAE decoder: latent grid -> pixel image."""

    def __init__(
        self,
        vae_model: nn.Module,
        latent_patch_size: int,
        latent_channel: int,
        latent_downsample: int,
    ):
        super().__init__()
        self.vae_model = vae_model
        self.latent_patch_size = latent_patch_size
        self.latent_channel = latent_channel
        self.latent_downsample = latent_downsample

    def forward(
        self,
        latents: torch.Tensor,
        image_h: torch.Tensor,
        image_w: torch.Tensor,
    ) -> NameToTensorList:
        H, W = image_h.item(), image_w.item()
        p = self.latent_patch_size
        h = H // self.latent_downsample
        w = W // self.latent_downsample

        # Unpatchify: [num_patches, patch_dim] -> [1, C, H_latent, W_latent]
        latent = latents.reshape(1, h, w, p, p, self.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(
            1, self.latent_channel, h * p, w * p
        )
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return {"image_output": [image]}


# ---------------------------------------------------------------------------
# BagelModel
# ---------------------------------------------------------------------------


class BagelModel(Model):
    """
    BAGEL unified multimodal model (ByteDance).

    Architecture: Qwen2 LLM with MoT + SigLIP2 ViT + FLUX VAE.
    The LLM serves as both the autoregressive text model and the denoiser
    for rectified flow image generation (no separate diffusion model).

    Stages (4):
        vit_encoder   (enc_dec) - SigLIP2 ViT + connector + pos embed
        vae_encoder   (enc_dec) - VAE encode + patchify + projection
        LLM           (ar)      - Fat stage: embed + Qwen2 + lm_head + CFG
        vae_decoder   (enc_dec) - VAE decode to pixels

    Phases (5):
        prefill_text  - Text token embedding + LLM prefill (causal)
        prefill_vit   - ViT encoding + LLM prefill (bidirectional)
        prefill_vae   - VAE encoding + LLM prefill (bidirectional)
        decode        - Autoregressive text generation
        image_gen     - Flow matching loop (3-pass CFG + Euler) + VAE decode

    Phase transitions are schedule-driven (no BOI token detection). The
    output mode is known upfront from the API request's output_modalities.
    Prefill steps are constructed as a sequential schedule that walks
    through interleaved text and image inputs.
    """

    def __init__(
        self,
        bagel_model=None,
        vae_model=None,
        tokenizer=None,
        new_token_ids: dict | None = None,
        num_timesteps: int = 50,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        think_mode: bool = False,
    ):
        self.bagel_model = bagel_model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.num_timesteps = num_timesteps
        self.cfg_text_scale = cfg_text_scale
        self.cfg_img_scale = cfg_img_scale
        self.think_mode = think_mode

        # Special token IDs
        token_ids = new_token_ids or {}
        self.boi_token_id = token_ids.get("boi_token_id")   # <|vision_start|>
        self.eoi_token_id = token_ids.get("eoi_token_id")   # <|vision_end|>
        self.eos_token_id = token_ids.get("eos_token_id")
        self.bos_token_id = token_ids.get("bos_token_id")

        # Lazy init cache -- submodules created on first access via
        # get_submodule(). A worker only instantiates the submodules it
        # actually needs (e.g., a worker running only vit_encoder never
        # creates the LLMSubmodule).
        self._submodule_cache: dict[str, StageSubmodule | None] = {}

    # -----------------------------------------------------------------------
    # Lazy submodule initialization
    # -----------------------------------------------------------------------

    def _create_submodule(self, stage_name: str) -> StageSubmodule | None:
        """Create a submodule wrapper on first access. Returns None in dummy mode."""
        if self.bagel_model is None:
            return None

        if stage_name == "LLM":
            return LLMSubmodule(
                language_model=self.bagel_model.language_model,
                llm2vae=self.bagel_model.llm2vae,
                boi_token_id=self.boi_token_id,
                eoi_token_id=self.eoi_token_id,
            )
        elif stage_name == "vit_encoder":
            if not hasattr(self.bagel_model, "vit_model"):
                return None
            return ViTEncoderSubmodule(
                vit_model=self.bagel_model.vit_model,
                connector=self.bagel_model.connector,
                vit_pos_embed=self.bagel_model.vit_pos_embed,
            )
        elif stage_name == "vae_encoder":
            if self.vae_model is None:
                return None
            return VAEEncoderSubmodule(
                vae_model=self.vae_model,
                vae2llm=self.bagel_model.vae2llm,
                time_embedder=self.bagel_model.time_embedder,
                latent_pos_embed=self.bagel_model.latent_pos_embed,
                latent_patch_size=self.bagel_model.latent_patch_size,
                latent_channel=self.bagel_model.latent_channel,
            )
        elif stage_name == "vae_decoder":
            if self.vae_model is None:
                return None
            return VAEDecoderSubmodule(
                vae_model=self.vae_model,
                latent_patch_size=self.bagel_model.latent_patch_size,
                latent_channel=self.bagel_model.latent_channel,
                latent_downsample=self.bagel_model.latent_downsample,
            )
        return None

    # -----------------------------------------------------------------------
    # Model ABC implementation
    # -----------------------------------------------------------------------

    def get_submodule(self, stage_name: str) -> torch.nn.Module | None:
        if stage_name in self._submodule_cache:
            return self._submodule_cache[stage_name]
        submodule = self._create_submodule(stage_name)
        self._submodule_cache[stage_name] = submodule
        return submodule

    def get_stage_engine_types(self) -> dict[str, str]:
        return {
            "vit_encoder": "enc_dec",
            "vae_encoder": "enc_dec",
            "LLM": "ar",
            "vae_decoder": "enc_dec",
        }

    def get_phase_graphs(self) -> dict[str, GraphSection]:
        # -- prefill_text: just the LLM stage (text embedding is internal) --
        prefill_text = GraphStage(
            name="LLM",
            input_ids=["text_inputs"],
            outputs=[
                GraphPointer(
                    next_stage=STREAM_OUT,
                    name="prefill_text_done",
                    output_modality="text",
                    is_new_token=False,
                ),
            ],
        )

        # -- prefill_vit: ViT encoder -> LLM --
        prefill_vit = Sequential([
            GraphStage(
                name="vit_encoder",
                input_ids=["image_inputs"],
                outputs=[
                    GraphPointer(next_stage="LLM", name="vit_emb"),
                ],
            ),
            GraphStage(
                name="LLM",
                input_ids=["vit_emb"],
                outputs=[
                    GraphPointer(
                        next_stage=STREAM_OUT,
                        name="prefill_vit_done",
                        output_modality="text",
                        is_new_token=False,
                    ),
                ],
            ),
        ])

        # -- prefill_vae: VAE encoder -> LLM --
        prefill_vae = Sequential([
            GraphStage(
                name="vae_encoder",
                input_ids=["image_inputs"],
                outputs=[
                    GraphPointer(next_stage="LLM", name="vae_emb"),
                ],
            ),
            GraphStage(
                name="LLM",
                input_ids=["vae_emb"],
                outputs=[
                    GraphPointer(
                        next_stage=STREAM_OUT,
                        name="prefill_vae_done",
                        output_modality="text",
                        is_new_token=False,
                    ),
                ],
            ),
        ])

        # -- decode: single LLM stage (embed + transformer + lm_head) --
        decode = GraphStage(
            name="LLM",
            input_ids=["text_inputs"],
            outputs=[
                GraphPointer(
                    next_stage=STREAM_OUT,
                    name="new_token",
                    output_modality="text",
                    is_new_token=True,
                ),
            ],
        )

        # -- image_gen: denoising loop (LLM does CFG+Euler) -> VAE decode --
        image_gen = Sequential([
            Loop(
                section=GraphStage(
                    name="LLM",
                    input_ids=["latents"],
                    outputs=[
                        GraphPointer(next_stage="LLM", name="latents"),
                    ],
                ),
                n_iters=self.num_timesteps - 1,
                outputs=[
                    GraphPointer(next_stage="vae_decoder", name="latents"),
                ],
            ),
            GraphStage(
                name="vae_decoder",
                input_ids=["latents"],
                outputs=[
                    GraphPointer(
                        next_stage=STREAM_OUT,
                        name="image_output",
                        output_modality="image",
                        back_to_conductor=True,
                    ),
                ],
            ),
        ])

        return dict(
            prefill_text=prefill_text,
            prefill_vit=prefill_vit,
            prefill_vae=prefill_vae,
            decode=decode,
            image_gen=image_gen,
        )

    def get_initial_forward_metadata(
        self,
        input_modalities: list[str],
        output_modalities: list[str],
    ) -> CurrentForwardMetadata:
        target_output = output_modalities[0]  # "text" or "image"
        is_understanding = (target_output == "text")

        # Build prefill schedule: sequential list of (phase_name, step_kwargs)
        schedule: list[tuple[str, dict]] = []

        # 1. System prompt (if think mode enabled)
        if self.think_mode:
            prompt = VLM_THINK_SYSTEM_PROMPT if is_understanding else GEN_THINK_SYSTEM_PROMPT
            schedule.append(("prefill_text", {"prompt": prompt}))

        # 2. Walk through interleaved inputs, building sequential steps
        text_idx, image_idx = 0, 0
        for mod in input_modalities:
            if mod == "text":
                schedule.append(("prefill_text", {"input_idx": text_idx}))
                text_idx += 1
            elif mod == "image":
                if is_understanding:
                    # Understanding: ViT only (no VAE encoding needed)
                    schedule.append(("prefill_vit", {"input_idx": image_idx}))
                else:
                    # Generation/editing: VAE encode the image
                    schedule.append(("prefill_vae", {"input_idx": image_idx}))
                image_idx += 1

        first_phase = schedule[0][0] if schedule else "decode"

        return CurrentForwardMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            phase=first_phase,
            is_prefill=bool(schedule),
            kwargs={
                "prefill_schedule": schedule,
                "prefill_step": 0,
                "target_output": target_output,
                "num_timesteps": self.num_timesteps,
                "cfg_text_scale": self.cfg_text_scale,
                "cfg_img_scale": self.cfg_img_scale,
            },
        )

    def get_forward_pass_inputs(
        self,
        metadata: CurrentForwardMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        prev_forward_metadata: CurrentForwardMetadata = None,
    ) -> list[GraphPointer]:
        """Construct the external inputs for the current forward pass.

        The conductor calls this to determine what tensors to send to
        workers at the start of each forward pass. For prefill phases,
        the schedule entry determines which input to route; for decode
        and image_gen, the previous output feeds back in.

        persist_signals key conventions:
            "text_inputs"    - list of per-turn text TensorPointerInfos
            "image_inputs"   - list of per-image TensorPointerInfos
            "system_prompt"  - tokenized system prompt (if think_mode)
            "new_token"      - last generated token (during decode)
            "latents"        - noise latents (for image_gen entry)
        """
        phase = metadata.phase

        if metadata.is_prefill:
            schedule = metadata.kwargs["prefill_schedule"]
            step = metadata.kwargs["prefill_step"]
            _, step_kwargs = schedule[step]

            if phase == "prefill_text":
                ptr = GraphPointer(next_stage="LLM", name="text_inputs")
                if "prompt" in step_kwargs:
                    # System prompt -- conductor tokenizes and stores it
                    ptr.tensor_info = persist_signals.get("system_prompt", [])
                else:
                    idx = step_kwargs["input_idx"]
                    all_text = persist_signals.get("text_inputs", [])
                    ptr.tensor_info = [all_text[idx]] if idx < len(all_text) else []
                return [ptr]

            elif phase == "prefill_vit":
                idx = step_kwargs["input_idx"]
                ptr = GraphPointer(next_stage="vit_encoder", name="image_inputs")
                all_images = persist_signals.get("image_inputs", [])
                ptr.tensor_info = [all_images[idx]] if idx < len(all_images) else []
                return [ptr]

            elif phase == "prefill_vae":
                idx = step_kwargs["input_idx"]
                ptr = GraphPointer(next_stage="vae_encoder", name="image_inputs")
                all_images = persist_signals.get("image_inputs", [])
                ptr.tensor_info = [all_images[idx]] if idx < len(all_images) else []
                return [ptr]

        elif phase == "decode":
            # Previous token feeds back as text_inputs
            ptr = GraphPointer(next_stage="LLM", name="text_inputs")
            ptr.tensor_info = persist_signals.get("new_token", [])
            return [ptr]

        elif phase == "image_gen":
            # Initial noise latents feed the LLM denoising loop
            ptr = GraphPointer(next_stage="LLM", name="latents")
            ptr.tensor_info = persist_signals.get("latents", [])
            return [ptr]

        return []

    def update_for_next_forward(
        self,
        metadata: CurrentForwardMetadata,
        new_tokens: dict[str, list[int]],
    ) -> CurrentForwardMetadata:
        """Advance phase transitions. Schedule-driven, no BOI detection.

        During prefill, steps through the schedule one entry at a time.
        After all prefill steps, transitions to decode (text output) or
        image_gen (image output) based on target_output set at init.

        During decode, checks for EOS token to mark request complete.
        After image_gen, marks request complete (one image per request).
        """
        if metadata.is_prefill:
            step = metadata.kwargs["prefill_step"] + 1
            schedule = metadata.kwargs["prefill_schedule"]

            if step < len(schedule):
                # More prefill steps remaining
                metadata.kwargs["prefill_step"] = step
                metadata.phase = schedule[step][0]
            else:
                # All prefill done -- transition based on target_output
                metadata.is_prefill = False
                target = metadata.kwargs["target_output"]
                if target == "text":
                    metadata.phase = "decode"
                elif target == "image":
                    metadata.phase = "image_gen"
            return metadata

        if metadata.phase == "decode":
            # Check for EOS
            tokens = new_tokens.get("new_token", [])
            if self.eos_token_id is not None and self.eos_token_id in tokens:
                metadata.kwargs["done"] = True
            # Otherwise stay in decode phase
            return metadata

        if metadata.phase == "image_gen":
            # Image generation complete (one image per request)
            metadata.kwargs["done"] = True
            return metadata

        return metadata
