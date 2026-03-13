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

import io
import json
import logging
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from torch import nn

from mminf.communication.tensors import NameToTensorList
from mminf.engine.ar_engine import KVCacheConfig
from mminf.engine.base import EngineType
from mminf.graph.base import (
    GraphPointer,
    GraphSection,
    GraphStage,
    Loop,
    Sequential,
    TensorPointerInfo,
)
from mminf.graph.special_destinations import STREAM_OUT
from mminf.model.bagel.components.autoencoder import BagelAutoEncoder
from mminf.model.bagel.components.language_model import BagelForCausalLM
from mminf.model.bagel.components.modeling_utils import BagelMLPconnector, PositionEmbedding, TimestepEmbedder
from mminf.model.bagel.components.tokenization import BagelTokenizer, add_special_tokens
from mminf.model.bagel.components.vit_encoder import BagelVisionModel
from mminf.model.bagel.config import load_bagel_config
from mminf.model.bagel.submodules import LLMSubmodule, VAEDecoderSubmodule, VAEEncoderSubmodule, ViTEncoderSubmodule
from mminf.model.base import CurrentForwardMetadata, ForwardPassArgs, Model, StageSubmodule
from mminf.model.utils import ModuleAndPrefix, load_weights_from_file

logger = logging.getLogger(__name__)

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
        model_path_hf: str,
        **kwargs
    ):
        config_path = hf_hub_download(repo_id=model_path_hf, filename="config.json", revision=None)
        with open(config_path) as f:
            self.config = load_bagel_config(json.load(f))

        self.model_path_hf = model_path_hf

        self.tokenizer = BagelTokenizer.from_pretrained(model_path_hf)
        self.tokenizer, new_token_ids, _ = add_special_tokens(self.tokenizer)

        # Special token IDs
        self.boi_token_id = new_token_ids.get("start_of_image")   # <|vision_start|>
        self.eoi_token_id = new_token_ids.get("end_of_image")   # <|vision_end|>
        self.eos_token_id = new_token_ids.get("eos_token_id")
        self.bos_token_id = new_token_ids.get("bos_token_id")

        # Lazy init cache -- submodules created on first access via
        # get_submodule(). A worker only instantiates the submodules it
        # actually needs (e.g., a worker running only vit_encoder never
        # creates the LLMSubmodule).
        self._submodule_cache: dict[str, StageSubmodule | None] = {}
        self.language_model = None
        self.llm2vae = None
        self.vae_model = None
        self.time_embedder = None
        self.vae2llm = None
        self.latent_pos_embed = None
        self.vit_model = None

        self.repo = None
        self.vae_initialized = False
        self.llm_initialized = False


    def _download_hf(self):
        if self.repo is not None:
            return
        cache_dir = snapshot_download(repo_id=self.model_path_hf)
        self.repo = Path(cache_dir)

    def _init_language_model_components(self, device):
        self._download_hf()
        self.llm_initialized = True
        with torch.device("meta"):
            self.language_model = BagelForCausalLM(self.config)
            self.llm2vae = nn.Linear(self.config.hidden_size, self.config.patch_latent_dim)

        ema_path = self.repo / "ema.safetensors"

        load_weights_from_file(
            ema_path,
            modules=[
                ModuleAndPrefix(self.language_model, prefix="language_model"),
                ModuleAndPrefix(self.llm2vae, prefix="llm2vae"),
            ],
            device=device
        )

        if not self.vae_initialized:
            # Need these for image gen
            with torch.device("meta"):
                self.latent_pos_embed = PositionEmbedding(
                    self.config.max_latent_size, self.config.hidden_size
                )
                self.time_embedder = TimestepEmbedder(self.config.hidden_size)
                self.vae2llm = nn.Linear(self.config.patch_latent_dim, self.config.hidden_size)
            load_weights_from_file(
                ema_path,
                modules=[
                    ModuleAndPrefix(self.vae2llm, prefix="vae2llm"),
                    ModuleAndPrefix(self.time_embedder, prefix="time_embedder"),
                    ModuleAndPrefix(self.latent_pos_embed, prefix="latent_pos_embed")
                ],
                device=device
            )

    def _init_vae_components(self, device):
        self._download_hf()
        if self.vae_initialized:
            return
        self.vae_initialized = True
        ae_params = self.config.vae_config
        with torch.device("meta"):
            self.vae_model = BagelAutoEncoder(ae_params)

        # Load in weights: VAE
        vae_path = self.repo / "ae.safetensors"
        load_weights_from_file(
            vae_path,
            modules=[ModuleAndPrefix(
                self.vae_model
            )],
            device=device
        )

        if not self.llm_initialized:
            # LLM components also need these for image gen, so these
            # might already be initialized by _init_language_model_components()
            with torch.device("meta"):
                self.latent_pos_embed = PositionEmbedding(
                    self.config.max_latent_size, self.config.hidden_size
                )
                self.time_embedder = TimestepEmbedder(self.config.hidden_size)
                self.vae2llm = nn.Linear(self.config.patch_latent_dim, self.config.hidden_size)
            ema_path = self.repo / "ema.safetensors"
            load_weights_from_file(
                ema_path,
                modules=[
                    ModuleAndPrefix(self.vae2llm, prefix="vae2llm"),
                    ModuleAndPrefix(self.time_embedder, prefix="time_embedder"),
                    ModuleAndPrefix(self.latent_pos_embed, prefix="latent_pos_embed")
                ],
                device=device
            )


    def _init_vit_components(self, device):
        self._download_hf()
        with torch.device("meta"):
            self.vit_model = BagelVisionModel(self.config.vit_config)
            self.connector = BagelMLPconnector(
                self.config.vit_config.hidden_size,
                self.config.hidden_size,
                self.config.connector_act
            )
            self.vit_pos_embed = PositionEmbedding(
                self.config.vit_max_num_patch_per_side,
                self.config.hidden_size
            )
        self.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
            self.config.vit_config,
        )

        ema_path = self.repo / "ema.safetensors"
        load_weights_from_file(
            ema_path,
            modules=[
                ModuleAndPrefix(self.vit_model, prefix="vit_model"),
                ModuleAndPrefix(self.connector, prefix="connector"),
                ModuleAndPrefix(self.vit_pos_embed, prefix="vit_pos_embed")
            ],
            device=device
        )


    # -----------------------------------------------------------------------
    # Lazy submodule initialization
    # -----------------------------------------------------------------------

    def _create_submodule(self, stage_name: str, device: str) -> StageSubmodule | None:
        """Create a submodule wrapper on first access."""
        logger.debug("Creating submodule for BAGEL model stage %s", stage_name)
        if stage_name == "LLM":
            self._init_language_model_components(device)
            return LLMSubmodule(
                language_model=self.language_model,
                llm2vae=self.llm2vae,
                vae2llm=self.vae2llm,
                time_embedder=self.time_embedder,
                latent_pos_embed=self.latent_pos_embed,
                config=self.config,
                boi_token_id=self.boi_token_id,
                eoi_token_id=self.eoi_token_id,
                bos_token_id=self.bos_token_id,
                eos_token_id=self.eos_token_id
            )
        elif stage_name == "vit_encoder":
            self._init_vit_components(device)
            return ViTEncoderSubmodule(
                vit_model=self.vit_model,
                connector=self.connector,
                vit_pos_embed=self.vit_pos_embed,
                vit_patch_size=self.config.vit_config.patch_size,
                vit_max_num_patch_per_side=self.config.vit_max_num_patch_per_side
            )
        elif stage_name == "vae_encoder":
            self._init_vae_components(device)
            return VAEEncoderSubmodule(
                vae_model=self.vae_model,
                vae2llm=self.vae2llm,
                time_embedder=self.time_embedder,
                latent_pos_embed=self.latent_pos_embed,
                latent_patch_size=self.config.latent_patch_size,
                latent_channel=self.config.vae_config.z_channels,
                latent_downsample=self.config.latent_downsample,
                max_latent_size=self.config.max_latent_size,
            )
        elif stage_name == "vae_decoder":
            self._init_vae_components(device)
            return VAEDecoderSubmodule(
                vae_model=self.vae_model,
                latent_patch_size=self.config.latent_patch_size,
                latent_channel=self.config.vae_config.z_channels,
                latent_downsample=self.config.latent_downsample,
            )
        return None

    # -----------------------------------------------------------------------
    # Model ABC implementation
    # -----------------------------------------------------------------------

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        **kwargs,
    ) -> NameToTensorList:
        """Tokenize user prompt and system prompt (if think_mode).

        Returns model-specific keys matching get_forward_pass_inputs:
            "text_inputs"    - tokenized user prompt
            "system_prompt"  - tokenized system prompt (think_mode only)
        """
        result: NameToTensorList = {}

        if prompt is not None:
            if self.tokenizer is not None:
                tokens = self.tokenizer.encode(prompt)
                result["text_inputs"] = [
                    torch.tensor(tokens, dtype=torch.long)
                ]
            else:
                # Fallback for testing without a tokenizer
                byte_data = prompt.encode("utf-8")
                result["text_inputs"] = [
                    torch.tensor(list(byte_data), dtype=torch.uint8)
                ]

        think_mode = kwargs.get("think_mode", self.config.think_mode)
        if think_mode and self.tokenizer is not None:
            target_output = output_modalities[0] if output_modalities else "text"
            is_understanding = (target_output == "text")
            sys_prompt = (
                VLM_THINK_SYSTEM_PROMPT if is_understanding
                else GEN_THINK_SYSTEM_PROMPT
            )
            sys_tokens = self.tokenizer.encode(sys_prompt)
            result["system_prompt"] = [
                torch.tensor(sys_tokens, dtype=torch.long)
            ]

        return result

    def postprocess(
        self, output: torch.Tensor,
        modality: str # text | image | video | audio
    ) -> bytes:
        if modality == "text":
            detok = self.tokenizer.decode(output)
            logger.debug("OUTPUT TEXT %s", detok)
            return detok.encode("utf-8")
        if modality == "image":
            output = output[0].permute(1, 2, 0) * 255
            img = Image.fromarray((output).to(torch.uint8).cpu().numpy())
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            return img_byte_arr.getvalue()
        raise ValueError(f"Unsupported modality: {modality!r}")

    def get_kv_cache_config(self) -> KVCacheConfig:
        return KVCacheConfig(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.hidden_size // self.config.num_attention_heads,
            max_seq_len=self.config.max_position_embeddings,
            num_qo_heads=self.config.num_attention_heads,
        )

    def get_submodule(self, stage_name: str, device: str="cpu") -> torch.nn.Module | None:
        if stage_name in self._submodule_cache:
            return self._submodule_cache[stage_name]
        submodule = self._create_submodule(stage_name, device)
        logger.info(f"Successfully loaded in BAGEL submodule for {stage_name}")
        self._submodule_cache[stage_name] = submodule
        return submodule

    def get_stage_engine_types(self) -> dict[str, EngineType]:
        return {
            "vit_encoder": EngineType.ENC_DEC,
            "vae_encoder": EngineType.ENC_DEC,
            "LLM": EngineType.AR,
            "vae_decoder": EngineType.ENC_DEC,
        }

    def get_phase_graphs(self) -> dict[str, GraphSection]:
        # -- prefill_text: just the LLM stage (text embedding is internal) --
        # No output needed — conductor is notified when the subgraph completes.
        prefill_text = GraphStage(
            name="LLM",
            input_ids=["text_inputs"],
            outputs=[],
        )

        # -- prefill_vit: ViT encoder -> LLM --
        prefill_vit = Sequential([
            GraphStage(
                name="vit_encoder",
                input_ids=["image_inputs"],
                outputs=[
                    GraphPointer(next_stage="LLM", name="img_emb"),
                ],
            ),
            GraphStage(
                name="LLM",
                input_ids=["img_emb"],
                outputs=[],
            ),
        ])

        # -- prefill_vae: VAE encoder -> LLM --
        prefill_vae = Sequential([
            GraphStage(
                name="vae_encoder",
                input_ids=["image_inputs"],
                outputs=[
                    GraphPointer(next_stage="LLM", name="img_emb"),
                ],
            ),
            GraphStage(
                name="LLM",
                input_ids=["img_emb"],
                outputs=[],
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
                    persist=True,
                ),
            ],
        )

        # -- image_gen: denoising loop (LLM does CFG+Euler) -> VAE decode --
        # n_iters = num_timesteps - 1 because the loop body performs one
        # Euler step per iteration. With N timestep boundaries (e.g. 50),
        # there are N-1 intervals, so N-1 Euler steps are needed.
        image_gen = Sequential([
            Loop(
                section=GraphStage(
                    name="LLM",
                    input_ids=["latents", "time_index"],
                    outputs=[
                        GraphPointer(next_stage="LLM", name="latents"),
                        GraphPointer(next_stage="LLM", name="time_index"),
                    ],
                ),
                n_iters=self.config.num_timesteps - 1,
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
                        persist=True,
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

    def _build_prefill_schedule(
        self, input_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        is_understanding: bool,
        think_mode: bool
    ):
        # Build prefill schedule: sequential list of (phase_name, input tensor info)
        schedule: list[tuple[str, TensorPointerInfo]] = []

        # 1. System prompt (if think mode enabled)
        if think_mode and "system_prompt" in input_signals:
            schedule.append(("prefill_text", input_signals["system_prompt"][0]))

        # 2. Walk through interleaved inputs, building sequential steps
        images = input_signals.get("image_inputs", [])
        texts = input_signals.get("text_inputs", [])

        text_idx, image_idx = 0, 0
        for mod in input_modalities:
            if mod == "text":
                if text_idx >= len(texts):
                    continue
                schedule.append(("prefill_text", texts[text_idx]))
                text_idx += 1
            elif mod == "image":
                if image_idx >= len(images):
                    continue
                if not is_understanding:
                    # Generation/editing: VAE encode the image
                    schedule.append(("prefill_vae", images[image_idx]))
                schedule.append(("prefill_vit", images[image_idx]))
                image_idx += 1
        return schedule

    def _get_step_metadata(
        self, full_metadata: CurrentForwardMetadata,
    ) -> dict:
        requires_cfg = (
            full_metadata.kwargs["target_output"] == "image" \
                and (
                    full_metadata.kwargs["cfg_img_scale"] > 1.0 \
                    or full_metadata.kwargs["cfg_text_scale"] > 1.0
                )
        )
        return {
            "requires_cfg": requires_cfg,
            "is_prefill": full_metadata.is_prefill,
            "cfg_text_scale": full_metadata.kwargs["cfg_text_scale"],
            "cfg_img_scale": full_metadata.kwargs["cfg_img_scale"],
            "cfg_interval": full_metadata.kwargs["cfg_interval"],
            "cfg_renorm_type": full_metadata.kwargs["cfg_renorm_type"],
            "cfg_renorm_min": full_metadata.kwargs["cfg_renorm_min"],
        }

    def _get_fwd_pass_inputs(
        self,
        metadata: CurrentForwardMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
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
            input_tensor_info = [schedule[step][1]]

            if phase == "prefill_text":
                ptr = GraphPointer(next_stage="LLM", name="text_inputs")
            elif phase == "prefill_vit":
                ptr = GraphPointer(next_stage="vit_encoder", name="image_inputs")
            elif phase == "prefill_vae":
                ptr = GraphPointer(next_stage="vae_encoder", name="image_inputs")
            else:
                raise ValueError(f"Unrecognized prefill phase {phase}")
            ptr.tensor_info = input_tensor_info
            return [ptr]

        elif phase == "decode":
            # Previous token feeds back as text_inputs
            ptr = GraphPointer(next_stage="LLM", name="text_inputs")
            ptr.tensor_info = persist_signals.get("new_token", [])
            return [ptr]

        elif phase == "image_gen":
            # Initial noise latents feed the LLM denoising loop.
            # Note: latents are typically initialized by the submodule's
            # preprocess() (random noise), not passed through persist_signals.
            # This lookup handles the case where latents are externally provided.
            ptr = GraphPointer(next_stage="LLM", name="latents")
            ptr.tensor_info = persist_signals.get("latents", [])
            return [
                ptr,
                GraphPointer(next_stage="LLM", name="time_index")
            ]

        return []

    def _get_unpersist_tensors(
        self, phase: str, inputs: list[GraphPointer]
    ) -> list[TensorPointerInfo]:
        """
        Lists the tensors that will be used for the last time in this forward pass
        """
        if phase == "prefill_vae":
            # If we have prefill_vae, we know that the image input will be
            # passed into the ViT encoder for the next forward pass, so it
            # has to stick around
            return []
        # otherwise, we can un-persist all tensors
        return sum(
            [inp.tensor_info for inp in inputs], start=[]
        )

    def get_initial_forward_pass_args(
        self, input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        target_output = output_modalities[0]  # "text" or "image"

        # Per-request overrides with config defaults
        overridable_keys = [
            "cfg_text_scale", "cfg_img_scale", "cfg_interval",
            "cfg_renorm_type", "cfg_renorm_min", "think_mode",
        ]
        params = {k: getattr(self.config, k) for k in overridable_keys}
        if model_kwargs:
            for key in overridable_keys:
                if key in model_kwargs:
                    params[key] = model_kwargs[key]

        think_mode = params.pop("think_mode") # used for schedule logic, not stored in params
        schedule = self._build_prefill_schedule(
            input_modalities=input_modalities,
            input_signals=input_signals,
            is_understanding=(target_output == "text"),
            think_mode=think_mode
        )

        first_phase = schedule[0][0] if schedule else "decode"
        full_metadata = CurrentForwardMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            phase=first_phase,
            is_prefill=bool(schedule),
            kwargs={
                "prefill_schedule": schedule,
                "prefill_step": 0,
                "target_output": target_output,
                "num_timesteps": self.config.num_timesteps,
                "think_mode": think_mode,
                **params,  # CFG params
            },
        )
        step_metadata =  self._get_step_metadata(full_metadata)
        inputs = self._get_fwd_pass_inputs(
            full_metadata, input_signals
        )
        unpersist_tensors = self._get_unpersist_tensors(first_phase, inputs)
        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata=step_metadata
        )

    def get_forward_pass_args(
        self, metadata: CurrentForwardMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        new_tokens: dict[str, list[int]],
    ) -> ForwardPassArgs:
        """Advance phase transitions. Schedule-driven, no BOI detection.

        During prefill, steps through the schedule one entry at a time.
        After all prefill steps, transitions to:
          - decode (text output)
          - decode (image output + think_mode: think first, then generate)
          - image_gen (image output, no think_mode)

        During decode:
          - Text output: EOS marks request complete.
          - Image output + think_mode: EOS transitions to image_gen
            (thinking is done, now generate the image).

        After image_gen, marks request complete (one image per request).
        """
        request_done = False
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
                    if metadata.kwargs.get("think_mode", False):
                        # Think first: decode to generate reasoning, then
                        # EOS triggers transition to image_gen.
                        metadata.phase = "decode"
                    else:
                        metadata.phase = "image_gen"
        elif metadata.phase == "decode":
            tokens = new_tokens.get("new_token", [])
            if self.eos_token_id is not None and self.eos_token_id in tokens:
                target = metadata.kwargs["target_output"]
                if metadata.kwargs.get("think_mode", False) and target == "image":
                    # Thinking phase complete — transition to image generation.
                    metadata.phase = "image_gen"
                else:
                    request_done = True

        elif metadata.phase == "image_gen":
            # Image generation complete (one image per request)
            request_done = True

        step_metadata =  self._get_step_metadata(metadata)
        inputs = self._get_fwd_pass_inputs(
            metadata, persist_signals
        )
        unpersist_tensors = self._get_unpersist_tensors(
            metadata.phase, inputs
        )
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata=step_metadata,
            request_done=request_done
        )
