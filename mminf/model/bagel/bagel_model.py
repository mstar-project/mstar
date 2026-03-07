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

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from safetensors.torch import load_file

from huggingface_hub import snapshot_download
import torch
import torch.nn as nn
import yaml

from mminf.communication.tensors import NameToTensorList
from mminf.graph.base import (
    GraphPointer,
    GraphSection,
    GraphStage,
    Loop,
    Sequential,
    TensorPointerInfo,
)
from mminf.engine.base import EngineType
from mminf.model.bagel.components.autoencoder import BagelAutoEncoder, BagelAutoEncoderConfig
from mminf.model.bagel.components.modeling_utils import BagelMLPconnector, PositionEmbedding, TimestepEmbedder
from mminf.model.bagel.components.language_model import BagelForCausalLM
from mminf.model.bagel.components.tokenization import BagelTokenizer
from mminf.model.bagel.components.vit_encoder import BagelVisionModel
from mminf.model.bagel.submodules import LLMSubmodule, VAEDecoderSubmodule, VAEEncoderSubmodule, ViTEncoderSubmodule
from mminf.model.bagel.utils import add_special_tokens
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
# Model Config
# ---------------------------------------------------------------------------


@dataclass
class BagelAutoEncoderConfig:
    resolution: int = 256
    in_channels: int = 3
    downsample: int = 8
    ch: int = 128
    out_ch: int = 3
    ch_mult: tuple[int] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    z_channels: int = 16
    scale_factor: float = 0.3611
    shift_factor: float = 0.1159

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "BagelAutoEncoderConfig":
        # Get field names from the dataclass
        field_names = {field.name for field in cls.__dataclass_fields__.values()}
        # Filter config_dict to only include known fields
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)


@dataclass
class BagelViTConfig:
    # ViT
    hidden_size=768
    intermediate_size=3072
    num_hidden_layers=12
    num_attention_heads=12
    num_channels=3
    image_size=224
    patch_size=16
    hidden_act="gelu_pytorch_tanh"
    layer_norm_eps=1e-6
    attention_dropout=0.0

    def __post_init__(self):
        self.rope = False
        self.num_hidden_layers -= 1

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "BagelModelConfig":
        # Get field names from the dataclass
        field_names = {field.name for field in cls.__dataclass_fields__.values()}
        # Filter config_dict to only include known fields
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)


@dataclass
class BagelModelConfig:
    vae_config: BagelAutoEncoderConfig
    vit_config: BagelViTConfig

    latent_patch_size: int = 2
    max_latent_size: int = 32
    num_timesteps: int = 50
    cfg_text_scale: float = 4.0
    cfg_img_scale: float = 1.5
    think_mode: bool = False
    vocab_size=151936
    hidden_size=4096
    intermediate_size=22016
    num_hidden_layers=32
    num_attention_heads=32
    num_key_value_heads=32
    hidden_act="silu"
    max_position_embeddings=32768
    initializer_range=0.02
    rms_norm_eps=1e-6
    use_cache=True
    rope_theta=10000.0
    rope_scaling=None
    use_sliding_window=False
    sliding_window=4096
    max_window_layers=28
    attention_dropout=0.0
    is_causal=True
    freeze_und=False
    connector_act="gelu_pytorch_tanh"
    vit_max_num_patch_per_side=70
    pad_token_id=1

    @classmethod
    def from_dict(
        cls, vae_config, vit_config,
        config_dict: dict[str, Any]
    ) -> "BagelModelConfig":
        # Get field names from the dataclass
        field_names = {field.name for field in cls.__dataclass_fields__.values()}
        # Filter config_dict to only include known fields
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(
            vae_config=vae_config,
            vit_config=vit_config,
            **filtered_dict
        )

    def __post_init__(self):
        self.latent_downsample = self.vae_config.downsample * self.latent_patch_size
        self.patch_latent_dim = self.latent_patch_size ** 2 \
            * self.vae_config.z_channels
        self.qk_norm = True
        self.tie_word_embeddings = False

# ---------------------------------------------------------------------------
# BagelModel
# ---------------------------------------------------------------------------

VAE_CONFIG_PATH = "vae_config.yaml"
VIT_CONFIG_PATH = "vit_config.yaml"
MODEL_CONFIG_PATH = "model_config.yaml"

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
        config_dir: str,
        model_path_hf: str,
    ):
        # Load configs
        vae_config_file = Path(config_dir) / VAE_CONFIG_PATH
        vit_config_file = Path(config_dir) / VIT_CONFIG_PATH
        model_config_file = Path(config_dir) / MODEL_CONFIG_PATH

        if vae_config_file.exists():
            with open(vae_config_file) as f:
                vae_config = BagelAutoEncoderConfig.from_dict(yaml.safe_load(f))
        else:
            vae_config = BagelAutoEncoderConfig()
        
        if vit_config_file.exists():
            with open(vit_config_file) as f:
                vit_config = BagelViTConfig.from_dict(yaml.safe_load(f))
        else:
            vit_config = BagelViTConfig()

        with open(model_config_file) as f:
            self.config = BagelModelConfig.from_dict(
                vae_config=vae_config,
                vit_config=vit_config,
                config_dict=yaml.safe_load(f)
            )

        self.model_path_hf = model_path_hf

        self.tokenizer = BagelTokenizer.from_pretrained(model_path_hf)
        self.tokenizer, new_token_ids, _ = add_special_tokens(self.tokenizer)

        # Special token IDs
        self.boi_token_id = new_token_ids.get("boi_token_id")   # <|vision_start|>
        self.eoi_token_id = new_token_ids.get("eoi_token_id")   # <|vision_end|>
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


    def _download_hf(self):
        if self.repo is not None:
            return
        cache_dir = snapshot_download(repo_id=self.model_path_hf)
        self.repo = Path(cache_dir)
    
    def _init_language_model_components(self):
        self._download_hf()
        self.language_model = BagelForCausalLM(self.config)
        self.llm2vae = nn.Linear(self.config.hidden_size, self.config.patch_latent_dim)

        ema_path = self.repo / "ema.safetensors"
        state_dict = load_file(ema_path)
        self.language_model.load_state_dict(state_dict, strict=False)
        self.llm2vae.load_state_dict(state_dict, strict=False)
        
    def _init_vae_components(self):
        self._download_hf()
        if self.vae_initialized:
            return
        self.vae_initialized = True
        self.latent_pos_embed = PositionEmbedding(
            self.config.max_latent_size, self.config.hidden_size
        )
        self.time_embedder = TimestepEmbedder(self.config.hidden_size)
        self.vae2llm = nn.Linear(self.config.patch_latent_dim, self.config.hidden_size)
        self.vae_model = BagelAutoEncoder(ae_params)

        # Load in weights: VAE
        ae_params = self.config.vae_config
        vae_path = self.repo / "ae.safetensors"
        state_dict = load_file(vae_path)
        self.vae_model.load_state_dict(state_dict, strict=False)

        # Load in weights: rest
        ema_path = self.repo / "ema.safetensors"
        state_dict = load_file(ema_path)
        self.time_embedder.load_state_dict(state_dict, strict=False)
        self.vae2llm.load_state_dict(state_dict, strict=False)
        self.latent_pos_embed.load_state_dict(state_dict, strict=False)

    def _init_vit_components(self):
        self._download_hf()
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

        # Load in weights
        ema_path = self.repo / "ema.safetensors"
        state_dict = load_file(ema_path)
        self.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
            self.config.vit_config, meta=True
        )
        self.vit_model.load_state_dict(state_dict, strict=False)
        self.connector.load_state_dict(state_dict, strict=False)
        self.vit_pos_embed.load_state_dict(state_dict, strict=False)


    # -----------------------------------------------------------------------
    # Lazy submodule initialization
    # -----------------------------------------------------------------------

    def _create_submodule(self, stage_name: str) -> StageSubmodule | None:
        """Create a submodule wrapper on first access."""

        if stage_name == "LLM":
            self._init_language_model_components()
            return LLMSubmodule(
                language_model=self.language_model,
                llm2vae=self.llm2vae,
                boi_token_id=self.boi_token_id,
                eoi_token_id=self.eoi_token_id,
            )
        elif stage_name == "vit_encoder":
            self._init_vit_components()
            return ViTEncoderSubmodule(
                vit_model=self.vit_model,
                connector=self.connector,
                vit_pos_embed=self.vit_pos_embed,
            )
        elif stage_name == "vae_encoder":
            self._init_vae_components()
            return VAEEncoderSubmodule(
                vae_model=self.vae_model,
                vae2llm=self.vae2llm,
                time_embedder=self.time_embedder,
                latent_pos_embed=self.latent_pos_embed,
                latent_patch_size=self.config.latent_patch_size,
                latent_channel=self.config.vae_config.z_channels,
                latent_downsample=self.config.latent_downsample,
            )
        elif stage_name == "vae_decoder":
            self._init_vit_components()
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

        if self.config.think_mode and self.tokenizer is not None:
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

    def get_submodule(self, stage_name: str) -> torch.nn.Module | None:
        if stage_name in self._submodule_cache:
            return self._submodule_cache[stage_name]
        submodule = self._create_submodule(stage_name)
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
                    GraphPointer(next_stage="LLM", name="vit_emb"),
                ],
            ),
            GraphStage(
                name="LLM",
                input_ids=["vit_emb"],
                outputs=[],
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
                    back_to_conductor=True,
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
                    input_ids=["latents"],
                    outputs=[
                        GraphPointer(next_stage="LLM", name="latents"),
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
        if self.config.think_mode:
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

        # 3. Annotate schedule with multi-cache metadata for generation mode.
        #    BAGEL's CFG requires 3 caches: main, cfg_img, cfg_text.
        #    - Text prefill: write to main + cfg_img (text-only cache)
        #    - Image prefill (vit/vae): write to main only
        #    - After last image: snapshot main -> cfg_text (system+image cache)
        #    Understanding mode: no annotations needed (default ["main"]).
        if not is_understanding:
            last_image_idx = None
            for i, (phase, _) in enumerate(schedule):
                if phase in ("prefill_vit", "prefill_vae"):
                    last_image_idx = i

            for i, (phase, step_kwargs) in enumerate(schedule):
                if phase == "prefill_text":
                    step_kwargs["cache_labels"] = ["main", "cfg_img"]
                elif phase in ("prefill_vit", "prefill_vae"):
                    step_kwargs["cache_labels"] = ["main"]
                    if i == last_image_idx:
                        step_kwargs["snapshot_after"] = ("main", "cfg_text")

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
                "num_timesteps": self.config.num_timesteps,
                "cfg_text_scale": self.config.cfg_text_scale,
                "cfg_img_scale": self.config.cfg_img_scale,
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
                    # System prompt -- tokenized by process_prompt() in data worker
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
            # Initial noise latents feed the LLM denoising loop.
            # Note: latents are typically initialized by the submodule's
            # preprocess() (random noise), not passed through persist_signals.
            # This lookup handles the case where latents are externally provided.
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
                    if self.config.think_mode:
                        # Think first: decode to generate reasoning, then
                        # EOS triggers transition to image_gen.
                        metadata.phase = "decode"
                    else:
                        metadata.phase = "image_gen"
            return metadata

        if metadata.phase == "decode":
            tokens = new_tokens.get("new_token", [])
            if self.eos_token_id is not None and self.eos_token_id in tokens:
                target = metadata.kwargs["target_output"]
                if self.config.think_mode and target == "image":
                    # Thinking phase complete — transition to image generation.
                    metadata.phase = "image_gen"
                else:
                    metadata.request_done = True
            # Otherwise stay in decode phase
            return metadata

        if metadata.phase == "image_gen":
            # Image generation complete (one image per request)
            metadata.request_done = True
            return metadata

        return metadata
