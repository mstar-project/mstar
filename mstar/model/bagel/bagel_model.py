"""
BagelModel: Model implementation for BAGEL (ByteDance) unified multimodal model.

BAGEL uses a Qwen2 LLM with MoT (Mixture-of-Transformers) architecture,
SigLIP2 ViT for image understanding, and FLUX VAE for image generation.
The LLM itself serves as the denoiser for rectified flow image generation
(no separate diffusion model).

Architecture (4 nodes):
    vit_encoder   (enc_dec) - SigLIP2 ViT + connector + pos embed
    vae_encoder   (enc_dec) - VAE encode + patchify + projection
    LLM           (ar)      - Fat node: embed + Qwen2 + lm_head + CFG + Euler
    vae_decoder   (enc_dec) - VAE decode to pixels

Graph walks (5):
    prefill_text  - Text token embedding + LLM prefill (causal)
    prefill_vit   - ViT encoding + LLM prefill (bidirectional for images)
    prefill_vae   - VAE encoding + LLM prefill (bidirectional for images)
    decode        - Autoregressive text generation
    image_gen     - Flow matching loop (3-pass CFG + Euler) + VAE decode

The LLM node absorbs text_emb, lm_head, and flow_proj because they are
always colocated on the same GPU. Keeping them as separate graph nodes
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

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.engine.base import EngineType
from mstar.engine.kv_store import KVCacheConfig
from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    GraphSection,
    Loop,
    Parallel,
    Sequential,
    TensorPointerInfo,
)
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.bagel.components.autoencoder import BagelAutoEncoder
from mstar.model.bagel.components.language_model import BagelForCausalLM
from mstar.model.bagel.components.modeling_utils import BagelMLPconnector, PositionEmbedding, TimestepEmbedder
from mstar.model.bagel.components.tokenization import BagelTokenizer, add_special_tokens
from mstar.model.bagel.components.vit_encoder import BagelVisionModel
from mstar.model.bagel.config import load_bagel_config
from mstar.model.bagel.submodules import (
    CombineCFGSubmodule,
    LLMSubmodule,
    VAEDecoderSubmodule,
    VAEEncoderSubmodule,
    ViTEncoderSubmodule,
)
from mstar.model.base import DECODE, ForwardPassArgs, Model
from mstar.model.loader import iter_safetensors_file, load_hf_weights
from mstar.model.loader.base import LLAMA_STACKED_PARAMS, StackedParamRule
from mstar.model.submodule_base import NodeSubmodule
from mstar.utils.sampling import SamplingConfig

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
# Weight-loading containers
# ---------------------------------------------------------------------------
#
# The HF safetensors files for BAGEL group multiple top-level modules under
# distinct key prefixes (``language_model.*``, ``llm2vae.*``, ``vit_model.*``,
# etc.). ``load_hf_weights`` takes a single ``nn.Module`` whose
# ``named_parameters()`` define the lookup table, so we wrap the relevant
# top-level modules in small file-local containers whose own attribute names
# reproduce the checkpoint prefixes. The container is discarded after the
# load — only the wrapped modules retain the materialised parameters.


class _BagelLLMEMA(nn.Module):
    """``language_model`` + ``llm2vae`` slice of ``ema.safetensors``."""

    def __init__(self, language_model: nn.Module, llm2vae: nn.Module):
        super().__init__()
        self.language_model = language_model
        self.llm2vae = llm2vae


class _BagelGenAuxEMA(nn.Module):
    """``vae2llm`` + ``time_embedder`` + ``latent_pos_embed`` slice of
    ``ema.safetensors`` (the auxiliary projections / embeddings the
    image-gen path needs in addition to the LLM)."""

    def __init__(
        self,
        vae2llm: nn.Module,
        time_embedder: nn.Module,
        latent_pos_embed: nn.Module,
    ):
        super().__init__()
        self.vae2llm = vae2llm
        self.time_embedder = time_embedder
        self.latent_pos_embed = latent_pos_embed


class _BagelViTEMA(nn.Module):
    """``vit_model`` + ``connector`` + ``vit_pos_embed`` slice of
    ``ema.safetensors`` (the SigLIP2 ViT and its projection)."""

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


def _stacked_source_keys(
    expected: set[str], stacked_params: list[StackedParamRule] | None,
) -> set[str]:
    """Expand fused parameter names into the per-shard checkpoint keys they
    are loaded from.
    ``_load_into`` filters the safetensors read to ``expected`` (the
    module's own parameter names). When ``stacked_params`` fuse several
    checkpoint tensors into one parameter (``q_proj`` / ``k_proj`` /
    ``v_proj`` → ``qkv_proj``), the source keys differ from the fused name,
    so they must be added to the read filter or the iterator would skip
    them.
    """
    if not stacked_params:
        return expected
    keys = set(expected)
    for name in expected:
        for rule in stacked_params:
            if rule.target_suffix in name:
                keys.add(name.replace(rule.target_suffix, rule.source_suffix))
    return keys


def _load_into(
    module: nn.Module, source_path: Path, device: str,
    stacked_params: list[StackedParamRule] | None = None,
) -> None:
    """Materialise ``module`` on ``device`` and load every parameter from
    ``source_path``. Raises ``KeyError`` if any parameter is missing from
    the checkpoint — the equivalent of the old ``enforce_missing_keys=True``
    safety net.
    ``stacked_params`` routes per-shard checkpoint tensors into fused
    parameters (e.g. ``q/k/v_proj`` → ``qkv_proj``); pass it for modules
    that use ``FusedColumnLinear`` projections.
    """
    module.to_empty(device=device)
    expected = set(dict(module.named_parameters()).keys())
    read_keys = _stacked_source_keys(expected, stacked_params)
    loaded = load_hf_weights(
        module,
        iter_safetensors_file(source_path, device=device, keys=read_keys),
        stacked_params=stacked_params
    )
    missing = expected - loaded
    if missing:
        sample = sorted(missing)[:10]
        more = "…" if len(missing) > 10 else ""
        raise KeyError(
            f"Missing {len(missing)} keys when loading from {source_path}: "
            f"{sample}{more}"
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

    Nodes (4):
        vit_encoder   (enc_dec) - SigLIP2 ViT + connector + pos embed
        vae_encoder   (enc_dec) - VAE encode + patchify + projection
        LLM           (ar)      - Fat node: embed + Qwen2 + lm_head + CFG
        vae_decoder   (enc_dec) - VAE decode to pixels

    Graph walks (5):
        prefill_text  - Text token embedding + LLM prefill (causal)
        prefill_vit   - ViT encoding + LLM prefill (bidirectional)
        prefill_vae   - VAE encoding + LLM prefill (bidirectional)
        decode        - Autoregressive text generation
        image_gen     - Flow matching loop (3-pass CFG + Euler) + VAE decode

    Graph walk transitions are schedule-driven (no BOI token detection). The
    output mode is known upfront from the API request's output_modalities.
    Prefill steps are constructed as a sequential schedule that walks
    through interleaved text and image inputs.
    """

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        **kwargs
    ):
        self.cache_dir = cache_dir

        config_path = hf_hub_download(
            repo_id=model_path_hf, filename="config.json",
            revision=None, cache_dir=cache_dir,
        )
        with open(config_path) as f:
            self.config = load_bagel_config(json.load(f))

        self.model_path_hf = model_path_hf

        self.tokenizer = BagelTokenizer.from_pretrained(model_path_hf, cache_dir=cache_dir)
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
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}
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

        # Set by get_worker_graphs() when config has LLM_cfg_text/LLM_cfg_img
        self._has_cfg_parallel = False

    @property
    def _image_gen_walk(self) -> str:
        """Return the appropriate image_gen graph walk based on config."""
        return "image_gen_cfg" if self._has_cfg_parallel else "image_gen"


    def _download_hf(self):
        if self.repo is not None:
            return
        local_dir = snapshot_download(
            repo_id=self.model_path_hf, cache_dir=self.cache_dir,
        )
        self.repo = Path(local_dir)

    def _init_language_model_components(self, device):
        if self.llm_initialized:
            return
        self._download_hf()
        self.llm_initialized = True
        with torch.device("meta"):
            self.language_model = BagelForCausalLM(self.config)
            self.llm2vae = nn.Linear(self.config.hidden_size, self.config.patch_latent_dim)

        ema_path = self.repo / "ema.safetensors"

        _load_into(
            _BagelLLMEMA(self.language_model, self.llm2vae),
            ema_path, device,
            stacked_params=LLAMA_STACKED_PARAMS,
        )

        if not self.vae_initialized:
            # Need these for image gen
            with torch.device("meta"):
                self.latent_pos_embed = PositionEmbedding(
                    self.config.max_latent_size, self.config.hidden_size
                )
                self.time_embedder = TimestepEmbedder(self.config.hidden_size)
                self.vae2llm = nn.Linear(self.config.patch_latent_dim, self.config.hidden_size)
            _load_into(
                _BagelGenAuxEMA(
                    self.vae2llm, self.time_embedder, self.latent_pos_embed,
                ),
                ema_path, device,
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
        _load_into(self.vae_model, vae_path, device)

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
            _load_into(
                _BagelGenAuxEMA(
                    self.vae2llm, self.time_embedder, self.latent_pos_embed,
                ),
                ema_path, device,
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
        _load_into(
            _BagelViTEMA(self.vit_model, self.connector, self.vit_pos_embed),
            ema_path, device,
        )


    # -----------------------------------------------------------------------
    # Lazy submodule initialization
    # -----------------------------------------------------------------------

    def _create_submodule(self, node_name: str, device: str) -> NodeSubmodule | None:
        """Create a submodule wrapper on first access."""
        logger.debug("Creating submodule for BAGEL model node %s", node_name)
        if node_name in ("LLM", "LLM_cfg_text", "LLM_cfg_img"):
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
                eos_token_id=self.eos_token_id,
                node_name=node_name,
            )
        elif node_name == "combine_cfg":
            self._init_language_model_components(device)
            return CombineCFGSubmodule(
                llm2vae=self.llm2vae,
                config=self.config,
            )
        elif node_name == "vit_encoder":
            self._init_vit_components(device)
            return ViTEncoderSubmodule(
                vit_model=self.vit_model,
                connector=self.connector,
                vit_pos_embed=self.vit_pos_embed,
                vit_patch_size=self.config.vit_config.patch_size,
                vit_max_num_patch_per_side=self.config.vit_max_num_patch_per_side
            )
        elif node_name == "vae_encoder":
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
        elif node_name == "vae_decoder":
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

    # System prompt for image understanding.
    # TODO: make this configurable
    BAGEL_DEFAULT_SYSTEM_PROMPT = "You are BAGEL, a helpful assistant created by ByteDance."

    # Chat-template envelope for image understanding (image-to-text).
    VLM_UNDERSTANDING_PREFIX = (
        "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n"
    )
    VLM_UNDERSTANDING_SUFFIX = "\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    # For text-to-text, BAGEL doesn't seem to do as well with the "user... assistant..." format
    VLM_UNDERSTANDING_TEMPLATE = "<|im_start|>{system_prompt}<|im_end|>" + \
        "<|im_start|>{prompt}<|im_end|><|im_start|>"

    # Image generation (T2I/I2I): bare wrapping, no system prompt or roles. Input
    # image (I2I) is positioned before this by the prefill_vae/vit walks.
    GEN_TEMPLATE = "<|im_start|>{prompt}<|im_end|><|im_start|>"
    BOS_EOS_TEMPL = "<|im_start|>{prompt}<|im_end|>"

    def _encode_text(self, text: str) -> torch.Tensor:
        """Tokenize a text segment, falling back to raw bytes when no tokenizer
        is available (dummy-model test path)."""
        if self.tokenizer is not None:
            return torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        return torch.tensor(list(text.encode("utf-8")), dtype=torch.uint8)

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """Tokenize user prompt and system prompt (if think_mode).

        Returns model-specific keys matching get_forward_pass_inputs:
            "text_inputs"    - tokenized user prompt
            "system_prompt"  - tokenized system prompt (think_mode only)

        Bagel doesn't need the raw multimodal tensors for process_prompt;
        images are loaded and handled as ``image_inputs`` by the data worker.
        """
        result: NameToTensorList = {}

        if prompt is not None:
            target_output = output_modalities[0] if output_modalities else "text"
            is_understanding = target_output == "text"
            has_image = "image" in input_modalities
            think_mode = kwargs.get("think_mode", self.config.think_mode)
            system_prompt = self.BAGEL_DEFAULT_SYSTEM_PROMPT

            if is_understanding and has_image:
                if think_mode:
                    system_prompt = f"{system_prompt} {VLM_THINK_SYSTEM_PROMPT}"
                segments = [
                    self.VLM_UNDERSTANDING_PREFIX.format(system_prompt=system_prompt),
                    self.VLM_UNDERSTANDING_SUFFIX.format(prompt=prompt),
                ]
            elif is_understanding:
                if think_mode:
                    system_prompt = f"{system_prompt} {VLM_THINK_SYSTEM_PROMPT}"
                segments = [self.VLM_UNDERSTANDING_TEMPLATE.format(
                    system_prompt=system_prompt, prompt=prompt
                )]
            else:
                # Image generation (T2I/I2I): the input image (I2I) is placed
                # before this text by the prefill_vae/vit walks.
                segments = [self.GEN_TEMPLATE.format(prompt=prompt)]
                if think_mode:
                    segments.insert(0, self.BOS_EOS_TEMPL.format(prompt=GEN_THINK_SYSTEM_PROMPT))
            result["text_inputs"] = [self._encode_text(seg) for seg in segments]

        # Image edit path: both input and output include "image".
        # request specifies a target width and/or height, resize the input
        # images to be the largest that fits within that box (preserving
        # aspect ratio) so the edited output matches the requested size.
        is_image_edit = "image" in input_modalities and "image" in output_modalities
        max_width = kwargs.get("width")
        max_height = kwargs.get("height")
        if (
            is_image_edit
            and tensors is not None
            and tensors.get("image_inputs")
        ):
            result["image_inputs"] = [
                self._resize_to_fit(img, max_width, max_height)
                for img in tensors["image_inputs"]
            ]

        return result

    @staticmethod
    def _resize_to_fit(
        image: torch.Tensor,
        max_width: int | None,
        max_height: int | None,
    ) -> torch.Tensor:
        """Resize a c x h x w image to the largest size fitting within
        max_width x max_height while preserving aspect ratio.

        If only one of max_width / max_height is given, only that dimension
        constrains the result. Both dimensions are capped at 1024.
        """
        _, h, w = image.shape
        scales = [1024 / w, 1024 / h]
        if max_width is not None:
            scales.append(max_width / w)
        if max_height is not None:
            scales.append(max_height / h)
        scale = min(scales)
        new_h = max(1, round(h * scale))
        new_w = max(1, round(w * scale))
        if new_h == h and new_w == w:
            return image
        logger.info(
            "Resizing input image from %dx%d to %dx%d (h x w)",
            h, w, new_h, new_w,
        )
        resized = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0)

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

    def get_kv_cache_config(self) -> list[KVCacheConfig]:
        return [KVCacheConfig(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.hidden_size // self.config.num_attention_heads,
            max_seq_len=self.config.max_position_embeddings,
            num_qo_heads=self.config.num_attention_heads,
        )]

    def get_submodule(self, node_name: str, device: str = "cpu", tp_group=None) -> torch.nn.Module | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(node_name, device)
        logger.info(f"Successfully loaded in BAGEL submodule for {node_name}")
        self._submodule_cache[node_name] = submodule
        return submodule

    def get_node_engine_types(self) -> dict[str, EngineType]:
        return {
            "vit_encoder": EngineType.STATELESS,
            "vae_encoder": EngineType.STATELESS,
            "init_latents": EngineType.STATELESS,
            "LLM": EngineType.KV_CACHE,
            "LLM_cfg_text": EngineType.KV_CACHE,
            "LLM_cfg_img": EngineType.KV_CACHE,
            "combine_cfg": EngineType.STATELESS,
            "vae_decoder": EngineType.STATELESS,
        }

    def get_worker_graphs(self, config_path: str):
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        node_groups = config.get("node_groups", [])
        all_node_names = {
            name for g in node_groups for name in g["node_names"]
        }
        self._has_cfg_parallel = "LLM_cfg_text" in all_node_names
        return super().get_worker_graphs(config_path)

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        # -- prefill_text: just the LLM node (text embedding is internal) --
        # No output needed — conductor is notified when the worker graph completes.
        prefill_text = GraphNode(
            name="LLM",
            input_names=["text_inputs"],
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="new_token",
                    output_modality="text",
                    persist=True
                ),
            ],
        )

        # -- prefill_vit: ViT encoder -> LLM --
        prefill_vit = Sequential([
            GraphNode(
                name="vit_encoder",
                input_names=["image_inputs"],
                outputs=[
                    GraphEdge(next_node="LLM", name="img_emb"),
                ],
            ),
            GraphNode(
                name="LLM",
                input_names=["img_emb"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                        persist=True
                    ),
                ],
            ),
        ])

        # -- prefill_vae: VAE encoder -> LLM --
        prefill_vae = Sequential([
            GraphNode(
                name="vae_encoder",
                input_names=["image_inputs"],
                outputs=[
                    GraphEdge(next_node="LLM", name="img_emb"),
                ],
            ),
            GraphNode(
                name="LLM",
                input_names=["img_emb"],
                outputs=[],
            ),
        ])

        # -- decode: single LLM node (embed + transformer + lm_head) --
        decode = Loop(
            name="decode_loop",
            section=GraphNode(
                name="LLM",
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                    ),
                    GraphEdge(
                        next_node="LLM",
                        name="text_inputs",
                    ),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )

        # -- image_gen: denoising loop (LLM does CFG+Euler) -> VAE decode --
        # n_iters = num_timesteps - 1 because the loop body performs one
        # Euler step per iteration. With N timestep boundaries (e.g. 50),
        # there are N-1 intervals, so N-1 Euler steps are needed.
        image_gen = Sequential([
            Loop(
                section=GraphNode(
                    name="LLM",
                    input_names=["latents", "time_index"],
                    outputs=[
                        GraphEdge(next_node="LLM", name="latents"),
                        GraphEdge(next_node="LLM", name="time_index"),
                    ],
                ),
                max_iters=self.config.num_timesteps - 1,
                outputs=[
                    GraphEdge(next_node="vae_decoder", name="latents"),
                ],
            ),
            GraphNode(
                name="vae_decoder",
                input_names=["latents"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="image_output",
                        output_modality="image",
                    ),
                ],
            ),
        ])

        # -- image_gen_cfg: parallel 3-branch CFG denoising loop -> VAE decode --
        # Each CFG branch (main, cfg_text, cfg_img) runs on its own GPU.
        # combine_cfg applies the CFG formula + Euler step after each iteration.
        image_gen_cfg = Sequential([
            Loop(
                section=Sequential([
                    Parallel([
                        GraphNode(
                            name="LLM",
                            input_names=["latents", "time_index"],
                            outputs=[
                                GraphEdge(next_node="combine_cfg", name="v_main"),
                            ],
                            enable_async_scheduling=False
                        ),
                        GraphNode(
                            name="LLM_cfg_text",
                            input_names=["latents", "time_index"],
                            outputs=[
                                GraphEdge(next_node="combine_cfg", name="v_cfg_text"),
                            ],
                            enable_async_scheduling=False
                        ),
                        GraphNode(
                            name="LLM_cfg_img",
                            input_names=["latents", "time_index"],
                            outputs=[
                                GraphEdge(next_node="combine_cfg", name="v_cfg_img"),
                            ],
                            enable_async_scheduling=False
                        ),
                    ]),
                    GraphNode(
                        name="combine_cfg",
                        input_names=[
                            "v_main", "v_cfg_text", "v_cfg_img",
                            "latents", "time_index",
                        ],
                        outputs=[
                            GraphEdge(next_node="LLM", name="latents"),
                            GraphEdge(next_node="LLM", name="time_index"),
                            GraphEdge(next_node="LLM_cfg_text", name="latents"),
                            GraphEdge(next_node="LLM_cfg_text", name="time_index"),
                            GraphEdge(next_node="LLM_cfg_img", name="latents"),
                            GraphEdge(next_node="LLM_cfg_img", name="time_index"),
                            GraphEdge(next_node="combine_cfg", name="latents"),
                            GraphEdge(next_node="combine_cfg", name="time_index"),
                        ],
                        enable_async_scheduling=False
                    ),
                ]),
                max_iters=self.config.num_timesteps - 1,
                outputs=[
                    GraphEdge(next_node="vae_decoder", name="latents"),
                ],
            ),
            GraphNode(
                name="vae_decoder",
                input_names=["latents"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="image_output",
                        output_modality="image",
                    ),
                ],
            ),
        ])

        walks = {
            "prefill_text": prefill_text,
            "prefill_vit": prefill_vit,
            "prefill_vae": prefill_vae,
            DECODE: decode,
            "image_gen": image_gen,
        }
        if self._has_cfg_parallel:
            walks["image_gen_cfg"] = image_gen_cfg
        return walks

    def _build_prefill_schedule(
        self, input_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        is_understanding: bool,
    ):
        # Build prefill schedule: sequential list of (graph_walk_name, input tensor info)
        schedule: list[tuple[str, TensorPointerInfo]] = []
        texts = input_signals.get("text_inputs", [])
        images = input_signals.get("image_inputs", [])

        # 1. System prompt

        if len(texts) == 2:
            # the first text block is the system prompt and should come at the very beginning
            # (in both thinking and non-thinking mode)
            input_modalities.insert(0, "text")

        # 2. Walk through interleaved inputs, building sequential steps
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

    def _requires_cfg(
        self, **kwargs,
    ):
        return (
            kwargs["target_output"] == "image" \
                and (
                    kwargs["cfg_img_scale"] > 1.0 \
                    or kwargs["cfg_text_scale"] > 1.0
                )
        )

    def _get_step_metadata(
        self, full_metadata: CurrentForwardConductorMetadata,
    ) -> dict:
        return {
            "is_prefill": full_metadata.is_prefill,
            "cfg_text_scale": full_metadata.kwargs["cfg_text_scale"],
            "cfg_img_scale": full_metadata.kwargs["cfg_img_scale"],
            "cfg_interval": full_metadata.kwargs["cfg_interval"],
            "cfg_renorm_type": full_metadata.kwargs["cfg_renorm_type"],
            "cfg_renorm_min": full_metadata.kwargs["cfg_renorm_min"],
            "width": full_metadata.kwargs["width"],
            "height": full_metadata.kwargs["height"],
            "image_preprocess": full_metadata.kwargs["image_preprocess"],
        }

    def _get_fwd_pass_inputs(
        self,
        metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
    ) -> list[GraphEdge]:
        """Construct the external inputs for the current forward pass.

        The conductor calls this to determine what tensors to send to
        workers at the start of each forward pass. For prefill graph walks,
        the schedule entry determines which input to route; for decode
        and image_gen, the previous output feeds back in.

        persist_signals key conventions:
            "text_inputs"    - list of per-turn text TensorPointerInfos
            "image_inputs"   - list of per-image TensorPointerInfos
            "system_prompt"  - tokenized system prompt (if think_mode)
            "new_token"      - last generated token (during decode)
            "latents"        - noise latents (for image_gen entry)
        """
        graph_walk = metadata.graph_walk

        if metadata.is_prefill:
            schedule = metadata.kwargs["prefill_schedule"]
            step = metadata.kwargs["prefill_step"]
            input_tensor_info = [schedule[step][1]]

            if graph_walk == "prefill_text":
                graph_edge = GraphEdge(next_node="LLM", name="text_inputs")
            elif graph_walk == "prefill_vit":
                graph_edge = GraphEdge(next_node="vit_encoder", name="image_inputs")
            elif graph_walk == "prefill_vae":
                graph_edge = GraphEdge(next_node="vae_encoder", name="image_inputs")
            else:
                raise ValueError(f"Unrecognized prefill graph_walk {graph_walk}")
            graph_edge.tensor_info = input_tensor_info
            return [graph_edge]

        elif graph_walk == DECODE:
            graph_edge = GraphEdge(
                next_node="LLM", name="text_inputs",
                tensor_info=persist_signals.get("new_token", [])
            )
            return [graph_edge]

        elif graph_walk == "image_gen":
            # Initial noise latents feed the LLM denoising loop.
            # Note: latents are typically initialized by the submodule's
            # preprocess() (random noise), not passed through persist_signals.
            # This lookup handles the case where latents are externally provided.
            graph_edge = GraphEdge(next_node="LLM", name="latents")
            graph_edge.tensor_info = persist_signals.get("latents", [])
            return [
                graph_edge,
                GraphEdge(next_node="LLM", name="time_index")
            ]

        elif graph_walk == "image_gen_cfg":
            return [
                GraphEdge(next_node="LLM", name="latents"),
                GraphEdge(next_node="LLM", name="time_index"),
                GraphEdge(next_node="LLM_cfg_text", name="latents"),
                GraphEdge(next_node="LLM_cfg_text", name="time_index"),
                GraphEdge(next_node="LLM_cfg_img", name="latents"),
                GraphEdge(next_node="LLM_cfg_img", name="time_index"),
                GraphEdge(next_node="combine_cfg", name="latents"),
                GraphEdge(next_node="combine_cfg", name="time_index"),
            ]

        return []

    def _get_unpersist_tensors(
        self, graph_walk: str, inputs: list[GraphEdge]
    ) -> list[TensorPointerInfo]:
        """
        Lists the tensors that will be used for the last time in this forward pass
        """
        if graph_walk == "prefill_vae":
            # If we have prefill_vae, we know that the image input will be
            # passed into the ViT encoder for the next forward pass, so it
            # has to stick around
            return []
        # otherwise, we can un-persist all tensors
        return sum(
            [inp.tensor_info for inp in inputs], start=[]
        )

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
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
        params["width"] = 1024
        params["height"] = 1024
        params["image_preprocess"] = "default"
        overridable_keys += ["width", "height", "image_preprocess"]

        if "image" in input_modalities and "image" in output_modalities:
            params["cfg_img_scale"] = 2.0
            params["cfg_interval"] = [0.0, 1.0]
            params["cfg_renorm_type"] = "text_channel"

        if model_kwargs:
            for key in overridable_keys:
                if key in model_kwargs:
                    params[key] = model_kwargs[key]

        if "image" in output_modalities and input_signals.get("image_inputs"):
            image = input_signals["image_inputs"][0]
            params["height"] = image.dims[1]
            params["width"] = image.dims[2]
            logger.info(f"Will generate an image of shape {params['height']} x {params['width']}")

        think_mode = params.pop("think_mode") # used for schedule logic, not stored in params
        schedule = self._build_prefill_schedule(
            input_modalities=input_modalities,
            input_signals=input_signals,
            is_understanding=(target_output == "text"),
        )

        first_graph_walk = schedule[0][0] if schedule else DECODE
        kwargs = {
            "prefill_schedule": schedule,
            "prefill_step": 0,
            "target_output": target_output,
            "num_timesteps": self.config.num_timesteps,
            "think_mode": think_mode,
            **params,  # CFG params  + gen width / height
        }
        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=first_graph_walk,
            is_prefill=bool(schedule),
            requires_cfg=self._requires_cfg(**kwargs),
            kwargs=kwargs,
        )
        step_metadata =  self._get_step_metadata(full_metadata)

        sample_prefill_token = (think_mode or target_output != "image") and len(schedule) == 1
        step_metadata["sample_prefill_token"] = sample_prefill_token

        inputs = self._get_fwd_pass_inputs(
            full_metadata, input_signals
        )
        unpersist_tensors = self._get_unpersist_tensors(first_graph_walk, inputs)
        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata=step_metadata
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections=None,
    ) -> ForwardPassArgs:
        """Advance graph walk transitions. Schedule-driven, no BOI detection.

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
        metadata = partition_metadata
        request_done = False

        has_text_output = (
            metadata.kwargs.get("think_mode", False) or metadata.kwargs["target_output"] != "image"
        )
        sample_prefill_token = False

        if metadata.is_prefill:
            step = metadata.kwargs["prefill_step"] + 1
            schedule = metadata.kwargs["prefill_schedule"]

            sample_prefill_token = has_text_output and (
                step == len(schedule) - 1
            )

            if step < len(schedule):
                # More prefill steps remaining
                metadata.kwargs["prefill_step"] = step
                metadata.graph_walk = schedule[step][0]
            else:
                # All prefill done -- transition based on target_output
                metadata.is_prefill = False
                target = metadata.kwargs["target_output"]
                if target == "text":
                    metadata.graph_walk = DECODE
                elif target == "image":
                    if metadata.kwargs.get("think_mode", False):
                        # Think first: decode to generate reasoning, then
                        # EOS triggers transition to image_gen.
                        metadata.graph_walk = DECODE
                    else:
                        metadata.graph_walk = self._image_gen_walk
        elif metadata.graph_walk == DECODE:
            target = metadata.kwargs["target_output"]
            if metadata.kwargs.get("think_mode", False) and target == "image":
                # Thinking graph walk complete — transition to image generation.
                metadata.graph_walk = self._image_gen_walk
            else:
                request_done = True

        elif metadata.graph_walk in ("image_gen", "image_gen_cfg"):
            # Image generation complete (one image per request), OR
            # text generation complete (happens in a dynamic loop)
            request_done = True

        step_metadata =  self._get_step_metadata(metadata)
        step_metadata["sample_prefill_token"] = sample_prefill_token
        inputs = self._get_fwd_pass_inputs(
            metadata, persist_signals
        )
        unpersist_tensors = self._get_unpersist_tensors(
            metadata.graph_walk, inputs
        )
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata=step_metadata,
            request_done=request_done
        )

    def get_sampling_config(
        self, node_name: str,
        model_kwargs: dict | None = None,
    )  -> SamplingConfig | None:
        keys = [
            "temperature", "top_k", "top_p", "repetition_penalty",
            "ignore_eos"
        ]
        params = {k: getattr(self.config, k) for k in keys}
        if model_kwargs:
            for key in keys:
                if key in model_kwargs:
                    params[key] = model_kwargs[key]
        return SamplingConfig(
            vocab_size=self.config.vocab_size,
            **params
        )
