"""
BagelModel: Model implementation for BAGEL (ByteDance) unified multimodal model.

BAGEL uses a Qwen2 LLM with MoT (Mixture-of-Transformers) architecture,
SigLIP2 ViT for image understanding, and FLUX VAE for image generation.
The LLM itself serves as the denoiser for rectified flow image generation
(no separate diffusion model).

Architecture (4 nodes, 7 with CFG-parallel placement):
    vit_encoder   (enc_dec) - SigLIP2 ViT + connector + pos embed
    vae_encoder   (enc_dec) - VAE encode + patchify + projection
    LLM           (ar)      - Fat node: embed + Qwen2 + lm_head + CFG + Euler
    vae_decoder   (enc_dec) - VAE decode to pixels
    LLM_cfg_text, LLM_cfg_img, combine_cfg - only when the deployment places
                  the guidance branches on their own workers (the CFG walks)

Graph walks (9, or 11 with CFG-parallel placement):
    prefill_text   - Text token embedding + LLM prefill (causal)
    prefill_vit    - ViT encoding + LLM prefill (bidirectional for images)
    prefill_vae    - VAE encoding + LLM prefill (bidirectional for images)
    decode         - Autoregressive text generation
    image_gen      - Flow matching loop (Euler, main branch only) + VAE decode
    A guided request's phases (``image_gen_phases``):
    image_gen_head - The steps before the CFG window, on LLM alone
    image_gen_guided2, image_gen_guided3 - The CFG window on LLM alone, every
                     branch in one combined forward
    image_gen_cfg2, image_gen_cfg3 - CFG-parallel only: the CFG window, one
                     node per guidance branch, the velocities combined in combine_cfg
    image_decode   - The VAE decode alone, when the window reaches the last step
    (the steps after the window run as image_gen, from the window's end)

The LLM node absorbs text_emb, lm_head, and flow_proj because they are
always colocated on the same GPU. Keeping them as separate graph nodes
would add unnecessary IPC overhead. A denoise step runs the guidance
branches it needs (one to three LLM passes: a branch whose scale or
interval leaves it out sits the step out) and combines their velocities.

Output mode is known upfront from the API request's output_modalities
field (no BOI token detection). Prefill is sequential: text tokens are
processed causally, then each image is processed bidirectionally.
"""

import json
import logging
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import DEFAULT_PARTITION, CurrentForwardConductorMetadata
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    KVReqConfig,
    KVSpec,
    NodeResourceSpec,
    PagedKVConfig,
    PositionConfig,
    PositionSpec,
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    ResourceReqConfig,
    SamplerSpec,
    SamplingReqConfig,
)
from mstar.engine.resources.sampler.utils import SamplingConfig
from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    GraphSection,
    Loop,
    Parallel,
    Sequential,
    TensorPointerInfo,
)
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.bagel.components.autoencoder import BagelAutoEncoder
from mstar.model.bagel.components.language_model import BagelForCausalLM
from mstar.model.bagel.components.modeling_utils import BagelMLPconnector, PositionEmbedding, TimestepEmbedder
from mstar.model.bagel.components.tokenization import BagelTokenizer, add_special_tokens
from mstar.model.bagel.components.vit_encoder import VIT_ATTN, BagelVisionModel
from mstar.model.bagel.config import MIXED_TEXT, load_bagel_config
from mstar.model.bagel.submodules import (
    CFG_WALKS,
    DENOISE_WALKS,
    IMAGE_ADVANCE,
    IMAGE_DECODE,
    IMAGE_GEN,
    IMAGE_SENTINELS,
    LLM_GRAPH_WALKS,
    NODE_TO_CFG_LABEL,
    CFGWindow,
    CombineCFGSubmodule,
    ImageGenPhase,
    LLMSubmodule,
    VAEDecoderSubmodule,
    VAEEncoderSubmodule,
    ViTEncoderSubmodule,
    cfg_window,
    denoise_loop,
    guided,
    image_gen_phases,
    vae_latents,
    vit_patches,
    vit_preprocess,
)
from mstar.model.base import (
    DECODE,
    ForwardPassArgs,
    KVLabel,
    LabelLayout,
    Model,
    PrefixKeys,
    ProcessPromptOutput,
    Span,
    prompt_spans,
)
from mstar.model.components.diffusion.image_io import encode_image, image_output_format
from mstar.model.loader import iter_safetensors_file, load_hf_weights
from mstar.model.loader.base import LLAMA_STACKED_PARAMS, StackedParamRule
from mstar.model.multimodal import (
    TEXT,
    PromptPart,
    chat_turns,
    check_attachments,
    check_plan,
    find_media_spans,
    parts_from_modalities,
    prefill_plan,
    split_around_spans,
)
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import TextOutput

logger = logging.getLogger(__name__)

# The request's `CFGWindow`, which process_prompt resolves with its guidance
# kwargs: the conductor's loop caps and walk schedule read it.
CFG_WINDOW = "cfg_window"

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
    autocast_dtype: torch.dtype | None = None,
) -> None:
    """Materialise ``module`` on ``device`` and load every parameter from
    ``source_path``. Raises ``KeyError`` if any parameter is missing from
    the checkpoint — the equivalent of the old ``enforce_missing_keys=True``
    safety net.
    ``stacked_params`` routes per-shard checkpoint tensors into fused
    parameters (e.g. ``q/k/v_proj`` → ``qkv_proj``); pass it for modules
    that use ``FusedColumnLinear`` projections.
    ``autocast_dtype`` (when not None) casts ``module`` on the meta device
    before ``to_empty`` so storage is allocated directly in that dtype
    instead of fp32-then-downcast (halves the load-time VRAM peak).
    """
    if autocast_dtype is not None:
        module = module.to(autocast_dtype)
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

    Nodes and walks: see the module docstring (the CFG-parallel walks and
    their three nodes exist only when a deployment places the guidance
    branches apart).

    Graph walk transitions are schedule-driven (no BOI token detection). The
    output mode is known upfront from the API request's output_modalities.
    Prefill steps are constructed as a sequential schedule that walks
    through interleaved text and image inputs.
    """


    INTERNAL_KWARGS = (CFG_WINDOW,)

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        accelerator_graph: bool | None = None,
        **kwargs
    ):
        if accelerator_graph is not None and not isinstance(accelerator_graph, bool):
            raise ValueError("accelerator_graph must be a boolean")
        self.cache_dir = cache_dir

        config_path = hf_hub_download(
            repo_id=model_path_hf, filename="config.json",
            revision=None, cache_dir=cache_dir,
        )
        with open(config_path) as f:
            self.config = load_bagel_config(json.load(f))
        if accelerator_graph is not None:
            self.config.accelerator_graph = accelerator_graph

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
        # Set when graph-walk-specific node groups place LLM cache consumers
        # in separate worker instances.
        self._has_llm_disaggregation = False
        # The separate image-generation placement needs serial denoising
        # until its published KV handoff and speculative scheduling agree.
        self._image_gen_remote_handoff = False

    def _image_gen_phases(self, window) -> tuple[ImageGenPhase, ...]:
        """The request's denoise walks on this deployment (`image_gen_phases`)."""
        return image_gen_phases(CFGWindow(*window), self.config.num_timesteps, self._has_cfg_parallel)

    def _download_hf(self):
        if self.repo is not None:
            return
        local_dir = snapshot_download(
            repo_id=self.model_path_hf, cache_dir=self.cache_dir,
        )
        self.repo = Path(local_dir)

    def _init_language_model_components(
        self, device, autocast_dtype=None, tp_group=None,
    ):
        if self.llm_initialized:
            return
        self._download_hf()
        self.llm_initialized = True
        with torch.device("meta"):
            self.language_model = BagelForCausalLM(
                self.config, comm_group=tp_group,
            )
            self.llm2vae = nn.Linear(self.config.hidden_size, self.config.patch_latent_dim)

        ema_path = self.repo / "ema.safetensors"

        _load_into(
            _BagelLLMEMA(self.language_model, self.llm2vae),
            ema_path, device,
            stacked_params=LLAMA_STACKED_PARAMS,
            autocast_dtype=autocast_dtype,
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
                autocast_dtype=autocast_dtype,
            )

    def _init_vae_components(self, device, autocast_dtype=None):
        self._download_hf()
        if self.vae_initialized:
            return
        self.vae_initialized = True
        ae_params = self.config.vae_config
        with torch.device("meta"):
            self.vae_model = BagelAutoEncoder(ae_params)

        # Load in weights: VAE
        vae_path = self.repo / "ae.safetensors"
        _load_into(self.vae_model, vae_path, device, autocast_dtype=autocast_dtype)

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
                autocast_dtype=autocast_dtype,
            )


    def _init_vit_components(self, device, autocast_dtype=None):
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
            autocast_dtype=autocast_dtype,
        )


    # -----------------------------------------------------------------------
    # Lazy submodule initialization
    # -----------------------------------------------------------------------

    def _create_submodule(
        self, node_name: str, device: str,
        autocast_dtype: torch.dtype | None = None,
        tp_group=None,
    ) -> NodeSubmodule | None:
        """Create a submodule wrapper on first access."""
        logger.debug("Creating submodule for BAGEL model node %s", node_name)
        if node_name in ("LLM", "LLM_cfg_text", "LLM_cfg_img"):
            self._init_language_model_components(
                device, autocast_dtype=autocast_dtype, tp_group=tp_group,
            )
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
            return CombineCFGSubmodule(config=self.config)
        elif node_name == "vit_encoder":
            self._init_vit_components(device, autocast_dtype=autocast_dtype)
            return ViTEncoderSubmodule(
                vit_model=self.vit_model,
                connector=self.connector,
                vit_pos_embed=self.vit_pos_embed,
                vit_patch_size=self.config.vit_config.patch_size,
                vit_max_num_patch_per_side=self.config.vit_max_num_patch_per_side,
                accelerator_graph=self.config.accelerator_graph,
            )
        elif node_name == "vae_encoder":
            self._init_vae_components(device, autocast_dtype=autocast_dtype)
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
            self._init_vae_components(device, autocast_dtype=autocast_dtype)
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

    # Image generation (T2I/I2I): bare wrapping, no system prompt or roles. Input
    # image (I2I) is positioned before this by the prefill_vae/vit walks.
    GEN_TEMPLATE = "<|im_start|>{prompt}<|im_end|><|im_start|>"
    BOS_EOS_TEMPL = "<|im_start|>{prompt}<|im_end|>"

    # The prefill_vit walk re-emits the two sentinels around its own
    # embeddings, so only the pad interior is dropped from the text spans.
    IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

    # One understanding turn, its attachments inside it: the system prompt's
    # comes first, and the reply opens after the last.
    VLM_TURN = "<|im_start|>{role}\n{content}<|im_end|>\n"
    VLM_REPLY = "<|im_start|>assistant\n"

    def _placeholder_specs(self) -> dict[str, tuple[int, int, int]]:
        """``(start, pad, end)`` sentinel ids, read off the tokenizer.

        The tokenizer produced the prompt, so it decides these ids; the config
        could name ones the prompt does not contain.
        """
        pad = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if pad is None or pad == self.tokenizer.unk_token_id:
            raise ValueError(
                "BAGEL's tokenizer has no <|image_pad|>, so an image cannot be "
                "placed in the prompt"
            )
        return {"image": (self.boi_token_id, pad, self.eoi_token_id)}

    def _render_prompt(
        self, parts: list[PromptPart], *, is_understanding: bool, system_prompt: str
    ) -> str:
        """Write the request out in order, attachments as placeholders.

        Understanding writes each turn in its own role block. A text run
        following an attachment carries the newline the old suffix supplied,
        and a turn ending on one closes after a newline too, so every image
        layout a single user turn could express renders to the string that
        template gave. Generation leaves attachments outside the turn — the
        prefill_vae/vit walks put the input image ahead of the prompt — and
        wraps what follows.
        """
        if is_understanding:
            blocks = [self.VLM_TURN.format(role="system", content=system_prompt)]
            for turn in chat_turns(parts):
                body, after_attachment = self._render_body(turn.parts, sep="\n")
                if after_attachment:
                    body.append("\n")
                blocks.append(self.VLM_TURN.format(role=turn.role, content="".join(body)))
            return "".join(blocks) + self.VLM_REPLY

        # Generation: attachments written before any text stay outside the turn.
        body, _ = self._render_body(parts, sep="")
        lead = 0
        while lead < len(body) and body[lead] == self.IMAGE_PLACEHOLDER:
            lead += 1
        return "".join(body[:lead]) + self.GEN_TEMPLATE.format(
            prompt="".join(body[lead:])
        )

    def _render_body(self, parts: list[PromptPart], sep: str) -> tuple[list[str], bool]:
        """Write ``parts`` out in order, and say whether the last is an attachment.

        ``sep`` opens a text run that follows an attachment: understanding's old
        suffix supplied that newline, generation's did not. Text that follows
        another role's text starts on a new line.
        """
        body: list[str] = []
        after_attachment = False
        role = None
        for part in parts:
            if part.modality == TEXT:
                if part.text:
                    if after_attachment:
                        body.append(sep + part.text)
                    elif body and part.role != role:
                        body.append("\n" + part.text)
                    else:
                        body.append(part.text)
                    after_attachment, role = False, part.role
                continue
            body.append(self.IMAGE_PLACEHOLDER)
            after_attachment = True
        return body, after_attachment

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
        prompt_parts: list[PromptPart] | None = None,
        **kwargs,
    ) -> NameToTensorList | ProcessPromptOutput:
        """Tokenize the prompt into one text span per prefill step.

        Returns model-specific keys matching get_forward_pass_inputs:
            "text_inputs"    - one tokenized span per text step of the plan
            "system_prompt"  - tokenized system prompt (think_mode only)

        The prompt is rendered once with a placeholder per image, tokenized
        once, and sliced around those placeholders, so text written between two
        attachments gets its own span and no seam is re-tokenized. The schedule
        builder walks the same :func:`prefill_plan`.

        Given the tensors, it also returns a ``kv_layout`` for each label the request may open.
        """
        params = self._request_params(input_modalities, output_modalities, kwargs)
        if "image" in output_modalities:
            image_output_format(kwargs)  # fail a bad output knob now, not after the denoise
            for key in ("width", "height"):
                value = params[key]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"BAGEL image {key} must be an integer, got {value!r}")
        result: NameToTensorList = {}

        if prompt is not None:
            target_output = output_modalities[0] if output_modalities else "text"
            is_understanding = target_output == "text"
            has_image = "image" in input_modalities
            think_mode = params["think_mode"]
            system_prompt = self.BAGEL_DEFAULT_SYSTEM_PROMPT

            texts, roles = prompt, None
            if prompt_parts is not None:
                texts = [p.text or "" for p in prompt_parts if p.modality == TEXT]
                roles = [p.role for p in prompt_parts]
            # a legacy layout can lack a text slot, and its prompt still needs one
            parts = parts_from_modalities(input_modalities or [TEXT], texts, roles)
            unsupported = {p.modality for p in parts} - {TEXT, "image"}
            if unsupported:
                raise ValueError(
                    f"BAGEL has no encoder for {', '.join(sorted(unsupported))}; "
                    "it accepts image attachments only"
                )
            if tensors is not None:
                check_attachments(
                    parts, {"image": len(tensors.get("image_inputs", []))},
                )

            body = parts
            if is_understanding and len(parts) > 1 and parts[0].role == "system":
                # a leading system message takes the default's place; a lone
                # one stays the turn it was, not an empty one after the default
                system_prompt, body = parts[0].text, parts[1:]
            if think_mode and is_understanding:
                system_prompt = f"{system_prompt} {VLM_THINK_SYSTEM_PROMPT}"

            text = self._render_prompt(
                body,
                is_understanding=is_understanding,
                system_prompt=system_prompt,
            )
            if is_understanding and not has_image:
                # No attachment: one span, one walk, nothing to scan.
                segments = [self._encode_text(text)]
            else:
                input_ids = self._encode_text(text)
                spans = find_media_spans(input_ids, self._placeholder_specs())
                segments = split_around_spans(input_ids, spans)
                check_plan(
                    prefill_plan(parts, leading_text=is_understanding),
                    spans,
                    len(segments),
                )
                if think_mode and not is_understanding:
                    # Not part of the request layout: tokenized on its own and
                    # prefilled ahead of the plan.
                    segments.insert(
                        0, self._encode_text(
                            self.BOS_EOS_TEMPL.format(prompt=GEN_THINK_SYSTEM_PROMPT)
                        )
                    )
            result["text_inputs"] = segments

        # Image edit path: both input and output include "image". Resize the
        # input images to the largest that fits within the requested width x
        # height (preserving aspect ratio) so the edited output matches it.
        is_image_edit = "image" in input_modalities and "image" in output_modalities
        if (
            is_image_edit
            and tensors is not None
            and tensors.get("image_inputs")
        ):
            result["image_inputs"] = [
                self._resize_to_fit(img, params["width"], params["height"])
                for img in tensors["image_inputs"]
            ]

        images = result.get("image_inputs") or (tensors or {}).get("image_inputs", [])
        if "image" in output_modalities:
            # an edit's output takes its (resized) input's size, a generation the requested one;
            # under 16 pixels a side (the VAE's 8x times the 2x2 patch) it has no latent patch,
            # and the loop would fail mid-request
            sides = images[0].shape[-2:] if images else (params["height"], params["width"])
            if any(side < 16 for side in sides):
                raise ValueError(f"BAGEL's output image must be at least 16 pixels a side, got {list(sides)} (h, w)")
        if tensors is None:
            return result
        if prompt is None:
            parts, segments = parts_from_modalities(input_modalities), []
        metadata = {}
        window = None
        if "image" in output_modalities:
            window = cfg_window(
                self.config.num_timesteps, self.config.timestep_shift, tuple(params["cfg_interval"]),
                params["cfg_text_scale"], params["cfg_img_scale"],
            )
            metadata[CFG_WINDOW] = tuple(window)
        metadata["kv_layout"] = {"kv": self._kv_layout(parts, segments, images, output_modalities, params, window)}
        return ProcessPromptOutput(result, metadata)

    def _kv_layout(
        self, parts: list[PromptPart], segments: list[torch.Tensor], images: list[torch.Tensor],
        output_modalities: list[str], params: dict, window: CFGWindow | None,
    ) -> dict[str, LabelLayout]:
        """Each label's writes, in the order `_build_prefill_schedule` runs them.

        ``main`` holds every span. A generation keys only its first span, if that is
        text: its images are resized before their walks see them, and a file's digest
        doesn't cover the resize. Under guidance, ``cfg_img`` holds the text alone.
        ``cfg_text`` forks before a text walk and after an image walk, so it holds all
        but the last span when the last span is text, and every span when the prompt
        ends on an image. A branch label denoises (its scratch) only where the
        request's ``window`` runs that branch.
        """
        cfg = self.config
        target_output = (output_modalities or ["text"])[0]
        understanding = target_output == "text"
        think_mode, image_preprocess = params["think_mode"], vit_preprocess(params["image_preprocess"])
        spans = []
        if think_mode and not understanding and segments:
            spans.append(Span("ids", "prefill_text", segments[0].numel(), segments[0].numel(), "text_inputs"))

        def image(step: PromptPart) -> list[Span]:
            _, height, width = images[step.index].shape
            item = (("image", step.index), (image_preprocess,))
            patches = vit_patches(
                height, width, image_preprocess, cfg.vit_config.patch_size, cfg.vit_max_num_patch_per_side,
            )
            vit = Span("digest", "prefill_vit", patches + IMAGE_SENTINELS, IMAGE_ADVANCE, *item)
            if understanding:
                return [vit]
            latents = vae_latents(height, width, image_preprocess, cfg.latent_downsample, cfg.max_latent_size)
            return [Span("digest", "prefill_vae", latents + IMAGE_SENTINELS, IMAGE_ADVANCE, *item), vit]

        texts = segments[len(spans):]
        # skip a text step the prompt has no tensor for, as `_build_prefill_schedule` does
        plan = prefill_plan(parts, leading_text=understanding)
        plan = [step for step in plan if step.modality != TEXT or step.index < len(texts)]
        spans += prompt_spans(plan, texts, "prefill_text", image)
        keyed = int(bool(spans) and spans[0].kind == "ids")
        if understanding:
            main = LabelLayout(
                spans=tuple(spans), growth=(("decode_loop", 1),),
                keyed_spans=None if "image" not in output_modalities else keyed,
            )
            return {"main": main}
        height, width = images[0].shape[-2:] if images else (params["height"], params["width"])
        scratch = (height // cfg.latent_downsample) * (width // cfg.latent_downsample) + IMAGE_SENTINELS
        growth = (("decode_loop", 1),) if think_mode else ()
        main = LabelLayout(spans=tuple(spans), growth=growth, scratch=scratch, keyed_spans=keyed)
        if not guided(window):
            return {"main": main, "cfg_text": LabelLayout(), "cfg_img": LabelLayout()}
        return {
            "main": main,
            "cfg_text": LabelLayout(
                spans=tuple(spans[:-1] if spans and spans[-1].walk == "prefill_text" else spans), scratch=scratch,
            ),
            "cfg_img": LabelLayout(
                spans=tuple(span for span in spans if span.walk == "prefill_text"), growth=growth,
                scratch=scratch if window.branches == 3 else 0,
            ),
        }

    @staticmethod
    def _resize_to_fit(
        image: torch.Tensor,
        max_width: int,
        max_height: int,
    ) -> torch.Tensor:
        """Resize a c x h x w image to the largest size fitting within
        max_width x max_height while preserving aspect ratio. Both dimensions
        are capped at 1024.
        """
        _, h, w = image.shape
        scale = min(1024 / w, 1024 / h, max_width / w, max_height / h)
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

    def text_output(self) -> TextOutput:
        return TextOutput(self._chunk_detokenizer(), (self.eos_token_id,))

    def postprocess(
        self, output: torch.Tensor,
        modality: str, # text | image | video | audio
        request_kwargs: dict | None = None,
    ) -> bytes:
        if modality == "text":
            return super().postprocess(output, modality, request_kwargs)
        if modality == "image":
            # truncated to uint8, through the shared writer: PIL's default PNG encode
            # (adaptive filters, zlib 6) took ~0.4 s of every 1024^2 request
            return encode_image((output[0] * 255).to(torch.uint8), request_kwargs)
        raise ValueError(f"Unsupported modality: {modality!r}")

    # CFG-parallel deployments add the two guidance branches. Ordinary
    # deployments only instantiate LLM, so do not make its TP resource depend
    # on absent, ungrouped nodes.
    _LLM_NODES = frozenset({"LLM", "LLM_cfg_text", "LLM_cfg_img"})

    def _kv_config(self) -> PagedKVConfig:
        return PagedKVConfig(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.hidden_size // self.config.num_attention_heads,
            max_seq_len=self.config.max_position_embeddings,
            num_qo_heads=self.config.num_attention_heads,
        )

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        """The text walk's prompt is its token ids, and so is every step it takes.

        An image in an understanding prompt is keyed by its file. An edit's VAE
        walk writes ``main`` too, but the keys of ``main`` leave that walk out. A
        chat sends a reply back as its own assistant turn, so its pages match.
        """
        decode = ((DECODE, "decode_loop"),)
        return {"kv": {
            "main": KVLabel(
                growth=decode, keys=PrefixKeys("text_inputs", ("prefill_text", "prefill_vit"), growth=True),
            ),
            # forks from main fill cfg_text, so it declares no growth
            "cfg_text": KVLabel(),
            "cfg_img": KVLabel(growth=decode),
        }}

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        # The stop check allows max_tokens - 1 iterations, and a loop runs at least one.
        caps = {"decode_loop": max(1, self.get_max_output_tokens(**model_kwargs) - 1)}
        if (window := model_kwargs.get(CFG_WINDOW)) is not None:
            # each denoise walk runs exactly its phase's steps
            caps.update(
                (denoise_loop(phase.walk), phase.steps) for phase in self._image_gen_phases(window) if phase.steps
            )
        return caps

    def checkpoint_path(self) -> str:
        return snapshot_download(
            repo_id=self.model_path_hf, cache_dir=self.cache_dir,
        )

    def get_node_resources(self) -> list[NodeResourceSpec]:
        """The KV cache, the attention over it, positions, and the sampler,
        plus the ViT's own cacheless attention.

        The labels here are the names the layers bind against
        (``Attention(attn_key=..., kv_key=..., pos_key=...)``) and the names
        ``declare_step`` keys its resource steps by, so they are one
        declaration read from three places.
        """
        kv_config = self._kv_config()
        nodes = set(self._LLM_NODES if self._has_cfg_parallel else {"LLM"})
        vit = self.config.vit_config
        return [
            # The ViT tower attends within one packed forward and caches
            # nothing, so this stands alone: no KV resource behind it, and only
            # the captured block loop plans it (eager runs flash-attn — see
            # `BagelViTAttention.attend`).
            RaggedAttentionSpec(
                resource_key=VIT_ATTN, nodes={"vit_encoder"},
                config=RaggedAttentionConfig(
                    num_qo_heads=vit.num_attention_heads,
                    num_kv_heads=vit.num_attention_heads,
                    head_dim=vit.hidden_size // vit.num_attention_heads,
                    # one image, hence one attending segment, per request
                    max_segments_per_request=1,
                ),
            ),
            KVSpec(resource_key="kv", nodes=nodes, config=kv_config, leader="LLM"),
            AttentionSpec(
                resource_key="attn", nodes=nodes,
                config=AttentionConfig(kv_cache="kv"),
            ),
            PositionSpec(
                resource_key="rope", nodes=nodes,
                config=PositionConfig(
                    kv_cache="kv", rope_theta=self.config.rope_theta,
                ),
            ),
            SamplerSpec(
                resource_key="sampler", nodes=nodes,
                vocab_size=self.config.vocab_size,
            ),
        ]

    def get_request_resource_configs(
        self, partition_fwd_args: dict[str, ForwardPassArgs],
        model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        """Per-resource knobs a new request is opened with.

        Two things are per-request rather than per-deployment: which cache
        labels the request will use (guidance doubles them, and it is a
        property of the request's arguments), and its sampling parameters.
        The conductor resolves this once and the engine hands each config to
        its resource at ingest.
        """
        from mstar.model.bagel.submodules import active_labels

        model_kwargs = model_kwargs or {}
        request = partition_fwd_args[DEFAULT_PARTITION].full_metadata.kwargs
        cfg = request["requires_cfg"]
        # only a generation's CFG walks run the CFG nodes
        nodes = self._LLM_NODES if self._has_cfg_parallel and request["target_output"] == "image" else {"LLM"}
        sampling = self.get_sampling_config("LLM", model_kwargs)
        publish_labels: dict[tuple[str, str], list[str]] = {}
        final_publish_labels: dict[tuple[str, str], list[str]] = {}
        if cfg:
            branch_labels = ["cfg_text", "cfg_img"]
            if self._has_cfg_parallel and not self._has_llm_disaggregation:
                # The branch labels' only readers are the CFG walk's branch nodes, one
                # label each. (A remote LLM, on an LLM-disaggregated deployment, may also
                # run decode, which reads cfg_img, and its single-node denoise walks still
                # name both labels per request, not per walk.)
                cfg_walk = next(p.walk for p in request["image_gen_phases"] if p.walk in CFG_WALKS)
                branch_labels = list(NODE_TO_CFG_LABEL.values())[1:CFG_WALKS[cfg_walk]]
            for (node, walk), labels in {
                ("LLM", "prefill_text"): ["cfg_text", "cfg_img"],
                ("LLM", "prefill_vit"): ["cfg_text"],
                ("LLM", "prefill_vae"): ["cfg_text"],
            }.items():
                if labels := [label for label in labels if label in branch_labels]:
                    publish_labels[(node, walk)] = labels
            if "cfg_img" in branch_labels:
                final_publish_labels[("LLM", "decode")] = ["cfg_img"]

        if self._has_llm_disaggregation:
            # Any prefill walk can be the last one before a remote decode or
            # image-generation worker. Decode itself may hand off to a remote
            # image-generation worker, but only its final state is needed.
            for walk in ("prefill_text", "prefill_vit", "prefill_vae"):
                publish_labels.setdefault(("LLM", walk), []).insert(0, "main")
            final_publish_labels.setdefault(
                ("LLM", "decode"), []
            ).insert(0, "main")

        return {
            # The KV resource reads this in admit_retrieve, to know which of a
            # published request's streams to pull in. Guidance-off requests
            # name only "main", so a transfer never drags branches that this
            # request will not attend.
            "kv": KVReqConfig(
                needed_labels_per_node_walk={
                    (node, walk): active_labels(walk, cfg, node)
                    for node in nodes
                    for walk in LLM_GRAPH_WALKS[self._has_cfg_parallel]
                },
                # Only the main LLM instance creates cache branches consumed
                # by the remote CFG replicas, and it exports only the ones the
                # request's CFG walk reads. Text-only requests export
                # nothing. Think-then-image exports cfg_img once after decode
                # stops, so the CFG walks see the final KV without rewriting
                # the SHM payload on every token. Disaggregated LLM placements
                # additionally export main at prefill boundaries and once when
                # decode stops.
                publish_labels_per_node_walk=publish_labels,
                final_publish_labels_per_node_walk=final_publish_labels,
            ),
            "sampler": SamplingReqConfig(
                temperature=sampling.temperature,
                top_k=sampling.top_k,
                top_p=sampling.top_p,
                ignore_eos=sampling.ignore_eos,
                repetition_penalty=sampling.repetition_penalty,
            ),
        }

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> torch.nn.Module | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(
            node_name, device, autocast_dtype=autocast_dtype, tp_group=tp_group,
        )
        logger.info(f"Successfully loaded in BAGEL submodule for {node_name}")
        self._submodule_cache[node_name] = submodule
        return submodule

    def get_default_sharding_config(self):
        from mstar.distributed.base import ShardingConfig

        return ShardingConfig(
            groups=[],
            tp_enabled_nodes={"LLM", "LLM_cfg_text", "LLM_cfg_img"},
            shard_dim={},
        )

    def get_worker_graphs(self, config_path: str):
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        node_groups = config.get("node_groups", [])
        all_node_names = {
            name for g in node_groups for name in g["node_names"]
        }
        self._has_cfg_parallel = "LLM_cfg_text" in all_node_names
        self._has_llm_disaggregation = sum(
            "LLM" in group["node_names"] for group in node_groups
        ) > 1
        def llm_ranks(walk: str) -> set[int]:
            return {
                rank
                for group in node_groups
                if "LLM" in group["node_names"]
                and (not group.get("graph_walks") or walk in group["graph_walks"])
                for rank in group["ranks"]
            }

        image_ranks = llm_ranks(IMAGE_GEN)
        self._image_gen_remote_handoff = bool(
            image_ranks
            and image_ranks.isdisjoint(llm_ranks("prefill_text"))
            and image_ranks.isdisjoint(llm_ranks("decode"))
        )
        return super().get_worker_graphs(config_path)

    def get_combined_graph_walks(self) -> dict[str, dict[str, set[str]]]:
        return {"LLM": {MIXED_TEXT: {"prefill_text", DECODE}}}

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
            max_iters=self.config.max_position_embeddings,
            outputs=[],
        )

        walks = {
            "prefill_text": prefill_text,
            "prefill_vit": prefill_vit,
            "prefill_vae": prefill_vae,
            DECODE: decode,
            IMAGE_DECODE: self._vae_decode(),
        }
        walks.update({
            walk: self._denoise_walk(walk) for walk in LLM_GRAPH_WALKS[self._has_cfg_parallel] if walk in DENOISE_WALKS
        })
        return walks

    @staticmethod
    def _denoise_nodes(walk: str) -> tuple[str, ...]:
        """The nodes of a denoise walk that read latents and time_index."""
        branches, split, _ = DENOISE_WALKS[walk]
        if not split:
            return ("LLM",)
        return (*list(NODE_TO_CFG_LABEL)[:branches], "combine_cfg")

    def _denoise_walk(self, walk: str) -> GraphSection:
        """A denoise walk, from its `DenoiseWalk` (branches, split, decode).

        One Euler step per iteration, so a request's N timestep boundaries take
        N - 1 iterations in all; the request's loop cap (`get_loop_caps`) sets
        the walk's share. Unsplit is the LLM node alone, running the walk's
        branches as one forward; split is a node per branch whose velocities
        combine_cfg combines into the step. Without the decode the loop's last
        latents and time_index persist for the request's next walk.
        """
        branches, split, decode = DENOISE_WALKS[walk]
        loop_back = [GraphEdge(next_node=node, name=name) for node in self._denoise_nodes(walk)
                     for name in ("latents", "time_index")]
        if not split:
            section = GraphNode(
                name="LLM",
                input_names=["latents", "time_index"],
                outputs=loop_back,
                # A separate image-generation worker combines a remote
                # KV handoff with this loop. The Rust speculative path
                # produced nondeterministic seeded output on that layout;
                # serial steps preserve the handoff's output.
                enable_async_scheduling=not self._image_gen_remote_handoff,
            )
        else:
            nodes = list(NODE_TO_CFG_LABEL.items())[:branches]
            section = Sequential([
                Parallel([
                    GraphNode(
                        name=node,
                        input_names=["latents", "time_index"],
                        outputs=[GraphEdge(next_node="combine_cfg", name=f"v_{label}")],
                        enable_async_scheduling=False,
                    )
                    for node, label in nodes
                ]),
                GraphNode(
                    name="combine_cfg",
                    input_names=[*(f"v_{label}" for _, label in nodes), "latents", "time_index"],
                    outputs=loop_back,
                    enable_async_scheduling=False,
                ),
            ])
        if decode:
            outputs = [GraphEdge(next_node="vae_decoder", name="latents")]
        else:
            outputs = [
                GraphEdge(next_node=EMPTY_DESTINATION, name=name, persist=True)
                for name in ("latents", "time_index")
            ]
        loop = Loop(
            name=denoise_loop(walk),
            section=section,
            max_iters=self.config.num_timesteps - 1,
            outputs=outputs,
        )
        return Sequential([loop, self._vae_decode()]) if decode else loop

    @staticmethod
    def _vae_decode() -> GraphNode:
        return GraphNode(
            name="vae_decoder",
            input_names=["latents"],
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="image_output",
                    output_modality="image",
                ),
            ],
        )

    def _build_prefill_schedule(
        self, input_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        is_understanding: bool,
        think_mode: bool = False,
    ):
        """Sequential list of ``(graph_walk_name, input tensor info)``.

        Walks the prefill plan, so attachments prefill where they were written
        and N of them get N walks. Understanding opens with the system span
        ahead of them; generation opens straight into them, after the
        think-mode span when there is one.
        """
        texts = list(input_signals.get("text_inputs", []))
        images = input_signals.get("image_inputs", [])

        schedule: list[tuple[str, TensorPointerInfo]] = []
        if think_mode and not is_understanding and texts:
            schedule.append(("prefill_text", texts.pop(0)))

        plan = prefill_plan(
            parts_from_modalities(input_modalities), leading_text=is_understanding,
        )
        for step in plan:
            pool = texts if step.modality == "text" else images
            if step.index >= len(pool):
                # ``check_plan`` already failed a mismatch at intake, where a
                # 400 can still be returned. This runs in the conductor, whose
                # loop only logs — raising here orphans the request and hangs
                # the client instead. Same choice as Qwen3-Omni's builder.
                logger.warning(
                    "BAGEL prefill plan wants a %s span at index %d but the "
                    "prompt produced %d; skipping it",
                    step.modality, step.index, len(pool),
                )
                continue
            if step.modality == "text":
                schedule.append(("prefill_text", texts[step.index]))
            else:
                if not is_understanding:
                    # Generation/editing: VAE encode the image
                    schedule.append(("prefill_vae", images[step.index]))
                schedule.append(("prefill_vit", images[step.index]))
        return schedule

    def _request_params(
        self, input_modalities: list[str], output_modalities: list[str], model_kwargs: dict | None,
    ) -> dict:
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
                if model_kwargs.get(key) is not None:  # an explicit null is the default
                    params[key] = model_kwargs[key]
        return params

    def _get_step_metadata(
        self, full_metadata: CurrentForwardConductorMetadata,
    ) -> dict:
        return {
            "is_prefill": full_metadata.is_prefill,
            "requires_cfg": full_metadata.kwargs["requires_cfg"],
            "cfg_text_scale": full_metadata.kwargs["cfg_text_scale"],
            "cfg_img_scale": full_metadata.kwargs["cfg_img_scale"],
            "cfg_interval": full_metadata.kwargs["cfg_interval"],
            "cfg_renorm_type": full_metadata.kwargs["cfg_renorm_type"],
            "cfg_renorm_min": full_metadata.kwargs["cfg_renorm_min"],
            "width": full_metadata.kwargs["width"],
            "height": full_metadata.kwargs["height"],
            "image_preprocess": full_metadata.kwargs["image_preprocess"],
        }

    @staticmethod
    def _image_gen_phase(metadata: CurrentForwardConductorMetadata) -> ImageGenPhase:
        return metadata.kwargs["image_gen_phases"][metadata.kwargs["image_gen_phase"]]

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
            "latents", "time_index" - where the previous denoise walk left off
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

        elif graph_walk in DENOISE_WALKS:
            # The previous denoise walk's persisted latents and time_index; the
            # request's first has none, and its nodes seed them.
            return [
                GraphEdge(next_node=node, name=name, tensor_info=list(persist_signals.get(name, [])))
                for node in self._denoise_nodes(graph_walk) for name in ("latents", "time_index")
            ]

        elif graph_walk == IMAGE_DECODE:
            # time_index, persisted beside them, is left to the request's teardown
            return [GraphEdge(next_node="vae_decoder", name="latents",
                              tensor_info=list(persist_signals.get("latents", [])))]

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
        params = self._request_params(input_modalities, output_modalities, model_kwargs)

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
            think_mode=think_mode,
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
        kwargs["requires_cfg"] = False
        if target_output == "image":
            # process_prompt resolved the window with the request's guidance kwargs
            window = CFGWindow(*(model_kwargs or {})[CFG_WINDOW])
            kwargs["requires_cfg"] = guided(window)
            kwargs["image_gen_phases"] = self._image_gen_phases(window)
            kwargs["image_gen_phase"] = 0
        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=first_graph_walk,
            is_prefill=bool(schedule),
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
          - the first denoise walk (image output, no think_mode)

        During decode:
          - Text output: EOS marks request complete.
          - Image output + think_mode: EOS transitions to the first denoise
            walk (thinking is done, now generate the image).

        The denoise walks run in the order of ``image_gen_phases``; after the
        last, marks request complete (one image per request).
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
                        metadata.graph_walk = self._image_gen_phase(metadata).walk
        elif metadata.graph_walk == DECODE:
            target = metadata.kwargs["target_output"]
            if metadata.kwargs.get("think_mode", False) and target == "image":
                # Thinking graph walk complete — transition to image generation.
                metadata.graph_walk = self._image_gen_phase(metadata).walk
            else:
                request_done = True

        elif metadata.graph_walk in DENOISE_WALKS or metadata.graph_walk == IMAGE_DECODE:
            phase = metadata.kwargs["image_gen_phase"] + 1
            if phase < len(metadata.kwargs["image_gen_phases"]):
                metadata.kwargs["image_gen_phase"] = phase
                metadata.graph_walk = self._image_gen_phase(metadata).walk
            else:
                # the image is out (one per request)
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
                if model_kwargs.get(key) is not None:  # an explicit null is the default
                    params[key] = model_kwargs[key]
        return SamplingConfig(
            vocab_size=self.config.vocab_size,
            **params
        )
