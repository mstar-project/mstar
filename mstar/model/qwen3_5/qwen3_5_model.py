import logging
from typing import Any

import torch
from jinja2 import TemplateError
from transformers import AutoTokenizer

from mstar.communication.tensors import NameToTensorList
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec
from mstar.engine.resources.attn.ragged.config import (
    RaggedAttentionConfig,
    RaggedAttentionSpec,
)
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.linear_attn.config import (
    LinearAttnConfig,
    LinearAttnSpec,
    LinearAttnVariant,
)
from mstar.engine.resources.position.config import PositionConfig, PositionSpec
from mstar.engine.resources.recurrent.config import DeltaNetGeometry, RecurrentStateConfig, RecurrentStateSpec
from mstar.engine.resources.sampler.config import SamplerSpec
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.model.base import ProcessPromptOutput, TensorAndMetadata
from mstar.model.multimodal import (
    MEDIA_LAYOUT,
    VIDEO,
    PromptPart,
    as_uint8_chw,
    messages_from_parts,
    vision_encoder,
)
from mstar.model.multimodal_decoder import (
    MIXED_SPLIT,
    MediaIntake,
    MultimodalDecoderModel,
    PreparedMedia,
)
from mstar.model.qwen3_5.components.language_model import Qwen3_5ForCausalLM
from mstar.model.qwen3_5.components.rope import vision_position_advance
from mstar.model.qwen3_5.config import (
    ATTN,
    GDN_STATE,
    KV_CACHE,
    LINEAR_ATTN,
    ROPE,
    SAMPLER,
    VISION_ATTN,
    Qwen3_5Config,
    Qwen3_5VisionConfig,
)
from mstar.model.qwen3_5.video import (
    ProcessedVideo,
    expand_video_pads,
    group_timestamps,
    sample_video,
)
from mstar.model.submodule_base import NodeSubmodule
from mstar.utils.fast_tokenize import encode_ids

logger = logging.getLogger(__name__)

# TODO: implement MoE variants
class Qwen3_5DenseModel(MultimodalDecoderModel):
    NAME = "Qwen3.5"
    METADATA_FILES = (
        "config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "tokenizer.json",
        "merges.txt",
    )

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(model_path_hf, cache_dir)
        self.config = Qwen3_5Config.from_hf(self.local_dir)
        # None on a text-only checkpoint (`_create_submodule` then refuses a
        # vision_encoder node)
        self.vision_config = Qwen3_5VisionConfig.from_hf_or_none(self.local_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.local_dir,
            cache_dir=cache_dir,
        )
        # config.json names <|endoftext|> but a chat turn ends on <|im_end|>,
        # which only the tokenizer knows. See `Qwen3_5Config.stop_token_ids`.
        if self.tokenizer.eos_token_id is not None:
            self.config.extra_stop_token_ids = (self.tokenizer.eos_token_id,)
        # the vision tower, whose output fills images' and videos' pads in
        # prompt order
        pad = self._pad_ids()
        tower = vision_encoder(tuple(pad.values())) if pad else None
        self.media = tuple(
            MediaIntake(m, tower, p, self._image_intake) if m == "image"
            # a video fills one pad per temporal group
            else MediaIntake(m, tower, p, self._video_intake, runs=self._video_runs)
            for m, p in pad.items()
        )
        self._image_processor = None
        self._video_processor = None

    # -----------------------------------------------------------------------
    # Model ABC: resources
    # -----------------------------------------------------------------------

    def _gdn_state_dtype(self) -> torch.dtype:
        """The checkpoint's ``mamba_ssm_dtype`` (fp32 for the released ones);
        fp32 wherever the pool's decode kernels cannot take it.

        ``gdn_state.state_dtype: bfloat16`` in the yaml overrides this. bf16
        halves state traffic, but only FlashInfer's fused bf16 decode kernel
        takes it, and that is K=V=128 only.
        """
        dtype = getattr(torch, self.config.mamba_ssm_dtype, None)
        head_dims_ok = (
            self.config.linear_key_head_dim == 128
            and self.config.linear_value_head_dim == 128
        )
        if dtype is torch.bfloat16 and head_dims_ok:
            return torch.bfloat16
        return torch.float32

    def get_node_resources(self) -> list[NodeResourceSpec]:
        num_kv_layers = len(self.config.full_layer_indices)
        num_gdn_layers = len(self.config.linear_layer_indices)
        return [
            KVSpec(
                resource_key=KV_CACHE, nodes={"LLM"},
                config=PagedKVConfig(
                    num_layers=num_kv_layers,
                    num_kv_heads=self.config.num_key_value_heads,
                    head_dim=self.config.head_dim,
                    max_seq_len=self.config.max_position_embeddings,
                    num_qo_heads=self.config.num_attention_heads
                )
            ),
            AttentionSpec(
                resource_key=ATTN, nodes={"LLM"},
                config=AttentionConfig(
                    kv_cache=KV_CACHE,
                    splits=(MIXED_SPLIT,),
                ),
            ),
            PositionSpec(
                resource_key=ROPE, nodes={"LLM"},
                config=PositionConfig(kv_cache=KV_CACHE),
            ),
            RecurrentStateSpec(
                resource_key=GDN_STATE, nodes={"LLM"},
                config=RecurrentStateConfig(
                    num_layers=num_gdn_layers,
                    blocks=DeltaNetGeometry(
                        num_k_heads=self.config.linear_num_key_heads,
                        num_v_heads=self.config.linear_num_value_heads,
                        head_k_dim=self.config.linear_key_head_dim,
                        head_v_dim=self.config.linear_value_head_dim,
                        conv_kernel_size=self.config.linear_conv_kernel_dim
                    ).to_blocks(state_dtype=self._gdn_state_dtype()),
                )
            ),
            LinearAttnSpec(
                resource_key=LINEAR_ATTN, nodes={"LLM"},
                config=LinearAttnConfig(
                    recurrent_state=GDN_STATE,
                    variant=LinearAttnVariant.GDN,
                    splits=(MIXED_SPLIT,),
                ),
            ),
            SamplerSpec(
                resource_key=SAMPLER, nodes={"LLM"},
                vocab_size=self.config.vocab_size,
                enable_repetion_penalty=True,
            ),
            *self._vision_resources(),
        ]

    def _vision_resources(self) -> list[NodeResourceSpec]:
        """The ViT tower's attention, its only resource; none when text-only."""
        if self.vision_config is None:
            return []
        from mstar.model.qwen3_5.submodules import VisionEncoderSubmodule

        vision = self.vision_config
        processor = self.image_processor
        min_patches = processor.size["shortest_edge"] // processor.patch_size ** 2
        return [
            RaggedAttentionSpec(
                resource_key=VISION_ATTN, nodes={"vision_encoder"},
                config=RaggedAttentionConfig(
                    num_qo_heads=vision.num_heads,
                    num_kv_heads=vision.num_heads,
                    head_dim=vision.head_dim,
                    # A segment per image, not per request: the resize floors
                    # each image at `shortest_edge` pixels. A video frame group
                    # has no floor; a prompt with more segments runs the block
                    # loop eagerly (`VisionEncoderSubmodule._fits_capture`).
                    max_segments_per_request=(
                        VisionEncoderSubmodule.BLOCK_LOOP_PATCH_BUCKETS[-1] // min_patches
                    ),
                ),
            ),
        ]

    # ------------------------------------------------------------------
    # Model ABC: I/O
    # ------------------------------------------------------------------
    # The chat template writes each image as `<|vision_start|><|image_pad|>
    # <|vision_end|>`; a video's triple is first written out per temporal
    # group (`expand_video_pads`). Each pad then expands to its tokens.
    _PAD_TOKENS: dict[str, str] = {"image": "<|image_pad|>", VIDEO: "<|video_pad|>"}

    def _pad_ids(self) -> dict[str, int]:
        """Pad ids, read off the tokenizer.

        Not off `config.json`: Qwen3-Omni's disagrees with its tokenizer, and
        wrong ids silently find no pads.
        """
        ids: dict[str, int] = {}
        for modality, token in self._PAD_TOKENS.items():
            i = self.tokenizer.convert_tokens_to_ids(token)
            if i is not None and i != self.tokenizer.unk_token_id:
                ids[modality] = i
        return ids

    @property
    def image_processor(self):
        """Lazily loaded: a text-only deployment never needs it."""
        if self._image_processor is None:
            from transformers import AutoImageProcessor

            self._image_processor = AutoImageProcessor.from_pretrained(
                self.local_dir, cache_dir=self.cache_dir,
            )
        return self._image_processor

    @property
    def video_processor(self):
        """The checkpoint's ``video_preprocessor_config.json``, lazily."""
        if self._video_processor is None:
            from transformers import AutoVideoProcessor

            self._video_processor = AutoVideoProcessor.from_pretrained(
                self.local_dir, cache_dir=self.cache_dir,
            )
        return self._video_processor

    def load_video(self, filepath: str, device: str) -> TensorAndMetadata:
        """Only the frames the video processor samples, on the host whatever
        ``device`` is (see `sample_video`)."""
        video = sample_video(filepath, self.video_processor)
        return TensorAndMetadata(data=video.frames, metadata=video.metadata)

    def _video_runs(self, metadata: dict) -> int:
        """A video's pad runs: one per temporal group."""
        return len(self._video_timestamps(metadata))

    def _video_timestamps(self, metadata: dict) -> list[float]:
        """Seconds per temporal group, from the sampling metadata alone."""
        return group_timestamps(
            metadata["frames_indices"], metadata.get("fps"),
            self.video_processor.temporal_patch_size,
        )

    def _preprocess_video(
        self, frames: torch.Tensor, metadata: dict, timestamps: list[float],
    ) -> ProcessedVideo:
        """Patches and grid for one sampled video.

        The frames are already sampled, so the processor must not sample
        again; everything else (resize budget, patchify) is its own.
        """
        from transformers.video_utils import VideoMetadata

        out = self.video_processor(
            videos=[frames.cpu()],
            video_metadata=[VideoMetadata(**metadata)],
            do_sample_frames=False,
            return_tensors="pt",
        )
        (grid,) = out["video_grid_thw"]
        if int(grid[0]) != len(timestamps):
            raise ValueError(
                f"video grid has {int(grid[0])} temporal groups but "
                f"{len(timestamps)} timestamps"
            )
        return ProcessedVideo(out["pixel_values_videos"], grid, timestamps)

    def _prompt_ids(
        self, parts: list[PromptPart], raw: dict[str, list[torch.Tensor]], **kwargs,
    ) -> torch.Tensor:
        """The chat template's prompt, each video's pad written out as one
        ``<t seconds>``-stamped pad per temporal group (`expand_video_pads`)."""
        raw_videos = raw.get(VIDEO, [])
        video_metadata = (kwargs.get("input_metadata") or {}).get("video_inputs", [])
        if len(video_metadata) != len(raw_videos):
            raise ValueError(
                f"Qwen3.5 got {len(raw_videos)} videos but sampling metadata "
                f"for {len(video_metadata)}; videos must come through `load_video`"
            )
        # From the metadata alone, so a bad layout 400s before the processors
        # spend seconds resizing a long video
        timestamps = [self._video_timestamps(metadata) for metadata in video_metadata]

        # The released checkpoints are chat models (base ones say `-Base`), so
        # always use the chat template. `enable_thinking=False` emits an empty,
        # pre-closed `<think>` block, which is how Qwen turns reasoning off.
        try:
            text = self.tokenizer.apply_chat_template(
                messages_from_parts(parts),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=kwargs.get("enable_thinking", True),
            )
        except TemplateError as exc:
            # a chat the template refuses (a system message after the first turn,
            # no user turn) is the client's to fix: a 400, not a 500
            raise ValueError(f"Qwen3.5's chat template refused the chat: {exc}") from exc
        if timestamps:
            text = expand_video_pads(
                text, self._PAD_TOKENS[VIDEO], "<|vision_start|>", "<|vision_end|>", timestamps,
            )
        # a long prompt is encoded as parallel chunks, identical ids
        return torch.tensor(encode_ids(self.tokenizer, text), dtype=torch.long)

    def _merge(self) -> int:
        if self.vision_config is None:
            raise ValueError("this Qwen3.5 checkpoint has no vision tower; send text only")
        return self.vision_config.spatial_merge_size

    def _image_intake(self, image: torch.Tensor, metadata: dict) -> PreparedMedia:
        merge = self._merge()
        # uint8, HWC: the processor rescales by default, so floats would be
        # rescaled twice and the model would see a near-black image
        out = self.image_processor(
            images=[as_uint8_chw(image).permute(1, 2, 0).numpy()], return_tensors="pt",
        )
        grid = out["image_grid_thw"][0]
        return PreparedMedia({"pixel_values": out["pixel_values"]}, [int(grid.prod()) // merge ** 2], [grid])

    def _video_intake(self, frames: torch.Tensor, metadata: dict) -> PreparedMedia:
        """Each temporal group is placed like a one-frame image."""
        merge = self._merge()
        video = self._preprocess_video(frames, metadata, self._video_timestamps(metadata))
        t, h, w = (int(n) for n in video.grid)
        group = torch.tensor([1, h, w])
        return PreparedMedia(
            {"pixel_values": video.pixel_values}, [int(group.prod()) // merge ** 2] * t, [group] * t,
        )

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        prompt_parts: list[PromptPart] | None = None,
        **kwargs,
    ) -> ProcessPromptOutput:
        """The shared intake, with MRoPE's advance in the KV layout.

        ``text_inputs`` is the whole prompt, one pad per merged vision token.
        ``pixel_values`` holds the images and videos in prompt order, as the
        tower's output fills the pads; the layout of the pads (offset, tokens
        and grid of each image and video frame group) goes back as
        ``MEDIA_LAYOUT`` metadata, for the tower's grid maths and the LLM's
        MRoPE positions.
        """
        prompt_in = self.intake(prompt, input_modalities, tensors, prompt_parts, **kwargs)
        pads, grids = prompt_in.pads, prompt_in.grids
        result: NameToTensorList = {"text_inputs": [pads.ids], **prompt_in.inputs}
        if (tensors or {}).get("video_inputs"):
            # the decoded frames are spent; don't hold them for the request's life
            result["video_inputs"] = []
        # an image spans max(h', w') MRoPE positions, not its t * h' * w' tokens
        advance = int(pads.ids.numel()) + sum(
            vision_position_advance(grids[p][i], self.vision_config.spatial_merge_size) - tokens
            for p, i, tokens in zip(pads.pad, pads.index, pads.tokens, strict=True)
        )
        layout = self.prompt_layout(prompt_in)
        return ProcessPromptOutput(
            result, {MEDIA_LAYOUT: layout, "kv_layout": self.kv_layout(layout, int(pads.ids.numel()), advance)},
        )

    # ------------------------------------------------------------------
    # Model ABC: submodule loading
    # ------------------------------------------------------------------

    def _create_submodule(
        self, node_name: str, device: str,
        autocast_dtype: torch.dtype | None = None,
        tp_group=None,
    ) -> NodeSubmodule | None:
        from mstar.model.qwen3_5.submodules import (
            LLMSubmodule,
            VisionEncoderSubmodule,
        )

        if node_name not in ("LLM", "vision_encoder"):
            return None

        dtype = autocast_dtype or torch.bfloat16
        weights_dir = self._resolve_weights()

        if node_name == "vision_encoder":
            from mstar.model.qwen3_5.components.vision import Qwen3_5VisionModel
            from mstar.model.qwen3_5.weight_loader import (
                load_qwen3_5_vision_weights,
            )

            if self.vision_config is None:
                raise ValueError(
                    f"{self.model_path_hf} is a text-only checkpoint, but the "
                    "deployment asked for a vision_encoder node. Drop it from "
                    "`node_groups`."
                )
            tower = self._build(
                lambda: Qwen3_5VisionModel(self.vision_config), dtype, device,
            )
            load_qwen3_5_vision_weights(tower, weights_dir, device=device)
            tower.requires_grad_(False).eval()
            logger.info("Loaded Qwen3.5 vision tower onto %s", device)
            return VisionEncoderSubmodule(
                tower, self.vision_config, self.encoder(node_name),
            )

        from mstar.model.qwen3_5.weight_loader import load_qwen3_5_weights

        model = self._build(
            lambda: Qwen3_5ForCausalLM(self.config, tp_group), dtype, device,
        )
        load_qwen3_5_weights(model, weights_dir, device=device)
        model.requires_grad_(False).eval()
        if tp_group is not None and tp_group.world_size > 1:
            # collective over the TP group; every rank builds the LLM here
            tp_group.init_allreduce_fusion(self.config.hidden_size, dtype)
        logger.info("Loaded Qwen3.5 LLM submodule onto %s", device)
        return LLMSubmodule(model, self.config, self.vision_config, self.encoders)
