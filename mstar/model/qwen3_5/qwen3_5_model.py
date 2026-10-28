import logging
from pathlib import Path
from typing import Any

import torch
from jinja2 import TemplateError
from transformers import AutoTokenizer

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mstar.distributed.base import ShardingConfig
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
from mstar.engine.resources.sampler.config import SamplerSpec, SamplingReqConfig
from mstar.engine.resources.spec import CombinedSplit, NodeResourceSpec, ResourceReqConfig
from mstar.graph.base import GraphSection, TensorPointerInfo
from mstar.model.base import (
    ForwardPassArgs,
    KVLabel,
    LabelLayout,
    Model,
    ProcessPromptOutput,
    Span,
    TensorAndMetadata,
)
from mstar.model.multimodal import (
    MEDIA_LAYOUT,
    TEXT,
    VIDEO,
    MediaEncoder,
    MediaPads,
    PrefillStep,
    PromptPart,
    check_attachments,
    expand_media_pads,
    initial_prefill_args,
    media_layout,
    media_prefill_schedule,
    media_walk_name,
    media_walks,
    messages_from_parts,
    next_prefill_args,
    parts_from_modalities,
    text_combined_walks,
    vision_encoder,
)
from mstar.model.qwen3_5.components.language_model import Qwen3_5ForCausalLM
from mstar.model.qwen3_5.components.rope import vision_position_advance
from mstar.model.qwen3_5.config import (
    ATTN,
    GDN_STATE,
    KV_CACHE,
    LINEAR_ATTN,
    LLM_MIXED,
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
from mstar.model.utils import TextChunkDetokenizer, drop_stop_tokens
from mstar.utils.fast_tokenize import encode_ids

logger = logging.getLogger(__name__)

# a mixed step's prompt rows (as many as `max_rows_in_combined` lets in) take
# the prefill kernels, its decode rows the decode ones
_MIXED_SPLIT = CombinedSplit(LLM_MIXED, prefill_walks=("prefill_text",))


def _resolve_model_metadata(repo_id: str, cache_dir: str | None) -> str:
    """Resolve only the files needed to construct config and tokenize input.

    The API and conductor processes never download the checkpoint; workers
    fetch the full snapshot lazily from ``get_submodule``.
    """
    local_path = Path(repo_id)
    if local_path.is_dir():
        return str(local_path)

    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json",
            "tokenizer.json",
            "merges.txt",
        ],
    )


def _as_hwc_uint8(image: torch.Tensor):
    """What the HF image processor expects, from what the data worker sends.

    Images arrive ``(C, H, W)`` float32 in [0, 1] on the GPU. The processor
    defaults to ``do_rescale=True``, so floats would be rescaled twice and the
    model would silently see a near-black image.
    """
    if image.dtype.is_floating_point:
        image = (image * 255.0).clamp(0, 255).to(torch.uint8)
    if image.dim() == 3 and image.shape[0] == 4:
        # the processor refuses a 4-channel (RGBA) tensor, so drop alpha
        image = image[:3]
    if image.dim() == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    return image.cpu().contiguous().numpy()


# TODO: implement MoE variants
class Qwen3_5DenseModel(Model):
    # Image preprocessing is a few small ops per request, and a thread per
    # core only slows them: each op waits on whichever thread a co-tenant
    # descheduled. At 64 threads c=1 TTFT p95 was 152 ms (some preprocesses
    # 35-130 ms); at 4, preprocessing held at 7-10 ms and p95 at 49 ms. One
    # thread is steadier but slower (13 ms).
    PREPROCESS_TORCH_THREADS = 4

    _text_detokenizer: TextChunkDetokenizer | None = None

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        **kwargs: Any,
    ):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir

        self.local_dir = _resolve_model_metadata(model_path_hf, cache_dir)
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
        pad = tuple(self._pad_ids().values())
        self.encoders: tuple[MediaEncoder, ...] = (vision_encoder(pad),) if pad else ()

        # so weights are materialized at most once per node
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}
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
                    splits=(_MIXED_SPLIT,),
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
                    splits=(_MIXED_SPLIT,),
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

    # -----------------------------------------------------------------------
    # Model ABC: walk graph declaration
    # -----------------------------------------------------------------------
    def get_combined_graph_walks(self) -> dict[str, dict[str, set[str]]]:
        return text_combined_walks()

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        # a safety bound on decode, not the per-request budget
        return media_walks(self.config.max_position_embeddings, self.encoders)

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

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        prompt_parts: list[PromptPart] | None = None,
        **kwargs,
    ) -> ProcessPromptOutput:
        """Render the request once, tokenize once, expand each attachment's pad.

        ``text_inputs`` is the whole prompt, one pad per merged vision token.
        ``pixel_values`` and ``image_grid_thw`` hold the images and videos in
        prompt order, as the tower's output fills the pads; the layout of the
        pads (offset, tokens and grid of each image and video frame group)
        goes back as ``MEDIA_LAYOUT`` metadata for the LLM's MRoPE positions.
        """
        if prompt is None:
            # An image-only message renders with an empty text span. With no
            # attachments either, fail here, where it reaches the client as a
            # 400, not in the conductor, where the request would hang.
            if not any(m != TEXT for m in input_modalities):
                raise ValueError(
                    "Qwen3.5 got a request with no text and no attachments; "
                    "send a prompt, an image, a video, or a mix"
                )
            prompt = ""

        parts = parts_from_modalities(
            input_modalities,
            [p.text or "" for p in prompt_parts if p.modality == TEXT]
            if prompt_parts is not None else prompt,
            [p.role for p in prompt_parts or ()],
        )
        unsupported = {p.modality for p in parts} - {TEXT, "image", VIDEO}
        if unsupported:
            raise ValueError(
                f"Qwen3.5 here has no {', '.join(sorted(unsupported))} path; "
                "text, image and video attachments only"
            )
        tensors = tensors or {}
        raw_images = tensors.get("image_inputs", [])
        raw_videos = tensors.get("video_inputs", [])
        # always, or a named but unsent attachment is quietly dropped
        check_attachments(parts, {"image": len(raw_images), VIDEO: len(raw_videos)})
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
        input_ids = torch.tensor(encode_ids(self.tokenizer, text), dtype=torch.long)

        pad = self._pad_ids()
        # before the video processor spends seconds resizing a long video
        found = {m: int((input_ids == pad[m]).sum()) for m in pad}
        want = {"image": len(raw_images), VIDEO: sum(len(times) for times in timestamps)}
        for m, n in want.items():
            if found.get(m, 0) != n:
                raise ValueError(
                    f"multimodal placement mismatch: the prompt holds {found.get(m, 0)} {m} "
                    f"placeholders for {n} {m} inputs"
                )

        images = [
            self.image_processor(images=[_as_hwc_uint8(image)], return_tensors="pt")
            for image in raw_images
        ]
        videos = [
            self._preprocess_video(frames, metadata, times)
            for frames, metadata, times in zip(raw_videos, video_metadata, timestamps, strict=True)
        ]

        if (images or videos) and self.vision_config is None:
            raise ValueError("this Qwen3.5 checkpoint has no vision tower; send text only")
        merge = self.vision_config.spatial_merge_size if self.vision_config is not None else 1
        image_grids = [out["image_grid_thw"][0] for out in images]
        # each video frame group is placed like a one-frame image
        group_grids = [
            torch.tensor([1, int(v.grid[1]), int(v.grid[2])]) for v in videos for _ in range(int(v.grid[0]))
        ]
        grids = {pad["image"]: image_grids} if "image" in pad else {}
        if VIDEO in pad:
            grids[pad[VIDEO]] = group_grids
        pads = expand_media_pads(
            input_ids, {p: [int(g.prod()) // merge ** 2 for g in gs] for p, gs in grids.items()},
        )

        result: NameToTensorList = {"text_inputs": [pads.ids]}
        # the tower's inputs in prompt order, one entry an image or a video
        ordered = []
        for part in parts:
            if part.modality == "image":
                ordered.append((images[part.index]["pixel_values"], image_grids[part.index]))
            elif part.modality == VIDEO:
                ordered.append((videos[part.index].pixel_values, videos[part.index].grid))
        if ordered:
            result["pixel_values"] = [pixels for pixels, _ in ordered]
            result["image_grid_thw"] = [grid for _, grid in ordered]
        if videos:
            # the decoded frames are spent; don't hold them for the request's life
            result["video_inputs"] = []
        # always set, so a layout the client sent never reaches the LLM
        layout = media_layout(pads, dict.fromkeys(grids, "vision_embeds"), grids)
        # an image spans max(h', w') MRoPE positions, not its t * h' * w' tokens
        advance = int(pads.ids.numel()) + sum(
            vision_position_advance(grids[p][i], merge) - tokens
            for p, i, tokens in zip(pads.pad, pads.index, pads.tokens, strict=True)
        )
        return ProcessPromptOutput(result, {MEDIA_LAYOUT: layout, "kv_layout": self.kv_layout(pads, advance)})

    def kv_layout(self, pads: MediaPads, advance: int) -> dict:
        """`process_prompt`'s ``kv_layout``: the whole prompt, written by its
        one prefill walk and moving the position counter by ``advance``, then a
        slot per decode step."""
        present = [e for e in self.encoders if set(e.pad_ids) & set(pads.pad)]
        length = int(pads.ids.numel())
        span = Span(
            "rows" if present else "ids",
            media_walk_name(present) if present else "prefill_text",
            length, advance, "text_inputs",
        )
        return {KV_CACHE: {"main": LabelLayout(spans=(span,), growth=(("decode_loop", 1),))}}

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        # the request's output budget: the stop check no longer counts tokens
        return {"decode_loop": self.get_max_output_tokens(**model_kwargs)}

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        # unkeyed: prefix reuse stays off
        return {KV_CACHE: {"main": KVLabel(
            growth=(("decode", "decode_loop"),),
        )}}


    def postprocess(
            self,
            output: torch.Tensor,
            modality: str,
            request_kwargs: dict | None = None,
        ) -> bytes:
            if modality == "text":
                ids = drop_stop_tokens(output, self.config.stop_token_ids, request_kwargs)
                if self._text_detokenizer is None or self._text_detokenizer.tokenizer is not self.tokenizer:
                    self._text_detokenizer = TextChunkDetokenizer(self.tokenizer)
                return self._text_detokenizer.to_bytes(ids)
            raise ValueError(f"Unsupported modality for Qwen 3.5: {modality!r}")

    # ------------------------------------------------------------------
    # Model ABC: forward pass args
    # ------------------------------------------------------------------

    def _prefill_schedule(
        self,
        input_modalities: list[str],
        signals: dict[str, list[TensorPointerInfo]],
        layout: dict | None = None,
    ) -> list[PrefillStep]:
        """`media_prefill_schedule`, carrying `process_prompt`'s media layout."""
        return media_prefill_schedule(signals, self.encoders, layout, model_name="Qwen3.5")

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        schedule = self._prefill_schedule(
            input_modalities, input_signals, (model_kwargs or {}).get(MEDIA_LAYOUT),
        )
        return initial_prefill_args(
            schedule, input_modalities, output_modalities, input_signals, model_name="Qwen3.5",
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        return next_prefill_args(partition_metadata, persist_signals)

    # ------------------------------------------------------------------
    # Model ABC: per-request resource configs
    # ------------------------------------------------------------------

    def get_request_resource_configs(
        self,
        partition_fwd_args: dict[str, ForwardPassArgs],
        model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        model_kwargs = model_kwargs or {}
        keys = ["temperature", "top_p", "top_k", "repetition_penalty", "ignore_eos"]
        return {
            SAMPLER: SamplingReqConfig(
                **{k: model_kwargs[k] for k in keys if k in model_kwargs}
            )
        }

    # ------------------------------------------------------------------
    # Model ABC: submodule loading
    # ------------------------------------------------------------------

    def get_default_sharding_config(self) -> ShardingConfig:
        return ShardingConfig(
            groups=[], tp_enabled_nodes={"LLM"}, shard_dim={},
        )

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(
            node_name, device, autocast_dtype, tp_group,
        )
        self._submodule_cache[node_name] = submodule
        return submodule

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
                tower, self.vision_config, next(e for e in self.encoders if e.node == node_name),
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

    @staticmethod
    def _build(make, dtype: torch.dtype, device: str) -> torch.nn.Module:
        """Construct under `dtype` so no fp32 copy of the weights is
        allocated. fp32-only parameters restore themselves in `_apply`."""
        torch.set_default_dtype(dtype)
        try:
            return make().to(device)
        finally:
            torch.set_default_dtype(torch.float32)

    def _resolve_weights(self) -> str:
        """The full snapshot; only workers call this."""
        local_path = Path(self.model_path_hf)
        if local_path.is_dir():
            return str(local_path)
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=self.model_path_hf, cache_dir=self.cache_dir,
        )
