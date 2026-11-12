import logging
from typing import Any

import torch
from jinja2 import TemplateError
from transformers import AutoTokenizer

from mstar.engine.state import (
    Array,
    Attend,
    Block,
    Causal,
    CausalConv,
    Entry,
    GatedDeltaNet,
    KVEntry,
    Layout,
    Positions,
    Read,
    Sampler,
    SampleSettings,
    SeqShard,
    TensorType,
    Value,
)
from mstar.model.base import ProcessPromptOutput, TensorAndMetadata
from mstar.model.components.mrope import image_position_advance
from mstar.model.keys import MIXED, PREFILL_TEXT
from mstar.model.media_limits import read_image
from mstar.model.multimodal import (
    VIDEO,
    PromptPart,
    as_uint8_chw,
    media_walk_names,
    messages_from_parts,
    vision_encoder,
)
from mstar.model.multimodal_decoder import (
    MIXED_SPLIT,
    SAMPLING_KEYS,
    MediaIntake,
    MultimodalDecoderModel,
    PreparedMedia,
)
from mstar.model.qwen3_5.components.language_model import Qwen3_5ForCausalLM
from mstar.model.qwen3_5.config import (
    ATTN,
    CONV,
    GDN_STATE,
    KV_CACHE,
    LINEAR_ATTN,
    LLM_SHARD,
    SAMPLER,
    VISION_ATTN,
    VISION_TOKENS,
    Qwen3_5Config,
    Qwen3_5VisionConfig,
)
from mstar.model.qwen3_5.video import (
    ProcessedVideo,
    expand_video_pads,
    group_timestamps,
    sample_video,
)
from mstar.model.submodule_base import ChunkingPolicy, NodeSubmodule
from mstar.utils.fast_tokenize import encode_ids

logger = logging.getLogger(__name__)

# TODO: implement MoE variants
class Qwen3_5DenseModel(MultimodalDecoderModel):
    NAME = "Qwen3.5"
    REASONING_END = "</think>"
    METADATA_FILES = (
        "config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "tokenizer.json",
        "merges.txt",
        # the tokenizer prefers it to tokenizer_config.json's template, and
        # some fine-tunes (LightOnOCR-3, Surya) keep their template nowhere else
        "chat_template.jinja",
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
        # prompt order. A text-only checkpoint's tokenizer still has the pad
        # tokens, so the tower decides whether the model takes media.
        pad = self._pad_ids() if self.vision_config is not None else {}
        tower = vision_encoder() if pad else None
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

    def declare_state(self) -> list[Array | Sampler | SeqShard]:
        c, v = self.config, self.vision_config
        conv_dim = (
            2 * c.linear_num_key_heads * c.linear_key_head_dim
            + c.linear_num_value_heads * c.linear_value_head_dim
        )
        gdn = Entry({
            "state": TensorType(
                (c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim),
                dtype=getattr(torch, c.mamba_ssm_dtype, torch.float32), shard=0,
            ),
            "conv": TensorType((conv_dim, c.linear_conv_kernel_dim - 1), dtype=None, shard=0),
        })
        state = [
            Array(
                KV_CACHE, nodes={"LLM"}, layers=c.full_layer_indices,
                entry=KVEntry(heads=c.num_key_value_heads, head_dim=c.head_dim),
                max_len=c.max_position_embeddings,
                positions=Positions(),
                reads={ATTN: Read(mask=Causal(), op=Attend(q_heads=c.num_attention_heads))},
            ),
            # written out because Qwen3.5 diverges silently without qk_l2norm
            Value(
                GDN_STATE, nodes={"LLM"}, layers=c.linear_layer_indices, entry=gdn,
                updates={CONV: CausalConv(), LINEAR_ATTN: GatedDeltaNet(qk_l2norm=True)},
            ),
            Sampler(
                SAMPLER, nodes={"LLM"}, vocab_size=c.vocab_size, penalty=True,
                min_p=self._sampler_spec(c.vocab_size).enable_min_p,
            ),
            # Prefill steps may split over the LLM's SP ranks, and decode never does, so decode's compiled frame
            # traces the bracket straight through. "none" until a deployment opts in: the LLM on SP ranks over KV
            # stored by head is refused until it sets ulysses, which has run on CPU only, and over KV stored by
            # position it runs copied unless it sets all_gather or ring.
            SeqShard(
                LLM_SHARD, nodes={"LLM"}, walks={PREFILL_TEXT, *media_walk_names(self.encoders)}, exchange="none",
            ),
        ]
        if v is not None:
            state.append(Array(
                VISION_TOKENS, nodes={"vision_encoder"}, layers=frozenset(range(v.depth)),
                entry=KVEntry(heads=v.num_heads, head_dim=v.head_dim), max_len=0,
                reads={VISION_ATTN: Read(mask=Block(), op=Attend(q_heads=v.num_heads))},
            ))
        return state

    def kernel_settings(self) -> dict[str, dict[str, Any]]:
        settings = {
            ATTN: {"splits": (MIXED_SPLIT,), "decode_fmha": True},
            LINEAR_ATTN: {"splits": (MIXED_SPLIT,)},
        }
        if self.vision_config is not None:
            settings[VISION_ATTN] = {"max_segments_per_step": self._vision_segment_ceiling()}
        return settings

    def get_node_resources(self):
        from mstar.model.qwen3_5.lora import qwen3_5_lora_spec

        return [qwen3_5_lora_spec(self.config, self.vision_config)]

    def _vision_segment_ceiling(self) -> int:
        """Segments a captured block loop's plan holds. A segment per image, not per request: the resize floors each
        image at `shortest_edge` pixels. A video frame group has no floor; a step with more segments runs the block
        loop eagerly (`PackedBlockLoopEncoder._fits_capture`)."""
        from mstar.model.qwen3_5.submodules import VisionEncoderSubmodule

        processor = self.image_processor
        min_patches = processor.size["shortest_edge"] // processor.patch_size ** 2
        return VisionEncoderSubmodule.BLOCK_LOOP_PATCH_BUCKETS[-1] // min_patches

    def get_request_resource_configs(
        self, input_modalities: list[str], output_modalities: list[str], model_kwargs: dict,
    ) -> dict[str, SampleSettings]:
        config = super().get_request_resource_configs(input_modalities, output_modalities, model_kwargs)[SAMPLER]
        return {SAMPLER: SampleSettings(**{k: getattr(config, k) for k in SAMPLING_KEYS})}

    # -----------------------------------------------------------------------
    # Chunked prefill
    # -----------------------------------------------------------------------

    def get_chunking_policies(self) -> dict[str, tuple[ChunkingPolicy, ...]]:
        from mstar.model.qwen3_5.submodules import LLMSubmodule

        budget = LLMSubmodule.MAX_BATCH_TOKENS
        return {"LLM": (
            ChunkingPolicy(walks=(LLMSubmodule.PREFILL_WALK,), chunkable=True, max_batch_tokens=budget),
            # carries the budget for the steps that mix a chunk with decode rows
            ChunkingPolicy(walks=(MIXED,), max_batch_tokens=budget),
        )}

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
        """The checkpoint's ``video_preprocessor_config.json``, lazily. Its
        patches are normalized by the image processor's settings
        (`pixel_normalizer`), so it must agree with them; checked here, on the
        first video, so a deployment that never sees one never loads it."""
        if self._video_processor is None:
            from transformers import AutoVideoProcessor

            video = AutoVideoProcessor.from_pretrained(
                self.local_dir, cache_dir=self.cache_dir,
            )
            image = self.image_processor
            for name in ("rescale_factor", "image_mean", "image_std", "temporal_patch_size", "patch_size"):
                if getattr(video, name) != getattr(image, name):
                    raise ValueError(f"image and video processors disagree on {name}")
            self._video_processor = video
        return self._video_processor

    def pixel_normalizer(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-column mean and std that turn the intake's byte patches into
        the processors' normalized pixels: their fused rescale-and-normalize,
        ``(x - mean / rescale) / (std / rescale)``, element for element. One
        for images and videos, whose processors share it (`video_processor`)."""
        processor = self.image_processor
        scale = 1.0 / processor.rescale_factor
        mean = torch.tensor(processor.image_mean) * scale
        std = torch.tensor(processor.image_std) * scale
        # a patch's columns are channel-major (C x T x P x P)
        per_channel = processor.temporal_patch_size * processor.patch_size ** 2
        return mean.repeat_interleave(per_channel), std.repeat_interleave(per_channel)

    def load_image(self, filepath: str, device: str) -> TensorAndMetadata:
        """The decoded RGB image, left uint8 ``(C, H, W)``.

        The base class hands out floats in [0, 1], which would only be turned
        back into bytes for the processor: two full-image passes that cost
        more than the processor itself on a large image.
        """
        return TensorAndMetadata(read_image(filepath, rgb=True).to(device))

    def load_video(self, filepath: str, device: str) -> TensorAndMetadata:
        """Only the frames the video processor samples, on the host whatever
        ``device`` is (see `sample_video`)."""
        video = sample_video(filepath, self.video_processor)
        return TensorAndMetadata(data=video.frames, metadata=video.metadata)

    def _video_runs(self, video: torch.Tensor, metadata: dict) -> int:
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
            # bytes, normalized on the device like images' (`pixel_normalizer`)
            do_rescale=False,
            do_normalize=False,
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
        enable_thinking = kwargs.get("enable_thinking", True)
        if not isinstance(enable_thinking, bool):
            raise ValueError(f"enable_thinking must be a boolean; got {enable_thinking!r}")
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
                enable_thinking=enable_thinking,
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
        # Resized and patchified here, still bytes: the vision encoder
        # rescales and normalizes on the device (`pixel_normalizer`). That is
        # a quarter of the bytes to ship, and the float work was most of this
        # call (43 of 52 ms for a 2560 px image).
        out = self.image_processor(
            images=[as_uint8_chw(image, truncate=True).permute(1, 2, 0).numpy()], return_tensors="pt",
            do_rescale=False, do_normalize=False,
            # inferred, a 1, 3 or 4 px tall image would be read as channels first
            input_data_format="channels_last",
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

    def media_advance(self, tokens: int, grid: list[int]) -> int:
        # an image spans max(h', w') MRoPE positions, not its t * h' * w' tokens
        return image_position_advance(torch.tensor(grid), self._merge())

    def request_kwargs(self) -> frozenset[str]:
        return super().request_kwargs() | {"enable_thinking"}

    def process_prompt(self, *args: Any, **kwargs: Any) -> ProcessPromptOutput:
        out = super().process_prompt(*args, **kwargs)
        kv = out.metadata.pop("kv_layout")[KV_CACHE]["main"]
        out.metadata["state_layout"] = {
            KV_CACHE: {"main": Layout(spans=kv.spans, growth=kv.growth)}, GDN_STATE: {"main": Layout()},
        }
        if kv.keyed_spans is not None:
            out.metadata["keyed_spans"] = {KV_CACHE: {"main": kv.keyed_spans}}
        return out

    # ------------------------------------------------------------------
    # Model ABC: submodule loading
    # ------------------------------------------------------------------

    def _create_submodule(
        self, node_name: str, device: str, dtype: torch.dtype, tp_group,
    ) -> NodeSubmodule | None:
        from mstar.model.qwen3_5.submodules import (
            LLMSubmodule,
            VisionEncoderSubmodule,
        )

        if node_name not in ("LLM", "vision_encoder"):
            return None

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
                tower, self.vision_config, self.encoder(node_name).output, self.pixel_normalizer(),
            )

        from mstar.model.qwen3_5.weight_loader import load_qwen3_5_weights

        model = self._build(
            lambda: Qwen3_5ForCausalLM(self.config, tp_group), dtype, device,
        )
        load_qwen3_5_weights(model, weights_dir, device=device)
        model.requires_grad_(False).eval()
        logger.info("Loaded Qwen3.5 LLM submodule onto %s", device)
        return LLMSubmodule(model, self.config, self.vision_config, self.encoders)
