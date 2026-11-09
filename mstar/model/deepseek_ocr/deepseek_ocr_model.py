"""DeepSeek-OCR: page images in, text out, on the shared VLM scaffold.

Two nodes: ``vision_encoder`` (DeepEncoder: SAM-B + CLIP-L + projector) turns
a request's image views into the rows of its ``<image>`` tokens, and ``LLM``
(a DeepSeek-V2 MoE decoder with plain attention, a `SampledTokenDecoder`)
splices them in and prefills the prompt as one row, then decodes. The views
are cut on the API server, as the model card's ``infer`` cuts them; see
``preprocessing``.
"""

import dataclasses
from itertools import chain

import torch

from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    KVSpec,
    PagedKVConfig,
    PositionConfig,
    PositionSpec,
)
from mstar.model.base import ProcessPromptOutput, TensorAndMetadata
from mstar.model.deepseek_ocr.config import ATTN, IMAGE_TOKEN, MODES, DeepseekOCRConfig
from mstar.model.deepseek_ocr.preprocessing import image_views, num_image_tokens, to_pil
from mstar.model.keys import KV_CACHE, ROPE
from mstar.model.multimodal import MediaEncoder, PromptPart
from mstar.model.multimodal_decoder import MIXED_SPLIT, MediaIntake, MultimodalDecoderModel, PreparedMedia
from mstar.model.submodule_base import NodeSubmodule

TEXT_WEIGHTS = ("model.embed_tokens.", "model.layers.", "model.norm.", "lm_head.")
VISION_WEIGHTS = ("model.sam_model.", "model.vision_model.", "model.projector.",
                  "model.image_newline", "model.view_seperator")
# room for the BOS and an OCR instruction ("Free OCR.", the grounding prompts) beside the image
PROMPT_TEXT_ROOM = 64


class DeepseekOCRModel(MultimodalDecoderModel):
    NAME = "DeepSeek-OCR"
    METADATA_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    # The API server cuts a page's views with PIL (~100 ms a page, the GIL released);
    # one page at a time, that capped c=64 at 4.3 pages/s with the encoder starved.
    # process_prompt reads only its request and state set in __init__.
    PREPROCESS_PARALLEL = True

    def __init__(
        self, model_path_hf: str, cache_dir: str | None = None, max_seq_len: int | None = None,
        image_mode: str = "gundam", max_crops: int | None = None, **kwargs,
    ):
        super().__init__(model_path_hf, cache_dir)
        self.full_config = DeepseekOCRConfig.from_json(f"{self.local_dir}/config.json")
        text = self.full_config.text
        if text.sliding_window is not None:
            # Unlimited-OCR: generated tokens see the prompt plus a window of
            # generated ones. No attention resource expresses a per-request
            # prompt sink beside a window yet, and full attention diverges
            # from the reference once a reply outgrows the window.
            raise NotImplementedError(
                f"{model_path_hf}: prompt-sink sliding-window attention (window "
                f"{text.sliding_window}) is not supported yet"
            )
        if image_mode not in MODES:
            raise ValueError(f"image_mode must be one of {sorted(MODES)}, got {image_mode!r}")
        mode = MODES[image_mode]
        if max_crops is not None:
            if not mode.crop_mode:
                raise ValueError(f"max_crops caps gundam's tiles; image_mode {image_mode!r} cuts none")
            if type(max_crops) is not int:
                raise ValueError(f"max_crops must be an int, got {max_crops!r}")
            mode = dataclasses.replace(mode, max_crops=max_crops)
        self.image_mode = mode
        limit = text.max_position_embeddings
        self.max_seq_len = limit if max_seq_len is None else max_seq_len
        if type(self.max_seq_len) is not int or not 0 < self.max_seq_len <= limit:
            raise ValueError(f"max_seq_len must be in 1..{limit}")

        # Not AutoTokenizer: it reads config.json, whose auto_map asks to run
        # the checkpoint's remote code.
        from transformers import PreTrainedTokenizerFast

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.local_dir)
        # a fine-tune may set the tokenizer's EOS apart from the config's
        eos = self.tokenizer.eos_token_id
        self.config = dataclasses.replace(text, extra_stop_token_ids=() if eos is None else (eos,))
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        # one input per view size: an image smaller than a tile has no tiles
        encoder = MediaEncoder("vision", "vision_encoder", ("global_views", "tiles"), "vision_embeds",
                               (self.image_token_id,))
        self.media = (MediaIntake("image", encoder, self.image_token_id, self._image_intake),)
        # the model card decodes greedily; the checkpoint ships no generation_config
        self.sampling_defaults = {"temperature": 0.0, **self.sampling_defaults}

    # ------------------------------------------------------------------
    # resources

    def get_node_resources(self):
        config = self.config
        return [
            KVSpec(resource_key=KV_CACHE, nodes={"LLM"}, config=PagedKVConfig(
                num_layers=config.num_hidden_layers, num_qo_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads, head_dim=config.head_dim,
                max_seq_len=self.max_seq_len,
            )),
            AttentionSpec(resource_key=ATTN, nodes={"LLM"}, config=AttentionConfig(
                kv_cache=KV_CACHE, splits=(MIXED_SPLIT,),
            )),
            # The reference rotates with a bf16 cos/sin table (Llama's rotary
            # embedding casts it to the activations' dtype).
            PositionSpec(resource_key=ROPE, nodes={"LLM"}, config=PositionConfig(
                kv_cache=KV_CACHE, rotary_dim=config.head_dim, interleave=False,
                rope_theta=config.rope_theta, rope_table_dtype=torch.bfloat16,
            )),
            self._sampler_spec(config.vocab_size),
        ]

    def get_max_output_tokens(self, **model_kwargs):
        return model_kwargs.get("max_output_tokens", 4096)

    # ------------------------------------------------------------------
    # weights

    def _create_submodule(self, node_name: str, device: str, dtype: torch.dtype, tp_group) -> NodeSubmodule | None:
        if node_name not in ("LLM", "vision_encoder"):
            return None
        from mstar.model.deepseek_ocr.components.deepencoder import DeepEncoder
        from mstar.model.deepseek_ocr.components.language_model import DeepseekOCRLanguageModel
        from mstar.model.deepseek_ocr.submodules import DeepseekOCRLLMSubmodule, DeepseekOCRVisionSubmodule
        from mstar.model.loader import iter_safetensors_shards

        local_dir = self._resolve_weights()
        prefixes = TEXT_WEIGHTS if node_name == "LLM" else VISION_WEIGHTS
        weights = chain.from_iterable(
            iter_safetensors_shards(local_dir, device="cpu", prefix=prefix) for prefix in prefixes
        )
        # checkpoint names under ``model.`` drop it; ``lm_head`` has none
        weights = ((name.removeprefix("model."), tensor) for name, tensor in weights)
        if node_name == "vision_encoder":
            with torch.device("meta"):
                encoder = DeepEncoder(self.full_config.projector_input_dim, self.full_config.projector_dim)
            encoder = encoder.to(dtype=dtype).to_empty(device=device)
            _load_strict(encoder, weights, "DeepEncoder")
            mode = self.image_mode
            sizes = (mode.global_size, mode.image_size) if mode.crop_mode else (mode.global_size,)
            return DeepseekOCRVisionSubmodule(encoder, self.encoder(node_name).output, view_sizes=sizes).eval()
        with torch.device("meta"):
            language_model = DeepseekOCRLanguageModel(self.config, tp_group)
        language_model = language_model.to(dtype=dtype).to_empty(device=device)
        language_model.load_weights(weights)
        return DeepseekOCRLLMSubmodule(
            language_model.eval(), self.config, self.encoders, max_prompt_tokens=self.max_prompt_tokens(),
        ).eval()

    # ------------------------------------------------------------------
    # prompts

    def max_prompt_tokens(self) -> int:
        """An image prompt's tokens at most: the largest image the mode cuts,
        plus room for the BOS and a short instruction."""
        from mstar.model.deepseek_ocr.preprocessing import max_image_tokens

        return max_image_tokens(self.image_mode) + PROMPT_TEXT_ROOM

    def load_image(self, filepath: str, device: str) -> TensorAndMetadata:
        """The decoded RGB image, left uint8 ``(C, H, W)``: PIL cuts the views from bytes."""
        from mstar.model.media_limits import read_image

        return TensorAndMetadata(read_image(filepath, rgb=True))

    @staticmethod
    def render(parts: list[PromptPart], num_images: int) -> str:
        """The prompt with ``<image>`` where each image goes.

        Text that places ``<image>`` itself is taken as written. Otherwise the
        images lead, each as ``<image>\\n``, then the text, whatever order a
        chat sends its parts in: the model card's prompts
        (``<image>\\nFree OCR.``), as vLLM places them. With the instruction
        before the page the model mostly loops (olmOCR-bench 33% against 75%).
        """
        text = "\n".join(p.text or "" for p in parts if p.modality == "text")
        if IMAGE_TOKEN in text:
            return text.strip()
        return (f"{IMAGE_TOKEN}\n" * num_images + text).strip()

    def _prompt_ids(self, parts: list[PromptPart], raw: dict[str, list[torch.Tensor]], **kwargs) -> torch.Tensor:
        text = self.render(parts, len(raw.get("image", ())))
        if not text:
            # with no image, the decoder would generate from BOS alone
            raise ValueError("DeepSeek-OCR needs an image or text")
        # One call: ``<image>`` is an added token, so the text around it
        # encodes as the reference's per-piece encode does.
        ids = [self.config.bos_token_id, *self.tokenizer.encode(text, add_special_tokens=False)]
        return torch.tensor(ids, dtype=torch.long)

    def _image_intake(self, image: torch.Tensor, metadata: dict) -> PreparedMedia:
        cut = image_views(to_pil(image), self.image_mode)
        inputs = {"global_views": cut.global_view}
        if cut.crops is not None:
            inputs["tiles"] = cut.crops
        tokens = num_image_tokens(self.image_mode, cut.cols, cut.rows)
        return PreparedMedia(inputs, [tokens], [(cut.cols, cut.rows)])

    def process_prompt(
        self, prompt, input_modalities, output_modalities, tensors=None, prompt_parts=None, **kwargs,
    ) -> ProcessPromptOutput:
        if output_modalities != ["text"]:
            raise ValueError("DeepSeek-OCR produces text only")
        out = super().process_prompt(prompt, input_modalities, output_modalities, tensors, prompt_parts, **kwargs)
        length = out.new_input_tensors["text_inputs"][0].numel()
        max_output_tokens = self.get_max_output_tokens(**kwargs)
        if length + max_output_tokens > self.max_seq_len:
            raise ValueError(
                f"prompt ({length} tokens) + max_output_tokens ({max_output_tokens}) "
                f"exceeds max_seq_len ({self.max_seq_len})"
            )
        return out


def _load_strict(module: torch.nn.Module, weights, what: str) -> None:
    """Name-matched copy; raises on any parameter left unfilled."""
    params = dict(module.named_parameters())
    loaded = set()
    for name, tensor in weights:
        if name in params:
            params[name].data.copy_(tensor)
            loaded.add(name)
    missing = sorted(set(params) - loaded)
    if missing:
        raise RuntimeError(f"{what} left {len(missing)} parameter(s) unloaded, e.g. {missing[:5]}")
