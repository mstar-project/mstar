"""DeepSeek-OCR: page images in, text out.

Two nodes: ``vision_encoder`` (DeepEncoder: SAM-B + CLIP-L + projector) turns a
request's image views into the rows of its ``<image>`` tokens, and ``LLM`` (a
DeepSeek-V2 MoE decoder with plain attention) prefills them with the prompt
and decodes. The views are cut on the API server, as the model card's ``infer``
cuts them; see ``preprocessing``.
"""

import dataclasses
import threading
from itertools import chain
from pathlib import Path

import torch

from mstar.conductor.request_info import DEFAULT_PARTITION, CurrentForwardConductorMetadata
from mstar.distributed.base import ShardingConfig
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    KVSpec,
    PagedKVConfig,
    PositionConfig,
    PositionSpec,
    SamplerSpec,
    SamplingReqConfig,
)
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import (
    ForwardPassArgs,
    KVLabel,
    LabelLayout,
    Model,
    ProcessPromptOutput,
    Span,
    TensorAndMetadata,
)
from mstar.model.deepseek_ocr.config import (
    ATTN,
    IMAGE_GRIDS_KEY,
    IMAGE_TOKEN,
    KV_CACHE,
    MODES,
    PREFILL_WALKS,
    ROPE,
    SAMPLER,
    STEP_GRIDS_KEY,
    DeepseekOCRConfig,
)
from mstar.model.deepseek_ocr.preprocessing import image_views, num_image_tokens, to_pil
from mstar.model.utils import TextOutput

TEXT_WEIGHTS = ("model.embed_tokens.", "model.layers.", "model.norm.", "lm_head.")
VISION_WEIGHTS = ("model.sam_model.", "model.vision_model.", "model.projector.",
                  "model.image_newline", "model.view_seperator")
_CHECKPOINT_FILES = ["*.json", "*.safetensors"]
# room for the BOS and an OCR instruction ("Free OCR.", the grounding prompts) beside the image
PROMPT_TEXT_ROOM = 64
# one lazy tokenizer load at a time (see `DeepseekOCRModel.tokenizer`)
_TOKENIZER_LOCK = threading.Lock()


class DeepseekOCRModel(Model):
    # The API server cuts a page's views with PIL (~100 ms a page, the GIL released);
    # one page at a time, that capped c=64 at 4.3 pages/s with the encoder starved.
    # process_prompt reads only its request and state that is set once, under a lock.
    PREPROCESS_PARALLEL = True

    def __init__(
        self, model_path_hf: str, cache_dir: str | None = None, max_seq_len: int | None = None,
        image_mode: str = "gundam", max_crops: int | None = None, **kwargs,
    ):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.local_dir = self._resolve(allow_patterns=["*.json"])
        self.full_config = DeepseekOCRConfig.from_json(self.local_dir / "config.json")
        self.config = self.full_config.text
        if self.config.sliding_window is not None:
            # Unlimited-OCR: generated tokens see the prompt plus a window of
            # generated ones. No attention resource expresses a per-request
            # prompt sink beside a window yet, and full attention diverges
            # from the reference once a reply outgrows the window.
            raise NotImplementedError(
                f"{model_path_hf}: prompt-sink sliding-window attention (window "
                f"{self.config.sliding_window}) is not supported yet"
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
        limit = self.config.max_position_embeddings
        self.max_seq_len = limit if max_seq_len is None else max_seq_len
        if type(self.max_seq_len) is not int or not 0 < self.max_seq_len <= limit:
            raise ValueError(f"max_seq_len must be in 1..{limit}")
        self._tokenizer = None
        self._image_token_id = None

    def _resolve(self, allow_patterns: list[str]) -> Path:
        local = Path(self.model_path_hf)
        if local.is_dir():
            return local
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=self.model_path_hf, cache_dir=self.cache_dir,
                                      allow_patterns=allow_patterns))

    # ------------------------------------------------------------------
    # graph

    def get_graph_walk_graphs(self):
        def prefill(*inputs):
            return GraphNode(
                name="LLM", input_names=["text_inputs", *inputs], outputs=[
                    GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token",
                              output_modality="text", conductor_new_token=True),
                    # Absent when prefill already finishes the request.
                    GraphEdge(next_node=EMPTY_DESTINATION, name="decode_input", persist=True),
                ],
            )

        return {
            "prefill": prefill(),
            "prefill_image": Sequential([
                GraphNode(name="vision_encoder", input_names=["image_views"], outputs=[
                    GraphEdge(next_node="LLM", name="image_embeds"),
                ]),
                prefill("image_embeds"),
            ]),
            "decode": Loop(
                # the request's max_output_tokens ends it earlier (get_loop_caps)
                name="decode_loop", max_iters=self.max_seq_len, outputs=[],
                section=GraphNode(name="LLM", input_names=["text_inputs"], outputs=[
                    # one signal, so the token loops back on the device and its client copy
                    # rides inline (`DeviceLoopback`); the loop caps end decode at
                    # max_output_tokens, so the conductor needn't count it
                    GraphEdge(next_node="LLM", name="text_inputs"),
                    GraphEdge(next_node=EMIT_TO_CLIENT, name="text_inputs", output_modality="text"),
                ]),
            ),
        }

    @staticmethod
    def _check_partition(partition_name):
        if partition_name != DEFAULT_PARTITION:
            raise ValueError(f"Unknown DeepSeek-OCR partition: {partition_name!r}")

    def get_initial_forward_pass_args(
        self, partition_name, input_modalities, output_modalities, input_signals, model_kwargs=None,
    ):
        self._check_partition(partition_name)
        if not input_signals.get("text_inputs"):
            raise ValueError("DeepSeek-OCR prefill requires text_inputs")
        views = input_signals.get("image_views", [])
        inputs = [GraphEdge(next_node="LLM", name="text_inputs", tensor_info=input_signals["text_inputs"])]
        step_metadata = {}
        if views:
            inputs.append(GraphEdge(next_node="vision_encoder", name="image_views", tensor_info=views))
            # known on the host since process_prompt, so it travels as metadata, not a tensor
            step_metadata[STEP_GRIDS_KEY] = (model_kwargs or {})[IMAGE_GRIDS_KEY]
        return ForwardPassArgs(
            full_metadata=CurrentForwardConductorMetadata(
                graph_walk="prefill_image" if views else "prefill", is_prefill=True,
                input_modalities=input_modalities, output_modalities=output_modalities,
            ),
            inputs=inputs, unpersist_tensors=[info for edge in inputs for info in edge.tensor_info],
            step_metadata=step_metadata,
        )

    def get_partition_forward_pass_args(
        self, partition_name, partition_metadata, persist_signals, incoming_connections=None,
    ):
        self._check_partition(partition_name)
        metadata = partition_metadata
        seeds = persist_signals.get("decode_input", [])
        if metadata.graph_walk == "decode" or not seeds:
            return ForwardPassArgs(metadata, [], list(seeds), request_done=True)
        if metadata.graph_walk not in PREFILL_WALKS:
            raise ValueError(f"Unexpected graph walk: {metadata.graph_walk!r}")
        metadata.graph_walk = "decode"
        metadata.is_prefill = False
        edge = GraphEdge(next_node="LLM", name="text_inputs", tensor_info=seeds)
        return ForwardPassArgs(metadata, [edge], list(seeds))

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
            AttentionSpec(resource_key=ATTN, nodes={"LLM"}, config=AttentionConfig(kv_cache=KV_CACHE)),
            # The reference rotates with a bf16 cos/sin table (Llama's rotary
            # embedding casts it to the activations' dtype).
            PositionSpec(resource_key=ROPE, nodes={"LLM"}, config=PositionConfig(
                kv_cache=KV_CACHE, rotary_dim=config.head_dim, interleave=False,
                rope_theta=config.rope_theta, rope_table_dtype=torch.bfloat16,
            )),
            SamplerSpec(resource_key=SAMPLER, nodes={"LLM"}, vocab_size=config.vocab_size,
                        enable_repetion_penalty=True),
        ]

    def get_request_resource_configs(self, partition_fwd_args, model_kwargs=None):
        # The model card decodes greedily.
        # min_p too: the sampler has no min-p filter here, and its check refuses the request
        defaults = dict(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, repetition_penalty=1.0,
                        ignore_eos=False, return_logprobs=False)
        supplied = model_kwargs or {}
        return {SAMPLER: SamplingReqConfig(**{key: supplied.get(key, val) for key, val in defaults.items()})}

    def get_max_output_tokens(self, **model_kwargs):
        return model_kwargs.get("max_output_tokens", 4096)

    def get_loop_caps(self, **model_kwargs):
        # Prefill samples the first token, so decode runs one fewer.
        return {"decode_loop": max(1, self.get_max_output_tokens(**model_kwargs) - 1)}

    def kv_labels(self):
        return {KV_CACHE: {"main": KVLabel(growth=(("decode", "decode_loop"),))}}

    def get_default_sharding_config(self):
        return ShardingConfig(groups=[], tp_enabled_nodes={"LLM"}, shard_dim={})

    # ------------------------------------------------------------------
    # weights

    def get_submodule(self, node_name, device="cpu", tp_group=None, autocast_dtype=None, sp_group=None):
        if node_name not in ("LLM", "vision_encoder"):
            raise ValueError(f"Unknown DeepSeek-OCR node: {node_name!r}")
        if sp_group is not None and sp_group.world_size != 1:
            raise ValueError("DeepSeek-OCR sequence parallelism is not implemented")
        from mstar.model.deepseek_ocr.components.deepencoder import DeepEncoder
        from mstar.model.deepseek_ocr.components.language_model import DeepseekOCRLanguageModel
        from mstar.model.deepseek_ocr.submodules import DeepseekOCRLLMSubmodule, DeepseekOCRVisionSubmodule
        from mstar.model.loader import iter_safetensors_shards

        local_dir = self._resolve(allow_patterns=_CHECKPOINT_FILES)
        dtype = autocast_dtype if autocast_dtype is not None else torch.float32
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
            return DeepseekOCRVisionSubmodule(encoder, view_sizes=sizes).eval()
        with torch.device("meta"):
            language_model = DeepseekOCRLanguageModel(self.config, tp_group)
        language_model = language_model.to(dtype=dtype).to_empty(device=device)
        language_model.load_weights(weights)
        return DeepseekOCRLLMSubmodule(
            language_model.eval(), self.config, self._get_image_token_id(), self.stop_token_ids(),
            max_prompt_tokens=self.max_prompt_tokens(),
        ).eval()

    # ------------------------------------------------------------------
    # prompts

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            # The prepare pool's first requests arrive together, and transformers'
            # lazy import fails when two threads take it at once.
            with _TOKENIZER_LOCK:
                if self._tokenizer is None:
                    # Not AutoTokenizer: it reads config.json, whose auto_map asks to
                    # run the checkpoint's remote code.
                    from transformers import PreTrainedTokenizerFast

                    self._tokenizer = PreTrainedTokenizerFast.from_pretrained(self.local_dir)
        return self._tokenizer

    def max_prompt_tokens(self) -> int:
        """An image prompt's tokens at most: the largest image the mode cuts,
        plus room for the BOS and a short instruction."""
        from mstar.model.deepseek_ocr.preprocessing import max_image_tokens

        return max_image_tokens(self.image_mode) + PROMPT_TEXT_ROOM

    def _get_image_token_id(self) -> int:
        if self._image_token_id is None:
            self._image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        return self._image_token_id

    def load_image(self, filepath: str, device: str) -> TensorAndMetadata:
        """The decoded RGB image, left uint8 ``(C, H, W)``: PIL cuts the views from bytes."""
        from mstar.model.media_limits import read_image

        return TensorAndMetadata(read_image(filepath, rgb=True))

    @staticmethod
    def _render(prompt, prompt_parts, num_images: int) -> str:
        """The prompt with ``<image>`` where each image goes.

        Text that places ``<image>`` itself is taken as written. Otherwise the
        images lead, each as ``<image>\\n``, then the text, whatever order a
        chat sends its parts in: the model card's prompts
        (``<image>\\nFree OCR.``), as vLLM places them. With the instruction
        before the page the model mostly loops (olmOCR-bench 33% against 75%).
        """
        if prompt_parts is None:
            if not isinstance(prompt, str):
                raise ValueError("Expected a text prompt")
            text = prompt
        else:
            text = "\n".join(p.text or "" for p in prompt_parts if p.modality == "text")
        if IMAGE_TOKEN in text:
            return text.strip()
        return (f"{IMAGE_TOKEN}\n" * num_images + text).strip()

    def process_prompt(
        self, prompt, input_modalities, output_modalities, tensors=None, prompt_parts=None, **kwargs,
    ):
        if output_modalities != ["text"] or not set(input_modalities) <= {"text", "image"}:
            raise ValueError("DeepSeek-OCR takes text and image input and produces text")
        tensors = tensors or {}
        images = tensors.get("image_inputs", [])
        text = self._render(prompt, prompt_parts, len(images))
        if not text:
            # with no image, the decoder would generate from BOS alone
            raise ValueError("DeepSeek-OCR needs an image or text")
        splits = text.split(IMAGE_TOKEN)
        if len(splits) - 1 != len(images):
            raise ValueError(f"The prompt places {len(splits) - 1} images but the request has {len(images)}")
        tokenizer, image_id = self.tokenizer, self._get_image_token_id()
        # Each text piece is encoded on its own, as the reference does.
        ids = [self.config.bos_token_id]
        views, grid = [], []
        for piece, image in zip(splits, [*images, None], strict=True):
            ids += tokenizer.encode(piece, add_special_tokens=False)
            if image is None:
                continue
            cut = image_views(to_pil(image), self.image_mode)
            views.append(cut.global_view)
            if cut.crops is not None:
                views.append(cut.crops)
            grid.append([cut.cols, cut.rows])
            ids += [image_id] * num_image_tokens(self.image_mode, cut.cols, cut.rows)
        ids = torch.tensor(ids, dtype=torch.long)
        max_output_tokens = self.get_max_output_tokens(**kwargs)
        if ids.numel() + max_output_tokens > self.max_seq_len:
            raise ValueError(
                f"prompt ({ids.numel()} tokens) + max_output_tokens ({max_output_tokens}) "
                f"exceeds max_seq_len ({self.max_seq_len})"
            )
        result = {"text_inputs": [ids]}
        metadata = {}
        if views:
            # the decoded pages are spent once cut: don't hold them for the request's life
            result["image_inputs"] = []
            result["image_views"] = views
            metadata[IMAGE_GRIDS_KEY] = grid
        # Image rows are not keyed by the placeholder ids they replace.
        span = Span("rows" if views else "ids", "prefill_image" if views else "prefill",
                    ids.numel(), ids.numel(), "text_inputs")
        layout = LabelLayout(spans=(span,), growth=(("decode_loop", 1),))
        return ProcessPromptOutput(result, {**metadata, "kv_layout": {KV_CACHE: {"main": layout}}})

    def stop_token_ids(self) -> frozenset[int]:
        """The config's EOS and the tokenizer's, which a fine-tune may set apart."""
        stop = {self.config.eos_token_id}
        if self.tokenizer.eos_token_id is not None:
            stop.add(self.tokenizer.eos_token_id)
        return frozenset(stop)

    def text_output(self) -> TextOutput:
        return TextOutput(self._chunk_detokenizer(), self.stop_token_ids())

    def postprocess(self, output, modality, request_kwargs=None):
        if modality != "text":
            raise ValueError("DeepSeek-OCR produces text only")
        return super().postprocess(output, modality, request_kwargs)


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
