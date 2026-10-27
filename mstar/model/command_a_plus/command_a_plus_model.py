"""Text and image-in, text-out generation with pinned metadata and explicitly supplied weights."""

import json
from itertools import chain

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
from mstar.model.base import ForwardPassArgs, KVLabel, LabelLayout, Model, ProcessPromptOutput, Span
from mstar.model.command_a_plus.assets import resolve_metadata
from mstar.model.command_a_plus.config import (
    GLOBAL_ATTN,
    KV_CACHE,
    LOCAL_ATTN,
    PREFILL_WALKS,
    ROPE,
    SAMPLER,
    CommandAPlusConfig,
)
from mstar.model.multimodal import messages_from_parts, parts_from_modalities

TEXT_WEIGHTS = ("model.language_model.",)
VISION_WEIGHTS = ("model.vision_tower.", "model.multi_modal_projector.")


class CommandAPlusModel(Model):
    def __init__(
        self, model_path_hf: str, cache_dir: str | None = None,
        max_seq_len: int | None = None, checkpoint_dir: str | None = None, **kwargs,
    ):
        self.local_dir = resolve_metadata(checkpoint_dir or model_path_hf, cache_dir)
        config = CommandAPlusConfig.from_json(self.local_dir / "config.json")
        self.config = config.text_config
        self.vision_config = config.vision_config
        self.max_seq_len = min(8192, self.config.max_position_embeddings) if max_seq_len is None else max_seq_len
        if type(self.max_seq_len) is not int or not 0 < self.max_seq_len <= self.config.max_position_embeddings:
            raise ValueError("max_seq_len must be positive and within max_position_embeddings")
        self._tokenizer = None
        self._detokenizer = None
        self._image_processor = None
        self._image_tokens = None

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
                GraphNode(name="vision_encoder", input_names=["pixel_values"], outputs=[
                    GraphEdge(next_node="LLM", name="image_embeds"),
                ]),
                prefill("image_embeds"),
            ]),
            "decode": Loop(
                # Per-request max_output_tokens ends the loop earlier; this is
                # only the hard bound, so it must not cap the request's choice.
                name="decode_loop", max_iters=self.max_seq_len, outputs=[],
                section=GraphNode(name="LLM", input_names=["text_inputs"], outputs=[
                    GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token",
                              output_modality="text", conductor_new_token=True),
                    GraphEdge(next_node="LLM", name="text_inputs"),
                ]),
            ),
        }

    @staticmethod
    def _check_partition(partition_name):
        if partition_name != DEFAULT_PARTITION:
            raise ValueError(f"Unknown Command A+ partition: {partition_name!r}")

    def get_initial_forward_pass_args(
        self, partition_name, input_modalities, output_modalities, input_signals,
        model_kwargs=None,
    ):
        self._check_partition(partition_name)
        if not input_signals.get("text_inputs"):
            raise ValueError("Command A+ prefill requires text_inputs")
        pixels = input_signals.get("pixel_values", [])
        inputs = [GraphEdge(next_node="LLM", name="text_inputs", tensor_info=input_signals["text_inputs"])]
        if pixels:
            inputs.append(GraphEdge(next_node="vision_encoder", name="pixel_values", tensor_info=pixels))
        return ForwardPassArgs(
            full_metadata=CurrentForwardConductorMetadata(
                graph_walk="prefill_image" if pixels else "prefill", is_prefill=True,
                input_modalities=input_modalities, output_modalities=output_modalities,
            ),
            inputs=inputs, unpersist_tensors=[info for edge in inputs for info in edge.tensor_info],
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

    def get_node_resources(self):
        config = self.config
        return [
            KVSpec(resource_key=KV_CACHE, nodes={"LLM"}, config=PagedKVConfig(
                num_layers=config.num_hidden_layers, num_qo_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads, head_dim=config.head_dim,
                max_seq_len=self.max_seq_len,
            )),
            AttentionSpec(resource_key=LOCAL_ATTN, nodes={"LLM"}, config=AttentionConfig(
                kv_cache=KV_CACHE, sliding_window=config.sliding_window,
            )),
            AttentionSpec(resource_key=GLOBAL_ATTN, nodes={"LLM"}, config=AttentionConfig(
                kv_cache=KV_CACHE,
            )),
            PositionSpec(resource_key=ROPE, nodes={"LLM"}, config=PositionConfig(
                kv_cache=KV_CACHE, rotary_dim=config.head_dim, interleave=True,
                rope_theta=config.rope_theta,
            )),
            SamplerSpec(resource_key=SAMPLER, nodes={"LLM"}, vocab_size=config.vocab_size,
                        enable_repetion_penalty=True),
        ]

    def get_request_resource_configs(self, partition_fwd_args, model_kwargs=None):
        defaults = dict(temperature=0.9, top_p=0.95, top_k=0,
                        repetition_penalty=1.04, ignore_eos=False)
        supplied = model_kwargs or {}
        return {SAMPLER: SamplingReqConfig(**{key: supplied.get(key, val) for key, val in defaults.items()})}

    def request_kwargs(self):
        return frozenset({
            "messages", "max_output_tokens", "temperature", "top_p", "top_k", "repetition_penalty", "ignore_eos",
        })

    def get_loop_caps(self, **model_kwargs):
        # Prefill samples the first token, so decode runs one fewer.
        return {"decode_loop": max(1, self.get_max_output_tokens(**model_kwargs) - 1)}

    def kv_labels(self):
        return {KV_CACHE: {"main": KVLabel(growth=(("decode", "decode_loop"),))}}

    def get_default_sharding_config(self):
        return ShardingConfig(groups=[], tp_enabled_nodes={"LLM"}, shard_dim={})

    def get_submodule(
        self, node_name, device="cpu", tp_group=None, autocast_dtype=None, sp_group=None,
    ):
        if node_name not in ("LLM", "vision_encoder"):
            raise ValueError(f"Unknown Command A+ node: {node_name!r}")
        if sp_group is not None and sp_group.world_size != 1:
            raise ValueError("Command A+ sequence parallelism is not implemented")
        from mstar.model.command_a_plus.components.language_model import CommandAPlusForCausalLM
        from mstar.model.command_a_plus.components.vision import CommandAPlusVisionEncoder
        from mstar.model.command_a_plus.submodules import CommandAPlusLLMSubmodule, CommandAPlusVisionSubmodule
        from mstar.model.loader import iter_safetensors_shards

        prefixes = TEXT_WEIGHTS if node_name == "LLM" else VISION_WEIGHTS
        # Validate availability before allocating hundreds of GB of parameters.
        index = self.local_dir / "model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            required = {shard for name, shard in weight_map.items() if name.startswith(prefixes)}
        else:
            required = {"model.safetensors"}
        missing = [name for name in sorted(required) if not (self.local_dir / name).is_file()]
        if not required or missing:
            raise FileNotFoundError(
                "Command A+ weights are not present locally; obtain the checkpoint explicitly "
                f"before starting workers. Missing shards: {missing[:3]}"
            )

        dtype = autocast_dtype if autocast_dtype is not None else torch.float32
        # Stream source tensors through CPU; loaders copy just this rank's slice.
        weights = chain.from_iterable(
            iter_safetensors_shards(self.local_dir, device="cpu", prefix=prefix) for prefix in prefixes
        )
        if node_name == "vision_encoder":
            with torch.device("meta"):
                encoder = CommandAPlusVisionEncoder(self.vision_config, self.config.hidden_size).to(dtype=dtype)
            encoder.to_empty(device=device)
            encoder.load_weights(weights)
            return CommandAPlusVisionSubmodule(encoder).eval()
        with torch.device("meta"):
            language_model = CommandAPlusForCausalLM(self.config, tp_group).to(dtype=dtype)
        language_model.to_empty(device=device)
        language_model.load_weights(weights)
        language_model.eval()
        if torch.device(device).type == "cuda":
            language_model.model.fuse_for_inference()
        return CommandAPlusLLMSubmodule(language_model, self.config, self.vision_config.image_token_id).eval()

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.local_dir, local_files_only=True)
        return self._tokenizer

    def _get_image_processor(self):
        if self._image_processor is None:
            from transformers import AutoImageProcessor

            self._image_processor = AutoImageProcessor.from_pretrained(self.local_dir, local_files_only=True)
        return self._image_processor

    def _get_image_tokens(self):
        """The start and end of image ids, and one tile's tokens ending in its line break."""
        if self._image_tokens is None:
            start, end, line_break = self._get_tokenizer().convert_tokens_to_ids(
                ["<|START_OF_IMG|>", "<|END_OF_IMG|>", "<|IMG_LINE_BREAK|>"],
            )
            tile = [self.vision_config.image_token_id] * self.vision_config.tokens_per_tile + [line_break]
            self._image_tokens = start, end, tile
        return self._image_tokens

    def process_prompt(
        self, prompt, input_modalities, output_modalities, tensors=None, prompt_parts=None, **kwargs,
    ):
        if output_modalities != ["text"] or not set(input_modalities) <= {"text", "image"}:
            raise ValueError("Command A+ takes text and image input and produces text")
        tensors = tensors or {}
        if "text_inputs" in tensors:
            ids = tensors["text_inputs"]
            if len(ids) != 1:
                raise ValueError("Expected one prompt token tensor")
            ids = ids[0]
        else:
            ids = torch.tensor(self._get_tokenizer().apply_chat_template(
                self._chat_messages(prompt, input_modalities, prompt_parts, kwargs.get("messages")),
                tokenize=True, add_generation_prompt=True, return_dict=False,
            ), dtype=torch.long)
        max_output_tokens = self.get_max_output_tokens(**kwargs)
        ids, pixel_values = self._expand_images(ids, tensors.get("image_inputs", []), max_output_tokens)
        if ids.ndim != 1 or ids.numel() == 0 or ids.numel() > self.max_seq_len:
            raise ValueError("Prompt must contain 1..max_seq_len token IDs")
        if ids.numel() + max_output_tokens > self.max_seq_len:
            raise ValueError(
                f"prompt ({ids.numel()} tokens) + max_output_tokens ({max_output_tokens}) "
                f"exceeds max_seq_len ({self.max_seq_len})"
            )
        if ids.dtype not in (torch.int32, torch.int64) or bool(((ids < 0) | (ids >= self.config.vocab_size)).any()):
            raise ValueError("Prompt token IDs must be integers within the vocabulary")
        result = {"text_inputs": [ids.to(dtype=torch.long)]}
        if pixel_values:
            result["pixel_values"] = pixel_values
        # Image rows are not keyed by the placeholder ids they replace.
        span = Span("rows" if pixel_values else "ids", "prefill_image" if pixel_values else "prefill",
                    ids.numel(), ids.numel(), "text_inputs")
        layout = LabelLayout(spans=(span,), growth=(("decode_loop", 1),))
        return ProcessPromptOutput(result, {"kv_layout": {KV_CACHE: {"main": layout}}})

    @staticmethod
    def _chat_messages(prompt, input_modalities, prompt_parts, messages):
        if messages is not None:
            if not (
                isinstance(messages, list) and messages
                and all(isinstance(m, dict) and set(m) == {"role", "content"}
                        and m["role"] in ("system", "user", "assistant")
                        and isinstance(m["content"], (str, list)) for m in messages)
            ):
                raise ValueError("messages must be a non-empty list of {role, content} chat turns")
            return messages
        if "image" in input_modalities:
            texts = prompt if prompt_parts is None else [p.text or "" for p in prompt_parts if p.modality == "text"]
            roles = [p.role for p in prompt_parts or ()]
            return messages_from_parts(parts_from_modalities(input_modalities, texts, roles))
        if prompt_parts is not None:
            prompt = "".join(part.text or "" for part in prompt_parts)
        if not isinstance(prompt, str):
            raise ValueError("Expected a text prompt")
        return [{"role": "user", "content": prompt}]

    def _expand_images(self, ids, images, max_output_tokens):
        """Tile each image and widen its one ``<|IMG_PATCH|>`` placeholder to the tiles' tokens."""
        image_token = self.vision_config.image_token_id
        positions = (ids == image_token).nonzero().flatten().tolist()
        if len(positions) != len(images):
            raise ValueError(f"The prompt has {len(positions)} image placeholders for {len(images)} images")
        if not images:
            return ids, []
        start_of_image, end_of_image, tile = self._get_image_tokens()
        pieces, pixel_values, start = [], [], 0
        length = ids.numel() - len(images)
        for position, image in zip(positions, images, strict=True):
            # load_image gives floats in [0, 1]; the processor expects the image's bytes.
            image = (image.cpu() * 255).round().clamp(0, 255).to(torch.uint8)
            tiles = self._get_image_processor()(images=[image], return_tensors="pt")["pixel_values"]
            length += len(tiles) * len(tile) + 2
            if length + max_output_tokens > self.max_seq_len:
                raise ValueError(
                    f"prompt (at least {length} tokens) + max_output_tokens ({max_output_tokens}) "
                    f"exceeds max_seq_len ({self.max_seq_len})"
                )
            pieces += [ids[start:position], torch.tensor([start_of_image, *tile * len(tiles), end_of_image])]
            pixel_values.append(tiles.to(torch.bfloat16))
            start = position + 1
        pieces.append(ids[start:])
        return torch.cat(pieces), pixel_values

    def postprocess(self, output, modality, request_kwargs=None):
        if modality != "text":
            raise ValueError("Command A+ produces text only")
        tokenizer = self._get_tokenizer()
        if self._detokenizer is None:
            # Per-token decode can corrupt UTF-8 characters split across tokens.
            # Only enable this byte mapping for a matching tokenizer decoder.
            decoder = json.loads(tokenizer.backend_tokenizer.decoder.__getstate__())
            if decoder.get("type") != "ByteLevel":
                raise ValueError("Command A+ streaming requires a ByteLevel tokenizer decoder")
            from mstar.model.utils import ByteLevelDetokenizer

            self._detokenizer = ByteLevelDetokenizer(tokenizer)
        chunks = []
        for token in output.reshape(-1).tolist():
            if token in (self.config.bos_token_id, self.config.eos_token_id, self.config.pad_token_id):
                continue
            if token in self._detokenizer.special_ids:
                # Preserve reasoning markers instead of filtering all specials.
                chunks.append(tokenizer.convert_ids_to_tokens(token).encode("utf-8"))
            else:
                chunks.append(self._detokenizer.to_bytes([token]))
        return b"".join(chunks)
