"""M* model wrapper for the Kimi-K2.7 text backbone."""
from __future__ import annotations

import logging
from dataclasses import replace

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    AttnBackend,
    KVConfig,
    KVLayout,
    KVSpec,
    NodeResourceSpec,
    PositionConfig,
    PositionSpec,
    ResourceReqConfig,
    SamplerSpec,
    SamplingReqConfig,
)
from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    GraphSection,
    Loop,
    Sequential,
    TensorPointerInfo,
)
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.kimi_k2_7.config import ATTN, KV_CACHE, ROPE, SAMPLER, KimiK2Config
from mstar.model.multimodal import find_media_spans, split_around_spans
from mstar.model.submodule_base import NodeSubmodule

logger = logging.getLogger(__name__)

LLM_NODE = "LLM"
VISION_NODE = "vision_encoder"
DECODE_LOOP = "decode_loop"


def _resolve_local_hf_snapshot(repo_id: str, cache_dir: str | None = None) -> str:
    from pathlib import Path

    from huggingface_hub import snapshot_download

    try:
        local_dir = snapshot_download(
            repo_id=repo_id, cache_dir=cache_dir, local_files_only=False,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Error downloading %r from huggingface: %s", repo_id, e)
        return repo_id
    return str(Path(local_dir))


def _resolve_checkpoint_config_dir(path: str, cache_dir: str | None = None) -> str | None:
    """Local directory holding ``config.json`` (and, if present,
    ``generation_config.json`` / ``preprocessor_config.json``) for ``path``,
    without pulling the full (multi-hundred-GB) checkpoint snapshot just to
    read a few small JSON files."""
    from pathlib import Path

    if not path:
        return None
    if Path(path).is_dir():
        return str(path)

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        config_path = hf_hub_download(
            repo_id=path, filename="config.json", cache_dir=cache_dir,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Error downloading config.json for %r from huggingface: %s", path, e,
        )
        return None
    for optional_file in ("generation_config.json", "preprocessor_config.json"):
        try:
            hf_hub_download(
                repo_id=path, filename=optional_file, cache_dir=cache_dir,
            )
        except EntryNotFoundError:
            pass  # optional file — not every checkpoint ships one
    return str(Path(config_path).parent)


class KimiK2Model(Model):
    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        **kwargs,
    ):
        self.cache_dir = cache_dir
        # A local checkpoint directory or an HF repo id. Deployments point this at
        # their own copy with ``mstar serve --model-path`` / ``mstar-serve
        # --model-path`` rather than hardcoding a path in the config yaml.
        self.model_path_hf = model_path_hf
        self._config_variant = kwargs.get("config_variant", "full")
        if self._config_variant == "reduced":
            self.config = KimiK2Config.reduced()
        elif self._config_variant == "reduced_quantized":
            self.config = KimiK2Config.reduced_quantized()
        elif self._config_variant == "reduced_quantized_inkernel":
            self.config = KimiK2Config.reduced_quantized_inkernel()
        elif self._config_variant == "reduced_marlin":
            self.config = KimiK2Config.reduced_marlin()
        elif self._config_variant == "k27_code":
            self.config = KimiK2Config.k27_code()
        else:
            self.config = KimiK2Config()
        if self._config_variant in ("full", "k27_code"):
            # Every process that builds this model (API server, conductor,
            # worker) must agree on the checkpoint's config before anything
            # reads it (e.g. get_node_resources() for the KV cache shape),
            # so this runs here rather than lazily in _create_submodule.
            config_dir = _resolve_checkpoint_config_dir(model_path_hf, cache_dir=cache_dir)
            if config_dir is not None:
                self.config = KimiK2Config.from_checkpoint(config_dir, base=self.config)
        if not kwargs.get("vision", True):
            # Text-only deployment: drops the vision tower and the
            # ``prefill_vision`` walk; the chat path then treats image parts
            # as unsupported.
            self.config = replace(self.config, vision=None)
        self._tokenizer_mode = kwargs.get("tokenizer_mode", "hf")
        self._tokenizer = None
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path_hf,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
        return self._tokenizer

    def _kv_and_attn_specs(self) -> list[NodeResourceSpec]:
        """The cache's shape and the backend over it, which Kimi picks
        together: absorbed MLA caches one latent per token, naive full K/V."""
        cfg = self.config
        if not cfg.mla_absorb:
            return [
                KVConfig(
                    num_layers=cfg.num_hidden_layers,
                    num_kv_heads=cfg.num_attention_heads,
                    head_dim=cfg.padded_head_dim,
                    max_seq_len=cfg.max_position_embeddings,
                    num_qo_heads=cfg.num_attention_heads,
                ),
                AttentionConfig(kv_cache=KV_CACHE),
            ]

        from mstar.model.kimi_k2_7.components.rope import yarn_get_mscale

        rope = cfg.rope_scaling
        mscale = yarn_get_mscale(rope["factor"], rope.get("mscale_all_dim", 0.0))
        return [
            KVConfig(
                num_layers=cfg.num_hidden_layers,
                num_kv_heads=1,
                head_dim=cfg.kv_lora_rank + cfg.qk_rope_head_dim,
                max_seq_len=cfg.max_position_embeddings,
                num_qo_heads=cfg.num_attention_heads,
                layout=KVLayout.MLA,
            ),
            AttentionConfig(
                kv_cache=KV_CACHE,
                backend=AttnBackend.MLA,
                # MLA scales by the unabsorbed qk_head_dim, which the latent
                # width does not carry
                softmax_scale=cfg.qk_head_dim ** -0.5 * mscale * mscale,
                mla_ckv_dim=cfg.kv_lora_rank,
            ),
        ]

    def get_node_resources(self) -> list[NodeResourceSpec]:
        kv_config, attn_config = self._kv_and_attn_specs()
        return [
            KVSpec(resource_key=KV_CACHE, nodes={LLM_NODE}, config=kv_config),
            AttentionSpec(resource_key=ATTN, nodes={LLM_NODE}, config=attn_config),
            SamplerSpec(
                resource_key=SAMPLER,
                nodes={LLM_NODE},
                vocab_size=self.config.vocab_size,
                enable_repetion_penalty=True,
            ),
            # Kimi applies its own DeepSeek-yarn rotary in the layer, so this
            # only tracks the position counters; the rope params here are
            # unused (see `KimiMLAAttention.position_ids`).
            PositionSpec(
                resource_key=ROPE,
                nodes={LLM_NODE},
                config=PositionConfig(
                    kv_cache=KV_CACHE,
                    rotary_dim=self.config.qk_rope_head_dim,
                    rope_theta=self.config.rope_theta,
                ),
            ),
        ]

    def get_request_resource_configs(
        self, partition_fwd_args: dict[str, ForwardPassArgs],
        model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        del partition_fwd_args
        model_kwargs = model_kwargs or {}
        return {
            SAMPLER: SamplingReqConfig(
                **{
                    key: model_kwargs.get(key, getattr(self.config, key))
                    for key in (
                        "temperature", "top_p", "repetition_penalty", "ignore_eos",
                    )
                }
            )
        }

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill = GraphNode(
            name=LLM_NODE,
            input_names=["text_inputs"],
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="new_token",
                    output_modality="text",
                    conductor_new_token=True,
                    persist=True,
                ),
            ],
        )

        decode = Loop(
            name=DECODE_LOOP,
            section=GraphNode(
                name=LLM_NODE,
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                        conductor_new_token=True,
                    ),
                    GraphEdge(
                        next_node=LLM_NODE,
                        name="text_inputs",
                    ),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )

        walks = dict(prefill=prefill, decode=decode)
        if self.config.vision is not None:
            walks["prefill_vision"] = Sequential([
                GraphNode(
                    name=VISION_NODE,
                    input_names=["image_inputs", "image_grids"],
                    outputs=[GraphEdge(next_node=LLM_NODE, name="image_embeds")],
                ),
                GraphNode(
                    name=LLM_NODE,
                    input_names=["image_embeds"],
                    outputs=[
                        GraphEdge(
                            next_node=EMIT_TO_CLIENT,
                            name="new_token",
                            output_modality="text",
                            conductor_new_token=True,
                            persist=True,
                        ),
                    ],
                ),
            ])
        return walks

    def _prefill_schedule_from_signals(
        self, input_signals: dict[str, list[TensorPointerInfo]],
    ) -> list[tuple[str, dict[str, TensorPointerInfo]]]:
        """One ``prefill`` step per text span, one ``prefill_vision`` step per
        image, interleaved the way the chat template wrote them: text, image,
        text, image, ..., text. Text-only requests get a single-entry, single-
        step schedule."""
        texts = input_signals.get("text_inputs", [])
        images = input_signals.get("image_inputs", [])
        grids = input_signals.get("image_grids", [])
        schedule: list[tuple[str, dict[str, TensorPointerInfo]]] = []
        for i in range(len(images)):
            schedule.append(("prefill", {"text_inputs": texts[i]}))
            schedule.append(
                ("prefill_vision", {"image_inputs": images[i], "image_grids": grids[i]})
            )
        if images:
            schedule.append(("prefill", {"text_inputs": texts[len(images)]}))
        else:
            # No images: reproduce the pre-vision edge exactly, the whole
            # (possibly empty) list, since dummy/fallback signals may omit
            # ``text_inputs`` entirely.
            schedule.append(("prefill", {"text_inputs": texts}))
        return schedule

    def _prefill_step_inputs(
        self, entry: tuple[str, dict[str, TensorPointerInfo]],
    ) -> list[GraphEdge]:
        walk_name, tensor_dict = entry
        target_node = VISION_NODE if walk_name == "prefill_vision" else LLM_NODE
        inputs = []
        for name, info in tensor_dict.items():
            edge = GraphEdge(next_node=target_node, name=name)
            edge.tensor_info = info if isinstance(info, list) else [info]
            inputs.append(edge)
        return inputs

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        schedule = self._prefill_schedule_from_signals(input_signals)
        first_walk = schedule[0][0]
        inputs = self._prefill_step_inputs(schedule[0])
        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])

        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=first_walk,
            is_prefill=True,
            kwargs={"prefill_schedule": schedule, "prefill_step": 0},
        )

        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={
                "is_prefill": True,
                "sample_prefill_token": len(schedule) == 1,
            },
        )

    def _advance_prefill_schedule(
        self,
        metadata: CurrentForwardConductorMetadata,
        schedule: list[tuple[str, dict[str, TensorPointerInfo]]],
        persist_signals: dict[str, list[TensorPointerInfo]],
    ) -> ForwardPassArgs:
        step = metadata.kwargs["prefill_step"] + 1
        if step < len(schedule):
            metadata.kwargs["prefill_step"] = step
            metadata.graph_walk = schedule[step][0]
            inputs = self._prefill_step_inputs(schedule[step])
            sample_prefill_token = step == len(schedule) - 1
        else:
            metadata.is_prefill = False
            metadata.graph_walk = "decode"
            graph_edge = GraphEdge(next_node=LLM_NODE, name="text_inputs")
            graph_edge.tensor_info = persist_signals.get("new_token", [])
            inputs = [graph_edge]
            sample_prefill_token = True

        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={
                "is_prefill": metadata.is_prefill,
                "sample_prefill_token": sample_prefill_token,
            },
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        metadata = partition_metadata
        schedule = metadata.kwargs.get("prefill_schedule")
        if metadata.is_prefill and schedule is not None:
            return self._advance_prefill_schedule(metadata, schedule, persist_signals)

        request_done = False
        if metadata.is_prefill:
            metadata.is_prefill = False
            metadata.graph_walk = "decode"
        elif metadata.graph_walk == "decode":
            request_done = True
            metadata.kwargs["decode_finished"] = True

        if request_done:
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True,
            )

        graph_edge = GraphEdge(next_node=LLM_NODE, name="text_inputs")
        graph_edge.tensor_info = persist_signals.get("new_token", [])
        inputs = [graph_edge]
        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])

        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={"is_prefill": metadata.is_prefill},
        )

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> NameToTensorList:
        messages = kwargs.pop("messages", None)
        raw_images = (tensors or {}).get("image_inputs") or []
        if raw_images:
            if messages is None:
                raise ValueError(
                    "Kimi-K2.7: image input requires the chat-template "
                    "(messages) path."
                )
            return self._process_image_messages(messages, raw_images, **kwargs)
        if messages is not None:
            if self._tokenizer_mode == "byte":
                raise ValueError(
                    "Kimi-K2.7 byte tokenizer_mode is synthetic-only and does "
                    "not support chat-template messages."
                )
            tools = kwargs.get("tools")
            if kwargs.get("tool_choice") == "none":
                tools = None
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                **(kwargs.get("chat_template_kwargs") or {}),
            )
            # transformers versions differ: a tensor/list of ids directly, or a
            # dict/BatchEncoding carrying "input_ids".
            if isinstance(encoded, (torch.Tensor, list, tuple)):
                input_ids = encoded
            else:
                input_ids = encoded["input_ids"]
            input_ids = torch.as_tensor(input_ids).reshape(-1).long()
            return {"text_inputs": [input_ids]}
        if prompt is None:
            return {}
        if self._tokenizer_mode == "byte":
            # Reduced serve maps UTF-8 bytes directly to token ids, avoiding HF IO.
            vocab = self.config.vocab_size
            byte_ids = [min(b, vocab - 1) for b in prompt.encode("utf-8")] or [0]
            input_ids = torch.tensor(byte_ids, dtype=torch.long)
            return {"text_inputs": [input_ids]}
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
        return {"text_inputs": [input_ids]}

    def _process_image_messages(
        self, messages: list, raw_images: list[torch.Tensor], **kwargs,
    ) -> NameToTensorList:
        """Render the chat template's own ``image`` marker, then splice each
        attachment's patches in for the single ``<|media_pad|>`` token the
        template leaves per image."""
        from mstar.model.kimi_k2_7.components.vision import preprocess_image

        vision_cfg = self.config.vision
        tools = kwargs.get("tools")
        if kwargs.get("tool_choice") == "none":
            tools = None
        text = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            **(kwargs.get("chat_template_kwargs") or {}),
        )
        token_ids = self.tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0].tolist()

        patches, grids = [], []
        for image in raw_images:
            image_patches, gh, gw = preprocess_image(image, vision_cfg)
            patches.append(image_patches)
            grids.append(torch.tensor([gh, gw]))

        pad_id, merge = vision_cfg.media_pad_token_id, vision_cfg.merge_kernel_size
        expanded, image_idx = [], 0
        for token_id in token_ids:
            if token_id == pad_id:
                gh, gw = grids[image_idx].tolist()
                expanded.extend([pad_id] * ((gh // merge) * (gw // merge)))
                image_idx += 1
            else:
                expanded.append(token_id)
        input_ids = torch.tensor(expanded, dtype=torch.long)

        specs = {
            "image": (
                vision_cfg.media_content_token_id, pad_id, vision_cfg.media_end_token_id,
            )
        }
        spans = find_media_spans(input_ids, specs)
        segments = split_around_spans(input_ids, spans)
        if len(segments) != len(raw_images) + 1:
            raise ValueError(
                f"Kimi-K2.7: expected {len(raw_images) + 1} text segments around "
                f"{len(raw_images)} image(s), found {len(segments)}"
            )
        return {"text_inputs": segments, "image_inputs": patches, "image_grids": grids}

    def postprocess(
        self,
        output: torch.Tensor,
        modality: str,
        request_kwargs: dict | None = None,
    ) -> bytes:
        if modality == "text":
            token_ids = output.tolist() if output.numel() else []
            if self._tokenizer_mode == "byte":
                # Synthetic reduced models emit arbitrary byte ids; return raw bytes.
                return bytes((t & 0xFF) for t in token_ids)
            # eos ids are dropped; a kwarg-less decode routes to tiktoken's Rust decoder (renders
            # <think>/tool-call markers verbatim), while any kwarg falls back to HF's slow per-id
            # path (~17x slower, adds spaces around special tokens).
            new_ids = [t for t in token_ids if t not in self.config.eos_token_ids]
            text = self.tokenizer.decode(new_ids)
            return text.encode("utf-8")
        raise ValueError(f"Unsupported modality for Kimi-K2.7: {modality!r}")

    def get_default_sharding_config(self):
        from mstar.distributed.base import ShardingConfig

        return ShardingConfig(groups=[], tp_enabled_nodes={LLM_NODE}, shard_dim={})

    def get_submodule(
        self,
        node_name: str,
        device: str = "cpu",
        tp_group=None,
        autocast_dtype: torch.dtype | None = None,
        sp_group=None,
    ) -> NodeSubmodule | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(
            node_name, device, tp_group=tp_group, autocast_dtype=autocast_dtype,
        )
        self._submodule_cache[node_name] = submodule
        return submodule

    def _create_submodule(
        self,
        node_name: str,
        device: str,
        tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name == VISION_NODE:
            return self._create_vision_submodule(device)
        if node_name != LLM_NODE:
            return None

        source = self._resolve_checkpoint()
        if source is None:
            logger.info(
                "KimiK2Model: no checkpoint resolved for node %r — dummy mode (None).",
                node_name,
            )
            return None

        # self.config is already final (resolved in __init__ for full/k27_code);
        # this is a no-op for those variants and unchanged behavior for reduced*.
        self._maybe_apply_checkpoint_quant_config(source)

        from mstar.model.kimi_k2_7.components.causal_lm import KimiForCausalLM
        from mstar.model.kimi_k2_7.submodules import KimiLLMSubmodule
        from mstar.model.loader import load_weights

        with torch.device("meta"):
            language_model = KimiForCausalLM(self.config, comm_group=tp_group)
        if autocast_dtype is not None:
            language_model = language_model.to(autocast_dtype)
        language_model.to_empty(device=device)
        loaded = load_weights(language_model, source, device=device)
        expected = set(dict(language_model.named_parameters()).keys())
        missing = sorted(expected - loaded)
        if missing:
            shown = missing[:20]
            more = f" (+{len(missing) - 20} more)" if len(missing) > 20 else ""
            raise RuntimeError(
                f"KimiK2Model: {len(missing)} parameter(s) were not loaded from "
                f"checkpoint {source!r}: {shown}{more}"
            )
        from mstar.model.components.quantization import process_weights_after_loading

        process_weights_after_loading(language_model, torch.device(device))
        language_model.eval()

        logger.info("Successfully loaded Kimi-K2.7 submodule for %s", node_name)
        return KimiLLMSubmodule(language_model=language_model, config=self.config)

    def _create_vision_submodule(self, device: str) -> NodeSubmodule | None:
        source = self._resolve_checkpoint()
        if source is None:
            logger.info(
                "KimiK2Model: no checkpoint resolved for node %r — dummy mode (None).",
                VISION_NODE,
            )
            return None

        from mstar.model.kimi_k2_7.components.vision import KimiMMProjector, KimiVisionTower
        from mstar.model.kimi_k2_7.submodules import KimiVisionEncoderSubmodule
        from mstar.model.utils import ModuleAndPrefix, load_weights_from_hf_shards

        with torch.device("meta"):
            vision_tower = KimiVisionTower(self.config.vision)
            mm_projector = KimiMMProjector(self.config.vision)
        vision_tower.to_empty(device=device)
        mm_projector.to_empty(device=device)
        load_weights_from_hf_shards(
            repo_dir=source,
            modules=[
                ModuleAndPrefix(vision_tower, prefix="vision_tower"),
                ModuleAndPrefix(mm_projector, prefix="mm_projector"),
            ],
            device=device,
        )
        vision_tower.eval()
        mm_projector.eval()

        logger.info("Successfully loaded Kimi-K2.7 submodule for %s", VISION_NODE)
        return KimiVisionEncoderSubmodule(vision_tower=vision_tower, mm_projector=mm_projector)

    def _resolve_checkpoint(self) -> str | None:
        from pathlib import Path

        path = getattr(self, "model_path_hf", None)
        if not path:
            return None
        if Path(path).exists():
            return str(path)
        return _resolve_local_hf_snapshot(path, cache_dir=getattr(self, "cache_dir", None))

    def _maybe_apply_checkpoint_quant_config(self, source: str) -> None:
        import json
        from pathlib import Path

        from mstar.model.components.quantization import CompressedTensorsQuantConfig
        from mstar.model.kimi_k2_7.config import _quant_raw_from_config_dict

        if self.config.quantization_config is not None:
            return
        config_json = Path(source) / "config.json"
        if not config_json.is_file():
            return
        try:
            with open(config_json) as f:
                raw = json.load(f)
        except (OSError, ValueError) as e:  # unreadable / malformed — stay bf16
            logger.warning("KimiK2Model: could not read %s: %s", config_json, e)
            return
        quant = CompressedTensorsQuantConfig.from_hf_config_dict(
            _quant_raw_from_config_dict(raw)
        )
        if quant is not None:
            logger.info(
                "KimiK2Model: compressed-tensors checkpoint (%d-bit, group_size=%d) "
                "— dequantizing on load.", quant.num_bits, quant.group_size,
            )
            self.config.quantization_config = quant
