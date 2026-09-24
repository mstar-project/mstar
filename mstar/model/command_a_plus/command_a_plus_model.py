"""Single-node text generation with pinned metadata and explicitly supplied weights."""

import json

import torch

from mstar.conductor.request_info import DEFAULT_PARTITION, CurrentForwardConductorMetadata
from mstar.distributed.base import ShardingConfig
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    KVConfig,
    KVSpec,
    PositionConfig,
    PositionSpec,
    SamplerSpec,
    SamplingReqConfig,
)
from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.command_a_plus.assets import resolve_metadata
from mstar.model.command_a_plus.config import (
    GLOBAL_ATTN,
    KV_CACHE,
    LOCAL_ATTN,
    ROPE,
    SAMPLER,
    CommandAPlusConfig,
)


class CommandAPlusModel(Model):
    def __init__(
        self, model_path_hf: str, cache_dir: str | None = None,
        max_seq_len: int | None = None, checkpoint_dir: str | None = None, **kwargs,
    ):
        self.local_dir = resolve_metadata(checkpoint_dir or model_path_hf, cache_dir)
        self.config = CommandAPlusConfig.from_json(self.local_dir / "config.json").text_config
        self.max_seq_len = min(8192, self.config.max_position_embeddings) if max_seq_len is None else max_seq_len
        if type(self.max_seq_len) is not int or not 0 < self.max_seq_len <= self.config.max_position_embeddings:
            raise ValueError("max_seq_len must be positive and within max_position_embeddings")
        self._tokenizer = None
        self._detokenizer = None

    def get_graph_walk_graphs(self):
        return {
            "prefill": GraphNode(
                name="LLM", input_names=["text_inputs"], outputs=[
                    GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token",
                              output_modality="text", conductor_new_token=True),
                    # Absent when prefill already finishes the request.
                    GraphEdge(next_node=EMPTY_DESTINATION, name="decode_input", persist=True),
                ],
            ),
            "decode": Loop(
                name="decode_loop", max_iters=self.get_max_output_tokens(), outputs=[],
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
        edge = GraphEdge(next_node="LLM", name="text_inputs", tensor_info=input_signals["text_inputs"])
        return ForwardPassArgs(
            full_metadata=CurrentForwardConductorMetadata(
                graph_walk="prefill", is_prefill=True,
                input_modalities=input_modalities, output_modalities=output_modalities,
            ),
            inputs=[edge], unpersist_tensors=edge.tensor_info,
        )

    def get_partition_forward_pass_args(
        self, partition_name, partition_metadata, persist_signals, incoming_connections=None,
    ):
        self._check_partition(partition_name)
        metadata = partition_metadata
        seeds = persist_signals.get("decode_input", [])
        if metadata.graph_walk == "decode" or not seeds:
            return ForwardPassArgs(metadata, [], list(seeds), request_done=True)
        if metadata.graph_walk != "prefill":
            raise ValueError(f"Unexpected graph walk: {metadata.graph_walk!r}")
        metadata.graph_walk = "decode"
        metadata.is_prefill = False
        edge = GraphEdge(next_node="LLM", name="text_inputs", tensor_info=seeds)
        return ForwardPassArgs(metadata, [edge], list(seeds))

    def get_node_resources(self):
        config = self.config
        return [
            KVSpec(resource_key=KV_CACHE, nodes={"LLM"}, config=KVConfig(
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

    def get_default_sharding_config(self):
        return ShardingConfig(groups=[], tp_enabled_nodes={"LLM"}, shard_dim={})

    def get_submodule(
        self, node_name, device="cpu", tp_group=None, autocast_dtype=None, sp_group=None,
    ):
        if node_name != "LLM":
            raise ValueError(f"Unknown Command A+ node: {node_name!r}")
        if sp_group is not None and sp_group.world_size != 1:
            raise ValueError("Command A+ sequence parallelism is not implemented")
        from mstar.model.command_a_plus.components.language_model import CommandAPlusForCausalLM
        from mstar.model.command_a_plus.submodules import CommandAPlusLLMSubmodule
        from mstar.model.loader import iter_safetensors_shards

        # Validate availability before allocating hundreds of GB of parameters.
        index = self.local_dir / "model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            required = {shard for name, shard in weight_map.items()
                        if name.startswith("model.language_model.")}
        else:
            required = {"model.safetensors"}
        missing = [name for name in sorted(required) if not (self.local_dir / name).is_file()]
        if not required or missing:
            raise FileNotFoundError(
                "Command A+ weights are not present locally; obtain the checkpoint explicitly "
                f"before starting workers. Missing shards: {missing[:3]}"
            )

        dtype = autocast_dtype if autocast_dtype is not None else torch.float32
        with torch.device("meta"):
            language_model = CommandAPlusForCausalLM(self.config, tp_group).to(dtype=dtype)
        language_model.to_empty(device=device)
        # Stream source tensors through CPU; loaders copy just this rank's slice.
        language_model.load_weights(iter_safetensors_shards(
            self.local_dir, device="cpu", prefix="model.language_model.",
        ))
        language_model.eval()
        return CommandAPlusLLMSubmodule(language_model, self.config).eval()

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.local_dir, local_files_only=True)
        return self._tokenizer

    def process_prompt(
        self, prompt, input_modalities, output_modalities, tensors=None, prompt_parts=None, **kwargs,
    ):
        if any(modality != "text" for modality in input_modalities) or output_modalities != ["text"]:
            raise ValueError("Command A+ currently supports text input and output only")
        if prompt_parts is not None:
            if any(part.modality != "text" for part in prompt_parts):
                raise ValueError("Command A+ currently supports text input only")
            prompt = "".join(part.text or "" for part in prompt_parts)
        if tensors and "text_inputs" in tensors:
            ids = tensors["text_inputs"]
            if len(ids) != 1:
                raise ValueError("Expected one prompt token tensor")
            ids = ids[0]
        else:
            if not isinstance(prompt, str):
                raise ValueError("Expected a text prompt")
            ids = torch.tensor(self._get_tokenizer().apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True, return_dict=False,
            ), dtype=torch.long)
        if ids.ndim != 1 or ids.numel() == 0 or ids.numel() > self.max_seq_len:
            raise ValueError("Prompt must contain 1..max_seq_len token IDs")
        if ids.dtype not in (torch.int32, torch.int64) or bool(((ids < 0) | (ids >= self.config.vocab_size)).any()):
            raise ValueError("Prompt token IDs must be integers within the vocabulary")
        return {"text_inputs": [ids.to(dtype=torch.long)]}

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
