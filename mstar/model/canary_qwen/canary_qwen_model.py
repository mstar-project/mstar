"""
CanaryQwenModel: NVIDIA Canary-Qwen-2.5B, English speech recognition by a
speech-augmented LLM (NeMo's SALM).

A FastConformer encoder hears the clip once and a linear layer maps its
frames into Qwen3-1.7B's embedding space; they take the place of the audio
placeholder in a chat prompt ("Transcribe the following: <audio>") and the LLM
writes the transcript. The checkpoint's LoRA on the LLM's query and value
projections is merged into them at load.

Architecture (2 nodes, single partition):
    audio_encoder — log-mel + FastConformer + projection, one-shot at prefill
    llm           — Qwen3 with a paged KV cache

Graph walks:
    prefill — audio_encoder -> llm (the prompt with the audio spliced in)
    decode  — llm loop
"""

import json
import logging
import re
from collections.abc import Iterable, Iterator

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardConductorMetadata, StreamingConnectionState
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    KVSpec,
    NodeResourceSpec,
    PagedKVConfig,
    PositionConfig,
    PositionSpec,
    ResourceReqConfig,
    SamplerSpec,
    SamplingReqConfig,
)
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, KVLabel, LabelLayout, Model, ProcessPromptOutput, Span, TensorAndMetadata
from mstar.model.canary_qwen.config import (
    ATTN,
    DECODE_LOOP,
    DECODE_WALK,
    KV_CACHE,
    PREFILL_WALK,
    ROPE,
    SAMPLER,
    CanaryQwenModelConfig,
)
from mstar.model.components.audio_features import HOP_LENGTH, SAMPLE_RATE, clip_samples, load_speech_file
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import check_kv_pools_fit, check_loaded, kv_pages
from mstar.utils.hf_snapshot import resolve_snapshot_dir

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTION = "Transcribe the following:"
_LORA = re.compile(r"^layers\.(\d+)\.self_attn\.([qv])_proj\.(base_layer|lora_A\.default|lora_B\.default)\.weight$")


def merge_lora(
    weights: Iterable[tuple[str, torch.Tensor]], scale: float,
) -> Iterator[tuple[str, torch.Tensor]]:
    """PEFT-named LLM weights (``layers.N.self_attn.q_proj.base_layer`` /
    ``lora_A.default`` / ``lora_B.default``) -> HF names with each adapter
    merged, ``W + scale * B @ A`` in fp32; everything else passes through."""
    pending: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for name, tensor in weights:
        match = _LORA.match(name)
        if match is None:
            yield name, tensor
            continue
        layer, proj, part = match.groups()
        parts = pending.setdefault((layer, proj), {})
        parts[part.split(".")[0]] = tensor
        if len(parts) == 3:
            merged = parts["base_layer"].float() + scale * (parts["lora_B"].float() @ parts["lora_A"].float())
            yield f"layers.{layer}.self_attn.{proj}_proj.weight", merged
            del pending[(layer, proj)]
    if pending:
        raise KeyError(f"incomplete LoRA weights for {sorted(pending)[:3]}")


class CanaryQwenModel(Model):
    """Canary-Qwen speech recognition: FastConformer + projection feeding a
    LoRA-adapted Qwen3-1.7B."""

    # trained on clips of up to 40 s, the transcription route's window; the
    # slack covers containers that decode longer than their header said, and
    # stays inside the largest encoder graph (4096 mel frames, 512 audio tokens)
    MAX_AUDIO_SECONDS = 40.96
    # a few tiny ops per upload; see "Preprocessing threads" in docs/adding_models.rst
    PREPROCESS_TORCH_THREADS = 1
    MAX_SEQ_LEN = 2048
    # the requests the default cache holds at full length; configs/canary_qwen.yaml
    # caps max_concurrent_requests here, so the conductor queues the rest
    MAX_CONCURRENT_REQUESTS = 64
    MAX_OUTPUT_TOKENS = 1024
    # the chat template and instruction: with 512 audio tokens and a full
    # transcript, a request stays inside MAX_SEQ_LEN
    MAX_TEXT_TOKENS = 512

    def __init__(self, model_path_hf: str, cache_dir: str | None = None, **kwargs):
        from transformers import AutoTokenizer

        from mstar.model.utils import ByteLevelDetokenizer

        self.model_path_hf = model_path_hf
        self.local_dir = resolve_snapshot_dir(model_path_hf, cache_dir, allow_patterns=["*.json", "*.safetensors"])
        llm_repo = json.loads((self.local_dir / "config.json").read_text())["pretrained_llm"]
        self.llm_dir = resolve_snapshot_dir(llm_repo, cache_dir, allow_patterns=["*.json", "*.txt"])
        self.config = CanaryQwenModelConfig.from_pretrained(self.local_dir, self.llm_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_dir)
        # NeMo registers the placeholder as a special token at load, as here
        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.config.audio_locator_tag]})
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.config.audio_locator_tag)
        self.stop_token_ids = {self.tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
        self._detokenizer = ByteLevelDetokenizer(self.tokenizer)
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    def checkpoint_path(self) -> str | None:
        return str(self.local_dir)

    # -------------------------------------------------------------------
    # resources
    # -------------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        llm = self.config.llm
        kv_config = PagedKVConfig(
            num_layers=llm.num_hidden_layers,
            num_kv_heads=llm.num_key_value_heads,
            head_dim=llm.head_dim,
            max_seq_len=self.MAX_SEQ_LEN,
            num_qo_heads=llm.num_attention_heads,
            page_size=128,
            # the request cap at MAX_SEQ_LEN, 16 pages each; at Qwen3-1.7B's
            # 112 KiB per token in bf16 that is about 15 GB
            max_num_pages=kv_pages(self.MAX_SEQ_LEN, 128, self.MAX_CONCURRENT_REQUESTS),
        )
        return [
            KVSpec(resource_key=KV_CACHE, nodes={"llm"}, config=kv_config),
            AttentionSpec(resource_key=ATTN, nodes={"llm"}, config=AttentionConfig(kv_cache=KV_CACHE)),
            PositionSpec(
                resource_key=ROPE, nodes={"llm"},
                config=PositionConfig(kv_cache=KV_CACHE, rope_theta=llm.rope_theta),
            ),
            SamplerSpec(resource_key=SAMPLER, nodes={"llm"}, vocab_size=llm.vocab_size, enable_repetion_penalty=False),
        ]

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        check_kv_pools_fit(self, config, config_path, logger)

    def get_request_resource_configs(
        self, partition_fwd_args: dict[str, ForwardPassArgs],
        model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        del partition_fwd_args
        model_kwargs = model_kwargs or {}
        return {
            SAMPLER: SamplingReqConfig(
                # NeMo decodes greedily
                temperature=model_kwargs.get("temperature", 0.0),
                top_p=model_kwargs.get("top_p", 1.0),
                ignore_eos=model_kwargs.get("ignore_eos", False),
            )
        }

    # -------------------------------------------------------------------
    # graph walks
    # -------------------------------------------------------------------

    def get_max_output_tokens(self, **model_kwargs):
        requested = model_kwargs.get("max_output_tokens")
        return self.MAX_OUTPUT_TOKENS if requested is None else min(requested, self.MAX_OUTPUT_TOKENS)

    def get_max_output_tokens_limit(self) -> int:
        return self.MAX_OUTPUT_TOKENS

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        # the prefill samples the first token, so decode runs one fewer
        return {DECODE_LOOP: max(1, self.get_max_output_tokens(**model_kwargs) - 1)}

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill = Sequential([
            GraphNode(
                name="audio_encoder",
                input_names=["audio"],
                outputs=[GraphEdge(next_node="llm", name="encoder_states")],
            ),
            GraphNode(
                name="llm",
                input_names=["encoder_states", "text_inputs"],
                outputs=[
                    # counted, so a one-token request ends here: a loop runs at least once
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text", persist=True,
                        conductor_new_token=True,
                    ),
                ],
            ),
        ])
        decode = Loop(
            name=DECODE_LOOP,
            section=GraphNode(
                name="llm",
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text"),
                    GraphEdge(next_node="llm", name="text_inputs"),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )
        return {PREFILL_WALK: prefill, DECODE_WALK: decode}

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=PREFILL_WALK,
            is_prefill=True,
        )
        audio = GraphEdge(next_node="audio_encoder", name="audio")
        audio.tensor_info = input_signals.get("audio", [])
        text = GraphEdge(next_node="llm", name="text_inputs")
        text.tensor_info = input_signals.get("text_inputs", [])
        inputs = [audio, text]
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=sum([inp.tensor_info for inp in inputs], start=[]),
            step_metadata={"is_prefill": True},
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        """prefill -> decode loop -> done."""
        metadata = partition_metadata
        if metadata.is_prefill:
            metadata.is_prefill = False
            metadata.graph_walk = DECODE_WALK
        elif metadata.graph_walk == DECODE_WALK:
            return ForwardPassArgs(full_metadata=metadata, inputs=[], unpersist_tensors=[], request_done=True)
        edge = GraphEdge(next_node="llm", name="text_inputs")
        edge.tensor_info = persist_signals.get("new_token", [])
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=[edge],
            unpersist_tensors=list(edge.tensor_info),
            step_metadata={"is_prefill": False},
        )

    # -------------------------------------------------------------------
    # prompt / output
    # -------------------------------------------------------------------

    def prompt_ids(self, num_audio_frames: int, instruction: str = DEFAULT_INSTRUCTION) -> list[int]:
        """NeMo's ``qwen`` prompt format for one user turn and the assistant
        prefix, the placeholder repeated once per audio frame."""
        text = (
            f"<|im_start|>user\n{instruction} {self.config.audio_locator_tag}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if ids.count(self.audio_token_id) != 1:
            raise ValueError("the instruction must not contain the audio placeholder")
        if len(ids) > self.MAX_TEXT_TOKENS:
            raise ValueError(f"the prompt is {len(ids)} tokens with its template; the limit is {self.MAX_TEXT_TOKENS}")
        at = ids.index(self.audio_token_id)
        return ids[:at] + [self.audio_token_id] * num_audio_frames + ids[at + 1:]

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> ProcessPromptOutput:
        """The clip's samples and the chat prompt. The text ``prompt`` is
        unused: the ASR harnesses send a generic one, which would replace the
        trained instruction. ``instruction`` replaces it on purpose (default
        "Transcribe the following:")."""
        samples = clip_samples(tensors, self.MAX_AUDIO_SECONDS, "Canary-Qwen")
        instruction = kwargs.get("instruction") or DEFAULT_INSTRUCTION
        if not isinstance(instruction, str):
            raise ValueError(f"instruction must be a string; got {instruction!r}")
        frames = self.config.encoder.output_lengths(samples.shape[0] // HOP_LENGTH)
        ids = self.prompt_ids(frames, instruction.strip() or DEFAULT_INSTRUCTION)
        # rows, not ids: the audio placeholders are the same for every clip of a length
        kv_layout = {KV_CACHE: {"main": LabelLayout(
            spans=(Span("rows", PREFILL_WALK, len(ids), len(ids), "text_inputs"),),
            growth=((DECODE_LOOP, 1),),
        )}}
        return ProcessPromptOutput(
            {"audio": [samples], "text_inputs": [torch.tensor(ids, dtype=torch.long)]},
            {"kv_layout": kv_layout},
        )

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        return {KV_CACHE: {"main": KVLabel(growth=((DECODE_WALK, DECODE_LOOP),))}}

    def max_input_files(self) -> dict[str, int]:
        return {"audio": 1}

    def load_audio(self, filepath: str, device: str) -> TensorAndMetadata:
        audio = load_speech_file(filepath, SAMPLE_RATE, self.MAX_AUDIO_SECONDS)
        return TensorAndMetadata(data=audio, metadata=dict(sample_rate=SAMPLE_RATE, num_channels=1))

    def postprocess(self, output: torch.Tensor, modality: str, **kwargs) -> bytes:
        if modality == "text":
            return self._detokenizer.to_bytes(output.reshape(-1).tolist())
        raise ValueError(f"Unsupported modality for Canary-Qwen: {modality!r}")

    # -------------------------------------------------------------------
    # submodules
    # -------------------------------------------------------------------

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name not in self._submodule_cache:
            self._submodule_cache[node_name] = self._create_submodule(node_name, device, autocast_dtype)
        return self._submodule_cache[node_name]

    def _weights(self, device: str, prefix: str, strip: bool = True) -> Iterator[tuple[str, torch.Tensor]]:
        from mstar.model.loader.iterators import iter_safetensors_shards

        for name, tensor in iter_safetensors_shards(self.local_dir, device=device, prefix=prefix):
            yield (name.removeprefix(prefix) if strip else name), tensor

    def _llm_weights(self, device: str) -> Iterator[tuple[str, torch.Tensor]]:
        # NeMo keeps the token embedding outside the LLM
        yield from self._weights(device, "embed_tokens.", strip=False)
        yield from merge_lora(self._weights(device, "llm.base_model.model.model."), self.config.lora_scale)

    def _create_submodule(self, node_name: str, device: str, autocast_dtype: torch.dtype | None):
        if node_name == "audio_encoder":
            from mstar.model.components.audio_features import NemoLogMel
            from mstar.model.components.fastconformer import FastConformerEncoder, nemo_to_hf_name
            from mstar.model.components.fastconformer_submodule import FastConformerEncoderSubmodule

            with torch.device("meta"):
                encoder = FastConformerEncoder(self.config.encoder)
                proj = torch.nn.Linear(self.config.encoder.hidden_size, self.config.llm.hidden_size)
            if autocast_dtype is not None:
                encoder, proj = encoder.to(autocast_dtype), proj.to(autocast_dtype)
            encoder, proj = encoder.to_empty(device=device), proj.to_empty(device=device)
            loaded = encoder.load_hf_weights(
                (nemo_to_hf_name(k), v) for k, v in self._weights(device, "perception.encoder.")
            )
            check_loaded(encoder, loaded, "Canary-Qwen encoder")
            featurizer = dict(self._weights(device, "perception.preprocessor.featurizer."))
            head = dict(self._weights(device, "perception.proj."))
            with torch.no_grad():
                proj.weight.copy_(head["weight"])
                proj.bias.copy_(head["bias"])
            # the filters NeMo's preprocessor runs with: the checkpoint's copies
            log_mel = NemoLogMel(
                num_mel_bins=self.config.encoder.num_mel_bins, device=device,
                mel_filters=featurizer["fb"], window=featurizer["window"],
            )
            return FastConformerEncoderSubmodule(encoder.eval(), projection=proj.eval(), log_mel=log_mel)
        if node_name == "llm":
            from mstar.model.canary_qwen.submodules import CanaryQwenLLMSubmodule
            from mstar.model.components.qwen3_lm import Qwen3DenseLM

            with torch.device("meta"):
                llm = Qwen3DenseLM(self.config.llm, attn_key=ATTN, kv_key=KV_CACHE, pos_key=ROPE)
            if autocast_dtype is not None:
                llm = llm.to(autocast_dtype)
            llm = llm.to_empty(device=device)
            llm.load_weights(self._llm_weights(device))
            return CanaryQwenLLMSubmodule(llm.eval(), self.audio_token_id, self.stop_token_ids)
        return None
