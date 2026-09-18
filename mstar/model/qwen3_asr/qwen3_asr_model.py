"""
Qwen3ASRModel: speech-to-text LLM (Qwen/Qwen3-ASR-1.7B, Qwen/Qwen3-ASR-0.6B).

An AuT audio encoder turns a log-mel clip of up to 20 minutes into ~13
tokens per second in LLM space; a dense Qwen3 decoder reads them inside a
ChatML prompt and writes ``language {Name}<asr_text>{transcript}``. Forcing
the language pre-fills ``language {Name}<asr_text>`` into the assistant
turn, so the model emits the transcript alone.

Architecture (2 nodes, single partition):
    audio_encoder  — packed AuT, 8 s attention windows through the ragged
                     attention resource (no cache)
    LLM            — dense Qwen3; paged KV cache; prefill captured on packed
                     token buckets, decode per batch size

Graph walks:
    prefill — audio_encoder -> LLM over the whole prompt with the audio
              embeddings spliced over the ``<|audio_pad|>`` placeholders;
              samples the first token
    decode  — LLM loop; each step feeds the sampled token back

Prompt (the reference SDK's chat template, system turn included):
    <|im_start|>system\\n{context}<|im_end|>\\n
    <|im_start|>user\\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\\n
    <|im_start|>assistant\\n[language {Name}<asr_text>]
"""

import logging
from pathlib import Path

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    KVConfig,
    KVSpec,
    NodeResourceSpec,
    PositionConfig,
    PositionSpec,
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    ResourceReqConfig,
    SamplerSpec,
    SamplingReqConfig,
)
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, Model, TensorAndMetadata
from mstar.model.components.audio_features import LogMelSpectrogram, load_audio_file
from mstar.model.qwen3_asr.config import (
    ASR_TEXT_TAG,
    ATTN,
    AUT_ATTN,
    DECODE_LOOP,
    DECODE_WALK,
    ENCODER_NODE,
    KV_CACHE,
    LANGUAGE_PREFIX,
    LLM_NODE,
    PREFILL_WALK,
    ROPE,
    SAMPLER,
    Qwen3ASRModelConfig,
)
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import ByteLevelDetokenizer

logger = logging.getLogger(__name__)


def _resolve_local_hf_snapshot(repo_id: str, cache_dir: str | None = None) -> str:
    from huggingface_hub import snapshot_download

    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=False,
        )
    except Exception as e:
        logger.warning("Error downloading from HuggingFace: %s", str(e))
        return repo_id
    return str(Path(local_dir))


class Qwen3ASRModel(Model):
    """Qwen3-ASR: AuT encoder + dense Qwen3 decoder."""

    # Concurrency the default cache sizing targets: 20 minutes of audio is
    # ~16k tokens, a LibriSpeech utterance ~300, so the pool is sized for a
    # mix (1024 pages of 128 = 131k tokens). Deployments retune under
    # ``resources:`` in their YAML.
    KV_PAGES = 1024
    MAX_WINDOWS_PER_REQUEST = 160  # 20 min / 8 s, rounded up

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        **kwargs,
    ):
        self.cache_dir = cache_dir
        self.model_path_hf = model_path_hf
        self.local_dir = _resolve_local_hf_snapshot(model_path_hf, cache_dir=cache_dir)
        self.config = Qwen3ASRModelConfig.from_pretrained(self.local_dir)

        from transformers import AutoTokenizer

        # The checkpoint's tokenizer.json carries the pre-tokenizer regex bug
        # transformers warns about; the fix flag is what the reference uses.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.local_dir, cache_dir=cache_dir, fix_mistral_regex=True,
        )
        self.log_mel = LogMelSpectrogram(
            num_mel_bins=self.config.num_mel_bins,
            sampling_rate=self.config.sampling_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
        )
        self._detokenizer = ByteLevelDetokenizer(self.tokenizer)
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    # -------------------------------------------------------------------
    # Model ABC: resources
    # -------------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        text = self.config.text
        audio = self.config.audio
        kv_config = KVConfig(
            num_layers=text.num_hidden_layers,
            num_kv_heads=text.num_key_value_heads,
            head_dim=text.head_dim,
            max_seq_len=text.max_position_embeddings,
            num_qo_heads=text.num_attention_heads,
            max_num_pages=self.KV_PAGES,
        )
        return [
            RaggedAttentionSpec(
                resource_key=AUT_ATTN, nodes={ENCODER_NODE},
                config=RaggedAttentionConfig(
                    num_qo_heads=audio.encoder_attention_heads,
                    num_kv_heads=audio.encoder_attention_heads,
                    head_dim=audio.head_dim,
                    # every 8 s window of a request is its own segment
                    max_segments_per_request=self.MAX_WINDOWS_PER_REQUEST,
                ),
            ),
            KVSpec(resource_key=KV_CACHE, nodes={LLM_NODE}, config=kv_config),
            AttentionSpec(
                resource_key=ATTN, nodes={LLM_NODE},
                config=AttentionConfig(kv_cache=KV_CACHE),
            ),
            PositionSpec(
                resource_key=ROPE, nodes={LLM_NODE},
                config=PositionConfig(kv_cache=KV_CACHE, rope_theta=text.rope_theta),
            ),
            SamplerSpec(
                resource_key=SAMPLER, nodes={LLM_NODE},
                vocab_size=text.vocab_size,
                # the reference decodes greedily and exposes no penalty knob
                enable_repetion_penalty=False,
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
                temperature=model_kwargs.get("temperature", 0.0),
                top_p=model_kwargs.get("top_p", 1.0),
                ignore_eos=model_kwargs.get("ignore_eos", False),
            )
        }

    # -------------------------------------------------------------------
    # Model ABC: graph walk definitions
    # -------------------------------------------------------------------

    def get_max_output_tokens(self, **model_kwargs):
        return model_kwargs.get("max_output_tokens", self.config.max_new_tokens)

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill = Sequential([
            GraphNode(
                name=ENCODER_NODE,
                input_names=["audio_features"],
                outputs=[GraphEdge(next_node=LLM_NODE, name="audio_embeds")],
            ),
            GraphNode(
                name=LLM_NODE,
                input_names=["audio_embeds", "text_inputs"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                        persist=True,
                    ),
                ],
            ),
        ])
        decode = Loop(
            name=DECODE_LOOP,
            section=GraphNode(
                name=LLM_NODE,
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text"),
                    GraphEdge(next_node=LLM_NODE, name="text_inputs"),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )
        return {PREFILL_WALK: prefill, DECODE_WALK: decode}

    # -------------------------------------------------------------------
    # Model ABC: forward pass args
    # -------------------------------------------------------------------

    @staticmethod
    def _edge(node: str, name: str, tensor_info: list[TensorPointerInfo]) -> GraphEdge:
        edge = GraphEdge(next_node=node, name=name)
        edge.tensor_info = list(tensor_info)
        return edge

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        inputs = [
            self._edge(ENCODER_NODE, "audio_features", input_signals.get("audio_features", [])),
            self._edge(LLM_NODE, "text_inputs", input_signals.get("text_inputs", [])),
        ]
        return ForwardPassArgs(
            full_metadata=CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk=PREFILL_WALK,
                is_prefill=True,
            ),
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
        """Single-partition state machine: prefill -> decode loop -> done."""
        metadata = partition_metadata
        if metadata.is_prefill:
            metadata.is_prefill = False
            metadata.graph_walk = DECODE_WALK
        elif metadata.graph_walk == DECODE_WALK:
            return ForwardPassArgs(
                full_metadata=metadata, inputs=[], unpersist_tensors=[], request_done=True,
            )
        inputs = [self._edge(LLM_NODE, "text_inputs", persist_signals.get("new_token", []))]
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=sum([inp.tensor_info for inp in inputs], start=[]),
            step_metadata={"is_prefill": False},
        )

    # -------------------------------------------------------------------
    # Model ABC: media loading + prompt processing
    # -------------------------------------------------------------------

    def load_audio(self, filepath: str, device: str) -> TensorAndMetadata:
        del device
        waveform = load_audio_file(filepath, self.config.sampling_rate)
        return TensorAndMetadata(
            data=waveform, metadata=dict(sample_rate=self.config.sampling_rate, num_channels=1),
        )

    def prompt_text(
        self, num_audio_tokens: int, context: str = "", language: str | None = None,
        assistant_prefix: str = "",
    ) -> str:
        """The reference SDK's ChatML prompt, with ``num_audio_tokens``
        placeholders and, for a forced language, the ``language X<asr_text>``
        assistant prefix. ``assistant_prefix`` is raw model output the
        assistant turn continues from (streaming re-decodes the audio so far
        with the stable part of the previous hypothesis prefilled)."""
        prompt = (
            f"<|im_start|>system\n{context}<|im_end|>\n"
            f"<|im_start|>user\n<|audio_start|>{'<|audio_pad|>' * num_audio_tokens}<|audio_end|><|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        if language:
            prompt += f"{LANGUAGE_PREFIX}{language}{ASR_TEXT_TAG}"
        return prompt + assistant_prefix

    def prompt_ids(
        self, num_audio_tokens: int, context: str = "", language: str | None = None,
        assistant_prefix: str = "",
    ) -> list[int]:
        # One placeholder in the text, expanded in ids: tokenizing thousands
        # of repeated special tokens is slow and adds nothing.
        ids = self.tokenizer.encode(
            self.prompt_text(1, context, language, assistant_prefix), add_special_tokens=False,
        )
        slot = ids.index(self.config.audio_token_id)
        return ids[:slot] + [self.config.audio_token_id] * num_audio_tokens + ids[slot + 1:]

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """Log-mel of the whole clip plus the ChatML prompt.

        The text ``prompt`` (or ``initial_prompt``) is the system-turn context
        the reference SDK calls ``context`` (hot words, domain hints);
        ``language`` (ISO code or name) forces the output language and turns
        off the model's own language line; ``assistant_prefix`` is model
        output to continue from (the streaming route's stable hypothesis).
        """
        raw_audio_inputs = (tensors or {}).get("audio_inputs", [])
        if len(raw_audio_inputs) != 1:
            raise ValueError(
                f"Qwen3-ASR expects exactly one audio input per request; got {len(raw_audio_inputs)}."
            )
        waveform = raw_audio_inputs[0].reshape(-1).to(torch.float32).cpu()
        if waveform.numel() == 0:
            raise ValueError("Qwen3-ASR received an empty audio input.")
        if waveform.numel() > self.config.max_audio_samples:
            raise ValueError(
                f"Qwen3-ASR takes at most {self.config.max_audio_seconds:.0f} s of audio per request; "
                f"got {waveform.numel() / self.config.sampling_rate:.1f} s. Split the file first."
            )
        # the encoder's first conv needs a few frames; the reference pads
        # sub-0.5 s clips with silence too
        min_samples = self.config.sampling_rate // 2
        if waveform.numel() < min_samples:
            waveform = torch.nn.functional.pad(waveform, (0, min_samples - waveform.numel()))

        audio_features = self.log_mel(waveform)  # (num_mel_bins, T)
        num_audio_tokens = self.config.audio.tokens_for_frames(audio_features.shape[-1])
        context = kwargs.get("initial_prompt") or prompt or ""
        language = self.config.language_name(kwargs.get("language"))
        ids = self.prompt_ids(
            num_audio_tokens, context=context, language=language,
            assistant_prefix=kwargs.get("assistant_prefix") or "",
        )
        return {
            "audio_features": [audio_features],
            "text_inputs": [torch.tensor(ids, dtype=torch.long)],
        }

    # -------------------------------------------------------------------
    # Model ABC: postprocess
    # -------------------------------------------------------------------

    def postprocess(self, output: torch.Tensor, modality: str, **kwargs) -> bytes:
        if modality == "text":
            return self._detokenizer.to_bytes(output.reshape(-1).tolist())
        raise ValueError(f"Unsupported modality for Qwen3-ASR: {modality!r}")

    # -------------------------------------------------------------------
    # Model ABC: sharding + submodule loading
    # -------------------------------------------------------------------

    def get_default_sharding_config(self):
        from mstar.distributed.base import ShardingConfig

        return ShardingConfig(groups=[], tp_enabled_nodes={LLM_NODE}, shard_dim={})

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        if node_name == ENCODER_NODE:
            submodule = self._create_encoder_submodule(device, autocast_dtype)
        elif node_name == LLM_NODE:
            submodule = self._create_llm_submodule(device, tp_group, autocast_dtype)
        else:
            submodule = None
        logger.info("Successfully loaded Qwen3-ASR submodule for %s", node_name)
        self._submodule_cache[node_name] = submodule
        return submodule

    def _create_encoder_submodule(self, device: str, autocast_dtype: torch.dtype | None) -> NodeSubmodule:
        from mstar.model.components.aut_encoder import AuTEncoder
        from mstar.model.loader.iterators import iter_safetensors_shards

        with torch.device("meta"):
            encoder = AuTEncoder(self.config.audio, attn_key=AUT_ATTN)
        if autocast_dtype is not None:
            encoder = encoder.to(autocast_dtype)
        encoder.to_empty(device=device)
        prefix = "thinker.audio_tower."
        weights = iter_safetensors_shards(self.local_dir, device=device, prefix=prefix)
        encoder.load_weights((k.removeprefix(prefix), v) for k, v in weights)
        encoder.eval()

        from mstar.model.qwen3_asr.submodules import Qwen3ASREncoderSubmodule
        return Qwen3ASREncoderSubmodule(encoder=encoder, config=self.config)

    @staticmethod
    def _llm_remap(name: str) -> str | None:
        if name.startswith("thinker.model."):
            return name.removeprefix("thinker.model.")
        if name == "thinker.lm_head.weight":
            return "lm_head.weight"
        return None

    def _create_llm_submodule(
        self, device: str, tp_group, autocast_dtype: torch.dtype | None,
    ) -> NodeSubmodule:
        from mstar.model.components.qwen3_lm import Qwen3DenseLM
        from mstar.model.loader.iterators import iter_safetensors_shards

        with torch.device("meta"):
            llm = Qwen3DenseLM(
                self.config.text, attn_key=ATTN, kv_key=KV_CACHE, pos_key=ROPE, comm_group=tp_group,
            )
        if autocast_dtype is not None:
            llm = llm.to(autocast_dtype)
        llm.to_empty(device=device)
        weights = iter_safetensors_shards(self.local_dir, device=device, prefix="thinker.")
        llm.load_weights(
            (mapped, v) for k, v in weights
            if (mapped := self._llm_remap(k)) is not None
        )
        llm.eval()

        from mstar.model.qwen3_asr.submodules import Qwen3ASRLLMSubmodule
        return Qwen3ASRLLMSubmodule(llm=llm, config=self.config)
