"""
HiggsAudioModel: speech-to-text model (bosonai/higgs-audio-v3-stt).

Higgs-Audio STT couples a Whisper-style audio encoder ("audio_tower")
and MLP projector ("audio_encoder_proj") with a dense Qwen3-1.7B text
LLM. Audio is split into fixed 4 s chunks, each chunk is encoded to
12.5 embeddings/s in LLM space, and the embeddings are spliced into the
ChatML prompt between ``<|audio_bos|>`` and ``<|audio_eos|>``:

    <|im_start|>user\\n{prompt}<|audio_bos|>[audio embeds]<|audio_eos|><|im_end|>\\n
    <|im_start|>assistant\\n -> transcript

Architecture (2 nodes, single partition):
    audio_encoder  (enc_dec) — audio_tower + projector (checkpoint code)
    LLM            (ar)      — dense Qwen3, paged KV cache

Graph walks (sequential prefill schedule, qwen3_omni-style):
    prefill_text  — text span -> LLM KV cache (the second/post-audio
                    span is the last prefill: samples the first token)
    prefill_audio — audio_encoder -> LLM KV cache
    decode        — LLM loop

Not implemented from the reference pipeline: silero-VAD chunk
boundaries (fixed 4 s chunks only) and the client-side n-gram loop fix
(``ngram_loop_fix.py``), which is post-hoc string cleanup.
"""

import logging
from pathlib import Path

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mstar.engine.base import EngineType
from mstar.engine.kv_store import KVCacheConfig
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.higgs_audio.config import HiggsAudioModelConfig
from mstar.model.submodule_base import NodeSubmodule
from mstar.utils.sampling import SamplingConfig

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


class HiggsAudioModel(Model):
    """Higgs-Audio STT: audio_tower + projector + dense Qwen3 LLM."""

    # The reference pipeline extracts mel features with the
    # whisper-large-v3 processor (not from the higgs checkpoint).
    WHISPER_PROCESSOR_ID = "openai/whisper-large-v3"

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        **kwargs,
    ):
        self.cache_dir = cache_dir
        self.model_path_hf = model_path_hf

        self.local_dir = _resolve_local_hf_snapshot(model_path_hf, cache_dir=cache_dir)
        self.config = HiggsAudioModelConfig.from_pretrained(self.local_dir)

        from transformers import AutoFeatureExtractor, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.local_dir, cache_dir=cache_dir,
        )
        whisper_dir = _resolve_local_hf_snapshot(
            self.WHISPER_PROCESSOR_ID, cache_dir=cache_dir,
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            whisper_dir, cache_dir=cache_dir,
        )

        from mstar.model.utils import ByteLevelDetokenizer
        self._detokenizer = ByteLevelDetokenizer(self.tokenizer)

        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    # -------------------------------------------------------------------
    # Model ABC: KV cache config
    # -------------------------------------------------------------------

    def get_kv_cache_config(self) -> list[KVCacheConfig]:
        return [KVCacheConfig(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            max_seq_len=self.config.max_position_embeddings,
            num_qo_heads=self.config.num_attention_heads,
            nodes=["LLM"],
            # ~2k tokens per ASR request (prompt + 12.5 audio embeds/s +
            # transcript); 256 pages = 32k tokens ≈ 16 concurrent
            # requests at ~3.7 GB.
            max_num_pages=256,
        )]

    # -------------------------------------------------------------------
    # Model ABC: node engine types
    # -------------------------------------------------------------------

    def get_node_engine_types(self) -> dict[str, EngineType]:
        return {
            "audio_encoder": EngineType.STATELESS,
            "LLM": EngineType.KV_CACHE,
        }

    # -------------------------------------------------------------------
    # Model ABC: graph walk definitions
    # -------------------------------------------------------------------

    def get_max_output_tokens(self, **model_kwargs):
        # Reference transcribe.py generates up to 1024 new tokens.
        return model_kwargs.get("max_output_tokens", 1024)

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        llm_prefill_outputs = [
            GraphEdge(
                next_node=EMIT_TO_CLIENT,
                name="new_token",
                output_modality="text",
                persist=True,
            ),
        ]

        prefill_text = GraphNode(
            name="LLM",
            input_names=["text_inputs"],
            outputs=llm_prefill_outputs,
        )

        prefill_audio = Sequential([
            GraphNode(
                name="audio_encoder",
                input_names=["audio_features", "audio_feature_lens"],
                outputs=[GraphEdge(next_node="LLM", name="audio_embeds")],
            ),
            GraphNode(
                name="LLM",
                input_names=["audio_embeds"],
                outputs=[edge.clone() for edge in llm_prefill_outputs],
            ),
        ])

        decode = Loop(
            name="decode_loop",
            section=GraphNode(
                name="LLM",
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                    ),
                    GraphEdge(
                        next_node="LLM",
                        name="text_inputs",
                    ),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )

        return dict(
            prefill_text=prefill_text,
            prefill_audio=prefill_audio,
            decode=decode,
        )

    # -------------------------------------------------------------------
    # Model ABC: forward pass args
    # -------------------------------------------------------------------

    def _build_prefill_schedule(
        self,
        input_signals: dict[str, list[TensorPointerInfo]],
    ) -> list[tuple[str, dict[str, TensorPointerInfo]]]:
        """[text-before-audio] + [audio] + [text-after-audio].

        ``process_prompt`` emits two ``text_inputs`` tensors (the ChatML
        prompt split at the audio splice point) plus the per-chunk mel
        features.
        """
        texts = input_signals.get("text_inputs", [])
        audio_features = input_signals.get("audio_features", [])
        audio_feature_lens = input_signals.get("audio_feature_lens", [])

        schedule: list[tuple[str, dict[str, TensorPointerInfo]]] = []
        if texts:
            schedule.append(("prefill_text", {"text_inputs": texts[0]}))
        if audio_features:
            entry = {"audio_features": audio_features[0]}
            if audio_feature_lens:
                entry["audio_feature_lens"] = audio_feature_lens[0]
            schedule.append(("prefill_audio", entry))
        for text in texts[1:]:
            schedule.append(("prefill_text", {"text_inputs": text}))
        return schedule

    def _prefill_inputs(
        self, metadata: CurrentForwardConductorMetadata,
    ) -> list[GraphEdge]:
        schedule = metadata.kwargs["prefill_schedule"]
        step = metadata.kwargs["prefill_step"]
        walk_name, tensor_dict = schedule[step]
        target_node = "audio_encoder" if walk_name == "prefill_audio" else "LLM"

        edges: list[GraphEdge] = []
        for input_name, tensor_info in tensor_dict.items():
            edge = GraphEdge(next_node=target_node, name=input_name)
            edge.tensor_info = [tensor_info]
            edges.append(edge)
        return edges

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        schedule = self._build_prefill_schedule(input_signals)
        if not schedule:
            raise ValueError("Higgs-Audio request has no prefill inputs.")

        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=schedule[0][0],
            is_prefill=True,
            kwargs={
                "prefill_schedule": schedule,
                "prefill_step": 0,
            },
        )

        inputs = self._prefill_inputs(full_metadata)
        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])

        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={
                "is_prefill": True,
                "is_last_prefill": len(schedule) == 1,
            },
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        """Single-partition state machine: prefill schedule -> decode -> done."""
        metadata = partition_metadata

        if metadata.is_prefill:
            step = metadata.kwargs["prefill_step"] + 1
            schedule = metadata.kwargs["prefill_schedule"]
            if step < len(schedule):
                metadata.kwargs["prefill_step"] = step
                metadata.graph_walk = schedule[step][0]
                inputs = self._prefill_inputs(metadata)
                unpersist_tensors = sum(
                    [inp.tensor_info for inp in inputs], start=[]
                )
                return ForwardPassArgs(
                    full_metadata=metadata,
                    inputs=inputs,
                    unpersist_tensors=unpersist_tensors,
                    step_metadata={
                        "is_prefill": True,
                        "is_last_prefill": step == len(schedule) - 1,
                    },
                )
            metadata.is_prefill = False
            metadata.graph_walk = "decode"
        elif metadata.graph_walk == "decode":
            # Decode loop returned to the conductor: EOS or max tokens.
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True,
            )

        edge = GraphEdge(next_node="LLM", name="text_inputs")
        edge.tensor_info = persist_signals.get("new_token", [])
        inputs = [edge]
        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])

        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={"is_prefill": False},
        )

    # -------------------------------------------------------------------
    # Model ABC: prompt processing
    # -------------------------------------------------------------------

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """Chunk the audio, extract per-chunk mels, and split the ChatML
        prompt at the audio splice point.

        Matches the reference ``transcribe.py``: fixed
        ``chunk_size_seconds`` chunks (no VAD), whisper-large-v3 mel
        extraction padded to the longest chunk, and the transcription
        prompt wrapped in ChatML with the audio between ``<|audio_bos|>``
        and ``<|audio_eos|>``. ``enable_thinking=False`` (default)
        appends the empty think block so the transcript starts
        immediately.
        """
        raw_audio_inputs = (tensors or {}).get("audio_inputs", [])
        if len(raw_audio_inputs) != 1:
            raise ValueError(
                f"Higgs-Audio expects exactly one audio input per request; "
                f"got {len(raw_audio_inputs)}."
            )

        waveform = raw_audio_inputs[0].reshape(-1).cpu().numpy()
        chunk_samples = self.config.chunk_size_samples
        chunks = [
            waveform[i:i + chunk_samples]
            for i in range(0, max(len(waveform), 1), chunk_samples)
        ]

        feat = self.feature_extractor(
            chunks,
            sampling_rate=self.config.sampling_rate,
            padding="longest",
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        audio_features = feat["input_features"]           # (num_chunks, mels, T)
        audio_feature_lens = feat["attention_mask"].sum(-1).to(torch.long)

        user_prompt = prompt or self.config.default_prompt

        def enc(s: str) -> list[int]:
            return self.tokenizer.encode(s, add_special_tokens=False)

        pre_ids = (
            enc("<|im_start|>user\n")
            + enc(user_prompt)
            + enc("<|audio_bos|>")
        )
        post_ids = enc("<|audio_eos|>") + enc("<|im_end|>\n<|im_start|>assistant\n")
        if not kwargs.get("enable_thinking", False):
            post_ids += enc("<think>\n\n</think>\n\n")

        return {
            "text_inputs": [
                torch.tensor(pre_ids, dtype=torch.long),
                torch.tensor(post_ids, dtype=torch.long),
            ],
            "audio_features": [audio_features],
            "audio_feature_lens": [audio_feature_lens],
        }

    # -------------------------------------------------------------------
    # Model ABC: sampling / postprocess
    # -------------------------------------------------------------------

    def get_sampling_config(
        self, node_name: str,
        model_kwargs: dict | None = None,
    ) -> SamplingConfig | None:
        model_kwargs = model_kwargs or {}
        return SamplingConfig(
            vocab_size=self.config.vocab_size,
            # Reference eval decodes greedily (do_sample=False).
            temperature=model_kwargs.get("temperature", 0.0),
            top_p=model_kwargs.get("top_p", 1.0),
            ignore_eos=model_kwargs.get("ignore_eos", False),
        )

    def postprocess(
        self,
        output: torch.Tensor,
        modality: str,
        **kwargs,
    ) -> bytes:
        if modality == "text":
            return self._detokenizer.to_bytes(output.reshape(-1).tolist())
        raise ValueError(f"Unsupported modality for Higgs-Audio: {modality!r}")

    # -------------------------------------------------------------------
    # Model ABC: submodule loading
    # -------------------------------------------------------------------

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(
            node_name, device, tp_group=tp_group, autocast_dtype=autocast_dtype,
        )
        logger.info("Successfully loaded Higgs-Audio submodule for %s", node_name)
        self._submodule_cache[node_name] = submodule
        return submodule

    def _create_submodule(
        self, node_name: str, device: str, tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name == "audio_encoder":
            return self._create_encoder_submodule(device)
        elif node_name == "LLM":
            return self._create_llm_submodule(
                device, tp_group=tp_group, autocast_dtype=autocast_dtype,
            )
        return None

    def _create_encoder_submodule(self, device: str) -> NodeSubmodule:
        from mstar.model.higgs_audio.components.audio_tower import (
            HiggsAudioFeatureProjector,
            HiggsAudioTower,
        )
        from mstar.model.utils import ModuleAndPrefix, load_weights_from_hf_shards

        audio_tower = HiggsAudioTower(self.config)
        projector = HiggsAudioFeatureProjector(self.config)

        load_weights_from_hf_shards(
            repo_dir=self.local_dir,
            modules=[
                ModuleAndPrefix(audio_tower, prefix="audio_tower"),
                ModuleAndPrefix(projector, prefix="audio_encoder_proj"),
            ],
            device=device,
        )
        audio_tower.eval()
        projector.eval()

        from mstar.model.higgs_audio.submodules import HiggsAudioEncoderSubmodule
        return HiggsAudioEncoderSubmodule(
            audio_tower=audio_tower, projector=projector, config=self.config,
        )

    @staticmethod
    def _llm_remap(name: str) -> str | None:
        # Generation-only components (audio codec embeddings, audio LM
        # head) and the encoder/projector are not part of the LLM.
        if name.startswith((
            "audio_tower.", "audio_encoder_proj.",
            "audio_codebook_embeddings.",
        )):
            return None
        if name == "audio_decoder_proj.text_lm_head.weight":
            return "lm_head.weight"
        if name.startswith("audio_decoder_proj."):
            return None
        return name

    def _create_llm_submodule(
        self, device: str, tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule:
        from mstar.model.higgs_audio.components.llm import HiggsAudioLLM
        from mstar.model.loader import LLAMA_STACKED_PARAMS, load_hf_weights
        from mstar.model.loader.iterators import iter_safetensors_shards

        with torch.device("meta"):
            llm = HiggsAudioLLM(self.config, comm_group=tp_group)
        # Cast on meta (no allocation) so to_empty allocates directly in the
        # target dtype instead of fp32-then-downcast.
        if autocast_dtype is not None:
            llm = llm.to(autocast_dtype)
        llm.to_empty(device=device)

        weights = iter_safetensors_shards(self.local_dir, device=device)
        load_hf_weights(
            llm, weights,
            stacked_params=LLAMA_STACKED_PARAMS,
            name_remapper=self._llm_remap,
        )
        llm.eval()

        from mstar.model.higgs_audio.submodules import HiggsAudioLLMSubmodule
        return HiggsAudioLLMSubmodule(llm=llm, config=self.config)
