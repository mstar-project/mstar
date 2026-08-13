"""NemotronDuplexModel — NVIDIA NemotronLabs VoiceChat-11B in M*.

A half-duplex speech-to-speech model with three stages:

    conformer_encoder (STATELESS) — 16 kHz speech → fused embeds + RNN-T transcript
    nano_llm          (KV_CACHE)  — Nemotron-H hybrid Mamba-2/attn/MLP backbone (9B)
    eartts_talker     (KV_CACHE)  — Gemma3 talker → 31-codebook RVQ codes
    audio_codec       (STATELESS) — RVQ codes → 22.05 kHz PCM

Input:  text prompt + user speech.  Output: agent text + agent speech +
streaming user transcription.

PHASING (agreed): this draft wires the **text path** end-to-end
(``prefill_text`` → ``decode``) so the Nemotron-H backbone can be brought up and
verified against the HF reference first. The audio nodes are declared and their
submodules stubbed; the full duplex graph + async partitions (encoder →
nano_llm → talker → codec, with ``StreamingGraphEdge`` fan-out of transcript /
text / audio) is the Phase 6 extension sketched in ``_duplex_design`` below.

Contract methods crib from ``mstar/model/orpheus/orpheus_model.py`` (single
streaming LLM+codec) and ``mstar/model/qwen3_omni/qwen3_omni_model.py`` (omni,
multi-KV, aux sampling, async partitions).
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.engine.base import EngineType
from mstar.engine.kv_cache_engine import KVCacheConfig
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.nemotron_duplex.config import NemotronDuplexConfig
from mstar.model.submodule_base import NodeSubmodule
from mstar.utils.sampling import SamplingConfig

logger = logging.getLogger(__name__)


def _resolve_local_hf_snapshot(repo_id: str, cache_dir: str | None = None) -> str:
    from huggingface_hub import snapshot_download

    try:
        local_dir = snapshot_download(repo_id=repo_id, cache_dir=cache_dir, local_files_only=False)
    except Exception as e:  # noqa: BLE001 - fall back to the raw id, mirrors Orpheus
        logger.warning("Error downloading %s from huggingface: %s", repo_id, e)
        return repo_id
    return str(Path(local_dir))


class NemotronDuplexModel(Model):
    """NVIDIA NemotronLabs VoiceChat-11B (duplex S2S)."""

    def __init__(self, model_path_hf: str, cache_dir: str | None = None, **kwargs):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.config = NemotronDuplexConfig()
        self._tokenizer = None  # lazy — not needed in dummy mode
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    @property
    def tokenizer(self):
        # TODO(verify): confirm the text tokenizer source. The Spark repo ships
        # it under ``nano/``; the base repo ships only ``rnnt_tokenizer/`` (the
        # ASR head). Loaded lazily so dummy-mode tests never touch the network.
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            src = _resolve_local_hf_snapshot(self.model_path_hf, cache_dir=self.cache_dir)
            self._tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
        return self._tokenizer

    # -------------------------------------------------------------------
    # KV cache config — nano_llm (attention layers only) + eartts_talker
    # -------------------------------------------------------------------

    def get_kv_cache_config(self) -> list[KVCacheConfig]:
        nano = self.config.nano
        eartts = self.config.eartts
        return [
            KVCacheConfig(
                # Only the ``*`` (attention) layers hold a KV cache; the Mamba
                # and MLP layers do not. The submodule maps global layer -> this
                # dense attention index.
                num_layers=nano.num_attention_layers,
                num_kv_heads=nano.num_key_value_heads,
                head_dim=nano.head_dim,
                max_seq_len=nano.max_position_embeddings,
                num_qo_heads=nano.num_attention_heads,
                nodes=["nano_llm"],
            ),
            # Phase 5 — declared now so layout is stable; only used once the
            # talker submodule is implemented.
            KVCacheConfig(
                num_layers=eartts.num_hidden_layers,
                num_kv_heads=eartts.num_key_value_heads,
                head_dim=eartts.head_dim,
                max_seq_len=nano.max_position_embeddings,
                num_qo_heads=eartts.num_attention_heads,
                nodes=["eartts_talker"],
            ),
        ]

    def get_node_engine_types(self) -> dict[str, EngineType]:
        return {
            "conformer_encoder": EngineType.STATELESS,  # Phase 4
            "nano_llm": EngineType.KV_CACHE,            # implemented (text path)
            "eartts_talker": EngineType.KV_CACHE,        # Phase 5
            "audio_codec": EngineType.STATELESS,        # Phase 5
        }

    # -------------------------------------------------------------------
    # Graph walks. Text path is live; audio walks are Phase 6 (see design note).
    # -------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill_text = GraphNode(
            name="nano_llm",
            input_names=["text_inputs"],
            outputs=[
                GraphEdge(
                    next_node=EMPTY_DESTINATION,
                    name="new_token",
                    conductor_new_token=True,
                    persist=True,
                ),
            ],
        )
        decode = Loop(
            name="decode_loop",
            section=GraphNode(
                name="nano_llm",
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(next_node="nano_llm", name="text_inputs"),
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                        conductor_new_token=True,
                    ),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )
        return dict(prefill_text=prefill_text, decode=decode)

    @staticmethod
    def _duplex_design() -> str:
        """Phase 6 target (documentation only).

        Nodes/partitions:
            ENC  partition: conformer_encoder  (prefill_audio) — emits
                 ``combined_embeds`` -> nano_llm and streaming ``transcript_token``
                 -> EMIT_TO_CLIENT (modality "text").
            LLM  partition: nano_llm (prefill_text | prefill_audio | decode) —
                 streams ``new_token`` (agent text) to the client and a talker
                 conditioning signal to the TTS partition via StreamingGraphEdge.
            TTS  partition: eartts_talker (tts_decode Loop) -> audio_codec
                 (codec_chunk) -> EMIT_TO_CLIENT (modality "audio"), with a
                 SlidingWindowChunkPolicy on the talker->codec connection
                 (Orpheus pattern) for low-latency 22.05 kHz output.

        Requires overriding get_partition_topology / get_partitions and routing
        cross-partition tensors with StreamingGraphEdge (target_partition=...).
        """
        return "see docstring"

    # -------------------------------------------------------------------
    # Prompt processing
    # -------------------------------------------------------------------

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> NameToTensorList:
        # Phase 4: when audio is in ``input_modalities``, run the conformer
        # encoder path to derive ``combined_embeds`` from ``audio_inputs``.
        if "audio" in input_modalities:
            raise NotImplementedError("Audio input (conformer encoder) — Phase 4.")
        if prompt is None:
            return {}
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0].to(torch.long)
        return {"text_inputs": [ids]}

    # -------------------------------------------------------------------
    # Forward-pass-args state machine (single "default" partition, text path)
    # -------------------------------------------------------------------

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk="prefill_text",
            is_prefill=True,
        )
        graph_edge = GraphEdge(next_node="nano_llm", name="text_inputs")
        graph_edge.tensor_info = input_signals.get("text_inputs", [])
        inputs = [graph_edge]
        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=inputs,
            unpersist_tensors=sum([inp.tensor_info for inp in inputs], start=[]),
            step_metadata={"is_prefill": True},
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections=None,
    ) -> ForwardPassArgs:
        metadata = partition_metadata
        request_done = False
        if metadata.is_prefill:
            metadata.is_prefill = False
            metadata.graph_walk = "decode"
        elif metadata.graph_walk == "decode":
            request_done = True
            metadata.kwargs["decode_finished"] = True

        if request_done:
            return ForwardPassArgs(
                full_metadata=metadata, inputs=[], unpersist_tensors=[], request_done=True,
            )

        graph_edge = GraphEdge(next_node="nano_llm", name="text_inputs")
        graph_edge.tensor_info = persist_signals.get("new_token", [])
        inputs = [graph_edge]
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=sum([inp.tensor_info for inp in inputs], start=[]),
            step_metadata={"is_prefill": metadata.is_prefill},
        )

    # -------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------

    def get_sampling_config(self, node_name: str, model_kwargs: dict | None = None) -> SamplingConfig | None:
        if node_name != "nano_llm":
            return None  # eartts_talker sampling: Phase 5 (+ get_aux_sampling_configs)
        model_kwargs = model_kwargs or {}
        keys = ["temperature", "top_p", "repetition_penalty", "ignore_eos"]
        params = {k: model_kwargs.get(k, getattr(self.config, k)) for k in keys}
        return SamplingConfig(vocab_size=self.config.vocab_size, **params)

    def get_output_sample_rate(self, modality: str = "audio") -> int:
        return self.config.eartts.sample_rate

    # -------------------------------------------------------------------
    # Standalone offline inference (Phase A — mirrors the reference
    # NemotronVoiceChat.offline_inference; bypasses the M* serving engine).
    # -------------------------------------------------------------------

    @torch.no_grad()
    def offline_inference(
        self,
        input_signal: torch.Tensor,       # (1, num_samples) 16 kHz mono
        input_signal_lens: torch.Tensor,  # (1,)
        prompt_tokens: torch.Tensor | None = None,  # (1, P) text token ids
        device: str = "cuda",
        temperature: float = 0.0,
        decode_audio: bool = True,
    ) -> dict:
        """Duplex speech-to-speech offline inference (batch size 1, fp32, greedy).

        Frame-synchronous loop mirroring the reference: perception -> per-frame
        [AddFusion(prev agent text, audio, prev function) -> nano step -> sample
        text/function] -> talker emits RVQ codes -> codec -> waveform.

        The nano uses the reference's Nemotron no-cache path: full re-prefill over
        [0:t+1] each frame (reuses the verified ``offline_forward``).

        Returns ``{tokens_text, tokens_function, tokens_audio, audio, audio_len}``.
        TODO: talker sampling loop (``EarTTSTalker.infer_codes_one_step``) and the
        text tokenizer (for detokenised ``text``) are being finalised; this method
        returns token ids and — once those land — audio. Verify vs the NeMo ref.
        """
        cfg = self.config
        # --- build the component modules (loads real weights) ---
        conf_sub = self.get_submodule("conformer_encoder", device=device)
        nano_sub = self.get_submodule("nano_llm", device=device)
        perception = conf_sub.perception
        nano = nano_sub.language_model
        embed_tokens = nano.embed_tokens

        # --- perception: audio -> per-frame embeddings (B, T, H) ---
        audio_embeds, lengths = perception(input_signal.to(device), input_signal_lens.to(device))
        B, T, H = audio_embeds.shape

        # --- prepend the system prompt embeddings, if any ---
        prompt_len = 0
        if prompt_tokens is not None and prompt_tokens.numel() > 0:
            prompt_len = prompt_tokens.shape[1]
            prompt_emb = embed_tokens(prompt_tokens.to(device))            # (1, P, H)
            audio_embeds = torch.cat([prompt_emb, audio_embeds], dim=1)    # (1, P+T, H)
            T = audio_embeds.shape[1]

        gen_text = torch.full((B, T), cfg.text_pad_id, device=device, dtype=torch.long)
        gen_function = torch.full((B, T), cfg.text_pad_id, device=device, dtype=torch.long)
        input_embeds = torch.zeros_like(audio_embeds)

        def fuse(agent_emb, audio_emb, function_emb):
            out = agent_emb * cfg.agent_text_weight + audio_emb * cfg.user_audio_weight
            if cfg.use_function_head:
                out = out + function_emb * cfg.function_weight
            return out

        def nano_step(seq_embeds):
            # seq_embeds: (L, H) fused embeddings for frames [0:t+1]; returns the
            # last position's text + function logits (non-cache re-prefill).
            hidden = nano.llm.offline_forward(seq_embeds)      # (L, H)
            last = hidden[-1:]                                  # (1, H)
            return nano.lm_head(last)[0], nano.function_head(last)[0]

        # frame 0: agent channel is BOS
        bos_emb = embed_tokens(torch.tensor([cfg.text_bos_id], device=device))  # (1, H)
        input_embeds[:, 0] = fuse(bos_emb, audio_embeds[:, 0], embed_tokens(gen_function[:, 0]))
        text_logits, function_logits = nano_step(input_embeds[0, :1])
        if prompt_len == 0:
            gen_text[:, 0] = text_logits.argmax(-1)
            gen_function[:, 0] = function_logits.argmax(-1)

        for t in range(1, T):
            agent_emb = embed_tokens(gen_text[:, t - 1])
            function_emb = embed_tokens(gen_function[:, t - 1])
            input_embeds[:, t] = fuse(agent_emb, audio_embeds[:, t], function_emb)
            if t < prompt_len:
                continue  # prompt region: teacher-forced, no sampling
            text_logits, function_logits = nano_step(input_embeds[0, : t + 1])
            # greedy (temperature==0). TODO: top-p / repetition-penalty sampling.
            gen_text[:, t] = text_logits.argmax(-1)
            gen_function[:, t] = function_logits.argmax(-1)

        result = {
            "tokens_text": gen_text,
            "tokens_function": gen_function,
            "prompt_len": prompt_len,
        }

        # --- audio-out: talker -> RVQ codes -> codec -> waveform ---
        if decode_audio:
            result.update(self._decode_audio_path(gen_text, prompt_len, device, temperature))
        return result

    def _decode_audio_path(self, gen_text, prompt_len, device, temperature):
        """Talker (autoregressive RVQ codes) -> codec (waveform). Talker sampling
        loop is being finalised in EarTTSTalker.infer_codes_one_step."""
        talker = self.get_submodule("eartts_talker", device=device).talker
        codec = self.get_submodule("audio_codec", device=device).codec
        text_stream = gen_text[:, prompt_len:]                 # (1, T') agent text per frame
        num_q = self.config.eartts.num_quantizers

        if not hasattr(talker, "infer_codes_one_step"):
            return {}  # talker sampling loop not yet available

        state = talker.init_state(batch_size=text_stream.shape[0], device=device)
        codes = []
        prev_codes = None
        prev_subword = text_stream[:, 0:1]
        for t in range(text_stream.shape[1]):
            cur_subword = text_stream[:, t : t + 1]
            frame_codes, state = talker.infer_codes_one_step(
                state, cur_subword, prev_subword, prev_codes, temperature=temperature,
            )
            codes.append(frame_codes)
            prev_codes, prev_subword = frame_codes, cur_subword
        gen_codes = torch.stack(codes, dim=1)                  # (1, T', num_q)
        assert gen_codes.shape[-1] == num_q
        code_len = torch.tensor([gen_codes.shape[1]], device=device)
        with torch.autocast(device_type="cuda", enabled=False):
            audio, audio_len = codec.decode(gen_codes, code_len)
        return {"tokens_audio": gen_codes, "audio": audio.squeeze(1), "audio_len": audio_len}

    # -------------------------------------------------------------------
    # Postprocess
    # -------------------------------------------------------------------

    def postprocess(self, output: torch.Tensor, modality: str, **kwargs) -> bytes:
        if modality == "text":
            token_ids = output.tolist() if output.dim() else [int(output)]
            return self.tokenizer.decode(token_ids).encode("utf-8")
        if modality == "audio":
            if output.numel() == 0:
                return b""
            return output.cpu().numpy().tobytes()
        raise ValueError(f"Unsupported modality for NemotronDuplex: {modality!r}")

    # -------------------------------------------------------------------
    # Sharding — nano_llm attention/MLP are TP-eligible (Phase 10).
    # -------------------------------------------------------------------

    def get_default_sharding_config(self):
        from mstar.distributed.base import ShardingConfig

        # TODO(Phase 10): Mamba layers need head-sharded state; declare TP only
        # after the components are built from the distributed linears.
        return ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={})

    # -------------------------------------------------------------------
    # Submodule loading
    # -------------------------------------------------------------------

    def get_submodule(
        self,
        node_name: str,
        device: str = "cpu",
        tp_group=None,
        autocast_dtype: torch.dtype | None = None,
        **kwargs,
    ) -> NodeSubmodule | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(node_name, device, autocast_dtype=autocast_dtype)
        self._submodule_cache[node_name] = submodule
        return submodule

    def _create_submodule(
        self, node_name: str, device: str, autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name == "nano_llm":
            return self._create_nano_submodule(device, autocast_dtype=autocast_dtype)
        if node_name == "conformer_encoder":
            return self._create_conformer_submodule(device, autocast_dtype=autocast_dtype)
        if node_name == "eartts_talker":
            return self._create_talker_submodule(device, autocast_dtype=autocast_dtype)
        if node_name == "audio_codec":
            return self._create_codec_submodule(device, autocast_dtype=autocast_dtype)
        return None

    def _json_config(self, local_dir: str) -> NemotronDuplexConfig:
        """Config populated from the checkpoint's ``config.json`` (cached).

        Falls back to dataclass defaults if the file is absent (e.g. when
        ``local_dir`` is an unresolved repo id).
        """
        if getattr(self, "_json_config_cache", None) is None:
            try:
                self._json_config_cache = NemotronDuplexConfig.from_pretrained(local_dir)
            except (FileNotFoundError, KeyError) as e:
                logger.warning("Could not read config.json (%s); using config defaults.", e)
                self._json_config_cache = self.config
        return self._json_config_cache

    def _build_and_load(self, module: torch.nn.Module, device: str, autocast_dtype, local_dir: str):
        """meta-build -> cast -> materialize -> load the module's checkpoint subtree.

        Each component's ``load_weights`` streams the single composite
        ``model.safetensors`` and keeps only its own subtree (verified: every
        checkpoint tensor maps to exactly one component param).
        """
        from mstar.model.loader import load_weights

        if autocast_dtype is not None:
            module = module.to(autocast_dtype)
        module.to_empty(device=device)
        load_weights(module, local_dir, device=device)
        return module.eval()

    def _create_nano_submodule(
        self, device: str, autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule:
        from mstar.model.nemotron_duplex.components.nemotron_h import NemotronHForCausalLM
        from mstar.model.nemotron_duplex.submodules import NemotronHLLMSubmodule

        local_dir = _resolve_local_hf_snapshot(self.model_path_hf, cache_dir=self.cache_dir)
        with torch.device("meta"):
            language_model = NemotronHForCausalLM(self.config.nano)
        language_model = self._build_and_load(language_model, device, autocast_dtype, local_dir)
        return NemotronHLLMSubmodule(language_model=language_model, config=self.config)

    def _create_conformer_submodule(
        self, device: str, autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule:
        from mstar.model.nemotron_duplex.components.conformer import Perception
        from mstar.model.nemotron_duplex.components.rnnt import RnntDecoder, RnntJoint
        from mstar.model.nemotron_duplex.submodules import ConformerEncoderSubmodule

        local_dir = _resolve_local_hf_snapshot(self.model_path_hf, cache_dir=self.cache_dir)
        cfg = self._json_config(local_dir)
        with torch.device("meta"):
            perception = Perception(cfg.stt)
            rnnt_dec, rnnt_joint = RnntDecoder(cfg.rnnt), RnntJoint(cfg.rnnt)
        perception = self._build_and_load(perception, device, autocast_dtype, local_dir)
        rnnt_dec = self._build_and_load(rnnt_dec, device, autocast_dtype, local_dir)
        rnnt_joint = self._build_and_load(rnnt_joint, device, autocast_dtype, local_dir)
        return ConformerEncoderSubmodule(perception, rnnt_dec, rnnt_joint, config=self.config)

    def _create_talker_submodule(
        self, device: str, autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule:
        from mstar.model.nemotron_duplex.components.eartts_talker import EarTTSTalker
        from mstar.model.nemotron_duplex.submodules import EarTTSTalkerSubmodule

        local_dir = _resolve_local_hf_snapshot(self.model_path_hf, cache_dir=self.cache_dir)
        cfg = self._json_config(local_dir)
        with torch.device("meta"):
            talker = EarTTSTalker(cfg.eartts)
        talker = self._build_and_load(talker, device, autocast_dtype, local_dir)
        return EarTTSTalkerSubmodule(talker=talker, config=self.config)

    def _create_codec_submodule(
        self, device: str, autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule:
        from mstar.model.nemotron_duplex.components.audio_codec import AudioCodec
        from mstar.model.nemotron_duplex.submodules import AudioCodecDecoderSubmodule

        local_dir = _resolve_local_hf_snapshot(self.model_path_hf, cache_dir=self.cache_dir)
        cfg = self._json_config(local_dir)
        with torch.device("meta"):
            codec = AudioCodec(cfg.codec)
        codec = self._build_and_load(codec, device, autocast_dtype, local_dir)
        return AudioCodecDecoderSubmodule(codec=codec, config=self.config)
