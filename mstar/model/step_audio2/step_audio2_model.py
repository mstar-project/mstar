"""Step-Audio-2-mini: a Qwen2.5-7B LLM over a Whisper-large-v3 audio tower, speaking through
CosyVoice2's token2wav.

Graph:
    prefill_text   LLM
    prefill_audio  audio_encoder -> LLM
    decode         LLM, looped (a text reply)
    decode_spoken  LLM, looped, streaming its speech ids (a spoken reply)
    t2w_chunk      Token2Wav (its own partition) -> client audio

A request runs one prefill walk over its whole rendered chat, picked by whether it carries
audio, then decodes. Whether a reply is spoken is known at intake: the request asks for
audio out, and the assistant turn then opens ``<|BOT|>assistant\\n<tts_start>`` with no
EOT. A spoken reply interleaves one text token with four speech ids (``<tts_pad>`` in the
text slot once the text runs out) and ends on ``<tts_end>``; its speech ids stream to
Token2Wav in the reference client's windows (``examples-vllm-stream.py``: 28 codes, hop 25,
then a last flush). Text and spoken replies decode in one step as ``decode_mixed``.

Ported from the reference's ``stepaudio2.py`` and ``token2wav.py`` (Apache-2.0). Voices are
reference clips in the serving config's ``model_kwargs.voices_dir``; cloning a voice from
a request's own clip is not served yet.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import FINAL_PREFILL, PartitionDefinition
from mstar.distributed.base import ShardingConfig
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    KVSpec,
    NodeResourceSpec,
    PagedKVConfig,
    PositionConfig,
    PositionSpec,
    RaggedAttentionConfig,
    RaggedKeyPrefixAttentionSpec,
    ResourceReqConfig,
    SamplerSpec,
    SamplingReqConfig,
)
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import KVLabel, LabelLayout, Model, ProcessPromptOutput, Span, WalkStep
from mstar.model.components.token2wav.keys import Token2WavKeys
from mstar.model.components.token2wav.model import DIT_COMPUTE_DTYPE, HOP, WINDOW, load_token2wav
from mstar.model.components.token2wav.resources import check_token2wav_slots, token2wav_resources
from mstar.model.components.token2wav.submodule import Token2WavSubmodule
from mstar.model.multimodal import TEXT, PromptPart, chat_turns, check_attachments, parts_from_modalities
from mstar.model.step_audio2 import processor
from mstar.model.step_audio2.components.audio import StepAudio2Audio, request_layout
from mstar.model.step_audio2.config import (
    AUDIO_ATTN,
    AUDIO_EMBEDS,
    AUDIO_MEL_FRAMES,
    AUDIO_MELS,
    DECODE,
    DECODE_LOOP,
    DECODE_MIXED,
    DECODE_SPOKEN,
    DEFAULT_SAMPLING,
    DEFAULT_SYSTEM_PROMPT,
    LLM_ATTN,
    LLM_KV,
    LLM_POS,
    LLM_SAMPLER,
    PREFILL_AUDIO,
    PREFILL_TEXT,
    SAMPLING_KEYS,
    SPEECH_BASE,
    SPEECH_IDS,
    SPEECH_MAX_TOKENS,
    T2W_CHUNK,
    T2W_KV,
    T2W_STATE,
    TEXT_VOCAB_END,
    StepAudio2Config,
    audio_tokens,
)
from mstar.model.step_audio2.token2wav import CONFORMER_REPLY_TOKENS, T2W_STREAMING
from mstar.model.step_audio2.weight_loader import load_audio_weights, load_llm_weights
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import ByteLevelDetokenizer, TextOutput
from mstar.streaming.chunk_policy import SlidingWindowChunkPolicy
from mstar.streaming.topology import Connection, PartitionTopology, StreamingGraphEdge

logger = logging.getLogger(__name__)

LLM = "LLM"
AUDIO = "audio_encoder"
TOKEN2WAV = "Token2Wav"

# async partitions: the main one runs the tower and the LLM; Token2Wav vocodes the
# spoken walk's speech ids as they arrive
MAIN = "main"

T2W_KEYS = Token2WavKeys(codes=SPEECH_IDS, state=T2W_STATE, kv=T2W_KV)
T2W_DEFAULT_SLOTS = 9
DEFAULT_VOICE = "default_female"
# A spoken reply's tokens: each is at most one speech id, so this many keep the vocoder's
# conformer caches within what they hold (``CONFORMER_REPLY_TOKENS``).
SPEECH_MAX_OUTPUT_TOKENS = CONFORMER_REPLY_TOKENS

REF_AUDIO_REFUSAL = (
    "Step-Audio2 does not clone a voice from ref_audio yet; choose one of its voices with 'voice'"
)


class SpokenTextDetokenizer:
    """A reply's text as the client gets it: a text token's raw bytes (byte-level BPE),
    and nothing for a speech id, ``<tts_pad>`` or any special token. The reference decodes
    only the ids below the first multimodal token (``stepaudio2.py:55``)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._bytes = ByteLevelDetokenizer(tokenizer, keep_special=True)
        self._skip = frozenset(i for i, t in tokenizer.added_tokens_decoder.items() if t.special)

    def token_bytes(self, token_id: int) -> bytes:
        if token_id >= TEXT_VOCAB_END or token_id in self._skip:
            return b""
        return self._bytes.token_bytes(token_id)

    def to_bytes(self, token_ids: list[int]) -> bytes:
        return b"".join(self.token_bytes(i) for i in token_ids)


def _resolve_metadata(repo_id: str, cache_dir: str | None) -> str:
    """Config and tokenizer files only; workers fetch weights."""
    if Path(repo_id).is_dir():
        return repo_id
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, cache_dir=cache_dir, allow_patterns=["*.json", "*.txt"])


def _voice_files(voices_dir: str | None) -> dict[str, Path]:
    """The ``.wav`` clips in ``voices_dir``, by file stem: the names a request picks."""
    if not voices_dir:
        raise ValueError(
            "Step-Audio2 needs model_kwargs.voices_dir in the serving config: a directory of reference "
            "clips (the GitHub repo's assets/default_female.wav and default_male.wav)"
        )
    root = Path(voices_dir).expanduser()
    voices = {p.stem: p for p in sorted(root.glob("*.wav"))}
    if not voices:
        raise ValueError(f"Step-Audio2's voices_dir {str(root)!r} holds no .wav clips")
    return voices


def _load_voice_clip(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """A voice clip as the reference's ``_prepare_prompt`` reads it: the first channel at
    16 kHz (``s3tokenizer.load_audio``) for its tokens and x-vector, and the channels'
    mean at 24 kHz, resampled from the clip's own rate, for its mel."""
    import soundfile as sf
    import torchaudio

    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.T.copy())

    def at(target: int, x: torch.Tensor) -> torch.Tensor:
        return x if rate == target else torchaudio.transforms.Resample(rate, target)(x)

    return at(16000, audio)[0], at(24000, audio.mean(dim=0, keepdim=True))[0]


class StepAudio2Model(Model):
    PREPROCESS_TORCH_THREADS = 4
    # the clips' mel frames and a spoken reply's token cap, which process_prompt settles
    INTERNAL_KWARGS = (AUDIO_MEL_FRAMES, SPEECH_MAX_TOKENS)

    def __init__(
        self, model_path_hf: str, cache_dir: str | None = None, max_seq_len: int | None = None,
        voices_dir: str | None = None, **kwargs: Any,
    ):
        from transformers import Qwen2TokenizerFast

        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.local_dir = _resolve_metadata(model_path_hf, cache_dir)
        self.config = StepAudio2Config.from_hf(self.local_dir)
        limit = self.config.llm.max_position_embeddings
        self.max_seq_len = limit if max_seq_len is None else max_seq_len
        if type(self.max_seq_len) is not int or not 0 < self.max_seq_len <= limit:
            raise ValueError(f"max_seq_len must be in 1..{limit} (max_position_embeddings), got {max_seq_len!r}")
        # the checkpoint's own class is Qwen2's (``tokenizer_config.json``); AutoTokenizer would
        # ask to run the checkpoint's remote code for its config
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(self.local_dir)
        self.voices = _voice_files(voices_dir)
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}
        self._detokenizer: SpokenTextDetokenizer | None = None

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        """A spoken reply holds a slot of each ``t2w_*`` resource, so one count sizes them
        all: ``t2w_state`` has one more (its sink)."""
        check_token2wav_slots(T2W_KEYS, config.get("resources") or {}, T2W_DEFAULT_SLOTS, config_path)

    def checkpoint_path(self) -> str | None:
        return self.local_dir

    def load_audio(self, filepath: str, device: str):
        """The default decode, resampled to 16 kHz as the reference's ``utils.load_audio``
        does (``transforms.Resample``; ``functional.resample`` moves its log-mel by up to
        3e-5)."""
        import torchaudio

        from mstar.api_server.media_io import check_resample_rate
        from mstar.model.base import TensorAndMetadata
        from mstar.model.components.audio_features import _decode
        from mstar.model.media_limits import decoding_upload

        rate = self.AUDIO_INPUT_RATE
        with decoding_upload(filepath, "the audio", ffmpeg=True):
            audio, sr = _decode(filepath, rate, self.MAX_AUDIO_SECONDS)
        check_resample_rate(sr, rate)
        if sr != rate:
            audio = torchaudio.transforms.Resample(sr, rate)(audio)
        return TensorAndMetadata(data=audio.contiguous(), metadata=dict(sample_rate=rate, num_channels=1))

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        llm, audio = self.config.llm, self.config.audio
        return [
            KVSpec(
                resource_key=LLM_KV, nodes={LLM},
                config=PagedKVConfig(
                    num_layers=llm.num_hidden_layers,
                    num_kv_heads=llm.num_key_value_heads,
                    head_dim=llm.head_dim,
                    max_seq_len=self.max_seq_len,
                    num_qo_heads=llm.num_attention_heads,
                ),
            ),
            AttentionSpec(resource_key=LLM_ATTN, nodes={LLM}, config=AttentionConfig(kv_cache=LLM_KV)),
            PositionSpec(
                resource_key=LLM_POS, nodes={LLM},
                config=PositionConfig(kv_cache=LLM_KV, rope_theta=llm.rope_theta),
            ),
            SamplerSpec(
                resource_key=LLM_SAMPLER, nodes={LLM}, vocab_size=llm.vocab_size, enable_repetition_penalty=True,
            ),
            RaggedKeyPrefixAttentionSpec(
                resource_key=AUDIO_ATTN, nodes={AUDIO},
                config=RaggedAttentionConfig(
                    num_qo_heads=audio.n_audio_head, num_kv_heads=audio.n_audio_head, head_dim=audio.head_dim,
                    # a request's 25 s clips; more run eagerly once the tower captures
                    max_segments_per_request=8,
                ),
            ),
            *token2wav_resources(T2W_KEYS, TOKEN2WAV, self._voice_seconds(), T2W_DEFAULT_SLOTS, T2W_STREAMING),
        ]

    def _voice_seconds(self) -> tuple[float, float]:
        """The shortest and longest voice clip."""
        import soundfile as sf

        seconds = [sf.info(str(p)).duration for p in self.voices.values()]
        return min(seconds), max(seconds)

    def get_request_resource_configs(
        self, input_modalities: list[str], output_modalities: list[str], model_kwargs: dict,
    ) -> dict[str, ResourceReqConfig]:
        knobs = {**DEFAULT_SAMPLING, **{k: model_kwargs[k] for k in SAMPLING_KEYS if k in model_kwargs}}
        return {LLM_SAMPLER: SamplingReqConfig(**knobs)}

    def request_kwargs(self) -> frozenset[str]:
        # ref_audio is read only to refuse it (``process_prompt``)
        return frozenset({*SAMPLING_KEYS, "voice", "system_prompt", "ref_audio"})

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        def prefill(inputs: list[str]) -> GraphNode:
            return GraphNode(
                name=LLM, input_names=inputs,
                outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", persist=True, output_modality="text")],
            )

        def decode(spoken: bool) -> Loop:
            # A text reply's only output is its token, which the LLM keeps on the device. A
            # spoken reply's also streams to the vocoder, an item per speech id.
            stream = [StreamingGraphEdge(next_node=TOKEN2WAV, name=SPEECH_IDS, target_partition=TOKEN2WAV)]
            return Loop(
                name=DECODE_LOOP,
                section=GraphNode(
                    name=LLM,
                    input_names=["text_inputs"],
                    outputs=[
                        GraphEdge(next_node=LLM, name="text_inputs"),
                        GraphEdge(next_node=EMIT_TO_CLIENT, name="text_inputs", output_modality="text"),
                        *(stream if spoken else ()),
                    ],
                ),
                # a safety bound; the per-request budget is `get_loop_caps`'
                max_iters=self.max_seq_len,
                outputs=[],
            )

        audio = GraphNode(name=AUDIO, input_names=[AUDIO_MELS], outputs=[GraphEdge(next_node=LLM, name=AUDIO_EMBEDS)])
        return {
            PREFILL_TEXT: prefill(["text_inputs"]),
            PREFILL_AUDIO: Sequential([audio, prefill(["text_inputs", AUDIO_EMBEDS])]),
            DECODE: decode(spoken=False),
            DECODE_SPOKEN: decode(spoken=True),
            T2W_CHUNK: GraphNode(
                name=TOKEN2WAV, input_names=[SPEECH_IDS],
                outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name=T2W_KEYS.audio, output_modality="audio")],
                # Not speculated: a window's step syncs on the host (the inverse STFT), as
                # MiniCPM-o's does.
                enable_async_scheduling=False,
            ),
        }

    def get_combined_graph_walks(self) -> dict[str, dict[str, set[str]]]:
        # text and spoken replies decode in one step; a text row still loops back
        return {LLM: {DECODE_MIXED: {DECODE, DECODE_SPOKEN}}}

    def get_partitions(self) -> list[PartitionDefinition]:
        main = set(self.get_graph_walk_graphs()) - {T2W_CHUNK}
        return [
            PartitionDefinition(name=MAIN, graph_walks=main),
            PartitionDefinition(name=TOKEN2WAV, graph_walks={T2W_CHUNK}),
        ]

    def get_partition_topology(self) -> PartitionTopology:
        return PartitionTopology(
            partitions=[MAIN, TOKEN2WAV],
            connections=[Connection(
                from_partition=MAIN, to_partition=TOKEN2WAV, edge_name=SPEECH_IDS,
                # the reference client's windows: 25 new codes and the 3 the flow looks
                # ahead to, then whatever is left once the reply ends
                chunk_policy_factory=lambda: SlidingWindowChunkPolicy(window=WINDOW, stride=HOP),
            )],
        )

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _messages(self, parts: list[PromptPart], system_prompt: str | None, speech: bool) -> list[dict]:
        """The reference's chat messages (``StepAudio2.apply_chat_template``): a turn per run
        of one role's parts, a text-only turn as one string (the reference tokenizes it as one
        piece), and the open assistant turn, ``<tts_start>`` for a spoken reply."""
        messages = []
        for turn in chat_turns(parts):
            if all(p.modality == TEXT for p in turn.parts):
                content: Any = "\n".join(p.text or "" for p in turn.parts)
            else:
                content = [
                    {"type": "text", "text": p.text or ""} if p.modality == TEXT
                    else {"type": "audio", "audio": p.index}
                    for p in turn.parts
                ]
            messages.append({"role": turn.role, "content": content})
        if not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT})
        messages.append({"role": "assistant", "content": "<tts_start>" if speech else None, "eot": False})
        return messages

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        prompt_parts: list[PromptPart] | None = None,
        **kwargs,
    ) -> ProcessPromptOutput:
        unsupported = set(output_modalities) - {TEXT, "audio"}
        if unsupported:
            raise ValueError(f"Step-Audio2 outputs text and audio; got {sorted(unsupported)}")
        if kwargs.get("ref_audio"):
            raise ValueError(REF_AUDIO_REFUSAL)
        speech = "audio" in output_modalities
        if speech:
            self.check_voice(kwargs.get("voice"))
        tensors = tensors or {}
        parts = parts_from_modalities(
            input_modalities,
            [p.text or "" for p in prompt_parts if p.modality == TEXT] if prompt_parts is not None else prompt,
            roles=[p.role for p in prompt_parts] if prompt_parts is not None else None,
        )
        bad = {p.modality for p in parts} - {TEXT, "audio"}
        if bad:
            raise ValueError(f"Step-Audio2 takes text and audio inputs; got {sorted(bad)}")
        audios = [a.reshape(-1).float().cpu() for a in tensors.get("audio_inputs", [])]
        check_attachments(parts, {"audio": len(audios)})
        if not any(p.text for p in parts if p.modality == TEXT) and not audios:
            raise ValueError("Step-Audio2 got a request with no text and no audio")

        pieces, mels = processor.render(
            self._messages(parts, kwargs.get("system_prompt"), speech), lambda index: audios[index],
        )
        ids = processor.tokenize(pieces, lambda text: self.tokenizer(text, add_special_tokens=False)["input_ids"])
        result: NameToTensorList = {"text_inputs": [torch.tensor(ids, dtype=torch.long)]}
        metadata: dict[str, Any] = {}
        if mels:
            frames = [m.shape[-1] for m in mels]
            request_layout(frames)  # refuses a clip too short to encode, here rather than on the worker
            processor.audio_positions(ids, [audio_tokens(n) for n in frames])
            # bf16 is what the tower casts them to before its convolutions, so the cast is exact
            result[AUDIO_MELS] = [torch.cat(mels, dim=-1).to(torch.bfloat16)]
            metadata[AUDIO_MEL_FRAMES] = frames
        if speech:
            requested = kwargs.get("max_output_tokens")
            if requested is not None and int(requested) > SPEECH_MAX_OUTPUT_TOKENS:
                raise ValueError(
                    f"a spoken Step-Audio2 reply holds at most {SPEECH_MAX_OUTPUT_TOKENS} tokens (the vocoder's "
                    f"caches); got max_output_tokens={requested}"
                )
            metadata[SPEECH_MAX_TOKENS] = SPEECH_MAX_OUTPUT_TOKENS
        # the prefill writes every prompt row, placeholders included, then a row per decode step
        walk = PREFILL_AUDIO if mels else PREFILL_TEXT
        layout = LabelLayout(spans=(Span("rows", walk, len(ids), len(ids), "text_inputs"),), growth=((DECODE_LOOP, 1),))
        metadata["kv_layout"] = {LLM_KV: {"main": layout}}
        return ProcessPromptOutput(result, metadata)

    def get_max_output_tokens(self, **model_kwargs) -> int:
        cap = super().get_max_output_tokens(**model_kwargs)
        return min(cap, model_kwargs.get(SPEECH_MAX_TOKENS, cap))

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        # the prefill samples the first token, so the loop runs one fewer
        return {DECODE_LOOP: max(1, self.get_max_output_tokens(**model_kwargs) - 1)}

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        return {LLM_KV: {"main": KVLabel(growth=((DECODE, DECODE_LOOP), (DECODE_SPOKEN, DECODE_LOOP)))}}

    def get_voices(self) -> list[str]:
        return list(self.voices)

    def get_default_voice(self) -> str:
        return DEFAULT_VOICE if DEFAULT_VOICE in self.voices else next(iter(self.voices))

    def check_voice(self, voice: str | None) -> str:
        name = voice or self.get_default_voice()
        if name not in self.voices:
            raise ValueError(f"unknown Step-Audio2 voice {name!r}; choose one of {sorted(self.voices)}")
        return name

    def text_output(self) -> TextOutput:
        if self._detokenizer is None:
            self._detokenizer = SpokenTextDetokenizer(self.tokenizer)
        return TextOutput(self._detokenizer, self.config.stop_token_ids)

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        if modality == "audio":
            # the reference's PCM (``token2wav.py:164-168``)
            pcm = (output.reshape(-1).float().clamp(-1.0, 1.0) * 32767.0).to(torch.int16)
            return pcm.cpu().numpy().tobytes()
        return super().postprocess(output, modality, request_kwargs)

    # ------------------------------------------------------------------
    # Walks
    # ------------------------------------------------------------------

    def walk_plan(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict,
    ) -> list[WalkStep]:
        """Main: one prefill walk over the whole prompt, then decode fed the prefill's
        token, on ``decode_spoken`` for a spoken reply. Token2Wav runs a chunk per streamed
        window, and only for a spoken reply."""
        speech = "audio" in output_modalities
        if partition_name == TOKEN2WAV:
            if not speech:
                return []
            return [WalkStep(T2W_CHUNK, step_metadata={T2W_KEYS.voice: self.check_voice(model_kwargs.get("voice"))})]
        settled = {k: model_kwargs[k] for k in self.INTERNAL_KWARGS if k in model_kwargs}
        walk = PREFILL_AUDIO if input_signals.get(AUDIO_MELS) else PREFILL_TEXT
        return [
            WalkStep(walk, step_metadata={FINAL_PREFILL: True, **settled}),
            WalkStep(DECODE_SPOKEN if speech else DECODE, bind={"text_inputs": "new_token"}),
        ]

    # ------------------------------------------------------------------
    # Submodules
    # ------------------------------------------------------------------

    def get_default_sharding_config(self) -> ShardingConfig:
        return ShardingConfig(groups=[], tp_enabled_nodes={LLM}, shard_dim={})

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None, sp_group=None, **kwargs,
    ) -> NodeSubmodule | None:
        if node_name not in self._submodule_cache:
            self._submodule_cache[node_name] = self._create_submodule(
                node_name, device, tp_group, autocast_dtype or torch.bfloat16,
            )
        return self._submodule_cache[node_name]

    def _weights_dir(self) -> str:
        if Path(self.model_path_hf).is_dir():
            return self.model_path_hf
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=self.model_path_hf, cache_dir=self.cache_dir,
            allow_patterns=["*.safetensors", "*.json", "token2wav/*"],
        )

    @staticmethod
    def _build(make, dtype: torch.dtype, device: str) -> torch.nn.Module:
        """Construct on ``meta``, then give it uninitialized ``dtype`` storage on ``device``
        for the loader to fill (it raises on any parameter left unfilled)."""
        with torch.device("meta"):
            module = make()
        return module.to(dtype).to_empty(device=device)

    def _create_token2wav(self, weights: str, device: str) -> NodeSubmodule:
        t2w = load_token2wav(str(Path(weights) / "token2wav"), device=device, streaming=T2W_STREAMING)
        # The reference prepares voice prompts on the CPU (but its tokenizer, on CUDA, which
        # lands on the same tokens up to a near-tie); it is once per voice.
        t2w.voice_encoder.cpu()
        voices = {
            name: t2w.prepare_voice(t2w.voice_encoder(*_load_voice_clip(path))) for name, path in self.voices.items()
        }
        logger.info("Loaded Step-Audio2 token2wav with voices %s onto %s", sorted(voices), device)
        return Token2WavSubmodule(
            t2w.requires_grad_(False), T2W_KEYS, voices, eos_code=None, dit_dtype=DIT_COMPUTE_DTYPE,
            code_offset=SPEECH_BASE,
        )

    def _create_submodule(self, node_name: str, device: str, tp_group, dtype: torch.dtype) -> NodeSubmodule | None:
        from mstar.model.step_audio2.submodules import AudioEncoderSubmodule, LLMSubmodule

        if node_name not in (LLM, AUDIO, TOKEN2WAV):
            return None
        weights = self._weights_dir()
        if node_name == TOKEN2WAV:
            return self._create_token2wav(weights, device)
        if node_name == AUDIO:
            tower = self._build(lambda: StepAudio2Audio(self.config.audio, attn_key=AUDIO_ATTN), dtype, device)
            load_audio_weights(tower, weights, device)
            return AudioEncoderSubmodule(tower.requires_grad_(False).eval(), self.config)
        from mstar.model.components.qwen3_lm import Qwen3DenseLM

        lm = self._build(
            lambda: Qwen3DenseLM(
                self.config.llm, attn_key=LLM_ATTN, kv_key=LLM_KV, pos_key=LLM_POS, comm_group=tp_group,
            ),
            dtype, device,
        )
        load_llm_weights(lm, weights, device)
        logger.info("Loaded Step-Audio2 LLM onto %s", device)
        return LLMSubmodule(lm.requires_grad_(False).eval(), self.config)
