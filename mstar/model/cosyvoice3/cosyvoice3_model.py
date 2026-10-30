"""Fun-CosyVoice3 text-to-speech with voice cloning for M*.

Architecture (four nodes, two asynchronous partitions)::

    speaker_encoder  reference clip -> speech tokens for the LM prompt
                     (Chatterbox's S3 tokenizer with the v3 weights)
    llm              Qwen2 0.5B over [sos | prompt text + text | task_id |
                     prompt speech] -> speech tokens, 25 Hz, repetition-aware
                     sampling
    flow             speech tokens -> mel (DiT flow matching conditioned on the
                     reference mel and x-vector), settled in 25-token chunks
    hift             mel -> 24 kHz PCM16 (causal HiFT, bounded-window streaming)

Walks and partitions::

    LM partition:     prefill        (cross-lingual / instruct)  llm
                      prefill_voice  (zero-shot, with ref_text)  speaker_encoder -> llm
                      decode         Loop over llm, one speech token per step
    Token2Wav:        token2wav      flow -> hift, fed by the ``speech_tokens``
                                     stream from the LM

Every request needs a reference voice: an uploaded clip (``ref_audio``) or a
preset from ``voices_dir``; the checkpoint ships none. With its transcript
(``ref_text``, or ``<voice>.txt`` beside a preset) the LM is prompted with the
clip's speech (zero-shot cloning); without it the LM sees only the text and the
clip conditions the flow (the reference's cross-lingual mode). ``instruct``
replaces the transcript with a style instruction (``instruct2``).

The reference implementation (``cosyvoice/cli/{cosyvoice,frontend,model}.py``)
was read for every contract here; nothing of it runs at serve time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    PartitionDefinition,
    StreamingConnectionState,
)
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    KVReqConfig,
    KVSpec,
    NodeResourceSpec,
    PagedKVConfig,
    PositionConfig,
    PositionSpec,
    ResourceReqConfig,
)
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Sequential, TensorPointerInfo
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
from mstar.model.chatterbox.chatterbox_model import (
    ChatterboxModel,
    _flag,
    _integer,
    _number,
    voice_key_for,
)
from mstar.model.chatterbox.components.audio_frontend import resample, trim_silence
from mstar.model.chatterbox.loader import resolve_snapshot
from mstar.model.cosyvoice3.config import (
    END_OF_PROMPT,
    END_OF_PROMPT_ID,
    FLOW_NODE,
    HIFT_NODE,
    LM_ATTN,
    LM_KV,
    LM_LABEL,
    LM_NODE,
    LM_POS,
    SAMPLE_RATE,
    SPEAKER_ENCODER_NODE,
    SYSTEM_PROMPT,
    TOKENIZER_SR,
    CosyVoice3Config,
)
from mstar.model.cosyvoice3.submodules import (
    AUDIO_CHUNK,
    DECODE_LOOP,
    MAX_SPEECH_TOKENS,
    MEL,
    MEL_FINAL,
    MIN_SPEECH_TOKENS,
    PREV_TOKEN,
    PROMPT_TOKENS,
    REF_AUDIO,
    SPEECH_TOKENS,
    TEXT_INPUTS,
    VOICE_KEY,
    reference_tokens,
)
from mstar.model.submodule_base import NodeSubmodule
from mstar.streaming.chunk_policy import ScheduledLeftContextChunkPolicy
from mstar.streaming.topology import Connection, PartitionTopology, StreamingGraphEdge

logger = logging.getLogger(__name__)

LM_PARTITION = "LM"
TOKEN2WAV_PARTITION = "Token2Wav"

_LM_KNOBS = ("temperature", "top_p", "top_k", "ras_window", "ras_tau", "ignore_eos")
_FLOW_KNOBS = ("n_cfm_timesteps",)
# settled by process_prompt and folded into model_kwargs, so the nodes read
# them from step_metadata instead of from device tensors
_LM_FACTS = (MIN_SPEECH_TOKENS, MAX_SPEECH_TOKENS, VOICE_KEY)
_FLOW_FACTS = (VOICE_KEY,)
# the cloning mode process_prompt built the prompt for; the conductor picks the
# LM walk from it rather than deciding again from the caller's input_modalities
PROMPT_MODE = "prompt_mode"

# A clip quieter or shorter than these clones noise
MIN_REFERENCE_SECONDS = 0.5
MIN_REFERENCE_PEAK = 1e-3
MAX_CFM_TIMESTEPS = 100
MAX_SEED = 2**63 - 1
MAX_TEMPERATURE = 5.0
MAX_RAS_WINDOW = 64


class CosyVoice3Model(Model):
    """Model contract: prompt processing, graph, partitions, resources and the
    per-partition state machine. No GPU compute lives here."""

    INTERNAL_KWARGS = (MIN_SPEECH_TOKENS, MAX_SPEECH_TOKENS, VOICE_KEY, PROMPT_MODE)
    LENGTH_KWARGS = ("max_output_tokens", "max_new_tokens")

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        voices_dir: str | None = None,
        llm_weights: str | None = None,
        stream_hop_tokens: int | None = None,
        stream_max_hop_tokens: int | None = None,
        stream_poll_tokens: int | None = None,
        hift_context_frames: int | None = None,
        flow_estimator_dtype: str | None = None,
        lm_dtype: str | None = None,
        max_new_tokens_limit: int | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self._lm_dtype = _parse_dtype(lm_dtype) if lm_dtype else torch.bfloat16
        if self._lm_dtype == torch.float32:
            raise ValueError("lm_dtype float32 cannot be served: the paged attention runs in bfloat16/float16 only")
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        cfg = self.config = CosyVoice3Config()
        if llm_weights is not None:
            cfg.llm_weights = str(llm_weights)
        if stream_hop_tokens is not None:
            cfg.stream_hop_tokens = _integer("stream_hop_tokens", stream_hop_tokens, low=0)
        if stream_max_hop_tokens is not None:
            cfg.stream_max_hop_tokens = _integer("stream_max_hop_tokens", stream_max_hop_tokens, low=1)
        if stream_poll_tokens is not None:
            cfg.stream_poll_tokens = _integer("stream_poll_tokens", stream_poll_tokens, low=1)
        if hift_context_frames is not None:
            cfg.hift_context_frames = _integer("hift_context_frames", hift_context_frames, low=0)
        if flow_estimator_dtype is not None:
            _parse_dtype(str(flow_estimator_dtype))
            cfg.flow_estimator_dtype = str(flow_estimator_dtype)
        if max_new_tokens_limit is not None:
            cfg.max_new_tokens_limit = _integer(
                "max_new_tokens_limit",
                max_new_tokens_limit,
                low=1,
                high=cfg.max_speech_tokens,
            )
        if cfg.stream_hop_tokens and cfg.stream_hop_tokens % cfg.flow.chunk_tokens:
            raise ValueError(
                f"stream_hop_tokens must be a multiple of the flow's {cfg.flow.chunk_tokens}-token attention chunk"
            )
        self.voices_dir = Path(voices_dir) if voices_dir else None
        self.local_dir = resolve_snapshot(model_path_hf, cache_dir)
        self.tokenizer = self._build_text_tokenizer()
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}
        self._shared: dict[str, Any] = {}

    def _build_text_tokenizer(self):
        from mstar.model.cosyvoice3.components.text import CosyVoice3TextTokenizer

        return CosyVoice3TextTokenizer(Path(self.local_dir) / self.config.text_tokenizer_dir)

    # -----------------------------------------------------------------------
    # Resources
    # -----------------------------------------------------------------------

    @property
    def _speech_limit(self) -> int:
        return self.config.max_new_tokens_limit or self.config.max_speech_tokens

    def _max_seq_len(self) -> int:
        cfg = self.config
        prompt_speech = int(cfg.max_reference_seconds * 25)
        return 2 + cfg.max_text_tokens + prompt_speech + self._speech_limit

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        """Refuse a request cap the LM KV pool cannot hold: nothing preempts a
        stream once the pool is full."""
        cap = config.get("max_concurrent_requests")
        if cap is None:
            logger.warning("%s sets no max_concurrent_requests: a burst can fill the LM KV cache", config_path)
            return
        kv = next(s for s in self.get_node_resources() if s.resource_key == LM_KV).config
        overrides = (config.get("resources") or {}).get(LM_KV) or {}
        page_size = int(overrides.get("page_size", kv.page_size))
        max_num_pages = int(overrides.get("max_num_pages", kv.max_num_pages))
        need = 1 + int(cap) * -(-self._max_seq_len() // page_size)
        if need > max_num_pages:
            raise ValueError(
                f"{config_path}: {cap} concurrent requests at {self._max_seq_len()} tokens each need "
                f"{need} pages of resources.{LM_KV}, which has {max_num_pages}"
            )

    def get_node_resources(self) -> list[NodeResourceSpec]:
        lm = self.config.lm
        kv = PagedKVConfig(
            num_layers=lm.num_hidden_layers,
            num_kv_heads=lm.num_key_value_heads,
            head_dim=lm.head_dim,
            max_seq_len=self._max_seq_len(),
            num_qo_heads=lm.num_attention_heads,
        )
        return [
            KVSpec(resource_key=LM_KV, nodes={LM_NODE}, config=kv),
            AttentionSpec(resource_key=LM_ATTN, nodes={LM_NODE}, config=AttentionConfig(kv_cache=LM_KV)),
            PositionSpec(
                resource_key=LM_POS,
                nodes={LM_NODE},
                config=PositionConfig(kv_cache=LM_KV, rope_theta=lm.rope_theta),
            ),
        ]

    def get_request_resource_configs(
        self,
        partition_fwd_args: dict[str, ForwardPassArgs],
        model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        del partition_fwd_args, model_kwargs
        return {LM_KV: KVReqConfig(needed_labels=[LM_LABEL])}

    def resolve_generation_kwargs(self, model_kwargs: dict | None) -> dict[str, Any]:
        """Every public knob, defaulted and range-checked, so a bad value is a
        400 from ``process_prompt`` rather than a hung request."""
        mk = dict(model_kwargs or {})
        s = self.config.sampling
        top_k = _integer("top_k", mk.get("top_k", s.top_k), low=1, high=128)
        knobs = {
            "temperature": _number("temperature", mk.get("temperature", s.temperature), low=0.0, high=MAX_TEMPERATURE),
            "top_p": _number("top_p", mk.get("top_p", s.top_p), low=0.0, high=1.0),
            "top_k": top_k,
            "ras_window": _integer("ras_window", mk.get("ras_window", s.ras_window), low=0, high=MAX_RAS_WINDOW),
            "ras_tau": _number("ras_tau", mk.get("ras_tau", s.ras_tau), low=0.0, high=1.0),
            "ignore_eos": _flag("ignore_eos", mk.get("ignore_eos", False)),
            "n_cfm_timesteps": _integer(
                "n_cfm_timesteps",
                mk.get("n_cfm_timesteps", self.config.flow.cfm.n_timesteps),
                low=1,
                high=MAX_CFM_TIMESTEPS,
            ),
        }
        if knobs["ras_window"] > self.config.sampling.ras_window and knobs["ras_window"] != 0:
            # the captured decode carries a history of the default window
            raise ValueError(f"ras_window is at most {self.config.sampling.ras_window} on this deployment")
        limit = mk.get("max_new_tokens", mk.get("max_output_tokens"))
        knobs["max_new_tokens"] = (
            None if limit is None else _integer("max_new_tokens", limit, low=1, high=self._speech_limit)
        )
        if mk.get("seed") is not None:
            _integer("seed", mk["seed"], low=0, high=MAX_SEED)
        if mk.get("ref_audio") is not None:
            # only the OpenAI adapter turns a ref_audio string into an upload
            raise ValueError("ref_audio cannot be a model kwarg; upload the clip as a file")
        for key in ("ref_text", "instruct"):
            if mk.get(key) is not None and not isinstance(mk[key], str):
                raise ValueError(f"{key} must be a string")
        return knobs

    def get_max_output_tokens(self, **model_kwargs: Any) -> int:
        return self.resolve_generation_kwargs(model_kwargs)["max_new_tokens"] or self._speech_limit

    def get_loop_caps(self, **model_kwargs: Any) -> dict[str, int]:
        # the stop check counts the prefill's token, so decode runs one iteration fewer than the budget
        budget = self.get_max_output_tokens(**model_kwargs)
        if model_kwargs.get(MAX_SPEECH_TOKENS) is not None:
            budget = min(budget, int(model_kwargs[MAX_SPEECH_TOKENS]))
        return {DECODE_LOOP: max(1, budget - 1)}

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        return {LM_KV: {LM_LABEL: KVLabel(growth=(("decode", DECODE_LOOP),))}}

    @staticmethod
    def _lm_layout(walk: str, text_tokens: int, prompt_tokens: int) -> LabelLayout:
        """What the LM writes: ``[sos | text | task_id | prompt speech]`` as ``build_prefill_embeds``
        lays it out, then a row per decode step."""
        return LabelLayout(
            spans=(
                Span("rows", walk, 1, 1, TEXT_INPUTS),
                Span("ids", walk, text_tokens, text_tokens, TEXT_INPUTS),
                Span("rows", walk, 1 + prompt_tokens, 1 + prompt_tokens, PROMPT_TOKENS),
            ),
            growth=((DECODE_LOOP, 1),),
        )

    # -----------------------------------------------------------------------
    # Graph
    # -----------------------------------------------------------------------

    def _lm_outputs(self, first: bool) -> list[GraphEdge]:
        edges: list[GraphEdge] = [
            StreamingGraphEdge(next_node=FLOW_NODE, name=SPEECH_TOKENS, target_partition=TOKEN2WAV_PARTITION),
        ]
        if first:
            edges.insert(
                0,
                GraphEdge(
                    next_node=EMPTY_DESTINATION,
                    name=SPEECH_TOKENS,
                    conductor_new_token=True,
                    persist=True,
                ),
            )
        else:
            edges.insert(0, GraphEdge(next_node=LM_NODE, name=PREV_TOKEN))
        return edges

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill = GraphNode(name=LM_NODE, input_names=[TEXT_INPUTS], outputs=self._lm_outputs(first=True))
        prefill_voice = Sequential(
            [
                GraphNode(
                    name=SPEAKER_ENCODER_NODE,
                    input_names=[REF_AUDIO],
                    outputs=[GraphEdge(next_node=LM_NODE, name=PROMPT_TOKENS)],
                ),
                GraphNode(
                    name=LM_NODE,
                    input_names=[TEXT_INPUTS, PROMPT_TOKENS],
                    outputs=self._lm_outputs(first=True),
                ),
            ]
        )
        decode = Loop(
            name="decode_loop",
            section=GraphNode(name=LM_NODE, input_names=[PREV_TOKEN], outputs=self._lm_outputs(first=False)),
            max_iters=self._speech_limit,
            outputs=[],
        )
        token2wav = Sequential(
            [
                GraphNode(
                    name=FLOW_NODE,
                    input_names=[SPEECH_TOKENS, REF_AUDIO],
                    outputs=[GraphEdge(next_node=HIFT_NODE, name=MEL), GraphEdge(next_node=HIFT_NODE, name=MEL_FINAL)],
                ),
                GraphNode(
                    name=HIFT_NODE,
                    input_names=[MEL, MEL_FINAL],
                    outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name=AUDIO_CHUNK, output_modality="audio")],
                ),
            ]
        )
        return {"prefill": prefill, "prefill_voice": prefill_voice, "decode": decode, "token2wav": token2wav}

    def get_partitions(self) -> list[PartitionDefinition]:
        return [
            PartitionDefinition(
                name=LM_PARTITION,
                graph_walks={"prefill", "prefill_voice", "decode"},
                initial_walk="prefill",
                producer_partitions=[],
            ),
            PartitionDefinition(
                name=TOKEN2WAV_PARTITION,
                graph_walks={"token2wav"},
                initial_walk=None,
                producer_partitions=[LM_PARTITION],
            ),
        ]

    def _chunk_policy(self):
        """Every ``stream_poll_tokens`` tokens the flow node runs on the whole
        token history: all delivered tokens ride along as left context, so the
        node keeps no copy of its own and decides itself what can settle."""
        cfg = self.config
        return ScheduledLeftContextChunkPolicy(
            schedule=(),
            chunk=cfg.stream_poll_tokens,
            left_context=self._speech_limit + 1,
        )

    def get_partition_topology(self) -> PartitionTopology:
        return PartitionTopology(
            partitions=[LM_PARTITION, TOKEN2WAV_PARTITION],
            connections=[
                Connection(
                    from_partition=LM_PARTITION,
                    to_partition=TOKEN2WAV_PARTITION,
                    edge_name=SPEECH_TOKENS,
                    chunk_policy_factory=self._chunk_policy,
                )
            ],
        )

    # -----------------------------------------------------------------------
    # Prompt processing (API data worker)
    # -----------------------------------------------------------------------

    def load_audio(self, filepath: str, device: str) -> TensorAndMetadata:
        """A reference clip as 24 kHz mono float32."""
        audio = ChatterboxModel._decode_audio(filepath).to(device)
        return TensorAndMetadata(data=audio, metadata=dict(sample_rate=SAMPLE_RATE, num_channels=1))

    def _preset_paths(self, voice: str) -> tuple[Path, str | None]:
        """A preset's clip and, when ``<voice>.txt`` exists beside it, its transcript."""
        if self.voices_dir is None:
            raise ValueError(
                f"Unknown voice {voice!r}: CosyVoice3 has no built-in voice and no voices_dir is "
                "configured; upload a reference clip (ref_audio, with its transcript in ref_text)"
            )
        # checked before anything is opened, so a name cannot probe the disk
        if not voice or voice.startswith(".") or any(c in voice for c in "/\\\0"):
            raise ValueError(f"Invalid voice name {voice!r}")
        for ext in ("", ".wav", ".flac", ".mp3", ".ogg", ".m4a"):
            path = self.voices_dir / f"{voice}{ext}"
            try:
                found = path.is_file() and path.suffix != ".txt"
            except OSError:  # a name the filesystem refuses, e.g. longer than NAME_MAX
                found = False
            if found:
                text = path.with_suffix(".txt")
                return path, text.read_text(encoding="utf-8").strip() if text.is_file() else None
        available = sorted({p.stem for p in self.voices_dir.iterdir() if p.is_file() and p.suffix != ".txt"})
        raise ValueError(f"Unknown voice {voice!r}; presets: {available}")

    def get_voices(self) -> list[str] | None:
        if self.voices_dir is None or not self.voices_dir.is_dir():
            return None
        return sorted({p.stem for p in self.voices_dir.iterdir() if p.is_file() and p.suffix != ".txt"})

    def _prepare_reference(self, wav: torch.Tensor) -> torch.Tensor:
        wav = wav.detach().to("cpu", torch.float32).reshape(-1)
        if wav.numel() == 0:
            raise ValueError("Reference audio is empty")
        if not torch.isfinite(wav).all():
            raise ValueError("Reference audio has NaN or infinite samples")
        max_len = int(self.config.max_reference_seconds * SAMPLE_RATE)
        if wav.numel() > max_len:
            raise ValueError(
                f"Reference audio is {wav.numel() / SAMPLE_RATE:.1f} s; the speech tokenizer takes at most "
                f"{self.config.max_reference_seconds:.0f} s"
            )
        sound = trim_silence(resample(wav, SAMPLE_RATE, TOKENIZER_SR), top_db=20.0).numel() / TOKENIZER_SR
        if sound < MIN_REFERENCE_SECONDS:
            raise ValueError(
                f"Reference audio has {sound:.2f} s of sound once silence is trimmed; "
                f"at least {MIN_REFERENCE_SECONDS} s is needed"
            )
        if (wav - wav.mean()).abs().max() < MIN_REFERENCE_PEAK:
            raise ValueError("Reference audio is silent (it never swings past -60 dBFS)")
        return wav.contiguous()

    def resolve_mode(self, model_kwargs: dict | None, uploaded: bool) -> tuple[str, str | None]:
        """``(mode, transcript)``: ``zero_shot`` (the LM hears the clip and
        reads its transcript), ``instruct`` or ``cross_lingual``."""
        mk = model_kwargs or {}
        if mk.get("instruct"):
            return "instruct", None
        transcript = mk.get("ref_text")
        if not transcript and not uploaded and mk.get("voice") not in (None, ""):
            transcript = self._preset_paths(str(mk["voice"]))[1]
        if transcript and transcript.strip():
            return "zero_shot", transcript.strip()
        return "cross_lingual", None

    @staticmethod
    def prompt_text(mode: str, transcript: str | None, instruct: str | None, text: str) -> str:
        """The LM's prompt text (``SYSTEM_PROMPT<|endofprompt|>`` + transcript,
        or the instruction), empty if the text carries its own delimiter."""
        if mode == "zero_shot":
            return f"{SYSTEM_PROMPT}{END_OF_PROMPT}{transcript}"
        if mode == "instruct":
            instruct = instruct.strip()
            return instruct if END_OF_PROMPT in instruct else f"{SYSTEM_PROMPT} {instruct}{END_OF_PROMPT}"
        return "" if END_OF_PROMPT in text else f"{SYSTEM_PROMPT}{END_OF_PROMPT}"

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs: Any,
    ) -> ProcessPromptOutput:
        if not prompt or not prompt.strip():
            raise ValueError("CosyVoice3 requires a non-empty text prompt")
        if any(m not in ("text", "audio") for m in input_modalities):
            raise ValueError("CosyVoice3 takes text plus a reference audio clip")
        if set(output_modalities) != {"audio"}:
            raise ValueError("CosyVoice3 produces audio output only")
        knobs = self.resolve_generation_kwargs(kwargs)

        uploaded = (tensors or {}).get("audio_inputs") or []
        if len(uploaded) > 1:
            raise ValueError("Give one reference clip per request")
        if uploaded:
            wav = uploaded[0]
        elif kwargs.get("voice") not in (None, ""):
            wav = self.load_audio(str(self._preset_paths(str(kwargs["voice"]))[0]), "cpu").data
        else:
            raise ValueError(
                "CosyVoice3 needs a reference voice: upload one (ref_audio, with its transcript in "
                "ref_text) or name a preset (voice)"
            )
        wav = self._prepare_reference(wav)

        mode, transcript = self.resolve_mode(kwargs, uploaded=bool(uploaded))
        prompt_ids = self.tokenizer(self.prompt_text(mode, transcript, kwargs.get("instruct"), prompt))
        text_ids = self.tokenizer(prompt)
        ids = torch.cat([prompt_ids, text_ids])
        if ids.numel() > self.config.max_text_tokens:
            raise ValueError(
                f"Prompt text and text are {ids.numel()} tokens; the limit is {self.config.max_text_tokens}. "
                "Split the text into sentences."
            )
        if not (ids == END_OF_PROMPT_ID).any():
            raise ValueError(f"CosyVoice3 needs {END_OF_PROMPT} in the instruction or the text")
        s = self.config.sampling
        max_len = min(int(text_ids.numel() * s.max_token_text_ratio), self._speech_limit)
        if knobs["max_new_tokens"] is not None:
            max_len = min(max_len, knobs["max_new_tokens"])
        if knobs["ignore_eos"] and knobs["max_new_tokens"] is not None:
            max_len = knobs["max_new_tokens"]
        min_len = min(int(text_ids.numel() * s.min_token_text_ratio), max_len)
        zero_shot = mode == "zero_shot"
        layout = self._lm_layout(
            "prefill_voice" if zero_shot else "prefill",
            ids.numel(),
            reference_tokens(wav.numel(), self.config) if zero_shot else 0,
        )
        return ProcessPromptOutput(
            {TEXT_INPUTS: [ids], REF_AUDIO: [wav]},
            {
                MIN_SPEECH_TOKENS: min_len,
                MAX_SPEECH_TOKENS: max(max_len, 1),
                VOICE_KEY: int(voice_key_for(wav)[0]),
                PROMPT_MODE: mode,
                "kv_layout": {LM_KV: {LM_LABEL: layout}},
            },
        )

    # -----------------------------------------------------------------------
    # Conductor state machine
    # -----------------------------------------------------------------------

    @staticmethod
    def _prompt_facts(model_kwargs: dict | None, keys) -> dict[str, Any]:
        mk = model_kwargs or {}
        missing = [k for k in keys if k not in mk]
        if missing:
            raise ValueError(f"CosyVoice3 request is missing {missing}, which process_prompt sets")
        return {k: mk[k] for k in keys}

    @staticmethod
    def _step_metadata(metadata: CurrentForwardConductorMetadata, keys) -> dict[str, Any]:
        step = {k: metadata.kwargs[k] for k in keys}
        step["is_prefill"] = metadata.is_prefill
        return step

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        knobs = self.resolve_generation_kwargs(model_kwargs)
        wants_audio = "audio" in output_modalities
        if partition_name == LM_PARTITION:
            zero_shot = self._prompt_facts(model_kwargs, (PROMPT_MODE,))[PROMPT_MODE] == "zero_shot"
            metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk="prefill_voice" if zero_shot else "prefill",
                is_prefill=True,
                kwargs={**{k: knobs[k] for k in _LM_KNOBS}, **self._prompt_facts(model_kwargs, _LM_FACTS)},
            )
            names = [TEXT_INPUTS] + ([REF_AUDIO] if zero_shot else [])
            targets = {TEXT_INPUTS: LM_NODE, REF_AUDIO: SPEAKER_ENCODER_NODE}
            inputs = []
            for name in names:
                edge = GraphEdge(next_node=targets[name], name=name)
                edge.tensor_info = input_signals.get(name, [])
                inputs.append(edge)
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=inputs,
                # the clip stays persisted for the flow node
                unpersist_tensors=list(input_signals.get(TEXT_INPUTS, [])),
                request_done=not wants_audio,
                step_metadata=self._step_metadata(metadata, _LM_KNOBS + _LM_FACTS),
            )
        if partition_name == TOKEN2WAV_PARTITION:
            metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk="token2wav",
                is_prefill=False,
                kwargs={**{k: knobs[k] for k in _FLOW_KNOBS}, **self._prompt_facts(model_kwargs, _FLOW_FACTS)},
            )
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=self._voice_inputs(input_signals),
                unpersist_tensors=[],
                request_done=not wants_audio,
                step_metadata=self._step_metadata(metadata, _FLOW_KNOBS + _FLOW_FACTS),
            )
        raise ValueError(f"Unknown CosyVoice3 partition {partition_name!r}")

    @staticmethod
    def _voice_inputs(signals: dict[str, list[TensorPointerInfo]], with_tensors: bool = True) -> list[GraphEdge]:
        """The reference clip for the flow node: the tensor with the first chunk
        only, a signal-only edge after (see Chatterbox's S3Gen)."""
        edge = GraphEdge(next_node=FLOW_NODE, name=REF_AUDIO)
        edge.tensor_info = list(signals.get(REF_AUDIO, [])) if with_tensors else []
        return [edge]

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        del incoming_connections
        if partition_name == LM_PARTITION:
            if partition_metadata.is_prefill:
                partition_metadata.is_prefill = False
                partition_metadata.graph_walk = "decode"
                edge = GraphEdge(next_node=LM_NODE, name=PREV_TOKEN)
                edge.tensor_info = persist_signals.get(SPEECH_TOKENS, [])
                return ForwardPassArgs(
                    full_metadata=partition_metadata,
                    inputs=[edge],
                    unpersist_tensors=list(edge.tensor_info),
                    step_metadata=self._step_metadata(partition_metadata, _LM_KNOBS + _LM_FACTS),
                )
            if partition_metadata.graph_walk == "decode":
                return ForwardPassArgs(
                    full_metadata=partition_metadata,
                    inputs=[],
                    unpersist_tensors=[],
                    request_done=True,
                )
            raise ValueError(f"LM in unexpected walk {partition_metadata.graph_walk!r}")
        if partition_name == TOKEN2WAV_PARTITION:
            return ForwardPassArgs(
                full_metadata=partition_metadata,
                inputs=self._voice_inputs(persist_signals, with_tensors=False),
                unpersist_tensors=[],
                step_metadata=self._step_metadata(partition_metadata, _FLOW_KNOBS + _FLOW_FACTS),
            )
        raise ValueError(f"Unknown CosyVoice3 partition {partition_name!r}")

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------

    def get_autocast_dtype(self):
        return self._lm_dtype

    def get_output_sample_rate(self, modality: str = "audio") -> int:
        return SAMPLE_RATE

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        del request_kwargs
        if modality != "audio":
            raise ValueError(f"Unsupported CosyVoice3 output modality {modality!r}")
        if output.numel() == 0:
            return b""
        pcm = output.detach().cpu()
        if pcm.is_floating_point():
            pcm = (pcm.clamp(-1, 1) * 32767).to(torch.int16)
        elif pcm.dtype != torch.int16:
            pcm = pcm.to(torch.int16)
        return pcm.contiguous().numpy().tobytes()

    # -----------------------------------------------------------------------
    # Submodules (worker side)
    # -----------------------------------------------------------------------

    def get_default_sharding_config(self):
        from mstar.distributed.base import ShardingConfig

        return ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={})

    def get_submodule(
        self,
        node_name: str,
        device: str = "cpu",
        tp_group=None,
        autocast_dtype: torch.dtype | None = None,
        sp_group=None,
    ) -> NodeSubmodule | None:
        del sp_group
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        if node_name == SPEAKER_ENCODER_NODE:
            from mstar.model.cosyvoice3.submodules import SpeakerEncoderSubmodule

            submodule = SpeakerEncoderSubmodule(self._reference_encoder(device), self.config)
        elif node_name == LM_NODE:
            submodule = self._create_lm_submodule(device, tp_group, autocast_dtype)
        elif node_name == FLOW_NODE:
            submodule = self._create_flow_submodule(device)
        elif node_name == HIFT_NODE:
            submodule = self._create_hift_submodule(device)
        else:
            raise ValueError(f"Unknown CosyVoice3 node {node_name!r}")
        self._submodule_cache[node_name] = submodule
        logger.info("Loaded CosyVoice3 submodule %s on %s", node_name, device)
        return submodule

    def _path(self, name: str) -> Path:
        path = Path(self.local_dir) / name
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing from the checkpoint snapshot")
        return path

    def _state_dict(self, name: str):
        return torch.load(self._path(name), map_location="cpu", weights_only=True, mmap=True).items()

    def _reference_encoder(self, device: str):
        """The speech tokenizer, prompt mel and CAMPPlus, once per device: the
        speaker-encoder and flow nodes share them when colocated."""
        key = f"reference_encoder:{device}"
        if key not in self._shared:
            from mstar.model.chatterbox.components.audio import MelSpectrogram24k
            from mstar.model.chatterbox.components.s3_tokenizer import S3Tokenizer
            from mstar.model.chatterbox.components.s3gen_xvector import CAMPPlus
            from mstar.model.cosyvoice3.components.onnx_weights import load_campplus, load_speech_tokenizer
            from mstar.model.cosyvoice3.submodules import ReferenceEncoder

            cfg = self.config
            # built for real: mel filters, windows and RoPE tables are computed
            tokenizer = S3Tokenizer(cfg.tokenizer).to(device).eval()
            load_speech_tokenizer(tokenizer, self._path(cfg.speech_tokenizer_onnx))
            campplus = CAMPPlus(cfg.xvector).to(device).eval()
            load_campplus(campplus, self._path(cfg.campplus_onnx), cfg.xvector.feat_dim)
            mel = MelSpectrogram24k(cfg.mel).to(device)
            self._shared[key] = ReferenceEncoder(tokenizer, mel, campplus)
        return self._shared[key]

    def _create_lm_submodule(self, device: str, tp_group=None, autocast_dtype: torch.dtype | None = None):
        from mstar.model.chatterbox.loader import materialize
        from mstar.model.cosyvoice3.components.lm import CosyVoice3LM
        from mstar.model.cosyvoice3.submodules import LMSubmodule

        with torch.device("meta"):
            model = CosyVoice3LM(self.config.lm, comm_group=tp_group)
        materialize(model, device, autocast_dtype or self._lm_dtype)
        model.keep_head_in_float32()  # before loading, so the checkpoint's head is not rounded through bfloat16
        model.load_weights((k, v.to(device)) for k, v in self._state_dict(self.config.llm_weights))
        return LMSubmodule(model.eval(), self.config)

    def _create_flow_submodule(self, device: str):
        from mstar.model.cosyvoice3.components.flow import CosyVoice3Flow
        from mstar.model.cosyvoice3.submodules import FlowSubmodule

        # the fixed noise field is computed in __init__ -> built for real
        flow = CosyVoice3Flow(self.config.flow).to(device)
        flow.load_weights((k, v.to(device)) for k, v in self._state_dict(self.config.flow_weights))
        dtype = _parse_dtype(self.config.flow_estimator_dtype)
        if dtype != torch.float32:
            flow.set_estimator_dtype(dtype)
        return FlowSubmodule(flow.eval(), self._reference_encoder(device), self.config)

    def _create_hift_submodule(self, device: str):
        from mstar.model.cosyvoice3.components.hift import CausalHiFTGenerator
        from mstar.model.cosyvoice3.submodules import HiFTSubmodule

        hift = CausalHiFTGenerator(self.config.hift).to(device)
        hift.load_weights((k, v.to(device)) for k, v in self._state_dict(self.config.hift_weights))
        hift.keep_f0_predictor_in_float64()
        return HiFTSubmodule(hift.eval(), self.config)


def _parse_dtype(name: str) -> torch.dtype:
    dtypes = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return dtypes[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown dtype {name!r}; use bfloat16, float16 or float32") from None
