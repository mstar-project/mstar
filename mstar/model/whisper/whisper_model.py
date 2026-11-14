"""
WhisperModel: encoder-decoder ASR (openai/whisper-large-v3-turbo, -large-v3, ...).

Whisper transcribes speech: a log-mel spectrogram of one 30 s window runs
through a 32-layer audio encoder once, and a text decoder (4 layers for
turbo, 32 for large-v3) generates the transcript autoregressively,
attending to the encoder output through cross-attention at every step.
One class serves every size: dims and token ids come from the checkpoint's
``config.json`` / ``generation_config.json`` / ``preprocessor_config.json``.

Architecture (2 nodes, single partition):
    audio_encoder  — native batched encoder, one CUDA graph per batch size
    decoder        — native decoder; paged self-attn KV cache + write-once
                     cross-attention context; decode captured per batch size

Graph walks:
    prefill          — audio_encoder -> decoder over the forced prompt
                       ``[<|startofprev|> ctx] <|sot|><|lang|><|task|>[<|notimestamps|>]``;
                       samples the first transcript token
    detect_language  — audio_encoder -> decoder over ``<|sot|>`` alone, sampling
                       restricted to language tokens (``language`` not given)
    prefill_prompt   — decoder only: ``<|lang|>`` (detected) + ``<|task|>[<|notimestamps|>]``
                       over the context the previous walk wrote
    decode           — decoder loop; each step feeds the sampled token back
    align            — decoder only, after decode when word timestamps were
                       asked for: emits the cross-attention alignment that
                       ``postprocess`` turns into word timings

Walk plan (``[-> align]`` for word timestamps):
    language given:  prefill [-> decode] [-> align]
    language absent: detect_language -> prefill_prompt [-> decode] [-> align]
The prompt walk samples the first of ``max_output_tokens``, so a budget of one
runs no decode; a stop id sampled there ends the request.

The transcript is streamed token by token as Whisper's own token stream
rendered to text: spoken words as bytes, plus the language token (``<|en|>``)
and, when timestamps were requested, the ``<|s.ss|>`` markers. The OpenAI layer's
adapter lifts those into ``language`` / ``segments``.
"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.engine.state import (
    Array,
    Attend,
    Bidirectional,
    Causal,
    Entry,
    KVEntry,
    Layout,
    Positions,
    Read,
    Sampler,
    SampleSettings,
    Span,
    TensorType,
    Value,
)
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import KVLabel, Model, ProcessPromptOutput, WalkStep
from mstar.model.components.audio_features import LogMelSpectrogram
from mstar.model.components.request_kwargs import (
    GREEDY_SAMPLING,
    SAMPLING_KEYS,
    checked_request_kwargs,
    sampler_capabilities,
    sampling_config,
    sampling_defaults,
)
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import ByteLevelDetokenizer, TextOutput, check_kv_pools_fit, kv_pages
from mstar.model.whisper.components import alignment
from mstar.model.whisper.components.timestamps import STATE_SIZE
from mstar.model.whisper.config import (
    ALIGN_WALK,
    ATTN,
    CONTEXT_LABEL,
    CROSS_ATTN,
    CROSS_KV_CACHE,
    DECODE_LOOP,
    DECODE_WALK,
    DECODER_NODE,
    DECODER_STATE,
    DETECT_LANGUAGE_WALK,
    ENCODER_NODE,
    KV_CACHE,
    PAGE_TOKENS,
    PREFILL_PROMPT_WALK,
    PREFILL_WALK,
    SAMPLER,
    WORDS_MARKER,
    WhisperModelConfig,
)

logger = logging.getLogger(__name__)

# greedy, as openai-whisper decodes; the checkpoint's generation_config.json overrides
DEFAULT_SAMPLING = GREEDY_SAMPLING


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


class WhisperDetokenizer(ByteLevelDetokenizer):
    """Byte-level detokenizer that keeps the informative control tokens.

    Language tokens and timestamp tokens are rendered as their literal
    ``<|xx|>`` / ``<|s.ss|>`` text so the serving layer can lift them out;
    the structural specials (``<|startoftranscript|>``, ``<|transcribe|>``,
    ``<|notimestamps|>``, ``<|endoftext|>``, ...) are dropped as before.
    """

    def __init__(self, tokenizer, config: WhisperModelConfig):
        super().__init__(tokenizer)
        self.config = config
        self._render_ids = set(config.language_token_ids)

    def _rendered(self, token_id: int) -> bool:
        return token_id in self._render_ids or self.config.is_timestamp(token_id)

    def to_bytes(self, token_ids: list[int]) -> bytes:
        raw = bytearray()
        for token_id in token_ids:
            if self._rendered(token_id):
                raw.extend(self.tokenizer.convert_ids_to_tokens(token_id).encode("utf-8"))
            elif token_id not in self.special_ids:
                token = self.tokenizer.convert_ids_to_tokens(token_id)
                raw.extend(self.byte_decoder[c] for c in token)
        return bytes(raw)


class WhisperModel(Model):
    """Whisper ASR: native batched encoder + native AR decoder."""

    DECODE_LOOP = DECODE_LOOP

    # the checkpoint's sampling defaults, set by __init__; a model built without it has the built-in ones
    sampling_defaults: Mapping[str, object] = DEFAULT_SAMPLING

    # A requested length that fits the table is cut to the room a longer prompt
    # leaves rather than refused: long-form windows carry the transcript so far
    # as <|startofprev|> context. One past the table is still refused (see
    # get_max_output_tokens).
    LENGTH_KWARGS = ()

    # Requests the default decoder caches hold at once, each at full length.
    # A deployment retunes the page counts under ``resources:`` in its YAML.
    MAX_CONCURRENT_REQUESTS = 64
    # The API server decodes uploads and stages their tensors with small CPU
    # ops; a thread per core spins between them and starves its event loop.
    PREPROCESS_TORCH_THREADS = 4
    # Set by ``process_prompt`` for the align walk: the mel frames that carry
    # audio, and where the transcript and a detected language sit in its row
    INTERNAL_KWARGS = ("audio_frames", "transcript_start", "language_slot")
    # Set from the deployment by validate_config_yaml: the align walk re-runs
    # the decoder unsharded, so word timestamps need tp_size 1.
    _decoder_tp_size = 1

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        **kwargs,
    ):
        self.cache_dir = cache_dir
        self.model_path_hf = model_path_hf

        self.local_dir = _resolve_local_hf_snapshot(model_path_hf, cache_dir=cache_dir)
        self.config = WhisperModelConfig.from_pretrained(self.local_dir)
        self.sampling_defaults = sampling_defaults(self.local_dir)
        self.AUDIO_INPUT_RATE = self.config.sampling_rate

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.local_dir, cache_dir=cache_dir,
        )
        self.log_mel = LogMelSpectrogram(
            num_mel_bins=self.config.num_mel_bins,
            sampling_rate=self.config.sampling_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            chunk_length=self.config.chunk_length,
        )
        self._detokenizer = WhisperDetokenizer(self.tokenizer, self.config)

        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    # -------------------------------------------------------------------
    # Model ABC: state
    # -------------------------------------------------------------------

    def declare_state(self) -> list[Array | Sampler]:
        """Decoder self-attention KV, a separate encoder-context KV, and each
        request's decoder state.

        Two caches rather than two labels on one: the self-attention resource
        plans a wrapper per label of the cache it names, so a shared cache
        would have it planning the 1500-token context every step for nothing.
        ``audio_encoder`` holds no state.
        """
        c = self.config
        kv = KVEntry(heads=c.decoder_attention_heads, head_dim=c.head_dim)
        attend = Attend(q_heads=c.decoder_attention_heads)
        layers = frozenset(range(c.decoder_layers))
        capabilities = sampler_capabilities(self.sampling_defaults)
        return [
            # No RoPE: the counter's planned ids drive the learned ``embed_positions`` lookup.
            Array(KV_CACHE, nodes={DECODER_NODE}, layers=layers, entry=kv, max_len=c.max_target_positions,
                  positions=Positions(), reads={ATTN: Read(mask=Causal(), op=attend)}),
            Array(CROSS_KV_CACHE, nodes={DECODER_NODE}, layers=layers, entry=kv, max_len=c.max_source_positions,
                  labels={CONTEXT_LABEL},
                  reads={CROSS_ATTN: Read(mask=Bidirectional(), op=attend, queries=KV_CACHE)}),
            # The transcript holds each sampled token at the position it is fed back at, so the token sampled at
            # the last position sits one past it.
            Value(DECODER_STATE, nodes={DECODER_NODE}, layers=frozenset({0}), entry=Entry({
                "ts_rules": TensorType((STATE_SIZE,), dtype=torch.long),
                "transcript": TensorType((c.max_target_positions + 1,), dtype=torch.long),
            })),
            # ASR decodes greedily; without the capability a request penalty or min_p is refused at admission.
            Sampler(
                SAMPLER, nodes={DECODER_NODE}, vocab_size=c.vocab_size,
                penalty=capabilities["enable_repetition_penalty"], min_p=capabilities["enable_min_p"],
            ),
        ]

    def kernel_settings(self) -> dict[str, dict[str, Any]]:
        requests = self.MAX_CONCURRENT_REQUESTS
        return {
            # Whisper is MHA: the CUDA-core decode kernel beats the
            # tensor-core one, which pads one query head per KV head to a tile
            ATTN: {"decode_tensor_cores": False},
            CROSS_ATTN: {"decode_tensor_cores": False},
            # 448 positions are 4 pages a request. The 30 s window's 1500
            # tokens are 12, held from prefill to the last token; 769 pages at
            # ~15 GiB for large-v3.
            KV_CACHE: {"max_num_pages": kv_pages(self.config.max_target_positions, PAGE_TOKENS, requests)},
            CROSS_KV_CACHE: {"max_num_pages": kv_pages(self.config.max_source_positions, PAGE_TOKENS, requests)},
        }

    def get_request_resource_configs(
        self, input_modalities: list[str], output_modalities: list[str], model_kwargs: dict,
    ) -> dict[str, SampleSettings]:
        del input_modalities, output_modalities
        config = sampling_config(self.sampling_defaults, model_kwargs)
        return {SAMPLER: SampleSettings(**{key: getattr(config, key) for key in SAMPLING_KEYS})}

    def request_kwargs(self) -> frozenset[str]:
        return frozenset({
            "language", "task", "timestamps", "initial_prompt", "max_output_tokens", *SAMPLING_KEYS,
        })

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        """Refuse a request cap the decoder caches cannot hold: a request keeps
        its encoder context from prefill to its last token, so nothing frees
        cross-attention pages but a finished request. Also note the decoder's
        tensor-parallel size, which word timestamps need to be 1, and refuse a
        placement that splits the decoder's walks across ranks."""
        decoder_groups = [
            group for group in config.get("node_groups") or []
            if DECODER_NODE in (group.get("node_names") or [])
        ]
        self._decoder_tp_size = max((int(group.get("tp_size", 1)) for group in decoder_groups), default=1)

        def serving(walk: str) -> tuple:
            return tuple(sorted(
                tuple(group.get("ranks") or ()) for group in decoder_groups
                if group.get("graph_walks") is None or walk in group["graph_walks"]
            ))

        # the align walk reads the transcript the other walks wrote to their
        # worker's decoder state, which doesn't move between workers
        walks = (PREFILL_WALK, DETECT_LANGUAGE_WALK, PREFILL_PROMPT_WALK, DECODE_WALK, ALIGN_WALK)
        served = {walk: serving(walk) for walk in walks}
        if len(set(served.values())) > 1:
            raise ValueError(
                f"{config_path}: the {DECODER_NODE} node's {', '.join(map(repr, walks))} walks must "
                f"run on the same ranks (word timestamps align the tokens those walks kept on their "
                f"worker); got " + ", ".join(f"{walk}: {list(ranks)}" for walk, ranks in served.items())
            )
        check_kv_pools_fit(self, config, config_path, logger)

    # -------------------------------------------------------------------
    # Model ABC: graph walk definitions
    # -------------------------------------------------------------------

    def get_max_output_tokens(self, **model_kwargs):
        # The learned position table caps prompt + generated tokens at
        # max_target_positions (448), less the plain forced prompt. A larger
        # request is refused by the conductor, not clamped; it cuts a longer
        # prompt's cap to what is left.
        limit = self.config.max_target_positions - self.config.decoder_prompt_len()
        return checked_request_kwargs(model_kwargs).get("max_output_tokens", limit)

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        def emit_first_token() -> list[GraphEdge]:
            return [
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="new_token",
                    output_modality="text",
                    persist=True,
                ),
            ]

        def encoder_then_decoder() -> GraphSection:
            return Sequential([
                GraphNode(
                    name=ENCODER_NODE,
                    input_names=["audio"],
                    # kept (persist) so the align walk can attend it again
                    outputs=[GraphEdge(next_node=DECODER_NODE, name="encoder_states", persist=True)],
                ),
                GraphNode(
                    name=DECODER_NODE,
                    input_names=["encoder_states", "text_inputs"],
                    outputs=emit_first_token(),
                ),
            ])

        prefill_prompt = GraphNode(
            name=DECODER_NODE,
            input_names=["text_inputs", "prompt_tail"],
            outputs=emit_first_token(),
        )

        decode = Loop(
            name=DECODE_LOOP,
            section=GraphNode(
                name=DECODER_NODE,
                input_names=["text_inputs"],
                outputs=[
                    # The sampled token loops back as the next input and goes to
                    # the client. The node keeps it on the device by sampler
                    # slot and the worker sends the batch's tokens inline, one
                    # frame a step, so a decode step stores no tensor per
                    # request (the client joins the text chunks; the align walk
                    # reads the transcript off the decoder state).
                    GraphEdge(next_node=DECODER_NODE, name="text_inputs"),
                    GraphEdge(next_node=EMIT_TO_CLIENT, name="text_inputs", output_modality="text"),
                ],
            ),
            # the default request's iterations: past them a request overruns the position table
            max_iters=self.get_loop_caps()[DECODE_LOOP],
            outputs=[],
        )

        align = GraphNode(
            name=DECODER_NODE,
            input_names=["encoder_states"],
            # float32, which is how ``postprocess`` tells it from token ids
            outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="alignment", output_modality="text")],
        )

        return {
            PREFILL_WALK: encoder_then_decoder(),
            DETECT_LANGUAGE_WALK: encoder_then_decoder(),
            PREFILL_PROMPT_WALK: prefill_prompt,
            DECODE_WALK: decode,
            ALIGN_WALK: align,
        }

    # -------------------------------------------------------------------
    # Model ABC: walk plan
    # -------------------------------------------------------------------

    def walk_plan(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict,
    ) -> list[WalkStep]:
        """``[detect_language -> prefill_prompt | prefill] -> [decode] [-> align]``."""
        word_timestamps = model_kwargs.get("timestamps") == "word"
        if word_timestamps and self._decoder_tp_size > 1:
            # a 400 now, rather than a failure after the whole decode has run
            raise ValueError(
                "word timestamps need the decoder on one GPU; this deployment shards it "
                f"(tp_size {self._decoder_tp_size})"
            )
        # What the decoder needs about the forced prompt, read as ``process_prompt``
        # reads it: any truthy ``timestamps`` (True, "word", "segment") omits
        # <|notimestamps|>.
        prompt = {"timestamps": bool(model_kwargs.get("timestamps", False))}
        if input_signals.get("prompt_tail"):
            # ``process_prompt`` holds the prompt's tail back when the language is
            # to be detected: the detected language token leads it in the second walk
            plan = [
                WalkStep(DETECT_LANGUAGE_WALK, step_metadata=prompt),
                WalkStep(PREFILL_PROMPT_WALK, bind={"text_inputs": "new_token"}, step_metadata=prompt),
            ]
        else:
            plan = [WalkStep(PREFILL_WALK, step_metadata=prompt)]
        # the prompt walk samples the budget's first token, so one token leaves decode
        # nothing to do. The conductor's fit to the caches never cuts a budget to one:
        # a request with no room for a decode step is refused.
        if self.get_max_output_tokens(**model_kwargs) > 1:
            plan.append(WalkStep(DECODE_WALK, bind={"text_inputs": "new_token"}))
        if word_timestamps:
            # The transcript the decoder state keeps, aligned against the persisted encoder
            # output under the forced prompt. Its tokens are resolved here, not read off
            # the prompt tensor on the GPU thread (a D2H sync); a detected language is
            # in the decoder state too.
            language = model_kwargs.get("language")
            plan.append(WalkStep(ALIGN_WALK, step_metadata={
                "language": None if language is None else self.config.language_token(language),
                "task": self.config.task_token(model_kwargs.get("task", "transcribe")),
                # how much of the window carries audio, and where its transcript row
                # holds the transcript and a detected language
                "audio_frames": model_kwargs.get("audio_frames"),
                "transcript_start": model_kwargs.get("transcript_start"),
                "language_slot": model_kwargs.get("language_slot"),
            }))
        return plan

    # -------------------------------------------------------------------
    # Model ABC: media loading + prompt processing
    # -------------------------------------------------------------------

    def _initial_prompt_ids(self, prompt: str | None) -> list[int]:
        """The ``<|startofprev|>`` context: prior transcript text, tokenized
        the way openai-whisper does (a leading space, no specials)."""
        if not prompt or not prompt.strip():
            return []
        return self.tokenizer.encode(" " + prompt.strip(), add_special_tokens=False)

    def max_input_files(self) -> dict[str, int]:
        return {"audio": 1}

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> ProcessPromptOutput:
        """Extract the log-mel window and build the forced decoder prompt.

        The text ``prompt`` is unused — Whisper is conditioned via its forced
        token sequence, controlled by ``language`` (ISO-639-1; ``None`` means
        detect it), ``task`` (``transcribe`` / ``translate``), ``timestamps``
        and ``initial_prompt`` (prior text carried over as ``<|startofprev|>``
        context) in model kwargs.
        """
        raw_audio_inputs = (tensors or {}).get("audio_inputs", [])
        if len(raw_audio_inputs) != 1:
            raise ValueError(
                f"Whisper expects one audio input per request, "
                f"got {len(raw_audio_inputs)}."
            )
        waveform = raw_audio_inputs[0].reshape(-1).to(torch.float32).cpu()
        if waveform.numel() == 0:
            raise ValueError("Whisper received an empty audio input.")

        # One fixed 30 s window: audio beyond it is dropped (long-form
        # chunking is the transcription route's job). The samples go to the
        # encoder as they are. The log-mel is its first step on the GPU. (On
        # the data worker's CPU threads the spectrogram was the serving
        # bottleneck: torch's intra-op pool spun a whole core per thread.)
        window = waveform[: self.config.n_samples].contiguous()

        kw = checked_request_kwargs(kwargs)
        language = kw.get("language")
        if language is not None:
            self.config.language_token(language)  # raises for a code the model lacks
        task = kw.get("task", "transcribe")
        if task not in self.config.task_to_id:
            raise ValueError(f"Whisper task must be one of {sorted(self.config.task_to_id)}, got {task!r}")
        timestamps = bool(kw.get("timestamps", False))
        prev_tokens = self._initial_prompt_ids(kw.get("initial_prompt"))
        prompt_ids = self.config.decoder_prompt_ids(
            language=language, task=task, timestamps=timestamps, prev_tokens=prev_tokens,
        )

        out: NameToTensorList = {
            "audio": [window],
            "text_inputs": [torch.tensor(prompt_ids, dtype=torch.long)],
        }
        # the encoder takes the padded window whole, so its context is always this long
        context = self.config.max_source_positions
        if language is None:
            tail = self.config.prompt_tail_ids(task=task, timestamps=timestamps)
            out["prompt_tail"] = [torch.tensor(tail, dtype=torch.long)]
            # detection commits the prompt up to <|sot|>; the prompt walk commits the
            # detected language token and the tail
            spans = (
                Span("ids", DETECT_LANGUAGE_WALK, len(prompt_ids), len(prompt_ids), "text_inputs"),
                Span("rows", PREFILL_PROMPT_WALK, 1 + len(tail), 1 + len(tail), "prompt_tail"),
            )
            encoder_walk = DETECT_LANGUAGE_WALK
            language_slot, transcript_start = len(prompt_ids), len(prompt_ids) + 1 + len(tail)
        else:
            spans = (Span("ids", PREFILL_WALK, len(prompt_ids), len(prompt_ids), "text_inputs"),)
            encoder_walk = PREFILL_WALK
            language_slot, transcript_start = None, len(prompt_ids)
        state_layout = {
            KV_CACHE: {"main": Layout(spans=spans, growth=((DECODE_LOOP, 1),))},
            CROSS_KV_CACHE: {CONTEXT_LABEL: Layout(
                spans=(Span("rows", encoder_walk, context, context, "encoder_states"),),
            )},
        }
        return ProcessPromptOutput(out, {
            "state_layout": state_layout,
            # mel frames that carry audio, word timestamps align within them
            "audio_frames": self.log_mel.num_frames(window.numel()),
            "transcript_start": transcript_start,
            "language_slot": language_slot,
        })

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        return {
            KV_CACHE: {"main": KVLabel(growth=((DECODE_WALK, DECODE_LOOP),))},
            CROSS_KV_CACHE: {CONTEXT_LABEL: KVLabel()},
        }

    # -------------------------------------------------------------------
    # Model ABC: postprocess
    # -------------------------------------------------------------------

    def postprocess(
        self,
        output: torch.Tensor,
        modality: str,
        **kwargs,
    ) -> bytes:
        if modality != "text":
            raise ValueError(f"Unsupported modality for Whisper: {modality!r}")
        if output.is_floating_point():
            # the align walk's alignment; every other text output is token ids
            return self._word_timings_text(output)
        return super().postprocess(output, modality, **kwargs)

    def text_output(self) -> TextOutput:
        return TextOutput(self._detokenizer)

    def warmup_preprocess(self) -> None:
        # postprocess runs the word timestamps' DTW in this process
        alignment.warm_up()

    def _decode_raw(self, ids: list[int]) -> str:
        """Word grouping's decode: no clean-up, which rewrites text across token
        boundaries (openai-whisper's tiktoken decode has none)."""
        return self.tokenizer.decode(ids, clean_up_tokenization_spaces=False)

    def _word_timings_text(self, output: torch.Tensor) -> bytes:
        """Word timings in Whisper's timestamp vocabulary, as the transcription
        route parses them: ``WORDS_MARKER`` then ``<|start|> word <|end|>``
        per word."""
        matrix, text, language = alignment.unpack_output(output)
        words = alignment.word_timings(matrix, text, self._decode_raw, self.config.language_of(language))
        tb, top = self.config.timestamp_begin, self.config.vocab_size - 1

        def stamp(seconds: float) -> bytes:
            token = min(top, tb + int(round(seconds / self.config.timestamp_precision)))
            return self._detokenizer.to_bytes([token])

        parts = [WORDS_MARKER.encode()]
        for w in words:
            parts += [stamp(w["start"]), w["word"].encode("utf-8"), stamp(w["end"])]
        return b"".join(parts)

    # -------------------------------------------------------------------
    # Model ABC: sharding
    # -------------------------------------------------------------------

    def get_default_sharding_config(self):
        from mstar.distributed.base import ShardingConfig

        return ShardingConfig(groups=[], tp_enabled_nodes={DECODER_NODE}, shard_dim={})

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
        logger.info("Successfully loaded Whisper submodule for %s", node_name)
        self._submodule_cache[node_name] = submodule
        return submodule

    def _create_submodule(
        self, node_name: str, device: str, tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name == ENCODER_NODE:
            return self._create_encoder_submodule(device, autocast_dtype=autocast_dtype)
        elif node_name == DECODER_NODE:
            return self._create_decoder_submodule(
                device, autocast_dtype=autocast_dtype, tp_group=tp_group,
            )
        return None

    def _create_encoder_submodule(
        self, device: str, autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule:
        from mstar.model.loader.iterators import iter_safetensors_shards
        from mstar.model.whisper.components.encoder import WhisperEncoderModel

        with torch.device("meta"):
            encoder = WhisperEncoderModel(self.config)
        # Cast on meta (no allocation) so to_empty allocates directly in the
        # target dtype instead of fp32-then-downcast.
        if autocast_dtype is not None:
            encoder = encoder.to(autocast_dtype)
        encoder.to_empty(device=device)

        prefix = "model.encoder."
        weights = iter_safetensors_shards(self.local_dir, device=device, prefix=prefix)
        encoder.load_weights((k.removeprefix(prefix), v) for k, v in weights)
        encoder.eval()

        from mstar.model.whisper.submodules import WhisperEncoderSubmodule
        return WhisperEncoderSubmodule(encoder=encoder, config=self.config)

    @staticmethod
    def _decoder_remap(name: str) -> str:
        # The shared Attention component names its output projection
        # ``o_proj``; the cross-attn module keeps HF's ``out_proj``.
        return name.replace("self_attn.out_proj", "self_attn.o_proj")

    def _create_decoder_submodule(
        self, device: str, autocast_dtype: torch.dtype | None = None, tp_group=None
    ) -> NodeSubmodule:
        from mstar.model.loader import WHISPER_STACKED_PARAMS, load_hf_weights
        from mstar.model.loader.iterators import iter_safetensors_shards
        from mstar.model.whisper.components.decoder import WhisperDecoderModel

        with torch.device("meta"):
            decoder = WhisperDecoderModel(self.config, comm_group=tp_group)
        # Cast on meta (no allocation) so to_empty allocates directly in the
        # target dtype instead of fp32-then-downcast.
        if autocast_dtype is not None:
            decoder = decoder.to(autocast_dtype)
        decoder.to_empty(device=device)

        weights = iter_safetensors_shards(
            self.local_dir, device=device, prefix="model.decoder.",
        )
        weights = ((k.removeprefix("model.decoder."), v) for k, v in weights)
        load_hf_weights(
            decoder,
            weights,
            stacked_params=WHISPER_STACKED_PARAMS,
            name_remapper=self._decoder_remap,
        )
        decoder.zero_missing_biases()
        decoder.eval()

        from mstar.model.whisper.submodules import WhisperDecoderSubmodule
        return WhisperDecoderSubmodule(decoder=decoder, config=self.config)
