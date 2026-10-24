"""PersonaPlex-7B: offline, preloaded-input replay of a full-duplex speech model.

Nodes:
    MimiEncoder - user PCM -> 8-codebook Mimi codes (whole sequence, ring mask)
    LM          - temporal transformer + depformer + delayed token cache;
                  one 80 ms frame (one temporal forward) per iteration
    MimiDecoder - accumulated agent codes -> 24 kHz PCM trimmed to the input

Walks, stepped through ``kwargs["walk_schedule"]`` (Wan2.2 pattern):
    encode_user_audio - MimiEncoder; persists ``user_audio_codes``
    conditioning      - Loop over LM: voice embeddings, 6 silence, persona text,
                        6 silence ([PP] models/lm.py:1110-1121); persists the
                        final ``lm_state`` (token cache, mask, offset, step)
    generate[_audio]  - Loop over LM, one iteration per user frame, then
                        MimiDecoder; ``agent_frame`` accumulates across
                        iterations, and ``generate`` also sends each frame's
                        ``text_frame`` to the client as it is produced

The LM is one node name in two walks (``Sequential`` keys a walk's nodes by name).
Its loop-backs are consumed and seeded empty at iteration 0, and async
scheduling is off because an overshoot iteration would append to the temporal
KV. Loops stop by finish signal from the LM submodule; ``max_iters`` is the
budget ceiling.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardConductorMetadata, StreamingConnectionState
from mstar.engine.resources import NodeResourceSpec, ResourceReqConfig
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.position.config import PositionConfig, PositionSpec
from mstar.engine.resources.sampler.config import SamplerSpec, SamplingReqConfig
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import ForwardPassArgs, KVLabel, LabelLayout, Model, ProcessPromptOutput, TensorAndMetadata

from . import config as cfg
from .voice import VOICES_ARCHIVE, extract_voice_presets, load_voice_preset, resolve_voice_path

logger = logging.getLogger(__name__)

DEFAULT_REPO = "nvidia/personaplex-7b-v1"  # [PP] models/loaders.py:45
TEXT_TOKENIZER_NAME = "tokenizer_spm_32k_3.model"  # [PP] models/loaders.py:42
LM_CHECKPOINT_NAME = "model.safetensors"  # [PP] models/loaders.py:43
MIMI_CHECKPOINT_NAME = "tokenizer-e351c8d8-checkpoint125.safetensors"  # [PP] models/loaders.py:44
GENERATE_AUDIO_WALK = cfg.GENERATE_AUDIO_WALK
LM_LOOP_BACKS = {
    cfg.CONDITIONING_WALK: ("lm_state",),
    cfg.GENERATE_WALK: ("lm_state", "agent_frame", "text_frame"),
}
# 128 pages x 128 tokens x 512 KiB a token (32 layers x 32 heads x 128 dims, K+V, bf16) = 8 GiB
DEFAULT_KV_PAGES = 128
# process_prompt's frame counts, read back by get_loop_caps; a client cannot set it.
LOOP_FRAMES_KWARG = "personaplex_loop_frames"


class PersonaPlexModel(Model):
    """NVIDIA PersonaPlex-7B (``nvidia/personaplex-7b-v1``), single GPU."""

    INTERNAL_KWARGS = (LOOP_FRAMES_KWARG,)

    def __init__(
        self,
        model_path_hf: str = DEFAULT_REPO,
        cache_dir: str | None = None,
        voice_dir: str | None = None,
        max_batch_size: int = cfg.DEFAULT_MAX_BATCH_SIZE,
        skip_weight_loading: bool = False,
        **kwargs,
    ):
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int) or max_batch_size < 1:
            raise ValueError(f"max_batch_size must be an integer >= 1, got {max_batch_size!r}")
        self.model_path_hf = model_path_hf or DEFAULT_REPO
        self.cache_dir = cache_dir
        self.voice_dir = voice_dir
        self.max_batch_size = max_batch_size
        # Dummy mode: get_submodule returns None, so the graph runs without weights.
        self.skip_weight_loading = skip_weight_loading
        self.lm_config = cfg.LMArchConfig()
        # Loaded lazily on the data worker so construction never touches the (gated) repo.
        self.tokenizer = None
        self._submodule_cache: dict[str, torch.nn.Module | None] = {}

    # ------------------------------------------------------------------
    # Model ABC: structure
    # ------------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        """Paged temporal KV, RoPE and the two samplers, all on the LM node.

        The depformer's 16-step KV is frame-local scratch inside the LM
        submodule, not a resource: the reference rebuilds it every frame
        ([PP] models/lm.py:1142), as Qwen3-TTS's code predictor does. Pool size
        is a deployment choice (``resources.lm_kv`` in the YAML); the default
        here only keeps a YAML without one off the generic 2048-page pool,
        which at this KV geometry would not fit a 96 GB GPU.
        """
        c = self.lm_config
        kv = PagedKVConfig(
            num_layers=c.num_layers, num_kv_heads=c.num_heads, head_dim=c.head_dim,
            max_seq_len=cfg.MAX_TEMPORAL_FORWARDS, num_qo_heads=c.num_heads,
            max_num_pages=DEFAULT_KV_PAGES,
        )
        lm = {cfg.LM_NODE}
        return [
            KVSpec(resource_key=cfg.LM_KV, nodes=lm, config=kv),
            AttentionSpec(resource_key=cfg.LM_ATTN, nodes=lm, config=AttentionConfig(kv_cache=cfg.LM_KV)),
            # Moshi rotates adjacent pairs ([PP] modules/rope.py): FlashInfer's interleave=True.
            PositionSpec(
                resource_key=cfg.LM_POS, nodes=lm,
                config=PositionConfig(kv_cache=cfg.LM_KV, interleave=True, rope_theta=float(c.rope_theta)),
            ),
            SamplerSpec(resource_key=cfg.TEXT_SAMPLER, nodes=lm, vocab_size=c.text_card,
                        enable_repetion_penalty=False),
            SamplerSpec(resource_key=cfg.DEPTH_SAMPLER, nodes=lm, vocab_size=c.card, enable_repetion_penalty=False),
        ]

    def _request_kv_pages(self, page_size: int) -> int:
        """Pages one request can hold: one KV entry per temporal forward, at most
        ``MAX_TEMPORAL_FORWARDS`` (async scheduling is off, so no overshoot step)."""
        return -(-cfg.MAX_TEMPORAL_FORWARDS // page_size)

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        """Refuse a request cap the temporal KV pool cannot hold.

        Nothing preempts a request once the pool is full, so
        ``max_concurrent_requests`` full-length requests must fit next to the
        pool's reserved sink page; requests over the cap wait in the conductor.
        """
        cap = config.get("max_concurrent_requests")
        if cap is None:
            logger.warning(
                "%s sets no max_concurrent_requests: long concurrent requests can fill "
                "the PersonaPlex temporal KV pool and stall", config_path,
            )
            return
        kv = next(s for s in self.get_node_resources() if s.resource_key == cfg.LM_KV).config
        overrides = (config.get("resources") or {}).get(cfg.LM_KV) or {}
        page_size = int(overrides.get("page_size", kv.page_size))
        max_num_pages = int(overrides.get("max_num_pages", kv.max_num_pages))
        need = 1 + int(cap) * self._request_kv_pages(page_size)
        if need > max_num_pages:
            raise ValueError(
                f"{config_path}: {cap} concurrent requests (max_concurrent_requests) of up to "
                f"{cfg.MAX_TEMPORAL_FORWARDS} forwards need {need} pages of resources.{cfg.LM_KV}, "
                f"which has {max_num_pages}; raise max_num_pages or lower max_concurrent_requests"
            )

    def _lm_node(self, walk: str) -> GraphNode:
        inputs = {
            cfg.CONDITIONING_WALK: ["lm_state", "voice_embeddings", "voice_cache", "persona_tokens"],
            cfg.GENERATE_WALK: ["lm_state", "agent_frame", "text_frame", "user_audio_codes"],
        }[walk]
        return GraphNode(
            name=cfg.LM_NODE, input_names=inputs,
            outputs=[GraphEdge(next_node=cfg.LM_NODE, name=name) for name in LM_LOOP_BACKS[walk]],
            enable_async_scheduling=False,
        )

    def _generate_walk(self, with_text: bool) -> GraphSection:
        accumulated = [GraphEdge(next_node=cfg.MIMI_DECODER_NODE, name="agent_frame")]
        lm = self._lm_node(cfg.GENERATE_WALK)
        if with_text:
            # Text goes to the client every frame instead of accumulating: an
            # accumulated client output is sent only when the loop ends, and by
            # then the worker has reused the pinned host buffers that staged the
            # earlier frames, so every frame arrived as the last token. Only the
            # Mimi decoder needs the agent codes as a sequence.
            lm.outputs.append(GraphEdge(next_node=EMIT_TO_CLIENT, name="text_frame", output_modality="text"))
        decode_loop = Loop(
            name=cfg.DECODE_LOOP, section=lm,
            max_iters=cfg.MAX_TEMPORAL_FORWARDS, outputs=[], accumulated_outputs=accumulated,
        )
        decoder = GraphNode(
            name=cfg.MIMI_DECODER_NODE, input_names=["agent_frame"],
            outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="audio_output", output_modality="audio")],
        )
        return Sequential(sections=[decode_loop, decoder])

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        encode = GraphNode(
            name=cfg.MIMI_ENCODER_NODE, input_names=["audio_inputs"],
            outputs=[GraphEdge(next_node=EMPTY_DESTINATION, name="user_audio_codes", persist=True)],
        )
        conditioning = Loop(
            name=cfg.CONDITIONING_LOOP, section=self._lm_node(cfg.CONDITIONING_WALK),
            max_iters=cfg.MAX_TEMPORAL_FORWARDS,
            # The final iteration's token-cache state carries into the generate walk.
            outputs=[GraphEdge(next_node=EMPTY_DESTINATION, name="lm_state", persist=True)],
        )
        return {
            cfg.ENCODE_USER_AUDIO_WALK: encode,
            cfg.CONDITIONING_WALK: conditioning,
            cfg.GENERATE_WALK: self._generate_walk(with_text=True),
            GENERATE_AUDIO_WALK: self._generate_walk(with_text=False),
        }

    # ------------------------------------------------------------------
    # Model ABC: request I/O (data worker)
    # ------------------------------------------------------------------

    def load_audio(self, filepath: str, device: str) -> TensorAndMetadata:
        """Decode at 24 kHz and keep channel 0, not a downmix ([PP] models/lm.py:121-129; offline.py:259)."""
        # Outside the try: a missing FFmpeg runtime fails here and is a server fault (500).
        from torchcodec.decoders import AudioDecoder

        limit_seconds = cfg.MAX_USER_AUDIO_SAMPLES / cfg.SAMPLE_RATE
        too_long = (f"PersonaPlex user audio is longer than {limit_seconds:.1f} s, the most "
                    f"{cfg.MAX_TEMPORAL_FORWARDS} temporal forwards can hold")
        try:
            decoder = AudioDecoder(filepath, sample_rate=cfg.SAMPLE_RATE)
            # Bound the upload before decoding it: a small compressed file can hold hours.
            header = decoder.metadata.duration_seconds_from_header
            if header is not None and header > limit_seconds:
                raise ValueError(f"{too_long} ({header:.1f} s)")
            # Headers can be absent or wrong, so decode at most one frame past the limit.
            stop = (cfg.MAX_USER_AUDIO_SAMPLES + cfg.FRAME_SIZE) / cfg.SAMPLE_RATE
            samples = decoder.get_samples_played_in_range(0.0, stop).data
        except RuntimeError as exc:
            # torchcodec reports undecodable bytes as RuntimeError; the data worker maps
            # ValueError to a 400. The message omits the server-side upload path.
            raise ValueError("PersonaPlex cannot decode the uploaded audio") from exc
        if samples.dim() != 2 or samples.shape[0] < 1:
            raise ValueError("PersonaPlex cannot decode the uploaded audio")
        if samples.shape[-1] > cfg.MAX_USER_AUDIO_SAMPLES:
            raise ValueError(too_long)
        audio = samples[0].to(torch.float32)
        return TensorAndMetadata(
            data=audio,
            metadata=dict(sample_rate=cfg.SAMPLE_RATE, num_channels=1, source_channels=int(samples.shape[0]),
                          num_samples=int(audio.shape[-1])),
        )

    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            import sentencepiece
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(self.model_path_hf, TEXT_TOKENIZER_NAME, cache_dir=self.cache_dir)
            self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=path)
        return self.tokenizer

    def _voice_dir(self) -> str | None:
        """``voice_dir``, or the repo's extracted ``voices.tgz`` when unset (not in dummy mode)."""
        if self.voice_dir is None and not self.skip_weight_loading:
            self.voice_dir = str(extract_voice_presets(self._download(VOICES_ARCHIVE)))
        return self.voice_dir

    def warmup_preprocess(self) -> None:
        """Fetch the tokenizer and voice presets before the first request."""
        if self.skip_weight_loading:
            return
        self._ensure_tokenizer()
        self._voice_dir()

    def get_voices(self) -> list[str] | None:
        """Preset names for ``GET /v1/audio/voices``; never downloads (runs in the API process)."""
        voice_dir = self.voice_dir
        if voice_dir is None and not self.skip_weight_loading:
            from huggingface_hub import try_to_load_from_cache

            archive = try_to_load_from_cache(self.model_path_hf, VOICES_ARCHIVE, cache_dir=self.cache_dir)
            voice_dir = Path(archive).parent / "voices" if isinstance(archive, str) else None
        if voice_dir is None or not Path(voice_dir).is_dir():
            return None
        return sorted(path.stem for path in Path(voice_dir).glob("*.pt")) or None

    @staticmethod
    def wrap_with_system_tags(text: str) -> str:
        """[PP] offline.py:83-90."""
        cleaned = text.strip()
        if cleaned.startswith(cfg.SYSTEM_TAG) and cleaned.endswith(cfg.SYSTEM_TAG):
            return cleaned
        return f"{cfg.SYSTEM_TAG} {cleaned} {cfg.SYSTEM_TAG}"

    @staticmethod
    def _validate_sampling(kwargs: dict) -> None:
        for name in ("temp_text", "temp_audio"):
            value = kwargs.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value < 0):
                raise ValueError(f"PersonaPlex {name} must be a finite number >= 0, got {value!r}")
        for name in ("topk_text", "topk_audio"):
            value = kwargs.get(name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"PersonaPlex {name} must be an integer >= 1, got {value!r}")
        if kwargs.get("greedy") is not None and not isinstance(kwargs["greedy"], bool):
            raise ValueError(f"PersonaPlex greedy must be a boolean, got {kwargs['greedy']!r}")

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> ProcessPromptOutput:
        """Validate the request and build the conditioning tensors.

        Request checks run on the data worker (here and in ``load_audio``), where
        a ValueError becomes a 400 before any GPU work; the conductor-side checks
        in ``get_initial_forward_pass_args`` repeat the context budget as a backstop.
        """
        if "audio" not in output_modalities or not set(output_modalities) <= {"audio", "text"}:
            raise ValueError(f"PersonaPlex outputs audio (optionally with text); got {output_modalities}")
        audio = (tensors or {}).get("audio_inputs") or []
        if "audio" not in input_modalities or len(audio) != 1:
            raise ValueError("PersonaPlex needs exactly one user audio input (24 kHz; channel 0 is used)")
        num_samples = int(audio[0].shape[-1])
        if num_samples == 0:
            raise ValueError("PersonaPlex user audio is empty")
        self._validate_sampling(kwargs)
        if prompt and kwargs.get("text_prompt") is not None:
            raise ValueError("give the persona as the prompt or as model_kwargs['text_prompt'], not both")
        persona = prompt if prompt else kwargs.get("text_prompt", cfg.DEFAULT_TEXT_PROMPT)
        if not isinstance(persona, str) or not persona.strip():
            # The reference would iterate None for an empty prompt and fail ([PP] offline.py:241-243).
            raise ValueError("PersonaPlex persona text prompt must be non-empty")
        preset = load_voice_preset(resolve_voice_path(self._voice_dir(), kwargs.get("voice")),
                                   dim=self.lm_config.dim)
        tokens = list(self._ensure_tokenizer().encode(self.wrap_with_system_tags(persona)))
        if not tokens:
            raise ValueError("PersonaPlex persona text prompt tokenized to nothing")
        persona_tokens = torch.tensor(tokens, dtype=torch.long)
        if persona_tokens.min() < 0 or persona_tokens.max() >= cfg.TEXT_CARD:
            raise ValueError("persona tokens fall outside the 32000-token text vocabulary")
        user_frames = math.ceil(num_samples / cfg.FRAME_SIZE)
        cfg.validate_context_budget(
            voice_embeddings=preset.num_embeddings, text_tokens=len(tokens), user_frames=user_frames,
        )
        conditioning = preset.num_embeddings + 2 * cfg.SILENCE_FRAMES + len(tokens)
        # One temporal KV entry per LM iteration, in both loops.
        layout = LabelLayout(growth=((cfg.CONDITIONING_LOOP, 1), (cfg.DECODE_LOOP, 1)))
        return ProcessPromptOutput({
            "voice_embeddings": [preset.embeddings],
            "voice_cache": [preset.cache],
            "persona_tokens": [persona_tokens],
        }, {
            "kv_layout": {cfg.LM_KV: {cfg.KV_LABEL: layout}},
            LOOP_FRAMES_KWARG: {cfg.CONDITIONING_LOOP: conditioning, cfg.DECODE_LOOP: user_frames},
        })

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        growth = ((cfg.CONDITIONING_WALK, cfg.CONDITIONING_LOOP), (cfg.GENERATE_WALK, cfg.DECODE_LOOP),
                  (cfg.GENERATE_AUDIO_WALK, cfg.DECODE_LOOP))
        return {cfg.LM_KV: {cfg.KV_LABEL: KVLabel(growth=growth)}}

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        # Each loop's stop check ends it after exactly this many frames, so the cap is that count.
        return dict(model_kwargs.get(LOOP_FRAMES_KWARG) or {})

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        """Audio: PCM16. Text: one frame's token as its SentencePiece piece ([PP] offline.py:286-295).

        Tokens 0-3 (EPAD/BOS/EOS/PAD) become empty bytes; NVIDIA's offline
        script logs them as labels instead.
        """
        del request_kwargs
        if modality == "audio":
            pcm = output.detach().cpu()
            if pcm.is_floating_point():
                pcm = (pcm.clamp(-1, 1) * 32767).to(torch.int16)
            return pcm.contiguous().numpy().tobytes()
        if modality == "text":
            if output.numel() != 1:
                raise ValueError(f"PersonaPlex emits one text token per frame, got shape {tuple(output.shape)}")
            token = int(output.reshape(-1)[0])
            if token <= cfg.TEXT_PAD_TOKEN_ID:
                return b""
            return self._ensure_tokenizer().id_to_piece(token).replace("▁", " ").encode("utf-8")
        raise ValueError(f"Unsupported PersonaPlex output modality: {modality!r}")

    # ------------------------------------------------------------------
    # Model ABC: forward-pass orchestration (conductor)
    # ------------------------------------------------------------------

    def get_max_output_tokens(self, **model_kwargs):
        # The input audio fixes the output length (one agent frame per user
        # frame), and process_prompt has already bounded it, so a client
        # max_output_tokens is ignored, as in Nemotron Duplex. Text frames carry
        # no conductor_new_token, so the conductor's cap never counts them.
        del model_kwargs
        return cfg.MAX_TEMPORAL_FORWARDS

    def get_autocast_dtype(self):
        # EngineManager also uses this as the paged-KV allocation dtype. Each
        # submodule disables engine autocast and keeps its own checkpoint dtype.
        return torch.bfloat16

    def get_request_resource_configs(
        self, partition_fwd_args: dict[str, ForwardPassArgs], model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        """Reference defaults ([PP] offline.py:363-374); ``greedy`` maps to temperature 0.

        An option sent as ``null`` is unset, as for the shared sampling parameters.
        """
        del partition_fwd_args
        kw = model_kwargs or {}

        def option(name, default):
            value = kw.get(name)
            return default if value is None else value

        greedy = bool(option("greedy", False))
        return {
            cfg.TEXT_SAMPLER: SamplingReqConfig(
                temperature=0.0 if greedy else float(option("temp_text", cfg.TEXT_TEMPERATURE)),
                top_k=int(option("topk_text", cfg.TEXT_TOP_K)), top_p=1.0,
            ),
            cfg.DEPTH_SAMPLER: SamplingReqConfig(
                temperature=0.0 if greedy else float(option("temp_audio", cfg.AUDIO_TEMPERATURE)),
                top_k=int(option("topk_audio", cfg.AUDIO_TOP_K)), top_p=1.0,
            ),
        }

    @staticmethod
    def _signals(persist: dict[str, list[TensorPointerInfo]], node: str, names) -> list[GraphEdge]:
        return [GraphEdge(next_node=node, name=name, tensor_info=list(persist.get(name, []))) for name in names]

    @staticmethod
    def _step_metadata(metadata: CurrentForwardConductorMetadata) -> dict:
        kw = metadata.kwargs
        return {key: kw[key] for key in (
            "num_voice_embeddings", "num_text_tokens", "num_user_frames", "num_samples", "conditioning_frames",
        )}

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        # Backstops only: process_prompt rejected these on the data worker.
        for name in ("audio_inputs", "voice_embeddings", "voice_cache", "persona_tokens"):
            if len(input_signals.get(name) or []) != 1:
                raise ValueError(f"PersonaPlex request is missing {name}")
        num_samples = int(input_signals["audio_inputs"][0].dims[-1])
        counts = {
            "num_voice_embeddings": int(input_signals["voice_embeddings"][0].dims[0]),
            "num_text_tokens": int(input_signals["persona_tokens"][0].dims[0]),
            "num_user_frames": math.ceil(num_samples / cfg.FRAME_SIZE),
            "num_samples": num_samples,
        }
        cfg.validate_context_budget(
            voice_embeddings=counts["num_voice_embeddings"], text_tokens=counts["num_text_tokens"],
            user_frames=counts["num_user_frames"],
        )
        generate = cfg.GENERATE_WALK if "text" in output_modalities else GENERATE_AUDIO_WALK
        kwargs = {
            "walk_schedule": [cfg.ENCODE_USER_AUDIO_WALK, cfg.CONDITIONING_WALK, generate],
            "walk_step": 0,
            # One LM iteration per voice embedding, silence frame and persona token.
            "conditioning_frames": counts["num_voice_embeddings"] + 2 * cfg.SILENCE_FRAMES
            + counts["num_text_tokens"],
            **counts,
        }
        metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities, output_modalities=output_modalities,
            graph_walk=cfg.ENCODE_USER_AUDIO_WALK, is_prefill=True, kwargs=kwargs,
        )
        inputs = self._signals(input_signals, cfg.MIMI_ENCODER_NODE, ["audio_inputs"])
        return ForwardPassArgs(
            full_metadata=metadata, inputs=inputs,
            unpersist_tensors=sum((e.tensor_info for e in inputs), start=[]),
            step_metadata=self._step_metadata(metadata),
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        """encode_user_audio -> conditioning -> generate[_audio] -> done."""
        metadata = partition_metadata
        schedule = metadata.kwargs["walk_schedule"]
        step = metadata.kwargs["walk_step"] + 1
        inputs: list[GraphEdge] = []
        request_done = step >= len(schedule)
        if not request_done:
            walk = schedule[step]
            metadata.kwargs["walk_step"] = step
            metadata.graph_walk = walk
            metadata.is_prefill = False
            if walk == cfg.CONDITIONING_WALK:
                inputs = self._signals(persist_signals, cfg.LM_NODE,
                                       ["voice_embeddings", "voice_cache", "persona_tokens"])
                inputs.append(GraphEdge(next_node=cfg.LM_NODE, name="lm_state"))
            else:
                inputs = self._signals(persist_signals, cfg.LM_NODE, ["lm_state", "user_audio_codes"])
                inputs += [GraphEdge(next_node=cfg.LM_NODE, name=n) for n in ("agent_frame", "text_frame")]
        # Each persisted tensor is used for the last time in the walk it feeds.
        return ForwardPassArgs(
            full_metadata=metadata, inputs=inputs,
            unpersist_tensors=sum((e.tensor_info for e in inputs), start=[]),
            step_metadata=self._step_metadata(metadata), request_done=request_done,
        )

    # ------------------------------------------------------------------
    # Model ABC: submodules
    # ------------------------------------------------------------------

    def get_submodule(
        self, node_name: str, device="cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None, sp_group=None,
    ) -> torch.nn.Module | None:
        del tp_group, autocast_dtype, sp_group  # single GPU; submodules load in their own dtypes
        if self.skip_weight_loading:
            return None
        if node_name not in self._submodule_cache:
            self._submodule_cache[node_name] = self._create_submodule(node_name, device)
        return self._submodule_cache[node_name]

    def _download(self, filename: str) -> str:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(self.model_path_hf, filename, cache_dir=self.cache_dir)

    def _create_submodule(self, node_name: str, device) -> torch.nn.Module:
        """Load one node from the gated checkpoint."""
        from .submodules import LMSubmodule, MimiDecoderSubmodule, MimiEncoderSubmodule

        if node_name == cfg.LM_NODE:
            from .weights import build_lm, iter_lm_checkpoint

            path = self._download(LM_CHECKPOINT_NAME)
            lm = build_lm(self.lm_config, iter_lm_checkpoint(path, device=device), device=device)
            logger.info("Loaded PersonaPlex LM from %s", path)
            return LMSubmodule(lm, max_batch_size=self.max_batch_size)
        if node_name in (cfg.MIMI_ENCODER_NODE, cfg.MIMI_DECODER_NODE):
            from safetensors.torch import load_file

            from .components.mimi import load_mimi_codec

            # Separate instances per node, like Wan2.2's VAEs; NVIDIA's other_mimi is omitted:
            # offline.py discards its outputs ([PP] offline.py:101,108,120).
            codec = load_mimi_codec(load_file(self._download(MIMI_CHECKPOINT_NAME)), device=device)
            cls = MimiEncoderSubmodule if node_name == cfg.MIMI_ENCODER_NODE else MimiDecoderSubmodule
            return cls(codec)
        raise ValueError(f"PersonaPlex has no node {node_name!r}")
