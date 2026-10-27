"""PersonaPlex node submodules: LM, MimiEncoder and MimiDecoder.

The LM runs one reference ``LMGen`` step per graph iteration ([PP]
models/lm.py:814-958). ``lm_state`` (the packed ``DelayedTokenCache`` plus a
per-request frame counter) is the loop-back edge, so every iteration is a pure
function of its inputs. Frame ``i`` of the request is:

    conditioning, i < N_emb:          replay voice embedding i (step_embeddings);
                                      restore the preset cache after the last
    conditioning, then 6 + N_text + 6: silence / persona token / silence frames
    generate, j = i - conditioning:   user frame j; sample text and 16 depth codes

Conditioning targets are fully provided (every channel is forced), so those
frames skip the text head, depformer and samplers: NVIDIA runs them, but only
to consume RNG, and the cache keeps the forced values either way. Whether a
generate channel is forced is read from the cache every frame (the first one
forces the agent acoustic codebooks).

Requests in the same walk batch: the temporal transformer, text head, samplers
and depformer run once over every row, while each row keeps its own cache
(offsets differ between requests) and commits it separately.

The depformer stops after the last depth step any row samples
(``depth_steps_needed``): the user's codebooks are always forced here, so a
generate frame runs 8 of its 16 depth steps, and the first one runs 1.

The generate step is captured as an accelerator graph: ``step`` is the
tensor-only part the graph holds (8 depth steps), and ``commit`` folds each
request's samples into its own cache afterwards, in ``postprocess``.
Conditioning frames run eagerly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)

from . import config as cfg
from .components.delayed_cache import DelayedTokenCache
from .components.depformer import depth_steps_needed
from .components.mimi import MimiCodec, pad_to_frames
from .weights import PersonaPlexLM

GENERATE_WALKS = frozenset({cfg.GENERATE_WALK, cfg.GENERATE_AUDIO_WALK})
LM_WALKS = GENERATE_WALKS | {cfg.CONDITIONING_WALK}


class ResourceTemporalBackend:
    """Temporal attention on M*'s paged KV, attention and position resources."""

    def __init__(self, kv, attn, pos, rope_theta: float, label: str = cfg.KV_LABEL):
        from mstar.engine.resources.convenience import AttentionCallable

        self.attend_fn = AttentionCallable(kv=kv, attn=attn)
        self.pos = pos
        self.rope_theta = rope_theta
        self.label = label

    def begin(self) -> None:
        self.attend_fn.bind_step(self.label)

    def rope(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Moshi's adjacent-pair rotation ([PP] modules/rope.py) is FlashInfer's interleave=True.
        return self.pos.apply_qk(q, k, label=self.label, interleave=True, rope_theta=self.rope_theta)

    def attend(self, layer_idx: int, q, k, v) -> torch.Tensor:
        self.attend_fn.set_layer_idx(layer_idx)
        return self.attend_fn(q, k, v)


@dataclass
class LMFrame:
    """One prepared LM iteration (cache already ``prepare``-d)."""

    walk: str
    frame: int  # frame index within the request, across both LM walks
    cache: DelayedTokenCache
    embeds: torch.Tensor  # [1, D] temporal input
    restore_cache: torch.Tensor | None = None  # voice preset cache, after the last embedding


def _sampled_token_ids(tokens: torch.Tensor, expected: torch.Tensor, source: str) -> torch.Tensor:
    """Adapt eager FlashInfer's int32 IDs to the cache's strict int64 contract."""
    if not isinstance(tokens, torch.Tensor):
        raise ValueError(f"{source} must return a tensor")
    if tokens.shape != expected.shape:
        raise ValueError(f"{source} must return shape {tuple(expected.shape)}, got {tuple(tokens.shape)}")
    if tokens.device != expected.device:
        raise ValueError(f"{source} must return tokens on {expected.device}, got {tokens.device}")
    if tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{source} must return int32 or int64 tokens, got {tokens.dtype}")
    return tokens.to(torch.long)


def _planned_frames(walk: str, meta: dict) -> tuple[int, int]:
    """(first, end) request frame indices covered by ``walk``."""
    conditioning = int(meta["conditioning_frames"])
    if walk == cfg.CONDITIONING_WALK:
        return 0, conditioning
    return conditioning, conditioning + int(meta["num_user_frames"])


class LMSubmodule(ARNodeSubmodule):
    """Temporal transformer + depformer + delayed token cache, up to ``max_batch_size`` requests a step."""

    disable_autocast = True
    disable_torch_compile = True

    def __init__(self, lm: PersonaPlexLM, max_batch_size: int = cfg.DEFAULT_MAX_BATCH_SIZE):
        super().__init__()
        self.lm = lm
        self._max_batch_size = max_batch_size
        self._backend: ResourceTemporalBackend | None = None

    def bind_node_resources(self, resources: dict[str, Any]) -> None:
        super().bind_node_resources(resources)
        self._backend = ResourceTemporalBackend(
            resources[cfg.LM_KV], resources[cfg.LM_ATTN], resources[cfg.LM_POS], float(self.lm.config.rope_theta)
        )

    # ------------------------------------------------------------------ frame logic (engine-free)

    def prepare_frame(self, graph_walk: str, step_metadata: dict, inputs: NameToTensorList) -> LMFrame | None:
        """Rebuild the cache from ``lm_state``, apply this frame's provided tokens, embed the input.

        Returns None for an iteration past the walk's planned frames (a vetoed
        overshoot), which the engine skips.
        """
        if graph_walk not in LM_WALKS:
            raise ValueError(f"unknown PersonaPlex LM walk {graph_walk!r}")
        device = next(self.lm.parameters()).device
        dtype = self.lm.temporal.text_emb.weight.dtype
        state = inputs.get("lm_state") or []
        if state:
            cache, frame = DelayedTokenCache.from_state_tensor(state[0].to(device))
        elif graph_walk == cfg.CONDITIONING_WALK:
            cache, frame = DelayedTokenCache(1, device=device), 0
        else:
            raise ValueError("the generate walk needs the conditioning lm_state")
        first, end = _planned_frames(graph_walk, step_metadata)
        if frame >= end:
            return None
        if frame < first:
            raise ValueError(f"lm_state is at frame {frame}, before this walk's first frame {first}")

        if graph_walk == cfg.CONDITIONING_WALK:
            return self._prepare_conditioning(frame, cache, step_metadata, inputs, device, dtype)
        codes = inputs["user_audio_codes"][0].to(device)
        if tuple(codes.shape) != (1, cfg.AUDIO_CODEBOOKS_PER_STREAM, int(step_metadata["num_user_frames"])):
            raise ValueError(f"user_audio_codes has shape {tuple(codes.shape)}")
        j = frame - first
        if not cache.prepare(user=codes[:, :, j : j + 1].long()):
            raise RuntimeError("a user frame must run a forward")
        return LMFrame(graph_walk, frame, cache, self.lm.temporal.embed_codes(cache.model_input)[:, 0].to(dtype))

    def _prepare_conditioning(self, frame, cache, meta, inputs, device, dtype) -> LMFrame:
        n_emb, n_text = int(meta["num_voice_embeddings"]), int(meta["num_text_tokens"])
        embeddings = inputs["voice_embeddings"][0]
        persona = inputs["persona_tokens"][0]
        if embeddings.shape[0] != n_emb or persona.shape[0] != n_text:
            raise ValueError("voice embeddings / persona tokens disagree with the request metadata")
        if frame < n_emb:
            # [PP] models/lm.py:853-873 and 1031-1038.
            cache.prepare_embedding()
            restore = inputs["voice_cache"][0].to(device).long() if frame == n_emb - 1 else None
            embeds = embeddings[frame].to(device=device, dtype=dtype).reshape(1, -1)
            return LMFrame(cfg.CONDITIONING_WALK, frame, cache, embeds, restore)
        # [PP] models/lm.py:1063-1121: agent = silence, user = sine, text = PAD or the persona token.
        index = frame - n_emb - cfg.SILENCE_FRAMES
        text = int(persona[index]) if 0 <= index < n_text else cfg.TEXT_PAD_TOKEN_ID
        def stream(tokens):
            return torch.tensor(tokens, dtype=torch.long, device=device).view(1, -1, 1)
        if not cache.prepare(user=stream(cfg.SINE_TOKENS), agent=stream(cfg.SILENCE_TOKENS),
                             text=torch.tensor([text], dtype=torch.long, device=device)):
            raise RuntimeError("conditioning token frames always run a forward after the voice replay")
        embeds = self.lm.temporal.embed_codes(cache.model_input)[:, 0].to(dtype)
        return LMFrame(cfg.CONDITIONING_WALK, frame, cache, embeds)

    def run_frame(
        self,
        frame: LMFrame,
        backend,
        sample_text: Callable[[torch.Tensor], torch.Tensor],
        sample_depth: Callable[[torch.Tensor, int], torch.Tensor],
    ) -> NameToTensorList:
        """``run_frames`` for one request."""
        return self.run_frames([frame], backend, sample_text, sample_depth)[0]

    @staticmethod
    def depth_steps(graph_walk: str, provided: torch.Tensor) -> int:
        """Depth steps a batch must run (0 for conditioning), from the ``[B, 17]`` forced mask; a host read.

        A generate frame always has the user's codebooks forced, so it never
        needs more than 8; the captured step runs exactly 8.
        """
        if graph_walk not in GENERATE_WALKS:
            return 0
        steps = depth_steps_needed(provided[:, 1:])
        if steps > cfg.AUDIO_CODEBOOKS_PER_STREAM:
            raise RuntimeError(f"a generate frame left a user codebook unforced ({steps} depth steps)")
        return steps

    def step(
        self,
        graph_walk: str,
        embeds: torch.Tensor,
        target: torch.Tensor,
        provided: torch.Tensor,
        backend,
        sample_text: Callable[[torch.Tensor], torch.Tensor],
        sample_depth: Callable[[torch.Tensor, int], torch.Tensor],
        depth_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Temporal forward, sampling and depformer for one frame of each row: tensors in, tensors out.

        ``embeds`` is ``[B, D]``; ``target``/``provided`` the cache's ``[B, 17]``
        target column and forced mask. Returns the raw sampled text ``[B]`` and
        audio ``[B, 16]`` that ``commit`` folds into each row's cache. Nothing
        here reads a tensor on the host, so the engine can capture it.
        """
        conditioning = graph_walk == cfg.CONDITIONING_WALK
        hidden, text_logits = self.lm.temporal.forward_embeddings(embeds, backend, text_logits=not conditioning)
        if conditioning:
            return target[:, 0], target[:, 1:]
        sampled_text = _sampled_token_ids(sample_text(text_logits.float()), target[:, 0], "text sampler")
        next_text = torch.where(provided[:, 0], target[:, 0], sampled_text)

        def sample_audio(logits: torch.Tensor, step: int) -> torch.Tensor:
            return _sampled_token_ids(sample_depth(logits, step), target[:, 0], f"depth sampler step {step}")

        audio = self.lm.depth.run_frame(
            next_text, hidden, target[:, 1:], provided[:, 1:], sample_audio, steps=depth_steps
        )
        return sampled_text, audio

    @staticmethod
    def commit(frame: LMFrame, sampled_text: torch.Tensor, audio: torch.Tensor) -> NameToTensorList:
        """Fold one row's ``[1]`` text and ``[1, 16]`` audio samples into its cache and emit its outputs."""
        cache = frame.cache
        output = cache.commit(sampled_text, audio)
        if frame.restore_cache is not None:
            cache.load(frame.restore_cache)
        outputs = {"lm_state": [cache.state_tensor(frame.frame + 1)]}
        if frame.walk in GENERATE_WALKS:
            if output is None:
                raise RuntimeError("generate frames always gather an output")
            # Delay-0 channels from the previous step, delay-1 from this one ([PP] models/lm.py:899-952).
            outputs["agent_frame"] = [output[:, 1 : 1 + cfg.AUDIO_CODEBOOKS_PER_STREAM].contiguous()]
            # B=1 column views can report contiguous while retaining stride 17;
            # worker byte serialization requires a physical stride of one.
            outputs["text_frame"] = [output[:, 0, 0].clone(memory_format=torch.contiguous_format)]
        return outputs

    @staticmethod
    def _target_rows(frames: list[LMFrame]) -> tuple[torch.Tensor, torch.Tensor]:
        """Snapshots of each row's target column and forced mask, ``[B, 17]``: commit rewrites the column."""
        target = torch.cat([frame.cache.target[:, :, 0] for frame in frames])
        provided = torch.cat([frame.cache.target_provided[:, :, 0] for frame in frames])
        return target, provided

    @staticmethod
    def _check_walk(graph_walk: str, frames: list[LMFrame], provided: torch.Tensor) -> None:
        if any(frame.walk != graph_walk for frame in frames):
            raise ValueError("a PersonaPlex LM batch must hold one walk")
        if graph_walk == cfg.CONDITIONING_WALK and not bool(provided.all()):
            raise RuntimeError(f"conditioning frame {frames[0].frame} has an unprovided target")

    def run_frames(
        self,
        frames: list[LMFrame],
        backend,
        sample_text: Callable[[torch.Tensor], torch.Tensor],
        sample_depth: Callable[[torch.Tensor, int], torch.Tensor],
        depth_steps: int | None = None,
    ) -> list[NameToTensorList]:
        """``step`` then ``commit`` for one frame of each request, as the engine runs them.

        Row ``i`` of the batch is ``frames[i]``; the samplers map ``[B, vocab]``
        logits to ``[B]`` tokens. ``depth_steps`` defaults to what the forced
        mask needs. Returns each request's outputs in order.
        """
        walk = frames[0].walk
        target, provided = self._target_rows(frames)
        self._check_walk(walk, frames, provided)
        if depth_steps is None:
            depth_steps = self.depth_steps(walk, provided)
        embeds = torch.cat([frame.embeds for frame in frames])
        text, audio = self.step(walk, embeds, target, provided, backend, sample_text, sample_depth, depth_steps)
        return [self.commit(frame, text[row : row + 1], audio[row : row + 1]) for row, frame in enumerate(frames)]

    # ------------------------------------------------------------------ engine contract
    #
    # prepare_inputs (per request) rebuilds the cache and stages its target
    # column; preprocess stacks the batch's tensors; forward/forward_batched run
    # ``step`` (captured for the generate walks); postprocess runs ``commit``
    # per request. Capture padding rows are clones of the config's template,
    # with tensors but no cache, so preprocess reads tensors only.

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs
    ) -> ARNodeInputs | None:
        del kwargs
        with torch.no_grad():
            frame = self.prepare_frame(graph_walk, fwd_info.step_metadata, inputs)
            if frame is None:
                return None
            target, provided = self._target_rows([frame])
        return ARNodeInputs(input_embeds=frame.embeds, input_seq_len=1,
                            tensor_inputs={"target": target, "provided": provided}, kwargs={"frame": frame})

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs]
    ) -> dict[str, Any]:
        del engine_inputs
        with torch.no_grad():
            target = torch.cat([inp.tensor_inputs["target"] for inp in inputs])
            provided = torch.cat([inp.tensor_inputs["provided"] for inp in inputs])
            frames = [inp.kwargs["frame"] for inp in inputs if "frame" in inp.kwargs]
            self._check_walk(graph_walk, frames, provided[: len(frames)])
            return {
                "input_embeds": torch.cat([inp.input_embeds for inp in inputs]),
                "target": target,
                "provided": provided,
                "depth_steps": self.depth_steps(graph_walk, provided),
            }

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        """One causal token per request; samplers only where frames sample."""
        del slot_lease, piecewise_leases, kwargs
        steps = {cfg.LM_KV: KVStep(), cfg.LM_ATTN: AttentionStep(causal=True), cfg.LM_POS: PositionStep()}
        if graph_walk in GENERATE_WALKS:
            steps[cfg.TEXT_SAMPLER] = SamplerStep(apply_penalty=False)
            steps[cfg.DEPTH_SAMPLER] = SamplerStep(apply_penalty=False)
        return SubmoduleStep(
            segments=[Segment(request_id=rid, label=cfg.KV_LABEL, span=inp.input_seq_len)
                      for rid, inp in zip(request_ids, inputs, strict=True)],
            steps=steps,
        )

    def _engine_step(self, graph_walk, engine_inputs, input_embeds, target, provided, depth_steps):
        if self._backend is None:
            raise RuntimeError("LM resources are not bound")
        self._backend.begin()
        rids = engine_inputs.request_ids
        resources = engine_inputs.resources
        text_sampler, depth_sampler = resources.get(cfg.TEXT_SAMPLER), resources.get(cfg.DEPTH_SAMPLER)
        return self.step(
            graph_walk, input_embeds, target, provided, self._backend,
            lambda logits: text_sampler.sample(rids, logits),
            lambda logits, _step: depth_sampler.sample(rids, logits.float()),
            depth_steps,
        )

    @torch.no_grad()
    def forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_embeds: torch.Tensor,
        target: torch.Tensor, provided: torch.Tensor, depth_steps: int, **kwargs
    ) -> NameToTensorList:
        del kwargs
        text, audio = self._engine_step(graph_walk, engine_inputs, input_embeds, target, provided, depth_steps)
        return {"sampled_text": [text], "sampled_audio": [audio]}

    @torch.no_grad()
    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_embeds: torch.Tensor,
        target: torch.Tensor, provided: torch.Tensor, depth_steps: int, **kwargs
    ) -> BatchedModelOutput:
        # Row outputs: the engine clones the real rows once, so a later replay
        # cannot overwrite them. No check_stop_buffers: the stop check must
        # read the lm_state that postprocess adds.
        del kwargs
        text, audio = self._engine_step(graph_walk, engine_inputs, input_embeds, target, provided, depth_steps)
        return BatchedModelOutput(row_outputs={"sampled_text": text, "sampled_audio": audio})

    def postprocess(self, request_id, request_info, outputs, inputs=None, **kwargs) -> None:
        """Commit this request's samples into its own cache (outside any captured graph)."""
        del request_id, request_info, kwargs
        frame = inputs.kwargs["frame"]
        text = outputs.pop("sampled_text")[0].reshape(1)
        audio = outputs.pop("sampled_audio")[0].reshape(1, cfg.NUM_AUDIO_CODEBOOKS)
        outputs.update(self.commit(frame, text, audio))

    def get_accelerator_graph_configs(self, device: torch.device, tp_world_size: int = 1) -> list:
        """Capture the generate step at batch sizes up to the deployment's ``max_batch_size``.

        The template row forces only the user's codebooks, as every generate
        frame does, so the captured depformer runs 8 steps. A request's first
        frame needs 1 of them; the other 7 are forced, so greedy output is
        unchanged (sampling draws 7 extra samples on that frame).
        """
        del tp_world_size
        from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig

        dtype = self.lm.temporal.text_emb.weight.dtype
        provided = torch.zeros(1, cfg.NUM_CODEBOOKS, dtype=torch.bool, device=device)
        provided[:, 1 + cfg.AUDIO_CODEBOOKS_PER_STREAM :] = True
        template = ARNodeInputs(
            input_embeds=torch.zeros(1, self.lm.config.dim, dtype=dtype, device=device), input_seq_len=1,
            tensor_inputs={"target": torch.zeros(1, cfg.NUM_CODEBOOKS, dtype=torch.long, device=device),
                           "provided": provided},
        )
        cap = self._max_batch_size
        sizes = sorted({size for size in (1, 2, 4, 8, 16, 32) if size <= cap} | {cap})
        return [BatchedAcceleratorGraphConfig(
            capture_graph_walk=cfg.GENERATE_WALK,
            replay_graph_walks=[cfg.GENERATE_WALK, cfg.GENERATE_AUDIO_WALK],
            single_request_inputs=template,
            capture_batch_sizes=sizes,
            compile=False,
            input_seq_dims={"input_embeds": 0, "target": 0, "provided": 0},
        )]

    def check_stop(self, request_id: str, request_info: CurrentForwardPassInfo, outputs) -> set[str]:
        """Finish the walk's loop after its last planned frame (read off ``lm_state``)."""
        del request_id
        state = (outputs.get("lm_state") or [None])[0]
        walk = request_info.graph_walk
        if state is None or walk not in LM_WALKS:
            return set()
        _, end = _planned_frames(walk, request_info.step_metadata)
        if int(state.reshape(-1)[-1]) >= end:
            return {cfg.CONDITIONING_LOOP if walk == cfg.CONDITIONING_WALK else cfg.DECODE_LOOP}
        return set()

    def can_batch(self, batch, model_inputs: list[NodeInputs]) -> bool:
        # A batch holds one walk, so its rows agree on which heads run; one row takes ``forward``.
        del batch
        return len(model_inputs) > 1

    def max_batch_size(self, graph_walk: str) -> int:
        del graph_walk
        return self._max_batch_size


class _MimiSubmodule(NodeSubmodule):
    """Mimi runs in fp32 like the reference ([PP] models/loaders.py:129-163).

    The encoder runs once per request over its whole clip and does not batch;
    the decoder batches chunks whose windows have the same length.
    """

    disable_autocast = True
    disable_torch_compile = True

    def __init__(self, codec: MimiCodec):
        super().__init__()
        self.codec = codec


class MimiEncoderSubmodule(_MimiSubmodule):
    """User PCM -> ``user_audio_codes`` ``[1, 8, N_user]``, the last frame zero-padded ([PP] models/lm.py:131-150)."""

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> NodeInputs:
        del graph_walk, kwargs
        audio = inputs["audio_inputs"][0]
        pcm = pad_to_frames(audio.to(device=self.get_device(), dtype=torch.float32).reshape(1, 1, -1))
        expected = int(fwd_info.step_metadata["num_user_frames"]) * cfg.FRAME_SIZE
        if pcm.shape[-1] != expected:
            raise ValueError(f"user audio pads to {pcm.shape[-1]} samples, metadata expects {expected}")
        return NodeInputs(tensor_inputs={"pcm": pcm})

    @torch.no_grad()
    def forward(self, graph_walk, engine_inputs, pcm: torch.Tensor, **kwargs) -> NameToTensorList:
        del graph_walk, engine_inputs, kwargs
        return {"user_audio_codes": [self.codec.encode(pcm)]}


class MimiDecoderSubmodule(_MimiSubmodule):
    """Streamed agent frames -> 24 kHz PCM ([PP] offline.py:300-307 decodes the whole track).

    Frames are decoded in fixed blocks (``block_frames``), each behind the
    request's last ``context_frames`` decoded frames, and only the block's
    samples are emitted. Frames that do not yet fill a block wait in the
    request's state until the next chunk or the stream's end, so a request's
    audio depends only on its frames, not on how the stream grouped them.
    Past the context, the decoder's 8 stacked windowed attention layers reach
    further back than the window, so a block can differ slightly from a
    whole-track decode (the same trade-off as Qwen3-TTS's codec). Emission
    stops at the input's sample count; the stream can end on an empty chunk,
    which still runs.
    """

    CONTEXT_KEY = "mimi_context"  # [8, n] agent codes, the last n <= context_frames decoded frames
    PENDING_KEY = "mimi_pending"  # [8, p] agent codes received but not yet decoded
    DECODED_KEY = "mimi_decoded"  # frames decoded so far
    EMITTED_KEY = "mimi_emitted"  # samples sent so far

    def __init__(self, codec: MimiCodec, context_frames: int = cfg.STREAM_CONTEXT_FRAMES,
                 max_batch_size: int = cfg.DEFAULT_MAX_BATCH_SIZE):
        super().__init__(codec)
        self.context_frames = context_frames
        self._max_batch_size = max_batch_size

    @staticmethod
    def stream_codes(chunk: torch.Tensor) -> torch.Tensor:
        """``[k, 1, 8, 1]`` (k stacked ``[1, 8, 1]`` frames) or one ``[1, 8, 1]`` frame -> ``[8, k]``, in order."""
        if chunk.dim() == 4 and chunk.shape[1:] == (1, cfg.AUDIO_CODEBOOKS_PER_STREAM, 1):
            frames = chunk[:, 0, :, 0]
        elif chunk.dim() == 3 and chunk.shape == (1, cfg.AUDIO_CODEBOOKS_PER_STREAM, 1):
            frames = chunk[:, :, 0]
        else:
            raise ValueError(f"agent frames must be [k, 1, 8, 1] or [1, 8, 1], got {tuple(chunk.shape)}")
        return frames.transpose(0, 1).contiguous()

    @staticmethod
    def block_frames(decoded: int) -> int:
        """Frames in the block that starts after ``decoded`` frames: ``STREAM_CHUNK_SCHEDULE``, then
        ``STREAM_CHUNK_FRAMES`` each, the same sizes the stream's chunk policy releases."""
        start = 0
        for size in cfg.STREAM_CHUNK_SCHEDULE:
            if decoded < start + size:
                return start + size - decoded
            start += size
        return cfg.STREAM_CHUNK_FRAMES - (decoded - start) % cfg.STREAM_CHUNK_FRAMES

    def prepare_inputs(self, graph_walk, fwd_info, inputs, is_final_stream_chunk=False, **kwargs) -> NodeInputs | None:
        del graph_walk, kwargs
        static = {"num_samples": int(fwd_info.step_metadata["num_samples"]), "final": bool(is_final_stream_chunk)}
        chunk = inputs.get("agent_frame") or []
        if not chunk:
            # A stream can end on an empty final chunk; it must still run to flush and close the stream.
            return NodeInputs(kwargs=static) if is_final_stream_chunk else None
        codes = self.stream_codes(chunk[0].to(self.get_device()).long())
        return NodeInputs(tensor_inputs={"codes": codes}, kwargs=static, input_seq_len=codes.shape[1])

    def preprocess(self, graph_walk, engine_inputs, inputs):
        del graph_walk, engine_inputs
        return {"codes": [inp.tensor_inputs.get("codes") for inp in inputs],
                "num_samples": [inp.kwargs["num_samples"] for inp in inputs],
                "final": [inp.kwargs["final"] for inp in inputs]}

    def can_batch(self, batch, model_inputs) -> bool:
        del batch, model_inputs
        return True

    def max_batch_size(self, graph_walk: str) -> int:
        # The deployment's cap, as for the LM: 1 decodes every request alone, so its audio does not
        # depend on what else is being served (batch shape changes TF32 rounding).
        del graph_walk
        return self._max_batch_size

    @torch.no_grad()
    def forward_batched(self, graph_walk, engine_inputs, codes, num_samples, final,
                        **kwargs) -> dict[str, NameToTensorList]:
        """Decode every full block (and, at the stream's end, the rest); rows whose blocks have equal
        shapes share one decode.

        The context and pending frames are read and written here, in the forward, so the next chunk
        of a request always sees this one's frames.
        """
        del graph_walk, kwargs
        rids = list(engine_inputs.request_ids)
        states = [self.request_state(rid) for rid in rids]
        pending = []
        for state, new in zip(states, codes, strict=True):
            held = state.get(self.PENDING_KEY)
            pending.append(new if held is None else held if new is None else torch.cat([held, new], dim=1))
        pieces: list[list[torch.Tensor]] = [[] for _ in rids]
        while True:
            groups: dict[tuple[int, int], list[tuple[int, torch.Tensor]]] = {}
            for i, state in enumerate(states):
                if pending[i] is None or pending[i].shape[1] == 0:
                    continue
                size = self.block_frames(state.get(self.DECODED_KEY, 0))
                if pending[i].shape[1] < size and not final[i]:
                    continue
                size = min(size, pending[i].shape[1])
                context = state.get(self.CONTEXT_KEY)
                block = pending[i][:, :size]
                window = block if context is None else torch.cat([context, block], dim=1)
                groups.setdefault((window.shape[1], size), []).append((i, window))
            if not groups:
                break
            for (_, size), rows in groups.items():
                pcm = self.codec.decode_last(torch.stack([w for _, w in rows]), size, cfg.SEANET_MARGIN_FRAMES)
                for row, (i, window) in enumerate(rows):
                    state = states[i]
                    pieces[i].append(pcm[row : row + 1])
                    # an explicit start: a -0 slice would keep the whole history
                    state.add(self.CONTEXT_KEY, window[:, max(window.shape[1] - self.context_frames, 0) :])
                    state.add(self.DECODED_KEY, state.get(self.DECODED_KEY, 0) + size)
                    pending[i] = pending[i][:, size:]
        out: dict[str, NameToTensorList] = {}
        device = self.get_device()
        for i, (rid, state) in enumerate(zip(rids, states, strict=True)):
            if pending[i] is not None and pending[i].shape[1]:
                state.add(self.PENDING_KEY, pending[i])
            else:
                state.remove(self.PENDING_KEY)
            emitted = state.get(self.EMITTED_KEY, 0)
            fresh = torch.cat(pieces[i], dim=-1) if pieces[i] else torch.zeros(1, 1, 0, device=device)
            fresh = fresh[..., : max(num_samples[i] - emitted, 0)]
            state.add(self.EMITTED_KEY, emitted + fresh.shape[-1])
            out[rid] = {"audio_output": [fresh.contiguous()]}
        return out

    def forward(self, graph_walk, engine_inputs, codes, num_samples, final, **kwargs) -> NameToTensorList:
        outputs = self.forward_batched(graph_walk, engine_inputs, codes, num_samples, final, **kwargs)
        return outputs[engine_inputs.request_ids[0]]
