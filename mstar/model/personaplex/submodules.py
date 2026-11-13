"""PersonaPlex node submodules: LM, MimiEncoder and MimiDecoder.

The LM runs one reference ``LMGen`` step per graph iteration ([PP]
models/lm.py:814-958). Frame ``i`` of the request is:

    conditioning, i < N_emb:          replay voice embedding i (step_embeddings);
                                      restore the preset cache after the last
    conditioning, then 6 + N_text + 6: silence / persona token / silence frames
    generate, j = i - conditioning:   user frame j; sample text and 16 depth codes

Conditioning targets are fully provided (every channel is forced), so those
frames skip the text head, depformer and samplers: NVIDIA runs them, but only
to consume RNG, and the cache keeps the forced values either way. So conditioning
does not depend on the model's output: its cache stays on the host, and each frame
writes the result to the request's slot in the ``LM_CACHE`` pool.

The generate step reads each row's cache from the pool, updates it and writes it
back, all as tensor work with no host reads, so the step is captured as one
accelerator graph. Requests in the same walk batch, at their own offsets. The user's
codebooks are always forced, so the depformer runs 8 of its 16 depth steps. A
request's first frame also forces the agent's acoustic codebooks, so only the first
of those samples anything that is kept; the other 7 draw samples that are dropped.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.engine.resources.recurrent import RecurrentStatePool, RecurrentStep
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
    StopPolicy,
)
from mstar.utils.h2d import PinnedStager

from . import config as cfg
from .components.delayed_cache import CacheRows, DelayedTokenCache, commit_rows, prepare_rows
from .components.mimi import MimiCodec, pad_to_frames
from .components.mimi_stream import IncrementalMimiDecoder
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


@dataclass(slots=True)
class ConditioningProgress:
    """A request's walk through conditioning, on the host."""

    cache: DelayedTokenCache
    persona: list[int]
    voice_cache: torch.Tensor
    next_frame: int = 0
    # The last frame's inputs, for an engine that prepares the same frame again.
    prepared: ARNodeInputs | None = None


class ConditioningFrame(NamedTuple):
    embeds: torch.Tensor  # [1, D] temporal input
    model_input: torch.Tensor  # [1, 17, 1] the tokens it embeds
    rows: CacheRows  # the cache after this frame


class GenerateFrame(NamedTuple):
    rows: CacheRows
    agent_frame: torch.Tensor  # [B, 8, 1]
    text_frame: torch.Tensor  # [B]
    model_input: torch.Tensor  # [B, 17, 1]
    forced: torch.Tensor  # [B, 17]


def read_cache_rows(pool: RecurrentStatePool) -> CacheRows:
    """This step's rows from the pool. Only a padding row is fresh: conditioning wrote the others.
    Zeros are valid tokens, so it needs no start state."""
    return CacheRows(pool.gather("tokens"), pool.gather("provided").bool(), pool.gather("offset"))


def write_cache_rows(pool: RecurrentStatePool, rows: CacheRows) -> None:
    pool.scatter_("tokens", rows.tokens)
    pool.scatter_("provided", rows.provided.long())
    pool.scatter_("offset", rows.offset)


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


class LMSubmodule(ARNodeSubmodule):
    """Temporal transformer + depformer + delayed token cache, up to ``max_batch_size`` requests a step."""

    disable_autocast = True
    disable_torch_compile = True

    PROGRESS_KEY = "conditioning_progress"

    def __init__(self, lm: PersonaPlexLM, max_batch_size: int = cfg.DEFAULT_MAX_BATCH_SIZE):
        super().__init__()
        self.lm = lm
        self._max_batch_size = max_batch_size
        self._backend: ResourceTemporalBackend | None = None
        self.register_buffer("_delays", torch.tensor(cfg.DELAYS, device=next(lm.parameters()).device),
                             persistent=False)

    def bind_node_resources(self, resources: dict[str, Any]) -> None:
        super().bind_node_resources(resources)
        self._backend = ResourceTemporalBackend(
            resources[cfg.LM_KV], resources[cfg.LM_ATTN], resources[cfg.LM_POS], float(self.lm.config.rope_theta)
        )

    # ------------------------------------------------------------------ frame logic (engine-free)

    @staticmethod
    def start_conditioning(step_metadata: dict, inputs: NameToTensorList) -> ConditioningProgress:
        persona = inputs["persona_tokens"][0]
        if inputs["voice_embeddings"][0].shape[0] != int(step_metadata["num_voice_embeddings"]) or (
            persona.shape[0] != int(step_metadata["num_text_tokens"])
        ):
            raise ValueError("voice embeddings / persona tokens disagree with the request metadata")
        return ConditioningProgress(DelayedTokenCache(1), persona.tolist(), inputs["voice_cache"][0].cpu().long())

    def conditioning_frame(
        self, progress: ConditioningProgress, step_metadata: dict, inputs: NameToTensorList,
    ) -> ConditioningFrame:
        """Advance the host cache through one conditioning frame and embed its input."""
        frame, cache = progress.next_frame, progress.cache
        device = self.get_device()
        dtype = self.lm.temporal.text_emb.weight.dtype
        n_emb = int(step_metadata["num_voice_embeddings"])
        if frame < n_emb:
            # [PP] models/lm.py:853-873 and 1031-1038.
            cache.prepare_embedding()
            model_input = cache.model_input.clone()
            embeds = inputs["voice_embeddings"][0][frame].to(device=device, dtype=dtype).reshape(1, -1)
        else:
            # [PP] models/lm.py:1063-1121: agent = silence, user = sine, text = PAD or the persona token.
            index = frame - n_emb - cfg.SILENCE_FRAMES
            text = progress.persona[index] if 0 <= index < len(progress.persona) else cfg.TEXT_PAD_TOKEN_ID

            def stream(tokens):
                return torch.tensor(tokens, dtype=torch.long).view(1, -1, 1)

            if not cache.prepare(user=stream(cfg.SINE_TOKENS), agent=stream(cfg.SILENCE_TOKENS),
                                 text=torch.tensor([text], dtype=torch.long)):
                raise RuntimeError("conditioning token frames always run a forward after the voice replay")
            model_input = cache.model_input.clone()
            embeds = self.lm.temporal.embed_codes(model_input.to(device))[:, 0].to(dtype)
        target = cache.target[:, :, 0]
        cache.commit(target[:, 0].clone(), target[:, 1:].clone())
        if frame == n_emb - 1:
            cache.load(progress.voice_cache)
        progress.next_frame = frame + 1
        return ConditioningFrame(embeds, model_input, cache.to_rows())

    def generate_frame(
        self,
        rows: CacheRows,
        user: torch.Tensor,
        backend,
        sample_text: Callable[[torch.Tensor], torch.Tensor],
        sample_depth: Callable[[torch.Tensor, int], torch.Tensor],
    ) -> GenerateFrame:
        """One generate frame of each row: tensors in, tensors out, with no host reads.

        ``user`` is each row's user codes ``[B, 8]``; the samplers map ``[B, vocab]``
        logits to ``[B]`` tokens.
        """
        prepared = prepare_rows(rows, user, self._delays)
        target, forced = prepared.target, prepared.forced
        dtype = self.lm.temporal.text_emb.weight.dtype
        embeds = self.lm.temporal.embed_codes(prepared.model_input)[:, 0].to(dtype)
        hidden, text_logits = self.lm.temporal.forward_embeddings(embeds, backend)
        sampled_text = _sampled_token_ids(sample_text(text_logits.float()), target[:, 0], "text sampler")
        next_text = torch.where(forced[:, 0], target[:, 0], sampled_text)

        def sample_audio(logits: torch.Tensor, step: int) -> torch.Tensor:
            return _sampled_token_ids(sample_depth(logits, step), target[:, 0], f"depth sampler step {step}")

        audio = self.lm.depth.run_frame(
            next_text, hidden, target[:, 1:], forced[:, 1:], sample_audio, steps=cfg.AUDIO_CODEBOOKS_PER_STREAM
        )
        committed, output = commit_rows(prepared, sampled_text, audio, self._delays)
        # Delay-0 channels from the previous step, delay-1 from this one ([PP] models/lm.py:899-952).
        # B=1 column views can report contiguous while retaining stride 17;
        # worker byte serialization requires a physical stride of one.
        return GenerateFrame(
            committed,
            output[:, 1 : 1 + cfg.AUDIO_CODEBOOKS_PER_STREAM].contiguous(),
            output[:, 0, 0].clone(memory_format=torch.contiguous_format),
            prepared.model_input,
            forced,
        )

    # ------------------------------------------------------------------ engine contract

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs
    ) -> ARNodeInputs | None:
        """Per request, on the host: the frame comes from the loop's own count, not from a tensor.

        Returns None for an iteration past the walk's planned frames, which the engine skips.
        """
        del kwargs
        if graph_walk == cfg.CONDITIONING_WALK:
            with torch.no_grad():
                return self._prepare_conditioning(fwd_info, inputs)
        if graph_walk not in GENERATE_WALKS:
            raise ValueError(f"unknown PersonaPlex LM walk {graph_walk!r}")
        j = fwd_info.dynamic_loop_iter_counts.get(cfg.DECODE_LOOP, 0)
        if j >= int(fwd_info.step_metadata["num_user_frames"]):
            return None
        codes = inputs["user_audio_codes"][0]
        if j == 0:
            self.request_state(fwd_info.rid_handle).remove(self.PROGRESS_KEY)
            expected = (1, cfg.AUDIO_CODEBOOKS_PER_STREAM, int(fwd_info.step_metadata["num_user_frames"]))
            if tuple(codes.shape) != expected:
                raise ValueError(f"user_audio_codes has shape {tuple(codes.shape)}")
        return ARNodeInputs(input_seq_len=1, tensor_inputs={"user_frame": codes[:, :, j].to(self.get_device()).long()})

    def _prepare_conditioning(self, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList) -> ARNodeInputs | None:
        meta = fwd_info.step_metadata
        frame = fwd_info.dynamic_loop_iter_counts.get(cfg.CONDITIONING_LOOP, 0)
        if frame >= int(meta["conditioning_frames"]):
            return None
        state = self.request_state(fwd_info.rid_handle)
        progress = state.get(self.PROGRESS_KEY)
        if progress is None:
            progress = self.start_conditioning(meta, inputs)
            state.add(self.PROGRESS_KEY, progress)
        if frame == progress.next_frame - 1:
            return progress.prepared
        if frame != progress.next_frame:
            raise RuntimeError(f"conditioning frame {frame} prepared after frame {progress.next_frame - 1}")
        result = self.conditioning_frame(progress, meta, inputs)
        progress.prepared = ARNodeInputs(input_embeds=result.embeds, input_seq_len=1,
                                         tensor_inputs=result.rows._asdict())
        return progress.prepared

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs]
    ) -> dict[str, Any]:
        del engine_inputs
        with torch.no_grad():
            if graph_walk in GENERATE_WALKS:
                return {"user_frame": torch.cat([inp.tensor_inputs["user_frame"] for inp in inputs])}
            device = self.get_device()
            return {
                "input_embeds": torch.cat([inp.input_embeds for inp in inputs]),
                **{name: torch.cat([inp.tensor_inputs[name] for inp in inputs]).to(device)
                   for name in CacheRows._fields},
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
        steps = {cfg.LM_KV: KVStep(), cfg.LM_ATTN: AttentionStep(causal=True), cfg.LM_POS: PositionStep(),
                 cfg.LM_CACHE: RecurrentStep()}
        if graph_walk in GENERATE_WALKS:
            steps[cfg.TEXT_SAMPLER] = SamplerStep(apply_penalty=False)
            steps[cfg.DEPTH_SAMPLER] = SamplerStep(apply_penalty=False)
        return SubmoduleStep(
            segments=[Segment(request_id=rid, label=cfg.KV_LABEL, span=inp.input_seq_len)
                      for rid, inp in zip(request_ids, inputs, strict=True)],
            steps=steps,
        )

    def _engine_step(self, graph_walk, engine_inputs, **staged) -> dict[str, torch.Tensor]:
        if self._backend is None:
            raise RuntimeError("LM resources are not bound")
        self._backend.begin()
        resources = engine_inputs.resources
        pool = resources[cfg.LM_CACHE]
        if graph_walk == cfg.CONDITIONING_WALK:
            self.lm.temporal.forward_embeddings(staged["input_embeds"], self._backend, text_logits=False)
            rows = CacheRows(staged["tokens"], staged["provided"], staged["offset"])
            write_cache_rows(pool, rows)
            return {cfg.CONDITIONING_TICK: rows.offset}
        rids = engine_inputs.request_ids
        text_sampler, depth_sampler = resources[cfg.TEXT_SAMPLER], resources[cfg.DEPTH_SAMPLER]
        frame = self.generate_frame(
            read_cache_rows(pool), staged["user_frame"], self._backend,
            lambda logits: text_sampler.sample(rids, logits),
            lambda logits, _step: depth_sampler.sample(rids, logits.float()),
        )
        write_cache_rows(pool, frame.rows)
        return {"agent_frame": frame.agent_frame, "text_frame": frame.text_frame}

    @torch.no_grad()
    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **staged) -> NameToTensorList:
        outputs = self._engine_step(graph_walk, engine_inputs, **staged)
        return {name: [tensor] for name, tensor in outputs.items()}

    @torch.no_grad()
    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **staged
    ) -> BatchedModelOutput:
        # Row outputs: the engine clones the real rows once, so a later replay cannot overwrite them.
        return BatchedModelOutput(row_outputs=self._engine_step(graph_walk, engine_inputs, **staged))

    def get_accelerator_graph_configs(self, device: torch.device, tp_world_size: int = 1) -> list:
        """Capture the generate step at batch sizes up to the deployment's ``max_batch_size``."""
        del tp_world_size
        from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig, geometric_batch_sizes

        template = ARNodeInputs(
            input_seq_len=1,
            tensor_inputs={"user_frame": torch.zeros(1, cfg.AUDIO_CODEBOOKS_PER_STREAM, dtype=torch.long,
                                                     device=device)},
        )
        return [BatchedAcceleratorGraphConfig(
            capture_graph_walk=cfg.GENERATE_WALK,
            replay_graph_walks=[cfg.GENERATE_WALK, cfg.GENERATE_AUDIO_WALK],
            single_request_inputs=template,
            capture_batch_sizes=geometric_batch_sizes(self._max_batch_size),
            compile=False,
            input_seq_dims={"user_frame": 0},
        )]

    def get_stop_policies(self, graph_walk: str) -> dict[str, StopPolicy] | None:
        """No row is read: each LM loop's cap is its exact frame count (``get_loop_caps``), and the
        runtime ends a loop at its cap."""
        return {} if graph_walk in LM_WALKS else None

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
    the decoder batches any chunks, padded to the widest.
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
    """Streamed agent frames -> 24 kHz PCM, equal to one whole-track decode ([PP] offline.py:300-307)
    up to floating-point rounding.

    Each chunk decodes only its new frames, from the stream's decoder state
    (``IncrementalMimiDecoder``), which lives in the ``mimi_state`` recurrent
    pool: a step gathers its rows' state and scatters the new state back. A
    batch pads its chunks to the widest, so on a GPU a row's rounding can
    depend on the chunks it is decoded with (``max_batch_size: 1`` decodes
    every request alone).
    Emission stops at the input's sample count; the stream can end on an empty
    chunk, which still runs.
    """

    EMITTED_KEY = "mimi_emitted"  # samples sent so far, kept by postprocess

    def __init__(self, codec: MimiCodec, max_batch_size: int = cfg.DEFAULT_MAX_BATCH_SIZE):
        super().__init__(codec)
        self.incremental = IncrementalMimiDecoder(codec)
        self._state_names = tuple(self.incremental.state_blocks())
        self._max_batch_size = max_batch_size
        self._lengths_stager = PinnedStager(torch.long, numel=max_batch_size)

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

    def prepare_inputs(self, graph_walk, fwd_info, inputs, is_final_stream_chunk=False, **kwargs) -> NodeInputs | None:
        del graph_walk, kwargs
        static = {"num_samples": int(fwd_info.step_metadata["num_samples"])}
        chunk = inputs.get("agent_frame") or []
        if not chunk:
            # A stream can end on an empty final chunk; it must still run to flush and close the stream.
            return NodeInputs(kwargs={**static, "frames": 0}) if is_final_stream_chunk else None
        codes = self.stream_codes(chunk[0].to(self.get_device()).long())
        return NodeInputs(tensor_inputs={"codes": codes}, kwargs={**static, "frames": codes.shape[1]},
                          input_seq_len=codes.shape[1])

    def declare_step(self, graph_walk, request_ids, inputs, slot_lease=None, piecewise_leases=None,
                     **kwargs) -> SubmoduleStep:
        """Every row reads and writes its stream's decoder state."""
        del graph_walk, inputs, slot_lease, piecewise_leases, kwargs
        return SubmoduleStep(segments=[Segment(request_id=rid, label="main", span=1) for rid in request_ids],
                             steps={cfg.MIMI_STATE: RecurrentStep()})

    def preprocess(self, graph_walk, engine_inputs, inputs):
        """The batch's chunks as ``[B, 8, width]``, right-padded to its widest, and each row's real length."""
        del graph_walk, engine_inputs
        frames = [inp.kwargs["frames"] for inp in inputs]
        width = max(frames)
        if width == 0:
            return {"codes": None, "lengths": None}
        device = self.get_device()
        rows = [inp.tensor_inputs["codes"] if n else torch.zeros(cfg.AUDIO_CODEBOOKS_PER_STREAM, 0, dtype=torch.long,
                                                                  device=device)
                for inp, n in zip(inputs, frames, strict=True)]
        codes = torch.stack([torch.nn.functional.pad(row, (0, width - row.shape[1])) for row in rows])
        lengths = torch.empty(len(inputs), dtype=torch.long, device=device)
        self._lengths_stager.copy_(lengths, frames)
        return {"codes": codes, "lengths": lengths}

    def can_batch(self, batch, model_inputs) -> bool:
        del batch, model_inputs
        return True

    def max_batch_size(self, graph_walk: str) -> int:
        # The deployment's cap, as for the LM: 1 decodes every request alone, so its audio does not
        # depend on what else is being served (batch shape changes TF32 rounding).
        del graph_walk
        return self._max_batch_size

    @torch.no_grad()
    def _decode(self, engine_inputs, codes, lengths) -> torch.Tensor:
        """``[B, 1, width * 1920]`` PCM from each row's state; a row's samples past its length are padding."""
        if codes is None:  # only empty final chunks: nothing to decode, no state to move
            return torch.zeros(len(engine_inputs.request_ids), 1, 0, device=self.get_device())
        pool: RecurrentStatePool = engine_inputs.resources[cfg.MIMI_STATE]
        # a fresh stream, and a padding row on the sink, read zeros: the decoder's own left padding
        state = {name: pool.gather(name) for name in self._state_names}
        pcm, new = self.incremental(codes, state, lengths)
        for name, value in new.items():
            pool.scatter_(name, value)
        return pcm

    def forward(self, graph_walk, engine_inputs, codes, lengths, **kwargs) -> NameToTensorList:
        del graph_walk, kwargs
        return {"audio_output": [self._decode(engine_inputs, codes, lengths)]}

    def forward_batched(self, graph_walk, engine_inputs, codes, lengths, **kwargs) -> BatchedModelOutput:
        del graph_walk, kwargs
        return BatchedModelOutput(row_outputs={"audio_output": self._decode(engine_inputs, codes, lengths)})

    def postprocess(self, request_id, request_info, outputs, inputs=None, **kwargs) -> None:
        """Cut the row to its chunk's frames and stop at the input's sample count.

        Runs once per executed step, in step order, so the count of samples sent
        is exact whatever the worker speculated.
        """
        del request_info, kwargs
        if "audio_output" not in outputs:
            return
        if inputs is None:
            raise ValueError("decoder postprocess needs the pass's inputs for its chunk length")
        state = self.request_state(request_id)
        emitted = state.get(self.EMITTED_KEY, 0)
        audio = outputs["audio_output"][0].reshape(-1)[: inputs.kwargs["frames"] * cfg.FRAME_SIZE]
        audio = audio[: max(inputs.kwargs["num_samples"] - emitted, 0)]
        state.add(self.EMITTED_KEY, emitted + audio.shape[-1])
        outputs["audio_output"][0] = audio.reshape(1, 1, -1)
