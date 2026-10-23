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
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.model.submodule_base import ARNodeInputs, ARNodeSubmodule, ModelInputsFromEngine, NodeInputs, NodeSubmodule

from . import config as cfg
from .components.delayed_cache import DelayedTokenCache
from .components.mimi import MimiCodec, pad_to_frames
from .weights import PersonaPlexLM

KV_LABEL = "main"
GENERATE_WALKS = frozenset({cfg.GENERATE_WALK, cfg.GENERATE_AUDIO_WALK})
LM_WALKS = GENERATE_WALKS | {cfg.CONDITIONING_WALK}


class ResourceTemporalBackend:
    """Temporal attention on M*'s paged KV, attention and position resources."""

    def __init__(self, kv, attn, pos, rope_theta: float, label: str = KV_LABEL):
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
    """Temporal transformer + depformer + delayed token cache, batch size 1."""

    disable_autocast = True
    disable_torch_compile = True

    def __init__(self, lm: PersonaPlexLM):
        super().__init__()
        self.lm = lm
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
        """Temporal forward, sampling, depformer and cache commit for one frame."""
        conditioning = frame.walk == cfg.CONDITIONING_WALK
        hidden, text_logits = self.lm.temporal.forward_embeddings(frame.embeds, backend, text_logits=not conditioning)
        cache = frame.cache
        target = cache.target[:, :, 0].clone()
        provided = cache.target_provided[:, :, 0].clone()
        if conditioning:
            if not cache.target_all_provided():
                raise RuntimeError(f"conditioning frame {frame.frame} has an unprovided target")
            output = cache.commit(target[:, 0], target[:, 1:])
        else:
            sampled_text = _sampled_token_ids(sample_text(text_logits.float()), target[:, 0], "text sampler")
            next_text = torch.where(provided[:, 0], target[:, 0], sampled_text)
            def sample_audio(logits: torch.Tensor, step: int) -> torch.Tensor:
                return _sampled_token_ids(sample_depth(logits, step), target[:, 0], f"depth sampler step {step}")

            audio = self.lm.depth.run_frame(next_text, hidden, target[:, 1:], provided[:, 1:], sample_audio)
            output = cache.commit(sampled_text, audio)
            if output is None:
                raise RuntimeError("generate frames always gather an output")
        if frame.restore_cache is not None:
            cache.load(frame.restore_cache)
        outputs = {"lm_state": [cache.state_tensor(frame.frame + 1)]}
        if not conditioning:
            # Delay-0 channels from the previous step, delay-1 from this one ([PP] models/lm.py:899-952).
            outputs["agent_frame"] = [output[:, 1 : 1 + cfg.AUDIO_CODEBOOKS_PER_STREAM].contiguous()]
            # B=1 column views can report contiguous while retaining stride 17;
            # worker byte serialization requires a physical stride of one.
            outputs["text_frame"] = [output[:, 0, 0].clone(memory_format=torch.contiguous_format)]
        return outputs

    # ------------------------------------------------------------------ engine contract

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs
    ) -> ARNodeInputs | None:
        del kwargs
        with torch.no_grad():
            frame = self.prepare_frame(graph_walk, fwd_info.step_metadata, inputs)
        if frame is None:
            return None
        return ARNodeInputs(input_embeds=frame.embeds, input_seq_len=1, kwargs={"frame": frame})

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs]
    ) -> dict[str, Any]:
        del graph_walk, engine_inputs
        if len(inputs) != 1:
            raise NotImplementedError("PersonaPlex serves batch size 1")
        return {"frame": inputs[0].kwargs["frame"]}

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
            segments=[Segment(request_id=rid, label=KV_LABEL, span=inp.input_seq_len)
                      for rid, inp in zip(request_ids, inputs, strict=True)],
            steps=steps,
        )

    @torch.no_grad()
    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, frame: LMFrame, **kwargs):
        del graph_walk, kwargs
        if self._backend is None:
            raise RuntimeError("LM resources are not bound")
        self._backend.begin()
        rids = engine_inputs.request_ids
        resources = engine_inputs.resources
        text_sampler, depth_sampler = resources.get(cfg.TEXT_SAMPLER), resources.get(cfg.DEPTH_SAMPLER)
        return self.run_frame(
            frame, self._backend,
            lambda logits: text_sampler.sample(rids, logits),
            lambda logits, _step: depth_sampler.sample(rids, logits.float()),
        )

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
        del batch, model_inputs
        return False


class _MimiSubmodule(NodeSubmodule):
    """Mimi runs in fp32 like the reference ([PP] models/loaders.py:129-163)."""

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
    """Accumulated agent frames -> PCM trimmed to the input length ([PP] offline.py:300-307)."""

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> NodeInputs:
        del graph_walk, kwargs
        frames = inputs["agent_frame"]
        meta = fwd_info.step_metadata
        if len(frames) != int(meta["num_user_frames"]):
            raise ValueError(f"MimiDecoder got {len(frames)} agent frames, expected {meta['num_user_frames']}")
        codes = torch.cat([f.to(self.get_device()).long().reshape(1, cfg.AUDIO_CODEBOOKS_PER_STREAM, 1)
                           for f in frames], dim=-1)
        return NodeInputs(tensor_inputs={"codes": codes}, kwargs={"num_samples": int(meta["num_samples"])})

    @torch.no_grad()
    def forward(self, graph_walk, engine_inputs, codes: torch.Tensor, num_samples: int, **kwargs):
        del graph_walk, engine_inputs, kwargs
        pcm = self.codec.decode(codes)
        if pcm.shape[-1] < num_samples:
            pcm = torch.nn.functional.pad(pcm, (0, num_samples - pcm.shape[-1]))
        return {"audio_output": [pcm[..., :num_samples]]}
