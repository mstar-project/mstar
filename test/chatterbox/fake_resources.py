"""CPU stand-ins for the KV, attention and position resources a T3 layer
stack attends through, so the components run under pytest without
FlashInfer or a GPU.

``FakeKVAttention`` follows ``test/modular/vjepa2/fake_resources.py``: the
K/V written this step are appended to a per-(label, layer, request) history
and attended with causal SDPA, which is what the paged kernel computes.
``FakePositionManager`` plans absolute positions from per-request counters
and applies Llama-3 RoPE the way HF does (rotate-half, smoothed inverse
frequencies), which is the convention FlashInfer's ``llama31`` kernel
implements. Both take the step layout from ``plan(seq_lens)``, and serve a
stack through its state handles (``FakeT3State``) or through the resource
cursors of a stack still on them (CosyVoice3's LM, ``FakeT3Resources``).
``FakeS3GenStream`` holds S3Gen's stream Value.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from mstar.engine.resources.position.config import PositionConfig
from mstar.engine.resources.step import StepContext
from mstar.engine.state import ArrayStep, Segment, SubmoduleStep
from mstar.engine.state.adapters import node_handles, node_resources, state_steps
from mstar.engine.state.host import HostFields
from mstar.model.chatterbox.chatterbox_model import ChatterboxModel
from mstar.model.chatterbox.config import (
    COND_LABEL,
    S3GEN_NODE,
    S3GEN_STREAM,
    T3_ATTN,
    T3_KV,
    T3_NODE,
    T3_POS,
    T3_SAMPLER,
    ChatterboxConfig,
)
from mstar.model.submodule_base import NodeInputs, engine_step


def _llama3_inv_freq(
    head_dim: int,
    rope_theta: float,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    old_context_len: float,
) -> torch.Tensor:
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    low_wavelen = old_context_len / low_freq_factor
    high_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    scaled = torch.where(wavelen > low_wavelen, inv_freq / factor, inv_freq)
    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    medium = ~(wavelen < high_wavelen) & ~(wavelen > low_wavelen)
    return torch.where(medium, smoothed, scaled)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class FakeKVAttention:
    """One object serving as both the KV and the attention resource."""

    requires_kv_write = True

    def __init__(self):
        self._label = "main"
        self._layer_idx = 0
        self._seq_lens: list[int] = [1]
        # (label, layer) -> per-request [k history, v history]
        self._kv: dict[tuple[str, int], list[tuple[list[torch.Tensor], list[torch.Tensor]]]] = {}
        self._pending: tuple[torch.Tensor, torch.Tensor] | None = None

    # resource cursors (see AttentionCallable)
    @property
    def default_label(self) -> str:
        return self._label

    def set_default_label(self, label: str) -> None:
        self._label = label

    def set_default_layer_idx(self, layer_idx: int) -> None:
        self._layer_idx = layer_idx

    def layer_view(self, layer: int | None = None) -> int:
        return self._layer_idx if layer is None else layer

    def opens_label(self, rid, label: str) -> bool:
        return True

    def plan(self, seq_lens: list[int]) -> None:
        """Per-request token counts of the packed step about to run."""
        self._seq_lens = list(seq_lens)

    def reset(self) -> None:
        self._kv.clear()
        self._pending = None

    def write_kv(self, k: torch.Tensor, v: torch.Tensor, layer_idx=None, label=None) -> None:
        self._pending = (k, v)

    def select_last_hidden(self, hidden: torch.Tensor, label: str = "main") -> torch.Tensor:
        """The last row of every packed segment, as the paged manager's
        ``select_last_hidden`` picks them off ``qo_indptr``."""
        del label
        ends = torch.tensor(self._seq_lens).cumsum(0) - 1
        return hidden.index_select(0, ends)

    def run(self, q: torch.Tensor, kv_cache_layer, k=None, v=None, label=None, layer_idx=None) -> torch.Tensor:
        del k, v, layer_idx
        assert self._pending is not None, "run without a preceding write_kv"
        k_all, v_all = self._pending
        self._pending = None
        key = (label or self._label, kv_cache_layer)
        if key not in self._kv:
            self._kv[key] = [([], []) for _ in self._seq_lens]
        histories = self._kv[key]
        assert len(histories) == len(self._seq_lens), "batch shape changed mid-request"

        outputs = []
        for i, (q_i, k_i, v_i) in enumerate(zip(
            torch.split(q, self._seq_lens),
            torch.split(k_all, self._seq_lens),
            torch.split(v_all, self._seq_lens),
            strict=True,
        )):
            histories[i][0].append(k_i)
            histories[i][1].append(v_i)
            keys = torch.cat(histories[i][0], dim=0)   # [S, Hkv, D]
            vals = torch.cat(histories[i][1], dim=0)
            n_q, n_k = q_i.shape[0], keys.shape[0]
            # this step's rows sit at the end of the stream: causal there,
            # unrestricted over the resident prefix
            q_pos = torch.arange(n_k - n_q, n_k).unsqueeze(1)
            k_pos = torch.arange(n_k).unsqueeze(0)
            mask = k_pos <= q_pos
            out = F.scaled_dot_product_attention(
                q_i.permute(1, 0, 2).unsqueeze(0),
                keys.permute(1, 0, 2).unsqueeze(0),
                vals.permute(1, 0, 2).unsqueeze(0),
                attn_mask=mask,
                enable_gqa=q_i.shape[1] != keys.shape[1],
            )
            outputs.append(out.squeeze(0).permute(1, 0, 2))
        return torch.cat(outputs, dim=0)


class FakePositionManager:
    """Absolute positions from per-request counters, HF-exact Llama-3 RoPE."""

    def __init__(self, config: PositionConfig | None = None):
        # the declared RoPE, for a handle's call, which passes no field of its own
        self.config = config
        self._counters: list[int] = []
        self._pos_ids: torch.Tensor | None = None
        self._label = "main"

    def plan(self, seq_lens: list[int]) -> torch.Tensor:
        """Positions of this step's packed rows; advances the counters."""
        if len(self._counters) != len(seq_lens):
            self._counters = [0] * len(seq_lens)
        ids = []
        for i, n in enumerate(seq_lens):
            start = self._counters[i]
            ids.extend(range(start, start + n))
            self._counters[i] = start + n
        self._pos_ids = torch.tensor(ids, dtype=torch.long)
        return self._pos_ids

    def reset(self) -> None:
        self._counters = []
        self._pos_ids = None

    def pos_ids(self, label: str = "main") -> torch.Tensor:
        del label
        assert self._pos_ids is not None, "plan the step first"
        return self._pos_ids

    def apply_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        label: str,
        rope_theta: float = 10_000.0,
        rope_scale: float = 1.0,
        low_freq_factor: float | None = None,
        high_freq_factor: float | None = None,
        old_context_len: float | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del label, kwargs
        if self.config is not None:
            rope_theta, rope_scale = self.config.rope_theta, self.config.rope_scale
            low_freq_factor, high_freq_factor = self.config.low_freq_factor, self.config.high_freq_factor
            old_context_len = self.config.old_context_len
        head_dim = q.shape[-1]
        pos = self._pos_ids[: q.shape[0]].to(torch.float32)
        if None not in (low_freq_factor, high_freq_factor, old_context_len):
            inv_freq = _llama3_inv_freq(
                head_dim, rope_theta, rope_scale,
                low_freq_factor, high_freq_factor, old_context_len,
            )
        else:
            inv_freq = 1.0 / (
                rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            )
            pos = pos / rope_scale
        freqs = torch.outer(pos, inv_freq)             # [N, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)        # [N, D]
        cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]
        q_out = (q.float() * cos + _rotate_half(q.float()) * sin).to(q.dtype)
        k_out = (k.float() * cos + _rotate_half(k.float()) * sin).to(k.dtype)
        return q_out, k_out


class FakeT3Resources:
    """The three fakes a T3 layer stack binds, with one ``plan`` for a step."""

    def __init__(self):
        self.kv_attn = FakeKVAttention()
        self.pos = FakePositionManager()

    def bind(self, module: torch.nn.Module, attn_key: str, kv_key: str, pos_key: str) -> None:
        resources = {attn_key: self.kv_attn, kv_key: self.kv_attn, pos_key: self.pos}
        for sub in module.modules():
            bind = getattr(sub, "bind_resources", None)
            if bind is not None and sub is not module:
                bind(resources)

    def plan(self, seq_lens: list[int]) -> torch.Tensor:
        self.kv_attn.plan(seq_lens)
        return self.pos.plan(seq_lens)

    def reset(self) -> None:
        self.kv_attn.reset()
        self.pos.reset()


def _declaring(config: ChatterboxConfig) -> ChatterboxModel:
    model = object.__new__(ChatterboxModel)
    model.config = config
    return model


class FakeT3State(FakeT3Resources):
    """The T3 node's state handles, as the engine builds them from the
    declarations of ``config``, over the fakes and ``sampler``."""

    def __init__(self, config: ChatterboxConfig, sampler=None):
        super().__init__()
        model = _declaring(config)
        self.pos = FakePositionManager(next(
            spec for spec in node_resources(model) if spec.resource_key == T3_POS
        ).config)
        resources = {T3_KV: self.kv_attn, T3_ATTN: self.kv_attn, T3_POS: self.pos, T3_SAMPLER: sampler}
        self.handles = node_handles(model, T3_NODE, resources)

    def bind_handles(self, module: torch.nn.Module) -> None:
        """Hand a T3 stack run without its submodule the handles its layers call."""
        for sub in module.modules():
            bind = getattr(sub, "bind_state", None)
            if bind is not None:
                bind(self.handles)

    def stack_step(self, seq_lens: list[int]) -> SubmoduleStep:
        """A step of one ``main`` segment per sequence of ``seq_lens``, for a stack run without its submodule."""
        step = SubmoduleStep(
            segments=[Segment(request_id=i, label=COND_LABEL, span=n) for i, n in enumerate(seq_lens)],
            state={T3_KV: ArrayStep()},
        )
        return state_steps(step, self.handles, graph_walk="prefill")

    @contextmanager
    def opened(self, step: SubmoduleStep, graph_walk: str) -> Iterator[torch.Tensor]:
        """``step``, as `engine_step` returns it, planned on the fakes and open
        for the calls of its forward, as the engine runs it; yields its
        position ids."""
        rows = list(dict.fromkeys(segment.request_id for segment in step.segments))
        step.set_ctx(StepContext(request_ids=rows, graph_walk=graph_walk, slot=None, capture=False))
        pos_ids = self.plan([segment.span for segment in step.segments])
        step.begin_calls()
        yield pos_ids
        step.check_calls()
        step.commit_calls()
        step.end_calls()


class _FakeStreamPool:
    """S3Gen's stream Values, an entry per request, read and written for the rows of the step."""

    def __init__(self, entry):
        self.entry = entry
        self.slots = {}
        self.rows = ()

    def block(self, field, layer):
        return torch.empty(1, *self.entry[field].shape, dtype=self.entry[field].dtype)

    def gather(self, field, layer, *, label, start=None):
        fresh = torch.zeros(self.entry[field].shape, dtype=self.entry[field].dtype)
        return torch.stack([self.slots.get(rid, {}).get(field, fresh) for rid in self.rows])

    def scatter_(self, field, value, layer, *, label):
        for rid, row in zip(self.rows, value, strict=True):
            self.slots.setdefault(rid, {})[field] = row.clone()


class FakeS3GenStream:
    """S3Gen's stream Value and its host fields, as the engine keeps them
    between a stream's chunks, bound to the node ``sub``."""

    def __init__(self, sub):
        model = _declaring(sub.config)
        stream = next(array for array in model.declare_state() if array.name == S3GEN_STREAM)
        self.pool = _FakeStreamPool(stream.entry.tensors)
        self.fields = HostFields()
        self.sub = sub
        sub.bind_node_state(node_handles(model, S3GEN_NODE, {S3GEN_STREAM: self.pool}, host_fields=self.fields))

    def held(self, rid) -> dict:
        return dict(self.fields.fields(rid, S3GEN_STREAM, "main", 0))

    def prepare(self, rid, tokens: torch.Tensor, is_final: bool, seed: int, n_timesteps: int) -> NodeInputs:
        """A chunk whose window is ``tokens``, every token the stream delivered so far."""
        if not self.fields.is_open(rid):
            self.fields.open(rid)
        info = SimpleNamespace(
            request_id=str(rid), rid_handle=rid, random_seed=seed, step_metadata={"n_cfm_timesteps": n_timesteps},
        )
        return self.sub.prepare_inputs("s3gen_chunk", info, {"speech_tokens": [tokens]}, is_final_stream_chunk=is_final)

    def step(self, prepared: dict) -> dict:
        return s3gen_step(self.sub, prepared)


def s3gen_step(sub, prepared: dict) -> dict:
    """One step of the S3Gen node ``sub``, its stream bound to a `_FakeStreamPool`, over ``prepared``, by rid handle,
    as the engine drives it: declared, the Value read and written, the host fields committed. Each row's PCM."""
    rids, inputs = list(prepared), list(prepared.values())
    step = engine_step(sub, graph_walk="s3gen_chunk", request_ids=rids, inputs=inputs)
    step.set_ctx(StepContext(request_ids=rids, graph_walk="s3gen_chunk", slot=None, capture=False))
    sub.stream._pool.rows = tuple(rids)
    step.begin_calls()
    packed = sub.preprocess("s3gen_chunk", None, inputs)
    if len(rids) > 1:
        out = sub.forward_batched("s3gen_chunk", None, **packed)
    else:
        out = {rids[0]: sub.forward("s3gen_chunk", None, **packed)}
    step.check_calls()
    step.commit_calls()
    step.end_calls()
    return {rid: out[rid]["audio_chunk"][0] for rid in rids}
