from typing import NamedTuple

import torch

from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.linear_attn.base import LinearAttnManager
from mstar.engine.resources.linear_attn.kda import KDAManager, KDAParams, KDAPlan, SpecBlocks
from mstar.engine.resources.recurrent.pool import RecurrentStatePool


class AttentionCallable:
    """A convenience wrapper around kv and attn that wraps the KV write and
    attention call in one pure-tensor function, with helper methods for
    setting layer and label information. ``write(k, v)`` writes the K/V alone.

    On a node whose steps split over its SP ranks, the call goes through the
    attention resource's exchange (`AttentionManager.exchange`), which hands the
    write and the attention the tokens and heads its pool's storage needs. One
    instance can drive a whole stack, so the label is one cursor for it; a
    model that varies its label per layer threads it explicitly instead.

    A cacheless (ragged) attention needs no callable: call the resource's
    ``run(q, k, v, label=...)`` with the span's label bound.
    """

    def __init__(self, kv: KVManager, attn: AttentionManager | None=None):
        self.kv = kv
        # Can be changed at runtime, e.g., for a model that switches
        self.attn = attn

    @torch.compiler.disable
    def bind_step(self, label: str, attn: AttentionManager | None = None) -> None:
        if attn is not None:
            self.attn = attn
        assert self.attn is not None, (
            "no attention resource: pass `attn` here or at construction"
        )
        self.attn.set_default_label(label)
        self.kv.set_default_label(label)

    @property
    def label(self) -> str:
        """This step's label. Read through to the resource, not stored here, so
        one instance can drive a whole stack of per-layer callables — and so a
        layer that no longer takes a label as an argument can still reach it
        (the position resource carries no cursor, so `apply_qk` is passed this).
        """
        return self.kv.default_label

    @torch.compiler.disable
    def set_layer_idx(self, layer_idx: int) -> None:
        self.attn.set_default_layer_idx(layer_idx)
        self.kv.set_default_layer_idx(layer_idx)

    @torch.compiler.disable
    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self._exchanged(q, k, v)

    @torch.compiler.disable
    def write(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write this step's K/V without attending (a second K over the same tokens, say), on SP
        ranks through the exchange, so it lands on the heads this rank stores. A no-op where the
        attention writes K/V inside its own run (``requires_kv_write`` False)."""
        self._exchanged(None, k, v)

    @property
    def _splits(self) -> bool:
        # a stand-in attention (Cosmos3's warmup's, the tests' fakes) is no AttentionManager; a node
        # that never splits keeps the direct call
        return isinstance(self.attn, AttentionManager) and self.attn.splits

    def _exchanged(self, q: torch.Tensor | None, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor | None:
        if self._splits:
            return self.attn.exchange(self._attend, q, k, v)
        return self._attend(q, k, v)

    def _attend(self, q: torch.Tensor | None, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor | None:
        # q None: the write alone, as the exchanges hand it on
        if self.attn.requires_kv_write:
            self.kv.write_kv(k, v)
        if q is None:
            return None
        return self.attn.run(q, kv_cache_layer=self.kv.layer_view(), k=k, v=v)

    @torch.compiler.disable
    def norm_rope_attend(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_norm: torch.nn.Module, k_norm: torch.nn.Module,
        pos_ids: torch.Tensor, rope_theta: float,
    ) -> torch.Tensor:
        """Per-head q/k RMSNorm (``q_norm``, ``k_norm``), plain RoPE at
        ``pos_ids`` and ``rope_theta``, and attention, on CUDA (a layer off it
        rotates through its position resource instead). For a backend that
        reads K/V from the paged cache, the step's KV write rides in the
        norm/RoPE kernel (``qk_norm_rope_kv_write``) rather than a launch of its
        own; a step that attends the fresh K/V (``reads_step_kv``) gets them
        normed and rotated. On a node whose attention splits, the write and the
        attention go through the exchange (the unfused branch) like ``__call__``'s."""
        from mstar.utils.attention import qk_norm_rope_inplace, qk_norm_rope_kv_write

        eps = _norm_eps(q_norm, k_norm)
        # the fused kernel leaves k as it was, so a step that attends its k and v needs them
        # normed and rotated in place; so does an exchange, which writes and attends the tokens and
        # heads this rank stores
        if self._splits or self.attn.reads_step_kv():
            qk_norm_rope_inplace(q, k, q_norm.weight, k_norm.weight, pos_ids, *eps, rope_theta)
            return self(q, k, v)
        layer, page_idx, slot_idx, n = self.kv.write_target()
        qk_norm_rope_kv_write(
            q, k, v, q_norm.weight, k_norm.weight, pos_ids, layer, page_idx, slot_idx, min(n, q.shape[0]),
            *eps, rope_theta,
        )
        return self.attn.run(q, kv_cache_layer=self.kv.layer_view())

    @torch.compiler.disable
    def norm_rope_table_attend(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_norm: torch.nn.Module, k_norm: torch.nn.Module,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """``norm_rope_attend`` rotating by each token's rotate-half ``cos_sin``
        rows (both halves equal), for a multimodal RoPE whose position grids one
        position can't express. The KV write rides in the kernel
        (``qk_norm_rope_table_kv_write``) on CUDA with a paged backend; anywhere
        else the norm and rotation run unfused."""
        from mstar.utils.attention import qk_norm_rope_table_kv_write

        eps = _norm_eps(q_norm, k_norm)
        if not q.is_cuda or self._splits or self.attn.reads_step_kv():
            cos, sin = (t[: q.shape[0], None] for t in cos_sin)
            q, k = (
                _rotate_half_rope(norm(x.reshape(-1, x.shape[-1])).view(x.shape), cos, sin)
                for x, norm in ((q, q_norm), (k, k_norm))
            )
            return self(q, k, v)
        layer, page_idx, slot_idx, n = self.kv.write_target()
        qk_norm_rope_table_kv_write(
            q, k, v, q_norm.weight, k_norm.weight, *cos_sin, layer, page_idx, slot_idx, min(n, q.shape[0]), *eps,
        )
        return self.attn.run(q, kv_cache_layer=self.kv.layer_view())


def _norm_eps(q_norm: torch.nn.Module, k_norm: torch.nn.Module) -> tuple[float, float]:
    """The q and k norms' epsilons. The fused kernels scale by ``weight``, so a
    Gemma-style norm (scale ``1 + weight``) is refused rather than mis-scaled."""
    if getattr(q_norm, "gemma_mode", False) or getattr(k_norm, "gemma_mode", False):
        raise ValueError("the fused q/k norm kernels scale by weight; a gemma_mode RMSNorm needs its own path")
    return float(q_norm.variance_epsilon), float(k_norm.variance_epsilon)


def _rotate_half_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x * cos + rotate_half(x) * sin``, the unfused rotation by rotate-half tables."""
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin


class LinearAttnCallable:
    """A recurrent state pool plus the resource running kernels against it,
    mirroring :class:`AttentionCallable`.

    The layer calls ``conv`` then the instance; this hands the pool's per-layer
    blocks over as plain tensors, so neither the layer nor the manager holds
    the pool.
    """

    def __init__(self, pool: RecurrentStatePool, attn: LinearAttnManager | None = None):
        self.pool = pool
        self.attn = attn
        self._layer_idx = 0

    @torch.compiler.disable
    def bind_step(self, label: str, attn: LinearAttnManager | None = None) -> None:
        if attn is not None:
            self.attn = attn
        assert self.attn is not None, (
            "no linear attention resource: pass `attn` here or at construction"
        )
        self.attn.set_default_label(label)

    @property
    def label(self) -> str:
        return self.attn.default_label

    @torch.compiler.disable
    def set_layer_idx(self, layer_idx: int) -> None:
        """The layer's index among the *recurrent* layers, not the stack's."""
        self._layer_idx = layer_idx
        self.attn.set_default_layer_idx(layer_idx)

    @torch.compiler.disable
    def conv(
        self, x: torch.Tensor, weight: torch.Tensor,
        bias: torch.Tensor | None = None, activation: str | None = "silu",
    ) -> torch.Tensor:
        return self.attn.run_conv(
            x,
            conv_layer=self.pool.block("conv", self._layer_idx),
            weight=weight,
            bias=bias,
            activation=activation,
        )

    @torch.compiler.disable
    def __call__(
        self,
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        a: torch.Tensor, b: torch.Tensor,
        a_log: torch.Tensor, dt_bias: torch.Tensor,
    ) -> torch.Tensor:
        return self.attn.run(
            q, k, v, a, b,
            state_layer=self.pool.block("state", self._layer_idx),
            a_log=a_log,
            dt_bias=dt_bias,
        )


class _KDALayerBlocks(NamedTuple):
    conv: torch.Tensor
    state: torch.Tensor
    spec: SpecBlocks | None  # the speculative blocks, on a pool that keeps them


class KDACallable:
    """:class:`LinearAttnCallable` for Kimi delta attention. The layer passes its
    projections and ``KDAParams``; this resolves the layer's conv, state and, for a
    verify step, speculative blocks, so neither the layer nor the manager holds the
    pool. The same cursor protocol, counting KDA layers only; a ``layer_idx``
    argument overrides the cursor.
    """

    def __init__(self, pool: RecurrentStatePool, attn: KDAManager | None = None):
        self.pool = pool
        self.attn = attn
        self._layer_idx = 0
        # layer -> its block views: the pool's tensors never move, so once per layer
        self._blocks: dict[int, _KDALayerBlocks] = {}

    @torch.compiler.disable
    def bind_step(self, label: str, attn: KDAManager | None = None) -> None:
        if attn is not None:
            self.attn = attn
        assert self.attn is not None, (
            "no KDA resource: pass `attn` here or at construction"
        )
        self.attn.set_default_label(label)

    @property
    def label(self) -> str:
        return self.attn.default_label

    @property
    def conv_dtype(self) -> torch.dtype:
        """The conv window's dtype in the pool. The kernels read the window beside
        the projected activations, so a layer checks its projections against it."""
        return self.pool.block("conv", 0).dtype

    @torch.compiler.disable
    def set_layer_idx(self, layer_idx: int) -> None:
        self._layer_idx = layer_idx
        self.attn.set_default_layer_idx(layer_idx)

    @torch.compiler.disable
    def current_plan(self) -> KDAPlan:
        """This step's plan under the bound label: whether it decodes or verifies."""
        return self.attn.current_plan()

    @torch.compiler.disable
    def __call__(
        self,
        qkv: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, params: KDAParams,
        gate: torch.Tensor | None = None, layer_idx: int | None = None,
    ) -> torch.Tensor:
        """``KDAManager.run`` on this layer's blocks; see it for the shapes."""
        layer = self._layer_idx if layer_idx is None else layer_idx
        blocks = self._blocks.get(layer)
        if blocks is None:
            blocks = self._blocks[layer] = _KDALayerBlocks(
                self.pool.block("conv", layer), self.pool.block("state", layer),
                SpecBlocks.of(self.pool, layer) if self.attn.speculative_tokens else None,
            )
        return self.attn.run(
            qkv, g, beta, blocks.conv, blocks.state, params, gate=gate, spec=blocks.spec,
        )

    @torch.compiler.disable
    def set_prefix_len(self, accepted: torch.Tensor) -> None:
        """After a verify step, ``accepted [rows]`` each row's accepted drafts: its
        first ``accepted + 1`` tokens (the drafts and the token after them) are the
        prefix the next step replays, from the side this step wrote. Device ops at
        the plan's slot ids, so a captured step records them; padding rows write
        the sink. Every layer shares the count and the side. Once per verify step, which the
        resource checks."""
        plan = self.attn.settle_verify()
        rows = plan.num_rows
        slots = plan.slot_ids[:rows].long()
        count = (accepted[:rows] + 1).to(torch.int32).unsqueeze(1)
        prefix_len, side = self.pool.block("spec_len"), self.pool.block("spec_side")
        prefix_len.index_copy_(0, slots, count)
        side.index_copy_(0, slots, 1 - side.index_select(0, slots))
class Mamba2Callable:
    """``LinearAttnCallable`` for a Mamba-2 layer: the conv, then the SSD
    recurrence, each handed this layer's block of the pool as a plain tensor.

    Same cursor protocol (``bind_step`` once per stack, ``set_layer_idx`` per
    recurrent layer). The layer index counts recurrent layers only; a hybrid
    stack interleaves them with attention layers, and the pool is sized by its
    own count.
    """

    def __init__(self, pool: RecurrentStatePool, attn: LinearAttnManager | None = None):
        self.pool = pool
        self.attn = attn
        self._layer_idx = 0

    @torch.compiler.disable
    def bind_step(self, label: str, attn: LinearAttnManager | None = None) -> None:
        if attn is not None:
            self.attn = attn
        assert self.attn is not None, (
            "no Mamba-2 resource: pass `attn` here or at construction"
        )
        self.attn.set_default_label(label)

    @property
    def label(self) -> str:
        return self.attn.default_label

    @torch.compiler.disable
    def set_layer_idx(self, layer_idx: int) -> None:
        self._layer_idx = layer_idx
        self.attn.set_default_layer_idx(layer_idx)

    @torch.compiler.disable
    def conv(
        self, x: torch.Tensor, weight: torch.Tensor,
        bias: torch.Tensor | None = None, activation: str | None = "silu",
    ) -> torch.Tensor:
        return self.attn.run_conv(
            x,
            conv_layer=self.pool.block("conv", self._layer_idx),
            weight=weight,
            bias=bias,
            activation=activation,
        )

    @torch.compiler.disable
    def __call__(
        self,
        x: torch.Tensor, dt: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
        a_log: torch.Tensor, d: torch.Tensor | None, dt_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.attn.run(
            x, dt, b, c,
            ssm_layer=self.pool.block("ssm", self._layer_idx),
            a_log=a_log, d=d, dt_bias=dt_bias,
        )
