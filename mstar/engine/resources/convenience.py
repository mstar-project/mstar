import torch

from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.linear_attn.base import LinearAttnManager
from mstar.engine.resources.recurrent.pool import RecurrentStatePool


class AttentionCallable:
    """A convenience wrapper around kv and attn that wraps the KV write and
    attention call in one pure-tensor function, with helper methods for
    setting layer and label information.

    Must be used for  ``ulysses_attention``, which expects such a pure tensor
    callable. Recommended to use instance per transformer, *not* one per layer,
    as Dynamo specializes ``ulysses_attention`` on the identity of its
    `run_attention` argument, so a per-layer callable retraces that frame once
    per layer and blows the recompile limit.

    That sharing is why the label is one cursor for the whole stack. No model
    varies its label per layer today; one that needs to should thread the label
    explicitly rather than use this.
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
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        if self.attn.requires_kv_write:
            self.kv.write_kv(k, v)
        return self.attn.run(q, kv_cache_layer=self.kv.layer_view(), k=k, v=v)


class LinearAttnCallable:
    """A convenience wrapper around a recurrent state pool and the resource
    running kernels against it, mirroring :class:`AttentionCallable`.

    The layer body calls ``conv`` and then the instance itself; this reads the
    pool's per-layer blocks and hands them over as plain tensors, so neither
    the layer nor the manager holds the pool.

    The layer cursor lives here as well as on the resource: the resource uses
    it for its own bookkeeping, and this needs it to pick the block.
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
        """The layer's index among the *recurrent* layers, not the stack's.

        A hybrid model interleaves these with full-attention layers, and the
        pool is sized by its own count; see the model's ``get_node_resources``.
        """
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
