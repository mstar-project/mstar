import torch

from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.kv.manager import KVManager


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
