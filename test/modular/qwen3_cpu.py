"""CPU stand-ins shared by the modular tests: CPU versions of kernels main lacks
(a reference paged attention over the KV pool's pages, the RoPE op, a CPU token
draw, and a transfer stub), and a dense reference's attention and positions to
compare a paged forward against.
"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F

from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import AttentionSpec, AttentionStep
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.sampler import utils as sampler_utils
from mstar.engine.resources.step import StepContext

PAGE = 4


# ── stand-ins for the CUDA-only kernels ──


class CPUPagedAttention(AttentionManager):
    """Reference paged attention, read straight off the KV pool's pages."""

    # it holds no request's state, so a rebuild policy (`Engine._check_rebuild_policies`) takes it
    supports_drop = True

    def __init__(self, kv_key: str):
        self._kv_key = kv_key
        self._views: dict[str, list] = {}
        self._step: AttentionStep | None = None

    @classmethod
    def build(cls, spec, info):
        return cls(spec.config.kv_cache)

    def depends_on(self):
        return {self._kv_key}

    def plan(self, step: AttentionStep, ctx: StepContext):
        self.reset_default_cursors()
        self._step = step
        self._views = {label: list(out.views) for label, out in ctx.plan_results[self._kv_key].items()}

    def run(self, q, label=None, kv_cache_layer=None, k=None, v=None, layer_idx=None):
        del k, v, layer_idx
        step = self._step
        out, row = [], 0
        for view in self._views[label or self._default_label]:
            n = view.to_compute
            keys = view.length - step.hide_last
            pos = torch.arange(keys)
            pages = torch.tensor(view.page_idxs)[pos // PAGE]
            kv = kv_cache_layer[pages, :, pos % PAGE]  # [keys, 2, heads, dim]
            kk, vv = kv[:, 0], kv[:, 1]
            rep = q.shape[1] // kk.shape[1]
            kk, vv = kk.repeat_interleave(rep, 1), vv.repeat_interleave(rep, 1)
            qq = q[row: row + n]
            mask = None
            if step.causal:
                at = torch.arange(view.length - n, view.length)
                mask = pos[None, :] <= at[:, None]
            o = F.scaled_dot_product_attention(
                qq.transpose(0, 1), kk.transpose(0, 1), vv.transpose(0, 1), attn_mask=mask,
            )
            out.append(o.transpose(0, 1))
            row += n
        return torch.cat(out)

    def qo_indptr_buf(self, label="main"):
        counts = torch.tensor([v.to_compute for v in self._views[label]])
        return torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])


@dataclasses.dataclass
class CPUAttentionSpec(AttentionSpec):
    @property
    def resource_class(self):
        return CPUPagedAttention


def _rope_cpu(q, k, pos_ids, cos_sin_cache, rotary_dim, interleave, rope_scale, rope_theta,
              low_freq_factor=None, high_freq_factor=None, old_context_len=None):
    assert not interleave and cos_sin_cache is None
    d = rotary_dim or q.shape[-1]
    half = d // 2
    inv = 1.0 / (rope_theta ** (torch.arange(0, half, dtype=torch.float64) * 2 / d))
    angle = pos_ids.to(torch.float64)[:, None] * inv[None]
    cos, sin = angle.cos()[:, None, :], angle.sin()[:, None, :]
    for x in (q, k):
        x1, x2 = x[..., :half].double(), x[..., half:d].double()
        x[..., :d] = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


_ROPE_REGISTERED = False


def register_cpu_rope() -> None:
    global _ROPE_REGISTERED
    if not _ROPE_REGISTERED:
        import mstar.engine.resources.position.rope  # noqa: F401  (defines the op)

        torch.library.register_kernel("mstar::rope_apply_qk_inplace", "cpu", _rope_cpu)
        _ROPE_REGISTERED = True


_device_sample_tokens = sampler_utils.sample_tokens


def _cpu_draw(logits, temperature=0.6, top_k=0, top_p=1.0, repetition_penalty=1.0, seen_token_mask=None,
              any_greedy=None, top_k_zero_count=None, seed=None, rand_offset=None, **kwargs):
    """The prefill's draw on the CPU: argmax at temperature 0, else a draw seeded by
    the row's seed and offset. A device's logits go to the real sampler."""
    if logits.device.type != "cpu":
        return _device_sample_tokens(
            logits, temperature, top_k, top_p, repetition_penalty, seen_token_mask, any_greedy,
            top_k_zero_count, seed, rand_offset, **kwargs,
        )
    n = logits.shape[0]
    temperature = torch.as_tensor(temperature, dtype=torch.float64).expand(n)
    out = logits.argmax(-1)
    for i in range(n):
        if temperature[i] > 0:
            draw_seed = 0 if seed is None else int(seed[i]) * 1_000_003 + int(rand_offset[i])
            g = torch.Generator().manual_seed(draw_seed)
            probs = torch.softmax(logits[i].double() / temperature[i], -1)
            out[i] = torch.multinomial(probs, 1, generator=g)[0]
    return out


class _StubTransfer:
    def __init__(self, *args, **kwargs):
        pass

    def get_kv_transfer_info(self, **kwargs):
        pass

    def remove_request(self, request_id):
        pass

    def cleanup(self):
        pass


def patch_cpu(monkeypatch) -> None:
    """The three stand-ins, and no KV transfer engine."""
    register_cpu_rope()
    monkeypatch.setattr(sampler_utils, "sample_tokens", _cpu_draw)
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def init_random(module: torch.nn.Module, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    for name, p in module.named_parameters():
        if name.endswith("norm.weight"):
            p.data.fill_(1.0)
        else:
            p.data.copy_(torch.randn(p.shape, generator=gen, dtype=torch.float64) / max(p.shape[-1], 1) ** 0.5)


def copy_weights(dst: torch.nn.Module, src: torch.nn.Module) -> None:
    """``src``'s parameters into ``dst``'s, by name, in ``dst``'s device and dtype: a
    float64 reference of a bfloat16 model computes on exactly its weights."""
    have = dict(src.named_parameters())
    for name, p in dst.named_parameters():
        p.data.copy_(have[name].detach().to(device=p.device, dtype=p.dtype))


class _DenseAttention:
    """Causal attention over one call's own K/V: the reference recomputes the whole sequence."""

    requires_kv_write = False
    default_label = "main"

    def set_default_label(self, label):
        pass

    def set_default_layer_idx(self, idx):
        pass

    def layer_view(self):
        return None

    def run(self, q, kv_cache_layer=None, k=None, v=None):
        rep = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        out = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True)
        return out.transpose(0, 1)


class _DensePositions:
    def __init__(self):
        self.n = 0

    def fusable_rope_theta(self, head_dim):
        return None

    def apply_qk(self, q, k, label=None, rope_theta=10_000.0, **kwargs):
        _rope_cpu(q, k, torch.arange(q.shape[0]), None, None, False, 1.0, rope_theta)
        return q, k


