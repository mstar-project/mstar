"""Native Qwen-Image dual-stream MMDiT (port of diffusers 0.39 ``QwenImageTransformer2DModel``).

One denoise step over a row's tokens, ``[txt (Lt) | img (target + references)]``:

    timestep -> sinusoidal(256, scale 1000) -> MLP -> temb     (bf16, like the reference)
    img tokens -> img_in (64 -> D);  txt tokens -> RMSNorm -> txt_in (3584 -> D)
    N dual-stream blocks: per-stream AdaLN modulation, q/k/v and MLP; one joint attention
    drop txt tokens -> AdaLN (norm_out) -> proj_out (D -> 64)

``zero_cond_t`` (Edit-2511): the reference images' tokens are modulated as at timestep 0,
the target's at the step's timestep (the layout's ``target_tokens`` splits them).

Each row's text is left-padded to the step's ``Lt``, so the joint sequence of a row is
``[pad | txt | img]`` and attends within the segments the step declares: the pad on its own,
and the real ``[txt | img]`` as the reference attends it. RoPE follows ``QwenEmbedRope``
(``scale_rope``): the target at frame 0 and reference ``i`` at frame ``i``, heights and widths
centred, text from ``max(h/2, w/2)`` over every image. The rotation is the reference's
complex product itself: its rounding is the device's complex multiply (fused multiply-adds on
CUDA), which no one real-arithmetic formula reproduces on every device.

On CUDA the elementwise work between the GEMMs runs as fused kernels (``kernels``) that round where
the eager ops do: the q/k norms with the rotation, the LayerNorms with the modulation after them (the
attention's gated residual too) and the gated residuals. Those are bit-identical to the eager ops (the CPU path),
in a fraction of the memory traffic. The one fusion that is not is the MLP's bias and tanh-GELU, in the
up-projection's GEMM epilogue (``ffn.addmm_gelu_tanh``, as SGLang runs it): it rounds once from the fp32
accumulator where eager rounds the GEMM and then the GELU, so it is held to the fp32 yardstick, not to bits.

Linears are tensor-parallel (``comm_group``): q/k/v and the MLP's first linear split by
column, the attention outputs and the MLP's second by row; the modulation linears stay whole.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.model.components.diffusion.ffn import addmm_gelu_tanh
from mstar.model.components.diffusion.qwen import dit_kernels as kernels
from mstar.model.components.distributed.linear import ColumnParallelLinear, RowParallelLinear
from mstar.model.components.lora.capable import LoraBound
from mstar.model.components.norm import QwenRMSNorm

# The rotary tables' extent in the reference (``QwenEmbedRope``'s 4096 positions per sign).
ROPE_POSITIONS = 4096


@dataclass(frozen=True)
class QwenImageTransformerConfig:
    """``transformer/config.json``."""

    in_channels: int = 64
    out_channels: int = 16
    num_layers: int = 60
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 3584
    axes_dims_rope: tuple[int, int, int] = (16, 56, 56)
    patch_size: int = 2
    zero_cond_t: bool = False

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @classmethod
    def from_dict(cls, cfg: dict) -> "QwenImageTransformerConfig":
        for unsupported in ("guidance_embeds", "use_additional_t_cond", "use_layer3d_rope"):
            if cfg.get(unsupported):
                raise ValueError(f"Qwen-Image transformer {unsupported}=True is not supported")
        return cls(
            in_channels=int(cfg["in_channels"]),
            out_channels=int(cfg.get("out_channels") or cfg["in_channels"]),
            num_layers=int(cfg["num_layers"]),
            attention_head_dim=int(cfg["attention_head_dim"]),
            num_attention_heads=int(cfg["num_attention_heads"]),
            joint_attention_dim=int(cfg["joint_attention_dim"]),
            axes_dims_rope=tuple(int(d) for d in cfg.get("axes_dims_rope", (16, 56, 56))),
            patch_size=int(cfg.get("patch_size", 2)),
            zero_cond_t=bool(cfg.get("zero_cond_t", False)),
        )


class QwenStepLayout(NamedTuple):
    """Per-step derived tables (``build_layout``): complex64 rotary tables and where the target's tokens end."""

    img_freqs: torch.Tensor        # [L_img, head_dim / 2]
    txt_freqs: torch.Tensor        # [branches, Lt, head_dim / 2], one row per local branch
    target_tokens: int             # the image tokens before the references'; zero_cond_t modulates the rest at t=0


def _rope_axis(index: torch.Tensor, dim: int, theta: float) -> torch.Tensor:
    """``QwenEmbedRope.rope_params``: complex64 ``[len(index), dim / 2]``."""
    freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
    return torch.polar(torch.ones_like(freqs), freqs)


@functools.lru_cache(maxsize=4)
def _rope_axes(axes_dims: tuple[int, ...], theta: float) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Each axis's rotations at positions ``0..ROPE_POSITIONS-1`` and at the negative ones, and the positive axes
    side by side (the text's table): constants, built once."""
    pos_index = torch.arange(ROPE_POSITIONS)
    neg_index = torch.arange(ROPE_POSITIONS).flip(0) * -1 - 1
    pos = [_rope_axis(pos_index, d, theta) for d in axes_dims]
    neg = [_rope_axis(neg_index, d, theta) for d in axes_dims]
    return pos, neg, torch.cat(pos, dim=1)


@functools.lru_cache(maxsize=16)
def _image_rope(axes_dims: tuple[int, ...], img_grids: tuple[tuple[int, int], ...], theta: float) -> torch.Tensor:
    """The image tokens' table, a function of the grids alone. Cached: building it is ~10^6-element host ops
    that run on torch's intra-op pool, and two ranks of one host building at once (cfg-parallel, each rank's
    first step of a request) contend for the cores: ~4 ms alone, ~300-400 ms together."""
    pos, neg, _ = _rope_axes(axes_dims, theta)
    img = []
    for frame, (h, w) in enumerate(img_grids):
        t_freqs = pos[0][frame:frame + 1].view(1, 1, 1, -1).expand(1, h, w, -1)
        h_freqs = torch.cat([neg[1][-(h - h // 2):], pos[1][:h // 2]]).view(1, h, 1, -1).expand(1, h, w, -1)
        w_freqs = torch.cat([neg[2][-(w - w // 2):], pos[2][:w // 2]]).view(1, 1, w, -1).expand(1, h, w, -1)
        img.append(torch.cat([t_freqs, h_freqs, w_freqs], dim=-1).reshape(h * w, -1))
    return torch.cat(img).contiguous()


def qwen_rope_tables(
    axes_dims: Sequence[int], img_grids: Sequence[tuple[int, int]], text_lengths: Sequence[int], text_rows: int,
    theta: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(img_freqs, txt_freqs)`` on the host, complex64, as ``QwenEmbedRope`` (``scale_rope``)
    builds them: ``img_grids`` are the target's then each reference's ``(h, w)`` in tokens, and
    ``text_lengths`` the real text of each local branch, left-padded to ``text_rows``. ``img_freqs`` is
    shared between calls: read it, don't write it."""
    axes_dims, img_grids = tuple(axes_dims), tuple(tuple(g) for g in img_grids)
    img_freqs = _image_rope(axes_dims, img_grids, theta)
    start = max(max(h // 2, w // 2) for h, w in img_grids)
    all_axes = _rope_axes(axes_dims, theta)[2]
    txt = []
    for length in text_lengths:
        # a pad row reads the first real position; it attends only itself
        index = (torch.arange(text_rows) - (text_rows - length)).clamp(min=0) + start
        txt.append(all_axes[index])
    txt_freqs = torch.stack(txt)
    return img_freqs, txt_freqs.contiguous()


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """``apply_rotary_emb_qwen(use_real=False)``: ``x [R, S, H, D]`` as ``D / 2`` complex pairs times ``freqs``
    (``[S, D / 2]`` or ``[R, S, D / 2]``), in complex64, back to ``x``'s dtype."""
    pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(pairs * freqs.unsqueeze(-2)).flatten(3).type_as(x)


def timestep_embedding(timesteps: torch.Tensor, dim: int = 256, scale: float = 1000.0) -> torch.Tensor:
    """diffusers ``Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)``: fp32 ``[N, dim]``
    laid out ``[cos | sin]``."""
    half = dim // 2
    exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    emb = scale * (timesteps[:, None].float() * torch.exp(exponent)[None, :])
    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


class _TimestepEmbedder(nn.Module):
    """diffusers ``TimestepEmbedding``: ``linear_2(silu(linear_1(x)))``."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(256, dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(x)))


class _TimeTextEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.timestep_embedder = _TimestepEmbedder(dim)

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self.timestep_embedder(timestep_embedding(timestep).to(dtype))


class _GeluProj(nn.Module):
    """diffusers ``GELU(approximate="tanh")``: a column-parallel projection and the activation, on CUDA in one
    GEMM (the bias and the GELU in its epilogue). The column shard is a local GEMM with its bias on every rank."""

    def __init__(self, comm_group: CommGroup, dim: int, inner: int):
        super().__init__()
        self.proj = ColumnParallelLinear(comm_group, dim, inner, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # a LoRA-bound projection adds its delta in its forward, before the GELU
        if x.is_cuda and not isinstance(self.proj, LoraBound):
            return addmm_gelu_tanh(x, self.proj.weight, self.proj.bias)
        return F.gelu(self.proj(x), approximate="tanh")


class _FeedForward(nn.Module):
    """diffusers ``FeedForward(dim, dim, activation_fn="gelu-approximate")``, ``net.0`` / ``net.2`` by name."""

    def __init__(self, comm_group: CommGroup, dim: int):
        super().__init__()
        self.net = nn.ModuleList([_GeluProj(comm_group, dim, 4 * dim), nn.Identity(),
                                  RowParallelLinear(comm_group, 4 * dim, dim, bias=True)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net[2](self.net[0](x))


class QwenImageJointAttention(nn.Module):
    """``QwenDoubleStreamAttnProcessor2_0``: per-stream q/k/v (column-parallel) and per-head RMSNorm
    of q and k, RoPE, one attention over each row's ``[txt | img]`` segments, per-stream outputs."""

    def __init__(self, comm_group: CommGroup, dim: int, num_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.head_dim = head_dim
        self.local_heads = divide(num_heads, comm_group.world_size)
        inner = num_heads * head_dim
        self.to_q, self.to_k, self.to_v = (ColumnParallelLinear(comm_group, dim, inner, bias=True) for _ in range(3))
        self.add_q_proj, self.add_k_proj, self.add_v_proj = (
            ColumnParallelLinear(comm_group, dim, inner, bias=True) for _ in range(3)
        )
        self.norm_q, self.norm_k = QwenRMSNorm(head_dim, eps), QwenRMSNorm(head_dim, eps)
        self.norm_added_q, self.norm_added_k = QwenRMSNorm(head_dim, eps), QwenRMSNorm(head_dim, eps)
        self.to_out = nn.ModuleList([RowParallelLinear(comm_group, inner, dim, bias=True)])
        self.to_add_out = RowParallelLinear(comm_group, inner, dim, bias=True)

    def forward(
        self, img: torch.Tensor, txt: torch.Tensor, layout: QwenStepLayout, txt_freqs: torch.Tensor,
        attend: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def heads(x: torch.Tensor) -> torch.Tensor:
            return x.unflatten(-1, (-1, self.head_dim))

        q, k = self._norm_rope(
            (heads(self.add_q_proj(txt)), heads(self.to_q(img)), self.norm_added_q, self.norm_q),
            (heads(self.add_k_proj(txt)), heads(self.to_k(img)), self.norm_added_k, self.norm_k),
            txt_freqs, layout.img_freqs,
        )
        v = heads(_joint_linear(self.add_v_proj, txt, self.to_v, img))
        out = attend(q, k, v).flatten(2, 3).to(q.dtype)
        num_txt = txt.shape[1]
        return _rows_linear(self.to_out[0], out[:, num_txt:]), _rows_linear(self.to_add_out, out[:, :num_txt])

    @staticmethod
    def _norm_rope(q_parts, k_parts, txt_freqs, img_freqs) -> tuple[torch.Tensor, torch.Tensor]:
        """q and k, each ``[txt | img]`` along the tokens with every part per-head normed and rotated: one fused
        pass per part for both on CUDA. Each ``*_parts`` is ``(txt, img, txt_norm, img_norm)``."""
        (q_txt, q_img, q_txt_norm, q_img_norm), (k_txt, k_img, k_txt_norm, k_img_norm) = q_parts, k_parts
        norms = (q_txt_norm, q_img_norm, k_txt_norm, k_img_norm)
        if (kernels.fits(q_txt, q_img, k_txt, k_img) and q_txt.shape[-1] == 128 and len({n.eps for n in norms}) == 1
                and q_txt.stride() == k_txt.stride() and q_img.stride() == k_img.stride()):
            return kernels.joint_qk_pair_norm_rope(
                q_txt, q_img, k_txt, k_img, *(n.weight for n in norms), txt_freqs, img_freqs, q_txt_norm.eps,
            )
        return tuple(
            torch.cat([apply_rope(tn(t), txt_freqs), apply_rope(im(i), img_freqs)], dim=1)
            for t, i, tn, im in (q_parts, k_parts)
        )


def _plain(linear: nn.Module, cls: type) -> bool:
    """A plain tensor-parallel linear (not a LoRA twin) whose forward is its GEMM with the bias."""
    return type(linear) is cls and linear.bias is not None and not linear.skip_bias_add


def _joint_linear(txt_linear: ColumnParallelLinear, txt: torch.Tensor, img_linear: ColumnParallelLinear,
                  img: torch.Tensor) -> torch.Tensor:
    """``cat([txt_linear(txt), img_linear(img)], dim=1)``, each row's GEMMs writing straight into its slice of the
    joint ``[R, Lt + Li, N]`` (a column shard is a local GEMM on every rank). Bit-exact with the cat at the served shape
    on H200 (a different GEMM shape may round differently elsewhere: the CUDA test holds it to bf16 rounding)."""
    if not (txt.is_cuda and _plain(txt_linear, ColumnParallelLinear) and _plain(img_linear, ColumnParallelLinear)
            and not txt_linear._local_output_start and not img_linear._local_output_start
            and not (txt_linear.gather_output and txt_linear.tp_size > 1)
            and not (img_linear.gather_output and img_linear.tp_size > 1)):
        return torch.cat([txt_linear(txt), img_linear(img)], dim=1)
    lt = txt.shape[1]
    out = img.new_empty(img.shape[0], lt + img.shape[1], img_linear.weight.shape[0])
    for r in range(img.shape[0]):
        torch.addmm(txt_linear.bias, txt[r], txt_linear.weight.t(), out=out[r, :lt])
        torch.addmm(img_linear.bias, img[r], img_linear.weight.t(), out=out[r, lt:])
    return out


def _rows_linear(linear: RowParallelLinear, x: torch.Tensor) -> torch.Tensor:
    """``linear(x)`` for a token slice ``x [R, L, K]`` of a larger tensor, one GEMM per row from its contiguous
    rows instead of a copy of the slice (one GPU; under TP the row shard's all-reduce runs as usual). Bit-exact with
    the copy at the served shape on H200 (see :func:`_joint_linear`)."""
    if not (x.is_cuda and _plain(linear, RowParallelLinear) and linear.tp_size == 1 and x[0].is_contiguous()):
        return linear(x.contiguous())
    out = x.new_empty(*x.shape[:2], linear.weight.shape[0])
    for r in range(x.shape[0]):
        torch.addmm(linear.bias, x[r], linear.weight.t(), out=out[r])
    return out


def _by_token(fn, split: int, tokens: tuple[torch.Tensor, ...], params: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """``fn(*tokens, *params)`` with ``params [sets * R, D]`` broadcast over the tokens: one set for every token, or
    (``zero_cond_t``) set 0 for the tokens before ``split`` and set 1 from it."""
    rows = tokens[0].shape[0]
    if params[0].shape[0] == rows:
        return fn(*tokens, *(p[:, None] for p in params))
    return torch.cat([fn(*(t[:, :split] for t in tokens), *(p[:rows, None] for p in params)),
                      fn(*(t[:, split:] for t in tokens), *(p[rows:, None] for p in params))], dim=1)


class _Mod(NamedTuple):
    """One modulation's parameters, each a contiguous ``[sets * B, D]`` (``zero_cond_t``: the step's timestep rows,
    then timestep 0's, the second set from token ``split``)."""

    shift: torch.Tensor
    scale1p: torch.Tensor
    gate: torch.Tensor


def _block_mods(linear: nn.Module, act: torch.Tensor) -> tuple[_Mod, _Mod]:
    """A block's two modulations from ``linear(act) [sets * B, 6D]``: one copy lays the six ``[sets * B, D]`` parts
    out contiguously (instead of one per part) and one add makes both ``1 + scale``."""
    params = linear(act)
    rows, dim = params.shape[0], params.shape[1] // 6
    parts = params.view(rows, 6, dim).transpose(0, 1).contiguous()
    scale1p = 1 + parts[1::3]
    return _Mod(parts[0], scale1p[0], parts[2]), _Mod(parts[3], scale1p[1], parts[5])


def _modulate(x: torch.Tensor, mod: _Mod, split: int):
    """``(x (1 + scale) + shift, gate)``; ``gate`` stays ``[sets * B, D]`` for :func:`_gated_add`."""
    shift, scale1p, gate = mod
    if kernels.fits(x, shift) and x.is_contiguous():
        return kernels.modulate(x, scale1p, shift, split), gate
    return _by_token(lambda t, s1p, sh: t * s1p + sh, split, (x,), (scale1p, shift)), gate


def _gated_add(x: torch.Tensor, gate: torch.Tensor, y: torch.Tensor, split: int) -> torch.Tensor:
    """``x + gate * y``, ``gate`` by token as in :func:`_modulate`."""
    if kernels.fits(x, gate, y) and x.is_contiguous() and y.is_contiguous():
        return kernels.gated_add(x, gate, y, split)
    return _by_token(lambda t, u, g: t + g * u, split, (x, y), (gate,))


def _norm_modulate(norm: nn.LayerNorm, x: torch.Tensor, mod: _Mod, split: int):
    """:func:`_modulate` of ``norm(x)``, in one pass where the kernel takes ``x``."""
    if kernels.fits(x, mod.shift) and kernels.norm_fits(x):
        return kernels.norm_modulate(x, mod.scale1p, mod.shift, split, norm.eps), mod.gate
    return _modulate(norm(x), mod, split)


def _gated_norm_modulate(
    x: torch.Tensor, gate: torch.Tensor, y: torch.Tensor, norm: nn.LayerNorm, mod: _Mod, split: int,
):
    """``r = _gated_add(x, gate, y)`` and ``_modulate(norm(r), mod)``: ``(r, modulated, next gate)``, from one
    read of ``x`` and ``y`` where the kernel takes them."""
    if kernels.fits(x, gate, y, mod.shift) and kernels.norm_fits(x) and y.is_contiguous():
        residual, out = kernels.gated_norm_modulate(x, gate, y, mod.scale1p, mod.shift, split, norm.eps)
        return residual, out, mod.gate
    residual = _gated_add(x, gate, y, split)
    return (residual, *_modulate(norm(residual), mod, split))


class QwenImageBlock(nn.Module):
    """``QwenImageTransformerBlock``."""

    def __init__(self, comm_group: CommGroup, config: QwenImageTransformerConfig, eps: float = 1e-6):
        super().__init__()
        dim = config.inner_dim
        self.zero_cond_t = config.zero_cond_t
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = QwenImageJointAttention(comm_group, dim, config.num_attention_heads, config.attention_head_dim, eps)
        self.img_mlp = _FeedForward(comm_group, dim)
        self.txt_mlp = _FeedForward(comm_group, dim)

    def forward(
        self, img: torch.Tensor, txt: torch.Tensor, temb_act: torch.Tensor, layout: QwenStepLayout,
        txt_freqs: torch.Tensor, attend,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``temb_act``: ``silu(temb)``, the modulations' input, the same for every block (``img_mod`` /
        ``txt_mod`` are ``SiLU`` then the linear)."""
        # zero_cond_t: the references' tokens (from target_tokens on) take timestep 0's parameters
        split = layout.target_tokens if self.zero_cond_t else img.shape[1]
        img_mod1, img_mod2 = _block_mods(self.img_mod[1], temb_act)
        txt_act = temb_act.chunk(2, dim=0)[0] if self.zero_cond_t else temb_act
        txt_mod1, txt_mod2 = _block_mods(self.txt_mod[1], txt_act)
        txt_split = txt.shape[1]

        img_in, img_gate1 = _norm_modulate(self.img_norm1, img, img_mod1, split)
        txt_in, txt_gate1 = _norm_modulate(self.txt_norm1, txt, txt_mod1, txt_split)
        img_attn, txt_attn = self.attn(img_in, txt_in, layout, txt_freqs, attend)

        img, img_in, img_gate2 = _gated_norm_modulate(img, img_gate1, img_attn, self.img_norm2, img_mod2, split)
        img = _gated_add(img, img_gate2, self.img_mlp(img_in), split)
        txt, txt_in, txt_gate2 = _gated_norm_modulate(txt, txt_gate1, txt_attn, self.txt_norm2, txt_mod2, txt_split)
        txt = _gated_add(txt, txt_gate2, self.txt_mlp(txt_in), txt_split)
        return img, txt


class _AdaNormOut(nn.Module):
    """diffusers ``AdaLayerNormContinuous(elementwise_affine=False, eps=1e-6)``: scale first, then shift."""

    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, 2 * dim)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(self.silu(temb).to(x.dtype)).chunk(2, dim=1)
        if kernels.fits(x, shift) and kernels.norm_fits(x):
            return kernels.norm_modulate(x, (1 + scale).contiguous(), shift.contiguous(), x.shape[1], self.norm.eps)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


class QwenImageTransformer(nn.Module):
    """``QwenImageTransformer2DModel``; parameter names follow the checkpoint's."""

    def __init__(self, config: QwenImageTransformerConfig, comm_group: CommGroup | None = None):
        super().__init__()
        comm_group = comm_group or CommGroup.trivial()
        self.config = config
        dim = config.inner_dim
        self.time_text_embed = _TimeTextEmbed(dim)
        self.txt_norm = QwenRMSNorm(config.joint_attention_dim, 1e-6)
        self.img_in = nn.Linear(config.in_channels, dim)
        self.txt_in = nn.Linear(config.joint_attention_dim, dim)
        self.transformer_blocks = nn.ModuleList(QwenImageBlock(comm_group, config) for _ in range(config.num_layers))
        self.norm_out = _AdaNormOut(dim)
        self.proj_out = nn.Linear(dim, config.patch_size**2 * config.out_channels)
        # set by a caller that compiles the blocks: their text axis is then dynamic
        self.dynamic_text = False

    @property
    def dtype(self) -> torch.dtype:
        return self.img_in.weight.dtype

    def forward(
        self, img: torch.Tensor, txt: torch.Tensor, timestep: torch.Tensor, layout: QwenStepLayout, attend,
    ) -> torch.Tensor:
        """``img [R, L_img, C_in]`` (each row's target then its references), ``txt [R, Lt, C_txt]`` (left-padded),
        ``timestep [R]`` in the transformer's dtype (the reference's ``t / 1000``) -> ``[R, L_img, C_out * 4]``.
        Rows are each request's local branches in turn, the layout's text tables one per branch."""
        rows = img.shape[0]
        img = self.img_in(img)
        if self.config.zero_cond_t:
            timestep = torch.cat([timestep, timestep * 0], dim=0)
        txt = self.txt_in(self.txt_norm(txt))
        temb = self.time_text_embed(timestep, img.dtype)
        txt_freqs = layout.txt_freqs.repeat(rows // layout.txt_freqs.shape[0], 1, 1)
        if self.dynamic_text:
            # compiled blocks: one graph for any prompt length, the image dims static
            for tensor in (txt, txt_freqs):
                torch._dynamo.mark_dynamic(tensor, 1)
        temb_act = F.silu(temb)
        for block in self.transformer_blocks:
            img, txt = block(img, txt, temb_act, layout, txt_freqs, attend)
        if self.config.zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]
        return self.proj_out(self.norm_out(img, temb))
