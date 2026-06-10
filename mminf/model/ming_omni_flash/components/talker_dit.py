"""CFM + DiT building blocks for the Ming-flash-omni-2.0 Talker (step 6b).

Ports the modeling primitives from vllm-omni's
``ming_flash_omni/talker_module.py`` (lines 1–402: DiT modules + CFM)
into mminf. Skips the vllm-only CFMGraphExecutor / Pool plumbing —
mminf has its own batching surface.

Upstream module layout (mirror the names so the loader can map
``talker/model.safetensors`` keys 1:1):

  flowmodel.x_embedder, .c_embedder, .t_embedder, .blocks.{N}.norm1,
  .blocks.{N}.attn.to_{q,k,v}, .blocks.{N}.attn.to_out.{0,1}, ..., .final_layer

Two external deps replaced with in-tree minimal ports to keep the
runtime dep surface small:

  * ``DiTTimestepEmbedding`` — SinusPositionEmbedding + 2-layer MLP.
    Mirrors vllm-omni's ``timestep_embedding.DiTTimestepEmbedding``.
  * ``RotaryEmbedding`` — non-xpos 1-D RoPE matching x_transformers'
    ``RotaryEmbedding.forward_from_seq_len`` exactly so the same
    apply pattern works. We port both classes (without the xpos
    branch — the talker config doesn't enable it).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

# ===========================================================================
# Sinusoidal timestep embedding (port of vllm-omni's DiTTimestepEmbedding)
# ===========================================================================


class _SinusPositionEmbedding(nn.Module):
    """Sinusoidal embedding for scalar timesteps (DDPM / DiT convention).

    Mirrors vllm-omni's ``SinusPositionEmbedding`` exactly:
    ``scale * x * exp(-log(10000) * k / (half_dim - 1))`` for
    ``k in [0, half_dim)``, then concat(sin, cos).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"freq_embed_dim must be even, got {dim}")
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: float = 1000.0) -> torch.Tensor:
        device = x.device
        half = self.dim // 2
        # log-spaced inverse frequencies
        emb = math.log(10000.0) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1).float() * emb.unsqueeze(0)
        out = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return out.to(x.dtype)


class DiTTimestepEmbedding(nn.Module):
    """SinusPosEmb → Linear → SiLU → Linear. Output is ``hidden_size``-dim."""

    def __init__(self, dim: int, freq_embed_dim: int = 256) -> None:
        super().__init__()
        self.time_embed = _SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        h = self.time_embed(timestep)
        h = h.to(timestep.dtype)
        return self.time_mlp(h)


# ===========================================================================
# RoPE — non-xpos 1-D variant (port of x_transformers.RotaryEmbedding)
# ===========================================================================
#
# x_transformers uses an INTERLEAVED pair layout: freqs are stacked as
# ``(f, f)`` per dim and then flattened, and ``rotate_half`` permutes
# adjacent pairs as ``(x1, x2) -> (-x2, x1)`` rather than the neox-cat
# split-by-halves convention used by Ling-2.0's thinker.
# We must mirror this layout exactly because the released ckpt's
# weights were trained against it.


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Pair-wise rotation: ``(..., d, 2) -> stack(-x2, x1)`` then flatten."""
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Standard partial-rotary apply with the interleaved pair layout.

    Args:
        t: ``(B, H, T, head_dim)`` queries or keys.
        freqs: ``(1, T, head_dim)`` rotary frequency table.
    """
    rot_dim = freqs.shape[-1]
    seq_len = t.shape[-2]
    freqs = freqs[:, -seq_len:, :]
    # Broadcast (1, T, D) to match (B, H, T, D) along the heads axis.
    if t.ndim == 4 and freqs.ndim == 3:
        freqs = freqs.unsqueeze(1)  # (1, 1, T, D)

    rotated = t[..., :rot_dim]
    passed = t[..., rot_dim:]
    orig_dtype = rotated.dtype
    cos = freqs.cos().to(orig_dtype)
    sin = freqs.sin().to(orig_dtype)
    rotated = (rotated * cos) + (_rotate_half_interleaved(rotated) * sin)
    out = torch.cat([rotated, passed], dim=-1)
    return out


class RotaryEmbedding(nn.Module):
    """Non-xpos 1-D rotary embeddings matching x_transformers' interleaved layout.

    ``forward_from_seq_len(T)`` returns ``(freqs, xpos_scale=None)`` where
    freqs is ``(1, T, dim)``. The DiT only ever uses ``xpos_scale=None``
    (released ckpt's ``use_xpos`` is implicitly False).
    """

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward_from_seq_len(
        self, seq_len: int,
    ) -> tuple[torch.Tensor, None]:
        t = torch.arange(seq_len, device=self.inv_freq.device)
        # einsum('b i, j -> b i j') with t unsqueezed to (1, T) and
        # inv_freq as (D//2,). Result: (1, T, D//2).
        freqs = torch.einsum(
            "i,j->ij", t.type_as(self.inv_freq), self.inv_freq,
        ).unsqueeze(0)  # (1, T, D//2)
        # Stack pair-wise then flatten so each adjacent (f, f) pair lines
        # up with ``rotate_half_interleaved``'s (-x2, x1) layout.
        freqs = torch.stack((freqs, freqs), dim=-1).flatten(-2)  # (1, T, D)
        return freqs, None


# ===========================================================================
# DiT building blocks (RMSNorm, FeedForward, Attention, DiTBlock, FinalLayer,
# CondEmbedder)
# ===========================================================================


class _RMSNorm(nn.Module):
    """Plain RMSNorm with a learnable scale (mirrors upstream)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.to(self.weight.dtype)
        return F.rms_norm(
            x, normalized_shape=(x.shape[-1],), weight=self.weight, eps=self.eps,
        )


class _FeedForward(nn.Module):
    """Linear → GELU → Dropout → Linear (port of upstream FeedForward).

    Layer indices in the released ckpt: ``ff.0.0`` (first Linear),
    ``ff.0.1`` (GELU, no params), ``ff.1`` (Dropout, no params),
    ``ff.2`` (second Linear). Match by integer index.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: float = 4,
        dropout: float = 0.0,
        approximate: str = "none",
    ) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(approximate=approximate),
        )
        self.ff = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class _Attention(nn.Module):
    """Single-block attention with optional QK-norm, RoPE, and key-padding mask.

    Param names — `to_q`, `to_k`, `to_v`, `to_out.0`, (`to_out.1` is a
    Dropout, no params) — mirror upstream exactly so the talker ckpt's
    ``blocks.N.attn.to_q.weight`` etc. load by state_dict equality.
    `q_norm` / `k_norm` are present only when ``qk_norm="rms_norm"``
    (released ckpt sets qk_norm=None, so both are None and absent from
    state_dict).

    Mask handling matches upstream (`talker_module.Attention.forward`):
      * ``mask`` is a ``(B, T)`` boolean key-padding mask — True for
        valid positions, False for padding.
      * When ``attn_mask_enabled=True``: build an SDPA attention mask
        from ``mask`` so padded keys are excluded from softmax.
      * Regardless of `attn_mask_enabled`: zero out output rows at
        masked-out positions before returning (matches upstream's
        unconditional ``x.masked_fill(~mask, 0.0)``).

    The released flowmodel + aggregator configs set
    ``attn_mask_enabled=False`` so the SDPA mask branch is a no-op on
    the live model; we still preserve the parameter for parity.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        qk_norm: str | None = None,
        attn_mask_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads
        self.dropout = dropout
        self.attn_mask_enabled = attn_mask_enabled

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        if qk_norm is None:
            self.q_norm = None
            self.k_norm = None
        elif qk_norm == "rms_norm":
            self.q_norm = _RMSNorm(dim_head)
            self.k_norm = _RMSNorm(dim_head)
        else:
            raise ValueError(f"Unimplemented qk_norm: {qk_norm!r}")

        # ``to_out`` is a ModuleList of [Linear, Dropout] (matches
        # upstream so ckpt keys ``to_out.0.weight`` etc. land).
        self.to_out = nn.ModuleList([
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout),
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope: tuple[torch.Tensor, torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        q = self.to_q(x).view(B, -1, self.heads, self.dim_head).transpose(1, 2)
        k = self.to_k(x).view(B, -1, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(x).view(B, -1, self.heads, self.dim_head).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        if rope is not None:
            freqs, _xpos_scale = rope  # xpos_scale always None on this path
            q = _apply_rotary_pos_emb(q, freqs)
            k = _apply_rotary_pos_emb(k, freqs)

        # SDPA mask. Upstream builds a (B', H, T, T) bool mask from a
        # (B, T) key-padding mask and uses additive masking via SDPA's
        # attn_mask kwarg. We replicate the same shape so float weights
        # see identical attention patterns.
        attn_mask = None
        if self.attn_mask_enabled and mask is not None:
            # mask shape: (B, T). Expand to (B, H, Tq, Tk).
            attn_mask = mask[:, None, None, :].expand(B, self.heads, q.shape[-2], k.shape[-2])

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
        )
        out = out.transpose(1, 2).reshape(B, -1, self.inner_dim)
        out = self.to_out[0](out)
        out = self.to_out[1](out)

        if mask is not None:
            # Unconditional output-zeroing at masked positions (matches
            # upstream's ``x.masked_fill(~mask, 0.0)``, executed even
            # when attn_mask_enabled is False).
            out = out.masked_fill(~mask[:, :, None], 0.0)
        return out


class _DiTBlock(nn.Module):
    """Pre-norm attention + pre-norm FFN with residuals (upstream DiTBlock).

    Forward signature matches upstream `(x, mask, rope)` so the
    Aggregator can pass a key-padding mask through to the attention.
    For the CFM DiT path the caller passes mask=None.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        qk_norm: str | None = None,
        attn_mask_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = _RMSNorm(hidden_size)
        self.attn = _Attention(
            dim=hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            attn_mask_enabled=attn_mask_enabled,
        )
        self.norm2 = _RMSNorm(hidden_size)
        self.mlp = _FeedForward(
            dim=hidden_size, mult=mlp_ratio, dropout=dropout, approximate="tanh",
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        rope: tuple[torch.Tensor, torch.Tensor | None] | None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask, rope=rope)
        x = x + self.mlp(self.norm2(x))
        return x


class _FinalLayer(nn.Module):
    """RMSNorm → Linear; projects DiT hidden states back to ``out_channels``."""

    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = _RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm_final(x))


class _CondEmbedder(nn.Module):
    """Projects LLM hidden states (cond) into the DiT hidden space."""

    def __init__(self, input_feature_size: int, hidden_size: int) -> None:
        super().__init__()
        self.cond_embedder = nn.Linear(input_feature_size, hidden_size)

    def forward(self, llm_cond: torch.Tensor) -> torch.Tensor:
        return self.cond_embedder(llm_cond)


# ===========================================================================
# DiT (assembles N DiTBlocks + embedders + final layer)
# ===========================================================================


class DiT(nn.Module):
    """Diffusion-transformer for audio-latent generation (port of upstream DiT).

    Forward signature mirrors upstream so the calling code in
    ``CFM.sample`` (and `forward_with_cfg`) works unchanged. The
    optional ``spk_embedder`` is omitted on the released ckpt (the
    flowmodel config has no ``spk_dim``).
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_size: int = 1024,
        depth: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        llm_cond_dim: int = 896,
        dropout: float = 0.0,
        qk_norm: str | None = None,
        spk_dim: int | None = None,
        attn_mask_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.t_embedder = DiTTimestepEmbedding(hidden_size)
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.c_embedder = _CondEmbedder(llm_cond_dim, hidden_size)
        self.spk_embedder = (
            nn.Linear(spk_dim, hidden_size) if spk_dim is not None else None
        )

        self.rotary_embed = RotaryEmbedding(hidden_size // num_heads)
        self.blocks = nn.ModuleList([
            _DiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                qk_norm=qk_norm,
                attn_mask_enabled=attn_mask_enabled,
            )
            for _ in range(depth)
        ])
        self.final_layer = _FinalLayer(hidden_size, self.out_channels)

    def forward(
        self,
        x: torch.Tensor,                # (B, patch_size, in_channels)
        t: torch.Tensor,                # (B,) or scalar
        c: torch.Tensor,                # (B, 1, llm_cond_dim)
        latent_history: torch.Tensor,   # (B, his_patch_size, in_channels)
        spk_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns hidden states of shape ``(B, prefix + T, out_channels)``.

        ``prefix`` is 1 (t+c) plus 1 if spk_embedder is set; the caller
        is expected to take the last ``T`` rows where ``T`` is the
        sum of ``latent_history`` and ``x`` lengths.
        """
        x = torch.cat([latent_history, x], dim=1)
        x = self.x_embedder(x)
        t_h = self.t_embedder(t).unsqueeze(1)
        c_h = self.c_embedder(c)
        y = t_h + c_h
        if spk_emb is None:
            if self.spk_embedder is not None:
                raise AssertionError(
                    "DiT was built with spk_embedder but spk_emb was None at forward."
                )
            x = torch.cat([y, x], dim=1)
        else:
            assert self.spk_embedder is not None, "spk_emb provided but spk_embedder=None"
            x = torch.cat([self.spk_embedder(spk_emb), y, x], dim=1)

        rope = self.rotary_embed.forward_from_seq_len(x.shape[1])
        for block in self.blocks:
            # DiT path: mask=None (CFM only uses RoPE; the Aggregator is
            # what actually exercises the mask branch).
            x = block(x, None, rope)
        return self.final_layer(x)

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        latent_history: torch.Tensor,
        spk_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Classifier-free guidance: double the batch and pass null cond.

        Returns only the last ``x.shape[1]`` rows (the denoised x).
        """
        x_cat = torch.cat([x, x], dim=0)
        lh_cat = torch.cat([latent_history, latent_history], dim=0)
        null_c = torch.zeros_like(c)
        c_cat = torch.cat([c, null_c], dim=0)
        if t.ndim == 0:
            t = t.repeat(x_cat.shape[0])
        spk_cat = None if spk_emb is None else torch.cat([spk_emb, spk_emb], dim=0)
        out = self.forward(x_cat, t, c_cat, lh_cat, spk_cat)
        return out[:, -x.shape[1]:, :]


# ===========================================================================
# CFM (Conditional Flow Matching sampler)
# ===========================================================================


def get_epss_timesteps(
    n: int, device: torch.device | str, dtype: torch.dtype,
) -> torch.Tensor:
    """EPSS schedule (port of upstream ``get_epss_timesteps``).

    Returns ``n + 1`` integration timesteps in [0, 1]. Predefined
    fixed-step schedules (5, 6, 7, 10, 12, 16) match the upstream's
    empirically-tuned packing of more steps near t=0 where prediction
    error is highest; other ``n`` values fall back to linspace.
    """
    dt = 1 / 32
    predefined = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    schedule = predefined.get(n)
    if not schedule:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return dt * torch.tensor(schedule, device=device, dtype=dtype)


class CFM(nn.Module):
    """Conditional Flow Matching sampler over a wrapped DiT.

    Single ``sample`` entry point — given an LLM condition and a noise
    latent, integrate the velocity field for ``steps`` substeps with
    classifier-free guidance.
    """

    def __init__(
        self,
        model: nn.Module,
        steps: int = 10,
        sway_sampling_coef: float | None = -1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.steps = steps
        self.sway_sampling_coef = sway_sampling_coef

    @torch.no_grad()
    def sample(
        self,
        llm_cond: torch.Tensor,           # (B, 1, llm_cond_dim)
        lat_cond: torch.Tensor,           # (B, his_patch_size, latent_dim)
        y0: torch.Tensor,                 # (B, patch_size, latent_dim) — initial noise
        t: torch.Tensor,                  # (steps + 1,) — from get_epss_timesteps
        sde_args: torch.Tensor,           # (3,) — [cfg_strength, sigma, temperature]
        sde_rnd: torch.Tensor,            # (steps, B, patch_size, latent_dim)
    ) -> torch.Tensor:
        """Returns the denoised latent ``(B, patch_size, latent_dim)``."""
        if t.shape[0] != self.steps + 1:
            raise ValueError(
                f"CFM.sample: expected t of length steps+1 = {self.steps + 1}, got {t.shape[0]}"
            )
        if sde_rnd.shape[0] != self.steps:
            raise ValueError(
                f"CFM.sample: expected sde_rnd[0] = {self.steps}, got {sde_rnd.shape[0]}"
            )

        def velocity(fn_t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            pred_cfg = self.model.forward_with_cfg(x, fn_t, llm_cond, lat_cond, None)
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            # Standard CFG composition.
            return pred + (pred - null_pred) * sde_args[0]

        if self.sway_sampling_coef is not None:
            t = t + self.sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        for step in range(self.steps):
            dt = t[step + 1] - t[step]
            y0 = y0 + velocity(t[step], y0) * dt
            # SDE noise term: sigma * sqrt(temperature) * sqrt(|dt|) * eps
            y0 = y0 + sde_args[1] * (sde_args[2] ** 0.5) * (dt.abs() ** 0.5) * sde_rnd[step]
        return y0


# ===========================================================================
# Factory: build a DiT + CFM from TalkerConfig
# ===========================================================================


def build_talker_cfm(
    talker_config,
    llm_cond_dim: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
) -> CFM:
    """Construct DiT + CFM from a :class:`TalkerConfig`.

    The released ckpt's flowmodel block carries
    ``in_channels=64, hidden_size=1024, depth=8, num_heads=16, mlp_ratio=4``
    with no spk_dim. ``llm_cond_dim`` defaults to the talker LLM hidden
    size (896) when not specified.
    """
    flow = talker_config.flowmodel
    if llm_cond_dim is None:
        llm_cond_dim = talker_config.llm.hidden_size
    dit = DiT(
        in_channels=flow.in_channels,
        hidden_size=flow.hidden_size,
        depth=flow.depth,
        num_heads=flow.num_heads,
        mlp_ratio=flow.mlp_ratio,
        llm_cond_dim=llm_cond_dim,
        dropout=flow.dropout,
        qk_norm=flow.qk_norm,
        attn_mask_enabled=flow.attn_mask_enabled,
    )
    cfm = CFM(model=dit, steps=talker_config.steps)
    cfm = cfm.to(dtype=dtype, device=device)
    cfm.eval()
    return cfm


# ===========================================================================
# Aggregator (DiT-shaped, maps audio latents back to LLM cond space)
# ===========================================================================


class Aggregator(nn.Module):
    """Maps generated audio-latent patches back to LLM embedding space.

    Port of upstream `talker_module.Aggregator` (lines 702-744). Same
    DiTBlock stack as the CFM head but the input embedder is `nn.Linear`
    (audio-latent → hidden) plus a learnable [CLS]-style `word_embedder`
    prepended to the sequence; the output is the `[CLS]` row only,
    projected to `llm_input_dim` so it can re-enter the talker LLM's
    embedding space (closing the conditional-history loop).

    The released aggregator block matches the flowmodel shape
    (`depth=8, hidden_size=1024, num_heads=16, mlp_ratio=4, in_channels=64`)
    except `dropout=0.1` and an `attn_mask_enabled=False` default that
    still preserves the output-masking branch.
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_size: int = 1024,
        depth: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        llm_input_dim: int = 896,
        dropout: float = 0.1,
        qk_norm: str | None = None,
        attn_mask_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # Learnable [CLS] token (single-row embedding table — exactly as
        # upstream uses ``nn.Embedding(1, hidden_size)`` indexed at 0).
        self.word_embedder = nn.Embedding(1, hidden_size)
        self.x_embedder = nn.Linear(in_channels, hidden_size)

        self.rotary_embed = RotaryEmbedding(hidden_size // num_heads)
        self.blocks = nn.ModuleList([
            _DiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                qk_norm=qk_norm,
                attn_mask_enabled=attn_mask_enabled,
            )
            for _ in range(depth)
        ])
        self.final_layer = _FinalLayer(hidden_size, llm_input_dim)

    def forward(
        self,
        x: torch.Tensor,                 # (B, T, in_channels) audio latents
        mask: torch.Tensor | None = None,   # (B, T) key-padding mask, True = valid
    ) -> torch.Tensor:
        """Returns the [CLS] row only: ``(B, 1, llm_input_dim)``.

        Mirrors upstream `Aggregator.forward`: prepend a single learnable
        [CLS] token, prepend a True-cell to the mask, run all DiT blocks,
        project to ``llm_input_dim`` via `final_layer`, return the
        leading row.
        """
        B = x.shape[0]
        h = self.x_embedder(x)
        cls_ids = torch.zeros((B, 1), dtype=torch.long, device=h.device)
        cls_embed = self.word_embedder(cls_ids)
        h = torch.cat([cls_embed, h], dim=1)

        rope = self.rotary_embed.forward_from_seq_len(h.shape[1])
        if mask is not None:
            # Prepend a True column so the [CLS] row is never masked.
            mask_pad = mask[:, :1].clone().detach()
            mask = torch.cat([mask_pad, mask], dim=-1)

        for block in self.blocks:
            h = block(h, mask, rope)
        h = self.final_layer(h)
        return h[:, :1, :]


def build_aggregator(
    talker_config,
    llm_input_dim: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
) -> Aggregator:
    """Construct an :class:`Aggregator` from a :class:`TalkerConfig`.

    The released ckpt's aggregator block carries
    ``in_channels=64, hidden_size=1024, depth=8, num_heads=16,
    mlp_ratio=4, dropout=0.1``. ``llm_input_dim`` defaults to
    ``talker_config.llm.hidden_size`` (896).
    """
    agg = talker_config.aggregator
    if llm_input_dim is None:
        llm_input_dim = talker_config.llm.hidden_size
    module = Aggregator(
        in_channels=agg.in_channels,
        hidden_size=agg.hidden_size,
        depth=agg.depth,
        num_heads=agg.num_heads,
        mlp_ratio=agg.mlp_ratio,
        llm_input_dim=llm_input_dim,
        dropout=agg.dropout,
        qk_norm=agg.qk_norm,
        attn_mask_enabled=agg.attn_mask_enabled,
    )
    module = module.to(dtype=dtype, device=device)
    module.eval()
    return module


# ===========================================================================
# Talker LLM backbone (Qwen2)
# ===========================================================================


def build_talker_llm(
    talker_llm_config,
    attn_implementation: str = "sdpa",
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
):
    """Construct a HF `Qwen2Model` from our `TalkerLLMConfig`.

    The talker's LLM is a stock Qwen2 model — no custom modules, no
    TP needed in the typical topology (the talker colocates on a
    single rank). Reusing `transformers.Qwen2Model` keeps the surface
    small and inherits HF's KV-cache + attention impl. The ckpt's
    weight keys under `talker/model.safetensors` start with `model.`
    and follow the standard Qwen2 layout, so the eventual loader
    will be a simple prefix strip.

    Args:
        talker_llm_config: `TalkerLLMConfig` instance.
        attn_implementation: passed through to Qwen2Config so the
            model can use FA2 / SDPA. The upstream vllm-omni talker
            uses ``"sdpa"`` (the ckpt's Qwen2 has
            `_attn_implementation: flash_attention_2` baked into its
            config dict but the vllm-omni runtime forcibly overrides
            to sdpa to play nicely with vLLM's attention machinery
            — we follow the same default).
        dtype: cast the model to this dtype after construction.
        device: device to materialise the model on.

    Returns:
        A `transformers.models.qwen2.modeling_qwen2.Qwen2Model`
        instance with all parameters allocated (weights are still
        random; the loader populates them later).
    """
    try:
        from transformers import Qwen2Config, Qwen2Model
    except ImportError as e:
        raise ImportError(
            "build_talker_llm requires transformers >= 4.43 (Qwen2 support). "
            f"Original error: {e}"
        ) from e

    llm_cfg = Qwen2Config(
        vocab_size=talker_llm_config.vocab_size,
        hidden_size=talker_llm_config.hidden_size,
        intermediate_size=talker_llm_config.intermediate_size,
        num_hidden_layers=talker_llm_config.num_hidden_layers,
        num_attention_heads=talker_llm_config.num_attention_heads,
        num_key_value_heads=talker_llm_config.num_key_value_heads,
        hidden_act=talker_llm_config.hidden_act,
        max_position_embeddings=talker_llm_config.max_position_embeddings,
        rms_norm_eps=talker_llm_config.rms_norm_eps,
        rope_theta=talker_llm_config.rope_theta,
        use_sliding_window=talker_llm_config.use_sliding_window,
        sliding_window=talker_llm_config.sliding_window,
        max_window_layers=talker_llm_config.max_window_layers,
        tie_word_embeddings=talker_llm_config.tie_word_embeddings,
        attention_dropout=talker_llm_config.attention_dropout,
        use_cache=talker_llm_config.use_cache,
        bos_token_id=talker_llm_config.bos_token_id,
        eos_token_id=talker_llm_config.eos_token_id,
        attn_implementation=attn_implementation,
    )
    model = Qwen2Model(llm_cfg)
    model = model.to(dtype=dtype, device=device)
    model.eval()
    return model


def build_talker_heads(
    talker_config,
    spk_embed_dim: int = 192,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
) -> dict[str, nn.Module]:
    """Build the talker's small per-purpose Linear heads.

    Two heads sit alongside the LLM + CFM + Aggregator + AudioVAE:

      * ``stop_head`` — ``Linear(hidden_size, 2, bias=True)``: binary
        end-of-audio classifier consumed during the generation loop
        to decide when to stop.
      * ``spk_head`` — ``Linear(spk_embed_dim=192, hidden_size,
        bias=True)``: projects a CAMPPlus speaker embedding into the
        LLM hidden space; the projected embedding is prepended to
        the prompt as a voice-condition token.

    Returned as a dict so callers can wire them into the talker
    forward without depending on a specific module-tree shape.
    """
    hidden = talker_config.llm.hidden_size
    stop_head = nn.Linear(hidden, 2, bias=True)
    spk_head = nn.Linear(spk_embed_dim, hidden, bias=True)
    stop_head = stop_head.to(dtype=dtype, device=device)
    spk_head = spk_head.to(dtype=dtype, device=device)
    stop_head.eval()
    spk_head.eval()
    return {"stop_head": stop_head, "spk_head": spk_head}


__all__ = [
    "DiT",
    "CFM",
    "Aggregator",
    "DiTTimestepEmbedding",
    "RotaryEmbedding",
    "get_epss_timesteps",
    "build_talker_cfm",
    "build_aggregator",
    "build_talker_llm",
    "build_talker_heads",
]
