"""The Praxis-1 action expert, from layers to the complete trunk-rope tower.

The expert is a second, smaller transformer beside the video trunk. It reads
the trunk's hidden states and never writes back, and it predicts the action
velocity. Nothing here is shared with the trunk, and that is the first thing
a port gets wrong: the two halves of the checkpoint use **opposite
conventions for nearly every convention they both have**.

Where the expert disagrees with the trunk:

* its RMS norm carries **no weight at all** and uses eps ``1e-4``, two orders
  larger than the trunk's ``1e-6``. ``norm_time.expert_rms_norm`` is that
  norm, and it is reused here rather than re-derived,
* the AdaLN's ``1 +`` applies to a **projection output**, not to a stored
  weight. The trunk's norms store ``weight - 1`` and add the 1 on load; the
  expert's AdaLN adds it per token, per forward, to ``proj(silu(cond))``.
  Folding it into the weight the trunk way gives a layer whose scale is
  always ``proj(...)``, which at the zero init of ``proj`` kills the branch
  entirely and after training is simply a different number,
* ``tap_norm_scale`` is a **plain ones-init weight**, the ordinary
  convention, sitting in the same checkpoint as the trunk's ``1 + w`` norms.
  Loading it as ``1 + w`` doubles every tap,
* the output norm is a **LayerNorm**: it subtracts the mean, has no affine
  and uses eps ``1e-6``. Every other norm in the expert is an RMS norm, so
  this is the one place where reaching for the familiar one is wrong.

Three more orderings that run fine when inverted:

* **RoPE before the q/k norm**, as in the trunk (``order: before_norm``), and
  **one table per side**. A cross-attention layer's keys are trunk tokens at
  trunk positions while its queries are expert tokens at the positions of
  their own noised frames, so the two sides take different tables. The
  self-attention layers pass the query table twice, which is the caller's
  decision and not this module's,
* the expert's rotary ``effective_dim`` equals its ``dim``, which equals
  ``head_dim`` (64). Unlike the trunk, where half the pairs have zero
  frequency, here **every pair rotates**,
* GEGLU gates on the **second** half and applies ``gelu(tanh)`` to the
  **first**. Swapping them runs and is a different function. The expansion
  is 4x, so ``mlp_in`` is ``1024 -> 8192`` and the gated hidden width is
  4096, not 8192.

The timestep enters **twice, through two separate embedders with their own
weights**: once fused into the action tokens (:class:`ActionTauEncoder`) and
once as the AdaLN condition shared by both halves of every block. Both use
features ``[cos, sin]`` in that order over a raw ``t`` in ``[0, 1]`` with
``frequency_dim`` 256 and no ``1000x`` scaling. ``[sin, cos]`` would permute
the first linear's input columns and still run; the missing ``1000x`` means
the high frequencies barely move over the interval, which is the
checkpoint's choice and not a bug to fix.

There is no classifier-free guidance anywhere in the expert. Guidance, where
the reference has it, lives in the sampler above this.

The reference's fused Triton kernels (``FUSED_AE_NORM``, ``FUSED_KV_SOURCE``)
are bit-identical to the eager path by construction and are gated off for
non-CUDA inputs, so the eager order written here is the contract.

Weight initialization is deliberately not replicated: these modules are
filled by the weight loader, as with
:mod:`mstar.model.praxis1.components.attention`.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.praxis1.components.norm_time import expert_rms_norm
from mstar.model.praxis1.components.rope import RotaryEmbedding, apply_rope


def geglu(x: torch.Tensor) -> torch.Tensor:
    """Gated GELU over the last dim: ``gelu_tanh(first half) * second half``.

    Which half gates is not interchangeable, and neither is the GELU: the
    ``tanh`` approximation is what the checkpoint was trained with, and the
    exact erf form differs in the last bits everywhere.
    """
    value, gate = torch.chunk(x, 2, dim=-1)
    return F.gelu(value, approximate="tanh") * gate


class ExpertTimestepEmbedder(nn.Module):
    """Sinusoidal flow-time features, then a two-layer MLP.

    Not the trunk's timestep embedding, which reads fixed random Fourier
    features from checkpoint buffers. This one is an ordinary sinusoidal
    ladder of ``frequency_dim // 2`` frequencies, concatenated as
    ``[cos, sin]``.

    Two casts are load bearing. The frequencies are built in float32 and then
    **rounded to the timestep's dtype** before the product, so a bfloat16
    timestep quantizes the frequencies as well as the angle. The features are
    then cast to the first linear's weight dtype, not to the timestep's.

    ``mlp`` is a :class:`~torch.nn.Sequential` so the parameter suffixes stay
    the checkpoint's (``mlp.0.*``, ``mlp.2.*``).
    """

    def __init__(self, dim: int, frequency_dim: int = 256, max_period: float = 10_000.0) -> None:
        super().__init__()
        self.frequency_dim = frequency_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(nn.Linear(frequency_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """``t (...)`` to ``(..., dim)``. ``t`` is the raw flow time in ``[0, 1]``."""
        half = self.frequency_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half, device=t.device, dtype=torch.float32) / half
        ).to(dtype=t.dtype)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        # [cos, sin], in that order. The reverse is a column permutation of
        # mlp.0 and runs identically.
        features = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(features.to(dtype=self.mlp[0].weight.dtype))


class ExpertAdaLN(nn.Module):
    """Scale-only adaptive norm: ``(1 + proj(silu(cond))) * expert_rms_norm(x)``.

    Scale only: there is no shift and no gate, so the condition can only
    rescale the normed tokens.

    The ``1 +`` is added to the **projection's output** every forward. See the
    module docstring for why that is not the same as the trunk's stored
    ``1 + weight``, and why both still run.

    The sum and the product are formed in float32 and the result is cast back
    to the tokens' dtype. ``expert_rms_norm`` has already returned the tokens'
    dtype at that point, so the float32 here re-widens it: that double
    rounding is the reference's, and removing it changes bfloat16 output.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """``tokens (B, S, D)`` and ``cond (B, S, D)`` to ``(B, S, D)``."""
        scale = self.proj(F.silu(cond))
        return ((1 + scale.float()) * expert_rms_norm(tokens)).to(dtype=tokens.dtype)


class ActionTauEncoder(nn.Module):
    """Action latents with the flow time fused in at the input.

    ``w3(silu(w2([w1(actions) ++ tau_features])))``. The timestep is
    concatenated, not added, so ``w2`` is twice as wide as ``dim``, and this
    embedder's weights are separate from the AdaLN condition's: the same
    ``tau`` goes through two different MLPs on the way into one block.

    The ``tau`` features are cast to the **action branch's** dtype before the
    concatenation, so in bfloat16 the timestep path is rounded to match the
    actions rather than the reverse.
    """

    def __init__(self, action_patch_dim: int, dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(action_patch_dim, dim)
        self.tau = ExpertTimestepEmbedder(dim)
        self.w2 = nn.Linear(2 * dim, dim)
        self.w3 = nn.Linear(dim, dim)

    def forward(self, actions: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """``actions (B, T, action_patch_dim)`` and ``tau (B, T)`` to ``(B, T, dim)``.

        ``tau`` is per action token, not per sample: every token of a chunk
        carries that chunk's flow time.
        """
        a = self.w1(actions)
        t = self.tau(tau).to(a.dtype)
        if t.shape[:2] != a.shape[:2]:
            raise ValueError(f"tau of {tuple(tau.shape)} does not cover actions of {tuple(actions.shape[:2])}")
        return self.w3(F.silu(self.w2(torch.cat([a, t], dim=-1))))


class ExpertKVSource(nn.Module):
    """Trunk tensors to the cross-attention key/value sequence.

    The KV of a cross-attention layer is ``[global tokens | one tap]``. The
    two arrive through **different projections of the same trunk width**:
    the global tokens (registers, anchor frame, text, taken from the trunk's
    *input* sequence) through ``text_proj``, and the tap (a trunk *block
    output*) through an RMS norm, ``tap_norm_scale`` and ``tap_proj``. One
    ``tap_proj`` serves all taps.

    ``tap_norm_scale`` is ones-init and multiplies directly. It is not
    ``1 + w`` despite sitting next to the trunk's norms that are, and the
    product is cast back to the tap's dtype before ``tap_proj``, not after.

    ``global_kv`` is computed once per expert forward and reused by every
    cross layer, which is why it is a separate call rather than an argument
    to :meth:`forward`.
    """

    tap_norm_scale: nn.Parameter | None

    def __init__(self, trunk_dim: int, dim: int, tap_norm: bool = True) -> None:
        super().__init__()
        self.tap_proj = nn.Linear(trunk_dim, dim, bias=False)
        self.text_proj = nn.Linear(trunk_dim, dim, bias=False)
        self.tap_norm_scale = nn.Parameter(torch.ones(trunk_dim)) if tap_norm else None

    def global_kv(self, global_tokens: torch.Tensor) -> torch.Tensor:
        """``(B, G, trunk_dim)`` to ``(B, G, dim)``. No norm on this side."""
        return self.text_proj(global_tokens)

    def forward(self, tap: torch.Tensor, global_kv: torch.Tensor | None = None) -> torch.Tensor:
        """``tap (B, S, trunk_dim)`` to ``(B, G + S, dim)``, globals first."""
        feat = tap
        if self.tap_norm_scale is not None:
            feat = (expert_rms_norm(feat) * self.tap_norm_scale).to(feat.dtype)
        feat = self.tap_proj(feat)
        if global_kv is None:
            return feat
        return torch.cat([global_kv.to(feat.dtype), feat], dim=1)


def dense_sdpa_attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: object) -> torch.Tensor:
    """The parity driver: SDPA over a dense ``(Q, KV)`` boolean mask.

    The mask is per row of the expert sequence and is shared by every batch
    entry and head, so it broadcasts in rather than being materialized per
    head. The served path wants the engine's kernel over the same predicate;
    this is here so the tests can drive the block without one.

    ``enable_gqa`` is deliberately absent. The expert has no GQA: 16 heads of
    64 is the full width 1024 on all three of q, k and v.
    """
    if mask is not None and not isinstance(mask, torch.Tensor):
        raise TypeError("dense SDPA needs a tensor mask")
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None if mask is None else mask[None, None])


class ActionExpertBlock(nn.Module):
    """One expert layer: attention, then a GEGLU MLP, each behind its own AdaLN.

    Both halves are pre-norm with a residual, and **both** take a scale-only
    AdaLN on the same shared condition. The two AdaLNs have separate weights.

    ``is_cross`` selects what the keys and values come from. Even layers of
    the expert are cross-attention over a trunk tap and odd layers are
    self-attention among the expert tokens; the alternation is the caller's,
    and a block only knows which of the two it is. A self-attention block
    reads the **AdaLN'd** tokens for its KV, not the residual stream, so
    there the q and the kv side share one norm and in a cross block they do
    not.

    The two residual adds are on ``x``, never on the normed copy, so a layer
    passes the unnormalized stream through untouched. That is what makes the
    LayerNorm in :class:`ActionHead` necessary at the end of the stack.
    """

    def __init__(self, *, dim: int, num_heads: int, head_dim: int, is_cross: bool, kv_dim: int) -> None:
        super().__init__()
        self.is_cross = is_cross
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim

        self.adaln_attn = ExpertAdaLN(dim)
        self.to_q = nn.Linear(dim, inner, bias=False)
        # One fused k/v projection, unlike the trunk's three separate
        # matrices: the checkpoint stores it as one tensor and the reference
        # chunks it, so splitting it here would only move the same split into
        # the loader.
        self.to_kv = nn.Linear(kv_dim if is_cross else dim, 2 * inner, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)

        self.adaln_mlp = ExpertAdaLN(dim)
        hidden = 4 * dim
        self.mlp_in = nn.Linear(dim, 2 * hidden)
        self.mlp_out = nn.Linear(hidden, dim)

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        """``(B, N, H * D)`` to ``(B, H, N, D)``."""
        return t.view(t.shape[0], t.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        *,
        kv_source: torch.Tensor | None,
        mask: object,
        q_table: torch.Tensor,
        k_table: torch.Tensor,
        attend: Callable = dense_sdpa_attend,
    ) -> torch.Tensor:
        """``x (B, Q, D)`` and ``cond (B, Q, D)`` to ``(B, Q, D)``.

        ``q_table`` and ``k_table`` are rotary tables ``(B, N, D/2, 2)`` for
        the query and the key positions. A self-attention block gets the
        query table twice; a cross-attention block gets the trunk's KV table
        as ``k_table``, and handing it the query table instead misplaces
        every key while leaving every shape intact.
        """
        b, q_len, _ = x.shape
        h = self.adaln_attn(x, cond)
        q = self.to_q(h)
        if self.is_cross:
            if kv_source is None:
                raise ValueError("a cross-attention block needs a kv_source")
            kv_in = kv_source
        else:
            if kv_source is not None:
                raise ValueError("a self-attention block reads its own tokens; drop the kv_source")
            kv_in = h
        k, v = self.to_kv(kv_in).chunk(2, dim=-1)

        q, k = self._split(q), self._split(k)
        # apply_rope rotates a pair with one table, and the two sides need
        # different tables here, so each side passes itself twice and keeps
        # its own output. Both outputs of a call are equal when both inputs
        # are, so which index is kept is cosmetic; it follows the reference
        # so the two lines read the same.
        q = apply_rope(q, q, q_table)[0]
        k = apply_rope(k, k, k_table)[1]
        # Norm after the rotation, and per head dimension. The expert's norm
        # has no weight, so unlike the trunk there is nothing to load here.
        attn = attend(expert_rms_norm(q), expert_rms_norm(k), self._split(v), mask)

        x = x + self.to_out(attn.transpose(1, 2).reshape(b, q_len, -1))
        u = geglu(self.mlp_in(self.adaln_mlp(x, cond)))
        return x + self.mlp_out(u)


class ActionHead(nn.Module):
    """The expert's output: LayerNorm over the action tokens, then a linear.

    ``norm_out`` is a LayerNorm, which **subtracts the mean**, has no affine
    parameters and uses eps ``1e-6``. It is the only non-RMS norm in the
    expert, and swapping in an RMS norm is the mistake this class exists to
    label: it runs, it has no weights either way, and it silently drops the
    centering.

    Only the trailing ``num_action_tokens`` rows are read. The proprio and
    future tokens that precede them in the expert sequence condition the
    layers and are then dropped.

    ``head_plus_zero`` adds ``+0.0`` to the result, which the shipped
    checkpoint config sets. Its only observable effect is turning ``-0.0``
    into ``+0.0``, which is a bit difference and therefore part of the
    contract.
    """

    def __init__(self, dim: int, action_patch_dim: int, head_plus_zero: bool = True) -> None:
        super().__init__()
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.head = nn.Linear(dim, action_patch_dim)
        self.head_plus_zero = head_plus_zero

    def forward(self, x: torch.Tensor, num_action_tokens: int) -> torch.Tensor:
        """``x (B, Q, D)`` to the action velocity ``(B, T, action_patch_dim)``."""
        out = self.head(self.norm_out(x[:, -num_action_tokens:]))
        if self.head_plus_zero:
            out = out + 0.0
        return out


def trunk_query_rope_ids(
    kv_rope_ids: torch.Tensor,
    *,
    num_global_tokens: int,
    num_clean_frames: int,
    num_noised_frames: int,
    tokens_per_frame: int,
    num_future_per_chunk: int,
    chunk_size: int,
) -> torch.Tensor:
    """Gather one trunk position per expert query from ``[global | clean | noised]``.

    Proprio and future queries use the first noised frame of their chunk.
    Action query ``i`` uses noised frame ``i``. The minimum position in a
    frame is its top-left token, including after the trunk's position shifts.
    """
    batch = kv_rope_ids.shape[0]
    frame_tokens = kv_rope_ids[:, num_global_tokens:]
    n_frames = num_clean_frames + num_noised_frames
    if frame_tokens.shape[1] != n_frames * tokens_per_frame:
        raise ValueError("KV positions do not cover the clean and noised frames")
    if not chunk_size or num_noised_frames % chunk_size:
        raise ValueError("noised frames must split into full action chunks")
    noised_ids = frame_tokens.view(batch, n_frames, tokens_per_frame).amin(dim=-1)
    noised_ids = noised_ids[:, num_clean_frames:]
    n_chunks = num_noised_frames // chunk_size
    device = kv_rope_ids.device
    chunks = torch.arange(n_chunks, device=device)
    prefix_frames = torch.cat([chunks, chunks.repeat_interleave(num_future_per_chunk)]) * chunk_size
    action_frames = torch.arange(num_noised_frames, device=device)
    return noised_ids.index_select(1, torch.cat([prefix_frames, action_frames]))


class ActionExpert(nn.Module):
    """The trunk-rope action expert over one chunk-causal encoder window.

    The caller supplies cross and self attention masks and the attention
    callable. The masks must encode ``window_size`` and
    ``clean_only_conditioning`` from the checkpoint config. This keeps mask
    planning with the attention resource while the model owns its
    projections, query layout and flow-time conditioning.
    The parameter names match the reference after ``action_expert.``.
    """

    def __init__(
        self,
        *,
        action_patch_dim: int,
        proprio_patch_dim: int,
        trunk_dim: int,
        dim: int = 1024,
        num_layers: int = 16,
        num_heads: int = 16,
        head_dim: int = 64,
        num_future_per_chunk: int = 8,
        max_chunks: int = 16,
        max_action_tokens: int = 64,
        tap_norm: bool = True,
        clean_only_conditioning: bool = False,
        rope: Mapping[str, int | float | str],
        global_order: Sequence[str] = ("register", "anchor", "text"),
        head_plus_zero: bool = False,
        layout_stamp: int | None = None,
    ) -> None:
        super().__init__()
        if num_layers < 2 or num_layers % 2:
            raise ValueError("num_layers must be a positive even count")
        if tuple(sorted(global_order)) != ("anchor", "register", "text"):
            raise ValueError("global_order must permute register, anchor and text")
        if rope.get("kind") != "trunk" or rope.get("order") != "before_norm":
            raise ValueError("the port supports trunk RoPE before the q/k norm")
        self.rope_table = RotaryEmbedding(
            dim=int(rope["dim"]),
            effective_dim=int(rope["effective_dim"]),
            base=int(rope["base"]),
            table_size=int(rope["table_size"]),
        )
        if int(rope["dim"]) != head_dim:
            raise ValueError("expert RoPE width must equal head_dim")
        self.global_order = tuple(global_order)
        self.num_future_per_chunk = num_future_per_chunk
        self.num_cross_layers = num_layers // 2
        self.dim = dim
        self.action_patch_dim = action_patch_dim
        self.trunk_dim = trunk_dim
        self.clean_only_conditioning = clean_only_conditioning
        self.head_plus_zero = head_plus_zero
        self.layout_stamp = layout_stamp
        self.max_action_tokens = max_action_tokens

        self.action_encoder = ActionTauEncoder(action_patch_dim, dim)
        self.timestep_embedder = ExpertTimestepEmbedder(dim)
        self.future_tokens = nn.Parameter(torch.zeros(num_future_per_chunk, dim))
        self.future_chunk_embed = nn.Embedding(max_chunks, dim)
        self.proprio_token = nn.Parameter(torch.zeros(1, dim))
        self.embed_proprio = nn.Linear(proprio_patch_dim, dim) if proprio_patch_dim > 0 else None
        self.tap_proj = nn.Linear(trunk_dim, dim, bias=False)
        self.text_proj = nn.Linear(trunk_dim, dim, bias=False)
        self.tap_norm_scale = nn.Parameter(torch.ones(trunk_dim)) if tap_norm else None
        self.blocks = nn.ModuleList(
            [
                ActionExpertBlock(
                    dim=dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    is_cross=idx % 2 == 0,
                    kv_dim=dim,
                )
                for idx in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.head = nn.Linear(dim, action_patch_dim)

    def forward_trunk_rope(
        self,
        *,
        noised_action_latents: torch.Tensor,
        action_timesteps: torch.Tensor,
        trunk_features: Sequence[torch.Tensor],
        global_tokens: torch.Tensor | None,
        proprio_latents: torch.Tensor | None,
        kv_rope_ids: torch.Tensor,
        chunk_size: int,
        num_clean_frames: int,
        num_noised_frames: int,
        tokens_per_frame: int,
        window_size: int,
        cross_mask: object,
        self_mask: object,
        attend: Callable = dense_sdpa_attend,
    ) -> torch.Tensor:
        """Predict action velocity; masks encode the caller's window policy."""
        if len(trunk_features) != self.num_cross_layers:
            raise ValueError(f"expected {self.num_cross_layers} taps, got {len(trunk_features)}")
        batch, n_actions, _ = noised_action_latents.shape
        if batch != 1 or n_actions != num_noised_frames:
            raise ValueError("one sample and one action per noised frame are required")
        if not chunk_size or n_actions % chunk_size:
            raise ValueError("action tokens must split into full chunks")
        n_chunks = n_actions // chunk_size
        if n_chunks > self.future_chunk_embed.num_embeddings:
            raise ValueError("future chunk embedding has too few rows")
        if action_timesteps.ndim == 1:
            tau = action_timesteps[:, None].expand(-1, n_actions)
        elif action_timesteps.shape == (batch, n_actions):
            tau = action_timesteps
        else:
            raise ValueError("action timesteps do not cover the action tokens")

        actions = self.action_encoder(noised_action_latents, tau)
        future = self.future_tokens[None, None].expand(batch, n_chunks, -1, -1)
        future = future + self.future_chunk_embed.weight[:n_chunks][None, :, None, :].to(future.dtype)
        future = future.reshape(batch, n_chunks * self.num_future_per_chunk, self.dim)
        proprio = self.proprio_token[None].expand(batch, n_chunks, -1).to(actions.dtype)
        if self.embed_proprio is not None and proprio_latents is not None:
            idx = torch.arange(n_chunks, device=actions.device) * chunk_size
            proprio = proprio + self.embed_proprio(proprio_latents.index_select(1, idx).to(proprio.dtype))
        x = torch.cat([proprio, future.to(actions.dtype), actions], dim=1)

        chunks = torch.arange(n_chunks, device=actions.device)
        query_chunks = torch.cat(
            [
                chunks,
                chunks.repeat_interleave(self.num_future_per_chunk),
                torch.arange(n_actions, device=actions.device) // chunk_size,
            ]
        )
        query_times = tau.index_select(1, (query_chunks * chunk_size).clamp(max=n_actions - 1))
        cond = self.timestep_embedder(query_times).to(x.dtype)

        n_globals = 0 if global_tokens is None else global_tokens.shape[1]
        expected_kv = n_globals + (num_clean_frames + num_noised_frames) * tokens_per_frame
        if kv_rope_ids.shape != (batch, expected_kv):
            raise ValueError("KV positions do not match the supplied frame layout")
        expected_frames = (batch, expected_kv - n_globals, self.trunk_dim)
        if any(tap.shape != expected_frames for tap in trunk_features):
            raise ValueError("tap shape does not match the supplied frame layout")
        if global_tokens is not None and global_tokens.shape != (batch, n_globals, self.trunk_dim):
            raise ValueError("global tokens must have the trunk width")
        q_ids = trunk_query_rope_ids(
            kv_rope_ids,
            num_global_tokens=n_globals,
            num_clean_frames=num_clean_frames,
            num_noised_frames=num_noised_frames,
            tokens_per_frame=tokens_per_frame,
            num_future_per_chunk=self.num_future_per_chunk,
            chunk_size=chunk_size,
        )
        q_table = self.rope_table(q_ids)
        k_table = self.rope_table(kv_rope_ids)
        if cross_mask is None or self_mask is None:
            raise ValueError("expert cross and self masks are required")
        if isinstance(cross_mask, torch.Tensor) and cross_mask.shape != (x.shape[1], expected_kv):
            raise ValueError("cross mask does not match expert queries and trunk keys")
        if isinstance(self_mask, torch.Tensor) and self_mask.shape != (x.shape[1], x.shape[1]):
            raise ValueError("self mask does not match expert queries")
        global_kv = None if global_tokens is None else self.text_proj(global_tokens)
        cross_idx = 0
        for block in self.blocks:
            if block.is_cross:
                tap = trunk_features[cross_idx]
                if self.tap_norm_scale is not None:
                    tap = (expert_rms_norm(tap) * self.tap_norm_scale).to(tap.dtype)
                frame_kv = self.tap_proj(tap)
                kv = frame_kv if global_kv is None else torch.cat([global_kv.to(frame_kv.dtype), frame_kv], dim=1)
                x = block(x, cond, kv_source=kv, mask=cross_mask, q_table=q_table, k_table=k_table, attend=attend)
                cross_idx += 1
            else:
                x = block(x, cond, kv_source=None, mask=self_mask, q_table=q_table, k_table=q_table, attend=attend)
        out = self.head(self.norm_out(x[:, -n_actions:]))
        if self.head_plus_zero:
            out = out + 0.0
        return out.view(batch, n_actions, self.action_patch_dim)
