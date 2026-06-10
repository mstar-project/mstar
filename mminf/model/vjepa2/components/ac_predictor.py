"""V-JEPA 2 action-conditioned predictor (for V-JEPA 2-AC).

Port of ``VisionTransformerPredictorAC`` and supporting blocks from
``vjepa2/src/models/ac_predictor.py`` + ``vjepa2/src/models/utils/modules.py``.
The HuggingFace Transformers port does NOT include the AC variant, so this
file stays close to the upstream naming to preserve checkpoint-key parity
with the upstream ``vjepa2-ac-vitg`` weights.

Key differences from the masked predictor:

- Fused ``qkv`` Linear (``dim -> dim*3``) per layer (upstream layout).
- Action + state + (optional) extrinsics tokens are interleaved into the
  spatial sequence per timestep: ``[a, s, x_0, ..., x_{H*W-1}]`` (+ ``e``
  if ``use_extrinsics``).  Action tokens rotate only along the depth axis.
- Causal attention across frames via ``build_action_block_causal_attention_mask``.
- Uses ``F.scaled_dot_product_attention`` (SDPA) — the attention mask is
  always present, so the eager fallback is unreachable.
"""

from __future__ import annotations


from mminf.engine.cache_manager import BatchedCacheManager
import torch
import torch.nn.functional as F
from torch import nn

from mminf.model.vjepa2.components.rope_utils import rotate_queries_or_keys, rotate_queries_or_keys_BNHD
from mminf.model.vjepa2.config import VJepa2ACPredictorConfig


def build_action_block_causal_attention_mask(
    grid_depth: int,
    grid_height: int,
    grid_width: int,
    add_tokens: int = 1,
) -> torch.Tensor:
    """Build a ``[N, N]`` boolean mask where frame ``t`` attends only to frames 0..t.

    Each frame contributes ``add_tokens + grid_height * grid_width`` tokens.
    """
    tokens_per_frame = add_tokens + (grid_height * grid_width)
    n = grid_depth * tokens_per_frame
    mask = torch.zeros(n, n, dtype=torch.bool)
    block = torch.ones(tokens_per_frame, tokens_per_frame, dtype=torch.bool)
    for t1 in range(grid_depth):
        for t2 in range(0, t1 + 1):
            mask[
                t1 * tokens_per_frame : (t1 + 1) * tokens_per_frame,
                t2 * tokens_per_frame : (t2 + 1) * tokens_per_frame,
            ] = block
    return mask


class _MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class ACRoPEAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        grid_size: int = 16,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} not divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        third = 2 * ((self.head_dim // 3) // 2)
        self.d_dim = third
        self.h_dim = third
        self.w_dim = third
        self.grid_size = grid_size

    @staticmethod
    def _separate_positions(ids: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens_per_frame = h * w
        frame_ids = ids // tokens_per_frame
        rem = ids - tokens_per_frame * frame_ids
        height_ids = rem // w
        width_ids = rem - w * height_ids
        return 1.0 * frame_ids, 1.0 * height_ids, 1.0 * width_ids

    def _compute_positions(
        self,
        t_0: int,
        h: int,
        w: int,
        action_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Compute RoPE position tensors for one cached frame step.

        Separated from forward_cached so callers can hoist this computation
        out of CUDA-graph-captured regions.  The returned tensors are ordinary
        (non-static) GPU tensors; the CUDA-graph path instead pre-allocates
        static GPU buffers and updates them with .copy_() before each replay.
        """
        spatial_ids = torch.arange(t_0 * h * w, (t_0 + 1) * h * w, device=device)
        d_pos, h_pos, w_pos = self._separate_positions(spatial_ids, h, w)
        h_pos = h_pos * (self.grid_size / h)
        w_pos = w_pos * (self.grid_size / w)
        time_pos: torch.Tensor | None = None
        if action_tokens > 0:
            time_pos = torch.full((action_tokens,), float(t_0), device=device, dtype=dtype)
        return d_pos, h_pos, w_pos, time_pos

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        t: int,
        h: int,
        w: int,
        action_tokens: int,
        t_0: int = 0,
        cache_handle: BatchedCacheManager | None = None,
        # Pre-computed position tensors for the cached path.  When provided
        # (CUDA-graph path), they are static GPU buffers already on device and
        # no torch.arange / torch.full calls happen inside the captured region.
        # When None (eager path), they are computed from t_0 here.
        d_pos: torch.Tensor | None = None,
        h_pos: torch.Tensor | None = None,
        w_pos: torch.Tensor | None = None,
        time_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cache_handle is not None:
            if d_pos is None:
                d_pos, h_pos, w_pos, time_pos = self._compute_positions(
                    t_0, h, w, action_tokens, x.device, x.dtype
                )
            return self.forward_cached(
                x=x,
                d_pos=d_pos, h_pos=h_pos, w_pos=w_pos, time_pos=time_pos,
                action_tokens=action_tokens,
                cache_handle=cache_handle,
            )
        b, n, c = x.size()

        # Position ids for the spatial part of each frame
        spatial_ids = torch.arange(t * h * w, device=x.device)
        d_pos, h_pos, w_pos = self._separate_positions(spatial_ids, h, w)
        # Upstream snaps to the RoPE grid in case inference H/W differ
        # from training; these are no-ops when grid_size matches.
        h_pos = h_pos * (self.grid_size / h)
        w_pos = w_pos * (self.grid_size / w)

        if action_tokens > 0:
            x = x.view(b, -1, action_tokens + h * w, c)  # [B, T, A+H*W, C]

            action_q, action_k, action_v = [], [], []
            for i in range(action_tokens):
                a = x[:, :, i : i + 1, :].flatten(1, 2)  # [B, T, C]
                qkv = (
                    self.qkv(a).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
                )  # [3, B, num_heads, T, head_dim]
                q, k, v = qkv[0], qkv[1], qkv[2]
                time_pos = torch.arange(t, device=x.device)
                qd = rotate_queries_or_keys(q[..., : self.d_dim], pos=time_pos)
                kd = rotate_queries_or_keys(k[..., : self.d_dim], pos=time_pos)
                qr = q[..., self.d_dim :]
                kr = k[..., self.d_dim :]
                action_q.append(torch.cat([qd, qr], dim=-1).view(b, self.num_heads, t, 1, -1))
                action_k.append(torch.cat([kd, kr], dim=-1).view(b, self.num_heads, t, 1, -1))
                action_v.append(v.view(b, self.num_heads, t, 1, -1))

            action_q = torch.cat(action_q, dim=3).flatten(2, 3)
            action_k = torch.cat(action_k, dim=3).flatten(2, 3)
            action_v = torch.cat(action_v, dim=3).flatten(2, 3)
            x = x[:, :, action_tokens:, :].flatten(1, 2)  # [B, T*H*W, C]

        # Spatial qkv + 3D RoPE
        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        s = 0
        qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], pos=d_pos)
        kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], pos=d_pos)
        s += self.d_dim
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], pos=h_pos)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], pos=h_pos)
        s += self.h_dim
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], pos=w_pos)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], pos=w_pos)
        s += self.w_dim
        if s < self.head_dim:
            qr, kr = q[..., s:], k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        if action_tokens > 0:
            # Interleave back: per frame, [A action tokens, H*W spatial tokens]
            def merge_(tx: torch.Tensor, ta: torch.Tensor) -> torch.Tensor:
                tx = tx.view(b, self.num_heads, t, h * w, -1)
                ta = ta.view(b, self.num_heads, t, action_tokens, -1)
                return torch.cat([ta, tx], dim=3).flatten(2, 3)

            q = merge_(q, action_q)
            k = merge_(k, action_k)
            v = merge_(v, action_v)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj(x)

    def forward_cached(
        self,
        x: torch.Tensor,                   # [B, L, C]
        d_pos: torch.Tensor,               # [H*W]  — depth (frame) positions for spatial tokens
        h_pos: torch.Tensor,               # [H*W]  — height positions
        w_pos: torch.Tensor,               # [H*W]  — width positions
        time_pos: torch.Tensor | None,     # [action_tokens] or None
        action_tokens: int,
        cache_handle: BatchedCacheManager,
    ) -> torch.Tensor:
        """Single-frame cached attention with pre-computed position tensors.

        All position tensors are expected to already be on the correct device.
        Callers must compute them via _compute_positions (or the model-level
        _compute_rope_positions) and may store them in static GPU buffers
        updated via .copy_() so the surrounding CUDA graph sees the new values.

        Parity with the regular forward was partially validated in
        test/modular/vjepa2/test_ac_rope_parity.py and more thoroughly in
        test/modular/vjepa2/test_ac_kv_cache_parity.py.
        """
        b, n, c = x.shape
        hd = self.head_dim

        qkv: torch.Tensor = self.qkv(x).view(b, n, 3, self.num_heads, hd)
        q, k, v = qkv.unbind(dim=2)

        if action_tokens > 0:
            q_action = q[:, :action_tokens]
            k_action = k[:, :action_tokens]
            v_action = v[:, :action_tokens]
            q_spatial = q[:, action_tokens:]
            k_spatial = k[:, action_tokens:]
            v_spatial = v[:, action_tokens:]
        else:
            q_spatial, k_spatial, v_spatial = q, k, v

        # 3-axis RoPE for spatial tokens
        s = 0
        qd = rotate_queries_or_keys_BNHD(q_spatial[..., s:s + self.d_dim], d_pos)
        kd = rotate_queries_or_keys_BNHD(k_spatial[..., s:s + self.d_dim], d_pos)
        s += self.d_dim
        qh = rotate_queries_or_keys_BNHD(q_spatial[..., s:s + self.h_dim], h_pos)
        kh = rotate_queries_or_keys_BNHD(k_spatial[..., s:s + self.h_dim], h_pos)
        s += self.h_dim
        qw = rotate_queries_or_keys_BNHD(q_spatial[..., s:s + self.w_dim], w_pos)
        kw = rotate_queries_or_keys_BNHD(k_spatial[..., s:s + self.w_dim], w_pos)
        s += self.w_dim
        if s < hd:
            q_spatial = torch.cat([qd, qh, qw, q_spatial[..., s:]], dim=-1)
            k_spatial = torch.cat([kd, kh, kw, k_spatial[..., s:]], dim=-1)
        else:
            q_spatial = torch.cat([qd, qh, qw], dim=-1)
            k_spatial = torch.cat([kd, kh, kw], dim=-1)

        # Temporal RoPE for action tokens
        if action_tokens > 0 and time_pos is not None:
            qd = rotate_queries_or_keys_BNHD(q_action[..., :self.d_dim], time_pos)
            kd = rotate_queries_or_keys_BNHD(k_action[..., :self.d_dim], time_pos)
            q_action = torch.cat([qd, q_action[..., self.d_dim:]], dim=-1)
            k_action = torch.cat([kd, k_action[..., self.d_dim:]], dim=-1)
            q = torch.cat([q_action, q_spatial], dim=1)
            k = torch.cat([k_action, k_spatial], dim=1)
            v = torch.cat([v_action, v_spatial], dim=1)
        else:
            q, k, v = q_spatial, k_spatial, v_spatial

        # [B, L, H, D] -> [B*L, H, D] for FlashInfer
        q = q.reshape(b * n, self.num_heads, hd)
        k = k.reshape(b * n, self.num_heads, hd)
        v = v.reshape(b * n, self.num_heads, hd)

        x = cache_handle.run_attention(q, k, v)
        x = x.reshape(b, n, c)
        return self.proj(x)


class ACBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        layer_norm_eps: float,
        grid_size: int,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.attn = ACRoPEAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            grid_size=grid_size,
        )
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.mlp = _MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        t: int,
        h: int,
        w: int,
        action_tokens: int,
        t_0: int = 0,
        cache_handle: BatchedCacheManager | None = None,
        d_pos: torch.Tensor | None = None,
        h_pos: torch.Tensor | None = None,
        w_pos: torch.Tensor | None = None,
        time_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x), attn_mask=attn_mask, t=t, h=h, w=w,
            action_tokens=action_tokens, t_0=t_0, cache_handle=cache_handle,
            d_pos=d_pos, h_pos=h_pos, w_pos=w_pos, time_pos=time_pos,
        )
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerPredictorAC(nn.Module):
    """Action-conditioned V-JEPA 2 predictor.

    Forward signature matches the upstream class so parity tests can pass
    outputs directly.  Expects encoder context embeddings plus per-timestep
    action / state (and optional extrinsics) tensors.
    """

    def __init__(self, config: VJepa2ACPredictorConfig):
        super().__init__()
        self.config = config
        self.is_frame_causal = config.is_frame_causal
        self.use_extrinsics = config.use_extrinsics
        self.img_height, self.img_width = config.img_size
        self.patch_size = config.patch_size
        self.num_frames = config.num_frames
        self.tubelet_size = config.tubelet_size
        self.grid_height = config.img_size[0] // config.patch_size
        self.grid_width = config.img_size[1] // config.patch_size

        # Input projections
        self.predictor_embed = nn.Linear(config.embed_dim, config.predictor_embed_dim, bias=True)
        self.action_encoder = nn.Linear(config.action_embed_dim, config.predictor_embed_dim, bias=True)
        self.state_encoder = nn.Linear(config.action_embed_dim, config.predictor_embed_dim, bias=True)
        # Extrinsics encoder uses one fewer input dim (matches upstream).
        self.extrinsics_encoder = nn.Linear(config.action_embed_dim - 1, config.predictor_embed_dim, bias=True)

        # Transformer blocks
        self.predictor_blocks = nn.ModuleList(
            [
                ACBlock(
                    dim=config.predictor_embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    layer_norm_eps=config.layer_norm_eps,
                    grid_size=self.grid_height,
                )
                for _ in range(config.depth)
            ]
        )

        self.predictor_norm = nn.LayerNorm(config.predictor_embed_dim, eps=config.layer_norm_eps)
        self.predictor_proj = nn.Linear(config.predictor_embed_dim, config.embed_dim, bias=True)

        # Causal attention mask is fully derived from config, so we cache it
        # lazily on the first forward rather than storing a buffer.  A buffer
        # would be zeroed out by ``meta_device → to_empty(device)``, which is
        # the pattern the model class uses to avoid a throwaway CPU init.
        self._attn_mask_cache: torch.Tensor | None = None

    @property
    def attn_mask(self) -> torch.Tensor | None:
        """Back-compat accessor used by tests.  Builds the mask on CPU if
        it hasn't been built yet."""
        if self._attn_mask_cache is None and self.config.is_frame_causal:
            self._attn_mask_cache = self._build_attn_mask(torch.device("cpu"))
        return self._attn_mask_cache

    def _build_attn_mask(self, device: torch.device) -> torch.Tensor:
        grid_depth = self.config.num_frames // self.config.tubelet_size
        add_tokens = 3 if self.config.use_extrinsics else 2
        return build_action_block_causal_attention_mask(
            grid_depth, self.grid_height, self.grid_width, add_tokens=add_tokens
        ).to(device)

    def _get_attn_mask(self, device: torch.device) -> torch.Tensor:
        cache = self._attn_mask_cache
        if cache is None or cache.device != device:
            cache = self._build_attn_mask(device)
            self._attn_mask_cache = cache
        return cache

    def _compute_rope_positions(
        self,
        t_0: int,
        h: int,
        w: int,
        action_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Compute RoPE position tensors for one cached rollout step.

        Delegates to ACRoPEAttention._compute_positions using the first block's
        grid_size, which is constant across all blocks for a given config.
        Called once before the block loop so the computation is hoisted out of
        any CUDA-graph-captured region.
        """
        return self.predictor_blocks[0].attn._compute_positions(
            t_0, h, w, action_tokens, device, dtype
        )

    def _prepare_sequence(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int, int, int]:
        """Embed and interleave action/state/spatial tokens.

        Returns ``(x, cond_tokens, b, t)`` where ``x`` is the full interleaved
        sequence ``[B, T*(cond_tokens + H*W), D]`` ready for the block loop.
        """
        x = self.predictor_embed(x)
        b, n_ctxt, d = x.size()
        t = n_ctxt // (self.grid_height * self.grid_width)
        s = self.state_encoder(states).unsqueeze(2)   # [B, T, 1, D]
        a = self.action_encoder(actions).unsqueeze(2)
        x = x.view(b, t, self.grid_height * self.grid_width, d)
        if self.use_extrinsics:
            if extrinsics is None:
                raise ValueError("extrinsics required when use_extrinsics=True")
            e = self.extrinsics_encoder(extrinsics).unsqueeze(2)
            x = torch.cat([a, s, e, x], dim=2).flatten(1, 2)
            cond_tokens = 3
        else:
            x = torch.cat([a, s, x], dim=2).flatten(1, 2)
            cond_tokens = 2
        return x, cond_tokens, b, t

    def _decode_sequence(
        self,
        x: torch.Tensor,
        cond_tokens: int,
        b: int,
        t: int,
    ) -> torch.Tensor:
        """Drop action/state tokens, apply norm + projection."""
        d = x.size(-1)
        x = x.view(b, t, cond_tokens + self.grid_height * self.grid_width, d)
        x = x[:, :, cond_tokens:, :].flatten(1, 2)
        x = self.predictor_norm(x)
        x = self.predictor_proj(x)
        return x

    def make_block_loop_fn(
        self,
        static_cache_manager,           # BatchedCacheManager with persistent wrappers, or None
        static_pos_bufs: dict,          # {"d_pos": Tensor, "h_pos": Tensor, ...}
        cond_tokens: int,
    ):
        """Return a closure capturing the block loop for PiecewiseCudaGraphRunner.

        The returned ``fn(x) -> x`` reads position tensors from
        ``static_pos_bufs`` (which the runner updates via ``.copy_()`` before
        each replay) and uses ``static_cache_manager`` (whose FlashInfer
        wrapper is re-planned outside the graph before each replay).

        ``advance_seq_len`` is NOT called inside this closure — the runner
        calls it after ``graph.replay()`` so it stays outside the captured
        region.
        """
        blocks = self.predictor_blocks
        gh, gw = self.grid_height, self.grid_width

        def fn(x: torch.Tensor) -> torch.Tensor:
            d_pos   = static_pos_bufs["d_pos"]
            h_pos   = static_pos_bufs["h_pos"]
            w_pos   = static_pos_bufs["w_pos"]
            time_pos = static_pos_bufs.get("time_pos")
            for blk_num, blk in enumerate(blocks):
                if static_cache_manager is not None:
                    static_cache_manager.set_layer_idx(blk_num)
                x = blk(
                    x,
                    attn_mask=None,
                    t=1,                   # always 1 frame per step in rollout
                    h=gh, w=gw,
                    action_tokens=cond_tokens,
                    cache_handle=static_cache_manager,
                    d_pos=d_pos, h_pos=h_pos, w_pos=w_pos, time_pos=time_pos,
                )
            return x

        return fn

    def forward(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        t_0: int = 0,
        cache_handle: BatchedCacheManager | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: encoder context embeddings ``[B, N_ctxt, embed_dim]``.
            actions: ``[B, T, action_embed_dim]``.
            states: ``[B, T, action_embed_dim]``.
            extrinsics: ``[B, T, action_embed_dim - 1]`` (only when
                ``use_extrinsics=True``).

        Returns:
            Predicted embeddings, ``[B, N_ctxt, embed_dim]``.
        """
        x, cond_tokens, b, t = self._prepare_sequence(x, actions, states, extrinsics)

        assert self.config.is_frame_causal, "non-causal AC predictor is not implemented"

        if cache_handle is None:
            attn_mask = self._get_attn_mask(x.device)[: x.size(1), : x.size(1)]
            d_pos = h_pos = w_pos = time_pos = None
        else:
            attn_mask = None
            # Compute positions once before the block loop so this work stays
            # outside any CUDA-graph-captured region (see PiecewiseCudaGraphRunner).
            d_pos, h_pos, w_pos, time_pos = self._compute_rope_positions(
                t_0, self.grid_height, self.grid_width, cond_tokens, x.device, x.dtype
            )

        for blk_num, blk in enumerate(self.predictor_blocks):
            if cache_handle is not None:
                cache_handle.set_layer_idx(blk_num)
            x = blk(
                x,
                attn_mask=attn_mask,
                t=t,
                h=self.grid_height,
                w=self.grid_width,
                action_tokens=cond_tokens,
                t_0=t_0,
                cache_handle=cache_handle,
                d_pos=d_pos, h_pos=h_pos, w_pos=w_pos, time_pos=time_pos,
            )

        if cache_handle is not None:
            cache_handle.advance_seq_len()

        return self._decode_sequence(x, cond_tokens, b, t)

