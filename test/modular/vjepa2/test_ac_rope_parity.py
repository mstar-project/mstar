import torch


def rotate_queries_or_keys(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings along the last ``D`` dims of ``x``.

    Args:
        x: ``[B, num_heads, N, D]`` (or broadcastable) — queries or keys.
        pos: positions of shape ``[N]`` or ``[B, num_heads, N]``.

    Returns:
        Rotated tensor, same shape as ``x``.
    """
    _, _, _, D = x.size()

    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega
    freq = pos.unsqueeze(-1) * omega

    emb_sin = freq.sin()
    emb_cos = freq.cos()

    emb_sin = emb_sin.repeat(1, 1, 1, 2)
    emb_cos = emb_cos.repeat(1, 1, 1, 2)

    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return (x * emb_cos) + (y * emb_sin)

def rotate_queries_or_keys_BNHD(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    _, _, _, D = x.size()
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega

    if pos.dim() == 1:
        freq = pos.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * omega
    else:
        freq = pos.unsqueeze(-1) * omega  # [B, N, num_heads, D//2]

    emb_sin = freq.sin().repeat(1, 1, 1, 2)  # [B, N, num_heads, D]
    emb_cos = freq.cos().repeat(1, 1, 1, 2)

    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return (x * emb_cos) + (y * emb_sin)

class DummyAttention:
    def __init__(self, device: torch.device):
        # dummy attributes to make the code run
        self.head_dim = 64
        self.num_heads = 8
        third = 2 * ((self.head_dim // 3) // 2)
        self.d_dim = third
        self.h_dim = third
        self.w_dim = third
        self.grid_size = 16

        self.qkv = torch.nn.Linear(128, 3 * self.num_heads * self.head_dim).to(device)

        # random initialization for testing
        torch.nn.init.normal_(self.qkv.weight)

    @staticmethod
    def _separate_positions(ids: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens_per_frame = h * w
        frame_ids = ids // tokens_per_frame
        rem = ids - tokens_per_frame * frame_ids
        height_ids = rem // w
        width_ids = rem - w * height_ids
        return 1.0 * frame_ids, 1.0 * height_ids, 1.0 * width_ids

    def forward_cached(
        self,
        x: torch.Tensor,          # [B, L, C]
        t_0: int,
        h: int,
        w: int,
        action_tokens: int,
    ) -> torch.Tensor:
        b, n, c = x.shape
        hd = self.head_dim

        spatial_tokens = h * w

        # ------------------------------------------------------------
        # Build spatial positions for this frame
        # ------------------------------------------------------------
        spatial_ids = torch.arange(
            t_0 * spatial_tokens, (t_0 + 1) * spatial_tokens, device=x.device,
        )

        d_pos, h_pos, w_pos = self._separate_positions(spatial_ids, h, w)

        h_pos = h_pos * (self.grid_size / h)
        w_pos = w_pos * (self.grid_size / w)

        # ------------------------------------------------------------
        # Single QKV projection
        # q/k/v -> [B, L, H, D]
        # ------------------------------------------------------------
        qkv: torch.Tensor = self.qkv(x).view(b, n, 3, self.num_heads, hd)
        q, k, v = qkv.unbind(dim=2)

        # ------------------------------------------------------------
        # Split action + spatial tokens
        # ------------------------------------------------------------
        if action_tokens > 0:
            q_action = q[:, :action_tokens]
            k_action = k[:, :action_tokens]
            v_action = v[:, :action_tokens]

            q_spatial = q[:, action_tokens:]
            k_spatial = k[:, action_tokens:]
            v_spatial = v[:, action_tokens:]
        else:
            q_spatial, k_spatial, v_spatial = q, k, v

        # ------------------------------------------------------------
        # Apply RoPE to spatial tokens
        # ------------------------------------------------------------
        s = 0
        print(q_spatial[..., s:s+self.d_dim].shape, d_pos.shape)
        qd = rotate_queries_or_keys_BNHD(q_spatial[..., s:s+self.d_dim], d_pos)
        kd = rotate_queries_or_keys_BNHD(k_spatial[..., s:s+self.d_dim], d_pos)
        s += self.d_dim

        qh = rotate_queries_or_keys_BNHD(q_spatial[..., s:s+self.h_dim], h_pos)
        kh = rotate_queries_or_keys_BNHD(k_spatial[..., s:s+self.h_dim], h_pos)
        s += self.h_dim

        qw = rotate_queries_or_keys_BNHD(q_spatial[..., s:s+self.w_dim], w_pos)
        kw = rotate_queries_or_keys_BNHD(k_spatial[..., s:s+self.w_dim], w_pos)
        s += self.w_dim

        if s < hd:
            q_spatial = torch.cat(
                [qd, qh, qw, q_spatial[..., s:]],
                dim=-1,
            )
            k_spatial = torch.cat(
                [kd, kh, kw, k_spatial[..., s:]],
                dim=-1,
            )
        else:
            q_spatial = torch.cat([qd, qh, qw], dim=-1)
            k_spatial = torch.cat([kd, kh, kw], dim=-1)

        # ------------------------------------------------------------
        # Apply temporal RoPE to action tokens (all at once)
        # ------------------------------------------------------------
        if action_tokens > 0:
            time_pos = torch.full(
                (action_tokens,),
                t_0,
                device=x.device,
                dtype=x.dtype,
            )

            qd = rotate_queries_or_keys_BNHD(q_action[..., :self.d_dim], time_pos)
            kd = rotate_queries_or_keys_BNHD(k_action[..., :self.d_dim], time_pos)

            q_action = torch.cat(
                [qd, q_action[..., self.d_dim:]],
                dim=-1,
            )
            k_action = torch.cat(
                [kd, k_action[..., self.d_dim:]],
                dim=-1,
            )

            q = torch.cat([q_action, q_spatial], dim=1)
            k = torch.cat([k_action, k_spatial], dim=1)
            v = torch.cat([v_action, v_spatial], dim=1)
        else:
            q, k, v = q_spatial, k_spatial, v_spatial

        # ------------------------------------------------------------
        # Flatten directly for cache attention
        # [B, L, H, D] -> [B*L, H, D]
        # ------------------------------------------------------------
        q = q.reshape(b * n, self.num_heads, hd)
        k = k.reshape(b * n, self.num_heads, hd)
        v = v.reshape(b * n, self.num_heads, hd)

        return q, k, v

    def forward_cached_old(
        self,
        x: torch.Tensor, # one chunk of tokens from one encoded frame
        t_0: int,
        h: int,
        w: int,
        action_tokens: int,
    ) -> torch.Tensor:
        b, n, c = x.size()

        # Position ids for the spatial part of each frame
        # We will only do this kv cache fwd function if we are running one block at a time
        # (that's what matches the forward call). We would need the attention mask if we did
        # more blocks, and that is not supported.
        spatial_ids = torch.arange(t_0 * h * w, (t_0 + 1) * h * w, device=x.device)
        d_pos, h_pos, w_pos = self._separate_positions(spatial_ids, h, w)

        # Upstream snaps to the RoPE grid in case inference H/W differ
        # from training; these are no-ops when grid_size matches.
        h_pos = h_pos * (self.grid_size / h)
        w_pos = w_pos * (self.grid_size / w)

        if action_tokens > 0:
            x = x.view(b, -1, action_tokens + h * w, c)  # [B, 1, A+H*W, C]

            action_q, action_k, action_v = [], [], []
            for i in range(action_tokens):
                a = x[:, :, i : i + 1, :].flatten(1, 2)  # [B, 1, C]
                qkv = (
                    self.qkv(a).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
                )  # [3, B, num_heads, 1, head_dim]
                q, k, v = qkv[0], qkv[1], qkv[2]
                time_pos = torch.tensor(t_0, device=x.device)
                qd = rotate_queries_or_keys(q[..., : self.d_dim], pos=time_pos)
                kd = rotate_queries_or_keys(k[..., : self.d_dim], pos=time_pos)
                qr = q[..., self.d_dim :]
                kr = k[..., self.d_dim :]
                action_q.append(torch.cat([qd, qr], dim=-1).view(b, self.num_heads, 1, 1, -1))
                action_k.append(torch.cat([kd, kr], dim=-1).view(b, self.num_heads, 1, 1, -1))
                action_v.append(v.view(b, self.num_heads, 1, 1, -1))

            action_q = torch.cat(action_q, dim=3).flatten(2, 3)
            action_k = torch.cat(action_k, dim=3).flatten(2, 3)
            action_v = torch.cat(action_v, dim=3).flatten(2, 3)
            x = x[:, :, action_tokens:, :].flatten(1, 2)  # [B, 1*H*W, C]

        # Spatial qkv + 3D RoPE
        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        s = 0
        qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], pos=d_pos)
        kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], pos=d_pos)
        print(q[..., s : s + self.d_dim].shape)
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
                tx = tx.view(b, self.num_heads, 1, h * w, -1)
                ta = ta.view(b, self.num_heads, 1, action_tokens, -1)
                return torch.cat([ta, tx], dim=3).flatten(2, 3)

            q = merge_(q, action_q)
            k = merge_(k, action_k)
            v = merge_(v, action_v)

        # qkv shape should be: B, num_heads, block_size, head_dim
        def _fix_shapes_for_attn(x):
            return x.transpose(1,2).flatten(0,1) # B*block_size, num_heads, head_dim

        return _fix_shapes_for_attn(q), \
            _fix_shapes_for_attn(k), \
            _fix_shapes_for_attn(v)

DEVICE = torch.device('cuda:3')
dummy_attn = DummyAttention(device=DEVICE)
x = torch.randn(2, 16*16+4, 128, device=DEVICE)
t0 = 5

q2, k2, v2 = dummy_attn.forward_cached_old(x, t0, h=16, w=16, action_tokens=4)
q, k, v = dummy_attn.forward_cached(x, t0, h=16, w=16, action_tokens=4)
print((q - q2).abs().max())
(print(v - v2).abs().max())
(print(k - k2).abs().max())
