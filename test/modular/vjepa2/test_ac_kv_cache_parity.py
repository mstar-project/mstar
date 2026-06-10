"""Parity test: VisionTransformerPredictorAC cached vs non-cached forward.

The KV-cached path (forward with cache_handle, one frame at a time) must produce
the same output per frame as the non-cached path (all T frames at once with a
causal block attention mask).

Why they must match:
  Non-cached: frame t_0 attends to frames 0..t_0 (blocked by the causal mask).
  Cached:     at step t_0 the cache holds K/V for frames 0..t_0 exactly, so
              full (non-causal) attention over the cache equals the masked result.

Uses a FakeCacheManager backed by F.scaled_dot_product_attention — no FlashInfer
or GPU required.
"""

from __future__ import annotations


import torch
import torch.nn.functional as F

from mminf.model.vjepa2.components.ac_predictor import VisionTransformerPredictorAC
from mminf.model.vjepa2.config import VJepa2ACPredictorConfig


# ---------------------------------------------------------------------------
# Fake cache manager
# ---------------------------------------------------------------------------

class FakeCacheManager:
    """Emulates BatchedCacheManager.run_attention with plain Python lists.

    Per layer and per request, we accumulate K/V tokens across time steps.
    This matches the real FlashInfer path where each request has its own KV
    pages and full attention is computed over that request's accumulated history.

    plan_attention must be called before each forward pass (as preprocess does
    in production) so run_attention knows how to split the flattened
    [sum(seq_lens), H, D] tensor back into per-request slices.
    """

    def __init__(self):
        self.layer_idx: int = 0
        self._seq_lens: list[int] = []
        # layer -> list of per-request (list[k], list[v])
        self._kv: dict[int, list[tuple[list[torch.Tensor], list[torch.Tensor]]]] = {}

    def set_layer_idx(self, idx: int) -> None:
        self.layer_idx = idx

    def plan_attention(self, seq_lens: list[int], is_causal: bool = False) -> None:
        """Record per-request token counts for the upcoming forward pass."""
        self._seq_lens = list(seq_lens)
        n_req = len(seq_lens)
        # Initialise KV history for any new layer that hasn't been seen yet.
        for layer, req_kvs in self._kv.items():
            if len(req_kvs) != n_req:
                self._kv[layer] = [([], []) for _ in range(n_req)]

    def advance_seq_len(self, n: int | None = None) -> None:
        pass  # seq tracking not needed; accumulated K/V lists are the history

    def run_attention(
        self,
        q: torch.Tensor,  # [sum(seq_lens), num_heads, head_dim]
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        layer = layer_idx if layer_idx is not None else self.layer_idx

        # Fall back to single-request if plan_attention wasn't called.
        seq_lens = self._seq_lens if self._seq_lens else [q.shape[0]]
        n_req = len(seq_lens)

        if layer not in self._kv:
            self._kv[layer] = [([], []) for _ in range(n_req)]

        req_kvs = self._kv[layer]
        q_splits = torch.split(q, seq_lens)
        k_splits = torch.split(k, seq_lens)
        v_splits = torch.split(v, seq_lens)

        outputs: list[torch.Tensor] = []
        for i, (q_i, k_i, v_i) in enumerate(zip(q_splits, k_splits, v_splits)):
            req_kvs[i][0].append(k_i)
            req_kvs[i][1].append(v_i)

            all_k = torch.cat(req_kvs[i][0], dim=0)  # [ctx_tokens, H, D]
            all_v = torch.cat(req_kvs[i][1], dim=0)

            # SDPA: [1, H, L, D] x [1, H, S, D]
            q_t = q_i.permute(1, 0, 2).unsqueeze(0)
            k_t = all_k.permute(1, 0, 2).unsqueeze(0)
            v_t = all_v.permute(1, 0, 2).unsqueeze(0)
            out_i = F.scaled_dot_product_attention(q_t, k_t, v_t)
            outputs.append(out_i.squeeze(0).permute(1, 0, 2))  # [L, H, D]

        return torch.cat(outputs, dim=0)  # [sum(seq_lens), H, D]


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

def _tiny_ac_config(use_extrinsics: bool = False) -> VJepa2ACPredictorConfig:
    """Small config: head_dim = 32/4 = 8, 4 frames → 2 grid-depth steps."""
    return VJepa2ACPredictorConfig(
        img_size=(16, 16),
        patch_size=4,
        num_frames=8,       # grid_depth = 8 // 2 = 4 steps
        tubelet_size=2,
        embed_dim=32,
        predictor_embed_dim=32,
        depth=2,
        num_heads=4,        # head_dim = 32 // 4 = 8
        mlp_ratio=2.0,
        qkv_bias=True,
        layer_norm_eps=1e-6,
        is_frame_causal=True,
        action_embed_dim=7,
        use_extrinsics=use_extrinsics,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKVCacheParity:
    def test_frame_by_frame_matches_full_sequence(self):
        """Core parity: cached step t_0 output == non-cached output[:, t_0*N:(t_0+1)*N].

        Both paths use the same model weights and F.scaled_dot_product_attention,
        so results should be numerically very close (atol=1e-5 on float32).
        """
        torch.manual_seed(42)
        cfg = _tiny_ac_config()
        model = VisionTransformerPredictorAC(cfg).eval()

        B = 1
        T = cfg.num_frames // cfg.tubelet_size        # 4
        N = (cfg.img_size[0] // cfg.patch_size) ** 2  # 16 spatial tokens / frame

        x_full  = torch.randn(B, T * N, cfg.embed_dim)
        actions = torch.randn(B, T, cfg.action_embed_dim)
        states  = torch.randn(B, T, cfg.action_embed_dim)

        # Non-cached: all T frames at once with causal block mask.
        with torch.no_grad():
            out_full = model(x_full, actions, states)   # [B, T*N, embed_dim]

        # Cached: one frame at a time, accumulating K/V in FakeCacheManager.
        cache = FakeCacheManager()
        with torch.no_grad():
            for t_0 in range(T):
                x_t   = x_full[:, t_0 * N : (t_0 + 1) * N, :]
                act_t = actions[:, t_0 : t_0 + 1, :]
                st_t  = states[:, t_0 : t_0 + 1, :]

                out_t = model(x_t, act_t, st_t, t_0=t_0, cache_handle=cache)
                # [B, N, embed_dim]

                expected = out_full[:, t_0 * N : (t_0 + 1) * N, :]
                torch.testing.assert_close(
                    out_t, expected, atol=1e-5, rtol=1e-5,
                    msg=f"mismatch at t_0={t_0}",
                )

    def test_parity_with_extrinsics(self):
        """Same check with use_extrinsics=True (3 cond tokens per frame)."""
        torch.manual_seed(7)
        cfg = _tiny_ac_config(use_extrinsics=True)
        model = VisionTransformerPredictorAC(cfg).eval()

        B = 1
        T = cfg.num_frames // cfg.tubelet_size
        N = (cfg.img_size[0] // cfg.patch_size) ** 2

        x_full     = torch.randn(B, T * N, cfg.embed_dim)
        actions    = torch.randn(B, T, cfg.action_embed_dim)
        states     = torch.randn(B, T, cfg.action_embed_dim)
        extrinsics = torch.randn(B, T, cfg.action_embed_dim - 1)

        with torch.no_grad():
            out_full = model(x_full, actions, states, extrinsics=extrinsics)

        cache = FakeCacheManager()
        with torch.no_grad():
            for t_0 in range(T):
                x_t   = x_full[:, t_0 * N : (t_0 + 1) * N, :]
                act_t = actions[:, t_0 : t_0 + 1, :]
                st_t  = states[:, t_0 : t_0 + 1, :]
                ext_t = extrinsics[:, t_0 : t_0 + 1, :]

                out_t = model(
                    x_t, act_t, st_t, extrinsics=ext_t,
                    t_0=t_0, cache_handle=cache,
                )

                expected = out_full[:, t_0 * N : (t_0 + 1) * N, :]
                torch.testing.assert_close(
                    out_t, expected, atol=1e-5, rtol=1e-5,
                    msg=f"mismatch at t_0={t_0}",
                )

    def test_parity_batch_size_2(self):
        """Parity holds for B=2 (two independent sequences in the same forward).

        plan_attention must be called before each frame to tell FakeCacheManager
        (and, in production, BatchedCacheManager) the per-request token count so
        run_attention can split the flattened [B*n, H, D] tensor correctly.
        """
        torch.manual_seed(99)
        cfg = _tiny_ac_config()
        model = VisionTransformerPredictorAC(cfg).eval()

        B = 2
        T = cfg.num_frames // cfg.tubelet_size
        N = (cfg.img_size[0] // cfg.patch_size) ** 2
        cond_tokens = 3 if cfg.use_extrinsics else 2
        tokens_per_req = N + cond_tokens  # tokens per request per frame

        x_full  = torch.randn(B, T * N, cfg.embed_dim)
        actions = torch.randn(B, T, cfg.action_embed_dim)
        states  = torch.randn(B, T, cfg.action_embed_dim)

        with torch.no_grad():
            out_full = model(x_full, actions, states)

        cache = FakeCacheManager()
        with torch.no_grad():
            for t_0 in range(T):
                x_t   = x_full[:, t_0 * N : (t_0 + 1) * N, :]
                act_t = actions[:, t_0 : t_0 + 1, :]
                st_t  = states[:, t_0 : t_0 + 1, :]

                # Mirror what preprocess does in production.
                cache.plan_attention(seq_lens=[tokens_per_req] * B, is_causal=False)
                out_t = model(x_t, act_t, st_t, t_0=t_0, cache_handle=cache)

                expected = out_full[:, t_0 * N : (t_0 + 1) * N, :]
                torch.testing.assert_close(
                    out_t, expected, atol=1e-5, rtol=1e-5,
                    msg=f"mismatch at t_0={t_0}",
                )

    def test_output_shape_cached(self):
        """Cached forward returns [B, N, embed_dim] per step (spatial tokens only)."""
        torch.manual_seed(0)
        cfg = _tiny_ac_config()
        model = VisionTransformerPredictorAC(cfg).eval()

        B, N = 1, (cfg.img_size[0] // cfg.patch_size) ** 2
        x   = torch.randn(B, N, cfg.embed_dim)
        act = torch.randn(B, 1, cfg.action_embed_dim)
        st  = torch.randn(B, 1, cfg.action_embed_dim)

        cache = FakeCacheManager()
        with torch.no_grad():
            out = model(x, act, st, t_0=0, cache_handle=cache)

        assert out.shape == (B, N, cfg.embed_dim)

    def test_cached_is_causal_no_future_leakage(self):
        """Perturbing frame k must not change the cached output at frame k-1.

        Reuses the causality invariant from test_ac_predictor_parity.py but
        exercises the cached code path: the K/V for frame k-1 are already
        committed to the cache before frame k is processed, so frame k-1's
        output cannot change.
        """
        torch.manual_seed(3)
        cfg = _tiny_ac_config()
        model = VisionTransformerPredictorAC(cfg).eval()

        B = 1
        T = cfg.num_frames // cfg.tubelet_size
        N = (cfg.img_size[0] // cfg.patch_size) ** 2

        x_base   = torch.randn(B, T * N, cfg.embed_dim)
        act_base = torch.randn(B, T, cfg.action_embed_dim)
        st_base  = torch.randn(B, T, cfg.action_embed_dim)

        x_pert   = x_base.clone()
        act_pert = act_base.clone()
        st_pert  = st_base.clone()
        # Perturb only the last frame.
        x_pert[:, -N:, :]   += 100.0
        act_pert[:, -1, :]  += 100.0
        st_pert[:, -1, :]   += 100.0

        def _run_cached(x_seq, actions, states):
            cache = FakeCacheManager()
            outs = []
            with torch.no_grad():
                for t_0 in range(T):
                    x_t   = x_seq[:, t_0 * N : (t_0 + 1) * N, :]
                    act_t = actions[:, t_0 : t_0 + 1, :]
                    st_t  = states[:, t_0 : t_0 + 1, :]
                    outs.append(model(x_t, act_t, st_t, t_0=t_0, cache_handle=cache))
            return outs

        base_outs = _run_cached(x_base, act_base, st_base)
        pert_outs = _run_cached(x_pert, act_pert, st_pert)

        # All frames except the last must be unaffected.
        for t_0 in range(T - 1):
            torch.testing.assert_close(
                base_outs[t_0], pert_outs[t_0], atol=0.0, rtol=0.0,
                msg=f"future-leakage at t_0={t_0}",
            )

        # Last frame output must differ.
        assert not torch.allclose(base_outs[-1], pert_outs[-1], atol=1e-3), \
            "last-frame perturbation had no effect"
