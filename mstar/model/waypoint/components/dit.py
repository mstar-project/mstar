"""The Waypoint-1.5 DiT: 24 blocks, the 4+1 pass driver, and the world clock.

Five forwards per generated frame: four frozen Euler denoise passes over
``config.scheduler_sigmas``, then one committing pass at sigma=0. Only the last
writes the ring. The loop is plain Python here, not engine steps.

Two clocks, threaded separately: ``f_pos`` drives buckets, slots and
visibility, ``t_pos = f_pos * config.ts_mult`` is the RoPE time coordinate.
They are numerically equal at this checkpoint's ``ts_mult == 1``; conflating
them is silent drift at any other serving fps.

``cond_proj`` is physically shared by all 24 blocks, and ``retie_cond_proj()``
is public because ``to_empty(device)`` silently un-ties it. Controller
conditioning is fused on 8 of 24 layers (``i % 3 == 0``); the other 16 carry no
``ctrl_mlpfusion`` submodule at all.

The port collapses the reference's ``WorldModel``/``WorldDiT`` pair into one
module, so blocks live at ``blocks.{i}``, not ``transformer.blocks.{i}``.
Everything below that prefix keeps the reference's spelling.
"""

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mstar.model.waypoint.components.attention import WaypointAttention
from mstar.model.waypoint.components.layers import (
    FP32_MODULE_PATHS,
    MLP,
    AdaLN,
    CondHead,
    ControllerInputEmbedding,
    DeviceTableCache,
    MLPFusion,
    NoiseConditioner,
    ada_gate,
    ada_rmsnorm,
    rms_norm,
)
from mstar.model.waypoint.components.rope import OrthoRoPEAngles
from mstar.model.waypoint.config import WaypointConfig

__all__ = ["WaypointDiT", "WaypointDiTBlock", "WaypointPosIds"]


class WaypointPosIds(NamedTuple):
    """The four position streams for one frame.

    ``f_pos`` is the ``[]`` int64 ring clock (the KV resource's ``upsert`` takes
    the scalar); ``t_pos`` is the ``[B, T]`` RoPE time coordinate
    ``f_pos * ts_mult``; ``y_pos``/``x_pos`` are ``[B, T]`` token-grid
    coordinates, ``row = i // width`` and ``col = i % width``.
    """

    f_pos: Tensor
    t_pos: Tensor
    y_pos: Tensor
    x_pos: Tensor


class WaypointDiTBlock(nn.Module):
    """One DiT block: adaLN-modulated causal frame attention, optional controller
    fusion, adaLN-modulated MLP.

    The six modulation tensors come from this block's ``cond_head``, whose
    ``bias_in`` is per-layer and whose ``cond_proj`` matrices alias block 0's.
    """

    def __init__(self, config: WaypointConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attn = WaypointAttention(config, layer_idx)
        self.mlp = MLP(config.d_model, config.d_model * config.mlp_ratio, config.d_model)
        self.cond_head = CondHead(config)

        # Absent, not None-gated, on the other 16 layers: the parameter tree
        # itself records which layers fuse.
        self.ctrl_mlpfusion = MLPFusion(config) if layer_idx in config.ctrl_layers else None

    def forward(
        self,
        x: Tensor,
        frame_pos: Tensor,
        rope_angles: tuple[Tensor, Tensor],
        cond: Tensor,
        ctrl_emb: Tensor,
        v1: Tensor | None,
        *,
        commit: bool,
    ) -> tuple[Tensor, Tensor]:
        """``x`` ``[B, N*T, D]``, ``cond``/``ctrl_emb`` ``[B, N, D]`` (per frame)
        -> ``(x, v1)``. ``v1`` is layer 0's pre-lerp V, threaded down the stack.

        Only ``f_pos`` reaches this far -- the RoPE angles are built once at the
        root -- so the scalar ring clock is passed rather than the whole bundle.
        """
        s0, b0, g0, s1, b1, g1 = self.cond_head(cond)

        residual = x
        x = ada_rmsnorm(x, s0, b0)
        x, v1 = self.attn(x, frame_pos, rope_angles, v1, commit=commit)
        x = ada_gate(x, g0) + residual

        # Both operands are bare-RMS-normed (no adaLN scale here) and the fusion
        # output is added ungated.
        if self.ctrl_mlpfusion is not None:
            x = self.ctrl_mlpfusion(rms_norm(x), rms_norm(ctrl_emb)) + x

        x = ada_gate(self.mlp(ada_rmsnorm(x, s1, b1)), g1) + x

        return x, v1


class WaypointDiT(nn.Module):
    """Conv2d patchify -> 24 blocks -> adaLN head -> unpatchify, plus the 4+1
    per-frame driver.

    Built on the meta device and materialized by ``weight_loader`` in a fixed
    order::

        with torch.device("meta"):
            dit = WaypointDiT(config)
        dit.cast_serving_dtypes()   # bf16 + fp32 islands, on meta
        dit.to_empty(device=device) # un-ties cond_proj
        dit.retie_cond_proj()       # MUST follow to_empty
        load_weights_into(dit, ...)

    The KV ring is NOT part of this module: it is derived state owned by an
    engine resource bound after materialization, so no ``to_empty`` or
    ``state_dict`` walk can leave it holding garbage.
    """

    def __init__(self, config: WaypointConfig):
        super().__init__()
        self.config = config
        self.patch = tuple(config.patch)

        self.denoise_step_emb = NoiseConditioner(
            config.d_model,
            reference_compat=config.reference_compat,
            cached_sigmas=config.scheduler_sigmas,
        )
        self.ctrl_emb = ControllerInputEmbedding(config)
        self.rope_angles = OrthoRoPEAngles(config)
        self.blocks = nn.ModuleList(
            WaypointDiTBlock(config, layer_idx) for layer_idx in range(config.n_layers)
        )

        C, D = config.channels, config.d_model
        ph, pw = self.patch
        self.patchify = nn.Conv2d(C, D, kernel_size=self.patch, stride=self.patch, bias=False)
        self.out_norm = AdaLN(D)
        # The checkpoint stores this as a [D, C, ph, pw] conv kernel; the loader
        # permutes it and expands the [C] bias over the patch (transforms 1-2).
        self.unpatchify = nn.Linear(D, C * ph * pw, bias=True)

        # Token-grid coordinates: derived state, held outside the module tree so
        # `to_empty` cannot leave it uninitialized. device="cpu" is required --
        # this runs under `with torch.device("meta")`.
        idx = torch.arange(config.tokens_per_frame, dtype=torch.long, device="cpu")
        self._grid = DeviceTableCache(
            idx.div(config.width, rounding_mode="floor"), idx.remainder(config.width)
        )
        # Per (device, dtype) sigma schedule; see _sigma_schedule.
        self._sigma_cache: dict[tuple[torch.device, torch.dtype], Tensor] = {}
        self._regions_compiled = False

        # Tie now so a plain (non-meta) build is already correct; retie after
        # to_empty for the meta build path.
        self.retie_cond_proj()

    # No `bind_resources` here: the driver holds no resource handle, it passes
    # `commit` down to the layers, which own the only calls into the ring.

    # ---- Build-time surface ------------------------------------------------

    @property
    def dtype(self) -> torch.dtype:
        """Bulk compute dtype (the non-island weights); callers cast latents to
        this."""
        return self.patchify.weight.dtype

    def cast_serving_dtypes(self) -> "WaypointDiT":
        """bf16 everywhere, then the fp32 islands back to fp32.

        Called on the **meta** module before ``to_empty(device)`` so storage is
        allocated directly in the serving dtype. ``.to(dtype)`` on meta
        preserves the ``cond_proj`` aliasing (``to_empty`` does not), so no
        retie is needed here.
        """
        self.to(torch.bfloat16)
        for path in FP32_MODULE_PATHS:
            self.get_submodule(path).to(torch.float32)
        return self

    def retie_cond_proj(self) -> "WaypointDiT":
        """Alias blocks 1..23's six ``cond_proj`` matrices onto block 0's.

        Public, and separate from ``__init__``, because ``to_empty(device)``
        destroys the tying: ``Module._apply`` allocates per parameter with no
        cross-module memo, so the 24 blocks come out holding 24 independent
        copies. Nothing raises; the symptoms are +0.6B resident parameters and
        23 unfilled ``cond_proj`` sets.

        ``bias_in`` is deliberately NOT tied -- it is genuinely per-layer.
        """
        ref_proj = self.blocks[0].cond_head.cond_proj
        for block in self.blocks[1:]:
            for blk_mod, ref_mod in zip(block.cond_head.cond_proj, ref_proj, strict=True):
                blk_mod.weight = ref_mod.weight
        return self

    def compile_regions(self, **compile_kwargs) -> "WaypointDiT":
        """Compile the denoise and cache passes, one region each. Idempotent.

        ``materialize_runtime_tables`` must run first: the derived token grid,
        RoPE tables and conditioner LUT live outside the module tree, and a first
        touch inside a ``fullgraph=True`` region is at best a specialization and
        at worst a graph break.
        """
        if not self._regions_compiled:
            options = {"fullgraph": True, "dynamic": False, **compile_kwargs}
            self._denoise_pass = torch.compile(self._denoise_pass, **options)
            self._cache_pass = torch.compile(self._cache_pass, **options)
            self._regions_compiled = True
        return self

    def materialize_runtime_tables(self, device: torch.device | str) -> "WaypointDiT":
        """Create every derived table after weight loading and before compile."""
        device = torch.device(device)
        self._grid.get(device)
        self.rope_angles.materialize(device)
        self.denoise_step_emb.materialize(device)
        self._sigma_schedule(device, self.dtype)
        return self

    # ---- Positions ---------------------------------------------------------

    def _pos_ids(self, frame_pos: Tensor) -> WaypointPosIds:
        """Build one frame's position streams from the ring clock."""
        if not torch.compiler.is_compiling():
            torch._check(
                frame_pos.ndim == 0 and frame_pos.dtype == torch.int64,
                lambda: f"frame_pos must be a [] int64 tensor; got {tuple(frame_pos.shape)} "
                f"{frame_pos.dtype}",
            )
        y_pos, x_pos = self._grid.get(frame_pos.device)
        # A no-op at ts_mult == 1, kept because it is the only place the two
        # clocks are related; deleting it drifts silently at another fps.
        t_pos = (frame_pos * self.config.ts_mult).reshape(1, 1).expand(1, y_pos.numel())
        return WaypointPosIds(f_pos=frame_pos, t_pos=t_pos, y_pos=y_pos[None], x_pos=x_pos[None])

    # ---- One forward -------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        sigma: Tensor,
        frame_pos: Tensor,
        *,
        mouse: Tensor,
        button: Tensor,
        scroll: Tensor,
        commit: bool,
    ) -> Tensor:
        """One pass over one latent frame; returns the rectified-flow velocity.

        ``x`` ``[B, N, C, H, W]`` latent (B == N == 1), ``sigma`` ``[B, N]``,
        ``frame_pos`` ``[]`` int64 ring clock, controller inputs ``[B, N, 2]`` /
        ``[B, N, n_buttons]`` / ``[B, N, 1]``. Returns ``[B, N, C, H, W]``.

        ``commit`` says whether this pass keeps its K/V: False for the four
        denoise passes, True for the fifth. An argument rather than resource
        state -- all five passes sit inside one engine step.
        """
        B, N, C, H, W = x.shape
        ph, pw = self.patch
        if H % ph or W % pw:
            raise ValueError(f"latent {H}x{W} is not divisible by patch {self.patch}.")
        Hp, Wp = H // ph, W // pw
        torch._assert(
            Hp * Wp == self.config.tokens_per_frame,
            f"{Hp} * {Wp} != {self.config.tokens_per_frame}",
        )
        # One frame per call, batch 1: the ring cache indexes a single frame per
        # upsert and the whole driver is built on that.
        torch._assert(B == 1 and N == 1, "WaypointDiT.forward supports B == 1, N == 1")

        pos_ids = self._pos_ids(frame_pos)
        # Keyword arguments on purpose: a silent x/y swap on a non-square grid
        # is a wrong-video-no-error bug.
        rope_angles = self.rope_angles(
            x_pos=pos_ids.x_pos, y_pos=pos_ids.y_pos, t_pos=pos_ids.t_pos
        )

        cond = self.denoise_step_emb(sigma)  # [B, N, D], fp32 island
        # Positional, and in this order — see ControllerInputEmbedding.
        ctrl_emb = self.ctrl_emb(mouse, button, scroll)  # [B, N, D]

        D = self.config.d_model
        h = self.patchify(x.reshape(B * N, C, H, W))  # [B*N, D, Hp, Wp]
        h = h.view(B, N, D, Hp * Wp).transpose(2, 3).flatten(1, 2)  # [B, N*T, D]

        v1 = None  # layer 0's pre-lerp V, threaded through all 24 blocks
        for block in self.blocks:
            h, v1 = block(
                h, pos_ids.f_pos, rope_angles, cond, ctrl_emb, v1, commit=commit
            )

        # silu sits BETWEEN the adaLN norm and the unpatchify projection
        # (reference world_model.py:348-352), not after it.
        h = F.silu(self.out_norm(h, cond))
        h = self.unpatchify(h)  # [B, N*T, C*ph*pw]
        h = h.view(B, N, Hp, Wp, C, ph, pw).permute(0, 1, 4, 2, 5, 3, 6)
        return h.reshape(B, N, C, Hp * ph, Wp * pw)

    # ---- The 4+1 driver ----------------------------------------------------

    def _sigma_schedule(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """The sigma table, memoized per (device, dtype) and resolved by the
        caller *outside* the compiled region -- materializing a tensor from a
        Python list inside ``fullgraph=True`` is a graph break.

        The dtype is load-bearing: the reference builds this table in the serving
        dtype and takes ``.diff()`` there, so the Euler step sizes are bf16
        differences of bf16 sigmas. fp32 changes two of the four steps.
        """
        key = (device, dtype)
        schedule = self._sigma_cache.get(key)
        if schedule is None:
            schedule = torch.tensor(self.config.scheduler_sigmas, dtype=dtype, device=device)
            self._sigma_cache[key] = schedule
        return schedule

    def _denoise_pass(
        self,
        x: Tensor,
        frame_pos: Tensor,
        sigmas: Tensor,
        *,
        mouse: Tensor,
        button: Tensor,
        scroll: Tensor,
    ) -> Tensor:
        """Four frozen Euler steps of the rectified-flow ODE. Returns the settled
        latent; **does not write the ring**.

        ``commit=False`` matters: each step attends to a different noisy version
        of the same frame, so none of them may keep its K/V. They still see
        themselves, through the unconditional scratch write at the ring tail.
        """
        # One reused sigma buffer, filled per step; a fresh allocation per step
        # would defeat cudagraph capture.
        sigma = x.new_empty((x.size(0), x.size(1)))
        # 5 sigmas, 4 diffs: the trailing 0.0 exists only to produce the last
        # step size. Sliced, not zipped ragged -- dynamo rejects a ragged zip
        # under fullgraph.
        for step_sigma, step_dsigma in zip(sigmas[:-1], sigmas.diff(), strict=True):
            v = self(
                x,
                sigma.fill_(step_sigma),
                frame_pos,
                mouse=mouse,
                button=button,
                scroll=scroll,
                commit=False,
            )
            # fp32 accumulate, back to the latent dtype: the add in bf16 loses
            # the small late steps.
            x = (x.float() + step_dsigma.float() * v.float()).type_as(x)
        return x

    def _cache_pass(
        self,
        x: Tensor,
        frame_pos: Tensor,
        *,
        mouse: Tensor,
        button: Tensor,
        scroll: Tensor,
    ) -> None:
        """The committing pass: one ``commit=True`` forward at sigma=0 on the
        settled latent, for the side effect alone -- the K/V it writes into every
        layer's ring. The returned velocity is discarded.
        """
        self(
            x,
            x.new_zeros((x.size(0), x.size(1))),
            frame_pos,
            mouse=mouse,
            button=button,
            scroll=scroll,
            commit=True,
        )

    def generate_frame(
        self,
        noise: Tensor,
        frame_pos: Tensor,
        *,
        mouse: Tensor,
        button: Tensor,
        scroll: Tensor,
    ) -> Tensor:
        """Denoise one frame from ``noise`` ``[B, N, C, H, W]`` and commit it.

        Five forwards: 4 frozen + 1 committing, all sharing one ``frame_pos``.
        The caller owns the ring clock and advances it once per committed frame.
        """
        # The .clone() is load-bearing, and must stay OUTSIDE the compiled
        # region. Both passes run inside ONE CUDA-graph capture, so the cache
        # pass allocates from the graph's private pool -- where the denoise
        # pass's output buffer is a free block. The cache pass's first
        # allocation can land on it and stomp the latent it is reading, at an
        # address baked into the graph and repeated on every replay.
        sigmas = self._sigma_schedule(noise.device, noise.dtype)
        x0 = self._denoise_pass(
            noise, frame_pos, sigmas, mouse=mouse, button=button, scroll=scroll
        ).clone()
        self._cache_pass(x0, frame_pos, mouse=mouse, button=button, scroll=scroll)
        return x0

    def append_frame(
        self,
        latent: Tensor,
        frame_pos: Tensor,
        *,
        mouse: Tensor,
        button: Tensor,
        scroll: Tensor,
    ) -> Tensor:
        """Prime the world state from a real (VAE-encoded) frame: the committing
        pass alone, one forward.

        ``latent`` is already the settled x0, so there is nothing to solve and
        nothing to clone. Returned unchanged for the caller to decode.
        """
        self._cache_pass(latent, frame_pos, mouse=mouse, button=button, scroll=scroll)
        return latent
