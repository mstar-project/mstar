"""The Waypoint-1.5 DiT: 24 blocks, the 4+1 pass driver, and the world clock.

Port of ``world_engine/src/model/world_model.py`` (``WorldDiTBlock``,
``WorldDiT``, ``WorldModel``) plus the per-frame driver from
``world_engine/src/world_engine.py`` (``gen_frame`` / ``append_frame`` /
``_denoise_pass`` / ``_cache_pass``). Read ``docs/waypoint/CONTRACTS.md``
sections 1, 4.4 and 4.6 alongside it.

Facts that are load-bearing:

  * **Five forwards per generated frame.** Four frozen Euler denoise passes over
    ``config.scheduler_sigmas``, then one unfrozen committing pass at sigma=0.
    Only the last writes the ring. The loop is plain Python inside this module,
    not engine steps (DECISIONS D4): with a model-owned cache the engine has
    nothing to do between denoise passes.
  * **The ``.clone()`` between denoise and commit is load-bearing**, not
    defensive copying. See ``generate_frame``.
  * **Two clocks.** ``f_pos`` (ring clock: buckets, slots, visibility) and
    ``t_pos = f_pos * config.ts_mult`` (RoPE time coordinate). ``ts_mult == 1``
    for this checkpoint so they are numerically equal, and they are still
    threaded separately -- conflating them is silent drift at any other serving
    fps (CONTRACTS section 4.4).
  * **``cond_proj`` is physically shared by all 24 blocks.** ``__init__`` ties
    it, and ``retie_cond_proj()`` exists as a public method because
    ``to_empty(device)`` silently un-ties it (``Module._apply`` has no
    cross-module memo). The loader MUST call it after ``to_empty`` or it gets
    +0.6B resident parameters and 23 blocks of ``cond_proj`` nobody fills, with
    no error (CONTRACTS section 6.1).
  * **Controller conditioning is fused on 8 of 24 layers** (``i % 3 == 0``), on
    ``rms_norm(x)`` and ``rms_norm(ctrl_emb)``. The other 16 blocks have no
    ``ctrl_mlpfusion`` submodule at all.
  * **No prompt cross-attention.** ``WaypointConfig.__post_init__`` rejects
    ``prompt_conditioning``, so there is no dead branch here to mislead a reader
    into thinking the checkpoint has one.

**Module-tree deviation, for the weight loader.** The reference splits this into
``WorldModel`` (embeddings, patchify, head) wrapping ``WorldDiT``
(``transformer.blocks``); the port collapses them into one ``WaypointDiT``, so
blocks live at ``blocks.{i}`` and not ``transformer.blocks.{i}``. That is one
extra prefix rename on top of CONTRACTS section 6's seven transforms
(``transformer.blocks.`` -> ``blocks.``); everything below that prefix keeps the
reference's spelling exactly. ``layers.FP32_MODULE_PATHS`` already assumes the
collapse (``denoise_step_emb`` is named relative to this root). See
DECISIONS D12.
"""

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mstar.model.waypoint.components.attention import WaypointAttention
from mstar.model.waypoint.components.kv_backend import WaypointKVBackend
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

    The reference packs these into a ``TensorDict`` keyed ``f_pos``/``t_pos``/
    ``y_pos``/``x_pos``; a ``NamedTuple`` says the same thing without pulling
    ``tensordict`` into mstar's dependency set, and it is a pytree so it survives
    ``torch.compile`` unchanged.

      * ``f_pos`` -- ``[]`` int64, the ring clock. The reference broadcasts it to
        ``[B, T]`` only because ``TensorDict`` demands a uniform batch shape; the
        cache reads ``f_pos[0, 0]`` and nothing else ever looks at it. The port
        keeps it a scalar, which is exactly what ``WaypointKVBackend.upsert``
        takes.
      * ``t_pos`` -- ``[B, T]`` int64, the RoPE time coordinate,
        ``f_pos * ts_mult``. Equal to ``f_pos`` for this checkpoint; see the
        module docstring.
      * ``y_pos`` / ``x_pos`` -- ``[B, T]`` int64 token-grid coordinates,
        ``row = i // width``, ``col = i % width``.
    """

    f_pos: Tensor
    t_pos: Tensor
    y_pos: Tensor
    x_pos: Tensor


class WaypointDiTBlock(nn.Module):
    """One DiT block: adaLN-modulated causal frame attention, optional controller
    fusion, adaLN-modulated MLP.

    The six modulation tensors come from this block's ``cond_head``; its
    ``bias_in`` is genuinely per-layer while its ``cond_proj`` matrices are
    aliases of block 0's (see ``WaypointDiT.retie_cond_proj``).
    """

    def __init__(self, config: WaypointConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attn = WaypointAttention(config, layer_idx)
        self.mlp = MLP(config.d_model, config.d_model * config.mlp_ratio, config.d_model)
        self.cond_head = CondHead(config)

        # 8 of 24 layers. Absent -- not None-gated at the tensor level -- on the
        # other 16, so the parameter tree itself records which layers fuse.
        self.ctrl_mlpfusion = MLPFusion(config) if layer_idx in config.ctrl_layers else None

    def forward(
        self,
        x: Tensor,
        frame_pos: Tensor,
        rope_angles: tuple[Tensor, Tensor],
        cond: Tensor,
        ctrl_emb: Tensor,
        v1: Tensor | None,
        backend: WaypointKVBackend,
    ) -> tuple[Tensor, Tensor]:
        """``x`` ``[B, N*T, D]``, ``cond``/``ctrl_emb`` ``[B, N, D]`` (per frame)
        -> ``(x, v1)``. ``v1`` is layer 0's pre-lerp V, threaded down the stack.

        Only ``f_pos`` of the reference's ``pos_ids`` reaches this far -- the
        RoPE angles are built once at the root from ``t/y/x_pos`` -- so the
        scalar ring clock is passed directly rather than the whole bundle.
        """
        s0, b0, g0, s1, b1, g1 = self.cond_head(cond)

        # Causal frame attention.
        residual = x
        x = ada_rmsnorm(x, s0, b0)
        x, v1 = self.attn(x, frame_pos, rope_angles, v1, backend)
        x = ada_gate(x, g0) + residual

        # Controller conditioning. Both operands are bare-RMS-normed (no adaLN
        # scale here); the fusion output is added ungated.
        if self.ctrl_mlpfusion is not None:
            x = self.ctrl_mlpfusion(rms_norm(x), rms_norm(ctrl_emb)) + x

        # MLP.
        x = ada_gate(self.mlp(ada_rmsnorm(x, s1, b1)), g1) + x

        return x, v1


class WaypointDiT(nn.Module):
    """Conv2d patchify -> 24 blocks -> adaLN head -> unpatchify, plus the 4+1
    per-frame driver.

    Built on the meta device and materialized by ``weight_loader`` in a fixed
    order (CONTRACTS section 6.1)::

        with torch.device("meta"):
            dit = WaypointDiT(config)
        dit.cast_serving_dtypes()   # bf16 + fp32 islands, on meta
        dit.to_empty(device=device) # un-ties cond_proj
        dit.retie_cond_proj()       # MUST follow to_empty
        load_weights_into(dit, ...)

    The KV ring is NOT part of this module: it is derived state owned by a
    ``WaypointKVBackend`` built after materialization, so no ``to_empty`` or
    ``state_dict`` walk can leave it holding garbage (DECISIONS D1/D3).
    """

    def __init__(self, config: WaypointConfig):
        super().__init__()
        self.config = config
        self.patch = tuple(config.patch)

        self.denoise_step_emb = NoiseConditioner(config.d_model)
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
        # permutes it into this Linear's [C*ph*pw, D] and expands the [C] bias
        # over the patch (CONTRACTS section 6, transforms 1-2).
        self.unpatchify = nn.Linear(D, C * ph * pw, bias=True)

        # Token-grid coordinates: derived state, so they live outside the module
        # tree (a non-persistent buffer would survive `to_empty` as
        # uninitialized garbage that no completeness check covers -- the stance
        # layers.DeviceTableCache exists for). device="cpu" is required: this
        # runs under `with torch.device("meta")`.
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

    # ---- Build-time surface ------------------------------------------------

    @property
    def dtype(self) -> torch.dtype:
        """Bulk compute dtype (the non-island weights); callers cast latents to
        this, mirroring ``Wan22DiT.dtype``."""
        return self.patchify.weight.dtype

    def cast_serving_dtypes(self) -> "WaypointDiT":
        """bf16 everywhere, then the fp32 islands back to fp32.

        Called on the **meta** module before ``to_empty(device)`` so storage is
        allocated directly in the serving dtype. ``.to(dtype)`` on meta
        preserves the ``cond_proj`` aliasing (``to_empty`` does not), so no
        retie is needed here.

        The island list is ``layers.FP32_MODULE_PATHS`` rather than a literal:
        the modules that must stay fp32 are a fact of the reference's
        ``NoCastModule`` set, and it belongs next to the modules themselves
        (DECISIONS D7).
        """
        self.to(torch.bfloat16)
        for path in FP32_MODULE_PATHS:
            self.get_submodule(path).to(torch.float32)
        return self

    def retie_cond_proj(self) -> "WaypointDiT":
        """Alias blocks 1..23's six ``cond_proj`` matrices onto block 0's.

        **Public, and separate from ``__init__``, because ``to_empty(device)``
        destroys the tying.** ``Module._apply`` allocates per parameter object
        with no cross-module memo, so the 24 blocks come out of ``to_empty``
        holding 24 independent copies. Nothing raises; the symptoms are +0.6B
        resident parameters and 23 unfilled ``cond_proj`` sets. Once re-tied,
        ``named_parameters()`` deduplicates and the loader's completeness check
        sees block 0's set only (CONTRACTS section 6.1).

        ``bias_in`` is deliberately NOT tied -- it is genuinely per-layer, and
        the checkpoint has 24 distinct values for it.
        """
        ref_proj = self.blocks[0].cond_head.cond_proj
        for block in self.blocks[1:]:
            for blk_mod, ref_mod in zip(block.cond_head.cond_proj, ref_proj, strict=True):
                blk_mod.weight = ref_mod.weight
        return self

    def compile_regions(self, **compile_kwargs) -> "WaypointDiT":
        """Compile the denoise and cache passes, one region each, matching the
        reference's two ``@torch.compile(fullgraph=True, dynamic=False)`` sites.

        Not done in ``__init__``: the serving layer decides, from
        ``config.compile_dit``, whether to compile at all (the eager path is the
        bit-exact reference and the parity harness wants it). Idempotent.

        Run one eager frame first. The derived tables (this module's token grid,
        ``OrthoRoPEAngles``' frequency tables) are copied to the device on first
        use by design -- they are deliberately outside the module tree, so
        nothing else materializes them -- and a first touch inside a
        ``fullgraph=True`` region is at best a specialization and at worst a
        graph break.
        """
        if not self._regions_compiled:
            options = {"fullgraph": True, "dynamic": False, **compile_kwargs}
            self._denoise_pass = torch.compile(self._denoise_pass, **options)
            self._cache_pass = torch.compile(self._cache_pass, **options)
            self._regions_compiled = True
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
        # The multiply is a no-op at ts_mult == 1 and is kept anyway: it is the
        # only place the two clocks are related, and deleting it is how a
        # different-fps checkpoint would start drifting silently.
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
        backend: WaypointKVBackend,
    ) -> Tensor:
        """One pass over one latent frame; returns the rectified-flow velocity.

        ``x`` ``[B, N, C, H, W]`` latent (B == N == 1), ``sigma`` ``[B, N]``,
        ``frame_pos`` ``[]`` int64 ring clock, controller inputs ``[B, N, 2]`` /
        ``[B, N, n_buttons]`` / ``[B, N, 1]``. Returns ``[B, N, C, H, W]``.

        Whether this pass commits to the ring is the *backend's* state
        (``set_frozen``), not an argument here -- exactly as in the reference,
        where the two compiled driver regions set it and the model is unaware.
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
        # upsert and the whole driver is built on that (reference asserts the
        # same thing).
        torch._assert(B == 1 and N == 1, "WaypointDiT.forward supports B == 1, N == 1")

        pos_ids = self._pos_ids(frame_pos)
        # Keyword arguments on purpose: (x, y, t) here vs the reference's
        # dict lookup, and a silent x/y swap on a non-square grid is a
        # wrong-video-no-error bug.
        rope_angles = self.rope_angles(
            x_pos=pos_ids.x_pos, y_pos=pos_ids.y_pos, t_pos=pos_ids.t_pos
        )

        cond = self.denoise_step_emb(sigma)  # [B, N, D], fp32 island
        # Positional and in this order: (mouse, button, scroll) is a checkpoint
        # fact whose widths sum correctly under any permutation.
        ctrl_emb = self.ctrl_emb(mouse, button, scroll)  # [B, N, D]

        D = self.config.d_model
        h = self.patchify(x.reshape(B * N, C, H, W))  # [B*N, D, Hp, Wp]
        h = h.view(B, N, D, Hp * Wp).transpose(2, 3).flatten(1, 2)  # [B, N*T, D]

        v1 = None  # layer 0's pre-lerp V, threaded through all 24 blocks
        for block in self.blocks:
            h, v1 = block(h, pos_ids.f_pos, rope_angles, cond, ctrl_emb, v1, backend)

        # Output head: silu sits BETWEEN the adaLN norm and the unpatchify
        # projection (reference world_model.py:348-352), not after it.
        h = F.silu(self.out_norm(h, cond))
        h = self.unpatchify(h)  # [B, N*T, C*ph*pw]
        h = h.view(B, N, Hp, Wp, C, ph, pw).permute(0, 1, 4, 2, 5, 3, 6)
        return h.reshape(B, N, C, Hp * ph, Wp * pw)

    # ---- The 4+1 driver ----------------------------------------------------

    def _sigma_schedule(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """The sigma table, memoized per (device, dtype). Resolved by the caller
        *outside* the compiled region -- materializing a tensor from a Python
        list inside ``fullgraph=True`` is a graph break, which is also why the
        reference keeps this table on the engine and only reads it in
        ``_denoise_pass``.

        **The dtype is load-bearing.** The reference builds this table in the
        serving dtype and takes ``.diff()`` there, so the Euler step sizes are
        bf16 differences of bf16 sigmas: ``bf16(0.9) - 1.0 == -0.1015625``,
        whereas an fp32 diff rounded to bf16 is ``-0.10009765625`` (two of the
        four steps differ). Building the table in fp32 "for precision" would
        change the ODE.
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
        backend: WaypointKVBackend,
        *,
        mouse: Tensor,
        button: Tensor,
        scroll: Tensor,
    ) -> Tensor:
        """Four frozen Euler steps of the rectified-flow ODE. Returns the settled
        latent; **does not write the ring**.

        Frozen matters: each step attends to a different noisy version of the
        same frame, so none of them may commit. They still see themselves,
        through the unconditional scratch write at the ring tail.

        ``sigmas`` is the ``[5]`` schedule in ``x``'s dtype, passed in rather
        than built here -- see ``_sigma_schedule``.
        """
        backend.set_frozen(True)
        # One reused sigma buffer, filled per step (the reference's shape --
        # a fresh allocation per step would defeat cudagraph capture).
        sigma = x.new_empty((x.size(0), x.size(1)))
        # strict=False is deliberate: there are 5 sigmas and 4 diffs, the
        # trailing 0.0 exists only to produce the last step size, and that
        # truncation IS the "4 denoise passes" of the 4+1 structure.
        for step_sigma, step_dsigma in zip(sigmas, sigmas.diff(), strict=False):
            v = self(
                x,
                sigma.fill_(step_sigma),
                frame_pos,
                mouse=mouse,
                button=button,
                scroll=scroll,
                backend=backend,
            )
            # fp32 accumulate, back to the latent dtype -- the reference's exact
            # expression; doing the add in bf16 loses the small late steps.
            x = (x.float() + step_dsigma.float() * v.float()).type_as(x)
        return x

    def _cache_pass(
        self,
        x: Tensor,
        frame_pos: Tensor,
        backend: WaypointKVBackend,
        *,
        mouse: Tensor,
        button: Tensor,
        scroll: Tensor,
    ) -> None:
        """The committing pass: one unfrozen forward at sigma=0 on the settled
        latent. Its only purpose is the side effect -- the K/V it writes into
        every layer's ring. The returned velocity is discarded.
        """
        backend.set_frozen(False)
        self(
            x,
            x.new_zeros((x.size(0), x.size(1))),
            frame_pos,
            mouse=mouse,
            button=button,
            scroll=scroll,
            backend=backend,
        )

    def generate_frame(
        self,
        noise: Tensor,
        frame_pos: Tensor,
        backend: WaypointKVBackend,
        *,
        mouse: Tensor,
        button: Tensor,
        scroll: Tensor,
    ) -> Tensor:
        """Denoise one frame from ``noise`` ``[B, N, C, H, W]`` and commit it.

        Five forwards: 4 frozen + 1 committing (CONTRACTS section 1). The caller
        owns the ring clock and must advance ``frame_pos`` by exactly one per
        committed frame -- all five passes of a frame share the same value.
        """
        # The .clone() is load-bearing, not hygiene: _denoise_pass is a compiled
        # region and inductor/cudagraphs reuse its output buffer, so the cache
        # pass's own allocations would stomp the latent it is supposed to be
        # reading. It must stay OUTSIDE the compiled region, on the returned
        # tensor, so the copy lands in caller-owned memory.
        sigmas = self._sigma_schedule(noise.device, noise.dtype)
        x0 = self._denoise_pass(
            noise, frame_pos, sigmas, backend, mouse=mouse, button=button, scroll=scroll
        ).clone()
        self._cache_pass(x0, frame_pos, backend, mouse=mouse, button=button, scroll=scroll)
        return x0

    def append_frame(
        self,
        latent: Tensor,
        frame_pos: Tensor,
        backend: WaypointKVBackend,
        *,
        mouse: Tensor,
        button: Tensor,
        scroll: Tensor,
    ) -> Tensor:
        """Prime the world state from a real (VAE-encoded) frame: the committing
        pass alone, no denoising, one forward.

        ``latent`` is already the settled x0, so there is nothing to solve and
        nothing to clone. Returned unchanged for the caller to decode.
        """
        self._cache_pass(latent, frame_pos, backend, mouse=mouse, button=button, scroll=scroll)
        return latent
