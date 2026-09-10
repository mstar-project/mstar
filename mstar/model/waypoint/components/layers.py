"""Stateless layer primitives for the Waypoint-1.5 DiT.

Faithful port of ``world_engine/src/model/nn.py`` plus the three conditioning
modules from ``world_engine/src/model/world_model.py``
(``ControllerInputEmbedding``, ``MLPFusion``, ``CondHead``). Attention, the
transformer block and the DiT itself live elsewhere; nothing here holds
sequence state, and nothing here touches the KV ring.

Two things in this file are load-bearing beyond "it is the same arithmetic":

  * **Parameter names are checkpoint keys.** ``fc1``/``fc2``, ``bias_in``,
    ``cond_proj``, ``mlp`` are the reference's attribute names and therefore
    the names ``weight_loader`` remaps onto. mstar's shared
    ``components.mlp.MLP`` spells its projections ``linear_in``/``linear_out``,
    so it is deliberately NOT reused: a local two-line ``MLP`` that keeps the
    checkpoint spelling is worth more than the shared class.
  * **``NoiseConditioner`` is an fp32 island.** The reference marks it
    ``NoCastModule``, i.e. it silently ignores ``.to(dtype)``. That trick is a
    bad fit for mstar (it warns, and it fights ``to_empty``), so the port keeps
    it an ordinary ``nn.Module`` that runs its own body under
    ``autocast(enabled=False)`` on ``.float()`` inputs, and publishes
    ``FP32_MODULE_PATHS`` for ``WaypointDiT.cast_serving_dtypes()`` to re-pin
    after the global bf16 cast.

The Fourier frequency table is derived state, not checkpoint state. The
reference registers it as a non-persistent buffer, which under mstar's meta
build would survive ``to_empty(device)`` as uninitialized garbage — a silent
wrong-numbers bug, since no loader completeness check covers buffers. It is
held outside the module tree instead, built on CPU at init and copied per
device on first use (the ``wan22.components.dit.Wan22RoPE3D`` approach).
"""

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.waypoint.config import WaypointConfig

# Submodule paths, relative to the ``WaypointDiT`` root, whose parameters must
# be restored to fp32 after a global ``.to(torch.bfloat16)``. This is the
# contract ``WaypointDiT.cast_serving_dtypes()`` consumes: "everything goes
# bf16, then these paths go back to fp32", mirroring
# ``Wan22DiT.cast_serving_dtypes``.
#
# The reference's ``NoCastModule`` set has three members; only one of them
# appears here, because the other two (``OrthoRoPEAngles``, ``OrthoRoPE`` in
# ``rope.py``) carry no parameters and no buffers at all — their tables live
# outside the module tree and are always fp32 — so there is nothing for a dtype
# cast to corrupt and nothing to pin back. See rope.py's module docstring.
FP32_MODULE_PATHS: tuple[str, ...] = ("denoise_step_emb",)


class DeviceTableCache:
    """Per-device replicas of small derived fp32 tables (RoPE frequencies,
    Fourier frequencies).

    These tables are a pure function of the config, so they are neither
    checkpoint state nor something a dtype cast should reach. Registering them
    as non-persistent buffers would put them inside the module tree, where
    ``to_empty(device)`` replaces their storage with uninitialized memory and
    ``.to(bfloat16)`` would truncate their precision. Holding them here instead
    keeps them out of ``state_dict``, out of ``to_empty``'s reach, and fp32
    forever.

    Callers MUST build the CPU tables with an explicit ``device="cpu"``: module
    ``__init__`` runs under ``with torch.device("meta")`` in mstar's build path,
    and an ambient-device ``torch.arange`` would produce data-less meta tensors.
    """

    _CPU = torch.device("cpu")

    def __init__(self, *tables: torch.Tensor):
        for table in tables:
            if table.device.type != "cpu":
                raise ValueError(
                    f"DeviceTableCache expects CPU-built tables (got {table.device}); "
                    "pass device='cpu' explicitly so the meta build context cannot capture it."
                )
        self._by_device: dict[torch.device, tuple[torch.Tensor, ...]] = {self._CPU: tables}

    def get(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        device = torch.device(device)
        if device not in self._by_device:
            self._by_device[device] = tuple(t.to(device) for t in self._by_device[self._CPU])
        return self._by_device[device]


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """Unweighted RMS norm over the last dim (reference ``nn.rms_norm``).

    No learned gain: every use site in Waypoint either has its scale supplied
    by adaLN (``ada_rmsnorm``, ``AdaLN``) or wants a bare normalization (Q/K
    norm, the two ``MLPFusion`` inputs).
    """
    return F.rms_norm(x, (x.size(-1),))


def ada_rmsnorm(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Per-frame adaLN modulation: ``rms_norm(x) * (1 + scale) + bias``.

    ``x`` is ``[B, N*T, D]`` (N frames of T tokens, flattened); ``scale`` and
    ``bias`` are ``[B, N, D]``, one modulation vector per *frame*. The unflatten
    exists purely so the per-frame vectors broadcast over that frame's tokens —
    it is the reference's ``eo.rearrange(x, 'b (n m) d -> b n m d')``.
    """
    x4 = x.unflatten(1, (scale.size(1), -1))
    y4 = rms_norm(x4) * (1 + scale.unsqueeze(2)) + bias.unsqueeze(2)
    return y4.flatten(1, 2)


def ada_gate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Per-frame adaLN output gate: ``x * gate``, ``gate`` broadcast over the
    tokens of its frame. Same shapes as ``ada_rmsnorm``. Note there is no
    ``1 +`` here — the gate multiplies the sublayer output before the residual
    add, so a zero gate means "contribute nothing"."""
    x4 = x.unflatten(1, (gate.size(1), -1))
    return (x4 * gate.unsqueeze(2)).flatten(1, 2)


class AdaLN(nn.Module):
    """adaLN with the scale/shift projection folded in: one Linear produces
    ``[scale | shift]`` from ``silu(cond)``.

    Used only for the DiT's output head (``out_norm``); the blocks get their
    six modulation tensors from ``CondHead`` and apply them with
    ``ada_rmsnorm``/``ada_gate`` instead. As there, ``cond`` is per-frame
    ``[B, N, D]`` and is expanded over each frame's tokens.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, n, d = cond.shape
        _, nm, _ = x.shape
        m = nm // n

        ab = self.fc(F.silu(cond))  # [b, n, 2d]
        ab = ab.view(b, n, 1, 2 * d).expand(-1, -1, m, -1).reshape(b, nm, 2 * d)
        scale, shift = ab.chunk(2, dim=-1)
        return rms_norm(x) * (1 + scale) + shift


class MLP(nn.Module):
    """Two-layer SiLU MLP, both projections bias-free.

    ``fc1``/``fc2`` and ``bias=False`` are checkpoint facts, not style: the
    reference's ``MLPFusion`` calls ``F.linear(h, self.mlp.fc2.weight)`` with no
    bias argument at all, which is only correct because there is no bias to
    pass. Adding one would load nothing and silently change the output.
    """

    def __init__(self, dim_in: int, dim_middle: int, dim_out: int):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_middle, bias=False)
        self.fc2 = nn.Linear(dim_middle, dim_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class NoiseConditioner(nn.Module):
    """sigma -> Fourier features -> MLP, the DiT's only noise-level input.

    fp32 island (reference ``NoCastModule``). ``FP32_MODULE_PATHS`` pins
    ``self.mlp`` back to fp32 after the serving bf16 cast, and the body runs
    under ``autocast(enabled=False)`` on a ``.float()`` sigma so nothing
    downcasts it again. The precision matters more than the 2 MB it costs: the
    four denoise sigmas are close together (1.0, 0.9, 0.75, 0.3) and the whole
    schedule is only distinguishable to the model through this embedding.

    The ``* 1000`` before the phase computation is the reference's scaling of
    the [0, 1] sigma range into a range where the Fourier basis actually
    rotates; ``* 2**0.5`` restores unit variance after the sin/cos concat.
    """

    def __init__(self, dim: int, fourier_dim: int = 512, base: float = 10_000.0):
        super().__init__()
        if fourier_dim % 2:
            raise ValueError(f"NoiseConditioner needs an even fourier_dim; got {fourier_dim}.")
        self.fourier_dim = fourier_dim
        # Derived, not checkpoint state — see the module docstring. device="cpu"
        # is required: __init__ runs under the meta-device build context.
        self._freq = DeviceTableCache(
            torch.logspace(0, -1, steps=fourier_dim // 2, base=base, dtype=torch.float32, device="cpu")
        )
        self.mlp = MLP(fourier_dim, dim * 4, dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """``s`` is ``[B, N]`` sigma; returns ``[B, N, dim]`` in ``s``'s dtype."""
        orig_dtype, shape = s.dtype, s.shape
        (freq,) = self._freq.get(s.device)

        with torch.amp.autocast("cuda", enabled=False):
            s = s.reshape(-1).float() * 1000  # fp32 for Fourier stability; x1000 for rotation range
            phase = s[:, None] * freq[None, :]
            emb = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
            emb = self.mlp(emb * 2**0.5)

        return emb.to(orig_dtype).view(*shape, -1)


class ControllerInputEmbedding(nn.Module):
    """Controller state -> one conditioning vector per frame.

    The concat order is ``(mouse, button, scroll)``: 2 + n_buttons + 1 = 259 for
    this checkpoint. **A permuted order does not raise** — the widths still sum
    to 259 and every downstream shape checks out — it just reads the mouse
    velocity out of button columns and produces plausible, wrong video. Treat
    this ordering as part of the checkpoint, and keep callers passing the three
    tensors positionally in this order.
    """

    def __init__(self, config: WaypointConfig):
        super().__init__()
        self.mlp = MLP(config.d_ctrl_in, config.d_model * config.mlp_ratio, config.d_model)

    def forward(self, mouse: torch.Tensor, button: torch.Tensor, scroll: torch.Tensor) -> torch.Tensor:
        """``mouse`` ``[B, N, 2]``, ``button`` ``[B, N, n_buttons]``, ``scroll``
        ``[B, N, 1]`` -> ``[B, N, d_model]``."""
        x = torch.cat((mouse, button, scroll), dim=-1)
        return self.mlp(x)


class MLPFusion(nn.Module):
    """Fuses a per-frame conditioning vector into that frame's tokens.

    Nominally ``MLP(2*D, D, D)`` applied to ``cat([x, cond])``, and the
    parameter tree says exactly that — ``mlp.fc1`` is one ``[D, 2D]`` matrix, so
    ``weight_loader`` has a single key to fill. The *compute* splits it:
    ``fc1.weight.chunk(2, dim=1)`` gives the x-half and the cond-half, each
    ``[D, D]``, which lets ``cond`` broadcast over the T tokens of its frame
    instead of being repeated into a ``[B, N*T, 2D]`` concat. Same arithmetic,
    no materialized copy; this is the path the reference actually ships (and
    what its ``SplitMLPFusion`` inference patch bakes in).

    The split is at compute time only. Do not turn it into stored ``fc1_x`` /
    ``fc1_c`` parameters: the checkpoint stores them split, and the loader's job
    is to ``cat(dim=1)`` them back into ``mlp.fc1`` (loader transform 7).
    """

    def __init__(self, config: WaypointConfig):
        super().__init__()
        self.mlp = MLP(2 * config.d_model, config.d_model, config.d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """``x`` ``[B, N*T, D]``, ``cond`` ``[B, N, D]`` -> ``[B, N*T, D]``."""
        B, _, D = x.shape
        L = cond.shape[1]

        Wx, Wc = self.mlp.fc1.weight.chunk(2, dim=1)  # each [D, D]

        x = x.view(B, L, -1, D)
        h = F.linear(x, Wx) + F.linear(cond, Wc).unsqueeze(2)  # broadcast, no repeat/cat
        h = F.silu(h)
        y = F.linear(h, self.mlp.fc2.weight)
        return y.flatten(1, 2)


class CondHead(nn.Module):
    """Per-layer WAN-style conditioning head: the noise embedding becomes the
    block's six adaLN modulation tensors (scale/shift/gate for attention, then
    the same three for the MLP).

    **The parameters here are split between per-layer and shared, and the split
    is the reason the checkpoint is 1.86B on disk but 1.28B resident:**

      * ``bias_in`` — a plain ``[d_model]`` ``nn.Parameter``, genuinely
        per-layer, 24 distinct copies. Present only for
        ``noise_conditioning == "wan"``.
      * ``cond_proj`` — a ``nn.ModuleList`` of 6 bias-free ``[D, D]`` Linears
        that is **physically shared across all 24 blocks**. The DiT ties them
        after construction by aliasing the ``.weight`` of blocks 1..23 onto
        block 0's (reference ``WorldDiT.__init__``), and the loader loads block
        0's copies and drops the other 23 sets. That is why ``cond_proj`` is a
        plain ``ModuleList`` of ``nn.Linear`` and nothing cleverer: aliasing
        ``blk.cond_head.cond_proj[j].weight = ref.cond_head.cond_proj[j].weight``
        has to remain a one-line assignment.

    **Tie after ``to_empty``, not in ``__init__``.** ``Module._apply`` allocates
    per parameter with no cross-module memo, so ``to_empty(device)`` silently
    un-aliases tied weights (``.to(dtype)`` on meta does not). Tying only in the
    constructor therefore yields 24 independent copies at serve time — no error,
    just 0.6B of duplicated resident weights and 23 blocks whose ``cond_proj``
    the loader never fills. Once tied, ``named_parameters()`` deduplicates, so
    the loader's completeness check sees block 0's set only, which is what it
    assumes.

    The checkpoint spells this head as two half-heads, ``attn_cond_head``
    (indices 0..2) and ``mlp_cond_head`` (indices 3..5), with a ``bias_in`` on
    each; the loader merges them and keeps the mlp one.
    """

    n_cond = 6

    def __init__(self, config: WaypointConfig):
        super().__init__()
        if config.noise_conditioning == "wan":
            self.bias_in = nn.Parameter(torch.zeros(config.d_model))
        else:
            # register_parameter(None) rather than a bare attribute so the
            # `is not None` branch below stays cheap and state_dict stays clean.
            self.register_parameter("bias_in", None)
        self.cond_proj = nn.ModuleList(
            nn.Linear(config.d_model, config.d_model, bias=False) for _ in range(self.n_cond)
        )

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """``cond`` ``[B, N, D]`` -> six ``[B, N, D]`` tensors, in block order
        ``(s0, b0, g0, s1, b1, g1)``."""
        cond = cond + self.bias_in if self.bias_in is not None else cond
        h = F.silu(cond)
        return tuple(p(h) for p in self.cond_proj)
