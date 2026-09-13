"""Stateless layer primitives for the Waypoint-1.5 DiT.

Port of ``world_engine/src/model/nn.py`` plus the three conditioning modules
from ``world_engine/src/model/world_model.py``. Nothing here holds sequence
state and nothing here touches the KV ring.

Parameter names are checkpoint keys: ``fc1``/``fc2``, ``bias_in``, ``cond_proj``
and ``mlp`` are the reference's attribute names and the names ``weight_loader``
remaps onto. mstar's shared ``components.mlp.MLP`` spells its projections
``linear_in``/``linear_out``, so it is not reused.

``NoiseConditioner`` is an fp32 island: it runs its body under
``autocast(enabled=False)`` on ``.float()`` inputs and publishes
``FP32_MODULE_PATHS`` for ``WaypointDiT.cast_serving_dtypes()`` to re-pin after
the global bf16 cast. Its Fourier frequency table is derived state, held outside
the module tree so ``to_empty(device)`` cannot leave it uninitialized -- no
loader completeness check covers buffers.
"""

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.waypoint.config import WaypointConfig

# Submodule paths, relative to the WaypointDiT root, whose parameters must be
# restored to fp32 after a global .to(torch.bfloat16). The reference's
# NoCastModule set has three members; the other two (OrthoRoPEAngles, OrthoRoPE)
# carry no parameters or buffers for a cast to corrupt.
FP32_MODULE_PATHS: tuple[str, ...] = ("denoise_step_emb",)


def bf16_roundtrip(table: torch.Tensor) -> torch.Tensor:
    """Quantize an fp32 derived table the way ``NoCastModule._apply`` does.
    Only reachable under ``WaypointConfig.reference_compat``.
    """
    return table.to(torch.bfloat16).to(table.dtype)


def _bf16_bits(x: torch.Tensor) -> torch.Tensor:
    """bf16 storage reinterpreted as an unsigned ``0..65535`` LUT index."""
    return x.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF


class DeviceTableCache:
    """Per-device replicas of small derived fp32 tables (RoPE frequencies,
    Fourier frequencies).

    A pure function of the config, so neither checkpoint state nor something a
    dtype cast should reach. Holding them here rather than as non-persistent
    buffers keeps them out of ``state_dict``, out of ``to_empty``'s reach, and
    fp32 forever.

    Callers MUST build the CPU tables with an explicit ``device="cpu"``: module
    ``__init__`` runs under ``with torch.device("meta")``, where an
    ambient-device ``torch.arange`` produces a data-less meta tensor.
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

    No learned gain: every use site either gets its scale from adaLN or wants a
    bare normalization (Q/K norm, the two ``MLPFusion`` inputs).
    """
    return F.rms_norm(x, (x.size(-1),))


def ada_rmsnorm(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Per-frame adaLN modulation: ``rms_norm(x) * (1 + scale) + bias``.

    ``x`` is ``[B, N*T, D]``; ``scale`` and ``bias`` are ``[B, N, D]``, one
    modulation vector per *frame*. The unflatten is what makes those vectors
    broadcast over their own frame's tokens.
    """
    x4 = x.unflatten(1, (scale.size(1), -1))
    y4 = rms_norm(x4) * (1 + scale.unsqueeze(2)) + bias.unsqueeze(2)
    return y4.flatten(1, 2)


def ada_gate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Per-frame adaLN output gate: ``x * gate``, same shapes as
    ``ada_rmsnorm``. No ``1 +`` here -- the gate multiplies the sublayer output
    before the residual add, so a zero gate contributes nothing."""
    x4 = x.unflatten(1, (gate.size(1), -1))
    return (x4 * gate.unsqueeze(2)).flatten(1, 2)


class AdaLN(nn.Module):
    """adaLN with the scale/shift projection folded in: one Linear produces
    ``[scale | shift]`` from ``silu(cond)``.

    The DiT's output head only; blocks get their six modulation tensors from
    ``CondHead``. ``cond`` is per-frame ``[B, N, D]``, expanded over its tokens.
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

    ``bias=False`` is a checkpoint fact: the reference's ``MLPFusion`` calls
    ``F.linear(h, self.mlp.fc2.weight)`` with no bias argument, which is only
    correct because there is no bias to pass.
    """

    def __init__(self, dim_in: int, dim_middle: int, dim_out: int):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_middle, bias=False)
        self.fc2 = nn.Linear(dim_middle, dim_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class NoiseConditioner(nn.Module):
    """sigma -> Fourier features -> MLP, the DiT's only noise-level input.

    fp32 island: ``FP32_MODULE_PATHS`` pins ``self.mlp`` back to fp32 after the
    serving bf16 cast, and the body runs under ``autocast(enabled=False)`` on a
    ``.float()`` sigma. The four denoise sigmas are close together
    (1.0, 0.9, 0.75, 0.3) and this embedding is the only thing that separates
    them. ``* 1000`` scales [0, 1] into the Fourier basis's rotating range;
    ``* 2**0.5`` restores unit variance after the sin/cos concat.
    """

    def __init__(
        self,
        dim: int,
        fourier_dim: int = 512,
        base: float = 10_000.0,
        *,
        reference_compat: bool = False,
        cached_sigmas: tuple[float, ...] = (),
    ):
        super().__init__()
        if fourier_dim % 2:
            raise ValueError(f"NoiseConditioner needs an even fourier_dim; got {fourier_dim}.")
        self.fourier_dim = fourier_dim
        # Derived, not checkpoint state. device="cpu" is required: __init__ runs
        # under the meta-device build context.
        freq = torch.logspace(
            0, -1, steps=fourier_dim // 2, base=base, dtype=torch.float32, device="cpu"
        )
        self._freq = DeviceTableCache(bf16_roundtrip(freq) if reference_compat else freq)
        self.mlp = MLP(fourier_dim, dim * 4, dim)

        self.reference_compat = reference_compat
        self.cached_sigmas = tuple(float(s) for s in cached_sigmas)
        self._lut: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        if reference_compat:
            if not self.cached_sigmas:
                raise ValueError(
                    "reference_compat NoiseConditioner needs the sigma schedule to cache."
                )
            levels = torch.tensor(self.cached_sigmas, dtype=torch.bfloat16, device="cpu")
            if len(set(_bf16_bits(levels).tolist())) != len(self.cached_sigmas):
                raise ValueError(
                    f"scheduler_sigmas {self.cached_sigmas} collide in bf16; the reference's "
                    "LUT would be ambiguous."
                )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """``s`` is ``[B, N]`` sigma; returns ``[B, N, dim]`` in ``s``'s dtype
        (bf16 under ``reference_compat``, whose table is stored bf16)."""
        if self.reference_compat:
            return self._cached_embed(s)
        return self._embed(s)

    def materialize(self, device: torch.device | str) -> None:
        """Materialize frequency data and the optional post-load sigma LUT."""
        device = torch.device(device)
        self._freq.get(device)
        if self.reference_compat:
            self._reference_lut(device)

    def _embed(self, s: torch.Tensor) -> torch.Tensor:
        orig_dtype, shape = s.dtype, s.shape
        (freq,) = self._freq.get(s.device)

        with torch.amp.autocast("cuda", enabled=False):
            s = s.reshape(-1).float() * 1000  # fp32 for Fourier stability; x1000 for rotation range
            phase = s[:, None] * freq[None, :]
            emb = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
            emb = self.mlp(emb * 2**0.5)

        return emb.to(orig_dtype).view(*shape, -1)

    def _cached_embed(self, s: torch.Tensor) -> torch.Tensor:
        """The reference's ``CachedDenoiseStepEmb``: read the embedding out of a
        bf16 table keyed on sigma's own bf16 bits. A sigma that is not on the
        schedule indexes one past the table and raises, rather than returning a
        neighbouring row."""
        if s.dtype is not torch.bfloat16:
            raise RuntimeError(f"reference_compat NoiseConditioner expects bf16 sigma; got {s.dtype}.")
        table, lut, oob = self._reference_lut(s.device)
        idx = lut[_bf16_bits(s)]
        return table[torch.where(idx >= 0, idx, oob).to(torch.int64)]

    def _reference_lut(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build (once per device) the reference's sigma table.

        The batch shape is load-bearing: all ``S`` sigmas in one call is an M=S
        GEMM, which under ``float32_matmul_precision('high')`` rounds through
        TF32 where the served M=1 GEMV stays exact fp32. Built on first use
        because it needs loaded weights, so an eager forward must warm it before
        ``compile_regions``.
        """
        device = torch.device(device)
        if device not in self._lut:
            levels = torch.tensor(self.cached_sigmas, dtype=torch.bfloat16, device=device)
            with torch.no_grad():
                table = self._embed(levels[:, None]).squeeze(1).to(torch.bfloat16).contiguous()
            lut = torch.full((65536,), -1, dtype=torch.int32, device=device)
            lut[_bf16_bits(levels)] = torch.arange(
                len(self.cached_sigmas), dtype=torch.int32, device=device
            )
            oob = torch.tensor(len(self.cached_sigmas), dtype=torch.int32, device=device)
            self._lut[device] = (table, lut, oob)
        return self._lut[device]


class ControllerInputEmbedding(nn.Module):
    """Controller state -> one conditioning vector per frame.

    The concat order is ``(mouse, button, scroll)`` -- 2 + n_buttons + 1 = 259
    here -- and it is a checkpoint fact. A permuted order does not raise: the
    widths still sum to 259 and every downstream shape checks out, it just reads
    mouse velocity out of button columns. Callers pass the three positionally.
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

    The parameter tree is ``MLP(2*D, D, D)`` over ``cat([x, cond])`` -- one
    ``[D, 2D]`` ``mlp.fc1`` for the loader to fill. The compute splits that
    matrix instead (``chunk(2, dim=1)``) so ``cond`` broadcasts over the T tokens
    of its frame rather than being repeated into a ``[B, N*T, 2D]`` concat.

    That split is compute-time only. Stored ``fc1_x``/``fc1_c`` parameters would
    undo transform 7, whose job is to ``cat(dim=1)`` them into ``mlp.fc1``.
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

    The per-layer/shared split is why the checkpoint is 1.86B on disk and 1.28B
    resident: ``bias_in`` is genuinely per-layer (24 copies, present only for
    ``noise_conditioning == "wan"``), while the six ``cond_proj`` Linears are
    physically shared across all 24 blocks -- the DiT aliases blocks 1..23's
    ``.weight`` onto block 0's and the loader drops the other 23 sets.

    The aliasing is established by ``WaypointDiT.retie_cond_proj``, which must run
    after ``to_empty``; see its docstring for what un-ties it.

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
