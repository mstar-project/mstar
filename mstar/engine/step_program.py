"""A captured step as an ordered task list, for executors other than the graph.

Today a captured step is the compiled forward replayed as an accelerator graph
(``GraphExecutor``, the default everywhere). A ``StepProgram`` describes the
same step as phases an executor can run without leaving the device: each phase
names its op, the module parameters it reads, the scratch it writes and the
per-row sums of squares (``ss`` slots) it consumes and produces for the
RMSNorms between phases. It is built from what the submodule already declares
(its modules and their configs, the resources its step binds), so no model code
names a kernel: the model declares, the engine picks the executor.

The first program is the residual-codebook depth loop of a code predictor
(Qwen3-Omni and Qwen3-TTS talkers): per depth position a stack of decoder layers
(QKV, attention, O + residual, gate/up + SwiGLU, down + residual), and per
position after the first an LM head and a sampler that feeds the next
position's embedding. ``PersistentExecutor`` (``persistent_executor.py``) runs
it as one resident kernel; ``lower`` turns it into the kernel's phase table.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import torch


class OpKind(enum.IntEnum):
    """Phase ops of the depth-loop program (values are the kernel's)."""
    PRO = 0      # residual <- the position-0 input (the talker's last hidden state)
    QKV = 1      # q, k, v <- RMSNorm(residual) @ Wqkv
    ATTN = 2     # q/k RMSNorm, rope, K/V append, attention over the frame's positions
    OPROJ = 3    # residual += attn @ Wo
    GATEUP = 4   # act <- silu(x @ Wg) * (x @ Wu), x = RMSNorm(residual)
    DOWN = 5     # residual += act @ Wd
    INPUT = 6    # residual <- the position-1 input (the layer-0 code's embedding)
    LMHEAD = 7   # logits <- RMSNorm(residual) @ LM[group]
    SAMPLE = 8   # code <- sampler(logits); residual <- embedding(code); embedding sum += it


GEMM_OPS = frozenset({OpKind.QKV, OpKind.OPROJ, OpKind.GATEUP, OpKind.DOWN, OpKind.LMHEAD})
ROW_OPS = frozenset({OpKind.PRO, OpKind.INPUT, OpKind.SAMPLE})
TILE_N = 16      # output columns per GEMM task
CHUNK_K = 1024   # K per weight chunk a task streams


class DepthDecoderShape(NamedTuple):
    """The dimensions a depth-loop program is built for."""
    layers: int
    positions: int       # code groups: position 0 is the talker's hidden state, 1..positions-1 sample
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    eps: float
    rope_theta: float


@dataclasses.dataclass(frozen=True)
class StepOp:
    """One phase. ``tiles`` are the GEMM's 16-column output tiles from
    ``first_tile`` (or the rows of a per-row op, or the (row, kv head) pairs of
    an attention op, both given per step rather than here). ``weight`` and
    ``norm`` are parameter names on the submodule; ``ss_in`` / ``ss_out`` are
    the sum-of-squares slots read (the RMSNorm of the A operand) and written."""
    kind: OpKind
    position: int
    layer: int = -1
    weight: str | None = None
    norm: str | None = None
    norm2: str | None = None
    tiles: int = 0
    first_tile: int = 0
    chunks: int = 1
    ldw: int = 0
    ss_in: int = -1
    ss_out: int = -1
    aux: int = 0
    # index into a stacked ``weight`` (the LM head of one group), -1 for the whole tensor
    weight_index: int = -1


@dataclasses.dataclass(frozen=True)
class StepProgram:
    """A step's phases in execution order, the shape they were built for, and
    the resources the step binds (resource key -> what the program uses it for)."""
    shape: DepthDecoderShape
    ops: tuple[StepOp, ...]
    resources: Mapping[str, str]
    ss_slots: int

    def gemm_bytes(self, dtype_bytes: int = 2) -> int:
        """Weight bytes one step streams through its GEMMs."""
        total = 0
        for op in self.ops:
            if op.kind in GEMM_OPS:
                k = op.chunks * CHUNK_K if op.kind != OpKind.GATEUP else CHUNK_K
                cols = (op.tiles - op.first_tile) * TILE_N * (2 if op.kind == OpKind.GATEUP else 1)
                total += cols * k * dtype_bytes
        return total

    def barriers(self) -> int:
        """Synchronisation points a phase-at-a-time executor pays (one per phase)."""
        return len(self.ops)


def ss_in_slot(position: int, layer: int, layers: int) -> int:
    return (position * layers + layer) * 2


def ss_post_slot(position: int, layer: int, layers: int) -> int:
    return (position * layers + layer) * 2 + 1


def ss_final_slot(position: int, layers: int, positions: int) -> int:
    return positions * layers * 2 + position


def depth_decoder_program(shape: DepthDecoderShape, sampler_key: str, param_prefix: str = "") -> StepProgram:
    """The depth loop of one frame as phases.

    Position 0 runs the layers on the talker's hidden state only to fill the
    K/V cache, so its last layer stops after the K/V columns of QKV and their
    append. Every later position ends with the LM head of its group and the
    sampler, which writes the code and loads the next position's input."""
    s = shape
    if s.hidden != CHUNK_K or s.intermediate % CHUNK_K or (s.heads * s.head_dim) % CHUNK_K:
        raise ValueError(f"depth program needs hidden = {CHUNK_K} and K multiples of it, got {s}")
    qkv_cols = (s.heads + 2 * s.kv_heads) * s.head_dim
    layer = f"{param_prefix}model.layers.{{}}"
    ops: list[StepOp] = [StepOp(OpKind.PRO, 0, ss_out=ss_in_slot(0, 0, s.layers))]
    for p in range(s.positions):
        for li in range(s.layers):
            pre = layer.format(li)
            cache_only = p == 0 and li == s.layers - 1
            ops.append(StepOp(
                OpKind.QKV, p, li, weight=f"{pre}.self_attn.qkv_proj.weight", norm=f"{pre}.input_layernorm.weight",
                tiles=qkv_cols // TILE_N, first_tile=(s.heads * s.head_dim) // TILE_N if cache_only else 0,
                ldw=s.hidden, ss_in=ss_in_slot(p, li, s.layers)))
            ops.append(StepOp(
                OpKind.ATTN, p, li, norm=f"{pre}.self_attn.q_norm.weight", norm2=f"{pre}.self_attn.k_norm.weight",
                aux=0 if cache_only else 1))
            if cache_only:
                continue
            ops.append(StepOp(
                OpKind.OPROJ, p, li, weight=f"{pre}.self_attn.o_proj.weight", tiles=s.hidden // TILE_N,
                chunks=(s.heads * s.head_dim) // CHUNK_K, ldw=s.heads * s.head_dim,
                ss_out=ss_post_slot(p, li, s.layers)))
            ops.append(StepOp(
                OpKind.GATEUP, p, li, weight=f"{pre}.mlp.gate_up_proj.weight",
                norm=f"{pre}.post_attention_layernorm.weight", tiles=s.intermediate // TILE_N, chunks=2,
                ldw=s.hidden, ss_in=ss_post_slot(p, li, s.layers)))
            last = li == s.layers - 1
            ops.append(StepOp(
                OpKind.DOWN, p, li, weight=f"{pre}.mlp.down_proj.weight", tiles=s.hidden // TILE_N,
                chunks=s.intermediate // CHUNK_K, ldw=s.intermediate,
                ss_out=ss_final_slot(p, s.layers, s.positions) if last else ss_in_slot(p, li + 1, s.layers)))
        if p == 0:
            ops.append(StepOp(OpKind.INPUT, 0, ss_out=ss_in_slot(1, 0, s.layers)))
            continue
        ops.append(StepOp(
            OpKind.LMHEAD, p, weight=f"{param_prefix}lm_head_weight", norm=f"{param_prefix}model.norm.weight",
            tiles=s.vocab // TILE_N, ldw=s.hidden, ss_in=ss_final_slot(p, s.layers, s.positions), aux=p - 1,
            weight_index=p - 1))
        ops.append(StepOp(
            OpKind.SAMPLE, p, ss_out=ss_in_slot(p + 1, 0, s.layers) if p + 1 < s.positions else -1))
    return StepProgram(shape=s, ops=tuple(ops), resources={sampler_key: "sampler"},
                       ss_slots=s.positions * s.layers * 2 + s.positions)


def code_predictor_shape(code_predictor: torch.nn.Module, positions: int) -> DepthDecoderShape:
    """The program shape read off a code predictor's modules (layers, attention,
    MLP, heads), so a checkpoint with other sizes builds its own program."""
    layers = code_predictor.model.layers
    attn = layers[0].self_attn
    mlp = layers[0].mlp
    return DepthDecoderShape(
        layers=len(layers), positions=positions, hidden=attn.hidden_size if hasattr(attn, "hidden_size")
        else layers[0].input_layernorm.weight.shape[0], heads=attn.num_heads, kv_heads=attn.num_kv_heads,
        head_dim=attn.head_dim, intermediate=mlp.intermediate_size, vocab=code_predictor.lm_head[0].out_features,
        eps=float(layers[0].input_layernorm.variance_epsilon), rope_theta=float(attn.rope_theta))


def program_for_talker(submodule, sampler_key: str) -> StepProgram:
    """The depth-loop program of a talker submodule that holds a ``code_predictor``
    and binds the code predictor's sampler as ``sampler_key`` in its decode step."""
    shape = code_predictor_shape(submodule.code_predictor, positions=submodule.num_codes)
    return depth_decoder_program(shape, sampler_key, param_prefix="code_predictor.")


# Fields of one row of the lowered table (int64), in the kernel's order.
PHASE_FIELDS = ("kind", "pos", "layer", "n_tasks", "chunks", "ldw", "task_begin", "ss_in", "ss_out", "w", "norm_w",
                "norm_w2", "aux")


def lower(program: StepProgram, params: Mapping[str, torch.Tensor], device: torch.device | str | None = None
          ) -> torch.Tensor:
    """The kernel's phase table: one int64 row per op, parameters as device
    addresses (so the table is only valid while ``params`` live)."""

    def addr(name: str | None, index: int = -1) -> int:
        if name is None:
            return 0
        t = params[name]
        if index >= 0:
            t = t[index]
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        return t.data_ptr()

    rows = []
    for op in program.ops:
        rows.append([int(op.kind), op.position, op.layer, op.tiles, op.chunks, op.ldw, op.first_tile, op.ss_in,
                     op.ss_out, addr(op.weight, op.weight_index), addr(op.norm), addr(op.norm2), op.aux])
    return torch.tensor(rows, dtype=torch.int64, device=device)


def check(program: StepProgram) -> list[str]:
    """Structural problems of a program: an ss slot read before any op wrote it,
    a slot written twice, or an op whose operands do not exist. Empty when sound."""
    problems = []
    written: set[int] = set()
    for i, op in enumerate(program.ops):
        if op.ss_in >= 0 and op.ss_in not in written:
            problems.append(f"op {i} {op.kind.name} p{op.position} l{op.layer} reads ss slot {op.ss_in} "
                            "before it is written")
        if op.ss_out >= 0:
            if op.ss_out in written:
                problems.append(f"op {i} {op.kind.name} writes ss slot {op.ss_out} twice")
            written.add(op.ss_out)
        if op.kind in GEMM_OPS and (op.weight is None or op.tiles <= op.first_tile):
            problems.append(f"op {i} {op.kind.name} has no weight or no tiles")
    return problems


def required_params(program: StepProgram) -> Sequence[str]:
    names = []
    for op in program.ops:
        for n in (op.weight, op.norm, op.norm2):
            if n is not None and n not in names:
                names.append(n)
    return names


def rope_table(shape: DepthDecoderShape, device=None) -> torch.Tensor:
    """[positions, head_dim]: cos of the head_dim / 2 rope frequencies at each position, then their sin (fp32)."""
    half = shape.head_dim // 2
    inv_freq = 1.0 / (shape.rope_theta ** (torch.arange(0, half, dtype=torch.float64) * 2 / shape.head_dim))
    angles = torch.arange(shape.positions, dtype=torch.float64)[:, None] * inv_freq[None, :]
    return torch.cat([angles.cos(), angles.sin()], dim=1).to(torch.float32).to(device).contiguous()



def interpret(program: StepProgram, w: Mapping[str, torch.Tensor], codebook: torch.Tensor, last_hidden: torch.Tensor,
              embed0: torch.Tensor, forced: torch.Tensor, *, kernel_rounding: bool = False,
              stop_after: int | None = None) -> dict[str, torch.Tensor]:
    """Run ``program`` in float64 with the codes ``forced`` [rows, positions]
    (teacher forcing): the reference executor any other executor is checked
    against. With ``kernel_rounding`` the GEMM operands, the K/V cache and the
    attention output are rounded to bf16 where the resident kernel rounds them;
    without, nothing is rounded but the weights. Returns the final (or, with
    ``stop_after``, the partial) state: h, qkv, attn, act, kv, logits, emb_sum."""
    s = program.shape
    dev = last_hidden.device
    f64 = torch.float64
    bs = last_hidden.shape[0]

    def rnd(x):
        return x.to(torch.bfloat16).to(f64) if kernel_rounding else x

    W = {k: v.to(f64) for k, v in w.items()}
    cb = codebook.to(f64)
    rope = rope_table(s, dev).to(f64)
    st = {"h": torch.zeros(bs, s.hidden, dtype=f64, device=dev),
          "qkv": torch.zeros(bs, (s.heads + 2 * s.kv_heads) * s.head_dim, dtype=f64, device=dev),
          "attn": torch.zeros(bs, s.heads * s.head_dim, dtype=f64, device=dev),
          "act": torch.zeros(bs, s.intermediate, dtype=f64, device=dev),
          "kv": torch.zeros(s.layers, bs, 2, s.positions, s.kv_heads, s.head_dim, dtype=f64, device=dev),
          "logits": torch.zeros(s.positions - 1, bs, s.vocab, dtype=f64, device=dev),
          "emb_sum": embed0.to(f64).clone(), "codes": forced.clone()}
    ss = torch.zeros(program.ss_slots, bs, dtype=f64, device=dev)

    def normed(slot, weight):
        return rnd(st["h"] * torch.rsqrt(ss[slot] / s.hidden + s.eps)[:, None] * W[weight][None, :])

    half = s.head_dim // 2
    ops = program.ops if stop_after is None else program.ops[:stop_after]
    for op in ops:
        k = op.kind
        if k in (OpKind.PRO, OpKind.INPUT):
            st["h"] = (last_hidden if k == OpKind.PRO else embed0).to(f64).clone()
            ss[op.ss_out] = (st["h"] ** 2).sum(1)
        elif k == OpKind.QKV:
            x = normed(op.ss_in, op.norm)
            c0 = op.first_tile * TILE_N
            st["qkv"][:, c0:] = x @ W[op.weight][c0:].T
        elif k == OpKind.ATTN:
            qkv = st["qkv"]
            q = qkv[:, :s.heads * s.head_dim].view(bs, s.heads, s.head_dim)
            kk = qkv[:, s.heads * s.head_dim:(s.heads + s.kv_heads) * s.head_dim].view(bs, s.kv_heads, s.head_dim)
            v = qkv[:, (s.heads + s.kv_heads) * s.head_dim:].view(bs, s.kv_heads, s.head_dim)
            cos = torch.cat([rope[op.position, :half]] * 2)
            sin = torch.cat([rope[op.position, half:]] * 2)

            def norm_rope(x, weight, cos, sin):
                x = x * torch.rsqrt((x ** 2).mean(-1, keepdim=True) + s.eps) * W[weight]
                rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
                return x * cos + rot * sin

            q = norm_rope(q, op.norm, cos, sin)
            kk = rnd(norm_rope(kk, op.norm2, cos, sin))
            st["kv"][op.layer, :, 0, op.position] = kk
            st["kv"][op.layer, :, 1, op.position] = rnd(v)
            if op.aux:
                keys = st["kv"][op.layer, :, 0, :op.position + 1]   # [bs, j, kvh, d]
                vals = st["kv"][op.layer, :, 1, :op.position + 1]
                qg = q.view(bs, s.kv_heads, s.heads // s.kv_heads, s.head_dim)
                scores = torch.einsum("bhgd,bjhd->bhgj", qg, keys) / math.sqrt(s.head_dim)
                out = torch.einsum("bhgj,bjhd->bhgd", scores.softmax(-1), vals)
                st["attn"] = rnd(out.reshape(bs, s.heads * s.head_dim))
        elif k in (OpKind.OPROJ, OpKind.DOWN):
            a = st["attn"] if k == OpKind.OPROJ else st["act"]
            st["h"] = st["h"] + a @ W[op.weight].T
            ss[op.ss_out] = (st["h"] ** 2).sum(1)
        elif k == OpKind.GATEUP:
            x = normed(op.ss_in, op.norm)
            gu = x @ W[op.weight].T
            g, u = gu[:, :s.intermediate], gu[:, s.intermediate:]
            st["act"] = rnd(g * torch.sigmoid(g) * u)
        elif k == OpKind.LMHEAD:
            x = normed(op.ss_in, op.norm)
            st["logits"][op.aux] = x @ W[op.weight][op.weight_index].T
        elif k == OpKind.SAMPLE:
            code = forced[:, op.position]
            e = cb[op.position - 1][code]
            st["emb_sum"] = st["emb_sum"] + e
            if op.ss_out >= 0:
                st["h"] = e.clone()
                ss[op.ss_out] = (e ** 2).sum(1)
    return st
