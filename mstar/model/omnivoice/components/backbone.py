"""The OmniVoice backbone, run as one packed bidirectional forward per step.

A request in a step becomes two documents — the conditional canvas and its CFG
counterpart — packed end to end into a single row and separated by flashinfer's
ragged attention rather than by a mask, or one document when it turned CFG off.
Nothing is padded, so a batch mixing a 2-second and a 25-second utterance wastes
nothing, and the head GEMM runs only at the positions whose logits are read.

The per-step kernels are the reference's own fastest mode: ``apply_flashinfer``
fuses QKV and gate/up, and the layer runs in ``(S, H, D)`` with flashinfer RoPE
and ``silu_and_mul`` (its RMSNorms go through M*'s shared ones).
``OmniVoiceBackbone.encode`` drives those fused weights with its own layer loop
rather than through the patched ``llm``: it norms q and k straight off the fused
QKV output (the reference copies each out first) and folds every residual add
into the next RMSNorm.  At 16 requests a
step those copies and adds were a tenth of the step's GPU time.

What M* adds on top is what the reference cannot do from inside one
``generate()`` call: the documents in a step come from *different concurrent
requests*, at different diffusion steps, with different per-request knobs.  Its
own server has no batching at all (k2-fsa/OmniVoice#251).

The coupling to ``omnivoice.models.omnivoice_flashinfer`` is deliberate but
narrow — four module-private names and the four fused-weight attributes its
``apply_flashinfer`` sets on each layer, checked at load by
``assert_flashinfer_api`` and ``assert_flashinfer_patched`` so an upstream
rename fails at boot rather than at the first request.
"""

import logging
import threading
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from mstar.engine.resources.rms_norm import run_rms_norm
from mstar.model.components.norm import add_rms_norm
from mstar.utils.h2d import to_device_nonblocking

logger = logging.getLogger(__name__)

# The private surface this integration rides on. Checked once at load: the
# module's names before ``apply_flashinfer`` runs, and the fused weights it
# leaves on every layer's attention and MLP (read by ``OmniVoiceBackbone.encode``)
# after.
_REQUIRED_FI_NAMES = ("_CTX", "PackedAttnRunner", "_forward_logits", "apply_flashinfer")
_REQUIRED_FI_LAYER_ATTRS = {
    "self_attn": ("_fi_w_qkv", "_fi_qkv_split", "_fi_rope_theta"),
    "mlp": ("_fi_w_gate_up",),
}


def assert_flashinfer_api():
    """Fail at boot if the reference's private surface moved.

    These names are private to ``omnivoice``. Pinning them here means an
    upgrade that renames one produces a clear error on startup instead of a
    silent fallback or a first-request crash.
    """
    from omnivoice.models import omnivoice_flashinfer as fi

    missing = [name for name in _REQUIRED_FI_NAMES if not hasattr(fi, name)]
    if missing:
        raise RuntimeError(
            "omnivoice.models.omnivoice_flashinfer is missing "
            f"{missing}; mstar's OmniVoice backbone drives its packed "
            "attention directly. Pin the omnivoice version, or port the "
            "packed forward into mstar."
        )

    from omnivoice.models import omnivoice as ov

    if not hasattr(ov, "_resolve_instruct"):
        raise RuntimeError(
            "omnivoice.models.omnivoice is missing _resolve_instruct; "
            "mstar validates voice-design instructs with it. Pin the "
            "omnivoice version, or port the validator into mstar."
        )
    return fi


def assert_flashinfer_patched(model: nn.Module) -> None:
    """Fail at boot if ``apply_flashinfer`` no longer leaves the fused weights
    ``OmniVoiceBackbone.encode`` reads, rather than at capture."""
    missing = sorted({
        f"{part}.{name}"
        for layer in model.llm.layers
        for part, names in _REQUIRED_FI_LAYER_ATTRS.items()
        for name in names
        if not hasattr(getattr(layer, part), name)
    })
    if missing:
        raise RuntimeError(
            f"omnivoice's apply_flashinfer no longer sets {missing} on the layers; "
            "mstar's OmniVoice layer loop reads them. Pin the omnivoice version, "
            "or port the fused weights into mstar."
        )


@dataclass
class CanvasItem:
    """One request's contribution to a step's packed batch.

    ``prefix_ids`` is ``[C, N]`` — style tokens, then text tokens, then the
    reference audio tokens when cloning.  ``tokens`` is ``[1, C, T]``, the live
    target canvas: all MASK at step 0, fully revealed when the loop ends.

    An item at ``guidance_scale == 0`` contributes one document instead of
    two. Scoring throws the unconditional logits away in that case, and the
    unconditional document is the whole target region plus its share of the
    head GEMM, so building it would be close to twice the work for nothing.
    """

    request_id: str
    prefix_ids: torch.Tensor
    prefix_audio_mask: torch.Tensor
    tokens: torch.Tensor
    guidance_scale: float
    # Filled in by build_packed_canvas; scoring reads them back.
    # flat_u_start stays -1 for an item with no unconditional document.
    flat_start: int = field(default=-1)
    flat_u_start: int = field(default=-1)

    @property
    def target_len(self) -> int:
        return self.tokens.shape[-1]

    @property
    def cond_len(self) -> int:
        return self.prefix_ids.shape[-1] + self.target_len

    @property
    def does_cfg(self) -> bool:
        return self.guidance_scale != 0


@dataclass
class PackedCanvas:
    """One step's packed inputs, plus the index that gathers its logits."""

    packed_ids: torch.Tensor       # [1, C, total]
    audio_mask: torch.Tensor       # [1, total]
    position_ids: torch.Tensor     # [1, total]
    doc_lens: list[int]            # [c_0, u_0, c_1, ...], u only where CFG is on
    tgt_index: torch.Tensor        # [flat_target_total + flat_uncond_total]
    flat_target_total: int
    flat_uncond_total: int
    items: list[CanvasItem]

    def slice_logits(
        self, logits: torch.Tensor, item: CanvasItem
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Conditional and unconditional target logits for one item.

        Laid out as every item's conditional target block followed by the
        unconditional blocks of the items that asked for CFG, which is the
        gather order ``tgt_index`` was built in.

        The unconditional half is ``None`` for an item at
        ``guidance_scale == 0``: it has no unconditional document, because
        scoring would discard those logits anyway.
        """
        start, length = item.flat_start, item.target_len
        c_logits = logits[:, :, start : start + length, :]
        if item.flat_u_start < 0:
            return c_logits, None
        u_start = self.flat_target_total + item.flat_u_start
        u_logits = logits[:, :, u_start : u_start + length, :]
        return c_logits, u_logits


def build_packed_canvas(
    items: list[CanvasItem],
    audio_mask_id: int,
    device: torch.device,
) -> PackedCanvas:
    """Pack a step's requests into one ragged row.

    Documents are laid out ``[cond_0, uncond_0, cond_1, uncond_1, ...]`` and
    kept apart by the attention plan's ``doc_lens``, not by a mask — so there
    are no pad positions at all, and therefore none of the fully-masked-row
    hazard a dense padded batch has to guard against.

    An item at ``guidance_scale == 0`` contributes only its conditional
    document, so a batch of those is close to half the packed length and half
    the head GEMM.  Mixing the two kinds in one step is fine: the layout is
    driven per item.

    ``position_ids`` restart at 0 in every document.  Without that, RoPE would
    read a later document's canvas as a continuation of the previous request's.
    """
    if not items:
        raise ValueError("build_packed_canvas called with no items")

    # One concatenation per device tensor, and the index tensors built on the
    # host and copied once: a step packs up to 16 requests, and ten small
    # kernels per request were a fifth of the step's host time.
    id_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []
    doc_lens: list[int] = []
    cond_rows: list[torch.Tensor] = []
    uncond_rows: list[torch.Tensor] = []
    longest = max(item.target_len for item in items)
    revealed = torch.ones(longest, dtype=torch.bool, device=device)

    cursor = flat_cursor = flat_u_cursor = 0
    for item in items:
        c_len, t_len = item.cond_len, item.target_len
        tokens = item.tokens[0].to(device)
        target_mask = revealed[:t_len]

        # Conditional document: [style | text | ref | target].
        id_parts += [item.prefix_ids.to(device), tokens]
        mask_parts += [item.prefix_audio_mask.to(device), target_mask]
        positions.append(torch.arange(c_len))
        doc_lens.append(c_len)
        item.flat_start = flat_cursor
        flat_cursor += t_len
        cond_rows.append(torch.arange(cursor + c_len - t_len, cursor + c_len))
        cursor += c_len

        if not item.does_cfg:
            item.flat_u_start = -1
            continue

        # Unconditional document: the target canvas alone. Dropping the
        # conditioning entirely *is* the null prompt here.
        id_parts.append(tokens)
        mask_parts.append(target_mask)
        positions.append(torch.arange(t_len))
        doc_lens.append(t_len)
        item.flat_u_start = flat_u_cursor
        flat_u_cursor += t_len
        uncond_rows.append(torch.arange(cursor, cursor + t_len))
        cursor += t_len

    return PackedCanvas(
        packed_ids=torch.cat(id_parts, dim=-1).unsqueeze(0),
        audio_mask=torch.cat(mask_parts).unsqueeze(0),
        position_ids=to_device_nonblocking(torch.cat(positions).unsqueeze(0), device),
        doc_lens=doc_lens,
        tgt_index=to_device_nonblocking(torch.cat(cond_rows + uncond_rows), device),
        flat_target_total=flat_cursor,
        flat_uncond_total=flat_u_cursor,
        items=items,
    )


class OmniVoiceBackbone(nn.Module):
    """The patched reference model, driven one packed step at a time.

    Holding the reference instance keeps the embedding merge, the audio head
    reshape and the fused weights the upstream ones. ``encode`` runs the layer
    stack over them itself (see the module docstring); ``forward`` is the
    reference's own packed forward, the path with backbone graphs off.
    """

    def __init__(self, model: nn.Module, plan_dtype: torch.dtype):
        super().__init__()
        self.model = model
        self.plan_dtype = plan_dtype
        # The engine runs forwards on a single dedicated GPU thread
        # (worker.py: ThreadPoolExecutor(max_workers=1)), so the module-global
        # _CTX the reference's attention reads is not contended today. The lock
        # costs nothing per step and makes that assumption survive a change to
        # the worker's threading, where the failure mode would otherwise be one
        # request attending with another's document boundaries.
        self._ctx_lock = threading.Lock()

    def encode(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        pos_ids: torch.Tensor,
        attn,
    ) -> torch.Tensor:
        """Embeddings and the Qwen3 body over one packed row: ``[N, hidden]``.

        Token-major inputs (``input_ids`` ``[N, C]``, the rest ``[N]``, ``pos_ids``
        int32) so a captured replay can copy a shorter row into its bucket.
        ``attn`` is the engine's ragged attention resource, planned for this
        row's documents.
        """
        import flashinfer

        hidden = self.model._prepare_embed_inputs(
            input_ids.t().unsqueeze(0), audio_mask.unsqueeze(0)
        )[0]
        llm = self.model.llm
        residual = None
        for layer in llm.layers:
            sa, mlp = layer.self_attn, layer.mlp
            norm = layer.input_layernorm
            if residual is None:
                residual = hidden
                x = run_rms_norm(hidden, norm.weight, eps=norm.variance_epsilon)
            else:
                # The fused add-norm normalizes the sum before rounding it to half
                # precision, where the reference rounds first: logits match it to
                # that rounding, not bit for bit.
                x, residual = add_rms_norm(hidden, residual, norm.weight, norm.variance_epsilon)
            n, d = x.shape[0], sa.head_dim
            heads_q, heads_k, _ = (width // d for width in sa._fi_qkv_split)
            qkv = F.linear(x, sa._fi_w_qkv).view(n, -1, d)
            # the q/k norms read their heads straight off the fused output
            q = run_rms_norm(qkv[:, :heads_q], sa.q_norm.weight, eps=sa.q_norm.variance_epsilon)
            k = run_rms_norm(qkv[:, heads_q:heads_q + heads_k], sa.k_norm.weight, eps=sa.k_norm.variance_epsilon)
            flashinfer.rope.apply_rope_pos_ids_inplace(
                q, k, pos_ids, rope_theta=sa._fi_rope_theta, interleave=False,
            )
            hidden = sa.o_proj(attn.run(q, k, qkv[:, heads_q + heads_k:]).reshape(n, -1))
            norm = layer.post_attention_layernorm
            x, residual = add_rms_norm(hidden, residual, norm.weight, norm.variance_epsilon)
            hidden = mlp.down_proj(flashinfer.activation.silu_and_mul(F.linear(x, mlp._fi_w_gate_up)))
        hidden, _ = add_rms_norm(hidden, residual, llm.norm.weight, llm.norm.variance_epsilon)
        return hidden

    def head(self, hidden: torch.Tensor, tgt_index: torch.Tensor) -> torch.Tensor:
        """The audio head at the logits-consuming positions, laid out as
        ``_forward_logits`` returns it: ``[1, C, len(tgt_index), V]``."""
        cfg = self.model.config
        tgt_hidden = hidden[tgt_index]
        logits = self.model.audio_heads(tgt_hidden)
        return logits.view(
            1, tgt_hidden.shape[0], cfg.num_audio_codebook, cfg.audio_vocab_size
        ).permute(0, 2, 1, 3)

    @torch.inference_mode()
    def forward(self, canvas: PackedCanvas) -> torch.Tensor:
        """One packed forward; returns ``[1, C, 2 * flat_target_total, V]``.

        Only the logits-consuming positions go through the 1024 -> 8200 head
        GEMM and the float32 upcast. Running them over the full packed length
        would be several times the work, most of it on prefix positions whose
        logits are never read.
        """
        fi = assert_flashinfer_api()

        with self._ctx_lock:
            self.model._fi_runner.plan(canvas.doc_lens, self.plan_dtype)
            fi._CTX["wrapper"] = self.model._fi_runner.wrapper
            fi._CTX["pos_ids"] = canvas.position_ids[0].to(torch.int32)
            # CUDA-graph bucketing is not used here; M* has its own capture
            # machinery and reconciling the two is a separate piece of work.
            fi._CTX["doc_slots"] = None
            return fi._forward_logits(
                self.model,
                canvas.packed_ids,
                canvas.audio_mask,
                canvas.position_ids,
                canvas.tgt_index,
            )
