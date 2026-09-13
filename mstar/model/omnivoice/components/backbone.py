"""The OmniVoice backbone, run as one packed bidirectional forward per step.

Every request in a step becomes two documents — the conditional canvas and its
CFG counterpart — packed end to end into a single row and separated by
flashinfer's ragged attention rather than by a mask.  Nothing is padded, so a
batch mixing a 2-second and a 25-second utterance wastes nothing, and the head
GEMM runs only at the positions whose logits are actually read.

The per-step kernel path is the reference's own fastest mode, reused rather than
re-derived: ``apply_flashinfer`` fuses QKV, swaps in flashinfer RoPE, RMSNorm
and ``silu_and_mul``, and keeps the whole layer in ``(S, H, D)`` so no transpose
copies happen.  Re-implementing that here would mean a second copy of tuned
kernels whose only job is to end up numerically identical.

What M* adds on top is what the reference cannot do from inside one
``generate()`` call: the documents in a step come from *different concurrent
requests*, at different diffusion steps, with different per-request knobs.  Its
own server has no batching at all (k2-fsa/OmniVoice#251).

The coupling to ``omnivoice.models.omnivoice_flashinfer`` is deliberate but
narrow — four module-private names, checked at load by ``assert_flashinfer_api``
so an upstream rename fails at boot rather than at the first request.
"""

import logging
import threading
from dataclasses import dataclass, field

import torch
from torch import nn

logger = logging.getLogger(__name__)

# The private surface this integration rides on. Checked once at load.
_REQUIRED_FI_NAMES = ("_CTX", "PackedAttnRunner", "_forward_logits", "apply_flashinfer")


def assert_flashinfer_api():
    """Fail at boot if the reference's packed-attention surface moved.

    These four names are private to ``omnivoice``. Pinning them here means an
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
    return fi


@dataclass
class CanvasItem:
    """One request's contribution to a step's packed batch.

    ``prefix_ids`` is ``[C, N]`` — style tokens, then text tokens, then the
    reference audio tokens when cloning.  ``tokens`` is ``[1, C, T]``, the live
    target canvas: all MASK at step 0, fully revealed when the loop ends.

    Both CFG documents are always built, even when ``guidance_scale`` is 0.
    The unconditional document is only the target region, so the waste is
    small, and a uniform two-documents-per-item layout keeps the packing and
    the gather index simple enough to be obviously correct.
    """

    request_id: str
    prefix_ids: torch.Tensor
    prefix_audio_mask: torch.Tensor
    tokens: torch.Tensor
    guidance_scale: float
    # Filled in by build_packed_canvas; scoring reads them back.
    flat_start: int = field(default=-1)

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
    doc_lens: list[int]            # [c_0, u_0, c_1, u_1, ...]
    tgt_index: torch.Tensor        # [2 * sum(target_len)]
    flat_target_total: int
    items: list[CanvasItem]

    def slice_logits(
        self, logits: torch.Tensor, item: CanvasItem
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditional and unconditional target logits for one item.

        ``logits`` is ``[1, C, 2 * flat_target_total, V]``, laid out as every
        item's conditional target block followed by every item's unconditional
        block — the gather order ``tgt_index`` was built in.
        """
        start, length = item.flat_start, item.target_len
        c_logits = logits[:, :, start : start + length, :]
        u_start = self.flat_target_total + start
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

    ``position_ids`` restart at 0 in every document.  Without that, RoPE would
    read a later document's canvas as a continuation of the previous request's.
    """
    if not items:
        raise ValueError("build_packed_canvas called with no items")

    num_codebook = items[0].prefix_ids.shape[0]

    doc_lens: list[int] = []
    for item in items:
        doc_lens.extend([item.cond_len, item.target_len])

    offsets = [0]
    for length in doc_lens[:-1]:
        offsets.append(offsets[-1] + length)
    total = sum(doc_lens)

    packed_ids = torch.full(
        (1, num_codebook, total), audio_mask_id, dtype=torch.long, device=device
    )
    audio_mask = torch.zeros((1, total), dtype=torch.bool, device=device)
    position_ids = torch.zeros((1, total), dtype=torch.long, device=device)

    cond_ranges: list[torch.Tensor] = []
    uncond_ranges: list[torch.Tensor] = []
    flat_cursor = 0

    for i, item in enumerate(items):
        c_off, u_off = offsets[2 * i], offsets[2 * i + 1]
        c_len, t_len = item.cond_len, item.target_len

        prefix_ids = item.prefix_ids.to(device)
        tokens = item.tokens[0].to(device)

        # Conditional document: [style | text | ref | target].
        packed_ids[0, :, c_off : c_off + c_len] = torch.cat([prefix_ids, tokens], dim=-1)
        audio_mask[0, c_off : c_off + c_len] = torch.cat(
            [
                item.prefix_audio_mask.to(device),
                torch.ones(t_len, dtype=torch.bool, device=device),
            ],
            dim=-1,
        )
        position_ids[0, c_off : c_off + c_len] = torch.arange(c_len, device=device)

        # Unconditional document: the target canvas alone. Dropping the
        # conditioning entirely *is* the null prompt here.
        packed_ids[0, :, u_off : u_off + t_len] = tokens
        audio_mask[0, u_off : u_off + t_len] = True
        position_ids[0, u_off : u_off + t_len] = torch.arange(t_len, device=device)

        item.flat_start = flat_cursor
        flat_cursor += t_len
        cond_ranges.append(
            torch.arange(c_off + c_len - t_len, c_off + c_len, device=device)
        )
        uncond_ranges.append(torch.arange(u_off, u_off + t_len, device=device))

    return PackedCanvas(
        packed_ids=packed_ids,
        audio_mask=audio_mask,
        position_ids=position_ids,
        doc_lens=doc_lens,
        tgt_index=torch.cat(cond_ranges + uncond_ranges),
        flat_target_total=flat_cursor,
        items=items,
    )


class OmniVoiceBackbone(nn.Module):
    """The patched reference model, driven one packed step at a time.

    Holding the reference instance rather than borrowing its submodules is what
    keeps the embedding merge, the audio head reshape and the fused attention
    path byte-for-byte the upstream ones; every line of per-step math that could
    drift is theirs.
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
