"""Bagel says how much each cache label holds, because nothing else can.

The pool admits a request only when the pages it can still take fit, and for
a text prompt its keys say how long the prompt is. An image request writes
what no key describes: an attachment's ViT and VAE tokens, the latents
image_gen puts on every label without committing them, and the guidance
labels' copies. Under-counting any of them lets the pool admit a request
whose own step then finds no page, which is the hold admission exists to end.
"""

from __future__ import annotations

import re
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.model.multimodal import PromptPart


class _StubTokenizer:
    """One id per character; the specials Bagel scans for are single ids."""

    SPECIALS = {
        "<|im_start|>": 101, "<|im_end|>": 102, "<|vision_start|>": 103,
        "<|image_pad|>": 104, "<|vision_end|>": 105,
    }
    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return self.SPECIALS.get(token, self.unk_token_id)

    def encode(self, text):
        pattern = "|".join(re.escape(t) for t in self.SPECIALS)
        ids = []
        for piece in re.split(f"({pattern})", text):
            if piece in self.SPECIALS:
                ids.append(self.SPECIALS[piece])
            else:
                ids.extend(1000 + ord(c) for c in piece)
        return ids


@pytest.fixture
def bagel():
    """A BagelModel with only what process_prompt needs."""
    from mstar.model.bagel.bagel_model import BagelModel

    class _Bagel(BagelModel):
        def __init__(self):
            self.config = SimpleNamespace(
                think_mode=False, vit_max_num_patch_per_side=70,
                max_latent_size=64, latent_downsample=16,
                vit_config=SimpleNamespace(patch_size=14),
            )
            self.tokenizer = _StubTokenizer()
            self.boi_token_id = _StubTokenizer.SPECIALS["<|vision_start|>"]
            self.eoi_token_id = _StubTokenizer.SPECIALS["<|vision_end|>"]

    return _Bagel()


def _text_tokens(out) -> int:
    return sum(len(span) for span in out.new_input_tensors["text_inputs"])


def test_a_text_request_reserves_its_prompt_and_decodes_on_main(bagel):
    out = bagel.process_prompt("hello", ["text"], ["text"])

    assert out.metadata == {
        "prompt_slots": {"kv": {"main": _text_tokens(out)}},
        "decode_labels": {"kv": ["main"]},
    }, "a text request was sized other than by its prompt, growing by decode"


class _StubTransfer:
    """No engine, no bytes moved."""

    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs


def test_an_attachment_counts_at_the_size_its_walk_writes(bagel, monkeypatch):
    from mstar.engine.resources.kv import manager as manager_mod
    from mstar.engine.resources.kv.config import KVReqConfig, PagedKVConfig

    parts = [
        PromptPart(modality="image", index=0),
        PromptPart(modality="text", text="What is this?"),
    ]
    out = bagel.process_prompt(
        "What is this?", ["image", "text"], ["text"],
        tensors={"image_inputs": [torch.zeros(3, 480, 640)]}, prompt_parts=parts,
    )

    # the walk resizes 480x640 to 512x688 for the VAE, then to 518x686: 37x49 patches
    slots = out.metadata["prompt_slots"]["kv"]["main"]
    assert slots == _text_tokens(out) + 37 * 49 + 2
    assert -(-slots // 128) == 15, (
        "a 480x640 attachment was counted at the ViT's largest grid, not its own size"
    )

    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    kv = manager_mod.KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=32768,
            max_num_pages=20, page_size=128,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv.ingest_request("look", KVReqConfig(
        max_tokens=256, prompt_slots=out.metadata["prompt_slots"]["kv"],
        decode_labels=out.metadata["decode_labels"]["kv"],
    ))

    assert kv.admit_retrieve("look", "LLM", "prefill_vit", None).ready, (
        "a request whose prompt and decode fit in 20 pages was not admitted"
    )


def test_an_image_request_counts_its_latents_on_every_guidance_label(bagel):
    out = bagel.process_prompt(
        "a red cube", ["text"], ["image"], height=512, width=512,
    )

    latents = (512 // 16) * (512 // 16) + 2
    main = _text_tokens(out) + latents
    assert out.metadata == {
        "prompt_slots": {"kv": {"main": main, "cfg_text": main, "cfg_img": main}},
        "decode_labels": {"kv": []},
    }, (
        "image_gen's latents land on every guidance label and nothing is "
        "decoded, but the count said otherwise"
    )


def test_an_edit_sizes_its_latents_off_its_input(bagel):
    parts = [
        PromptPart(modality="image", index=0),
        PromptPart(modality="text", text="make it blue"),
    ]
    out = bagel.process_prompt(
        "make it blue", ["image", "text"], ["image"],
        tensors={"image_inputs": [torch.zeros(3, 512, 1024)]}, prompt_parts=parts,
    )

    # already 1024 on its long edge, so the edit keeps its size; the ViT walk
    # takes it to 490x980 (35x70 patches), the VAE walk keeps it (32x64)
    vit, vae = 35 * 70 + 2, 32 * 64 + 2
    latents = (512 // 16) * (1024 // 16) + 2
    slots = out.metadata["prompt_slots"]["kv"]["main"]
    assert slots == _text_tokens(out) + vit + vae + latents, (
        "an edit generates at its input's size, and its attachment writes "
        "both a ViT and a VAE walk"
    )
