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
from api_uploads import png, preprocess

LABELS = ("main", "cfg_text", "cfg_img")


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


# the walk resizes 480x640 to 512x688 for the VAE, then to 518x686: 37x49 ViT patches
ATTACHED = 37 * 49 + 2
# a 512x1024 edit is already 1024 on its long edge, so it generates at its size; the ViT
# walk takes it to 490x980 (35x70 patches), the VAE walk keeps it (32x64)
EDIT_VIT, EDIT_VAE, EDIT_LATENTS = 35 * 70 + 2, 32 * 64 + 2, 32 * 64 + 2
EDIT = EDIT_VIT + EDIT_VAE + EDIT_LATENTS


@pytest.mark.parametrize(("prompt", "images", "output_modalities", "kwargs", "slots", "decoded"), [
    ("hello", [], ["text"], {}, {"main": 0}, ["main"]),
    ("What is this?", [(480, 640)], ["text"], {}, {"main": ATTACHED}, ["main"]),
    # an OpenAI chat whose only part is an image arrives with no prompt
    (None, [(480, 640)], ["text"], {}, {"main": ATTACHED}, ["main"]),
    # image_gen's latents land on every guidance label, and nothing decodes
    ("a red cube", [], ["image"], {"height": 512, "width": 512}, dict.fromkeys(LABELS, 32 * 32 + 2), []),
    # the first output picks the walks: text then image decodes text
    ("a red cube", [], ["text", "image"], {"height": 16, "width": 16}, {"main": 0}, ["main"]),
    # no walk writes the attachment to cfg_img
    ("make it blue", [(512, 1024)], ["image"], {}, {"main": EDIT, "cfg_text": EDIT, "cfg_img": EDIT_LATENTS}, []),
], ids=["text", "attachment", "no_text", "image_gen", "text_first", "edit"])
def test_each_label_reserves_the_prompt_plus_what_its_walks_write(
    bagel, prompt, images, output_modalities, kwargs, slots, decoded, tmp_path,
):
    files = [png(height, width) for height, width in images]
    tensors, declared = preprocess(bagel, None, prompt, files, output_modalities, kwargs, tmp_path)

    text = sum(len(span) for span in tensors.get("text_inputs") or [])
    assert (declared["prompt_slots"], declared["decode_labels"]) == (
        {"kv": {label: text + extra for label, extra in slots.items()}}, {"kv": decoded},
    ), "a label was counted other than its prompt plus the attachments and latents its walks write"
