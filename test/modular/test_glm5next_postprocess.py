"""GLM-5.3-Flash emits each token's raw bytes, so a character split across
tokens survives per-token postprocess (per-token decode gave U+FFFD)."""

import torch

from mstar.model.glm5_next.glm5_next_model import Glm5NextModel


class _ByteLevelTokenizer:
    # GPT-2 byte-level token strings: "Ã" + "©" are the bytes of "é", "Ġ" is a space
    all_special_ids = [0]
    _tokens = {0: "<|endoftext|>", 1: "Ã", 2: "©", 3: "Ġhi", 4: "</think>"}

    def convert_ids_to_tokens(self, ids):
        return [self._tokens[i] for i in ids]


def test_postprocess_emits_raw_token_bytes():
    m = Glm5NextModel("x")
    m._tokenizer = _ByteLevelTokenizer()
    out = [m.postprocess(torch.tensor([i]), "text") for i in (1, 2)]
    assert out == [b"\xc3", b"\xa9"]
    assert b"".join(out).decode("utf-8") == "é"
    # special tokens drop; added non-special ones stay
    assert m.postprocess(torch.tensor([4, 3, 0]), "text") == b"</think> hi"
