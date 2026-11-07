"""Kimi K3's tokenizer: the checkpoint's tiktoken ``TikTokenTokenizer`` (``tokenization_kimi.py``),
whose chat template is ``encoding_k3.py``. Both are the checkpoint's remote code, loaded with
``trust_remote_code`` as vLLM and SGLang load them. After merceod's #257.
"""
from __future__ import annotations

from pathlib import Path

# the reserved-range ids start here ([BOS]); the structure markers kept in replies
_FIRST_SPECIAL_ID = 163584
_KEPT_SPECIAL = ("<|open|>", "<|close|>", "<|sep|>")


class KimiK3Tokenizer:
    def __init__(self, model_dir: str | Path):
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        self.eos_ids = tuple(int(self._tok.convert_tokens_to_ids(t))
                             for t in ("<|end_of_msg|>", "[EOS]", "[EOT]"))
        self._kept = {int(self._tok.convert_tokens_to_ids(t)) for t in _KEPT_SPECIAL}

    @property
    def hf(self):
        return self._tok

    def encode(self, text: str) -> list[int]:
        """Raw text; special-token strings in it are encoded as text (Kimi's encode bypasses the HF
        wrapper only without HF-style kwargs)."""
        return [int(i) for i in self._tok.encode(text, allow_special_tokens=False)]

    def apply_chat_template(
        self, messages: list[dict], *, add_generation_prompt: bool = True, thinking: bool = True,
        thinking_effort: str | None = None,
    ) -> list[int]:
        kwargs = dict(add_generation_prompt=add_generation_prompt, tokenize=True, thinking=thinking)
        if thinking_effort is not None:
            kwargs["thinking_effort"] = thinking_effort
        ids = self._tok.apply_chat_template(messages, **kwargs)
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        return [int(i) for i in ids]

    def decode_bytes(self, ids: list[int]) -> bytes:
        """The tokens' raw bytes (tiktoken), the reserved range dropped but its structure markers,
        as ``skip_special_tokens`` would."""
        ids = [i for i in ids if i < _FIRST_SPECIAL_ID or i in self._kept]
        return self._tok.model.decode_bytes(ids)
