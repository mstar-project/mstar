r"""Kimi K3 tokenizer wrapper.

The checkpoint ships a tiktoken-backed ``TikTokenTokenizer`` (``tokenization_kimi.py``)
whose ``apply_chat_template`` renders the XTML chat format (``encoding_k3.py``); both load
through ``AutoTokenizer.from_pretrained(..., trust_remote_code=True)``. This wrapper hides
the HF object behind the small surface M\* needs: chat rendering to ids, raw text
encoding, byte-faithful incremental decoding, and the special ids.
"""
from __future__ import annotations

from pathlib import Path


class KimiK3Tokenizer:
    def __init__(self, model_dir: str | Path):
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        self.bos_id: int = int(self._tok.convert_tokens_to_ids("[BOS]"))
        self.eos_id: int = int(self._tok.convert_tokens_to_ids("<|end_of_msg|>"))
        self.eot_id: int = int(self._tok.convert_tokens_to_ids("[EOT]"))
        self.pad_id: int = int(self._tok.convert_tokens_to_ids("[PAD]"))
        self.media_pad_id: int = int(self._tok.convert_tokens_to_ids("<|media_pad|>"))
        self.open_id: int = int(self._tok.convert_tokens_to_ids("<|open|>"))
        self.close_id: int = int(self._tok.convert_tokens_to_ids("<|close|>"))
        self.sep_id: int = int(self._tok.convert_tokens_to_ids("<|sep|>"))
        self._first_special_id: int = self.bos_id

    @property
    def hf(self):
        return self._tok

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size

    # Kimi's TikTokenTokenizer.encode/decode bypass the HF machinery only when called
    # without HF-style kwargs; the HF path would insert spaces around special tokens.
    def encode(self, text: str, allow_special_tokens: bool = False) -> list[int]:
        """Raw text -> ids. Special-token strings in ``text`` are encoded as ordinary text
        unless ``allow_special_tokens`` (injection-safe default, like the chat renderer)."""
        return [int(i) for i in self._tok.encode(text, allow_special_tokens=allow_special_tokens)]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        """Exact tiktoken decode. ``skip_special_tokens`` drops the reserved-range ids
        (>= 163584) except the three XTML structure markers, which downstream parsers need."""
        if skip_special_tokens:
            keep = {self.open_id, self.close_id, self.sep_id}
            ids = [i for i in ids if i < self._first_special_id or i in keep]
        return self._tok.decode([int(i) for i in ids])

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        add_generation_prompt: bool = True,
        thinking: bool = True,
        thinking_effort: str | None = None,
        tools: list | None = None,
    ) -> list[int]:
        kwargs = dict(add_generation_prompt=add_generation_prompt, tokenize=True, thinking=thinking)
        if thinking_effort is not None:
            kwargs["thinking_effort"] = thinking_effort
        if tools is not None:
            kwargs["tools"] = tools
        ids = self._tok.apply_chat_template(messages, **kwargs)
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        return [int(i) for i in ids]

    def render_chat(self, messages: list[dict], **kwargs) -> str:
        kwargs.setdefault("add_generation_prompt", True)
        return self._tok.apply_chat_template(messages, tokenize=False, **kwargs)
