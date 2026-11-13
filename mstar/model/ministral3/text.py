"""Checkpoint-backed text intake/output, independent of weight loading and serving.

The Model wrapper delegates prompt IDs and text_output to this object.
Transformers is used for tokenization only; all model computation stays native.
"""

from pathlib import Path

import torch
from jinja2 import TemplateError

from mstar.model.ministral3.config import Ministral3TextConfig
from mstar.model.multimodal import TEXT, PromptPart, messages_from_parts
from mstar.model.utils import ByteLevelDetokenizer, TextChunkDetokenizer, TextOutput


class Ministral3TextProcessor:
    """Load local metadata, render text chats, and declare byte-faithful output."""

    def __init__(self, directory: str | Path):
        from transformers import AutoTokenizer

        # Preserve the published Tekken pattern. The fixture's token IDs were
        # checked against mistral-common, independently of the HF tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False, fix_mistral_regex=False,
        )
        self.config = Ministral3TextConfig.from_hf(
            directory, tokenizer_eos_token_id=self.tokenizer.eos_token_id,
        )
        if len(self.tokenizer) != self.config.vocab_size:
            raise ValueError("Ministral 3 tokenizer vocabulary does not match text_config")
        if not self.tokenizer.chat_template:
            raise ValueError("Ministral 3 Instruct requires the checkpoint's chat template")
        if not ByteLevelDetokenizer.is_byte_level(self.tokenizer):
            raise ValueError("Ministral 3 text streaming requires a byte-level tokenizer")
        self.output = TextOutput(TextChunkDetokenizer(self.tokenizer), self.config.stop_token_ids)

    def prompt_ids(self, parts: list[PromptPart], *, allow_images: bool = False) -> torch.Tensor:
        """Host token IDs; image-enabled intake expands the template's placeholders."""
        if not parts:
            raise ValueError("Ministral 3 requires a text prompt")
        allowed = {TEXT, "image"} if allow_images else {TEXT}
        if any(part.modality not in allowed for part in parts):
            raise ValueError("Unsupported Ministral 3 prompt modality")
        if any(part.role not in (None, "system", "user", "assistant") for part in parts):
            raise ValueError("Ministral 3 text supports system, user and assistant messages only")
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages_from_parts(parts), tokenize=False, add_generation_prompt=True,
            )
        except TemplateError as exc:
            raise ValueError(f"Ministral 3's chat template refused the chat: {exc}") from exc
        # The template already writes BOS and prior turns' EOS. Adding special
        # tokens a second time would change the prompt the native decoder sees.
        ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        if len(ids) > self.config.max_position_embeddings:
            raise ValueError("Ministral 3 prompt exceeds max_position_embeddings")
        return torch.tensor(ids, dtype=torch.long)
