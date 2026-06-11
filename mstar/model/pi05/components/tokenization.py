"""Tokenization wrapper for Pi0.5: PaliGemma tokenizer + state discretization."""

import logging

import torch

from mstar.model.pi05.config import Pi05Config

logger = logging.getLogger(__name__)


def normalize_prompt(prompt: str) -> str:
    """Lowercase + strip whitespace, matching openpi's PaligemmaTokenizer."""
    return prompt.strip().lower()


class Pi05Tokenizer:
    """Wrapper around the HF PaliGemma tokenizer that also tokenizes robot state.

    Robot state values are discretized into ``state_token_bins`` bins and mapped
    to language token IDs starting at ``state_token_offset``. Pi0.5 reuses
    bottom-of-vocab tokens for state bins so that PaliGemma's embedding table
    can embed them directly.
    """

    def __init__(self, hf_tokenizer, config: Pi05Config):
        self.hf_tokenizer = hf_tokenizer
        self.config = config

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        text = normalize_prompt(prompt)
        ids = self.hf_tokenizer(text, add_special_tokens=True).input_ids
        ids = ids[: self.config.max_lang_tokens]
        return torch.tensor(ids, dtype=torch.long)

