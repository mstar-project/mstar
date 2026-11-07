"""Tokenization wrapper for Pi0.5: PaliGemma tokenizer and openpi's prompt text."""

import logging

import torch

from mstar.model.pi05.config import Pi05Config

logger = logging.getLogger(__name__)


def pi05_prompt(task: str | None, state_bins: list[int] | None = None) -> str:
    """The text openpi's PaligemmaTokenizer tokenizes for pi0.5.

    Underscores and newlines in the task become spaces. The case is kept, and
    so is the space after "Action:", since openpi tokenizes the string as built.
    """
    cleaned = (task or "").strip().replace("_", " ").replace("\n", " ")
    if state_bins is None:
        return cleaned
    return f"Task: {cleaned}, State: {' '.join(map(str, state_bins))};\nAction: "


class Pi05Tokenizer:
    """Wrapper around the HF PaliGemma tokenizer. The robot state reaches it as
    text inside the prompt (see :func:`pi05_prompt`)."""

    def __init__(self, hf_tokenizer, config: Pi05Config):
        self.hf_tokenizer = hf_tokenizer
        self.config = config

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Refuses a prompt over ``max_lang_tokens``. openpi truncates it instead,
        which cuts off the state and "Action:" that end the prompt."""
        ids = self.hf_tokenizer(prompt, add_special_tokens=True).input_ids
        limit = self.config.max_lang_tokens
        if len(ids) > limit:
            raise ValueError(f"the prompt is {len(ids)} tokens, task and robot state together; "
                             f"pi0.5 takes at most {limit} (max_lang_tokens)")
        return torch.tensor(ids, dtype=torch.long)
