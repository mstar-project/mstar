# Copyright (c) 2024 Alibaba Inc.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified for M* in 2026: the special-token list written
# down without the reference tokenizer class.
#
# Original file (FunAudioLLM/CosyVoice, cosyvoice/tokenizer/tokenizer.py) was
# released under Apache-2.0, with the full license text available at
# https://github.com/FunAudioLLM/CosyVoice/blob/main/LICENSE.
#
# This modified file is released under the same license.

"""CosyVoice3's text tokenizer: the Qwen2 BPE of ``CosyVoice-BlankEN`` plus the
special tokens CosyVoice3 adds (prompt delimiters, paralinguistic tags, and
English and pinyin phonemes for pronunciation fixes such as ``[j][ǐ]``).

The ids follow from the order the tokens are added, so the list below is the
reference's (``cosyvoice/tokenizer/tokenizer.py``, ``CosyVoice3Tokenizer``);
``<|endofprompt|>`` must come out as 151646, which the LM was trained on.
"""

from __future__ import annotations

from pathlib import Path

import torch

from mstar.model.cosyvoice3.config import END_OF_PROMPT, END_OF_PROMPT_ID

_TAGS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endofprompt|>",
    "[breath]",
    "<strong>",
    "</strong>",
    "[noise]",
    "[laughter]",
    "[cough]",
    "[clucking]",
    "[accent]",
    "[quick_breath]",
    "<laughter>",
    "</laughter>",
    "[hissing]",
    "[sigh]",
    "[vocalized-noise]",
    "[lipsmack]",
    "[mn]",
    "<|endofsystem|>",
)

_EN_PHONEMES = (
    "AA AA0 AA1 AA2 AE AE0 AE1 AE2 AH AH0 AH1 AH2 AO AO0 AO1 AO2 AW AW0 AW1 AW2 AY AY0 AY1 AY2 "
    "B CH D DH EH EH0 EH1 EH2 ER ER0 ER1 ER2 EY EY0 EY1 EY2 F G HH IH IH0 IH1 IH2 IY IY0 IY1 IY2 "
    "JH K L M N NG OW OW0 OW1 OW2 OY OY0 OY1 OY2 P R S SH T TH UH UH0 UH1 UH2 UW UW0 UW1 UW2 "
    "V W Y Z ZH"
).split()

_PINYIN = (
    "a ai an ang ao b c ch d e ei en eng f g h i ian in ing iu ià iàn iàng iào iá ián iáng iáo "
    "iè ié iòng ióng iù iú iā iān iāng iāo iē iě iōng iū iǎ iǎn iǎng iǎo iǒng iǔ j k l m n o ong "
    "ou p q r s sh t u uang ue un uo uà uài uàn uàng uá uái uán uáng uè ué uì uí uò uó uā uāi "
    "uān uāng uē uě uī uō uǎ uǎi uǎn uǎng uǐ uǒ vè w x y z zh à ài àn àng ào á ái án áng áo è "
    "èi èn èng èr é éi én éng ér ì ìn ìng í ín íng ò òng òu ó óng óu ù ùn ú ún ā āi ān āng āo "
    "ē ēi ēn ēng ě ěi ěn ěng ěr ī īn īng ō ōng ōu ū ūn ǎ ǎi ǎn ǎng ǎo ǐ ǐn ǐng ǒ ǒng ǒu ǔ ǔn ǘ "
    "ǚ ǜ"
).split()

SPECIAL_TOKENS = _TAGS + tuple(f"[{p}]" for p in _EN_PHONEMES) + tuple(f"[{p}]" for p in _PINYIN)


class CosyVoice3TextTokenizer:
    def __init__(self, path: str | Path):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.tokenizer.add_special_tokens(
            {
                "eos_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
                "additional_special_tokens": list(SPECIAL_TOKENS),
            }
        )
        got = self.tokenizer.convert_tokens_to_ids(END_OF_PROMPT)
        if got != END_OF_PROMPT_ID:
            raise RuntimeError(f"{END_OF_PROMPT} tokenizes to {got}; the LM was trained with {END_OF_PROMPT_ID}")

    def __call__(self, text: str) -> torch.Tensor:
        """``[N]`` long ids; no BOS/EOS is added (Qwen2 adds none)."""
        if not text:
            return torch.zeros(0, dtype=torch.long)
        return torch.tensor(self.tokenizer([text])["input_ids"][0], dtype=torch.long)
