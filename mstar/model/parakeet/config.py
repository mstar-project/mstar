"""Config for NVIDIA's FastConformer transducer ASR checkpoints: Parakeet-TDT
(``parakeet_tdt``) and the cache-aware streaming Nemotron-3.5 ASR
(``nemotron3_5_asr``), read from the HF ``config.json``,
``generation_config.json`` and ``processor_config.json``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mstar.model.components.fastconformer import FastConformerConfig
from mstar.model.components.transducer import TransducerConfig
from mstar.model.utils import GRANARY_LANGUAGES, base_language

SUPPORTED_MODEL_TYPES = ("parakeet_tdt", "parakeet_rnnt", "nemotron3_5_asr")

WALK, DECODE_LOOP = "transcribe", "decode_loop"
# each request's decoding state between iterations of the decode loop
TRANSDUCER_STATE = "transducer_state"
# model_kwargs and step_metadata keys, from process_prompt: the clip's encoder
# frames, and Nemotron's language prompt
NUM_FRAMES, PROMPT_ID = "num_frames", "prompt_id"


def check_detected_language(language: str | None) -> None:
    """Parakeet-TDT detects the language from the speech. A request may name
    one it knows, or ``auto``, which it then ignores; any other is refused."""
    if language is None or language == "":
        return
    # base_language refuses a language that isn't a string
    if base_language(language) not in ("auto", *GRANARY_LANGUAGES):
        raise ValueError(
            f"unsupported language {language!r}; Parakeet detects one of: {', '.join(GRANARY_LANGUAGES)}"
        )


@dataclass
class ParakeetModelConfig:
    model_type: str
    encoder: FastConformerConfig
    transducer: TransducerConfig
    # offline checkpoints standardize each mel bin over the clip; streaming
    # ones cannot (the clip isn't known yet) and don't
    normalize_features: bool = True
    # Nemotron's language prompt (the projector itself is the transducer's)
    default_prompt_id: int = 0
    prompt_dictionary: dict[str, int] = field(default_factory=dict)
    sampling_rate: int = 16_000

    @property
    def has_prompt(self) -> bool:
        return self.transducer.num_prompts > 0

    @classmethod
    def from_pretrained(cls, local_dir: str | Path, lookahead: int | None = None) -> ParakeetModelConfig:
        local_dir = Path(local_dir)
        hf = json.loads((local_dir / "config.json").read_text())
        model_type = hf["model_type"]
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(f"unsupported transducer checkpoint {model_type!r}; supported: {SUPPORTED_MODEL_TYPES}")
        gen = json.loads((local_dir / "generation_config.json").read_text())
        proc = json.loads((local_dir / "processor_config.json").read_text())
        supported = hf["encoder_config"].get("supported_num_lookahead_tokens") or []
        if lookahead is None and supported:
            # M* hears whole clips, so the widest lookahead the checkpoint was
            # trained with: the most accurate, at the same cost
            lookahead = max(supported)
        elif lookahead is not None and supported and lookahead not in supported:
            raise ValueError(f"lookahead {lookahead} is not one of the checkpoint's {supported}")
        encoder = FastConformerConfig.from_hf(hf["encoder_config"], lookahead=lookahead)
        prompts = dict(proc.get("prompt_dictionary", {}))
        return cls(
            model_type=model_type,
            encoder=encoder,
            transducer=TransducerConfig.from_hf(hf, start_token_id=gen["decoder_start_token_id"]),
            normalize_features=not encoder.causal,
            default_prompt_id=hf.get("default_prompt_id", prompts.get("auto", 0)),
            prompt_dictionary=prompts,
        )

    def prompt_id(self, language: str | None) -> int:
        """Nemotron's prompt slot for ``language``: a locale like ``en-US``, a
        bare code (``ja`` finds ``ja-JP``, ``de-AT`` finds ``de``), or
        ``auto``; any case."""
        if language is not None and not isinstance(language, str):
            raise ValueError(f"a language is a locale such as 'en-US'; got {language!r}")
        if not language or language.lower() == "auto":
            return self.default_prompt_id
        table = {name.lower(): slot for name, slot in self.prompt_dictionary.items()}
        key = language.replace("_", "-").lower()
        code = key.split("-")[0]
        # the locale, else its language (de-AT finds de)
        for name in (key, code):
            if name in table:
                return table[name]
        locales = sorted(name for name in self.prompt_dictionary if name.lower().split("-")[0] == code)
        if len({table[name.lower()] for name in locales}) == 1:
            return table[locales[0].lower()]
        if locales:
            raise ValueError(f"language {language!r} is ambiguous here; pick one of {', '.join(locales)}")
        raise ValueError(
            f"unsupported language {language!r}; supported: {', '.join(sorted(self.prompt_dictionary))}"
        )
