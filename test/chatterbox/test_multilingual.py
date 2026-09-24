"""Chatterbox Multilingual against the reference package: the grapheme
tokenizer's ids per language and the T3 greedy tokens through the engine's
calling sequence. Needs the ``ResembleAI/chatterbox`` snapshot (the
multilingual files ship in the same repo) and the ``chatterbox`` package."""

from __future__ import annotations

import importlib.util
import os
from types import SimpleNamespace

import pytest
import torch

chatterbox = pytest.importorskip("chatterbox")
from chatterbox.models.t3 import T3 as RefT3  # noqa: E402, N811
from chatterbox.models.t3.modules.cond_enc import T3Cond  # noqa: E402
from chatterbox.models.t3.modules.t3_config import T3Config as RefT3Config  # noqa: E402
from chatterbox.models.tokenizers import MTLTokenizer  # noqa: E402
from chatterbox.mtl_tts import punc_norm as ref_punc_norm  # noqa: E402
from fake_resources import FakeT3Resources  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from test_t3_pipeline import FakeGreedySampler, _reference_greedy  # noqa: E402

from mstar.model.chatterbox.components.text import (  # noqa: E402
    SUPPORTED_LANGUAGES,
    MultilingualTextTokenizer,
    punc_norm,
)
from mstar.model.chatterbox.config import ChatterboxConfig  # noqa: E402
from mstar.model.chatterbox.loader import resolve_snapshot  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")

SENTENCES = {
    "en": "The quick brown fox jumps over the lazy dog",
    "de": "Der schnelle braune Fuchs springt über den faulen Hund.",
    "fr": "Le renard brun rapide saute par-dessus le chien paresseux!",
    "es": "¿Dónde está la biblioteca? Está a la vuelta de la esquina.",
    "ko": "안녕하세요, 만나서 반갑습니다.",
    "hi": "नमस्ते, आप कैसे हैं?",
    "ar": "مرحبا بكم في المدينة.",
    "el": "Καλημέρα, τι κάνεις σήμερα;",
    "zh": "你好，今天天气很好。",
    "ja": "今日はいい天気ですね。",
    "ru": "Привет, как дела?",
    "he": "שלום, מה שלומך?",
}


@pytest.fixture(scope="module")
def snapshot() -> str:
    try:
        snap = resolve_snapshot("ResembleAI/chatterbox")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"checkpoint not available: {exc}")
    if not os.path.isfile(f"{snap}/grapheme_mtl_merged_expanded_v1.json"):
        pytest.skip("multilingual tokenizer file is not in the snapshot")
    return snap


@pytest.fixture(scope="module")
def tokenizers(snapshot):
    ours = MultilingualTextTokenizer(
        f"{snapshot}/grapheme_mtl_merged_expanded_v1.json", f"{snapshot}/Cangjie5_TC.json",
    )
    theirs = MTLTokenizer(f"{snapshot}/grapheme_mtl_merged_expanded_v1.json")
    if not theirs.cangjie_converter.word2cj:
        # the package fetches the table through the hub cache, which is off here
        theirs.cangjie_converter.word2cj = ours.cangjie.word2cj
        theirs.cangjie_converter.cj2word = ours.cangjie.cj2word
    return ours, theirs


def _reference_ids(theirs: MTLTokenizer, text: str, lang: str) -> list[int]:
    ids = theirs.text_to_tokens(ref_punc_norm(text), language_id=lang)[0].tolist()
    return [255, *ids, 0]


@pytest.mark.parametrize("lang", sorted(SENTENCES))
def test_tokenizer_matches_reference_per_language(tokenizers, lang):
    """Same ids as the package's tokenizer, including the per-language step
    (both sides skip a step whose optional package is missing)."""
    ours, theirs = tokenizers
    text = SENTENCES[lang]
    assert punc_norm(text, multilingual=True) == ref_punc_norm(text)
    got = ours(text, language_id=lang).tolist()
    assert got == _reference_ids(theirs, text, lang)
    assert got[0] == 255 and got[-1] == 0
    vocab = ours.tokenizer.get_vocab()
    assert got[1] == vocab[f"[{lang}]"]  # the language token leads
    # kanji are only in the vocabulary as kana: without pykakasi they are
    # unknown tokens on both sides, as in the reference
    if lang != "ja" or importlib.util.find_spec("pykakasi") is not None:
        assert vocab["[UNK]"] not in got[1:-1], f"{lang}: unknown tokens for {text!r}"


def test_chinese_is_spelled_in_cangjie_codes(tokenizers):
    ours, _ = tokenizers
    spelled = ours.preprocess("你好", "zh")
    assert spelled.startswith("[zh]") and spelled.count("[cj_.]") == 2
    vocab = ours.tokenizer.get_vocab()
    assert all(tok in vocab for tok in ("[cj_.]", "[cj_a]", "[cj_o]"))


def test_unknown_language_is_refused(tokenizers):
    ours, _ = tokenizers
    with pytest.raises(ValueError, match="Unsupported language_id"):
        ours("Hello", language_id="xx")
    assert set(SUPPORTED_LANGUAGES) == set(chatterbox.mtl_tts.SUPPORTED_LANGUAGES)
    no_lang = ours("Hello", language_id=None).tolist()
    assert no_lang[1] != ours.tokenizer.get_vocab()["[en]"]


def test_t3_greedy_tokens_match_reference(snapshot, tokenizers):
    """Greedy T3 speech tokens for a German sentence, our node against the
    reference ``T3.inference`` loop with argmax sampling (as in
    ``test_t3_pipeline`` for the English and Turbo checkpoints)."""
    from mstar.model.chatterbox.components.t3 import T3Model
    from mstar.model.chatterbox.config import T3_ATTN, T3_KV, T3_POS, T3_SAMPLER
    from mstar.model.chatterbox.loader import iter_weights
    from mstar.model.chatterbox.submodules import (
        PREV_TOKEN,
        SPEECH_TOKENS,
        TEXT_INPUTS,
        BuiltinT3Voice,
        T3Submodule,
    )
    from mstar.model.submodule_base import ModelInputsFromEngine

    ours, _ = tokenizers
    config = ChatterboxConfig.multilingual()
    assert config.t3.text_vocab_size == 2454 and config.t3.cond_len == 34
    text_tokens = ours(SENTENCES["de"], language_id="de")
    n_steps, cfg_weight, exaggeration = 12, 0.5, 0.5

    ref = RefT3(RefT3Config.multilingual())
    ref.load_state_dict(load_file(f"{snapshot}/{config.t3_weights}"), strict=False)
    ref.eval()
    conds = torch.load(f"{snapshot}/conds.pt", map_location="cpu", weights_only=True)["t3"]
    cond = T3Cond(
        speaker_emb=conds["speaker_emb"], cond_prompt_speech_tokens=conds["cond_prompt_speech_tokens"],
        emotion_adv=exaggeration * torch.ones(1, 1, 1),
    )

    model = T3Model(config.t3)
    model.load_weights(iter_weights(f"{snapshot}/{config.t3_weights}"))
    model.eval()
    builtin = BuiltinT3Voice(
        speaker_emb=conds["speaker_emb"].reshape(-1), prompt_tokens=conds["cond_prompt_speech_tokens"].reshape(-1),
    )
    sub = T3Submodule(model, config, builtin_voice=builtin)
    res = FakeT3Resources()
    res.bind(model, T3_ATTN, T3_KV, T3_POS)
    resources = {T3_ATTN: res.kv_attn, T3_KV: res.kv_attn, T3_POS: res.pos, T3_SAMPLER: FakeGreedySampler()}
    info = SimpleNamespace(
        request_id="r",
        step_metadata={"cfg_weight": cfg_weight, "exaggeration": exaggeration, "min_p": 0.05,
                       "max_new_tokens": 1000, "is_prefill": True},
        resource_configs={T3_SAMPLER: SimpleNamespace(temperature=0.0, ignore_eos=False)},
        max_tokens=4096, random_seed=0,
    )
    engine = ModelInputsFromEngine(request_ids=["r"], per_request_info={"r": info}, resources=resources)

    def run(walk, inputs):
        prepared = sub.prepare_inputs(walk, info, inputs)
        step = sub.declare_step(walk, ["r"], [prepared])
        res.plan([seg.span for seg in step.segments])
        packed = sub.preprocess(walk, engine, [prepared])
        out = sub.forward(walk, engine, **packed)
        sub.postprocess("r", info, out)
        return out[SPEECH_TOKENS][0]

    with torch.no_grad():
        expected = _reference_greedy(ref, cond, text_tokens, cfg_weight, n_steps)
        got = [int(run("prefill", {TEXT_INPUTS: [text_tokens]}))]
        prev = torch.tensor([got[-1]])
        for _ in range(n_steps - 1):
            prev = run("decode", {PREV_TOKEN: [prev]})
            got.append(int(prev))
    print(f"[multilingual de] ref={expected}\n mstar={got}")
    assert got == expected
