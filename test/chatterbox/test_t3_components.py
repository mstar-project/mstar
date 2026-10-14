"""CPU parity of the M* T3 components against the reference ``chatterbox``
package, on the real weights, for both variants.

Runs in fp32 with the fake CPU resources from ``fake_resources.py``: the
conditioning prefix, the assembled prefill sequence (cond and uncond rows),
a packed two-request prefill through the backbone and three batched decode
steps against the HF backbone's KV cache.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("chatterbox")

from chatterbox.models.t3 import T3  # noqa: E402
from chatterbox.models.t3.modules.cond_enc import T3Cond  # noqa: E402
from chatterbox.models.t3.modules.t3_config import T3Config as RefT3Config  # noqa: E402
from fake_resources import FakeT3Resources  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from mstar.model.chatterbox.components.t3 import T3Model  # noqa: E402
from mstar.model.chatterbox.config import (  # noqa: E402
    T3_ATTN,
    T3_KV,
    T3_POS,
    ChatterboxConfig,
)
from mstar.model.chatterbox.loader import (  # noqa: E402
    iter_weights,
    materialize,
    resolve_snapshot,
)

REPOS = {"chatterbox": "ResembleAI/chatterbox", "turbo": "ResembleAI/chatterbox-turbo"}
VARIANTS = ("chatterbox", "turbo")


def _snapshot(variant: str) -> str:
    try:
        return resolve_snapshot(REPOS[variant])
    except Exception as exc:  # offline and not cached
        pytest.skip(f"{REPOS[variant]} is not available locally: {exc}")


def _reference(variant: str, snapshot: str, weights_file: str) -> T3:
    if variant == "turbo":
        hp = RefT3Config(text_tokens_dict_size=50276)
        hp.llama_config_name = "GPT2_medium"
        hp.speech_tokens_dict_size = 6563
        hp.input_pos_emb = None
        hp.speech_cond_prompt_len = 375
        hp.use_perceiver_resampler = False
        hp.emotion_adv = False
        ref = T3(hp)
    else:
        ref = T3()
    ref.load_state_dict(load_file(f"{snapshot}/{weights_file}"))
    return ref.eval()


class Pair:
    """Reference model, M* model, fakes and seeded conditioning inputs."""

    def __init__(self, variant: str):
        self.variant = variant
        self.config = ChatterboxConfig.from_variant(variant)
        snapshot = _snapshot(variant)
        self.ref = _reference(variant, snapshot, self.config.t3_weights)
        with torch.device("meta"):
            model = T3Model(self.config.t3)
        self.model = materialize(model, "cpu", torch.float32)
        self.model.load_weights(iter_weights(f"{snapshot}/{self.config.t3_weights}"))
        self.model.eval()
        self.fakes = FakeT3Resources()
        self.fakes.bind(self.model, T3_ATTN, T3_KV, T3_POS)

        gen = torch.Generator().manual_seed(1234)
        t3 = self.config.t3
        self.speaker_emb = torch.randn(1, t3.speaker_embed_size, generator=gen)
        self.prompt_tokens = torch.randint(
            0, 6561, (1, t3.speech_cond_prompt_len), generator=gen
        )
        self.emotion = torch.full((1, 1, 1), 0.5)
        self.gen = gen

    def ref_cond(self) -> T3Cond:
        return T3Cond(
            speaker_emb=self.speaker_emb.clone(),
            cond_prompt_speech_tokens=self.prompt_tokens.clone(),
            emotion_adv=self.emotion.clone(),
        )

    def text_ids(self, n: int) -> torch.Tensor:
        t3 = self.config.t3
        body = torch.randint(1, t3.text_vocab_size, (n,), generator=self.gen)
        if self.variant == "turbo":
            return body
        return torch.cat([
            torch.tensor([t3.start_text_token]), body, torch.tensor([t3.stop_text_token]),
        ])


_PAIRS: dict[str, Pair] = {}


@pytest.fixture(params=VARIANTS, scope="module")
def pair(request) -> Pair:
    variant = request.param
    if variant not in _PAIRS:
        _PAIRS[variant] = Pair(variant)
    return _PAIRS[variant]


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


@torch.no_grad()
def test_conditioning_matches_reference(pair: Pair):
    ref = pair.ref.prepare_conditioning(pair.ref_cond())
    emotion = pair.emotion if pair.config.t3.emotion_adv else None
    ours = pair.model.conditioning(pair.speaker_emb, pair.prompt_tokens, emotion)
    assert ours.shape == ref.shape == (1, pair.config.t3.cond_len, pair.config.t3.hidden_size)
    diff = _max_diff(ours, ref)
    print(f"[{pair.variant}] conditioning max abs diff {diff:.2e}")
    assert diff < 1e-4


@torch.no_grad()
def _reference_prefill(pair: Pair, text_ids: torch.Tensor) -> torch.Tensor:
    """The sequence the reference sampler feeds its backbone: rows are the
    conditional and, for Chatterbox, the unconditional branch."""
    ref, t3 = pair.ref, pair.config.t3
    cond = pair.ref_cond()
    if pair.variant == "turbo":
        text = text_ids[None]
        bos = torch.full((1, 1), t3.start_speech_token, dtype=torch.long)
        embeds, _ = ref.prepare_input_embeds(
            t3_cond=cond, text_tokens=text, speech_tokens=bos, cfg_weight=0.0,
        )
        return embeds
    text = text_ids[None].repeat(2, 1)
    bos = torch.full((2, 1), t3.start_speech_token, dtype=torch.long)
    embeds, _ = ref.prepare_input_embeds(
        t3_cond=cond, text_tokens=text, speech_tokens=bos, cfg_weight=0.5,
    )
    bos_embed = ref.speech_emb(bos[:1]) + ref.speech_pos_emb.get_fixed_embedding(0)
    return torch.cat([embeds, bos_embed.repeat(2, 1, 1)], dim=1)


@torch.no_grad()
def test_prefill_embeds_match_reference(pair: Pair):
    text_ids = pair.text_ids(17)
    ref_rows = _reference_prefill(pair, text_ids)
    emotion = pair.emotion if pair.config.t3.emotion_adv else None
    cond_emb = pair.model.conditioning(pair.speaker_emb, pair.prompt_tokens, emotion)[0]
    cond_row = pair.model.build_prefill_embeds(cond_emb, text_ids, uncond=False)
    expected_len = pair.config.t3.cond_len + text_ids.shape[0] + (
        2 if pair.config.t3.duplicate_bos_in_prefill else 1
    )
    assert cond_row.shape == (expected_len, pair.config.t3.hidden_size)
    assert ref_rows.shape[1] == expected_len
    diff = _max_diff(cond_row, ref_rows[0])
    print(f"[{pair.variant}] prefill cond row max abs diff {diff:.2e}")
    assert diff < 1e-4
    if pair.variant != "turbo":
        uncond_row = pair.model.build_prefill_embeds(cond_emb, text_ids, uncond=True)
        diff = _max_diff(uncond_row, ref_rows[1])
        print(f"[{pair.variant}] prefill uncond row max abs diff {diff:.2e}")
        assert diff < 1e-4
        # the uncond text rows are the position table alone, not zeros
        n_cond = pair.config.t3.cond_len
        text_rows = uncond_row[n_cond:n_cond + text_ids.shape[0]]
        assert torch.allclose(
            text_rows, pair.model.text_pos_emb.weight[: text_ids.shape[0]], atol=1e-6
        )


@torch.no_grad()
def _ref_forward(pair: Pair, embeds: torch.Tensor, past=None):
    """HF backbone last hidden -> speech logits, with its own KV cache."""
    out = pair.ref.tfmr(inputs_embeds=embeds, past_key_values=past, use_cache=True)
    hidden = out.last_hidden_state
    return hidden, pair.ref.speech_head(hidden), out.past_key_values


@torch.no_grad()
def test_backbone_prefill_and_decode_match_reference(pair: Pair):
    t3 = pair.config.t3
    emotion = pair.emotion if t3.emotion_adv else None
    cond_emb = pair.model.conditioning(pair.speaker_emb, pair.prompt_tokens, emotion)[0]
    texts = [pair.text_ids(12), pair.text_ids(19)]
    rows = [pair.model.build_prefill_embeds(cond_emb, ids) for ids in texts]
    seq_lens = [row.shape[0] for row in rows]

    # reference: one unpadded request at a time, each with its own cache
    ref_hidden, ref_logits, pasts = [], [], []
    for row in rows:
        hidden, logits, past = _ref_forward(pair, row[None])
        ref_hidden.append(hidden[0])
        ref_logits.append(logits[0, -1])
        pasts.append(past)

    # ours: both requests packed into one step
    pair.fakes.reset()
    pos_ids = pair.fakes.plan(seq_lens)
    hidden = pair.model.hidden(torch.cat(rows), label="main", position_ids=pos_ids)
    logits = pair.model.logits(hidden)
    ends = torch.tensor(seq_lens).cumsum(0)
    starts = ends - torch.tensor(seq_lens)
    worst_hidden = worst_logits = 0.0
    for i in range(2):
        worst_hidden = max(worst_hidden, _max_diff(hidden[starts[i]:ends[i]], ref_hidden[i]))
        worst_logits = max(worst_logits, _max_diff(logits[ends[i] - 1], ref_logits[i]))
        assert logits[ends[i] - 1].argmax() == ref_logits[i].argmax()
    print(f"[{pair.variant}] prefill hidden max abs diff {worst_hidden:.2e}, logits {worst_logits:.2e}")
    assert worst_hidden < 1e-3
    assert worst_logits < 1e-3

    # three greedy decode steps, batched on our side, cached on both sides
    next_tokens = [logit.argmax() for logit in ref_logits]
    for step in range(3):
        speech_pos = torch.full((1,), step + 1, dtype=torch.long)
        step_rows, ref_step_logits = [], []
        for i in range(2):
            tok = next_tokens[i].view(1)
            emb = pair.model.embed_speech(tok, speech_pos)
            step_rows.append(emb)
            if t3.learned_pos_emb:
                ref_emb = pair.ref.speech_emb(tok[None]) + pair.ref.speech_pos_emb.get_fixed_embedding(step + 1)
            else:
                ref_emb = pair.ref.speech_emb(tok[None])
            assert _max_diff(emb, ref_emb[0]) < 1e-6
            _, logits_i, pasts[i] = _ref_forward(pair, ref_emb, pasts[i])
            ref_step_logits.append(logits_i[0, -1])
        pos_ids = pair.fakes.plan([1, 1])
        hidden = pair.model.hidden(torch.cat(step_rows), label="main", position_ids=pos_ids)
        logits = pair.model.logits(hidden)
        worst = max(_max_diff(logits[i], ref_step_logits[i]) for i in range(2))
        print(f"[{pair.variant}] decode step {step} logits max abs diff {worst:.2e}")
        assert worst < 1e-3
        for i in range(2):
            assert logits[i].argmax() == ref_step_logits[i].argmax()
            next_tokens[i] = ref_step_logits[i].argmax()


@torch.no_grad()
def test_backbone_ignores_or_requires_position_ids(pair: Pair):
    """Llama takes its positions from the resource; GPT-2 must be handed them."""
    emb = torch.zeros(3, pair.config.t3.hidden_size)
    pair.fakes.reset()
    pair.fakes.plan([3])
    if pair.config.t3.backbone.is_gpt2:
        with pytest.raises(ValueError, match="position_ids"):
            pair.model.hidden(emb, label="main")
    else:
        out = pair.model.hidden(emb, label="main")
        assert out.shape == emb.shape
