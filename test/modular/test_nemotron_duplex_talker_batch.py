"""The talker advances N sessions in one backbone pass exactly like N sequential
single-session steps: unequal caches are right-padded and masked, RoPE positions
are per row, and each row draws its own seeded noise. Tiny random weights, fp32,
CPU."""
from dataclasses import replace

import pytest
import torch

from mstar.model.nemotron_duplex.components.eartts_talker import EarTTSTalker
from mstar.model.nemotron_duplex.config import EarTTSConfig

TINY = replace(
    EarTTSConfig(),
    hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=2,
    num_key_value_heads=2, head_dim=16, num_quantizers=4, codebook_size=16, code_dim=8,
    mog_num_predictions=8, mog_low_rank=4, mog_num_layers=1, mog_intermediate_size=32,
)
S2C = {5: (10, 11), 7: (12,), 9: (13, 14, 15), 11: (16,), 13: (17, 18)}
CHAR_PAD = 256
SAMPLING = dict(text_eos_id=2, num_iter=4, guidance_scale=0.5, noise_scale=0.8, top_p=0.8)


def make_talker(seed=0) -> EarTTSTalker:
    torch.manual_seed(seed)
    t = EarTTSTalker(TINY)
    for name, prm in t.named_parameters():
        if prm.is_floating_point():
            prm.data.normal_(0, 0.1)
        elif "is_continuation" in name:
            prm.data.copy_(torch.randint(0, 2, prm.shape))
        elif "special_flags" in name:
            prm.data.copy_(torch.randint(0, 3, prm.shape))
        else:  # control codes, codec silence tokens
            prm.data.copy_(torch.arange(prm.numel()).view(prm.shape) % TINY.codebook_size)
    return t.eval()


def fresh_state(t: EarTTSTalker) -> dict:
    st = t.init_state(1, speaker="Aria", subword_id_to_char_ids=S2C, char_pad_idx=CHAR_PAD,
                      text_pad_id=12, text_eos_id=2, speech_pad_id=TINY.codebook_size)
    st["_prev_codes"] = st["prev_codes"]
    return st


def seq_step(t, state, tok, gen):
    cur = torch.tensor([[tok]])
    cond = t.text_conditioning(cur, torch.ones_like(cur, dtype=torch.bool), S2C, CHAR_PAD)
    codes, new = t.infer_codes_one_step(state, cur, cur, state["_prev_codes"], cond=cond,
                                        generator=gen, **SAMPLING)
    new["_prev_codes"] = codes
    return codes, new


def assert_states_close(a, b):
    assert a["pos"] == b["pos"]
    for key in ("kv", "kv_uncond"):
        for (ka, va), (kb, vb) in zip(a[key], b[key], strict=True):
            torch.testing.assert_close(ka, kb, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(va, vb, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(a["_prev_codes"].reshape(-1), b["_prev_codes"].reshape(-1))


def test_text_conditioning_from_host_ids_matches():
    t = make_talker()
    ids = [5, 7, 9, 99]   # 99 has no char mapping
    batched = t.text_conditioning_from_ids(ids, S2C, CHAR_PAD)
    for i, tok in enumerate(ids):
        cur = torch.tensor([[tok]])
        one = t.text_conditioning(cur, torch.ones_like(cur, dtype=torch.bool), S2C, CHAR_PAD)
        torch.testing.assert_close(batched[i:i + 1], one)


def test_batched_step_matches_sequential_with_unequal_caches():
    t = make_talker()
    # session A: warm-up + 2 steps (pos P+2); session B: warm-up + 1 step (pos P+1); C: fresh
    ga, gb, gc = (torch.Generator().manual_seed(s) for s in (1, 2, 3))
    a = fresh_state(t)
    _, a = seq_step(t, a, 5, ga)
    _, a = seq_step(t, a, 7, ga)
    b = fresh_state(t)
    _, b = seq_step(t, b, 9, gb)
    c = fresh_state(t)
    assert a["pos"] != b["pos"] != c["pos"]

    # reference: each session steps alone, with a copy of its generator state
    def clone_gen(g):
        h = torch.Generator()
        h.set_state(g.get_state())
        return h
    refs = [seq_step(t, s, tok, clone_gen(g)) for s, tok, g in ((a, 11, ga), (b, 13, gb), (c, 7, gc))]

    ids = torch.tensor([11, 13, 7])
    conds = t.text_conditioning_from_ids(ids.tolist(), S2C, CHAR_PAD)
    codes, news = t.infer_codes_batched([a, b, c], ids, conds, generators=[ga, gb, gc], **SAMPLING)
    assert codes.shape == (3, TINY.num_quantizers)
    for i, (ref_codes, ref_state) in enumerate(refs):
        torch.testing.assert_close(codes[i], ref_codes.reshape(-1))
        assert_states_close(news[i], ref_state)


def test_batched_step_forces_silence_on_eos():
    t = make_talker()
    a, b = fresh_state(t), fresh_state(t)
    ga, gb = torch.Generator().manual_seed(4), torch.Generator().manual_seed(5)
    ids = torch.tensor([SAMPLING["text_eos_id"], 5])
    conds = t.text_conditioning_from_ids(ids.tolist(), S2C, CHAR_PAD)
    ref_a, _ = seq_step(t, a, SAMPLING["text_eos_id"], torch.Generator().manual_seed(4))
    ref_b, _ = seq_step(t, b, 5, torch.Generator().manual_seed(5))
    codes, _ = t.infer_codes_batched([a, b], ids, conds, generators=[ga, gb], **SAMPLING)
    torch.testing.assert_close(codes[0], ref_a.reshape(-1))
    torch.testing.assert_close(codes[1], ref_b.reshape(-1))


def test_batched_step_without_guidance_drops_uncond_like_sequential():
    t = make_talker()
    a = fresh_state(t)
    kw = dict(SAMPLING, guidance_scale=0.0)
    ids = torch.tensor([5])
    conds = t.text_conditioning_from_ids([5], S2C, CHAR_PAD)
    codes, news = t.infer_codes_batched([a], ids, conds, generators=[torch.Generator().manual_seed(6)], **kw)
    cur = torch.tensor([[5]])
    ref, ref_state = t.infer_codes_one_step(a, cur, cur, a["_prev_codes"], cond=conds,
                                            generator=torch.Generator().manual_seed(6), **kw)
    torch.testing.assert_close(codes[0], ref.reshape(-1))
    assert news[0]["kv_uncond"] is None and ref_state["kv_uncond"] is None


@pytest.mark.parametrize("rows", [1, 3])
def test_per_row_noise_is_batch_independent(rows):
    """A row's noise draws do not depend on how many rows share the batch."""
    from mstar.model.nemotron_duplex.components.eartts_talker import _rand_rows

    gens = [torch.Generator().manual_seed(10 + i) for i in range(rows)]
    batched = _rand_rows((rows, 2, 5), torch.device("cpu"), torch.float32, gens, torch.randn)
    for i in range(rows):
        alone = torch.randn((1, 2, 5), generator=torch.Generator().manual_seed(10 + i))
        torch.testing.assert_close(batched[i:i + 1], alone)
