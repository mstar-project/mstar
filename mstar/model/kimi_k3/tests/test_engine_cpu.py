"""The tiny Kimi K3 / K3.1 (five layers: KDA + dense MLP, two KDA + latent MoE, two gated MLA +
MoE; residual blocks of 2) through the real ``Engine`` on CPU, against the plain-torch reference
of the checkpoint's modeling code over the same random checkpoint (``reference.py``): prefill and
decode logits, two requests batched, the context cap, and K3.1's DSA past ``index_topk``."""
import pytest
import torch

from mstar.model.kimi_k3.config import KDA_STATE, KV_CACHE, KimiK3ModelConfig
from mstar.model.kimi_k3.tests import engine_helpers as H
from mstar.model.kimi_k3.tests.reference import reference_logits

PROMPT = torch.tensor([5, 9, 13, 7, 21, 2, 40, 33], dtype=torch.long)
# fp32 end to end: the port and the reference differ only in summation order (the latent MoE sums its up
# projection's and shared experts' partials in one K-packed GEMM: up to 6e-5 off the reference's two)
ATOL = 1e-4


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    H.install_cpu_stubs(monkeypatch)
    torch.set_num_threads(2)


def _logits(engine):
    return engine._resources["sampler"].logits[-1]


def _greedy(engine, sd, cfg, prompt, steps, use_dsa=False):
    """Prefill then ``steps`` decodes of one request; each step's logits against the reference
    forward of the whole prefix. Returns the generated tokens."""
    H.add_request(engine, "a")
    tokens = [H.run_step(engine, "prefill", {"a": prompt})["a"].view(1)]
    ref = reference_logits(sd, prompt, cfg, use_dsa)
    torch.testing.assert_close(_logits(engine)[-1], ref[-1], atol=ATOL, rtol=0)
    for _ in range(steps):
        tokens.append(H.run_step(engine, "decode", {"a": tokens[-1]})["a"].view(1))
        ref = reference_logits(sd, torch.cat([prompt, *tokens[:-1]]), cfg, use_dsa)
        torch.testing.assert_close(_logits(engine)[-1], ref[-1], atol=ATOL, rtol=0)
        assert tokens[-1].item() == ref[-1].argmax().item()
    engine.remove_request("a")
    return tokens


@pytest.mark.parametrize("variant", ["reduced", "reduced_nodsa"])
def test_prefill_and_decode_match_the_reference(variant):
    engine, sub, model, sd = H.build_engine(variant)
    _greedy(engine, sd, model.config, PROMPT, steps=4)


def test_two_requests_batched_match_each_alone():
    engine, sub, model, sd = H.build_engine(seed=1)
    kv, pool = engine._resources[KV_CACHE], engine._resources[KDA_STATE]
    a, b = PROMPT, torch.tensor([44, 3, 8], dtype=torch.long)
    H.add_request(engine, "a")
    H.add_request(engine, "b")
    first = H.run_step(engine, "prefill", {"a": a, "b": b})
    seq = {"a": [first["a"].view(1)], "b": [first["b"].view(1)]}
    for _ in range(3):
        step = H.run_step(engine, "decode", {rid: seq[rid][-1] for rid in seq})
        for rid, tokens in seq.items():
            tokens.append(step[rid].view(1))
    assert sub.request_state("a").get("context") == len(a) + 3
    assert kv._streams["a"]["main"].stored_len == len(a) + 3
    assert pool.num_free_slots == pool.config.usable_slots - 2
    for rid, prompt in (("a", a), ("b", b)):
        ref = reference_logits(sd, torch.cat([prompt, *seq[rid][:-1]]), model.config)
        assert ref[-1].argmax().item() == seq[rid][-1].item()
    for rid in ("a", "b"):
        engine.remove_request(rid)
    assert pool.num_free_slots == pool.config.usable_slots and not kv._streams


def test_dsa_long_context_matches_sparse_attention_past_topk():
    """Past index_topk (8 here) each MLA query attends its top-8 keys by the indexer's score:
    the port matches the reference with that selection, and not the dense one (the control)."""
    cfg = KimiK3ModelConfig.reduced(index_topk=8, max_seq_len=40, dsa_long_context=True)
    engine, sub, model, sd = H.build_engine(config=cfg)
    prompt = PROMPT[:6]
    tokens = _greedy(engine, sd, cfg, prompt, steps=8, use_dsa=True)
    full = torch.cat([prompt, *tokens[:-1]])
    sparse, dense = reference_logits(sd, full, cfg, True), reference_logits(sd, full, cfg, False)
    assert (sparse[-1] - dense[-1]).abs().max() > 100 * ATOL


@pytest.mark.parametrize("prompt_len", [20, 33])
def test_dsa_prefill_past_topk_matches_sparse_attention(prompt_len):
    """A prompt far past index_topk (8): every prefill row from the 9th on attends its own top-8,
    through the DSA resource's prefill path, then decode goes on past it; the sparse reference
    matches, the dense one doesn't."""
    cfg = KimiK3ModelConfig.reduced(index_topk=8, max_seq_len=64, dsa_long_context=True)
    engine, sub, model, sd = H.build_engine(config=cfg)
    prompt = torch.randint(0, cfg.vocab_size, (prompt_len,), generator=torch.Generator().manual_seed(7))
    assert model.prompt_limit() == 62
    tokens = _greedy(engine, sd, cfg, prompt, steps=3, use_dsa=True)
    first = reference_logits(sd, prompt, cfg, use_dsa=True)[-1]
    assert (first - reference_logits(sd, prompt, cfg, use_dsa=False)[-1]).abs().max() > 100 * ATOL
    assert len(tokens) == 4


def test_dsa_two_requests_past_topk_read_their_own_keys():
    """Two prompts past index_topk, prefilled one after the other, then decoded in one step past
    it: every row reads its own request's keys and pages."""
    cfg = KimiK3ModelConfig.reduced(index_topk=8, max_seq_len=64, dsa_long_context=True)
    engine, sub, model, sd = H.build_engine(config=cfg, seed=3)
    g = torch.Generator().manual_seed(8)
    a = torch.randint(0, cfg.vocab_size, (14,), generator=g)
    b = torch.randint(0, cfg.vocab_size, (19,), generator=g)
    H.add_request(engine, "a")
    tok_a = H.run_step(engine, "prefill", {"a": a})["a"].view(1)
    H.add_request(engine, "b")
    # the engine batches by walk: b's prefill, then a's decode, each past topk
    tok_b = H.run_step(engine, "prefill", {"b": b})["b"].view(1)
    ref_b = reference_logits(sd, b, cfg, use_dsa=True)[-1]
    torch.testing.assert_close(engine._resources["sampler"].logits[-1][-1], ref_b, atol=2 * ATOL, rtol=0)
    step = H.run_step(engine, "decode", {"a": tok_a, "b": tok_b})
    logits = engine._resources["sampler"].logits[-1]
    for row, (prompt, tok) in enumerate(((a, tok_a), (b, tok_b))):
        ref = reference_logits(sd, torch.cat([prompt, tok]), cfg, use_dsa=True)[-1]
        torch.testing.assert_close(logits[row], ref, atol=2 * ATOL, rtol=0)
    assert set(step) == {"a", "b"}


def test_context_cap_fails_only_that_request_and_decode_stops_inside_it(caplog):
    engine, sub, model, sd = H.build_engine(seed=2)
    limit = model.config.context_limit  # 64
    H.add_request(engine, "long")
    assert H.failed_requests(engine, "prefill", {"long": torch.zeros(limit + 1, dtype=torch.long)}) == {"long"}
    assert "exceeds the 64-token limit" in caplog.text
    engine.remove_request("long")
    H.add_request(engine, "r")
    token = H.run_step(engine, "prefill", {"r": torch.arange(limit - 5)})["r"]
    for i in range(limit):
        token = H.run_step(engine, "decode", {"r": token.view(1)})["r"]
        info = H._info("r", "decode", max_tokens=10 * limit)
        info.dynamic_loop_iter_counts["decode_loop"] = i
        if sub._stops("r", info, token.item()):
            break
    # 1 prefill token + 4 decodes: one short of the cap, so the step already in flight fits
    assert i == 3 and sub.request_state("r").get("context") == limit - 1
    H.run_step(engine, "decode", {"r": token.view(1)})
    assert sub.request_state("r").get("context") == limit


@pytest.mark.parametrize("dsa", [False, True])
def test_a_prompt_prefilled_in_chunks_matches_it_whole(dsa):
    """Chunked prefill as the engine cuts it: one request's prompt in three consecutive prefill
    steps continues the KDA state, the latent cache and (K3.1, past an index_topk of 8) the index
    keys where the last chunk left them; the final chunk's logits and the decode after it match
    the reference over the whole prompt."""
    cfg = (KimiK3ModelConfig.reduced(index_topk=8, max_seq_len=64, dsa_long_context=True) if dsa
           else KimiK3ModelConfig.reduced())
    engine, sub, model, sd = H.build_engine(config=cfg, seed=4)
    prompt = torch.randint(0, cfg.vocab_size, (29,), generator=torch.Generator().manual_seed(9))
    H.add_request(engine, "a")
    for start, end in ((0, 11), (11, 20), (20, 29)):
        tok = H.run_step(engine, "prefill", {"a": prompt[start:end]})["a"].view(1)
    assert sub.request_state("a").get("context") == 29
    ref = reference_logits(sd, prompt, cfg, use_dsa=dsa)[-1]
    torch.testing.assert_close(engine._resources["sampler"].logits[-1][-1], ref, atol=2 * ATOL, rtol=0)
    assert tok.item() == ref.argmax().item()
    H.run_step(engine, "decode", {"a": tok})
    ref = reference_logits(sd, torch.cat([prompt, tok]), cfg, use_dsa=dsa)[-1]
    torch.testing.assert_close(engine._resources["sampler"].logits[-1][-1], ref, atol=2 * ATOL, rtol=0)


def test_prefill_runs_in_chunks_of_the_configured_size():
    from mstar.model.kimi_k3.kimi_k3_model import KimiK31Model

    model = KimiK31Model("x", config_variant="reduced", tokenizer_mode="byte", prefill_chunk_tokens=4096)
    (policy,) = model.chunking_policy_of().values()
    assert policy.chunkable and policy.max_batch_tokens == 4096
    assert type(model.get_graph_walk_graphs()["prefill"]).__name__ == "Loop"
    assert KimiK31Model("x", config_variant="reduced", prefill_chunk_tokens=0).chunking_policy_of() == {}


def test_a_rows_logits_do_not_depend_on_the_rows_beside_it():
    """Decode of three requests alone and beside a fourth (what a 3-row batch replaying a 4-row graph's
    padding row amounts to, row for row): their rows agree to fp32 rounding, so a gap between a padded
    replay and a 3-row eager step is the kernels' rounding at another batch shape, not one row reaching
    into another (MoE routing, KDA slots, the MLA cache, the attention-residual reads are all per row)."""
    cfg = KimiK3ModelConfig.reduced()
    g = torch.Generator().manual_seed(11)
    prompts = {f"r{i}": torch.randint(0, cfg.vocab_size, (n,), generator=g) for i, n in enumerate((7, 12, 9))}
    extra = torch.randint(0, cfg.vocab_size, (10,), generator=g)
    rows = {}
    for label, batch in (("alone", prompts), ("beside", {**prompts, "x": extra})):
        engine, sub, model, sd = H.build_engine(config=cfg, seed=6)
        for rid in batch:
            H.add_request(engine, rid)
        tok = H.run_step(engine, "prefill", batch)
        steps = []
        for _ in range(3):
            tok = H.run_step(engine, "decode", {rid: tok[rid].view(1) for rid in batch})
            steps.append(engine._resources["sampler"].logits[-1][:3].clone())
        rows[label] = torch.stack(steps)
    # fp32 rounding at another batch shape (up to 1.6e-5): a row reaching into another moves it by far more
    torch.testing.assert_close(rows["beside"], rows["alone"], atol=3e-5, rtol=0)


@pytest.mark.parametrize("dsa", [False, True])
def test_a_prompt_sharing_a_cached_prefix_skips_it_and_matches_the_reference(dsa):
    """Prefix reuse through the engine: a second prompt sharing the first's two pages restores the
    KDA state kept at the second page's end and runs only its own tokens over the latent pages
    (and, K3.1 past an index_topk of 8, the index keys) the first wrote; its prefill logits and
    the decode after it match the reference over the whole prompt, and its context counts the
    skipped tokens."""
    cfg = (KimiK3ModelConfig.reduced(index_topk=8, max_seq_len=256, dsa_long_context=True) if dsa
           else KimiK3ModelConfig.reduced(dsa=False, max_seq_len=256))
    engine, sub, model, sd = H.build_engine(config=cfg, seed=5, prefix_slots=4)
    gen = torch.Generator().manual_seed(11)
    first = torch.randint(0, 250, (150,), generator=gen)
    second = torch.cat([first[:128], torch.randint(0, 250, (20,), generator=gen)])
    cuts = []
    split = sub.split_inputs

    def spy(graph_walk, fwd_info, inputs, start, end):
        cuts.append((fwd_info.rid_handle, start, end))
        return split(graph_walk, fwd_info, inputs, start, end)

    sub.split_inputs = spy
    H.add_keyed_request(engine, model, "a", first)
    tok = H.run_step(engine, "prefill", {"a": first})["a"].view(1)
    H.run_step(engine, "decode", {"a": tok})
    engine.remove_request("a")
    assert not cuts, "the first prompt found a prefix in an empty cache"

    H.add_keyed_request(engine, model, "b", second)
    tok = H.run_step(engine, "prefill", {"b": second})["b"].view(1)
    assert cuts == [("b", 128, 148)], "b didn't skip the two pages a cached"
    assert sub.request_state("b").get("context") == 148
    ref = reference_logits(sd, second, cfg, use_dsa=dsa)
    torch.testing.assert_close(_logits(engine)[-1], ref[-1], atol=2 * ATOL, rtol=0)
    assert tok.item() == ref[-1].argmax().item()
    H.run_step(engine, "decode", {"b": tok})
    ref = reference_logits(sd, torch.cat([second, tok]), cfg, use_dsa=dsa)
    torch.testing.assert_close(_logits(engine)[-1], ref[-1], atol=2 * ATOL, rtol=0)
    assert sub.request_state("b").get("context") == 149
