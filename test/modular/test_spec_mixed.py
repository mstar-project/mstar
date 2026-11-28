"""Mixed prompt-plus-verify steps: a verifying node's prompts run in its verify steps (`draft_mixed_step`), the
verify rows first, k + 1 tokens each, then the prompt rows. On the toy target (`spec_toy`) through the real
engine's KV, position and sampler resources: a mixed step's verify rows verify, and take back, as in a verify
step of their own, and a prompt row leaves with its first token, as from a prompt step."""
from __future__ import annotations

import sys

import pytest
import torch

sys.path.insert(0, ".")

from spec_toy import NODE, STOP, Driver, ToyModel, carry_oracle, engine, every, oracle, reference  # noqa: E402

from mstar.engine.resources.draft.config import DraftMode  # noqa: E402
from mstar.engine.resources.draft.drafter import ChunkBoundary, DrafterNeeds, mixed_rows  # noqa: E402
from mstar.engine.resources.draft.startup import combined_problem, resolve_mode  # noqa: E402
from mstar.engine.resources.kv import manager as manager_mod  # noqa: E402
from mstar.engine.resources.sampler.verify import chain_verify_rows, greedy_verify  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

K = 3
PROMPTS = {1: [5, 9, 2], 2: [17], 3: [1, 2, 3, 4, 5, 6, 7, 8, 9]}
LATE = {4: [11, 13, 2, 8, 30], 5: [40], 6: [3, 3, 3, 3, 3, 3, 3]}
MAX = 14


class _StubTransfer:
    def __init__(self, *args, **kwargs):
        pass

    def get_kv_transfer_info(self, **kwargs):
        pass

    def remove_request(self, request_id):
        pass

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    # every take-back keeps a row's position at its KV length plus a fixed offset (`check_offsets`)
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)


# ── the layout ──


def test_the_layout_reads_each_rows_block_and_a_prompt_rows_last_token():
    block = K + 1
    # two verify rows, a prompt row of 3 tokens, an empty padding row
    kind = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, -1, -1, -1])
    qo = torch.tensor([0, 4, 8, 11, 11])
    rows = mixed_rows(qo, kind, block)
    assert rows.prompt.tolist() == [False, False, True, True]
    # the padding row reads the last real token, so it is marked for writes to skip
    assert rows.at.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [10, 10, 10, 10], [10, 10, 10, 10]]
    assert rows.empty.tolist() == [False, False, False, True]
    assert rows.tokens == 11


def test_a_refused_row_tests_no_draft_and_samples_its_first_position():
    """Greedy: the argmax at position 0. Sampled: a plain draw from p_0 at the bonus number, its draft
    never taken out of p."""
    torch.manual_seed(0)
    n, vocab = 4, 16
    logits = torch.randn(n, K + 1, vocab)
    drafts = logits.argmax(-1)[:, :K]  # drafts the target would accept in full
    refuse = torch.tensor([False, True, False, True])
    tokens, accepted = greedy_verify(logits.reshape(-1, vocab), drafts, refuse)
    assert accepted.tolist() == [K, 0, K, 0]
    assert torch.equal(tokens[refuse, 0], logits[refuse, 0].argmax(-1))
    probs = torch.softmax(logits, -1)
    seed, offset = torch.arange(n), torch.zeros(n, dtype=torch.long)
    tokens, accepted = chain_verify_rows(probs.reshape(-1, vocab), n, drafts, seed, offset, refuse)
    assert accepted[refuse].tolist() == [0, 0]
    from mstar.engine.resources.sampler.verify import _inverse_cdf, _uniforms

    u = _uniforms(seed, offset, K + 1, probs.dtype)[:, K]
    want = _inverse_cdf(probs[:, 0], u)
    assert torch.equal(tokens[refuse, 0], want[refuse])


def test_a_mixed_step_hides_its_verify_rows_last_keys_alone():
    """A drafter attention that hides each block's last row (`AttentionStep.hide_last`, DSpark's) hides a mixed
    step's verify rows' alone: a one-token prompt with nothing before it, and an empty padding row, keep their
    keys; a verify step still refuses a row it can't hide."""
    from mstar.engine.resources.kv.plan import SequenceView, hide_tail_views

    verify = SequenceView(1, "main", [0, 1], length=7, to_compute=K + 1)
    prompt = SequenceView(4, "main", [2], length=1, to_compute=1)
    padding = SequenceView(-1, "main", [], length=0, to_compute=0)
    out = hide_tail_views([verify, prompt, padding], 1, 4, rows=[True, False])
    assert out[0].length == 6 and out[1:] == [prompt, padding]
    with pytest.raises(ValueError, match="hide_last=1 on a row of 0 new tokens"):
        hide_tail_views([verify, padding], 1, 4)


# ── through the engine ──


def _verified(driver, prompts, steps_before, temperature, seed, filters):
    """``prompts`` prefilled and verified ``steps_before`` steps; the rows still running."""
    for rid, prompt in prompts.items():
        driver.add(rid, prompt, MAX, temperature=temperature, seed=seed + rid, **filters)
    live = list(prompts)
    for _ in range(steps_before):
        stopped = driver.step(live)
        live = [rid for rid in live if not stopped[rid]]
    return live


def _separate(prompts, late, drafter=oracle, steps_before=2, temperature=0.0, seed=0, filters=None):
    """``prompts`` prefilled and verified ``steps_before`` steps, then ``late``'s prompt steps on their own.
    ``filters``: every request's top_k, top_p, min_p and top_p_first."""
    filters = filters or {}
    driver = Driver(engine(K, drafter, hf_filters=_hf(filters)))
    live = _verified(driver, prompts, steps_before, temperature, seed, filters)
    for rid, prompt in late.items():
        driver.add(rid, prompt, MAX, temperature=temperature, seed=seed + rid, **filters)
    driver.run(live + list(late))
    return driver.out


def _mixed(prompts, late, drafter=oracle, steps_before=2, temperature=0.0, seed=0, filters=None):
    """The same, ``late``'s prompts riding in one mixed step with ``prompts``' verify rows."""
    filters = filters or {}
    driver = Driver(engine(K, drafter, mixed=True, hf_filters=_hf(filters)))
    live = _verified(driver, prompts, steps_before, temperature, seed, filters)
    for rid in late:
        driver.add(rid, None, MAX, temperature=temperature, seed=seed + rid, **filters)
    stopped = driver.mixed_step(live, late)
    driver.run([rid for rid in live + list(late) if not stopped[rid]])
    return driver, stopped


@pytest.mark.parametrize(
    "drafter", [oracle, every(2), every(1), carry_oracle], ids=["oracle", "every-2nd-wrong", "all-wrong", "carrying"],
)
def test_a_mixed_steps_rows_decode_as_in_steps_of_their_own(drafter):
    """Greedy parity: every reply is the target's, as when the late prompts ran their own prompt steps; the
    verify rows' take-backs keep positions at their KV lengths through every step (`check_offsets`)."""
    driver, _ = _mixed(PROMPTS, LATE, drafter)
    want = {rid: reference(prompt, MAX) for rid, prompt in {**PROMPTS, **LATE}.items()}
    assert driver.out == want
    assert driver.out == _separate(PROMPTS, LATE, drafter)


def _hf(filters: dict) -> bool:
    # by name, so a request that sets top_p_first False runs on the sampler one that sets it True does
    return "min_p" in filters or "top_p_first" in filters


def _top_p_first_on_the_cpu(monkeypatch):
    """top_p_first is CUDA-only: the sampler refuses it on the CPU, and the verify filter raises on an order.
    Lift the refusal, and stand in for the CUDA filter with torch on the rows that ask: top-p, then top-k."""
    from mstar.engine.resources.sampler import verify
    from mstar.engine.resources.sampler.resource import SamplerResource

    plain, problem = verify.filtered_probs, SamplerResource._generation_controls_problem

    def ordered(logits, temperature, top_k, top_p, min_p=None, order=None):
        k_first = plain(logits, temperature, top_k, top_p, min_p)
        if order is None:
            return k_first
        p_first = plain(logits, temperature, torch.zeros_like(top_k), top_p, min_p)
        ranked, at = p_first.sort(dim=-1, descending=True, stable=True)
        k = torch.where(top_k > 0, top_k, logits.shape[-1])
        ranked = torch.where(torch.arange(logits.shape[-1])[None, :] < k[:, None], ranked, 0.0)
        p_first = torch.zeros_like(ranked).scatter(-1, at, ranked)
        p_first = p_first / p_first.sum(-1, keepdim=True)
        return torch.where(order.top_p_first[:, None], p_first, k_first)

    monkeypatch.setattr(verify, "filtered_probs", ordered)
    monkeypatch.setattr(
        SamplerResource, "_generation_controls_problem",
        lambda self, cfg: None if "CUDA-only" in (why := problem(self, cfg) or "CUDA-only") else why,
    )


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("filters", [
    {}, {"top_k": 20, "top_p": 0.7}, {"min_p": 0.5}, {"top_k": 20, "top_p": 0.7, "top_p_first": True},
], ids=["unfiltered", "top-k-top-p", "min-p", "top-p-first"])
@pytest.mark.parametrize("drafter", [oracle, every(2)], ids=["oracle", "every-2nd-wrong"])
def test_a_seeded_reply_is_the_same_whether_its_prompt_rode_a_mixed_step(monkeypatch, drafter, filters, seed):
    """A prompt step's token is drawn as a mixed step's prompt row draws it, from the request's own stream at
    the same number and through the request's own filters (top-k, top-p, min-p and their order), so a seeded
    reply is the same whichever its prompt rode. At T=20 the toy's one-hot logits leave every token a few
    percent, so a different draw shows: top-p first keeps 20 tokens here where top-k first keeps 14, and
    min-p 0.5 keeps the argmax alone."""
    if filters.get("top_p_first"):
        _top_p_first_on_the_cpu(monkeypatch)
    driver, _ = _mixed(PROMPTS, LATE, drafter, temperature=20.0, seed=seed, filters=filters)
    assert driver.out == _separate(PROMPTS, LATE, drafter, temperature=20.0, seed=seed, filters=filters)


@pytest.mark.parametrize("path", ["mixed", "separate"])
def test_top_p_first_changes_some_seeded_reply_on_a_verifying_node(monkeypatch, path):
    """The case above compares the two paths with each other, so a verifying node that dropped the filter
    order on both would pass it. Over its seeds, some reply with top-p first differs from the same request's
    with top-k first, on either path."""
    _top_p_first_on_the_cpu(monkeypatch)
    run = _mixed if path == "mixed" else _separate

    def replies(top_p_first: bool) -> list:
        filters = {"top_k": 20, "top_p": 0.7, "top_p_first": top_p_first}
        outs = [run(PROMPTS, LATE, oracle, temperature=20.0, seed=seed, filters=filters) for seed in range(8)]
        return [out[0].out if path == "mixed" else out for out in outs]

    assert replies(True) != replies(False)


def test_a_carrying_drafter_reads_its_carry_and_stores_a_prompt_rows_first():
    """The carrying oracle drafts off its carry alone: it is accepted in full on the mixed step's verify row,
    and on the late prompt's first verify step, whose carry the mixed step stored."""
    driver = Driver(engine(K, carry_oracle, mixed=True))
    driver.add(1, PROMPTS[1], MAX)
    driver.step([1])
    driver.add(4, None, MAX)
    draft = driver.eng._drafts[NODE]
    before = draft.accepted
    driver.mixed_step([1], {4: LATE[4]})
    assert draft.accepted == before + K
    before = draft.accepted
    driver.step([1, 4])
    assert draft.accepted == before + 2 * K


def test_a_mixed_steps_captures_read_the_carry_staged():
    """A captured mixed step reads its rows' carry as a verify step's does, staged off the masters
    (`bind_graph_reads`), so a growth of the masters between capture and replay never strands its read."""
    from mstar.engine.resources.draft.drafter import DRAFT_CARRY

    reads = engine(K, carry_oracle, mixed=True)._resources["sampler"]._graph_reads
    assert DRAFT_CARRY in reads["decode"] and DRAFT_CARRY in reads["mixed"]


@pytest.mark.parametrize("mode", ["synchronous", "asynchronous"])
def test_a_node_whose_resources_correct_a_verify_steps_state_refuses_mixed_steps(monkeypatch, mode):
    """A resource that corrects a verify step's state in its graph (GDN's) reads every row as a verify
    block, which a mixed step's prompt rows aren't: the node refuses at load in either mode, before any
    resource binds its verify walk, and the refusal offers no mode; without a mixed walk it loads."""
    from mstar.engine.resources.position.manager import RopeManager

    bound = []
    monkeypatch.setattr(RopeManager, "correct_verify_state", lambda self, result: None)
    monkeypatch.setattr(RopeManager, "bind_verify_walk", lambda self, *args: bound.append(args))
    with pytest.raises(ValueError, match=r"resources \['rope'\] correct a verify step's state in its graph") as info:
        engine(K, oracle, mode=mode, mixed=True)
    assert "asynchronously" not in str(info.value) and "'mixed'" in str(info.value)
    assert not bound, "refused after binding"
    engine(K, oracle, mode=mode)


def test_a_prompt_row_leaves_with_its_first_token_and_counts_no_verify_row():
    driver = Driver(engine(K, oracle, mixed=True))
    driver.add(1, PROMPTS[1], MAX)
    driver.step([1])
    draft = driver.eng._drafts[NODE]
    before = (draft.verify_rows, draft.accepted, len(driver.out[1]))
    driver.add(4, None, MAX)
    driver.mixed_step([1], {4: LATE[4]})
    assert driver.out[4] == reference(LATE[4], MAX)[:1]
    # the verify row counted one verify row, and accepted the oracle's drafts in full: they reached the
    # target in its block; the prompt row counted none
    assert draft.verify_rows == before[0] + 1 and draft.accepted == before[1] + K
    assert len(driver.out[1]) == before[2] + K + 1


def test_a_mixed_step_runs_its_verify_rows_first_and_refuses_its_prompt_rows_drafts(monkeypatch):
    """The engine orders the rows (`order_split_rows`), and the sampler tests no draft of a prompt row."""
    from mstar.engine.resources.sampler.resource import SamplerResource

    seen = []
    verify = SamplerResource.verify

    def recording(self, request_ids, logits, drafts, refuse=None):
        seen.append((tuple(request_ids), None if refuse is None else refuse.tolist()))
        return verify(self, request_ids, logits, drafts, refuse)

    monkeypatch.setattr(SamplerResource, "verify", recording)
    driver = Driver(engine(K, oracle, mixed=True))
    for rid in (1, 2):
        driver.add(rid, PROMPTS[rid], MAX)
    for rid in (4, 5):
        driver.add(rid, None, MAX)
    seen.clear()
    driver.mixed_step([1, 2], {4: LATE[4], 5: LATE[5]})
    assert seen == [((1, 2, 4, 5), [False, False, True, True])]


def test_a_mixed_steps_verify_rows_alone_enter_the_ledger():
    driver = Driver(engine(K, every(2), mixed=True))
    for rid in (1, 2):
        driver.add(rid, PROMPTS[rid], MAX)
    driver.step([1, 2])
    for rid in (4, 5):
        driver.add(rid, None, MAX)
    ledger = driver.eng._drafts[NODE].ledger
    seen = []
    committed = ledger.committed

    def recording(rids, labels=None, counted=None):
        seen.append((tuple(rids or ()), tuple(counted or ())))
        return committed(rids, labels, counted)

    ledger.committed = recording
    driver.mixed_step([1, 2], {4: LATE[4], 5: LATE[5]})
    assert seen == [((1, 2), (K + 1, K + 1))]
    # their results came due and were applied: nothing a prompt row could have left pending
    driver.run([1, 2, 4, 5])
    assert not ledger.unrecorded_rows()


def test_a_prompt_rows_stop_token_or_one_token_budget_ends_it():
    """A one-token budget stops the reply at its first token, as a prompt step's does."""
    driver = Driver(engine(K, oracle, mixed=True))
    driver.add(1, PROMPTS[1], MAX)
    driver.step([1])
    driver.add(4, None, 1)
    stopped = driver.mixed_step([1], {4: LATE[4]})
    assert stopped[4] and driver.out[4] == reference(LATE[4], 1)
    # a prompt whose first token is the stop id
    prompt = next(p for p in ([a, b] for a in range(60) for b in range(60)) if reference(p, 1) == [STOP])
    driver.add(5, None, MAX)
    stopped = driver.mixed_step([1], {5: prompt})
    assert stopped[5] and driver.out[5] == [STOP]


# ── what refuses mixed steps ──


def _decl(mixed=True, needs_mixed=True, mode="auto"):
    model = ToyModel(K, mode, needs=DrafterNeeds(chunk_boundary=ChunkBoundary.NONE, mixed=needs_mixed), mixed=mixed)
    return model.verify_nodes()[0], model.combined_walk_of() or {(NODE, "prefill"): "mixed", (NODE, "decode"): "mixed"}


def test_startup_takes_a_combined_walk_with_the_verify_walk_only_as_the_named_mixed_walk():
    decl, combined = _decl()
    assert combined_problem(decl, combined) is None
    decl, combined = _decl(mixed=False)
    assert "doesn't run as mixed prompt-plus-verify steps" in combined_problem(decl, combined)
    decl, combined = _decl(needs_mixed=False)
    assert "DrafterNeeds.mixed" in combined_problem(decl, combined)


def test_a_lagged_drafter_refuses_a_mixed_walk():
    """Its passes trail each row's verify block (`Lag`), planned over every row as one."""
    from mstar.engine.resources.step import Lag

    model = ToyModel(K, needs=DrafterNeeds(chunk_boundary=ChunkBoundary.NONE, mixed=True, lag=Lag(1, (2, 1, 1))),
                     mixed=True)
    assert "lags the target" in combined_problem(model.verify_nodes()[0], model.combined_walk_of())


def test_a_mixed_node_verifies_asynchronously_as_a_verify_node_does():
    """Its mixed steps are resident in their verify rows (`ResidentPlanner.mixed_walks`), so the mixed walk holds
    nothing back: the mode follows the node's resources and scheduling alone."""
    for mode in ("auto", "asynchronous"):
        decl, _ = _decl(mode=mode)
        assert resolve_mode(decl, {}, True)[0] == DraftMode.ASYNCHRONOUS
    decl, _ = _decl(mode="auto")
    mode, reason = resolve_mode(decl, {}, False)
    assert mode == DraftMode.SYNCHRONOUS and "async scheduling" in reason


def test_verify_rows_in_a_combined_step_the_node_doesnt_run_mixed_are_refused():
    driver = Driver(engine(K, oracle))
    driver.add(1, PROMPTS[1], MAX)
    driver.add(4, None, MAX)
    driver.eng._split_prompt_walks[(NODE, "mixed")] = frozenset({"prefill"})
    with pytest.raises(RuntimeError, match="rows inside a combined step"):
        driver.mixed_step([1], {4: LATE[4]})


# ── lean mixed steps ──


def test_a_lean_mixed_step_verifies_by_the_argmax_and_replies_as_a_full_one(monkeypatch):
    """An all-greedy mixed step runs its sampler lean (`lean_for` over every row, its prompt rows' too): its verify
    rows take `greedy_verify`, a refused prompt row the argmax at its first position, and it builds no
    vocabulary's probs. A full mixed step builds them. The replies agree."""
    from test_spec_greedy_verify import _counting

    calls = _counting(monkeypatch)
    outs = {}
    for lean in (False, True):
        driver = Driver(engine(K, oracle, mixed=True, greedy_capture=lean))
        for rid, prompt in PROMPTS.items():
            driver.add(rid, prompt, MAX)
        for _ in range(2):
            driver.step(list(PROMPTS))
        for rid in LATE:
            driver.add(rid, None, MAX)
        calls.update(greedy=0, probs=0)
        stopped = driver.mixed_step(list(PROMPTS), LATE)
        assert (calls["greedy"] > 0, calls["probs"] > 0) == (lean, not lean), calls
        driver.run([rid for rid in [*PROMPTS, *LATE] if not stopped[rid]])
        outs[lean] = driver.out
    assert outs[True] == outs[False]


def test_a_mixed_step_advances_its_verify_rows_resident_lengths_by_what_each_keeps(monkeypatch):
    """Under a resident capture the graph advances each verify row's KV length cell by the advance buffer, which
    the step's verify writes (accepted + 1), as a verify step's does: a mixed step that left it zero held its
    verify rows' device lengths back by what they kept, and their next steps re-verified tokens already
    emitted. Its prompt rows' entries land on the trash row, which the advance skips."""
    from types import SimpleNamespace

    import mstar.engine.resources.draft.drafter as drafter_mod

    real, seen = drafter_mod.draft_mixed_step, []

    def resident(*args, **kwargs):
        advance = torch.zeros(8, dtype=torch.long)
        kwargs["resident"] = SimpleNamespace(buffers=SimpleNamespace(advance=advance))
        result = real(*args, **kwargs)
        seen.append((advance.clone(), result.accepted.clone()))
        return result

    monkeypatch.setattr(drafter_mod, "draft_mixed_step", resident)
    driver = Driver(engine(K, every(2), mixed=True))
    for rid, prompt in PROMPTS.items():
        driver.add(rid, prompt, MAX)
    for _ in range(2):
        driver.step(list(PROMPTS))
    for rid in LATE:
        driver.add(rid, None, MAX)
    driver.mixed_step(list(PROMPTS), LATE)
    ((advance, accepted),) = seen
    n = accepted.shape[0]
    assert advance[:n].tolist() == (accepted + 1).tolist() and advance[n:].eq(0).all()
