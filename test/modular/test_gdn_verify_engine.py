"""A hybrid GDN + attention model verifying drafts on the real Engine, in
synchronous mode (`hybrid_toy`): greedy output equals plain greedy decoding
whatever the drafts, every accepted count 0..k running, the GDN state settled
by the in-graph rerun at every step."""
from __future__ import annotations

import collections

import pytest
from spec_toy import Driver

from mstar.engine.resources.kv import manager as manager_mod

# a one-token prompt would take the single-token kernels, which have no CPU stand-in
PROMPTS = {1: [5, 9, 2], 2: [17, 4], 3: [1, 2, 3, 4, 5, 6, 7, 8, 9]}
MAX = 20


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    import gdn_cpu
    import qwen3_cpu

    qwen3_cpu.patch_cpu(monkeypatch)
    gdn_cpu.install(monkeypatch)
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)


def _references(extra: int = 8) -> dict[int, list[int]]:
    from hybrid_toy import reference_greedy

    return {rid: p + reference_greedy(p, MAX + extra) for rid, p in PROMPTS.items()}


def _decode(k, spoil, batched=True):
    import hybrid_toy as toy

    eng = toy.engine(k, toy.ForcedDrafter(_references(), spoil))
    driver = Driver(eng)
    for group in ([list(PROMPTS)] if batched else [[rid] for rid in PROMPTS]):
        for rid in group:
            driver.add(rid, PROMPTS[rid], MAX)
        driver.run(group)
    return driver.out


def _check(out):
    from hybrid_toy import reference_greedy

    for rid, prompt in PROMPTS.items():
        margins: list[float] = []
        ref = reference_greedy(prompt, MAX, margins=margins)
        n = min(len(ref), len(out[rid]))
        at = next((j for j in range(n) if ref[j] != out[rid][j]), None)
        if at is None and len(ref) != len(out[rid]):
            at = n
        assert at is None or margins[at] < 1e-4, (rid, at, ref, out[rid])


def test_the_verify_step_corrects_only_the_resources_bound_at_load():
    """correcting_keys: the node's resources that correct the block's state in the graph, found once at
    load in the runner's order; the verify step calls those and no other."""
    import hybrid_toy as toy
    from spec_toy import NODE as SPEC_NODE
    from spec_toy import engine as spec_engine
    from spec_toy import oracle

    eng = toy.engine(3, toy.ForcedDrafter(_references()))
    assert eng._submodules[toy.NODE].submodule.drafter_share.correcting_keys == ("linear_attn",)
    plain = spec_engine(3, oracle)
    assert plain._submodules[SPEC_NODE].submodule.drafter_share.correcting_keys == ()


def test_a_verifying_node_with_a_pool_nothing_reruns_is_refused_before_and_at_load(monkeypatch):
    """A second recurrent pool on the node that no linear-attention resource names would keep a
    verify step's rejected tokens folded into its state. The spec check is the only guard, so it
    is tested where it runs: validation's `check_deployment` and the engine's load."""
    import hybrid_toy as toy
    import torch

    from mstar.engine.resources.draft.startup import check_deployment
    from mstar.engine.resources.recurrent.config import RecurrentStateConfig, RecurrentStateSpec

    orphan = RecurrentStateSpec("orphan_state", {toy.NODE}, RecurrentStateConfig(
        num_layers=1, blocks=toy.CPU_SHAPES.geometry.to_blocks(state_dtype=torch.float32, conv_dtype=torch.float32),
        max_slots=16,
    ))
    specs = toy.specs(3) + [orphan]
    decls = toy.HybridModel(3, "synchronous", None, True, False).verify_nodes()
    problems = check_deployment(decls, {s.resource_key: s for s in specs}, {}, (), {}, True)
    assert any("'orphan_state'" in p and "no linear-attention resource reruns its state" in p for p in problems)

    real = toy.specs
    monkeypatch.setattr(toy, "specs", lambda *args, **kwargs: real(*args, **kwargs) + [orphan])
    with pytest.raises(RuntimeError, match="no linear-attention resource reruns its state"):
        toy.engine(3, toy.ForcedDrafter(_references()))


@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("k, first_wrong", [
    (k, wrong) for k in (1, 3) for wrong in (None, 0, 1, 2) if wrong is None or wrong < k
])
def test_greedy_output_is_plain_greedy_whatever_the_drafts(k, first_wrong, batched):
    out = _decode(k, lambda j: j == first_wrong, batched)
    _check(out)


def test_without_the_rerun_the_state_keeps_rejected_tokens(monkeypatch):
    """Negative control: with correct_verify_state a no-op, rejected tokens stay folded
    into the GDN state and the output leaves plain greedy decoding."""
    from mstar.engine.resources.linear_attn.gdn import GDNManager

    monkeypatch.setattr(GDNManager, "correct_verify_state", lambda self, result: None)
    out = _decode(3, lambda j: j == 1)
    with pytest.raises(AssertionError):
        _check(out)


def test_every_accepted_count_runs():
    import hybrid_toy as toy

    counts = collections.Counter()
    for first_wrong in (None, 0, 1, 2):
        eng = toy.engine(3, toy.ForcedDrafter(_references(), lambda j, f=first_wrong: j == f))
        driver = Driver(eng)
        for rid, prompt in PROMPTS.items():
            driver.add(rid, prompt, MAX)
        seen = []
        real = eng.cut_verify_step

        def cut(batch, outputs, host, real=real, seen=seen):
            seen += host.buffers["verify_accepted"].tolist()
            return real(batch, outputs, host)

        eng.cut_verify_step = cut
        driver.run(list(PROMPTS))
        counts.update(seen)
    assert {0, 1, 2, 3} <= set(counts), counts


def test_only_the_cpu_attention_keeps_the_toy_from_asynchronous_mode():
    """The recurrent pool and GDN resource no longer refuse asynchronous verify
    steps; on the CPU the toy's attention stand-in, which keeps no resident
    table, and its uncaptured walks still do. Auto runs synchronous."""
    import hybrid_toy as toy

    with pytest.raises((RuntimeError, ValueError), match="asynchronous") as refused:
        toy.engine(3, toy.ForcedDrafter(_references()), mode="asynchronous")
    assert "gdn_state" not in str(refused.value) and "linear_attn" not in str(refused.value)
    eng = toy.engine(3, toy.ForcedDrafter(_references()), mode="auto")
    assert not eng._drafts[toy.NODE].asynchronous


def _plain(prompts=PROMPTS):
    import hybrid_toy as toy

    driver = toy.PlainDriver(toy.engine(3, None))
    for rid, prompt in prompts.items():
        driver.add(rid, prompt, MAX)
    driver.run(list(prompts))
    return driver.out


def test_the_plain_run_is_plain_greedy():
    """The same build with no drafter: one-token GDN decode steps on the CPU
    stand-ins, which must reproduce the dense reference."""
    _check(_plain())


@pytest.mark.parametrize("k", [1, 3])
def test_an_always_rejected_drafter_equals_the_plain_run(k):
    """The RFC's bar on one build and its kernels: every draft wrong, every
    step keeps only its first token, and the output is the plain run's."""
    assert _decode(k, lambda j: True) == _plain()


def test_a_prompt_prefix_hit_restores_the_recurrent_snapshot_and_greedy_parity_holds():
    """Prefix caching on a verifying GDN node: the first request's prompt files
    a page-boundary snapshot of its state; the same prompt again hits, restores
    the snapshot into its slot, prefills the rest and verifies drafts, and its
    reply is still plain greedy decoding's."""
    import hybrid_toy as toy

    # three whole pages of 4: the KV cache serves the first two, and the first request's
    # snapshot sits at the last boundary before its last token, 8, so the pool serves them too
    prompt = list(range(1, 13))
    refs = {rid: prompt + toy.reference_greedy(prompt, MAX + 8) for rid in (1, 2)}
    eng = toy.engine(3, toy.ForcedDrafter(refs, lambda j: j == 1), prefix=True)
    pool = eng._resources["gdn_state"]
    assert pool._kv is not None, "the pool keeps no prompt snapshots"
    driver = Driver(eng)
    assert toy.add_keyed(driver, 1, prompt, MAX, 3, toy.PAGE) == 0
    driver.run([1])
    assert pool._snap_by_key, "the prompt filed no snapshot"
    hit = toy.add_keyed(driver, 2, prompt, MAX, 3, toy.PAGE)
    assert hit == 2 * toy.PAGE
    driver.run([2])
    want = toy.reference_greedy(prompt, MAX)
    assert driver.out[1] == want and driver.out[2] == want
