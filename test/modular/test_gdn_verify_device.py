"""The hybrid GDN + attention toy (`hybrid_toy`) verifying drafts on a CUDA
device: FlashInfer's multi-token GDN kernel and paged attention, the multi-token
conv, the device sampler; eager and captured, synchronous. The verify step's
draft pass, target, verify and closed-gate rerun replay as one graph per
bucket. Greedy output must be plain greedy decoding's by the margin rule (bf16:
where they first differ, the float32 reference's top-2 margin is a near-tie),
for every accepted count. Skips without CUDA and FlashInfer.

The drafter is a device table (`TableDrafter`), so a replayed step drafts what
an eager one does; its CPU case below checks the table itself everywhere.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch
from spec_toy import Driver

from mstar.engine.resources.base import Residency

PROMPTS = {1: [5, 9, 2], 2: [17, 4], 3: [1, 2, 3, 4, 5, 6, 7, 8, 9]}
MAX = 20
K = 3
MARGIN = 0.1


def _cuda_reason() -> str | None:
    if not torch.cuda.is_available():
        return "needs CUDA"
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        return "needs flashinfer"
    return None


def _fa3_reason() -> str | None:
    """Asynchronous verify steps read the attention's resident tables: FA3's paged kernel (Hopper)."""
    from mstar.engine.resources.attn.fa3_verify import fa3_verify_unavailable_reason

    return _cuda_reason() or fa3_verify_unavailable_reason(torch.device("cuda"), 128, torch.bfloat16)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    import qwen3_cpu

    from mstar.engine.resources.kv import manager as manager_mod

    # no KV transfer engine; a device's draws go to the real sampler
    qwen3_cpu.patch_cpu(monkeypatch)
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)


def _bf16_twin(seed: int = 0):
    """The device model's weights, on the CPU: drawn on the CPU and rounded to bf16, as the device's are."""
    import hybrid_toy as toy

    return toy.TinyHybrid(seed, toy.Shapes(**{**toy.CUDA_SHAPES.__dict__, "device": "cpu"}))


def _references(twin, extra: int = 8):
    from hybrid_toy import reference_greedy

    return {rid: p + reference_greedy(p, MAX + extra, like=twin) for rid, p in PROMPTS.items()}


def _check(out, twin, margin: float = MARGIN):
    from hybrid_toy import reference_greedy

    for rid, prompt in PROMPTS.items():
        margins: list[float] = []
        ref = reference_greedy(prompt, MAX, margins=margins, like=twin)
        n = min(len(ref), len(out[rid]))
        at = next((j for j in range(n) if ref[j] != out[rid][j]), None)
        if at is None and len(ref) != len(out[rid]):
            at = n
        assert at is None or margins[at] <= margin, (rid, at, ref, out[rid])


def _replays_resident(eng) -> list[bool]:
    """Per verify step from here on, whether it replayed a capture on resident
    tables; filled as the engine leases each step's capture."""
    flags: list[bool] = []
    real = eng.reserve_replay_slot

    def reserve(batch):
        lease = real(batch)
        if batch.step_context.graph_walk == "decode":
            flags.append(lease is not None and batch.step_context.resident is not None)
        return lease

    eng.reserve_replay_slot = reserve
    return flags


def _check_resident(eng, flags: list[bool]) -> None:
    """A captured case replays every verify step on resident tables: the GDN
    lift's resident gather (slots at span k + 1) ran, not a host plan."""
    import hybrid_toy as toy

    assert eng._submodules[toy.NODE].resident is not None, "the node built no resident planner"
    assert flags and all(flags), flags


def _check_no_bucket_dropped(eng) -> None:
    """Every verify bucket captured: a dropped one would run its steps eagerly, and the
    rerun's kernel variants for its rows would never have been captured."""
    import hybrid_toy as toy

    runner = eng._submodules[toy.NODE].accelerator_graph_runner
    assert runner.dropped_buckets == [], runner.dropped_buckets
    assert runner.max_captured_bs("decode") == max(toy.HybridLM.DECODE_CAPTURE_BATCH_SIZES) == 4


def _decode(eng, prompts=None):
    """The run's replies, and per verify step and row (rid, accepted). A
    captured engine's verify steps must all replay on resident tables."""
    prompts = prompts or PROMPTS
    resident = _replays_resident(eng)
    driver = Driver(eng)
    for rid, prompt in prompts.items():
        driver.add(rid, prompt, MAX)
    seen = []
    real = eng.cut_verify_step

    def cut(batch, outputs, host):
        accepted = host.buffers["verify_accepted"].tolist()
        seen.extend(zip(host.request_ids, accepted[: len(host.request_ids)], strict=True))
        return real(batch, outputs, host)

    eng.cut_verify_step = cut
    driver.run(list(prompts))
    if eng._submodules[next(iter(eng._submodules))].submodule.accelerator_graph:
        _check_resident(eng, resident)
    return driver.out, seen


def _plain(captured: bool, max_tokens: int = MAX):
    """The same build with no drafter, on the device."""
    import hybrid_toy as toy

    driver = toy.PlainDriver(toy.engine(K, None, shapes=toy.CUDA_SHAPES, captured=captured))
    for rid, prompt in PROMPTS.items():
        driver.add(rid, prompt, max_tokens)
    driver.run(list(PROMPTS))
    return driver.out


def _counted(seen, out, source) -> list[int]:
    """Each row's accepted counts while its drafts were the target's own
    tokens: up to the first step whose draft window reaches where the reply
    leaves ``source`` (a bf16 near-tie there changes the count, not a bug), and
    never a row's last step (cut by its length)."""
    counted = []
    for rid, prompt in PROMPTS.items():
        ref = source[rid][len(prompt):]
        reply = out[rid]
        n = min(len(ref), len(reply))
        diverge = next((j for j in range(n) if ref[j] != reply[j]), n)
        steps = [a for r, a in seen if r == rid][:-1]
        at = 0  # the anchor's index in the reply
        for accepted in steps:
            if diverge <= at + K:
                break
            counted.append(accepted)
            at += accepted + 1
    return counted


def test_the_table_drafter_drafts_the_reference_on_the_cpu(monkeypatch):
    import gdn_cpu
    import hybrid_toy as toy

    gdn_cpu.install(monkeypatch)
    refs = {rid: p + toy.reference_greedy(p, MAX + 8) for rid, p in PROMPTS.items()}
    eng = toy.engine(K, toy.TableDrafter(refs, K, lambda j: j == 2, prompts=PROMPTS))
    out, seen = _decode(eng)
    for rid, prompt in PROMPTS.items():
        assert out[rid] == toy.reference_greedy(prompt, MAX)
    # the spoiled third draft caps every uncut step at two
    counts = [a for _, a in seen]
    assert set(counts) <= {0, 1, 2} and _counted(seen, out, refs) and set(_counted(seen, out, refs)) == {2}


@pytest.mark.parametrize("captured", [False, True], ids=["eager", "captured"])
@pytest.mark.parametrize("first_wrong", [None, 0, 1, 2])
def test_greedy_output_is_plain_greedy_on_the_device(captured, first_wrong):
    if (reason := _cuda_reason()) is not None:
        pytest.skip(reason)
    import hybrid_toy as toy

    twin = _bf16_twin()
    # drafted from the device's own plain run, so a bf16 tie the margin rule allows drafts what verify picks
    plain = _plain(captured, MAX + 8)
    source = {rid: p + plain[rid] for rid, p in PROMPTS.items()}
    drafter = toy.TableDrafter(source, K, lambda j: j == first_wrong, device="cuda", prompts=PROMPTS)
    eng = toy.engine(K, drafter, shapes=toy.CUDA_SHAPES, captured=captured)
    if captured:
        _check_no_bucket_dropped(eng)
    out, seen = _decode(eng)
    _check(out, twin)
    want = K if first_wrong is None else first_wrong
    counted = _counted(seen, out, source)
    assert counted and set(counted) == {want}, seen


@pytest.mark.parametrize("captured", [False, True], ids=["eager", "captured"])
def test_an_always_rejected_drafter_equals_the_plain_run_on_the_device(captured):
    """The RFC's bar on one build: every draft wrong, so every verify step
    keeps its first token, the plain run's. In bf16 the two runs take
    different kernels (multi-token GDN and verify attention, against one-token
    decode), so they may part only where the float32 twin's top-2 margin there
    is a near-tie; byte equality is for float32 toys (the CPU test)."""
    if (reason := _cuda_reason()) is not None:
        pytest.skip(reason)
    import hybrid_toy as toy

    twin = _bf16_twin()
    plain = _plain(captured)
    _check(plain, twin)
    source = {rid: p + plain[rid] for rid, p in PROMPTS.items()}
    eng = toy.engine(K, toy.TableDrafter(source, K, lambda j: True, device="cuda", prompts=PROMPTS),
                     shapes=toy.CUDA_SHAPES, captured=captured)
    out, seen = _decode(eng)
    for rid, prompt in PROMPTS.items():
        n = min(len(out[rid]), len(plain[rid]))
        at = next((j for j in range(n) if out[rid][j] != plain[rid][j]), None)
        if at is None:
            assert len(out[rid]) == len(plain[rid]), (rid, out[rid], plain[rid])
        else:
            assert toy.margin_at(prompt + plain[rid][:at], like=twin) <= MARGIN, (rid, at, out[rid], plain[rid])
    assert set(_counted(seen, out, source)) <= {0}, seen


def test_without_the_rerun_the_device_output_leaves_plain_greedy(monkeypatch):
    if (reason := _cuda_reason()) is not None:
        pytest.skip(reason)
    import hybrid_toy as toy

    from mstar.engine.resources.linear_attn.gdn import GDNManager

    monkeypatch.setattr(GDNManager, "settle_verify_step", lambda self, result: None)
    twin = _bf16_twin()
    # every draft rejected: three kept-out tokens a step, which this toy's greedy output
    # can't absorb at a near-tie the margin rule allows (one a step, it can)
    eng = toy.engine(K, toy.TableDrafter(_references(twin), K, lambda j: j == 0, device="cuda", prompts=PROMPTS),
                     shapes=toy.CUDA_SHAPES, captured=True)
    out, _ = _decode(eng)
    with pytest.raises(AssertionError):
        _check(out, twin)


def test_a_padding_rows_unwritten_conv_output_stays_out_of_the_real_rows(monkeypatch):
    """Three requests in a four-row verify capture: the padding row is on the
    null slot, where the multi-token conv writes nothing. With that output NaN
    (`GDNVerifyWrapper.run_conv`'s `torch.empty` made to fill it), the captured
    run still matches plain greedy."""
    if (reason := _cuda_reason()) is not None:
        pytest.skip(reason)
    import hybrid_toy as toy

    from mstar.engine.resources.linear_attn import wrappers

    class NanTorch:
        """The GDN wrappers' `torch`, whose `empty` fills with NaN."""

        def empty(self, *args, **kwargs):
            out = torch.empty(*args, **kwargs)
            return out.fill_(float("nan")) if out.is_floating_point() else out

        def __getattr__(self, name):
            return getattr(torch, name)

    monkeypatch.setattr(wrappers, "torch", NanTorch())
    twin = _bf16_twin()
    eng = toy.engine(K, toy.TableDrafter(_references(twin), K, lambda j: j == 0, device="cuda", prompts=PROMPTS),
                     shapes=toy.CUDA_SHAPES, captured=True)
    out, _ = _decode(eng)
    _check(out, twin)


def _state_pair(stage: bool, monkeypatch, device: str = "cuda"):
    """Two pools and GDN resources on the device, FlashInfer's kernels, the
    staged input path or the eager one: one verifies, one runs plainly."""
    from mstar.engine.resources.base import EngineResourceInfo
    from mstar.engine.resources.linear_attn import gdn as gdn_mod
    from mstar.engine.resources.linear_attn.base import LinearAttnManager
    from mstar.engine.resources.linear_attn.config import LinearAttnConfig, LinearAttnSpec, LinearAttnVariant
    from mstar.engine.resources.recurrent.config import DeltaNetGeometry, RecurrentStateConfig, RecurrentStateSpec
    from mstar.engine.resources.recurrent.pool import RecurrentStatePool

    # the staged path is gated to SM100 by measurement, not by what it computes
    monkeypatch.setattr(gdn_mod, "gdn_stage_applies", lambda device, geometry: stage)
    device = torch.device(device)
    geometry = DeltaNetGeometry(num_k_heads=1, num_v_heads=2, head_k_dim=128, head_v_dim=128, conv_kernel_size=4)
    pairs = []
    for verifies in (True, False):
        pool_spec = RecurrentStateSpec("gdn_state", {"LLM"}, RecurrentStateConfig(
            num_layers=1, blocks=geometry.to_blocks(
                state_dtype=torch.float32, conv_dtype=torch.float32 if device.type == "cpu" else torch.bfloat16,
            ),
            max_slots=4,
        ))
        pool = RecurrentStatePool.build(pool_spec, EngineResourceInfo(device=device))
        manager = LinearAttnManager.build(
            LinearAttnSpec("linear_attn", {"LLM"},
                           LinearAttnConfig(recurrent_state="gdn_state", variant=LinearAttnVariant.GDN)),
            EngineResourceInfo(device=device, dependencies={"gdn_state": pool_spec}),
        )
        manager.bind_dependencies({"gdn_state": pool})
        if verifies:
            pool.bind_verify_walk("decode", K + 1, 0)
            manager.bind_verify_walk("decode", K + 1, 0)
        pool.ingest_request(0)
        if pool.ask(0, "LLM", "prefill") is not None:
            pool.reserve(0, "LLM", "prefill", None)
        pairs.append((pool, manager))
    return geometry, pairs


def _gdn_step(pool, manager, geometry, span: int, walk: str, x, a, b, weights) -> None:
    """One row's step through the real resource, as `GatedDeltaNet` drives it: a prompt
    step on the chunked kernel, a verify step on the multi-token one."""
    from mstar.engine.resources.linear_attn.config import LinearAttnStep
    from mstar.engine.resources.recurrent.config import RecurrentStep
    from mstar.engine.resources.step import Segment, StepContext

    ctx = StepContext(request_ids=[0], graph_walk=walk, slot=0, capture=False)
    ctx.force_prefill = True  # one kernel for every span, a single kept token's too
    segments = (Segment(0, "main", span),)
    rstep = RecurrentStep(segments=segments)
    assert pool.admit(rstep, ctx).ok
    ctx.plan_results["gdn_state"] = pool.plan(rstep, ctx)
    manager.plan(LinearAttnStep(segments=segments), ctx)
    manager.set_default_layer_idx(0)
    conv_w, a_log, dt_bias = weights
    qkv = manager.run_conv(x, pool.block("conv", 0), conv_w, None)
    kd, vd = geometry.num_k_heads * geometry.head_k_dim, geometry.num_v_heads * geometry.head_v_dim
    q, k, v = torch.split(qkv, [kd, kd, vd], dim=-1)
    n = x.shape[0]
    # column views of the conv output and of one gate projection, as `GatedDeltaNet` passes them:
    # the kernels read the strides
    ab = torch.cat([a, b], dim=-1)
    a, b = ab[:, : a.shape[-1]], ab[:, a.shape[-1]:]
    assert not any(t.is_contiguous() for t in (q, k, v, a, b))
    out = manager.run(q.reshape(n, geometry.num_k_heads, -1), k.reshape(n, geometry.num_k_heads, -1),
                      v.reshape(n, geometry.num_v_heads, -1), a, b, pool.block("state", 0), a_log, dt_bias)
    pool.commit(rstep, ctx)
    return out


def _fp64_reference(geometry, weights, x, a, b):
    """One GDN layer over ``x``'s tokens from zeros, in float64 on the CPU, as
    `GatedDeltaNet` and the resource compute it: the conv (silu), q and k
    l2-normalised, the gates, the delta rule token by token. Returns the state,
    the conv window and every token's output."""
    import gdn_cpu

    conv_w, a_log, dt_bias = (t.detach().double().cpu() for t in weights)
    x, a, b = (t.detach().double().cpu() for t in (x, a, b))
    width = geometry.conv_kernel_size - 1
    seq = torch.cat([torch.zeros(width, x.shape[1], dtype=torch.float64), x])
    y = torch.nn.functional.silu(sum(conv_w[:, j] * seq[j: j + x.shape[0]] for j in range(width + 1)))
    kd, vd = geometry.num_k_heads * geometry.head_k_dim, geometry.num_v_heads * geometry.head_v_dim
    q, k, v = torch.split(y, [kd, kd, vd], dim=-1)
    n = x.shape[0]
    q = torch.nn.functional.normalize(q.reshape(n, geometry.num_k_heads, -1), dim=-1)
    k = torch.nn.functional.normalize(k.reshape(n, geometry.num_k_heads, -1), dim=-1)
    v = v.reshape(n, geometry.num_v_heads, -1)
    alpha = torch.exp(-torch.exp(a_log) * torch.nn.functional.softplus(a + dt_bias))
    beta = torch.sigmoid(b)
    zero = torch.zeros(geometry.num_v_heads, geometry.head_v_dim, geometry.head_k_dim, dtype=torch.float64)
    out, state = gdn_cpu.delta_rule(q, k, v, alpha, beta, zero, geometry.head_k_dim ** -0.5)
    return state, seq[-width:].T, out


def _rel(got, want) -> float:
    got = got.detach().double().cpu()
    return float((got - want).norm() / want.norm())


# A settled state and a verify forward's outputs against the float64 reference: the multi-token
# kernel computes in tf32 (~5e-4 off on the state, ~2e-3 on its bf16 outputs at the checkpoints'
# shapes); one token too many kept, or none undone, lands 0.6-1.4 off.
STATE_REL = 5e-3


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("accepted", range(K + 1))
def test_the_rerun_leaves_the_state_of_the_kept_tokens_alone(monkeypatch, device, accepted):
    """State equality, the RFC's first condition: a verify step settled at
    ``accepted`` leaves the slot's state and conv window those of the prompt
    and the kept tokens alone, against a float64 reference of the layer, to
    `STATE_REL`; and its forward's outputs are the reference's for the block.
    One token too many kept, or none undone, is far outside it. The CPU case
    runs `gdn_cpu`'s stand-ins, which checks the test itself."""
    if device == "cpu":
        import gdn_cpu

        gdn_cpu.install(monkeypatch)
    elif (reason := _cuda_reason()) is not None:
        pytest.skip(reason)
    from types import SimpleNamespace

    geometry, ((vpool, verifier), _) = _state_pair(False, monkeypatch, device)
    gen = torch.Generator(device=device).manual_seed(accepted)

    # the stand-ins compute in their inputs' dtype
    activations = torch.float32 if device == "cpu" else torch.bfloat16

    def draw(*shape, scale=1.0, dtype=activations):
        return (torch.randn(*shape, generator=gen, device=device) * scale).to(dtype)

    hv = geometry.num_v_heads
    weights = (draw(geometry.conv_dim, geometry.conv_kernel_size, scale=0.3), draw(hv, scale=0.3, dtype=torch.float32),
               draw(hv, scale=0.3, dtype=torch.float32))
    prompt = [draw(5, geometry.conv_dim), draw(5, hv), draw(5, hv)]
    block = [draw(K + 1, geometry.conv_dim), draw(K + 1, hv), draw(K + 1, hv)]
    _gdn_step(vpool, verifier, geometry, 5, "prefill", *prompt, weights)
    # copied before the settle, whose rerun writes the same buffer (the model has read it by then)
    out = _gdn_step(vpool, verifier, geometry, K + 1, "decode", *block, weights).clone()
    verifier.settle_verify_step(SimpleNamespace(accepted=torch.tensor([accepted], device=device)))

    def upto(n):
        return _fp64_reference(geometry, weights, *(torch.cat([p, t[:n]]) for p, t in zip(prompt, block, strict=True)))

    slot = vpool._slots[0]["main"].index
    state, window, outputs = upto(accepted + 1)
    assert _rel(vpool.block("state", 0)[slot], state) <= STATE_REL
    assert _rel(vpool.block("conv", 0)[slot], window) <= STATE_REL
    # the forward over the whole block, from the prompt's state
    full_outputs = upto(K + 1)[2][5:]
    assert _rel(out.reshape(K + 1, hv, -1), full_outputs) <= STATE_REL
    # the bar tells a wrong undo apart: one more token kept, or none undone
    wrong = upto(min(accepted + 2, K + 1))[0] if accepted < K else upto(accepted)[0]
    assert _rel(vpool.block("state", 0)[slot], wrong) > 20 * STATE_REL


@pytest.mark.parametrize("first_wrong", [None, 0, 1, 2])
def test_asynchronous_resident_steps_match_the_plain_run(first_wrong):
    """Asynchronous mode: every verify step a resident replay that gathers its
    rows' pool slots off the tables, checkpoints the conv window, and settles in
    the graph. Drafted from the device's own plain run; output by the margin
    rule, and every counted step at the accepted count the drafts force."""
    if (reason := _fa3_reason()) is not None:
        pytest.skip(reason)
    import hybrid_toy as toy

    twin = _bf16_twin()
    plain = _plain(True, MAX + 8)
    source = {rid: p + plain[rid] for rid, p in PROMPTS.items()}
    drafter = toy.TableDrafter(source, K, lambda j: j == first_wrong, device="cuda", prompts=PROMPTS)
    eng = toy.engine(K, drafter, mode="asynchronous", shapes=toy.CUDA_SHAPES, captured=True)
    assert eng._drafts[toy.NODE].asynchronous
    out, seen = _decode(eng)
    _check(out, twin)
    want = K if first_wrong is None else first_wrong
    counted = _counted(seen, out, source)
    assert counted and set(counted) == {want}, seen


@pytest.mark.parametrize("mode", ["synchronous", "asynchronous"])
def test_a_prompt_prefix_hit_restores_the_snapshot_on_the_device(mode):
    """Prefix caching on a verifying GDN node, captured: the first request's
    prompt files a snapshot at a page boundary; the same prompt again hits,
    restores it, prefills the rest, and verifies on resident tables. Both
    replies are the plain run's by the margin rule."""
    if (reason := (_fa3_reason() if mode == "asynchronous" else _cuda_reason())) is not None:
        pytest.skip(reason)
    import hybrid_toy as toy

    # snapshots fall every page, and FlashInfer's chunked kernel checkpoints only at
    # multiples of its 64-token chunk
    shapes = dataclasses.replace(toy.CUDA_SHAPES, page=64)
    page = shapes.page
    # three whole pages: the cache serves the first two, where the first request's snapshot sits
    prompt = [1 + (i * 7) % 60 for i in range(3 * page)]
    twin = _bf16_twin()
    plain = toy.PlainDriver(toy.engine(K, None, shapes=shapes, captured=True))
    plain.add(1, prompt, MAX + 8)
    plain.run([1])
    source = {rid: prompt + plain.out[1] for rid in (1, 2)}
    drafter = toy.TableDrafter(source, K, lambda j: j == 1, device="cuda", prompts={1: prompt, 2: prompt})
    eng = toy.engine(K, drafter, mode=mode, shapes=shapes, captured=True, prefix=True)
    resident = _replays_resident(eng)
    driver = Driver(eng)
    assert toy.add_keyed(driver, 1, prompt, MAX, K, page) == 0
    driver.run([1])
    assert toy.add_keyed(driver, 2, prompt, MAX, K, page) == 2 * page
    driver.run([2])
    _check_resident(eng, resident)
    margins: list[float] = []
    want = toy.reference_greedy(prompt, MAX, margins=margins, like=twin)
    for rid in (1, 2):
        out = driver.out[rid]
        n = min(len(want), len(out))
        at = next((j for j in range(n) if want[j] != out[j]), None if len(want) == len(out) else n)
        assert at is None or margins[at] <= MARGIN, (rid, at, want, out)


# ── the verify conv leaves the window alone; one kernel settles it ──


@pytest.mark.parametrize("dim", [96, 8192])
def test_the_verify_conv_outputs_do_not_depend_on_updating_the_window(dim):
    """`causal_conv1d_update(update_state=False)` writes the same outputs, bit for
    bit, and leaves every window as it was; a null-slot row is skipped either way."""
    if (reason := _cuda_reason()) is not None:
        pytest.skip(reason)
    from mstar.utils.causal_conv1d import causal_conv1d_update

    gen = torch.Generator(device="cuda").manual_seed(0)
    rows, block, width, slots = 4, K + 1, 4, 6
    conv = torch.randn(slots, dim, width - 1, generator=gen, device="cuda").bfloat16()
    # [rows, block, dim] packed, read as [rows, dim, block], as the verify wrapper passes it
    x = torch.randn(rows, block, dim, generator=gen, device="cuda").bfloat16()
    weight = torch.randn(dim, width, generator=gen, device="cuda").bfloat16()
    bias = torch.randn(dim, generator=gen, device="cuda").bfloat16()
    index = torch.tensor([3, 0, 5, 1], dtype=torch.int32, device="cuda")
    outs, states = [], []
    for update in (True, False):
        state = conv.clone()
        out = torch.zeros(rows, block, dim, dtype=torch.bfloat16, device="cuda")
        causal_conv1d_update(x=x.transpose(1, 2), conv_state=state, weight=weight, bias=bias, activation="silu",
                             conv_state_indices=index, null_block_id=0, out=out.transpose(1, 2), update_state=update)
        outs.append(out)
        states.append(state)
    assert torch.equal(outs[0], outs[1])
    assert torch.equal(states[1], conv)
    assert not torch.equal(states[0][3], conv[3]) and torch.equal(states[0][0], conv[0])


def _old_settle_conv(conv, window, x, addr, starts, kept):
    """The settle before the window stayed in its slot: from a pre-step copy of each row's window."""
    rows, dim, width = window.shape
    src = kept[:, None] + torch.arange(width, device=kept.device)[None, :]
    from_window = src < width
    old = window.gather(2, src.clamp(max=width - 1)[:, None, :].expand(rows, dim, width))
    at = (starts[:, None] + src - width).clamp(min=0, max=x.shape[0] - 1)
    new = x[at.reshape(-1)].view(rows, width, dim).transpose(1, 2).to(conv.dtype)
    conv.index_copy_(0, addr, torch.where(from_window[:, None, :], old, new))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("accepted", range(K + 2))
def test_the_settle_kernel_is_the_old_settle_bit_for_bit(dtype, accepted):
    """`settle_conv` on CUDA against the old torch settle, for every accepted count
    (and one past k, clamped), with a padding row on the sink and ``x`` and
    ``accepted`` strided views: the real rows' windows are equal bit for bit and
    the sink is untouched."""
    if (reason := _cuda_reason()) is not None:
        pytest.skip(reason)
    from mstar.engine.resources.linear_attn.settle_conv import settle_conv

    gen = torch.Generator(device="cuda").manual_seed(accepted)
    rows, block, dim, width, slots = 5, K + 1, 8192 + 64, 3, 8
    conv = torch.randn(slots, dim, width, generator=gen, device="cuda").to(dtype)
    # x as the projection's column view: strided rows
    wide = torch.randn(rows * block, dim + 128, generator=gen, device="cuda").bfloat16()
    x = wide[:, 64: 64 + dim]
    slot_ids = torch.tensor([4, 2, 0, 7, 1], dtype=torch.int32, device="cuda")
    # a column of a packed [rows, block + 1] result, as the sampler's verify returns it
    packed = torch.zeros(rows, block + 1, dtype=torch.long, device="cuda")
    packed[:, block] = torch.tensor([accepted, (accepted + 1) % (K + 1), 0, K, accepted], device="cuda")
    per_row = packed[:, block]
    want = conv.clone()
    addr = slot_ids.long()
    kept = per_row.clamp(max=block - 1) + 1
    _old_settle_conv(want, conv.index_select(0, addr), x, addr, torch.arange(rows, device="cuda") * block, kept)
    got = conv.clone()
    settle_conv(got, slot_ids, per_row, x, block, null_slot_id=0)
    real = torch.tensor([s for s in range(slots) if s != 0], device="cuda")
    assert torch.equal(got.index_select(0, real), want.index_select(0, real))
    assert torch.equal(got[0], conv[0])


# ── an offload with a verify result pending, under MSTAR_KV_DEBUG_ASSERTS ──


@pytest.mark.parametrize("corrupt", [None, "kv", "gdn", "carry"])
def test_an_offload_with_a_pending_verify_result_brings_every_resource_back(monkeypatch, corrupt):
    """Row 2's verify result is recorded but not applied when it is offloaded:
    the offload settles it first, moves its KV pages to host, and gives up its
    resident row; the reload brings the pages back, and its GDN slot and the
    sampler's rows, which stayed on device, are as they were (the engine's
    fingerprints, under the debug switch). The run is then plain greedy. The
    controls change the GDN slot or the sampler's last-token row while the
    request is offloaded, or its KV pages' host copy, and the reload raises."""
    if (reason := _cuda_reason()) is not None:
        pytest.skip(reason)
    import hybrid_toy as toy

    from mstar.engine.resources.kv import manager as manager_mod

    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    twin = _bf16_twin()
    eng = toy.engine(K, toy.TableDrafter(_references(twin), K, lambda j: j == 1, device="cuda", prompts=PROMPTS),
                     shapes=toy.CUDA_SHAPES, offload=True)
    driver = Driver(eng)
    for rid, prompt in PROMPTS.items():
        driver.add(rid, prompt, MAX)
    driver.step([1, 2])
    driver.step([1])
    ledger = eng._drafts[toy.NODE].ledger
    assert ledger.pending(2) and not eng.verify_unrecorded(2)
    assert eng.offload_request(toy.NODE, 2) > 0
    assert not ledger.pending(2) and eng.residency(toy.NODE, 2) == Residency.ON_HOST
    if corrupt == "kv":
        host = eng._resources["kv"]._cpu_pool
        (state,) = host.offloaded[2].values()
        host.cpu_kv_cache[:, state.host_pages[0]] += 1
    elif corrupt == "gdn":
        pool = eng._resources["gdn_state"]
        slot = pool._slots[2]["main"].index
        pool._blocks["state"][:, slot] += 1.0
    elif corrupt == "carry":
        bufs = eng._resources["sampler"]._cg_buffers
        bufs.last_token.master[bufs._rid_to_slot[2]] += 1
    if corrupt is not None:
        with pytest.raises(AssertionError, match="differs from the offload"):
            eng.reload_request(toy.NODE, 2)
            # the device digests compare in the graph's stream and land later, as shutdown reads them
            eng._poll_print_checks(wait=True)
        return
    assert eng.reload_request(toy.NODE, 2)
    driver.run(list(PROMPTS))
    _check(driver.out, twin)
