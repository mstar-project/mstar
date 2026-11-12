"""Qwen3.5 drafting with its MTP head on a CUDA device: a tiny random model
(`qwen3_5_cpu.CUDA_TEXT`, head dims 128) with the device's own kernels
(FlashInfer's GDN and attention, FA3 for an asynchronous verify, the Triton
conv), bf16, eager and captured. The captured cases replay every verify step
on resident tables: the MTP head's draft passes and observe, the target, the
verify, the GDN rerun and the carry in one graph.

What it checks, since greedy output alone holds whatever the drafts are:

- every verify step's drafts, read through a capture-safe probe, against the
  MTP chain recomputed on what the step kept, teacher-forced on the served
  drafts (`forced_draft_logprobs`, the bf16-rounded weights in float32), by a
  bar calibrated on a bf16 run of the same reference (see `_check_drafts`;
  the CPU suite holds the chain to the strict margin rule);
- forced acceptance inside the graphs: drafts gathered from a static device
  table of the plain run's continuation, spoiled from a chosen position, so
  every accepted count 0..k runs captured;
- synchronous == asynchronous exactly, one request at a time;
- a padded packed prefill (three prompts on a four-row capture) and more
  requests registered than the largest decode capture, run in groups that
  replay with sampler rows past it (the carry's loop-back master);
- greedy output by the margin rule against a float32 reference.

Skips without CUDA and FlashInfer (and, asynchronous, FA3's paged kernel)."""
from __future__ import annotations

import collections

import pytest
import torch

PROMPTS = {1: [5, 9, 2, 7, 11, 3], 2: [17, 4], 3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
# more than the largest decode capture (4): a second group replays with sampler rows past it
MANY = {**PROMPTS, 4: [33, 34, 35], 5: [60, 2], 6: [7, 7, 7, 7, 1]}
MAX = 16
K = 3
MARGIN = 0.1
DEVICE = "cuda"
# The draft check's bar is the reference's own bf16 run against its float32 run, pooled over
# every device test's steps in the module: on these random weights a chained bf16 pass flips
# drafts at gaps up to ~0.75 by itself. The device is a second, independent bf16 computation:
# pooled over 261 steps its largest gap ran 0.33 past the control's at a draft index (0.475 vs
# 0.15 at draft 1), and its exact steps 8 short of the control's 214. A wrong chained hidden or a
# shifted rotary row lands 5-10 below the top-1, with exact steps falling to 7 and 74.
DRAFT_GAP_SLACK = 0.35
EXACT_STEP_FRACTION = 0.1
# filled by `_check_drafts`, judged by the module's last test
_POOLED = {"control_worst": [0.0] * K, "device": [], "exact": 0, "control_exact": 0, "steps": 0}


def _cuda_reason() -> str | None:
    if not torch.cuda.is_available():
        return "needs CUDA"
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        return "needs flashinfer"
    return None


def _fa3_reason(head_dim: int = 128) -> str | None:
    from mstar.engine.resources.attn.fa3_verify import fa3_verify_unavailable_reason

    return _cuda_reason() or fa3_verify_unavailable_reason(torch.device("cuda"), head_dim, torch.bfloat16)


def _skip(mode: str, head_dim: int = 128) -> None:
    if (reason := (_fa3_reason(head_dim) if mode == "asynchronous" else _cuda_reason())) is not None:
        pytest.skip(reason)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    from qwen3_cpu import _StubTransfer

    from mstar.engine.resources.kv import manager as manager_mod
    from mstar.model.qwen3_5.submodules import LLMSubmodule

    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    # a single-process engine has no transfer engine for its KV pool
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    # a few captures, not the production ladders
    monkeypatch.setattr(LLMSubmodule, "DECODE_CAPTURE_BATCH_SIZES", [1, 2, 4])
    monkeypatch.setattr(LLMSubmodule, "PREFILL_CAPTURE_BATCH_SIZES", [1, 2, 4])
    monkeypatch.setattr(LLMSubmodule, "PREFILL_MEDIA_TOKEN_BUCKETS", [64])
    monkeypatch.setattr(LLMSubmodule, "PREFILL_MEDIA_CAPTURE_BATCH_SIZES", [1])


def _model(tmp_path, monkeypatch, k=None, mode="synchronous", head_dim=128):
    from qwen3_5_cpu import CUDA_TEXT, CUDA_TEXT_256, make_model, write_checkpoint

    text = CUDA_TEXT_256 if head_dim == 256 else CUDA_TEXT
    return make_model(monkeypatch, write_checkpoint(tmp_path, **text), k=k, mode=mode,
                      prefill_token_buckets=[32, 64])


class _Probe:
    """The MTP head's drafts of the last verify step, copied into a static
    buffer inside ``propose`` (a capture records the copy, so a replayed step's
    drafts are read as an eager one's); ``table`` forces the proposals to a
    device table of a continuation, keyed by (position, anchor token)."""

    def __init__(self, monkeypatch, rows: int = 8, table: torch.Tensor | None = None):
        from mstar.model.qwen3_5.components.mtp import Qwen3_5MTPDrafter

        self.buf = torch.zeros(rows, K, dtype=torch.long, device=DEVICE)
        real, probe = Qwen3_5MTPDrafter.propose, self

        def propose(drafter, inputs):
            drafts = real(drafter, inputs)
            n = drafts.shape[0]
            probe.buf[:n].copy_(drafts)
            if table is None:
                return drafts
            at = inputs.resources["rope"].pos_ids("main")[: n * (K + 1)].view(n, K + 1)[:, 0].long()
            return table[at.clamp(max=table.shape[0] - 1), inputs.input_ids.reshape(-1).long()]

        monkeypatch.setattr(Qwen3_5MTPDrafter, "propose", propose)


def _table(source: dict[int, list[int]], prompts: dict[int, list[int]], spoil, vocab: int, positions: int = 256):
    """Each (position, anchor) of ``source``'s continuations to its next ``K``
    tokens, spoiled where ``spoil(j)``; prompt positions are never anchors."""
    table = torch.zeros(positions, vocab, K, dtype=torch.long)
    for rid, seq in source.items():
        for at in range(len(prompts[rid]) - 1, len(seq) - 1):
            row = [seq[at + 1 + j] if at + 1 + j < len(seq) else 0 for j in range(K)]
            table[at, seq[at]] = torch.tensor([(t + 1) % vocab if spoil(j) else t for j, t in enumerate(row)])
    return table.to(DEVICE)


def _run(model, captured: bool, prompts=PROMPTS, groups=None, probe: _Probe | None = None, batched: bool = True,
         chunks: dict[int, list[int]] | None = None, leases: list | None = None):
    """Prefill ``prompts`` (packed, unless not ``batched``; a row in
    ``chunks`` in chunks ending at those bounds) and decode them, in
    ``groups`` of rows when given. Returns the driver and each verify step's
    (rid, reply length before it, drafts); every captured verify step must
    have replayed on resident tables. ``leases`` collects each prompt step's
    leased bucket (None when it ran eager)."""
    from qwen3_5_cpu import Driver, engine

    eng = engine(model, device=DEVICE, captured=captured)
    resident: list[bool] = []
    real = eng.reserve_replay_slot

    def reserve(batch):
        lease = real(batch)
        if batch.step_context.graph_walk == "decode":
            resident.append(lease is not None and batch.step_context.resident is not None)
        elif leases is not None:
            leases.append(None if lease is None else lease.bucket)
        return lease

    eng.reserve_replay_slot = reserve
    steps = []

    class _Probed(Driver):
        def step(self, rids):
            before = {rid: len(self.out[rid]) for rid in rids}
            stopped = super().step(rids)
            if probe is not None:
                drafts = probe.buf[: len(rids)].cpu()
                steps.extend((rid, before[rid], drafts[i].tolist()) for i, rid in enumerate(rids))
            return stopped

    driver = _Probed(eng, model)
    for rid, prompt in prompts.items():
        driver.add(rid, prompt, MAX)
    chunks = chunks or {}
    for rid in chunks:
        driver.prefill_chunks(rid, prompts[rid], chunks[rid])
    rest = [rid for rid in prompts if rid not in chunks]
    if batched and rest:
        driver.prefill(rest, prompts)
    else:
        for rid in rest:
            driver.prefill([rid], prompts)
    for group in groups or [list(prompts)]:
        driver.run(group)
    if captured:
        assert eng._submodules["LLM"].resident is not None, "the node built no resident planner"
        assert resident and all(resident), resident
    return driver, steps


def _check_output(model, out, prompts=PROMPTS):
    from qwen3_5_cpu import reference_greedy

    for rid, prompt in prompts.items():
        margins: list[float] = []
        ref = reference_greedy(model, prompt, MAX, margins=margins, round_to=torch.bfloat16)
        n = min(len(ref), len(out[rid]))
        at = next((j for j in range(n) if ref[j] != out[rid][j]), None if len(ref) == len(out[rid]) else n)
        assert at is None or margins[at] <= MARGIN, (rid, at, ref, out[rid])


def _check_drafts(model, prompts, out, steps) -> int:
    """Scores each step's drafts against the MTP chain recomputed on what the row
    had kept, teacher-forced on the served drafts, and the bf16 control (the same
    reference under bf16 autocast) on the same steps, into `_POOLED`; the module's
    last test judges them together. Returns how many steps were scored."""
    from qwen3_5_cpu import forced_draft_logprobs

    for rid, emitted, drafts in steps:
        seq = prompts[rid] + out[rid][:emitted]
        ref = forced_draft_logprobs(model, seq, drafts)
        control = forced_draft_logprobs(model, seq, drafts, bf16=True)
        _POOLED["exact"] += all(int(lp.argmax()) == d for lp, d in zip(ref, drafts, strict=True))
        _POOLED["control_exact"] += all(
            int(lp.argmax()) == int(c.argmax()) for lp, c in zip(ref, control, strict=True)
        )
        _POOLED["steps"] += 1
        for i, (lp, c, d) in enumerate(zip(ref, control, drafts, strict=True)):
            top = int(lp.argmax())
            _POOLED["control_worst"][i] = max(_POOLED["control_worst"][i], float(lp[top] - lp[int(c.argmax())]))
            if d != top:
                _POOLED["device"].append((i, float(lp[top] - lp[d]), rid, emitted))
    return len(steps)


@pytest.mark.parametrize("captured", [False, True], ids=["eager", "captured"])
def test_plain_qwen3_5_on_the_device_matches_the_reference(tmp_path, monkeypatch, captured):
    _skip("synchronous")
    model = _model(tmp_path, monkeypatch)
    driver, _ = _run(model, captured)
    _check_output(model, driver.out)


@pytest.mark.parametrize("mode, captured", [
    ("synchronous", False), ("synchronous", True), ("asynchronous", True),
], ids=["sync-eager", "sync-captured", "async-captured"])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_every_verify_steps_drafts_are_the_mtp_chain(tmp_path, monkeypatch, mode, captured, head_dim):
    """Three prompts packed into a four-row prefill capture (a padding row),
    then decoded; each verify step's drafts by the margin rule. Head dims 128
    and 256, the released checkpoints'."""
    _skip(mode, head_dim)
    model = _model(tmp_path, monkeypatch, k=K, mode=mode, head_dim=head_dim)
    probe = _Probe(monkeypatch)
    leases: list = []
    driver, steps = _run(model, captured, probe=probe, leases=leases)
    if captured:
        # the three prompts replayed the four-row prefill capture: one padding row
        assert leases and leases[0] is not None and leases[0].bs == 4, leases
    assert driver.eng._drafts["LLM"].asynchronous == (mode == "asynchronous")
    _check_output(model, driver.out)
    assert _check_drafts(model, PROMPTS, driver.out, steps) >= len(PROMPTS)


@pytest.mark.parametrize("mode", ["synchronous", "asynchronous"])
def test_more_requests_than_the_largest_capture_keep_their_carry(tmp_path, monkeypatch, mode):
    """Six requests registered, decoded in groups of four and two: the second
    group replays with sampler rows past the capture's batch size, where a
    loop-back master that grew would be a tensor the graph never saw."""
    _skip(mode)
    model = _model(tmp_path, monkeypatch, k=K, mode=mode)
    probe = _Probe(monkeypatch)
    driver, steps = _run(model, True, prompts=MANY, groups=[[1, 2, 3, 4], [5, 6]], probe=probe, batched=False)
    _check_output(model, driver.out, MANY)
    assert _check_drafts(model, MANY, driver.out, steps) >= len(MANY)


@pytest.mark.parametrize("mode", ["synchronous", "asynchronous"])
@pytest.mark.parametrize("first_wrong", [None, 0, 1, 2])
def test_every_accepted_count_runs_inside_the_graphs(tmp_path, monkeypatch, first_wrong, mode):
    """Drafts forced from a static device table of the plain run's
    continuation, spoiled from ``first_wrong`` on: each accepted count runs
    captured (the GDN rerun, the KV take-back, the head's observe at that
    count), and the drafts the head made are checked as above."""
    _skip(mode)
    plain_model = _model(tmp_path / "plain", monkeypatch)
    plain, _ = _run(plain_model, True)
    source = {rid: p + plain.out[rid] for rid, p in PROMPTS.items()}
    model = _model(tmp_path / "mtp", monkeypatch, k=K, mode=mode)
    spoil = (lambda j: False) if first_wrong is None else (lambda j: j >= first_wrong)
    probe = _Probe(monkeypatch, table=_table(source, PROMPTS, spoil, model.config.vocab_size))
    driver, steps = _run(model, True, probe=probe)
    _check_output(model, driver.out)
    want = K if first_wrong is None else first_wrong
    counts = collections.Counter(driver.accepted)
    assert counts[want] >= 1, counts
    _check_drafts(model, PROMPTS, driver.out, steps)


def test_synchronous_and_asynchronous_agree_exactly_one_request_at_a_time(tmp_path, monkeypatch):
    _skip("asynchronous")
    runs = []
    for mode in ("synchronous", "asynchronous"):
        model = _model(tmp_path / mode, monkeypatch, k=K, mode=mode)
        driver, _ = _run(model, True, groups=[[rid] for rid in PROMPTS], batched=False)
        runs.append((driver.out, driver.accepted))
    assert runs[0] == runs[1]


def test_one_token_prompts_and_final_chunks_run_the_mtp_prompt_pass(tmp_path, monkeypatch):
    """Eager prompt steps whose rows are one token each plan FlashInfer's
    decode wrapper, which keeps no query offsets: a one-token prompt, and a
    prompt whose final chunk is its last token. The MTP head's prompt pass
    takes each row as its own last, and the drafts after are checked."""
    _skip("synchronous")
    model = _model(tmp_path, monkeypatch, k=K)
    prompts = {1: [42], 2: [5, 9, 2, 7, 11, 3]}
    probe = _Probe(monkeypatch)
    driver, steps = _run(model, False, prompts=prompts, probe=probe, chunks={2: [5, 6]}, batched=False)
    _check_output(model, driver.out, prompts)
    assert _check_drafts(model, prompts, driver.out, steps) >= len(prompts)


def test_the_drafts_hold_to_the_pooled_bf16_control():
    """Last in the module: every scored draft against the bf16 control pooled over
    all the module's steps. A draft's gap below the reference's top-1 may pass the
    control's largest at its index by `DRAFT_GAP_SLACK`; the exact steps may fall
    short of the control's by `EXACT_STEP_FRACTION` of the steps."""
    if not _POOLED["steps"]:
        pytest.skip("no device test scored drafts")
    worst = _POOLED["control_worst"]
    print(f"pooled over {_POOLED['steps']} steps: exact {_POOLED['exact']}, control {_POOLED['control_exact']}, "
          f"control worst {[round(w, 3) for w in worst]}, device worst "
          f"{[round(max([g for j, g, *_ in _POOLED['device'] if j == i], default=0.0), 3) for i in range(K)]}")
    for i, gap, rid, emitted in _POOLED["device"]:
        assert gap <= worst[i] + DRAFT_GAP_SLACK, (
            f"rid {rid} after {emitted}: draft {i} is {gap:.3f} below the reference's top-1; "
            f"bf16 alone reaches {worst[i]:.3f} there"
        )
    slack = EXACT_STEP_FRACTION * _POOLED["steps"]
    assert _POOLED["exact"] >= _POOLED["control_exact"] - slack, (_POOLED["exact"], _POOLED["control_exact"])
