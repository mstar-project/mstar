"""The fp32 screen: a speculative arm's greedy replies against plain's at the same batch shapes, judged by a
float32 reference model teacher-forced on each distinct served reply.

The reference gives the served token's rank and deficit at every position (`Fp32Reference.scan`). Per arm
(`screen`): a one-sided exact test of the departure rate against plain's (rho = 1), a hypergeometric test of
departures past plain's same-shape 99th percentile, both failing at p < ``Gates.p_fail``, and tripwires (a
served token at rank >= ``Gates.trip_rank`` or deficit >= ``Gates.trip_deficit``, ``Gates.burst[0]``
departures within ``Gates.burst[1]`` tokens of a reload, or a reply that ends on no stop token where its pair
goes on). Where a reply first differs from plain's is reported, not gated.

Shared by the served Qwen3.5 MTP and Qwen3-8B drafter modules; ``test/modular/test_fp32_screen.py`` pins its
decisions on synthetic inputs. The integration tests are no package: load it by path.
"""
from __future__ import annotations

import random
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable, NamedTuple

import torch


class Gates(NamedTuple):
    """The screen's thresholds: a p-value below ``p_fail`` fails; the tripwires on a served token (rank, and
    deficit unless None), and on ``burst[0]`` departures within ``burst[1]`` tokens after a reload."""

    p_fail: float
    trip_rank: int
    trip_deficit: float | None
    burst: tuple[int, int]


class Scan(NamedTuple):
    """fp32, teacher-forced on a reply: at each position, its argmax, its top-2 log-prob gap, and the
    served token's rank (0: the argmax) and deficit (the argmax's log-prob minus the served token's)."""

    argmax: list[int]
    gaps: list[float]
    ranks: list[int]
    deficits: list[float]


def scan_logprobs(lp: torch.Tensor, ids: list[int]) -> Scan:
    """A reply's `Scan` from the reference's log-probs at its positions (``[len(ids), vocab]``)."""
    top = lp.topk(2)
    served = lp.gather(1, torch.tensor(ids, dtype=torch.long, device=lp.device)[:, None]).squeeze(1)
    return Scan(
        top.indices[:, 0].tolist(), (top.values[:, 0] - top.values[:, 1]).tolist(),
        (lp > served[:, None]).sum(-1).tolist(), (top.values[:, 0] - served).tolist(),
    )


class Fp32Reference:
    """``scan(prompt, ids, image=False)``: the reference model in fp32 over the reply ``ids`` to ``prompt``,
    memoized by (prompt, image, ids): a reply served twice is one sample. A subclass loads the model
    (`_load`) and renders a prompt's inputs as M* prefilled them (`_inputs`). Load it once the compared
    servers have stopped, and release it before the next one starts."""

    def __init__(self):
        self.lm = None
        self.scans: dict[tuple, Scan] = {}

    def _load(self):
        raise NotImplementedError

    def _inputs(self, prompt: str, image: bool) -> dict:
        raise NotImplementedError

    def load(self):
        if self.lm is None:
            self.lm = self._load()
        return self.lm

    def scan(self, prompt: str, ids: list[int], image: bool = False) -> Scan:
        key = (prompt, image, tuple(ids))
        if key in self.scans:
            return self.scans[key]
        self.load()
        inputs = self._inputs(prompt, image)
        start = inputs["input_ids"].shape[1] - 1
        reply = torch.tensor([ids], dtype=inputs["input_ids"].dtype)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], reply], dim=1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        if "mm_token_type_ids" in inputs:
            # the reply is text
            types = inputs["mm_token_type_ids"]
            inputs["mm_token_type_ids"] = torch.cat([types, types.new_zeros(1, reply.shape[1])], dim=1)
        with torch.no_grad():
            logits = self.lm(**{k: v.cuda() for k, v in inputs.items()}).logits
        self.scans[key] = scan_logprobs(logits[0, start: start + len(ids)].float().log_softmax(-1), ids)
        return self.scans[key]

    def release(self) -> None:
        self.lm = None
        torch.cuda.empty_cache()


# ── a run's replies and server events ──


def rounds(reply: Callable[[str, str, str], dict], url: str, name: str, prompts: list[str], copies: int,
           n: int) -> list[dict[int, dict]]:
    """``n`` rounds of ``copies`` copies of ``prompts`` at once, each round in its own seeded arrival order;
    ``reply(url, prompt, request_id)`` is one streamed reply (``ids``, ``rid`` and ``chunks``: each
    chunk's receive time and ids)."""
    prompts = prompts * copies
    out = []
    for r in range(n):
        order = list(range(len(prompts)))
        random.Random(1000 * copies + r).shuffle(order)
        with ThreadPoolExecutor(len(order)) as pool:
            futures = {c: pool.submit(reply, url, prompts[c], f"{name}-r{r}-c{c}") for c in order}
            out.append({c: f.result() for c, f in futures.items()})
    return out


_RELOAD = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) .*?Reloaded request (\S+) on")


def events(log: str) -> list[list]:
    """``[time, request]`` for each reload in a server log whose asctime is UTC (serve with ``TZ=UTC``). The
    engine logs every reload; the moves to the host are rate-limited in the log, so they are not read here."""
    out = []
    for line in log.splitlines():
        m = _RELOAD.match(line)
        if m:
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=timezone.utc).timestamp()
            out.append([t, m.group(2)])
    return out


def _reloads(events) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for t, rid in events or ():
        out.setdefault(rid.split("#")[0], []).append(t)
    return out


def token_at(reply: dict, t: float) -> int:
    """Tokens the client had received of ``reply`` by time ``t``."""
    seen = 0
    for at, ids in reply["chunks"]:
        if at > t:
            break
        seen += len(ids)
    return seen


# ── the screen ──


class Pair(NamedTuple):
    """A speculative reply and the plain reply it is compared with (the same prompt and batch shape)."""

    prompt: str
    image: bool
    plain: dict
    other: dict
    label: str


def pairs_rounds(plain: dict, run: dict, prompts: list[str]) -> list[Pair]:
    """Round by round, copy by copy: both runs took the same arrival orders (`rounds`)."""
    return [Pair(prompts[c % len(prompts)], False, rnd_p[c], rnd[c], f"r{r} c{c} p{c % len(prompts)}")
            for r, (rnd_p, rnd) in enumerate(zip(plain["rounds"], run["rounds"], strict=True)) for c in rnd]


class Departures(NamedTuple):
    """An arm's positions that leave fp32's argmax, each distinct prefix counted once: replies that share
    a prefix share its positions, which are one observation, not several."""

    positions: int
    deficits: list[float]
    private: int
    trips: list[str]


def departures(pairs: list[Pair], ref, side: str, gates: Gates, events=None) -> Departures:
    """``side``'s replies ("plain" or "other") in ``pairs``, each position counted once per distinct prefix
    (a per-prompt trie of the replies): their departures from fp32, those after the pair's split (private),
    and the tripwires. A reply's ids end with its stop token (served replies carry it)."""
    seen, nodes, positions, deficits, private, trips = set(), set(), 0, [], 0, []
    reloads = _reloads(events)
    for p in pairs:
        reply, other = getattr(p, side), (p.other if side == "plain" else p.plain)
        key = (p.prompt, p.image, tuple(reply["ids"]))
        if key in seen:
            continue
        seen.add(key)
        scan = ref.scan(p.prompt, reply["ids"], p.image)
        n = min(len(reply["ids"]), len(other["ids"]))
        split = next((j for j in range(n) if reply["ids"][j] != other["ids"][j]), n)
        ids = reply["ids"]
        # the positions no earlier reply of this prompt reached by the same prefix
        new = [j for j in range(len(ids)) if (p.prompt, p.image, tuple(ids[:j + 1])) not in nodes]
        nodes.update((p.prompt, p.image, tuple(ids[:j + 1])) for j in new)
        positions += len(new)
        left = [j for j in new if scan.ranks[j] > 0]
        deficits += [scan.deficits[j] for j in left]
        mine = [j for j in left if j >= split]
        private += len(mine)
        for j in left:
            if scan.ranks[j] >= gates.trip_rank or (
                    gates.trip_deficit is not None and scan.deficits[j] >= gates.trip_deficit):
                trips.append(f"{p.label}: token {j} at fp32 rank {scan.ranks[j]}, deficit {scan.deficits[j]:.3f}")
        if len(ids) < len(other["ids"]) and split == len(ids):
            # a reply's ids end with its stop token, which the scan scores like any other; a strict prefix of its
            # pair stopped on no token (truncated, or empty)
            trips.append(f"{p.label}: ended at token {len(ids)} on no stop token where its pair went on")
        for t in reloads.get(reply["rid"] or "", ()):
            at = token_at(reply, t)
            if sum(at <= j < at + gates.burst[1] for j in mine) >= gates.burst[0]:
                trips.append(f"{p.label}: {gates.burst[0]}+ private departures within {gates.burst[1]} tokens "
                             f"of a reload at {at}")
    return Departures(positions, deficits, private, trips)


def rate_test(spec: Departures, plain: Departures) -> tuple[float, float, tuple[float, float]]:
    """One-sided exact test of spec's departure rate against plain's (rho = 1), conditional on their total:
    p, the rate ratio, and its 95% CI (Clopper-Pearson on the conditional binomial)."""
    from scipy.stats import beta, binom

    xs, xp, ns, np_ = len(spec.deficits), len(plain.deficits), spec.positions, plain.positions
    total, share = xs + xp, ns / (ns + np_)
    p = float(binom.sf(xs - 1, total, share)) if total else 1.0
    ratio = (xs / ns) / (xp / np_) if xp and np_ else float("inf") if xs else 1.0

    def rho(pi: float) -> float:
        return float("inf") if pi >= 1 else pi * np_ / ((1 - pi) * ns)

    lo = rho(beta.ppf(0.025, xs, total - xs + 1)) if xs else 0.0
    hi = rho(beta.ppf(0.975, xs + 1, total - xs)) if xs < total else float("inf")
    return p, ratio, (lo, hi)


def tail_test(spec: Departures, plain: Departures) -> tuple[float, float, int, int]:
    """Spec departures deeper than plain's same-shape 99th percentile, against plain's own, by the
    hypergeometric draw of spec's departures from the pooled ones: p, the threshold, and the two counts."""
    from scipy.stats import hypergeom

    if not plain.deficits:
        return 1.0, float("inf"), 0, 0
    q = float(torch.tensor(plain.deficits).quantile(0.99))
    ks, kp = sum(d > q for d in spec.deficits), sum(d > q for d in plain.deficits)
    total, xs = len(spec.deficits) + len(plain.deficits), len(spec.deficits)
    p = float(hypergeom.sf(ks - 1, total, ks + kp, xs)) if ks else 1.0
    return p, q, ks, kp


def splits(pairs: list[Pair], ref, events=None) -> list[str]:
    """Where each speculative reply first leaves its plain pair: reported, not gated."""
    reloads = _reloads(events)
    out = []
    for p in pairs:
        a, b = p.plain["ids"], p.other["ids"]
        n = min(len(a), len(b))
        at = next((j for j in range(n) if a[j] != b[j]), None)
        if at is None:
            continue
        sp, ss = ref.scan(p.prompt, a, p.image), ref.scan(p.prompt, b, p.image)
        left = "both" if sp.ranks[at] and ss.ranks[at] else "plain" if sp.ranks[at] else "spec" if ss.ranks[at] else "?"
        reloaded = any(token_at(p.other, t) <= at for t in reloads.get(p.other["rid"] or "", ()))
        out.append(f"{p.label} at {at}: t_p {a[at]} t_s {b[at]}, t_s fp32 rank {ss.ranks[at]} deficit "
                   f"{ss.deficits[at]:.3f}, left fp32: {left}" + (", reloaded before" if reloaded else ""))
    return out


def screen(label: str, groups: list[tuple[list[Pair], list[Pair] | None, list | None]], ref,
           gates: Gates) -> list[str]:
    """One speculative arm against plain: ``groups`` are (pairs, the plain tail pool's pairs, the arm's
    server events) per batch shape. Prints the counts, the rate ratio and its CI, the tail test and the
    splits; returns the failed gates."""
    spec = plain = Departures(0, [], 0, [])
    plain_tail = Departures(0, [], 0, [])
    found = []
    for pairs, tail_pairs, evts in groups:
        s, p = departures(pairs, ref, "other", gates, evts), departures(pairs, ref, "plain", gates)
        t = departures(tail_pairs or pairs, ref, "plain", gates)
        spec = Departures(spec.positions + s.positions, spec.deficits + s.deficits, spec.private + s.private,
                          spec.trips + s.trips)
        plain = Departures(plain.positions + p.positions, plain.deficits + p.deficits, plain.private + p.private,
                           plain.trips + p.trips)
        plain_tail = Departures(plain_tail.positions + t.positions, plain_tail.deficits + t.deficits, 0, [])
        found += splits(pairs, ref, evts)
    p_rate, ratio, ci = rate_test(spec, plain)
    p_tail, q, ks, kp = tail_test(spec, plain_tail)
    print(f"{label}: departures spec {len(spec.deficits)}/{spec.positions} ({spec.private} private), plain "
          f"{len(plain.deficits)}/{plain.positions} ({plain.private} private); rate ratio {ratio:.2f} "
          f"(95% CI {ci[0]:.2f}-{ci[1]:.2f}), p {p_rate:.2g}; past plain's q99 {q:.3f}: spec {ks}, plain {kp}, "
          f"p {p_tail:.2g}")
    for line in found:
        print(f"  split {label} {line}")
    failed = []
    if p_rate < gates.p_fail:
        failed.append(f"{label}: departure rate {ratio:.2f}x plain's, p {p_rate:.2g}")
    if p_tail < gates.p_fail:
        failed.append(f"{label}: {ks} departures past plain's q99 {q:.3f} (plain {kp}), p {p_tail:.2g}")
    failed += [f"{label} tripwire: {t}" for t in spec.trips]
    for t in plain.trips:
        # plain's own: reported beside the arm, not counted against it
        print(f"  plain tripwire (beside {label}): {t}")
    return failed
