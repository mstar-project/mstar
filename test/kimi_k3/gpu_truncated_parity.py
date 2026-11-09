"""Kimi K3.1 against its reference forward on a truncated release checkpoint
(``python -m mstar.model.kimi_k3.truncate``: the first layers and the head, experts dequantized
to bf16), on GPU in bf16 or on CPU in fp32.

The real ``Engine`` prefills four prompts in one batch and decodes them greedily; the plain-torch
reference (``mstar/model/kimi_k3/tests/reference.py``, the checkpoint's modeling code) runs over
each whole sequence in fp32 and bf16, teacher-forced on the engine's own tokens (the model is
causal: one forward gives every step's logits). A case passes when the engine's error is within a
few times the reference's own bf16-vs-fp32 spread, its picks agree where the fp32 top-1 leads by
more than twice that spread, and a control (the reference without the MLA output gate; for DSA,
the reference replayed on keys the engine did not pick) is clearly further off. K3.1's DSA runs
with index_topk cut to 16 so decode
goes past it, or (--dsa-topk 2048 --dsa-prompt-len 8192 --prefill-chunk 2048) at the release's top-k
with long prompts prefilled in chunks: the engine's picks (prefill rows' too) are recorded and the
reference is replayed with them
(``forced``), which must match the engine as the dense case does; the picks may differ from the
fp32 reference's own no more often than the bf16 reference's do (an fp32 engine: never).

``--mode tp1`` (engine on --engine-device, reference on --ref-device); ``--mode tp2`` (two ranks:
NCCL on cuda:0-1, or gloo with --engine-device cpu), checked the same way against the reference,
the two ranks' tokens and picks compared, and its prefill rows against the tp1 run's (--tp1).
``--dtype float32`` (CPU) holds the engine to fp32 accuracy. ``--ref-ckpt`` gives the reference other
weights than the engine's (the bf16 cut of an MXFP4 cut: what the packing stands for); ``--compare``
prints the prefill rows against another run's. ``--mode capture``: decode replayed from its captured
graphs against the same decode eager, teacher-forced on the eager tokens (4 rows, an exact bucket,
and 3, padded); with ``--dsa`` past ``--dsa-topk``, the captured DSA decode. Exits nonzero on any failed check.

    python test/kimi_k3/gpu_truncated_parity.py --ckpt <dir> --mode tp1 --out tp1.pt
    python test/kimi_k3/gpu_truncated_parity.py --ckpt <dir> --mode tp2 --out tp2.pt --tp1 tp1.pt
    python test/kimi_k3/gpu_truncated_parity.py --ckpt <dir> --mode tp2 --out c.pt --engine-device cpu \
        --ref-device cpu --dtype float32 --cpu
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import socket
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp

PROMPT_SEEDS = ((0, 24), (1, 37), (2, 30), (3, 45))
DSA_TOPK = 16
# fp32 engine against the fp32 reference: summation order only
FP32_ATOL = 1e-2


def _prompts(vocab: int, device) -> dict[str, torch.Tensor]:
    """Fixed id sequences clear of the reserved range (the release's specials start at 163584)."""
    out = {}
    for seed, length in PROMPT_SEEDS:
        g = torch.Generator().manual_seed(seed)
        high = min(vocab, 160000)
        out[f"p{seed}"] = torch.randint(min(1000, high // 4), high, (length,), generator=g).to(device)
    return out


def _empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def force_small_m_backends(sub) -> dict:
    """Route every small-M GEMM of ``sub`` (its shapes as this process holds them: a TP rank's own) to
    FlashInfer: bf16 products on ``tinygemm`` up to 8 rows and ``cublaslt`` up to 64, fp32 ones on
    ``cublaslt`` (H200 has neither ``tgv`` nor ``cute-dsl``), by loading a table of exactly those shapes.
    A table keyed by TP8 shapes never hits a TP1 or TP2 cut. Returns the shapes."""
    import json
    import tempfile

    from mstar.utils import small_m_gemm

    shapes = {}
    for mod in sub.modules():
        for attr, out in (("_in_proj_weight", "bf16"), ("_qkv_a_g_weight", "bf16"), ("_in_weight", "fp32"),
                          ("_out_weight", "bf16")):
            w = getattr(mod, attr, None)
            if isinstance(w, torch.Tensor) and w.dim() == 2:
                shapes[(int(w.shape[0]), int(w.shape[1]), out)] = attr
        o_proj = getattr(mod, "o_proj", None)
        if hasattr(mod, "_project_out") and o_proj is not None and getattr(mod, "tp_size", 1) > 1:
            shapes[(int(o_proj.weight.shape[0]), int(o_proj.weight.shape[1]), "bf16")] = "kda o_proj"
    entries = []
    for (n, k, out) in shapes:
        rows = [(8, "fi_tinygemm"), (64, "fi_cublaslt")] if out == "bf16" else [(64, "fi_cublaslt")]
        entries += [{"n": n, "k": k, "out": out, "m": m, "backend": b} for m, b in rows]
    cap = list(torch.cuda.get_device_capability())
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"capability": cap, "entries": entries}, f)
    small_m_gemm.load_table(f.name)
    print(f"small_m_gemm forced on {len(shapes)} shapes: {sorted(shapes)}", flush=True)
    return shapes


def build(ckpt: str, device: str, overrides: dict | None = None, group=None, groups=None,
          dtype: torch.dtype = torch.bfloat16, page_size: int | None = None, prefix_slots: int = 0,
          prefill_buckets: list[int] | None = None, force_small_m: bool = False, model_kwargs: dict | None = None):
    """``(engine, model)`` with the checkpoint loaded through ``KimiK31Model.get_submodule``. ``page_size``:
    the latent cache's, as a serving config sets it; ``prefix_slots``: the prefix cache open, rooted in
    the checkpoint, with that many KDA snapshots; ``model_kwargs``: serving model kwargs (--model-kwargs)."""
    from mstar.communication.tensors import LocalTransferEngine
    from mstar.distributed.communication import WorkerParallelGroups
    from mstar.engine.engine import Engine
    from mstar.engine.resources.kv import manager as kv_mod
    from mstar.engine.resources.kv.transfer import TransferEngineInfo
    from mstar.model.kimi_k3.config import KDA_STATE, KV_CACHE, SAMPLER
    from mstar.model.kimi_k3.kimi_k3_model import KimiK31Model
    from mstar.model.kimi_k3.tests import engine_helpers as H
    from mstar.quantization.scope import load_submodule

    kv_mod.KVTransferManager = H._StubTransfer  # one engine, nothing to transfer
    model = KimiK31Model(ckpt, strict_config=False, kda_max_requests=4, tokenizer_mode="byte",
                         kda_conv_dtype=dtype, **(model_kwargs or {}), **({"prefill_token_buckets": prefill_buckets,
                                                  "prefill_capture_batch_sizes": (1, 2, 4)}
                                                 if prefill_buckets else {}))
    if overrides:
        model.config = dataclasses.replace(model.config, **overrides)
        model._check_checkpoint_config = lambda source: None  # the overrides are deliberate
    # as the engine loads a node: under its quantization scope (MXFP4 experts load as stored)
    sub = load_submodule(model, "LLM", {}, device, tp_group=group, autocast_dtype=dtype)
    # the step's logits come back as a row output: a captured decode replays the graph, not the
    # sampler's Python, so a list the sampler appends to would hold the last eager step's (job 1461)
    sub.__class__ = _logits_submodule_class()
    if force_small_m:
        force_small_m_backends(sub)
    if model.config.dsa_long_context:
        # the DSA picks are recorded from Python (generate), which a captured decode skips: this case
        # decodes eager; --mode capture --dsa checks the captured DSA decode against it
        sub.get_accelerator_graph_configs = lambda *args, **kwargs: []
    block = next(m for m in sub.modules() if hasattr(m, "experts_packed"))
    print(f"experts: {block.experts_packed.name if block.experts_packed else 'bf16'}", flush=True)
    engine = Engine(None, autocast_dtype=dtype)
    if groups is None:
        groups = WorkerParallelGroups(num_workers=1, global_rank=0)
    specs = model.get_node_resources()
    for spec in specs:
        if spec.resource_key == KV_CACHE and page_size:
            # above one rank the prefix cache opens only with rank 0 deciding the hits (prefix_cache_tp)
            tp = bool(prefix_slots) and group is not None and group.world_size > 1
            spec.apply_yaml_overrides(page_size=page_size, **({"prefix_cache_tp": True} if tp else {}))
        elif spec.resource_key == KDA_STATE and prefix_slots:
            spec.apply_yaml_overrides(prefix_slots=prefix_slots)
    engine.load_model(
        {"LLM": sub}, specs, parallel_groups=groups, device=torch.device(device),
        model=model if prefix_slots else None,
        transfer_engine_info=TransferEngineInfo(
            my_entity_id=f"e{device}", my_session_id="s", transfer_engine=LocalTransferEngine("localhost")),
        kv_cache_type=dtype)
    greedy = H.GreedySampler()
    engine._resources[SAMPLER] = engine._submodules["LLM"].resources[SAMPLER] = greedy
    sub.node_resources[SAMPLER] = greedy
    engine.warmup()
    return engine, model



def _step(engine, walk: str, inputs: dict[str, torch.Tensor]) -> tuple[dict, dict]:
    """One engine step (as ``engine_helpers.run_step``): each request's token and logits, both read
    from the step's outputs, so a captured replay's are what is checked. A token that is not its
    logits' argmax means the logits are not the step's own."""
    from mstar.model.kimi_k3.tests import engine_helpers as H

    batch = H._batch(walk, inputs)
    engine.prepare_inputs(batch)
    assert not batch.failed_requests, batch.failed_requests
    outputs = engine.exec_and_postprocess(batch)
    assert batch.admit_error is None, batch.admit_error
    engine.finalize_batch(batch)
    tokens, logits = {}, {}
    for rid in inputs:
        row = outputs.per_rid_outputs[rid]
        tokens[rid] = row["new_token"][0]
        logits[rid] = row["logits"][0].reshape(-1).float().cpu()
        if int(logits[rid].argmax()) != int(tokens[rid].reshape(-1)[0]):
            raise RuntimeError(f"{walk} {rid}: token {int(tokens[rid].reshape(-1)[0])} is not its logits' "
                               f"argmax {int(logits[rid].argmax())}: the logits are not this step's")
    return tokens, logits


def _slots_to_positions(selection: torch.Tensor, ctx) -> torch.Tensor:
    """The paged DSA resource's picks (``[rows, topk]`` MLA-cache slots, entries past each row's
    ``attn_lens`` unused) as request-local key positions, -1 padded: a slot's page is the request's
    ``i``-th, so its position is ``i * page_size + offset``. An eager step's context only."""
    sel = selection.detach().cpu()
    pages = ctx.pages.cpu().tolist()
    starts = [*ctx.host_page_start, len(pages)]
    lens = ctx.attn_lens()
    out = torch.full_like(sel, -1)
    for r0, n, req in ctx.spans:
        index = {page: i for i, page in enumerate(pages[starts[req]:starts[req + 1]])}
        for r in range(r0, r0 + n):
            live = sel[r, :lens[r]].tolist()
            out[r, :len(live)] = torch.tensor(
                [index[s // ctx.page_size] * ctx.page_size + s % ctx.page_size for s in live], dtype=out.dtype)
    return out


def generate(engine, prompts: dict[str, torch.Tensor], steps: int, chunk: int | None = None) -> dict[str, dict]:
    """Batched greedy: per request, the tokens, each step's logits and the DSA picks of every row
    that picked (``dsa[query position][layer]``, request-local key positions, -1 padding). With
    ``chunk``, each prompt's prefill runs as consecutive steps of that many tokens, as the engine
    cuts a long prompt (the prompts then share a length)."""
    from mstar.model.kimi_k3.tests import engine_helpers as H

    rids = list(prompts)
    out = {rid: {"tokens": [], "logits": [], "dsa": {}, "prompt": prompts[rid].cpu()} for rid in rids}
    picks: list[tuple[int, torch.Tensor]] = []
    rows: list[tuple[str, int]] = []  # the step's rows in batch order: (request, position)
    dsa = engine._resources.get("dsa")
    if dsa is not None:
        select = dsa.select
        mla_layers = engine._submodules["LLM"].submodule.config.full_attn_layers

        def recording(plane, q, weights, group=None, label=None):
            selection = select(plane, q, weights, group=group, label=label)
            picks.append((mla_layers[plane], _slots_to_positions(selection, dsa.current(label))))
            return selection

        dsa.select = recording

    def run(walk, inputs, positions):
        rows[:] = positions
        result = _step(engine, walk, inputs)
        for layer, selection in picks:
            if selection.shape[0] != len(rows):
                raise RuntimeError(f"DSA picks of {selection.shape[0]} rows for a step of {len(rows)}")
            for r, (rid, pos) in enumerate(rows):
                out[rid]["dsa"].setdefault(pos, {})[layer] = selection[r]
        picks.clear()
        return result

    for rid in rids:
        H.add_request(engine, rid)
    length = len(next(iter(prompts.values())))
    if chunk and any(len(p) != length for p in prompts.values()):
        raise ValueError("chunked prefill here takes prompts of one length")
    bounds = [(s, min(s + chunk, length)) for s in range(0, length, chunk)] if chunk else [(0, None)]
    for start, end in bounds:
        pieces = {rid: p[start:end] for rid, p in prompts.items()}
        step, logits = run("prefill", pieces,
                           [(rid, start + i) for rid, piece in pieces.items() for i in range(len(piece))])
    for k in range(steps + 1):
        for rid in rids:
            out[rid]["tokens"].append(int(step[rid].reshape(-1)[0]))
            out[rid]["logits"].append(logits[rid])
        if k == steps:
            break
        step, logits = run("decode", {rid: step[rid].reshape(1) for rid in rids},
                           [(rid, len(prompts[rid]) + k) for rid in rids])
    for rid in rids:
        engine.remove_request(rid)
        out[rid]["logits"] = torch.stack(out[rid]["logits"])
    return out


def load_reference_weights(ckpt: str, dtype, device) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    index = json.loads((Path(ckpt) / "model.safetensors.index.json").read_text())["weight_map"]
    sd = {}
    for shard in sorted(set(index.values())):
        with safe_open(str(Path(ckpt) / shard), framework="pt", device=str(device)) as f:
            for name in f.keys():
                t = f.get_tensor(name)
                # fp32 tensors (A_log, dt_bias, the convs, the router bias) stay fp32
                sd[name[len("language_model."):]] = t if t.dtype == torch.float32 else t.to(dtype)
    return sd


def _forced(run: dict) -> dict[int, dict[int, torch.Tensor]]:
    """The run's picks as the reference replays them: layer -> query position -> keys."""
    out: dict[int, dict[int, torch.Tensor]] = {}
    for pos, layers in run["dsa"].items():
        for layer, keys in layers.items():
            out.setdefault(layer, {})[pos] = keys
    return out


def _unpicked(run: dict, topk: int) -> dict[int, dict[int, torch.Tensor]]:
    """The DSA control: each query past ``topk`` replayed on visible keys the engine did not pick (the
    earliest, at most ``topk``), so the check must tell a wrong selection apart. The dense reference was
    the control before: on a 4-layer cut at top-16 it sits only ~1 logit from the sparse one, under twice
    a B300 bf16 engine's error on a sensitive prompt (04c dsa/p3: 0.90 vs 0.58, the engine 1.6x the bf16
    reference's spread, the same against a CPU fp32 reference)."""
    out: dict[int, dict[int, torch.Tensor]] = {}
    for pos, layers in run["dsa"].items():
        if pos + 1 <= topk:
            continue
        for layer, keys in layers.items():
            keep = torch.ones(pos + 1, dtype=torch.bool)
            keep[keys[keys >= 0].long().cpu()] = False
            out.setdefault(layer, {})[pos] = torch.arange(pos + 1)[keep][:topk]
    return out


def reference_rows(sd, cfg, run: dict, use_dsa: bool, forced=None, trace=None) -> torch.Tensor:
    """The reference's logits at each step's position, teacher-forced on the engine's tokens."""
    from mstar.model.kimi_k3.tests.reference import reference_logits

    prompt, tokens = run["prompt"], run["tokens"]
    device = next(iter(sd.values())).device
    seq = torch.cat([prompt, torch.tensor(tokens[:-1])]).to(device)
    return reference_logits(sd, seq, cfg, use_dsa=use_dsa, forced=forced, trace=trace,
                            logits_from=len(prompt) - 1).float().cpu()


def _max_err(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).abs().amax(dim=-1)


def check(name: str, logits, ref32, ref16, control, failures: list | None, fp32_engine: bool) -> dict:
    """The engine against the fp32 reference, on the scale of the reference's own bf16-vs-fp32
    spread: its error over that spread, and its token picks on decisive steps (the fp32 top-1
    leading by more than twice the median spread) against the bf16 reference's picks on the same
    steps, since bf16 flips some of those too. ``failures`` None: report only."""
    err, spread, ctrl = _max_err(logits, ref32), _max_err(ref16, ref32), _max_err(control, ref32)
    top2 = ref32.topk(2, dim=-1).values
    decisive = (top2[:, 0] - top2[:, 1]) > 2 * spread.median()
    target = ref32.argmax(-1)
    flips = int((logits.argmax(-1) != target)[decisive].sum())
    ref16_flips = int((ref16.argmax(-1) != target)[decisive].sum())
    ratio = (err / spread.clamp_min(1e-3)).median().item()
    row = {"case": name, "steps": len(err), "err_max": err.max().item(), "err_median": err.median().item(),
           "bf16_spread_median": spread.median().item(), "err_over_spread": ratio,
           "control_median": ctrl.median().item(), "decisive_steps": int(decisive.sum()),
           "decisive_flips": flips, "bf16_ref_decisive_flips": ref16_flips,
           "argmax_agree_all": (logits.argmax(-1) == target).float().mean().item()}
    print(json.dumps(row), flush=True)
    if failures is None:
        return row
    if fp32_engine:
        if err.max() > FP32_ATOL or flips:
            failures.append(f"{name}: an fp32 engine is {err.max():.3g} off the fp32 reference ({flips} flips)")
    else:
        if ratio > 4.0:
            failures.append(f"{name}: engine-vs-fp32 error is {ratio:.1f}x the reference's own bf16 spread")
        if flips > 2 * ref16_flips + 1:
            failures.append(f"{name}: {flips} token flips on {int(decisive.sum())} decisive steps, the bf16 "
                            f"reference {ref16_flips}")
    if ctrl.median() < 2 * err.median():
        failures.append(f"{name}: the control is not clearly off ({ctrl.median():.3g} vs {err.median():.3g})")
    return row


def _top_sets(scores: torch.Tensor, positions: torch.Tensor, topk: int) -> list[set[int]]:
    """Each row's top ``min(topk, position + 1)`` keys: the visible ones only (the rest are -inf)."""
    order = scores.sort(dim=-1, descending=True, stable=True).indices[:, :topk].cpu()
    return [set(row[:min(topk, int(pos) + 1)].tolist()) for row, pos in zip(order, positions, strict=True)]


def pick_stats(run: dict, trace32: dict, trace16: dict, topk: int) -> dict:
    """The engine's DSA picks against the fp32 reference's own, beside the bf16 reference's against
    the same: per (row, layer) whether the top-k set differs. A key swapped across the cut is
    measured by its fp32 score's distance from the cut over the spread (std) of that query's scores."""
    compared = engine_diff = ref16_diff = 0
    gaps: list[float] = []
    ref16_gaps: list[float] = []
    layers = sorted({layer for picks in run["dsa"].values() for layer in picks})
    for layer in layers:
        pos = sorted(p for p, picks in run["dsa"].items() if layer in picks)
        for first in range(0, len(pos), 1024):
            chunk = torch.tensor(pos[first:first + 1024])
            s32 = trace32[layer]["scores"][chunk.to(trace32[layer]["scores"].device)].float()
            s16 = trace16[layer]["scores"][chunk.to(trace16[layer]["scores"].device)].float()
            r32s, r16s = _top_sets(s32, chunk, topk), _top_sets(s16, chunk, topk)
            s32 = s32.cpu()
            for i, p in enumerate(chunk.tolist()):
                keys = run["dsa"][p][layer]
                ours, r32 = set(keys[keys >= 0].tolist()), r32s[i]
                compared += 1
                if ours == r32 and r16s[i] == r32:
                    continue
                row = s32[i]
                visible = row[torch.isfinite(row)]
                cut = min(row[k].item() for k in r32)
                scale = max(visible.std().item(), 1e-12) if visible.numel() > 1 else 1.0
                if r16s[i] != r32:
                    ref16_diff += 1
                    ref16_gaps.extend(abs(row[k].item() - cut) / scale for k in r16s[i] ^ r32)
                if ours != r32:
                    engine_diff += 1
                    gaps.extend(abs(row[k].item() - cut) / scale for k in ours ^ r32)
    return {"picks_compared": compared, "engine_picks_differing": engine_diff,
            "bf16_ref_picks_differing": ref16_diff, "swap_gap_over_std_max": max(gaps, default=0.0),
            "bf16_ref_swap_gap_over_std_max": max(ref16_gaps, default=0.0)}


def evaluate(cases: list[tuple], args, failures: list) -> list[dict]:
    """``cases``: ``(name, config, runs, use_dsa)``. The references load once per dtype."""
    fp32_engine = args.dtype == torch.float32
    rows, ref = [], {}
    for dtype in (torch.float32, torch.bfloat16):
        sd = load_reference_weights(args.ref_ckpt or args.ckpt, dtype, args.ref_device)
        for name, cfg, runs, use_dsa in cases:
            for rid, run in runs.items():
                key = (name, rid)
                trace = {} if use_dsa else None
                ref[key, dtype] = reference_rows(sd, cfg, run, use_dsa=use_dsa, trace=trace)
                if trace is not None:
                    ref[key, dtype, "trace"] = trace
                if use_dsa:
                    ref[key, dtype, "forced"] = reference_rows(sd, cfg, run, use_dsa=True, forced=_forced(run))
                if dtype == torch.float32:
                    if use_dsa:
                        control = reference_rows(sd, cfg, run, use_dsa=True, forced=_unpicked(run, cfg.index_topk))
                    else:
                        control = reference_rows(sd, dataclasses.replace(cfg, mla_use_output_gate=False), run,
                                                 use_dsa=False)
                    ref[key, "control"] = control
        del sd
        _empty_cache()
    for name, cfg, runs, use_dsa in cases:
        for rid, run in runs.items():
            key = (name, rid)
            logits = run["logits"]
            r32, r16, ctrl = ref[key, torch.float32], ref[key, torch.bfloat16], ref[key, "control"]
            if not use_dsa:
                rows.append(check(f"{name}/{rid}", logits, r32, r16, ctrl, failures, fp32_engine))
                continue
            # only the steps past index_topk test the selection
            past = len(run["prompt"]) + torch.arange(len(run["tokens"])) > cfg.index_topk
            own: list[str] = []
            row = check(f"{name}/{rid}", logits[past], r32[past], r16[past], ctrl[past], own, fp32_engine)
            forced = check(f"{name}/{rid}/forced", logits[past], ref[key, torch.float32, "forced"][past],
                           ref[key, torch.bfloat16, "forced"][past], ctrl[past], failures, fp32_engine)
            stats = pick_stats(run, ref[key, torch.float32, "trace"], ref[key, torch.bfloat16, "trace"],
                               cfg.index_topk)
            row.update(stats, forced_err_over_spread=forced["err_over_spread"])
            print(json.dumps({"case": f"{name}/{rid}/picks", **stats}), flush=True)
            if stats["picks_compared"] == 0:
                failures.append(f"{name}/{rid}: the engine recorded no DSA picks past index_topk")
            # Replayed with the engine's picks the reference must match the engine (forced, above).
            # The picks themselves may differ from the fp32 reference's no more often than the bf16
            # reference's do (an fp32 engine: not at all), or only by near-ties: keys swapped across
            # the cut whose fp32 scores sit within bf16's own swap distance of it (twice the bf16
            # reference's largest, at least 0.25 std). On a 28-pick sample the bf16 reference's count
            # alone is noisy (job 1461: 4 on one token sequence, 13 on another for the same prompt).
            allowed = 0 if fp32_engine else 2 * stats["bf16_ref_picks_differing"] + 2
            near_tie = max(0.25, 2 * stats["bf16_ref_swap_gap_over_std_max"])
            far = stats["swap_gap_over_std_max"] > near_tie
            if stats["engine_picks_differing"] > allowed and (fp32_engine or far):
                failures.append(f"{name}/{rid}: {stats['engine_picks_differing']} of {stats['picks_compared']} "
                                f"pick sets differ from the fp32 reference's (bf16 reference: "
                                f"{stats['bf16_ref_picks_differing']}), swaps up to "
                                f"{stats['swap_gap_over_std_max']:.3g} std from the cut (near-tie bound "
                                f"{near_tie:.3g})")
                failures.extend(own)
            rows.append(row)
    return rows


def _dsa_overrides(args, prompt_len: int) -> dict:
    return {"index_topk": args.dsa_topk, "dsa_long_context": True, "max_seq_len": prompt_len + args.steps + 8}


def _dsa_prompts(args, vocab: int, device) -> dict[str, torch.Tensor]:
    """The DSA case's prompts: the dense ones cut under the top-k (decode goes past it), or, with
    --dsa-prompt-len, prompts of that length (the prefill too runs past it)."""
    if not args.dsa_prompt_len:
        return {rid: p[:args.dsa_topk - 4] for rid, p in _prompts(vocab, device).items()}
    out = {}
    for seed in range(args.dsa_prompts):
        g = torch.Generator().manual_seed(100 + seed)
        high = min(vocab, 160000)
        out[f"l{seed}"] = torch.randint(min(1000, high // 4), high, (args.dsa_prompt_len,), generator=g).to(device)
    return out


def run_cases(args, device: str, group=None, groups=None) -> list[tuple]:
    """Dense, then DSA past the top-k: ``(name, config, runs, use_dsa)``."""
    cases = []
    if not args.skip_dense:
        engine, model = build(args.ckpt, device, dtype=args.dtype, group=group, groups=groups,
                              prefill_buckets=args.prefill_buckets, force_small_m=args.force_small_m,
                              model_kwargs=json.loads(args.model_kwargs or "{}"))
        cfg = model.config
        print(f"config: {cfg.num_hidden_layers} layers, kda {cfg.kda_layers}, mla {cfg.full_attn_layers}, "
              f"{cfg.num_experts} experts", flush=True)
        cases.append(("dense", cfg, generate(engine, _prompts(cfg.vocab_size, device), args.steps), False))
        del engine
        _empty_cache()
    vocab = _configs(args)[0].vocab_size
    prompts = _dsa_prompts(args, vocab, device)
    length = max(len(p) for p in prompts.values())
    if length + args.steps <= args.dsa_topk:
        raise SystemExit(f"the DSA case never passes top-k {args.dsa_topk}: prompts of {length} tokens and "
                         f"{args.steps} steps (raise --steps or --dsa-prompt-len)")
    engine, model = build(args.ckpt, device, overrides=_dsa_overrides(args, length), dtype=args.dtype,
                          force_small_m=args.force_small_m, model_kwargs=json.loads(args.model_kwargs or "{}"),
                          group=group, groups=groups)
    print(f"dsa: index_topk {model.config.index_topk}, prompts of {length} tokens"
          f"{f', prefill in chunks of {args.prefill_chunk}' if args.prefill_chunk else ''}", flush=True)
    cases.append(("dsa", model.config, generate(engine, prompts, args.steps, args.prefill_chunk), True))
    del engine
    _empty_cache()
    return cases


def _report(failures: list) -> int:
    for f in failures:
        print("FAIL", f, flush=True)
    print("PASS" if not failures else f"{len(failures)} check(s) failed", flush=True)
    return 1 if failures else 0


def compare_prefill(label: str, runs_by_case: dict, other_path: str) -> None:
    """Each case's prefill rows (the same inputs in both runs) against another run's."""
    other = dict(torch.load(other_path)["cases"])
    for name, runs in runs_by_case.items():
        for rid, run in runs.items():
            diff = (run["logits"][0] - other[name][rid]["logits"][0]).abs().max().item()
            print(json.dumps({"case": f"{label}/{name}/{rid}", "prefill_row_max_diff": diff}), flush=True)


def mode_tp1(args) -> int:
    failures: list[str] = []
    cases = run_cases(args, args.engine_device)
    rows = evaluate(cases, args, failures)
    torch.save({"cases": [(n, r) for n, _, r, _ in cases], "rows": rows}, args.out)
    if args.compare:
        compare_prefill("vs-compare", {n: r for n, _, r, _ in cases}, args.compare)
    return _report(failures)


def _cpu_stubs() -> None:
    from mstar.model.kimi_k3.tests import engine_helpers as H

    class _Patch:
        def setitem(self, d, k, v):
            d[k] = v

        def setattr(self, o, n, v):
            setattr(o, n, v)

    H.install_cpu_stubs(_Patch())


def _tp2_rank(rank: int, args, port: int) -> None:
    from mstar.distributed.all_reduce import AllReduce
    from mstar.distributed.communication import CommGroup, WorkerParallelGroups

    group = CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=[0, 1])
    groups = WorkerParallelGroups(num_workers=2, global_rank=rank, any_parallelism=True,
                                  world_parallel_groups=[(0, 1)], dist_timeout_s=1800)
    groups.add("LLM", group, "tp")
    groups.node_to_instance_comm["LLM"] = group
    if args.engine_device.startswith("cuda"):
        device = f"cuda:{rank}"
        groups.init_dist(init_method=f"tcp://127.0.0.1:{port}", device=torch.device("cuda", rank))
    else:  # gloo, as test_tp_cpu.py
        _cpu_stubs()
        device = "cpu"
        torch.distributed.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                                             world_size=2, timeout=datetime.timedelta(seconds=1800))
        group.device_group = group.cpu_group = torch.distributed.new_group([0, 1])
        group.initialized = True
        group._all_reduce = AllReduce()
    if device == "cpu":
        try:
            cases = run_cases(args, device, group=group, groups=groups)
            torch.save({"cases": [(n, r) for n, _, r, _ in cases]}, f"{args.out}.rank{rank}")
            torch.distributed.barrier()
        finally:
            group.device_group = group.cpu_group = group._all_reduce = None
            torch.distributed.destroy_process_group()
        return
    # On GPUs a rank exits without tearing down: job 1461's ranks saved their results, then hung at the
    # barrier / NCCL teardown with their decode graphs captured (0% GPU, host futex waits) until the
    # job's time limit. mp.spawn's join still waits for both ranks' files.
    import traceback

    code = 0
    try:
        cases = run_cases(args, device, group=group, groups=groups)
        torch.save({"cases": [(n, r) for n, _, r, _ in cases]}, f"{args.out}.rank{rank}")
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def mode_tp2(args) -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(_tp2_rank, args=(args, port), nprocs=2)
    ranks = [dict(torch.load(f"{args.out}.rank{r}")["cases"]) for r in (0, 1)]
    failures: list[str] = []
    # every rank must sample the same tokens and pick the same keys: a rank's own choice would
    # attend other rows than its peer's heads do
    for name, runs in ranks[0].items():
        for rid, run in runs.items():
            peer = ranks[1][name][rid]
            same_picks = all(torch.equal(run["dsa"][p][layer], peer["dsa"][p][layer])
                             for p in run["dsa"] for layer in run["dsa"][p])
            row = {"case": f"tp2-ranks/{name}/{rid}", "tokens_equal": run["tokens"] == peer["tokens"],
                   "logits_max_diff": (run["logits"] - peer["logits"]).abs().max().item(), "picks_equal": same_picks}
            print(json.dumps(row), flush=True)
            if not (row["tokens_equal"] and same_picks):
                failures.append(f"tp2 {name}/{rid}: the two ranks disagree ({row})")
    # rank 0's runs against the reference, as tp1's are
    cases = []
    dense_cfg, dsa_cfg = _configs(args)
    for name, runs in ranks[0].items():
        cases.append((name, dsa_cfg if name == "dsa" else dense_cfg, runs, name == "dsa"))
    rows = evaluate(cases, args, failures)
    # prefill rows: the same inputs at TP1 and TP2, so they differ only by bf16 summation order
    if args.tp1:
        compare_prefill("tp2-vs-tp1", ranks[0], args.tp1)
    torch.save({"rows": rows}, args.out)
    return _report(failures)


def _logits_submodule_class():
    from mstar.model.kimi_k3.submodules import KimiK3LLMSubmodule
    from mstar.model.submodule_base import BatchedModelOutput

    class LogitsSubmodule(KimiK3LLMSubmodule):
        """The submodule, its logits emitted beside the tokens so a replay's can be read back."""

        def forward_batched(self, graph_walk, engine_inputs, input_ids, **kwargs):
            logits = self._logits(graph_walk, engine_inputs, input_ids, kwargs.get("num_valid_tokens"))
            tokens = engine_inputs.resources["sampler"].sample(engine_inputs.request_ids, logits=logits)
            return BatchedModelOutput(row_outputs={"new_token": tokens, "logits": logits},
                                      check_stop_buffers={"new_token": tokens})

    return LogitsSubmodule


class Stepper:
    """The checkpoint's submodule on its own resources and an ``AcceleratorGraphRunner``, stepped as the
    engine steps it (lease, pad, declare, admit, plan, run or replay, commit), after
    test/modular/test_glm5next_mtp_graph.py."""

    def __init__(self, args):
        from mstar.distributed.communication import CommGroup, JointGroups
        from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner
        from mstar.engine.resources import StepRunner
        from mstar.engine.resources.base import EngineResourceInfo, build_resource
        from mstar.engine.resources.kv import manager as kv_mod
        from mstar.engine.resources.spec import resolve_spec_dependencies
        from mstar.model.kimi_k3.kimi_k3_model import KimiK31Model
        from mstar.model.kimi_k3.tests import engine_helpers as H
        from mstar.quantization.scope import load_submodule

        kv_mod.KVTransferManager = H._StubTransfer
        self.dev = torch.device(args.engine_device)
        buckets = getattr(args, "prefill_buckets", None)
        model = KimiK31Model(args.ckpt, strict_config=False, kda_max_requests=8, tokenizer_mode="byte",
                             **json.loads(getattr(args, "model_kwargs", None) or "{}"),
                             **({"prefill_token_buckets": buckets, "prefill_capture_batch_sizes": (1, 2, 4)}
                                if buckets else {}))
        if buckets and model.config.context_limit < 512:  # a tiny dry-run cut's top-k: room for the cases
            model.config = dataclasses.replace(model.config, index_topk=1024, max_seq_len=1024)
            model._check_checkpoint_config = lambda source: None
        if args.dsa:  # past --dsa-topk keys every row selects: the prompts' prefill and the decode
            longest = max(length for _, length in PROMPT_SEEDS)
            model.config = dataclasses.replace(model.config, **_dsa_overrides(args, longest))
            model._check_checkpoint_config = lambda source: None
        sub = load_submodule(model, "LLM", {}, self.dev, autocast_dtype=args.dtype)
        sub.__class__ = _logits_submodule_class()
        if getattr(args, "force_small_m", False):
            force_small_m_backends(sub)
        specs = model.get_node_resources()
        by_key = resolve_spec_dependencies(specs)
        self.resources = {}
        for spec in specs:
            if hasattr(getattr(spec, "config", None), "max_num_pages"):
                spec.config.max_num_pages = 256
            info = EngineResourceInfo(device=self.dev, dependencies={k: by_key[k] for k in spec.depends_on()})
            self.resources[spec.resource_key] = build_resource(spec, info)
        for spec in specs:  # as Engine.load_model: DSA reads its KV cache's pages
            self.resources[spec.resource_key].bind_dependencies(
                {k: self.resources[k] for k in spec.depends_on()})
        if self.dev.type == "cpu":  # a dry run: the sampler's kernels are GPU-only
            self.resources["sampler"] = H.GreedySampler()
        self.runner = StepRunner(self.resources, node_resources={"LLM": list(self.resources)})
        sub.bind_node_resources(self.resources)
        self.sub, self.model = sub, model
        trivial = JointGroups(tp_group=CommGroup.trivial(), sp_group=CommGroup.trivial())
        self.cg = AcceleratorGraphRunner(
            submodule_name="LLM", submodule=sub, resources=self.resources, step_runner=self.runner,
            device=self.dev, autocast_dtype=None, joint_comm_group=trivial, num_slots=2)
        self.cg.warmup_and_capture(None)

    def _info(self, rid: str, walk: str):
        from mstar.conductor.request_info import CurrentForwardPassInfo
        from mstar.engine.resources import SamplingReqConfig

        return CurrentForwardPassInfo(
            request_id=f"wire-{rid}", rid_handle=rid, graph_walk=walk, fwd_index=0, random_seed=0,
            max_tokens=4096, resource_configs={"sampler": SamplingReqConfig(temperature=0.0)},
            dynamic_loop_iter_counts={"decode_loop": 0})

    def step(self, walk: str, inputs: dict[str, torch.Tensor], captured: bool):
        """One step: ``(tokens (n,), logits (n, vocab) fp32 on the host, captured?)``."""
        from mstar.engine.resources import SamplingReqConfig
        from mstar.engine.resources.step import StepContext
        from mstar.model.submodule_base import BatchedModelOutput, ModelInputsFromEngine

        rids = list(inputs)
        if walk == "prefill":
            for rid in rids:
                self.runner.ingest_request(rid, {"sampler": SamplingReqConfig(temperature=0.0)})
        infos = {rid: self._info(rid, walk) for rid in rids}
        node_inputs = [self.sub.prepare_inputs(walk, infos[rid], {"text_inputs": [ids]})
                       for rid, ids in inputs.items()]
        real_inputs = list(node_inputs)
        # the capture the engine would pick: under dsa_long_context, dense or selecting by the rows' contexts
        key = self.sub.cg_key_info(walk, infos)
        tokens = sum(inp.input_seq_len for inp in node_inputs) if walk == "prefill" else None
        lease = (self.cg.lease_slot(walk, bs=len(rids), slot=0, num_tokens=tokens, cg_key_info=key)
                 if captured else None)
        if captured and lease is None:
            raise RuntimeError(f"no captured {walk} graph fits {len(rids)} rows")
        ctx = StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False, slot_lease=lease)
        if lease is not None:
            node_inputs = self.cg.pad_inputs(lease, node_inputs)
            infos = self.cg.step_metadata(lease, rids, infos)
            ctx.set_padded_rids(self.cg.step_ids(lease, rids))
        padded = list(ctx.padded_request_ids)
        step = self.sub.declare_step(walk, padded, node_inputs, slot_lease=lease)
        step.set_ctx(ctx)
        assert self.runner.admit(step).ok
        self.runner.plan(step)
        # as the engine stages a padded step: the real rows' token count (a model may mask the rest)
        ei = ModelInputsFromEngine(request_ids=padded, per_request_info=infos, resources=self.resources,
                                   captured=lease is not None, step=step,
                                   real_seq_len=sum(inp.input_seq_len for inp in real_inputs))
        pre = self.sub.preprocess(walk, ei, node_inputs)
        with torch.no_grad():
            raw = self.cg.run_forward(lease, pre) if lease is not None else self.sub.forward_batched(walk, ei, **pre)
        self.runner.commit(step)
        raw = BatchedModelOutput.coerce(raw)
        n = len(rids)
        tokens = raw.row_outputs["new_token"][:n].reshape(n).clone()
        logits = raw.row_outputs["logits"][:n].float().cpu()
        for i, rid in enumerate(rids):
            self.sub.postprocess(rid, infos[rid], {"new_token": [tokens[i:i + 1]]}, inputs=real_inputs[i])
        if lease is not None:
            self.cg.release(lease, n)
        return tokens, logits

    def remove(self, rids) -> None:
        for rid in rids:
            self.runner.remove_request(rid)
            self.sub.request_states.pop(rid, None)


def mode_capture(args) -> int:
    """Decode replayed from its captured graphs against the same decode run eager, teacher-forced on the
    eager run's tokens: the same rows, the same inputs, step by step. A batch of 3 replays the 4-row graph
    with a padding row; to tell padding from the 4-row graph's kernels (a GEMM or MoE kernel picked for
    4 rows rounds bf16 differently from one picked for 3), the 3 requests also run eager beside a real
    4th request: replay against that is the padding row's effect alone (expected ~0), eager 4 against
    eager 3 the kernels' alone."""
    failures: list[str] = []
    st = Stepper(args)
    vocab = st.model.config.vocab_size
    every = list(_prompts(vocab, st.dev).items())

    def teacher_forced(prefix: str, prompts: dict, tokens: list, captured: bool, extra=None):
        """Logits of ``prompts``' rows over the steps, fed ``tokens``; ``extra`` (rid, prompt) runs beside
        them on its own greedy tokens."""
        batch = {f"{prefix}{r}": p for r, p in prompts.items()}
        if extra is not None:
            batch[f"{prefix}x"] = extra
        tok, _ = st.step("prefill", batch, captured=False)
        own = tok[len(prompts):]
        out = []
        for k in range(args.steps):
            feed = {f"{prefix}{r}": tokens[k][i:i + 1] for i, r in enumerate(prompts)}
            if extra is not None:
                feed[f"{prefix}x"] = own
            tok, logits = st.step("decode", feed, captured=captured)
            own = tok[len(prompts):]
            out.append(logits[:len(prompts)])
        st.remove(list(batch))
        return out

    def compare(name: str, got: list, ref: list) -> dict:
        diffs = [_max_err(g, r).max().item() for g, r in zip(got, ref, strict=True)]
        sure = [(r.topk(2, dim=-1).values[:, 0] - r.topk(2, dim=-1).values[:, 1]) > 0.5 for r in ref]
        flips = sum(int((g.argmax(-1) != r.argmax(-1))[m].sum()) for g, r, m in zip(got, ref, sure, strict=True))
        row = {"case": name, "steps": len(diffs), "max_logit_diff": max(diffs),
               "median_logit_diff": sorted(diffs)[len(diffs) // 2], "decisive": sum(int(m.sum()) for m in sure),
               "flips": flips, "per_step_max": [round(d, 4) for d in diffs]}
        print(json.dumps(row), flush=True)
        return row

    for n in (4, 3):  # an exact capture bucket, and one padded with a dummy row
        prompts = dict(every[:n])
        eager_tokens, eager_logits = [], []
        tok, logits = st.step("prefill", {f"e{r}": p for r, p in prompts.items()}, captured=False)
        for _ in range(args.steps):
            eager_tokens.append(tok)
            tok, logits = st.step("decode", {f"e{r}": tok[i:i + 1] for i, r in enumerate(prompts)}, captured=False)
            eager_logits.append(logits)
        st.remove([f"e{r}" for r in prompts])
        # a CPU dry run has no graphs: its "replay" runs eager (the comparisons then test the rows'
        # independence of the batch around them, in fp32)
        replay = teacher_forced("c", prompts, eager_tokens, captured=st.dev.type != "cpu")
        row = compare(f"capture/bs{n}", replay, eager_logits)
        if n == 4:
            # the same kernels at the same shape: the replay is the eager step
            if row["max_logit_diff"] > 0.25 or row["flips"]:
                failures.append(f"capture/bs4: replay vs eager off by {row['max_logit_diff']:.3g} "
                                f"({row['flips']} flips)")
            continue
        filler = every[3][1]
        beside = teacher_forced("f", prompts, eager_tokens, captured=False, extra=filler)
        # Eager, no padding anywhere: the same 4-row step with only the 4th row changed. Job 1480 found
        # the padded replay 1.19 off the eager 4-row step (median 0.06, no flips), as far as eager 4 rows
        # is from eager 3 (1.46). Rows do not mix logically (CPU: 2e-6 in fp32), so the hypothesis is
        # rounding that depends on the neighbours: the MoE kernels' tiles follow every token's routing
        # (the 4th row's tokens), FlashInfer's MLA split-KV plan every row's length (its length).
        again = teacher_forced("g", prompts, eager_tokens, captured=False, extra=filler)
        high = min(vocab, 160000)
        recolored = torch.randint(min(1000, high // 4), high, filler.shape, generator=torch.Generator().manual_seed(77))
        tokens4 = teacher_forced("h", prompts, eager_tokens, captured=False, extra=recolored.to(filler.device))
        length4 = teacher_forced("i", prompts, eager_tokens, captured=False, extra=filler[:12])
        repeat = compare("eager/bs4_repeat", again, beside)  # determinism: must be 0
        by_tokens = compare("eager/bs4_4th_row_tokens", tokens4, beside)  # same lengths: routing only
        by_length = compare("eager/bs4_4th_row_length", length4, beside)  # lengths (and routing)
        pad = compare("capture/bs3_vs_eager_bs4", replay, beside)  # the padding row in place of the 4th
        kernels = compare("eager/bs4_vs_bs3", beside, eager_logits)  # 4-row kernels vs 3-row ones
        moe = _moe_neighbour_probe(st)
        neighbour = max(by_tokens["max_logit_diff"], by_length["max_logit_diff"])
        if repeat["max_logit_diff"] > 0:
            failures.append(f"capture/bs3: two identical eager 4-row runs differ by {repeat['max_logit_diff']:.3g}")
        # The padding row is one more neighbour: it may move real rows as far as a real neighbour does
        # (with slack for which side of the noise a step lands on), never flip a decisive step.
        if pad["flips"] or pad["max_logit_diff"] > max(0.25, 1.5 * neighbour):
            failures.append(f"capture/bs3: the padding row moves the real rows by {pad['max_logit_diff']:.3g} "
                            f"({pad['flips']} flips); a real 4th row moves them by {neighbour:.3g}")
        if row["flips"] or row["max_logit_diff"] > max(0.25, 1.5 * max(kernels["max_logit_diff"], neighbour)):
            failures.append(f"capture/bs3: replay vs eager off by {row['max_logit_diff']:.3g} ({row['flips']} "
                            f"flips), 4-row vs 3-row kernels eager {kernels['max_logit_diff']:.3g}")
        print(json.dumps({"case": "capture/bs3_attribution", "padding": pad["max_logit_diff"],
                          "neighbour_tokens": by_tokens["max_logit_diff"],
                          "neighbour_length": by_length["max_logit_diff"], "batch_shape": kernels["max_logit_diff"],
                          "moe_rows_moved_by_a_neighbour": moe}), flush=True)
    return _report(failures)


def mode_prefill_capture(args) -> int:
    """Prefill replayed from its packed captures (``prefill_token_buckets``) against the same prefill run
    eager, then decode teacher-forced on the eager run's tokens from what each left. An exact bucket (one
    256-token row) runs the eager step's kernels at its shape: the replay must be the eager step. A padded
    one (fewer tokens than the bucket, or rows) may move the real rows as far as a real neighbour row does
    eager, never flip a decisive step. A continuation chunk replays past the context an eager chunk wrote,
    where that ends on a 64-key boundary (``KimiK3LLMSubmodule.can_replay``)."""
    failures: list[str] = []
    args.prefill_buckets = [64, 128, 256, 512]
    st = Stepper(args)
    vocab = st.model.config.vocab_size
    high = min(vocab, 160000)

    def ids(n: int, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        return torch.randint(min(1000, high // 4), high, (n,), generator=g).to(st.dev)

    def run(prefix: str, prompts: dict, captured: bool, head: dict | None = None, steps: int = 4, feed=None):
        """Prefill ``prompts`` (after ``head``, prefilled eager first, if given), then ``steps`` decodes fed
        ``feed`` (the eager run's tokens) or their own; the prefill's and each decode's logits."""
        rids = {r: f"{prefix}{r}" for r in prompts}
        if head is not None:
            st.step("prefill", {rids[r]: h for r, h in head.items()}, captured=False)
        tok, logits = st.step("prefill", {rids[r]: p for r, p in prompts.items()}, captured=captured)
        rows, tokens = [logits], [tok]
        for k in range(steps):
            src = tokens[-1] if feed is None else feed[k]
            tok, logits = st.step("decode", {rids[r]: src[i:i + 1] for i, r in enumerate(prompts)},
                                  captured=st.dev.type != "cpu")
            rows.append(logits)
            tokens.append(tok)
        st.remove(list(rids.values()))
        return rows, tokens

    def compare(name: str, got: list, ref: list) -> dict:
        diffs = [_max_err(g, r).max().item() for g, r in zip(got, ref, strict=True)]
        sure = [(r.topk(2, dim=-1).values[:, 0] - r.topk(2, dim=-1).values[:, 1]) > 0.5 for r in ref]
        flips = sum(int((g.argmax(-1) != r.argmax(-1))[m].sum()) for g, r, m in zip(got, ref, sure, strict=True))
        row = {"case": name, "prefill_max_diff": round(diffs[0], 4), "decode_max_diff": round(max(diffs[1:]), 4),
               "decisive": sum(int(m.sum()) for m in sure), "flips": flips}
        print(json.dumps(row), flush=True)
        return row

    captured = st.dev.type != "cpu"
    cases = [
        ("exact_1x256", {"a": ids(256, 1)}, None, True),
        ("padded_1x200", {"a": ids(200, 2)}, None, False),
        ("padded_3rows", {"a": ids(100, 3), "b": ids(70, 4), "c": ids(60, 5)}, None, False),
        # a chunk past the 256 tokens an eager chunk wrote: on a 64-key boundary, so it replays
        ("continuation_150_after_256", {"a": ids(150, 6)}, {"a": ids(256, 7)}, False),
    ]
    from mstar.model.submodule_base import ARNodeInputs

    # past a cut the scheduler made mid-way the MLA plan peels the row, which no capture holds: eager
    off = ARNodeInputs(input_seq_len=150, chunk_start=200)
    if captured and st.sub.can_replay("prefill", [off]):
        failures.append("a prefill row starting 200 keys in would replay a capture its peeled plan overflows")
    padded, spreads = [], []
    for name, prompts, head, exact in cases:
        eager, eager_tokens = run("e", prompts, False, head)
        replay, _ = run("c", prompts, captured, head, feed=eager_tokens)
        row = compare(f"prefill_capture/{name}", replay, eager)
        if exact:
            if row["prefill_max_diff"] > 0.25 or row["flips"]:
                failures.append(f"{name}: the replay of an exact bucket is off the eager step ({row})")
            continue
        # a real neighbour row in the padding's place, eager: how far a row's neighbours move it
        filler = {"z": ids(48, 8)}
        beside, _ = run("f", {**prompts, **filler}, False,
                        None if head is None else {**head, "z": ids(16, 9)}, feed=[
                            torch.cat([t, t[:1]]) for t in eager_tokens[:4]])
        n = len(prompts)
        neighbour = compare(f"prefill_capture/{name}/eager_neighbour", [b[:n] for b in beside], eager)
        padded.append((name, row))
        spreads.append(max(neighbour["prefill_max_diff"], neighbour["decode_max_diff"]))
    # One neighbour row is one draw of the bf16 noise a row's company moves it by (job 1733: 0.08 to 1.41
    # across the cases); a padded replay may move a row as far as the run's real neighbours do, never flip
    # a decisive step.
    bound = max([0.25, *(1.5 * s for s in spreads)])
    for name, row in padded:
        if row["flips"] or max(row["prefill_max_diff"], row["decode_max_diff"]) > bound:
            failures.append(f"{name}: the padded replay moves the rows past real neighbours' {bound:.3g} ({row})")
    return _report(failures)


def _moe_neighbour_probe(st) -> dict:
    """One MoE layer alone on 4 tokens: tokens 0-2 fixed, token 3 drawn twice. Rows 0-2 come out
    bitwise equal unless the experts' kernels round a token by its neighbours' routing."""
    layer = next(l for l in st.sub.language_model.model.layers if hasattr(l, "block_sparse_moe"))
    block = layer.block_sparse_moe
    dtype = next(p for p in block.parameters() if p.is_floating_point()).dtype
    g = torch.Generator().manual_seed(5)
    base = torch.randn(4, st.model.config.hidden_size, generator=g).to(st.dev, dtype)
    other = base.clone()
    other[3] = torch.randn(st.model.config.hidden_size, generator=g).to(st.dev, dtype)
    with torch.no_grad():
        a, b, a2 = block(base.clone()), block(other.clone()), block(base.clone())
    return {"layer": layer.layer_idx, "rows_0_2_max_diff": (a[:3] - b[:3]).abs().max().item(),
            "repeat_max_diff": (a - a2).abs().max().item()}


def _configs(args):
    """The dense and DSA configs ``run_cases`` builds, without a model."""
    from mstar.model.kimi_k3.kimi_k3_model import KimiK31Model

    model = KimiK31Model(args.ckpt, strict_config=False, tokenizer_mode="byte")
    length = args.dsa_prompt_len or args.dsa_topk - 4
    return model.config, dataclasses.replace(model.config, **_dsa_overrides(args, length))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=("tp1", "tp2", "capture", "prefill-capture"), default="tp1")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp1", help="the tp1 run's --out, for --mode tp2")
    ap.add_argument("--ref-ckpt", help="the reference's weights (default --ckpt): the bf16 cut of an MXFP4 one")
    ap.add_argument("--compare", help="another tp1 run's --out, whose prefill rows to compare with")
    ap.add_argument("--dsa-topk", type=int, default=DSA_TOPK, help="the DSA case's index_topk (the release: 2048)")
    ap.add_argument("--dsa-prompt-len", type=int, default=0,
                    help="the DSA case's prompt length; 0: the dense prompts cut under the top-k")
    ap.add_argument("--dsa-prompts", type=int, default=2, help="prompts of --dsa-prompt-len")
    ap.add_argument("--prefill-chunk", type=int, default=0, help="prefill the DSA case in chunks of this many tokens")
    ap.add_argument("--skip-dense", action="store_true", help="the DSA case alone")
    ap.add_argument("--dsa", action="store_true",
                    help="--mode capture with DSA past --dsa-topk: captured DSA decode against it eager")
    ap.add_argument("--engine-device", default="cuda:0")
    ap.add_argument("--prefill-buckets", type=lambda v: [int(n) for n in v.split(",")], default=None,
                    help="capture prefill at these packed token counts (within index_topk): the engine replays them")
    ap.add_argument("--force-small-m", action="store_true",
                    help="route every small-M GEMM to FlashInfer at the shapes each process holds")
    ap.add_argument("--ref-device", default="cuda:1")
    ap.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    ap.add_argument("--model-kwargs", default="", help="JSON serving model kwargs for the engine's model, every mode")
    # CPU: the FlashInfer stub and no CUDA (a tiny checkpoint's dry run, or the fp32 check)
    ap.add_argument("--cpu", "--cpu-dry-run", dest="cpu", action="store_true")
    args = ap.parse_args()
    args.dtype = getattr(torch, args.dtype)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.cpu:
        _cpu_stubs()
    return {"tp1": mode_tp1, "tp2": mode_tp2, "capture": mode_capture,
            "prefill-capture": mode_prefill_capture}[args.mode](args)


if __name__ == "__main__":
    os.environ.setdefault("MSTAR_PRE_PLAN_SPEC", "0")
    sys.exit(main())
