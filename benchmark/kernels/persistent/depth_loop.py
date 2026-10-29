"""The code-predictor depth loop: resident kernel against the captured per-op path.

    python -m benchmark.kernels.persistent.depth_loop --out DIR [--weights checkpoint|random] [--bs 27]

At one batch size, for the Qwen3-Omni code predictor's shapes (5 layers, 16
positions, hidden 1024, 16/8 heads, inter 3072, vocab 2048):

1. phase-by-phase agreement of the resident kernel with a float64 mirror of its
   own rounding (stopping the kernel after a phase), teacher-forced, so a wrong
   phase is named instead of only the logits being off;
2. parity: the logits of every sampled group, teacher-forced, from the kernel
   and from the per-op path, each against a float64 evaluation of the same
   weights (relative error per group, its sum over the groups, argmax
   agreement);
3. the sampler: codes drawn by the kernel for one group's fixed logits against
   the top-k / top-p distribution, beside the same count of draws from the
   SamplerResource kernels (total variation distance);
4. time per frame of each, captured: the per-op frame as one graph, the kernel
   as one launch (plus its scratch reset) in a graph, by row-group count, sync
   scope and grid; one replay per sync, median over replays.

The float64 evaluation interprets the same ``StepProgram`` the kernel runs
(``step_program.interpret``).
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from benchmark.kernels.persistent import perop
from mstar.engine import step_program as sp
from mstar.kernels import depth_step as ds

OMNI = sp.DepthDecoderShape(layers=5, positions=16, hidden=1024, heads=16, kv_heads=8, head_dim=128,
                            intermediate=3072, vocab=2048, eps=1e-6, rope_theta=1e6)
REPO = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


# ---------------------------------------------------------------------------------------------------------
# weights and inputs

def checkpoint_weights(device) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """The code predictor's weights from the cached checkpoint, in the program's
    names (q/k/v and gate/up fused, the LM heads stacked); the residual codebooks;
    the talker's layer-0 codec embedding."""
    import json as _json

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    root = Path(snapshot_download(REPO, allow_patterns=["*.json"]))
    index = _json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    pre = "talker.code_predictor."
    wanted = [k for k in index if k.startswith(pre)] + ["talker.model.codec_embedding.weight"]
    raw: dict[str, torch.Tensor] = {}
    for shard in sorted({index[k] for k in wanted}):
        with safe_open(str(Path(snapshot_download(REPO, allow_patterns=[shard])) / shard), "pt") as f:
            for k in wanted:
                if index[k] == shard:
                    raw[k] = f.get_tensor(k).to(torch.bfloat16)
    w: dict[str, torch.Tensor] = {}
    for li in range(OMNI.layers):
        lp = f"{pre}model.layers.{li}."
        o = f"model.layers.{li}."
        w[o + "self_attn.qkv_proj.weight"] = torch.cat(
            [raw[lp + f"self_attn.{n}_proj.weight"] for n in ("q", "k", "v")], dim=0)
        w[o + "mlp.gate_up_proj.weight"] = torch.cat([raw[lp + "mlp.gate_proj.weight"], raw[lp + "mlp.up_proj.weight"]],
                                                     dim=0)
        for n in ("self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                  "input_layernorm.weight", "post_attention_layernorm.weight", "mlp.down_proj.weight"):
            w[o + n] = raw[lp + n]
    w["model.norm.weight"] = raw[pre + "model.norm.weight"]
    w["lm_head_weight"] = torch.stack([raw[pre + f"lm_head.{g}.weight"] for g in range(OMNI.positions - 1)])
    codebook = torch.stack([raw[pre + f"model.codec_embedding.{g}.weight"] for g in range(OMNI.positions - 1)])
    w = {k: v.to(device).contiguous() for k, v in w.items()}
    return w, codebook.to(device).contiguous(), raw["talker.model.codec_embedding.weight"].to(device)


def random_weights(device, seed: int = 0) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    s = OMNI
    g = torch.Generator(device="cpu").manual_seed(seed)

    def lin(n, k):
        return (torch.randn(n, k, generator=g) / math.sqrt(k)).to(torch.bfloat16)

    def norm(n):
        return (1.0 + 0.1 * torch.randn(n, generator=g)).to(torch.bfloat16)

    w: dict[str, torch.Tensor] = {}
    q = s.heads * s.head_dim
    for li in range(s.layers):
        o = f"model.layers.{li}."
        w[o + "self_attn.qkv_proj.weight"] = lin(q + 2 * s.kv_heads * s.head_dim, s.hidden)
        w[o + "self_attn.o_proj.weight"] = lin(s.hidden, q)
        w[o + "self_attn.q_norm.weight"] = norm(s.head_dim)
        w[o + "self_attn.k_norm.weight"] = norm(s.head_dim)
        w[o + "input_layernorm.weight"] = norm(s.hidden)
        w[o + "post_attention_layernorm.weight"] = norm(s.hidden)
        w[o + "mlp.gate_up_proj.weight"] = lin(2 * s.intermediate, s.hidden)
        w[o + "mlp.down_proj.weight"] = lin(s.hidden, s.intermediate)
    w["model.norm.weight"] = norm(s.hidden)
    w["lm_head_weight"] = torch.stack([lin(s.vocab, s.hidden) for _ in range(s.positions - 1)])
    codebook = (0.05 * torch.randn(s.positions - 1, s.vocab, s.hidden, generator=g)).to(torch.bfloat16)
    talker = (0.05 * torch.randn(3072, s.hidden, generator=g)).to(torch.bfloat16)
    return ({k: v.to(device).contiguous() for k, v in w.items()}, codebook.to(device).contiguous(),
            talker.to(device))


def make_inputs(bs: int, talker_codec: torch.Tensor, device, seed: int = 1, same_rows: bool = False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    last_hidden = torch.randn(bs, OMNI.hidden, generator=g).to(torch.bfloat16)
    layer0 = torch.randint(0, talker_codec.shape[0], (bs,), generator=g)
    forced = torch.randint(0, OMNI.vocab, (bs, OMNI.positions), generator=g)
    if same_rows:
        last_hidden[:] = last_hidden[0]
        layer0[:] = layer0[0]
        forced[:] = forced[0]
    last_hidden = last_hidden.to(device)
    embed0 = talker_codec[layer0.to(device)].contiguous()
    forced[:, 0] = layer0
    return last_hidden, embed0, forced.to(device)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def group_errors(logits: torch.Tensor, ref: torch.Tensor) -> dict[str, Any]:
    """Per sampled group: relative L2 error over the batch and argmax agreement."""
    errs, agree, worst_row = [], [], 0.0
    for g in range(ref.shape[0]):
        errs.append(rel(logits[g], ref[g]))
        agree.append(float((logits[g].double().argmax(-1) == ref[g].argmax(-1)).double().mean()))
        rows = ((logits[g].double() - ref[g]).norm(dim=-1) / ref[g].norm(dim=-1)).max()
        worst_row = max(worst_row, float(rows))
    return {"rel_per_group": errs, "rel_sum": sum(errs), "argmax_agreement": statistics.mean(agree),
            "worst_row_rel": worst_row}


# ---------------------------------------------------------------------------------------------------------

class Kernel:
    """The resident kernel at one configuration, with its own scratch."""

    def __init__(self, program, w, codebook, bs, groups, scope, grid, device, top_k, top_p, temperature,
                 sync_mode=0):
        self.program, self.codebook, self.bs = program, codebook, bs
        self.groups, self.scope, self.grid, self.sync_mode = groups, scope, grid, sync_mode
        self.table = sp.lower(program, w, device)
        self.rope = sp.rope_table(program.shape, device)
        self.scratch = ds.DepthStepScratch.allocate(program.shape, bs, groups, program.ss_slots, device)
        self.top_k, self.top_p, self.temperature = top_k, top_p, temperature

    def run(self, last_hidden, embed0, emb_sum, codes, *, seed=0, forced=None, stop_after=-1, trace=None):
        ds.launch(self.table, self.rope, last_hidden, embed0, self.codebook, emb_sum, codes, self.scratch,
                  groups=self.groups, scope=self.scope, grid=self.grid, shape=self.program.shape, top_k=self.top_k,
                  top_p=self.top_p, temperature=self.temperature, seed=seed, forced=forced, stop_after=stop_after,
                  sync_mode=self.sync_mode, trace=trace)

    def ctl(self) -> list[int]:
        return [int(v) for v in self.scratch.ctl.tolist()]

    def zero_state(self):
        for t in (self.scratch.h, self.scratch.qkv, self.scratch.attn, self.scratch.act, self.scratch.kv,
                  self.scratch.logits):
            t.zero_()


def phase_checks(kernel: Kernel, program, w, codebook, last_hidden, embed0, forced) -> list[dict[str, Any]]:
    """Kernel state after a phase against the mirror stopped at the same phase."""
    kinds = [op.kind for op in program.ops]
    stops = [1]
    for want in (sp.OpKind.QKV, sp.OpKind.ATTN, sp.OpKind.OPROJ, sp.OpKind.GATEUP, sp.OpKind.DOWN):
        stops.append(kinds.index(want) + 1)
    first_lm = kinds.index(sp.OpKind.LMHEAD)
    stops += [kinds.index(sp.OpKind.INPUT) + 1, first_lm + 1, first_lm + 2, first_lm + 7, len(kinds)]
    out = []
    for stop in sorted(set(stops)):
        kernel.zero_state()
        emb_sum = embed0.float().clone()
        codes = forced.clone()
        kernel.run(last_hidden, embed0, emb_sum, codes, forced=forced, stop_after=stop)
        torch.cuda.synchronize()
        ref = sp.interpret(program, w, codebook, last_hidden, embed0, forced, kernel_rounding=True, stop_after=stop)
        sc = kernel.scratch
        last = program.ops[stop - 1]
        entry = {"stop_after": stop, "last_op": f"{last.kind.name} p{last.position} l{last.layer}", "ctl": kernel.ctl()}
        for name, got in (("h", sc.h), ("qkv", sc.qkv), ("attn", sc.attn), ("act", sc.act), ("kv", sc.kv),
                          ("logits", sc.logits), ("emb_sum", emb_sum)):
            r = ref[name]
            entry[name] = None if float(r.abs().max()) == 0.0 else rel(got, r)
        out.append(entry)
    return out


class ModelLoop:
    """The code predictor's own depth loop as the served Talker step runs it
    (``Qwen3OmniCodePredictor.forward_depth_unrolled`` per position, the LM head,
    the SamplerResource kernels, the embedding and its sum), on the same weights."""

    def __init__(self, w, codebook, bs, device, state: perop.PerOpState):
        from mstar.model.qwen3_omni.components.talker import Qwen3OmniCodePredictor
        from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig

        cfg = Qwen3OmniModelConfig()
        cp_cfg = cfg.code_predictor
        cp_cfg.rms_norm_eps, cp_cfg.rope_theta = OMNI.eps, OMNI.rope_theta
        with torch.device("meta"):
            cp = Qwen3OmniCodePredictor(cfg)
        cp = cp.to(torch.bfloat16).to_empty(device=device)
        with torch.no_grad():
            for name, t in cp.named_parameters():
                if name.startswith("model.codec_embedding."):
                    t.copy_(codebook[int(name.split(".")[2])])
                elif name.startswith("lm_head."):
                    t.copy_(w["lm_head_weight"][int(name.split(".")[1])])
                else:
                    t.copy_(w[name])
        cp.consolidate_stacked_weights()
        self.cp, self.state, self.bs = cp.eval(), state, bs
        self.kv = torch.zeros(OMNI.layers, bs, 2, OMNI.positions, OMNI.kv_heads, OMNI.head_dim, dtype=torch.bfloat16,
                              device=device)
        self.pos = torch.zeros(bs, 1, dtype=torch.long, device=device)

    @torch.no_grad()
    def frame(self, last_hidden, embed0, emb_sum, codes):
        cp = self.cp
        self.pos.zero_()
        cp.forward_depth_unrolled(last_hidden.unsqueeze(1), self.pos, self.kv, cache_pos=0)
        self.pos += 1
        embed = embed0
        for g in range(1, OMNI.positions):
            hidden = cp.forward_depth_unrolled(embed.unsqueeze(1), self.pos, self.kv, cache_pos=g).squeeze(1)
            self.pos += 1
            logits = torch.matmul(hidden, cp.lm_head_weight[g - 1].t())
            tokens = perop.sampler(self.state, logits)
            codes[:, g] = tokens
            embed = cp.model.codec_embedding[g - 1](tokens)
            emb_sum.add_(embed)


def time_graph(fn, replays: int = 30, warmup: int = 3, before=None) -> dict[str, Any]:
    """``fn`` captured once and replayed one replay per sync; ``before`` (not
    timed) runs ahead of each timed replay, e.g. to evict the caches."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(warmup):
        graph.replay()
        torch.cuda.synchronize()
    times = []
    for _ in range(replays):
        if before is not None:
            before()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return {"ms": statistics.median(times), "ms_min": min(times), "ms_max": max(times), "replays": replays,
            "graph": graph}


def phase_trace(kernel: Kernel, last_hidden, embed0, emb_sum, codes, runs: int = 5) -> dict[str, Any]:
    """Where a frame's time goes, from group 0's first workgroup: per op kind, the time from the previous
    barrier's exit to this one's arrival (work) and from arrival to exit (waiting for the group), in us per
    frame summed over the kind's phases; medians over ``runs`` launches after one warm-up."""
    program = kernel.program
    n = len(program.ops)
    khz = ds.wall_clock_khz()
    trace = torch.zeros(2 * n, dtype=torch.int64, device=last_hidden.device)
    per_run = []
    for i in range(runs + 1):
        kernel.run(last_hidden, embed0, emb_sum, codes, seed=11 + i, trace=trace)
        torch.cuda.synchronize()
        if i == 0:
            continue
        t = trace.view(n, 2).double().cpu() / (khz / 1e3)  # us
        work, wait = {}, {}
        for ph, op in enumerate(program.ops):
            start = t[ph - 1, 1] if ph else t[0, 0]
            work[op.kind.name] = work.get(op.kind.name, 0.0) + float(t[ph, 0] - start)
            wait[op.kind.name] = wait.get(op.kind.name, 0.0) + float(t[ph, 1] - t[ph, 0])
        per_run.append((work, wait, float(t[n - 1, 1] - t[0, 0])))
    kinds = list(per_run[0][0])
    return {"wall_clock_khz": khz, "frame_us": statistics.median(r[2] for r in per_run),
            "work_us": {k: statistics.median(r[0][k] for r in per_run) for k in kinds},
            "wait_us": {k: statistics.median(r[1][k] for r in per_run) for k in kinds},
            "phases": dict(collections.Counter(op.kind.name for op in program.ops)), "ctl": kernel.ctl()}


def topk_topp_probs(logits: torch.Tensor, top_k: int, top_p: float, temperature: float) -> torch.Tensor:
    probs = torch.softmax(logits.double() / temperature, -1)
    vals, idx = probs.topk(top_k)
    vals = vals / vals.sum()
    excl = vals.cumsum(0) - vals
    vals = torch.where(excl < top_p, vals, torch.zeros_like(vals))
    out = torch.zeros_like(probs)
    out[idx] = vals / vals.sum()
    return out


def sampler_check(kernel: Kernel, program, w, codebook, talker_codec, device, draws_per_row: int,
                  top_k, top_p, temperature) -> dict[str, Any]:
    """Group-1 codes over many seeds for identical rows: the kernel's and the
    SamplerResource's draws against the filtered distribution of the kernel's
    own group-1 logits."""
    bs = kernel.bs
    last_hidden, embed0, forced = make_inputs(bs, talker_codec, device, same_rows=True)
    counts = torch.zeros(program.shape.vocab, dtype=torch.float64, device=device)
    emb_sum = torch.zeros(bs, program.shape.hidden, dtype=torch.float32, device=device)
    codes = torch.zeros(bs, program.shape.positions, dtype=torch.int64, device=device)
    for seed in range(draws_per_row):
        kernel.run(last_hidden, embed0, emb_sum, codes, seed=seed + 1)
        counts += torch.bincount(codes[:, 1], minlength=program.shape.vocab).double()
    logits = kernel.scratch.logits[0, 0].clone()
    expected = topk_topp_probs(logits, top_k, top_p, temperature)
    n = counts.sum()
    tv_kernel = 0.5 * float((counts / n - expected).abs().sum())
    state = perop.PerOpState.allocate(program.shape, bs, device, top_k, top_p, temperature, seed=0)
    rep = logits[None].expand(bs, -1).contiguous()
    ref_counts = torch.zeros_like(counts)
    for seed in range(draws_per_row):
        state.seed.fill_(seed + 1)
        ref_counts += torch.bincount(perop.sampler(state, rep).long(), minlength=program.shape.vocab).double()
    tv_ref = 0.5 * float((ref_counts / ref_counts.sum() - expected).abs().sum())
    support = int((expected > 0).sum())
    outside = float(counts[expected == 0].sum() / n)
    return {"draws": int(n), "support": support, "tv_kernel": tv_kernel, "tv_sampler_resource": tv_ref,
            "kernel_mass_outside_support": outside,
            "expected_tv_noise": math.sqrt(support / (2 * math.pi * float(n)))}


def _stage(args, result: dict[str, Any], name: str) -> None:
    """Keep what is measured so far, in case a later stage takes the process down."""
    (args.out / f"partial_{args.weights}.json").write_text(json.dumps(result, indent=1, default=str))
    print(f"[depth_loop] {time.strftime('%H:%M:%S')} {name} done", flush=True)


def run(args) -> dict[str, Any]:
    device = torch.device("cuda")
    import mstar.engine.resources.rms_norm.rms_norm  # noqa: F401  registers mstar::rms_norm / fused_add_rms_norm
    from mstar.platforms import current_platform

    facts = current_platform.device_facts()
    t0 = time.time()
    if args.weights == "checkpoint":
        w, codebook, talker_codec = checkpoint_weights(device)
    else:
        w, codebook, talker_codec = random_weights(device)
    program = sp.depth_decoder_program(OMNI, sampler_key="code_sampler")
    problems = sp.check(program)
    result: dict[str, Any] = {"device": torch.cuda.get_device_name(0), "arch": facts.arch, "cus": facts.compute_units,
                              "weights": args.weights, "bs": args.bs, "phases": len(program.ops),
                              "program_problems": problems, "weight_bytes_per_frame": program.gemm_bytes(),
                              "load_s": time.time() - t0}
    bs = args.bs
    last_hidden, embed0, forced = make_inputs(bs, talker_codec, device)
    cus = facts.compute_units or 304
    top_k, top_p, temperature = 50, 0.8, 1.0

    t1 = time.time()
    ds.extension().load()
    result["kernel_build_s"] = time.time() - t1

    def guarded(name, fn):
        """A stage's result, or its error, so one broken stage leaves the others standing."""
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001  recorded
            import traceback

            traceback.print_exc()
            return {"error": f"{type(exc).__name__}: {str(exc)[:400]}"}
        finally:
            _stage(args, result, name)

    # 1. phase-by-phase agreement with the mirror (one configuration per sync scope)
    def stage_checks():
        checks = {}
        for groups, scope in ((8, ds.SCOPE_XCD), (4, ds.SCOPE_AGENT)):
            kc = Kernel(program, w, codebook, bs, groups, scope, ds.default_grid(cus, groups), device, top_k, top_p,
                        temperature)
            checks[f"g{groups}_{'xcd' if scope else 'agent'}"] = phase_checks(kc, program, w, codebook, last_hidden,
                                                                              embed0, forced)
        return checks

    result["phase_checks"] = guarded("phase checks", stage_checks)

    # 2. parity, teacher-forced, against float64
    ref = sp.interpret(program, w, codebook, last_hidden, embed0, forced, kernel_rounding=False)
    state = perop.PerOpState.allocate(OMNI, bs, device, top_k, top_p, temperature, seed=0)
    rope = sp.rope_table(OMNI, device)
    clean: list[Kernel] = []

    def kernel_parity(groups, scope):
        kp = Kernel(program, w, codebook, bs, groups, scope, ds.default_grid(cus, groups), device, top_k, top_p,
                    temperature)

        def run_it():
            emb_sum = embed0.float().clone()
            codes = forced.clone()
            kp.zero_state()
            kp.run(last_hidden, embed0, emb_sum, codes, forced=forced)
            torch.cuda.synchronize()
            out = group_errors(kp.scratch.logits, ref["logits"])
            out["emb_sum_rel"] = rel(emb_sum, ref["emb_sum"])
            out["ctl"] = kp.ctl()
            if out["ctl"][0] == 0:
                clean.append(kp)
            return out
        return run_it

    def stage_per_op_parity():
        emb_sum_p = embed0.float().clone()
        codes_p = forced.clone()
        perop.depth_frame(w, OMNI, rope, state, last_hidden, embed0, codebook, emb_sum_p, codes_p, forced=forced)
        torch.cuda.synchronize()
        out = group_errors(state.logits, ref["logits"])
        out["emb_sum_rel"] = rel(emb_sum_p, ref["emb_sum"])
        out["kernel_vs_per_op_rel_sum"] = sum(rel(k.scratch.logits[g], state.logits[g]) for g in range(15))
        return out

    parity = {"kernel": guarded("kernel parity", kernel_parity(8, ds.SCOPE_XCD)),
              "kernel_g4_agent": guarded("kernel parity g4", kernel_parity(4, ds.SCOPE_AGENT))}
    k = clean[0] if clean else Kernel(program, w, codebook, bs, 4, ds.SCOPE_AGENT, ds.default_grid(cus, 4), device,
                                      top_k, top_p, temperature)
    parity["per_op"] = guarded("per-op parity", stage_per_op_parity)
    result["parity"] = parity

    # 3. sampler distribution
    result["sampler"] = guarded("sampler", lambda: sampler_check(k, program, w, codebook, talker_codec, device,
                                                                 args.draws, top_k, top_p, temperature))

    # 4. captured time per frame
    timings = {}
    emb_sum_t = embed0.float().clone()
    codes_t = forced.clone()

    def per_op_frame(forced_codes=None):
        emb_sum_t.copy_(embed0.float())
        perop.depth_frame(w, OMNI, rope, state, last_hidden, embed0, codebook, emb_sum_t, codes_t,
                          forced=forced_codes)

    for name, fn in (("per_op_sampled", per_op_frame), ("per_op_forced", lambda: per_op_frame(forced))):
        try:
            t = time_graph(fn, args.replays)
            t.pop("graph")
            timings[name] = t
        except Exception as exc:  # noqa: BLE001  recorded
            timings[name] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        result["timing"] = timings
        _stage(args, result, f"timing {name}")
    # the sampler's share of the per-op frame: its kernels for the 15 groups on fixed logits
    fixed_logits = torch.randn(OMNI.positions - 1, bs, OMNI.vocab, device=device).to(torch.bfloat16)

    def samplers_only():
        for g in range(OMNI.positions - 1):
            perop.sampler(state, fixed_logits[g])

    try:
        t = time_graph(samplers_only, args.replays)
        t.pop("graph")
        timings["per_op_samplers_only"] = t
    except Exception as exc:  # noqa: BLE001  recorded
        timings["per_op_samplers_only"] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    try:
        loop = ModelLoop(w, codebook, bs, device, state)
        t = time_graph(lambda: loop.frame(last_hidden, embed0, emb_sum_t, codes_t), args.replays)
        t.pop("graph")
        timings["model_loop_sampled"] = t
    except Exception as exc:  # noqa: BLE001  recorded
        timings["model_loop_sampled"] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    result["timing"] = timings
    _stage(args, result, "timing model_loop_sampled")
    configs = [(8, ds.SCOPE_XCD, 1), (8, ds.SCOPE_XCD, 2), (8, ds.SCOPE_AGENT, 1), (4, ds.SCOPE_AGENT, 1),
               (4, ds.SCOPE_AGENT, 2), (2, ds.SCOPE_AGENT, 1), (2, ds.SCOPE_AGENT, 2)]
    for groups, scope, per_cu in configs:
        grid = ds.default_grid(cus, groups, per_cu)
        name = f"kernel_g{groups}_{'xcd' if scope else 'agent'}_grid{grid}"
        kk = Kernel(program, w, codebook, bs, groups, scope, grid, device, top_k, top_p, temperature)
        try:
            timings[f"{name}_trace"] = phase_trace(kk, last_hidden, embed0, emb_sum_t, codes_t)
        except Exception as exc:  # noqa: BLE001  recorded
            timings[f"{name}_trace"] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        for mode, fc in (("sampled", None), ("forced", forced)):
            try:
                t = time_graph(lambda kk=kk, fc=fc: kk.run(last_hidden, embed0, emb_sum_t, codes_t, seed=7,
                                                             forced=fc), args.replays)
                t.pop("graph")
                t["ctl"] = kk.ctl()
                timings[f"{name}_{mode}"] = t
            except Exception as exc:  # noqa: BLE001  recorded
                timings[f"{name}_{mode}"] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            result["timing"] = timings
            _stage(args, result, f"timing {name}_{mode}")
    # the served step streams the talker's experts between frames, so the depth loop's weights start outside the
    # memory-side cache: the same arms after reading 1 GiB before each frame
    flush_buf = torch.empty(1 << 28, dtype=torch.float32, device=device)
    cold = {"per_op_sampled_cold": per_op_frame}
    for groups, scope in ((8, ds.SCOPE_XCD), (4, ds.SCOPE_AGENT)):
        kc = Kernel(program, w, codebook, bs, groups, scope, ds.default_grid(cus, groups), device, top_k, top_p,
                    temperature)
        name = f"kernel_g{groups}_{'xcd' if scope else 'agent'}_grid{kc.grid}_sampled_cold"
        cold[name] = functools.partial(kc.run, last_hidden, embed0, emb_sum_t, codes_t, seed=7)
    for name, fn in cold.items():
        try:
            t = time_graph(fn, args.replays, before=flush_buf.sum)
            t.pop("graph")
            timings[name] = t
        except Exception as exc:  # noqa: BLE001  recorded
            timings[name] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        result["timing"] = timings
        _stage(args, result, f"timing {name}")
    result["timing"] = timings
    base = timings.get("per_op_sampled", {}).get("ms")
    if base is None and "ms" in timings.get("per_op_forced", {}) and "ms" in timings.get("per_op_samplers_only", {}):
        # the sampled frame did not capture: the forced frame plus the samplers stands in for it
        base = timings["per_op_forced"]["ms"] + timings["per_op_samplers_only"]["ms"]
        result["per_op_base_note"] = "per_op_forced + per_op_samplers_only"
    if base:
        result["ratio_to_per_op"] = {k2: v["ms"] / base for k2, v in timings.items() if "ms" in v}
    return result


def run_variants(args) -> dict[str, Any]:
    """Barrier variants of the resident kernel: each one's teacher-forced logits
    against the default configuration's, then its captured time per frame."""
    device = torch.device("cuda")
    from mstar.platforms import current_platform

    cus = current_platform.device_facts().compute_units or 304
    w, codebook, talker_codec = (checkpoint_weights if args.weights == "checkpoint" else random_weights)(device)
    program = sp.depth_decoder_program(OMNI, sampler_key="code_sampler")
    bs = args.bs
    last_hidden, embed0, forced = make_inputs(bs, talker_codec, device)
    top_k, top_p, temperature = 50, 0.8, 1.0

    def forced_logits(k: Kernel) -> torch.Tensor:
        emb_sum = embed0.float().clone()
        codes = forced.clone()
        k.zero_state()
        k.run(last_hidden, embed0, emb_sum, codes, forced=forced)
        torch.cuda.synchronize()
        return k.scratch.logits.clone()

    ref = forced_logits(Kernel(program, w, codebook, bs, 8, ds.SCOPE_XCD, ds.default_grid(cus, 8), device, top_k,
                               top_p, temperature))
    out: dict[str, Any] = {"weights": args.weights, "bs": bs, "variants": {}}
    emb_sum_t = embed0.float().clone()
    codes_t = forced.clone()
    configs = []
    for per_cu in (1, 2):
        for mode in (0, ds.SYNC_LOAD_POLL, ds.SYNC_GATHER, ds.SYNC_GATHER | ds.SYNC_LOAD_POLL):
            configs.append((8, ds.SCOPE_XCD, per_cu, mode))
        configs.append((8, ds.SCOPE_XCD, per_cu, ds.SPLIT_K))
        for groups in (4, 2):
            for mode in (0, ds.SYNC_GATHER, ds.SYNC_HIER, ds.SYNC_HIER | ds.SYNC_LOAD_POLL, ds.SYNC_HIER | ds.SPLIT_K):
                configs.append((groups, ds.SCOPE_AGENT, per_cu, mode))
    for groups, scope, per_cu, mode in configs:
        grid = ds.default_grid(cus, groups, per_cu)
        algo = "hier" if mode & ds.SYNC_HIER else "gather" if mode & ds.SYNC_GATHER else "counter"
        name = (f"g{groups}_{'xcd' if scope else 'agent'}_grid{grid}_{algo}"
                f"{'_loadpoll' if mode & ds.SYNC_LOAD_POLL else ''}{'_splitk' if mode & ds.SPLIT_K else ''}")
        entry: dict[str, Any] = {}
        try:
            k = Kernel(program, w, codebook, bs, groups, scope, grid, device, top_k, top_p, temperature, mode)
            entry["rel_to_default"] = rel(forced_logits(k), ref)
            entry["ctl_forced"] = k.ctl()
            t = time_graph(functools.partial(k.run, last_hidden, embed0, emb_sum_t, codes_t, seed=7), args.replays)
            t.pop("graph")
            entry.update(t)
            entry["ctl"] = k.ctl()
        except Exception as exc:  # noqa: BLE001  recorded
            entry["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        out["variants"][name] = entry
        (args.out / f"partial_variants_{args.weights}.json").write_text(json.dumps(out, indent=1, default=str))
        print(f"[depth_loop] {time.strftime('%H:%M:%S')} variant {name}: {entry.get('ms', entry.get('error'))}",
              flush=True)
    return out


def table(r: dict[str, Any]) -> str:
    lines = [f"# depth loop, bs {r['bs']}: {r['device']} ({r['arch']}), {r['weights']} weights, {r['phases']} phases",
             "", "## time per frame, captured (one replay per sync, median)", "",
             "| arm | ms | min | max | x per-op sampled | ctl |", "|---|---|---|---|---|---|"]
    ratio = r.get("ratio_to_per_op", {})
    for name, t in r["timing"].items():
        if "ms" in t:
            lines.append(f"| {name} | {t['ms']:.3f} | {t['ms_min']:.3f} | {t['ms_max']:.3f} | "
                         f"{ratio.get(name, float('nan')):.3f} | {t.get('ctl', '')} |")
        else:
            lines.append(f"| {name} | {t.get('error')} | | | | |")
    p = r["parity"]
    lines += ["", "## parity, teacher-forced, logits vs float64 (sum over the 15 groups of the relative L2 error)", "",
              "| path | rel sum | worst row rel | argmax agreement |", "|---|---|---|---|"]
    for name in ("kernel", "kernel_g4_agent", "per_op"):
        e = p.get(name, {"error": "not run"})
        if "error" in e:
            lines.append(f"| {name} | {e['error']} | | |")
            continue
        lines.append(f"| {name} | {e['rel_sum']:.4f} | {e['worst_row_rel']:.4f} | {e['argmax_agreement']:.4f} |")
    if "kernel_vs_per_op_rel_sum" in p["per_op"]:
        lines.append(f"\nkernel vs per-op, rel sum {p['per_op']['kernel_vs_per_op_rel_sum']:.4f}")
    s = r["sampler"]
    if "error" in s:
        lines += ["", f"## sampler: {s['error']}"]
    else:
        lines += ["", f"## sampler: {s['draws']} draws over a support of {s['support']}: TV kernel "
                  f"{s['tv_kernel']:.4f}, SamplerResource {s['tv_sampler_resource']:.4f}, noise "
                  f"~{s['expected_tv_noise']:.4f}, kernel mass outside the support "
                  f"{s['kernel_mass_outside_support']:.2e}"]
    lines += ["", "## phase checks (rel error vs mirror)"]
    checks = r["phase_checks"]
    if "error" in checks:
        lines.append(checks["error"])
        checks = {}
    for cfg, rows in checks.items():
        lines.append(f"\n{cfg}")
        for e in rows:
            vals = " ".join(f"{k}={e[k]:.1e}" if e[k] is not None else f"{k}=-" for k in
                            ("h", "qkv", "attn", "act", "kv", "logits", "emb_sum"))
            lines.append(f"- after {e['stop_after']} ({e['last_op']}): {vals} ctl={e['ctl']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", choices=("checkpoint", "random"), default="checkpoint")
    ap.add_argument("--bs", type=int, default=27)
    ap.add_argument("--replays", type=int, default=30)
    ap.add_argument("--draws", type=int, default=150, help="sampler draws per row")
    ap.add_argument("--variants", action="store_true", help="only the barrier variants")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.variants:
        res = run_variants(args)
        (args.out / f"depth_loop_variants_{args.weights}.json").write_text(json.dumps(res, indent=1, default=str))
        for name, e in res["variants"].items():
            print(name, {k: e[k] for k in ("ms", "rel_to_default", "ctl", "error") if k in e})
        return 0
    result = run(args)
    (args.out / f"depth_loop_{args.weights}.json").write_text(json.dumps(result, indent=1, default=str))
    text = table(result)
    (args.out / f"depth_loop_{args.weights}.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
