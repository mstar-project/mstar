"""Hierarchical optimality report for a Cosmos3 Walk, in M*'s abstraction.

Reports recoverable optimization headroom at every level of M*'s model
abstraction:

    Walk  ->  component / graph node  ->  op-class  ->  kernel

For the `image_gen` walk the components are the DiT denoise step (run
`num_inference_steps` times) and the Wan-VAE decode (run once). For each level
it shows measured device time, a hardware-necessary roofline, the recoverable
headroom, and -- the point of the report -- the NAMED bucket that headroom falls
into (fusion / redundant / occupancy / at-roofline), so you know *what* to fix.

The Walk cost is multiplicity-weighted (dit x N steps, vae x 1), so the ranking
reflects real end-to-end priority, not per-invocation time.

Method (see also cosmos3_compiled_kernel_analysis.py for the kernel<->source
detail): the DiT is profiled as actually served (torch.compile + CUDA-graph
replay) with an analytic per-op-class roofline; the VAE is profiled eager with
FLOPs from torch's FlopCounterMode. Peaks are measured live. Absolute ms carry
the caveats documented in the sibling script; treat gaps as ROI signals.

Run:
  cd <mstar repo>
  COSMOS3_NANO_DIR=/path CUDA_VISIBLE_DEVICES=<gpu> python perf_testing/cosmos3_optimality_report.py
Report is tee'd to perf_testing/results/ by default (COSMOS3_FEEDBACK_OUT to override).
"""
import os, sys, re, glob, shutil, tempfile, datetime, collections, torch

# ---- output recording (default on) ----
_OUT = os.environ.get("COSMOS3_FEEDBACK_OUT")
if _OUT is None:
    _r = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(_r, exist_ok=True)
    _OUT = os.path.join(_r, f"cosmos3_optimality_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt")
if _OUT:
    class _Tee:
        def __init__(self, *s): self.streams = s
        def write(self, x):
            for st in self.streams: st.write(x)
        def flush(self):
            for st in self.streams: st.flush()
    _outfh = open(_OUT, "w"); sys.stdout = _Tee(sys.__stdout__, _outfh)
    print(f"# recording report to {_OUT}")

SNAP = os.environ.get("COSMOS3_NANO_DIR",
    "/raid/hf/hub/models--nvidia--Cosmos3-Nano/snapshots/411f42a8fdfb8c5b2583cb8786e0938f49796eaa")
if not os.path.isdir(SNAP): raise SystemExit(f"checkpoint not found: {SNAP} (set COSMOS3_NANO_DIR)")
os.environ["COSMOS3_NANO_DIR"] = SNAP
TRACE_DIR = tempfile.mkdtemp(prefix="cosmos3_opt_")
import torch._inductor.config as ind_cfg
ind_cfg.trace.enabled = True; ind_cfg.trace.debug_dir = TRACE_DIR
ind_cfg.force_disable_caches = True; os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"

from torch.profiler import profile, ProfilerActivity
from torch.utils.flop_counter import FlopCounterMode
import mstar.model.cosmos3.tests.test_engine_cache as T

REPLAYS = 20

# =================================================================================
# hardware peaks (measured live)
# =================================================================================
def _bench(fn, iters=40, warm=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

# =================================================================================
# profiler helpers
# =================================================================================
def _dev_us(k): return getattr(k, "self_device_time_total", getattr(k, "self_cuda_time_total", 0))

def profile_calls(fn, n):
    """Run fn n times under the profiler; return {kernel_name: (ms_total_over_n, calls)}."""
    for _ in range(3): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(n): fn()
        torch.cuda.synchronize()
    out = collections.defaultdict(lambda: [0.0, 0])
    for k in p.key_averages():
        us = _dev_us(k)
        if us > 0: out[k.key][0] += us; out[k.key][1] += k.count
    return out

def family(key):
    k = key.lower()
    if key.startswith("triton_"): return "triton"
    if "conv" in k or "nchwtonhwc" in k or "implicit_gemm" in k: return "conv"   # cuDNN/cuTLASS conv
    if any(x in k for x in ("nvjet", "cublas", "cutlass", "splitkreduce")): return "gemm"
    if any(x in k for x in ("flashinfer", "fmha", "flash", "scaled_dot")): return "attention"
    if any(x in k for x in ("memcpy", "memset", "cudagraphlaunch")): return "overhead"
    if "radix_sort" in k or "sort" in k or "cub::" in k: return "overhead"
    if "elementwise" in k or "reduce_kernel" in k or k.startswith("void at::native"): return "pointwise"
    return "pointwise"

# map raw family -> report op-class
OPCLASS = {"gemm": "GEMM", "conv": "convolution", "attention": "attention",
           "triton": "norm/adaLN/pointwise", "pointwise": "norm/adaLN/pointwise", "overhead": "overhead"}

# =================================================================================
# Inductor kernel -> source (for the DiT top kernels)
# =================================================================================
FILE_RE = re.compile(r'#\s*File:\s*(.+?):(\d+)\s+in\s+(\w+),\s*code:\s*(.*)')
NODE_RE = re.compile(r'^\s*(\w+)\s*:\s*"')
SRC_RE  = re.compile(r'Source Nodes:\s*\[(.*?)\],\s*Original ATen:\s*\[(.*?)\]')
DEF_RE  = re.compile(r'^\s*(triton_\w+)\s*=\s*async_compile')
RUN_RE  = re.compile(r'(triton_\w+)\.run\(')
def _node_src(fx):
    m, cur = {}, None
    if not os.path.exists(fx): return m
    for line in open(fx):
        f = FILE_RE.search(line)
        if f: cur = (os.path.basename(f.group(1)), f.group(2), f.group(4).strip()); continue
        n = NODE_RE.match(line)
        if n and cur: m[n.group(1)] = cur
    return m
def dit_kernel_sources():
    KSRC = collections.defaultdict(collections.OrderedDict)
    for oc in glob.glob(f"{TRACE_DIR}/**/output_code.py", recursive=True):
        n2s = _node_src(os.path.join(os.path.dirname(oc), "fx_graph_readable.py"))
        last = None
        for line in open(oc):
            s = SRC_RE.search(line)
            if s: last = [x.strip() for x in s.group(1).split(",") if x.strip()]; continue
            for rx in (DEF_RE, RUN_RE):
                mm = rx.search(line)
                if mm and last:
                    for nd in last:
                        if nd in n2s:
                            f, ln, code = n2s[nd]; KSRC[mm.group(1)][f"{f}:{ln}"] = code
    return KSRC

# =================================================================================
# build model + scenario
# =================================================================================
ctx = T._scenario(1)
if ctx is None: raise SystemExit("scenario unavailable (needs COSMOS3_NANO_DIR + CUDA).")
model, device, dtype = ctx["model"], ctx["device"], ctx["dtype"]
mpipe = ctx["mpipe"]
c = model.config
STEPS_WALK = c.num_inference_steps      # denoise steps per image_gen walk (multiplicity of the DiT)

print("measuring hardware peaks...", flush=True)
_a = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
_b = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
PEAK_FLOPS = (2 * 8192 ** 3) / (_bench(lambda: torch.mm(_a, _b)) * 1e-3)
_src = torch.randn(1 << 29, device=device, dtype=torch.bfloat16); _dst = torch.empty_like(_src)
PEAK_BW = (2 * _src.numel() * 2) / (_bench(lambda: _dst.copy_(_src)) * 1e-3)
del _a, _b, _src, _dst
def roof(fl, by): return max(fl / PEAK_FLOPS, by / PEAK_BW) * 1e3      # ms

# =================================================================================
# COMPONENT 1 — DiT (served: torch.compile + CUDA graph), analytic op-class roofline
# =================================================================================
def collect_dit():
    model.config.compile_denoise = True
    model._submodule_cache.pop("dit", None)
    dit = model.get_submodule("dit", device=device)
    from mstar.conductor.request_info import CurrentForwardPassInfo
    from mstar.distributed.communication import CommGroup
    from mstar.engine.cuda_graph_runner import CudaGraphRunner
    from mstar.model.submodule_base import ModelInputsFromEngine
    from mstar.utils.sampling import MultiSampler, MultiSamplingConfig
    dit.gen_capture_resolutions = ((T.H, T.W),)
    rid = "opt0"
    shared = T._flashinfer_shared(model, [rid], device, dtype)
    md = {"height": T.H, "width": T.W, "num_frames": 1, "fps": 24.0,
          "guidance_scale": T.GS, "num_inference_steps": T.STEPS}
    fwd = CurrentForwardPassInfo(request_id=rid, graph_walk="prefill", requires_cfg=False,
          fwd_index=0, random_seed=T.SEED, max_tokens=0, sampling_config={}, step_metadata=md)
    cm = T._mk_cm(shared, [rid])
    ei = ModelInputsFromEngine(request_ids=[rid], per_request_info={rid: fwd}, cache_manager=cm)
    ti = [torch.tensor(ctx["cond"], dtype=torch.long, device=device),
          torch.tensor(ctx["uncond"], dtype=torch.long, device=device)]
    ni = dit.prepare_inputs("prefill", fwd, {"text_inputs": ti})
    dit.forward("prefill", ei, **dit.preprocess("prefill", ei, [ni]))
    runner = CudaGraphRunner(submodule_name="dit", submodule=dit, kv_cache_config=shared["cfg"],
        alloc_manager=shared["alloc"], sampler=MultiSampler.new(aux_labels=[], device=torch.device(device),
        tp_group=CommGroup.trivial()), buffer_manager=shared["buf"], device=torch.device(device),
        autocast_dtype=dtype, default_sampling_config=MultiSamplingConfig(), tp_group=CommGroup.trivial())
    print("  DiT: warmup_and_capture (compile + cuda-graph)...", flush=True)
    runner.warmup_and_capture(); runner.register_request(rid)
    fwd.graph_walk = "image_gen"
    ni_f = dit.prepare_inputs("image_gen", fwd,
        {"latents": [ctx["init"].clone()], "time_index": [torch.zeros(1, dtype=torch.long, device=device)]})
    KT = profile_calls(lambda: runner.run(graph_walk="image_gen", requires_cfg=False, request_ids=[rid],
                       inputs=[ni_f], per_request_info={rid: fwd}, submodule=dit), REPLAYS)
    KSRC = dit_kernel_sources()

    # analytic necessary work per op-class (GEN pathway, both CFG branches, all layers)
    d, L = c.hidden_size, c.num_hidden_layers
    Hh, Hkv, Dh, I = c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.intermediate_size
    qd, kd = Hh * Dh, Hkv * Dh
    Tg = max(1, (T.H // 16 // 2) * (T.W // 16 // 2)); Tctx = len(ctx["cond"]) + Tg; CFG = 2; bpe = 2
    gf = gb = 0.0
    for M, N, K in [(Tg, qd, d), (Tg, kd, d), (Tg, kd, d), (Tg, d, qd), (Tg, I, d), (Tg, I, d), (Tg, d, I)]:
        gf += 2 * M * N * K; gb += (M * K + K * N + M * N) * bpe
    mlt = CFG * L
    roof_cls = {
        "GEMM": roof(mlt * gf, mlt * gb),
        "attention": roof(mlt * 2 * 2 * Hh * Tg * Tctx * Dh,
                          mlt * (2 * Hh * Tg * Dh + 2 * Hkv * Tctx * Dh) * bpe),
        "norm/adaLN/pointwise": roof(0, mlt * (6 * Tg * d + Tg * qd + Tg * kd) * 2 * bpe),
        "overhead": 0.0,
    }
    meta = f"served compile+graph, Tg={Tg} Tctx={Tctx} CFG={CFG} L={L}"
    return dict(name="dit", walk_mult=STEPS_WALK, KT=KT, KSRC=KSRC, roof_cls=roof_cls, meta=meta)

# =================================================================================
# COMPONENT 2 — VAE decoder (eager real module), FLOPs from FlopCounterMode
# =================================================================================
def collect_vae():
    lat = ctx["lat_fused"]
    dec = lambda: mpipe._decode(lat)
    dec(); torch.cuda.synchronize()
    KT = profile_calls(dec, 4)                       # decode is cheap; fewer reps
    for k in KT: KT[k][0] /= (4 / REPLAYS)           # normalize to REPLAYS-window like the DiT (so /REPLAYS = per-call)
    # FLOPs per op-class via FlopCounterMode (compute-bound classes: conv, gemm, attention)
    fc = FlopCounterMode(display=False)
    with fc: dec()
    flops = collections.Counter()
    for mod, opd in fc.get_flop_counts().items():
        for aten, fl in opd.items():
            a = str(aten).lower()
            cls = ("convolution" if "conv" in a else
                   "GEMM" if any(x in a for x in ("mm", "addmm", "bmm", "linear")) else
                   "attention" if "scaled_dot" in a or "sdpa" in a else None)
            if cls: flops[cls] += fl
    # only the "Global" module double-counts; use per-op sums but FlopCounter nests -> take the top-level only
    fcg = fc.get_flop_counts().get("Global", {})
    flops = collections.Counter()
    for aten, fl in fcg.items():
        a = str(aten).lower()
        cls = ("convolution" if "conv" in a else
               "GEMM" if any(x in a for x in ("mm", "addmm", "bmm", "linear")) else
               "attention" if "scaled_dot" in a or "sdpa" in a else None)
        if cls: flops[cls] += fl
    roof_cls = {cls: (fl / PEAK_FLOPS) * 1e3 for cls, fl in flops.items()}   # compute roofline (ms)
    return dict(name="vae_decoder", walk_mult=1, KT=KT, KSRC={}, roof_cls=roof_cls,
                meta="eager real module, FLOPs via FlopCounterMode (compute roofline; pointwise=fusion est.)")

# =================================================================================
# per-component analysis: op-class rollup + buckets
# =================================================================================
def bucket(op_class, gap):
    if op_class == "overhead":
        return "REDUNDANT/artifact — sampling/copies (some determinism-only); remove or fuse"
    if gap is None:
        return "measured-only (no roofline)"
    if gap <= 1.3:
        return "at roofline — no recoverable headroom"
    if op_class in ("norm/adaLN/pointwise",):
        return "FUSION — unfused/eager pointwise; pull into compiled region or a fused kernel"
    if op_class == "GEMM":
        return "OCCUPANCY/precision — small-shape GEMM; larger batch / fp8 / better tile"
    if op_class == "convolution":
        return "OCCUPANCY/layout — channels-last / conv+pointwise fusion / precision"
    if op_class == "attention":
        return "KERNEL/launch — small-shape attention; fuse or better kernel"
    return "off roofline"

def analyze(comp):
    """op-class -> dict(measured_ms_per_call, roofline_ms, gap, headroom, bucket, kernels[])."""
    cls_ms = collections.defaultdict(float)
    cls_kernels = collections.defaultdict(list)
    for key, (us, cnt) in comp["KT"].items():
        oc = OPCLASS[family(key)]
        ms = us / 1e3 / REPLAYS
        cls_ms[oc] += ms
        cls_kernels[oc].append((key, ms, cnt // REPLAYS))
    out = {}
    for oc, m in cls_ms.items():
        rf = comp["roof_cls"].get(oc)
        gap = (m / rf) if (rf and rf > 0) else None
        hr = max(0.0, m - rf) if (rf and rf > 0) else (m if oc == "overhead" else 0.0)
        # pointwise with no analytic roofline in vae -> treat as fusion headroom (measured - tiny)
        if oc == "norm/adaLN/pointwise" and rf is None:
            hr = m * 0.9; gap = None
        out[oc] = dict(measured=m, roofline=rf, gap=gap, headroom=hr,
                       bucket=bucket(oc, gap if gap is not None else (99 if hr > 0 else 1)),
                       kernels=sorted(cls_kernels[oc], key=lambda r: -r[1]))
    return out

# =================================================================================
# collect + render
# =================================================================================
print("\ncollecting component: dit ...", flush=True)
DIT = collect_dit()
print("collecting component: vae_decoder ...", flush=True)
try:
    VAE = collect_vae()
except Exception as e:
    print(f"  (vae_decoder skipped: {e})"); VAE = None
components = [DIT] + ([VAE] if VAE else [])

print("\n" + "#" * 100)
print(f"# OPTIMALITY REPORT — Walk 'image_gen'  (t2i, {STEPS_WALK} denoise steps)")
print(f"# peaks: {PEAK_FLOPS/1e12:.0f} TFLOP/s, {PEAK_BW/1e9:.0f} GB/s   |   hierarchy: Walk > component > op-class > kernel")
print("#" * 100)

walk_cost = walk_head = 0.0
comp_rows = []
for comp in components:
    A = analyze(comp)
    per_call = sum(v["measured"] for v in A.values())
    per_call_hr = sum(v["headroom"] for v in A.values())
    mult = comp["walk_mult"]
    comp_rows.append((comp, A, per_call, per_call_hr, mult))
    walk_cost += per_call * mult; walk_head += per_call_hr * mult

print(f"\nWALK TOTAL:  ~{walk_cost:.0f} ms   |   recoverable ~{walk_head:.0f} ms "
      f"({100*walk_head/walk_cost if walk_cost else 0:.0f}%)")
print(f"  (multiplicity-weighted: dit x{STEPS_WALK} steps, vae_decoder x1)\n")
print(f"  {'component':<16}{'per-call ms':>12}{'xN':>6}{'walk ms':>10}{'share':>7}{'recoverable ms':>16}")
print("  " + "-" * 70)
for comp, A, pc, pch, mult in sorted(comp_rows, key=lambda r: -r[2] * r[4]):
    wc = pc * mult
    print(f"  {comp['name']:<16}{pc:>12.3f}{mult:>6}{wc:>10.0f}{100*wc/walk_cost if walk_cost else 0:>6.0f}%{pch*mult:>16.0f}")

# per-component detail
for comp, A, pc, pch, mult in sorted(comp_rows, key=lambda r: -r[3] * r[4]):
    print("\n" + "=" * 100)
    print(f"COMPONENT: {comp['name']}   (x{mult}/walk)   per-call {pc:.3f} ms  ->  walk {pc*mult:.0f} ms")
    print(f"           {comp['meta']}")
    print("=" * 100)
    print(f"   {'op-class':<24}{'measured':>10}{'roofline':>10}{'gap':>7}{'walk-headroom':>15}   bucket")
    for oc, v in sorted(A.items(), key=lambda kv: -kv[1]["headroom"] * mult):
        rf = f"{v['roofline']:.3f}" if v['roofline'] is not None else "   n/a"
        gp = f"{v['gap']:.0f}x" if v['gap'] is not None else "  n/a"
        print(f"   {oc:<24}{v['measured']:>10.3f}{rf:>10}{gp:>7}{v['headroom']*mult:>15.0f}   {v['bucket']}")
    # top kernels within each op-class, with source (DiT only has source map)
    for oc, v in sorted(A.items(), key=lambda kv: -kv[1]["headroom"] * mult):
        if v["headroom"] * mult < 1: continue
        print(f"\n   op-class '{oc}'  top kernels:")
        for key, ms, cps in v["kernels"][:3]:
            print(f"      {ms:>7.3f} ms/call  x{cps:<4} {key[:60]}")
            src = comp["KSRC"].get(key)
            if src:
                for loc, code in list(src.items())[:3]:
                    print(f"          {loc:<24} {code}")

print("\n" + "#" * 100)
print("# READ: 'walk-headroom' = per-call headroom x multiplicity = recoverable ms per generated image.")
print("# Rank components/op-classes by walk-headroom to pick what to optimize first.")
print("#" * 100)

if not os.environ.get("COSMOS3_KEEP_TRACE"): shutil.rmtree(TRACE_DIR, ignore_errors=True)
else: print(f"\n# Inductor dump kept at {TRACE_DIR}")
if _OUT:
    sys.stdout = sys.__stdout__; _outfh.close(); print(f"# report written to {_OUT}")
