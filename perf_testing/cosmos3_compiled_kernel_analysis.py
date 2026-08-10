"""Op-/kernel-level feedback for the SERVED Cosmos3-Nano DiT (torch.compile + CUDA graph).

Builds the DiT the way it is actually served (``compile_denoise=True`` + the real
``CudaGraphRunner`` capture), then produces a per-step, per-kernel breakdown that
backtracks each kernel to its M* source line -- so you can see where the served
denoise step spends its time and which ``transformer.py`` lines to optimize.

What it prints, for one denoise step:
  1. every compiled subgraph, and for each:
       - the fused Triton kernels  -> the source line(s) each one fuses
         (via Inductor ``output_code.py`` provenance + ``fx_graph_readable.py``)
       - the NON-fused extern/cuBLAS calls (matmuls) -> their source line(s)
  2. a FULL measured device-time breakdown across ALL kernels
       (Triton fused / cuBLAS GEMM / FlashInfer attention / eager-uncompiled / ...),
       measured by replaying the captured graph under torch.profiler.
  3. the source lines the most fused kernels touch (fusion-boundary hot spots).

Why it is built this way (compile + CUDA-graph gotchas this handles):
  - torch.compile silently no-ops if the DiT submodule was already built eager and
    cached -> we evict ``_submodule_cache['dit']`` before rebuilding.
  - Under CUDA-graph *replay* the profiler loses CPU<->kernel correlation, so we
    attribute source at *compile* time (Inductor provenance) and measure time at
    *replay* time, then join by kernel name.
  - Inductor serves compiles from an on-disk cache and skips the debug dump, so we
    ``force_disable_caches`` for a cold recompile that re-emits the provenance.

Requirements:
  - the ``mstar`` conda env (torch + flashinfer + diffusers + mstar)
  - a CUDA GPU with a free ~24 GB
  - a local Cosmos3-Nano checkpoint. Point COSMOS3_NANO_DIR at it, or edit the
    default below. (HF: nvidia/Cosmos3-Nano.)

Run:
  cd <mstar repo>
  COSMOS3_NANO_DIR=/path/to/Cosmos3-Nano \
    CUDA_VISIBLE_DEVICES=<free gpu> python perf_testing/cosmos3_compiled_kernel_analysis.py

NOTE: this uses the offline test scaffolding (``tests/test_engine_cache``) which
runs with ``use_deterministic_algorithms(True)`` -- so a few ``_assert_async``
kernels appear in the breakdown that are absent in production. Treat absolute ms
as indicative; the relative split and the kernel->source mapping are the point.
"""
import os, re, sys, glob, shutil, tempfile, datetime, collections, torch

# ---- output recording ----
# By default the report is tee'd to perf_testing/results/cosmos3_feedback_<ts>.txt
# (and still printed to the console). Override the path with COSMOS3_FEEDBACK_OUT,
# or set it empty (COSMOS3_FEEDBACK_OUT=) to disable file recording.
# COSMOS3_KEEP_TRACE=1 keeps the Inductor debug dump (output_code.py, fx_graph_readable.py).
_OUT = os.environ.get("COSMOS3_FEEDBACK_OUT")
if _OUT is None:
    _results = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(_results, exist_ok=True)
    _OUT = os.path.join(_results, f"cosmos3_feedback_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt")
if _OUT:
    class _Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, s):
            for st in self.streams: st.write(s)
        def flush(self):
            for st in self.streams: st.flush()
    _outfh = open(_OUT, "w")
    sys.stdout = _Tee(sys.__stdout__, _outfh)
    print(f"# recording report to {_OUT}")

# ---- config ----
SNAP = os.environ.get(
    "COSMOS3_NANO_DIR",
    "/raid/hf/hub/models--nvidia--Cosmos3-Nano/snapshots/411f42a8fdfb8c5b2583cb8786e0938f49796eaa",
)
if not os.path.isdir(SNAP):
    raise SystemExit(f"Cosmos3-Nano checkpoint not found: {SNAP}\n"
                     f"Set COSMOS3_NANO_DIR to a local checkpoint dir.")
os.environ["COSMOS3_NANO_DIR"] = SNAP
TRACE_DIR = tempfile.mkdtemp(prefix="cosmos3_inductor_")   # Inductor debug dump

import torch._inductor.config as ind_cfg
import torch._inductor.metrics as ind_metrics
ind_cfg.trace.enabled = True
ind_cfg.trace.debug_dir = TRACE_DIR
ind_cfg.force_disable_caches = True                        # cold recompile -> re-emit provenance
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"

from torch.profiler import profile, ProfilerActivity
import mstar.model.cosmos3.tests.test_engine_cache as T

# ---------- 1. build served DiT (compile ON) + capture ----------
ctx = T._scenario(1)                                       # image (num_frames=1)
if ctx is None:
    raise SystemExit("scenario unavailable (needs COSMOS3_NANO_DIR + CUDA).")
model, device, dtype = ctx["model"], ctx["device"], ctx["dtype"]
model.config.compile_denoise = True
model._submodule_cache.pop("dit", None)                    # evict eager cache -> rebuild WITH compile
dit = model.get_submodule("dit", device=device)

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import CommGroup
from mstar.engine.cuda_graph_runner import CudaGraphRunner
from mstar.model.submodule_base import ModelInputsFromEngine
from mstar.utils.sampling import MultiSampler, MultiSamplingConfig

ind_metrics.reset()
dit.gen_capture_resolutions = ((T.H, T.W),)
rid = "cgr0"
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
print(">> warmup_and_capture (torch.compile + cuda-graph capture)...", flush=True)
runner.warmup_and_capture()
runner.register_request(rid)
print(f">> compiled: {ind_metrics.generated_kernel_count} Triton kernels\n", flush=True)

# ---------- 2. measured device-ms per kernel name via graph replay ----------
REPLAYS = 20
KTIME = collections.defaultdict(lambda: [0.0, 0])          # kernel name -> [us, calls]
try:
    fwd.graph_walk = "image_gen"
    ni_fixed = dit.prepare_inputs("image_gen", fwd,
        {"latents": [ctx["init"].clone()], "time_index": [torch.zeros(1, dtype=torch.long, device=device)]})
    replay = lambda: runner.run(graph_walk="image_gen", requires_cfg=False, request_ids=[rid],
                                inputs=[ni_fixed], per_request_info={rid: fwd}, submodule=dit)
    for _ in range(3): replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(REPLAYS): replay()
        torch.cuda.synchronize()
    for k in p.key_averages():
        us = getattr(k, "self_device_time_total", getattr(k, "self_cuda_time_total", 0))
        if us > 0: KTIME[k.key][0] += us; KTIME[k.key][1] += k.count
except Exception as e:
    print(f"(replay timing skipped: {e})\n")

def lookup_ms(name):
    for key, (us, cnt) in KTIME.items():
        if name in key: return us / 1e3 / REPLAYS, cnt
    return None, None

# ---------- 3. parse each subgraph: kernel -> source nodes -> source lines ----------
FILE_RE = re.compile(r'#\s*File:\s*(.+?):(\d+)\s+in\s+(\w+),\s*code:\s*(.*)')
NODE_RE = re.compile(r'^\s*(\w+)\s*:\s*"')                 # `pow_1: "f32[...]" = ...`
SRC_RE  = re.compile(r'Source Nodes:\s*\[(.*?)\],\s*Original ATen:\s*\[(.*?)\]')
DEF_RE  = re.compile(r'^\s*(triton_\w+)\s*=\s*async_compile')
RUN_RE  = re.compile(r'(triton_\w+)\.run\(')
META_RE = re.compile(r"'num_load':\s*(\d+),\s*'num_store':\s*(\d+),\s*'num_reduction':\s*(\d+)")
EXTERN_RE = re.compile(r'extern_kernels\.(\w+)\(')         # non-fused cuBLAS/library calls

def node_to_source(fx_path):
    m, cur = {}, None
    if not os.path.exists(fx_path): return m
    for line in open(fx_path):
        f = FILE_RE.search(line)
        if f: cur = (os.path.basename(f.group(1)), f.group(2), f.group(4).strip()); continue
        n = NODE_RE.match(line)
        if n and cur: m[n.group(1)] = cur
    return m

def kernels_of(oc_path):
    """returns (triton: name -> [set(nodes), set(aten), (nl,ns,nr)],
                extern: list of (op, set(nodes), set(aten)))."""
    lines = open(oc_path).read().splitlines()
    K = collections.defaultdict(lambda: [set(), set(), None])
    externs, last_src = [], None
    for i, line in enumerate(lines):
        s = SRC_RE.search(line)
        if s:
            nodes = [x.strip() for x in s.group(1).split(",") if x.strip()]
            aten  = [x.strip() for x in s.group(2).split(",") if x.strip()]
            last_src = (nodes, aten); continue
        e = EXTERN_RE.search(line)
        if e:
            n, a = last_src if last_src else ([], [])
            externs.append((e.group(1), set(n), set(a)))
        for rx in (DEF_RE, RUN_RE):
            mm = rx.search(line)
            if mm and last_src:
                K[mm.group(1)][0].update(last_src[0]); K[mm.group(1)][1].update(last_src[1])
        d = DEF_RE.search(line)
        if d:
            for j in range(i, min(i + 40, len(lines))):
                mt = META_RE.search(lines[j])
                if mt: K[d.group(1)][2] = tuple(map(int, mt.groups())); break
    return K, externs

subgraphs = sorted(glob.glob(f"{TRACE_DIR}/**/output_code.py", recursive=True))
print(f"# {len(subgraphs)} compiled subgraph(s); {ind_metrics.generated_kernel_count} Triton kernels total")
print("# each kernel below -> the M* source line(s) it fuses (transformer.py unless noted)\n")

srcline_hits = collections.Counter()
KSRC = collections.defaultdict(collections.OrderedDict)   # triton kernel name -> {loc: source-line text}
EXT_SRC = collections.OrderedDict()                       # extern/GEMM call-site loc -> source-line text
def src_of(nodes, n2s):
    srcs = collections.OrderedDict()
    for nd in nodes:
        if nd in n2s:
            f, ln, code = n2s[nd]; srcs[f"{f}:{ln}"] = code; srcline_hits[f"{f}:{ln}"] += 1
    return srcs

for gi, oc in enumerate(subgraphs):
    d = os.path.dirname(oc)
    n2s = node_to_source(os.path.join(d, "fx_graph_readable.py"))
    K, externs = kernels_of(oc)
    print("=" * 100)
    print(f"SUBGRAPH {gi}: {os.path.basename(d)}   ({len(K)} fused Triton, {len(externs)} extern/library calls)")
    print("=" * 100)
    for name, (nodes, aten, meta) in sorted(K.items()):
        ms, calls = lookup_ms(name)
        srcs = src_of(nodes, n2s)
        for loc, code in srcs.items(): KSRC[name][loc] = code
        tag = f"{ms:.3f} ms/step" if ms is not None else "n/a"
        memhint = f" loads/stores/reductions={meta}" if meta else ""
        ops = ",".join(a.replace("aten.", "") for a in sorted(aten))
        print(f"\n  > [Triton] {name}")
        print(f"      device time : {tag}   ({calls or 0} calls/{REPLAYS}-step-window)")
        print(f"      fused ATen  : {ops}{memhint}")
        print(f"      SOURCE LINES ({len(srcs)}):")
        for loc, code in srcs.items():
            print(f"         {loc:<26} {code}")
    for op, nodes, aten in externs:
        srcs = src_of(nodes, n2s)
        for loc, code in srcs.items(): EXT_SRC[loc] = code
        atens = ",".join(a.replace("aten.", "") for a in sorted(aten))
        print(f"\n  > [extern/cuBLAS] extern_kernels.{op}   (NON-fused library call; {atens or 'matmul'})")
        for loc, code in srcs.items():
            print(f"         {loc:<26} {code}")

# ---------- 4. FULL measured per-step breakdown by kernel family ----------
def classify(key):
    k = key.lower()
    if key.startswith("triton_"): return "Triton (fused compiled)"
    if any(x in k for x in ("nvjet", "cublas", "cutlass", "gemm")): return "cuBLAS GEMM (non-fused)"
    if any(x in k for x in ("flashinfer", "fmha", "attention", "flash")): return "attention (FlashInfer)"
    if "elementwise" in k or "reduce_kernel" in k or k.startswith("void at::native"): return "eager aten (uncompiled)"
    if any(x in k for x in ("memcpy", "memset", "graphlaunch")): return "memcpy/launch"
    return "other"

groups = collections.defaultdict(lambda: [0.0, 0]); rowsK = []
for key, (us, cnt) in KTIME.items():
    groups[classify(key)][0] += us; groups[classify(key)][1] += cnt
    rowsK.append((key, us / 1e3 / REPLAYS, cnt // REPLAYS))
tot = sum(v[0] for v in groups.values()) / 1e3 / REPLAYS
print("\n" + "=" * 100)
print(f"FULL measured per-step device breakdown (ALL kernels incl. NON-fused)   total ~{tot:.2f} ms/step")
print("=" * 100)
for g, (us, cnt) in sorted(groups.items(), key=lambda x: -x[1][0]):
    print(f"   {g:<28}{us / 1e3 / REPLAYS:>8.3f} ms/step{100 * us / (tot * 1e3 * REPLAYS) if tot else 0:>7.1f}%")

# ---------- 5. OPTIMALITY LADDER (VibeSim-style) + per-op-class metrics ----------
# Per op-class: measured device time (R0) vs the hardware-necessary roofline
# (R6 = max(FLOPs/peak_compute, bytes/peak_bw)). gap = measured/roofline;
# headroom = measured - roofline = recoverable ms/step. Peaks measured live.
def _bench(fn, iters=40, warm=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters
try:
    _a = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    _b = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    PEAK_FLOPS = (2 * 8192 ** 3) / (_bench(lambda: torch.mm(_a, _b)) * 1e-3)      # FLOP/s
    _src = torch.randn(1 << 29, device=device, dtype=torch.bfloat16)             # 1 GiB
    _dst = torch.empty_like(_src)                                                # preallocated (no per-iter alloc)
    PEAK_BW = (2 * _src.numel() * 2) / (_bench(lambda: _dst.copy_(_src)) * 1e-3)  # B/s (r+w)
    del _a, _b, _src, _dst
except Exception as e:
    PEAK_FLOPS, PEAK_BW = 2.0e15, 7.0e12
    print(f"(peak bench failed, using datasheet B200 defaults: {e})")

c = model.config
d, L = c.hidden_size, c.num_hidden_layers
Hh, Hkv, Dh, I = c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.intermediate_size
qdim, kvdim = Hh * Dh, Hkv * Dh
Tg = max(1, (T.H // 16 // 2) * (T.W // 16 // 2))       # gen latent-patch tokens (image, 1 frame)
Tund = len(ctx["cond"]); Tctx = Tund + Tg; CFG = 2; bpe = 2

def _gemm(M, N, K): return 2 * M * N * K, (M * K + K * N + M * N) * bpe
gf = gb = 0.0                                          # per-layer GEN pathway: qkv + o + gate/up/down
for M, N, K in [(Tg, qdim, d), (Tg, kvdim, d), (Tg, kvdim, d), (Tg, d, qdim),
                (Tg, I, d), (Tg, I, d), (Tg, d, I)]:
    f, b = _gemm(M, N, K); gf += f; gb += b
mult = CFG * L
gemm_f, gemm_b = mult * gf, mult * gb
attn_f = mult * (2 * 2 * Hh * Tg * Tctx * Dh)
attn_b = mult * ((Hh * Tg * Dh + 2 * Hkv * Tctx * Dh + Hh * Tg * Dh) * bpe)
norm_b = mult * ((6 * Tg * d + Tg * qdim + Tg * kvdim) * 2 * bpe)   # 4 layernorms + qk-norm + mod/gate

def _roof(fl, by): return max(fl / PEAK_FLOPS, by / PEAK_BW) * 1e3   # ms
ms_of = lambda g: groups.get(g, [0.0, 0])[0] / 1e3 / REPLAYS
CLASS_DEF = [
    ("GEMM (proj+MLP)",      ms_of("cuBLAS GEMM (non-fused)"), _roof(gemm_f, gemm_b)),
    ("attention",            ms_of("attention (FlashInfer)"),  _roof(attn_f, attn_b)),
    ("norm/adaLN/pointwise", ms_of("Triton (fused compiled)") + ms_of("eager aten (uncompiled)")
                             + ms_of("other") + ms_of("memcpy/launch"), _roof(0, norm_b)),
]
CLASS_OPT = {}                                          # op-class -> (measured, roofline, gap, headroom)
R0 = R6 = rec = 0.0
for name, m, roof in CLASS_DEF:
    gap = m / roof if roof > 0 else float('inf'); hr = max(0.0, m - roof)
    CLASS_OPT[name] = (m, roof, gap, hr); R0 += m; R6 += roof; rec += hr
print("\n" + "=" * 100)
print("OPTIMALITY LADDER (VibeSim-style)   peak: "
      f"{PEAK_FLOPS/1e12:.0f} TFLOP/s, {PEAK_BW/1e9:.0f} GB/s | shape Tg={Tg} Tctx={Tctx} CFG={CFG} L={L}")
print("=" * 100)
print(f"   {'op-class':<24}{'measured':>10}{'roofline':>10}{'gap':>7}{'headroom':>10}   (ms/step)")
for name, (m, roof, gap, hr) in sorted(CLASS_OPT.items(), key=lambda r: -r[1][3]):
    print(f"   {name:<24}{m:>10.3f}{roof:>10.3f}{gap:>7.1f}{hr:>10.3f}")
print("   " + "-" * 68)
print(f"   {'R0 measured step':<24}{R0:>10.3f}")
print(f"   {'R6 necessary (roofline)':<24}{'':<10}{R6:>10.3f}")
print(f"   {'recoverable headroom':<24}{'':<10}{'':<10}{'':<7}{rec:>10.3f}  ({100*rec/R0 if R0 else 0:.0f}% of step)")

# ---------- 6. TOP KERNELS by execution time: source line (+ copy) + optimality ----------
FAM2LADDER = {"cuBLAS GEMM (non-fused)": "GEMM (proj+MLP)",
              "attention (FlashInfer)": "attention",
              "Triton (fused compiled)": "norm/adaLN/pointwise",
              "eager aten (uncompiled)": "norm/adaLN/pointwise"}
print("\n" + "=" * 100)
print("TOP KERNELS by execution time  (for each: source line + copy of the line, and optimality)")
print("=" * 100)
for rank, (key, ms, cps) in enumerate(sorted(rowsK, key=lambda r: -r[1])[:14], 1):
    fam = classify(key)
    print(f"\n#{rank}  {ms:.3f} ms/step   x{cps}/step   [{fam}]")
    print(f"    kernel: {key}")
    # (1) source line(s) + a copy of each line
    if key in KSRC and KSRC[key]:
        print("    source:")
        for loc, code in KSRC[key].items():
            print(f"        {loc:<24} {code}")
    elif fam == "cuBLAS GEMM (non-fused)":
        print("    source: cuBLAS matmul (library kernel) -> GEN-pathway projection/MLP call sites:")
        for loc, code in list(EXT_SRC.items())[:8]:
            print(f"        {loc:<24} {code}")
    elif fam == "attention (FlashInfer)":
        print("    source: FlashInfer paged attention (library kernel); GEN attends [text K/V | gen]")
    elif fam == "eager aten (uncompiled)":
        print("    source: uncompiled eager kernel (not in the Inductor graph) -> attribute via the")
        print("            capture-time module-hook profiler (eager copies/casts/modulation/residuals")
        print("            around the attention graph break; the kernel name above hints at the op).")
    else:
        print("    source: (no source mapping for this kernel class)")
    # (2) optimality analysis (op-class roofline this kernel belongs to)
    lc = FAM2LADDER.get(fam)
    if lc and lc in CLASS_OPT:
        m, roof, gap, hr = CLASS_OPT[lc]
        verdict = ("at/below roofline -- no recoverable headroom" if gap <= 1.05
                   else f"{gap:.0f}x off roofline; op-class has {hr:.2f} ms/step recoverable")
        print(f"    optimality: op-class '{lc}'  measured {m:.2f} vs roofline {roof:.3f} ms/step  ->  {verdict}")
    else:
        print("    optimality: overhead/other (no roofline)")

# ---------- 7. hottest source lines by #kernels touching them ----------
print("\n" + "=" * 100)
print("SOURCE LINES that the most FUSED kernels touch (fusion-boundary hot spots):")
for loc, n in srcline_hits.most_common(15):
    print(f"   {n:>3} kernels   {loc}")

if os.environ.get("COSMOS3_KEEP_TRACE"):
    print(f"\n# Inductor debug dump kept at: {TRACE_DIR}")
else:
    shutil.rmtree(TRACE_DIR, ignore_errors=True)
if _OUT:
    sys.stdout = sys.__stdout__      # restore before closing so no flush hits a closed file
    _outfh.close()
    print(f"# report written to {_OUT}")
