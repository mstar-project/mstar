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
# By default the whole report is written to a FILE (not the console):
# perf_testing/results/cosmos3_feedback_<ts>.txt. Override the path with
# COSMOS3_FEEDBACK_OUT, or set it empty (COSMOS3_FEEDBACK_OUT=) to print to stdout.
# COSMOS3_KEEP_TRACE=1 keeps the Inductor debug dump (output_code.py, fx_graph_readable.py).
_OUT = os.environ.get("COSMOS3_FEEDBACK_OUT")
if _OUT is None:
    _results = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(_results, exist_ok=True)
    _OUT = os.path.join(_results, f"cosmos3_feedback_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt")
if _OUT:
    print(f"# writing report to {_OUT}", file=sys.__stdout__, flush=True)   # only console line
    _outfh = open(_OUT, "w")
    sys.stdout = _outfh                                                     # everything else -> file

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

# ---------- 4. EVERY kernel ranked by execution time, mapped to source ----------
def kind(key):
    k = key.lower()
    if key.startswith("triton_"): return "triton"
    if "conv" in k or "nchwtonhwc" in k or "implicit_gemm" in k: return "conv"
    if any(x in k for x in ("nvjet", "cublas", "cutlass", "splitkreduce")): return "gemm"
    if any(x in k for x in ("flashinfer", "fmha", "flash", "scaled_dot")): return "attention"
    if any(x in k for x in ("memcpy", "memset", "graphlaunch")): return "memcpy"
    return "eager"

rows = sorted(((key, us / 1e3 / REPLAYS, cnt // REPLAYS) for key, (us, cnt) in KTIME.items()),
              key=lambda r: -r[1])
total = sum(r[1] for r in rows)
print("\n" + "=" * 100)
print(f"EVERY KERNEL by execution time -> source   (total ~{total:.2f} ms/step, {len(rows)} kernels)")
print("=" * 100)
for rank, (key, ms, cps) in enumerate(rows, 1):
    k = kind(key)
    print(f"\n#{rank}  {ms:.3f} ms/step   x{cps}/step")
    print(f"    kernel: {key}")
    if key in KSRC and KSRC[key]:                                # exact: Inductor provenance
        for loc, code in KSRC[key].items():
            print(f"    source: {loc:<24} {code}")
    elif k in ("gemm", "conv"):                                  # library matmul/conv: the extern call sites
        print(f"    source: {k} library kernel (no per-kernel line); extern call sites:")
        for loc, code in list(EXT_SRC.items())[:8]:
            print(f"            {loc:<24} {code}")
    elif k == "attention":
        print(f"    source: FlashInfer paged attention (library kernel)")
    elif k == "eager":
        print(f"    source: uncompiled eager kernel (not in the Inductor graph); the kernel name above")
        print(f"            names the aten op -- run the capture-time module-hook profiler to pin the line")
    else:
        print(f"    source: (runtime/library kernel; no source mapping)")

if os.environ.get("COSMOS3_KEEP_TRACE"):
    print(f"\n# Inductor debug dump kept at: {TRACE_DIR}")
else:
    shutil.rmtree(TRACE_DIR, ignore_errors=True)
if _OUT:
    sys.stdout = sys.__stdout__      # restore before closing so no flush hits a closed file
    _outfh.close()
    print(f"# report written to {_OUT}")
