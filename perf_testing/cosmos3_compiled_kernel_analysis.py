"""Kernel-level HEADROOM feedback for the SERVED Cosmos3-Nano DiT (torch.compile + CUDA graph).

Builds the DiT the way it is actually served (``compile_denoise=True`` + the real
``CudaGraphRunner`` capture), measures every kernel of one denoise step, and — via
the model-agnostic ``perf_testing/kernel_feedback`` package — scores each kernel
against the GPU's speed-of-light and buckets the gap:

    observed -> kernel_quality_gap -> speed_of_light
             -> movement_elimination -> fusion_in_graph -> estimated_floor

so the report ranks kernels by *recoverable* ms (better kernel / fuse / eliminate),
not merely by observed ms, and maps each back to its M* source line.

How the three measurement passes fit together (compile + CUDA-graph gotchas):
  - torch.compile silently no-ops if the DiT submodule was already built eager and
    cached -> we evict ``_submodule_cache['dit']`` before rebuilding.
  - Inductor serves compiles from an on-disk cache and skips the debug dump, so we
    ``force_disable_caches`` for a cold recompile that re-emits provenance. The dump
    gives every fused Triton kernel its exact tensor traffic + source lines, and
    every extern GEMM its shapes (-> exact FLOPs).
  - Under CUDA-graph *replay* the profiler loses CPU<->kernel correlation, so times
    come from replaying the captured graph under torch.profiler (join by kernel name).
  - Eager glue kernels (around the attention graph break) have no Inductor
    provenance; a separate single EAGER step under ``with_stack``+``record_shapes``
    attributes them to source and estimates their bytes.

Requirements: the ``mstar`` conda env, a CUDA GPU with ~24 GB free, and a local
Cosmos3-Nano checkpoint (COSMOS3_NANO_DIR, HF: nvidia/Cosmos3-Nano).

Run:
  cd <mstar repo>
  COSMOS3_NANO_DIR=/path/to/Cosmos3-Nano \
    CUDA_VISIBLE_DEVICES=<free gpu> python perf_testing/cosmos3_compiled_kernel_analysis.py

NOTE: the offline scaffolding (``tests/test_engine_cache``) runs with
``use_deterministic_algorithms(True)``; the induced ``_assert_async`` kernels are
bucketed as ``harness_artifacts`` and excluded from the floor.
"""
import collections
import datetime
import json
import os
import shutil
import sys
import tempfile

import torch

# ---- output recording ----
# By default the whole report is written to FILES (not the console):
# perf_testing/results/cosmos3_feedback_<ts>.txt (+ .json). Override the path with
# COSMOS3_FEEDBACK_OUT, or set it empty (COSMOS3_FEEDBACK_OUT=) to print to stdout.
# COSMOS3_KEEP_TRACE=1 keeps the Inductor debug dump.
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

from torch.profiler import ProfilerActivity, profile

import mstar.model.cosmos3.tests.test_engine_cache as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel_feedback import build_report, parse_dump, profile_eager_attribution, render_text

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
print(f">> compiled: {ind_metrics.generated_kernel_count} Triton kernels", flush=True)

# ---------- 2. measured device-ms per kernel name via graph replay ----------
REPLAYS = 20
KTIME = collections.defaultdict(lambda: [0.0, 0])          # kernel name -> [us, calls]
fwd.graph_walk = "image_gen"
ni_fixed = dit.prepare_inputs("image_gen", fwd,
    {"latents": [ctx["init"].clone()], "time_index": [torch.zeros(1, dtype=torch.long, device=device)]})
def replay():
    return runner.run(graph_walk="image_gen", requires_cfg=False, request_ids=[rid],
                      inputs=[ni_fixed], per_request_info={rid: fwd}, submodule=dit)
try:
    for _ in range(3):
        replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(REPLAYS):
            replay()
        torch.cuda.synchronize()
    for k in p.key_averages():
        us = getattr(k, "self_device_time_total", getattr(k, "self_cuda_time_total", 0))
        if us > 0:
            KTIME[k.key][0] += us
            KTIME[k.key][1] += k.count
except Exception as e:
    print(f"(replay timing skipped: {e})\n")

_total_ms = sum(us for us, _ in KTIME.values()) / 1e3 / REPLAYS
print(f">> replay timing: {len(KTIME)} kernels, ~{_total_ms:.2f} ms/step", flush=True)

# ---------- 3. parse the dump BEFORE the eager pass ----------
# (the eager step may compile extra dynamic-shape variants into TRACE_DIR;
# freeze the subgraph set that corresponds to the captured graph first)
subgraphs = parse_dump(TRACE_DIR)

# ---------- 4. one EAGER step under with_stack -> eager-kernel attribution ----------
# The compiled subgraphs are warm by now, so this profiles only real execution
# (not compilation); correlation works because nothing replays a CUDA graph here.
# The eager path must use the SAME attention backend as the captured graph:
# cosmos3 defaults to "dense_gen" (FlashAttention-3), which has no sm100 kernel
# image and aborts the process on B200 — the capture itself runs FlashInfer.
eager_attr = {}
try:
    _saved_backend = shared["cfg"].attention_backend
    shared["cfg"].attention_backend = "flashinfer"
    try:
        cm_e = T._mk_cm(shared, [rid])
    finally:
        shared["cfg"].attention_backend = _saved_backend
    ei_e = ModelInputsFromEngine(request_ids=[rid], per_request_info={rid: fwd}, cache_manager=cm_e)

    def _eager_step():
        ni_e = dit.prepare_inputs("image_gen", fwd,
            {"latents": [ctx["init"].clone()],
             "time_index": [torch.zeros(1, dtype=torch.long, device=device)]})
        with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
            dit.forward("image_gen", ei_e, **dit.preprocess("image_gen", ei_e, [ni_e]))
        torch.cuda.synchronize()

    _eager_step()                                          # settle: compiles eager-path variants
    eager_attr = profile_eager_attribution(_eager_step, project_markers=("mstar",))
    print(f">> eager attribution: {len(eager_attr)} kernel names mapped", flush=True)
except Exception as e:
    print(f"(eager attribution skipped: {e})", flush=True)

# ---------- 5. build the headroom report ----------
report = build_report(
    subgraphs,
    {k: tuple(v) for k, v in KTIME.items()},
    replays=REPLAYS,
    gpu_name=torch.cuda.get_device_name(),
    eager_attr=eager_attr,
    meta={"model": "Cosmos3-Nano DiT", "walk": "image_gen", "height": T.H, "width": T.W,
          "num_frames": 1, "steps": T.STEPS, "guidance_scale": T.GS, "replays": REPLAYS,
          "compiled_subgraphs": len(subgraphs),
          "triton_kernels_compiled": ind_metrics.generated_kernel_count},
)
print()
print(render_text(report))

if _OUT:
    json_path = os.path.splitext(_OUT)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(report.to_json(), f, indent=1)

if os.environ.get("COSMOS3_KEEP_TRACE"):
    print(f"\n# Inductor debug dump kept at: {TRACE_DIR}")
else:
    shutil.rmtree(TRACE_DIR, ignore_errors=True)
if _OUT:
    sys.stdout = sys.__stdout__      # restore before closing so no flush hits a closed file
    _outfh.close()
    print(f"# report written to {_OUT}")
    print(f"# json    written to {json_path}")
