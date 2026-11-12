"""Shared pieces of the draft-model and EAGLE-3 measurements (RFC #1186, "Phases"): a Qwen3 node on its
checkpoint's real weights on the real Engine, captured, bf16; a stand-in drafter that makes a verifying node's
step the target's verify alone; and per-step device timing.

The stand-in drafter proposes each row's anchor ``k`` times and keeps nothing: the verify step then runs the
target's ``k + 1`` rows a request through the verify attention and sampler, and nothing of a drafter. Any Qwen3
checkpoint takes it, so the same step measures Qwen3-8B as a target and Qwen3-0.6B as a draft model running its
``k + 1``-row block (drafter shape (a), one pass). Benchmark only: it patches `Qwen3Model._drafter`, not model code.

The driver is the CPU tests' (`test/modular/qwen3_cpu.py`), on the device, with ``ignore_eos`` so no row stops
inside a measurement. Each arm runs in a fresh process (`run_arm`): no engine, compile cache or memory carries over.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from mstar.model.components.hosted_drafter import HostedDrafter, HostedDrafterDecl

REPO = Path(__file__).resolve().parents[2]
TARGET = "Qwen/Qwen3-8B"
DRAFT = "Qwen/Qwen3-0.6B"


class _NullDrafter(nn.Module, HostedDrafter):
    """Drafts each row's anchor ``k`` times; no state, no resources, no weights."""

    def __init__(self, needs):
        super().__init__()
        self.needs = needs

    def propose(self, inputs) -> torch.Tensor:
        return inputs.input_ids.reshape(-1, 1).long().expand(-1, inputs.num_draft_tokens).contiguous()

    def prefill(self, inputs) -> None:
        return None


@dataclass(frozen=True)
class NullDrafterDecl(HostedDrafterDecl):
    """A drafter declaration with nothing to declare: the node verifies, and the step is the target's."""

    num_draft_tokens: int
    needs: object = field(default=None)

    def __post_init__(self):
        from mstar.engine.resources.draft.drafter import ChunkBoundary, DrafterNeeds

        object.__setattr__(self, "needs", DrafterNeeds(chunk_boundary=ChunkBoundary.NONE))

    def node_resources(self, nodes):
        return []

    def prefill_steps(self):
        return {}

    def verify_steps(self):
        return {}

    def build(self, *, device, comm_group=None, dtype=None, target=None):
        return _NullDrafter(self.needs).to(device)


def qwen3_model(path: str, k: int | None, prefill_buckets: list[int] | None = None, mode: str = "synchronous"):
    """``Qwen3Model`` on ``path``; with ``k``, verifying ``k`` drafts a step through the stand-in drafter.
    ``prefill_buckets``: the captured prompt ladder, in place of the node's."""
    from mstar.model.qwen3.qwen3_model import Qwen3Model
    from mstar.model.qwen3.submodules import Qwen3LMSubmodule

    if prefill_buckets is not None:
        Qwen3LMSubmodule.PREFILL_TOKEN_BUCKETS = sorted(prefill_buckets)

    draft = None
    if k is not None:
        def _drafter(self, cfg):
            # the prefix root reads the drafter's checkpoint directory: the target's own stands in
            self._drafter_dir = self.local_dir
            return NullDrafterDecl(cfg.num_draft_tokens)

        Qwen3Model._drafter = _drafter
        draft = {"method": "dspark", "model": "stand-in", "num_draft_tokens": k, "mode": mode}
    return Qwen3Model(model_path_hf=path, draft=draft)


def engine(model, rows: list[int], max_num_pages: int = 1024, captured: bool = True):
    """The LLM node on the real Engine, its decode captures at ``rows``. 1024 pages of 128 tokens hold 64 rows of
    about 1.1k tokens or one 32k prompt (Qwen3-8B: about 19 GB of KV)."""
    from mstar.distributed.communication import WorkerParallelGroups
    from mstar.engine.engine import Engine
    from mstar.engine.resources.kv import manager as manager_mod
    from mstar.engine.resources.kv.config import KVSpec
    from mstar.model.qwen3.submodules import Qwen3LMSubmodule

    sys.path.insert(0, str(REPO / "test" / "modular"))
    from qwen3_cpu import _StubTransfer

    # an engine outside a worker has no transfer engine for its KV pool
    manager_mod.KVTransferManager = _StubTransfer
    Qwen3LMSubmodule.DECODE_CAPTURE_BATCH_SIZES = sorted(set(rows))
    if not captured:
        model.accelerator_graph = False
    sub = model.get_submodule("LLM", device="cuda", autocast_dtype=torch.bfloat16)
    specs = model.get_node_resources()
    for spec in specs:
        if isinstance(spec, KVSpec):
            spec.config.max_num_pages = max_num_pages
    eng = Engine(graph_runtime=None, autocast_dtype=torch.bfloat16)
    eng.load_model(
        {"LLM": sub}, specs, WorkerParallelGroups(num_workers=1, global_rank=0),
        torch.device("cuda", torch.cuda.current_device()),
        transfer_engine_info=None, kv_cache_type=torch.bfloat16, model=model,
    )
    if captured:
        eng.warmup()
    return eng


def driver(eng, model):
    sys.path.insert(0, str(REPO / "test" / "modular"))
    import qwen3_cpu

    real = qwen3_cpu.SamplingReqConfig
    if not getattr(real, "_ignores_eos", False):
        # no row stops inside a measurement (a benchmark-parity knob), set before the engine takes the config
        def config(**kw):
            return real(ignore_eos=True, **kw)

        config._ignores_eos = True
        qwen3_cpu.SamplingReqConfig = config
    return qwen3_cpu.Driver(eng, model)


def prompts(rows: int, length: int, seed: int = 0) -> dict[int, list[int]]:
    gen = torch.Generator().manual_seed(seed)
    return {rid: torch.randint(1000, 20000, (length,), generator=gen).tolist() for rid in range(1, rows + 1)}


def step_times(drv, live: list[int], warmup: int, steps: int) -> dict:
    """Decode (or verify) steps over ``live``: device ms per step as the kernels' summed device time (inside
    graph replays too) and as CUDA-event wall time around the steps, which also holds the host's time between
    them."""
    for _ in range(warmup):
        drv.step(live)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        drv.step(live)
    end.record()
    torch.cuda.synchronize()
    wall = start.elapsed_time(end) / steps
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            drv.step(live)
        torch.cuda.synchronize()
    kernels = sum(e.self_device_time_total for e in prof.key_averages()) / 1000 / steps
    return {"kernel_ms": kernels, "wall_ms": wall}


def weight_bytes(model) -> int:
    """Bytes of the node's weights in bf16, the head counted once (a tied head reads the embedding again)."""
    cfg = model.config
    per_layer = (
        cfg.hidden_size * (cfg.num_attention_heads + 2 * cfg.num_key_value_heads) * cfg.head_dim  # q, k, v
        + cfg.num_attention_heads * cfg.head_dim * cfg.hidden_size  # o
        + 3 * cfg.hidden_size * cfg.intermediate_size  # gate, up, down
    )
    head = cfg.vocab_size * cfg.hidden_size
    # the embedding is a gather (a row a token), the head a full read
    return 2 * (cfg.num_hidden_layers * per_layer + head)


def kv_bytes(model, rows: int, context: int) -> int:
    """Bytes of KV a step's attention reads: every layer's K and V over ``context`` tokens a row, bf16."""
    cfg = model.config
    return 2 * 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * context * rows


def run_arm(module: str, arm: dict) -> dict:
    """``module``'s ``--one`` entry point in a fresh process; its last stdout line is the arm's JSON."""
    command = [sys.executable, "-m", module, "--one", json.dumps(arm)]
    done = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
    if done.returncode:
        sys.stderr.write(done.stdout[-4000:] + done.stderr[-8000:])
        raise RuntimeError(f"{module} {arm}: the arm failed (output above)")
    return json.loads(done.stdout.strip().splitlines()[-1])
