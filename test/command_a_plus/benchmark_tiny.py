"""Warm single-GPU BF16 timing probe; run as a module from the checkout.

Uses random tiny weights. These timings are not production throughput estimates.
Requires the same environment as test_reference's GPU test. No downloads.
"""

import json
import statistics
import tempfile
import time
from pathlib import Path

import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.engine.resources import SamplingReqConfig
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.step import StepContext
from mstar.model.command_a_plus.command_a_plus_model import CommandAPlusModel
from mstar.model.command_a_plus.config import KV_CACHE, SAMPLER
from mstar.model.command_a_plus.submodules import CommandAPlusLLMSubmodule
from mstar.model.submodule_base import ModelInputsFromEngine
from test.command_a_plus.test_integration import info
from test.command_a_plus.test_reference import make_pair


@torch.inference_mode()
def main():
    raw, reference, actual = make_pair(gpu=True)
    del reference
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "config.json").write_text(json.dumps(raw))
        model = CommandAPlusModel(directory)
        specs = {s.resource_key: s for s in model.get_node_resources()}
    specs[KV_CACHE].config.page_size = 16
    specs[KV_CACHE].config.max_num_pages = 16
    transfer = TransferEngineInfo("benchmark", "benchmark", LocalTransferEngine("localhost"))
    resources = {key: build_resource(spec, EngineResourceInfo(
        device=torch.device("cuda"), kv_dtype=torch.bfloat16, transfer_engine_info=transfer,
        dependencies={dependency: specs[dependency] for dependency in spec.depends_on()},
    )) for key, spec in specs.items()}
    runner = StepRunner(resources)
    node = CommandAPlusLLMSubmodule(actual, actual.config).eval()
    node.bind_node_resources(resources)
    engine = ModelInputsFromEngine(["probe"], {}, resources)
    measurements = []
    try:
        for repetition, decode_steps in enumerate((5, 32)):
            runner.ingest_request("probe", {SAMPLER: SamplingReqConfig(
                temperature=0, repetition_penalty=1, top_p=1, ignore_eos=True,
            )})
            tokens = torch.arange(2, 21, device="cuda")
            for index in range(decode_steps + 1):
                walk = "prefill" if index == 0 else "decode"
                torch.cuda.synchronize()
                start = time.perf_counter()
                inputs = [node.prepare_inputs(walk, info(walk), {"text_inputs": [tokens]})]
                step = node.declare_step(walk, ["probe"], inputs)
                step.set_ctx(StepContext(["probe"], walk, index % 2, False))
                outcome = runner.admit(step)
                if not outcome.ok or not outcome.ready:
                    raise RuntimeError(outcome)
                runner.plan(step)
                planned = time.perf_counter()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                result = node.forward_batched(walk, engine, **node.preprocess(walk, engine, inputs))
                end.record()
                runner.commit(step)
                tokens = result["probe"]["new_token"][0]
                end.synchronize()
                if repetition:
                    measurements.append(dict(walk=walk, planning_ms=(planned-start)*1000,
                                             forward_gpu_ms=begin.elapsed_time(end),
                                             total_ms=(time.perf_counter()-start)*1000))
            runner.remove_request("probe")
            node.cleanup_request("probe")
        report = {
            "device": torch.cuda.get_device_name(), "dtype": "bfloat16", "tp": 1,
            "parameters": sum(p.numel() for p in actual.parameters()),
            "prefill_tokens": 19, "decode_steps": 32,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "prefill": measurements[0],
            "decode_median": {key: statistics.median(row[key] for row in measurements[1:])
                              for key in ("planning_ms", "forward_gpu_ms", "total_ms")},
            "scope": "Tiny eager model + resources + greedy sampling; excludes HTTP/worker scheduling and HF reference",
        }
        print(json.dumps(report, indent=2))
    finally:
        runner.remove_request("probe")
        for resource in resources.values():
            resource.cleanup()


if __name__ == "__main__":
    main()
