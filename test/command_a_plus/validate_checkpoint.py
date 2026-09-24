"""Opt-in full-checkpoint reference, TP parity, and warmed timing runs.

This script never downloads weights. Run `reference` in one process, let it exit,
then run `mstar` with torchrun using the saved reference directory. Both modes
require an explicitly supplied local checkpoint. Outputs belong in scratch space.
"""

import argparse
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from mstar.model.command_a_plus.command_a_plus_model import CommandAPlusModel

PROMPTS = [
    "What is 2 + 2? Answer briefly.",
    "Complete this sentence: The capital of France is",
    "Write one short sentence greeting a new colleague.",
]


def sync_all():
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


@torch.inference_mode()
def reference(args):
    from transformers import Cohere2VisionForConditionalGeneration

    adapter = CommandAPlusModel(str(args.checkpoint))
    tokenizer = adapter._get_tokenizer()
    devices = torch.cuda.device_count()
    device_map = {
        "model.vision_tower": 0, "model.multi_modal_projector": 0,
        "model.language_model.embed_tokens": 0,
        "model.language_model.rotary_emb": 0,
        "model.language_model.norm": devices - 1, "lm_head": 0,
    }
    for index in range(adapter.config.num_hidden_layers):
        device_map[f"model.language_model.layers.{index}"] = min(
            devices - 1, index * devices // adapter.config.num_hidden_layers,
        )
    print("Loading HF reference", device_map, flush=True)
    started = time.perf_counter()
    model, loading = Cohere2VisionForConditionalGeneration.from_pretrained(
        args.checkpoint, local_files_only=True, dtype=torch.bfloat16,
        device_map=device_map, attn_implementation="eager", experts_implementation="eager",
        output_loading_info=True,
    )
    model.eval()
    sync_all()
    load_seconds = time.perf_counter() - started
    write_json(args.output / "hf_loading.json", {
        key: sorted(value) if isinstance(value, set) else value for key, value in loading.items()
    })
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"HF loading was not exact: {loading}")
    print(f"HF loaded in {load_seconds:.1f}s", flush=True)
    trace = None
    if args.trace:
        from test.command_a_plus.activation_trace import ActivationTrace

        trace = ActivationTrace(model.model.language_model, reference=True)
    tensors, cases = {}, []
    for case_index, prompt in enumerate(PROMPTS):
        ids = adapter.process_prompt(prompt, ["text"], ["text"])["text_inputs"][0]
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        )
        inputs, cache, generated, logits_rows, timings = ids[None].cuda(0), None, [], [], []
        for _step in range(args.tokens):
            if trace is not None:
                trace.step = _step
            sync_all()
            start = time.perf_counter()
            output = model(input_ids=inputs, past_key_values=cache, use_cache=True, logits_to_keep=1)
            sync_all()
            timings.append((time.perf_counter() - start) * 1000)
            cache = output.past_key_values
            logits = output.logits[0, -1].float().cpu()
            if not torch.isfinite(logits).all():
                raise AssertionError("HF produced non-finite logits")
            logits_rows.append(logits)
            token = logits.argmax().item()
            generated.append(token)
            inputs = torch.tensor([[token]], device="cuda:0")
            if token == adapter.config.eos_token_id:
                break
        tensors[f"case_{case_index}"] = torch.stack(logits_rows)
        top = logits_rows[0].topk(10)
        case = dict(prompt=prompt, rendered_prompt=rendered, input_ids=ids.tolist(),
                    generated_ids=generated, text=tokenizer.decode(generated, skip_special_tokens=False),
                    first_top_ids=top.indices.tolist(), first_top_logits=top.values.tolist(),
                    step_ms=timings)
        cases.append(case)
        if trace is not None:
            trace.save(args.output / f"hf_trace_{case_index}.safetensors")
        print("HF case", case_index, repr(case["text"]), flush=True)
        del output, cache
    if trace is not None:
        trace.close()
    save_file(tensors, args.output / "hf_logits.safetensors")
    write_json(args.output / "hf_reference.json", dict(
        checkpoint=str(args.checkpoint.resolve()), load_seconds=load_seconds, cases=cases,
        peak_allocated_gib=[torch.cuda.max_memory_allocated(i) / 2**30 for i in range(devices)],
    ))


class Runtime:
    def __init__(self, args, group):
        from mstar.communication.tensors import LocalTransferEngine
        from mstar.distributed.communication import CommGroup, JointGroups
        from mstar.engine.resources.base import EngineResourceInfo, build_resource
        from mstar.engine.resources.kv.transfer import TransferEngineInfo
        from mstar.engine.resources.runner import StepRunner
        from mstar.model.command_a_plus.config import KV_CACHE

        self.model = CommandAPlusModel(str(args.checkpoint), max_seq_len=args.context)
        started = time.perf_counter()
        self.node = self.model.get_submodule("LLM", device=torch.device("cuda", group.rank),
                                             tp_group=group, autocast_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started
        print(f"rank {group.rank}: weights loaded in {self.load_seconds:.1f}s", flush=True)
        specs = {spec.resource_key: spec for spec in self.model.get_node_resources()}
        specs[KV_CACHE].config.max_num_pages = 256
        specs[KV_CACHE].config.page_size = 128
        transfer = TransferEngineInfo(f"validation_{group.rank}", "validation", LocalTransferEngine("localhost"))
        self.resources = {key: build_resource(spec, EngineResourceInfo(
            device=torch.device("cuda", group.rank), kv_dtype=torch.bfloat16,
            joint_comm_group=JointGroups(group, CommGroup.trivial()), transfer_engine_info=transfer,
            dependencies={dep: specs[dep] for dep in spec.depends_on()},
        )) for key, spec in specs.items()}
        self.runner = StepRunner(self.resources)
        self.node.bind_node_resources(self.resources)
        self.initial_free_pages = self.resources[KV_CACHE]._arena.num_free

    def start(self, rid):
        from mstar.engine.resources import SamplingReqConfig
        from mstar.model.command_a_plus.config import SAMPLER

        self.runner.ingest_request(rid, {SAMPLER: SamplingReqConfig(
            temperature=0, top_p=1, repetition_penalty=1, ignore_eos=True,
        )})

    def step(self, rids, rows, iteration):
        from mstar.engine.resources.step import StepContext
        from mstar.model.command_a_plus.config import LOCAL_ATTN, SAMPLER
        from mstar.model.submodule_base import ModelInputsFromEngine

        walk = "prefill" if iteration == 0 else "decode"
        started = time.perf_counter()
        inputs = [self.node.prepare_inputs(walk, None, {"text_inputs": [row]}) for row in rows]
        step = self.node.declare_step(walk, rids, inputs)
        step.set_ctx(StepContext(rids, walk, iteration % 2, False))
        admitted = self.runner.admit(step)
        if not admitted.ok or not admitted.ready:
            raise RuntimeError(admitted)
        self.runner.plan(step)
        planned = time.perf_counter()
        engine = ModelInputsFromEngine(rids, {}, self.resources)
        ids = self.node.preprocess(walk, engine, inputs)["text_inputs"]
        lm = self.node.language_model
        hidden = lm(lm.model.embed_tokens(ids), label="main")
        if iteration == 0:
            hidden = self.resources[LOCAL_ATTN].select_last_hidden(hidden, label="main")
        logits = lm.compute_logits(hidden)
        tokens = self.resources[SAMPLER].sample(rids, logits=logits)
        self.runner.commit(step)
        torch.cuda.synchronize()
        return logits, tokens, dict(planning_ms=(planned-started)*1000,
                                   total_ms=(time.perf_counter()-started)*1000)

    def remove(self, rid):
        self.runner.remove_request(rid)
        self.node.cleanup_request(rid)

    def check_empty(self):
        from mstar.model.command_a_plus.config import KV_CACHE

        if self.resources[KV_CACHE]._arena.num_free != self.initial_free_pages:
            raise AssertionError("KV pages were not returned after request cleanup")

    def close(self):
        for resource in self.resources.values():
            resource.cleanup()


@torch.inference_mode()
def mstar(args):
    import torch.distributed as dist

    from mstar.distributed.communication import CommGroup

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    group = CommGroup(dist.get_rank(), dist.get_rank(), list(range(dist.get_world_size())))
    group.device_group = dist.group.WORLD
    runtime = None
    try:
        runtime = Runtime(args, group)
        trace = None
        if args.trace and rank == 0:
            from test.command_a_plus.activation_trace import ActivationTrace

            trace = ActivationTrace(runtime.node.language_model.model, reference=False)
        reference_info = json.loads((args.output / "hf_reference.json").read_text())
        expected = load_file(args.output / "hf_logits.safetensors")
        results, tensors = [], {}
        for index, case in enumerate(reference_info["cases"]):
            ids = runtime.model.process_prompt(case["prompt"], ["text"], ["text"])["text_inputs"][0]
            if ids.tolist() != case["input_ids"]:
                raise AssertionError("HF and M* input tokens differ")
            rid = f"parity_{index}"
            runtime.start(rid)
            rows, metrics, logits_rows = [ids.cuda()], [], []
            for step, reference_token in enumerate(case["generated_ids"]):
                if trace is not None:
                    trace.step = step
                logits, tokens, timing = runtime.step([rid], rows, step)
                ours = logits[0].float().cpu()
                theirs = expected[f"case_{index}"][step]
                if not torch.isfinite(ours).all() or not torch.isfinite(theirs).all():
                    raise AssertionError(f"Non-finite logits in case {index}, step {step}")
                delta = ours - theirs
                token = tokens.item()
                gap = (theirs.max() - theirs[token]).item()
                metrics.append(dict(**timing, max_abs=delta.abs().max().item(),
                                    rmse=delta.square().mean().sqrt().item(),
                                    cosine=torch.nn.functional.cosine_similarity(ours, theirs, dim=0).item(),
                                    token=token, reference_token=reference_token, reference_gap=gap,
                                    top10_overlap=len(set(ours.topk(10).indices.tolist())
                                                      & set(theirs.topk(10).indices.tolist()))))
                logits_rows.append(ours)
                # Teacher forcing compares the same prefix after any BF16 near tie.
                rows = [torch.tensor([reference_token], device="cuda")]
            runtime.remove(rid)
            runtime.check_empty()
            tensors[f"case_{index}"] = torch.stack(logits_rows)
            results.append(dict(prompt=case["prompt"], metrics=metrics))
            if rank == 0:
                if trace is not None:
                    trace.save(args.output / f"mstar_trace_{index}.safetensors")
                print("M* parity", index, json.dumps(metrics), flush=True)
        if trace is not None:
            trace.close()
        if rank == 0:
            save_file(tensors, args.output / "mstar_logits.safetensors")
            write_json(args.output / "mstar_parity.json", results)
        if args.trace:
            from test.command_a_plus.activation_trace import compare_reference_routing

            compare_reference_routing(runtime, args, rank)
        # Stop before benchmarks if the same-prefix logits do not match. Keep
        # artifacts even on failure so discrepancies can be diagnosed.
        for result in results:
            for metric in result["metrics"]:
                if metric["rmse"] > args.max_rmse or metric["cosine"] < args.min_cosine:
                    raise AssertionError(f"Logit parity failed: {metric}")
                if metric["max_abs"] > args.max_abs:
                    raise AssertionError(f"Maximum logit error exceeded its bound: {metric}")
                if metric["reference_gap"] > 2 * metric["max_abs"] + 1e-6:
                    raise AssertionError(f"Greedy token outside measured rounding bound: {metric}")
        # A fresh free-running request verifies actual generated text as well.
        generated_cases = []
        for index, case in enumerate(reference_info["cases"]):
            rid = f"greedy_{index}"
            runtime.start(rid)
            rows, generated, timings = [torch.tensor(case["input_ids"], device="cuda")], [], []
            for step in range(args.tokens):
                _, tokens, timing = runtime.step([rid], rows, step)
                token = tokens.item()
                generated.append(token)
                timings.append(timing)
                rows = [tokens]
                if token == runtime.model.config.eos_token_id:
                    break
            runtime.remove(rid)
            runtime.check_empty()
            text = runtime.model._get_tokenizer().decode(generated, skip_special_tokens=False)
            generated_cases.append(dict(prompt=case["prompt"], ids=generated, text=text, timings=timings))
            if rank == 0:
                print("M* greedy", index, repr(generated_cases[-1]["text"]), flush=True)
        benchmark = []
        for length in args.lengths:
            if length + args.benchmark_tokens > args.context:
                raise ValueError("benchmark prompt plus output exceeds configured context")
            seed = runtime.model._get_tokenizer().encode("The quick brown fox jumps over the lazy dog. ",
                                                         add_special_tokens=False)
            prompt = (seed * (length // len(seed) + 1))[:length]
            # Warm each prefill shape before measuring; the first encounter can
            # compile a different fused-MoE configuration even after decode is warm.
            for repetition in range(2):
                rid = f"benchmark_{length}_{repetition}"
                runtime.start(rid)
                rows, timings = [torch.tensor(prompt, device="cuda")], []
                count = min(4, args.benchmark_tokens) if repetition == 0 else args.benchmark_tokens
                for step in range(count):
                    _, tokens, timing = runtime.step([rid], rows, step)
                    timings.append(timing)
                    rows = [tokens]
                runtime.remove(rid)
                runtime.check_empty()
            benchmark.append(dict(prompt_tokens=length, output_tokens=args.benchmark_tokens,
                                  prefill_ms=timings[0]["total_ms"],
                                  decode_median_ms=statistics.median(t["total_ms"] for t in timings[1:])))
            if rank == 0:
                print("M* benchmark", benchmark[-1], flush=True)
        memory = [None] * group.world_size
        dist.all_gather_object(memory, torch.cuda.max_memory_allocated() / 2**30)
        if rank == 0:
            write_json(args.output / "mstar_results.json", dict(
                tp=group.world_size, load_seconds=runtime.load_seconds, parity=results,
                greedy=generated_cases, benchmark=benchmark, peak_allocated_gib=memory,
            ))
    finally:
        if runtime is not None:
            runtime.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reference", "mstar"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--lengths", type=int, nargs="*", default=[])
    parser.add_argument("--benchmark-tokens", type=int, default=32)
    parser.add_argument("--max-rmse", type=float, default=0.15)
    parser.add_argument("--max-abs", type=float, default=1.0)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--trace", action="store_true",
                        help="save activations and diagnose M* with saved HF routing choices and weights")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not (args.checkpoint / "model.safetensors.index.json").is_file():
        parser.error("a complete local sharded checkpoint is required")
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())
    missing = [name for name in set(index["weight_map"].values()) if not (args.checkpoint / name).is_file()]
    if missing:
        parser.error(f"checkpoint download is incomplete; missing {sorted(missing)}")
    if args.tokens < 1 or args.benchmark_tokens < 2:
        parser.error("tokens must be positive and benchmark-tokens must be at least two")
    (reference if args.mode == "reference" else mstar)(args)


if __name__ == "__main__":
    main()
