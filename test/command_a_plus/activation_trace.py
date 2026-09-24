"""Opt-in activation snapshots for diagnosing full-checkpoint parity failures."""

import json

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file


class ActivationTrace:
    def __init__(self, backbone, *, reference):
        self.tensors = {}
        self.step = 0
        self.handles = []
        self.reference = reference
        self.watch(backbone.embed_tokens, "embedding")
        self.watch(backbone.norm, "final_norm")
        for i, layer in enumerate(backbone.layers):
            prefix = f"layer_{i}"
            self.handles.append(layer.register_forward_pre_hook(
                self.input_hook(prefix + ".input"), with_kwargs=True,
            ))
            for module, name in ((layer, "output"), (layer.input_layernorm, "norm"),
                                 (layer.self_attn, "attention"), (layer.mlp, "moe"),
                                 (layer.mlp.shared_experts, "shared")):
                self.watch(module, f"{prefix}.{name}")
            self.handles.append(layer.mlp.gate.register_forward_hook(self.router_hook(prefix)))

    def put(self, name, tensor):
        tensor = tensor.detach().reshape(-1, tensor.shape[-1])
        self.tensors[f"step_{self.step}.{name}"] = tensor.to("cpu", copy=True).contiguous()

    def input_hook(self, name):
        def capture(module, args, kwargs):
            self.put(name, args[0] if args else kwargs["hidden_states"])
        return capture

    def watch(self, module, name):
        def capture(module, args, output):
            self.put(name, output[0] if isinstance(output, tuple) else output)
        self.handles.append(module.register_forward_hook(capture))

    def router_hook(self, prefix):
        def capture(module, args, output):
            if self.reference:
                logits, weights, indices = output
            else:
                weights, indices, _ = output
                logits = F.linear(args[0], module.weight)
            self.put(prefix + ".router_logits", logits)
            self.put(prefix + ".router_weights", weights)
            self.put(prefix + ".router_indices", indices)
        return capture

    def save(self, path):
        save_file(self.tensors, path)
        self.tensors.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.inference_mode()
def compare_reference_routing(runtime, args, rank):
    """Counterfactual: hold all routing IDs AND weights at the HF values.

    Run fresh requests with the original teacher-forced prefixes. This is a
    diagnostic intervention, never an alternative implementation or pass gate.
    """
    reference = json.loads((args.output / "hf_reference.json").read_text())
    expected = load_file(args.output / "hf_logits.safetensors")
    cases, outputs = [], {}
    for case_index, case in enumerate(reference["cases"]):
        with safe_open(args.output / f"hf_trace_{case_index}.safetensors", framework="pt") as trace:
            routing = {key: trace.get_tensor(key).cuda() for key in trace.keys()
                       if key.endswith(("router_indices", "router_weights"))}
        current_step = [0]
        handles = []

        def hook(layer_index, current_step=current_step, routing=routing):
            def replace(module, inputs, output):
                prefix = f"step_{current_step[0]}.layer_{layer_index}.router_"
                weights, indices = routing[prefix + "weights"], routing[prefix + "indices"]
                if weights.shape != output[0].shape or indices.shape != output[1].shape:
                    raise ValueError("routing intervention token shapes differ")
                return weights, indices, output[2]
            return replace

        rid = f"reference_routing_{case_index}"
        runtime.start(rid)
        try:
            for i, layer in enumerate(runtime.node.language_model.model.layers):
                handles.append(layer.mlp.gate.register_forward_hook(hook(i)))
            rows = [torch.tensor(case["input_ids"], device="cuda")]
            metrics, logits_rows = [], []
            for step, token in enumerate(case["generated_ids"]):
                current_step[0] = step
                logits, tokens, _ = runtime.step([rid], rows, step)
                ours, theirs = logits[0].float().cpu(), expected[f"case_{case_index}"][step]
                delta = ours - theirs
                metrics.append(dict(step=step, rmse=delta.square().mean().sqrt().item(),
                                    max_abs=delta.abs().max().item(),
                                    cosine=F.cosine_similarity(ours, theirs, dim=0).item(),
                                    token=tokens.item(), reference_token=token))
                logits_rows.append(ours)
                rows = [torch.tensor([token], device="cuda")]
            cases.append(dict(case=case_index, metrics=metrics))
            outputs[f"case_{case_index}"] = torch.stack(logits_rows)
            if rank == 0:
                print("Reference-routing intervention", case_index, json.dumps(metrics), flush=True)
        finally:
            for handle in handles:
                handle.remove()
            runtime.remove(rid)
            runtime.check_empty()
    if rank == 0:
        save_file(outputs, args.output / "reference_routing_logits.safetensors")
        (args.output / "reference_routing.json").write_text(json.dumps(cases, indent=2) + "\n")
