"""Check trained routers against HF snapshots with identical inputs on one GPU."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from mstar.model.command_a_plus.components.language_model import CommandAPlusRouter


@torch.inference_mode()
def replay(checkpoint, traces):
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    config = json.loads((checkpoint / "config.json").read_text())["text_config"]
    router = CommandAPlusRouter(config["hidden_size"], config["num_experts"],
                                config["num_experts_per_tok"]).to(device="cuda", dtype=torch.bfloat16)
    results = []
    for path in sorted(traces.glob("hf_trace_*.safetensors")):
        with safe_open(path, framework="pt") as trace:
            steps = sorted({int(key.split(".")[0].removeprefix("step_")) for key in trace.keys()})
            for layer in range(config["num_hidden_layers"]):
                key = f"model.language_model.layers.{layer}.mlp.gate.weight"
                with safe_open(checkpoint / index[key], framework="pt") as weights:
                    router.weight.copy_(weights.get_tensor(key))
                for step in steps:
                    prefix = f"step_{step}.layer_{layer}."
                    inputs = trace.get_tensor(prefix + "norm").cuda()
                    actual_weights, actual_ids, _ = router(inputs)
                    logits = F.linear(inputs, router.weight).cpu()
                    expected_weights = trace.get_tensor(prefix + "router_weights")
                    expected_ids = trace.get_tensor(prefix + "router_indices")
                    expected_logits = trace.get_tensor(prefix + "router_logits")
                    results.append(dict(case=path.stem, layer=layer, step=step,
                                        indices_exact=torch.equal(actual_ids.cpu(), expected_ids),
                                        weights_exact=torch.equal(actual_weights.cpu(), expected_weights),
                                        logits_exact=torch.equal(logits, expected_logits)))
    destination = traces / "router_replay.json"
    destination.write_text(json.dumps(results, indent=2) + "\n")
    failures = [r for r in results if not all(r[k] for k in ("indices_exact", "weights_exact", "logits_exact"))]
    print(f"Router replay: {len(results)} snapshots; {len(failures)} differences; {destination}")
    if not results or failures:
        raise AssertionError(f"Router replay differences: {failures[:5]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    args = parser.parse_args()
    replay(args.checkpoint, args.traces)


if __name__ == "__main__":
    main()
