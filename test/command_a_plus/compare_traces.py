"""Compare saved HF and M* traces without allocating GPU memory."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def differences(reference, actual):
    reference, actual = reference.float(), actual.float()
    delta = actual - reference
    return dict(max_abs=delta.abs().max().item(), rmse=delta.square().mean().sqrt().item(),
                relative_l2=(delta.norm() / reference.norm().clamp_min(1e-12)).item())


def compare(directory):
    results = []
    for path in sorted(directory.glob("hf_trace_*.safetensors")):
        case = int(path.stem.removeprefix("hf_trace_"))
        other = directory / f"mstar_trace_{case}.safetensors"
        with safe_open(path, framework="pt") as hf, safe_open(other, framework="pt") as mstar:
            if set(hf.keys()) != set(mstar.keys()):
                raise ValueError(f"trace keys differ for case {case}")
            for key in hf.keys():
                x, y = hf.get_tensor(key), mstar.get_tensor(key)
                if x.shape != y.shape:
                    raise ValueError(f"trace shape differs for {key}: {x.shape} vs {y.shape}")
                step_name, component = key.split(".", 1)
                row = dict(case=case, step=int(step_name.removeprefix("step_")), component=component)
                if key.endswith("router_indices"):
                    matches = (x.sort(-1).values == y.sort(-1).values).all(-1)
                    changed = (~matches).nonzero().flatten().tolist()
                    row.update(changed_tokens=changed, token_count=x.shape[0],
                               reference_last=x[-1].tolist(), actual_last=y[-1].tolist())
                    if changed:
                        logits_key = key.removesuffix("indices") + "logits"
                        logits = hf.get_tensor(logits_key).float()
                        scores = logits.topk(x.shape[-1] + 1, dim=-1).values
                        row["reference_boundary_gaps"] = (scores[changed, -2] - scores[changed, -1]).tolist()
                else:
                    row.update(all_tokens=differences(x, y), last_token=differences(x[-1], y[-1]))
                results.append(row)
    if not results:
        raise ValueError("no activation traces found")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = compare(args.directory)
    destination = args.directory / "activation_comparison.json"
    destination.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Compared {len(results)} activation snapshots; wrote {destination}")


if __name__ == "__main__":
    main()
