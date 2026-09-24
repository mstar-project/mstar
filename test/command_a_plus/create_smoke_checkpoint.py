"""Create random tiny weights and a TP1 config for an HTTP serving smoke test.

The supplied metadata directory must already contain the official tokenizer.
No model weights are downloaded. Generated text has no semantic meaning.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file
from transformers import Cohere2MoeConfig, Cohere2MoeForCausalLM

from test.command_a_plus.test_integration import FIXTURE
from test.command_a_plus.test_reference import reference_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "model.safetensors").exists():
        parser.error("output directory already contains weights")
    raw = json.loads(FIXTURE.read_text())
    # Also identifies the synthetic config as a modern non-Mistral model to
    # Transformers' legacy tokenizer-regex detection.
    import transformers
    raw["transformers_version"] = transformers.__version__
    official = json.loads((args.metadata_dir / "config.json").read_text())["text_config"]
    raw["text_config"].update(hidden_size=64, head_dim=128, intermediate_size=64,
                              max_position_embeddings=512,
                              **{key: official[key] for key in (
                                  "vocab_size", "bos_token_id", "eos_token_id", "pad_token_id",
                              )})
    (args.output_dir / "config.json").write_text(json.dumps(raw, indent=2))
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"):
        shutil.copyfile(args.metadata_dir / name, args.output_dir / name)
    torch.manual_seed(123)
    model = Cohere2MoeForCausalLM(Cohere2MoeConfig(**raw["text_config"])).to(dtype=torch.bfloat16)
    save_file({name: tensor.clone().contiguous() for name, tensor in reference_weights(model)},
              args.output_dir / "model.safetensors")
    deployment = {
        "model": "command_a_plus", "max_seq_len": 512,
        "model_kwargs": {"checkpoint_dir": str(args.output_dir.resolve()), "max_seq_len": 512},
        "resources": {"kv_cache": {"max_num_pages": 64, "page_size": 16}},
        "node_groups": [{"node_names": ["LLM"], "ranks": [0], "tp_size": 1,
                         "graph_walks": ["prefill", "decode"]}],
    }
    path = args.output_dir / "deployment.yaml"
    path.write_text(yaml.safe_dump(deployment))
    print(path)


if __name__ == "__main__":
    main()
