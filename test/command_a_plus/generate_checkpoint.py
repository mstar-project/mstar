"""Free-running text smoke test using a local checkpoint and real M* resources.

This tests generation separately from numerical parity. It does not override
the parity runner's acceptance limits or claim HTTP/serving validation.
"""

import argparse
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from mstar.distributed.communication import CommGroup
from test.command_a_plus.validate_checkpoint import PROMPTS, Runtime, write_json


@torch.inference_mode()
def generate(args):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    group = CommGroup(dist.get_rank(), dist.get_rank(), list(range(dist.get_world_size())))
    group.device_group = dist.group.WORLD
    runtime = None
    try:
        runtime = Runtime(args, group)
        cases = []
        tokenizer = runtime.model._get_tokenizer()
        for index, prompt in enumerate(PROMPTS):
            ids = runtime.model.process_prompt(prompt, ["text"], ["text"])["text_inputs"][0]
            if len(ids) + args.tokens > args.context:
                raise ValueError("prompt plus generation budget exceeds context")
            rid = f"text_smoke_{index}"
            runtime.start(rid)
            generated = []
            ended = False
            try:
                rows = [ids.cuda()]
                for step in range(args.tokens):
                    logits, tokens, _ = runtime.step([rid], rows, step)
                    if not torch.isfinite(logits).all():
                        raise AssertionError("generation produced non-finite logits")
                    token = tokens.item()
                    generated.append(token)
                    rows = [tokens]
                    if token == runtime.model.config.eos_token_id:
                        ended = True
                        break
            finally:
                runtime.remove(rid)
                runtime.check_empty()
            text = tokenizer.decode(generated, skip_special_tokens=False)
            case = dict(prompt=prompt, generated_ids=generated, text=text,
                        generated_tokens=len(generated), stopped_on_eos=ended)
            cases.append(case)
            if rank == 0:
                print(f"Case {index}: {len(generated)} tokens; EOS={ended}; {text!r}", flush=True)
                write_json(args.output, dict(tp=group.world_size, checkpoint=str(args.checkpoint.resolve()),
                                            load_seconds=runtime.load_seconds, cases=cases))
    finally:
        if runtime is not None:
            runtime.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--context", type=int, default=8192)
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("tokens must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate(args)


if __name__ == "__main__":
    main()
