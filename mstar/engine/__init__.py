import os

import torch


def recompile_limit() -> int:
    return min(max(int(os.environ.get("MSTAR_RECOMPILE_LIMIT") or 84), 8), 256)


def apply_torch_config() -> None:
    # dynamo config is per-thread since torch 2.12, so every compiling thread calls this
    torch._dynamo.config.recompile_limit = recompile_limit()
    torch._dynamo.config.allow_unspec_int_on_nn_module = True
    torch._dynamo.config.specialize_int = False
    torch.set_float32_matmul_precision('high')


apply_torch_config()
