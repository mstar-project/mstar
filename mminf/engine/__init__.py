import torch

torch._dynamo.config.recompile_limit = 64
torch._dynamo.config.allow_unspec_int_on_nn_module = True
torch._dynamo.config.specialize_int = False
torch.set_float32_matmul_precision('high')
