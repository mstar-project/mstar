import torch

# torch._dynamo ConfigModule stores user overrides in a ContextVar (thread-local
# as of torch 2.13+). Import-time assignments only affect the importing thread;
# the dedicated GPU executor thread that actually compiles never sees them.
# Call apply_dynamo_config() on any thread that may trigger dynamo tracing.
RECOMPILE_LIMIT = 84


def apply_dynamo_config() -> None:
    """Apply dynamo settings to the *calling* thread.

    torch's ConfigModule keeps user overrides in a ContextVar, so these are
    thread-local — any thread that may trigger a compile must call this.
    """
    torch._dynamo.config.recompile_limit = RECOMPILE_LIMIT
    torch._dynamo.config.allow_unspec_int_on_nn_module = True
    torch._dynamo.config.specialize_int = False


apply_dynamo_config()
torch.set_float32_matmul_precision("high")
