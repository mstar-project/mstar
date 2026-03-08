import torch


def run_rms_norm(
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-06
    ):
    # TODO: this should maybe not be in CacheHandle, but it might make
    # sense to still be defined on the engine level so that we can easily
    # swap out flashinfer for anything else
    import flashinfer
    return flashinfer.norm.rmsnorm(
        input, weight, eps=eps
    )
