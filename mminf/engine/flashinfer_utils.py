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


def run_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float=1.0,
    causal: bool=True,
):
    import flashinfer
    return flashinfer.ops.attention(
        q,
        k,
        v,
        causal=causal,
        sm_scale=scale,
    )
