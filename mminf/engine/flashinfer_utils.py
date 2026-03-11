import torch


def run_rms_norm(
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-06
    ):
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
    return flashinfer.single_prefill_with_kv_cache(
        q,
        k,
        v,
        causal=causal,
        sm_scale=scale,
    )
