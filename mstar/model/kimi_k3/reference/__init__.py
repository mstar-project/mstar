"""Pure-PyTorch reference implementations of the Kimi K3 building blocks.

These are the numerical ground truth for the serving kernels (tests compare every fast
path against them) and double as the eager fallback where no kernel applies. They
follow ``KIMI_K3_ARCH_SPEC.md`` and the Hugging Face modeling code line by line, with
the same dtype discipline (fp32 math where the reference computes in fp32).
"""
