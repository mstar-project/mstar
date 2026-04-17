"""Code2Wav vocoder wrapper for Qwen3-Omni.

Per the design doc, we REUSE HuggingFace's ``Qwen3OmniMoeCode2Wav`` class
directly rather than reimplementing it.  Code2Wav is a complex streaming
ConvNet vocoder (SnakeBeta activation, CausalTransConvNet, ConvNeXtBlock,
etc.) and is not performance-critical enough to justify a custom
implementation for our system.

This mirrors sglang-omni's approach (which also reuses HF directly).
vllm-omni reimplements a wrapper class, but only because vLLM's architecture
registry requires top-level class registration — we have no such constraint
in mminf.

The wrapper factory returns an HF-loaded instance that Code2WavSubmodule
can call via its ``forward()`` / ``chunked_decode()`` methods.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def load_code2wav_model(
    local_dir: str,
    device: str | torch.device = "cuda",
) -> torch.nn.Module:
    """Load HF's ``Qwen3OmniMoeCode2Wav`` from a local checkpoint directory.

    Follows sglang-omni's pattern:
      1. Load the top-level ``Qwen3OmniMoeConfig`` via ``AutoConfig``.
      2. Extract ``code2wav_config``.
      3. Instantiate HF's ``Qwen3OmniMoeCode2Wav`` via ``_from_config``.
      4. Load weights from the checkpoint shards using the ``code2wav.`` prefix.

    Args:
        local_dir: Path to the Qwen3-Omni HF checkpoint directory.
        device: Target device for the model weights.

    Returns:
        An instance of ``Qwen3OmniMoeCode2Wav`` in eval mode on ``device``.
    """
    from transformers import AutoConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeCode2Wav,
    )

    from mminf.model.utils import ModuleAndPrefix, load_weights_from_hf_shards

    config = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)
    code2wav_config = getattr(config, "code2wav_config", None)
    if code2wav_config is None:
        raise ValueError(
            f"No code2wav_config found in {local_dir}. "
            "The checkpoint does not appear to be a Qwen3-Omni model "
            "with audio output enabled."
        )

    with torch.device("meta"):
        model = Qwen3OmniMoeCode2Wav._from_config(code2wav_config)

    load_weights_from_hf_shards(
        repo_dir=local_dir,
        modules=[ModuleAndPrefix(model, prefix="code2wav")],
        device=device,
    )
    model.eval()
    return model


class Qwen3OmniCode2Wav(torch.nn.Module):
    """Thin wrapper around HF's ``Qwen3OmniMoeCode2Wav``.

    This class exists purely so that callers (e.g., ``Qwen3OmniModel._create_code2wav_submodule``)
    can construct the vocoder uniformly without knowing about HF's config
    nesting.  The forward pass delegates directly to the HF model.

    Usage::

        model = Qwen3OmniCode2Wav(config)   # meta-device, unloaded
        load_weights_from_hf_shards(...)    # caller loads weights

    or (preferred) use the module-level ``load_code2wav_model()`` helper
    which handles config extraction and weight loading in one call.
    """

    def __init__(self, hf_cfg, hf_model):
        super().__init__()
        self._hf_model = hf_model
        self.config = hf_cfg

    def forward(self, codes: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._hf_model(codes, **kwargs)

    def chunked_decode(
        self,
        codes: torch.Tensor,
        chunk_size: int = 300,
        left_context_size: int = 25,
    ) -> torch.Tensor:
        return self._hf_model.chunked_decode(
            codes, chunk_size=chunk_size, left_context_size=left_context_size
        )
