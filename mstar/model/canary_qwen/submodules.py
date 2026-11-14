"""The LLM node of Canary-Qwen: dense Qwen3 on the shared `SampledTokenDecoder`,
the whole prompt prefilled as one row with the encoder's audio frames spliced
in at its placeholders. The encoder node is the shared
``FastConformerEncoderSubmodule`` with the projection into the LLM's width."""
from __future__ import annotations

from mstar.model.components.sampled_token_decoder import DeclaredTokenDecoder


class CanaryQwenLLMSubmodule(DeclaredTokenDecoder):
    # off: with torch.compile, Canary-Qwen's throughput was 7-15% lower at c=1-64 (H100 A/B)
    disable_torch_compile = True
    # the ladder it captured before moving onto the scaffold
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    # decode only, as before the scaffold: a prefill's length follows its clip
    CAPTURE_PREFILL = False
    # Prefill isn't captured; this caps its batch, so one prefill's
    # reservation stays well inside the cache.
    PREFILL_MEDIA_CAPTURE_BATCH_SIZES = [16]
    # decode reads its ids back from the routed token and emits a tensor a
    # request, as before the scaffold; the device path is a measured change
    device_loopback = None
