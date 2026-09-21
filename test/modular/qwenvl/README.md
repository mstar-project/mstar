# Qwen3-VL-30B-A3B platform scenarios

These tests are organized around the capabilities an inference platform must
provide, rather than the private helper that happens to implement them.

| Scenario | What a failure means |
| --- | --- |
| `test_checkpoint_onboarding.py` | The published Qwen3-VL MoE config cannot be safely admitted, or interleaved MRoPE semantics drift. |
| `test_checkpoint_loading.py` | A 30B-A3B checkpoint can be partially or incorrectly mapped into the serving graph. |
| `test_image_chat_serving.py` | An API image-plus-text chat request cannot be processed, routed through graph walks, or deployed with the declared TP topology. |
| `test_continuous_batch_serving.py` | Packed LLM preprocessing, cache positions, last-token sampling, and decode state preserve per-request boundaries. This is a component contract, not proof that different graph walks co-schedule. |
| `test_reference_compatibility.py` | Tiny MoE text and DeepStack image execution no longer match Hugging Face's Qwen3-VL reference. |

Run the platform contract locally:

```bash
uv run --extra qwenvl --extra dev --with pytest-cov \
  pytest -q test/modular/qwenvl \
  --cov=mstar.model.qwenvl --cov-report=term-missing
```

The suite is intentionally CPU-sized. It proves graph, loading, packed-LLM,
and reference contracts. It does not prove end-to-end scheduler co-batching
between `prefill` and `prefill_vision`, actual 30B checkpoint admission,
single-GPU CUDA serving, TP2 numerical parity, or throughput.
