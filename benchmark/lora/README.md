# LoRA checks on a GPU

Nothing here runs in CI.

## 1. Kernels, eagerly and inside one captured graph (minutes)

```
pytest test/integration/test_lora_triton_gpu.py -v
```

* `test_kernels_match_the_reference[...]`: `lora_apply_` (split-K, ranks 16 to 512
  including tiled ranks past 64, token tails from 1 to 1500) and the embedding's
  `lora_gather_shrink` + `lora_expand_` (ids off the vocabulary) against the fp32
  reference ops, with a rank-0 slot and base rows left untouched.
* `test_a_tiled_rank_inside_a_captured_graph`: `lora_apply_` at max rank 128
  captured once and replayed for changing slot mixes and ranks, bit for bit against
  eager.

The CPU tests (no GPU):

```
pytest test/modular/test_lora_pool.py test/modular/test_lora_tp_lockstep.py \
    test/modular/test_lora_api.py test/modular/test_lora_graphs.py \
    test/modular/test_lora_prefix_keys.py test/modular/test_lora_targets.py -v
```

## Behaviour changes for non-LoRA deployments

The LoRA stack changes three things a deployment without a `resources: lora:` block can
see. Each is deliberate:

* **`autocast_dtype` is parsed.** A string in the serving config (`float16`,
  `torch.bfloat16`) becomes the `torch.dtype` it names before any node is built; a value
  that names no dtype fails at startup instead of reaching the engine as a string.
  `autocast_dtype: null` is passed through as None, as before.
* **The conductor validates requests against the deployment's specs.** It resolves
  resource specs as the workers do (`serving_resources`: the model's declaration with the
  YAML's `resources:` overrides applied), where it used to validate against the model's
  defaults. A request is now checked against the limits the workers actually built.
* **`lora_adapter` on a deployment without LoRA is a 400.** A request naming an adapter
  (`model_kwargs: {lora_adapter: ...}` on `/generate`) is refused rather than run on the
  base model, which would answer it with weights it did not ask for. The OpenAI `model`
  field is unaffected: without LoRA it is not read, as before.

No cost without a `lora:` block (`non_lora_ab`): Qwen3.5-0.8B, Orpheus and Kokoro
served from this branch and from a base, alternating per round, each run with the
worker phase tables:

```
python -m benchmark.lora.non_lora_ab --phase setup --work ~/ab --base 44ab46571 --branch lora/generic
python -m benchmark.lora.non_lora_ab --phase run --work ~/ab --out ~/ab/results --rounds 3
python -m benchmark.lora.non_lora_ab --phase compare --out ~/ab/results
```
