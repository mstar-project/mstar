# Registering a model

Five places, four of them easy to forget. A model can be entirely correct and still be unreachable because one of these is missing, and no unit test catches it. `test/modular/test_model_registration.py` covers the first, second and fourth.

## 1. `mstar/model/registry.py` — required

`MODEL_REGISTRY` maps the config-YAML `model:` string to a **`(module_path, class_name)` tuple**, not to an imported class. The module is imported lazily through `import_module`, which is what keeps `mstar.model.registry` cheap to import; a direct class import here would drag every model's dependencies into every process.

```python
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "your_model": ("mstar.model.your_model.your_model_model", "YourModel"),
}
```

Add to `HF_MODELS` too if weights come from Hugging Face:

```python
HF_MODELS: dict[str, dict] = {
    "your_model": {"model_path_hf": "org/your-model-id"},
}
```

Note that `docs/adding_models.rst` showed the wrong signature for a long time (`dict[str, type[Model]]` with a direct import). Trust the file.

## 2. `mstar/cli/main.py` — required

`DEFAULT_CONFIGS` maps the model name to its default config filename, relative to `configs/`. **Without an entry, `mstar serve <your_model>` exits with `error: unknown model` before anything loads**, however correct the model is.

```python
DEFAULT_CONFIGS: dict[str, str] = {
    "your_model": "your_model.yaml",
}
```

This namespace is a superset of `MODEL_REGISTRY`: a key here may be a deployment alias for a model registered under another name (`bagel_cfg_parallel` → the `bagel` model with a different config).

## 3. `mstar/cli/main.py` — the help snippets, cosmetic but user-facing

The same file builds the post-launch client example from a series of `if model in (...)` tuples — the plain `client.generate(...)` form, and an OpenAI-compatible block for models with OpenAI semantics. A model absent from all of them still serves, but prints no usage example. Add it to whichever tuples match its modalities.

## 4. `docs/models.rst` — required for a released model

One row in the registry-key table, marked `*(Beta)*` if that applies. This is the list users actually read.

## 5. `mstar/api_server/openai/adapters.py` — only if the model needs OpenAI routes

Subclass `OpenAIAdapter` and add it to `ADAPTER_REGISTRY`. This is genuinely optional: roughly half the registered models have an adapter, because it only makes sense where the model maps onto OpenAI semantics (`chat.completions`, `audio.speech`, `images.generate`). A model reachable only through `POST /generate` needs no adapter and should not get an empty one.

## Also

- `configs/<your_model>.yaml` with `node_groups` mapping node names to ranks. **You do not need to copy it into `mstar/default_configs/`** — `setup.py` copies `configs/` there at build time so `mstar serve` works from a pip install. That directory holds only a marker in a source checkout.
- A `pyproject.toml` optional-dependency extra, if the model pulls in deps the base install should not carry. Say in the PR description why.
