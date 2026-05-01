# Repository Guidelines

## Project Structure & Module Organization
`mminf/` is the main Python package. Core runtime pieces live in `mminf/api_server/` (FastAPI entrypoint), `mminf/conductor/`, `mminf/worker/`, `mminf/engine/`, `mminf/graph/`, `mminf/communication/`, and `mminf/streaming/`. Model implementations and loaders live under `mminf/model/` by family (`bagel/`, `qwen3_omni/`, `pi05/`, `vjepa2/`, etc.). Deployment configs are in `configs/`. Benchmarks live in `benchmark/`. Tests are under `test/`, split into `test/modular/` for unit-style coverage and `test/integration/` plus model-specific folders for end-to-end flows. Treat `ref/` as reference material, not primary code to extend.

## Build, Test, and Development Commands
Use Python 3.12.

- `pip install -e ".[dev]"`: install the package in editable mode with lint/test tools.
- `ruff check .`: run lint checks used in CI.
- `ruff format .`: apply the repo formatter.
- `pytest test/modular/`: run fast modular tests.
- `pytest test/integration/`: run GPU- and weights-dependent integration tests.
- `mminf-serve --config configs/<model>.yaml --host 0.0.0.0 --port 8000`: start the server.
- `bash benchmark/run_benchmark.sh`: run the benchmark wrapper against a running server.

## Coding Style & Naming Conventions
Follow Ruff defaults configured in `pyproject.toml`: 120-character lines, double quotes, Python 3.12 syntax. Use 4-space indentation. Keep modules, functions, and test files in `snake_case`; use `CamelCase` for classes. Match existing model package patterns: config in `config.py`, orchestration in `*_model.py`, shared layers in `submodules.py` or `components/`.

## Testing Guidelines
Add modular tests next to the subsystem they cover and name files `test_<feature>.py`. Reserve integration tests for flows that require CUDA, extra packages, or downloaded weights; gate those with `pytest.mark.skipif` as existing tests do. Before opening a PR, run `ruff check .` and the narrowest relevant `pytest` target.

## Commit & Pull Request Guidelines
Recent commits use concise, scoped subjects such as `worker: ...` or `cache_manager: ...`. Follow that pattern: `<area>: imperative summary`. PRs should explain the behavioral change, list configs or model paths touched, mention any required weights/hardware, and include logs or screenshots only when API behavior or profiling output changes.

## Security & Configuration Tips
Do not commit `.env`, generated outputs, or profiling artifacts. Keep local experiments in `test/scratch/` or ignored files, and avoid editing `nsys_profiles/` unless the change is explicitly about profiling data.
