# Contributing to M*

Thanks for your interest in M\*! Contributions of all kinds are welcome, and you don't
need to read a rulebook first.

## Issues

- **Bug?** Open a [bug report](https://github.com/mstar-project/mstar/issues/new?template=bug_report.yml).
- **Idea or feature?** Open a [feature request](https://github.com/mstar-project/mstar/issues/new?template=feature_request.yml).
- **Want a model supported?** Open a [new model request](https://github.com/mstar-project/mstar/issues/new?template=new_model.yml).

General questions are welcome too — a plain issue or an email to <atindra@cs.stanford.edu> is fine.

## Pull requests

1. Fork and create a branch.
2. Make a focused change.
3. Run `ruff check .` (CI enforces it).
4. Open a PR against `main`. It runs CI and gets a review before merging.

### CPU core tests

The **CPU Core** CI job runs the graph I/O, resource runner, admission failure,
micro-scheduler, and worker drain tests on a GitHub-hosted Ubuntu runner whenever
a PR targeting `main` is opened, updated, or reopened. It needs no GPU or model
weights. New updates cancel an older CPU Core run for the same PR.

To reproduce the job in a clean Linux environment with Python 3.12:

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m pip check
HF_HUB_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -q -ra --strict-markers --durations=20 \
  --junitxml=cpu-core.xml \
  test/modular/test_graph_io.py \
  test/modular/test_resource_runner.py \
  test/modular/test_admit_failure_handling.py \
  test/modular/test_micro_scheduler.py \
  test/modular/test_worker_drain.py
```

Keep the file list explicit: other modules under `test/modular/` require GPU
backends during collection. CI prints the slowest tests and saves the JUnit report
as the `cpu-core-results` artifact for seven days. Making **CPU Core** a required
merge check is a separate repository ruleset or branch protection setting.

## Adding a model

Start with the [Adding a New Model](https://mstar-project.github.io/mstar/adding_models.html)
guide — it walks through the component graph, Walks, and the submodule pattern.

By contributing, you agree that your contributions are licensed under the repository's
[Apache-2.0 license](LICENSE).
