# Packaging and releases

## Install names

The engine is published as **`mstar-ai`** on PyPI. The bare `mstar` name is
taken by an unrelated project, and PyPI's similarity rule (separators are
dropped before names are compared) refuses `m-star` while that project
exists. The import package and console scripts are unchanged:

```
pip install mstar-ai
python -c "import mstar"
mstar --help
```

`mstar-project` and `mstar-serve` are alias packages (under
`packaging/aliases/`) that carry no code and just depend on `mstar-ai`, so
`pip install mstar-project` resolves to the same thing. They mirror mstar-ai's
extras, so `pip install "mstar-project[all]"` forwards to `mstar-ai[all]`. Keep
their `[project.optional-dependencies]` in sync when mstar-ai's extras change.
PyPI treats `mstar-ai`, `mstar_ai`, `mstar.ai` and `MSTAR-AI` as one name.

## Default configs

`configs/` stays the single source of truth at the repo root (unchanged for
checkouts). At build time `setup.py` copies `configs/*.yaml` into the
`mstar/default_configs/` package, so the wheel ships them under the `mstar`
namespace (not as a top-level `configs` package, which would clash with any
other distribution shipping one). `MANIFEST.in` grafts `configs/` into the
sdist so a wheel built from the sdist copies them too. Nothing is duplicated
in git. The CLI resolves via `importlib.resources.files("mstar.default_configs")`
for a pip install and falls back to the repo `configs/` for checkouts.

## Cutting a release

Version lives in `pyproject.toml` (`[project] version`), and the alias
packages carry the same version in `packaging/aliases/*/pyproject.toml`. To
release:

1. Bump the version in all three pyproject files and land it on `main`.
2. Publish a GitHub Release with a matching tag (e.g. `v0.2.1`). The workflow
   refuses a tag that does not match the version.
3. The `Publish to PyPI` workflow builds the engine and the alias packages and
   uploads them over Trusted Publishing (OIDC, no token stored in the repo).
   The `pypi` environment asks one of its reviewers to approve the run first.

One-time PyPI setup: add a pending Trusted Publisher for each project
(`mstar-ai`, `mstar-project` and `mstar-serve`, all with owner `mstar-project`,
repo `mstar`, workflow `release.yml`, environment `pypi`). PyPI activates one
pending publisher per upload step, so on the very first release a later upload
step can fail as not authorized. Re-running the failed job finishes it, the
uploads skip files that already exist.

## Local build / dry run

```
pip install build twine
python -m build                                        # -> dist/mstar_ai-<ver>.tar.gz + .whl
python -m build --outdir dist packaging/aliases/mstar-project
python -m build --outdir dist packaging/aliases/mstar-serve
twine check dist/*
twine upload --repository testpypi dist/*              # optional TestPyPI dry run
```

The alias packages are published by the release workflow together with the
engine, so they need no separate upload.
