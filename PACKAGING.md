# Packaging and releases

## Install names

The engine is published as **`m-star`** on PyPI (the bare `mstar` name was
already taken). The import package and console scripts are unchanged:

```
pip install m-star
python -c "import mstar"
mstar --help
```

`mstar-ai` and `mstar-project` are alias packages (under
`packaging/aliases/`) that carry no code and just depend on `m-star`, so
`pip install mstar-ai` resolves to the same thing. They mirror m-star's
extras, so `pip install "mstar-ai[all]"` forwards to `m-star[all]`; keep
their `[project.optional-dependencies]` in sync when m-star's extras change.
PyPI treats `m-star`, `m_star`, `m.star`, and `M-Star` as one name, but
`mstar` (no separator) is a separate project.

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

Version lives in `pyproject.toml` (`[project] version`). To release:

1. Bump the version and land it on `main`.
2. Publish a GitHub Release with a matching tag (e.g. `v0.2.0`).
3. The `Publish to PyPI` workflow builds the sdist and wheel and uploads
   them over Trusted Publishing (OIDC — no token stored in the repo).

One-time PyPI setup: add a pending Trusted Publisher for project `m-star`
(owner `mstar-project`, repo `mstar`, workflow `release.yml`, environment
`pypi`).

## Local build / dry run

```
pip install build twine
python -m build                       # -> dist/m_star-<ver>.tar.gz + .whl
twine upload --repository testpypi dist/*     # optional TestPyPI dry run
```

The alias packages are built and published from their own directories, e.g.
`cd packaging/aliases/mstar-ai && python -m build`, and only need
re-publishing if their metadata changes.
