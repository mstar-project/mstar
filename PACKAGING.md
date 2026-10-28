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

## Rust extension and frontend in the Linux wheels

Besides the sdist and a pure-Python `py3-none-any` wheel, each release builds
a manylinux wheel per CPython version (3.12, 3.13, 3.14) and architecture
(x86_64, aarch64) with cibuildwheel. Those wheels also carry the `mstar_rust`
extension (`rust/`) and the `mstar-server` HTTP frontend binary
(`rust/server/`, installed into the environment's `bin/`). `setup.py` compiles
both with setuptools-rust only when `CIBUILDWHEEL=1`; every other build,
including `pip install -e .`, is pure Python, and a checkout builds the Rust
parts with maturin and cargo (docs/installation.rst). pip picks the matching
manylinux wheel and falls back to the pure one elsewhere, which runs on the
Python transport and graph runtime.

- **Compiler.** `rust-toolchain.toml` at the repo root pins the Rust version
  for CI, the release wheels and local builds. Bump it there.
- **Third-party licenses.** Before building, the workflow runs
  [cargo-about](https://github.com/EmbarkStudios/cargo-about) with
  `rust/about.toml` over `rust/server` (whose dependency tree contains
  `rust/`'s) and writes `LICENSES-THIRD-PARTY.txt`, which ships in each Rust
  wheel's `dist-info/licenses/`. A crate under a license not in `accepted`
  fails the `wheels` job. The C/C++ that crates vendor is declared there by
  hand (libzmq via zeromq-src, Oniguruma via onig_sys, esaxx via esaxx-rs);
  check those entries when bumping those crates.
- **Not published yet.** libzmq 4.3.4 (vendored by zeromq-src, statically
  linked into both binaries) is LGPL-3.0-or-later with its own static-linking
  exception, and that license is not in `accepted` until a maintainer signs off
  on distributing it. Until then the license step fails, so `wheels` fails and
  `pypi-wheels` is skipped; `pypi` still publishes the sdist, the pure wheel
  and the aliases.
- **Checks.** Each wheel is installed without dependencies into a clean venv
  outside the build container and must import `mstar_rust`, run
  `mstar-server --help` and list the license file. A failure there keeps the
  Rust wheels off PyPI.

## Cutting a release

Version lives in `pyproject.toml` (`[project] version`), and the alias
packages carry the same version in `packaging/aliases/*/pyproject.toml`. To
release:

1. Bump the version in all three pyproject files and land it on `main`.
2. Publish a GitHub Release with a matching tag (e.g. `v0.2.1`). The workflow
   refuses a tag that does not match the version.
3. The `Publish to PyPI` workflow builds the engine and the alias packages
   and uploads them over Trusted Publishing (OIDC, no token stored in the
   repo); the Linux Rust wheels above go up in a job of their own. The `pypi`
   environment asks one of its reviewers to approve each publishing job first.

One-time PyPI setup: each project needs a pending Trusted Publisher (owner
`mstar-project`, repo `mstar`, workflow `release.yml`, environment `pypi`).
PyPI allows only one pending publisher per configuration at a time and turns
it into a real one at the first upload, so the three projects are bootstrapped
in turn: add the publisher for `mstar-ai` and publish the release (the alias
steps fail as not authorized), add the one for `mstar-project` and re-run the
failed job, then the same for `mstar-serve`. The uploads skip files that
already exist, so re-runs are safe. Once all three projects exist, a single run
publishes all of them.

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

`python -m build` makes only the pure-Python wheel. The Linux Rust wheels need
cibuildwheel (Docker) with the workflow's `CIBW_*` settings; to try the
workflow without publishing, run its `wheels` job on a fork branch.
