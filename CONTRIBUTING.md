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

### Automated review

A bot posts an advisory review when a contributor (the team, or anyone who has had a PR merged here) opens a non-draft PR, from a fork or not. It checks the invariants in [AGENTS.md](AGENTS.md) (the model/engine boundary, resource declarations, rank lockstep, Python/Rust parity, documented environment variables) and cites them by number. It is not a merge gate and a maintainer still reviews your PR. It doesn't rerun on every push: a contributor can comment `/review` on their own PR for a re-review, and someone with write access can do so on any PR, or comment `/review thorough` for the thorough mode. Reply to disagree; apply the `skip-ai-review` label to turn the automatic review off for a PR. Details in [.github/review/](.github/review/).

### AI-assisted contributions

These are welcome, with two conditions: read [AGENTS.md](AGENTS.md) first (point your agent at it, or symlink it to whatever filename your tooling expects), and say in the PR description that AI assistance was used, which tests you ran, and what the results were. You need to understand and be able to defend every line you submit.

### Optional Mac checks

On **Apple Silicon macOS**, you can check your current worktree before
committing or pushing:

```bash
./scripts/ci/mac.sh all
```

This is opt-in: no Git hooks are installed and running it is not a contribution
requirement. It uses native `linux/arm64` Docker containers. **Passing these Mac
checks does not establish that the full x86 CPU suite or GPU tests pass.** GitHub
continues to run the full CPU manifest on its Ubuntu runners.

#### Setup

You need Git, Python 3, and an ARM Docker engine, such as Docker Desktop or
Docker with Colima on your Mac. Remote x86 Docker engines are unsupported.
Colima is not an mstar dependency. Allocate at least **8 GiB
of VM memory** for the Rust build; validation used 4 CPUs and 8 GiB. Rust release
linking can exhaust a 2 GiB VM even with one compilation job.

The runner uses your current Docker context. You can select another configured
context with `DOCKER_CONTEXT=your-context ./scripts/ci/mac.sh all`. It does not
create, resize, or reconfigure your VM. The initial run downloads images,
Python packages and Rust dependencies; no GPU or model weights are needed.

This first version supports Apple Silicon macOS with rootful Docker, without
user-namespace remapping. Intel Macs, other host operating systems and x86
emulation are outside its supported scope. In particular, setting
`CI_PLATFORM=linux/amd64` is rejected.

#### Checks and coverage

| Command | Checks |
| --- | --- |
| `./scripts/ci/mac.sh lint` | Ruff and runner regression tests |
| `./scripts/ci/mac.sh cpu-smoke` | ARM-compatible subset of the CPU core manifest |
| `./scripts/ci/mac.sh dynamo` | Dynamo bridges and registration |
| `./scripts/ci/mac.sh rust` | Native Rust tests, server/extension builds and Python integration tests |

`all` runs these four checks sequentially and stops at the first failure.
`cpu-smoke` reads [test/cpu-core.txt](test/cpu-core.txt), excluding only the
modules listed in [test/cpu-mac-exclude.txt](test/cpu-mac-exclude.txt):

- `test_waypoint_components.py`
- `test_waypoint_dit.py`
- `test_flex_attention_resource.py`
- `test_waypoint_benchmark_videos.py`

The first three invoke compiled FlexAttention, which requires AVX2 with the
pinned PyTorch 2.9.1 CPU implementation. The video test hits an illegal
instruction in PyAV 19.0.1 encoding in the tested Apple VM; this also reproduces
without mstar. The entire modules are excluded,
including their portable structural cases. Currently the Mac subset selects
49 of the 53 core modules. It prints the exclusions and saves the exact
selection in `cpu-smoke-tests.txt` with the reports. New core modules are
included automatically unless explicitly excluded; invalid or stale exclusions
fail the check. Existing GPU/optional-feature skips are reported by pytest.
No test assertions are changed for the Mac runner.

CPU smoke also disables PyTorch's oneDNN backend in its test process: the
pinned ARM wheel crashes on a standalone BF16 Linear operation in the tested
Apple VM. BF16 tests still execute through PyTorch's CPU fallback with their
original assertions. These checks do not cover optimized oneDNN kernels.
The setting does not affect GitHub or other jobs.

The local installation and check scripts follow the existing GitHub jobs, with
Python 3.12 and a separate environment for each check. The local image uses
Debian ARM while GitHub uses Ubuntu x86. Some dependencies and Rust stable are
unpinned; resolved packages and tool versions are recorded per run. When
updating CI dependencies, update `scripts/ci/install.sh` and
`scripts/ci/cpu-torch.txt` alongside `.github/workflows/ci.yml`.

#### Source, reports and caches

The snapshot includes current tracked files and eligible untracked source,
including uncommitted edits. Untracked ignored files, local environments and
build outputs are excluded; tracked files are retained. Tests run on copies
inside disposable containers and do not modify your checkout. This checks the
working tree, rather than only staged changes or GitHub's PR merge commit.

Logs, timings and pytest reports go to `.ci-results/`. Download and Rust build
caches use Docker volumes named `mstar-ci-*`. Images and caches remain for later
runs; containers and temporary snapshots are removed. Ctrl-C also removes the
active container and retains its logs. There is no automatic local job timeout.
Rust defaults to two parallel compilation jobs; use `CI_BUILD_JOBS=1` to reduce
parallelism when needed.

The native Rust server links against the build interpreter's `libpython`.
When launching the binary separately, make that matching library available to
the dynamic loader. The local Rust check configures `LD_LIBRARY_PATH` for it.

### CPU core tests on GitHub

The **CPU Core** job runs the full [test/cpu-core.txt](test/cpu-core.txt) manifest
on a GitHub-hosted Ubuntu runner whenever a PR targeting `main` is opened,
updated or reopened. It needs no GPU or model weights; mixed modules skip their
GPU cases explicitly. New updates cancel an older CPU Core run for the same PR.

The installation and invocation commands are in
[the CI workflow](.github/workflows/ci.yml). To add a CPU-compatible module,
add its repository-relative path to the manifest after validating it without
a GPU or model downloads. New cases within a listed module are automatically
included. Check ARM compatibility before extending the Mac exclusion list.
The pytest `@file` syntax requires pytest 8.2 or newer, installed by the `dev`
extra.

Keep the selection explicit: other modules under `test/modular/` require GPU
backends during collection. `test_ragged_attention_cpu.py` contains padding
validation; GPU attention and capture tests remain in `test_ragged_attention.py`.
CI prints the slowest tests and retains JUnit reports for seven days. Making
**CPU Core** a required merge check is a separate repository ruleset or branch
protection setting.

## Adding a model

Start with the [Adding a New Model](https://mstar-project.github.io/mstar/adding_models.html)
guide — it walks through the component graph, Walks, and the submodule pattern.

By contributing, you agree that your contributions are licensed under the repository's
[Apache-2.0 license](LICENSE).
