#!/usr/bin/env bash
# Optional checks for Apple Silicon macOS using native ARM Linux containers.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/ci/mac.sh [all|lint|cpu-smoke|dynamo|rust]

Requires Apple Silicon macOS, Docker, Git, and Python 3. Tests the current
worktree in native ARM Linux containers, including uncommitted and untracked
source files. Reports are saved in .ci-results/. No Git hooks are installed.

Mac checks; full x86 CPU suite not run. The cpu-smoke selection is smaller
than GitHub's CPU Core job. GitHub Actions remains the full CI gate.

Optional environment:
  CI_BUILD_JOBS   Parallel Rust compilation jobs (default: 2)
EOF
}

job=${1:-all}
if [[ $# -gt 1 ]]; then
    usage >&2
    exit 2
fi
case "$job" in
    -h|--help) usage; exit 0 ;;
    all) jobs=(lint cpu-smoke dynamo rust) ;;
    lint|cpu-smoke|dynamo|rust) jobs=("$job") ;;
    *) usage >&2; exit 2 ;;
esac

if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
    echo "This runner supports Apple Silicon macOS only." >&2
    exit 2
fi
if [[ ${CI_PLATFORM:-linux/arm64} != linux/arm64 ]]; then
    echo "CI_PLATFORM must be linux/arm64 for Mac checks; x86 emulation is unsupported." >&2
    exit 2
fi
platform=linux/arm64
architecture=arm64
build_jobs=${CI_BUILD_JOBS:-2}
if [[ ! "$build_jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "CI_BUILD_JOBS must be a positive integer" >&2
    exit 2
fi
for command in docker git python3; do
    command -v "$command" >/dev/null || {
        echo "Required command not found: $command" >&2
        exit 1
    }
done
security_options=$(docker info --format '{{json .SecurityOptions}}')
if [[ $security_options == *'"name=rootless"'* || $security_options == *'"name=userns"'* ]]; then
    echo "Mac checks currently require rootful Docker without user-namespace remapping." >&2
    echo "Rootless Docker and userns-remap are not supported because report ownership differs." >&2
    exit 1
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$repo_root"
mkdir -p .ci-results
reports=$(mktemp -d "$repo_root/.ci-results/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
# Keep the bind mount beneath the checkout: macOS VM runtimes commonly share
# the home directory, but not the host's per-user temporary directory.
temporary=$(mktemp -d "$repo_root/.ci-results/source-XXXXXX")
active_pid=
container_name=
cleanup() {
    if [[ -n "$container_name" ]]; then
        docker rm --force "$container_name" >/dev/null 2>&1 || true
    fi
    if [[ -n "$active_pid" ]]; then
        kill "$active_pid" 2>/dev/null || true
        wait "$active_pid" 2>/dev/null || true
    fi
    rm -rf -- "$temporary"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python3 scripts/ci/snapshot.py "$repo_root" "$temporary/source.tar"
echo 'Mac checks; full x86 CPU suite not run.'
printf 'Platform: %s\nReports: %s\n' "$platform" "$reports"
printf 'Platform: %s\nBase commit: %s\n' "$platform" "$(git rev-parse HEAD)" > "$reports/summary.txt"
echo 'Mac checks; full x86 CPU suite not run.' >> "$reports/summary.txt"

for job in "${jobs[@]}"; do
    target=python
    if [[ "$job" == rust ]]; then
        target=rust
    fi
    image="mstar-local-ci-${target}:${architecture}"
    job_reports="$reports/$job"
    mkdir -p "$job_reports"
    echo "=== $job ($platform) ==="
    started=$SECONDS
    docker build --platform "$platform" --target "$target" \
        --tag "$image" --file scripts/ci/Dockerfile scripts/ci \
        > "$job_reports/image-build.log" 2>&1 &
    active_pid=$!
    if ! wait "$active_pid"; then
        active_pid=
        cat "$job_reports/image-build.log" >&2
        echo "$job: image build failed" | tee -a "$reports/summary.txt"
        exit 1
    fi
    active_pid=
    mounts=(
        --mount "type=bind,source=$temporary/source.tar,target=/source.tar,readonly"
        --mount "type=bind,source=$job_reports,target=/reports"
        --mount "type=volume,source=mstar-ci-${architecture}-py312-pip,target=/root/.cache/pip"
    )
    if [[ "$job" == rust ]]; then
        mounts+=(
            --mount "type=volume,source=mstar-ci-${architecture}-cargo-registry,target=/opt/cargo/registry"
            --mount "type=volume,source=mstar-ci-${architecture}-cargo-git,target=/opt/cargo/git"
            --mount "type=volume,source=mstar-ci-${architecture}-py312-rust-target,target=/work/rust/target"
            --mount "type=volume,source=mstar-ci-${architecture}-py312-server-target,target=/work/rust/server/target"
        )
    fi
    # A known unique name lets cancellation remove this job even if the Docker
    # client disconnects. Waiting on background commands makes signal traps prompt.
    container_name="mstar-ci-${temporary##*/}-$job"
    docker run --rm --init --name "$container_name" --platform "$platform" --shm-size=256m \
        "${mounts[@]}" \
        --env "CARGO_BUILD_JOBS=$build_jobs" \
        --env CI_REPORT_DIR=/reports \
        --env "REPORT_UID=$(id -u)" --env "REPORT_GID=$(id -g)" \
        --workdir /work "$image" bash -euo pipefail -c '
            finish() {
                python -m pip list --format=columns > /reports/packages.txt || true
                chown -R "$REPORT_UID:$REPORT_GID" /reports
            }
            trap finish EXIT
            # Cargo target caches are shared across runs. Fresh source mtimes
            # prevent an older checkout from reusing a newer cached binary.
            tar -mxf /source.tar -C /work
            {
                uname -m
                python --version
                if command -v rustc >/dev/null; then rustc --version; fi
            } > /reports/toolchain.txt
            bash scripts/ci/install.sh "$1"
            bash scripts/ci/check.sh "$1"
        ' bash "$job" 2>&1 | tee "$job_reports/check.log" &
    active_pid=$!
    if wait "$active_pid"; then
        active_pid=
        container_name=
        echo "$job: passed ($((SECONDS - started))s)" | tee -a "$reports/summary.txt"
    else
        active_pid=
        echo "$job: FAILED ($((SECONDS - started))s); see $job_reports" | tee -a "$reports/summary.txt"
        exit 1
    fi
done
echo "All selected Mac checks passed; full x86 CPU suite not run. Reports: $reports"
