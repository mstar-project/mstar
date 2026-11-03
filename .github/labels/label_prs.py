#!/usr/bin/env python3
"""Label PRs by the areas they touch (area/*, model/*) and their size (size/*).

Run by .github/workflows/pr-labels.yml with the default branch checked out;
reads only the PR's file list through the API (for a stacked PR, its diff
against the PR it is stacked on), never its code. Locally:

    python3 .github/labels/label_prs.py --repo mstar-project/mstar --all --dry-run

Only labels under the managed prefixes are added or removed; anything else on
the PR (type/*, set by the review bot, or a human's labels) is left alone.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from urllib.parse import quote

MANAGED = ("area/", "model/", "size/")

# First matching prefix wins; a file can have more than one area. mstar/model/<name>/
# and configs/ are handled in areas_of(), since their label depends on the
# model name. A file nothing matches is area/other, which shows a rule is missing.
AREAS = [
    # The layers follow AGENTS.md's layer map and the PR title tags in use.
    ("mstar/__init__.py", "area/api"),
    ("mstar/api_server/", "area/api"),
    ("mstar/client/", "area/api"),
    ("mstar/cli/", "area/api"),
    ("mstar/conductor/", "area/conductor"),
    ("mstar/graph/", "area/graph"),
    ("mstar/worker/worker_metrics.py", ("area/observability", "area/worker")),
    ("mstar/worker/", "area/worker"),
    ("mstar/streaming/", "area/worker"),
    ("mstar/communication/", "area/transport"),
    ("mstar/engine/resources/", "area/resources"),
    ("mstar/engine/", "area/engine"),
    ("mstar/platforms/", "area/engine"),
    ("mstar/distributed/", "area/distributed"),
    ("mstar/observability/", "area/observability"),
    ("mstar/profile/", "area/observability"),
    ("mstar/validation/", "area/conductor"),
    # GPU kernels models build on.
    ("mstar/utils/adarms_norm.py", "area/model-shared"),
    ("mstar/utils/attention.py", "area/model-shared"),
    ("mstar/utils/causal_conv1d/", "area/model-shared"),
    ("mstar/utils/fused_moe/", "area/model-shared"),
    # The messages to and from workers.
    ("mstar/utils/ipc_format.py", "area/worker"),
    # Helpers for the API server and the Rust frontend.
    ("mstar/utils/client_errors.py", "area/api"),
    ("mstar/utils/runtime_dir.py", "area/api"),
    ("mstar/utils/socket_prefix.py", "area/api"),
    ("mstar/utils/utf8_stream.py", "area/api"),
    ("mstar/utils/profiler.py", "area/observability"),
    # The rest are shared by several layers, so they name none of them.
    ("mstar/utils/", "area/utils"),
    # The model API (base.py, registry.py, ...) and components/ and loader/.
    ("mstar/model/", "area/model-shared"),
    ("mstar/integrations/", "area/integrations"),
    ("mstar/default_configs/", "area/infra"),
    # The Rust frontend, graph runtime and transport mirror their Python
    # counterparts and change with them (AGENTS.md invariant 5).
    ("rust/server/Cargo.lock", "area/infra"),
    ("rust/server/", ("area/rust", "area/api")),
    ("rust/src/graph/", ("area/rust", "area/graph")),
    ("rust/src/communicator.rs", ("area/rust", "area/transport")),
    ("rust/src/shm.rs", ("area/rust", "area/transport")),
    ("rust/src/tensors.rs", ("area/rust", "area/transport")),
    ("rust/Cargo.lock", "area/infra"),
    ("rust/", "area/rust"),
    ("benchmark/", "area/benchmark"),
    ("perf_testing/", "area/benchmark"),
    ("results/", "area/benchmark"),
    (".github/", "area/infra"),
    ("packaging/", "area/infra"),
    ("scripts/", "area/infra"),
    ("tools/", "area/infra"),
    ("docker/", "area/infra"),
    ("pyproject.toml", "area/infra"),
    ("setup.py", "area/infra"),
    ("MANIFEST.in", "area/infra"),
    ("uv.lock", "area/infra"),
    (".gitignore", "area/infra"),
    (".sample.env", "area/infra"),
    # Agent skills are docs, like AGENTS.md.
    (".claude/", "area/docs"),
    ("docs/", "area/docs"),
    ("examples/", "area/docs"),
    ("assets/", "area/docs"),
    ("LICENSE", "area/docs"),
    ("test/", "area/tests"),
]
# Lockfiles say nothing about review effort, so they don't count toward size.
NOT_SIZED = ("uv.lock", "rust/Cargo.lock", "rust/server/Cargo.lock")

# A layer (area/*) needs this many changed lines, so a one-line touch doesn't
# label a PR with every layer it brushed against. Any change to a model's code
# or config labels that model.
MIN_LINES = 10
# Files every model and request goes through: any change to one labels its area.
CORE_FILES = {
    "mstar/conductor/conductor.py",
    "mstar/engine/engine.py",
    "mstar/engine/resources/base.py",
    "mstar/engine/resources/runner.py",
    "mstar/graph/runtime/base.py",
    "mstar/model/base.py",
    "mstar/model/submodule_base.py",
    "mstar/worker/micro_scheduler.py",
    "mstar/worker/worker.py",
}
# A change across many models (an engine API every model implements, say) gets
# one model/many label rather than one per model.
MAX_MODELS = 3
SIZES = [(20, "size/XS"), (100, "size/S"), (700, "size/M"), (2000, "size/L"), (5000, "size/XL")]

COLORS = {"area/": "1d76db", "model/": "5319e7", "size/": "c5def5"}

# Registry entries, `"<key>": ("mstar.model.<folder>...` or
# `MODEL_REGISTRY["<key>"] = ("mstar.model.<folder>...`, so a config named after
# a registry key (qwen3_8_27b_tp4.yaml) finds its model folder.
REGISTRY_ENTRY = re.compile(r'"([\w.-]+)"\]?\s*[:=]\s*\(\s*"mstar\.model\.(\w+)\.')

# A stacked PR from a fork targets main, since it can't target its parent's
# branch, and says "Stacked on #N" before its body's first heading or toggle.
STACKED_ON = re.compile(r"\bstacked on #(\d+)", re.IGNORECASE)
BODY_SECTION = re.compile(r"^\s*(?:#{1,6}\s|<details)", re.MULTILINE)
# The compare API lists at most this many files.
COMPARE_MAX_FILES = 300


def gh(*args, data=None):
    for attempt in range(3):
        out = subprocess.run(["gh", *args], input=data, capture_output=True, text=True, check=False)
        if not out.returncode:
            return out.stdout
        time.sleep(2 ** attempt)
    raise RuntimeError(f"gh {' '.join(args)}: {out.stderr.strip()}")


def gh_pages(path):
    """All items of a paginated list endpoint."""
    out = gh("api", "--paginate", "--slurp", path)
    return [item for page in json.loads(out) for item in page]


def model_of_config(path, aliases):
    """The model folder a config belongs to, from its file name.

    Matched on whole underscore-separated words, ignoring the underscores, so
    qwen3omni_2gpu.yaml finds qwen3_omni. The longest match wins: qwen3_5_9b
    over a shorter name that is also a prefix.
    """
    words = os.path.basename(path).split(".")[0].split("_")
    for n in range(len(words), 0, -1):
        folder = aliases.get("".join(words[:n]))
        if folder:
            return folder
    return None


def areas_of(path, aliases):
    # Tests kept next to the code they test (mstar/model/<name>/tests/) are tests.
    if "/tests/" in path:
        return ("area/tests",)
    if path.startswith("mstar/model/"):
        parts = path.split("/")
        if len(parts) > 3 and parts[2] not in ("components", "loader"):
            return (f"model/{parts[2]}",)
    if path.startswith("configs/"):
        folder = model_of_config(path, aliases)
        return (f"model/{folder}" if folder else "area/configs",)
    for prefix, labels in AREAS:
        if path.startswith(prefix):
            return labels if isinstance(labels, tuple) else (labels,)
    if "/" not in path and path.endswith(".md"):
        return ("area/docs",)
    return ("area/other",)


def labels_for(files, aliases):
    lines = defaultdict(int)
    core = set()
    size_lines = 0
    for f in files:
        areas = areas_of(f["filename"], aliases)
        n = f["additions"] + f["deletions"]
        for area in areas:
            lines[area] += n
        if f["filename"] in CORE_FILES:
            core.update(areas)
        if not {"area/tests", "area/docs"} & set(areas) and f["filename"] not in NOT_SIZED:
            size_lines += n
    # Tests count toward no area, and docs only on a docs-only PR: nearly every
    # PR touches them. Test-only and docs-only PRs still get a label.
    code = {a: n for a, n in lines.items() if a not in ("area/tests", "area/docs")}
    if code:
        wanted = {a for a, n in code.items() if n >= MIN_LINES or a in core or a.startswith("model/")}
        if not wanted:
            wanted = {max(code, key=code.get)}
    elif "area/docs" in lines:
        wanted = {"area/docs"}
        size_lines = lines["area/docs"]
    elif "area/tests" in lines:
        wanted = {"area/tests"}
        size_lines = lines["area/tests"]
    else:
        wanted = set()
    models = {a for a in wanted if a.startswith("model/")}
    if len(models) > MAX_MODELS:
        wanted = (wanted - models) | {"model/many"}
    wanted.add(next((label for limit, label in SIZES if size_lines < limit), "size/XXL"))
    return wanted


def model_aliases(repo_root, files):
    """Model folder names and registry keys, without underscores, to the folder."""
    names = set()
    model_dir = os.path.join(repo_root, "mstar", "model")
    if os.path.isdir(model_dir):
        names |= {d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))}
    # A new-model PR adds a folder the base doesn't have yet.
    for f in files:
        parts = f["filename"].split("/")
        if len(parts) > 3 and parts[:2] == ["mstar", "model"]:
            names.add(parts[2])
    names -= {"components", "loader", "__pycache__"}
    aliases = {}
    try:
        with open(os.path.join(model_dir, "registry.py")) as fh:
            for key, folder in REGISTRY_ENTRY.findall(fh.read()):
                if folder in names:
                    aliases[key.replace("_", "")] = folder
    except OSError:
        pass
    aliases.update({name.replace("_", ""): name for name in names})
    return aliases


def stacked_on(body, number):
    """The PR named by "Stacked on #N" in the body's opening lines, or None."""
    lead = BODY_SECTION.split(body or "", maxsplit=1)[0]
    m = STACKED_ON.search(lead)
    return int(m.group(1)) if m and int(m.group(1)) != number else None


def stacked_files(repo, pr, files, open_prs):
    """The files a stacked PR's own commits change, or None to use its file list.

    Its file list against main carries its parent's changes too; its head
    against the parent's head leaves only its own. None when the PR isn't
    stacked, the parent is closed or merged, the compare fails or may be cut
    off, or the compare lists a file the PR doesn't change, as it does when the
    head isn't built on the parent's history (a wrong number, or a child
    rebased onto a newer main than its parent). A push to the parent doesn't
    relabel the child; the child's next push or a manual run does.

    open_prs maps number to PR for every open PR (--all), or is None to look
    the parent up.
    """
    parent = stacked_on(pr["body"], pr["number"])
    if parent is None:
        return None
    try:
        if open_prs is None:
            parent_pr = json.loads(gh("api", f"repos/{repo}/pulls/{parent}"))
        else:
            parent_pr = open_prs.get(parent)
        if not parent_pr or parent_pr["state"] != "open":
            return None
        # Only the first page of a compare lists the files; per_page only
        # pages the commits.
        cmp = json.loads(gh("api", f"repos/{repo}/compare/{parent_pr['head']['sha']}"
                                   f"...{pr['head']['sha']}?per_page=1"))
    except RuntimeError as e:
        print(f"#{pr['number']}: sized against main, not #{parent}: {e}", file=sys.stderr)
        return None
    own = cmp.get("files") or []
    if not own or len(own) >= COMPARE_MAX_FILES:
        return None
    if not {f["filename"] for f in own} <= {f["filename"] for f in files}:
        return None
    return own


def sync(repo, number, wanted, current, existing, dry_run):
    managed = {lb for lb in current if lb.startswith(MANAGED)}
    add, remove = sorted(wanted - managed), sorted(managed - wanted)
    print(f"#{number}: {' '.join(sorted(wanted))}"
          + (f"  (+{','.join(add)})" if add else "") + (f"  (-{','.join(remove)})" if remove else ""))
    if dry_run:
        return
    for label in add:
        if label not in existing:
            color = next(c for p, c in COLORS.items() if label.startswith(p))
            gh("api", "-X", "POST", f"repos/{repo}/labels", "-f", f"name={label}", "-f", f"color={color}")
            existing.add(label)
    if add:
        gh("api", "-X", "POST", f"repos/{repo}/issues/{number}/labels",
           "--input", "-", data=json.dumps({"labels": add}))
    for label in remove:
        gh("api", "-X", "DELETE", f"repos/{repo}/issues/{number}/labels/{quote(label, safe='')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("prs", nargs="*", type=int)
    ap.add_argument("--all", action="store_true", help="every open PR")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", default=".", help="checkout of the default branch, for the model list")
    a = ap.parse_args()

    if a.all:
        pulls = gh_pages(f"repos/{a.repo}/pulls?state=open&per_page=100")
        open_prs = {p["number"]: p for p in pulls}
    else:
        pulls = [json.loads(gh("api", f"repos/{a.repo}/pulls/{n}")) for n in a.prs]
        open_prs = None
    existing = set() if a.dry_run else {lb["name"] for lb in gh_pages(f"repos/{a.repo}/labels?per_page=100")}
    failed = 0
    for pr in pulls:
        number = pr["number"]
        try:
            files = gh_pages(f"repos/{a.repo}/pulls/{number}/files?per_page=100")
            own = stacked_files(a.repo, pr, files, open_prs) or files
            sync(a.repo, number, labels_for(own, model_aliases(a.root, files)),
                 {lb["name"] for lb in pr["labels"]}, existing, a.dry_run)
        except RuntimeError as e:
            print(f"#{number}: {e}", file=sys.stderr)
            failed += 1
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
