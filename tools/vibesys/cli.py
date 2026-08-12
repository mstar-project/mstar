"""``mstar-vibesys`` — drive VibeSys against mstar for a registered task type.

    mstar-vibesys <task-type> <instance> <action> [options]

Actions:
    show          Print the resolved spec and generated paths.
    build-image   Build the reproducible Docker eval image.
    build         Create the worktree seed + render the bundle + seed files.
    run           build (if needed) then exec the vibesys CLI.
    clean         Remove the worktree seed and generated bundle.

Invoke as ``python -m tools.vibesys ...`` from the mstar repo root, or via the
``mstar-vibesys`` console script once installed.
"""

from __future__ import annotations

import argparse
import shutil
import sys

from tools.vibesys.core import image as image_mod
from tools.vibesys.core import runner as runner_mod
from tools.vibesys.core import seed as seed_mod
from tools.vibesys.core.layout import default_layout
from tools.vibesys.tasks import REGISTRY


def _resolve(task_type: str, instance: str):
    if task_type not in REGISTRY:
        raise SystemExit(f"unknown task type {task_type!r}; known: {sorted(REGISTRY)}")
    task = REGISTRY[task_type]
    inst_path = task.instance_path(instance)
    if not inst_path.exists():
        raise SystemExit(f"no instance {instance!r} for {task_type!r} (expected {inst_path})")
    return task, task.load_spec(inst_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mstar-vibesys")
    ap.add_argument("task_type", help="e.g. add-model")
    ap.add_argument("instance", help="e.g. lingbot")
    ap.add_argument("action", choices=["show", "build-image", "build", "run", "clean"])
    ap.add_argument("--commit", default="HEAD", help="mstar rev to seed the worktree from")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--backend", default="cuda")
    ap.add_argument("--cli-provider", default="claude")
    ap.add_argument("--no-docker", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="docker build --no-cache")
    ap.add_argument("--force", action="store_true", help="recreate an existing worktree/bundle")
    ap.add_argument("--dry-run", action="store_true", help="print the vibesys command without running it")
    args = ap.parse_args(argv)

    task, spec = _resolve(args.task_type, args.instance)
    layout = default_layout()
    bundle_dir = layout.bundle_dir(task.name, args.instance)
    seed_dir = layout.seed_dir(task.name, args.instance)
    image_tag = layout.image_tag(task.name, args.instance)

    if args.action == "show":
        print(f"task     : {task.name}  (domain={task.domain})")
        print(f"instance : {args.instance}")
        print(f"spec     : {spec}")
        print(f"bundle   : {bundle_dir}")
        print(f"seed     : {seed_dir}")
        print(f"image    : {image_tag}")
        return 0

    if args.action == "clean":
        seed_mod.remove_seed(seed_dir)
        shutil.rmtree(bundle_dir, ignore_errors=True)
        print(f"removed {seed_dir} and {bundle_dir}")
        return 0

    if args.action == "build-image":
        spec_img = task.image(spec)
        if spec_img is None:
            print("this task declares no image; nothing to build")
            return 0
        image_mod.build_image(
            spec_img.dockerfile, layout.root, image_tag,
            build_args=spec_img.build_args, no_cache=args.no_cache,
        )
        print(f"built {image_tag}")
        return 0

    if args.action in ("build", "run"):
        commit = seed_mod.resolve_commit(layout.root, args.commit)
        if args.force:
            shutil.rmtree(bundle_dir, ignore_errors=True)
        seed_mod.create_seed(layout.root, commit, seed_dir, force=args.force)
        task.render_bundle(spec, bundle_dir)
        task.seed_files(spec, seed_dir)
        print(f"seeded worktree @ {commit[:12]} -> {seed_dir}")
        print(f"rendered bundle -> {bundle_dir}")
        if args.action == "build":
            return 0

        docker = not args.no_docker
        docker_image = image_tag if (docker and image_mod.image_exists(image_tag)) else None
        if docker and docker_image is None:
            print(f"note: image {image_tag} not built; run `build-image` first or pass --no-docker", file=sys.stderr)
        inputs = task.synthesis_inputs(spec, bundle_dir, seed_dir)
        # A persistent HF cache so the ~12GB checkpoint downloads once and is
        # reused across runs (the candidate loads weights by HF id in local mode).
        hf_cache = layout.base / "hf_cache"
        opts = task.run_options(
            spec, exp_name=f"{task.name}-{args.instance}", docker_image=docker_image or "",
            rounds=args.rounds, backend=args.backend, docker=docker, cli_provider=args.cli_provider,
            extra_env={"HF_HOME": str(hf_cache)}, runs_dir=layout.base / "exp_env",
        )
        argv_out = runner_mod.build_argv(inputs, opts)
        return runner_mod.exec_vibesys(argv_out, dry_run=args.dry_run, extra_env=opts.extra_env)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
