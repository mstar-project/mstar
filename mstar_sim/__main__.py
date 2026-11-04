"""Run mstar (or any Python command) with every process on an emulated GPU.

    python -m mstar_sim [--gpu h100] [--num-gpus 8] [--memory-gb N]
                        [--profile-db FILE] [--trace FILE]
                        <mstar command ...> | -- <python args ...>

``<mstar command>`` is anything ``mstar`` takes (``serve qwen3_5_0.8b
--config ...``). After ``--`` come plain Python arguments, e.g.
``-- -m pytest test/modular/test_x.py`` or ``-- my_script.py``.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MSTAR_COMMANDS = {"serve"}


def environment(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    env["MSTAR_SIM"] = "1"
    env["MSTAR_SIM_GPU"] = args.gpu
    env["MSTAR_SIM_NUM_GPUS"] = str(args.num_gpus)
    env["MSTAR_SIM_MAX_CAPTURE_TOKENS"] = str(args.max_capture_tokens)
    env["MSTAR_SIM_MAX_CAPTURE_BATCH"] = str(args.max_capture_batch)
    if args.cost:
        env["MSTAR_SIM_COST"] = args.cost
    if args.no_graphs or args.eager_nodes:
        env["MSTAR_SIM_NO_GRAPHS"] = "1" if args.no_graphs else args.eager_nodes
    if args.memory_gb:
        env["MSTAR_SIM_MEMORY_GB"] = str(args.memory_gb)
    if args.profile_db:
        env["MSTAR_SIM_PROFILE_DB"] = os.path.abspath(args.profile_db)
    if args.trace:
        env["MSTAR_SIM_TRACE"] = os.path.abspath(args.trace)
    if args.report:
        env["MSTAR_SIM_REPORT"] = os.path.abspath(args.report)
    # the site hook installs the emulator in every interpreter that starts
    paths = [os.path.join(HERE, "_site"), os.path.dirname(HERE)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="python -m mstar_sim", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", default="h100", help="simulated GPU (h100, h200, b200, a100, l40s, ...)")
    parser.add_argument("--num-gpus", type=int, default=8, help="GPUs the box appears to have")
    parser.add_argument("--memory-gb", type=float, default=None,
                        help="memory each simulated GPU reports (default: the card's)")
    parser.add_argument("--max-capture-tokens", type=int, default=0,
                        help="drop accelerator-graph buckets above this many tokens, for a "
                             "faster start-up (default 0: keep all)")
    parser.add_argument("--max-capture-batch", type=int, default=0,
                        help="drop accelerator-graph buckets above this batch size (default 0: "
                             "keep all)")
    parser.add_argument("--no-graphs", action="store_true",
                        help="capture no accelerator graphs: every step runs eagerly "
                             "(for debugging: module hooks fire on every forward)")
    parser.add_argument("--eager-nodes", default=None, metavar="A,B",
                        help="run only these graph nodes eagerly, capturing the rest (a node "
                             "whose capture needs a real Triton build)")
    parser.add_argument("--cost", choices=["tables", "roofline"], default=None,
                        help="kernel times: 'tables' (default when the AIConfigurator data is "
                             "found: measured per-GPU tables, roofline for the rest) or 'roofline'")
    parser.add_argument("--profile-db", default=None,
                        help="measured kernel timings (python -m mstar_sim.profile) to use "
                             "before the roofline model")
    parser.add_argument("--trace", default=None,
                        help="append every simulated kernel's shape key here, for profiling")
    parser.add_argument("--report", default=None,
                        help="write per-process simulated-device stats here at exit")
    args, rest = parser.parse_known_args(argv)
    if rest and rest[0] == "--":
        command = rest[1:]
        isolated = {"-I", "-E"} & set(command[: command.index("-m")] if "-m" in command else command[:1])
        if isolated:
            parser.error(f"{' '.join(sorted(isolated))} makes Python ignore PYTHONPATH, so the emulator "
                         "would never install (and the installed mstar, not this tree's, would load)")
    elif rest and rest[0] in MSTAR_COMMANDS:
        command = ["-m", "mstar.cli.main", *rest]
    else:
        parser.error("give an mstar command (e.g. `serve <model>`) or `-- <python args>`")
    if not args.cost and not os.environ.get("MSTAR_SIM_COST"):
        from mstar_sim import tables

        if not os.path.isdir(tables.data_dir()):
            print("mstar_sim: kernel times from the roofline model; for NVIDIA's measured tables: "
                  f"{tables.FETCH}", file=sys.stderr)
    os.execvpe(sys.executable, [sys.executable, *command], environment(args))


if __name__ == "__main__":
    main()
