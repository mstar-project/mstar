"""The command line of tier 1.

    python -m fuzzer.tier1 list
    python -m fuzzer.tier1 run --seeds 500
    python -m fuzzer.tier1 run --machine model_run --time 60 --all
    python -m fuzzer.tier1 run --machine model_run --save wide-loop-tail
    python -m fuzzer.tier1 replay fuzzer/tier1/corpus/model_run/<case>.json

The ``run`` command exits with the code 1 if any machine fails. It stops at the
first failure of each machine. Use ``--all`` to collect every signature.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fuzzer.common.case import Case
from fuzzer.common.driver import run_case, save_to_corpus, search
from fuzzer.tier1.machines import MACHINES, TIER


def _report(machine_name: str, case: Case, failure) -> None:
    """Print one failure: the signature, the message and the case."""
    print(f"\n{'=' * 72}")
    print(f"FAIL  {machine_name}  [{failure.signature[0]}] {failure.signature[1]}")
    print(f"{'=' * 72}")
    print(failure.message)
    print()
    print(case.pretty())
    print()


def _cmd_list(_args: argparse.Namespace) -> int:
    """Print the name and the subject of each machine."""
    import importlib

    for name, machine in MACHINES.items():
        # The subject line of a machine is the first line of the docstring of
        # its module. It is not on the class.
        doc = importlib.import_module(machine.__module__).__doc__ or name
        print(f"  {name:18s} {doc.strip().splitlines()[0]}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Search for failures, shrink them, and print each one."""
    names = [args.machine] if args.machine else list(MACHINES)
    unknown = [name for name in names if name not in MACHINES]
    if unknown:
        print(f"unknown machine(s): {unknown}; known: {list(MACHINES)}", file=sys.stderr)
        return 2

    budget = None if args.time is None else args.time / len(names)
    total_failures = 0
    for name in names:
        machine = MACHINES[name]
        result = search(
            machine,
            seeds=range(args.start, args.start + args.seeds),
            num_ops=args.ops,
            time_budget=budget,
            do_shrink=not args.no_shrink,
            stop_after=None if args.all else 1,
        )
        status = "ok" if result.ok else f"{len(result.failures)} FAILING"
        print(
            f"{name:18s} {result.cases_run:6d} cases  "
            f"{result.elapsed:6.2f}s  {status}"
        )
        for case, failure in result.failures:
            total_failures += 1
            _report(name, case, failure)
            if args.save:
                case.notes["status"] = args.status
                case.notes["invariant"] = failure.signature[1]
                case.notes["summary"] = failure.message.split("\n")[0][:200]
                label = f"{args.save}-{failure.signature[1]}".replace("/", "_")
                path = save_to_corpus(TIER, case, label)
                print(f"saved -> {path}")
    return 1 if total_failures else 0


def _cmd_replay(args: argparse.Namespace) -> int:
    """Run one saved case again and report what it does now."""
    case = Case.load(Path(args.path))
    machine = MACHINES[case.machine]
    print(case.pretty())
    failure = run_case(machine, case)
    if failure is None:
        print("\nPASSED (the case no longer reproduces)")
        return 0
    _report(case.machine, case, failure)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Parse the arguments and run the command that they name."""
    parser = argparse.ArgumentParser(prog="python -m fuzzer.tier1")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "list", help="print the machines of this tier"
    ).set_defaults(func=_cmd_list)

    run = sub.add_parser("run", help="generate cases and shrink each failure")
    run.add_argument("--machine", help="one machine; the default is every machine")
    run.add_argument("--seeds", type=int, default=500)
    run.add_argument("--start", type=int, default=0, help="the first seed")
    run.add_argument("--ops", type=int, default=40, help="the number of ops in a case")
    run.add_argument(
        "--time", type=float,
        help="the budget in seconds, divided equally between the machines",
    )
    run.add_argument("--no-shrink", action="store_true")
    run.add_argument(
        "--all", action="store_true",
        help="continue after the first signature, to find the other failures",
    )
    run.add_argument(
        "--save", metavar="LABEL",
        help="write each small failure to the corpus, with this name",
    )
    run.add_argument(
        "--status", choices=["open", "fixed"], default="open",
        help="the status in the corpus: 'open' is a known bug, and the tests "
             "expect it to fail again; 'fixed' is a regression guard, and the "
             "tests expect it to pass",
    )
    run.set_defaults(func=_cmd_run)

    replay = sub.add_parser("replay", help="run one case from the corpus again")
    replay.add_argument("path")
    replay.set_defaults(func=_cmd_replay)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
