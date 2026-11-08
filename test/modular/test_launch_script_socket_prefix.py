"""Every server binds the same IPC names under its ``--socket-path-prefix``, so
two servers on one host need two prefixes. A launch script whose prefix is
keyed only on the user, or that relied on the entrypoint's old shared
``/tmp/mstar`` default, gives every server it starts the same sockets: the
second one takes over the first one's API inboxes before the deployment lock
refuses it."""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = sorted(p for d in ("test", "benchmark") for p in (REPO / d).rglob("*.sh"))
PREFIX_ARG = re.compile(r"--socket-path-prefix[ =]+(\"[^\"]*\"|'[^']*'|\S+)")
VARIABLE = re.compile(r"\$(?:\{(\w+)[^}]*\}|(\w+)|(\$)|(\())")
ENTRYPOINT = re.compile(r"api_server/entrypoint\.py|mstar\.api_server\.entrypoint|\bmstar-serve\b")
PER_USER = {"WHO", "USER", "USERNAME", "LOGNAME", "HOME"}


def _assignment(text: str, name: str) -> str | None:
    match = re.search(rf"^\s*(?:local\s+|export\s+)?{name}=(\S+)", text, re.MULTILINE)
    return match.group(1) if match else None


def _keys(text: str, value: str, seen: frozenset = frozenset()) -> set[str]:
    """The variables a prefix depends on, following same-file assignments."""
    keys = set()
    for braced, bare, pid, subshell in VARIABLE.findall(value):
        name = braced or bare or pid or subshell
        if name in seen:  # PORT="${PORT:-8000}": the caller's PORT
            keys.add(name)
            continue
        assigned = _assignment(text, name) if (braced or bare) else None
        keys |= _keys(text, assigned, seen | {name}) if assigned is not None else {name}
    return keys


def _code(path: Path) -> str:
    """The script without comment lines, one command per line."""
    lines = (line for line in path.read_text().splitlines() if not line.lstrip().startswith("#"))
    return re.sub(r"\\\n", " ", "\n".join(lines))


def _prefixes(path: Path) -> list[str]:
    return PREFIX_ARG.findall(_code(path))


def test_the_scan_finds_the_launch_scripts():
    assert sum(len(_prefixes(p)) for p in SCRIPTS) >= 14


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: str(p.relative_to(REPO)))
def test_launch_script_passes_a_socket_prefix(script):
    for command in _code(script).splitlines():
        if ENTRYPOINT.search(command):
            assert "--socket-path-prefix" in command, (
                f"{script.relative_to(REPO)} starts the entrypoint without --socket-path-prefix: {command.strip()}"
            )


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: str(p.relative_to(REPO)))
def test_launch_script_socket_prefix_is_unique_per_server(script):
    text = _code(script)
    for value in _prefixes(script):
        keys = _keys(text, value)
        assert keys - PER_USER, (
            f"{script.relative_to(REPO)}: --socket-path-prefix {value} depends only on {sorted(keys) or 'literals'}; "
            "key it on the port too (e.g. /tmp/mstar_${WHO}_${PORT}/) so two servers on one host do not share sockets"
        )
