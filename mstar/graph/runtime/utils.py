import tomllib
from pathlib import Path

# A checkout's rust/, beside its mstar/ package. An installed wheel has none.
_CHECKOUT_RUST = Path(__file__).resolve().parents[3] / "rust"


def require_rust_extension(rust_dir: Path = _CHECKOUT_RUST) -> None:
    """Raise, saying how to get it, if the ``mstar_rust`` extension is missing
    or, in a checkout, was built from other sources than ``rust_dir``.

    A build from older sources at the same version imports fine and then fails
    mid-request. Each worker checks its own host's build.
    """
    try:
        import mstar_rust
    except ImportError as e:
        raise ImportError(
            "The graph runtime needs the mstar_rust extension, which this environment "
            "does not have. The Linux wheels of mstar-ai on PyPI carry it; from a "
            "checkout, build it with `maturin develop --release -m rust/Cargo.toml` "
            "(see docs/installation.rst)."
        ) from e
    if not _is_mstar_rust_crate(rust_dir):
        return
    # Content, not mtimes: a branch switch or stash pop that rewrites a file
    # unchanged leaves the build valid.
    built = getattr(mstar_rust, "BUILD_SOURCE_HASH", None)
    if built is None:
        problem = "predates the source check"
    else:
        current = mstar_rust.source_hash(str(rust_dir))
        if current == built:
            return
        problem = f"was built from sources hashing to {built}, but {rust_dir} hashes to {current}"
    # The touch makes cargo rerun build.rs and so recompile the crate. Cargo
    # goes by mtimes, so without it sources copied in with older ones (`cp -p`,
    # `rsync -a`, tar) are not rebuilt and this repeats.
    raise ImportError(
        f"The mstar_rust build at {mstar_rust.__file__} {problem}. Rebuild it with "
        f"`touch {rust_dir / 'build.rs'} && maturin develop --release -m {rust_dir / 'Cargo.toml'}`."
    )


def _is_mstar_rust_crate(rust_dir: Path) -> bool:
    try:
        with open(rust_dir / "Cargo.toml", "rb") as f:
            return tomllib.load(f).get("lib", {}).get("name") == "mstar_rust"
    except (OSError, tomllib.TOMLDecodeError):
        return False
