def require_rust_extension() -> None:
    """Raise, saying how to get it, if the ``mstar_rust`` extension is missing."""
    try:
        import mstar_rust  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The graph runtime needs the mstar_rust extension, which this environment "
            "does not have. The Linux wheels of mstar-ai on PyPI carry it; from a "
            "checkout, build it with `maturin develop --release -m rust/Cargo.toml` "
            "(see docs/installation.rst)."
        ) from e
