"""Upstream compatibility audit, run explicitly by emulator maintainers.

    python -m pytest mstar_sim/audits -q

These checks inspect mstar's current interfaces, so they are outside CPU Core.
Emulator behavior and its pinned planner golden remain in mstar_sim/tests.
"""

from mstar_sim.tests.test_device import ROOT, run_sim


def test_every_cuda_seam_mstar_reaches_is_emulated():
    """The contract with mstar: every CUDA entry point, kernel-library name, ctypes
    library and torch.ops namespace mstar's source reaches is a stand-in, or is
    listed with a reason in the table of the module that owns that seam. A new one
    is reported here for emulator maintenance, outside the required CPU Core suite."""
    out = run_sim("""
        import ast, importlib, pathlib, re, torch
        from mstar_sim import cuda, device, libcuda, patches

        assert not patches.SKIPPED, f"patches no longer apply to this mstar: {patches.SKIPPED}"

        def dotted(node):
            parts = []
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            return ".".join([node.id, *reversed(parts)]) if isinstance(node, ast.Name) else None

        found = {}
        for path in pathlib.Path(%r, "mstar").rglob("*.py"):
            if "tests" in path.parts:
                continue
            tree = ast.parse(path.read_text())
            uses_ctypes = "ctypes" in path.read_text()
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Attribute) and (d := dotted(node)):
                    p = d.split(".")
                    if p[:2] in (["torch", "cuda"], ["torch", "accelerator"]) and len(p) > 2:
                        names.append(".".join(p[:3]))
                    elif p[:2] == ["torch", "_C"] and len(p) > 2 and p[2].startswith("_cuda"):
                        names.append(".".join(p[:3]))
                    elif p[:2] == ["torch", "ops"] and len(p) > 2:
                        names.append("ops:" + p[2])
                    elif p[0] == "flashinfer" and len(p) > 1:
                        names.append(d)
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    top = node.module.split(".")[0]
                    if top == "flashinfer":
                        names += [f"{node.module}.{a.name}" for a in node.names]
                    elif top == "cuda":
                        names.append("module:cuda")
                    elif "gluon" in node.module.split(".") or any(a.name == "gluon" for a in node.names):
                        names.append("gluon")
                elif isinstance(node, ast.Import):
                    for a in node.names:
                        if a.name.split(".")[0] == "cuda":
                            names.append("module:cuda")
                        elif a.name.startswith("flashinfer."):
                            names.append(a.name)
                elif uses_ctypes and isinstance(node, ast.Constant) and isinstance(node.value, str):
                    names += ["lib:" + m for m in re.findall(r"\\b(lib[a-z0-9_]+?)(?:\\.so|$)", node.value)]
                for n in names:
                    found.setdefault(n, f"{path}:{node.lineno}")

        import flashinfer  # under the emulator, the whole package is the stand-in
        assert "fake_libs" in flashinfer.__file__, flashinfer.__file__
        from mstar.engine.resources.attn import wrappers  # mstar's pin moves, so must the stand-in
        assert flashinfer.__version__ in wrappers.DIRECT_PLAN_FLASHINFER_VERSIONS, flashinfer.__version__

        def resolves(name):
            obj = importlib.import_module(name.split(".")[0])
            for part in name.split(".")[1:]:
                try:
                    obj = getattr(obj, part)
                except AttributeError:
                    try:
                        obj = importlib.import_module(f"{obj.__name__}.{part}")
                    except ImportError:
                        return None
            return obj

        missing = []
        for name, where in sorted(found.items()):
            if name == "gluon":
                ok = resolves("triton.experimental.gluon") is not None
            elif name.startswith("ops:"):
                ok = name[4:] in device.OPS_NAMESPACES
            elif name.startswith("module:"):
                ok = name[7:] in device.ABSENT_MODULES
            elif name.startswith("lib:"):
                ok = name[4:] in libcuda.LIBRARIES
            elif name.startswith("flashinfer"):
                try:
                    ok = resolves(name) is not None
                except (AttributeError, ImportError):
                    ok = False
            else:
                obj = resolves(name)
                ok = name in cuda.PASSTHROUGH or (
                    obj is not None and (getattr(obj, "__module__", "") or "").startswith("mstar_sim"))
            if not ok:
                missing.append(f"{name} (first at {where})")
        assert not missing, (
            "mstar now reaches CUDA through names the emulator doesn't cover; add a stand-in in "
            "mstar_sim/ (or a row with the reason in that module's table):\\n  " + "\\n  ".join(missing))
        print("ok", len(found))
    """ % ROOT)
    assert out.startswith("ok")


def test_kept_resident_kernel_parameters_match_the_triton_functions():
    run_sim("""
        import inspect
        from mstar.engine.resources import resident
        from mstar.engine.resources.attn import wrappers
        from mstar.engine.resources.kv import resident as kv_resident
        from mstar_sim.kernels import KEPT_WRITES

        modules = resident, kv_resident, wrappers
        mismatches = []
        for name, kept in KEPT_WRITES.items():
            originals = [getattr(module, name) for module in modules if hasattr(module, name)]
            assert len(originals) == 1, f"{name}: expected one resident Triton kernel"
            expected = list(inspect.signature(originals[0].fn).parameters)
            # **_ receives Triton launch options (e.g. num_warps), not kernel arguments.
            actual = [p.name for p in inspect.signature(kept).parameters.values()
                      if p.kind != inspect.Parameter.VAR_KEYWORD]
            if actual != expected:
                mismatches.append(f"{name}: stand-in {actual}; Triton {expected}")
        assert not mismatches, "\\n".join(mismatches)
    """)
