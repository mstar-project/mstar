// Under the `python` feature, builds the torch ABI shim the extension embeds.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/torch_abi/shim.cpp");
    println!("cargo:rustc-check-cfg=cfg(mstar_torch_shim)");
    #[cfg(feature = "python")]
    torch_shim::build(std::path::Path::new(
        &std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"),
    ));
}

/// The shim is its own shared library so the extension never links torch:
/// `THPVariableClass` is a data symbol, so a linked extension could not import
/// before torch. It is embedded in the extension (`include_bytes!` in
/// src/torch_abi/mod.rs) and dlopened after torch is imported. Embedded rather
/// than installed beside it because the extension has two build paths with
/// different layouts (maturin, setuptools-rust), auditwheel would try to graft
/// the libtorch a wheel file names, and an embedded shim cannot go stale against
/// the extension the source-hash check vouches for.
#[cfg(feature = "python")]
mod torch_shim {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    const PROBE: &str = "import importlib.util as u, sysconfig\n\
        s = u.find_spec('torch')\n\
        print(sysconfig.get_path('include'))\n\
        print(s.submodule_search_locations[0] if s else '')\n";

    /// The interpreter's headers, and torch: `MSTAR_TORCH_DIR` for a build
    /// whose interpreter has none (PEP 517 isolation, see .github/workflows/
    /// release.yml), else the build interpreter's own.
    fn locate() -> Option<(String, PathBuf)> {
        println!("cargo:rerun-if-env-changed=MSTAR_TORCH_DIR");
        println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
        let python = std::env::var("PYO3_PYTHON").unwrap_or_else(|_| "python3".to_string());
        let out = Command::new(&python).args(["-c", PROBE]).output().ok()?;
        if !out.status.success() {
            return None;
        }
        let out = String::from_utf8(out.stdout).ok()?;
        let mut lines = out.lines();
        let py_include = lines.next()?.to_string();
        let torch = match std::env::var("MSTAR_TORCH_DIR") {
            Ok(dir) => PathBuf::from(dir),
            Err(_) => PathBuf::from(lines.next().filter(|s| !s.is_empty())?),
        };
        Some((py_include, torch))
    }

    pub fn build(root: &Path) {
        println!("cargo:rerun-if-env-changed=CXX");
        // Mach-O links differently, memfd is Linux's, and only Linux wheels serve.
        if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
            return;
        }
        let Some((py_include, torch)) = locate() else {
            println!(
                "cargo:warning=no torch found: building mstar_rust without the torch ABI shim, \
                 so the SHM arena will refuse to start. Install torch, then `touch rust/build.rs` \
                 and rebuild, or set MSTAR_TORCH_DIR."
            );
            return;
        };
        let version_file = torch.join("include/torch/csrc/api/include/torch/version.h");
        println!("cargo:rerun-if-changed={}", version_file.display());
        let source = root.join("src/torch_abi/shim.cpp");
        let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
        let shim = out_dir.join("libmstar_torch_v2.so");

        let compiler = cc::Build::new().cpp(true).get_compiler();
        let stamp = format!(
            "{:?} {} {} {:?} {:?}",
            compiler.path(),
            py_include,
            torch.display(),
            fs::read(&source).expect("read shim.cpp"),
            fs::read(&version_file).unwrap_or_default(),
        );
        // The shim takes tens of seconds: compile only when its own inputs changed.
        let stamp_file = out_dir.join("shim.stamp");
        let fresh = shim.is_file() && fs::read_to_string(&stamp_file).ok().as_ref() == Some(&stamp);
        if !fresh {
            let output = compiler
                .to_command()
                .args(["-std=c++17", "-O2", "-fPIC", "-shared", "-fvisibility=hidden", "-s"])
                .arg("-Wl,-soname,libmstar_torch_v2.so")
                .arg(format!("-I{py_include}"))
                .arg(format!("-I{}", torch.join("include").display()))
                .arg(&source)
                .arg("-o")
                .arg(&shim)
                .arg(format!("-L{}", torch.join("lib").display()))
                .args(["-ltorch_python", "-ltorch_cpu", "-lc10"])
                .output()
                .expect("run the C++ compiler for the torch ABI shim");
            assert!(
                output.status.success(),
                "compiling {} against {} failed:\n{}",
                source.display(),
                torch.display(),
                String::from_utf8_lossy(&output.stderr),
            );
            fs::write(&stamp_file, &stamp).expect("write the shim stamp");
        }
        println!("cargo:rustc-cfg=mstar_torch_shim");
    }
}
