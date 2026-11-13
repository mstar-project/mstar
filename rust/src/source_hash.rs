// The content hash a checkout compares its mstar_rust and mstar-server builds
// with. build.rs and server/build.rs `include!` this file to embed it, and the
// extension exposes the same function as `source_hash`, so there is one
// implementation.
// Changing it is changing a hashed source, so a build made with older hashing
// code is refused like any other stale build. (No `//!` here: `include!`
// cannot take inner attributes.)
//
// FNV-1a 64, which is fixed by definition (std's DefaultHasher may change
// between Rust versions), over each hashed file in byte order of its
// `/`-separated path: the path, a NUL, the length as a little-endian u64, then
// the bytes.

use std::fs;
use std::io;
use std::path::Path;

/// Hashed beside every file under `src/`; missing ones are skipped. The
/// crate's own inputs only, so a build from a copy of rust/ hashes the same.
pub const TOP_FILES: [&str; 3] = ["build.rs", "Cargo.toml", "Cargo.lock"];

pub fn source_hash(rust_dir: &Path) -> io::Result<String> {
    let mut files: Vec<String> = TOP_FILES
        .iter()
        .filter(|f| rust_dir.join(f).is_file())
        .map(|f| f.to_string())
        .collect();
    collect_sources(rust_dir, "src", &mut files)?;
    files.sort();

    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |bytes: &[u8]| {
        for &b in bytes {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
    };
    for rel in &files {
        let bytes = fs::read(rust_dir.join(rel))?;
        feed(rel.as_bytes());
        feed(&[0]);
        feed(&(bytes.len() as u64).to_le_bytes());
        feed(&bytes);
    }
    Ok(format!("{h:016x}"))
}

/// Files under `root/rel`, skipping dot-names (.DS_Store, editor swap files)
/// and not following symlinks.
fn collect_sources(root: &Path, rel: &str, out: &mut Vec<String>) -> io::Result<()> {
    for entry in fs::read_dir(root.join(rel))? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let path = format!("{rel}/{name}");
        let kind = entry.file_type()?;
        if kind.is_dir() {
            collect_sources(root, &path, out)?;
        } else if kind.is_file() {
            out.push(path);
        }
    }
    Ok(())
}
