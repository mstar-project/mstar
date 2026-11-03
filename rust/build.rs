// Embeds the content hash of this crate's sources as MSTAR_RUST_SOURCE_HASH,
// which the extension exposes as BUILD_SOURCE_HASH (see src/source_hash.rs).

include!("src/source_hash.rs");

fn main() {
    let root = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let hash = source_hash(Path::new(&root)).expect("hash the crate's sources");
    println!("cargo:rustc-env=MSTAR_RUST_SOURCE_HASH={hash}");
    for f in TOP_FILES {
        println!("cargo:rerun-if-changed={f}");
    }
    // A directory: cargo reruns this when anything under it changes.
    println!("cargo:rerun-if-changed=src");
}
