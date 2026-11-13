// Embeds the content hash of this crate's sources as MSTAR_SERVER_SOURCE_HASH,
// hashed as for the extension (../src/source_hash.rs). The rust/ half of what
// the binary reports is the rlib's own BUILD_SOURCE_HASH, so it describes the
// library code linked in even when only this crate was rebuilt.

include!("../src/source_hash.rs");

fn main() {
    let root = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let hash = source_hash(Path::new(&root)).expect("hash the crate's sources");
    println!("cargo:rustc-env=MSTAR_SERVER_SOURCE_HASH={hash}");
    for f in TOP_FILES {
        println!("cargo:rerun-if-changed={f}");
    }
    println!("cargo:rerun-if-changed=src");
}
