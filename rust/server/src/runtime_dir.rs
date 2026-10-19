//! The frontend's runtime directories (bridge socket dir, uploaded media) and
//! the files written into them. Twin of `mstar/utils/runtime_dir.py`; keep the
//! checks and the error text in step.

use std::fs::{DirBuilder, OpenOptions};
use std::io::{self, Write};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

/// Create `path` with mode 0700 if it is missing, then require a real
/// directory owned by this user and not writable by group or others.
///
/// The default locations sit in a shared `/tmp` under predictable names, so an
/// existing directory is only accepted after this check; anything else is a
/// `PermissionDenied` error naming the path. Only the final component has to
/// be ours; missing parents are created 0700 too, so a parent that is itself a
/// runtime dir (a socket dir holding the upload dir) passes its own check under
/// any umask.
pub fn ensure_private_dir(path: &Path) -> io::Result<PathBuf> {
    // SAFETY: geteuid has no preconditions and cannot fail.
    ensure_private_dir_for(path, unsafe { libc::geteuid() })
}

fn ensure_private_dir_for(path: &Path, uid: u32) -> io::Result<PathBuf> {
    // Rebuilding from components drops a trailing slash, without which lstat
    // would follow a symlink.
    let path: PathBuf = path.components().collect();
    let meta = match std::fs::symlink_metadata(&path) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                DirBuilder::new().recursive(true).mode(0o700).create(parent)?;
            }
            match DirBuilder::new().mode(0o700).create(&path) {
                Err(e) if e.kind() != io::ErrorKind::AlreadyExists => return Err(e),
                _ => {}
            }
            std::fs::symlink_metadata(&path)?
        }
        Err(e) => return Err(e),
    };
    let kind = meta.file_type();
    let mode = meta.permissions().mode() & 0o7777;
    let problem = if kind.is_symlink() {
        "it is a symlink".to_string()
    } else if !kind.is_dir() {
        "it is not a directory".to_string()
    } else if meta.uid() != uid {
        format!("it is owned by uid {}", meta.uid())
    } else if mode & 0o022 != 0 {
        format!("its mode is {mode:o}")
    } else {
        return Ok(path);
    };
    Err(io::Error::new(
        io::ErrorKind::PermissionDenied,
        format!(
            "refusing {}: {problem}. It must be a directory owned by this user \
             (uid {uid}) and not writable by group or others; remove it, or pass \
             another path with --socket-path-prefix / --upload-dir.",
            path.display()
        ),
    ))
}

/// Write `data` to a new file at `path`, mode 0600; an existing file or
/// symlink there is an `AlreadyExists` error instead of being written through.
pub fn write_private_file(path: &Path, data: &[u8]) -> io::Result<()> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?
        .write_all(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    fn uid() -> u32 {
        unsafe { libc::geteuid() }
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("mstar_rtdir_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        DirBuilder::new().mode(0o700).create(&dir).unwrap();
        dir
    }

    fn mode_of(path: &Path) -> u32 {
        std::fs::symlink_metadata(path).unwrap().permissions().mode() & 0o7777
    }

    #[test]
    fn fresh_dir_is_created_0700_under_umask_022_and_002() {
        let root = scratch("fresh");
        for umask in [0o022, 0o002] {
            let old = unsafe { libc::umask(umask) };
            let made = ensure_private_dir(&root.join(format!("{umask:o}/run/")));
            unsafe { libc::umask(old) };
            let made = made.unwrap();
            assert_eq!(made, root.join(format!("{umask:o}/run")));
            assert_eq!(mode_of(&made), 0o700);
            // An existing private dir is accepted as is.
            ensure_private_dir(&made).unwrap();
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_parent_created_for_a_nested_dir_passes_its_own_check() {
        let root = scratch("nested");
        for umask in [0o022, 0o002] {
            let old = unsafe { libc::umask(umask) };
            let runtime = root.join(format!("{umask:o}/runtime"));
            let made = ensure_private_dir(&runtime.join("uploads"))
                .and_then(|_| ensure_private_dir(&runtime));
            unsafe { libc::umask(old) };
            assert_eq!(made.unwrap(), runtime);
            assert_eq!(mode_of(&runtime), 0o700);
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn foreign_owner_is_refused() {
        let root = scratch("owner");
        let err = ensure_private_dir_for(&root, uid() + 1).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);
        let msg = err.to_string();
        assert!(msg.contains(&format!("owned by uid {}", uid())), "{msg}");
        assert!(msg.contains("--upload-dir"), "{msg}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn symlink_is_refused_even_with_a_trailing_slash() {
        let root = scratch("link");
        let target = root.join("target");
        DirBuilder::new().mode(0o700).create(&target).unwrap();
        let link = root.join("link");
        symlink(&target, &link).unwrap();
        let mut with_slash = link.clone().into_os_string();
        with_slash.push("/");
        let err = ensure_private_dir(Path::new(&with_slash)).unwrap_err();
        assert!(err.to_string().contains("is a symlink"), "{err}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn group_or_world_writable_dir_is_refused() {
        let root = scratch("mode");
        for mode in [0o777, 0o770, 0o702] {
            std::fs::set_permissions(&root, std::fs::Permissions::from_mode(mode)).unwrap();
            let err = ensure_private_dir(&root).unwrap_err();
            assert!(err.to_string().contains(&format!("mode is {mode:o}")), "{err}");
        }
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn private_file_is_0600_and_never_overwrites() {
        let root = scratch("file");
        let path = root.join("upload.wav");
        write_private_file(&path, b"first").unwrap();
        assert_eq!(mode_of(&path), 0o600);
        let err = write_private_file(&path, b"second").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(&path).unwrap(), b"first");
        // A symlink planted at the name is not followed.
        let outside = root.join("outside");
        std::fs::write(&outside, b"keep").unwrap();
        let link = root.join("planted.wav");
        symlink(&outside, &link).unwrap();
        assert!(write_private_file(&link, b"x").is_err());
        assert_eq!(std::fs::read(&outside).unwrap(), b"keep");
        std::fs::remove_dir_all(&root).unwrap();
    }
}
