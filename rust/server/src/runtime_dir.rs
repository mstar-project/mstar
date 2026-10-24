//! The frontend's runtime directories (bridge socket dir, uploaded media) and
//! the files written into them. Twin of `mstar/utils/runtime_dir.py`; keep the
//! checks and the error text in step.

use std::fs::{DirBuilder, OpenOptions};
use std::io::{self, Write};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

/// `/tmp/mstar_uploads_<user>/`, as `default_upload_dir` in
/// `mstar/utils/socket_prefix.py`: the ownership check refuses another user's
/// upload dir, so one shared default would stop the second user on a host.
pub fn default_upload_dir() -> PathBuf {
    let vars = ["LOGNAME", "USER", "LNAME", "USERNAME"].map(std::env::var_os);
    let name = login_name(&vars).or_else(passwd_name);
    // SAFETY: getuid has no preconditions and cannot fail.
    upload_dir_for(name.as_deref(), unsafe { libc::getuid() })
}

/// getpass.getuser's pick: the first non-empty variable, even one that is not
/// UTF-8 (it then fails the name rule and the uid is used, as in Python).
fn login_name(vars: &[Option<std::ffi::OsString>]) -> Option<String> {
    vars.iter()
        .flatten()
        .find(|v| !v.is_empty())
        .map(|v| v.to_str().unwrap_or_default().to_string())
}

/// The user name when it is a short ASCII name, else the uid (the rule of
/// `_user` in `mstar/utils/socket_prefix.py`).
fn upload_dir_for(name: Option<&str>, uid: u32) -> PathBuf {
    let tag = match name {
        Some(n) if (1..=32).contains(&n.len())
            && n.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b)) => n.to_string(),
        _ => uid.to_string(),
    };
    PathBuf::from(format!("/tmp/mstar_uploads_{tag}/"))
}

fn passwd_name() -> Option<String> {
    // SAFETY: called once at startup, before other threads could call getpw*;
    // the entry is copied out before returning.
    unsafe {
        let pw = libc::getpwuid(libc::getuid());
        if pw.is_null() || (*pw).pw_name.is_null() {
            return None;
        }
        std::ffi::CStr::from_ptr((*pw).pw_name).to_str().ok().map(str::to_string)
    }
}

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

    #[test]
    fn the_default_upload_dir_is_per_user() {
        // the cases of test_the_upload_dir_default_holds_the_user_name_only_when_short_and_ascii
        let long = "a".repeat(32);
        let accented = "\u{e9}".repeat(30);
        for (name, kept) in [
            (Some("alice"), true),
            (Some(long.as_str()), true),
            (Some(&"a".repeat(33)[..]), false),
            (Some(accented.as_str()), false),
            (Some("a b"), false),
            (Some(""), false),
            (None, false),
        ] {
            let tag = if kept { name.unwrap().to_string() } else { "1234".to_string() };
            assert_eq!(upload_dir_for(name, 1234), PathBuf::from(format!("/tmp/mstar_uploads_{tag}/")), "{name:?}");
        }
        assert_ne!(upload_dir_for(Some("alice"), 1), upload_dir_for(Some("bob"), 2));
        assert!(default_upload_dir().starts_with("/tmp"));
    }

    #[test]
    fn the_login_name_is_the_first_non_empty_variable_even_if_not_utf8() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;
        let bad = Some(OsString::from_vec(vec![0xff]));
        let user = Some(OsString::from("alice"));
        assert_eq!(login_name(&[None, Some(OsString::new()), user.clone()]).as_deref(), Some("alice"));
        // Python takes LOGNAME here, fails its name rule and uses the uid
        let picked = login_name(&[bad, user]);
        assert_eq!(upload_dir_for(picked.as_deref(), 1001), PathBuf::from("/tmp/mstar_uploads_1001/"));
        assert_eq!(login_name(&[None, None]), None);
    }

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
