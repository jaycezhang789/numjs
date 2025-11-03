use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    napi_build::setup();
    configure_gpu_toolkits();
}

fn configure_gpu_toolkits() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if env::var("CARGO_FEATURE_GPU_CUDA").is_ok() {
        println!("cargo:rerun-if-env-changed=CUDA_HOME");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        println!("cargo:rerun-if-env-changed=CUDA_ROOT");
        if let Some(root) = find_cuda_root(&target_os) {
            emit_link_search_dirs("CUDA", &root, &target_os);
        } else {
            println!(
                "cargo:warning=CUDA feature enabled but CUDA toolkit not found. Set CUDA_HOME/CUDA_PATH."
            );
        }
        emit_cuda_links(&target_os);
    }

    if env::var("CARGO_FEATURE_GPU_ROCM").is_ok() {
        println!("cargo:rerun-if-env-changed=ROCM_HOME");
        println!("cargo:rerun-if-env-changed=ROCM_PATH");
        println!("cargo:rerun-if-env-changed=ROCM_ROOT");
        if let Some(root) = find_rocm_root(&target_os) {
            emit_link_search_dirs("ROCM", &root, &target_os);
        } else {
            println!(
                "cargo:warning=ROCm feature enabled but ROCm toolkit not found. Set ROCM_HOME/ROCM_PATH."
            );
        }
        emit_rocm_links(&target_os);
    }
}

fn find_cuda_root(target_os: &str) -> Option<PathBuf> {
    let candidates = [
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
    ];
    if let Some(path) = first_existing_path(&candidates) {
        return Some(path);
    }
    match target_os {
        "windows" => find_latest_version_dir(Path::new(
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        )),
        "linux" | "macos" => {
            let fallback = Path::new("/usr/local/cuda");
            fallback.exists().then(|| fallback.to_path_buf())
        }
        _ => None,
    }
}

fn find_rocm_root(target_os: &str) -> Option<PathBuf> {
    let candidates = ["ROCM_HOME", "ROCM_PATH", "ROCM_ROOT"];
    if let Some(path) = first_existing_path(&candidates) {
        return Some(path);
    }
    match target_os {
        "windows" => env::var("HIP_PATH").ok().map(PathBuf::from),
        "linux" => {
            let fallback = Path::new("/opt/rocm");
            fallback.exists().then(|| fallback.to_path_buf())
        }
        "macos" => None,
        _ => None,
    }
}

fn first_existing_path(keys: &[&str]) -> Option<PathBuf> {
    for key in keys {
        if let Ok(value) = env::var(key) {
            if !value.is_empty() {
                let path = PathBuf::from(value);
                if path.exists() {
                    return Some(path);
                }
            }
        }
    }
    None
}

fn find_latest_version_dir(base: &Path) -> Option<PathBuf> {
    let entries = fs::read_dir(base).ok()?;
    let mut versions: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.is_dir())
        .collect();
    versions.sort();
    versions.pop()
}

fn emit_link_search_dirs(name: &str, root: &Path, target_os: &str) {
    let mut candidates = Vec::new();
    match target_os {
        "windows" => {
            candidates.push(root.join("lib").join("x64"));
        }
        "linux" => {
            candidates.push(root.join("lib64"));
            candidates.push(root.join("lib"));
        }
        "macos" => {
            candidates.push(root.join("lib"));
        }
        _ => {}
    }
    for dir in candidates {
        if dir.exists() {
            println!("cargo:rustc-link-search=native={}", dir.to_string_lossy());
        }
    }
    println!(
        "cargo:warning={name} root detected at {}",
        root.to_string_lossy()
    );
}

fn emit_cuda_links(target_os: &str) {
    // Driver API is required by the cust crate; BLAS/SOLVER libraries will be
    // needed once the GPU kernels are implemented.
    let libs = match target_os {
        "windows" => ["cuda", "cudart", "cublas", "cusolver"],
        _ => ["cuda", "cudart", "cublas", "cusolver"],
    };
    for lib in libs {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    if target_os == "macos" {
        println!("cargo:warning=CUDA on macOS is deprecated; ensure legacy drivers are installed.");
    }
}

fn emit_rocm_links(target_os: &str) {
    let libs = match target_os {
        "windows" => ["amdhip64", "rocblas", "rocsolver"],
        _ => ["amdhip64", "rocblas", "rocsolver"],
    };
    for lib in libs {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    if target_os == "macos" {
        println!("cargo:warning=ROCm is not officially supported on macOS.");
    }
}
