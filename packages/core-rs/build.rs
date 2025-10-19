fn main() {
    #[cfg(feature = "sparse-suitesparse")]
    build_suitesparse();
}

#[cfg(feature = "sparse-suitesparse")]
fn build_suitesparse() {
    use std::env;
    use std::path::PathBuf;

    println!("cargo:rerun-if-changed=src/sparse/suitesparse/ffi.c");
    println!("cargo:rerun-if-changed=src/sparse/suitesparse/ffi.h");
    println!("cargo:rerun-if-env-changed=NUMRS_SUITESPARSE_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=NUMRS_SUITESPARSE_LIB_DIR");
    println!("cargo:rerun-if-env-changed=NUMRS_SUITESPARSE_LINK_LIBS");
    println!("cargo:rerun-if-env-changed=NUMRS_SUITESPARSE_EMULATION");

    let mut build = cc::Build::new();
    build.file("src/sparse/suitesparse/ffi.c");
    build.include("src/sparse/suitesparse");
    build.flag_if_supported("-std=c11");

    if let Ok(include_dir) = env::var("NUMRS_SUITESPARSE_INCLUDE_DIR") {
        build.include(include_dir);
    }

    let mut emulate = true;
    if let Ok(value) = env::var("NUMRS_SUITESPARSE_EMULATION") {
        emulate = value != "0";
    }

    if emulate {
        build.define("NUMRS_SUITESPARSE_EMULATION", None);
    } else {
        build.define("NUMRS_SUITESPARSE_USE_SYSTEM", None);
        if let Ok(link_dir) = env::var("NUMRS_SUITESPARSE_LIB_DIR") {
            println!("cargo:rustc-link-search=native={link_dir}");
        }
        let libs = env::var("NUMRS_SUITESPARSE_LINK_LIBS")
            .unwrap_or_else(|_| "cholmod,amd,colamd,suitesparseconfig".to_string());
        for lib in libs.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            println!("cargo:rustc-link-lib={lib}");
        }
    }

    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // Allow cross compilation scenarios where the SuiteSparse headers or libs
    // live outside the workspace by appending an arbitrary sysroot include dir.
    if let Ok(sysroot) = env::var("NUMRS_SYSROOT") {
        let sysroot_path = PathBuf::from(&sysroot);
        let include_path = sysroot_path.join("include");
        build.include(&include_path);
        let lib_path = sysroot_path.join("lib");
        println!("cargo:rustc-link-search=native={}", lib_path.display());
    }

    build.compile("numrs_suitesparse");
}
