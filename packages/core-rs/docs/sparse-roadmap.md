# SuiteSparse Integration Roadmap

The purpose of this roadmap is to take the sparse layer in `num_rs_core` from the original dense fallback implementation to a fully fledged native stack powered by SuiteSparse (with `sprs` acting as the portable baseline). Each phase is broken down into concrete engineering tasks, recommended tooling, and validation steps so that contributors can pick up the work without guesswork.

---

## Phase A – Technology Assessment & Build Foundations

1. **Dependency analysis**
   - Survey the SuiteSparse components we intend to bind first (`CSparse` for CSR primitives, `CHOLMOD` for factorisation, `SPMM`/`SPMV` helpers). Document version requirements and transitive dependencies (e.g. BLAS, METIS).
   - Confirm `sprs` coverage for portable builds (WASM, environments without native libraries). Record gaps where SuiteSparse will eventually be required (e.g. double precision `csrgemm` performance).
   - Define feature flags:
     - `sparse-native` – enables the pure Rust `sprs` backend.
     - `sparse-suitesparse` – compiles the C shim and links against SuiteSparse. Requires toolchain checks during `build.rs`.

2. **Cross-platform build strategy**
   - **macOS / Linux**: prefer system packages (`brew install suite-sparse`, `apt-get install libsuitesparse-dev`). Provide fallbacks via `NUMRS_SUITESPARSE_INCLUDE_DIR` and `NUMRS_SUITESPARSE_LIB_DIR` environment variables.
   - **Windows**: experiment with vcpkg triplets (`vcpkg install suitesparse`) and document the required MSVC runtime libraries. Ensure path handling in `build.rs` supports Windows separators.
   - **WASM**: keep bundling the `sprs` backend; SuiteSparse is currently out of scope for wasm32 targets. Note the limitations explicitly (no GPU, medium-sized matrices only).
   - Set up `build.rs` to honour `NUMRS_SUITESPARSE_EMULATION` so local devs can opt into the built-in C fallback without installing SuiteSparse.

3. **Build script prototype**
   - Detect the active feature set at compile time and only compile the C shim when `sparse-suitesparse` is enabled.
   - Emit `cargo:rustc-link-search` and `cargo:rustc-link-lib` directives according to the active platform and environment variables.
   - Re-export informative build messages (`cargo:warning=`) when required headers or libraries are missing so CI failures are obvious.

Deliverables for Phase A: build script merged, documentation for required toolchains, CI smoke job that compiles with `--features sparse-suitesparse` on at least one OS.

---

## Phase B – Core Implementation

1. **Shared data structures**
   - Finalise `CsrMatrixView` validation (shape checks, dtype guards, accessor helpers) so that both SuiteSparse and `sprs` can trust the payload without re-validating.
   - Introduce `SparseBackend` trait (`csrmv`, `csrgemm`, `csradd`, `transpose`) and ensure the dense fallback implements it for all dtypes.

2. **SuiteSparse FFI**
   - Author the minimal C shim (`src/sparse/suitesparse/ffi.c/h`) that:
     - Converts CSR input into SuiteSparse’s CSC representation when needed.
     - Calls into `cs_gaxpy`, `cs_transpose`, and future `cholmod_*` entry points.
     - Normalises error codes to the small status enum consumed by Rust.
   - Expose a safe Rust wrapper (`suitesparse::SuiteSparseBackend`) that:
     - Converts `float32` values into `f64` buffers eagerly (SuiteSparse operates in double precision).
     - Allocates destination buffers with deterministic layouts (row-major dense as expected by `MatrixBuffer`).
     - Maps failure codes to `E_SHAPE_MISMATCH` or `E_NUMERIC_ISSUE` with actionable diagnostics.

3. **Fallback and chaining strategy**
   - Keep the dense fallback as the universal last resort. All exported helpers (`sparse_matmul`, `sparse_matvec`, `sparse_add`, `sparse_transpose`) should attempt SuiteSparse → `sprs` → dense in that order, logging the reason for each downgrade.
   - Track feature availability in metrics so later telemetry can show how often projects run without SuiteSparse.

Deliverables for Phase B: feature-gated SuiteSparse backend merged, dense fallback untouched, public API unchanged (callers remain agnostic to the active backend).

---

## Phase C – Validation & Benchmarking

1. **Unit tests**
   - Extend `packages/core-rs/tests/sparse.rs` to execute against every backend configuration:
     - default (dense fallback),
     - `--features sparse-native`,
     - `--features sparse-suitesparse`,
     - combined (for chained downgrades).
   - Add regression tests for malformed CSR payloads (non-monotonic `row_ptr`, column indices out of bounds) to guarantee the viewer rejects invalid inputs early.

2. **Property tests**
   - Leverage `proptest` to generate random CSR matrices (with controllable sparsity) and confirm that SuiteSparse and `sprs` produce identical results for matvec and gemm.

3. **Benchmarks**
   - Expand `benches/sparse.rs` to include realistic workloads (block sparse matmul, batched matvec, add operations) and record throughput for:
     - dense fallback,
     - `sprs`,
     - SuiteSparse with system BLAS.
   - Automate comparison runs in CI (at least nightly) to detect performance regressions.

4. **Integration testing**
   - Run the N-API binding against SuiteSparse-enabled builds to ensure the FFI path survives the JS boundary.
   - Confirm the WASM bundle still works (should continue using `sprs`).

Deliverables for Phase C: green test suite across feature combinations, benchmark numbers captured in CHANGELOG or internal dashboards.

---

## Phase D – Documentation, Distribution & Release

1. **Documentation**
   - Update `packages/js/docs/design/sparse-matrix.md` with the final API surface and backend behaviour.
   - Create “Getting started with SuiteSparse” instructions (environment variables, required libraries, verification steps).
   - Document fallback behaviour and troubleshooting (e.g. what happens when SuiteSparse is missing at runtime).

2. **Distribution & CI**
   - Extend CI matrix to compile and test both `sparse-native` and `sparse-suitesparse` variants on Linux/macOS/Windows.
   - For N-API releases, publish supplementary platform packages that bundle SuiteSparse binaries or clearly document the manual installation steps.

3. **Release checklist**
   - `cargo fmt`, `cargo clippy --all-targets --all-features`, `cargo test --all-features`, `npm run build:js`.
   - Regenerate sparse benchmarks and capture deltas in the release notes.
   - Announce feature flags and required dependencies prominently in the changelog/readme so downstream users know how to opt in.

By the end of Phase D the sparse stack should be production ready: predictable build tooling, well-documented, covered by tests, and benchmarked against the dense fallback. Subsequent phases (factorisation, iterative solvers, GPU integration) can then build on top of this foundation.
