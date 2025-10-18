# Changelog

## Unreleased

- Added `sparse-native` feature gate in `num_rs_core::sparse`, enabling native CSR operations via sprs with automatic fallback.
- Introduced CSR unit tests (`packages/core-rs/tests/sparse.rs`) covering validation and dense parity.
- Added Criterion benchmark `cargo bench --bench sparse` for comparing fallback vs native sparse kernels.
- Updated N-API sparse handlers to reuse `CsrMatrixView`, ensuring consistent dtype handling.
- Documented sparse build/test commands in `README.md` and roadmap docs.
