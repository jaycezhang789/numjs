# Changelog

## Unreleased

- Nothing yet.

## 0.2.0 - 2025-10-24

### Added

- Switched CUDA GEMM entry points to cuBLAS with transpose, batched, and strided support, plus alias detection for in-place mistakes.
- Implemented two-pass CUDA reductions for `reduce_sum_f32`, `reduce_max_f32_with_policy`, and `argmax_f32_with_policy`, exposing `gpu::NanPolicy` and host/device helpers while avoiding NVRTC header lookup issues.
- Added Criterion harness `cargo bench -p num_rs_core --features gpu-cuda --bench gpu_vs_cpu` to compare GPU and CPU performance.
- Added `sparse-native` feature gate in `num_rs_core::sparse`, enabling native CSR operations via sprs with automatic fallback.
- Introduced CSR unit tests (`packages/core-rs/tests/sparse.rs`) covering validation and dense parity.
- Added Criterion benchmark `cargo bench --bench sparse` for comparing fallback vs native sparse kernels.
- Updated N-API sparse handlers to reuse `CsrMatrixView`, ensuring consistent dtype handling.
- Documented sparse build/test commands in `README.md` and roadmap docs.
