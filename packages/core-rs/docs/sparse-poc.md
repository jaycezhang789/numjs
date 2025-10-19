# Sparse Proof-of-Concept with `sprs`

This note documents the original proof-of-concept that validated the `sprs` crate as a viable sparse backend for `num_rs_core`. While we now have a multi-backend architecture (SuiteSparse, `sprs`, and dense fallback), the experiment remains a useful reference for contributors who want to understand how CSR data flows through the system.

## Prerequisites

- Rust toolchain (1.74 or newer recommended).
- `sprs = "0.11"` is already listed as an optional dependency; the example activates it via the `sparse-native` feature.
- The repository must be built with `--features sparse-native` so that the `sprs` backend is compiled.

## Running the example

```bash
cargo run --example sparse_poc --features sparse-native
```

What the example does:

1. Builds a 3×3 CSR matrix using `sprs::CsMat` (two non-zero entries per row) to keep the math easy to audit.
2. Wraps the CSR payload in `num_rs_core::sparse::CsrMatrixView`, triggering the same validation logic production code uses.
3. Calls into the public helpers:
   - `sparse_matmul` (CSR × dense column vector),
   - `sparse_add` (CSR + dense matrix),
   - `sparse_transpose`.
4. Prints the results and compares them against the dense baseline computed with `sprs` alone. Any discrepancy causes the example to panic.

You can tweak the matrices or add additional checks to explore other operations—`sparse_poc.rs` is intentionally small and self-contained.

## Takeaways

- **API compatibility** – `CsrMatrixView` provides the minimal metadata (rows, cols, dtype, indptr, indices, values) required by both `sprs` and SuiteSparse. The example mimics the data flow that happens inside the real backend dispatcher.
- **Fallback strategy** – even when `sparse-native` is enabled the public helpers still fall back to the dense implementation if the backend errors out. You can simulate this by forcing `sprs` to reject the input (e.g. feed an invalid `row_ptr`) and observing the logged downgrade.
- **Cross-compilation** – the PoC runs on every target supported by Rust because `sprs` is a pure-Rust dependency. This makes it ideal for WASM builds and CI smoke tests.

## Where to go from here

- Use the example as a template for additional regression tests. For instance, you can port the logic into `tests/sparse.rs` to cover more edge cases.
- Extend the example to benchmark `sprs` vs. the dense fallback (`MatrixBuffer::from_vec`). Enable Criterion to capture runtime numbers.
- Investigate larger, randomly generated sparse matrices (e.g. with `proptest`) to stress-test the `CsrMatrixView` validation.

As SuiteSparse integration matures, we will add a sibling example (`cargo run --example suitesparse_poc --features sparse-suitesparse`) so contributors can quickly verify their local native setup.
