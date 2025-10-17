# numjs Workspace

`numjs` is a multi-language workspace that delivers a NumPy-inspired API for JavaScript and TypeScript. The project combines a Rust core (`packages/core-rs`), a Node.js N-API binding (`packages/napi`), a WebAssembly build (`packages/wasm`), and the published JavaScript wrapper (`packages/js`).

## Workspace Layout

- `packages/core-rs` – Rust implementation of matrix storage, broadcasting, fancy indexing, dtype management, and numerically stable primitives.
- `packages/napi` – Node.js binding built on `napi-rs`, exposing the Rust core as a native module.
- `packages/wasm` – `wasm-bindgen` build that powers the browser/WebAssembly fallback.
- `packages/js` – user-facing npm package that selects the best backend at runtime and provides ergonomic helpers.
- `examples/` – sample scripts and integration entry points.

## Stable Reductions by Default

- `sum` and `dot` now call numerically stable reducers. Native backends perform pairwise accumulation; the WebAssembly/JS path uses Kahan compensation to avoid the worst cancellation cases.
- `sumUnsafe` and `dotUnsafe` remain available for benchmarking or when you explicitly need the historical behaviour.

## Fixed64 (Alpha)

Fixed-point matrices are backed by signed 64-bit integers plus a per-matrix scale.

- Construct via `Matrix.fromFixed` or `matrixFromFixed`, or use the Rust bindings directly.
- Supported operations: construction, elementwise add, `concat`, `stack`, `where`, `take`, `put`, `gather`, `gatherPairs`, `scatter`, `scatterPairs`, `transpose`, and `broadcastTo`.
- Unsupported operations currently throw descriptive errors (e.g. `matmul`, `clip`, `writeNpy`, `writeNpz`). Cast to `float64` before invoking those APIs.
- All operands must share the same scale; mixed-dtype operations promote to `float64`.
- The feature is still in alpha—performance is lower than float dtypes and overflow checks are intentionally conservative. Feedback is encouraged.

## New JavaScript Helpers

- `matrix.transpose()` / `transpose(matrix)` provide zero-copy views across dtypes, including fixed64.
- `broadcastTo(matrix, rows, cols)` expands singleton dimensions with dtype-aware copies and keeps fixed64 scale metadata.

## Error Model & Numeric Tolerances

- Core errors now surface stable codes alongside human-readable messages. The current codes are `E_SHAPE_MISMATCH` (dimension/shape contract failures), `E_NUMERIC_ISSUE` (overflow, NaN/Inf in forbidden contexts), and `E_CHOLESKY_NOT_SPD` (Cholesky factorisation rejected a non-SPD input).
- Both the WASM and N-API backends normalise these codes: thrown errors expose `error.code` while keeping the descriptive message for display/logging.
- Default floating-point comparisons share exported constants `DEFAULT_RTOL = 1e-12` and `DEFAULT_ATOL = 0`. These values balance double-precision arithmetic on the native backend with the WebAssembly/JS fallbacks.
- Minor cross-backend differences remain possible (WebAssembly paths occasionally round through `float32`). When comparing results fetched from different backends, use the shared tolerances or adjust them for tighter/looser acceptance depending on your workload.

## Parallelism & Performance

- Large GEMMs now use cache-aware tiling with adaptive threading. The Rust core combines Rayon with vendor BLAS backends (OpenBLAS/BLIS/MKL) when available.
- Environment variables let you tune concurrency without recompilation:
  - `NUMJS_CPU_THREADS` fixes the thread count for Rayon and BLAS backends.
  - `NUMJS_DISABLE_PARALLEL=1` disables automatic parallel scheduling entirely.
  - `NUMJS_PARALLEL_MIN_ELEMS` / `NUMJS_PARALLEL_MIN_FLOPS` adjust the size thresholds that trigger parallel execution.
- Small matrices automatically fall back to single-threaded execution to avoid oversubscription.
- `matmul_batched` performs batched GEMM across 3D tensors and parallelises over batch items when profitable.
- In browsers, opt-in WASM threads via `await init({ threads: true })` (requires `SharedArrayBuffer` and a cross-origin isolated context). You can cap the worker count with `init({ threads: 4 })`.

## Licensing and Feedback

The project is currently pre-release. File issues or pull requests if you encounter bugs, missing algorithms, or performance regressions.
