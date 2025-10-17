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

## Licensing and Feedback

The project is currently pre-release. File issues or pull requests if you encounter bugs, missing algorithms, or performance regressions.
