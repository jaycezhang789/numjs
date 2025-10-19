# SparseMatrix Architecture

This document outlines the design goals, data structures, and backend integration plan for the sparse matrix stack in `@jayce789/numjs`. It is intentionally detailed so that both maintainers and external contributors can understand how the JavaScript API relates to the Rust core (`num_rs_core`) and to the optional native dependencies (SuiteSparse and `sprs`).

## Data model

- **Primary type: `SparseMatrix`**
  - Canonical in-memory representation is CSR (Compressed Sparse Row). Internally we store:
    - `rowPtr: Uint32Array` – length is `rows + 1`, monotonic, final entry equals `nnz`.
    - `colIdx: Uint32Array` – length `nnz`, column indices in `[0, cols)`.
    - `values: Float32Array | Float64Array | BigInt64Array` – payload matches the declared `dtype`.
  - Convenience constructors convert alternative formats into CSR:
    - `fromCSR(rowPtr, colIdx, values, shape, dtype?)` – zero-copy when buffers are already CSR.
    - `fromCSC(colPtr, rowIdx, values, shape, dtype?)` – transposes during construction.
    - `fromCOO(indices, values, shape, dtype?)` – sorts and compresses coordinate triples.
  - Export helpers never mutate internal buffers:
    - `toCSR()` returns `{ rowPtr, colIdx, values }` views.
    - `toCOO()` materialises coordinate lists for interoperability (e.g. with SciPy).
    - `toDense()` is available for debugging or fallbacks, but clearly documented as O(rows × cols).

- **Supported dtypes**
  - `float32` and `float64` are first-class and map directly onto SuiteSparse / `sprs`.
  - `int32` and `bool` are supported through conversion utilities (convert to `float64` internally).
  - `fixed64` mirrors the dense `Matrix` capability; conversion preserves scale metadata during CSR rebuilds.

- **Memory views**
  - Node.js / N-API: the wrapper passes the typed arrays directly into Rust, letting the active backend (`SuiteSparseBackend`, `SprsBackend`, or fallback) access them without copying.
  - WASM: the wasm bundle uses `sprs` compiled to wasm or a pure-JS fallback when compiling `sprs` is not feasible.
  - `SparseMatrix` keeps the buffers immutable from the JS perspective; structural changes are performed by allocating new instances.

## API surface

```ts
export type SparseFormat = "csr" | "csc" | "coo";

export interface SparseMatrixInit {
  format: SparseFormat;
  data: {
    rowPtr?: Uint32Array;
    colPtr?: Uint32Array;
    indices: Uint32Array;
    values: Float32Array | Float64Array | BigInt64Array;
  };
  shape: { rows: number; cols: number };
  dtype?: DType;
}

export class SparseMatrix {
  constructor(init: SparseMatrixInit);
  static fromCSR(args: CSRInitArgs): SparseMatrix;
  static fromCOO(args: COOInitArgs): SparseMatrix;
  static fromDense(matrix: Matrix, options?: { zeroThreshold?: number }): SparseMatrix;

  readonly rows: number;
  readonly cols: number;
  readonly nnz: number;
  readonly dtype: DType;
  readonly format: SparseFormat;

  toCSR(): { rowPtr: Uint32Array; colIdx: Uint32Array; values: TypedArray };
  toCOO(options?: { sort?: boolean }): { rowIdx: Uint32Array; colIdx: Uint32Array; values: TypedArray };
  toDense(): Matrix;
}
```

### Sparse–dense operations

| Operation | Description |
| --- | --- |
| `sparseMatmul(sparse: SparseMatrix, dense: Matrix, opts?)` | CSR × dense multiply. Delegates to SuiteSparse (`csrgemm`) or `sprs`, with dense fallback. Supports both `sparse @ dense` and `dense @ sparse.transpose()`. |
| `sparseMatvec(sparse: SparseMatrix, vector: Matrix | TypedArray)` | Specialised matvec fast path. Enforces column-vector shape on input, returns a dense column vector. |
| `sparseAdd(sparse: SparseMatrix, dense: Matrix)` | Elementwise addition. Uses backend acceleration when available, otherwise densifies the sparse operand once. |
| `sparseTranspose(sparse: SparseMatrix, format?: "csr" | "csc")` | Returns a new `SparseMatrix` in the requested storage format without touching values. |
| `sparseDiag(values)` / `sparseEye(size)` | Utility constructors for common sparse patterns (diagonal and identity matrices). |

The JS API mirrors the Rust helpers (`num_rs_core::sparse::*`). Additional operations (slicing, stacking, block assembly) are part of the backlog and will share the same backend dispatch path.

### Autograd compatibility

Phase 1 focuses on forward computations. Autograd support will arrive in a later milestone via one of two approaches:

1. Sparse-aware gradient accumulation (store gradients as CSR when the sparsity structure is static).
2. On-demand densification before backprop (suitable for small matrices).

The public API is designed so that `SparseMatrix` can be adapted to `Tensor` when autograd is ready—every method returns either `Matrix` or `SparseMatrix`, making it easy to plug into mixed pipelines.

## Backend integration plan

| Phase | Objective | Implementation notes |
| --- | --- | --- |
| Phase 1 | Implement `SparseMatrix` in JS with CSR storage and dense fallback for all operations. | TypedArray-backed CSR; operations executed in JS/TS or via dense helpers. |
| Phase 2 | Wire the N-API binding to the Rust sparse layer (`sprs` → SuiteSparse). | `num_rs_core` exposes `sparse_matmul`, `sparse_matvec`, `sparse_add`, `sparse_transpose`. N-API marshals buffers directly. |
| Phase 3 | WASM backend support. | Compile `sprs` to wasm where possible. Otherwise ship a pure JS fallback tuned for medium-sized matrices. |
| Phase 4 | Out-of-core & advanced features. | Memory mapping for gigantic CSR matrices, chunked execution helpers, Arrow/Parquet ingestion. |

> The WASM build currently uses `sprs` in emulation mode. Native SuiteSparse is not yet available on wasm32, but we keep the API identical so later upgrades are drop-in.

## Ecosystem interoperability

- **SuiteSparse** – Backend of choice on platforms where native libraries are available. Primary entry points are `cs_gaxpy` (`csrmv`), `cs_sparse` × `dense` multiplication, and (future) `cholmod` for factorisation.
- **sprs** – Pure Rust fallback used for WASM and for development. Provides identical semantics for CSR operations.
- **Arrow/Parquet** – Planned interoperability path: convert Arrow’s `SparseCSRMatrix` representation into `SparseMatrix` without copying.
- **Graph libraries** – Keep column indices as 32-bit to align with common graph datasets. Provide adapters for libraries such as `graphlib` or WebGPU-based solvers.

## Testing strategy

- **Unit tests** – Focus on round-tripping CSR ↔ COO, verifying shape/dtype metadata, and ensuring sparse–dense operations match dense baselines.
- **Integration tests** – Use existing high-level APIs (e.g. linear regression demo) with sparse inputs to guarantee that fallback logic works in real scenarios.
- **Backend parity tests** – Run the same test suite against all backend combinations (`auto`, `sprs`, `SuiteSparse`, fallback) and compare results bit-for-bit.
- **Property tests** – Use `proptest` on the Rust side to generate random CSR matrices and confirm invariants for `CsrMatrixView`.

## Immediate next steps

1. Implement the `SparseMatrix` class in `packages/js/src`, including format conversion utilities and metadata validation.
2. Expose JS fallbacks for `sparseMatmul`, `sparseAdd`, `sparseTranspose`, and `sparseMatvec`, reusing the Rust helpers when the backend is available.
3. Extend the documentation tutorials with end-to-end sparse examples (construction, matmul, exporter).
4. Finish bundling SuiteSparse in the N-API distribution or document the manual installation path.
5. Prototype `mmapSparseCSR(path)` for large datasets and add chunked execution helpers to the roadmap.

Contributors interested in the sparse stack are encouraged to read `packages/core-rs/docs/sparse-roadmap.md` for the full backend plan and `packages/core-rs/docs/sparse-poc.md` for a minimal example using `sprs`.
