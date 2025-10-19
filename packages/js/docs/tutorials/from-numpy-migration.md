# Migrating from NumPy to NumJS

> Audience: engineers who already rely on Python + NumPy and want to port data-processing or numerical workloads to TypeScript/JavaScript using `@jayce789/numjs`.

The goal of this guide is to map familiar NumPy concepts to their NumJS equivalents, highlight the practical differences (initialisation, async loading, typing), and provide copy-pasteable examples for the most common migration scenarios.

## Conceptual mapping

| NumPy concept | NumJS equivalent | Notes |
| --- | --- | --- |
| `numpy.ndarray` | `Matrix` | Stored column-major to interoperate efficiently with BLAS/LAPACK. Same semantics as a 2‑D ndarray. |
| `dtype` | `DType` | Names match (`float32`, `float64`, `int32`, …). NumJS adds `fixed64` for decimal fixed-point. |
| Broadcasting | `broadcastTo`, implicit broadcasting in elementwise ops | Follows NumPy’s rules exactly; dimension mismatches surface as `E_SHAPE_MISMATCH`. |
| UFuncs | Top-level helpers (`add`, `exp`, `sin`, …) | Most scalar/matrix operators have a direct equivalent. Missing functions can often be expressed via `map` or pending work in the roadmap. |
| Views / copy-on-write | Lazy copy semantics | Operations return lightweight views when safe; explicit copying is available via `clone()` or `Matrix.from`. |

## Migration checklist

1. **Identify data sources** – Decide whether inputs come from files, network responses, or typed arrays. Pick the appropriate constructor (`readNpy`, `matrixFromBytes`, `new Matrix(typedArray, rows, cols)`).
2. **Port linear algebra kernels** – `matmul`, `svd`, `qr`, `solve`, `pinv` mirror NumPy 1:1. Most changes are mechanical renames.
3. **Handle asynchronous initialisation** – Call `await init()` once near process start to load the backend (N-API or WASM). This is often the biggest structural change compared to Python.
4. **Add TypeScript typings** – Annotate functions to accept `Matrix` or plain typed arrays. Inspect `matrix.dtype` when branching on behaviour.
5. **Recreate test coverage** – Replace `numpy.testing.assert_allclose` with `allClose(matrixA, matrixB, rtol, atol)`. Defaults `DEFAULT_RTOL` and `DEFAULT_ATOL` match NumPy’s double precision.

```ts
// NumPy: c = numpy.matmul(a, b) + bias
import { init, Matrix, matmul, add } from "@jayce789/numjs";

await init();

const a = Matrix.fromArray([1, 2, 3, 4], { rows: 2, cols: 2 });
const b = Matrix.fromArray([5, 6, 7, 8], { rows: 2, cols: 2 });
const bias = Matrix.full(2, 2, 1);

const c = add(matmul(a, b.transpose()), bias);
console.log(c.toArray()); // Float64Array [20, 27, 32, 43]
```

## Typical migration scenarios

### 1. Loading `.npy` / `.npz` files

- Use `readNpy` or `readNpz` from `@jayce789/numjs/io`. Both functions accept `Uint8Array` buffers and return `Matrix` instances while preserving dtype and shape.
- In browsers, fetch the file as an `ArrayBuffer` before passing it to the loader.

```ts
import { readNpy } from "@jayce789/numjs/io";

const buffer = await fetch("/weights.fc1.npy").then(r => r.arrayBuffer());
const matrix = readNpy(new Uint8Array(buffer));
```

### 2. Interoperating with Pandas / Arrow data

- Bridge through Apache Arrow or Polars: convert columns to typed arrays, then wrap them using `Matrix.fromArray`.
- When the dataset is columnar, you can reuse the underlying `Float64Array` without copying, which keeps migration costs low.

### 3. Replacing SciPy linear algebra

- `solve`, `qr`, `svd`, `eigen`, and related operations are implemented in Rust on top of vendor BLAS. Their signatures mirror NumPy for drop-in replacement.
- For features still missing (e.g. sparse solvers, FFT beyond the current coverage) you can keep the Python implementation temporarily and call it via a service or worker until the NumJS equivalent lands.

## Performance tips

- **Prefer the N-API backend** – On Node.js, the native module offers performance comparable to NumPy’s C extensions. Ensure your deployment includes the platform package (`@jayce789/numjs-linux-x64`, etc.).
- **Batch operations** – Group small matrix operations to reduce FFI round trips. Utilities such as `stack` and `concat` help vectorise workloads.
- **Avoid repeated `toArray()`** – Keep results as `Matrix` until you genuinely need a typed array for interoperability.
- **Deterministic randomness** – Use `seedRandom` (or module-specific seeders) to reproduce experiments after migration.

## Additional resources

- Interactive playgrounds: see the [StackBlitz and CodeSandbox guide](../interactive/README.md) for ready-to-run environments.
- Repository examples: `examples/numjs-interactive/src/migration.ts` contains an end-to-end port of a NumPy script.
- Join the discussion: search existing issues/discussions for migration topics or open a new thread with your findings.

If you encounter API gaps or performance regressions during migration, please file an issue—we treat migration blockers as high priority.
