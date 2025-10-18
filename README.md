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

## GPU Acceleration (Beta)

- **WebGPU hot paths.** `matmulAsync`, `sumAsync`, and the new `conv2d` helper offload to WebGPU compute shaders when `navigator.gpu` is available. The JavaScript wrapper coordinates scheduling, automatically falling back to WASM/JS when WebGPU initialisation fails. These APIs always resolve to the same results as their synchronous counterparts, so you can feature-detect with `if (webGpuAvailable())`.
- **Node CUDA binding.** The N-API module now exposes `gpuAvailable()`, `gpuBackendKind()`, `gpuMatmul()`, and `gpuSum()`. When the binary is built with `--features gpu-cuda`, the wrapper will try the CUDA kernels first and transparently return to the CPU implementation if the GPU is missing or rejects a workload.
- **Build switches.** CUDA support is off by default. Enable it by rebuilding the native addon with:

  ```shell
  cargo build -p num_rs_napi --release --features gpu-cuda
  ```

  The JS package will automatically detect the accelerated backend once the addon is on the module search path.

- **Usage example.**

  ```ts
  import { init, Matrix, matmulAsync, gpuBackendKind } from "@jayce789/numjs";

  await init();
  console.log("GPU:", gpuBackendKind() ?? "cpu");

  const a = new Matrix(new Float32Array([1, 2, 3, 4]), 2, 2, { dtype: "float32" });
  const b = new Matrix(new Float32Array([4, 3, 2, 1]), 2, 2, { dtype: "float32" });

  const accelerated = await matmulAsync(a, b);
  console.log(accelerated.toArray()); // falls back gracefully if GPU is unavailable
  ```

- **Fallback guarantees.** Both the WebGPU and CUDA paths are optional. When a GPU path fails initialisation or throws, the library reverts to the existing WASM/native implementations without changing return types or throwing additional errors.

## Arrow & Polars Interop (Preview)

- **Current behaviour.** The newly added bridges in `packages/js/src/io` let you convert between `Matrix` instances and Apache Arrow (browser/Node) or Polars DataFrames (Node). Right now the helpers materialise intermediate row-major buffers when moving from columnar Arrow/Polars data into NumJS matrices. As a result, these conversions are **not** zero-copy yet.
- **Zero-copy roadmap.** The Rust core already stores row/column strides, but to avoid copies we need a stride-aware `Matrix` view pipeline that can stitch together Arrow column buffers without re-packing them. That work is tracked and will land before we stabilise the IO API.
- **Optional dependencies.** These bridges rely on external packages which are not bundled by default:
  - Browser/Node: install `apache-arrow` alongside `@jayce789/numjs`.
  - Node (Polars): install `nodejs-polars` (preferred) or the pure JS `polars` package.
  If the modules are missing, the helpers throw informative errors explaining which dependency to add.
- **CI follow-up.** Our GitHub Actions matrix will gain conditional jobs that install these optional dependencies and run interop smoke tests once the zero-copy pipeline lands. Until then the code paths remain best-effort and are guarded by runtime type-checks.

## DataFrame View & CSV/Parquet IO (Preview)

- **Lightweight DataFrameView.** `DataFrameView` wraps a 2D `Matrix` and tracks column names plus per-column dtypes. It supports column selection, renaming, per-column dtype overrides, and projection back to plain matrices/row objects.
- **CSV helpers.** New Node utilities `readCsvDataFrame` / `writeDataFrameToCsv` load and persist numeric/boolean tables. Browser code can feed a `ReadableStream<Uint8Array>` into `readCsvDataFrameFromStream` for progressive parsing. The parser currently targets numeric and boolean columns; non-numeric cells become `NaN` and can be coerced later.
- **Parquet helpers.** On Node, `readParquetDataFrame` / `writeParquetDataFrame` integrate with Polars. Install `nodejs-polars` (or `polars`) and the helpers will bridge through the existing Polars converters. Without the dependency the functions raise clear guidance.
- **Interoperability.** `DataFrameView` cooperates with the Arrow/Polars bridges, so you can move between CSV ↔︎ DataFrameView ↔︎ Arrow/Parquet with minimal glue. The class deliberately keeps the underlying `Matrix` in `float64`; dtype hints drive conversions when you request individual columns.

## Image & Signal Processing (Preview)

- **Spatial kernels.** `im2col`, `maxPool`, and `avgPool` provide the building blocks for classic convolution pipelines and pooling layers. `sobelFilter` (returns gradient X/Y plus optional magnitude) and `gaussianBlur` (kernel derived from `sigma`) wrap the existing `conv2d` accelerator so they benefit from WebGPU in the browser or CUDA on Node.
- **FFT utilities.** New helpers `fftAxis`, `fft2d`, `ifftAxis`, `ifft2d`, and `powerSpectrum` expose the Rust `rustfft` backend through both the WASM and N-API backends. They return complex-valued pairs (`real`/`imag` matrices) so you can hop between spatial and frequency domains without leaving the NumJS API.

## Python & ONNX Interop (Preview)

- **Python bridge.** `runPythonScript` executes inline Python via the system interpreter (configurable `pythonPath`/`env`/`args`) and returns captured stdout/stderr. `pythonTransformMatrix` streams a matrix as JSON into Python and expects JSON back, making it easy to validate kernels against NumPy/SciPy. Install CPython and author scripts that read from stdin; if Python is missing, the helpers throw guidance.
- **ONNX Runtime.** `loadOnnxModel` dynamically imports `onnxruntime-node`, creates an inference session, and exposes a lightweight `OnnxModel` wrapper. Feed `Matrix` or typed arrays to `model.run` and receive output matrices. The dependency is optional—if it’s absent the helper raises an informative error.
- **Safety defaults.** Both bridges are Node-only and lazy-load their optional dependencies. No extra code ships to the browser bundles unless you explicitly call the APIs.

## Debugging & Diagnostics

- Set `DEBUG=1` before invoking `init()` to enable verbose logging. NumJS reports backend selection, fallback decisions (N-API → WASM → WebGPU), and emits low-level metrics to help you diagnose performance and interop issues.
- Use `init({ preferBackend: "napi" | "wasm", onBackendReady(info) { … } })` to steer backend selection. When a requested backend fails to load, NumJS logs guidance—e.g., installing the appropriate N-API binary or enabling WebGPU—and automatically moves down the fallback chain.
- Debug mode also captures timing/fallback information during GPU initialisation and Python/ONNX integrations, making it easier to follow the control flow when bridging ecosystems.

## Testing & Benchmarks

- **Property tests.** The Rust core uses [`proptest`](https://docs.rs/proptest) to exercise arithmetic and matmul kernels across random shapes, while the JavaScript wrapper mirrors this with [`fast-check`](https://github.com/dubzzz/fast-check). Run `cargo test -p num_rs_core` and `npm --prefix packages/js run test` to execute both suites.
- **Backend consistency.** JavaScript tests spawn N-API and WASM processes to ensure numerical parity (within tolerance) across the two runtimes. Skips automatically if a backend is missing on the host.
- **Benchmarks.** `npm --prefix packages/js run bench` records micro/macro benchmarks (matmul, convolution, pooling pipelines) and emits an HTML report under `tmp/numjs-bench.html` so you can track performance regressions over time.
- **Sparse matrix validation.** `cargo test --features sparse-native` exercises the CSR-native path (fallback tests run without the feature). Compare fallback/native performance via `cargo bench --bench sparse [--features sparse-native]`.

## Licensing and Feedback

The project is currently pre-release. File issues or pull requests if you encounter bugs, missing algorithms, or performance regressions.
