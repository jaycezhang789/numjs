# Looking Ahead: Upcoming Features & Research Directions

NumJS already covers dense matrix operations, dual N-API/WASM backends, and an experimental WebGPU executor. This document highlights the initiatives currently on the roadmap so the community knows where contributions are most welcome. Nothing here is set in stone—please open issues or discussions if your use case requires a specific feature.

## Automatic differentiation (Autograd)

- **Computation graph** – Design a lightweight graph representation that interoperates with existing `Matrix` objects. Distinguish leaf nodes (constants / inputs) from intermediate nodes (operator outputs) and support immediate execution or deferred graph construction.
- **Forward/backward rules** – Implement dual operations for the core operator set (`add`, `sub`, `mul`, `div`, `matmul`, `broadcastTo`, `sum`, `exp`, `log`, etc.). Provide a `backward()` entry point that performs reverse-mode accumulation.
- **Gradient storage** – Attach an optional `grad` view to `Matrix` so gradients can be accumulated or reset in place. Include gradient checking utilities to detect NaNs or exploding values during long training runs.
- **Optimiser examples** – Ship reference implementations of `sgd`, `adam`, and `rmsprop` to validate the autograd stack on small models (linear regression, multilayer perceptrons).
- **Backend harmony** – Ensure gradients computed via N-API, WASM, and WebGPU produce identical results. Introduce mixed-precision or fallback modes when necessary.

## Randomness and probability distributions

- **PRNG engines** – Implement well-tested generators such as PCG-XSL-RR and Philox with 32/64-bit output, seed control, and jump-ahead support.
- **Reproducible streams** – Provide counter-based APIs that split streams deterministically across threads, WebWorkers, and GPU kernels.
- **Distribution toolkit** – Offer vectorised samplers for `uniform`, `normal`, `bernoulli`, `poisson`, and more, returning `Matrix` objects directly.
- **Statistical diagnostics** – Include helpers such as `chiSquareTest` and `ksTest` to evaluate sample quality. Emit seed metadata with failures for easy reproduction.
- **Autograd hooks** – Reserve extension points so probabilistic models can register reparameterised gradients in future releases.

> These features are under active design. If you need a particular function urgently, please file an issue or submit a draft PR to influence prioritisation.

## Sparse matrices and SuiteSparse integration

- **Data structures** – Finalise the `SparseMatrix` abstraction with CSR/CSC/COO storage and seamless conversion to/from dense `Matrix`.
- **Core operators** – Provide accelerated sparse–dense matmul, sparse add, transpose, slicing, and stacking. WASM will reuse `sprs`; N-API will call into SuiteSparse.
- **Distribution** – Extend optional dependencies to bundle SuiteSparse binaries via platform packages where licensing allows. Document manual installation for other environments.
- **Fallback story** – Preserve pure Rust/JS fallbacks and surface warnings when performance will be degraded.

## Large data & out-of-core processing

- **Memory mapping** – Expose Node.js-friendly `mmap` utilities so CSV/Parquet/Arrow chunks can be accessed as `Matrix`/`SparseMatrix` storage without copying.
- **Chunked algorithms** – Offer helpers like `chunkedMap`, `chunkedReduce`, and `outOfCoreMatmul` to keep memory footprints stable.
- **Schedulers** – Add multi-threaded / multi-worker pipelines with progress callbacks and cancellation support.
- **Arrow/Parquet integration** – Allow columns or sparse structures to be mapped directly from Arrow record batches.

## Numerical robustness

- **Accurate reductions** – Add Kahan and pairwise summation options for `sum`, `mean`, `var`, automatically enabling them when instability is detected.
- **Stable softmax** – Introduce `stableSoftmax`, `logSumExp`, and related helpers that subtract the maximum value and avoid overflow by design.
- **Diagnostics** – Provide utilities such as `detectCancellation` and `numericStats` to inspect intermediate tensors for numerical issues.

## Mixed precision & tolerance control

- **New dtypes** – Introduce `float16`, `bfloat16`, and `tf32` descriptors with cast helpers. Execute them in native precision on GPU/N-API backends where possible.
- **Automatic precision selection** – Choose computation precision based on matrix size and user-specified error budgets, with fallbacks to `float32`/`float64` when required.
- **Tolerance configuration** – Extend `init({ tolerances })` and add `withNumericTolerance` to make error bounds explicit at the API level.
- **WebGPU support** – Detect device capabilities at runtime and select the appropriate shader variant (`f16` vs `f32`).

## JIT compilation & custom operators

- **Cranelift JIT** – On Node/N-API, use Cranelift to JIT compile user-defined elementwise kernels (`jitCompile(fn, signature)` style).
- **WGSL code generation** – In WebGPU/WASM builds, template WGSL kernels for elementwise, reduction, and broadcast operations to enable zero-copy custom operators.
- **Sandboxing** – Constrain JIT inputs, add caching, and produce diagnostic logs to keep custom kernels safe.
- **Autograd integration** – Allow custom operators to register gradient implementations so they participate in the automatic differentiation graph.

The roadmap evolves alongside community feedback. If you would like to own a specific milestone or propose an alternative approach, please open a discussion—we are eager to collaborate.
