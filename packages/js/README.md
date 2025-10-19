# @jayce789/numjs

[![GitHub](https://img.shields.io/badge/github-numjs-24292f?logo=github)](https://github.com/jaycezhang789/numjs)

`@jayce789/numjs` brings a Rust-powered numerical core to the JavaScript and TypeScript ecosystem. It exposes a NumPy-inspired matrix API, negotiates the best available backend (Node.js N-API, WebAssembly, or pure JS fallback), and runs unchanged across Node.js, browsers, Electron, and serverless platforms.

---

## Contents

- [Features at a Glance](#features-at-a-glance)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Everyday Operations](#everyday-operations)
- [Backend Selection](#backend-selection)
- [Using numjs in the Browser](#using-numjs-in-the-browser)
- [Error Handling & Diagnostics](#error-handling--diagnostics)
- [Performance Checklist](#performance-checklist)
- [Fixed-Point Matrices (Fixed64)](#fixed-point-matrices-fixed64)
- [Documentation & Tutorials](#documentation--tutorials)
- [Publishing the Package](#publishing-the-package)
- [Contributing](#contributing)

---

## Features at a Glance

- **Dual backend architecture** – loads the prebuilt N-API addon when available, otherwise streams and instantiates the WebAssembly module. Both expose identical APIs.
- **Rich dtype coverage** – floats, integers, booleans, and an experimental fixed-point (`fixed64`) representation.
- **Copy-on-write semantics** – matrix views are zero-copy until mutation or dtype transitions require new storage.
- **Typed array interop** – dense matrices can be created from or exported to existing typed arrays without copying when layouts align.
- **Numerically stable primitives** – reductions (`sum`, `dot`, etc.) promote to pairwise/Kahan implementations to minimise cancellation.
- **Tree-shakable distribution** – ships ESM and CJS entry points with side-effect-friendly exports for bundlers.
- **Future-ready** – sparse matrices, SuiteSparse integration, WebGPU acceleration, autograd, and probabilistic helpers are actively developed.

## Installation

```bash
npm install @jayce789/numjs
# or
pnpm add @jayce789/numjs
yarn add @jayce789/numjs
```

The install command pulls the JavaScript wrapper plus optional platform-specific packages (e.g. `@jayce789/numjs-linux-x64`). Package managers treat them as `optionalDependencies`, so unsupported platforms fall back automatically.

## Quick Start

```ts
import { init, Matrix, add, matmul, backendKind } from "@jayce789/numjs";

await init(); // loads N-API when available, otherwise falls back to WebAssembly

const a = Matrix.fromArray([1, 2, 3, 4], { rows: 2, cols: 2 });
const b = Matrix.fromArray([5, 6, 7, 8], { rows: 2, cols: 2 });

console.log(backendKind());            // "napi" or "wasm"
console.log(add(a, b).toArray());      // Float64Array [6, 8, 10, 12]
console.log(matmul(a, b.transpose()).toArray());
```

Call `init()` once near process startup (or application mount) to negotiate the backend. Repeated calls are cheap; the loader caches the active backend.

## Everyday Operations

- **Construction**
  - `Matrix.fromArray(data, { rows, cols, dtype? })`
  - `Matrix.eye(size, dtype?)`, `Matrix.zeros(rows, cols, dtype?)`
- **Casting & dtype control**
  - `matrix.astype("float32", { casting: "clip" | "round_floor" | "unsafe" })`
  - Mixed-dtype operators promote according to a deterministic ladder (integers → float32 → float64).
- **Elementwise & reductions**
  - `add`, `sub`, `mul`, `div`, `pow`, `exp`, `log`
  - `sum`, `mean`, `dot`, `norm`, `whereSelect`
- **Linear algebra**
  - `matmul`, `solve`, `svd`, `qr`, `eigen`, `cholesky` (availability depends on backend features such as BLAS/LAPACK support).
- **Shape helpers**
  - `matrix.transpose()`, `transpose(matrix)`
  - `broadcastTo(matrix, rows, cols)`
  - `concat(a, b, axis)` and `stack(a, b, axis)`
- **Output helpers**
  - `matrix.toArray()` – returns a zero-copy view when layout permits.
  - `withOutputFormat({ decimals })` and `round(matrix, decimals)` for display-friendly formatting.

Consult the generated declarations in `dist/index.d.ts` or the API reference in `packages/js/docs` for the full surface area.

## Backend Selection

`init()` follows a deterministic probe order:

1. **N-API backend** – attempts to load the precompiled `.node` binary (located in `dist/bindings/napi/` or in a hoisted platform package).
2. **WebAssembly backend** – downloads and instantiates `dist/bindings/wasm/num_rs_wasm.wasm` if native loading fails.

Use `backendKind()` to inspect the result. To override the detection logic:

```ts
await init({
  preferBackend: "napi",  // "auto" | "napi" | "wasm"
  threads: true,          // enable the WASM thread pool when available
  webGpu: { forceFallback: false } // lazily initialise the experimental WebGPU executor
});
```

When running with the WASM backend you can pass a number to `threads` (e.g. `{ threads: 4 }`) to cap the worker count. Threaded WASM requires `SharedArrayBuffer` and a cross-origin isolated context.

## Using numjs in the Browser

```ts
import { init, Matrix, broadcastTo } from "@jayce789/numjs";

await init(); // streams and instantiates the wasm bundle

const vector = Matrix.fromArray([1, 2, 3], { rows: 3, cols: 1 });
console.log(broadcastTo(vector, 3, 2).toArray());
```

**Bundler tips**

- Configure your bundler (Vite, Webpack, Rollup, Parcel) to treat `.wasm` as an asset. The default project templates already include the necessary loader configuration.
- When targeting browsers without `SharedArrayBuffer`, omit `threads: true` or include a fallback path.
- WebGPU demos require Chromium ≥ 113, or Firefox Nightly with the WebGPU flag.

## Error Handling & Diagnostics

- Errors carry stable codes: `E_SHAPE_MISMATCH`, `E_NUMERIC_ISSUE`, `E_CHOLESKY_NOT_SPD`, etc. Inspect `err.code` instead of parsing message strings.
- Floating-point comparisons share exported tolerances `DEFAULT_RTOL` (`1e-12`) and `DEFAULT_ATOL` (`0`). `isClose` and `allClose` accept overrides when needed.
- Monitor internal copies with `copyBytesTotal()`, `takeCopyBytes()`, and `resetCopyBytes()`. These counters help identify unexpected buffer materialisations.
- `backendKind()` and future `enableDebugLogging()` hooks surface loader decisions and fallback reasons.

## Performance Checklist

- Deploy the N-API backend whenever possible; it leverages native BLAS implementations and Rayon for parallelism.
- Batch small operations to reduce overhead when crossing the JS ↔ Rust boundary.
- Avoid eagerly calling `toArray()` unless you truly need a plain typed array—stay in `Matrix` form for subsequent operations.
- Use the benchmarking scaffolding in `packages/core-rs/benches` (Criterion) or the JS profiling scripts in `examples/numjs-interactive` to validate performance regressions.

## Fixed-Point Matrices (Fixed64)

`fixed64` matrices store signed 64-bit integers plus a shared decimal scale. They are ideal for scenarios requiring deterministic decimal arithmetic (financial calculations, exact rounding rules).

- Create via `Matrix.fromFixed(data, { rows, cols, scale })` or by casting (`matrix.astype("fixed64", { scale })`).
- Supported today: construction, elementwise add, `concat`, `stack`, `where`, `take`, `put`, `gather`, `scatter`, `transpose`, `broadcastTo`, and explicit casts.
- All operands must use the same scale; mixed-dtype operations promote to `float64`.
- Unsupported operations (`matmul`, `clip`, `writeNpy`, `writeNpz`) currently throw descriptive errors—cast to `float64` before invoking them.
- The feature remains experimental; overflow checks are intentionally conservative. Feedback is encouraged before stabilisation.

## Documentation & Tutorials

The `packages/js/docs` directory hosts in-depth guides:

- [`docs/tutorials/from-numpy-migration.md`](docs/tutorials/from-numpy-migration.md) – step-by-step migration from NumPy ecosystems.
- [`docs/tutorials/backends.md`](docs/tutorials/backends.md) – deep dive into backend detection, environment quirks, and troubleshooting.
- [`docs/tutorials/webgpu.md`](docs/tutorials/webgpu.md) – enabling and tuning the experimental WebGPU executor.
- [`docs/interactive/README.md`](docs/interactive/README.md) – StackBlitz/CodeSandbox playgrounds and cloneable demos.
- [`docs/design/sparse-matrix.md`](docs/design/sparse-matrix.md) – CSR-first sparse matrix architecture and SuiteSparse integration plan.
- [`docs/future.md`](docs/future.md) – autograd, random distributions, sparse roadmap, and other upcoming features.

Explore the examples under `examples/numjs-interactive/` for runnable code snippets that mirror the documentation.

## Publishing the Package

Run the full build before publishing so consumers receive both backends:

```bash
npm run build
npm publish --workspace packages/js
# add --access public on first publish if required
```

Ensure CI has produced the N-API binaries for every supported platform prior to publishing.

## Contributing

- Star or fork the repository: <https://github.com/jaycezhang789/numjs>
- File issues for bugs, feature requests, or usability improvements.
- Pull requests are welcome—please run `npm run build` (and any relevant tests) before submitting.
- Documentation contributions are just as valuable as code; see the guides above for context.

---

Questions, feedback, or ideas? Reach us via [GitHub issues](https://github.com/jaycezhang789/numjs/issues). We appreciate community involvement and respond as quickly as we can.
