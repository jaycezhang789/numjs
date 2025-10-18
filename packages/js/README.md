# @jayce789/numjs

[![GitHub](https://img.shields.io/badge/github-numjs-24292f?logo=github)](https://github.com/jaycezhang789/numjs)

`@jayce789/numjs` brings a Rust-powered numerical core to the JavaScript ecosystem, delivering a NumPy-inspired matrix API. The package ships with a high-performance Node N-API backend and a WebAssembly fallback, automatically loading the best option at runtime so the same code runs in Node.js and modern browsers.

## Table of Contents
- [Project Links](#project-links)
- [Installation](#installation)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
  - [Common Operations](#common-operations)
- [文档与教程](#文档与教程)
- [Backend Strategy](#backend-strategy)
- [Browser Usage](#browser-usage)
- [Node.js Compatibility](#nodejs-compatibility)
- [Error Handling & Diagnostics](#error-handling--diagnostics)
- [Performance Tips](#performance-tips)
- [Output & Rounding (Display Only)](#output--rounding-display-only)
- [Stable Reductions](#stable-reductions)
- [Fixed64 (alpha)](#fixed64-alpha)
- [Publishing](#publishing)
- [Contributing](#contributing)

## Project Links
- **Repository:** <https://github.com/jaycezhang789/numjs>
- **Issue Tracker:** <https://github.com/jaycezhang789/numjs/issues>

## Installation
```bash
npm install @jayce789/numjs
# or
pnpm add @jayce789/numjs
yarn add @jayce789/numjs
```

## Key Features
- Unified API that works in Node.js and the browser.
- Automatic backend selection with graceful fallback when a native binary is unavailable.
- Rich dtype support covering floats, integers, booleans, and experimental fixed-point data.
- Rust core for predictable performance and copy-on-write semantics.
- Tree-shakable distribution that ships both ESM and CJS bundles.
- Zero-copy typed array views for Float32/Float64 matrices on the WASM backend, plus SIMD-accelerated fused kernels for elementwise math and matmul.

## Quick Start
```ts
import { init, Matrix, add, matmul, backendKind } from "@jayce789/numjs";

await init(); // loads N-API when available, otherwise uses WebAssembly

const a = new Matrix([1, 2, 3, 4], 2, 2);
const b = new Matrix([5, 6, 7, 8], 2, 2);

console.log(backendKind()); // "napi" or "wasm"
console.log(add(a, b).toArray());    // Float64Array [6, 8, 10, 12]
console.log(matmul(a, b.transpose()).toArray());
```

### Common Operations
- `Matrix.fromArray(data, rows, cols, dtype?)` to build matrices from plain arrays or typed arrays.
- `matrix.astype("float32", { casting: "round_floor|clip" })` to cast between dtypes with explicit rounding/overflow strategies.
- `whereSelect(condition, truthy, falsy)` for broadcasted conditional selection.
- `concat(matrixA, matrixB, axis)` / `stack(matrixA, matrixB, axis)` for joins with dtype promotion.
- `matrix.transpose()` / `transpose(matrix)` for zero-copy axis swaps.
- `broadcastTo(matrix, rows, cols)` to expand singleton dimensions (works for fixed64 as well).
- `sum(matrix)` and `dot(a, b)` now default to numerically stable reducers.
- Linear algebra helpers such as `svd`, `qr`, `solve`, and `eigen` when supported by the active backend.

Refer to the generated TypeScript declarations in `dist/index.d.ts` for the complete surface area.

## 文档与教程

- [从 NumPy 迁移指南](docs/tutorials/from-numpy-migration.md)：对照表与迁移实践，帮助 Python 团队快速上手。
- [WASM 与 N-API 后端选择与原理](docs/tutorials/backends.md)：剖析双后端加载流程、手动控制与常见故障排查。
- [WebGPU 加速篇](docs/tutorials/webgpu.md)：介绍如何启用 GPU 管线与性能调优策略。
- [StackBlitz/CodeSandbox 交互式文档](docs/interactive/README.md)：在线运行 Playground，附复制即跑 Demo。

## Backend Strategy
`init()` negotiates the backend in the following order:
1. **N-API backend** – loads the precompiled `.node` binary shipped under `dist/bindings/napi/`.
2. **WebAssembly backend** – falls back to `dist/bindings/wasm/num_rs_wasm.*` when the native module is unavailable (for example in the browser).

You can inspect the decision with `backendKind()` or provide your own loader if you need custom logic.

- When the WASM backend is active, call `await init({ threads: true })` to spin up a WebAssembly thread pool (requires `SharedArrayBuffer` and `crossOriginIsolated`). Pass a number (for example `init({ threads: 4 })`) to clamp the worker count.

## Browser Usage
```ts
import { init, Matrix, broadcastTo } from "@jayce789/numjs";

await init(); // downloads and instantiates the wasm module
const vector = new Matrix([1, 2, 3], 3, 1);
console.log(broadcastTo(vector, 3, 2).toArray());
```
Ensure your bundler knows how to serve `.wasm` assets (Vite, Webpack, Rollup, etc.).

On the WASM backend, `matrix.toArray()` now reuses the underlying memory for `float32`/`float64` matrices whenever the layout is contiguous, returning a typed array view instead of copying.

## Node.js Compatibility
- Supports Node.js 16 and newer.
- Bundles CommonJS (`dist/index.cjs`) and ES Module (`dist/index.js`) entry points.
- Works in Electron/NW.js – when the native module cannot be loaded the code automatically falls back to WebAssembly.

## Error Handling & Diagnostics
- Core operations surface structured error codes (`E_SHAPE_MISMATCH`, `E_NUMERIC_ISSUE`, `E_CHOLESKY_NOT_SPD`) so you can branch on failures without parsing strings. Errors thrown from the package expose `error.code` alongside human-readable messages.
- Use the shared tolerances `DEFAULT_RTOL` (`1e-12`) and `DEFAULT_ATOL` (`0`) for backend-neutral floating-point comparisons. Adjust them when porting workloads that demand tighter or looser error bounds.
- The helper functions `isClose` and `allClose` accept optional overrides if you need to change tolerances, and default to the shared constants above.
- When debugging backend choices or performance, pair diagnostics with `backendKind()` and the copy-byte counters detailed below.
- For memory investigations, `copyBytesTotal()`, `takeCopyBytes()`, and `resetCopyBytes()` provide visibility into internal copies.

## Performance Tips
- Prefer the N-API backend for heavy workloads; prebuild binaries for your deployment targets when possible.
- Use `copyBytesTotal()`, `takeCopyBytes()`, and `resetCopyBytes()` to monitor internal copies when tuning code.

## Output & Rounding (Display Only)
The library keeps computations in native float dtypes but lets you control presentation.
- Global configuration via `setOutputFormat({ as, decimals, scale, ... })`.
- Scoped overrides via `withOutputFormat(opts, fn)` or `scopedOutputFormat(opts)`.
- Helpers such as `toOutputArray`, `toOutput2D`, and instance `matrix.toString()` make printing predictable.
- Optional `round(matrix, decimals)` returns a new matrix with quantised values when you need to clamp results before exporting.

## Stable Reductions
- `sum(matrix)` and `dot(a, b)` now default to numerically stable implementations. The N-API backend performs native pairwise summation; the WebAssembly/JS fallback uses Kahan compensation.
- `sumUnsafe` and `dotUnsafe` remain available when you need the historical linear accumulators for performance comparisons.

## Fixed64 (alpha)
Fixed-point matrices are represented as signed 64-bit integers plus a per-buffer decimal scale.
- Construct via `Matrix.fromFixed(data, rows, cols, scale)` or the helper `matrixFromFixed(...)`.
- Supported operators today: construction, elementwise add, `concat`, `stack`, `where`, `take`, `put`, `gather`, `gatherPairs`, `scatter`, `scatterPairs`, `matrix.transpose()`, `broadcastTo`, and explicit dtype casts.
- All operands must share the same scale; mixed-dtype operations promote to `float64`.
- Unsupported paths (currently throw informative errors): `matmul`, `clip`, `writeNpy`, and `writeNpz`. Convert to `float64` before invoking those APIs.
- The feature is still experimental; performance is lower than native floats and overflow checks are deliberately conservative. Feedback is welcome before promoting it to a stable contract.

## Publishing
Before publishing to npm, build all artefacts so consumers receive both backends:
```bash
npm run build
npm publish --workspace packages/js
```
Add `--access public` on first publish if the package is not yet public.

## Contributing
- Star or fork the project on GitHub: <https://github.com/jaycezhang789/numjs>
- File issues for bugs, feature requests, or usability feedback.
- Pull requests are welcome—please run `npm run build` and any relevant tests before submitting.

---
Questions or ideas? Issues and pull requests are always welcome via [GitHub](https://github.com/jaycezhang789/numjs/issues).
