# @jayce789/numjs

`@jayce789/numjs` brings a Rust-powered numerical core to the JavaScript ecosystem, delivering a NumPy-inspired matrix API. The package ships with a high-performance Node N-API backend and a WebAssembly fallback, automatically loading the best option at runtime so the same code runs in Node.js and modern browsers.

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
- `matrix.astype("float32")` to cast between dtypes (copy-on-write when possible).
- `whereSelect(condition, truthy, falsy)` for broadcasted conditional selection.
- `concat(matrixA, matrixB, axis)` / `stack(matrixA, matrixB, axis)` for joins with dtype promotion.
- `matrix.transpose()` / `transpose(matrix)` for zero-copy axis swaps.
- `broadcastTo(matrix, rows, cols)` to expand singleton dimensions (works for fixed64 as well).
- `sum(matrix)` and `dot(a, b)` now default to numerically stable reducers.
- Linear algebra helpers such as `svd`, `qr`, `solve`, and `eigen` when supported by the active backend.

Refer to the generated TypeScript declarations in `dist/index.d.ts` for the complete surface area.

## Backend Strategy
`init()` negotiates the backend in the following order:
1. **N-API backend** – loads the precompiled `.node` binary shipped under `dist/bindings/napi/`.
2. **WebAssembly backend** – falls back to `dist/bindings/wasm/num_rs_wasm.*` when the native module is unavailable (for example in the browser).

You can inspect the decision with `backendKind()` or provide your own loader if you need custom logic.

## Browser Usage
```ts
import { init, Matrix, broadcastTo } from "@jayce789/numjs";

await init(); // downloads and instantiates the wasm module
const vector = new Matrix([1, 2, 3], 3, 1);
console.log(broadcastTo(vector, 3, 2).toArray());
```
Ensure your bundler knows how to serve `.wasm` assets (Vite, Webpack, Rollup, etc.).

## Node.js Compatibility
- Supports Node.js 16 and newer.
- Bundles CommonJS (`dist/index.cjs`) and ES Module (`dist/index.js`) entry points.
- Works in Electron/NW.js – when the native module cannot be loaded the code automatically falls back to WebAssembly.

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

---
Questions or ideas? Issues and pull requests are always welcome.
