# @jayce789/numjs

`@jayce789/numjs` brings a Rust-powered numerical core to the JavaScript ecosystem, delivering a NumPy-inspired matrix API. The package ships with both a high-performance Node N-API backend and a WebAssembly fallback, automatically loading the best option at runtime and running seamlessly in Node.js or modern browsers.

## Installation
```bash
npm install @jayce789/numjs
```
or, if you prefer other package managers:
```bash
pnpm add @jayce789/numjs
yarn add @jayce789/numjs
```

## Key Features
- **Unified API** – The same Matrix and operator interfaces work in Node.js and the browser.
- **Automatic backend selection** – Prefers the prebuilt N-API binary and falls back to WebAssembly when necessary.
- **Rich dtype support** – Covers common numerical dtypes such as `float64`, `int32`, `bool`, and more.
- **Rust core** – Critical logic is implemented in Rust for predictable, high performance.
- **Tree-shakable outputs** – Distributed as both ESM and CJS bundles for smooth integration with modern build tools.

## Quick Start
```ts
import { init, Matrix, add, matmul, backendKind } from "@jayce789/numjs";

await init(); // resolves and loads the optimal backend

const a = new Matrix([1, 2, 3, 4], 2, 2);
const b = new Matrix([5, 6, 7, 8], 2, 2);

console.log(backendKind());   // "napi" or "wasm"
console.log(add(a, b).toArray());    // Float64Array [6, 8, 10, 12]
console.log(matmul(a, b).toArray()); // Float64Array [19, 22, 43, 50]
```

### Common Operations
- `Matrix.fromArray(array, rows, cols, dtype?)` – Create a matrix from plain arrays.
- `matrix.astype("float32")` – Convert between supported dtypes.
- `whereSelect(condition, truthy, falsy)` – Apply boolean masks to select values.
- `concat(matrixA, matrixB, axis)` and `stack(matrixA, matrixB, axis)` – Concatenate or stack matrices.
- Linear algebra helpers such as `svd`, `qr`, `solve`, and `eigen`, when supported by the active backend.

Refer to the bundled TypeScript definitions or source comments for the full API surface.

## Backend Strategy
`init()` negotiates the backend in the following order:
1. **N-API backend** – Loads the precompiled `.node` binary under `dist/bindings/napi/`. Multiple platform targets are shipped with the package.
2. **WebAssembly backend** – If the N-API binary is unavailable or the runtime is not Node.js (for example, the browser), the loader falls back to `dist/bindings/wasm/num_rs_wasm.*`.

Need to pin a specific backend? Inspect `backendKind()` or customize your initialization sequence as required.

## Browser Usage
The WebAssembly backend allows the library to run directly in modern browsers:
1. Ensure your bundler can load `.wasm` assets (configure Vite, Webpack, Rollup, etc.).
2. Import `init` and the APIs you need from the ESM entry point.
3. Call `await init()` before executing matrix operations.

```ts
import { init, Matrix } from "@jayce789/numjs";

await init(); // downloads and instantiates the wasm module in the browser
const vector = new Matrix([1, 2, 3], 3, 1);
console.log(vector.toArray());
```

## Node.js Compatibility
- Supports Node.js 16 and newer.
- Available as both CommonJS (`dist/index.cjs`) and ES Module (`dist/index.js`) builds, selected automatically based on how you import the package.
- Works in Electron, NW.js, and similar runtimes—if the environment cannot load the N-API binary, it automatically falls back to WebAssembly.

## Performance Tips
- For heavier workloads, prefer the N-API backend; prebuild platform-specific binaries in your deployment pipeline when possible.
- Use `copyBytesTotal()`, `takeCopyBytes()`, and `resetCopyBytes()` to monitor internal buffer copies and optimize data movement.

## Output & Rounding (Display-Only)

JavaScript numbers use IEEE‑754 binary floating‑point, so decimal values like `0.1 + 0.2` cannot be represented exactly. This library keeps computations in native float dtypes for performance, and adds a configurable output layer to make results look clean when printing/exporting.

- Default output behavior:
  - `as: 'number'`, `decimals: 12`, `trimTrailingZeros: true`
  - Affects only `toString()`, `toJSON()`, `toOutputArray()`, and `toOutput2D()`.
  - Does not change internal computation or matrix storage.

- Configure globally or per scope:
  - `setOutputFormat({ as?: 'string'|'number'|'bigint', decimals?, scale?, trimTrailingZeros? })`
  - `withOutputFormat(opts, fn)` to apply within a limited scope
  - `getOutputFormat()` to inspect current settings

- Export helpers:
  - `toOutputArray(matrix, options?)` → 1D array
  - `toOutput2D(matrix, options?)` → 2D rows×cols array
  - Instance: `matrix.toOutputArray(options?)`, `matrix.toString()`, `matrix.toJSON()`

- Output modes and trade‑offs:
  - `as: 'string'` (precise decimal rendering). Best for reports/JSON. Controlled by `decimals` and `trimTrailingZeros`.
  - `as: 'number'` (rounded Numbers). Looks clean for display, but follow‑up math is still binary float and can reintroduce artifacts.
  - `as: 'bigint'` + `scale` (fixed‑point integers). Great for currency exporting (e.g., cents). Not suitable for general arithmetic in JS unless you implement your own fixed‑point operators.

Example:
```ts
import {
  setOutputFormat, withOutputFormat,
  toOutputArray, toOutput2D, round,
} from '@jayce789/numjs';

// Global output as string with 2 decimals
setOutputFormat({ as: 'string', decimals: 2 });

// Per-call overrides
const pretty = toOutputArray(matrix, { as: 'number', decimals: 6 });
const cents  = toOutputArray(matrix, { as: 'bigint', scale: 2 });

// Scoped output format
await withOutputFormat({ as: 'string', decimals: 3 }, async () => {
  console.log(String(matrix));
});

// Explicit compute-layer quantization (optional, not default)
const quantized = round(matrix, 2); // returns a new Matrix
```

## Modes & Trade‑offs

- Float (default):
  - Fastest path (SIMD/FMA where applicable via Rust/N-API/wasm).
  - Display artifacts are handled by the output layer; internal computations remain floating‑point.
  - For comparisons, prefer `isClose`/`allClose` instead of `===`.

- Fixed‑point export (`as: 'bigint'` + `scale`):
  - Ideal for currency when exporting/serializing; keeps decimal semantics externally.
  - Not a compute dtype in JS; do not feed back into numeric math unless you convert deliberately.

- Decimal/Fixed dtypes (roadmap):
  - A true decimal dtype (`decimal` using `rust_decimal`) or fixed (`i64 + scale`) can provide decimal semantics in the core.
  - Trade‑off: linear algebra and vectorized ops will be slower vs float. Not enabled by default; will be introduced behind explicit opt‑in when available.

## Stability & Comparisons

- Use `isClose(a, b, { rtol=1e-12, atol=0, equalNaN=false })` for scalar comparisons.
- Use `allClose(A, B, same options)` for matrices.
- Future work may adopt numerically stable algorithms (e.g., pairwise/Kahan sum, Welford variance) in the core where it makes sense.

## Publishing
Before publishing to npm, build all artifacts (`dist/` and `dist/bindings/**`) so consumers receive both backends. Run the root build pipeline:
```bash
npm run build
```
Then publish from the workspace:
```bash
npm publish --workspace packages/js
```
Add `--access public` if you are releasing the package publicly for the first time.

---

Have feedback or ideas? Issues and pull requests are always welcome.
