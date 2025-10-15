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
