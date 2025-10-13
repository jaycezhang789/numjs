# @jayce789/numjs

NumPy-inspired numerical computing for JavaScript that runs on both Node.js and browsers. The library prefers a fast native N-API backend and automatically falls back to WebAssembly when the native binary is unavailable.

## Highlights
- Single API surface for Node.js and browser environments.
- Automatic backend detection (`"napi"` -> `"wasm"` fallback).
- Rust core (`packages/core-rs`) shared by both bindings.

## Quick Start
```ts
import { init, Matrix, add, matmul, backendKind } from "@jayce789/numjs";

await init(); // loads the best available backend
const a = new Matrix([1, 2, 3, 4], 2, 2);
const b = new Matrix([5, 6, 7, 8], 2, 2);

console.log(backendKind()); // "napi" or "wasm"
console.log(add(a, b).toArray());    // Float64Array [6, 8, 10, 12]
console.log(matmul(a, b).toArray()); // Float64Array [19, 22, 43, 50]
```

## Backends
- **N-API** (`dist/bindings/napi/index.node`): loaded in Node.js when a prebuilt binary exists for the current platform.
- **WebAssembly** (`dist/bindings/wasm/num_rs_wasm.*`): lazy-loaded module that works in browsers and Node.js.

Both backends expose the same Rust-powered API, so the JavaScript interface remains identical.

## Building from Source
The publish workflow expects prebuilt artifacts to live under `dist/bindings/**`. Generate them locally in three steps:

1. Build the N-API binary (requires Rust toolchain):
   ```bash
   npm --prefix packages/napi run build
   ```
2. Build the WebAssembly package (requires `wasm-pack`):
   ```bash
   npm --prefix packages/wasm run build
   ```
3. Bundle the JavaScript wrapper and copy the artifacts:
   ```bash
   npm --prefix packages/js run build
   ```

When releasing to npm, run the same commands (or rely on CI) before `npm publish` so that consumers receive both backends.
