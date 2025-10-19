# Choosing Between WASM and N-API Backends

`@jayce789/numjs` ships with a dual-backend architecture. Whenever the runtime environment allows it, we load the N-API native module for maximal throughput; otherwise we fall back to a portable WebAssembly build. This guide explains how the loader operates, which knobs you can adjust, and how to diagnose common deployment issues.

## High-level architecture

```
┌────────────┐    ┌─────────────────────┐
│ Application│───▶│@jayce789/numjs      │
└────────────┘    ├─────────────────────┤
                  │ N-API bridge (Rust) │──▶ Prebuilt `.node` binary (platform specific)
                  │ WASM bridge (wasm32)│──▶ `num_rs_wasm.wasm` + JS glue
                  └─────────────────────┘
```

The `init()` function coordinates the backend discovery process:

1. Detects the host platform (`process.platform`, `process.arch`, and libc variants) and looks up the matching entry in `NAPI_DISTRIBUTIONS` (`packages/js/src/index.ts`).
2. Attempts to load the prebuilt `.node` binary from the current package or from a hoisted dependency (e.g. `@jayce789/numjs-linux-x64`).
3. If native loading fails, downloads / instantiates `dist/bindings/wasm/num_rs_wasm.wasm` and its JavaScript glue. All public APIs are re-exported from the wasm bindings.
4. When the `threads` option is enabled, validates `SharedArrayBuffer` support and spins up the wasm thread pool.

## Recommended backend per scenario

| Environment | Recommended backend | Rationale |
| --- | --- | --- |
| CI, server-side rendering | N-API | Fast startup, uses native BLAS backends, full multithreading. |
| Electron apps | N-API first, WASM fallback | Electron ≥ 28 ships with a compatible N-API runtime; fallback avoids crashes when ABI mismatches. |
| Browsers, Edge runtime | WASM | Requires no native binary, streams easily over the network. |
| Serverless (Vercel/Netlify) | N-API on x64 instances, WASM on edge functions | Keep cold starts low by pre-warming the wasm module; native modules need to be bundled as assets. |

### Forcing a backend

```ts
import { init } from "@jayce789/numjs";

await init({
  preferBackend: "napi",           // "auto" | "napi" | "wasm"
  threads: true,                   // Enable WASM multithreading when supported
  webGpu: { forceFallback: false } // Optional: initialise WebGPU during startup
});
```

Setting `preferBackend: "napi"` skips the wasm probe entirely; `"wasm"` bypasses native loading (useful in sandboxed environments).

## N-API distribution packages

- All platform-specific binaries are listed under `optionalDependencies`. Package managers (npm, pnpm, yarn) install only the matching subset, reducing install failures.
- The build pipeline uses `scripts/copy-artifacts.mjs` to move compiled `.node` files into `dist/bindings/napi/` and to publish them under `npm/<platform>/` helper packages.
- When cutting a release, ensure CI produces binaries for every supported triplet (Linux x64, Linux arm64, macOS x64/arm64, Windows x64). `README.md` contains the end-to-end release checklist.

## WebAssembly module

- Built with `wasm-pack` and bundled via `tsup`. SIMD is enabled by default, so Node.js ≥ 18 or modern browsers are required.
- Calling `init({ threads: true })` lazily loads the worker glue (`num_rs_wasm_bg.wasm`) and initialises the thread pool. If `SharedArrayBuffer` is unavailable, the loader falls back to single-threaded execution and logs a warning.
- `init({ webGpu: {} })` optionally boots the experimental WebGPU pipeline. See the [WebGPU tutorial](./webgpu.md) for details.

## Diagnostics & troubleshooting

- `backendKind()` returns the active backend (`"napi"`, `"wasm"`, or `"fallback"`). You can register callbacks with `backendReadyHook()` to monitor transitions.
- Debug logging (coming soon) will stream verbose loader information. In the meantime you can set `process.env.NUMJS_DEBUG=1` to surface select warnings.
- Common errors:
  - `ERR_DLOPEN_FAILED` – the native module could not be loaded. Confirm that your runtime matches one of the published triplets and that `optionalDependencies` were not pruned.
  - `TypeError: WebAssembly is not defined` – occurs when running in an environment where WASM support is disabled. Provide a polyfill or switch to N-API.
  - `SharedArrayBuffer is undefined` – browsers need COOP/COEP headers for cross-origin isolation. Disable `threads` or configure the headers appropriately.
  - `Cannot find module '@jayce789/numjs-linux-x64'` – indicates the platform helper package was not installed; run `pnpm install --include-optional`.

## Example projects

- StackBlitz: the [`WASM vs N-API profiler`](../interactive/README.md#stackblitz) sandbox compares throughput and prints diagnostic information during backend selection.
- CodeSandbox: [`backend-selection.ts`](../interactive/README.md#codesandbox) demonstrates how to force a backend and display runtime metadata.
- Local demo: `examples/numjs-interactive/src/backends.ts` mirrors the sandbox code and can be executed with `pnpm dev`.

For implementation details inspect the `packages/napi/` (native binding) and `packages/wasm/` (wasm bundle) subdirectories. They contain the Rust crates, build scripts, and integration tests that underpin the high-level JS API.
