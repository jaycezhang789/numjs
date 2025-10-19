# Interactive Documentation & Online Playground

You can try `@jayce789/numjs` instantly on StackBlitz or CodeSandbox—no local setup required. The official sandboxes described below mirror the repository’s `examples/numjs-interactive` project, making it easy to experiment, share snippets, or reproduce bug reports.

## StackBlitz

- **Entry point**: <https://stackblitz.com/github/jaycezhang789/numjs/tree/master/examples/numjs-interactive?file=src/main.ts&terminal=dev>
- Preinstalled dependencies: `@jayce789/numjs`, `typescript`, `tslib`
- Default command: `pnpm dev` (Vite dev server)
- Included panels:
  - `src/main.ts` – matrix construction, chained operators, backend diagnostics.
  - `autograd.ts` – early autograd prototype showcasing forward and backward passes.
  - `training.ts` – linear regression training loop with SGD/Adam/RMSProp.
  - `webgpu.ts` – optional WebGPU acceleration demo with environment detection.
  - `migration.ts` – side-by-side NumPy → NumJS API mapping.

> The first time StackBlitz loads the project it runs `pnpm install`. Once the console prints `ready - server running...`, edits refresh instantly.

## CodeSandbox

- **Entry point**: <https://codesandbox.io/p/github/jaycezhang789/numjs/tree/master/examples/numjs-interactive>
- Preconfigured tasks:
  1. `pnpm install`
  2. `pnpm dev`
- Highlight: `src/backends.ts` shows how to force N-API, WASM, or WebGPU backends at runtime.
- Live collaboration is enabled—invite teammates to edit and observe output together.

## Clone-and-run demo

```bash
npx degit jaycezhang789/numjs/examples/numjs-interactive my-numjs-playground
cd my-numjs-playground
pnpm install   # or npm install / yarn
pnpm dev       # http://localhost:5173
```

This Vite + TypeScript project is identical to the online playgrounds. Key files:

- `src/main.ts` – quickstart script using `init()`, `Matrix`, `add`, `matmul`, and `backendKind()`.
- `src/backends.ts` – backend capability inspection and manual selection.
- `src/webgpu.ts` – WebGPU accelerated matmul and broadcast example.

For a purely Node.js experience, run `node scripts/node-demo.mjs` (also available via `pnpm node-demo`).

> Tip: When running WebGPU demos in online sandboxes, use a modern browser with WebGPU enabled (Chromium 113+). Some sandboxes disable experimental APIs in embedded iframes; open the project in a dedicated tab if necessary.
