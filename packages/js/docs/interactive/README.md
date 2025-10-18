# 交互式文档 & 在线 Playground

你可以通过 StackBlitz 或 CodeSandbox 即时体验 `@jayce789/numjs`，无需本地安装。我们提供了两个官方入口以及“复制即跑”的最小示例，方便你快速验证 API 或分享给团队成员。

## StackBlitz

- **入口**：<https://stackblitz.com/github/jaycezhang789/numjs/tree/master/examples/numjs-interactive?file=src/main.ts&terminal=dev>
- 预装依赖：`@jayce789/numjs`, `typescript`, `tslib`
- 默认启动脚本：`pnpm dev`（Vite）
- Playground 中包含以下面板：
  - `src/main.ts`：展示矩阵创建、算子链式调用、后端信息输出。
  - `autograd.ts`：新引入的自动微分示例，可查看梯度回传效果。
  - `webgpu.ts`：可选启用 WebGPU 加速，自动检测运行环境。
  - `migration.ts`：与 NumPy API 映射的对照示例。

> 第一次打开 Playground 时 StackBlitz 会自动执行 `pnpm install`，控制台出现 `"ready - server running..."` 后即可实时编辑并查看输出。

## CodeSandbox

- **入口**：<https://codesandbox.io/p/github/jaycezhang789/numjs/tree/master/examples/numjs-interactive>
- 预设任务：
  1. 执行 `pnpm install`
  2. 启动 `pnpm dev`
- 内置任务 `src/backends.ts` 示范如何手动指定 N-API / WASM / WebGPU。
- 支持 Live Collaboration，可直接邀请队友在同一环境修改代码并观察输出。

## 复制即跑的 Demo

```bash
npx degit jaycezhang789/numjs/examples/numjs-interactive my-numjs-playground
cd my-numjs-playground
pnpm install   # 或 npm install / yarn
pnpm dev       # http://localhost:5173
```

该 Demo 基于 Vite + TypeScript，结构与在线 Playground 完全一致。主要文件简介：

- `src/main.ts`：快速入门脚本，展示 `init()`, `Matrix`, `add`, `matmul`, `backendKind()` 的组合使用。
- `src/backends.ts`：提供后端能力检测与手动切换示例。
- `src/webgpu.ts`：演示 WebGPU 加速矩阵乘法与广播操作。

如需在 Node.js 纯后端环境运行，可参考 `scripts/node-demo.mjs`（同目录提供），执行 `pnpm node-demo` 或 `node scripts/node-demo.mjs` 即可。

> 提示：在 StackBlitz/CodeSandbox 中运行 WebGPU 相关示例时，请使用支持 WebGPU 的现代浏览器（Chromium 113+）。
