# numjs Workspace

多后端的数值计算实验项目，目标是在 JavaScript/TypeScript 中提供类似 NumPy 的体验，并能在不同运行时之间自动切换最快的后端。

## 包结构

- `packages/core-rs`：Rust 实现的核心数值运算（矩阵加法、矩阵乘法等），被所有绑定共享。
- `packages/napi`：基于 N-API 的 Node.js 原生扩展，构建后会生成 `index.node`，在 Node 环境使用以获得最佳性能。
- `packages/wasm`：使用 `wasm-bindgen` 编译到 WebAssembly，构建后生成 `pkg/` 目录，适用于浏览器或没有原生扩展的环境。
- `packages/js`：向外暴露的 npm 包 `@jayce789/numjs`，在运行时优先加载 N-API 后端，若不可用则回退到 WebAssembly。

## 构建流程

```bash
# 安装依赖
npm install

# 构建所有后端并产出 JS 发行物
npm run build
```

命令会按顺序执行：

1. 构建 N-API 后端（Rust → Node 原生）。
2. 构建 WebAssembly 后端（Rust → wasm）。
3. 使用 `tsup` 打包 TypeScript 外观层并复制所有二进制/wasm 产物到 `packages/js/dist/bindings/**`，供发布使用。

## 发布 npm 包

```bash
cd packages/js
npm publish --access public
```

请确认已通过 `npm login` 获取 `@jayce789` scope 的发布权限，并且在发布前执行了完整构建。

## 运行入口

```ts
import { init, Matrix, add, matmul, backendKind } from "@jayce789/numjs";

await init(); // 自动加载 N-API 或 wasm 后端
```

`init()` 会尝试加载 N-API，如果失败则回退到 wasm，确保在不同环境都能获得同一套 API。
