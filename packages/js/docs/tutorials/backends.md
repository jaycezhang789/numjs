# WASM 与 N-API 后端选择与原理

NumJS 采用“双后端”架构：Node.js 环境优先加载 N-API 原生模块，浏览器和受限运行环境则自动回退至 WebAssembly。理解背后的机制可以帮助你在多端场景中取得最佳性能与兼容性。

## 架构概览

```
┌────────────┐    ┌─────────────────────┐
│ Application│───▶│@jayce789/numjs      │
└────────────┘    ├─────────────────────┤
                   │ N-API bridge (Rust) │──▶ Prebuilt `.node` (平台特定)
                   │ WASM bridge (wasm32)│──▶ `num_rs_wasm.wasm` + JS glue
                   └─────────────────────┘
```

`init()` 会执行以下流程：

1. 解析 `process.platform` 与 `process.arch`，匹配 [`packages/js/src/index.ts`](../../src/index.ts) 中的 `NAPI_DISTRIBUTIONS`。
2. 尝试加载对应的 `.node` 或 npm 平台包（可跨项目 hoist）。
3. 如果原生模块不可用，则异步下载/实例化 `dist/bindings/wasm/num_rs_wasm.wasm`。
4. 若启用 `threads` 选项，会检查 `SharedArrayBuffer` 支持并初始化 wasm 线程池。

## 后端选择策略

| 场景 | 推荐后端 | 理由 |
| --- | --- | --- |
| CI/服务端渲染 | N-API | 启动快、吞吐高、支持多线程。 |
| Electron 应用 | N-API（首选）→ WASM | Electron 28+ 支持 N-API，若 ABI 不匹配会自动回退。 |
| 浏览器/Edge runtime | WASM | 纯前端部署，无需本地编译。 |
| Serverless (Vercel/Netlify) | N-API on x64，WASM on edge | 避免在 Hook 中下载 wasm，提前缓存。 |

### 强制指定后端

```ts
import { init } from "@jayce789/numjs";

await init({
  preferBackend: "napi",           // "auto" | "napi" | "wasm"
  threads: true,                   // 启用 wasm 多线程
  webGpu: { forceFallback: false } // 浏览器中预热 WebGPU（可选）
});
```

## N-API 平台包

- 所有平台包列在 `optionalDependencies`，npm/yarn 会按平台挑选可用项，降低安装失败概率。
- `scripts/copy-artifacts.mjs` 会在 `npm run build` 期间把编译好的 `.node` 文件放入 `dist/bindings/napi/`，并同步到 `npm/<platform>/` 子包。
- 发布时应确保 CI 产出所有平台二进制，参考 `README.md` 的发布流程。

## WebAssembly 模块

- 构建使用 `wasm-pack` + `tsup`，默认启用 SIMD，需 Node.js ≥ 18 或现代浏览器。
- `init({ threads: true })` 会动态导入 `num_rs_wasm_bg.wasm` 和 worker glue，若运行环境缺乏 `SharedArrayBuffer` 则退回单线程模式。
- 可通过 `init({ webGpu: {} })` 让 wasm backend 尝试启用 WebGPU 适配层（详见 [WebGPU 加速篇](./webgpu.md)）。

## 故障诊断

- 调用 `backendKind()` 查看当前后端；`backendReadyHook` 可注册监听。
- 设置 `enableDebugLogging()`（未来版本提供）后可打印详细加载日志。
- 常见错误：
  - `ERR_DLOPEN_FAILED`：平台包缺失或 ABI 不匹配，确认是否运行在受支持平台，并检查 `optionalDependencies` 是否被包管理器忽略。
  - `TypeError: WebAssembly is not defined`：在禁用 wasm 的环境运行，可通过 polyfill 或降级到 Node.js。
  - `SharedArrayBuffer is undefined`：浏览器缺乏跨源隔离，移除 `threads: true` 或配置 COOP/COEP 头。

## 参考示例

- StackBlitz：[`WASM vs N-API profiler`](../interactive/README.md#stackblitz)。
- CodeSandbox：[`backend-selection.ts`](../interactive/README.md#codesandbox) 演示如何显式指定后端。

更多内部实现细节可浏览 `packages/napi/` 与 `packages/wasm/` 子项目源码。
