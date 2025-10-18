# WebGPU 加速篇

> 适用环境：Chromium 113+、Firefox Nightly（需 flag）、Node.js 20+（`--experimental-webgpu`）。部分执行器仍处于实验阶段，请在生产环境启用时做好降级方案。

## 为什么需要 WebGPU

随着模型尺寸与矩阵规模的增长，纯 CPU 加速已难以满足实时场景。NumJS 提供实验性的 WebGPU 执行后端，可将部分算子 offload 到 GPU，显著提升批量矩阵乘法、卷积与广播计算的吞吐。

## 快速上手

```ts
import { init, initWebGpu, matmulAsync, Matrix } from "@jayce789/numjs";

await init({ preferBackend: "wasm", threads: true });

// 确认 WebGPU 可用
const supported = await initWebGpu();
if (!supported) {
  console.warn("当前环境不支持 WebGPU，将回退到 WASM CPU 实现");
}

const createRandomMatrix = (rows: number, cols: number) =>
  new Matrix(
    Float32Array.from({ length: rows * cols }, () => Math.random() - 0.5),
    rows,
    cols,
    { dtype: "float32" }
  );

const a = createRandomMatrix(512, 512);
const b = createRandomMatrix(512, 512);

console.time("gpu-matmul");
const c = await matmulAsync(a, b, { mode: "gpu-only" });
console.timeEnd("gpu-matmul");
```

> `initWebGpu()` 会在内部实例化 `navigator.gpu.requestAdapter()`（或 Node.js `gpu.requestAdapter()`），并注册 GPU pipeline，重用 wasm 内存以减少数据拷贝。

## 需要掌握的概念

- **Device & Queue**：当前实现会请求系统首个支持 FP32 的适配器与设备，如需暂时禁用 WebGPU，可调用 `initWebGpu({ forceFallback: true })`。
- **Shared Memory**：对大规模矩阵，分块（tile-based）矩阵乘法是性能关键。NumJS 在 GPU kernel 中采用 16×16 tile 并支持向量化加载。
- **同步与异步**：所有 WebGPU 算子均返回 `Promise<Matrix>`，与 CPU 版本保持一致。内部会在 GPU kernel 完成后将结果写回共享内存。

## 性能调优

| 策略 | 说明 |
| --- | --- |
| 减少 host↔device 往返 | 避免在 GPU 管线中频繁调用 `toArray()`，尽量串联多个 GPU 算子。 |
| 控制矩阵尺寸 | 小于 `128×128` 的矩阵在 CPU/N-API 上可能更快，可通过 `if (rows < 128)` 判断后择优化路径。 |
| 避免重复初始化 | 在单个会话内只调用一次 `initWebGpu()`，内部会缓存管线与缓冲区。 |
| 利用 WASM 线程池 | WebGPU 与 wasm 线程池并不冲突，可同时用于 CPU/GPU pipeline。 |

## 故障排查

- `DOMException: No adapter found`：浏览器未启用 WebGPU，检查 `chrome://flags/#enable-webgpu`。
- `ValidationError: binding size must be a multiple of 4`：自定义 kernel 时需按文档对齐缓冲区。
- Node.js 需要 `--experimental-webgpu` 标志，并确保 `@webgpu/types` 已安装（项目已在 devDependencies 中包含）。

## 示例与 Playground

- StackBlitz：[WebGPU 矩阵乘法对比](../interactive/README.md#stackblitz) 展示与纯 WASM/CPU 的性能差异。
- CodeSandbox：参见 [`webgpu-broadcast.ts`](../interactive/README.md#codesandbox) 演示 GPU 加速的广播操作。
- 本地示例：`examples/numjs-interactive` 包含完整的 Vite + TypeScript 项目，可直接 `pnpm install && pnpm dev` 运行。

> WebGPU 模块仍在快速迭代，建议在生产环境中保持开关可控，并在回退路径上保留 N-API 或 WASM CPU 实现。
