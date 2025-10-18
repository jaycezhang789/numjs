# NumJS Interactive Playground

一个基于 Vite 的前端示例项目，用于演示 `@jayce789/numjs` 的核心 API、后端选择逻辑以及 WebGPU 加速实验特性。该目录与 StackBlitz/CodeSandbox 在线文档同步。

## 快速开始

```bash
pnpm install
pnpm dev
```

浏览器访问 <http://localhost:5173>，修改 `src/*.ts` 观察热更新输出。

## 关键脚本

- `src/main.ts`：快速入门示例，展示矩阵操作与后端检测。
- `src/backends.ts`：遍历不同 `preferBackend` 选项验证后端可用性。
- `src/autograd.ts`：演示自动微分，展示梯度回传与广播梯度处理。
- `src/training.ts`：基于线性回归的训练示例，结合优化器封装实际更新参数。
- `src/migration.ts`：NumPy ➜ NumJS API 映射示例。
- `src/webgpu.ts`：WebGPU 启用与性能测量（实验性）。
- `scripts/node-demo.mjs`：在 Node.js 中运行的 CLI 版本，便于 CI 或 Benchmark。

> 注意：WebGPU 功能取决于运行环境，可参考 `packages/js/docs/tutorials/webgpu.md` 获取更多细节。
