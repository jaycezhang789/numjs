# 锦上添花与未来探索

NumJS 的基础能力已经覆盖矩阵算子、N-API/WebAssembly 双后端以及 WebGPU 实验支持。接下来，我们计划在以下两个方向持续迭代，欢迎在 Issue 或 Discussions 中分享需求与想法。

## 自动微分（Autograd）

目标是在不引入庞大的运行时依赖的前提下，为 NumJS 提供轻量级计算图与自动微分能力，适配优化算法与小型深度学习模型。

- **计算图表示**：设计原生 `Matrix` 兼容的节点结构，区分叶子节点（常量/输入）与中间节点（算子输出），支持即时前向执行与延迟构建。
- **前向/反向传播**：为核心算子（加减乘除、`matmul`、`broadcastTo`、`sum`、`exp`、`log` 等）实现对偶运算规则，并提供 `backward()` 触发的梯度回传。
- **梯度存储**：在 `Matrix` 上暴露可选的 `grad` 视图，支持累积与清零；为长链路计算考虑梯度检查与 NaN 检测。
- **优化器接口**：提供 `sgd`, `adam` 等基础优化器示例，验证 Autograd 在小模型训练（如线性回归、多层感知机）的可行性。
- **与现有后端协同**：确保 Autograd 在 N-API、WASM、WebGPU 下表现一致，必要时通过混合精度或回退策略保持稳定性。

## 随机与分布

高质量的随机数基础是统计模拟、模型初始化与强化学习的关键。我们计划引入现代 PRNG 与分布抽样工具包。

- **PRNG 引擎**：实现 PCG-XSL-RR 与 Philox 等通过测试的生成器，支持 32/64 位输出、自定义播种以及跳跃（`advance`）能力。
- **可复现的流与子流**：提供基于 counter 的流划分 API，方便在多线程、WebWorker、GPU 场景下保持跨环境 determinism。
- **分布族支持**：原生实现 `uniform`, `normal`, `bernoulli`, `poisson` 等常用分布，并暴露向量化采样接口（直接生成 `Matrix`）。
- **统计检验工具**：附带简单的 `chiSquareTest`, `ksTest` 帮助验证分布质量，同时输出种子复现的诊断信息。
- **与 Autograd 的结合**：为概率模型预留钩子，未来可将采样节点纳入计算图，实现 reparameterization trick 等技巧。

> 这些功能仍在规划阶段。如果你在项目中迫切需要部分能力，欢迎通过 GitHub Issue 反馈优先级，或提交草稿 PR 参与设计讨论。

## 稀疏矩阵与 SuiteSparse 集成

- **数据结构**：添加 `SparseMatrix` 抽象，首批支持 CSR/CSC/COO 存储格式，可与现有 `Matrix` 互转换。
- **核心算子**：实现稀疏-稠密矩阵乘、稀疏矩阵乘、转置、切片、堆叠等基础操作；在 WASM 端重用 Rust `sprs`，N-API 端对接 SuiteSparse。
- **编译与分发**：扩展可选依赖，提供平台包携带 SuiteSparse 预编译库，确保安装体验一致。
- **Fallback 策略**：在没有原生库的环境中降级到纯 Rust/JS 实现，同时警示性能影响。

## 大数据与外存处理

- **内存映射**：暴露 Node.js `mmap` 封装，允许直接把外部二进制文件映射为 `Matrix`/`SparseMatrix` 的底层存储。
- **分块算法**：提供 `chunkedMap`, `chunkedReduce`, `outOfCoreMatmul` 等辅助工具，在内存受限场景维持稳定运行。
- **调度策略**：支持按块调度的多线程/多 Worker 流水线，结合进度回调与取消机制。
- **与 Arrow/Parquet 集成**：允许基于 Arrow RecordBatch 按需映射列，减少数据复制。

## 数值稳健性增强

- **精确求和**：在 `sum`, `mean`, `var` 等聚合中增加 Kahan、pairwise 选项，并自动检测何时启用高精度路径。
- **稳定 softmax**：推出 `stableSoftmax`, `logSumExp` 等函数，默认减去最大值消除溢出，并提供日志/检验工具。
- **诊断工具**：新增 `detectCancellation`, `numericStats` 帮助分析数值不稳定的中间结果。

## 混合精度与容差控制

- **新 dtype**：为 `float16`, `bfloat16`, `tf32` 引入类型描述与转换函数，在 GPU/N-API 后端以原生精度执行。
- **自动调度**：根据运算规模与误差预算，自动选择运算精度，必要时回退到 `float32/float64`。
- **容差配置**：扩展 `init({ tolerances })` 与 `withNumericTolerance`，让用户在 API 级别声明误差上限。
- **WebGPU 支持**：检测浏览器/驱动能力，动态切换 `f16`/`f32` shader。

## JIT 与自定义算子

- **Cranelift JIT**：在 Node/N-API 环境下使用 Cranelift 动态生成本地 ufunc，实现 `jitCompile(fn, signature)` 风格 API。
- **WGSL 代码生成**：在 WebGPU + WASM 环境通过 WGSL 模板 JIT 自定义 kernel，支持 elementwise、reduction、broadcast 三大模式。
- **安全沙箱**：限制 JIT 输入类型与操作范围，提供诊断日志与缓存策略。
- **与 Autograd 集成**：允许为自定义算子注册梯度定义，使其在自动微分图中无缝使用。
