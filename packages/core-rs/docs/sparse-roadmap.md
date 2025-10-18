# SuiteSparse Integration Roadmap

本路线图旨在将 `num_rs_core::sparse` 的 fallback 替换为真正的 SuiteSparse/sprs 原生实现，并验证性能收益。

---

## Phase A: 技术选型与构建支持

1. **依赖分析**
   - SuiteSparse 原生库（`csparse`, `cholmod`, `spmm` 等）与 sprs（纯 Rust）。
   - 工程优先级：先用纯 Rust 的 `sprs` 验证接口，再接入 SuiteSparse 提升性能。

2. **跨平台构建方案**
   - macOS/Linux：使用系统包或脚本编译 SuiteSparse。
   - Windows：评估 vcpkg 或手动构建流程。
   - WASM：保持 sprs fallback，记录限制。

3. **构建脚本原型**
   - 在 `packages/core-rs/build.rs` 或独立脚本中检测 SuiteSparse。
   - 引入 cargo feature（例如 `sparse-native`）控制编译路径。

---

## Phase B: 核心实现替换

1. **共享数据结构**
   - 在 `CsrMatrixView` 上扩展 `values_f32` / `values_f64` 访问接口，便于传递给 native 内核。

2. **SuiteSparse 绑定**
   - 编写 FFI 包装（Rust -> C）：
     - `extern "C" { ... }`
     - 安全封装：行/列指针合法性检测在 Rust 端完成。
   - 实现 `sparse_matmul` / `sparse_add` / `sparse_transpose` native 版本。

3. **Fallback 策略**
   - 保留当前 dense fallback 作为 `cfg(not(feature = "sparse-native"))`。
   - 在 native 失败或特定 dtype 不支持时回退到 fallback。

---

## Phase C: 测试与性能

1. **单元测试**
   - 小规模 CSR 运算与 dense 结果一致性。
   - 异常输入（行列指针非法、越界等）。

2. **性能基准**
   - 在 `benches/` 或 `criterion` 中比较：
     - 纯 js fallback vs native SuiteSparse
     - 密度变化（1%, 10%, 50%）的性能。

3. **集成测试**
   - 与 `sparse_poc` 的结果对比。
   - 在 N-API/WASM 中端到端验证。

---

## Phase D: 文档与发布

1. 更新 `packages/js/docs/design/sparse-matrix.md`，说明 native 内核构建方法。
2. CI 任务区分有/无 SuiteSparse 的构建矩阵。
3. 发布前检查：
   - `cargo fmt`, `cargo clippy`, `cargo test`, `npm run build:js`
   - 生成性能报告并附在 release notes。
