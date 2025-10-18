# Sparse PoC with `sprs`

本实验帮助验证 `sprs` 在 `num_rs_core` 中的可用性，并为后续 SuiteSparse 集成提供参考。

## Experiment Steps

1. 安装依赖：`sprs = "0.11"` 已作为 `dev-dependencies` 添加到 `packages/core-rs/Cargo.toml`。
2. 运行示例（需启用 `sparse-native` 特性）：
   ```bash
   cargo run --example sparse_poc --features sparse-native
   ```
   该程序会：
   - 使用 `sprs::CsMat` 构造一个 3x3 CSR 矩阵。
   - 将 CSR 映射到 `num_rs_core::sparse::CsrMatrixView`。
   - 调用 `sparse_matmul` 与 `sparse_add` fallback。
   - 输出结果并与 `sprs` 的稠密乘积进行比对。
3. 交叉编译可行性：
   - `sprs` 仅依赖纯 Rust 代码（默认特性关闭），适合 wasm/跨平台场景。
   - 对于 N-API/SuiteSparse 方案，可在此基础上替换 fallback，实现原生稀疏算子。

## Next Steps

- 在 `num_rs_core::sparse` 中替换 fallback 为实际 SuiteSparse/sprs 内核。
- 增加基准测试比较 native 与 fallback 的性能差异。
