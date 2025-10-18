# SparseMatrix 设计草案

本文档描述 `@jayce789/numjs` 稀疏矩阵生态的第一阶段设计目标与技术路线，为后续实现与绑定 SuiteSparse/sprs 做准备。

## 数据结构

- **核心类型：`SparseMatrix`**
  - 默认内部存储采用 CSR（Compressed Sparse Row）格式，字段：
    - `rowPtr: Uint32Array`
    - `colIdx: Uint32Array`
    - `values: Float32Array | Float64Array | BigInt64Array (fixed64)`
  - 提供静态构造函数支持自定义格式：
    - `fromCSR(rowPtr, colIdx, values, shape, dtype)`
    - `fromCSC(...)`、`fromCOO(indices, values, shape, dtype)` —— 内部立即转换到 CSR。
  - 支持 `toCSR()`、`toCOO()` 导出原始数据，便于与外部工具交互。

- **DType 支持**
  - `float32` / `float64`：常规算子和 SuiteSparse 兼容
  - `int32` / `bool`：稀疏掩码与图算法需要（先提供转换接口，后续扩展）
  - `fixed64`：对齐现有 `Matrix` 能力，COO->CSR 时保持 64-bit 数据

- **内存视图**
  - 稀疏-稠密混合运算时，根据后端能力选择：
    - Node/NAPI：传递原始 TypedArray 到 Rust 层（sprs/SuiteSparse）
    - WASM：借助 `sprs` 的 wasm 版本或在 JS 中 fallback
  - 提供 `toDense()` 仅用于 debug / fallback，不推荐链式调用

## API 设计

### 类型与构造

```ts
export type SparseFormat = "csr" | "csc" | "coo";

export interface SparseMatrixInit {
  format: SparseFormat;
  data: {
    rowPtr?: Uint32Array;
    colPtr?: Uint32Array;
    indices: Uint32Array;
    values: Float32Array | Float64Array | BigInt64Array;
  };
  shape: { rows: number; cols: number };
  dtype?: DType;
}

export class SparseMatrix {
  constructor(init: SparseMatrixInit);
  static fromCSR(...): SparseMatrix;
  static fromDense(matrix: Matrix, threshold?: number): SparseMatrix;

  readonly rows: number;
  readonly cols: number;
  readonly nnz: number;
  readonly dtype: DType;
  readonly format: SparseFormat;

  toCSR(): { rowPtr: Uint32Array; colIdx: Uint32Array; values: TypedArray };
  toCOO(): { rowIdx: Uint32Array; colIdx: Uint32Array; values: TypedArray };
}
```

### 稀疏-稠密混合运算

| 运算 | 说明 |
| --- | --- |
| `sparseMatmul(sparse: SparseMatrix, dense: Matrix, options?)` | 支持 CSR * Dense 与 Dense * CSR；根据后端选择 N-API/WASM/JS fallback |
| `sparseAdd(sparse: SparseMatrix, dense: Matrix)` | 稀疏+稠密；fallback 版本为稠密化输入再调现有算子 |
| `sparseTranspose(sparse, format?)` | 原地切换 CSR/CSC，避免重复转换 |
| `sparseMatvec(sparse, vector)` | 快速路径，SuiteSparse/sprs 原生支持 |
| `sparseDiag` / `sparseEye` | 快速构造常见稀疏结构 |

### Autograd 兼容

第一阶段仅支持前向运算；自动微分计划在后续迭代中通过稀疏梯度表示或 callback 形式落地。接口保持与 `Tensor` 兼容（返回 `Matrix` 或 `SparseMatrix`），同时暴露 `toTensor()` 以便转换。

## 后端集成路线

| 阶段 | 目标 | 技术方案 |
| --- | --- | --- |
| Phase 1 | JS 层实现 `SparseMatrix` 类型与基础转换；稠密 fallback | 使用 TypedArray + CSR 实现；在 JS 中完成 `sparseMatmul` fallback（O(nnz * cols)) |
| Phase 2 | N-API 集成 SuiteSparse | Rust 中引入 `sprs` + SuiteSparse，导出 `sparse_matmul`, `sparse_add`, `sparse_transpose` 等 FFI；扩展 `copy-artifacts` 脚本分发 `.so/.dylib`。目前 `core-rs` 已有 PoC（见 `packages/core-rs/docs/sparse-poc.md`），等待原生实现落地。 |
|         |                        | **当前实现**：N-API 入口已就绪，但内部仍以稠密 fallback 实现。在 native kernels 落地后，将在这些入口中切换到真实稀疏实现。 |
| Phase 3 | WASM 后端支持 | 编译 `sprs`/`csparse` 到 wasm，或在 wasm 后端完成 CSR 算法；需关注内存访问与 SIMD |
| Phase 4 | 外存数据管道 | 利用 `mmap` + `SparseMatrix.fromCSR({rowPtr: MMapUint32(...)})`，结合 chunked 算子处理超大数据集 |

> 当前 WASM 仍保留纯 JS fallback；`packages/wasm` 尚未集成稀疏内核，但接口预留完毕，后续可按 Phase 3 路线接入。

## 与现有 生态的接口

- **SuiteSparse**：通过 `csrmv`, `csrgemm`、`cholmod` 等接口扩展；在 Rust 层封装统一 API。
- **sprs**：Rust 端提供 `CsMat` 类型，便于在 WASM/N-API 间共享。
- **Arrow/Parquet**：后续支持通过 Arrow CSR 表达（`SparseCSRMatrix`) 直接构造 `SparseMatrix`。

## 测试计划

- 单元测试覆盖：
  - CSR ↔ COO 转换、稀疏-稠密 matmul 与 Node/WASM fallback 一致性。
  - `nnz`、`dtype`、`shape` 等元数据校验。
  - 与 `Matrix` 的互转（小规模 dense）。
- 集成测试：
  - 调用 `trainLinearRegression` 等现有 API 时引入稀疏特征矩阵，验证 fallback 正常工作。
  - 待 SuiteSparse 集成后增加差分测试，确保 N-API 输出与 JS fallback 匹配。

## Roadmap 下一步

1. 在 `packages/js/src` 中实现 `SparseMatrix` 类与基础转换工具。
2. 新增 `sparseMatmul`, `sparseAdd`, `sparseTranspose` JS fallback，并在 docs/tutorial 中加入使用示例。
3. 扩展 `gradcheck` 与 Autograd，让稀疏结果能转换为稠密后参与梯度计算（过渡策略）。
4. N-API 层引入 SuiteSparse（需要额外二进制依赖与构建脚本调整）。
5. 实现 `mmapSparseCSR(path)` 辅助函数，结合 chunked 算法验证外存场景。
