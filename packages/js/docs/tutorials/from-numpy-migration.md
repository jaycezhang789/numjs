# 从 NumPy 迁移指南

> 目标读者：已经熟悉 Python/NumPy，希望将现有科学计算或数据处理代码迁移到 JavaScript/TypeScript 与 `@jayce789/numjs` 的开发者。

## 核心理念对齐

| NumPy | NumJS (`@jayce789/numjs`) | 备注 |
| --- | --- | --- |
| `numpy.ndarray` | `Matrix` | 默认存储为列主序（column-major），便于与 BLAS/LAPACK 互操作。 |
| `dtype` | `DType` | 名称一致：`float32/float64/int32/...`。`fixed64` 是 numjs 特有的十进制定点格式。 |
| Broadcasting | `broadcastTo`, 自动广播规则 | 与 NumPy 一致，严格检查维度。 |
| UFunc | 顶层函数（如 `add`, `exp`, `sin`） | 大多数标量/矩阵算子覆盖；更多函数持续补充。 |
| 视图 / Copy-on-write | 默认延迟复制 | 仅在需要保留旧数据时进行 copy，节省内存。 |

## 迁移步骤速览

1. **梳理数据入口**：确认数据是从文件、网络还是内存中生成，决定是否需要 `readNpy`, `matrixFromBytes`, `new Matrix(typedArray, rows, cols)` 等入口。
2. **映射线性代数操作**：`matmul`, `svd`, `qr`, `solve` 等与 NumPy 接口保持一致，通常仅需替换命名。
3. **处理异步初始化**：NumJS 需要 `await init()` 以加载 N-API 或 WebAssembly 后端，可在顶层 `async` 函数或框架启动阶段调用。
4. **类型系统对接**：在 TypeScript 中，函数签名可声明为 `Matrix | Float64Array` 等联合类型，配合 `matrix.dtype` 判别具体实现。
5. **测试与校验**：使用 `allClose` 替换 NumPy 的 `assert_allclose`，并重用公差 `DEFAULT_RTOL`, `DEFAULT_ATOL`。

```ts
// numpy: c = numpy.matmul(a, b) + bias
import { init, Matrix, matmul, add } from "@jayce789/numjs";

await init();

const a = new Matrix([1, 2, 3, 4], 2, 2);
const b = new Matrix([5, 6, 7, 8], 2, 2);
const bias = new Matrix([1, 1, 1, 1], 2, 2);

const c = add(matmul(a, b.transpose()), bias);
console.log(c.toArray()); // Float64Array [20, 27, 32, 43]
```

## 常见迁移场景

### 1. NPY / NPZ 文件读取

- 使用 `readNpy` 或 `readNpz`（多矩阵）读取由 NumPy 导出的二进制文件。
- 浏览器环境需确保 `.npy` 文件以 `ArrayBuffer` 形式获取后再传递给 NumJS。

```ts
import { readNpy } from "@jayce789/numjs/io";
const buffer = await fetch("/weights.fc1.npy").then(r => r.arrayBuffer());
const matrix = readNpy(new Uint8Array(buffer));
```

### 2. Pandas / Arrow 数据互通

- 推荐使用 `apache-arrow` 或 `nodejs-polars` 作为桥梁：先把 DataFrame 转为 TypedArray，再构建 `Matrix`。
- 对于列式数据，可直接复用底层 `Float64Array`，避免额外复制。

### 3. SciPy 线性代数迁移

- `solve`, `qr`, `svd`, `eigen` 等操作由 Rust/BLAS 支持，接口设计保持 NumPy 风格。
- 若依赖稀疏矩阵或 FFT，可在过渡期将相关部分保留在 Python，通过 WebSocket / WASM worker 进行远程调用，逐步替换。

## 性能调优建议

- **优先使用 N-API 后端**：在 Node.js 环境中默认加载原生 `.node`，与 Python C 扩展性能接近。
- **批量操作**：合并小矩阵运算，减少 FFI 往返；`stack` / `concat` 可帮助组批。
- **避免频繁 `toArray()`**：只在需要与第三方库交互时输出数据，内部计算保持 `Matrix` 类型。
- **固定随机种子**：NumJS 提供 `seedRandom` 以便重现实验。

## 参考资源

- StackBlitz Demo：见 [交互式文档](../interactive/README.md)。
- GitHub `examples/`：`examples/numjs-interactive/src/migration.ts` 展示从 NumPy 脚本移植的端到端案例。
- 常见问题可搜索仓库 Discussions 或开启 Issue。

迁移过程中若发现 API 或性能差异，欢迎反馈，我们会在后续版本中优先修复。
