# numjs Workspace

`numjs` 是一个多后端（N-API / WebAssembly）协同的数值计算项目，目标是在 JavaScript / TypeScript 中提供接近 NumPy 的体验，并在不同运行时之间自动选择最快的实现。

Workspace 目录结构：

- `packages/core-rs`：Rust 实现的核心算子引擎（矩阵存储、广播、花式索引、dtype 管理等）。
- `packages/napi`：Node.js N-API 绑定，提供最高性能的原生实现。
- `packages/wasm`：使用 `wasm-bindgen` 导出的 WebAssembly 版本，面向浏览器及受限环境。
- `packages/js`：对外发布的 npm 包，负责后端加载、API 暴露与 JS 侧兼容逻辑。

## 安装

```bash
npm install @jayce789/numjs
# 或
yarn add @jayce789/numjs
# 或
pnpm add @jayce789/numjs
```

## 快速上手

```ts
import {
  init,
  Matrix,
  add,
  matmul,
  where,
  take,
  gather,
  scatter,
  backendKind,
} from "@jayce789/numjs";

await init(); // 自动加载 N-API，失败时回落到 wasm

console.log(backendKind()); // "napi" | "wasm"

const a = new Matrix([1, 2, 3, 4], 2, 2);
const b = new Matrix([5, 6, 7, 8], 2, 2);

const sum = add(a, b);
const product = matmul(a, b.transpose());

const mask = new Matrix([true, false, false, true], 2, 2);
const choice = new Matrix([10, 10, 10, 10], 2, 2).astype("int32");
const fallback = new Matrix([0, 0, 0, 0], 2, 2).astype("int32");

const selected = where([mask], [choice], fallback);
console.log(Array.from(selected.toArray())); // [10, 0, 0, 10]

const rows = take(a, 0, [1, 0]);                     // 花式行索引
const picks = gather(a, [0, 1], [1]);                // 外积式取值
const updated = scatter(a, [0, 1], [1], new Matrix([9, 9], 2, 1));
```

## Matrix API 概览

### 构造

```ts
const base = new Matrix(data, rows, cols, { dtype?: DType });
const fromBytes = Matrix.fromBytes(byteView, rows, cols, dtype);
```

支持的 `data`：

- `number[]` / `boolean[]`
- 常见 `TypedArray`（含 `BigInt64Array` / `BigUint64Array`）

构造时可显式制定 `dtype`，否则依据输入自动推断。

### 属性 & 转换

```ts
base.rows;            // number
base.cols;            // number
base.dtype;           // "float64" 等
base.dtypeInfo;       // { size, kind, isFloat, isSigned }
base.toArray();       // 根据 dtype 返回对应 TypedArray/数组
base.toBytes();       // Uint8Array 形式的原始缓冲
base.astype("int32"); // 智能复用或复制，宽度相同的 dtype 会 reinterpret
base.astype("int32", { copy: true }); // 强制复制
```

Rust 端实现了 copy-on-write，切片、转置、宽度相同的 dtype 转换等都不会复制底层缓冲区。

## 顶层函数

顶层函数全部返回新的 `Matrix`，并自动处理 dtype 对齐、广播与视图展开：

| 函数 | 说明 |
| ---- | ---- |
| `add(a, b)` | 逐元素加法，自动 dtype 提升 |
| `matmul(a, b)` | 常规矩阵乘 |
| `clip(matrix, min, max)` | 对元素做区间裁剪 |
| `where(condition, truthy, falsy?, options?)` | 支持单条件或多条件广播选择，可指定默认值 |
| `concat(a, b, axis)` / `stack(a, b, axis)` | 沿行/列拼接，内部做 dtype 对齐 |
| `take(matrix, axis, indices)` | 沿指定轴进行花式索引 |
| `put(matrix, axis, indices, values)` | 沿指定轴写入数据（外积模式） |
| `gather(matrix, rowIndices, colIndices)` | 外积式选择子矩阵 |
| `gatherPairs(matrix, rowIndices, colIndices)` | 配对选择（索引一一对应） |
| `scatter(matrix, rowIndices, colIndices, values)` | 外积式写入 |
| `scatterPairs(matrix, rowIndices, colIndices, values)` | 配对写入 |
| `copyBytesTotal()` / `takeCopyBytes()` / `resetCopyBytes()` | 报告自 Rust 侧记录的复制字节数 |
| `backendKind()` / `backendCapabilities()` | 查询当前后端类型及可用特性 |

当所选后端缺失某些算子（例如 wasm 暂未实现 `where_select_multi`）时，JS 层会自动回退到广播 + 显式复制的实现，保证功能行为一致。

## 复制监测

可依赖指标评估方案性能：

```ts
takeCopyBytes(); // 返回自上次调用以来的复制字节数并清零
resetCopyBytes(); // 手动清零
```

在测试或性能分析时，可以在关键操作前后调用该接口，判断是否触发了额外复制。

## 运行时差异

`backendCapabilities()` 返回如下信息：

```ts
{
  kind: "napi" | "wasm",
  supportsMatrixFromBytes: boolean,
  supportsReadNpy: boolean,
  supportsWriteNpy: boolean,
  supportsCopyMetrics: boolean,
  supportedDTypes: DType[]
}
```

可据此决定是否下发 `from_bytes`、`writeNpy` 等特性。

## 开发提示

仓库包含多语言链路，建议在提交前运行：

```bash
cargo check                         # 校验核心/绑定层 Rust 代码
npm --prefix packages/js test       # 运行 JS 层测试
```

欢迎通过 issue / PR 讨论算子、dtype、新后端等话题。
