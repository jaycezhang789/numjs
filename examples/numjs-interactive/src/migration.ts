import { Matrix, matmul, add, allClose } from "@jayce789/numjs";

type Logger = (message: string) => void;

const numpyReference = new Float64Array([20, 27, 32, 43]);

export async function runMigrationSample(log: Logger) {
  log("📘 NumPy ➜ NumJS 迁移动作演示：");

  const a = new Matrix([1, 2, 3, 4], 2, 2);
  const b = new Matrix([5, 6, 7, 8], 2, 2);
  const bias = new Matrix([1, 1, 1, 1], 2, 2);

  const c = add(matmul(a, b.transpose()), bias);

  log(`• 结果: ${JSON.stringify(Array.from(c.toArray()))}`);
  log(`• 与 NumPy 参考比较: ${allClose(c, new Matrix(numpyReference, 2, 2)) ? "通过" : "未通过"}`);
  log("• 可尝试修改数据或算子，观察实时刷新。");
}
