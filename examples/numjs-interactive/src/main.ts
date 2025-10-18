import { init, Matrix, add, matmul, backendKind } from "@jayce789/numjs";
import { logBackendSupport } from "./backends";
import { runMigrationSample } from "./migration";

const output = document.querySelector<HTMLPreElement>("#output");

function append(message: string) {
  if (!output) return;
  output.textContent += `${message}\n`;
}

async function main() {
  output?.replaceChildren();
  append("🔄 初始化 NumJS...");

  await init();
  append(`✅ 当前后端: ${backendKind()}`);

  const a = new Matrix([1, 2, 3, 4], 2, 2);
  const b = new Matrix([5, 6, 7, 8], 2, 2);
  const identity = new Matrix([1, 0, 0, 1], 2, 2);
  const result = add(matmul(a, b.transpose()), identity);

  append("🧮 计算 (A @ Bᵀ) + I:");
  append(JSON.stringify(Array.from(result.toArray())));

  await logBackendSupport(append);
  await runMigrationSample(append);

  append("ℹ️ 试着修改 src/*.ts 查看实时结果。");
}

main().catch((error) => {
  console.error(error);
  append(`❌ 初始化失败: ${error instanceof Error ? error.message : String(error)}`);
});
