#!/usr/bin/env node
import { init, backendKind, Matrix, matmul } from "@jayce789/numjs";

async function main() {
  await init();
  console.log(`backendKind(): ${backendKind()}`);

  const createRandomMatrix = (rows, cols) =>
    new Matrix(
      Float64Array.from({ length: rows * cols }, () => Math.random() - 0.5),
      rows,
      cols
    );

  const a = createRandomMatrix(128, 128);
  const b = createRandomMatrix(128, 128);
  console.time("matmul");
  matmul(a, b);
  console.timeEnd("matmul");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
