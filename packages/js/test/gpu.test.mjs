import test from "node:test";
import assert from "node:assert/strict";

import {
  init,
  Matrix,
  matmul,
  matmulAsync,
  sum,
  sumAsync,
} from "../dist/index.js";

test("matmulAsync matches CPU matmul result", async () => {
  await init();
  const a = new Matrix(
    new Float32Array([1, 2, 3, 4, 5, 6]),
    2,
    3,
    { dtype: "float32" }
  );
  const b = new Matrix(
    new Float32Array([7, 8, 9, 10, 11, 12]),
    3,
    2,
    { dtype: "float32" }
  );
  const asyncResult = await matmulAsync(a, b);
  const cpuResult = matmul(a, b).astype("float32", { copy: true });
  assert.deepEqual(
    Array.from(asyncResult.toArray()),
    Array.from(cpuResult.toArray())
  );
});

test("sumAsync matches CPU sum result", async () => {
  await init();
  const matrix = new Matrix(
    new Float32Array([1, 2, 3, 4]),
    2,
    2,
    { dtype: "float32" }
  );
  const asyncSum = await sumAsync(matrix, { dtype: "float32" });
  const cpuSum = sum(matrix, { dtype: "float32" });
  assert.deepEqual(
    Array.from(asyncSum.toArray()),
    Array.from(cpuSum.toArray())
  );
});
