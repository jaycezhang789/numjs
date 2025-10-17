import test from "node:test";
import assert from "node:assert/strict";

let fc;
try {
  ({ default: fc } = await import("fast-check"));
} catch (error) {
  test("property tests skipped (fast-check missing)", (t) => {
    t.skip("Install dev dependency fast-check to run property tests");
  });
}

if (fc) {
  const { init, Matrix, add, matmul } = await import("../dist/index.js");

  await init({ threads: false, webGpu: { forceFallback: true } });

  function expandArray(values, size) {
    if (values.length === size) {
      return values;
    }
    if (values.length > size) {
      return values.slice(0, size);
    }
    const out = new Array(size);
    for (let i = 0; i < size; i += 1) {
      out[i] = values[i % values.length];
    }
    return out;
  }

  function matrixArb() {
    const valueArb = fc.double({ min: -10, max: 10, noDefaultInfinity: true, noNaN: true });
    return fc.record({
      rows: fc.integer({ min: 1, max: 4 }),
      cols: fc.integer({ min: 1, max: 4 }),
      values: fc.array(valueArb, { minLength: 1, maxLength: 16 }),
    }).map(({ rows, cols, values }) => {
      const data = expandArray(values, rows * cols);
      return new Matrix(Float64Array.from(data), rows, cols, { dtype: "float64" });
    });
  }

  test("add agrees with reference addition", async () => {
    await fc.assert(
      fc.asyncProperty(matrixArb(), matrixArb(), async (lhs, rhs) => {
        if (lhs.rows !== rhs.rows || lhs.cols !== rhs.cols) {
          const data = expandArray(Array.from(rhs.toArray()), lhs.rows * lhs.cols);
          rhs = new Matrix(Float64Array.from(data), lhs.rows, lhs.cols, { dtype: "float64" });
        }
        const leftArray = Array.from(lhs.toArray());
        const rightArray = Array.from(rhs.toArray());
        const result = add(lhs, rhs).toArray();
        const reference = leftArray.map((value, index) => value + rightArray[index]);
        assert.deepEqual(Array.from(result), reference);
      }),
      { numRuns: 50 }
    );
  });

  test("matmul agrees with reference matmul", async () => {
    const tupleArb = fc.record({
      m: fc.integer({ min: 1, max: 3 }),
      k: fc.integer({ min: 1, max: 3 }),
      n: fc.integer({ min: 1, max: 3 }),
      leftValues: fc.array(fc.double({ min: -5, max: 5, noNaN: true, noDefaultInfinity: true }), { minLength: 1, maxLength: 16 }),
      rightValues: fc.array(fc.double({ min: -5, max: 5, noNaN: true, noDefaultInfinity: true }), { minLength: 1, maxLength: 16 }),
    });

    await fc.assert(
      fc.asyncProperty(tupleArb, async ({ m, k, n, leftValues, rightValues }) => {
        const left = new Matrix(Float64Array.from(expandArray(leftValues, m * k)), m, k, { dtype: "float64" });
        const right = new Matrix(Float64Array.from(expandArray(rightValues, k * n)), k, n, { dtype: "float64" });
        const product = matmul(left, right).astype("float64", { copy: false }).toArray();
        const leftArray = left.toArray();
        const rightArray = right.toArray();
        for (let row = 0; row < m; row += 1) {
          for (let col = 0; col < n; col += 1) {
            let acc = 0;
            for (let shared = 0; shared < k; shared += 1) {
              acc += leftArray[row * k + shared] * rightArray[shared * n + col];
            }
            const idx = row * n + col;
            assert.ok(Math.abs(product[idx] - acc) < 1e-9);
          }
        }
      }),
      { numRuns: 40 }
    );
  });
}
