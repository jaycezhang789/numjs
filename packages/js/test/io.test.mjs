import test from "node:test";
import assert from "node:assert/strict";
import "./helpers/polyfill-fetch.mjs";

const { init, Matrix, matrixToArrowTable, matrixToPolarsDataFrame, arrowTableToMatrix } =
  await import("../dist/index.js");

await init({ threads: false });

test("matrixToArrowTable throws when apache-arrow is missing", async () => {
  const matrix = new Matrix(new Float64Array([1, 2, 3, 4]), 2, 2, { dtype: "float64" });
  await assert.rejects(
    matrixToArrowTable(matrix, ["a", "b"]),
    /apache-arrow/i,
    "should instruct the user to install apache-arrow"
  );
});

test("matrixToPolarsDataFrame throws when Polars module is missing", async () => {
  const matrix = new Matrix(new Float64Array([1, 2, 3, 4]), 2, 2, { dtype: "float64" });
  await assert.rejects(
    matrixToPolarsDataFrame(matrix, { columnNames: ["x", "y"] }),
    /Polars module/i,
    "should explain how to install polars bindings"
  );
});

test("arrowTableToMatrix validates column lengths", () => {
  const dummyColumn = {
    name: "a",
    toArray() {
      return new Float64Array([1, 2, 3]);
    },
  };
  const table = {
    schema: { fields: [{ name: "a" }] },
    numRows: 2,
    getColumnAt() {
      return dummyColumn;
    },
  };
  assert.throws(
    () => arrowTableToMatrix(table),
    /does not match row count/,
    "should guard against inconsistent table metadata"
  );
});
