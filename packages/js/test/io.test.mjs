import test from "node:test";
import assert from "node:assert/strict";
import { promises as fs } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { Readable } from "node:stream";
import "./helpers/polyfill-fetch.mjs";

const {
  init,
  Matrix,
  DataFrameView,
  matrixToArrowTable,
  matrixToPolarsDataFrame,
  arrowTableToMatrix,
  readCsvDataFrame,
  writeDataFrameToCsv,
  readCsvDataFrameFromStream,
  readParquetDataFrame,
  writeParquetDataFrame,
} =
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

test("DataFrameView selection and dtype metadata", () => {
  const base = new Matrix(new Float64Array([1, 0, 0, 1]), 2, 2, { dtype: "float64" });
  const frame = DataFrameView.fromMatrix(base, {
    columns: ["alpha", "beta"],
  }).withColumnDTypes({ beta: "bool" });
  const betaColumn = frame.column("beta");
  assert.equal(betaColumn.dtype, "bool");
  const selected = frame.select(["beta"]);
  assert.deepEqual(selected.columnNames, ["beta"]);
  assert.equal(selected.column("beta").dtype, "bool");
});

test("readCsvDataFrame and writeDataFrameToCsv round-trip", async () => {
  const matrix = new Matrix(new Float64Array([1, 0, 0, 1]), 2, 2, { dtype: "float64" });
  const frame = DataFrameView.fromMatrix(matrix, {
    columns: ["alpha", "beta"],
  }).withColumnDTypes({ beta: "bool" });
  const tempPath = join(tmpdir(), `numjs-${randomUUID()}.csv`);
  await writeDataFrameToCsv(frame, tempPath, { delimiter: "," });
  const read = await readCsvDataFrame(tempPath, {});
  assert.deepEqual(read.columnNames, frame.columnNames);
  assert.equal(read.column("beta").dtype, "bool");
  const rows = read.toObjectRows();
  assert.equal(rows.length, 2);
  await fs.unlink(tempPath);
});

test("readCsvDataFrameFromStream parses streamed input", async () => {
  const csv = "x,y\n1,0\n0,1\n";
  const stream = Readable.toWeb(Readable.from([Buffer.from(csv)]));
  const frame = await readCsvDataFrameFromStream(stream, {});
  assert.deepEqual(frame.columnNames, ["x", "y"]);
  assert.equal(frame.rowCount, 2);
});

test("Parquet helpers inform about missing Polars", async () => {
  await assert.rejects(readParquetDataFrame("/tmp/does-not-exist.parquet"), /polars/i);
  const matrix = new Matrix(new Float64Array([1, 2, 3, 4]), 2, 2, { dtype: "float64" });
  const frame = DataFrameView.fromMatrix(matrix, { columns: ["a", "b"] });
  await assert.rejects(writeParquetDataFrame(frame, "/tmp/does-not-exist.parquet"), /polars/i);
});
