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
  im2col,
  maxPool,
  avgPool,
  fftAxis,
  ifftAxis,
  powerSpectrum,
  runPythonScript,
  pythonTransformMatrix,
  loadOnnxModel,
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

test("im2col expands patches into columns", () => {
  const input = new Matrix(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]), 3, 3, {
    dtype: "float32",
  });
  const colsMatrix = im2col(input, 2, 2);
  assert.equal(colsMatrix.rows, 4);
  assert.equal(colsMatrix.cols, 4);
  const expected = [
    [1, 2, 4, 5],
    [2, 3, 5, 6],
    [4, 5, 7, 8],
    [5, 6, 8, 9],
  ];
  const actual = colsMatrix.toArray();
  for (let col = 0; col < colsMatrix.cols; col += 1) {
    for (let row = 0; row < colsMatrix.rows; row += 1) {
      assert.equal(actual[row * colsMatrix.cols + col], expected[row][col]);
    }
  }
});

test("maxPool and avgPool reduce windows", () => {
  const input = new Matrix(
    new Float32Array([
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16,
    ]),
    4,
    4,
    {
    dtype: "float32",
  }
  );
  const maxed = maxPool(input, 2, 2, { stride: 2 });
  const avged = avgPool(input, 2, 2, { stride: 2 });
  assert.deepEqual(Array.from(maxed.toArray()), [6, 8, 14, 16]);
  assert.deepEqual(Array.from(avged.toArray()), [3.5, 5.5, 11.5, 13.5]);
});

test("fftAxis and ifftAxis round-trip real signals", () => {
  const signal = new Matrix(new Float64Array([1, 0, 0, 0]), 1, 4, { dtype: "float64" });
  const spectrum = fftAxis(signal, 1);
  assert.deepEqual(Array.from(spectrum.real.toArray()), [1, 1, 1, 1]);
  assert.deepEqual(Array.from(spectrum.imag.toArray()), [0, 0, 0, 0]);
  const inverted = ifftAxis(spectrum.real, spectrum.imag, 1);
  const recovered = inverted.real.toArray();
  assert.ok(
    Array.from(recovered).every((value, index) =>
      Math.abs(value - (index === 0 ? 1 : 0)) < 1e-6
    )
  );
});

test("powerSpectrum computes magnitude", () => {
  const signal = new Matrix(new Float64Array([0, 1, 0, -1]), 1, 4, { dtype: "float64" });
  const spectrum = powerSpectrum(signal, 1);
  const values = spectrum.toArray();
  assert.equal(values.length, 4);
  assert.ok(values.every((v) => v >= 0));
});

test("runPythonScript surfaces interpreter errors", async () => {
  await assert.rejects(
    runPythonScript("print('hello')", { pythonPath: "__nonexistent_python__" }),
    /requires a functional Python interpreter/
  );
});

test("pythonTransformMatrix propagates run errors", async () => {
  const matrix = new Matrix(new Float64Array([1, 2, 3, 4]), 2, 2, { dtype: "float64" });
  await assert.rejects(
    pythonTransformMatrix(matrix, "print('noop')", { pythonPath: "__nonexistent_python__" }),
    /requires a functional Python interpreter/
  );
});

test("loadOnnxModel requires onnxruntime-node", async () => {
  await assert.rejects(loadOnnxModel(join(tmpdir(), `${randomUUID()}.onnx`)), /onnxruntime-node/);
});
