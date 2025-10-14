import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const originalFetch = globalThis.fetch;
globalThis.fetch = async (resource, options) => {
  if (resource instanceof URL && resource.protocol === "file:") {
    const path = fileURLToPath(resource);
    const bytes = await readFile(path);
    return new Response(bytes);
  }
  if (typeof resource === "string" && resource.startsWith("file://")) {
    const path = fileURLToPath(new URL(resource));
    const bytes = await readFile(path);
    return new Response(bytes);
  }
  if (originalFetch) {
    return originalFetch(resource, options);
  }
  throw new Error("No fetch implementation available");
};

import {
  init,
  Matrix,
  add,
  stack,
  concat,
  DTYPE_INFO,
  writeNpy,
  readNpy,
  writeNpz,
  readNpz,
  matrixFromBytes,
  where,
  take,
  put,
  gather,
  gatherPairs,
  scatter,
  scatterPairs,
  backendCapabilities,
  takeCopyBytes,
} from "../dist/index.js";

await init();

test("Matrix.astype returns original instance for view casts", () => {
  const matrix = new Matrix([1, 2], 1, 2);
  const same = matrix.astype("float64");

  assert.strictEqual(same, matrix);
  assert.equal(same.dtype, "float64");
  assert.ok(same.toArray() instanceof Float64Array);
});

test("Matrix.astype respects copy option when dtype matches", () => {
  const matrix = new Matrix([3, 4], 1, 2);
  const copy = matrix.astype("float64", { copy: true });

  assert.notStrictEqual(copy, matrix);
  assert.equal(copy.dtype, "float64");
  assert.deepEqual(Array.from(copy.toArray()), [3, 4]);
});


test("Matrix.astype reinterprets same-width dtype without copy", () => {
  const matrix = new Matrix(new Uint32Array([1, 0]), 1, 2);
  takeCopyBytes();
  const view = matrix.astype("int32");
  assert.notStrictEqual(view, matrix);
  assert.equal(view.dtype, "int32");
  assert.deepEqual(Array.from(view.toArray()), [1, 0]);
  assert.equal(takeCopyBytes(), 0);

  const copied = matrix.astype("int32", { copy: true });
  assert.equal(copied.dtype, "int32");
  assert.deepEqual(Array.from(copied.toArray()), [1, 0]);
  const expectedBytes = matrix.dtypeInfo.size * matrix.rows * matrix.cols;
  assert.equal(takeCopyBytes(), expectedBytes);
});

test("Binary ops promote dtypes using promotion table", () => {
  const lhs = new Matrix([1, 2], 1, 2).astype("int16");
  const rhs = new Matrix([3, 4], 1, 2).astype("uint8");

  const sum = add(lhs, rhs);
  assert.equal(sum.dtype, "int32");
  const sumArray = sum.toArray();
  assert.ok(sumArray instanceof Int32Array);
  assert.deepEqual(Array.from(sumArray), [4, 6]);

  const stacked = concat(lhs, rhs, 0);
  assert.equal(stacked.dtype, "int32");
  const stackedArray = stacked.toArray();
  assert.ok(stackedArray instanceof Int32Array);
  assert.deepEqual(Array.from(stackedArray), [1, 2, 3, 4]);

  const stackedMany = concat(lhs, concat(lhs, rhs, 0), 0);
  assert.equal(stackedMany.dtype, "int32");
  assert.deepEqual(Array.from(stackedMany.toArray()), [1, 2, 1, 2, 3, 4]);

  const boolLift = add(
    new Matrix([true, false], 1, 2).astype("bool"),
    new Matrix([1, 1], 1, 2).astype("uint8")
  );
  assert.equal(boolLift.dtype, "uint8");
  assert.deepEqual(Array.from(boolLift.toArray()), [2, 1]);
  const concatAxis1 = concat(lhs, rhs, 1);
  assert.equal(concatAxis1.dtype, "int32");
  assert.deepEqual(Array.from(concatAxis1.toArray()), [1, 2, 3, 4]);

  const stackedAxis = stack(lhs, rhs, 0);
  assert.equal(stackedAxis.dtype, "int32");
  assert.deepEqual(Array.from(stackedAxis.toArray()), [1, 2, 3, 4]);
});

test("dtype metadata exposes size and kind information", () => {
  assert.equal(DTYPE_INFO.float32.size, 4);
  assert.equal(DTYPE_INFO.uint8.kind, "unsigned");
  assert.equal(DTYPE_INFO.int16.isSigned, true);
  const matrix = new Matrix([1, 0], 1, 2).astype("uint8");
  assert.equal(matrix.dtypeInfo.kind, "unsigned");
  assert.equal(matrix.dtypeInfo.size, 1);
  const boolMatrix = new Matrix([0, 1], 1, 2).astype("bool");
  assert.equal(boolMatrix.dtypeInfo.kind, "bool");
  assert.equal(boolMatrix.dtypeInfo.size, 1);
});

test("Matrix constructor infers dtype from typed arrays", () => {
  const matrix = new Matrix(new Int16Array([1, -2, 3, -4]), 2, 2);
  assert.equal(matrix.dtype, "int16");
  const array = matrix.toArray();
  assert.ok(array instanceof Int16Array);
  assert.deepEqual(Array.from(array), [1, -2, 3, -4]);
});

test("Matrix constructor allows overriding dtype", () => {
  const matrix = new Matrix(new Float32Array([1.1, 2.2]), 1, 2, { dtype: "float64" });
  assert.equal(matrix.dtype, "float64");
  const array = matrix.toArray();
  assert.ok(array instanceof Float64Array);
  const values = Array.from(array);
  assert.ok(values.every((value, index) => Math.abs(value - [1.1, 2.2][index]) < 1e-6));
});

test("Matrix constructor validates length against shape", () => {
  assert.throws(
    () => new Matrix([1, 2, 3], 2, 2),
    /does not match shape/i
  );
  assert.throws(
    () => new Matrix(new Int8Array([1, 2, 3]), 1, 4),
    /does not match shape/i
  );
});

test("Matrix preserves int64 typed array values without precision loss", () => {
  const big = new BigInt64Array([
    9007199254740993n,
    -9007199254740993n,
  ]);
  const matrix = new Matrix(big, 1, 2);
  assert.equal(matrix.dtype, "int64");
  const array = matrix.toArray();
  assert.ok(array instanceof BigInt64Array);
  assert.deepEqual(Array.from(array), Array.from(big));
});

test("writeNpy/readNpy preserve dtype and data", (t) => {
  const original = new Matrix(new Uint16Array([1, 65535]), 1, 2);
  let bytes;
  try {
    bytes = writeNpy(original);
  } catch (error) {
    if (String(error).includes("not supported")) {
      t.skip("Current backend does not support write_npy");
      return;
    }
    throw error;
  }
  const roundtrip = readNpy(bytes);
  assert.equal(roundtrip.dtype, "uint16");
  const array = roundtrip.toArray();
  assert.ok(array instanceof Uint16Array);
  assert.deepEqual(Array.from(array), [1, 65535]);
});

test("writeNpz/readNpz roundtrip keeps dtype metadata", (t) => {
  const lhs = new Matrix(new Int8Array([1, -2]), 1, 2);
  const rhs = new Matrix(new Float32Array([0.1, 0.2]), 1, 2);
  let archive;
  try {
    archive = writeNpz([
      { name: "lhs", matrix: lhs },
      { name: "rhs", matrix: rhs },
    ]);
  } catch (error) {
    if (String(error).includes("not supported")) {
      t.skip("Current backend does not support write_npy/write_npz");
      return;
    }
    throw error;
  }
  const entries = readNpz(archive);
  const left = entries.find((entry) => entry.name === "lhs");
  const right = entries.find((entry) => entry.name === "rhs");
  assert.ok(left && right);
  assert.equal(left.matrix.dtype, "int8");
  assert.equal(right.matrix.dtype, "float32");
  assert.deepEqual(Array.from(left.matrix.toArray()), [1, -2]);
  const rightArray = Array.from(right.matrix.toArray());
  assert.ok(rightArray.every((value, index) => Math.abs(value - [0.1, 0.2][index]) < 1e-6));
});

test("where supports multiple conditions with fallback", () => {
  const condA = new Matrix([true, false, false, true], 2, 2);
  const condB = new Matrix([false, true, false, false], 2, 2);
  const choiceA = new Matrix([10, 10, 10, 10], 2, 2).astype("int32");
  const choiceB = new Matrix([20, 20, 20, 20], 2, 2).astype("int32");
  const fallback = new Matrix([0, 0, 0, 0], 2, 2).astype("int32");

  const result = where([condA, condB], [choiceA, choiceB], fallback);
  assert.equal(result.dtype, "int32");
  assert.deepEqual(Array.from(result.toArray()), [10, 20, 0, 10]);
});

test("take and put operate along specified axis", () => {
  const base = new Matrix([1, 2, 3, 4, 5, 6], 3, 2).astype("int32");
  const taken = take(base, 0, [2, 0]);
  assert.equal(taken.rows, 2);
  assert.equal(taken.cols, 2);
  assert.deepEqual(Array.from(taken.toArray()), [5, 6, 1, 2]);

  const updates = new Matrix([10, 11, 12, 13], 2, 2).astype("int32");
  const updated = put(base, 0, [0, 2], updates);
  assert.deepEqual(Array.from(updated.toArray()), [10, 11, 3, 4, 12, 13]);
});

test("gather and scatter support fancy indexing", () => {
  const source = new Matrix([1, 2, 3, 4, 5, 6], 3, 2).astype("int32");
  const gathered = gather(source, [0, 2], [1]);
  assert.equal(gathered.rows, 2);
  assert.equal(gathered.cols, 1);
  assert.deepEqual(Array.from(gathered.toArray()), [2, 6]);

  const gatheredPairs = gatherPairs(source, [0, 2], [0, 1]);
  assert.equal(gatheredPairs.rows, 2);
  assert.equal(gatheredPairs.cols, 1);
  assert.deepEqual(Array.from(gatheredPairs.toArray()), [1, 6]);

  const replacements = new Matrix([100, 200], 2, 1).astype("int32");
  const scattered = scatter(source, [0, 2], [1], replacements);
  assert.deepEqual(Array.from(scattered.toArray()), [1, 100, 3, 4, 5, 200]);

  const pairValues = new Matrix([7, 8], 2, 1).astype("int32");
  const scatteredPairs = scatterPairs(source, [1, 2], [0, 1], pairValues);
  assert.deepEqual(Array.from(scatteredPairs.toArray()), [1, 2, 7, 4, 5, 8]);
});

test("backendCapabilities exposes feature flags", () => {
  const caps = backendCapabilities();
  assert.ok(Array.isArray(caps.supportedDTypes));
  assert.ok(caps.supportedDTypes.includes("float64"));
  assert.equal(typeof caps.supportsMatrixFromBytes, "boolean");
});

test("matrixFromBytes constructs matrices from raw buffers", (t) => {
  const caps = backendCapabilities();
  if (!caps.supportsMatrixFromBytes) {
    t.skip("Current backend does not support Matrix.from_bytes");
    return;
  }
  const source = new Uint8Array(new Uint32Array([10, 20, 30, 40]).buffer);
  const matrix = matrixFromBytes(source, 2, 2, "uint32");
  assert.equal(matrix.dtype, "uint32");
  assert.deepEqual(Array.from(matrix.toArray()), [10, 20, 30, 40]);
});
