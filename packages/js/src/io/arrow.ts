import type { DType } from "../index";
import { Matrix } from "../index";

type NumericTypedArray = Float64Array | Float32Array | Int32Array;

type ArrowVectorLike = {
  length: number;
  toArray(): ArrayLike<number> | ArrayLike<bigint> | ArrayLike<boolean>;
  name?: string;
  type?: { toString(): string };
};

type ArrowColumnAccessor = (index: number) => ArrowVectorLike | null | undefined;

export type ArrowTableLike = {
  numRows?: number;
  length?: number;
  numCols?: number;
  schema?: { fields?: Array<{ name: string }> };
  getColumnAt?: ArrowColumnAccessor;
  getChildAt?: ArrowColumnAccessor;
  getColumn?: ArrowColumnAccessor;
  [Symbol.iterator]?(): IterableIterator<ArrowVectorLike>;
};

export type ArrowToMatrixOptions = {
  columns?: readonly string[];
  dtype?: DType;
};

export type ArrowConversionResult = {
  matrix: Matrix;
  columnNames: string[];
};

export async function matrixToArrowTable(
  matrix: Matrix,
  columnNames: readonly string[],
  options: {
    arrowModule?: unknown;
    coerceTo?: "float64" | "float32" | "int32";
  } = {}
): Promise<unknown> {
  if (columnNames.length === 0) {
    throw new Error("matrixToArrowTable requires at least one column name");
  }
  if (columnNames.length !== matrix.cols) {
    throw new Error(
      `matrixToArrowTable: expected ${matrix.cols} column names, received ${columnNames.length}`
    );
  }
  const arrow = await loadArrowModule(options.arrowModule);
  const converter = resolveArrowTableBuilder(arrow);
  const coerceTarget = options.coerceTo ?? "float64";
  const arrays: Record<string, NumericTypedArray> = {};
  for (let col = 0; col < matrix.cols; col += 1) {
    const columnData = extractMatrixColumn(matrix, col, coerceTarget);
    arrays[columnNames[col]] = columnData;
  }
  return converter(arrays, arrow);
}

export function arrowTableToMatrix(
  table: ArrowTableLike,
  options: ArrowToMatrixOptions = {}
): ArrowConversionResult {
  const columnNames = resolveColumnNames(table);
  const selectedNames =
    options.columns && options.columns.length > 0 ? options.columns : columnNames;
  const columnCount = selectedNames.length;
  if (columnCount === 0) {
    throw new Error("arrowTableToMatrix: no columns selected");
  }
  const rows = resolveRowCount(table);
  const dtype = options.dtype ?? "float64";
  const buffer = new Float64Array(rows * columnCount);
  const resolvedNames: string[] = [];
  for (let columnIndex = 0; columnIndex < columnCount; columnIndex += 1) {
    const name = selectedNames[columnIndex];
    const vector = getVectorByName(table, columnNames, name, columnIndex);
    const values = normalizeArrowVector(vector);
    if (values.length !== rows) {
      throw new Error(
        `arrowTableToMatrix: column "${name}" length (${values.length}) does not match row count (${rows})`
      );
    }
    for (let row = 0; row < rows; row += 1) {
      buffer[row * columnCount + columnIndex] = Number(values[row]);
    }
    resolvedNames.push(name);
  }
  const matrix = new Matrix(buffer, rows, columnCount, { dtype });
  if (dtype !== "float64") {
    return {
      matrix: matrix.astype(dtype, { copy: false }),
      columnNames: resolvedNames,
    };
  }
  return { matrix, columnNames: resolvedNames };
}

function normalizeArrowVector(
  vector: ArrowVectorLike
): ArrayLike<number | bigint | boolean> {
  if (!vector || typeof vector.toArray !== "function") {
    throw new Error("arrowTableToMatrix: encountered column without toArray()");
  }
  const array = vector.toArray();
  if (!Array.isArray(array) && !ArrayBuffer.isView(array)) {
    return Array.from(array as ArrayLike<number>);
  }
  return array;
}

function resolveColumnNames(table: ArrowTableLike): string[] {
  if (table.schema?.fields?.length) {
    return table.schema.fields.map((field) => field.name);
  }
  const names: string[] = [];
  const iterator = table[Symbol.iterator]?.();
  if (iterator) {
    let index = 0;
    for (const column of iterator) {
      names.push(column.name ?? `col_${index}`);
      index += 1;
    }
    return names;
  }
  const numCols = table.numCols ?? 0;
  const accessor = table.getColumnAt ?? table.getChildAt ?? table.getColumn;
  if (accessor && numCols > 0) {
    for (let index = 0; index < numCols; index += 1) {
      const vector = accessor.call(table, index);
      names.push(vector?.name ?? `col_${index}`);
    }
    return names;
  }
  throw new Error("arrowTableToMatrix: unable to determine column names");
}

function resolveRowCount(table: ArrowTableLike): number {
  if (typeof table.numRows === "number") return table.numRows;
  if (typeof table.length === "number") return table.length;
  const iterator = table[Symbol.iterator]?.();
  if (iterator) {
    const first = iterator.next();
    if (first.value && typeof first.value.length === "number") {
      return first.value.length;
    }
  }
  throw new Error("arrowTableToMatrix: unable to determine row count");
}

function getVectorByName(
  table: ArrowTableLike,
  columnNames: string[],
  name: string,
  fallbackIndex: number
): ArrowVectorLike {
  const index = columnNames.indexOf(name);
  const resolvedIndex = index >= 0 ? index : fallbackIndex;
  const accessor = table.getColumnAt ?? table.getChildAt ?? table.getColumn;
  if (accessor) {
    const column = accessor.call(table, resolvedIndex);
    if (column) {
      return column;
    }
  }
  const iterator = table[Symbol.iterator]?.();
  if (iterator) {
    let currentIndex = 0;
    for (const column of iterator) {
      if (currentIndex === resolvedIndex) {
        return column;
      }
      currentIndex += 1;
    }
  }
  throw new Error(`arrowTableToMatrix: failed to access column "${name}"`);
}

function extractMatrixColumn(
  matrix: Matrix,
  columnIndex: number,
  target: "float64" | "float32" | "int32"
): NumericTypedArray {
  const colMatrix = matrix.column(columnIndex).astype("float64", { copy: false });
  const data = colMatrix.toArray();
  if (data instanceof Float64Array) {
    switch (target) {
      case "float64":
        return data.slice();
      case "float32":
        return Float32Array.from(data);
      case "int32":
        return Int32Array.from(data);
    }
  }
  const float64 = Float64Array.from(data as ArrayLike<number>);
  if (target === "float64") {
    return float64;
  }
  if (target === "float32") {
    return Float32Array.from(float64);
  }
  return Int32Array.from(float64);
}

async function loadArrowModule(provided: unknown): Promise<any> {
  if (provided) {
    return provided;
  }
  const dynamicImport = new Function("specifier", "return import(specifier);");
  try {
    return await dynamicImport("apache-arrow");
  } catch (error) {
    throw new Error(
      'Unable to load "apache-arrow". Install it and pass the module via options.arrowModule if bundlers prevent dynamic import.'
    );
  }
}

function resolveArrowTableBuilder(
  arrow: any
): (arrays: Record<string, NumericTypedArray>, module: any) => unknown {
  if (typeof arrow?.tableFromArrays === "function") {
    return (arrays) => arrow.tableFromArrays(arrays);
  }
  if (typeof arrow?.Table?.from === "function") {
    return (arrays) => arrow.Table.from(arrays);
  }
  if (typeof arrow?.Table?.new === "function") {
    return (arrays, module) => {
      const fields = Object.keys(arrays).map(
        (name) => new module.Field(name, new module.Float64(), true)
      );
      const columns = Object.values(arrays).map(
        (array: NumericTypedArray) => module.Vector.from(array)
      );
      const schema = new module.Schema(fields);
      return module.Table.new(columns, schema);
    };
  }
  throw new Error(
    "The loaded apache-arrow module does not expose a recognised factory (tableFromArrays/Table.from/Table.new)"
  );
}
