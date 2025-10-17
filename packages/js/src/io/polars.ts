import { Matrix } from "../index";

export type PolarsModuleLike = {
  DataFrame: new (data: Record<string, unknown[]>) => any;
  Series?: new (name: string, values: unknown[]) => any;
  readParquet?: (path: string, options?: unknown) => Promise<unknown>;
  writeParquet?: (path: string, dataframe: unknown, options?: unknown) => Promise<void>;
};

type PolarsSeriesLike = {
  name?: string;
  toArray(): unknown[];
  dtype?: string;
};

type PolarsDataFrameLike = {
  height?: number;
  width?: number;
  columns?: string[];
  shape?: [number, number];
  getColumns?(): PolarsSeriesLike[];
  getColumn?(name: string): PolarsSeriesLike;
  column?(name: string): PolarsSeriesLike;
  toStruct?(): Record<string, unknown[]>;
};

export type PolarsToMatrixOptions = {
  columns?: readonly string[];
};

export type MatrixToPolarsOptions = {
  columnNames: readonly string[];
  polarsModule?: unknown;
};

export function polarsDataFrameToMatrix(
  dataframe: PolarsDataFrameLike,
  options: PolarsToMatrixOptions = {}
): { matrix: Matrix; columnNames: string[] } {
  const columns = resolvePolarsColumns(dataframe);
  const selected =
    options.columns && options.columns.length > 0 ? options.columns : columns.map((col) => col.name);
  if (!selected || selected.length === 0) {
    throw new Error("polarsDataFrameToMatrix: no columns available");
  }
  const rows = resolvePolarsRowCount(dataframe);
  const buffer = new Float64Array(rows * selected.length);
  const resolvedNames: string[] = [];
  for (let columnIndex = 0; columnIndex < selected.length; columnIndex += 1) {
    const name = selected[columnIndex];
    const series = resolveSeries(dataframe, columns, name);
    const values = series.toArray();
    if (values.length !== rows) {
      throw new Error(
        `polarsDataFrameToMatrix: column "${name}" length (${values.length}) does not match row count (${rows})`
      );
    }
    for (let row = 0; row < rows; row += 1) {
      buffer[row * selected.length + columnIndex] = Number(values[row]);
    }
    resolvedNames.push(name);
  }
  return {
    matrix: new Matrix(buffer, rows, selected.length, { dtype: "float64" }),
    columnNames: resolvedNames,
  };
}

export async function matrixToPolarsDataFrame(
  matrix: Matrix,
  options: MatrixToPolarsOptions
): Promise<unknown> {
  if (!options.columnNames || options.columnNames.length === 0) {
    throw new Error("matrixToPolarsDataFrame requires columnNames");
  }
  if (options.columnNames.length !== matrix.cols) {
    throw new Error(
      `matrixToPolarsDataFrame: expected ${matrix.cols} column names, received ${options.columnNames.length}`
    );
  }
  const polars = await ensurePolarsModule(options.polarsModule);
  const columns: Record<string, unknown[]> = {};
  for (let col = 0; col < matrix.cols; col += 1) {
    const seriesData = matrix
      .column(col)
      .astype("float64", { copy: false })
      .toArray() as ArrayLike<number>;
    columns[options.columnNames[col]] = Array.from(seriesData);
  }
  if (typeof polars.DataFrame === "function") {
    return new polars.DataFrame(columns);
  }
  throw new Error("Loaded Polars module does not expose a DataFrame constructor");
}

function resolvePolarsColumns(dataframe: PolarsDataFrameLike): Array<{ name: string; series: PolarsSeriesLike }> {
  if (typeof dataframe.getColumns === "function") {
    return dataframe.getColumns().map((series) => ({
      name: series.name ?? "",
      series,
    }));
  }
  if (Array.isArray(dataframe.columns)) {
    return dataframe.columns.map((name) => ({
      name,
      series: resolveSeries(dataframe, [], name),
    }));
  }
  if (typeof dataframe.toStruct === "function") {
    const struct = dataframe.toStruct();
    return Object.entries(struct).map(([name, values]) => ({
      name,
      series: createSeriesShim(name, values),
    }));
  }
  throw new Error("Unable to resolve Polars columns");
}

function resolveSeries(
  dataframe: PolarsDataFrameLike,
  known: Array<{ name: string; series: PolarsSeriesLike }>,
  name: string
): PolarsSeriesLike {
  const match = known.find((entry) => entry.name === name);
  if (match) {
    return match.series;
  }
  if (typeof dataframe.getColumn === "function") {
    return dataframe.getColumn(name);
  }
  if (typeof dataframe.column === "function") {
    return dataframe.column(name);
  }
  throw new Error(`Polars dataframe does not expose column "${name}"`);
}

function resolvePolarsRowCount(dataframe: PolarsDataFrameLike): number {
  if (typeof dataframe.height === "number") {
    return dataframe.height;
  }
  if (dataframe.shape && typeof dataframe.shape[0] === "number") {
    return dataframe.shape[0];
  }
  if (Array.isArray(dataframe.columns) && dataframe.columns.length > 0) {
    const first = resolveSeries(dataframe, [], dataframe.columns[0]).toArray();
    return first.length;
  }
  throw new Error("Unable to infer Polars dataframe height");
}

function createSeriesShim(name: string, values: unknown[]): PolarsSeriesLike {
  return {
    name,
    toArray: () => values,
  };
}

export async function ensurePolarsModule(provided: unknown): Promise<PolarsModuleLike> {
  if (provided) {
    return provided as PolarsModuleLike;
  }
  const dynamicImport = new Function("specifier", "return import(specifier);");
  const candidates = ["nodejs-polars", "polars"];
  for (const specifier of candidates) {
    try {
      const mod = await dynamicImport(specifier);
      if (mod?.default) {
        return mod.default as PolarsModuleLike;
      }
      return mod as PolarsModuleLike;
    } catch {
      // try next candidate
    }
  }
  throw new Error(
    'Unable to load a Polars module. Install "nodejs-polars" or "polars" and optionally pass it through options.polarsModule.'
  );
}
