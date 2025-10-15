import { strFromU8, strToU8, unzipSync, zipSync } from "fflate";

type BackendKind = "napi" | "wasm";

export type DType =
  | "bool"
  | "int8"
  | "int16"
  | "int32"
  | "int64"
  | "uint8"
  | "uint16"
  | "uint32"
  | "uint64"
  | "float32"
  | "float64";

type BackendMatrixHandle = {
  readonly rows: number;
  readonly cols: number;
  // dtype may be missing on some backends; we cache it separately when needed
  readonly dtype?: DType;
  to_vec?(): Float64Array | number[];
  toVec?(): Float64Array | number[];
  astype(dtype: DType, copy?: boolean): BackendMatrixHandle;
  to_bytes?(): Uint8Array;
  toBytes?(): Uint8Array;
};

type BackendMatrixConstructor = {
  new (
    data: Float64Array,
    rows: number,
    cols: number
  ): BackendMatrixHandle;
  from_bytes?(
    data: Uint8Array,
    rows: number,
    cols: number,
    dtype: DType
  ): BackendMatrixHandle;
  fromBytes?(
    data: Uint8Array,
    rows: number,
    cols: number,
    dtype: DType
  ): BackendMatrixHandle;
};

type BackendModule = {
  Matrix: BackendMatrixConstructor;
  add(a: BackendMatrixHandle, b: BackendMatrixHandle): BackendMatrixHandle;
  matmul(a: BackendMatrixHandle, b: BackendMatrixHandle): BackendMatrixHandle;
  clip?(
    matrix: BackendMatrixHandle,
    min: number,
    max: number
  ): BackendMatrixHandle;
  where_select?(
    condition: BackendMatrixHandle,
    truthy: BackendMatrixHandle,
    falsy: BackendMatrixHandle
  ): BackendMatrixHandle;
  where_select_multi?(
    conditions: readonly BackendMatrixHandle[],
    choices: readonly BackendMatrixHandle[],
    defaultValue?: BackendMatrixHandle
  ): BackendMatrixHandle;
  concat?(
    a: BackendMatrixHandle,
    b: BackendMatrixHandle,
    axis: number
  ): BackendMatrixHandle;
  stack?(
    a: BackendMatrixHandle,
    b: BackendMatrixHandle,
    axis: number
  ): BackendMatrixHandle;
  svd?(
    matrix: BackendMatrixHandle
  ): {
    u: BackendMatrixHandle;
    s: Float64Array;
    vt: BackendMatrixHandle;
  };
  qr?(
    matrix: BackendMatrixHandle
  ): {
    q: BackendMatrixHandle;
    r: BackendMatrixHandle;
  };
  solve?(a: BackendMatrixHandle, b: BackendMatrixHandle): BackendMatrixHandle;
  eigen?(
    matrix: BackendMatrixHandle
  ): {
    values: Float64Array;
    vectors: BackendMatrixHandle;
  };
  take?(
    matrix: BackendMatrixHandle,
    axis: number,
    indices: readonly number[]
  ): BackendMatrixHandle;
  put?(
    matrix: BackendMatrixHandle,
    axis: number,
    indices: readonly number[],
    values: BackendMatrixHandle
  ): BackendMatrixHandle;
  gather?(
    matrix: BackendMatrixHandle,
    rowIndices: readonly number[],
    colIndices: readonly number[]
  ): BackendMatrixHandle;
  gather_pairs?(
    matrix: BackendMatrixHandle,
    rowIndices: readonly number[],
    colIndices: readonly number[]
  ): BackendMatrixHandle;
  scatter?(
    matrix: BackendMatrixHandle,
    rowIndices: readonly number[],
    colIndices: readonly number[],
    values: BackendMatrixHandle
  ): BackendMatrixHandle;
  scatter_pairs?(
    matrix: BackendMatrixHandle,
    rowIndices: readonly number[],
    colIndices: readonly number[],
    values: BackendMatrixHandle
  ): BackendMatrixHandle;
  read_npy?(buffer: Uint8Array): BackendMatrixHandle;
  write_npy?(matrix: BackendMatrixHandle): Uint8Array;
  copy_bytes_total?: () => number;
  take_copy_bytes?: () => number;
  reset_copy_bytes?: () => void;
};

const WASM_ENTRY = "./bindings/wasm/num_rs_wasm.js";
const NAPI_ENTRY = "./bindings/napi/index.node";
const NAPI_BINDING_PREFIX = "./bindings/napi/index";

type NapiDistribution = {
  packages: readonly string[];
  binaries: readonly string[];
};

const NAPI_DISTRIBUTIONS: Record<string, NapiDistribution> = {
  "win32-x64": {
    packages: ["@jayce789/numjs-win32-x64-msvc"],
    binaries: ["win32-x64-msvc"],
  },
  "win32-arm64": {
    packages: ["@jayce789/numjs-win32-arm64-msvc"],
    binaries: ["win32-arm64-msvc"],
  },
  "darwin-x64": {
    packages: ["@jayce789/numjs-darwin-x64"],
    binaries: ["darwin-x64"],
  },
  "darwin-arm64": {
    packages: [
      "@jayce789/numjs-darwin-arm64",
      "@jayce789/numjs-darwin-x64",
    ],
    binaries: ["darwin-arm64", "darwin-x64"],
  },
  "linux-x64": {
    packages: ["@jayce789/numjs-linux-x64-gnu"],
    binaries: ["linux-x64-gnu"],
  },
  "linux-arm64": {
    packages: ["@jayce789/numjs-linux-arm64-gnu"],
    binaries: ["linux-arm64-gnu"],
  },
};

let activeBackend: BackendModule | null = null;
let activeKind: BackendKind | null = null;
let pendingLoad: Promise<void> | null = null;

export type NamedMatrix = { name: string; matrix: Matrix };

const isNode =
  typeof process !== "undefined" && typeof process.versions?.node === "string";

export type DTypeKind = "bool" | "unsigned" | "signed" | "float";

export type DTypeInfo = {
  size: number;
  kind: DTypeKind;
  isFloat: boolean;
  isSigned: boolean;
};

export const DTYPE_INFO: Record<DType, DTypeInfo> = {
  bool: { size: 1, kind: "bool", isFloat: false, isSigned: false },
  int8: { size: 1, kind: "signed", isFloat: false, isSigned: true },
  int16: { size: 2, kind: "signed", isFloat: false, isSigned: true },
  int32: { size: 4, kind: "signed", isFloat: false, isSigned: true },
  int64: { size: 8, kind: "signed", isFloat: false, isSigned: true },
  uint8: { size: 1, kind: "unsigned", isFloat: false, isSigned: false },
  uint16: { size: 2, kind: "unsigned", isFloat: false, isSigned: false },
  uint32: { size: 4, kind: "unsigned", isFloat: false, isSigned: false },
  uint64: { size: 8, kind: "unsigned", isFloat: false, isSigned: false },
  float32: { size: 4, kind: "float", isFloat: true, isSigned: true },
  float64: { size: 8, kind: "float", isFloat: true, isSigned: true },
};

export type BackendCapabilities = {
  kind: BackendKind;
  supportsMatrixFromBytes: boolean;
  supportsReadNpy: boolean;
  supportsWriteNpy: boolean;
  supportsCopyMetrics: boolean;
  supportedDTypes: readonly DType[];
};

const PROMOTION_TABLE: Record<DType, Record<DType, DType>> = {
  bool: {
    bool: "bool",
    int8: "int8",
    int16: "int16",
    int32: "int32",
    int64: "int64",
    uint8: "uint8",
    uint16: "uint16",
    uint32: "uint32",
    uint64: "uint64",
    float32: "float32",
    float64: "float64",
  },
  int8: {
    bool: "int8",
    int8: "int8",
    int16: "int16",
    int32: "int32",
    int64: "int64",
    uint8: "int16",
    uint16: "int32",
    uint32: "float64",
    uint64: "float64",
    float32: "float32",
    float64: "float64",
  },
  int16: {
    bool: "int16",
    int8: "int16",
    int16: "int16",
    int32: "int32",
    int64: "int64",
    uint8: "int32",
    uint16: "int32",
    uint32: "float64",
    uint64: "float64",
    float32: "float32",
    float64: "float64",
  },
  int32: {
    bool: "int32",
    int8: "int32",
    int16: "int32",
    int32: "int32",
    int64: "int64",
    uint8: "int32",
    uint16: "int32",
    uint32: "int64",
    uint64: "float64",
    float32: "float32",
    float64: "float64",
  },
  int64: {
    bool: "int64",
    int8: "int64",
    int16: "int64",
    int32: "int64",
    int64: "int64",
    uint8: "int64",
    uint16: "int64",
    uint32: "int64",
    uint64: "float64",
    float32: "float64",
    float64: "float64",
  },
  uint8: {
    bool: "uint8",
    int8: "int16",
    int16: "int32",
    int32: "int32",
    int64: "int64",
    uint8: "uint8",
    uint16: "uint16",
    uint32: "uint32",
    uint64: "uint64",
    float32: "float32",
    float64: "float64",
  },
  uint16: {
    bool: "uint16",
    int8: "int32",
    int16: "int32",
    int32: "int32",
    int64: "int64",
    uint8: "uint16",
    uint16: "uint16",
    uint32: "uint32",
    uint64: "uint64",
    float32: "float32",
    float64: "float64",
  },
  uint32: {
    bool: "uint32",
    int8: "float64",
    int16: "float64",
    int32: "int64",
    int64: "int64",
    uint8: "uint32",
    uint16: "uint32",
    uint32: "uint32",
    uint64: "uint64",
    float32: "float32",
    float64: "float64",
  },
  uint64: {
    bool: "uint64",
    int8: "float64",
    int16: "float64",
    int32: "float64",
    int64: "float64",
    uint8: "uint64",
    uint16: "uint64",
    uint32: "uint64",
    uint64: "uint64",
    float32: "float64",
    float64: "float64",
  },
  float32: {
    bool: "float32",
    int8: "float32",
    int16: "float32",
    int32: "float32",
    int64: "float64",
    uint8: "float32",
    uint16: "float32",
    uint32: "float32",
    uint64: "float64",
    float32: "float32",
    float64: "float64",
  },
  float64: {
    bool: "float64",
    int8: "float64",
    int16: "float64",
    int32: "float64",
    int64: "float64",
    uint8: "float64",
    uint16: "float64",
    uint32: "float64",
    uint64: "float64",
    float32: "float64",
    float64: "float64",
  },
};

function promoteBinaryDType(left: DType, right: DType): DType {
  const table = PROMOTION_TABLE[left];
  if (!table) {
    throw new Error(`promoteBinaryDType: unsupported dtype "${left}"`);
  }
  const promoted = table[right];
  if (!promoted) {
    throw new Error(
      `promoteBinaryDType: unable to promote "${left}" with "${right}"`
    );
  }
  return promoted;
}

function promoteManyDType(dtypes: readonly DType[]): DType {
  if (dtypes.length === 0) {
    throw new Error("promoteManyDType requires at least one dtype");
  }
  let result = dtypes[0];
  for (let index = 1; index < dtypes.length; index += 1) {
    result = promoteBinaryDType(result, dtypes[index]);
  }
  return result;
}
async function loadBackend(): Promise<void> {
  if (activeBackend) {
    return;
  }

  if (isNode) {
    const napi = await loadNapiBackend();
    if (napi) {
      activeBackend = napi;
      activeKind = "napi";
      return;
    }
  }

  const wasm = await loadWasmBackend();
  activeBackend = wasm;
  activeKind = "wasm";
}

async function loadNapiBackend(): Promise<BackendModule | null> {
  const candidates = resolveNapiCandidates();
  for (const specifier of candidates) {
    try {
      const required = (await getRequire())(specifier) as BackendModule;
      return required;
    } catch (error) {
      if (!isModuleNotFound(error)) {
        console.warn(
          `[numjs] Failed to load native backend from "${specifier}". Falling back if possible.`,
          error
        );
      }
    }
  }
  return null;
}

async function loadWasmBackend(): Promise<BackendModule> {
  const module = (await import(WASM_ENTRY)) as BackendModule & {
    default?: () => Promise<unknown>;
  };

  if (typeof module.default === "function") {
    await module.default();
  }

  return module;
}

function ensureBackend(): BackendModule {
  if (!activeBackend) {
    throw new Error("Backend not initialised. Call init() first.");
  }
  return activeBackend;
}

function wrapMatrix(handle: BackendMatrixHandle): Matrix {
  const matrix = Object.create(Matrix.prototype) as Matrix;
  matrix["_handle"] = handle;
  return matrix;
}

function getHandle(matrix: Matrix): BackendMatrixHandle {
  return matrix["_handle"];
}

function getMatrixDTypeFromHandle(handle: BackendMatrixHandle): DType {
  const dt = (handle as any).dtype;
  if (typeof dt === "string") return dt as DType;
  // Fallback: many minimal backends expose only float64 vectors
  return "float64";
}

function getMatrixDType(matrix: Matrix): DType {
  const cache = (matrix as any)._dtypeCache as DType | undefined;
  if (cache) return cache;
  const dtype = getMatrixDTypeFromHandle(getHandle(matrix));
  (matrix as any)._dtypeCache = dtype;
  return dtype;
}

function toBackendBytes(data: ArrayBuffer | Uint8Array): Uint8Array {
  return data instanceof Uint8Array ? data : new Uint8Array(data);
}

function callHandleToVec(handle: BackendMatrixHandle): Float64Array | number[] {
  if (typeof handle.to_vec === "function") return handle.to_vec();
  if (typeof handle.toVec === "function") return handle.toVec();
  throw new Error("Backend handle lacks to_vec/toVec method");
}

function callHandleToBytes(handle: BackendMatrixHandle): Uint8Array {
  if (typeof handle.to_bytes === "function") return handle.to_bytes();
  if (typeof handle.toBytes === "function") return handle.toBytes();
  throw new Error("Backend handle lacks to_bytes/toBytes method");
}

function handleToFloat64(handle: BackendMatrixHandle): Float64Array {
  const raw = callHandleToVec(handle);
  return raw instanceof Float64Array ? raw : Float64Array.from(raw);
}

type TypedArray =
  | Float64Array
  | Float32Array
  | Int8Array
  | Int16Array
  | Int32Array
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | BigInt64Array
  | BigUint64Array
  | Uint8ClampedArray;

type MatrixInputData = Float64Array | number[] | boolean[] | TypedArray;

export type MatrixOptions = {
  dtype?: DType;
};

const TYPED_ARRAY_TO_DTYPE: Array<[new (...args: any[]) => TypedArray, DType]> = [
  [Float64Array, "float64"],
  [Float32Array, "float32"],
  [Int8Array, "int8"],
  [Int16Array, "int16"],
  [Int32Array, "int32"],
  [Uint8Array, "uint8"],
  [Uint8ClampedArray, "uint8"],
  [Uint16Array, "uint16"],
  [Uint32Array, "uint32"],
  [BigInt64Array, "int64"],
  [BigUint64Array, "uint64"],
];

function inferDTypeFromData(data: MatrixInputData): DType | null {
  for (const [ctor, dtype] of TYPED_ARRAY_TO_DTYPE) {
    if (data instanceof ctor) {
      return dtype;
    }
  }
  return null;
}

function toFloat64Array(data: MatrixInputData): Float64Array {
  if (data instanceof Float64Array) {
    return data;
  }
  if (data instanceof BigInt64Array || data instanceof BigUint64Array) {
    return Float64Array.from(data, (value) => Number(value));
  }
  if (ArrayBuffer.isView(data)) {
    return Float64Array.from(data as ArrayLike<number>);
  }
  if (Array.isArray(data)) {
    if (data.length === 0) {
      return new Float64Array();
    }
    if (typeof data[0] === "boolean") {
      return Float64Array.from(
        data as boolean[],
        (value) => (value ? 1 : 0)
      );
    }
  }
  return Float64Array.from(data as number[]);
}

function copyTypedArrayToUint8(data: TypedArray): Uint8Array {
  return new Uint8Array(
    data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)
  );
}

function isTypedArrayData(data: MatrixInputData): data is TypedArray {
  return ArrayBuffer.isView(data) && !(data instanceof DataView);
}

function elementCount(data: MatrixInputData): number {
  return Array.isArray(data) ? data.length : data.length;
}

function typedArrayFromBytes(bytes: Uint8Array, dtype: DType): TypedArray {
  const view = bytes.slice();
  const buffer = view.buffer;
  switch (dtype) {
    case "float64":
      return new Float64Array(buffer);
    case "float32":
      return new Float32Array(buffer);
    case "int8":
      return new Int8Array(buffer);
    case "int16":
      return new Int16Array(buffer);
    case "int32":
      return new Int32Array(buffer);
    case "int64":
      return new BigInt64Array(buffer);
    case "uint8":
      return view;
    case "uint16":
      return new Uint16Array(buffer);
    case "uint32":
      return new Uint32Array(buffer);
    case "uint64":
      return new BigUint64Array(buffer);
    case "bool":
      return Uint8Array.from(view, (value) => (value !== 0 ? 1 : 0));
    default:
      return view;
  }
}

function handleToTypedArray(handle: BackendMatrixHandle): TypedArray {
  const dtype = getMatrixDTypeFromHandle(handle);
  if (dtype === "float64") {
    return handleToFloat64(handle);
  }
  const bytes = callHandleToBytes(handle);
  return typedArrayFromBytes(bytes, dtype);
}

export async function init(): Promise<void> {
  if (activeBackend) {
    return;
  }
  if (!pendingLoad) {
    pendingLoad = loadBackend().finally(() => {
      pendingLoad = null;
    });
  }
  await pendingLoad;
}

export class Matrix {
  private _handle: BackendMatrixHandle;
  private _dtypeCache?: DType;

  constructor(
    data: MatrixInputData,
    rows: number,
    cols: number,
    options: MatrixOptions = {}
  ) {
    const expectedLength = rows * cols;
    const length = elementCount(data);
    if (expectedLength !== length) {
      throw new Error(
        `Matrix data length (${length}) does not match shape (${rows} x ${cols})`
      );
    }

    const backend = ensureBackend();
    const inferred = inferDTypeFromData(data);
    const targetDType = options.dtype ?? inferred ?? "float64";
    const MatrixCtor = backend.Matrix as BackendMatrixConstructor;

    const fromBytes = (MatrixCtor as BackendMatrixConstructor).from_bytes ??
      (MatrixCtor as BackendMatrixConstructor).fromBytes;

    if (isTypedArrayData(data) && inferred && typeof fromBytes === "function") {
      const baseHandle = fromBytes.call(
        MatrixCtor,
        copyTypedArrayToUint8(data),
        rows,
        cols,
        inferred
      );
      this._handle =
        targetDType === inferred ? baseHandle : baseHandle.astype(targetDType);
      this._dtypeCache = targetDType;
      return;
    }

    const base = new MatrixCtor(toFloat64Array(data), rows, cols);
    this._handle =
      targetDType === "float64" ? base : base.astype(targetDType);
    this._dtypeCache = targetDType;
  }

  static fromHandle(handle: BackendMatrixHandle): Matrix {
    return wrapMatrix(handle);
  }

  static fromHandleWithDType(handle: BackendMatrixHandle, dtype: DType): Matrix {
    const m = wrapMatrix(handle);
    (m as any)._dtypeCache = dtype;
    return m;
  }

  static fromBytes(
    data: ArrayBuffer | Uint8Array,
    rows: number,
    cols: number,
    dtype: DType
  ): Matrix {
    return matrixFromBytes(data, rows, cols, dtype);
  }

  get rows(): number {
    return this._handle.rows;
  }

  get cols(): number {
    return this._handle.cols;
  }

  get dtype(): DType {
    return getMatrixDType(this);
  }

  toArray(): TypedArray {
    return handleToTypedArray(this._handle);
  }

  get dtypeInfo(): DTypeInfo {
    return DTYPE_INFO[this.dtype];
  }

  toBytes(): Uint8Array {
    return callHandleToBytes(this._handle);
  }

  clip(min: number, max: number): Matrix {
    return clip(this, min, max);
  }

  concat(other: Matrix, axis = 0): Matrix {
    return concat(this, other, axis);
  }

  stack(other: Matrix, axis = 0): Matrix {
    return stack(this, other, axis);
  }

  take(axis: number, indices: IndexCollection): Matrix {
    return take(this, axis, indices);
  }

  put(axis: number, indices: IndexCollection, values: Matrix): Matrix {
    return put(this, axis, indices, values);
  }

  gather(rowIndices: IndexCollection, colIndices: IndexCollection): Matrix {
    return gather(this, rowIndices, colIndices);
  }

  gatherPairs(rowIndices: IndexCollection, colIndices: IndexCollection): Matrix {
    return gatherPairs(this, rowIndices, colIndices);
  }

  scatter(
    rowIndices: IndexCollection,
    colIndices: IndexCollection,
    values: Matrix
  ): Matrix {
    return scatter(this, rowIndices, colIndices, values);
  }

  scatterPairs(
    rowIndices: IndexCollection,
    colIndices: IndexCollection,
    values: Matrix
  ): Matrix {
    return scatterPairs(this, rowIndices, colIndices, values);
  }

  astype(dtype: DType, options: { copy?: boolean } = {}): Matrix {
    const copy = options.copy ?? false;
    if (!copy && dtype === this.dtype) {
      return this;
    }
    const handle = getHandle(this).astype(dtype, copy);
    return Matrix.fromHandle(handle);
  }

  round(decimals = 12): Matrix {
    return round(this, decimals);
  }

  toOutputArray(options?: Partial<OutputFormat>): (number | string | bigint)[] | BigInt64Array | BigUint64Array {
    return toOutputArray(this, options);
  }

  toJSON(): unknown {
    const format = getOutputFormat();
    return {
      rows: this.rows,
      cols: this.cols,
      dtype: this.dtype,
      // JSON-safe: always stringify values to avoid BigInt serialization errors
      data: toOutput2D(this, { ...format, as: "string" }),
    };
  }

  toString(): string {
    const format = getOutputFormat();
    const rows = toOutput2D(this, { ...format, as: "string" });
    const delimiter = format.delimiter ?? "\t";
    const lineEnding = format.lineEnding ?? "\n";
    const padTo = format.padTo ?? 0;
    const align = format.align ?? "right";
    let lines: string[] = [];
    if (padTo > 0) {
      lines = rows.map((row) =>
        row
          .map((cell) => {
            const s = String(cell);
            return align === "left" ? s.padEnd(padTo) : s.padStart(padTo);
          })
          .join(delimiter)
      );
    } else {
      lines = rows.map((row) => row.join(delimiter));
    }
    return `Matrix(${this.rows}x${this.cols}, dtype=${this.dtype})` + lineEnding + lines.join(lineEnding);
  }
}

function castToDType(matrix: Matrix, dtype: DType): Matrix {
  return matrix.astype(dtype, { copy: false });
}

function castAllToDType(matrices: readonly Matrix[], dtype: DType): Matrix[] {
  return matrices.map((matrix) => castToDType(matrix, dtype));
}

export type WhereOptions = {
  default?: Matrix;
};

type IndexCollection = readonly number[] | ArrayLike<number>;

function normalizeIndices(indices: IndexCollection): number[] {
  return Array.from(indices as ArrayLike<number>, ensureIntegerIndex);
}

function ensureIntegerIndex(value: number): number {
  if (!Number.isFinite(value) || !Number.isInteger(value)) {
    throw new Error(`Index "${value}" must be a finite integer`);
  }
  return value;
}

type MatrixShape = { rows: number; cols: number };

function broadcastDim(current: number, next: number): number {
  if (current === 0) {
    return next;
  }
  if (next === 0) {
    return current;
  }
  if (current === next) {
    return current;
  }
  if (current === 1) {
    return next;
  }
  if (next === 1) {
    return current;
  }
  throw new Error(
    `Shapes are not broadcast compatible (${current} vs ${next})`
  );
}

function computeBroadcastShape(
  conditions: readonly Matrix[],
  choices: readonly Matrix[],
  fallback: Matrix | null
): MatrixShape {
  let rows = 0;
  let cols = 0;

  const updateShape = (matrix: Matrix) => {
    rows = broadcastDim(rows, matrix.rows);
    cols = broadcastDim(cols, matrix.cols);
  };

  for (const matrix of conditions) {
    updateShape(matrix);
  }
  for (const matrix of choices) {
    updateShape(matrix);
  }
  if (fallback) {
    updateShape(fallback);
  }
  if (rows === 0 || cols === 0) {
    throw new Error("Unable to infer broadcast shape for where operation");
  }
  return { rows, cols };
}

function allocateArrayForDType(dtype: DType, length: number): MatrixInputData {
  switch (dtype) {
    case "bool":
      return Array.from({ length }, () => false);
    case "int8":
      return new Int8Array(length);
    case "int16":
      return new Int16Array(length);
    case "int32":
      return new Int32Array(length);
    case "int64":
      return new BigInt64Array(length);
    case "uint8":
      return new Uint8Array(length);
    case "uint16":
      return new Uint16Array(length);
    case "uint32":
      return new Uint32Array(length);
    case "uint64":
      return new BigUint64Array(length);
    case "float32":
      return new Float32Array(length);
    case "float64":
      return new Float64Array(length);
    default:
      throw new Error(`Unsupported dtype "${dtype}"`);
  }
}

function fillZeros(data: MatrixInputData, dtype: DType): void {
  if (dtype === "bool") {
    (data as boolean[]).fill(false);
  }
}

function cloneMatrixData(matrix: Matrix, dtype: DType): MatrixInputData {
  if (dtype === "bool") {
    const source = matrix
    .astype("bool", { copy: true })
    .toArray() as ArrayLike<number | boolean>;
    return Array.from(source, (value) => Boolean(value));
  }
  const array = matrix.toArray();
  if (Array.isArray(array)) {
    return array.slice();
  }
  return array.slice();
}

function writeMatrixValue(
  target: MatrixInputData,
  index: number,
  value: number | bigint | boolean,
  dtype: DType
): void {
  switch (dtype) {
    case "bool":
      (target as boolean[])[index] = Boolean(value);
      return;
    case "int64":
      (target as BigInt64Array)[index] =
        typeof value === "bigint" ? value : BigInt(value as number);
      return;
    case "uint64":
      (target as BigUint64Array)[index] =
        typeof value === "bigint" ? value : BigInt(value as number);
      return;
    default:
      (target as Exclude<MatrixInputData, boolean[] | BigInt64Array | BigUint64Array>)[
        index
      ] = Number(value);
  }
}

function maskValueTrue(value: number | bigint | boolean): boolean {
  if (typeof value === "number") {
    return value !== 0;
  }
  if (typeof value === "bigint") {
    return value !== 0n;
  }
  return value === true;
}

function applyMaskToData(
  dest: MatrixInputData,
  mask: ArrayLike<number | bigint | boolean>,
  src: ArrayLike<number | bigint | boolean>,
  dtype: DType
): void {
  for (let index = 0; index < mask.length; index += 1) {
    if (maskValueTrue(mask[index])) {
      writeMatrixValue(dest, index, src[index], dtype);
    }
  }
}

function broadcastMatrixTo(matrix: Matrix, rows: number, cols: number): Matrix {
  if (matrix.rows === rows && matrix.cols === cols) {
    return matrix;
  }
  if (
    (matrix.rows !== 1 && matrix.rows !== rows) ||
    (matrix.cols !== 1 && matrix.cols !== cols)
  ) {
    throw new Error(
      `Cannot broadcast matrix of shape ${matrix.rows}x${matrix.cols} to ${rows}x${cols}`
    );
  }
  const dtype = matrix.dtype;
  const source = matrix.toArray();
  const target = allocateArrayForDType(dtype, rows * cols);
  const srcRows = matrix.rows;
  const srcCols = matrix.cols;
  for (let row = 0; row < rows; row += 1) {
    const srcRow = srcRows === 1 ? 0 : row;
    for (let col = 0; col < cols; col += 1) {
      const srcCol = srcCols === 1 ? 0 : col;
      const srcIndex = srcRow * srcCols + srcCol;
      const dstIndex = row * cols + col;
      writeMatrixValue(target, dstIndex, source[srcIndex], dtype);
    }
  }
  return new Matrix(target, rows, cols, { dtype });
}


export function matrixFromBytes(
  data: ArrayBuffer | Uint8Array,
  rows: number,
  cols: number,
  dtype: DType
): Matrix {
  const backend = ensureBackend();
  const MatrixCtor = backend.Matrix as BackendMatrixConstructor;
  const fromBytes = (MatrixCtor as BackendMatrixConstructor).from_bytes ??
    (MatrixCtor as BackendMatrixConstructor).fromBytes;
  if (typeof fromBytes !== "function") {
    throw new Error("Matrix.from_bytes is not supported by current backend");
  }
  const handle = fromBytes.call(MatrixCtor, toBackendBytes(data), rows, cols, dtype);
  return Matrix.fromHandleWithDType(handle, dtype);
}

export function add(a: Matrix, b: Matrix): Matrix {
  const dtype = promoteBinaryDType(a.dtype, b.dtype);
  const left = castToDType(a, dtype);
  const right = castToDType(b, dtype);
  const backend = ensureBackend();
  const result = backend.add(getHandle(left), getHandle(right));
  return Matrix.fromHandleWithDType(result, dtype);
}

export function matmul(a: Matrix, b: Matrix): Matrix {
  const dtype = promoteBinaryDType(a.dtype, b.dtype);
  const left = castToDType(a, dtype);
  const right = castToDType(b, dtype);
  const backend = ensureBackend();
  const result = backend.matmul(getHandle(left), getHandle(right));
  return Matrix.fromHandleWithDType(result, dtype);
}

export function clip(matrix: Matrix, min: number, max: number): Matrix {
  const backend = ensureBackend();
  if (!backend.clip) {
    throw new Error("clip is not supported by current backend");
  }
  const result = backend.clip(getHandle(matrix), min, max);
  return Matrix.fromHandleWithDType(result, matrix.dtype);
}

export function where(
  condition: Matrix | Matrix[],
  truthy: Matrix | Matrix[],
  falsy?: Matrix,
  options: WhereOptions = {}
): Matrix {
  const backend = ensureBackend();
  const defaultMatrix = falsy ?? options.default ?? null;

  if (!Array.isArray(condition) && !Array.isArray(truthy)) {
    if (!backend.where_select) {
      throw new Error("where_select is not supported by current backend");
    }
    if (!defaultMatrix) {
      throw new Error("where requires a falsy matrix or a default option");
    }
    const dtype = promoteBinaryDType(truthy.dtype, defaultMatrix.dtype);
    const cond = condition.astype("bool", { copy: false });
    const lhs = castToDType(truthy, dtype);
    const rhs = castToDType(defaultMatrix, dtype);
    const result = backend.where_select(
      getHandle(cond),
      getHandle(lhs),
      getHandle(rhs)
    );
    return Matrix.fromHandleWithDType(result, dtype);
  }

  const conditionList = Array.isArray(condition) ? condition : [condition];
  const choiceList = Array.isArray(truthy) ? truthy : [truthy];

  if (conditionList.length !== choiceList.length) {
    throw new Error(
      `where: expected ${choiceList.length} condition(s), received ${conditionList.length}`
    );
  }
  if (choiceList.length === 0) {
    throw new Error("where: at least one condition/choice pair is required");
  }

  const dtypeCandidates = choiceList.map((item) => item.dtype);
  if (defaultMatrix) {
    dtypeCandidates.push(defaultMatrix.dtype);
  }
  const dtype = promoteManyDType(dtypeCandidates);
  const targetShape = computeBroadcastShape(
    conditionList,
    choiceList,
    defaultMatrix
  );

  const boolConditions = conditionList.map((cond) =>
    broadcastMatrixTo(cond.astype("bool", { copy: false }), targetShape.rows, targetShape.cols)
  );
  const broadcastChoices = choiceList.map((choice) =>
    broadcastMatrixTo(choice, targetShape.rows, targetShape.cols)
  );
  const castChoices = castAllToDType(broadcastChoices, dtype);
  const defaultCast = defaultMatrix
    ? castToDType(
        broadcastMatrixTo(defaultMatrix, targetShape.rows, targetShape.cols),
        dtype
      )
    : undefined;

  if (backend.where_select_multi) {
    const result = backend.where_select_multi(
      boolConditions.map(getHandle),
      castChoices.map(getHandle),
      defaultCast ? getHandle(defaultCast) : undefined
    );
    return Matrix.fromHandleWithDType(result, dtype);
  }

  const resultData = defaultCast
    ? cloneMatrixData(defaultCast, dtype)
    : allocateArrayForDType(dtype, targetShape.rows * targetShape.cols);
  if (!defaultCast) {
    fillZeros(resultData, dtype);
  }
  for (let index = 0; index < boolConditions.length; index += 1) {
    const maskArray = boolConditions[index].toArray();
    const choiceArray = castChoices[index].toArray();
    applyMaskToData(resultData, maskArray, choiceArray, dtype);
  }
  return new Matrix(resultData, targetShape.rows, targetShape.cols, { dtype });
}

export function take(
  matrix: Matrix,
  axis: number,
  indices: IndexCollection
): Matrix {
  const backend = ensureBackend();
  if (!backend.take) {
    throw new Error("take is not supported by current backend");
  }
  const normalized = normalizeIndices(indices);
  const result = backend.take(getHandle(matrix), axis, normalized);
  return Matrix.fromHandleWithDType(result, matrix.dtype);
}

export function put(
  matrix: Matrix,
  axis: number,
  indices: IndexCollection,
  values: Matrix
): Matrix {
  const backend = ensureBackend();
  if (!backend.put) {
    throw new Error("put is not supported by current backend");
  }
  const normalized = normalizeIndices(indices);
  const castValues = castToDType(values, matrix.dtype);
  const result = backend.put(
    getHandle(matrix),
    axis,
    normalized,
    getHandle(castValues)
  );
  return Matrix.fromHandleWithDType(result, matrix.dtype);
}

export function gather(
  matrix: Matrix,
  rowIndices: IndexCollection,
  colIndices: IndexCollection
): Matrix {
  const backend = ensureBackend();
  if (!backend.gather) {
    throw new Error("gather is not supported by current backend");
  }
  const rows = normalizeIndices(rowIndices);
  const cols = normalizeIndices(colIndices);
  const result = backend.gather(getHandle(matrix), rows, cols);
  return Matrix.fromHandleWithDType(result, matrix.dtype);
}

export function gatherPairs(
  matrix: Matrix,
  rowIndices: IndexCollection,
  colIndices: IndexCollection
): Matrix {
  const backend = ensureBackend();
  if (!backend.gather_pairs) {
    throw new Error("gather_pairs is not supported by current backend");
  }
  const rows = normalizeIndices(rowIndices);
  const cols = normalizeIndices(colIndices);
  const result = backend.gather_pairs(getHandle(matrix), rows, cols);
  return Matrix.fromHandleWithDType(result, matrix.dtype);
}

export function scatter(
  matrix: Matrix,
  rowIndices: IndexCollection,
  colIndices: IndexCollection,
  values: Matrix
): Matrix {
  const backend = ensureBackend();
  if (!backend.scatter) {
    throw new Error("scatter is not supported by current backend");
  }
  const rows = normalizeIndices(rowIndices);
  const cols = normalizeIndices(colIndices);
  const castValues = castToDType(values, matrix.dtype);
  const result = backend.scatter(
    getHandle(matrix),
    rows,
    cols,
    getHandle(castValues)
  );
  return Matrix.fromHandleWithDType(result, matrix.dtype);
}

export function scatterPairs(
  matrix: Matrix,
  rowIndices: IndexCollection,
  colIndices: IndexCollection,
  values: Matrix
): Matrix {
  const backend = ensureBackend();
  if (!backend.scatter_pairs) {
    throw new Error("scatter_pairs is not supported by current backend");
  }
  const rows = normalizeIndices(rowIndices);
  const cols = normalizeIndices(colIndices);
  const castValues = castToDType(values, matrix.dtype);
  const result = backend.scatter_pairs(
    getHandle(matrix),
    rows,
    cols,
    getHandle(castValues)
  );
  return Matrix.fromHandleWithDType(result, matrix.dtype);
}

export function concat(a: Matrix, b: Matrix, axis = 0): Matrix {
  const dtype = promoteManyDType([a.dtype, b.dtype]);
  const [left, right] = castAllToDType([a, b], dtype);
  const backend = ensureBackend();
  if (!backend.concat) {
    throw new Error("concat is not supported by current backend");
  }
  const result = backend.concat(getHandle(left), getHandle(right), axis);
  return Matrix.fromHandleWithDType(result, dtype);
}

export function stack(a: Matrix, b: Matrix, axis = 0): Matrix {
  const dtype = promoteManyDType([a.dtype, b.dtype]);
  const [left, right] = castAllToDType([a, b], dtype);
  const backend = ensureBackend();
  if (!backend.stack) {
    throw new Error("stack is not supported by current backend");
  }
  const result = backend.stack(getHandle(left), getHandle(right), axis);
  return Matrix.fromHandleWithDType(result, dtype);
}

export function svd(matrix: Matrix): {
  u: Matrix;
  s: Float64Array;
  vt: Matrix;
} {
  const backend = ensureBackend();
  if (!backend.svd) {
    throw new Error("svd is not supported by current backend");
  }
  const result = backend.svd(getHandle(matrix));
  return {
    u: Matrix.fromHandleWithDType(result.u, matrix.dtype),
    s: result.s,
    vt: Matrix.fromHandleWithDType(result.vt, matrix.dtype),
  };
}

export function qr(matrix: Matrix): { q: Matrix; r: Matrix } {
  const backend = ensureBackend();
  if (!backend.qr) {
    throw new Error("qr is not supported by current backend");
  }
  const result = backend.qr(getHandle(matrix));
  return {
    q: Matrix.fromHandleWithDType(result.q, matrix.dtype),
    r: Matrix.fromHandleWithDType(result.r, matrix.dtype),
  };
}

export function solve(a: Matrix, b: Matrix): Matrix {
  const backend = ensureBackend();
  if (!backend.solve) {
    throw new Error("solve is not supported by current backend");
  }
  const result = backend.solve(getHandle(a), getHandle(b));
  return Matrix.fromHandleWithDType(result, a.dtype);
}

export function eigen(matrix: Matrix): {
  values: Float64Array;
  vectors: Matrix;
} {
  const backend = ensureBackend();
  if (!backend.eigen) {
    throw new Error("eigen is not supported by current backend");
  }
  const result = backend.eigen(getHandle(matrix));
  return {
    values: result.values,
    vectors: Matrix.fromHandleWithDType(result.vectors, matrix.dtype),
  };
}

export function readNpy(data: ArrayBuffer | Uint8Array): Matrix {
  const backend = ensureBackend();
  if (!backend.read_npy) {
    throw new Error("read_npy is not supported by current backend");
  }
  const readNpyFn = (backend as any).read_npy ?? (backend as any).readNpy;
  const handle = readNpyFn(toBackendBytes(data));
  // dtype unknown from interface; rely on handle or fallback
  const m = Matrix.fromHandle(handle);
  return m;
}

export function writeNpy(matrix: Matrix): Uint8Array {
  const backend = ensureBackend();
  if (!backend.write_npy) {
    throw new Error("write_npy is not supported by current backend");
  }
  const writeNpyFn = (backend as any).write_npy ?? (backend as any).writeNpy;
  return writeNpyFn(getHandle(matrix));
}

export function copyBytesTotal(): number {
  const backend = ensureBackend();
  if (!backend.copy_bytes_total) {
    throw new Error("copy_bytes_total is not supported by current backend");
  }
  const fn = (backend as any).copy_bytes_total ?? (backend as any).copyBytesTotal;
  return fn();
}

export function takeCopyBytes(): number {
  const backend = ensureBackend();
  if (!backend.take_copy_bytes) {
    throw new Error("take_copy_bytes is not supported by current backend");
  }
  const fn = (backend as any).take_copy_bytes ?? (backend as any).takeCopyBytes;
  return fn();
}

export function resetCopyBytes(): void {
  const backend = ensureBackend();
  if (!backend.reset_copy_bytes) {
    throw new Error("reset_copy_bytes is not supported by current backend");
  }
  const fn = (backend as any).reset_copy_bytes ?? (backend as any).resetCopyBytes;
  fn();
}

export function backendKind(): BackendKind {
  if (!activeKind) {
    throw new Error("Backend not initialised. Call init() first.");
  }
  return activeKind;
}

export function backendCapabilities(): BackendCapabilities {
  const backend = ensureBackend();
  const MatrixCtor = backend.Matrix as BackendMatrixConstructor;
  return {
    kind: backendKind(),
    supportsMatrixFromBytes: typeof MatrixCtor.from_bytes === "function",
    supportsReadNpy: typeof backend.read_npy === "function",
    supportsWriteNpy: typeof backend.write_npy === "function",
    supportsCopyMetrics:
      typeof backend.copy_bytes_total === "function" &&
      typeof backend.take_copy_bytes === "function" &&
      typeof backend.reset_copy_bytes === "function",
    supportedDTypes: Object.keys(DTYPE_INFO) as DType[],
  };
}

async function getRequire(): Promise<(id: string) => unknown> {
  if (typeof require === "function") {
    return require;
  }
  const { createRequire } = await import("node:module");
  const moduleFile = getCurrentModuleFile();
  if (!moduleFile) {
    throw new Error("Unable to determine current module path");
  }
  if (moduleFile.startsWith("file://")) {
    return createRequire(moduleFile);
  }
  const { pathToFileURL } = await import("node:url");
  return createRequire(pathToFileURL(moduleFile).href);
}

function roundNumber(value: number, decimals: number): number {
  if (!Number.isFinite(value)) {
    return value;
  }
  const factor = Math.pow(10, decimals);
  return Math.round((value + Number.EPSILON) * factor) / factor;
}

export function round(matrix: Matrix, decimals = 12): Matrix {
  const dtype = matrix.dtype;
  if (dtype !== "float32" && dtype !== "float64") {
    return matrix;
  }
  const src = matrix.toArray();
  const length = matrix.rows * matrix.cols;
  if (dtype === "float64") {
    const out = new Float64Array(length);
    for (let i = 0; i < length; i += 1) {
      out[i] = roundNumber((src as Float64Array)[i], decimals);
    }
    return new Matrix(out, matrix.rows, matrix.cols, { dtype });
  } else {
    const out = new Float32Array(length);
    for (let i = 0; i < length; i += 1) {
      out[i] = roundNumber((src as Float32Array)[i], decimals);
    }
    return new Matrix(out, matrix.rows, matrix.cols, { dtype });
  }
}

// ---------------------------------------------------------------------
// Output formatting context (affects only printing/JSON/export)
// ---------------------------------------------------------------------

export type OutputAs = "string" | "number" | "bigint";

export type OutputFormat = {
  as: OutputAs;
  decimals?: number; // for string/number modes
  scale?: number; // for bigint (fixed-point) mode
  trimTrailingZeros?: boolean; // for string mode
  bigintBits?: 64 | 128; // bounds check for bigint export
  bigintSigned?: boolean; // signed range (default true)
  useScientific?: boolean; // enable scientific notation for large/small
  sciMinExp?: number; // |exp| >= sciMinExp -> scientific
  suppressSmall?: boolean; // |x| < 10^-decimals renders as 0
  delimiter?: string; // toString delimiter
  lineEnding?: string; // toString line ending
  padTo?: number; // toString min width
  align?: "left" | "right"; // toString alignment
};

const DEFAULT_OUTPUT_FORMAT: OutputFormat = {
  as: "number",
  decimals: 12,
  trimTrailingZeros: true,
  useScientific: false,
  sciMinExp: 6,
  suppressSmall: true,
  delimiter: "\t",
  lineEnding: "\n",
  align: "right",
};

let CURRENT_OUTPUT_FORMAT: OutputFormat = { ...DEFAULT_OUTPUT_FORMAT };
const OUTPUT_FORMAT_STACK: OutputFormat[] = [];

export function setOutputFormat(options: Partial<OutputFormat>): void {
  CURRENT_OUTPUT_FORMAT = { ...CURRENT_OUTPUT_FORMAT, ...options };
}

export function getOutputFormat(): OutputFormat {
  return { ...CURRENT_OUTPUT_FORMAT };
}

export async function withOutputFormat<T>(
  options: Partial<OutputFormat>,
  fn: () => T | Promise<T>
): Promise<T> {
  OUTPUT_FORMAT_STACK.push(CURRENT_OUTPUT_FORMAT);
  try {
    setOutputFormat(options);
    return await fn();
  } finally {
    const prev = OUTPUT_FORMAT_STACK.pop();
    if (prev) CURRENT_OUTPUT_FORMAT = prev;
  }
}

// Manual scope helper for environments that cannot await (e.g. host not awaiting async callbacks)
export function scopedOutputFormat(options: Partial<OutputFormat>): { restore(): void } {
  const prev = CURRENT_OUTPUT_FORMAT;
  setOutputFormat(options);
  let restored = false;
  return {
    restore() {
      if (!restored) {
        CURRENT_OUTPUT_FORMAT = prev;
        restored = true;
      }
    },
  };
}

function isNegZero(n: number): boolean {
  return n === 0 && 1 / n === -Infinity;
}

function formatNumberToString(
  value: number,
  decimals = 12,
  trim = true,
  useScientific = false,
  sciMinExp = 6,
  suppressSmall = true
): string {
  if (!Number.isFinite(value)) {
    // Safe textual representation for NaN/Infinity
    return String(value);
  }
  let v = value;
  if (suppressSmall && Math.abs(v) < Math.pow(10, -Math.max(0, decimals))) {
    v = 0;
  }
  if (useScientific && v !== 0) {
    const exp = Math.floor(Math.log10(Math.abs(v)));
    if (Math.abs(exp) >= (sciMinExp ?? 6)) {
      const sExp = v.toExponential(decimals);
      return sExp;
    }
  }
  const s = v.toFixed(decimals);
  if (!trim) return s;
  // trim trailing zeros and an optional trailing dot
  const t = s.replace(/\.0+$/, "").replace(/(\.[0-9]*?)0+$/, "$1");
  // normalize "-0" -> "0"
  return t === "-0" ? "0" : t;
}

function toScaledBigIntFromNumber(value: number, scale: number): bigint {
  if (!Number.isFinite(value)) {
    throw new Error(`Cannot convert non-finite number ${value} to bigint with scale ${scale}`);
  }
  // Use string rounding to avoid binary float artifacts
  const s = value.toFixed(Math.max(0, scale));
  const neg = s.startsWith("-");
  const abs = neg ? s.slice(1) : s;
  const [intPart, fracPartRaw = ""] = abs.split(".");
  const fracPart = fracPartRaw.padEnd(scale, "0").slice(0, scale);
  const digits = (intPart + fracPart).replace(/^0+/, "");
  const bi = BigInt(digits.length ? digits : "0");
  return neg ? -bi : bi;
}

function bigIntBounds(bits: 64 | 128, signed: boolean): { min: bigint; max: bigint } {
  if (bits === 64) {
    if (signed) {
      return { min: -(1n << 63n), max: (1n << 63n) - 1n };
    }
    return { min: 0n, max: (1n << 64n) - 1n };
  }
  // 128-bit
  const one = 1n;
  const shift = 128n;
  if (signed) {
    return { min: -(one << (shift - 1n)), max: (one << (shift - 1n)) - 1n };
  }
  return { min: 0n, max: (one << shift) - 1n };
}

 
export function toOutputArray(
  matrix: Matrix,
  options: Partial<OutputFormat> = {}
): (number | string | bigint)[] | BigInt64Array | BigUint64Array {
  const fmt = { ...CURRENT_OUTPUT_FORMAT, ...options } as OutputFormat;
  const values = matrix.toArray();
  const length = matrix.rows * matrix.cols;

  if (fmt.as === "string") {
    const decimals = fmt.decimals ?? DEFAULT_OUTPUT_FORMAT.decimals!;
    const trim = fmt.trimTrailingZeros ?? DEFAULT_OUTPUT_FORMAT.trimTrailingZeros!;
    const useSci = fmt.useScientific ?? DEFAULT_OUTPUT_FORMAT.useScientific!;
    const sciMin = fmt.sciMinExp ?? DEFAULT_OUTPUT_FORMAT.sciMinExp!;
    const suppressSmall = fmt.suppressSmall ?? DEFAULT_OUTPUT_FORMAT.suppressSmall!;
    const out = new Array<string>(length);
    for (let i = 0; i < length; i += 1) {
      const v = (values as any)[i];
      if (typeof v === "number") {
        out[i] = formatNumberToString(v, decimals, trim, useSci, sciMin, suppressSmall);
      } else if (typeof v === "bigint") {
        out[i] = v.toString();
      } else if (typeof v === "boolean") {
        out[i] = v ? "true" : "false";
      } else {
        out[i] = String(v);
      }
    }
    return out;
  }

  if (fmt.as === "bigint") {
    const scale = fmt.scale;
    if (!Number.isInteger(scale) || (scale as number) < 0) {
      throw new RangeError("toOutputArray(as=\"bigint\"): a non-negative integer 'scale' is required");
    }
    const bits = fmt.bigintBits ?? 64;
    const signed = fmt.bigintSigned ?? true;
    const factor = BigInt("1" + "0".repeat(scale as number));
    const { min, max } = bigIntBounds(bits as 64 | 128, signed);
    if (bits === 64) {
      const out64 = signed ? new BigInt64Array(length) : new BigUint64Array(length);
      for (let i = 0; i < length; i += 1) {
        const v = (values as any)[i];
        if (typeof v === "number") {
          if (!Number.isFinite(v)) {
            throw new RangeError("toOutputArray(as=\\\"bigint\\\", bigintBits=64): cannot convert NaN/Infinity; use as:'string' or as:'number'");
          }
          const bi = toScaledBigIntFromNumber(v, scale as number);
          if (bi < min || bi > max) {
            throw new RangeError(`toOutputArray(as=\\\"bigint\\\"): scaled value overflows ${bits}-bit ${signed ? "signed" : "unsigned"} range (value=${v}, scale=${scale})`);
          }
          out64[i] = bi as any;
        } else if (typeof v === "bigint") {
          const scaled = v * factor;
          if (scaled < min || scaled > max) {
            throw new RangeError(`toOutputArray(as=\\\"bigint\\\"): scaled bigint overflows ${bits}-bit ${signed ? "signed" : "unsigned"} range (value=${v}n, scale=${scale})`);
          }
          out64[i] = scaled as any;
        } else if (typeof v === "boolean") {
          const scaled = (v ? 1n : 0n) * factor;
          out64[i] = scaled as any;
        } else {
          out64[i] = 0n as any;
        }
      }
      return out64;
    }
    const out: (bigint | string)[] = new Array(length);
    for (let i = 0; i < length; i += 1) {
      const v = (values as any)[i];
      if (typeof v === "number") {
        if (!Number.isFinite(v)) {
          // Represent non-finite as strings in bigint mode to be JSON-safe
          out[i] = String(v);
          continue;
        }
        const bi = toScaledBigIntFromNumber(v, scale as number);
        if (bi < min || bi > max) {
          throw new Error(
            `toOutputArray(as="bigint"): scaled value overflows ${bits}-bit ${signed ? "signed" : "unsigned"} range (value=${v}, scale=${scale})`
          );
        }
        out[i] = bi;
      } else if (typeof v === "bigint") {
        const scaled = v * factor;
        if (scaled < min || scaled > max) {
          throw new Error(
            `toOutputArray(as="bigint"): scaled bigint overflows ${bits}-bit ${signed ? "signed" : "unsigned"} range (value=${v}n, scale=${scale})`
          );
        }
        out[i] = scaled;
      } else if (typeof v === "boolean") {
        const scaled = (v ? 1n : 0n) * factor;
        if (scaled < min || scaled > max) {
          throw new Error(
            `toOutputArray(as="bigint"): scaled boolean overflows ${bits}-bit ${signed ? "signed" : "unsigned"} range (scale=${scale})`
          );
        }
        out[i] = scaled;
      } else {
        out[i] = "0";
      }
    }
    return out;
  }

  // default: as number
  const decimals = fmt.decimals ?? DEFAULT_OUTPUT_FORMAT.decimals!;
  const out = new Array<number>(length);
  for (let i = 0; i < length; i += 1) {
    const v = (values as any)[i];
    if (typeof v === "number") {
      const n = parseFloat((v as number ).toFixed(decimals));
      out[i] = n === 0 ? 0 : n;
    } else if (typeof v === "bigint") {
      out[i] = Number(v);
    } else if (typeof v === "boolean") {
      out[i] = v ? 1 : 0;
    } else {
      out[i] = Number(v);
    }
  }
  return out;
}

export function toOutput2D(
  matrix: Matrix,
  options: Partial<OutputFormat> = {}
): (number | string | bigint)[][] {
  const flat = toOutputArray(matrix, options) as (number | string | bigint)[];
  const rows: (number | string | bigint)[][] = [];
  let index = 0;
  for (let r = 0; r < matrix.rows; r += 1) {
    const row: (number | string | bigint)[] = [];
    for (let c = 0; c < matrix.cols; c += 1) {
      row.push(flat[index++]);
    }
    rows.push(row);
  }
  return rows;
}

export function forEachRowToOutput(
  matrix: Matrix,
  options: Partial<OutputFormat>,
  fn: (row: (number | string | bigint)[], rowIndex: number) => void
): void {
  const cols = matrix.cols;
  const flat = toOutputArray(matrix, options) as (number | string | bigint)[];
  let index = 0;
  for (let r = 0; r < matrix.rows; r += 1) {
    const row: (number | string | bigint)[] = new Array(cols);
    for (let c = 0; c < cols; c += 1) row[c] = flat[index++];
    fn(row, r);
  }
}

export async function* iterOutputRows(
  matrix: Matrix,
  options: Partial<OutputFormat> = {}
): AsyncIterable<(number | string | bigint)[]> {
  const cols = matrix.cols;
  const flat = toOutputArray(matrix, options) as (number | string | bigint)[];
  let index = 0;
  for (let r = 0; r < matrix.rows; r += 1) {
    const row: (number | string | bigint)[] = new Array(cols);
    for (let c = 0; c < cols; c += 1) row[c] = flat[index++];
    yield row;
  }
}

// ---------------------------------------------------------------------
// Float helpers for comparisons (stability-friendly)
// ---------------------------------------------------------------------

export function isClose(
  a: number,
  b: number,
  { rtol = 1e-12, atol = 0, equalNaN = false }: { rtol?: number; atol?: number; equalNaN?: boolean } = {}
): boolean {
  if (Number.isNaN(a) || Number.isNaN(b)) {
    return equalNaN && Number.isNaN(a) && Number.isNaN(b);
  }
  if (!Number.isFinite(a) || !Number.isFinite(b)) {
    return a === b;
  }
  const diff = Math.abs(a - b);
  return diff <= atol + rtol * Math.max(Math.abs(a), Math.abs(b));
}

export function allClose(
  a: Matrix,
  b: Matrix,
  { rtol = 1e-12, atol = 0, equalNaN = false }: { rtol?: number; atol?: number; equalNaN?: boolean } = {}
): boolean {
  if (a.rows !== b.rows || a.cols !== b.cols) return false;
  const aArr = a.astype("float64", { copy: false }).toArray() as Float64Array;
  const bArr = b.astype("float64", { copy: false }).toArray() as Float64Array;
  const n = a.rows * a.cols;
  for (let i = 0; i < n; i += 1) {
    if (!isClose(aArr[i], bArr[i], { rtol, atol, equalNaN })) {
      return false;
    }
  }
  return true;
}

function getCurrentModuleFile(): string | undefined {
  // Derive the current module path without relying on import.meta,
  // which is absent in the CommonJS build emitted by tsup.
  const previousPrepareStackTrace = Error.prepareStackTrace;
  try {
    Error.prepareStackTrace = (_, stack) => stack;
    const stack = new Error().stack as unknown as NodeJS.CallSite[] | undefined;
    if (!stack) {
      return undefined;
    }
    for (const frame of stack) {
      const fileName = frame.getFileName?.();
      if (fileName) {
        return fileName;
      }
    }
    return undefined;
  } finally {
    Error.prepareStackTrace = previousPrepareStackTrace;
  }
}

function resolveNapiCandidates(): string[] {
  if (!isNode) {
    return [];
  }
  const platformKey = `${process.platform}-${process.arch}`;
  const distribution = NAPI_DISTRIBUTIONS[platformKey];
  const candidates = new Set<string>();
  if (distribution) {
    for (const pkg of distribution.packages) {
      candidates.add(pkg);
    }
    for (const suffix of distribution.binaries) {
      candidates.add(`${NAPI_BINDING_PREFIX}.${suffix}.node`);
    }
  }
  candidates.add(NAPI_ENTRY);
  return Array.from(candidates);
}

function isModuleNotFound(error: unknown): boolean {
  if (!error || typeof error !== "object") {
    return false;
  }
  const candidate = error as { code?: unknown };
  return candidate.code === "MODULE_NOT_FOUND";
}

export function readNpz(data: ArrayBuffer | Uint8Array): NamedMatrix[] {
  const archive = unzipSync(toBackendBytes(data));
  const results: NamedMatrix[] = [];
  for (const [name, content] of Object.entries(archive)) {
    if (!name.endsWith(".npy")) {
      continue;
    }
    const matrix = readNpy(content);
    const baseName = name.slice(0, -4);
    results.push({ name: baseName, matrix });
  }
  return results;
}

export function writeNpz(entries: NamedMatrix[]): Uint8Array {
  if (entries.length === 0) {
    throw new Error("writeNpz requires at least one matrix entry");
  }
  const archive: Record<string, Uint8Array> = {};
  for (const entry of entries) {
    const key = entry.name?.length ? `${entry.name}.npy` : "array.npy";
    archive[key] = writeNpy(entry.matrix);
  }
  return zipSync(archive);
}











