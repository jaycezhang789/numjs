import { strFromU8, strToU8, unzipSync, zipSync } from "fflate";
import {
  createWebGpuEngine,
  isWebGpuSupported,
  WebGpuEngine,
} from "./gpu/webgpu";
import { LazyArray, LazyScalar, constant as lazyConstant } from "./lazy";
import {
  arrowTableToMatrix,
  matrixToArrowTable,
  type ArrowTableLike,
  type ArrowToMatrixOptions,
  type ArrowConversionResult,
} from "./io/arrow";
import {
  polarsDataFrameToMatrix,
  matrixToPolarsDataFrame,
  type PolarsToMatrixOptions,
  type MatrixToPolarsOptions,
} from "./io/polars";

type BackendKind = "napi" | "wasm";

export const DEFAULT_RTOL = 1e-12;
export const DEFAULT_ATOL = 0;

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
  | "float64"
  | "fixed64";

export type InitOptions = {
  threads?: boolean | number;
};

export type GpuBackendKind = "webgpu" | "cuda" | "rocm";

export type GpuExecutionMode = "auto" | "gpu-only" | "cpu-only";

export type WebGpuInitOptions = {
  forceFallback?: boolean;
};

export type GpuMatmulOptions = {
  mode?: GpuExecutionMode;
};

export type Conv2DOptions = {
  stride?: number;
  pad?: number;
  mode?: GpuExecutionMode;
};

export type FftResult = {
  real: Matrix;
  imag: Matrix;
};

export type DataFrameInitOptions = {
  columns?: readonly string[];
  columnDTypes?: Record<string, DType>;
  inferColumnDTypes?: boolean;
};

export type CsvReadOptions = {
  delimiter?: string;
  hasHeader?: boolean;
  columns?: readonly string[];
  columnTypes?: Record<string, DType>;
  encoding?: string;
  skipEmptyLines?: boolean;
};

export type CsvWriteOptions = {
  delimiter?: string;
  includeHeader?: boolean;
  newline?: "\n" | "\r\n";
  encoding?: string;
};

export type ParquetReadOptions = {
  columns?: readonly string[];
  polarsModule?: unknown;
};

export type ParquetWriteOptions = {
  polarsModule?: unknown;
  compression?: string;
};

type BackendMatrixHandle = {
  readonly rows: number;
  readonly cols: number;
  // dtype may be missing on some backends; we cache it separately when needed
  readonly dtype?: DType;


  to_vec?(): Float64Array | number[];
  toVec?(): Float64Array | number[];
  astype(dtype: DType, copy?: boolean, casting?: string | null): BackendMatrixHandle;
  to_bytes?(): Uint8Array;
  toBytes?(): Uint8Array;
  toFloat32Array?(): Float32Array;
  to_float32_array?(): Float32Array;
  toFloat64Array?(): Float64Array;
  to_float64_array?(): Float64Array;


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
  from_fixed_i64?(
    data: BigInt64Array,
    rows: number,
    cols: number,
    scale: number
  ): BackendMatrixHandle;
  fromFixedI64?(
    data: BigInt64Array,
    rows: number,
    cols: number,
    scale: number
  ): BackendMatrixHandle;
};

type BackendModule = {
  Matrix: BackendMatrixConstructor;
  add(a: BackendMatrixHandle, b: BackendMatrixHandle): BackendMatrixHandle;
  sub(a: BackendMatrixHandle, b: BackendMatrixHandle): BackendMatrixHandle;
  mul(a: BackendMatrixHandle, b: BackendMatrixHandle): BackendMatrixHandle;
  div(a: BackendMatrixHandle, b: BackendMatrixHandle): BackendMatrixHandle;
  neg(matrix: BackendMatrixHandle): BackendMatrixHandle;
  exp?(matrix: BackendMatrixHandle): BackendMatrixHandle;
  log?(matrix: BackendMatrixHandle): BackendMatrixHandle;
  sin?(matrix: BackendMatrixHandle): BackendMatrixHandle;
  cos?(matrix: BackendMatrixHandle): BackendMatrixHandle;
  tanh?(matrix: BackendMatrixHandle): BackendMatrixHandle;
  sigmoid?(matrix: BackendMatrixHandle): BackendMatrixHandle;
  matmul(a: BackendMatrixHandle, b: BackendMatrixHandle): BackendMatrixHandle;
  sum?(matrix: BackendMatrixHandle, dtype?: DType | null): BackendMatrixHandle;
  nansum?(matrix: BackendMatrixHandle, dtype?: DType | null): BackendMatrixHandle;
  nanmean?(matrix: BackendMatrixHandle, dtype?: DType | null): BackendMatrixHandle;
  median?(matrix: BackendMatrixHandle, dtype?: DType | null): BackendMatrixHandle;
  quantile?(
    matrix: BackendMatrixHandle,
    q: number,
    dtype?: DType | null
  ): BackendMatrixHandle;
  percentile?(
    matrix: BackendMatrixHandle,
    p: number,
    dtype?: DType | null
  ): BackendMatrixHandle;
  dot?(
    a: BackendMatrixHandle,
    b: BackendMatrixHandle,
    dtype?: DType | null
  ): BackendMatrixHandle;
  initThreads?(threads?: number | null): Promise<void>;
  init_threads?(threads?: number | null): Promise<void>;
  from_fixed_i64?(
    data: BigInt64Array,
    rows: number,
    cols: number,
    scale: number
  ): BackendMatrixHandle;
  fromFixedI64?(
    data: BigInt64Array,
    rows: number,
    cols: number,
    scale: number
  ): BackendMatrixHandle;
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
  fft_axis?(
    matrix: BackendMatrixHandle,
    axis: number
  ): { real: BackendMatrixHandle; imag: BackendMatrixHandle };
  fft2d?(
    matrix: BackendMatrixHandle
  ): { real: BackendMatrixHandle; imag: BackendMatrixHandle };
  ifft_axis?(
    real: BackendMatrixHandle,
    imag: BackendMatrixHandle,
    axis: number
  ): { real: BackendMatrixHandle; imag: BackendMatrixHandle };
  ifft2d?(
    real: BackendMatrixHandle,
    imag: BackendMatrixHandle
  ): { real: BackendMatrixHandle; imag: BackendMatrixHandle };
  transpose?(matrix: BackendMatrixHandle): BackendMatrixHandle;
  broadcast_to?(
    matrix: BackendMatrixHandle,
    rows: number,
    cols: number
  ): BackendMatrixHandle;
  broadcastTo?(
    matrix: BackendMatrixHandle,
    rows: number,
    cols: number
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
let pendingInitOptions: InitOptions = {};
let wasmThreadsInitialized = false;
let webGpuEngine: WebGpuEngine | null = null;
let webGpuEnginePromise: Promise<WebGpuEngine | null> | null = null;
let activeGpuKind: GpuBackendKind | null = null;

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
  fixed64: { size: 8, kind: "signed", isFloat: false, isSigned: true },
};

export type BackendCapabilities = {
  kind: BackendKind;
  supportsMatrixFromBytes: boolean;
  supportsReadNpy: boolean;
  supportsWriteNpy: boolean;
  supportsCopyMetrics: boolean;
  supportedDTypes: readonly DType[];
};

const PROMOTION_TABLE: Partial<Record<DType, Partial<Record<DType, DType>>>> = {
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
      fixed64: "float64",
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
      fixed64: "float64",
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
      fixed64: "float64",
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
      fixed64: "float64",
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
      fixed64: "float64",
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
      fixed64: "float64",
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
      fixed64: "float64",
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
      fixed64: "float64",
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
      fixed64: "float64",
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
      fixed64: "float64",
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
      fixed64: "float64",
    },
  fixed64: {
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
    fixed64: "fixed64",
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
async function loadBackend(options: InitOptions): Promise<void> {
  if (activeBackend) {
    return;
  }

  if (isNode) {
    const napi = await loadNapiBackend();
    if (napi) {
      activeBackend = wrapBackendErrors(napi);
      activeKind = "napi";
      return;
    }
  }

  const wasm = await loadWasmBackend();
  activeBackend = wrapBackendErrors(wasm);
  activeKind = "wasm";
  await ensureWasmThreadState(options);
}

function mergeInitOptions(base: InitOptions, extra: InitOptions): InitOptions {
  const merged: InitOptions = { ...base };
  if (extra.threads !== undefined) {
    merged.threads = extra.threads;
  }
  return merged;
}

function supportsSharedArrayBuffer(): boolean {
  if (typeof SharedArrayBuffer === "undefined" || typeof Atomics === "undefined") {
    return false;
  }
  if (
    typeof self !== "undefined" &&
    Object.prototype.hasOwnProperty.call(self, "crossOriginIsolated") &&
    (self as { crossOriginIsolated?: boolean }).crossOriginIsolated === false
  ) {
    return false;
  }
  return true;
}

async function ensureWasmThreadState(options: InitOptions): Promise<void> {
  if (activeKind !== "wasm") {
    return;
  }
  const requested = options.threads;
  if (!requested || wasmThreadsInitialized) {
    return;
  }
  if (!supportsSharedArrayBuffer()) {
    console.warn(
      "[numjs] SharedArrayBuffer/Atomics unavailable; skipping WASM thread pool initialisation."
    );
    return;
  }
  const backend = ensureBackend();
  const initThreads =
    (backend as { initThreads?: (threads?: number | null) => Promise<void> }).initThreads ??
    (backend as { init_threads?: (threads?: number | null) => Promise<void> }).init_threads;
  if (typeof initThreads !== "function") {
    console.warn(
      "[numjs] WASM backend does not expose initThreads(); skipping thread pool initialisation."
    );
    wasmThreadsInitialized = true;
    return;
  }
  const normalized =
    typeof requested === "number" ? Math.max(1, Math.floor(requested)) : undefined;
  try {
    await initThreads.call(backend, normalized ?? null);
  } catch (error) {
    console.warn("[numjs] Failed to initialise WASM thread pool:", error);
  }
  wasmThreadsInitialized = true;
}

async function loadNapiBackend(): Promise<BackendModule | null> {
  const candidates = resolveNapiCandidates();
  for (const specifier of candidates) {
    try {
      const required = (await getRequire())(specifier) as BackendModule;
      if (!isNapiBackendSufficient(required)) {
        continue;
      }
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

function wrapBackendErrors(module: BackendModule): BackendModule {
  return new Proxy(module, {
    get(target, prop, receiver) {
      const value = Reflect.get(target, prop, receiver);
      if (prop === "Matrix" || typeof value !== "function") {
        return value;
      }
      const wrapped = function (this: unknown, ...args: unknown[]) {
        try {
          const result = (value as (...fnArgs: unknown[]) => unknown).apply(
            this === undefined ? target : this,
            args
          );
          if (isPromiseLike(result)) {
            return (result as Promise<unknown>).catch((error) => {
              throw normalizeBackendError(error);
            });
          }
          return result;
        } catch (error) {
          throw normalizeBackendError(error);
        }
      };
      return wrapped as typeof value;
    },
  }) as BackendModule;
}

function normalizeBackendError(error: unknown): Error {
  if (error instanceof Error) {
    const existingCode = (error as { code?: unknown }).code;
    if (typeof existingCode === "string" && existingCode.startsWith("E_")) {
      return error;
    }
    const parsed = parseCoreErrorString(error.message);
    if (parsed) {
      error.message = parsed.message;
      (error as { code?: string }).code = parsed.code;
    }
    return error;
  }
  if (typeof error === "string") {
    const parsed = parseCoreErrorString(error);
    if (parsed) {
      const normalized = new Error(parsed.message);
      (normalized as { code?: string }).code = parsed.code;
      return normalized;
    }
    return new Error(error);
  }
  const coerced = String(error);
  const parsed = parseCoreErrorString(coerced);
  if (parsed) {
    const normalized = new Error(parsed.message);
    (normalized as { code?: string }).code = parsed.code;
    return normalized;
  }
  return new Error(coerced);
}

function parseCoreErrorString(message: string): { code: string; message: string } | null {
  const separatorIndex = message.indexOf(": ");
  if (separatorIndex <= 0) {
    return null;
  }
  const code = message.slice(0, separatorIndex);
  if (!code.startsWith("E_")) {
    return null;
  }
  const detail = message.slice(separatorIndex + 2);
  return { code, message: detail };
}

function isPromiseLike(value: unknown): value is Promise<unknown> {
  if (value === null) return false;
  const type = typeof value;
  if (type !== "object" && type !== "function") return false;
  return typeof (value as { then?: unknown }).then === "function";
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

function getMatrixFixedScaleFromHandle(handle: BackendMatrixHandle): number | null {
  const candidate = (handle as any).fixedScale ?? (handle as any).fixed_scale;
  if (typeof candidate === "number") {
    return candidate;
  }
  if (candidate === null) {
    return null;
  }
  const methodNames = ["fixedScale", "fixed_scale", "getFixedScale"];
  for (const name of methodNames) {
    const fn = (handle as any)[name];
    if (typeof fn === "function") {
      try {
        const value = fn.call(handle);
        if (typeof value === "number") {
          return value;
        }
        if (value === null || value === undefined) {
          return null;
        }
      } catch {
        // ignore backend getter failures and fall through
      }
    }
  }
  return null;
}

function getMatrixFixedScale(matrix: Matrix): number | null {
  const cache = (matrix as any)._fixedScaleCache as number | null | undefined;
  if (cache !== undefined) {
    return cache;
  }
  if (matrix.dtype !== "fixed64") {
    (matrix as any)._fixedScaleCache = null;
    return null;
  }
  const scale = getMatrixFixedScaleFromHandle(getHandle(matrix));
  (matrix as any)._fixedScaleCache = scale ?? null;
  return scale ?? null;
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

async function ensureWebGpuEngine(
  options?: WebGpuInitOptions
): Promise<WebGpuEngine | null> {
  if (options?.forceFallback) {
    webGpuEngine = null;
    webGpuEnginePromise = null;
    if (activeGpuKind === "webgpu") {
      activeGpuKind = null;
    }
    return null;
  }
  if (webGpuEngine) {
    return webGpuEngine;
  }
  if (!webGpuEnginePromise) {
    webGpuEnginePromise = createWebGpuEngine(options).catch((error) => {
      console.warn("[numjs] Failed to initialise WebGPU engine:", error);
      return null;
    });
  }
  const engine = await webGpuEnginePromise;
  if (engine) {
    webGpuEngine = engine;
    activeGpuKind = "webgpu";
  } else {
    if (activeGpuKind === "webgpu") {
      activeGpuKind = null;
    }
    webGpuEnginePromise = null;
  }
  return webGpuEngine;
}

export async function initWebGpu(
  options: WebGpuInitOptions = {}
): Promise<boolean> {
  const engine = await ensureWebGpuEngine(options);
  return engine !== null;
}

export function webGpuAvailable(): boolean {
  return !!webGpuEngine || isWebGpuSupported();
}

export function gpuBackendKind(): GpuBackendKind | null {
  if (activeGpuKind) {
    return activeGpuKind;
  }
  if (isNode) {
    try {
      const backend = ensureBackend();
      const kindFn =
        (backend as Record<string, unknown>).gpu_backend_kind ??
        (backend as Record<string, unknown>).gpuBackendKind;
      if (typeof kindFn === "function") {
        const result = kindFn.call(backend);
        if (result === "cuda") {
          activeGpuKind = "cuda";
          return activeGpuKind;
        }
        if (result === "rocm") {
          activeGpuKind = "rocm";
          return activeGpuKind;
        }
      }
    } catch (error) {
      console.warn("[numjs] Failed to query native GPU backend:", error);
    }
  }
  return activeGpuKind;
}

export function gpuAvailable(): boolean {
  return gpuBackendKind() !== null || webGpuAvailable();
}

async function selectGpuEngine(
  mode: GpuExecutionMode | undefined
): Promise<WebGpuEngine | null> {
  if (mode === "cpu-only") {
    return null;
  }
  const engine = await ensureWebGpuEngine();
  if (!engine && mode === "gpu-only") {
    throw new Error("WebGPU backend requested but unavailable in this environment");
  }
  return engine;
}

function toFloat32View(matrix: Matrix): {
  rows: number;
  cols: number;
  array: Float32Array;
} {
  const source =
    matrix.dtype === "float32" ? matrix : matrix.astype("float32", { copy: true });
  const arr = source.toArray();
  if (arr instanceof Float32Array) {
    return { rows: source.rows, cols: source.cols, array: arr };
  }
  return {
    rows: source.rows,
    cols: source.cols,
    array: Float32Array.from(arr as ArrayLike<number>),
  };
}

function computeConv2DOutputShape(
  inputRows: number,
  inputCols: number,
  kernelRows: number,
  kernelCols: number,
  stride: number,
  pad: number
): { rows: number; cols: number } {
  if (stride <= 0) {
    throw new Error("conv2d: stride must be positive");
  }
  if (kernelRows <= 0 || kernelCols <= 0) {
    throw new Error("conv2d: kernel dimensions must be positive");
  }
  const outRows = Math.floor((inputRows + 2 * pad - kernelRows) / stride + 1);
  const outCols = Math.floor((inputCols + 2 * pad - kernelCols) / stride + 1);
  if (outRows <= 0 || outCols <= 0) {
    throw new Error(
      `conv2d: invalid output shape (${outRows} x ${outCols}); check stride/padding/kernel dimensions`
    );
  }
  return { rows: outRows, cols: outCols };
}

function conv2dCpuImplementation(
  input: Float32Array,
  kernel: Float32Array,
  inputRows: number,
  inputCols: number,
  kernelRows: number,
  kernelCols: number,
  stride: number,
  pad: number
): Float32Array {
  const { rows: outRows, cols: outCols } = computeConv2DOutputShape(
    inputRows,
    inputCols,
    kernelRows,
    kernelCols,
    stride,
    pad
  );
  const output = new Float32Array(outRows * outCols);
  for (let orow = 0; orow < outRows; orow += 1) {
    for (let ocol = 0; ocol < outCols; ocol += 1) {
      let acc = 0;
      for (let krow = 0; krow < kernelRows; krow += 1) {
        for (let kcol = 0; kcol < kernelCols; kcol += 1) {
          const inRow = orow * stride + krow - pad;
          const inCol = ocol * stride + kcol - pad;
          if (inRow < 0 || inRow >= inputRows || inCol < 0 || inCol >= inputCols) {
            continue;
          }
          const inputIndex = inRow * inputCols + inCol;
          const kernelIndex = krow * kernelCols + kcol;
          acc += input[inputIndex] * kernel[kernelIndex];
        }
      }
      output[orow * outCols + ocol] = acc;
    }
  }
  return output;
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
  const { buffer, byteOffset, byteLength } = bytes;
  switch (dtype) {
    case "float64":
      return new Float64Array(
        buffer,
        byteOffset,
        byteLength / Float64Array.BYTES_PER_ELEMENT
      );
    case "float32":
      return new Float32Array(
        buffer,
        byteOffset,
        byteLength / Float32Array.BYTES_PER_ELEMENT
      );
    case "int8":
      return new Int8Array(buffer, byteOffset, byteLength);
    case "int16":
      return new Int16Array(
        buffer,
        byteOffset,
        byteLength / Int16Array.BYTES_PER_ELEMENT
      );
    case "int32":
      return new Int32Array(
        buffer,
        byteOffset,
        byteLength / Int32Array.BYTES_PER_ELEMENT
      );
    case "int64":
      return new BigInt64Array(
        buffer,
        byteOffset,
        byteLength / BigInt64Array.BYTES_PER_ELEMENT
      );
    case "uint8":
      return new Uint8Array(buffer, byteOffset, byteLength);
    case "uint16":
      return new Uint16Array(
        buffer,
        byteOffset,
        byteLength / Uint16Array.BYTES_PER_ELEMENT
      );
    case "uint32":
      return new Uint32Array(
        buffer,
        byteOffset,
        byteLength / Uint32Array.BYTES_PER_ELEMENT
      );
    case "uint64":
      return new BigUint64Array(
        buffer,
        byteOffset,
        byteLength / BigUint64Array.BYTES_PER_ELEMENT
      );
    case "fixed64":
      return new BigInt64Array(
        buffer,
        byteOffset,
        byteLength / BigInt64Array.BYTES_PER_ELEMENT
      );
    case "bool":
      return new Uint8Array(buffer, byteOffset, byteLength);
    default:
      return new Uint8Array(buffer, byteOffset, byteLength);
  }
}

function isTypedArray(value: unknown): value is TypedArray {
  return ArrayBuffer.isView(value) && !(value instanceof DataView);
}

function handleToTypedArray(handle: BackendMatrixHandle): TypedArray {
  const dtype = getMatrixDTypeFromHandle(handle);

  if (dtype === "float32") {
    const direct =
      (handle as { toFloat32Array?: () => unknown }).toFloat32Array ??
      (handle as { to_float32_array?: () => unknown }).to_float32_array;
    if (typeof direct === "function") {
      const result = direct.call(handle);
      if (isTypedArray(result)) {
        return result as Float32Array;
      }
    }
  } else if (dtype === "float64") {
    const direct =
      (handle as { toFloat64Array?: () => unknown }).toFloat64Array ??
      (handle as { to_float64_array?: () => unknown }).to_float64_array;
    if (typeof direct === "function") {
      const result = direct.call(handle);
      if (isTypedArray(result)) {
        return result as Float64Array;
      }
    }
  }

  const bytes = callHandleToBytes(handle);
  return typedArrayFromBytes(bytes, dtype);
}

export async function init(options: InitOptions = {}): Promise<void> {
  pendingInitOptions = mergeInitOptions(pendingInitOptions, options);
  if (activeBackend) {
    await ensureWasmThreadState(pendingInitOptions);
    if (!isNode) {
      await initWebGpu().catch(() => {
        /* noop - fall back to CPU */
      });
    }
    return;
  }
  if (!pendingLoad) {
    const scheduledOptions = pendingInitOptions;
    pendingLoad = loadBackend(scheduledOptions).finally(() => {
      pendingLoad = null;
    });
  }
  await pendingLoad;
  await ensureWasmThreadState(pendingInitOptions);
  if (!isNode) {
    await initWebGpu().catch(() => {
      /* noop - fall back to CPU */
    });
  }
}

export class Matrix {
  private _handle: BackendMatrixHandle;
  private _dtypeCache?: DType;
  private _fixedScaleCache?: number | null;

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
        this._fixedScaleCache =
          targetDType === "fixed64"
            ? getMatrixFixedScaleFromHandle(this._handle)
            : null;
        return;
      }

      const base = new MatrixCtor(toFloat64Array(data), rows, cols);
      this._handle =
        targetDType === "float64" ? base : base.astype(targetDType);
      this._dtypeCache = targetDType;
      this._fixedScaleCache =
        targetDType === "fixed64"
          ? getMatrixFixedScaleFromHandle(this._handle)
          : null;
  }

  static fromHandle(handle: BackendMatrixHandle): Matrix {
    return wrapMatrix(handle);
  }

  static fromHandleWithDType(
    handle: BackendMatrixHandle,
    dtype: DType,
    metadata?: { fixedScale?: number | null }
  ): Matrix {
    const m = wrapMatrix(handle);
    (m as any)._dtypeCache = dtype;
    if (dtype === "fixed64") {
      const scale =
        metadata && "fixedScale" in metadata
          ? metadata.fixedScale ?? null
          : getMatrixFixedScaleFromHandle(handle);
      (m as any)._fixedScaleCache = scale ?? null;
    } else {
      (m as any)._fixedScaleCache = null;
    }
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

  static fromFixed(
    data: BigInt64Array | readonly bigint[] | Iterable<bigint>,
    rows: number,
    cols: number,
    scale: number
  ): Matrix {
    const expectedLength = rows * cols;
    const typed =
      data instanceof BigInt64Array
        ? new BigInt64Array(data)
        : BigInt64Array.from(data as Iterable<bigint>);
    if (typed.length !== expectedLength) {
      throw new Error(
        `Matrix.fromFixed: data length (${typed.length}) does not match shape (${rows} x ${cols})`
      );
    }
    const backend = ensureBackend();
    const MatrixCtor = backend.Matrix as BackendMatrixConstructor & {
      from_fixed_i64?: (
        data: BigInt64Array,
        rows: number,
        cols: number,
        scale: number
      ) => BackendMatrixHandle;
      fromFixedI64?: (
        data: BigInt64Array,
        rows: number,
        cols: number,
        scale: number
      ) => BackendMatrixHandle;
    };
    let handle: BackendMatrixHandle | undefined;
    if (typeof MatrixCtor.from_fixed_i64 === "function") {
      handle = MatrixCtor.from_fixed_i64(typed, rows, cols, scale);
    } else if (typeof MatrixCtor.fromFixedI64 === "function") {
      handle = MatrixCtor.fromFixedI64(typed, rows, cols, scale);
    } else if (typeof (backend as any).from_fixed_i64 === "function") {
      handle = (backend as any).from_fixed_i64(typed, rows, cols, scale) as BackendMatrixHandle;
    } else if (typeof (backend as any).fromFixedI64 === "function") {
      handle = (backend as any).fromFixedI64(typed, rows, cols, scale) as BackendMatrixHandle;
    }
    if (!handle) {
      throw new Error("Matrix.fromFixed: current backend does not support fixed64 matrices");
    }
    return Matrix.fromHandleWithDType(handle, "fixed64", { fixedScale: scale });
  }

  static fromLazy(lazy: LazyArray, options: MatrixOptions = {}): Matrix {
    const { data, rows, cols } = lazy.evaluate();
    const base = new Matrix(data, rows, cols, { dtype: "float64" });
    const target = options.dtype ?? "float64";
    if (target !== "float64") {
      return base.astype(target, { copy: false });
    }
    return base;
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

  get fixedScale(): number | null {
    return getMatrixFixedScale(this);
  }

  toArray(): TypedArray {
    return handleToTypedArray(this._handle);
  }

  toLazy(): LazyArray {
    const float64 = this.astype("float64", { copy: false }).toArray();
    const data =
      float64 instanceof Float64Array
        ? float64
        : Float64Array.from(float64 as ArrayLike<number>);
    return LazyArray.fromDense(data, this.rows, this.cols);
  }

  get dtypeInfo(): DTypeInfo {
    return DTYPE_INFO[this.dtype];
  }


  row(index: number): Matrix {
    return row(this, index);
  }

  column(index: number): Matrix {
    return column(this, index);
  }

  slice(
    rows?: { start?: number; end?: number; step?: number },
    cols?: { start?: number; end?: number; step?: number }
  ): Matrix {
    return slice(this, rows, cols);
  }  toBytes(): Uint8Array {
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

  transpose(): Matrix {
    return transpose(this);
  }

  broadcastTo(rows: number, cols: number): Matrix {
    return broadcastTo(this, rows, cols);
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

  astype(
    dtype: DType,
    options: { copy?: boolean; casting?: string } = {}
  ): Matrix {
    const copy = options.copy ?? false;
    const casting = options.casting ?? null;
    if (!copy && dtype === this.dtype) {
      return this;
    }
    const handle = getHandle(this).astype(dtype, copy, casting);
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
    case "fixed64":
      return new BigInt64Array(length);
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
    case "fixed64":
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

function broadcastMatrixInJs(matrix: Matrix, rows: number, cols: number): Matrix {
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
  if (dtype === "fixed64") {
    const scale = matrix.fixedScale ?? 0;
    return Matrix.fromFixed(target as BigInt64Array, rows, cols, scale);
  }
  return new Matrix(target, rows, cols, { dtype });
}

function transposeMatrixInJs(matrix: Matrix): Matrix {
  const dtype = matrix.dtype;
  const srcRows = matrix.rows;
  const srcCols = matrix.cols;
  const rows = srcCols;
  const cols = srcRows;
  const source = matrix.toArray();
  const target = allocateArrayForDType(dtype, rows * cols);
  for (let destRow = 0; destRow < rows; destRow += 1) {
    for (let destCol = 0; destCol < cols; destCol += 1) {
      const srcRow = destCol;
      const srcCol = destRow;
      const srcIndex = srcRow * srcCols + srcCol;
      const dstIndex = destRow * cols + destCol;
      writeMatrixValue(target, dstIndex, source[srcIndex], dtype);
    }
  }
  if (dtype === "fixed64") {
    const scale = matrix.fixedScale ?? 0;
    return Matrix.fromFixed(target as BigInt64Array, rows, cols, scale);
  }
  return new Matrix(target, rows, cols, { dtype });
}


export function matrixFromBytes(
  data: ArrayBuffer | Uint8Array,
  rows: number,
  cols: number,
  dtype: DType
): Matrix {
  if (dtype === "fixed64") {
    throw new Error(
      "matrixFromBytes: fixed64 requires an explicit scale; use matrixFromFixed instead"
    );
  }
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

export function matrixFromFixed(
  data: BigInt64Array | readonly bigint[] | Iterable<bigint>,
  rows: number,
  cols: number,
  scale: number
): Matrix {
  return Matrix.fromFixed(data, rows, cols, scale);
}

export function add(a: Matrix, b: Matrix): Matrix {
  const dtype = promoteBinaryDType(a.dtype, b.dtype);
  const left = castToDType(a, dtype);
  const right = castToDType(b, dtype);
  const backend = ensureBackend();
  const result = backend.add(getHandle(left), getHandle(right));
  return Matrix.fromHandleWithDType(result, dtype);
}

export function sub(a: Matrix, b: Matrix): Matrix {
  const dtype = promoteBinaryDType(a.dtype, b.dtype);
  const left = castToDType(a, dtype);
  const right = castToDType(b, dtype);
  const backend = ensureBackend();
  const result = backend.sub(getHandle(left), getHandle(right));
  return Matrix.fromHandleWithDType(result, dtype);
}

export function mul(a: Matrix, b: Matrix): Matrix {
  const dtype = promoteBinaryDType(a.dtype, b.dtype);
  const left = castToDType(a, dtype);
  const right = castToDType(b, dtype);
  const backend = ensureBackend();
  const result = backend.mul(getHandle(left), getHandle(right));
  return Matrix.fromHandleWithDType(result, dtype);
}

export function div(a: Matrix, b: Matrix): Matrix {
  const dtype = promoteBinaryDType(a.dtype, b.dtype);
  const left = castToDType(a, dtype);
  const right = castToDType(b, dtype);
  const backend = ensureBackend();
  const result = backend.div(getHandle(left), getHandle(right));
  return Matrix.fromHandleWithDType(result, dtype);
}

export function neg(matrix: Matrix): Matrix {
  const backend = ensureBackend();
  const result = backend.neg(getHandle(matrix));
  return Matrix.fromHandleWithDType(result, matrix.dtype, {
    fixedScale: matrix.dtype === "fixed64" ? matrix.fixedScale : undefined,
  });
}

export function exp(matrix: Matrix): Matrix {
  const backend = ensureBackend();
  if (backend.exp) {
    const handle = backend.exp(getHandle(matrix));
    const dtype = getMatrixDTypeFromHandle(handle);
    const fixedScale = dtype === "fixed64" ? getMatrixFixedScaleFromHandle(handle) : undefined;
    return Matrix.fromHandleWithDType(handle, dtype, { fixedScale });
  }
  return unaryFloatOpInJs(matrix, Math.exp, matrix.dtype === "float32");
}

export function log(matrix: Matrix): Matrix {
  const backend = ensureBackend();
  if (backend.log) {
    const handle = backend.log(getHandle(matrix));
    const dtype = getMatrixDTypeFromHandle(handle);
    return Matrix.fromHandleWithDType(handle, dtype);
  }
  return unaryFloatOpInJs(matrix, Math.log, matrix.dtype === "float32");
}

export function sin(matrix: Matrix): Matrix {
  const backend = ensureBackend();
  if (backend.sin) {
    const handle = backend.sin(getHandle(matrix));
    const dtype = getMatrixDTypeFromHandle(handle);
    return Matrix.fromHandleWithDType(handle, dtype);
  }
  return unaryFloatOpInJs(matrix, Math.sin, matrix.dtype === "float32");
}

export function cos(matrix: Matrix): Matrix {
  const backend = ensureBackend();
  if (backend.cos) {
    const handle = backend.cos(getHandle(matrix));
    const dtype = getMatrixDTypeFromHandle(handle);
    return Matrix.fromHandleWithDType(handle, dtype);
  }
  return unaryFloatOpInJs(matrix, Math.cos, matrix.dtype === "float32");
}

export function tanh(matrix: Matrix): Matrix {
  const backend = ensureBackend();
  if (backend.tanh) {
    const handle = backend.tanh(getHandle(matrix));
    const dtype = getMatrixDTypeFromHandle(handle);
    return Matrix.fromHandleWithDType(handle, dtype);
  }
  return unaryFloatOpInJs(matrix, Math.tanh, matrix.dtype === "float32");
}

export function sigmoid(matrix: Matrix): Matrix {
  const backend = ensureBackend();
  if (backend.sigmoid) {
    const handle = backend.sigmoid(getHandle(matrix));
    const dtype = getMatrixDTypeFromHandle(handle);
    return Matrix.fromHandleWithDType(handle, dtype);
  }
  return unaryFloatOpInJs(matrix, sigmoidStable, matrix.dtype === "float32");
}

export function matmul(a: Matrix, b: Matrix): Matrix {
  if (a.dtype === "fixed64" || b.dtype === "fixed64") {
    throw new Error("matmul does not support fixed64 matrices; cast operands to float64 first");
  }
  const dtype = promoteBinaryDType(a.dtype, b.dtype);
  const left = castToDType(a, dtype);
  const right = castToDType(b, dtype);
  const backend = ensureBackend();
  const result = backend.matmul(getHandle(left), getHandle(right));
  return Matrix.fromHandleWithDType(result, dtype);
}

export async function matmulAsync(
  a: Matrix,
  b: Matrix,
  options: GpuMatmulOptions = {}
): Promise<Matrix> {
  if (a.dtype === "fixed64" || b.dtype === "fixed64") {
    throw new Error("matmulAsync does not support fixed64 matrices; cast operands to float32 first");
  }
  if (a.cols !== b.rows) {
    throw new Error(
      `matmulAsync dimension mismatch: left columns (${a.cols}) must equal right rows (${b.rows})`
    );
  }
  if (isNode) {
    const backend = ensureBackend();
    const nativeGpuMatmul =
      (backend as Record<string, unknown>).gpu_matmul ??
      (backend as Record<string, unknown>).gpuMatmul;
    if (typeof nativeGpuMatmul === "function") {
      try {
        const handle = (nativeGpuMatmul as (lhs: BackendMatrixHandle, rhs: BackendMatrixHandle) => BackendMatrixHandle)(
          getHandle(a),
          getHandle(b)
        );
        const dtype = getMatrixDTypeFromHandle(handle) ?? "float32";
        const metadata =
          dtype === "fixed64"
            ? { fixedScale: getMatrixFixedScaleFromHandle(handle) ?? null }
            : undefined;
        activeGpuKind = "cuda";
        return Matrix.fromHandleWithDType(handle, dtype, metadata);
      } catch (error) {
        console.warn("[numjs] N-API gpu_matmul failed; falling back to other accelerators.", error);
      }
    }
  }
  const engine = await selectGpuEngine(options.mode ?? "auto");
  if (!engine) {
    return Promise.resolve(matmul(a, b));
  }
  const left = toFloat32View(a);
  const right = toFloat32View(b);
  if (left.cols !== right.rows) {
    throw new Error(
      `matmulAsync dimension mismatch: left columns (${left.cols}) must equal right rows (${right.rows})`
    );
  }
  const result = await engine.matmul({
    a: left.array,
    b: right.array,
    rowsA: left.rows,
    colsA: left.cols,
    colsB: right.cols,
  });
  return new Matrix(result, left.rows, right.cols, { dtype: "float32" });
}

export async function conv2d(
  input: Matrix,
  kernel: Matrix,
  options: Conv2DOptions = {}
): Promise<Matrix> {
  if (input.dtype === "fixed64" || kernel.dtype === "fixed64") {
    throw new Error("conv2d does not support fixed64 matrices; cast to float32 first");
  }
  const stride = options.stride ?? 1;
  const pad = options.pad ?? 0;
  const inputView = toFloat32View(input);
  const kernelView = toFloat32View(kernel);
  const { rows: outRows, cols: outCols } = computeConv2DOutputShape(
    inputView.rows,
    inputView.cols,
    kernelView.rows,
    kernelView.cols,
    stride,
    pad
  );
  const engine = await selectGpuEngine(options.mode ?? "auto");
  if (engine) {
    const result = await engine.conv2d({
      input: inputView.array,
      kernel: kernelView.array,
      inputRows: inputView.rows,
      inputCols: inputView.cols,
      kernelRows: kernelView.rows,
      kernelCols: kernelView.cols,
      stride,
      pad,
    });
    return new Matrix(result, outRows, outCols, { dtype: "float32" });
  }
  const cpuResult = conv2dCpuImplementation(
    inputView.array,
    kernelView.array,
    inputView.rows,
    inputView.cols,
    kernelView.rows,
    kernelView.cols,
    stride,
    pad
  );
  return new Matrix(cpuResult, outRows, outCols, { dtype: "float32" });
}

export function im2col(
  input: Matrix,
  kernelRows: number,
  kernelCols: number,
  options: Conv2DOptions = {}
): Matrix {
  if (kernelRows <= 0 || kernelCols <= 0) {
    throw new Error("im2col: kernel dimensions must be positive");
  }
  const stride = options.stride ?? 1;
  const pad = options.pad ?? 0;
  if (stride <= 0) {
    throw new Error("im2col: stride must be positive");
  }
  const view = toFloat32View(input);
  const outRows = Math.floor((view.rows + 2 * pad - kernelRows) / stride + 1);
  const outCols = Math.floor((view.cols + 2 * pad - kernelCols) / stride + 1);
  if (outRows <= 0 || outCols <= 0) {
    throw new Error("im2col: output dimensions are non-positive; adjust stride/padding");
  }
  const columns = outRows * outCols;
  const rows = kernelRows * kernelCols;
  const data = new Float32Array(rows * columns);
  let colIndex = 0;
  for (let outRow = 0; outRow < outRows; outRow += 1) {
    for (let outCol = 0; outCol < outCols; outCol += 1) {
      for (let kRow = 0; kRow < kernelRows; kRow += 1) {
        for (let kCol = 0; kCol < kernelCols; kCol += 1) {
          const sourceRow = outRow * stride + kRow - pad;
          const sourceCol = outCol * stride + kCol - pad;
          const rowIndex = kRow * kernelCols + kCol;
          const destIndex = rowIndex * columns + colIndex;
          if (
            sourceRow < 0 ||
            sourceRow >= view.rows ||
            sourceCol < 0 ||
            sourceCol >= view.cols
          ) {
            data[destIndex] = 0;
          } else {
            data[destIndex] = view.array[sourceRow * view.cols + sourceCol];
          }
        }
      }
      colIndex += 1;
    }
  }
  return new Matrix(data, rows, columns, { dtype: "float32" });
}

function pool2d(
  input: Matrix,
  kernelRows: number,
  kernelCols: number,
  reducer: (acc: number, value: number) => number,
  initial: number,
  finalize: (acc: number, count: number) => number,
  options: Conv2DOptions = {}
): Matrix {
  if (kernelRows <= 0 || kernelCols <= 0) {
    throw new Error("pool2d: kernel dimensions must be positive");
  }
  const stride = options.stride ?? kernelRows;
  const pad = options.pad ?? 0;
  if (stride <= 0) {
    throw new Error("pool2d: stride must be positive");
  }
  const view = toFloat32View(input);
  const outRows = Math.floor((view.rows + 2 * pad - kernelRows) / stride + 1);
  const outCols = Math.floor((view.cols + 2 * pad - kernelCols) / stride + 1);
  if (outRows <= 0 || outCols <= 0) {
    throw new Error("pool2d: output dimensions are non-positive; adjust stride/padding");
  }
  const result = new Float32Array(outRows * outCols);
  let index = 0;
  for (let outRow = 0; outRow < outRows; outRow += 1) {
    for (let outCol = 0; outCol < outCols; outCol += 1) {
      let acc = initial;
      let count = 0;
      for (let kRow = 0; kRow < kernelRows; kRow += 1) {
        for (let kCol = 0; kCol < kernelCols; kCol += 1) {
          const sourceRow = outRow * stride + kRow - pad;
          const sourceCol = outCol * stride + kCol - pad;
          if (
            sourceRow < 0 ||
            sourceRow >= view.rows ||
            sourceCol < 0 ||
            sourceCol >= view.cols
          ) {
            continue;
          }
          const value = view.array[sourceRow * view.cols + sourceCol];
          acc = reducer(acc, value);
          count += 1;
        }
      }
      result[index++] = finalize(acc, count);
    }
  }
  return new Matrix(result, outRows, outCols, { dtype: "float32" });
}

export function maxPool(
  input: Matrix,
  kernelRows: number,
  kernelCols: number,
  options: Conv2DOptions = {}
): Matrix {
  return pool2d(
    input,
    kernelRows,
    kernelCols,
    (acc, value) => (acc > value ? acc : value),
    Number.NEGATIVE_INFINITY,
    (acc) => acc,
    options
  );
}

export function avgPool(
  input: Matrix,
  kernelRows: number,
  kernelCols: number,
  options: Conv2DOptions = {}
): Matrix {
  return pool2d(
    input,
    kernelRows,
    kernelCols,
    (acc, value) => acc + value,
    0,
    (acc, count) => (count === 0 ? 0 : acc / count),
    options
  );
}

export async function sobelFilter(
  input: Matrix,
  options: { mode?: Conv2DOptions; magnitude?: boolean } = {}
): Promise<{ gx: Matrix; gy: Matrix; magnitude?: Matrix }> {
  const kx = new Matrix(
    new Float32Array([1, 0, -1, 2, 0, -2, 1, 0, -1]),
    3,
    3,
    { dtype: "float32" }
  );
  const ky = new Matrix(
    new Float32Array([1, 2, 1, 0, 0, 0, -1, -2, -1]),
    3,
    3,
    { dtype: "float32" }
  );
  const gx = await conv2d(input, kx, options.mode ?? {});
  const gy = await conv2d(input, ky, options.mode ?? {});
  let magnitude: Matrix | undefined;
  if (options.magnitude !== false) {
    const gxArr = gx.astype("float64", { copy: false }).toArray() as Float64Array;
    const gyArr = gy.astype("float64", { copy: false }).toArray() as Float64Array;
    const mag = new Float64Array(gxArr.length);
    for (let i = 0; i < gxArr.length; i += 1) {
      const x = gxArr[i];
      const y = gyArr[i];
      mag[i] = Math.hypot(x, y);
    }
    magnitude = new Matrix(mag, gx.rows, gx.cols, { dtype: "float64" });
  }
  return { gx, gy, magnitude };
}

export async function gaussianBlur(
  input: Matrix,
  options: { sigma?: number; size?: number; mode?: Conv2DOptions } = {}
): Promise<Matrix> {
  const sigma = options.sigma ?? 1;
  const sizeInput = options.size ?? Math.ceil(sigma * 6);
  let size = sizeInput | 1;
  if (size < 3) size = 3;
  if (size % 2 === 0) size += 1;
  const radius = (size - 1) / 2;
  const kernel = new Float32Array(size * size);
  const sigmaSq = sigma * sigma;
  let sum = 0;
  for (let y = -radius; y <= radius; y += 1) {
    for (let x = -radius; x <= radius; x += 1) {
      const idx = (y + radius) * size + (x + radius);
      const value = Math.exp(-(x * x + y * y) / (2 * sigmaSq));
      kernel[idx] = value;
      sum += value;
    }
  }
  for (let i = 0; i < kernel.length; i += 1) {
    kernel[i] /= sum;
  }
  const kernelMatrix = new Matrix(kernel, size, size, { dtype: "float32" });
  return conv2d(input, kernelMatrix, options.mode ?? {});
}


export function row(matrix: Matrix, index: number): Matrix {
  const backend = ensureBackend();
  const fn = (backend as any).row ?? (backend as any).Row;
  if (typeof fn === "function") {
    const handle = fn(getHandle(matrix), index);
    const meta = matrix.dtype === "fixed64" ? { fixedScale: matrix.fixedScale } : undefined;
    return Matrix.fromHandleWithDType(handle, matrix.dtype, meta);
  }
  return take(matrix, 0, [index]);
}

export function column(matrix: Matrix, index: number): Matrix {
  const backend = ensureBackend();
  const fn = (backend as any).column ?? (backend as any).Column;
  if (typeof fn === "function") {
    const handle = fn(getHandle(matrix), index);
    const meta = matrix.dtype === "fixed64" ? { fixedScale: matrix.fixedScale } : undefined;
    return Matrix.fromHandleWithDType(handle, matrix.dtype, meta);
  }
  return take(matrix, 1, [index]);
}

export function slice(
  matrix: Matrix,
  rows?: { start?: number; end?: number; step?: number },
  cols?: { start?: number; end?: number; step?: number }
): Matrix {
  const backend = ensureBackend();
  const fn = (backend as any).slice ?? (backend as any).Slice;
  if (typeof fn !== "function") {
    throw new Error("slice is not supported by current backend");
  }
  const rs = rows ?? {};
  const cs = cols ?? {};
  const rowStart = (rs.start ?? null) as number | null;
  const rowEnd = (rs.end ?? null) as number | null;
  const rowStep = (rs.step ?? 1) as number | null;
  const colStart = (cs.start ?? null) as number | null;
  const colEnd = (cs.end ?? null) as number | null;
  const colStep = (cs.step ?? 1) as number | null;
  const handle = fn(
    getHandle(matrix),
    rowStart,
    rowEnd,
    rowStep,
    colStart,
    colEnd,
    colStep
  );
  const meta = matrix.dtype === "fixed64" ? { fixedScale: matrix.fixedScale } : undefined;
  return Matrix.fromHandleWithDType(handle, matrix.dtype, meta);
}export function clip(matrix: Matrix, min: number, max: number): Matrix {
  if (matrix.dtype === "fixed64") {
    throw new Error("clip does not support fixed64 matrices; cast to float64 before clipping");
  }
  const backend = ensureBackend();
  if (!backend.clip) {
    throw new Error("clip is not supported by current backend");
  }
  const result = backend.clip(getHandle(matrix), min, max);
  return Matrix.fromHandleWithDType(result, matrix.dtype);
}


export function compress(mask: Matrix, matrix: Matrix): Matrix {
  const backend = ensureBackend();
  const fn = (backend as any).compress ?? (backend as any).Compress;
  if (typeof fn === "function") {
    const shape = computeBroadcastShape([mask], [matrix], null);
    const maskB = broadcastTo(mask.astype("bool", { copy: false }), shape.rows, shape.cols);
    const dataB = broadcastTo(matrix, shape.rows, shape.cols);
    const handle = fn(getHandle(maskB), getHandle(dataB));
    const meta = matrix.dtype === "fixed64" ? { fixedScale: matrix.fixedScale } : undefined;
    return Matrix.fromHandleWithDType(handle, matrix.dtype, meta);
  }
  const shape = computeBroadcastShape([mask], [matrix], null);
  const maskB = broadcastTo(mask.astype("bool", { copy: false }), shape.rows, shape.cols);
  const dataB = broadcastTo(matrix, shape.rows, shape.cols);
  const maskArr = maskB.toArray() as ArrayLike<boolean | number>;
  const vals = dataB.toArray() as any;
  let count = 0;
  for (let i = 0; i < maskArr.length; i++) {
    if (maskArr[i] as any) count++;
  }
  const dtype = matrix.dtype;
  if (dtype === "fixed64") {
    const out = new BigInt64Array(count);
    let w = 0;
    for (let i = 0; i < maskArr.length; i++) {
      if (maskArr[i] as any) out[w++] = (vals as BigInt64Array | bigint[])[i] as bigint;
    }
    const scale = matrix.fixedScale ?? 0;
    return Matrix.fromFixed(out, count, 1, scale);
  }
  const out = allocateArrayForDType(dtype, count) as any;
  let w = 0;
  for (let i = 0; i < maskArr.length; i++) {
    if (maskArr[i] as any) out[w++] = (vals as any)[i];
  }
  return new Matrix(out, count, 1, { dtype });
}export function where(
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
    broadcastMatrixInJs(cond.astype("bool", { copy: false }), targetShape.rows, targetShape.cols)
  );
  const broadcastChoices = choiceList.map((choice) =>
    broadcastMatrixInJs(choice, targetShape.rows, targetShape.cols)
  );
  const castChoices = castAllToDType(broadcastChoices, dtype);
  const defaultCast = defaultMatrix
    ? castToDType(
        broadcastMatrixInJs(defaultMatrix, targetShape.rows, targetShape.cols),
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

export function transpose(matrix: Matrix): Matrix {
  const backend = ensureBackend();
  const fn =
    backend.transpose ??
    (backend as { transpose_matrix?: (m: BackendMatrixHandle) => BackendMatrixHandle }).transpose_matrix ??
    (backend as { transposeMatrix?: (m: BackendMatrixHandle) => BackendMatrixHandle }).transposeMatrix;
  if (typeof fn === "function") {
    const result = fn(getHandle(matrix));
    return Matrix.fromHandleWithDType(result, matrix.dtype, {
      fixedScale: matrix.fixedScale,
    });
  }
  return transposeMatrixInJs(matrix);
}

export function broadcastTo(matrix: Matrix, rows: number, cols: number): Matrix {
  const backend = ensureBackend();
  const fn =
    backend.broadcast_to ??
    backend.broadcastTo ??
    (backend as { broadcast?: (m: BackendMatrixHandle, r: number, c: number) => BackendMatrixHandle }).broadcast;
  if (typeof fn === "function") {
    const result = fn(getHandle(matrix), rows, cols);
    return Matrix.fromHandleWithDType(result, matrix.dtype, {
      fixedScale: matrix.fixedScale,
    });
  }
  return broadcastMatrixInJs(matrix, rows, cols);
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

export const INT64_MIN = -9223372036854775808n;
export const INT64_MAX = 9223372036854775807n;
export const UINT64_MAX = 0xffffffffffffffffn;

export type ReduceOptions = {
  dtype?: DType;
};

export type GpuReduceOptions = ReduceOptions & {
  mode?: GpuExecutionMode;
};

function unaryFloatOpInJs(
  matrix: Matrix,
  op: (value: number) => number,
  preferFloat32: boolean
): Matrix {
  const values = matrix.astype("float64", { copy: false }).toArray() as Float64Array;
  const out = new Float64Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    out[index] = op(values[index]);
  }
  const base = new Matrix(out, matrix.rows, matrix.cols, { dtype: "float64" });
  return preferFloat32 ? base.astype("float32", { copy: false }) : base;
}

function sigmoidStable(value: number): number {
  if (value >= 0) {
    const z = Math.exp(-value);
    return 1 / (1 + z);
  }
  const z = Math.exp(value);
  return z / (1 + z);
}

function nansumInJs(matrix: Matrix, options: ReduceOptions): Matrix {
  const preferFloat32 =
    matrix.dtype === "float32" && (!options.dtype || options.dtype === "float32");
  if (preferFloat32) {
    const data = matrix.astype("float32", { copy: false }).toArray() as Float32Array;
    let total = 0;
    for (let index = 0; index < data.length; index += 1) {
      const value = data[index];
      if (!Number.isNaN(value)) {
        total += value;
      }
    }
    const scalar = createScalarMatrix(Math.fround(total), "float32");
    return castReduceResult(scalar, options.dtype);
  }
  const data = matrix.astype("float64", { copy: false }).toArray() as Float64Array;
  let total = 0;
  for (let index = 0; index < data.length; index += 1) {
    const value = data[index];
    if (!Number.isNaN(value)) {
      total += value;
    }
  }
  const scalar = createScalarMatrix(total, "float64");
  return castReduceResult(scalar, options.dtype);
}

function nanmeanInJs(matrix: Matrix, options: ReduceOptions): Matrix {
  const preferFloat32 =
    matrix.dtype === "float32" && (!options.dtype || options.dtype === "float32");
  if (preferFloat32) {
    const data = matrix.astype("float32", { copy: false }).toArray() as Float32Array;
    let total = 0;
    let count = 0;
    for (let index = 0; index < data.length; index += 1) {
      const value = data[index];
      if (!Number.isNaN(value)) {
        total += value;
        count += 1;
      }
    }
    const mean = count === 0 ? Number.NaN : total / count;
    const scalar = createScalarMatrix(Math.fround(mean), "float32");
    return castReduceResult(scalar, options.dtype);
  }
  const data = matrix.astype("float64", { copy: false }).toArray() as Float64Array;
  let total = 0;
  let count = 0;
  for (let index = 0; index < data.length; index += 1) {
    const value = data[index];
    if (!Number.isNaN(value)) {
      total += value;
      count += 1;
    }
  }
  const mean = count === 0 ? Number.NaN : total / count;
  const scalar = createScalarMatrix(mean, "float64");
  return castReduceResult(scalar, options.dtype);
}

function computeLinearQuantileArray(values: Float64Array, q: number): number {
  if (values.length === 0) {
    return Number.NaN;
  }
  const sorted = Array.from(values);
  if (sorted.some((value) => Number.isNaN(value))) {
    return Number.NaN;
  }
  sorted.sort((a, b) => a - b);
  if (sorted.length === 1) {
    return sorted[0];
  }
  const pos = q * (sorted.length - 1);
  const lower = Math.floor(pos);
  const upper = Math.ceil(pos);
  if (lower === upper) {
    return sorted[lower];
  }
  const fraction = pos - lower;
  return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
}

function quantileInJs(matrix: Matrix, q: number, options: ReduceOptions): Matrix {
  const values = matrix.astype("float64", { copy: false }).toArray() as Float64Array;
  const value = computeLinearQuantileArray(values, q);
  const scalar = createScalarMatrix(value, "float64");
  return castReduceResult(scalar, options.dtype);
}

function medianInJs(matrix: Matrix, options: ReduceOptions): Matrix {
  return quantileInJs(matrix, 0.5, options);
}

function percentileInJs(matrix: Matrix, p: number, options: ReduceOptions): Matrix {
  const q = p / 100;
  return quantileInJs(matrix, q, options);
}

function reductionAccumulatorDType(dtype: DType): DType {
  switch (dtype) {
    case "bool":
    case "int8":
    case "int16":
    case "int32":
    case "int64":
      return "int64";
    case "uint8":
    case "uint16":
    case "uint32":
    case "uint64":
      return "uint64";
    case "float32":
      return "float32";
    case "float64":
      return "float64";
    case "fixed64":
      return "fixed64";
    default:
      return dtype;
  }
}

function ensureInt64Range(value: bigint, context: string): void {
  if (value < INT64_MIN || value > INT64_MAX) {
    throw new Error(`${context}: signed accumulator overflow`);
  }
}

function ensureUint64Range(value: bigint, context: string): void {
  if (value < 0n || value > UINT64_MAX) {
    throw new Error(`${context}: unsigned accumulator overflow`);
  }
}

function createScalarMatrix(
  value: number | bigint,
  dtype: DType,
  metadata?: { fixedScale?: number | null }
): Matrix {
  if (dtype === "fixed64") {
    const scale = metadata?.fixedScale ?? 0;
    const data = new BigInt64Array([BigInt(value)]);
    return Matrix.fromFixed(data, 1, 1, scale);
  }
  const data = allocateArrayForDType(dtype, 1);
  writeMatrixValue(data, 0, value, dtype);
  return new Matrix(data, 1, 1, { dtype });
}

function castReduceResult(result: Matrix, target?: DType): Matrix {
  if (!target || target === result.dtype) {
    return result;
  }
  return result.astype(target, { copy: false });
}

function sumInJs(matrix: Matrix, options: ReduceOptions): Matrix {
  const accumulatorDType = reductionAccumulatorDType(matrix.dtype);
  const working =
    matrix.dtype === accumulatorDType
      ? matrix
      : matrix.astype(accumulatorDType, { copy: true });
  const array = working.toArray();
  switch (accumulatorDType) {
    case "int64": {
      const data = array as BigInt64Array;
      let total = 0n;
      for (let index = 0; index < data.length; index += 1) {
        total += data[index];
      }
      ensureInt64Range(total, "sum");
      const scalar = createScalarMatrix(total, "int64");
      return castReduceResult(scalar, options.dtype);
    }
    case "uint64": {
      const data = array as BigUint64Array;
      let total = 0n;
      for (let index = 0; index < data.length; index += 1) {
        total += data[index];
      }
      ensureUint64Range(total, "sum");
      const scalar = createScalarMatrix(total, "uint64");
      return castReduceResult(scalar, options.dtype);
    }
    case "float32": {
      const data = array as Float32Array;
      let sum = 0;
      let compensation = 0;
      for (let i = 0; i < data.length; i += 1) {
        const y = data[i] - compensation;
        const t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
      }
      const scalar = createScalarMatrix(Math.fround(sum), "float32");
      return castReduceResult(scalar, options.dtype);
    }
    case "float64": {
      const data = array as Float64Array;
      let sum = 0;
      let compensation = 0;
      for (let i = 0; i < data.length; i += 1) {
        const y = data[i] - compensation;
        const t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
      }
      const scalar = createScalarMatrix(sum, "float64");
      return castReduceResult(scalar, options.dtype);
    }
    case "fixed64": {
      const data = array as BigInt64Array;
      let total = 0n;
      for (let index = 0; index < data.length; index += 1) {
        total += data[index];
      }
      ensureInt64Range(total, "sum(Fixed64)");
      const scale = working.fixedScale ?? matrix.fixedScale ?? 0;
      const scalar = Matrix.fromFixed(new BigInt64Array([total]), 1, 1, scale);
      return castReduceResult(scalar, options.dtype);
    }
    default:
      throw new Error(`sum: unsupported accumulator dtype ${accumulatorDType}`);
  }
}

function dotSignedArray(left: ArrayLike<number>, right: ArrayLike<number>): bigint {
  let total = 0n;
  for (let index = 0; index < left.length; index += 1) {
    total += BigInt(left[index]) * BigInt(right[index]);
  }
  return total;
}

function dotUnsignedArray(left: ArrayLike<number>, right: ArrayLike<number>): bigint {
  let total = 0n;
  for (let index = 0; index < left.length; index += 1) {
    total += BigInt(left[index]) * BigInt(right[index]);
  }
  return total;
}

function dotBigIntArray(left: BigInt64Array | BigUint64Array, right: BigInt64Array | BigUint64Array): bigint {
  let total = 0n;
  for (let index = 0; index < left.length; index += 1) {
    total += left[index] * right[index];
  }
  return total;
}

function dotInJs(a: Matrix, b: Matrix, options: ReduceOptions): Matrix {
  if (a.rows !== b.rows || a.cols !== b.cols) {
    throw new Error("dot: shape mismatch");
  }
  if (a.dtype === "fixed64" || b.dtype === "fixed64") {
    throw new Error("dot(Fixed64): convert operands to float64 before reducing");
  }
  const mulDType = promoteBinaryDType(a.dtype, b.dtype);
  const left = a.dtype === mulDType ? a : a.astype(mulDType, { copy: true });
  const right = b.dtype === mulDType ? b : b.astype(mulDType, { copy: true });
  const accumulatorDType = reductionAccumulatorDType(mulDType);
  const leftArray = left.toArray();
  const rightArray = right.toArray();

  switch (mulDType) {
    case "bool": {
      const lhs = leftArray as Uint8Array;
      const rhs = rightArray as Uint8Array;
      let total = 0n;
      for (let index = 0; index < lhs.length; index += 1) {
        if (lhs[index] !== 0 && rhs[index] !== 0) {
          total += 1n;
        }
      }
      ensureInt64Range(total, "dot(bool, bool)");
      const scalar = createScalarMatrix(total, "int64");
      return castReduceResult(scalar, options.dtype);
    }
    case "int8":
    case "int16":
    case "int32": {
      const lhs = leftArray as Int16Array | Int32Array | Int8Array;
      const rhs = rightArray as Int16Array | Int32Array | Int8Array;
      const total = dotSignedArray(lhs, rhs);
      ensureInt64Range(total, "dot(signed)");
      const scalar = createScalarMatrix(total, "int64");
      return castReduceResult(scalar, options.dtype);
    }
    case "int64": {
      const lhs = leftArray as BigInt64Array;
      const rhs = rightArray as BigInt64Array;
      const total = dotBigIntArray(lhs, rhs);
      ensureInt64Range(total, "dot(int64)");
      const scalar = createScalarMatrix(total, "int64");
      return castReduceResult(scalar, options.dtype);
    }
    case "uint8":
    case "uint16":
    case "uint32": {
      const lhs = leftArray as Uint16Array | Uint32Array | Uint8Array;
      const rhs = rightArray as Uint16Array | Uint32Array | Uint8Array;
      const total = dotUnsignedArray(lhs, rhs);
      ensureUint64Range(total, "dot(unsigned)");
      const scalar = createScalarMatrix(total, "uint64");
      return castReduceResult(scalar, options.dtype);
    }
    case "uint64": {
      const lhs = leftArray as BigUint64Array;
      const rhs = rightArray as BigUint64Array;
      const total = dotBigIntArray(lhs, rhs);
      ensureUint64Range(total, "dot(uint64)");
      const scalar = createScalarMatrix(total, "uint64");
      return castReduceResult(scalar, options.dtype);
    }
    case "float32": {
      const lhs = leftArray as Float32Array;
      const rhs = rightArray as Float32Array;
      let sum = 0;
      let compensation = 0;
      for (let index = 0; index < lhs.length; index += 1) {
        const product = lhs[index] * rhs[index];
        const y = product - compensation;
        const t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
      }
      const scalar = createScalarMatrix(Math.fround(sum), "float32");
      return castReduceResult(scalar, options.dtype);
    }
    case "float64": {
      const lhs = leftArray as Float64Array;
      const rhs = rightArray as Float64Array;
      let sum = 0;
      let compensation = 0;
      for (let index = 0; index < lhs.length; index += 1) {
        const product = lhs[index] * rhs[index];
        const y = product - compensation;
        const t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
      }
      const scalar = createScalarMatrix(sum, "float64");
      return castReduceResult(scalar, options.dtype);
    }
    case "fixed64":
      throw new Error("dot(Fixed64): convert operands to float64 before reducing");
    default:
      throw new Error(`dot: unsupported multiplicative dtype ${mulDType}`);
  }
}

export function sum(matrix: Matrix, options: ReduceOptions = {}): Matrix {
  const backend = ensureBackend();
  const target = options.dtype ?? null;
  if (backend.sum) {
    const handle = backend.sum(getHandle(matrix), target);
    const dtype = getMatrixDTypeFromHandle(handle);
    const fixedScale = dtype === "fixed64" ? getMatrixFixedScaleFromHandle(handle) : undefined;
    return Matrix.fromHandleWithDType(handle, dtype, { fixedScale });
  }
  return sumInJs(matrix, options);
}

export async function sumAsync(
  matrix: Matrix,
  options: GpuReduceOptions = {}
): Promise<Matrix> {
  const { mode = "auto", dtype } = options;
  if (isNode) {
    const backend = ensureBackend();
    const nativeGpuSum =
      (backend as Record<string, unknown>).gpu_sum ?? (backend as Record<string, unknown>).gpuSum;
    if (typeof nativeGpuSum === "function") {
      try {
        const handle = (nativeGpuSum as (
          source: BackendMatrixHandle,
          dtype: string | null
        ) => BackendMatrixHandle)(getHandle(matrix), dtype ?? null);
        const resolvedDType = getMatrixDTypeFromHandle(handle) ?? (dtype ?? "float32");
        const metadata =
          resolvedDType === "fixed64"
            ? { fixedScale: getMatrixFixedScaleFromHandle(handle) ?? null }
            : undefined;
        activeGpuKind = "cuda";
        return Matrix.fromHandleWithDType(handle, resolvedDType, metadata);
      } catch (error) {
        console.warn("[numjs] N-API gpu_sum failed; falling back to other accelerators.", error);
      }
    }
  }
  const engine = await selectGpuEngine(mode);
  if (!engine) {
    const reduceOptions: ReduceOptions = {};
    if (dtype) {
      reduceOptions.dtype = dtype;
    }
    return Promise.resolve(sum(matrix, reduceOptions));
  }
  if (matrix.dtype === "fixed64") {
    throw new Error("sumAsync does not support fixed64 matrices; cast to float32 first");
  }
  const view = toFloat32View(matrix);
  const total = await engine.reduceSum({ data: view.array });
  const result = new Matrix(new Float32Array([total]), 1, 1, { dtype: "float32" });
  if (dtype && dtype !== "float32") {
    return result.astype(dtype, { copy: false });
  }
  return result;
}

export function nansum(matrix: Matrix, options: ReduceOptions = {}): Matrix {
  const backend = ensureBackend();
  const target = options.dtype ?? null;
  if (backend.nansum) {
    const handle = backend.nansum(getHandle(matrix), target);
    const dtype = getMatrixDTypeFromHandle(handle);
    const fixedScale = dtype === "fixed64" ? getMatrixFixedScaleFromHandle(handle) : undefined;
    return Matrix.fromHandleWithDType(handle, dtype, { fixedScale });
  }
  return nansumInJs(matrix, options);
}

export function nanmean(matrix: Matrix, options: ReduceOptions = {}): Matrix {
  const backend = ensureBackend();
  const target = options.dtype ?? null;
  if (backend.nanmean) {
    const handle = backend.nanmean(getHandle(matrix), target);
    const dtype = getMatrixDTypeFromHandle(handle);
    return Matrix.fromHandleWithDType(handle, dtype);
  }
  return nanmeanInJs(matrix, options);
}

export function median(matrix: Matrix, options: ReduceOptions = {}): Matrix {
  const backend = ensureBackend();
  const target = options.dtype ?? null;
  if (backend.median) {
    const handle = backend.median(getHandle(matrix), target);
    const dtype = getMatrixDTypeFromHandle(handle);
    return Matrix.fromHandleWithDType(handle, dtype);
  }
  return medianInJs(matrix, options);
}

export function quantile(
  matrix: Matrix,
  q: number,
  options: ReduceOptions = {}
): Matrix {
  if (Number.isNaN(q) || q < 0 || q > 1) {
    throw new Error("quantile: q must be within [0, 1]");
  }
  const backend = ensureBackend();
  const target = options.dtype ?? null;
  if (backend.quantile) {
    const handle = backend.quantile(getHandle(matrix), q, target);
    const dtype = getMatrixDTypeFromHandle(handle);
    return Matrix.fromHandleWithDType(handle, dtype);
  }
  return quantileInJs(matrix, q, options);
}

export function percentile(
  matrix: Matrix,
  p: number,
  options: ReduceOptions = {}
): Matrix {
  if (Number.isNaN(p) || p < 0 || p > 100) {
    throw new Error("percentile: p must be within [0, 100]");
  }
  const backend = ensureBackend();
  const target = options.dtype ?? null;
  if (backend.percentile) {
    const handle = backend.percentile(getHandle(matrix), p, target);
    const dtype = getMatrixDTypeFromHandle(handle);
    return Matrix.fromHandleWithDType(handle, dtype);
  }
  return percentileInJs(matrix, p, options);
}

export function dot(a: Matrix, b: Matrix, options: ReduceOptions = {}): Matrix {
  if (a.rows !== b.rows || a.cols !== b.cols) {
    throw new Error("dot: shape mismatch");
  }
  const backend = ensureBackend();
  const target = options.dtype ?? null;
  if (backend.dot) {
    const handle = backend.dot(getHandle(a), getHandle(b), target);
    const dtype = getMatrixDTypeFromHandle(handle);
    const fixedScale = dtype === "fixed64" ? getMatrixFixedScaleFromHandle(handle) : undefined;
    return Matrix.fromHandleWithDType(handle, dtype, { fixedScale });
  }
  return dotInJs(a, b, options);
}

export function sumUnsafe(matrix: Matrix): number {
  const arr = matrix.astype("float64", { copy: false }).toArray() as Float64Array;
  let total = 0;
  for (let i = 0; i < arr.length; i += 1) {
    total += arr[i];
  }
  return total;
}

export function dotUnsafe(a: Matrix, b: Matrix): number {
  if (a.rows !== b.rows || a.cols !== b.cols) {
    throw new Error("dotUnsafe: shape mismatch");
  }
  const av = a.astype("float64", { copy: false }).toArray() as Float64Array;
  const bv = b.astype("float64", { copy: false }).toArray() as Float64Array;
  let total = 0;
  for (let i = 0; i < av.length; i += 1) {
    total += av[i] * bv[i];
  }
  return total;
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
  const dtype = getMatrixDTypeFromHandle(handle);
  const fixedScale =
    dtype === "fixed64" ? getMatrixFixedScaleFromHandle(handle) : null;
  return Matrix.fromHandleWithDType(handle, dtype, { fixedScale });
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
  if ((globalThis as any).__numjsCachedRequire) {
    return (globalThis as any).__numjsCachedRequire;
  }
  try {
    const nodeRequire = Function("return typeof require !== 'undefined' ? require : undefined;")();
    if (typeof nodeRequire === "function") {
      (globalThis as any).__numjsCachedRequire = nodeRequire;
      return nodeRequire;
    }
  } catch {
    // ignore environments that disallow Function evaluation
  }
  const { createRequire } = await import("node:module");
  const moduleFile = getCurrentModuleFile();
  if (!moduleFile) {
    throw new Error("Unable to determine current module path");
  }
  if (moduleFile.startsWith("file://")) {
    const req = createRequire(moduleFile);
    (globalThis as any).__numjsCachedRequire = req;
    return req;
  }
  const { pathToFileURL } = await import("node:url");
  const req = createRequire(pathToFileURL(moduleFile).href);
  (globalThis as any).__numjsCachedRequire = req;
  return req;
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
  {
    rtol = DEFAULT_RTOL,
    atol = DEFAULT_ATOL,
    equalNaN = false,
  }: { rtol?: number; atol?: number; equalNaN?: boolean } = {}
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
  {
    rtol = DEFAULT_RTOL,
    atol = DEFAULT_ATOL,
    equalNaN = false,
  }: { rtol?: number; atol?: number; equalNaN?: boolean } = {}
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
  candidates.add(NAPI_ENTRY);
  if (distribution) {
    for (const pkg of distribution.packages) {
      candidates.add(pkg);
    }
    for (const suffix of distribution.binaries) {
      candidates.add(`${NAPI_BINDING_PREFIX}.${suffix}.node`);
    }
  }
  return Array.from(candidates);
}

function isNapiBackendSufficient(candidate: BackendModule): boolean {
  const matrixExport = (candidate as { Matrix?: unknown }).Matrix;
  if (typeof matrixExport !== "function") {
    return false;
  }
  const proto = (matrixExport as { prototype?: unknown }).prototype as Record<string, unknown> | undefined;
  if (!proto || typeof proto.astype !== "function") {
    return false;
  }
  const ctorWithStatic = matrixExport as {
    from_fixed_i64?: (...args: unknown[]) => unknown;
    fromFixedI64?: (...args: unknown[]) => unknown;
  };
  const hasFixed64Factory =
    typeof ctorWithStatic.from_fixed_i64 === "function" ||
    typeof ctorWithStatic.fromFixedI64 === "function";
  if (!hasFixed64Factory) {
    return false;
  }
  const requiredOps: Array<keyof BackendModule> = ["sub", "mul", "div", "neg", "sum", "dot"];
  for (const op of requiredOps) {
    if (typeof (candidate as Record<string, unknown>)[op] !== "function") {
      return false;
    }
  }
  if (typeof (candidate as { gather_pairs?: unknown }).gather_pairs !== "function") {
    return false;
  }
  if (typeof (candidate as { scatter_pairs?: unknown }).scatter_pairs !== "function") {
    return false;
  }
  return true;
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

const DATAFRAME_DEFAULT_DTYPE: DType = "float64";
const CSV_DEFAULT_DELIMITER = ",";

export class DataFrameView {
  private readonly _matrix: Matrix;
  private readonly _columnNames: string[];
  private readonly _columnDTypes: Record<string, DType>;

  constructor(matrix: Matrix, columnNames?: readonly string[], columnDTypes?: Record<string, DType>) {
    const names =
      columnNames && columnNames.length === matrix.cols
        ? Array.from(columnNames)
        : generateDefaultColumnNames(matrix.cols);
    if (names.length !== matrix.cols) {
      throw new Error(
        `DataFrameView: expected ${matrix.cols} column names, received ${names.length}`
      );
    }
    this._matrix = matrix.dtype === "float64" ? matrix : matrix.astype("float64", { copy: true });
    this._columnNames = names;
    const dtypes: Record<string, DType> = {};
    for (const name of names) {
      dtypes[name] = columnDTypes?.[name] ?? DATAFRAME_DEFAULT_DTYPE;
    }
    this._columnDTypes = dtypes;
  }

  static fromMatrix(matrix: Matrix, options: DataFrameInitOptions = {}): DataFrameView {
    const view = new DataFrameView(matrix, options.columns, options.columnDTypes);
    if (options.inferColumnDTypes) {
      const inferred: Record<string, DType> = {};
      for (let index = 0; index < view._columnNames.length; index += 1) {
        const name = view._columnNames[index];
        inferred[name] = matrix.column(index).dtype;
      }
      return view.withColumnDTypes(inferred);
    }
    return view;
  }

  get rowCount(): number {
    return this._matrix.rows;
  }

  get columnCount(): number {
    return this._matrix.cols;
  }

  get columnNames(): string[] {
    return Array.from(this._columnNames);
  }

  get columnDTypes(): Record<string, DType> {
    return { ...this._columnDTypes };
  }

  toMatrix(copy = false): Matrix {
    return copy ? this._matrix.astype("float64", { copy: true }) : this._matrix;
  }

  column(name: string, dtype?: DType): Matrix {
    const index = this.columnIndex(name);
    const target = dtype ?? this._columnDTypes[name] ?? DATAFRAME_DEFAULT_DTYPE;
    let column = this._matrix.column(index);
    if (column.dtype !== target) {
      column = column.astype(target, { copy: true });
    }
    return column;
  }

  select(columnNames: readonly string[]): DataFrameView {
    if (columnNames.length === 0) {
      throw new Error("DataFrameView.select requires at least one column");
    }
    const matrices: Matrix[] = columnNames.map((name) =>
      this._matrix.column(this.columnIndex(name)).astype("float64", { copy: true })
    );
    const matrix = combineColumnsAsMatrix(matrices);
    const dtypes: Record<string, DType> = {};
    for (const name of columnNames) {
      dtypes[name] = this._columnDTypes[name] ?? DATAFRAME_DEFAULT_DTYPE;
    }
    return new DataFrameView(matrix, columnNames, dtypes);
  }

  withColumnDTypes(overrides: Record<string, DType>): DataFrameView {
    const merged = { ...this._columnDTypes };
    for (const [name, dtype] of Object.entries(overrides)) {
      if (!this._columnNames.includes(name)) {
        throw new Error(`DataFrameView: unknown column "${name}"`);
      }
      merged[name] = dtype;
    }
    return new DataFrameView(this._matrix, this._columnNames, merged);
  }

  renameColumns(mapping: Record<string, string>): DataFrameView {
    const renamed = this._columnNames.map((name) => mapping[name] ?? name);
    const dtypes: Record<string, DType> = {};
    for (let index = 0; index < renamed.length; index += 1) {
      dtypes[renamed[index]] = this._columnDTypes[this._columnNames[index]] ?? DATAFRAME_DEFAULT_DTYPE;
    }
    return new DataFrameView(this._matrix, renamed, dtypes);
  }

  toColumnArrays(): Record<string, Array<number | boolean>> {
    const result: Record<string, Array<number | boolean>> = {};
    for (const name of this._columnNames) {
      const dtype = this._columnDTypes[name] ?? DATAFRAME_DEFAULT_DTYPE;
      const column = this.column(name, dtype);
      const numeric = Array.from(column.toArray() as ArrayLike<number>);
      if (dtype === "bool") {
        result[name] = numeric.map((value) => Boolean(value));
      } else {
        result[name] = numeric;
      }
    }
    return result;
  }

  toObjectRows(limit?: number): Array<Record<string, number | boolean>> {
    const rows: Array<Record<string, number | boolean>> = [];
    const columnData = this.toColumnArrays();
    const max = typeof limit === "number" ? Math.min(limit, this.rowCount) : this.rowCount;
    for (let row = 0; row < max; row += 1) {
      const record: Record<string, number | boolean> = {};
      for (const name of this._columnNames) {
        record[name] = columnData[name][row];
      }
      rows.push(record);
    }
    return rows;
  }

  private columnIndex(name: string): number {
    const index = this._columnNames.indexOf(name);
    if (index === -1) {
      throw new Error(`DataFrameView: unknown column "${name}"`);
    }
    return index;
  }
}

function combineColumnsAsMatrix(columns: Matrix[]): Matrix {
  if (columns.length === 0) {
    throw new Error("combineColumnsAsMatrix requires at least one column matrix");
  }
  let combined = columns[0].astype("float64", { copy: true });
  for (let index = 1; index < columns.length; index += 1) {
    const next = columns[index].astype("float64", { copy: true });
    combined = concat(combined, next, 1);
  }
  return combined;
}

function generateDefaultColumnNames(count: number): string[] {
  const result: string[] = [];
  for (let index = 0; index < count; index += 1) {
    result.push(`col${index}`);
  }
  return result;
}

function parseCsvString(content: string, delimiter: string, skipEmptyLines: boolean): string[][] {
  const rows: string[][] = [];
  let field = "";
  let row: string[] = [];
  let inQuotes = false;
  for (let i = 0; i < content.length; i += 1) {
    const ch = content[i];
    if (ch === '"') {
      if (inQuotes && content[i + 1] === '"') {
        field += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }
    if (ch === delimiter && !inQuotes) {
      row.push(field);
      field = "";
      continue;
    }
    if ((ch === "\n" || ch === "\r") && !inQuotes) {
      if (ch === "\r" && content[i + 1] === "\n") {
        i += 1;
      }
      row.push(field);
      field = "";
      if (!(skipEmptyLines && row.every((cell) => cell.trim() === ""))) {
        rows.push(row);
      }
      row = [];
      continue;
    }
    field += ch;
  }
  row.push(field);
  if (!(skipEmptyLines && row.every((cell) => cell.trim() === ""))) {
    if (!(row.length === 1 && row[0] === "")) {
      rows.push(row);
    }
  }
  return rows;
}

function buildDataFrameFromCsvString(content: string, options: CsvReadOptions = {}): DataFrameView {
  const delimiter = options.delimiter ?? CSV_DEFAULT_DELIMITER;
  const rows = parseCsvString(content, delimiter, options.skipEmptyLines !== false);
  if (rows.length === 0) {
    throw new Error("readCsvDataFrame: CSV source is empty");
  }
  let headerRow: string[] | null = null;
  let bodyRows: string[][] = rows;
  if (options.hasHeader !== false) {
    headerRow = rows[0];
    bodyRows = rows.slice(1);
    if (!headerRow || headerRow.length === 0) {
      throw new Error("readCsvDataFrame: header row is empty");
    }
  }
  let columnNames: string[];
  if (options.hasHeader === false) {
    columnNames =
      options.columns && options.columns.length > 0
        ? Array.from(options.columns)
        : generateDefaultColumnNames(rows[0].length);
  } else if (options.columns && options.columns.length > 0) {
    if (!headerRow) {
      throw new Error("readCsvDataFrame: header row required when selecting columns");
    }
    const indices = options.columns.map((name) => {
      const index = headerRow!.indexOf(name);
      if (index === -1) {
        throw new Error(`readCsvDataFrame: column "${name}" not found in header`);
      }
      return index;
    });
    columnNames = Array.from(options.columns);
    bodyRows = bodyRows.map((row) => indices.map((idx) => row[idx] ?? ""));
  } else {
    columnNames = headerRow ?? generateDefaultColumnNames(rows[0].length);
  }
  if (columnNames.length === 0) {
    columnNames = generateDefaultColumnNames(rows[0].length);
  }
  const columnCount = columnNames.length;
  for (const row of bodyRows) {
    if (row.length !== columnCount) {
      throw new Error(
        `readCsvDataFrame: row length (${row.length}) does not match column count (${columnCount})`
      );
    }
  }
  const columnDTypes: Record<string, DType> = {};
  const columnValues: Array<string[]> = columnNames.map((_, index) => bodyRows.map((row) => row[index] ?? ""));
  for (let index = 0; index < columnNames.length; index += 1) {
    const name = columnNames[index];
    const explicit = options.columnTypes?.[name];
    columnDTypes[name] = explicit ?? inferColumnType(columnValues[index]);
  }
  const matrix = constructMatrixFromCsv(columnNames, columnValues, columnDTypes);
  return new DataFrameView(matrix, columnNames, columnDTypes);
}

function constructMatrixFromCsv(
  columnNames: string[],
  columnValues: Array<string[]>,
  columnDTypes: Record<string, DType>
): Matrix {
  const rows = columnValues[0]?.length ?? 0;
  const cols = columnNames.length;
  const buffer = new Float64Array(rows * cols);
  for (let col = 0; col < cols; col += 1) {
    const name = columnNames[col];
    const dtype = columnDTypes[name] ?? DATAFRAME_DEFAULT_DTYPE;
    const values = columnValues[col];
    for (let row = 0; row < rows; row += 1) {
      buffer[row * cols + col] = coerceValueToNumber(values[row] ?? "", dtype);
    }
  }
  return new Matrix(buffer, rows, cols, { dtype: "float64" });
}

function inferColumnType(values: readonly string[]): DType {
  let boolCandidate = values.length > 0;
  let intCandidate = values.length > 0;
  for (const value of values) {
    const normalized = value.trim().toLowerCase();
    if (normalized === "" || normalized === "nan") {
      boolCandidate = false;
      intCandidate = false;
      continue;
    }
    if (boolCandidate && normalized !== "true" && normalized !== "false" && normalized !== "1" && normalized !== "0") {
      boolCandidate = false;
    }
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || !Number.isInteger(numeric)) {
      intCandidate = false;
    }
    if (!boolCandidate && !intCandidate) {
      break;
    }
  }
  if (boolCandidate) {
    return "bool";
  }
  if (intCandidate) {
    return "int32";
  }
  return "float64";
}

function coerceValueToNumber(value: string, dtype: DType): number {
  const trimmed = value.trim();
  switch (dtype) {
    case "bool": {
      if (/^(true|1)$/i.test(trimmed)) return 1;
      if (/^(false|0)$/i.test(trimmed)) return 0;
      return trimmed.length > 0 ? 1 : 0;
    }
    case "int32": {
      const parsed = Number.parseInt(trimmed, 10);
      return Number.isFinite(parsed) ? parsed : NaN;
    }
    default: {
      const parsed = Number(trimmed);
      return Number.isFinite(parsed) ? parsed : NaN;
    }
  }
}

function ensureNodeEnvironment(feature: string): void {
  if (!isNode) {
    throw new Error(`${feature} is only available in Node.js environments`);
  }
}

async function loadNodeFs() {
  return import("node:fs/promises");
}

export async function readCsvDataFrame(
  path: string,
  options: CsvReadOptions = {}
): Promise<DataFrameView> {
  ensureNodeEnvironment("readCsvDataFrame");
  const fs = await loadNodeFs();
  const encoding = options.encoding ?? "utf8";
  const content = (await fs.readFile(path, { encoding } as any)) as unknown as string;
  return buildDataFrameFromCsvString(content, options);
}

export async function writeDataFrameToCsv(
  frame: DataFrameView,
  path: string,
  options: CsvWriteOptions = {}
): Promise<void> {
  ensureNodeEnvironment("writeDataFrameToCsv");
  const fs = await loadNodeFs();
  const delimiter = options.delimiter ?? CSV_DEFAULT_DELIMITER;
  const includeHeader = options.includeHeader !== false;
  const newline = options.newline ?? "\n";
  const pieces: string[] = [];
  if (includeHeader) {
    pieces.push(frame.columnNames.map((name) => escapeCsvValue(name, delimiter)).join(delimiter));
  }
  const columns = frame.toColumnArrays();
  for (let row = 0; row < frame.rowCount; row += 1) {
    const values = frame.columnNames.map((name) => {
      const dtype = frame.columnDTypes[name] ?? DATAFRAME_DEFAULT_DTYPE;
      const raw = columns[name][row];
      if (dtype === "bool") {
        return escapeCsvValue(String(Boolean(raw)), delimiter);
      }
      return escapeCsvValue(String(raw ?? ""), delimiter);
    });
    pieces.push(values.join(delimiter));
  }
  const payload = pieces.join(newline) + newline;
  await fs.writeFile(path, payload, { encoding: options.encoding ?? "utf8" } as any);
}

export async function readCsvDataFrameFromStream(
  stream: ReadableStream<Uint8Array>,
  options: CsvReadOptions = {}
): Promise<DataFrameView> {
  const decoder = new TextDecoder(options.encoding ?? "utf-8");
  const reader = stream.getReader();
  let content = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    content += decoder.decode(value, { stream: true });
  }
  content += decoder.decode(new Uint8Array(), { stream: false });
  return buildDataFrameFromCsvString(content, options);
}

export async function readParquetDataFrame(
  path: string,
  options: ParquetReadOptions = {}
): Promise<DataFrameView> {
  ensureNodeEnvironment("readParquetDataFrame");
  try {
    const polars = await import("./io/polars");
    const module = await polars.ensurePolarsModule(options.polarsModule);
    if (typeof module.readParquet !== "function") {
      throw new Error("Loaded Polars module does not expose readParquet");
    }
    const dataframe = (await module.readParquet(
      path,
      options.columns ? { columns: options.columns } : undefined
    )) as unknown;
    const { matrix, columnNames } = polars.polarsDataFrameToMatrix(dataframe as any, {
      columns: options.columns,
    });
    return DataFrameView.fromMatrix(matrix, {
      columns: columnNames,
      inferColumnDTypes: true,
    });
  } catch (error) {
    throw new Error(
      'readParquetDataFrame requires the "nodejs-polars" (preferred) or "polars" package. Install it and optionally pass the module via options.polarsModule.',
      { cause: error }
    );
  }
}

export async function writeParquetDataFrame(
  frame: DataFrameView,
  path: string,
  options: ParquetWriteOptions = {}
): Promise<void> {
  ensureNodeEnvironment("writeParquetDataFrame");
  try {
    const polars = await import("./io/polars");
    const module = await polars.ensurePolarsModule(options.polarsModule);
    const df = (await polars.matrixToPolarsDataFrame(frame.toMatrix(true), {
      columnNames: frame.columnNames,
      polarsModule: module,
    })) as { writeParquet?: (path: string, options?: unknown) => Promise<void> };
    if (df && typeof df.writeParquet === "function") {
      await df.writeParquet(path, options);
      return;
    }
    if (typeof module.writeParquet === "function") {
      await module.writeParquet(path, df, options);
      return;
    }
    throw new Error("Loaded Polars module does not expose writeParquet");
  } catch (error) {
    throw new Error(
      'writeParquetDataFrame requires the "nodejs-polars" (preferred) or "polars" package. Install it and optionally pass the module via options.polarsModule.',
      { cause: error }
    );
  }
}

function escapeCsvValue(value: string, delimiter: string): string {
  if (value.includes('"')) {
    value = value.replace(/"/g, '""');
  }
  if (value.includes(delimiter) || /\r|\n/.test(value) || value.includes('"')) {
    return `"${value}"`;
  }
  return value;
}

type BackendFftResult = {
  real: BackendMatrixHandle;
  imag: BackendMatrixHandle;
};

function convertBackendFftResult(result: BackendFftResult): FftResult {
  const real = Matrix.fromHandleWithDType(result.real, "float64");
  const imag = Matrix.fromHandleWithDType(result.imag, "float64");
  return { real, imag };
}

function resolveFftBackendFn(
  backend: BackendModule,
  name: "fft_axis" | "fft2d" | "ifft_axis" | "ifft2d"
): ((...args: unknown[]) => BackendFftResult) | null {
  const candidate = (backend as any)[name] ?? (backend as any)[toCamelCase(name)];
  return typeof candidate === "function" ? (candidate as (...args: unknown[]) => BackendFftResult) : null;
}

function toCamelCase(name: string): string {
  return name.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
}

function ensureFftAxis(axis: number, matrix: Matrix): number {
  if (axis !== 0 && axis !== 1) {
    throw new Error("fftAxis: axis must be 0 (columns) or 1 (rows)");
  }
  if (axis === 0 && matrix.rows < 1) {
    throw new Error("fftAxis: matrix must have at least one row");
  }
  if (axis === 1 && matrix.cols < 1) {
    throw new Error("fftAxis: matrix must have at least one column");
  }
  return axis;
}

export function fftAxis(matrix: Matrix, axis = 1): FftResult {
  const normalizedAxis = ensureFftAxis(axis, matrix);
  const backend = ensureBackend();
  const fn = resolveFftBackendFn(backend, "fft_axis");
  if (!fn) {
    throw new Error("fftAxis: current backend does not expose FFT support");
  }
  const result = fn(getHandle(matrix), normalizedAxis);
  return convertBackendFftResult(result);
}

export function ifftAxis(real: Matrix, imag: Matrix, axis = 1): FftResult {
  const normalizedAxis = ensureFftAxis(axis, real);
  if (real.rows !== imag.rows || real.cols !== imag.cols) {
    throw new Error("ifftAxis: real and imaginary matrices must have identical shapes");
  }
  const backend = ensureBackend();
  const fn = resolveFftBackendFn(backend, "ifft_axis");
  if (!fn) {
    throw new Error("ifftAxis: current backend does not expose inverse FFT support");
  }
  const result = fn(getHandle(real), getHandle(imag), normalizedAxis);
  return convertBackendFftResult(result);
}

export function fft2d(matrix: Matrix): FftResult {
  const backend = ensureBackend();
  const fn = resolveFftBackendFn(backend, "fft2d");
  if (!fn) {
    throw new Error("fft2d: current backend does not expose 2D FFT support");
  }
  const result = fn(getHandle(matrix));
  return convertBackendFftResult(result);
}

export function ifft2d(real: Matrix, imag: Matrix): FftResult {
  if (real.rows !== imag.rows || real.cols !== imag.cols) {
    throw new Error("ifft2d: real and imaginary matrices must have identical shapes");
  }
  const backend = ensureBackend();
  const fn = resolveFftBackendFn(backend, "ifft2d");
  if (!fn) {
    throw new Error("ifft2d: current backend does not expose 2D inverse FFT support");
  }
  const result = fn(getHandle(real), getHandle(imag));
  return convertBackendFftResult(result);
}

export function powerSpectrum(matrix: Matrix, axis = 1): Matrix {
  const { real, imag } = fftAxis(matrix, axis);
  const realArr = real.astype("float64", { copy: false }).toArray() as Float64Array;
  const imagArr = imag.astype("float64", { copy: false }).toArray() as Float64Array;
  const spectrum = new Float64Array(realArr.length);
  for (let i = 0; i < realArr.length; i += 1) {
    spectrum[i] = Math.hypot(realArr[i], imagArr[i]);
  }
  return new Matrix(spectrum, real.rows, real.cols, { dtype: "float64" });
}

export function lazyFromMatrix(matrix: Matrix): LazyArray {
  return matrix.toLazy();
}

export function realizeLazyArray(
  lazy: LazyArray,
  options: MatrixOptions = {}
): Matrix {
  return Matrix.fromLazy(lazy, options);
}

export function lazyScalar(value: number): LazyScalar {
  return LazyScalar.fromNumber(value);
}

export { LazyArray, LazyScalar, lazyConstant };
export {
  arrowTableToMatrix,
  matrixToArrowTable,
  polarsDataFrameToMatrix,
  matrixToPolarsDataFrame,
};
export type {
  ArrowTableLike,
  ArrowToMatrixOptions,
  ArrowConversionResult,
  PolarsToMatrixOptions,
  MatrixToPolarsOptions,
};
