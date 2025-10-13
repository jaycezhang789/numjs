
import { strFromU8, strToU8, unzipSync, zipSync } from "fflate";

type BackendKind = "napi" | "wasm";

type MatrixHandle = {
  rows: number;
  cols: number;
  to_vec(): Float64Array | number[];
};

type BackendModule = {
  Matrix: new (
    data: Float64Array,
    rows: number,
    cols: number
  ) => MatrixHandle;
  add(a: MatrixHandle, b: MatrixHandle): MatrixHandle;
  matmul(a: MatrixHandle, b: MatrixHandle): MatrixHandle;
  clip?(matrix: MatrixHandle, min: number, max: number): MatrixHandle;
  where_select?(
    condition: MatrixHandle,
    truthy: MatrixHandle,
    falsy: MatrixHandle
  ): MatrixHandle;
  concat?(a: MatrixHandle, b: MatrixHandle, axis: number): MatrixHandle;
  stack?(a: MatrixHandle, b: MatrixHandle, axis: number): MatrixHandle;
  svd?(
    matrix: MatrixHandle
  ): {
    u: MatrixHandle;
    s: Float64Array;
    vt: MatrixHandle;
  };
  qr?(
    matrix: MatrixHandle
  ): {
    q: MatrixHandle;
    r: MatrixHandle;
  };
  solve?(a: MatrixHandle, b: MatrixHandle): MatrixHandle;
  eigen?(
    matrix: MatrixHandle
  ): {
    values: Float64Array;
    vectors: MatrixHandle;
  };
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

function wrapMatrix(inner: MatrixHandle): Matrix {
  const matrix = Object.create(Matrix.prototype) as Matrix;
  (matrix as unknown as { inner: MatrixHandle }).inner = inner;
  return matrix;
}

function inner(matrix: Matrix): MatrixHandle {
  return (matrix as unknown as { inner: MatrixHandle }).inner;
}

function expectCapability<K extends keyof BackendModule>(
  backend: BackendModule,
  capability: K
): NonNullable<BackendModule[K]> {
  const impl = backend[capability];
  if (typeof impl !== "function") {
    const kind = activeKind ?? "unknown";
    throw new Error(
      `Backend "${kind}" does not support operation "${String(capability)}"`
    );
  }
  return impl as NonNullable<BackendModule[K]>;
}

function cloneBackendBytes(bytes: Uint8Array): Uint8Array {
  if (isNode && typeof Buffer !== "undefined" && bytes instanceof Buffer) {
    return new Uint8Array(bytes);
  }
  return Uint8Array.from(bytes);
}

function toBackendBytes(data: ArrayBuffer | Uint8Array): Uint8Array {
  return data instanceof Uint8Array ? data : new Uint8Array(data);
}

function normaliseData(data: Float64Array | number[]): Float64Array {
  return data instanceof Float64Array ? data : Float64Array.from(data);
}

function toUint8Array(data: ArrayBuffer | Uint8Array): Uint8Array {
  return data instanceof Uint8Array ? data : new Uint8Array(data);
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
  private inner: MatrixHandle;

  constructor(data: Float64Array | number[], rows: number, cols: number) {
    const backend = ensureBackend();
    this.inner = new backend.Matrix(normaliseData(data), rows, cols);
  }

  get rows(): number {
    return this.inner.rows;
  }

  get cols(): number {
    return this.inner.cols;
  }

  toArray(): Float64Array {
    const raw = this.inner.to_vec();
    return raw instanceof Float64Array
      ? raw
      : Float64Array.from(raw as number[]);
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

  astype(dtype: "float32" | "float64" = "float64"): Float32Array | Float64Array {
    const array = this.toArray();
    return dtype === "float32" ? new Float32Array(array) : new Float64Array(array);
  }
}

export function add(a: Matrix, b: Matrix): Matrix {
  const backend = ensureBackend();
  const result = backend.add(inner(a), inner(b));
  return wrapMatrix(result);
}

export function matmul(a: Matrix, b: Matrix): Matrix {
  const backend = ensureBackend();
  const result = backend.matmul(inner(a), inner(b));
  return wrapMatrix(result);
}

export function clip(matrix: Matrix, min: number, max: number): Matrix {
  const backend = ensureBackend();
  const impl = expectCapability(backend, "clip");
  return wrapMatrix(impl(inner(matrix), min, max));
}

export function where(
  condition: Matrix,
  truthy: Matrix,
  falsy: Matrix
): Matrix {
  const backend = ensureBackend();
  const impl = expectCapability(backend, "where_select");
  return wrapMatrix(impl(inner(condition), inner(truthy), inner(falsy)));
}

export function concat(a: Matrix, b: Matrix, axis = 0): Matrix {
  const backend = ensureBackend();
  const impl = expectCapability(backend, "concat");
  return wrapMatrix(impl(inner(a), inner(b), axis));
}

export function stack(a: Matrix, b: Matrix, axis = 0): Matrix {
  const backend = ensureBackend();
  const impl = expectCapability(backend, "stack");
  return wrapMatrix(impl(inner(a), inner(b), axis));
}

export function svd(matrix: Matrix): {
  u: Matrix;
  s: Float64Array;
  vt: Matrix;
} {
  const backend = ensureBackend();
  const impl = expectCapability(backend, "svd");
  const result = impl(inner(matrix));
  return {
    u: wrapMatrix(result.u),
    s:
      result.s instanceof Float64Array
        ? result.s
        : Float64Array.from(result.s as unknown as Iterable<number>),
    vt: wrapMatrix(result.vt),
  };
}

export function qr(matrix: Matrix): { q: Matrix; r: Matrix } {
  const backend = ensureBackend();
  const impl = expectCapability(backend, "qr");
  const result = impl(inner(matrix));
  return { q: wrapMatrix(result.q), r: wrapMatrix(result.r) };
}

export function solve(a: Matrix, b: Matrix): Matrix {
  const backend = ensureBackend();
  const impl = expectCapability(backend, "solve");
  return wrapMatrix(impl(inner(a), inner(b)));
}

export function eigen(matrix: Matrix): {
  values: Float64Array;
  vectors: Matrix;
} {
  const backend = ensureBackend();
  const impl = expectCapability(backend, "eigen");
  const result = impl(inner(matrix));
  return {
    values:
      result.values instanceof Float64Array
        ? result.values
        : Float64Array.from(result.values as unknown as Iterable<number>),
    vectors: wrapMatrix(result.vectors),
  };
}

export function readNpy(data: ArrayBuffer | Uint8Array): Matrix {
  const bytes = toUint8Array(data);
  const { shape, values } = parseNpy(bytes);
  const rows = shape[0] ?? 1;
  const cols = shape[1] ?? 1;
  return new Matrix(values, rows, cols);
}

export function writeNpy(matrix: Matrix): Uint8Array {
  const array = matrix.toArray();
  const shape = [matrix.rows, matrix.cols];
  return createNpy(array, shape);
}

export function readNpz(data: ArrayBuffer | Uint8Array): NamedMatrix[] {
  const archive = unzipSync(toUint8Array(data));
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

export function backendKind(): BackendKind {
  if (!activeKind) {
    throw new Error("Backend not initialised. Call init() first.");
  }
  return activeKind;
}

async function getRequire(): Promise<(id: string) => unknown> {
  if (typeof require === "function") {
    return require;
  }
  const { createRequire } = await import("node:module");
  return createRequire(import.meta.url);
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

type ParsedNpy = {
  shape: number[];
  values: Float64Array;
};

function parseNpy(bytes: Uint8Array): ParsedNpy {
  if (
    bytes[0] !== 0x93 ||
    bytes[1] !== 0x4e ||
    bytes[2] !== 0x55 ||
    bytes[3] !== 0x4d ||
    bytes[4] !== 0x50 ||
    bytes[5] !== 0x59
  ) {
    throw new Error("Invalid NPY magic header");
  }

  const major = bytes[6];
  const minor = bytes[7];
  let headerLength: number;
  let offset: number;

  if (major === 1) {
    headerLength = bytes[8] | (bytes[9] << 8);
    offset = 10;
  } else if (major === 2) {
    headerLength =
      bytes[8] |
      (bytes[9] << 8) |
      (bytes[10] << 16) |
      (bytes[11] << 24);
    offset = 12;
  } else {
    throw new Error(`Unsupported NPY version ${major}.${minor}`);
  }

  const headerText = strFromU8(bytes.subarray(offset, offset + headerLength));
  const descrMatch = headerText.match(/'descr':\s*'([^']+)'/);
  if (!descrMatch || descrMatch[1] !== "<f8") {
    throw new Error("Only little-endian float64 NPY files are supported");
  }
  const fortranMatch = headerText.match(/'fortran_order':\s*(False|True)/);
  if (!fortranMatch || fortranMatch[1] !== "False") {
    throw new Error("Only C-order NPY files are supported");
  }
  const shapeMatch = headerText.match(/'shape':\s*\(([^)]*)\)/);
  if (!shapeMatch) {
    throw new Error("Failed to parse NPY shape");
  }
  const rawDims = shapeMatch[1]
    .split(",")
    .map((dim) => dim.trim())
    .filter(Boolean);
  const shape = rawDims.map((dim) => Number.parseInt(dim, 10));
  if (shape.length === 0 || shape.some((dim) => Number.isNaN(dim))) {
    throw new Error("Invalid NPY shape");
  }

  const dataOffset = offset + headerLength;
  const dataBytes = bytes.subarray(dataOffset);
  const valueView = new Float64Array(
    dataBytes.buffer,
    dataBytes.byteOffset,
    dataBytes.byteLength / Float64Array.BYTES_PER_ELEMENT
  );
  const values = new Float64Array(valueView);
  return { shape, values };
}

function createNpy(values: Float64Array, shape: number[]): Uint8Array {
  if (shape.length === 0) {
    throw new Error("Shape must contain at least one dimension");
  }
  const expectedCount = shape.reduce((acc, dim) => acc * dim, 1);
  if (expectedCount !== values.length) {
    throw new Error("Data length does not match shape");
  }

  const shapeText =
    shape.length === 1 ? `${shape[0]},` : shape.join(", ");
  let header =
    `{'descr': '<f8', 'fortran_order': False, 'shape': (${shapeText}), }`;
  const baseLength = 10; // magic(6) + version(2) + header_len(2)
  while ((baseLength + header.length) % 16 !== 0) {
    header += " ";
  }
  header += "\n";
  const headerBytes = strToU8(header, true);
  const arrayBytes = new Uint8Array(
    values.buffer,
    values.byteOffset,
    values.byteLength
  );

  const output = new Uint8Array(baseLength + headerBytes.length + arrayBytes.length);
  output.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59], 0);
  output[6] = 1;
  output[7] = 0;
  output[8] = headerBytes.length & 0xff;
  output[9] = (headerBytes.length >> 8) & 0xff;
  output.set(headerBytes, baseLength);
  output.set(arrayBytes, baseLength + headerBytes.length);
  return output;
}
