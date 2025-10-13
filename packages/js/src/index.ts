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

function normaliseData(data: Float64Array | number[]): Float64Array {
  return data instanceof Float64Array ? data : Float64Array.from(data);
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
}

export function add(a: Matrix, b: Matrix): Matrix {
  const backend = ensureBackend();
  const result = backend.add(
    (a as unknown as { inner: MatrixHandle }).inner,
    (b as unknown as { inner: MatrixHandle }).inner
  );
  return wrapMatrix(result);
}

export function matmul(a: Matrix, b: Matrix): Matrix {
  const backend = ensureBackend();
  const result = backend.matmul(
    (a as unknown as { inner: MatrixHandle }).inner,
    (b as unknown as { inner: MatrixHandle }).inner
  );
  return wrapMatrix(result);
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
