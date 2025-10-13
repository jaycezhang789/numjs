import { access, cp, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const jsRoot = resolve(here, "..");
const distDir = resolve(jsRoot, "dist");

const wasmSource = resolve(jsRoot, "../wasm/pkg");
const wasmTarget = resolve(distDir, "bindings/wasm");

const napiSource = resolve(jsRoot, "../napi/index.node");
const napiTarget = resolve(distDir, "bindings/napi/index.node");

async function exists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

await mkdir(distDir, { recursive: true });

if (await exists(wasmSource)) {
  await mkdir(wasmTarget, { recursive: true });
  await cp(wasmSource, wasmTarget, { recursive: true, force: true });
} else {
  console.warn(
    "[copy-artifacts] wasm-pack output not found. Expected at",
    wasmSource
  );
}

if (await exists(napiSource)) {
  await mkdir(dirname(napiTarget), { recursive: true });
  await cp(napiSource, napiTarget, { recursive: false, force: true });
} else {
  console.warn(
    "[copy-artifacts] N-API artifact not found. Expected at",
    napiSource
  );
}
