import { access, cp, mkdir, readdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const jsRoot = resolve(here, "..");
const distDir = resolve(jsRoot, "dist");

const wasmSource = resolve(jsRoot, "../wasm/pkg");
const wasmTarget = resolve(distDir, "bindings/wasm");

const napiSourceDir = resolve(jsRoot, "../napi");
const napiTargetDir = resolve(distDir, "bindings/napi");
const napiTypeSource = resolve(jsRoot, "../napi/index.d.ts");
const napiTypeTarget = resolve(napiTargetDir, "index.d.ts");

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

await mkdir(napiTargetDir, { recursive: true });
let copiedNapi = false;
if (await exists(napiSourceDir)) {
  const entries = await readdir(napiSourceDir);
  for (const entry of entries) {
    if (!entry.startsWith("index") || !entry.endsWith(".node")) {
      continue;
    }
    const sourcePath = resolve(napiSourceDir, entry);
    const targetPath = resolve(napiTargetDir, entry);
    await cp(sourcePath, targetPath, { recursive: false, force: true });
    copiedNapi = true;
  }
  if (await exists(napiTypeSource)) {
    await cp(napiTypeSource, napiTypeTarget, { recursive: false, force: true });
  }
}
if (!copiedNapi) {
  console.warn(
    "[copy-artifacts] N-API artifacts not found. Expected at",
    napiSourceDir
  );
}
