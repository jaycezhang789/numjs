import { access, cp, mkdir, readdir, rm, readFile, writeFile } from "node:fs/promises";
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

// Map platform-specific suffix to per-platform npm subpackage directory
const PLATFORM_PACKAGE_DIRS = new Map([
  ["win32-x64-msvc", resolve(jsRoot, "npm/win32-x64-msvc")],
  ["win32-arm64-msvc", resolve(jsRoot, "npm/win32-arm64-msvc")],
  ["darwin-x64", resolve(jsRoot, "npm/darwin-x64")],
  ["darwin-arm64", resolve(jsRoot, "npm/darwin-arm64")],
  ["linux-x64-gnu", resolve(jsRoot, "npm/linux-x64-gnu")],
  ["linux-x64-musl", resolve(jsRoot, "npm/linux-x64-musl")],
  ["linux-arm64-gnu", resolve(jsRoot, "npm/linux-arm64-gnu")],
  ["linux-arm64-musl", resolve(jsRoot, "npm/linux-arm64-musl")],
]);

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
  // Remove .gitignore from copied wasm pkg to ensure npm includes files
  const wasmGitignore = resolve(wasmTarget, ".gitignore");
  if (await exists(wasmGitignore)) {
    await rm(wasmGitignore, { force: true });
  }
  const wasmPackageJson = resolve(wasmTarget, "package.json");
  if (await exists(wasmPackageJson)) {
    try {
      const contents = JSON.parse(await readFile(wasmPackageJson, "utf8"));
      if (!contents.type) {
        contents.type = "module";
        await writeFile(wasmPackageJson, `${JSON.stringify(contents, null, 2)}\n`, "utf8");
      }
    } catch (error) {
      console.warn("[copy-artifacts] Unable to update wasm package.json", error);
    }
  }
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

    // Also copy to platform-specific npm subpackages when suffix is present
    // Expected patterns: index.<suffix>.node
    const match = entry.match(/^index\.(.+)\.node$/);
    if (match) {
      const suffix = match[1];
      const pkgDir = PLATFORM_PACKAGE_DIRS.get(suffix);
      if (pkgDir) {
        await mkdir(pkgDir, { recursive: true });
        const perPlatformTarget = resolve(pkgDir, `index.${suffix}.node`);
        await cp(sourcePath, perPlatformTarget, { recursive: false, force: true });
      }
    }
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
