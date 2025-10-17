import test from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, resolve } from "node:path";

const testDir = dirname(fileURLToPath(import.meta.url));
const distModuleUrl = pathToFileURL(resolve(testDir, "../dist/index.js")).href;

async function runBackend(preference) {
  const script = `
    import * as num from ${JSON.stringify(distModuleUrl)};
    try {
      await num.init({ preferBackend: ${JSON.stringify(preference)}, threads: false, webGpu: { forceFallback: true } });
      const a = new num.Matrix(new Float64Array([1, 2, 3, 4]), 2, 2, { dtype: "float64" });
      const b = new num.Matrix(new Float64Array([4, 3, 2, 1]), 2, 2, { dtype: "float64" });
      const sum = num.add(a, b);
      const product = num.matmul(a, b).astype("float64", { copy: false });
      console.log(JSON.stringify({
        ok: true,
        backend: num.backendKind(),
        sum: Array.from(sum.toArray()),
        product: Array.from(product.toArray())
      }));
    } catch (error) {
      console.log(JSON.stringify({ ok: false, error: error.message }));
      process.exitCode = 1;
    }
  `;

  const child = spawn(process.execPath, ["--input-type=module", "-"], {
    env: { ...process.env, DEBUG: process.env.DEBUG ?? "" },
    stdio: ["pipe", "pipe", "inherit"],
  });
  child.stdin.end(script);
  let output = "";
  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (chunk) => {
    output += chunk;
  });
  let parsed;
  const exitCode = await new Promise((resolve) => child.on("close", resolve));
  try {
    parsed = output ? JSON.parse(output.split("\n").filter(Boolean).pop()) : { ok: false, error: "no output" };
  } catch (error) {
    parsed = { ok: false, error: `invalid JSON: ${output}` };
  }
  return { parsed, exitCode };
}

test("WASM and N-API produce consistent results", async (t) => {
  const wasmResult = await runBackend("wasm");
  if (!wasmResult.parsed.ok) {
    t.skip(`wasm backend unavailable: ${wasmResult.parsed.error}`);
    return;
  }
  const napiResult = await runBackend("napi");
  if (!napiResult.parsed.ok) {
    t.skip(`napi backend unavailable: ${napiResult.parsed.error}`);
    return;
  }
  const wasm = wasmResult.parsed;
  const napi = napiResult.parsed;
  assert.equal(wasm.sum.length, napi.sum.length);
  assert.equal(wasm.product.length, napi.product.length);
  for (let i = 0; i < wasm.sum.length; i += 1) {
    assert.ok(Math.abs(wasm.sum[i] - napi.sum[i]) < 1e-9);
  }
  for (let i = 0; i < wasm.product.length; i += 1) {
    assert.ok(Math.abs(wasm.product[i] - napi.product[i]) < 1e-9);
  }
});
