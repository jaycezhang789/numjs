import { mkdir, writeFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import { resolve } from "node:path";

const {
  init,
  Matrix,
  matmul,
  matmulAsync,
  conv2d,
  maxPool,
  gaussianBlur,
} = await import("../dist/index.js");

await init({ threads: false });

function time(fn, iterations = 25) {
  const start = performance.now();
  for (let i = 0; i < iterations; i += 1) {
    fn();
  }
  const duration = performance.now() - start;
  return duration / iterations;
}

async function timeAsync(fn, iterations = 10) {
  const start = performance.now();
  for (let i = 0; i < iterations; i += 1) {
    // eslint-disable-next-line no-await-in-loop
    await fn();
  }
  const duration = performance.now() - start;
  return duration / iterations;
}

function createMatrix(size, dtype = "float32") {
  const data = new Float32Array(size.rows * size.cols).map((_, i) => (i % 7) - 3);
  return new Matrix(data, size.rows, size.cols, { dtype });
}

const benchmarks = [];

const matA = createMatrix({ rows: 128, cols: 64 });
const matB = createMatrix({ rows: 64, cols: 128 });
benchmarks.push({
  name: "matmul (128x64 x 64x128)",
  type: "micro",
  unit: "ms/op",
  value: time(() => matmul(matA, matB)),
});

benchmarks.push({
  name: "matmulAsync (128x64 x 64x128)",
  type: "micro",
  unit: "ms/op",
  value: await timeAsync(() => matmulAsync(matA, matB)),
});

const image = createMatrix({ rows: 256, cols: 256 });
const kernel = new Matrix(
  new Float32Array([
    1, 0, -1,
    2, 0, -2,
    1, 0, -1,
  ]),
  3,
  3,
  { dtype: "float32" }
);
benchmarks.push({
  name: "conv2d (256x256, 3x3)",
  type: "micro",
  unit: "ms/op",
  value: await timeAsync(() => conv2d(image, kernel)),
});

benchmarks.push({
  name: "maxPool (256x256, 2x2 stride 2)",
  type: "micro",
  unit: "ms/op",
  value: time(() => maxPool(image, 2, 2, { stride: 2 })),
});

benchmarks.push({
  name: "gaussianBlur (256x256, sigma=1.2)",
  type: "micro",
  unit: "ms/op",
  value: await timeAsync(() => gaussianBlur(image, { sigma: 1.2 })),
});

const macroIterations = 5;
const macroStart = performance.now();
for (let i = 0; i < macroIterations; i += 1) {
  const mm = matmul(matA, matB);
  const blurred = await gaussianBlur(image, { sigma: 1 });
  await conv2d(blurred, kernel);
  await matmulAsync(mm, matB.transpose());
}
const macroDuration = (performance.now() - macroStart) / macroIterations;
benchmarks.push({
  name: "pipeline (matmul + blur + conv2d)",
  type: "macro",
  unit: "ms/op",
  value: macroDuration,
});

const rows = benchmarks
  .map((entry) =>
    `<tr><td>${entry.type}</td><td>${entry.name}</td><td>${entry.value.toFixed(3)}</td><td>${entry.unit}</td></tr>`
  )
  .join("\n");

const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>NumJS Benchmarks</title>
  <style>
    body { font-family: sans-serif; padding: 2rem; background: #f7f7f7; }
    table { border-collapse: collapse; width: 100%; background: #fff; }
    th, td { border: 1px solid #ccc; padding: 0.5rem 0.75rem; text-align: left; }
    th { background: #f0f0f0; }
  </style>
</head>
<body>
  <h1>NumJS Benchmarks</h1>
  <p>Generated at ${new Date().toISOString()}</p>
  <table>
    <thead><tr><th>Type</th><th>Benchmark</th><th>Time</th><th>Unit</th></tr></thead>
    <tbody>
      ${rows}
    </tbody>
  </table>
</body>
</html>`;

await mkdir('tmp', { recursive: true });
const reportPath = resolve('tmp', 'numjs-bench.html');
await writeFile(reportPath, html, 'utf8');
console.log(`Benchmark report written to ${reportPath}`);
