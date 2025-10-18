import { init, initWebGpu, matmulAsync, Matrix } from "@jayce789/numjs";

type Logger = (message: string) => void;

export async function tryWebGpuDemo(log: Logger) {
  log("⚙️ 尝试启用 WebGPU...");
  await init({ preferBackend: "wasm", threads: true });

  const enabled = await initWebGpu();
  if (!enabled) {
    log("❗ 当前环境缺少 WebGPU 支持，已自动回退。");
    return;
  }

  const createRandomMatrix = (rows: number, cols: number) =>
    new Matrix(
      Float32Array.from({ length: rows * cols }, () => Math.random() - 0.5),
      rows,
      cols,
      { dtype: "float32" }
    );

  const lhs = createRandomMatrix(256, 256);
  const rhs = createRandomMatrix(256, 256);

  console.time("webgpu:matmul");
  await matmulAsync(lhs, rhs, { mode: "gpu-only" });
  console.timeEnd("webgpu:matmul");

  log("✅ WebGPU matmul 完成，详见浏览器控制台耗时。");
}
