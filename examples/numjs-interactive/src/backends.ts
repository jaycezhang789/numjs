import { init, backendKind } from "@jayce789/numjs";

type Logger = (message: string) => void;

export async function logBackendSupport(log: Logger) {
  log("🔍 检查后端能力...");

  const preferOrder: Array<"napi" | "wasm"> = ["napi", "wasm"];

  for (const prefer of preferOrder) {
    try {
      await init({ preferBackend: prefer });
      log(`• preferBackend="${prefer}" ➜ 成功，backendKind() = ${backendKind()}`);
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error);
      log(`• preferBackend="${prefer}" ➜ 失败 (${reason})`);
    }
  }

  log("提示：在 Node.js 中运行 `pnpm node-demo` 查看 CLI 版本的加载日志。");
}
