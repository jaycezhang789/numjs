import Koa from "koa";
import Router from "koa-router";
import {
  Matrix,
  add,
  backendKind,
  init,
  matmul,
  writeNpy,
} from "@jayce789/numjs";

const app = new Koa();
const router = new Router();

let ready = false;

async function ensureReady() {
  if (!ready) {
    await init();
    ready = true;
  }
}

router.get("/health", async (ctx) => {
  await ensureReady();
  ctx.body = { status: "ok", backend: backendKind() };
});

router.get("/matmul", async (ctx) => {
  await ensureReady();
  const a = new Matrix([1, 2, 3, 4], 2, 2);
  const b = new Matrix([5, 6, 7, 8], 2, 2);
  const result = matmul(a, b).toArray();
  // Coerce potential bigint-typed arrays to numbers for JSON
  const numbers = Array.from(result as unknown as ArrayLike<any>, Number);
  ctx.body = { result: numbers };
});

router.get("/export.npy", async (ctx) => {
  await ensureReady();
  const matrix = add(
    new Matrix([1, 2, 3, 4], 2, 2),
    new Matrix([5, 6, 7, 8], 2, 2)
  );
  const bytes = writeNpy(matrix);
  ctx.set("Content-Type", "application/octet-stream");
  ctx.set("Content-Disposition", 'attachment; filename="sum.npy"');
  ctx.body = Buffer.from(bytes);
});

app.use(router.routes()).use(router.allowedMethods());

const port = Number.parseInt(process.env.PORT ?? "3000", 10);
app.listen(port, () => {
  console.log(`Koa example listening on http://localhost:${port}`);
});
