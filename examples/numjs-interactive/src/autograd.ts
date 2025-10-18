import {
  tensor,
  tensorAdd,
  tensorMatmul,
  tensorSum,
  tensorSumAxis,
  Tensor,
} from "@jayce789/numjs";

type Logger = (message: string) => void;

function formatMatrix(matrix: Tensor["value"]): string {
  return JSON.stringify(Array.from(matrix.toArray()));
}

export async function runAutogradDemo(log: Logger) {
  log("🧠 Autograd 演示：y = sum((W @ x) + b)");

  const x = tensor(
    new Float64Array([1, 2, 3, 4]),
    2,
    2,
    { requiresGrad: false, name: "x" }
  );
  const w = tensor(
    new Float64Array([0.5, -0.25, 1.0, -0.75]),
    2,
    2,
    { requiresGrad: true, name: "W" }
  );
  const b = tensor([0.1, -0.2], 1, 2, { requiresGrad: true, name: "b" });

  const wx = tensorMatmul(w, x);
  const logits = tensorAdd(wx, tensorSumAxis(b, 0));
  const loss = tensorSum(logits);

  log(`• loss 数值: ${formatMatrix(loss.value)}`);

  loss.backward();

  const weightGrad = w.grad?.toArray() ?? [];
  const biasGrad = b.grad?.toArray() ?? [];
  log(`• W.grad: ${JSON.stringify(Array.from(weightGrad))}`);
  log(`• b.grad: ${JSON.stringify(Array.from(biasGrad))}`);
  log("提示：修改 W/x/b 初始值可观察梯度变化。");
}
