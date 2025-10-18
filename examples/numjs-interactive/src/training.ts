import { Matrix, trainLinearRegression } from "@jayce789/numjs";

type Logger = (message: string) => void;

export async function runTrainingDemo(log: Logger) {
  log("🏋️ 线性回归训练示例：拟合 y = 3x + 2 + 噪声");

  const samples = 40;
  const features = new Float64Array(samples);
  const targets = new Float64Array(samples);
  for (let i = 0; i < samples; i += 1) {
    const x = (i - samples / 2) / (samples / 4);
    const noise = (Math.random() - 0.5) * 0.3;
    features[i] = x;
    targets[i] = 3 * x + 2 + noise;
  }

  const featureMatrix = new Matrix(features, samples, 1);
  const targetMatrix = new Matrix(targets, samples, 1);

  const result = trainLinearRegression(featureMatrix, targetMatrix, {
    epochs: 120,
    learningRate: 0.05,
    optimizer: "adam",
  });

  const learnedWeight = Array.from(result.weights.toArray())[0];
  const learnedBias = Array.from(result.bias.toArray())[0];
  const finalLoss = result.losses[result.losses.length - 1];

  log(`• 训练结束: w ≈ ${learnedWeight.toFixed(3)}, b ≈ ${learnedBias.toFixed(3)}`);
  log(`• 最终 MSE: ${finalLoss.toFixed(4)}`);
  log("提示：尝试调整 epochs/learningRate/optimizer 观察收敛差异。");
}
