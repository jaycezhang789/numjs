/* eslint-disable @typescript-eslint/no-non-null-assertion */
/**
 * WebGPU compute backend for accelerating hot math paths (matmul, conv2d, reduce).
 * These helpers operate on plain Float32Array buffers; higher level wrappers
 * inside src/index.ts convert to/from Matrix handles.
 */

const MATMUL_TILE_SIZE_X = 16;
const MATMUL_TILE_SIZE_Y = 16;
const REDUCE_WORKGROUP_SIZE = 256;
const CONV_TILE_X = 16;
const CONV_TILE_Y = 16;

type FloatArray = Float32Array;

export type WebGpuDeviceContext = {
  adapter: GPUAdapter;
  device: GPUDevice;
  queue: GPUQueue;
};

export type WebGpuMatmulInputs = {
  a: FloatArray;
  b: FloatArray;
  rowsA: number;
  colsA: number;
  colsB: number;
};

export type WebGpuReduceInputs = {
  data: FloatArray;
};

export type WebGpuConv2DInputs = {
  input: FloatArray;
  kernel: FloatArray;
  inputRows: number;
  inputCols: number;
  kernelRows: number;
  kernelCols: number;
  stride?: number;
  pad?: number;
};

export interface WebGpuEngine {
  context: WebGpuDeviceContext;
  matmul(inputs: WebGpuMatmulInputs): Promise<FloatArray>;
  reduceSum(inputs: WebGpuReduceInputs): Promise<number>;
  conv2d(inputs: WebGpuConv2DInputs): Promise<FloatArray>;
}

export type CreateWebGpuEngineOptions = {
  forceFallback?: boolean;
  useStub?: boolean;
};

export function isWebGpuSupported(): boolean {
  return typeof navigator !== "undefined" && typeof navigator.gpu !== "undefined";
}

export async function createWebGpuEngine(
  options: CreateWebGpuEngineOptions = {}
): Promise<WebGpuEngine | null> {
  if (options.forceFallback) {
    return null;
  }
  if (!isWebGpuSupported()) {
    return null;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    return null;
  }
  const device = await adapter.requestDevice();
  const queue = device.queue;
  const context: WebGpuDeviceContext = { adapter, device, queue };
  if (options.useStub) {
    const engine = new WebGpuStubEngine(context);
    await engine.ensurePipelines();
    return engine;
  }
  const engine = new WebGpuEngineImpl(context);
  await engine.ensurePipelines();
  return engine;
}

class WebGpuEngineImpl implements WebGpuEngine {
  readonly context: WebGpuDeviceContext;
  private matmulPipeline: GPUComputePipeline | null = null;
  private reducePipeline: GPUComputePipeline | null = null;
  private convPipeline: GPUComputePipeline | null = null;

  constructor(context: WebGpuDeviceContext) {
    this.context = context;
    this.context.device.lost.then((info) => {
      console.warn("[numjs] WebGPU device lost:", info.message);
      this.matmulPipeline = null;
      this.reducePipeline = null;
      this.convPipeline = null;
    });
  }

  async ensurePipelines(): Promise<void> {
    await Promise.all([
      this.ensureMatmulPipeline(),
      this.ensureReducePipeline(),
      this.ensureConvPipeline(),
    ]);
  }

  async matmul(inputs: WebGpuMatmulInputs): Promise<FloatArray> {
    const { rowsA, colsA, colsB } = inputs;
    if (rowsA === 0 || colsB === 0) {
      return new Float32Array(rowsA * colsB);
    }
    const pipeline = await this.ensureMatmulPipeline();
    const { device, queue } = this.context;
    const aBuffer = createStorageBuffer(device, inputs.a);
    const bBuffer = createStorageBuffer(device, inputs.b);
    const outputSize = rowsA * colsB * 4;
    const cBuffer = device.createBuffer({
      size: outputSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const params = new Uint32Array([rowsA, colsA, colsB]);
    const paramsBuffer = createUniformBuffer(device, params);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: aBuffer } },
        { binding: 1, resource: { buffer: bBuffer } },
        { binding: 2, resource: { buffer: cBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });
    const readBuffer = device.createBuffer({
      size: outputSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const dispatchX = Math.ceil(rowsA / MATMUL_TILE_SIZE_X);
    const dispatchY = Math.ceil(colsB / MATMUL_TILE_SIZE_Y);
    pass.dispatchWorkgroups(dispatchX, dispatchY);
    pass.end();
    encoder.copyBufferToBuffer(cBuffer, 0, readBuffer, 0, outputSize);
    queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const outCopy = readBuffer.getMappedRange();
    const result = new Float32Array(outCopy.slice(0));
    readBuffer.unmap();
    aBuffer.destroy();
    bBuffer.destroy();
    cBuffer.destroy();
    paramsBuffer.destroy();
    readBuffer.destroy();
    return result;
  }

  async reduceSum(inputs: WebGpuReduceInputs): Promise<number> {
    const { data } = inputs;
    if (data.length === 0) {
      return 0;
    }
    const pipeline = await this.ensureReducePipeline();
    const { device, queue } = this.context;
    const length = data.length >>> 0;
    const workgroupCount = Math.max(1, Math.ceil(length / REDUCE_WORKGROUP_SIZE));
    const params = new Uint32Array([length, workgroupCount]);
    const paramsBuffer = createUniformBuffer(device, params);
    const inputBuffer = createStorageBuffer(device, data);
    const partialSize = workgroupCount * 4;
    const partialBuffer = device.createBuffer({
      size: partialSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: partialBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });
    const readBuffer = device.createBuffer({
      size: partialSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupCount);
    pass.end();
    encoder.copyBufferToBuffer(partialBuffer, 0, readBuffer, 0, partialSize);
    queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const view = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    let total = 0;
    for (let i = 0; i < view.length; i += 1) {
      total += view[i];
    }
    paramsBuffer.destroy();
    inputBuffer.destroy();
    partialBuffer.destroy();
    readBuffer.destroy();
    return total;
  }

  async conv2d(inputs: WebGpuConv2DInputs): Promise<FloatArray> {
    const {
      input,
      kernel,
      inputRows,
      inputCols,
      kernelRows,
      kernelCols,
      stride = 1,
      pad = 0,
    } = inputs;
    const outRows = Math.floor((inputRows + 2 * pad - kernelRows) / stride + 1);
    const outCols = Math.floor((inputCols + 2 * pad - kernelCols) / stride + 1);
    if (outRows <= 0 || outCols <= 0) {
      throw new Error("conv2d: invalid output size; check stride/padding/kernel dimensions.");
    }
    const pipeline = await this.ensureConvPipeline();
    const { device, queue } = this.context;
    const inputBuffer = createStorageBuffer(device, input);
    const kernelBuffer = createStorageBuffer(device, kernel);
    const outSize = outRows * outCols * 4;
    const outputBuffer = device.createBuffer({
      size: outSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const params = new Uint32Array([
      inputRows,
      inputCols,
      kernelRows,
      kernelCols,
      outRows,
      outCols,
      stride,
      pad,
    ]);
    const paramsBuffer = createUniformBuffer(device, params);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: kernelBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });
    const readBuffer = device.createBuffer({
      size: outSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const dispatchX = Math.ceil(outRows / CONV_TILE_X);
    const dispatchY = Math.ceil(outCols / CONV_TILE_Y);
    pass.dispatchWorkgroups(dispatchX, dispatchY);
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outSize);
    queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const view = readBuffer.getMappedRange();
    const result = new Float32Array(view.slice(0));
    readBuffer.unmap();
    inputBuffer.destroy();
    kernelBuffer.destroy();
    outputBuffer.destroy();
    paramsBuffer.destroy();
    readBuffer.destroy();
    return result;
  }

  private async ensureMatmulPipeline(): Promise<GPUComputePipeline> {
    if (this.matmulPipeline) {
      return this.matmulPipeline;
    }
    const module = this.context.device.createShaderModule({
      code: MATMUL_SHADER,
    });
    this.matmulPipeline = await this.context.device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module,
        entryPoint: "main",
      },
    });
    return this.matmulPipeline;
  }

  private async ensureReducePipeline(): Promise<GPUComputePipeline> {
    if (this.reducePipeline) {
      return this.reducePipeline;
    }
    const module = this.context.device.createShaderModule({
      code: REDUCE_SHADER,
    });
    this.reducePipeline = await this.context.device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module,
        entryPoint: "main",
      },
    });
    return this.reducePipeline;
  }

  private async ensureConvPipeline(): Promise<GPUComputePipeline> {
    if (this.convPipeline) {
      return this.convPipeline;
    }
    const module = this.context.device.createShaderModule({
      code: CONV_SHADER,
    });
    this.convPipeline = await this.context.device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module,
        entryPoint: "main",
      },
    });
    return this.convPipeline;
  }
}

class WebGpuStubEngine implements WebGpuEngine {
  readonly context: WebGpuDeviceContext;
  private noopPipeline: GPUComputePipeline | null = null;

  constructor(context: WebGpuDeviceContext) {
    this.context = context;
    this.context.device.lost.then((info) => {
      console.warn("[numjs] WebGPU device lost (stub engine):", info.message);
      this.noopPipeline = null;
    });
  }

  async ensurePipelines(): Promise<void> {
    await this.ensureNoopPipeline();
  }

  async matmul(inputs: WebGpuMatmulInputs): Promise<FloatArray> {
    const pipeline = await this.ensureNoopPipeline();
    dispatchNoop(this.context, pipeline);
    return cpuMatmul(inputs);
  }

  async reduceSum(inputs: WebGpuReduceInputs): Promise<number> {
    const pipeline = await this.ensureNoopPipeline();
    dispatchNoop(this.context, pipeline);
    return cpuReduceSum(inputs);
  }

  async conv2d(inputs: WebGpuConv2DInputs): Promise<FloatArray> {
    const pipeline = await this.ensureNoopPipeline();
    dispatchNoop(this.context, pipeline);
    return cpuConv2D(inputs);
  }

  private async ensureNoopPipeline(): Promise<GPUComputePipeline> {
    if (this.noopPipeline) {
      return this.noopPipeline;
    }
    const module = this.context.device.createShaderModule({
      code: NOOP_SHADER,
    });
    const pipeline = this.context.device.createComputePipeline({
      layout: "auto",
      compute: {
        module,
        entryPoint: "main",
      },
    });
    this.noopPipeline = pipeline;
    return pipeline;
  }
}

function dispatchNoop(context: WebGpuDeviceContext, pipeline: GPUComputePipeline): void {
  const { device, queue } = context;
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.dispatchWorkgroups(1);
  pass.end();
  queue.submit([encoder.finish()]);
}

function cpuMatmul(inputs: WebGpuMatmulInputs): Float32Array {
  const { a, b, rowsA, colsA, colsB } = inputs;
  const out = new Float32Array(rowsA * colsB);
  for (let row = 0; row < rowsA; row += 1) {
    const rowOffset = row * colsA;
    const outOffset = row * colsB;
    for (let col = 0; col < colsB; col += 1) {
      let acc = 0;
      for (let k = 0; k < colsA; k += 1) {
        acc += a[rowOffset + k] * b[k * colsB + col];
      }
      out[outOffset + col] = acc;
    }
  }
  return out;
}

function cpuReduceSum(inputs: WebGpuReduceInputs): number {
  const { data } = inputs;
  let total = 0;
  for (let index = 0; index < data.length; index += 1) {
    total += data[index];
  }
  return total;
}

function cpuConv2D(inputs: WebGpuConv2DInputs): Float32Array {
  const {
    input,
    kernel,
    inputRows,
    inputCols,
    kernelRows,
    kernelCols,
    stride = 1,
    pad = 0,
  } = inputs;
  const outRows = Math.floor((inputRows + 2 * pad - kernelRows) / stride + 1);
  const outCols = Math.floor((inputCols + 2 * pad - kernelCols) / stride + 1);
  const output = new Float32Array(Math.max(outRows, 0) * Math.max(outCols, 0));
  if (outRows <= 0 || outCols <= 0) {
    return output;
  }
  for (let outRow = 0; outRow < outRows; outRow += 1) {
    for (let outCol = 0; outCol < outCols; outCol += 1) {
      let acc = 0;
      for (let kr = 0; kr < kernelRows; kr += 1) {
        for (let kc = 0; kc < kernelCols; kc += 1) {
          const inRow = outRow * stride + kr - pad;
          const inCol = outCol * stride + kc - pad;
          if (
            inRow >= 0 &&
            inRow < inputRows &&
            inCol >= 0 &&
            inCol < inputCols
          ) {
            const inputIndex = inRow * inputCols + inCol;
            const kernelIndex = kr * kernelCols + kc;
            acc += input[inputIndex] * kernel[kernelIndex];
          }
        }
      }
      output[outRow * outCols + outCol] = acc;
    }
  }
  return output;
}

function createStorageBuffer(device: GPUDevice, source: FloatArray): GPUBuffer {
  const buffer = device.createBuffer({
    size: alignTo(source.byteLength, 4),
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(buffer, 0, source.buffer, source.byteOffset, source.byteLength);
  return buffer;
}

function createUniformBuffer(device: GPUDevice, data: Uint32Array): GPUBuffer {
  const buffer = device.createBuffer({
    size: alignTo(data.byteLength, 16),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  return buffer;
}

function alignTo(value: number, multiple: number): number {
  const remainder = value % multiple;
  return remainder === 0 ? value : value + multiple - remainder;
}

const NOOP_SHADER = /* wgsl */ `
@compute @workgroup_size(1)
fn main() {
}
`;

const MATMUL_SHADER = /* wgsl */ `
struct Matrix {
  values: array<f32>;
};

struct Params {
  rowsA: u32;
  colsA: u32;
  colsB: u32;
};

@group(0) @binding(0) var<storage, read> lhs: Matrix;
@group(0) @binding(1) var<storage, read> rhs: Matrix;
@group(0) @binding(2) var<storage, read_write> out: Matrix;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${MATMUL_TILE_SIZE_X}, ${MATMUL_TILE_SIZE_Y}, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.x;
  let col = global_id.y;
  if (row >= params.rowsA || col >= params.colsB) {
    return;
  }
  var acc: f32 = 0.0;
  let colsA = params.colsA;
  let colsB = params.colsB;
  for (var k: u32 = 0u; k < colsA; k = k + 1u) {
    let aIndex = row * colsA + k;
    let bIndex = k * colsB + col;
    acc = acc + lhs.values[aIndex] * rhs.values[bIndex];
  }
  let cIndex = row * colsB + col;
  out.values[cIndex] = acc;
}
`;

const REDUCE_SHADER = /* wgsl */ `
struct Matrix {
  values: array<f32>;
};

struct Params {
  length: u32;
  workgroups: u32;
};

@group(0) @binding(0) var<storage, read> inputData: Matrix;
@group(0) @binding(1) var<storage, read_write> partialSums: Matrix;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> tile: array<f32, ${REDUCE_WORKGROUP_SIZE}>;

@compute @workgroup_size(${REDUCE_WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group_id: vec3<u32>
) {
  let workgroupSize: u32 = ${REDUCE_WORKGROUP_SIZE}u;
  let totalGroups = params.workgroups;
  let length = params.length;
  var sum: f32 = 0.0;
  var index: u32 = global_id.x;
  let stride: u32 = workgroupSize * totalGroups;
  while (index < length) {
    sum = sum + inputData.values[index];
    index = index + stride;
  }
  tile[local_id.x] = sum;
  workgroupBarrier();

  var offset: u32 = workgroupSize / 2u;
  loop {
    if (offset == 0u) {
      break;
    }
    if (local_id.x < offset) {
      tile[local_id.x] = tile[local_id.x] + tile[local_id.x + offset];
    }
    workgroupBarrier();
    offset = offset / 2u;
  }

  if (local_id.x == 0u) {
    partialSums.values[group_id.x] = tile[0];
  }
}
`;

const CONV_SHADER = /* wgsl */ `
struct Params {
  inRows: u32;
  inCols: u32;
  kernelRows: u32;
  kernelCols: u32;
  outRows: u32;
  outCols: u32;
  stride: u32;
  pad: u32;
};

struct Matrix {
  values: array<f32>;
};

@group(0) @binding(0) var<storage, read> inputData: Matrix;
@group(0) @binding(1) var<storage, read> kernelData: Matrix;
@group(0) @binding(2) var<storage, read_write> outputData: Matrix;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${CONV_TILE_X}, ${CONV_TILE_Y}, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let outRow = global_id.x;
  let outCol = global_id.y;
  if (outRow >= params.outRows || outCol >= params.outCols) {
    return;
  }

  let stride = params.stride;
  let pad = params.pad;
  let inRows = params.inRows;
  let inCols = params.inCols;
  let kRows = params.kernelRows;
  let kCols = params.kernelCols;

  var acc: f32 = 0.0;
  for (var kr: u32 = 0u; kr < kRows; kr = kr + 1u) {
    for (var kc: u32 = 0u; kc < kCols; kc = kc + 1u) {
      let inRow = i32(outRow) * i32(stride) + i32(kr) - i32(pad);
      let inCol = i32(outCol) * i32(stride) + i32(kc) - i32(pad);
      if (inRow < 0 || inRow >= i32(inRows) || inCol < 0 || inCol >= i32(inCols)) {
        continue;
      }
      let inputIndex = u32(inRow) * inCols + u32(inCol);
      let kernelIndex = kr * kCols + kc;
      acc = acc + inputData.values[inputIndex] * kernelData.values[kernelIndex];
    }
  }
  let outIndex = outRow * params.outCols + outCol;
  outputData.values[outIndex] = acc;
}
`;
